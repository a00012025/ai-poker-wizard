#!/usr/bin/env python3
"""Backfill `deviations.ev_loss_estimate` + meta fields for existing rows.

Idempotent, resumable, dry-run by default. Work queue is:
    SELECT * FROM deviations WHERE ev_loss_estimate IS NULL ORDER BY id

Each successful UPDATE removes the row from the queue automatically, so
crash mid-run → rerun picks up where it left off. No state file.

Solver calls hit the 3-tier `gto_cache.py` (in-memory + Postgres
gto_api_cache + local .gto_cache/*.json); historical deviations were
analyzed before so cache hit-rate should be ~100%, effectively zero API
round-trips.

Usage:
    python scripts/backfill_ev_loss.py                       # dry-run, all users
    python scripts/backfill_ev_loss.py --execute             # write
    python scripts/backfill_ev_loss.py --execute --limit 100
    python scripts/backfill_ev_loss.py --execute --chat-id 12345
    python scripts/backfill_ev_loss.py --execute --resume    # same as default

Requires: SUPABASE_CONN env var, valid GTO Wizard token (.tokens.json).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncpg  # noqa: E402

logger = logging.getLogger("backfill_ev_loss")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ── Pure helpers (extracted for testability) ──

def _walk_to_decision(
    analysis_result: dict,
    street: str,
    action_index: int,
) -> dict | None:
    """Find the specific solver snapshot for (street, action_index).

    `analysis_result` is the dict returned by `analyze_hand_full()`. It
    contains parallel lists `hero_spots` and `solutions` where each
    hero_spot has a `street` field. For a given street we pick the
    Nth hero_spot on that street (preflop: global counter; postflop:
    per-street counter) to match how `_extract_deviations` indexes them
    in src/gemini_session.py.

    Returns the solver snapshot dict (with `action_solutions`) or None
    if not found / no action_solutions present.
    """
    if not analysis_result:
        return None
    hero_spots = analysis_result.get("hero_spots") or []
    solutions = analysis_result.get("solutions") or []
    if not hero_spots or not solutions:
        return None

    is_preflop = street == "preflop"
    counter = 0
    for i, spot in enumerate(hero_spots):
        spot_street = spot.get("street", "")
        if is_preflop:
            if spot_street != "preflop":
                continue
            if counter == action_index:
                sol = solutions[i] if i < len(solutions) else None
                if sol and "action_solutions" in sol:
                    return sol
                return None
            counter += 1
        else:
            if spot_street != street:
                continue
            if counter == action_index:
                sol = solutions[i] if i < len(solutions) else None
                if sol and "action_solutions" in sol:
                    return sol
                return None
            counter += 1
    return None


def _extract_action_evs(
    decision_snapshot: dict,
    hero_hand: str,
    hero_pos: str,
    is_preflop: bool,
    combo_idx: int | None = None,
) -> dict[str, float] | None:
    """Returns {action_code: ev} or None. Delegates to hh_deviation_check."""
    if not decision_snapshot:
        return None
    try:
        from hh_deviation_check import (
            _get_action_evs_preflop,
            _get_action_evs_postflop,
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"failed to import action-EV helpers: {e}")
        return None
    try:
        if is_preflop:
            return _get_action_evs_preflop(decision_snapshot, hero_hand, hero_pos)
        return _get_action_evs_postflop(
            decision_snapshot, hero_hand, hero_pos, combo_idx=combo_idx
        )
    except Exception:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arg parser. Exposed for testing."""
    p = argparse.ArgumentParser(
        description="Backfill deviations.ev_loss_estimate + meta fields."
    )
    p.add_argument(
        "--execute", action="store_true",
        help="Actually write. Default is dry-run.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not write (default behavior).",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N rows.")
    p.add_argument("--chat-id", type=int, default=None,
                   help="Only process rows for this chat_id.")
    p.add_argument("--resume", action="store_true",
                   help="(No-op; already the default — queue skips done rows.)")
    ns = p.parse_args(argv)
    # Default is dry-run unless --execute is passed.
    ns.dry_run = not ns.execute
    return ns


# ── Backfill core ──

async def _connect():
    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        print("ERROR: SUPABASE_CONN not set. Run: set -a && source .env && set +a",
              file=sys.stderr)
        sys.exit(2)
    return await asyncpg.connect(dsn, statement_cache_size=0)


async def _fetch_queue(conn, chat_id: int | None, limit: int | None) -> list[dict]:
    sql = [
        "SELECT d.id, d.chat_id, d.hand_history_id, d.street, d.action_index,",
        "       d.hero_action, d.gto_action, d.spot_category, d.position,",
        "       d.meta, h.hand_id, h.hand_data",
        "  FROM deviations d",
        "  LEFT JOIN hand_histories h ON h.id = d.hand_history_id",
        " WHERE d.ev_loss_estimate IS NULL",
    ]
    args: list = []
    if chat_id is not None:
        args.append(chat_id)
        sql.append(f" AND d.chat_id = ${len(args)}")
    sql.append(" ORDER BY d.id")
    if limit is not None:
        args.append(limit)
        sql.append(f" LIMIT ${len(args)}")
    rows = await conn.fetch("\n".join(sql), *args)
    return [dict(r) for r in rows]


def _compute_update_for_row(row: dict) -> dict | None:
    """Given a queue row, compute the update payload.

    Returns a dict with keys {ev_loss, meta_updates, gto_best_ev_action,
    gto_dominant_action, pot_type, preflop_line_key, villain_pos, gtow_type,
    gtow_hero_role, aggression_direction}, or None if the hand is
    unreconstructable / no solver data.

    This is I/O-free except for the analyze_hand_full() call, which
    hits the GTO cache — no DB writes.
    """
    hand_data = row.get("hand_data")
    if hand_data is None:
        return None
    if isinstance(hand_data, str):
        try:
            hand_data = json.loads(hand_data)
        except Exception:
            return None
    if not isinstance(hand_data, dict):
        return None

    street = row["street"]
    action_index = row["action_index"]
    hero_action = row["hero_action"]

    try:
        from analyze_hand import analyze_hand_full
    except Exception as e:
        logger.warning(f"import analyze_hand failed: {e}")
        return None

    try:
        result = analyze_hand_full(hand_data)
    except Exception as e:
        logger.debug(f"analyze_hand_full failed id={row['id']}: {e}")
        return None

    hero_pos = result.get("hero_position", "") or row.get("position") or ""
    hero_hand = result.get("hero_hand", "") or ""
    hero_hand_raw = hand_data.get("hero_hand", "") or ""

    # Combo index for postflop
    try:
        from gto_formatter import combo_index_for_hand
        combo_idx = combo_index_for_hand(hero_hand_raw)
    except Exception:
        combo_idx = None

    sol = _walk_to_decision(result, street, action_index)
    if sol is None:
        return None

    is_preflop = street == "preflop"
    action_evs = _extract_action_evs(
        sol, hero_hand, hero_pos, is_preflop, combo_idx=combo_idx,
    )

    try:
        from leak_service import (
            compute_ev_loss,
            pick_best_ev_action,
            classify_aggression_direction,
        )
        from spot_categorizer import (
            compute_preflop_line_key,
            compute_pot_type,
            identify_primary_villain,
            map_spot_to_gtow,
            _identify_preflop_aggressor,
        )
    except Exception as e:
        logger.warning(f"import helpers failed: {e}")
        return None

    ev_loss = compute_ev_loss(action_evs, hero_action)
    gto_best_ev = pick_best_ev_action(action_evs)

    # Dominant (highest-frequency) action from the snapshot
    gto_dominant = None
    action_solutions = sol.get("action_solutions") or []
    if action_solutions:
        best = max(
            action_solutions,
            key=lambda a: a.get("total_frequency", 0),
            default=None,
        )
        if best:
            gto_dominant = best.get("action", {}).get("code")
    if not gto_dominant:
        gto_dominant = row.get("gto_action")

    preflop_actions_str = hand_data.get("preflop_actions", "") or ""
    num_players = hand_data.get("players_at_table", 8) or 8

    try:
        line_key = compute_preflop_line_key(
            preflop_actions_str, hero_pos,
            num_players=num_players,
            action_index=(action_index if is_preflop else 0),
        )
    except Exception:
        line_key = None
    pot_type = compute_pot_type(line_key) if line_key is not None else None

    try:
        villain_pos = identify_primary_villain(
            hand_data, hero_pos, street, None,
        )
    except Exception:
        villain_pos = None

    try:
        pf_agg = _identify_preflop_aggressor(preflop_actions_str, num_players)
    except Exception:
        pf_agg = None
    hero_is_pf_aggressor = (pf_agg == hero_pos)

    try:
        gtow_type, gtow_hero_role = map_spot_to_gtow(
            row.get("spot_category", ""), pot_type, street, hero_is_pf_aggressor,
        )
    except Exception:
        gtow_type, gtow_hero_role = None, None

    aggression_direction = classify_aggression_direction(hero_action, gto_best_ev)

    meta_updates = {
        k: v for k, v in {
            "villain_pos": villain_pos,
            "preflop_line_key": line_key,
            "pot_type": pot_type,
            "aggression_direction": aggression_direction,
            "gtow_type": gtow_type,
            "gtow_hero_role": gtow_hero_role,
            "gto_dominant_action": gto_dominant,
            "gto_best_ev_action": gto_best_ev,
        }.items() if v is not None
    }

    return {
        "ev_loss": ev_loss,
        "meta_updates": meta_updates,
        "has_solver_data": action_evs is not None,
    }


async def _apply_update(conn, dev_id: int, ev_loss: float | None,
                        meta_updates: dict) -> None:
    """UPDATE the deviations row with JSONB merge for meta."""
    meta_json = json.dumps(meta_updates or {})
    await conn.execute(
        """
        UPDATE deviations
           SET ev_loss_estimate = $2,
               meta = COALESCE(meta, '{}'::jsonb) || $3::jsonb
         WHERE id = $1
        """,
        dev_id, ev_loss, meta_json,
    )


async def run_backfill(args: argparse.Namespace) -> int:
    conn = await _connect()
    done = updated = skipped = errors = 0
    try:
        queue = await _fetch_queue(conn, args.chat_id, args.limit)
        total = len(queue)
        mode = "DRY-RUN" if args.dry_run else "EXECUTE"
        logger.info(f"backfill starting: mode={mode} queue={total}")
        if total == 0:
            logger.info("nothing to do — queue empty")
            return 0

        for row in queue:
            done += 1
            try:
                payload = _compute_update_for_row(row)
                if payload is None:
                    skipped += 1
                    if args.dry_run:
                        print(
                            f"[DRY] deviation id={row['id']} "
                            f"hand={row.get('hand_id')} street={row['street']} "
                            f"action={row['action_index']} → SKIP (no hand_data or solver)"
                        )
                    continue

                if not payload["has_solver_data"] and payload["ev_loss"] is None:
                    skipped += 1
                    if args.dry_run:
                        print(
                            f"[DRY] deviation id={row['id']} "
                            f"hand={row.get('hand_id')} street={row['street']} "
                            f"action={row['action_index']} → SKIP (no action_evs)"
                        )
                    continue

                if args.dry_run:
                    ev_disp = (
                        f"{payload['ev_loss']:.2f}bb"
                        if payload["ev_loss"] is not None else "None"
                    )
                    meta_short = ", ".join(
                        f"{k}={v}" for k, v in payload["meta_updates"].items()
                    )
                    print(
                        f"[DRY] deviation id={row['id']} chat={row['chat_id']} "
                        f"hand={row.get('hand_id')} street={row['street']} "
                        f"action={row['action_index']}\n"
                        f"      existing ev_loss=NULL  → {ev_disp}\n"
                        f"      meta updates: {meta_short}"
                    )
                    updated += 1
                else:
                    await _apply_update(
                        conn, row["id"], payload["ev_loss"], payload["meta_updates"],
                    )
                    updated += 1
            except Exception as e:
                errors += 1
                logger.warning(
                    f"row id={row.get('id')} failed: {type(e).__name__}: {e}"
                )
                # Do NOT mark as attempted — let next run retry.

            if done % 50 == 0:
                logger.info(
                    f"backfill: {done}/{total} processed, "
                    f"{updated} updated, {skipped} skipped, {errors} errors"
                )
    except KeyboardInterrupt:
        print("\n^C — aborting", file=sys.stderr)
    finally:
        await conn.close()

    print()
    if args.dry_run:
        print(
            f"DRY-RUN SUMMARY: {done} rows eligible, {updated} would update, "
            f"{skipped} skipped (no solver data), {errors} errors"
        )
    else:
        print(
            f"BACKFILL SUMMARY: {done} rows processed, "
            f"{updated} updated, {skipped} skipped, {errors} errors"
        )
    return 0 if errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run_backfill(args))
    except Exception as e:
        logger.error(f"fatal: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
