#!/usr/bin/env python3
"""Regrade confidently identified final-table preflop decisions with ICM.

FT detection is deliberately narrow: the first 9-handed hand, or the final
non-increasing player-count tail of a tournament that reaches heads-up.  Raw
postflop grades remain chipEV and are labelled as such.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from gto_api import find_closest_action, get_next_actions, get_spot_solution
from gto_formatter import normalize_hand_name
from hh_deviation_check import (
    _get_action_evs_preflop,
    _get_preflop_hand_freqs,
    _grade_action_choice,
)
from icm_modes import find_icm_params
from ledger_distill import decode_gtow_depth

DEFAULT_MAX_STACK_GAP_BB = 5.0
ICM_RAW = ROOT / "data" / "gtow_raw" / "icm_regrade"
_STREET = {"PREFLOP": "preflop", "FLOP": "flop", "TURN": "turn", "RIVER": "river"}


def detect_ft_windows(rows: list[dict]) -> dict[str, dict]:
    """Return high-confidence FT start timestamps keyed by tournament id."""
    grouped = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        if row.get("tournament_id") and row.get("played_at") is not None:
            grouped[str(row["tournament_id"])].append(row)

    windows = {}
    for tid, hands in grouped.items():
        hands.sort(key=lambda h: h["played_at"])
        nine = next((h for h in hands if int(h.get("total_players") or 0) == 9), None)
        if nine:
            windows[tid] = {"started_at": nine["played_at"], "reason": "nine_handed"}
            continue

        positive = [h for h in hands if 2 <= int(h.get("total_players") or 0) <= 8]
        if not positive or int(positive[-1]["total_players"]) != 2:
            continue
        last_increase = 0
        for i in range(1, len(positive)):
            if int(positive[i]["total_players"]) > int(positive[i - 1]["total_players"]):
                last_increase = i
        tail = positive[last_increase:]
        counts = [int(h["total_players"]) for h in tail]
        if len(set(counts)) >= 3 and all(a >= b for a, b in zip(counts, counts[1:])):
            windows[tid] = {
                "started_at": tail[0]["played_at"],
                "reason": "monotone_tail_to_heads_up",
            }
    return windows


def stack_match_quality(actual: list[float], solver: list[float],
                        max_gap_bb: float = DEFAULT_MAX_STACK_GAP_BB) -> dict:
    if len(actual) != len(solver) or not actual:
        return {"acceptable": False, "max_gap_bb": float("inf"), "rank_inversion": True}
    max_gap = max(abs(float(a) - float(b)) for a, b in zip(actual, solver))
    inversion = any(
        (actual[i] - actual[j]) * (solver[i] - solver[j]) < 0
        for i in range(len(actual)) for j in range(i + 1, len(actual))
    )
    return {
        "acceptable": not inversion and max_gap <= max_gap_bb,
        "max_gap_bb": round(max_gap, 3),
        "rank_inversion": inversion,
    }


def render_summary(counts: dict) -> str:
    headline = (
        "ICM_REGRADING "
        f"tournaments={counts.get('tournaments', 0)} "
        f"hands={counts.get('hands', 0)} "
        f"preflop_regraded={counts.get('preflop_regraded', 0)}"
    )
    return "\n".join([
        headline,
        f"• FT detector：9-handed {counts.get('ft_nine_handed', 0)} 場；"
        f"淘汰尾段 {counts.get('ft_monotone_tail', 0)} 場",
        f"• stack distribution 不夠接近：{counts.get('preflop_unmatched_stack', 0)}",
        f"• archive detail 缺失：{counts.get('preflop_missing_detail', 0)}",
        f"• detail 已抓到但 cache 唯讀：{counts.get('detail_cache_write_failed', 0)}",
        f"• ICM node 無法可靠評分：{counts.get('preflop_ungraded', 0)}",
        f"• ICM provider 暫時失敗：{counts.get('preflop_provider_error', 0)}",
        f"• stack match 合格、等待重評：{counts.get('preflop_ready', 0)}",
        f"• FT postflop 暫用 chipEV 近似：{counts.get('postflop_chipev', 0)}",
    ])


def _with_flag(flags, name: str) -> list[str]:
    if isinstance(flags, str):
        flags = json.loads(flags)
    return list(dict.fromkeys([*(flags or []), name]))


def _flags(flags) -> set[str]:
    if isinstance(flags, str):
        flags = json.loads(flags)
    return set(flags or [])


def _is_pko(name: str | None) -> bool:
    value = (name or "").lower()
    return "bounty" in value or "pko" in value or "knockout" in value


def ensure_cli_credentials(env=os.environ, bootstrap=None) -> bool:
    if env.get("GTOW_USER_ID") or env.get("GTOW_REFRESH_TOKEN"):
        return True
    if bootstrap is None:
        from gto_owner_token import bootstrap_owner_db_token
        bootstrap = bootstrap_owner_db_token
    return bool(bootstrap(verbose=True))


def cache_fetched_detail(detail: dict, path: Path, *, open_gzip=gzip.open) -> bool:
    """Best-effort raw cache; a container-owned volume must not block grading."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open_gzip(path, "wt") as fh:
            json.dump(detail, fh)
        return True
    except OSError:
        return False


async def fetch_missing_details(hand_ids: list[str], fetcher=None) -> dict:
    if not hand_ids:
        return {}
    if fetcher is None:
        from ledger_ingest import _fetch_details_concurrent
        fetcher = _fetch_details_concurrent
    return await fetcher(
        hand_ids,
        on_progress=lambda done, total: print(
            f"  ICM detail sweep: {done}/{total}", flush=True),
    )


def _preflop_points(detail: dict) -> list[dict]:
    return [
        gp for gp in ((detail.get("game_analysis") or {}).get("game_points") or [])
        if _STREET.get((((gp.get("real_game") or {}).get("current_street") or {}).get("type")))
        == "preflop"
    ]


def _stacks_from_point(gp: dict) -> list[float]:
    players = (gp.get("real_game") or {}).get("players") or []
    return [float(player.get("stack") or 0) for player in players]


def _resolve_action(action: dict, available: list[dict]) -> str | None:
    raw = str(action.get("code") or "")
    exact = next((a for a in available if (a.get("action") or {}).get("code") == raw), None)
    if exact:
        return raw
    if raw in {"F", "C", "X"}:
        return raw
    raises = [a for a in available
              if str((a.get("action") or {}).get("code") or "").startswith("R")]
    if raw in {"AI", "RAI"} or action.get("allin"):
        return next(((a.get("action") or {}).get("code") for a in raises
                     if (a.get("action") or {}).get("allin")), None)
    try:
        target = float(action.get("betsize") or raw[1:])
    except (TypeError, ValueError):
        return None
    return find_closest_action(raises, target) if raises else None


def regrade_preflop(detail: dict, hero_pos: str, hero_hand: str, params: dict,
                     *, next_actions=get_next_actions,
                     spot_solution=get_spot_solution) -> list[dict]:
    """Return hero preflop grade updates; provider calls are injectable."""
    prefix: list[str] = []
    hero_idx = 0
    updates = []
    hand_name = normalize_hand_name(hero_hand)
    for gp in _preflop_points(detail):
        action = gp.get("real_game_action") or {}
        actor = action.get("position")
        if not actor:
            continue
        kwargs = {
            "gametype": params["gametype"],
            "depth": params["depth"],
            "stacks": params.get("stacks") or "",
            "preflop_actions": "-".join(prefix),
        }
        if actor == hero_pos:
            solution = spot_solution(**kwargs)
            available = (solution or {}).get("action_solutions") or []
            code = _resolve_action(action, available)
            freqs = _get_preflop_hand_freqs(solution, hand_name, hero_pos) if solution else None
            evs = _get_action_evs_preflop(solution, hand_name, hero_pos) if solution else None
            recommendation, _best_ev, _hero_ev, loss = _grade_action_choice(
                freqs or {}, evs, code or "")
            updates.append({
                "decision_idx": hero_idx,
                "taken_code": "AI" if code == "RAI" else code,
                "best_code": "AI" if recommendation == "RAI" else recommendation,
                "taken_freq": (freqs or {}).get(code or ""),
                "freq_diff": (
                    max(freqs.values()) - (freqs or {}).get(code or "", 0.0)
                    if freqs else None
                ),
                "ev_loss_bb": loss,
                "correctness": (
                    "BEST_MOVE" if code == recommendation
                    else "CORRECT_MOVE" if (freqs or {}).get(code or "", 0.0) >= 0.01
                    else "BLUNDER"
                ) if code and freqs else None,
                "graded": bool(code and freqs and loss is not None),
            })
            hero_idx += 1
        else:
            response = next_actions(**kwargs)
            available = ((response or {}).get("next_actions") or {}).get("available_actions") or []
            code = _resolve_action(action, available)
        if not code:
            break
        prefix.append(code)
    return updates


FT_HANDS_SQL = """
SELECT gtow_hand_id,tournament_id,tournament_name,played_at,total_players,
       position,hero_hand,raw_path
FROM ledger_hands WHERE source='online' AND tournament_id IS NOT NULL
ORDER BY tournament_id,played_at
"""

async def _mark_ungraded(conn, hand_id: str, flag: str) -> int:
    rows = await conn.fetch(
        "SELECT decision_idx,approx_flags FROM ledger_decisions "
        "WHERE gtow_hand_id=$1 AND street='preflop'", hand_id)
    for row in rows:
        await conn.execute(
            "UPDATE ledger_decisions SET strategy_context='icm',excluded=true,"
            "confidence=0,approx_flags=$3 "
            "WHERE gtow_hand_id=$1 AND street='preflop' AND decision_idx=$2",
            hand_id, row["decision_idx"], json.dumps(_with_flag(row["approx_flags"], flag)))
    return len(rows)


async def run(conn, *, max_stack_gap_bb: float = DEFAULT_MAX_STACK_GAP_BB,
              dry_run: bool = False, scan_only: bool = False,
              verbose: bool = False, fetch_missing: bool = False) -> dict:
    hands = [dict(row) for row in await conn.fetch(FT_HANDS_SQL)]
    windows = detect_ft_windows(hands)
    ft_hands = [h for h in hands if str(h["tournament_id"]) in windows
                and h["played_at"] >= windows[str(h["tournament_id"])]["started_at"]]
    counts = defaultdict(int, tournaments=len(windows), hands=len(ft_hands))
    counts["ft_nine_handed"] = sum(
        window["reason"] == "nine_handed" for window in windows.values())
    counts["ft_monotone_tail"] = sum(
        window["reason"] == "monotone_tail_to_heads_up" for window in windows.values())
    decision_rows = await conn.fetch(
        "SELECT d.gtow_hand_id,d.street,d.decision_idx,d.approx_flags "
        "FROM ledger_decisions d JOIN ledger_hands h USING(gtow_hand_id) "
        "WHERE d.source='online' AND h.tournament_id = ANY($1::text[])",
        list(windows),
    ) if windows else []
    decisions_by_hand = defaultdict(list)
    for row in decision_rows:
        decisions_by_hand[row["gtow_hand_id"]].append(dict(row))

    final_flags = {"archive_icm_regraded", "icm_regrade_unmatched_stack",
                   "icm_regrade_ungraded"}
    missing_ids = []
    if fetch_missing and not scan_only:
        for hand in ft_hands:
            preflop = [d for d in decisions_by_hand[hand["gtow_hand_id"]]
                       if d["street"] == "preflop"]
            raw_path = ROOT / hand["raw_path"] if hand.get("raw_path") else None
            if (preflop
                    and not all(_flags(d["approx_flags"]) & final_flags for d in preflop)
                    and (not raw_path or not raw_path.exists())):
                missing_ids.append(hand["gtow_hand_id"])
    fetched_details = await fetch_missing_details(missing_ids) if missing_ids else {}

    for hand in ft_hands:
        decisions = decisions_by_hand[hand["gtow_hand_id"]]
        postflop = [d for d in decisions if d["street"] != "preflop"]
        postflop_pending = [d for d in postflop
                            if "ft_postflop_chipev_approx" not in _flags(d["approx_flags"])]
        counts["postflop_chipev"] += len(postflop_pending)
        if not dry_run:
            for dec in postflop_pending:
                await conn.execute(
                    "UPDATE ledger_decisions SET strategy_context='icm_postflop_chipev',"
                    "approx_flags=$4 "
                    "WHERE gtow_hand_id=$1 AND street=$2 AND decision_idx=$3",
                    hand["gtow_hand_id"], dec["street"], dec["decision_idx"],
                    json.dumps(_with_flag(dec["approx_flags"], "ft_postflop_chipev_approx")))

        preflop = [d for d in decisions if d["street"] == "preflop"]
        if not preflop:
            continue
        if all(_flags(d["approx_flags"]) & final_flags for d in preflop):
            continue
        raw_path = ROOT / hand["raw_path"] if hand.get("raw_path") else None
        detail = fetched_details.get(hand["gtow_hand_id"])
        if not isinstance(detail, dict):
            detail = None
        if detail and not dry_run:
            raw_path = ICM_RAW / f"{hand['gtow_hand_id']}.json.gz"
            if cache_fetched_detail(detail, raw_path):
                rel = str(raw_path.relative_to(ROOT))
                await conn.execute(
                    "UPDATE ledger_hands SET raw_path=$2,detail_status='fetched' "
                    "WHERE gtow_hand_id=$1", hand["gtow_hand_id"], rel)
            else:
                counts["detail_cache_write_failed"] += 1
        if detail is None and (not raw_path or not raw_path.exists()):
            counts["preflop_missing_detail"] += len(preflop)
            if not dry_run:
                await _mark_ungraded(conn, hand["gtow_hand_id"], "icm_regrade_missing_detail")
            continue
        if detail is None:
            with gzip.open(raw_path, "rt") as fh:
                detail = json.load(fh)
        points = _preflop_points(detail)
        if not points:
            counts["preflop_ungraded"] += len(preflop)
            if not dry_run:
                await _mark_ungraded(conn, hand["gtow_hand_id"], "icm_regrade_no_preflop_points")
            continue
        actual_stacks = _stacks_from_point(points[0])
        players = len(actual_stacks)
        params = find_icm_params(
            actual_stacks, pko=_is_pko(hand.get("tournament_name")),
            tournament_size=1000, players_remaining=players,
            phase="FT", players_at_table=players)
        quality = stack_match_quality(actual_stacks, params.get("solver_stacks") or [],
                                      max_stack_gap_bb)
        if not quality["acceptable"]:
            counts["preflop_unmatched_stack"] += len(preflop)
            if verbose:
                print(
                    f"UNMATCHED {hand['gtow_hand_id']} actual={actual_stacks} "
                    f"solver={params.get('solver_stacks')} max_gap={quality['max_gap_bb']} "
                    f"rank_inversion={quality['rank_inversion']}")
            if not dry_run:
                await _mark_ungraded(conn, hand["gtow_hand_id"], "icm_regrade_unmatched_stack")
            continue
        if scan_only:
            counts["preflop_ready"] += len(preflop)
            continue
        try:
            grades = regrade_preflop(
                detail, hand["position"], hand["hero_hand"], params)
        except Exception as exc:
            counts["preflop_provider_error"] += len(preflop)
            print(f"PROVIDER_ERROR {hand['gtow_hand_id']} {exc}")
            if not dry_run:
                await _mark_ungraded(conn, hand["gtow_hand_id"], "icm_regrade_provider_error")
            continue
        by_idx = {grade["decision_idx"]: grade for grade in grades}
        for dec in preflop:
            grade = by_idx.get(dec["decision_idx"])
            if not grade or not grade["graded"]:
                counts["preflop_ungraded"] += 1
                if not dry_run:
                    await conn.execute(
                        "UPDATE ledger_decisions SET strategy_context='icm',excluded=true,"
                        "confidence=0,approx_flags=$3 "
                        "WHERE gtow_hand_id=$1 AND street='preflop' AND decision_idx=$2",
                        hand["gtow_hand_id"], dec["decision_idx"],
                        json.dumps(_with_flag(dec["approx_flags"], "icm_regrade_ungraded")))
                continue
            counts["preflop_regraded"] += 1
            if not dry_run:
                flags = [f for f in _with_flag(dec["approx_flags"], "archive_icm_regraded")
                         if f != "chipev_grading"]
                flags.append(f"icm_stack_max_gap:{quality['max_gap_bb']:g}")
                await conn.execute(
                    "UPDATE ledger_decisions SET grader='archive_icm',strategy_context='icm',gametype=$3,"
                    "solver_depth_bb=$4,taken_code=$5,best_code=$6,ev_loss_bb=$7,"
                    "taken_freq=$8,freq_diff=$9,correctness=$10,ev_loss_pct_pot=NULL,"
                    "gto_score=NULL,confidence=1,excluded=false,approx_flags=$11 "
                    "WHERE gtow_hand_id=$1 AND street='preflop' AND decision_idx=$2",
                    hand["gtow_hand_id"], dec["decision_idx"],
                    params["gametype"], decode_gtow_depth(params["depth"]),
                    grade["taken_code"], grade["best_code"], grade["ev_loss_bb"],
                    grade["taken_freq"], grade["freq_diff"], grade["correctness"],
                    json.dumps(flags))

    if not dry_run:
        await conn.execute(
            "UPDATE ledger_hands SET total_ev_loss_bb=NULL,total_ev_loss_pct_pot=NULL,"
            "avg_gto_score=NULL,hand_correctness=NULL "
            "WHERE tournament_id = ANY($1::text[])", list(windows))
    return dict(counts)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scan-only", action="store_true",
                        help="detect FT/archive/stack coverage without solver calls or writes")
    parser.add_argument("--max-stack-gap-bb", type=float, default=DEFAULT_MAX_STACK_GAP_BB)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fetch-missing", action="store_true",
                        help="fetch missing FT details from GTOW before regrading")
    args = parser.parse_args()
    if not args.scan_only and not ensure_cli_credentials():
        print("ICM regrade requires a synchronized owner GTOW DB session")
        return 2
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        if args.dry_run or args.scan_only:
            counts = await run(conn, max_stack_gap_bb=args.max_stack_gap_bb,
                               dry_run=True, scan_only=args.scan_only,
                               verbose=args.verbose, fetch_missing=args.fetch_missing)
        else:
            counts = await run(conn, max_stack_gap_bb=args.max_stack_gap_bb,
                               verbose=args.verbose, fetch_missing=args.fetch_missing)
        print(render_summary(counts))
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
