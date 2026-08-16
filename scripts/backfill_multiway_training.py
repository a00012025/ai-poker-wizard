#!/usr/bin/env python3
"""Re-align live multiway ledger/queue rows with their graded HU projection.

The live grader solves postflop multiway hands through a deterministic HU
projection.  Older rows classified the original multiway action history
instead, so their learning leaf could contain discarded-player actions and
their GTOW Trainer link could not be rebuilt.  This backfill is idempotent and
updates every persisted ``multiway_recast`` decision plus every drill queue row
that references one of those decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from gto_owner_token import bootstrap_owner_db_token
from live_flow import training_hand_for_postflop
from queue_feed import (
    _as_list,
    _source_decisions,
    depths_for_scope,
    normalize_source_entries,
    queue_drill_url_from_sources,
)
from spot_naming import compact_spot_name, drill_depth_scope
from spot_taxonomy import walk_spots_from_parsed

_HANDS_SQL = """
SELECT DISTINCT h.gtow_hand_id, h.parsed_json
FROM ledger_hands h
JOIN ledger_decisions d USING (gtow_hand_id)
WHERE h.source='live' AND d.street <> 'preflop'
  AND d.approx_flags::text LIKE '%multiway_recast%'
ORDER BY h.gtow_hand_id
"""

_DECISIONS_SQL = """
SELECT street, decision_idx, spot_leaf
FROM ledger_decisions
WHERE gtow_hand_id=$1 AND street <> 'preflop'
  AND approx_flags::text LIKE '%multiway_recast%'
ORDER BY street, decision_idx
"""

_UPDATE_DECISION_SQL = """
UPDATE ledger_decisions SET
  spot_category=$4, spot_leaf=$5, spot_parent=$6, spot_keys=$7::jsonb,
  hero_cat=$8, villain_cat=$9, ip_oop=$10, facing=$11, pot_type=$12,
  flop_seq=$13, turn_seq=$14, eff_stack=$15, depth_band=$16,
  board_suit=$17, board_conn=$18, board_paired=$19,
  discarded=$20, limp_origin=$21
WHERE gtow_hand_id=$1 AND street=$2 AND decision_idx=$3
"""

_QUEUE_SQL = """
SELECT id, status, source_hands, spot_leaf, spot_category, label, drill_url,
       depth_scope
FROM drill_queue
WHERE kind='drill' AND source_hands::text LIKE '%live:%'
ORDER BY id
"""


def _parsed(value) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("live ledger hand has no parsed_json object")
    return value


def _spot_values(hand_id: str, spot: dict) -> tuple:
    tags = spot.get("tags") or {}
    return (
        hand_id,
        spot["street"],
        int(spot["decision_idx"]),
        spot["category"],
        spot["leaf"],
        spot.get("parent"),
        json.dumps(spot["keys"]),
        spot.get("hero_cat"),
        spot.get("villain_cat"),
        spot.get("ip_oop"),
        spot.get("facing"),
        spot.get("pot_type"),
        spot.get("flop_seq"),
        spot.get("turn_seq"),
        tags.get("eff_stack"),
        tags.get("depth_band"),
        tags.get("board_suit"),
        tags.get("board_conn"),
        tags.get("board_paired"),
        bool(spot.get("discarded")),
        bool(spot.get("limp_origin")),
    )


async def backfill(conn, *, dry_run: bool = True) -> dict:
    tally = {
        "hands": 0,
        "decisions_checked": 0,
        "decisions_updated": 0,
        "queue_checked": 0,
        "queue_updated": 0,
        "unresolved": [],
    }
    affected: set[tuple[str, str, int]] = set()
    tx = conn.transaction()
    await tx.start()
    try:
        for row in await conn.fetch(_HANDS_SQL):
            hand_id = row["gtow_hand_id"]
            tally["hands"] += 1
            try:
                projected = training_hand_for_postflop(_parsed(row["parsed_json"]))
            except Exception as exc:
                tally["unresolved"].append(f"{hand_id}: {exc}")
                continue
            projected_spots = {
                (spot["street"], int(spot["decision_idx"])): spot
                for spot in walk_spots_from_parsed(projected)
                if spot["street"] != "preflop"
            }
            for old in await conn.fetch(_DECISIONS_SQL, hand_id):
                key = (old["street"], int(old["decision_idx"]))
                tally["decisions_checked"] += 1
                spot = projected_spots.get(key)
                if spot is None:
                    tally["unresolved"].append(
                        f"{hand_id} {key[0]}[{key[1]}]: projection node missing"
                    )
                    continue
                affected.add((hand_id, key[0], key[1]))
                await conn.execute(_UPDATE_DECISION_SQL, *_spot_values(hand_id, spot))
                if old["spot_leaf"] != spot["leaf"]:
                    tally["decisions_updated"] += 1

        for raw_queue in await conn.fetch(_QUEUE_SQL):
            queue = dict(raw_queue)
            entries = _as_list(queue["source_hands"])
            if not any(
                (
                    entry.get("hand_id"),
                    entry.get("street"),
                    int(entry.get("decision_idx") or 0),
                )
                in affected
                for entry in entries
            ):
                continue
            tally["queue_checked"] += 1
            normalized = await normalize_source_entries(conn, entries)
            decisions = await _source_decisions(conn, normalized)
            if len(decisions) != len(normalized):
                tally["unresolved"].append(
                    f"queue {queue['id']}: {len(decisions)}/{len(normalized)} sources resolved"
                )
                continue
            identities = {
                (decision.get("spot_leaf"), decision.get("spot_category"))
                for decision in decisions
            }
            if len(identities) != 1:
                tally["unresolved"].append(
                    f"queue {queue['id']}: sources split across {sorted(identities)}"
                )
                continue
            representative = decisions[-1]
            rebuilt = await queue_drill_url_from_sources(
                conn,
                normalized,
                depths=depths_for_scope(queue["depth_scope"]),
            )
            scope = drill_depth_scope(
                {
                    "drill_url": rebuilt,
                    "eff_stack": representative.get("eff_stack"),
                }
            )
            leaf = representative["spot_leaf"]
            category = representative["spot_category"]
            label = compact_spot_name(
                {
                    **representative,
                    "hero_pos": representative.get("position"),
                    "drill_url": rebuilt,
                    "depth_scope": scope,
                }
            )
            collision = await conn.fetchval(
                "SELECT id FROM drill_queue WHERE id<>$1 AND kind='drill' "
                "AND spot_leaf=$2 AND depth_scope=$3 "
                "AND status IN ('pending','prescribed') LIMIT 1",
                queue["id"],
                leaf,
                scope,
            )
            if collision and queue["status"] in {"pending", "prescribed"}:
                tally["unresolved"].append(
                    f"queue {queue['id']}: identity collides with queue {collision}"
                )
                continue
            current = (
                queue["spot_leaf"],
                queue["spot_category"],
                queue["label"],
                queue["drill_url"],
                queue["depth_scope"],
                entries,
            )
            rebuilt_values = (leaf, category, label, rebuilt, scope, normalized)
            if current != rebuilt_values:
                await conn.execute(
                    "UPDATE drill_queue SET spot_leaf=$2, spot_category=$3, "
                    "label=$4, drill_url=$5, depth_scope=$6, "
                    "source_hands=$7::jsonb, gtow_settings_hash=NULL, "
                    "gtow_drill_synced_at=NULL, gtow_training_started_at=NULL, "
                    "gtow_baseline_totals=NULL WHERE id=$1",
                    queue["id"],
                    leaf,
                    category,
                    label,
                    rebuilt,
                    scope,
                    json.dumps(normalized),
                )
                tally["queue_updated"] += 1

        if tally["unresolved"]:
            await tx.rollback()
        elif dry_run:
            await tx.rollback()
        else:
            await tx.commit()
    except Exception:
        await tx.rollback()
        raise
    return tally


async def _run(dry_run: bool) -> int:
    bootstrap_owner_db_token()
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        tally = await backfill(conn, dry_run=dry_run)
    finally:
        await conn.close()
    print(json.dumps(tally, ensure_ascii=False, indent=2))
    if tally["unresolved"]:
        print("No changes committed because unresolved rows remain.", file=sys.stderr)
        return 2
    print("DRY-RUN (rolled back)" if dry_run else "COMMITTED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="commit the full ledger + queue repair (default is rollback dry-run)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(dry_run=not args.apply)))


if __name__ == "__main__":
    main()
