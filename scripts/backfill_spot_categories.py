#!/usr/bin/env python3
"""Recategorize deviations.spot_category for hands where hero_spots was
built without `street_actions_before_hero`.

Bug: before this fix, hero_spots didn't carry the street-action history, so
the categorizer treated every postflop decision by the PF aggressor as a
clean c-bet (`cbet_ip`/`cbet_oop`). Hands where hero actually faced a donk,
probe, or check-raise were mis-clustered into c-bet buckets.

Backfill: reconstruct `street_actions_before_hero` from
`hand_histories.hand_data` and re-run `categorize_spot` for each
deviation. Write the corrected spot_category back if it changed.

Usage:
    python scripts/backfill_spot_categories.py --dry-run            # preview
    python scripts/backfill_spot_categories.py --dry-run --chat 5..  # per-user
    python scripts/backfill_spot_categories.py                       # apply
    python scripts/backfill_spot_categories.py --since 30            # last N days
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncpg
from spot_categorizer import categorize_spot, classify_board_texture


_STREET_TO_IDX = {"flop": 0, "turn": 1, "river": 2}


def _reconstruct_before_hero(hand_data: dict, street_name: str,
                             hero_action_idx: int) -> list[dict] | None:
    """Return the actions on `street_name` that occurred BEFORE hero's
    (hero_action_idx+1)-th action on that street. Returns None if the
    street isn't in the stored data."""
    streets = hand_data.get("streets") or []
    target_idx = _STREET_TO_IDX.get(street_name, -1)
    if target_idx < 0 or target_idx >= len(streets):
        return None
    actions = streets[target_idx].get("actions") or []
    hero_pos = hand_data.get("hero_position", "")
    hero_seen = 0
    before: list[dict] = []
    for a in actions:
        if a.get("position") == hero_pos:
            if hero_seen == hero_action_idx:
                return before
            hero_seen += 1
        before.append(a)
    # Fewer hero actions than expected → return everything collected.
    return before


def _board_for_street(hand_data: dict, street_name: str) -> str | None:
    streets = hand_data.get("streets") or []
    target_idx = _STREET_TO_IDX.get(street_name, -1)
    if target_idx < 0 or target_idx >= len(streets):
        return None
    board = streets[0].get("board") or ""
    for i in range(1, target_idx + 1):
        board += streets[i].get("card") or ""
    return board or None


async def backfill(pool: asyncpg.Pool, *, chat_id: int | None,
                   since_days: int | None, dry_run: bool) -> dict:
    where = ["d.street != 'preflop'"]
    args: list = []
    if chat_id is not None:
        args.append(chat_id)
        where.append(f"d.chat_id = ${len(args)}")
    if since_days is not None:
        args.append(str(since_days))
        where.append(f"d.created_at >= NOW() - (${len(args)} || ' days')::interval")
    sql = f"""
        SELECT d.id, d.chat_id, d.hand_history_id, d.street, d.action_index,
               d.spot_category, d.board_texture, h.hand_data
        FROM deviations d
        JOIN hand_histories h ON h.id = d.hand_history_id
        WHERE {' AND '.join(where)}
        ORDER BY d.id
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)

    changes: list[tuple[int, str, str]] = []  # (deviation_id, old, new)
    transitions: Counter = Counter()
    skipped_no_street = 0
    unchanged = 0

    for r in rows:
        hd = r["hand_data"]
        if isinstance(hd, str):
            try:
                hd = json.loads(hd)
            except Exception:
                continue
        if not hd:
            continue

        before = _reconstruct_before_hero(hd, r["street"], r["action_index"] or 0)
        if before is None:
            skipped_no_street += 1
            continue

        try:
            new_cat, _ = categorize_spot(
                hd, street=r["street"], action_index=0,
                street_actions_before_hero=before,
            )
        except Exception:
            continue

        old_cat = r["spot_category"]
        if new_cat and new_cat != old_cat:
            changes.append((r["id"], old_cat, new_cat))
            transitions[(old_cat, new_cat)] += 1
        else:
            unchanged += 1

    if not dry_run and changes:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for dev_id, _old, new_cat in changes:
                    await conn.execute(
                        "UPDATE deviations SET spot_category = $1 WHERE id = $2",
                        new_cat, dev_id,
                    )

    return {
        "scanned":           len(rows),
        "changed":           len(changes),
        "unchanged":         unchanged,
        "skipped_no_street": skipped_no_street,
        "transitions":       transitions,
        "samples":           changes[:5],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chat", type=int, default=None,
                   help="Limit to a single chat_id")
    p.add_argument("--since", type=int, default=None,
                   help="Limit to deviations from the last N days")
    args = p.parse_args()

    dsn = os.environ["SUPABASE_CONN"]
    pool = await asyncpg.create_pool(dsn, statement_cache_size=0)
    try:
        result = await backfill(pool, chat_id=args.chat,
                                since_days=args.since, dry_run=args.dry_run)
    finally:
        await pool.close()

    print(f"scanned:           {result['scanned']}")
    print(f"would_change:      {result['changed']}" if args.dry_run
          else f"changed:           {result['changed']}")
    print(f"unchanged:         {result['unchanged']}")
    print(f"skipped_no_street: {result['skipped_no_street']}")
    print("\ntop transitions:")
    for (old, new), n in result["transitions"].most_common(20):
        print(f"  {old or '-':>16} → {new:<16} {n}")
    if result["samples"]:
        print("\nsample changes (dev_id, old → new):")
        for s in result["samples"]:
            print(f"  {s[0]}: {s[1]} → {s[2]}")


if __name__ == "__main__":
    asyncio.run(main())
