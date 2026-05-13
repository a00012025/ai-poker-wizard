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

    spot_changes: list[tuple[int, str, str]] = []   # (dev_id, old, new)
    tex_changes:  list[tuple[int, str | None, str | None]] = []
    spot_transitions: Counter = Counter()
    tex_transitions:  Counter = Counter()
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
        spot_dirty = bool(new_cat) and new_cat != old_cat
        if spot_dirty:
            spot_changes.append((r["id"], old_cat, new_cat))
            spot_transitions[(old_cat, new_cat)] += 1

        # Recompute board_texture (independent of spot_category).
        new_tex = classify_board_texture(_board_for_street(hd, r["street"]))
        old_tex = r["board_texture"]
        tex_dirty = (new_tex != old_tex)
        if tex_dirty:
            tex_changes.append((r["id"], old_tex, new_tex))
            tex_transitions[(old_tex, new_tex)] += 1

        if not spot_dirty and not tex_dirty:
            unchanged += 1

    if not dry_run:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for dev_id, _old, new_cat in spot_changes:
                    await conn.execute(
                        "UPDATE deviations SET spot_category = $1 WHERE id = $2",
                        new_cat, dev_id,
                    )
                for dev_id, _old, new_tex in tex_changes:
                    await conn.execute(
                        "UPDATE deviations SET board_texture = $1 WHERE id = $2",
                        new_tex, dev_id,
                    )

    return {
        "scanned":            len(rows),
        "spot_changed":       len(spot_changes),
        "tex_changed":        len(tex_changes),
        "unchanged":          unchanged,
        "skipped_no_street":  skipped_no_street,
        "spot_transitions":   spot_transitions,
        "tex_transitions":    tex_transitions,
        "spot_samples":       spot_changes[:5],
        "tex_samples":        tex_changes[:5],
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
    verb = "would_change" if args.dry_run else "changed"
    print(f"{verb} spot_category:  {result['spot_changed']}")
    print(f"{verb} board_texture:  {result['tex_changed']}")
    print(f"unchanged:         {result['unchanged']}")
    print(f"skipped_no_street: {result['skipped_no_street']}")
    print("\ntop spot transitions:")
    for (old, new), n in result["spot_transitions"].most_common(20):
        print(f"  {old or '-':>16} → {new:<16} {n}")
    print("\ntop texture transitions:")
    for (old, new), n in result["tex_transitions"].most_common(20):
        print(f"  {old or '-':>10} → {(new or '-'):<10} {n}")
    if result["spot_samples"]:
        print("\nsample spot changes (dev_id, old → new):")
        for s in result["spot_samples"]:
            print(f"  {s[0]}: {s[1]} → {s[2]}")


if __name__ == "__main__":
    asyncio.run(main())
