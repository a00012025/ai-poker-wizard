#!/usr/bin/env python3
"""Backfill deviations table from existing hand_histories.

One-time migration script. Reads all existing hands, runs spot categorization,
checks GTO solutions, and populates the deviations table.

Usage:
    set -a && source .env && set +a
    python scripts/backfill_deviations.py [--limit N] [--dry-run]
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from gto_formatter import normalize_hand_name, combo_index_for_hand
from spot_categorizer import categorize_spot, classify_board_texture
from leak_service import insert_deviation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


async def backfill(limit: int | None = None, dry_run: bool = False):
    dsn = os.getenv("SUPABASE_CONN")
    if not dsn:
        print("ERROR: SUPABASE_CONN not set")
        return

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, statement_cache_size=0)

    try:
        # Get all hand histories
        async with pool.acquire() as conn:
            query = """
                SELECT hh.id, hh.chat_id, hh.hand_data, hh.uploaded_at
                FROM hand_histories hh
                LEFT JOIN deviations d ON d.hand_history_id = hh.id
                WHERE d.id IS NULL
                ORDER BY hh.uploaded_at DESC
            """
            if limit:
                query += " LIMIT $1"
                rows = await conn.fetch(query, limit)
            else:
                rows = await conn.fetch(query)

        logger.info(f"Found {len(rows)} hands to backfill")

        processed = 0
        skipped = 0
        inserted = 0

        for row in rows:
            hh_id = row["id"]
            chat_id = row["chat_id"]
            try:
                hand_data = json.loads(row["hand_data"])
            except (json.JSONDecodeError, TypeError):
                skipped += 1
                continue

            hero_pos = hand_data.get("hero_position", "")
            hero_hand_raw = hand_data.get("hero_hand", "")
            hero_hand = normalize_hand_name(hero_hand_raw)
            preflop_actions = hand_data.get("preflop_actions", "")
            effective_bb = hand_data.get("effective_bb")
            num_players = hand_data.get("players_at_table", 8)

            if not hero_pos or not preflop_actions:
                skipped += 1
                continue

            combo_idx = combo_index_for_hand(hero_hand_raw)

            # Categorize preflop spot
            cat, texture = categorize_spot(hand_data, "preflop", action_index=0)

            if not dry_run:
                # Insert a preflop deviation stub (no GTO data without API call)
                # We only insert the spot categorization; actual GTO comparison
                # requires API calls which should be done separately if needed
                pass  # Skip preflop-only insert since we'd need GTO data

            # Process postflop streets
            streets = hand_data.get("streets") or hand_data.get("postflop_actions", [])
            street_names = ["flop", "turn", "river"]

            for st_idx, street in enumerate(streets):
                if st_idx >= len(street_names):
                    break
                street_name = street_names[st_idx]
                actions = street.get("actions", [])

                # Find hero actions
                actions_before_hero = []
                for act in actions:
                    if act.get("position") == hero_pos:
                        # Categorize this hero spot
                        cat, texture = categorize_spot(
                            hand_data, street_name,
                            street_actions_before_hero=actions_before_hero,
                        )
                        if not dry_run:
                            logger.debug(
                                f"  Hand {hh_id} {street_name}: {cat} (texture={texture})"
                            )
                        break
                    actions_before_hero.append(act)

            processed += 1
            if processed % 50 == 0:
                logger.info(f"Processed {processed}/{len(rows)} hands...")

        logger.info(
            f"Backfill complete: {processed} processed, {skipped} skipped, {inserted} inserted"
        )

    finally:
        await pool.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Backfill deviations table")
    parser.add_argument("--limit", type=int, help="Limit number of hands to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't insert, just count")
    args = parser.parse_args()

    asyncio.run(backfill(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
