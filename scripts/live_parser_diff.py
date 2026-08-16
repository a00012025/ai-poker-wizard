#!/usr/bin/env python3
"""Replay every stored live raw hand and produce a human golden-set review.

Read-only: this script never updates ledger tables. It compares the currently
stored ``ledger_hands.parsed_json`` with the production live tokenizer/replay
pipeline and writes every disagreement as an auditable text report.

Usage:
  python scripts/live_parser_diff.py \
      --out /tmp/live-parser-review.txt \
      --json-out /tmp/live-parser-review.json
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from live_flow import parse_block  # noqa: E402


def normalized_hand(hand: dict | None) -> dict | None:
    """Canonical comparison shape, including exact sizes and actor order."""
    if not hand:
        return None
    streets = []
    for street in hand.get("streets") or []:
        actions = []
        for action in street.get("actions") or []:
            row = {
                "position": action.get("position"),
                "action": action.get("action"),
            }
            if action.get("size") is not None:
                row["size"] = round(float(action["size"]), 3)
            if action.get("pot_fraction") is not None:
                row["pot_fraction"] = round(
                    float(action["pot_fraction"]), 6)
            actions.append(row)
        streets.append({
            "street": street.get("street"),
            "board": street.get("board"),
            "card": street.get("card"),
            "actions": actions,
        })
    return {
        "players_at_table": hand.get("players_at_table") or 8,
        "effective_bb": (
            round(float(hand["effective_bb"]), 3)
            if hand.get("effective_bb") is not None else None
        ),
        "hero_position": hand.get("hero_position"),
        "hero_hand": hand.get("hero_hand"),
        "preflop_actions": hand.get("preflop_actions"),
        "streets": streets,
    }


def parse_stored(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def field_summary(old: dict | None, new: dict | None) -> list[str]:
    if new is None:
        return ["new_parse_failed"]
    if new.get("_refused"):
        return ["new_parse_refused: " + "; ".join(new["_refused"])]
    old_n = normalized_hand(old)
    new_n = normalized_hand(new)
    if old_n == new_n:
        return []
    changed = []
    for key in (
        "players_at_table", "effective_bb", "hero_position", "hero_hand",
        "preflop_actions", "streets",
    ):
        if (old_n or {}).get(key) != (new_n or {}).get(key):
            changed.append(key)
    return changed


async def fetch_live_hands(limit: int = 0) -> list[dict]:
    import asyncpg

    conn = await asyncpg.connect(
        os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        sql = """
            SELECT gtow_hand_id, raw_text, parsed_json
            FROM ledger_hands
            WHERE source = 'live' AND raw_text IS NOT NULL
            ORDER BY played_at, gtow_hand_id
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [dict(row) for row in await conn.fetch(sql)]
    finally:
        await conn.close()


def replay_rows(rows: list[dict], model: str, workers: int) -> list[dict]:
    def one(row: dict) -> dict:
        try:
            new = parse_block(row["raw_text"], model=model)
            error = None
        except Exception as exc:  # report failures; never abort the corpus
            new = None
            error = f"{type(exc).__name__}: {exc}"
        old = parse_stored(row.get("parsed_json"))
        changed = field_summary(old, new)
        if error:
            changed = [f"new_parse_error: {error}"]
        return {
            "gtow_hand_id": row["gtow_hand_id"], "raw_text": row["raw_text"],
            "changed_fields": changed, "old": old, "new": new,
        }

    if workers <= 1:
        return [one(row) for row in rows]
    out = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in rows}
        for future in as_completed(futures):
            out.append(future.result())
    by_id = {row["gtow_hand_id"]: i for i, row in enumerate(rows)}
    return sorted(out, key=lambda row: by_id[row["gtow_hand_id"]])


def _pretty(hand: dict | None) -> str:
    return json.dumps(normalized_hand(hand), ensure_ascii=False, indent=2,
                      sort_keys=True)


def render_report(results: list[dict], model: str) -> str:
    changed = [row for row in results if row["changed_fields"]]
    refused = sum(
        any(field.startswith(("new_parse_refused", "new_parse_failed",
                              "new_parse_error"))
            for field in row["changed_fields"])
        for row in changed
    )
    lines = [
        "LIVE PARSER GOLDEN-SET REVIEW",
        "=" * 80,
        f"Model: {model}",
        f"Rows replayed: {len(results)}",
        f"Unchanged: {len(results) - len(changed)}",
        f"Changed/refused: {len(changed)}",
        f"New parser refused/failed: {refused}",
        "",
        "Review instructions:",
        "  [ ] OLD correct   [ ] NEW correct   [ ] BOTH wrong",
        "  Fill GOLD_JSON when BOTH wrong, then convert accepted cases into",
        "  regression fixtures. This report is read-only; DB was not modified.",
        "",
    ]
    for number, row in enumerate(changed, 1):
        old_text = _pretty(row["old"])
        new_hand = row["new"] if not (row["new"] or {}).get("_refused") else None
        new_text = _pretty(new_hand)
        diff = "\n".join(difflib.unified_diff(
            old_text.splitlines(), new_text.splitlines(),
            fromfile="OLD_DB", tofile="NEW_REPLAY", lineterm=""))
        lines.extend([
            f"#{number}  {row['gtow_hand_id']}",
            "-" * 80,
            "CHANGED_FIELDS: " + ", ".join(row["changed_fields"]),
            "RAW:",
            row["raw_text"],
            "",
            "DIFF:",
            diff or "(parse refused; see CHANGED_FIELDS)",
            "",
            "OLD_JSON:",
            old_text,
            "",
            "NEW_JSON:",
            new_text,
            "",
            "TOKEN_TRACE:",
            json.dumps((row["new"] or {}).get("_parse_trace") or [],
                       ensure_ascii=False),
            "PARSE_FLAGS:",
            json.dumps((row["new"] or {}).get("_parse_flags") or [],
                       ensure_ascii=False),
            "",
            "VERDICT: [ ] OLD correct   [ ] NEW correct   [ ] BOTH wrong",
            "NOTES:",
            "GOLD_JSON:",
            "",
            "=" * 80,
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--json-out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_LIVE_PARSE_MODEL", "gemini-3.6-flash"))
    args = parser.parse_args()

    rows = asyncio.run(fetch_live_hands(args.limit))
    results = replay_rows(rows, args.model, args.workers)
    Path(args.out).write_text(
        render_report(results, args.model), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
    changed = sum(bool(row["changed_fields"]) for row in results)
    print(f"replayed={len(results)} changed_or_refused={changed} out={args.out}")


if __name__ == "__main__":
    main()
