# scripts/ocr/classifier/extract_crops.py
"""Pull snapshots from Supabase, run Phase-0 localization, write labeled
crops to data/cards/{rank}/{suit}/{hand_id}_{source}_{slot}.png.

Label priority: expected_json overrides parsed_json. Rows where crop count
!= label count are logged to data/extract_crops.skipped.log and skipped —
we never invent labels to fill mismatches.

Usage:
    python -m scripts.ocr.classifier.extract_crops
    python -m scripts.ocr.classifier.extract_crops --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocr.region_detector import detect_regions  # noqa: E402
from ocr.table_parser import _locate_hero_cards, _locate_board_cards  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "cards"
SKIP_LOG = REPO_ROOT / "data" / "extract_crops.skipped.log"
_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")


def _parse_hand_labels(parsed: dict, expected: dict | None) -> tuple[list[str], list[str]]:
    """Return (hero_cards, board_cards) both as ['Xy', ...] in order."""
    src = dict(parsed or {})
    if expected:
        for k, v in expected.items():
            if v is not None:
                src[k] = v

    hero_raw = src.get("hero_hand") or ""
    hero: list[str] = []
    if len(hero_raw) >= 2:
        for i in range(0, len(hero_raw) - 1, 2):
            pair = hero_raw[i:i + 2]
            if _CARD_RE.match(pair):
                hero.append(pair)

    board: list[str] = []
    for street in src.get("streets", []) or []:
        b = street.get("board")
        if b:
            for i in range(0, len(b) - 1, 2):
                pair = b[i:i + 2]
                if _CARD_RE.match(pair):
                    board.append(pair)
        c = street.get("card")
        if c and _CARD_RE.match(c):
            board.append(c)
    return hero, board


def _decode_table_region(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    regions = detect_regions(img)
    if regions is None:
        return None
    return regions["table"]


def _save_crop(crop: np.ndarray, rank: str, suit: str, hand_id: str,
               source: str, slot: int) -> Path:
    out_dir = OUT_ROOT / rank / suit
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hand_id}_{source}_{slot}.png"
    cv2.imwrite(str(out_path), crop)
    return out_path


def _skip(hand_id: str, reason: str):
    with SKIP_LOG.open("a") as f:
        f.write(f"{hand_id}\t{reason}\n")


async def main(limit: int | None):
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        q = ("SELECT hand_id, image_data, parsed_json, expected_json "
             "FROM analysis_snapshots "
             "WHERE image_data IS NOT NULL AND parsed_json IS NOT NULL "
             "ORDER BY hand_id")
        rows = await conn.fetch(q + (f" LIMIT {int(limit)}" if limit else ""))
    finally:
        await conn.close()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.write_text("")

    total_saved = 0
    total_skipped_rows = 0

    for r in rows:
        hand_id = r["hand_id"]
        parsed = r["parsed_json"]
        expected = r["expected_json"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(expected, str):
            expected = json.loads(expected)

        table_region = _decode_table_region(bytes(r["image_data"]))
        if table_region is None:
            _skip(hand_id, "image_decode_or_region_detect_failed")
            total_skipped_rows += 1
            continue

        hero_labels, board_labels = _parse_hand_labels(parsed, expected)
        hero_crops = _locate_hero_cards(table_region)
        board_crops = _locate_board_cards(table_region)

        if hero_labels:
            if len(hero_crops) == len(hero_labels):
                for i, (crop, lbl) in enumerate(zip(hero_crops, hero_labels)):
                    m = _CARD_RE.match(lbl)
                    _save_crop(crop, m.group(1), m.group(2), hand_id, "hero", i)
                    total_saved += 1
            else:
                _skip(hand_id,
                      f"hero_mismatch crops={len(hero_crops)} labels={len(hero_labels)}")

        if board_labels:
            if len(board_crops) == len(board_labels):
                for i, (crop, lbl) in enumerate(zip(board_crops, board_labels)):
                    m = _CARD_RE.match(lbl)
                    _save_crop(crop, m.group(1), m.group(2), hand_id, "board", i)
                    total_saved += 1
            else:
                _skip(hand_id,
                      f"board_mismatch crops={len(board_crops)} labels={len(board_labels)}")

    print(f"saved {total_saved} crops")
    print(f"skipped rows (any mismatch logged): see {SKIP_LOG}")
    print(f"total rows processed: {len(rows)}, decode failures: {total_skipped_rows}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
