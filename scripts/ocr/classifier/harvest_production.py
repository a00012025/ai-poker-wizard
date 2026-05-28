"""Extract labeled card crops from analysis_snapshots for the
production_v1 corpus. Labels come from expected_json.hero_hand
(user-verified), not parsed_json (the previous OCR guess)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..region_detector import detect_regions
from ..table_parser import _locate_hero_cards, _trim_above_card_edge


def _parse_hand_into_two(hand: str | None) -> list[str] | None:
    if not hand or len(hand) != 4:
        return None
    return [hand[0:2], hand[2:4]]


def harvest_snapshot(*, hand_id: str, image_bytes: bytes,
                     expected: dict, out_dir: Path) -> int:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    regions = detect_regions(img)
    if not regions:
        return 0
    table = regions.get("table")
    if table is None:
        return 0
    hero_cards = _parse_hand_into_two((expected or {}).get("hero_hand"))
    if not (hero_cards and len(hero_cards) == 2):
        return 0
    raw_crops = _locate_hero_cards(table)
    if len(raw_crops) != 2:
        return 0
    crops = [_trim_above_card_edge(c) for c in raw_crops]
    n = 0
    out_dir = Path(out_dir)
    for slot, (crop, label) in enumerate(zip(crops, hero_cards)):
        dest = out_dir / label.lower() / f"{hand_id}_hero_{slot}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop)
        n += 1
    return n


def harvest_corpus(snapshots: list[dict], out_dir: Path) -> int:
    total = 0
    for snap in snapshots:
        expected = snap.get("expected_json")
        if not expected or not snap.get("image_data"):
            continue
        if isinstance(expected, str):
            expected = json.loads(expected)
        total += harvest_snapshot(
            hand_id=snap["hand_id"],
            image_bytes=bytes(snap["image_data"]),
            expected=expected,
            out_dir=out_dir,
        )
    return total
