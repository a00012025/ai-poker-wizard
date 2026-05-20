"""Extract labeled CardCNN crops from PokerCraft replay screenshots."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocr.region_detector import detect_regions  # noqa: E402
from ocr.table_parser import (  # noqa: E402
    _locate_board_cards,
    _locate_hero_cards,
    _mask_win_overlay,
    _trim_above_card_edge,
)

OUT_ROOT = REPO_ROOT / "data" / "cards_v2"
SKIP_LOG = REPO_ROOT / "data" / "cards_v2_skipped.log"
_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")
_RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA")}
_PAIR_SUIT_ORDER = {s: i for i, s in enumerate("chds")}


def _split_cards(value: str) -> list[str]:
    value = (value or "").replace(" ", "")
    return [value[i:i + 2] for i in range(0, len(value) - 1, 2)]


def _gt_labels(gt: dict) -> tuple[list[str], list[str]]:
    hero = [card for card in _split_cards(gt.get("hero_hand", "")) if _CARD_RE.match(card)]
    board: list[str] = []
    for street in gt.get("streets", []) or []:
        board.extend(card for card in _split_cards(street.get("board", "")) if _CARD_RE.match(card))
        card = street.get("card")
        if card and _CARD_RE.match(card):
            board.append(card)
    return hero, board


def _visual_hero_order(cards: list[str]) -> list[str]:
    if len(cards) != 2:
        return cards
    if cards[0][0] == cards[1][0]:
        return sorted(cards, key=lambda c: _PAIR_SUIT_ORDER.get(c[1], -1), reverse=True)
    return sorted(cards, key=lambda c: _RANK_ORDER.get(c[0], -1), reverse=True)


def extract_one(image_bytes: bytes, gt: dict) -> dict:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "hero_crops": [],
            "hero_labels": [],
            "board_crops": [],
            "board_labels": [],
            "reason": "decode_failed",
        }
    regions = detect_regions(img)
    if regions is None:
        return {
            "hero_crops": [],
            "hero_labels": [],
            "board_crops": [],
            "board_labels": [],
            "reason": "region_detect_failed",
        }
    hero_labels, board_labels = _gt_labels(gt)
    hero_labels = _visual_hero_order(hero_labels)
    table = regions["table"]
    hero_crops = [_trim_above_card_edge(c) for c in _locate_hero_cards(table)]
    return {
        "hero_crops": hero_crops,
        "hero_labels": hero_labels,
        "board_crops": _locate_board_cards(table),
        "board_labels": board_labels,
        "reason": None,
    }


def _save(crop: np.ndarray, label: str, hand_id: str, source: str, slot: int) -> None:
    match = _CARD_RE.match(label)
    if not match:
        return
    out_dir = OUT_ROOT / match.group(1) / match.group(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{hand_id}_{source}_{slot}.png"), crop)


def _log_skip(hand_id: str, reason: str) -> None:
    with SKIP_LOG.open("a") as fh:
        fh.write(f"{hand_id}\t{reason}\n")


def _load_gt(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            rows[row["hand_id"]] = row["ground_truth"]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--images", default="data/hand_images/img")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gt_rows = _load_gt(Path(args.gt))
    paths = sorted(Path(args.images).glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.write_text("")

    n_hero = 0
    n_board = 0
    n_skip = 0
    for idx, path in enumerate(paths, start=1):
        gt = gt_rows.get(path.stem)
        if gt is None:
            continue
        result = extract_one(path.read_bytes(), gt)
        if result["reason"]:
            _log_skip(path.stem, result["reason"])
            n_skip += 1
            continue
        hero_crops = result["hero_crops"]
        hero_labels = result["hero_labels"]
        board_crops = result["board_crops"]
        board_labels = result["board_labels"]
        if hero_labels and len(hero_crops) == len(hero_labels):
            for slot, (crop, label) in enumerate(zip(hero_crops, hero_labels)):
                _save(crop, label, path.stem, "hero", slot)
                _save(_mask_win_overlay(crop), label, path.stem, "hero_masked", slot)
                n_hero += 1
        elif hero_labels:
            _log_skip(path.stem, f"hero_count crops={len(hero_crops)} labels={len(hero_labels)}")
        if board_labels and len(board_crops) == len(board_labels):
            for slot, (crop, label) in enumerate(zip(board_crops, board_labels)):
                _save(crop, label, path.stem, "board", slot)
                n_board += 1
        elif board_labels:
            _log_skip(path.stem, f"board_count crops={len(board_crops)} labels={len(board_labels)}")
        if idx % 250 == 0:
            print(f"{idx}/{len(paths)} hero={n_hero} board={n_board} skipped={n_skip}", flush=True)

    print(f"DONE images={len(paths)} hero={n_hero} board={n_board} skipped={n_skip} log={SKIP_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
