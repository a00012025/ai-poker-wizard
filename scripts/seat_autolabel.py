#!/usr/bin/env python3
"""Avatar-anchored seat-reading auto-label harness (Phase C, D4a).

For every corpus image we already have the HH ground truth, which yields each
seat's stack at screenshot time (the PokerCraft replayer shows per-seat stacks;
for the population that matters these match the HH `stacks_bb` within OCR
tolerance — verified on the cache). The harness scores a seat-read source
against that expected multiset with ZERO manual labeling:

  * seat_recall    — fraction of expected GT stacks matched by some read
  * seat_precision — fraction of reads that matched an expected stack
                     (1 - phantom rate)
  * value_accuracy — among matched reads, fraction within a tight tolerance

Two sources:
  --score-current : the EXISTING pipeline's `named_stacks`, read straight from
                    data/effbb_cache/cache.jsonl (no re-OCR). This is the bar.
  --detector avatars : run scripts/ocr/seat_detector + seat_reader on the image.

Usage:
  python scripts/seat_autolabel.py --score-current --stride 24
  python scripts/seat_autolabel.py --detector avatars --stride 24
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "ocr"))

CACHE = ROOT / "data/effbb_cache/cache.jsonl"
GT = ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
IMG_DIR = ROOT / "data/hand_images/img"


def _match_tol(v):
    return max(0.3, 0.03 * abs(v))


def _greedy_match(reads, expected):
    """Greedily match read values to expected values within tolerance.
    Returns (matched_reads, matched_expected, exact_within_tight)."""
    exp = sorted(expected)
    used = [False] * len(exp)
    matched_reads = 0
    matched_exp = set()
    tight_ok = 0
    for rv in reads:
        best_j, best_d = None, None
        for j, ev in enumerate(exp):
            if used[j]:
                continue
            d = abs(rv - ev)
            if d <= _match_tol(ev) and (best_d is None or d < best_d):
                best_j, best_d = j, d
        if best_j is not None:
            used[best_j] = True
            matched_reads += 1
            matched_exp.add(best_j)
            if best_d <= max(0.15, 0.01 * abs(exp[best_j])):
                tight_ok += 1
    return matched_reads, len(matched_exp), tight_ok


def _load_gt():
    gt = {}
    for line in open(GT, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            gt[d["hand_id"]] = d.get("ground_truth") or {}
    return gt


def score(reads_for_hand, stride):
    """reads_for_hand(hand_id, inputs) -> list[float] of read stack values."""
    gt = _load_gt()
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    tot_exp = tot_reads = tot_matched_reads = tot_matched_exp = tot_tight = 0
    n_imgs = 0
    for i, r in enumerate(rows):
        if i % stride:
            continue
        hid = r["hand_id"]
        g = gt.get(hid) or r.get("gt") or {}
        expected = [v for v in (g.get("stacks_bb") or []) if v and v >= 1.0]
        if not expected:
            continue
        try:
            reads = reads_for_hand(hid, r.get("inputs") or {}) or []
        except Exception:
            reads = []
        reads = [v for v in reads if v and v >= 1.0]
        n_imgs += 1
        tot_exp += len(expected)
        tot_reads += len(reads)
        mr, me, tg = _greedy_match(reads, expected)
        tot_matched_reads += mr
        tot_matched_exp += me
        tot_tight += tg
    recall = 100 * tot_matched_exp / tot_exp if tot_exp else 0
    prec = 100 * tot_matched_reads / tot_reads if tot_reads else 0
    vacc = 100 * tot_tight / tot_matched_reads if tot_matched_reads else 0
    print(f"images={n_imgs}  expected_seats={tot_exp}  reads={tot_reads}")
    print(f"  seat_recall    = {recall:5.1f}%  ({tot_matched_exp}/{tot_exp})")
    print(f"  seat_precision = {prec:5.1f}%  ({tot_matched_reads}/{tot_reads})")
    print(f"  value_accuracy = {vacc:5.1f}%  (tight matches / matched reads)")
    return recall, prec, vacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-current", action="store_true",
                    help="score the existing cached named_stacks (the bar)")
    ap.add_argument("--detector", choices=["avatars"],
                    help="run a new seat-read source")
    ap.add_argument("--stride", type=int, default=24)
    args = ap.parse_args()

    if args.score_current:
        def reads_current(hid, inputs):
            return [s.get("stack") for s in (inputs.get("named_stacks") or [])]
        print("=== BASELINE: current pipeline named_stacks ===")
        score(reads_current, args.stride)
        return 0

    if args.detector == "avatars":
        import cv2
        from ocr.region_detector import detect_regions
        from ocr.seat_detector import detect_avatars
        from ocr.seat_reader import read_seats

        def reads_avatars(hid, inputs):
            path = IMG_DIR / f"{hid}.png"
            if not path.exists():
                return []
            im = cv2.imread(str(path))
            if im is None:
                return []
            reg = detect_regions(im)
            table = reg.get("table") if reg else None
            if table is None:
                table = im
            avatars = detect_avatars(table, None)
            return [row["stack"] for row in read_seats(table, avatars)
                    if row.get("stack")]
        print("=== DETECTOR: avatar-anchored seat reads ===")
        score(reads_avatars, args.stride)
        return 0

    ap.error("choose --score-current or --detector avatars")


if __name__ == "__main__":
    raise SystemExit(main())
