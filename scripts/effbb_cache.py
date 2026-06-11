#!/usr/bin/env python3
"""Build the effective_bb input cache: one full parse over the corpus,
capturing the inputs to _compute_effective_bb + HH ground truth + a hash of
the OCR modules that produce those inputs (for staleness detection).

Usage:
  EFFBB_CAPTURE=1 python scripts/effbb_cache.py \
      --images data/hand_images/img \
      --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
      --out data/effbb_cache/cache.jsonl [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ocr"))

from ocr.n8_parser import parse_n8_screenshot

# Modules whose changes invalidate the cache (see spec staleness contract).
_OCR_MODULE_FILES = [
    "ocr/panel_parser.py",
    "ocr/table_parser.py",
    "ocr/n8_parser.py",
]


def ocr_modules_hash() -> str:
    h = hashlib.sha256()
    base = Path(__file__).resolve().parent
    for rel in _OCR_MODULE_FILES:
        h.update((base / rel).read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.getenv("EFFBB_CAPTURE"):
        sys.exit("Set EFFBB_CAPTURE=1 so _assemble_hand stashes __effbb_inputs__.")

    gt = {}
    with open(args.ground_truth, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            gt[o["hand_id"]] = o["ground_truth"]

    imgs = sorted(Path(args.images).glob("*.png"))
    pairs = [(p, gt[p.stem]) for p in imgs if p.stem in gt]
    if args.limit:
        pairs = pairs[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mod_hash = ocr_modules_hash()
    n_ok = n_inputs = 0
    with out.open("w", encoding="utf-8") as wf:
        for i, (path, g) in enumerate(pairs):
            rec = {"hand_id": path.stem, "ocr_hash": mod_hash,
                   "gt": {k: g.get(k) for k in
                          ("effective_bb", "stacks_bb", "preflop_actions",
                           "num_players", "table_size", "hero_position")}}
            try:
                res = parse_n8_screenshot(path.read_bytes())
                hand = res.get("hand")
                rec["confidence"] = round(res.get("confidence", 0.0), 3)
                rec["hand_none"] = hand is None
                if hand and "__effbb_inputs__" in hand:
                    rec["inputs"] = hand["__effbb_inputs__"]
                    n_inputs += 1
                n_ok += 1
            except Exception as e:
                rec["err"] = str(e)[:120]
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(pairs)} (inputs={n_inputs})", flush=True)
    print(f"[effbb_cache] wrote {len(pairs)} rows -> {out} "
          f"(parsed={n_ok}, with_inputs={n_inputs}, ocr_hash={mod_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
