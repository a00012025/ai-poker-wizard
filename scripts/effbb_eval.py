#!/usr/bin/env python3
"""Evaluate _compute_effective_bb over the input cache in seconds.

Replays cached inputs through the CURRENT _compute_effective_bb, scores at the
solver depth bucket level vs HH ground truth, splits hero-active vs hero-folded,
and prints the fault breakdown + a precision/coverage curve over confidence.

Usage: python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ocr"))

from effbb_metrics import bucket_match, hero_folded_preflop, classify_fault, depth_bucket
from ocr.n8_parser import _compute_effective_bb


def recompute(inp):
    """Replay one cached input tuple through _compute_effective_bb.
    Tolerates both the 2-tuple (legacy) and 3-tuple (rewritten) returns."""
    res = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"],
    )
    if isinstance(res, tuple) and len(res) == 3:
        return res                      # (eff, hero_start, confidence)
    eff, hero_start = res
    return eff, hero_start, 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="emit only when confidence >= this (precision/coverage knob)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]
    active, folded = [], []
    for r in rows:
        gt = r.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in r:
            continue
        p_eff, hero_start, conf = recompute(r["inputs"])
        if conf < args.min_conf:
            p_eff = None
        rec = {"hid": r["hand_id"], "gt_eff": ge, "p_eff": p_eff,
               "hero_start": hero_start, "conf": conf,
               "gt_max": max(gt.get("stacks_bb") or [0]) or None}
        hf = hero_folded_preflop(gt)
        # CLEANUP: route by hero-folded status; skip when unknown (hf is None).
        if hf is True:
            folded.append(rec)
        elif hf is False:
            active.append(rec)

    def score(name, subset):
        emitted = [x for x in subset if x["p_eff"] is not None]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(subset) if subset else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"\n## {name}: n={len(subset)} emitted={len(emitted)} "
              f"coverage={cov:.1f}% bucket-precision={prec:.2f}% "
              f"({len(ok)}/{len(emitted)})")
        wrong = [x for x in emitted if x not in ok]
        faults = {}
        for x in wrong:
            f = classify_fault(p_eff=x["p_eff"], gt_eff=x["gt_eff"],
                               hero_start=x["hero_start"] or x["gt_eff"],
                               gt_max=x["gt_max"])
            faults[f] = faults.get(f, 0) + 1
        if wrong:
            print("   faults:", faults)
        return prec, cov

    score("HERO ACTIVE (target population)", active)
    score("HERO FOLDED (context only)", folded)

    print("\n--- precision/coverage curve (hero-active, by confidence floor) ---")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        emitted = [x for x in active if x["p_eff"] is not None and x["conf"] >= thr]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(active) if active else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"  conf>={thr:.1f}: coverage={cov:5.1f}%  precision={prec:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
