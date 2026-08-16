#!/usr/bin/env python3
"""Phase-0 oracle harness for effective_bb accuracy.

Quantifies, on the hero-active corpus, the SEPARATE ceilings of the two levers
so we invest in the right ones:

  * oracle_attribution : if seat/contestant selection were PERFECT but we kept
    the current OCR stack numbers, what bucket-precision/coverage is reachable?
    (= "is the right number already in the inputs?") -> the ATTRIBUTION lever.
  * input-bound floor  : hands where NO OCR input value lands in the GT bucket
    (even allowing a small added investment) -> the OCR-REREAD lever.
  * attribution ambiguity : among recoverable hands, how many DISTINCT seat
    stacks fall in the GT bucket? (1 => well-posed; >=2 => the betting-state
    engine must disambiguate the true contestant) -> sizes the Phase-2 problem.
  * input-bound severity : for input-bound hands, how many buckets away is the
    nearest input value, and is GT reachable from a small digit correction of
    some seat? -> sizes the Phase-3/5 reread problem.

Usage: python scripts/effbb_oracle.py --cache data/effbb_cache/cache.jsonl
"""
import argparse
import json
from collections import Counter


from effbb_metrics import depth_bucket, hero_folded_preflop

# Bucket order (short->deep) for "buckets-away" distance.
BUCKETS = [8, 9, 10, 12, 14, 17, 20, 25, 30, 35, 40, 50, 60, 80, 100]
BIDX = {b: i for i, b in enumerate(BUCKETS)}
INVEST_TRIES = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 11.0)


def gt_bucket_hits(stacks, gb):
    """Distinct input stacks whose (displayed [+small invest]) lands in bucket gb."""
    hits = set()
    for s in stacks:
        if not s or s < 1.0:
            continue
        for inv in INVEST_TRIES:
            if depth_bucket(s + inv) == gb:
                hits.add(round(s, 2))
                break
    return hits


def nearest_bucket_distance(stacks, gb):
    """Min |bucket(stack) - gb| in bucket-index units over all input stacks."""
    tgt = BIDX[gb]
    best = None
    for s in stacks:
        if not s or s < 1.0:
            continue
        b = depth_bucket(s)
        if b in BIDX:
            d = abs(BIDX[b] - tgt)
            best = d if best is None else min(best, d)
    return best


def digit_correctable(stacks, ge, gb):
    """Could a small OCR digit slip explain a missing GT value? Heuristic: some
    input stack, after a single-digit-ish perturbation (drop/insert a leading
    digit, swap a confusable), lands in the GT bucket. We approximate with:
    GT value within [s/10, s*10] ratio of some stack and shares a digit pattern.
    Cheap proxy: GT*10 or GT/10 or GT with one digit changed near an input."""
    for s in stacks:
        if not s or s < 1.0:
            continue
        # common misreads: dropped/added leading digit (x10), decimal slip
        for cand in (s * 10, s / 10, s + 10, s - 10):
            if cand >= 1.0 and depth_bucket(cand) == gb:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/effbb_cache/cache.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]

    n = 0
    recoverable = 0
    input_bound = 0
    ambiguity = Counter()          # #distinct GT-bucket stacks among recoverable
    ib_distance = Counter()        # buckets-away of nearest input for input-bound
    ib_digit_fixable = 0
    for r in rows:
        gt = r.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in r:
            continue
        if hero_folded_preflop(gt) is not False:
            continue
        n += 1
        gb = depth_bucket(ge)
        stacks = [s for s in (r["inputs"].get("stacks") or []) if s and s >= 1.0]
        hits = gt_bucket_hits(stacks, gb)
        if hits:
            recoverable += 1
            ambiguity[min(len(hits), 4)] += 1   # cap label at "4+"
        else:
            input_bound += 1
            d = nearest_bucket_distance(stacks, gb)
            ib_distance[d if d is not None else -1] += 1
            ib_digit_fixable += digit_correctable(stacks, ge, gb)

    print(f"Hero-active hands (GT effective>=1): {n}\n")
    print(f"ORACLE-ATTRIBUTION ceiling (right number present in inputs):")
    print(f"  recoverable : {recoverable} ({100*recoverable/n:.1f}%)  "
          f"-> perfect attribution => 100% precision @ {100*recoverable/n:.1f}% coverage")
    print(f"  input-bound : {input_bound} ({100*input_bound/n:.1f}%)  "
          f"-> needs OCR reread or abstain\n")

    print("ATTRIBUTION AMBIGUITY among recoverable (distinct GT-bucket seats):")
    for k in sorted(ambiguity):
        lbl = f"{k}+" if k == 4 else str(k)
        print(f"  {lbl} candidate seat(s) in GT bucket: {ambiguity[k]} "
              f"({100*ambiguity[k]/recoverable:.1f}%)")
    well_posed = ambiguity[1]
    print(f"  => {100*well_posed/recoverable:.1f}% of recoverable have a UNIQUE "
          f"GT-bucket seat (attribution well-posed);")
    print(f"     the rest need the betting-state engine to pick the true "
          f"contestant among same-bucket seats.\n")

    print("INPUT-BOUND severity (buckets-away of nearest input value):")
    for d in sorted(ib_distance):
        lbl = "no valid stack" if d == -1 else f"{d} bucket(s)"
        print(f"  {lbl}: {ib_distance[d]} ({100*ib_distance[d]/input_bound:.1f}%)")
    print(f"  digit-slip-explainable (x10 / decimal / ±10 of a seat): "
          f"{ib_digit_fixable} ({100*ib_digit_fixable/input_bound:.1f}% of input-bound)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
