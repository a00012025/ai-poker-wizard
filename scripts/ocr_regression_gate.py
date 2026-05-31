#!/usr/bin/env python3
"""OCR no-regression gate: compare two ocr_benchmark.py runs and block on regressions.

`ocr_benchmark.py` writes `<out>/diffs.jsonl` listing every hand that did NOT
match ground truth (parse failures + field mismatches). A change is a
*regression* if any hand that passed on the baseline now fails. This script
diffs the two failing-id sets and exits non-zero when that happens, so it can
gate a commit/merge.

Run the SAME image set + ground truth on both sides (same --images / --limit):

    # baseline (on main / merge-base):
    python scripts/ocr_benchmark.py --images data/hand_images/img \
        --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
        --out /tmp/ocr_base
    # candidate (on your branch, with the OCR change):
    python scripts/ocr_benchmark.py --images data/hand_images/img \
        --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
        --out /tmp/ocr_head
    # gate:
    python scripts/ocr_regression_gate.py --base /tmp/ocr_base --head /tmp/ocr_head

Exit 0 = no hand regressed (fixes/no-change only). Exit 1 = at least one hand
that passed on baseline now fails — BLOCK the change.
"""
import argparse
import json
import sys
from pathlib import Path


def _failing_ids(out: Path) -> dict[str, dict]:
    """hand_id -> diff record for every hand in <out>/diffs.jsonl."""
    p = out / "diffs.jsonl" if out.is_dir() else out
    if not p.exists():
        sys.exit(f"missing diffs file: {p} (did ocr_benchmark.py run with --out?)")
    failing = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        failing[rec["hand_id"]] = rec
    return failing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path,
                    help="baseline ocr_benchmark --out dir (or diffs.jsonl path)")
    ap.add_argument("--head", required=True, type=Path,
                    help="candidate ocr_benchmark --out dir (or diffs.jsonl path)")
    ap.add_argument("--show", type=int, default=20, help="max records to print")
    args = ap.parse_args()

    base_fail = _failing_ids(args.base)
    head_fail = _failing_ids(args.head)

    regressions = sorted(set(head_fail) - set(base_fail))
    fixes = sorted(set(base_fail) - set(head_fail))

    print(f"baseline failing: {len(base_fail)}   candidate failing: {len(head_fail)}")
    print(f"fixed (passed-now): {len(fixes)}   regressed (newly-failing): {len(regressions)}")

    if fixes:
        print("\n✅ FIXES (failed on baseline, pass now):")
        for hid in fixes[:args.show]:
            print(f"  {hid}")
        if len(fixes) > args.show:
            print(f"  … +{len(fixes) - args.show} more")

    if regressions:
        print("\n❌ REGRESSIONS (passed on baseline, FAIL now) — BLOCKING:")
        for hid in regressions[:args.show]:
            r = head_fail[hid]
            ocr, gt = r.get("ocr", {}), r.get("gt", {})
            print(f"  {hid}  ocr={ocr}  gt={gt}")
        if len(regressions) > args.show:
            print(f"  … +{len(regressions) - args.show} more")
        print(f"\nGATE FAILED: {len(regressions)} hand(s) regressed. Do NOT commit/merge.")
        sys.exit(1)

    print("\nGATE PASSED: no hand regressed.")


if __name__ == "__main__":
    main()
