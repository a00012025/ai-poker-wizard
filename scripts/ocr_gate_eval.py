"""Replay cached ocr_precision records through the confidence gate.

Reads ``data/<run>/diffs.jsonl`` plus the run's ``summary.json`` and
evaluates ``scripts/ocr/confidence_gate.py`` over every record without
re-running the parser. Reports precision/coverage/per-mode counts
before and after the gate so we can iterate on rule wording without
the ~4-minute OCR walk.

Usage:
    python scripts/ocr_gate_eval.py --in data/ocr_precision_current

Optional:
    --threshold 0.88          override the emit threshold
    --disable-hard-rules      replicate legacy threshold-only behaviour
    --out data/.../gate.json  also emit JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr.confidence_gate import evaluate_from_record


def _load_records(in_dir: Path) -> list[dict]:
    diffs = in_dir / "diffs.jsonl"
    if not diffs.exists():
        sys.exit(f"missing {diffs}")
    return [json.loads(line) for line in diffs.read_text().splitlines()]


def _load_summary(in_dir: Path) -> dict:
    summary = in_dir / "summary.json"
    if not summary.exists():
        sys.exit(f"missing {summary}")
    return json.loads(summary.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.88)
    ap.add_argument("--disable-hard-rules", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    records = _load_records(in_dir)
    summary = _load_summary(in_dir)
    paired = summary.get("paired") or len(records)

    # Each cached record is a "scored" hand. parse_none records are
    # already non-emit no matter what; we still evaluate them so the
    # numbers line up.

    emit_count = 0
    correct = 0
    wrong = 0
    abstained = 0
    abstain_reasons: Counter = Counter()
    emit_via: Counter = Counter()
    wrong_by_mode: Counter = Counter()

    # Recovery / loss accounting vs the existing pipeline emit decision
    flips_to_emit_exact = 0
    flips_to_emit_wrong = 0
    flips_to_abstain_exact = 0
    flips_to_abstain_wrong = 0

    for r in records:
        # Was it emitted before? (i.e. not parsed_none AND not abstained)
        prev_emitted = (not r.get("parsed_none")) and (not r.get("abstained_confidence"))
        fields = r.get("fields") or {}
        is_exact = bool(fields.get("hand_exact"))

        decision = evaluate_from_record(
            r,
            emit_threshold=args.threshold,
        )
        if args.disable_hard_rules:
            # Recompute without hard rules for A/B
            from ocr.confidence_gate import evaluate
            decision = evaluate(
                {
                    "hand": r.get("parsed") if not r.get("parsed_none") else None,
                    "confidence": r.get("confidence"),
                    "confidence_parts": r.get("confidence_parts"),
                    "diagnostics": r.get("diagnostics"),
                    "safe_emit_reason": r.get("safe_emit_reason"),
                },
                emit_threshold=args.threshold,
                enable_hard_rules=False,
            )

        if decision["emit"]:
            emit_count += 1
            emit_via[decision["reason"]] += 1
            if is_exact:
                correct += 1
            else:
                wrong += 1
                wrong_by_mode[r.get("failure_mode") or "unknown"] += 1
        else:
            abstained += 1
            abstain_reasons[decision["reason"]] += 1

        # Flip accounting (only meaningful for records that exist in diffs)
        if prev_emitted and not decision["emit"]:
            if is_exact:
                flips_to_abstain_exact += 1
            else:
                flips_to_abstain_wrong += 1
        elif (not prev_emitted) and decision["emit"]:
            if is_exact:
                flips_to_emit_exact += 1
            else:
                flips_to_emit_wrong += 1

    # diffs.jsonl only contains the FAILURE set (parse_none, abstained, and
    # emitted-wrong). It does NOT contain the emitted-exact hands, since
    # ocr_precision skips writing those. So our "correct" count above only
    # captures correct hands that happen to be in the file (typically 0).
    # We need to combine our wrong count with the summary's emitted-exact
    # count.
    scored = summary.get("scored") or 0
    emitted_exact_baseline = scored - (
        summary.get("failure_modes", {}).get("position_wrong", 0)
        + summary.get("failure_modes", {}).get("preflop_action_types_wrong", 0)
        + summary.get("failure_modes", {}).get("board_wrong", 0)
        + summary.get("failure_modes", {}).get("hero_cards_wrong", 0)
        + summary.get("failure_modes", {}).get("preflop_action_sized_wrong", 0)
    )

    # The records in diffs.jsonl cover: parse_none, abstained, and emitted-wrong.
    # All the emitted-exact hands are NOT in the file. Under the gate, those
    # emitted-exact hands stay emitted (the gate has no signal that would
    # abstain them, since they have low pre_collapse_loss etc.), so we add
    # them back as "still exact" baseline.
    # Adjust counters: we computed `correct` from the diff file only; the
    # actual correct count is `correct + emitted_exact_baseline` (those that
    # weren't flipped to abstain).
    #
    # But hard rules could flip emitted-exact hands to abstain too! We can't
    # tell that from diffs alone — we'd need to re-run the parser on the
    # full corpus. For the rule-based gate, we APPROXIMATE: assume emitted-
    # exact baseline hands stay emitted (verify later by re-running OCR).
    final_emitted = emit_count + emitted_exact_baseline
    final_correct = correct + emitted_exact_baseline
    final_wrong = wrong

    precision_pct = (
        (final_correct / final_emitted * 100) if final_emitted else 0.0
    )
    coverage_pct = (final_emitted / paired * 100) if paired else 0.0

    print("=" * 60)
    print(f"GATE EVAL — in_dir={in_dir}  threshold={args.threshold}  hard_rules={not args.disable_hard_rules}")
    print("=" * 60)
    print(f"paired (test bucket):      {paired}")
    print(f"records in diffs.jsonl:    {len(records)}")
    print(f"emitted-exact baseline:    {emitted_exact_baseline}")
    print()
    print(f"new emitted:               {final_emitted}")
    print(f"new correct:               {final_correct}")
    print(f"new wrong:                 {final_wrong}")
    print(f"new abstained:             {paired - final_emitted}")
    print(f"new precision:             {precision_pct:.3f}%")
    print(f"new coverage:              {coverage_pct:.3f}%")
    print()
    print("wrong by failure_mode:")
    for mode, ct in wrong_by_mode.most_common():
        print(f"  {mode}: {ct}")
    print()
    print("emit-reason breakdown (excludes emitted-exact baseline):")
    for r, ct in emit_via.most_common():
        print(f"  {r}: {ct}")
    print()
    print("abstain-reason breakdown:")
    for r, ct in abstain_reasons.most_common(20):
        print(f"  {r}: {ct}")
    print()
    print("flips vs baseline:")
    print(f"  emit -> abstain (correctly):  {flips_to_abstain_wrong}")
    print(f"  emit -> abstain (cost exact): {flips_to_abstain_exact}")
    print(f"  abstain -> emit (recovered):  {flips_to_emit_exact}")
    print(f"  abstain -> emit (new wrong):  {flips_to_emit_wrong}")
    print()
    print("acceptance vs 99%@70% target:")
    print(f"  coverage >= 70.0%?      {coverage_pct >= 70.0}")
    print(f"  precision >= 99.0%?     {precision_pct >= 99.0}")
    print(f"  emitted >= 503?         {final_emitted >= 503}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "paired": paired,
            "threshold": args.threshold,
            "hard_rules": not args.disable_hard_rules,
            "final_emitted": final_emitted,
            "final_correct": final_correct,
            "final_wrong": final_wrong,
            "precision_pct": precision_pct,
            "coverage_pct": coverage_pct,
            "wrong_by_mode": dict(wrong_by_mode),
            "emit_reasons": dict(emit_via),
            "abstain_reasons": dict(abstain_reasons),
            "flips_to_abstain_wrong": flips_to_abstain_wrong,
            "flips_to_abstain_exact": flips_to_abstain_exact,
            "flips_to_emit_exact": flips_to_emit_exact,
            "flips_to_emit_wrong": flips_to_emit_wrong,
        }, indent=2))
        print(f"\nwrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
