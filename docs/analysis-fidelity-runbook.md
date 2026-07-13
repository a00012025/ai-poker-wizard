# GTOW Analyzer ↔ `analyze_hand_full` Fidelity Runbook

## Purpose

Use the imported online ledger as a sampling index and GTOW Analyzer detail as
the oracle for validating new-hand analysis. The check is decision-level:
reconstruction, canonical solver node, taken action, acceptable actions,
frequency, and EV loss.

This does **not** replace GTOW Analyzer. It calibrates the repository's fallback
grader and keeps approximations observable, as required by `NORTH_STAR.md` §4.2
and §7.2.

## Important boundary

Ledger rows alone are insufficient because they intentionally store distilled
decision facts, not the complete action stream. The checker therefore uses:

1. `ledger_hands` / `ledger_decisions` for deterministic stratified sampling;
2. `data/gtow_raw/...` when the archived detail exists locally;
3. GTOW Analyze detail API as fallback;
4. `analyze_hand_full()` for the repository-side result.

GTOW hands marked `NO_GTO_SOLUTION`, `ZERO_PERCENT_ACTION`, or another non-OK
warning/solution status are reported as `skipped_gtow_unknown`. They do not enter
the fidelity denominator. Existing `simplify multiway` behavior remains useful
for those hands, but its approximate EV is not compared against a nonexistent
GTOW oracle.

## Commands

```bash
# Inspect the deterministic rare-first sample without making solver calls
python scripts/analysis_fidelity_check.py --sample-size 30 --dry-run

# Pilot
python scripts/analysis_fidelity_check.py --sample-size 30

# Main audit; resumable by default
python scripts/analysis_fidelity_check.py --sample-size 400 --resume

# Specific regression hand(s)
python scripts/analysis_fidelity_check.py \
  --hand-id bed8860a-442b-4478-a9b4-8acfd52b6143 \
  --hand-id eef0b07b-23b6-4fe0-bcc6-41d83629583c
```

Artifacts are ignored runtime data:

- `data/analysis_fidelity/results.jsonl` — append-only per-hand checkpoints;
- `data/analysis_fidelity/report.md` — aggregate and per-decision mismatch view.

Use `--no-resume` to replace the current JSONL run. Use another `--output-dir`
when comparing code revisions without overwriting the prior evidence.

## Sampling

The selector is deterministic for a given seed and intentionally over-samples:

- 5bet, 4bet, squeeze;
- heads-up and 9-max;
- all-in and repeated hero decisions;
- sizing/depth snaps and GTOW no-solution cases;
- high-EV-loss hands;
- then fills the remainder from the baseline population.

The default seed is `20260713`; pass `--seed` for a separate reproducible cohort.

## Status interpretation

| Status | Meaning |
|---|---|
| `match` | Same canonical node/action and EV/frequency within tolerance |
| `skipped_gtow_unknown` | GTOW cannot grade the hand; fallback is not judged |
| `node_mismatch` | Gametype/depth/board/action sequence differs |
| `taken_action_mismatch` | Repository mapped the real action to another code |
| `own_combo_off_range` | Exact suited combo does not reach the repository node |
| `best_action_mismatch` | Repository best action is incompatible with GTOW acceptable actions |
| `ev_mismatch` | Same node but EV loss differs by more than `0.05bb` |
| `frequency_mismatch` | Same node/EV but taken frequency differs by more than 5pp |
| `missing_own_solution` | GTOW solved it but the repository returned no solution |
| `missing_own_decision` / `extra_own_decision` | Decision replay counts diverged |

EV is only used as a parity assertion after canonical node equality. Comparing
EV across different depths/action trees would conflate approximation with a
calculation bug.

## Regression coverage

```bash
python scripts/regression_test.py -k fidelity -v
python scripts/regression_test.py
```

Frozen fixtures cover:

- simple preflop fold;
- exact-suit multi-street river blunder;
- rare 9-max squeeze with a large missed turn EV loss;
- GTOW-unknown skipping;
- node-before-EV comparison semantics;
- strategy-array extraction;
- deterministic rare-first sampling;
- resume checkpoints and report denominators.
