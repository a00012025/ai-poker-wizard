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

GTOW may also retain a real terminal action (usually an all-in/call) without a
selected Analyze action. Those are `skipped_gtow_ungraded`, also outside the
denominator: the local replay is real, but GTOW supplied no EV oracle.

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

## Current validation evidence (2026-07-13)

- Initial fixed 100-hand cohort: `95/164` comparable decisions matched (58%).
- After root-cause fixes, the same 100-hand cohort: `141/163` matched (86.5%),
  with 0 hand errors.
- Expanded rare-first cohort (`--sample-size 300 --seed 20260714`): 300 hands,
  633 replayed decisions, 0 hand errors; 179 GTOW-unknown and 6 GTOW-ungraded
  decisions skipped; `401/448` comparable decisions matched (89.5%).
- The 100- and 300-hand cohorts overlap by 33 hands, covering 367 unique hands.
- Expanded strata include 9-max, heads-up, 4bet/5bet, squeeze, all-in,
  multi-decision, sizing/depth snaps, high-loss, baseline, and no-solution.

Remaining non-matches stay explicit in `report.md`; they are not silently
accepted. Most are archived raw-depth/current-tree action-label or solver-data
drift, exact-combo off-range nodes, or material node/EV differences requiring a
future frozen-case investigation.

## Status interpretation

| Status | Meaning |
|---|---|
| `match` | Same canonical node/action and EV/frequency within tolerance |
| `skipped_gtow_unknown` | GTOW cannot grade the hand; fallback is not judged |
| `skipped_gtow_ungraded` | Real action exists, but GTOW emitted no selected/graded action |
| `node_mismatch` | Gametype/depth/board/action sequence differs |
| `taken_action_mismatch` | Repository mapped the real action to another code |
| `own_combo_off_range` | Exact suited combo does not reach the repository node |
| `best_action_mismatch` | Repository best action is incompatible with GTOW acceptable actions |
| `ev_mismatch` | Same node but EV loss differs by more than `0.05bb` |
| `frequency_mismatch` | Same node/EV but taken frequency differs by more than 5pp |
| `missing_own_solution` | GTOW solved it but the repository returned no solution |
| `missing_own_decision` / `extra_own_decision` | Decision replay counts diverged |

EV is only used as a parity assertion after canonical node equality. The check
compares local raw `best EV - taken EV` against the same delta recomputed from
GTOW's available-action EVs. GTOW's product `ev_loss` field may intentionally
be zero for `INACCURACY`; comparing that thresholded product field to a raw
delta would create false failures. Comparing EV across different depths/action
trees would likewise conflate approximation with a calculation bug.

Numeric raise codes within 15% are treated as the same sizing bucket. This
absorbs harmless archived raw-depth vs current canonical-tree labels such as
`R10` vs `R9.5`; all-in vs non-all-in and materially different sizes remain
strict mismatches.

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
- resume checkpoints and report denominators;
- first-action shove depth, short all-in side pots, and sub-8bb MTT/HU trees;
- 9-max safe mapping and fail-closed physical-UTG handling;
- postflop raise-increment/pot-fraction sizing;
- rare non-zero exact-combo EV rows;
- BB walks and GTOW-ungraded terminal decisions.
