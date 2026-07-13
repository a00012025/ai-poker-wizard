# GTOW Analyze Fidelity Findings — 2026-07-13

## Decision

The imported GTOW Analyze history is a useful oracle for validating
`analyze_hand_full()`, provided that comparisons are decision-level and only
include decisions that GTOW itself graded successfully.

GTOW decisions marked unknown, no-solution, zero-percent, or otherwise
ungraded remain outside the fidelity denominator. Existing `simplify multiway`
behavior is intentionally preserved for those hands: it can still provide an
approximation, but an approximation cannot be judged against a missing GTOW
solution.

## Audit evidence

Two deterministic cohorts were run:

| cohort | hands | replayed decisions | comparable | exact matches | match rate | hand errors |
|---|---:|---:|---:|---:|---:|---:|
| Fixed pilot, seed `20260713` | 100 | — | 163 | 141 | 86.5% | 0 |
| Rare-first expansion, seed `20260714` | 300 | 633 | 448 | 401 | 89.5% | 0 |

The cohorts overlap by 33 hands, so they cover 367 unique hands. The expanded
cohort deliberately over-sampled 9-max, heads-up, 4bet/5bet, squeeze, all-in,
multi-decision, sizing/depth snap, high-EV-loss, baseline, and GTOW no-solution
cases.

In the 300-hand cohort, 179 GTOW-unknown decisions and 6 GTOW-ungraded
decisions were skipped. They are reported separately and are not counted as
matches or failures.

## Remaining mismatch distribution

Of 448 comparable decisions, 47 did not meet the strict fidelity assertions:

| status | decisions | share of mismatches | interpretation |
|---|---:|---:|---|
| `node_mismatch` | 19 | 40.4% | Different depth, board, or action sequence |
| `ev_mismatch` | 7 | 14.9% | Same canonical node/action, EV loss differs by more than `0.05bb` |
| `taken_action_mismatch` | 6 | 12.8% | Real action mapped to a different solver code |
| `best_action_mismatch` | 5 | 10.6% | Local best action is incompatible with GTOW acceptable actions |
| `own_combo_off_range` | 5 | 10.6% | Exact suited combo does not reach the current local node |
| `frequency_mismatch` | 4 | 8.5% | Same node/EV, taken frequency differs by more than 5 percentage points |
| `ev_unavailable` | 1 | 2.1% | A comparable EV delta could not be obtained |

Mismatch concentration by street:

| street | mismatches | share |
|---|---:|---:|
| River | 30 | 63.8% |
| Preflop | 10 | 21.3% |
| Turn | 5 | 10.6% |
| Flop | 2 | 4.3% |

The remaining risk is therefore not evenly distributed. River action-tree
reconstruction is the dominant surface.

## Most common error sources

### 1. Canonical solver-node reconstruction

This is the largest remaining category. The 19 node mismatches contain:

- 9 depth differences;
- 9 river-action differences;
- 2 preflop-action differences.

One decision contains more than one differing field, so field counts exceed
the number of node-mismatch decisions.

Typical causes include effective-stack selection, depth snapping, accumulated
pot differences, raise-size normalization, and deciding whether a terminal
raise is a numeric sizing or `RAI`. A node mismatch must be resolved before EV
can be compared: EV values from different trees are not parity evidence.

### 2. River sizing and all-in semantics

Thirty of the 47 strict mismatches occur on the river. Small upstream
differences compound by the final street:

- an earlier bet may snap to a different sizing bucket;
- the resulting pot and remaining stack change the river percentage;
- a shove may be represented as a numeric raise in one tree and `RAI` in
  another;
- archived GTOW Analyze action labels may differ from the current live tree.

Numeric raise labels within 15% are considered compatible to absorb harmless
labels such as `R10` versus `R9.5`. All-in versus non-all-in remains a strict
difference.

### 3. Archived Analyze data versus the current GTOW tree

The ledger records a historical GTOW Analyze result, while
`analyze_hand_full()` queries the current solver API. GTOW can change depth
trees, available sizes, labels, strategy frequencies, or EV arrays over time.

Consequently, mismatches must be separated into:

- **reconstruction bugs**: wrong depth, pot, action sequence, or action family;
- **solver drift**: the physical hand is equivalent, but archived and current
  solver data differ;
- **harmless numerical drift**: the chosen action and practical conclusion are
  unchanged, but frequency or EV moves slightly.

Frequency-only differences are lower priority than material node, taken-action,
best-action, or EV-loss differences.

### 4. Exact-combo reach

Five decisions are `own_combo_off_range`. GTOW's archived node grades the exact
suited combo, but the current reconstructed node gives that combo zero or
effectively zero reach.

The old local `0.5%` reach cutoff was a genuine source of false off-range
results and has been removed. Any truly non-zero exact-combo reach, down to the
solver's numerical precision, is now retained. Remaining cases need frozen
per-hand investigation to distinguish action-tree drift from solver-version
drift.

### 5. GTOW unknown and ungraded decisions

These are the most numerous non-match statuses in the raw report, but they are
not analyzer failures:

- `skipped_gtow_unknown`: GTOW has no trustworthy oracle for the decision;
- `skipped_gtow_ungraded`: a real terminal action exists, but GTOW emitted no
  selected/graded action.

They remain visible for coverage accounting and stay outside the denominator.
Multiway fallback behavior must not be removed merely to increase the measured
match rate.

## Systematic bugs fixed during the audit

The fixed 100-hand cohort improved from `95/164` matches (58.0%) to `141/163`
(86.5%). Fixes were protected by regression tests and include:

- binding hero/aggressor stacks correctly when selecting effective depth;
- retaining the known played effective depth around short all-in side pots;
- supporting sub-8bb MTT/HU trees;
- preserving the real preflop structure when later folds leave exact heads-up
  postflop play;
- safe physical 9-max to supported solver-seat mapping, while failing closed
  for unsafe physical-UTG cases;
- computing the physical pot from the real table size rather than padded solver
  seats;
- respecting variable antes instead of assuming `0.125bb`;
- treating explicit all-in amounts as absolute amounts;
- computing postflop raise size from the raise increment and pot after call;
- snapping sizing midpoint ties upward like GTOW;
- preserving rare non-zero exact-combo EV rows;
- accepting N-1 all-fold BB walks;
- separating GTOW-ungraded terminal actions from fidelity failures.

No existing `simplify multiway` behavior was deleted.

## Prioritization for the next audit

North Star ranking is EV-weighted, so follow-up work should not be ordered by
raw mismatch frequency alone.

1. Freeze and investigate material node mismatches with GTOW EV loss at or
   above `3bb`, starting with the river cases.
2. Investigate high-loss exact-combo off-range cases (`13.18bb`, `5.81bb`, then
   the remaining cases).
3. Resolve best-action and taken-action direction disagreements.
4. Re-run the same frozen cohorts after each root-cause fix.
5. Expand with new deterministic seeds only after the material frozen cases are
   stable.
6. Defer frequency-only drift unless it changes the coaching conclusion.

Representative high-impact cases are kept in the ignored cohort report rather
than committed as user data. Reproduce or inspect them with the commands in
[`analysis-fidelity-runbook.md`](analysis-fidelity-runbook.md).

## Acceptance boundary

The fidelity check is considered healthy when:

- GTOW-unknown and GTOW-ungraded decisions are explicit and excluded;
- no missing/extra local decisions or hand-level crashes are hidden;
- node equality is established before EV comparison;
- material node/action/EV differences remain visible and reproducible;
- fixes add frozen regression coverage;
- multiway approximation remains clearly labeled rather than presented as an
  exact GTOW result.

The current 89.5% exact decision match is a useful baseline, not a reason to
silently accept the remaining 10.5%.
