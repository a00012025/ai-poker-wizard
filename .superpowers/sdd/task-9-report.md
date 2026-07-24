# Task 9 Report: Hand 2 `b4` small-blind bet parse bug

## Result
DONE_WITH_CONCERNS: fix implemented and committed. Targeted reproduction and live-flow regressions pass. Full regression suite still has pre-existing/environment failures unrelated to this change (missing `data/effbb_cache/cache.jsonl`, missing Pokercraft ground-truth fixture, and one GTOW snapshot frequency drift).

## Reproduction
`scripts/_tmp.py` with the real block reproduced the original hard validation failure before the fix:

- Raw street hints: `[['R4', 'C'], ['X', 'R8', 'F']]`
- Parsed flop before fix: `Ac5c6d` actions `[{'position': 'SB', 'action': 'C'}]`
- Validator before fix: `False`, `Ac5c6d：SB 跟注（Call）但前面沒有任何下注——Call 一定要有對象`

After the fix, the same reproduction prints:

- Parsed flop: `SB R4` / `BTN C`
- Repair note: `flop 補回原文開頭 bet`
- Validator: `True`, no hard errors

## Root Cause
`_extract_street_action_hints()` correctly parsed raw `Ac5c6d b4 call` as `[R4, C]` (`scripts/live_flow.py:398-407`). Gemini emitted only the caller for the flop (`SB C`). The existing `repair_street_actions_from_block()` only repaired one exact mismatch shape: dropped leading HU check before aggression (`scripts/live_flow.py:501-509` after this patch). It required two current street actors, so a one-action orphan-call street was skipped before `repair_hu_pot()` could reassign HU alternation.

## Fix
Extended `repair_street_actions_from_block()` with a second conservative, HU-only exact-mismatch repair (`scripts/live_flow.py:459-499`):

- infer exactly two HU actors from the raw preflop line first, falling back to all parsed street actors only if exactly two;
- require raw hint classes exactly `[R, C]`;
- require parsed classes exactly `[C]` or `[X, C]`;
- restore OOP `R{size}` and IP `C`, preserving the existing caller payload and adding a visible repair note.

No signature changes.

## Regression
Added `flop_b4_is_bet` in `scripts/regression_tests/test_live_flow.py:878-920`. It uses a fake Gemini client returning the observed bad parse, then asserts the repaired hand is legal and the flop opens with a bet.

## Verification
Passing:

```bash
python scripts/_tmp.py
python -m py_compile scripts/live_flow.py scripts/regression_tests/test_live_flow.py
python scripts/regression_test.py -k flop_b4_is_bet
python scripts/regression_test.py -k repair_street_actions
```

Additional run:

```bash
python scripts/regression_test.py -k live
```

Result: 70 passed, 4 failed. The 4 failures all require missing `data/effbb_cache/cache.jsonl` and are unrelated to this live parser change.

Full suite:

```bash
python scripts/regression_test.py
```

Result: 852 passed, 36 failed. Unrelated failures were missing local effbb/cache and Pokercraft fixtures plus one GTOW snapshot frequency drift (`H2494`). Cache drift created during the run was restored before commit.
