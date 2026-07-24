# Task 9 Report: Hand 2 `b4` small-blind bet parse bug

## Result
DONE_WITH_CONCERNS: fix implemented and quality follow-up tightened. Targeted reproduction, repair, and no-overrepair regressions pass. Full/live filtered suites still have pre-existing/environment failures unrelated to this change (missing `data/effbb_cache/cache.jsonl`, missing Pokercraft ground-truth fixture, and one GTOW snapshot frequency drift in the full run).

## Reproduction
`scripts/_tmp.py` with the real block reproduced the original hard validation failure before the fix:

- Raw street hints: `[['R4', 'C'], ['X', 'R8', 'F']]`
- Parsed flop before fix: `Ac5c6d` actions `[{'position': 'SB', 'action': 'C'}]`
- Validator before fix: `False`, `Ac5c6d：SB 跟注（Call）但前面沒有任何下注——Call 一定要有對象`

After the fix, the same Hand 2 shape repairs to:

- preflop solver line: `F-F-F-F-R2-C-R7-F-F-C`
- contribution line: `F-F-F-F-R2-C-R7-F-C`
- flop: `SB R4` size `4.0`, then `BTN C`
- turn: `SB X`, `BTN R8`, `SB F`
- repair note: `flop 補回原文開頭 bet`
- validator: `True`, no hard errors

## Root Cause
`_extract_street_action_hints()` correctly parsed raw `Ac5c6d b4 call` as `[R4, C]` (`scripts/live_flow.py:398-407`). Gemini emitted only the caller for the flop (`SB C`). The existing `repair_street_actions_from_block()` only repaired one exact mismatch shape: dropped leading HU check before aggression (`scripts/live_flow.py:501-509`). It required two current street actors, so a one-action orphan-call street was skipped before `repair_hu_pot()` could reassign HU alternation.

## Fix
Extended `repair_street_actions_from_block()` with a second conservative exact-mismatch repair (`scripts/live_flow.py:459-499`):

- infer exactly two HU actors from **raw preflop events only** for the new `[R,C]` orphan-call repair;
- do **not** use parsed street actors as fallback for this new repair, because parsed actors may be the corrupt LLM output;
- require raw hint classes exactly `[R, C]`;
- require parsed classes exactly `[C]` or `[X, C]`;
- restore OOP `R{size}` and IP `C`, preserving the existing caller payload and adding a visible repair note.

The older dropped-leading-check repair is preserved: it still uses the parsed street's two actors and exact `X`-missing mismatch shape, without depending on raw-preflop HU proof.

No signature changes.

## Regression
Updated/added tests in `scripts/regression_tests/test_live_flow.py`:

- `flop_b4_is_bet` asserts exact repaired preflop solver/contribution lines, flop `SB R4` size `4.0` then `BTN C`, turn `SB X` / `BTN R8` / `SB F`, validator success, and visible repair note.
- `test_live_repair_street_actions_does_not_restore_bet_without_raw_hu_proof` proves the new `[R,C]` repair does not use parsed street actor fallback when raw preflop cannot establish exactly two live HU actors.
- Existing `test_live_repair_street_actions_restores_dropped_leading_check` keeps coverage for the older dropped-leading-check behavior.

## Verification
Passing:

```bash
python -m py_compile scripts/live_flow.py scripts/regression_tests/test_live_flow.py
python scripts/regression_test.py -k flop_b4_is_bet
python scripts/regression_test.py -k does_not_restore_bet
python scripts/regression_test.py -k repair_street_actions
```

Additional run:

```bash
python scripts/regression_test.py -k live
```

Result after quality follow-up: 71 passed, 4 failed. The 4 failures all require missing `data/effbb_cache/cache.jsonl` and are unrelated to this live parser change.

Earlier full suite run:

```bash
python scripts/regression_test.py
```

Result: 852 passed, 36 failed. Unrelated failures were missing local effbb/cache and Pokercraft fixtures plus one GTOW snapshot frequency drift (`H2494`). Cache drift created during that run was restored before commit.
