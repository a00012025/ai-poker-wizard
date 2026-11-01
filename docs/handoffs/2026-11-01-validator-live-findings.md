# Validator findings surfaced during the H3985 release gate

The former `test_validator_corpus_no_new_false_positives` read the mutable
`analysis_snapshots` table. On 2026-11-01 it found three unreviewed hands that
were added after its triage list. These remain visible through
`python scripts/validation_report.py --worklist`; the offline regression test
now uses committed snapshots so an unrelated live upload cannot change CI.

| Hand | Finding | Next action |
| --- | --- | --- |
| H3962 | GTOW screenshot parser emitted hero `Ts9s` and flop `KsTs9s`, duplicating both hero cards on the board. | Confirm all card suits from the original screenshot, then fix the GTOW image path and lock a snapshot. |
| H3963 | GTOW screenshot parser emitted hero `Tc9c` and flop `KdTh9c` (duplicate `9c`), plus seven preflop actions for an eight-seat hand. | Confirm card suits and table size from the original screenshot, fix the GTOW image path, then lock a snapshot. |
| H3971 | French hypothetical cash-game question stops at BB's pending decision; the stored line has five actions for six seats. | Decide whether the validator should accept a pending hero action in hypothetical queries, then add a focused behavior test. |

These are data and parser follow-ups, not validator allowlist entries. Do not
mark them resolved merely because the deterministic test suite is green.
