# Initial coach model A/B — 2026-10-21

## Decision

Use `gpt-6-astra` at `reasoning.effort=low` as the default initial coach
narrator. Keep `OPENAI_COACH_MODEL` and `OPENAI_COACH_REASONING_EFFORT` as
operator overrides. Do not deploy until the PR is reviewed.

## Method

- Four saved real owner hands: H3943 (correct multistreet mix), H3932 (two
  decisions on both Flop and Turn), H3931 (Turn loss followed by off-tree
  River), and H3944 (AJs with zero reach after the prior 100% call).
- Rebuilt each teaching digest with current code and a read-only local solver
  cache loader. The same digest, user shorthand, prompt, fact audit, repair,
  and deterministic fallback were used for each model. `db=None` and
  `disable_tools=True`; no Telegram, database write, or live GTOW request.
  OpenAI generation was live.
- Two runs per hand/model, with the real final response recorded along with
  first-draft, repair, or fallback stage, audit violations, token usage, and
  wall time. H3944 also received three extra matched runs per Terra/Astra
  after clarifying the two range nodes in the prompt.
- One H3943 successor-node cache entry was unreadable locally; its response
  enrichment was omitted equally for all candidates. This is a coaching-flow
  replay, not a new image/HH parse or live GTOW solve.

## Observed results

| Configuration and test phase | First draft | Repair | Fallback | Final fact audit | Mean time | Estimated API cost/hand |
|---|---:|---:|---:|---:|---:|---:|
| Terra medium, original pipeline, 8 runs | 7 | 0 | 1 | 8/8 | 7.8s | $0.0209 |
| Luna high, existing 1,800-token cap, 8 runs | 0 | 0 | 8 no-text | 6/8 | 21.1s | $0.0015 |
| Luna high, 6,000-token cap, 8 runs | 4 | 2 | 2 | 8/8 | 46.9s | $0.0031 |
| Astra low, original auditor, 8 runs | 5 | 0 | 3 | 6/8 | 16.2s | $0.1161 |
| Astra low, selected final code, 8 runs | 7 | 1 | 0 | 8/8 | 16.8s | $0.0967 |

The original-auditor rows are not an apples-to-apples final-model ranking:
several grammatically correct outputs were rejected by a street-level or
actor-level false positive. Those drafts were preserved and re-audited offline
after each deterministic fix. The final Astra row is a fresh live-model E2E
run after the fixes. In three additional matched H3944 runs with the clarified
range-node prompt, Astra passed on the first draft 3/3 times versus Terra 1/3
first-draft and 2/3 repair.

The estimated costs use official standard text-token list prices and observed
prompt/completion tokens, including reasoning tokens. They are estimates,
not billing records ([Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra),
[Luna and Astra](https://developers.openai.com/api/docs/models)). The sample is small and intentionally weighted toward
historically difficult hands; it cannot prove a population-wide win.

## Pipeline changes validated by the replay

1. Distinguish Hero equity versus an all-in range from Villain's all-in
   frequency or response. Keep unsupported Villain action claims blocked.
2. Attribute same-street verdicts to their actual action and order, rather
   than transferring support from a prior check/call to a later losing call.
3. Keep an off-tree River ungraded without rejecting a correct explanation
   of the preceding Turn. Reject contradictory claims that grade that River.
4. Label the LJ call range as the **earlier** UTG-open node; never present it
   as LJ's response range to the later HJ jam.
5. On narrator failure, keep the scored decisions and group each street once
   in the deterministic fallback, rather than returning a generic no-data
   sentence. Do not explain a bad call using an alternative raise's bluff job.

## Remaining limits

- The final Astra run still required one factual repair out of eight. The
  fact audit and deterministic fallback remain necessary.
- Astra low cost approximately 4.6x the observed Terra cost per hand in this
  sample. The operator environment override is the immediate rollback path.
- The model/evaluator experiment is limited to four saved hands and two
  repeated generations each. Recheck fallback rate and player preference on
  a larger owner-snapshot sample before claiming a durable quality gain.
