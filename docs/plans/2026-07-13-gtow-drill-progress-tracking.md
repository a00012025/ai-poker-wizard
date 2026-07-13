# Deferred: GTOW drill launch and practice-progress tracking

Status: **product discovery deferred**  
Recorded: 2026-07-13  
North Star link: training → evaluation → attributable real-game EV-loss decline

## Intent

When we next design the training loop, discuss one coherent feature rather
than adding a premature weekly before/after verdict:

1. Automatically create/open the prescribed GTOW drill.
2. Use an authenticated GTOW API or another verifiable GTOW surface to ingest
   practice sessions, scores, attempts, timestamps and the exact trained spot.
3. Connect practice exposure to later **online real-hand** decisions.
4. Evaluate whether treated spot families improve relative to matched untreated
   families (difference-in-differences), with sample size and uncertainty.

## Required product decisions before implementation

- What “automatic open” means: Telegram button, browser deep-link, scheduled
  reminder, or a controlled browser action. No unsolicited external action.
- Which GTOW practice endpoint/data is available and permitted, including auth,
  token refresh, rate limits and stable identifiers for drill/session/spot.
- Which score is meaningful: GTOW score, EV loss, action accuracy, completion,
  streak, or a combination. Frequency counts must not become the ranking metric.
- How a GTOW trained spot maps to our `spot_parent` and exact `spot_leaf`.
- Treatment definition: prescribed, opened, attempted, or completed; duration
  and number of decisions required before a family is considered treated.
- How to handle repeated treatments, overlapping families and practice outside
  the system.
- User-visible privacy/retention controls for practice history.

## Future data contract sketch — not approved schema

- `drill_prescriptions`: family, representative leaf, GTOW drill identity,
  prescribed/opened/completed timestamps.
- `gtow_practice_sessions`: external session id, drill id, started/ended,
  decisions attempted, GTOW score/EV metrics, raw archive reference.
- `training_exposures`: normalized family, exposure strength and confidence.
- `training_readbacks`: treated pre/post, matched-control pre/post, DiD estimate,
  confidence interval, n and verdict state.

## Evaluation guardrails

- Online real-hand ledger remains the outcome source; selective live hands never
  enter the aggregate KPI.
- No “有進步” verdict from a single treated before/after delta.
- Minimum observation window and sample thresholds must be agreed during product
  discovery; the current North Star note suggests at least four weeks.
- Always show treated n, control n, baseline, post-period and uncertainty.
- Valid states should distinguish insufficient data, treated-specific change,
  global shift, no detectable effect and regression.

## Explicitly out of scope for the current P0 implementation

- Calling GTOW practice APIs.
- Automatically opening external drills.
- Recording practice scores.
- Implementing the causal/DiD verdict.

Current-state warning: the existing weekly readback is still a descriptive
single-cohort before/after readout. Any arrow or「有進步」copy is **not** causal
evidence and must be replaced by the treated/control contract when this P1 is
designed.

The current P0 work only makes prescriptions honest: decision-level solver
depth, confidence filtering and hierarchical family diagnosis.
