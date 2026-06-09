# Effective_bb Accuracy — Design Spec

- **Date:** 2026-06-10
- **Branch:** `feat/effective-bb-accuracy`
- **Status:** Approved design → ready for implementation plan
- **Author:** Harry + Claude

## 1. Problem

`effective_bb` is the effective stack depth in big blinds. It is the **sole
source of the solver depth bucket**: `analyze_hand.py:1158` does
`depth = nearest_depth(hand["effective_bb"])`. A wrong `effective_bb` → wrong
solver depth → wrong GTO node → wrong coaching. The MTT depth buckets are
`AVAILABLE_DEPTHS = [100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8]`
(dense at the short end, where it matters most).

Today `effective_bb` is computed by `_compute_effective_bb` in
`scripts/ocr/n8_parser.py`, which reconstructs each player's STARTING stack from
the N8 screenshot (displayed = remaining = starting − invested) and then a
sanity gate (`scripts/ocr/n8_parser.py:1778`, `effective_bb > hero_displayed*5`)
nulls implausible values. A prior negative result
([[effbb-gate-not-tweakable]]) proved the gate can't be loosened without
exposing garbage; the real fix is the reconstruction itself.

### 1.1 Measured baseline (2026-06-10 diagnostic)

A 400-hand strided sample (pure OCR via `parse_n8_screenshot`, compared at the
depth-bucket level to HH-derived ground truth, which carries true per-seat
`stacks_bb`):

| Population | n (of 400) | coverage | bucket-precision |
|---|---|---|---|
| Hero **folded** preflop (bot does not coach) | 287 (72%) | 82% | 32.6% |
| Hero **active** (coached hands) | 110 | 75% | **~52%** (±~9%, n small) |

Fault decomposition of the 39 hero-active wrong-bucket hands, against trusted
HH stacks:

- **31/39 overshoot** (`p_eff > gt`) — a one-directional, systematic bias.
- **21/39: `gt ≈ a shorter active villain`** — OCR returned a larger stack
  (often hero's own) and failed to take `min` with the shorter active villain.
  E.g. TM5901451744 (hero 109.9bb vs active caller 30.7bb → true 30.7, OCR
  109.9); TM5873208532 (hero 36.2 vs callers 29.3/30.4 → true 29.3, OCR 36.2).
- **Several over-compute past hero's OWN start** (49.0 when hero had 29.0; 43.5
  when hero had 12.6) — investment over-add.
- **8/39 undershoot** — dropped/short reconstruction.
- **Stack-digit OCR misreads are a minority.**

**Conclusion: the bottleneck is downstream reconstruction LOGIC, not OCR input.**
The stack numbers are largely read correctly; `_compute_effective_bb` (1) does
not take `min` with the shorter active villain, (2) over-adds investment past
hero's own stack, and (3) sometimes undershoots. This justifies a logic rewrite
plus an input-level cache for fast iteration. The earlier "~1% problem" framing
referred only to the gate's false-nulls and badly understated the real error.

## 2. Goals & Success Metrics

Three locked decisions:

1. **Metric = solver-depth-bucket match.** A hand is "correct" iff
   `nearest_depth(parsed_eff) == nearest_depth(gt_eff)`. Exact-bb equality is
   neither necessary nor measured (21bb vs 23bb both → 20bb bucket).
2. **Target = precision on emitted ≥ 99.5%**, maximize coverage. When the
   computation is not confident it returns `None` (abstain); downstream already
   degrades `None` to `max(max_raise·10, 20)` (`analyze_hand.py:1144-1156`), so
   abstain is a soft signal, **not** a hard validator warning. "Wrong bucket on
   an emitted value" is the number we drive to ~zero.
3. **Population = hero-active hands only** (hero did not fold preflop). The 72%
   hero-folded majority is excluded — the bot does not coach those, and their
   `effective_bb` does not affect output quality.

**Acceptance:** on the hero-active corpus, emitted bucket-precision ≥ 99.5%,
with coverage reported and an explicit operating point chosen. The
precision/coverage frontier is a deliverable, not just the single number.

## 3. Design

### Component 1 — `_compute_effective_bb` rewrite

Change the signature to return `(effective_bb, confidence)` (or
`(None, confidence)` to abstain) and **remove** the `displayed×5` gate
(`n8_parser.py:1760-1779`); the confidence-based abstain subsumes it and also
fixes the H3522 deep-invested false-null.

Algorithm:

1. **Per-player starting-stack estimators.** For hero and for each active
   (non-folded) villain, estimate `start = displayed_remaining + investment`
   using two independent investment estimators:
   - **(a) action-walk** — the current per-entry reconstruction.
   - **(b) pot-header** — derive each player's matched contribution from the
     pot deltas between street headers.
   For a player who is all-in, a third estimator is the all-in amount itself.
2. **Selection fix.** `effective = min(hero_start, min over ALL active villains
   of villain_start)`. This is the single biggest fix (the 21/39 cases).
3. **Over-compute guard.** Bound each player's investment by pot conservation:
   reconstructed `start` may not exceed `displayed + (that player's maximum
   possible contribution given the pot)`. An estimate implying
   `start > displayed + pot` is rejected.
4. **Confidence = estimator agreement on the effective player.** If estimators
   (a) and (b) for the player that determines the effective stack land in the
   **same depth bucket**, emit with high confidence. If they diverge across a
   bucket boundary, or the effective player is near-all-in with ambiguous
   sizing, **abstain** (`None`).

`hero_starting_stack` continues to be returned for downstream
`_preflop_allin_effective_bb` use.

### Component 2 — input-level feature cache (`scripts/effbb_cache.py`, new)

A standalone script (chosen over extending `ocr_benchmark.py`, which re-parses
fresh every run). One full ~3h parse over the 7,183-image corpus that captures,
per hand:

- **Inputs to `_compute_effective_bb`:** `columns`, `all_stacks`,
  `named_stacks`, `hero_position`, `hero_displayed`.
- **Ground truth:** `effective_bb`, `stacks_bb`, `preflop_actions`,
  `num_players`, `hero_position`.
- **A hash of the relevant OCR modules** (`panel_parser.py`, `table_parser.py`,
  and the position-assignment parts of `n8_parser.py`) for staleness detection.

Caching the **inputs** (not the outputs) is the property that makes Phase 1
iterate in seconds: each candidate `_compute_effective_bb` is re-invoked over
cached `columns`/stacks without re-OCR. Output: a single JSONL/pickle under
`data/effbb_cache/`.

**Staleness contract** (answers "when must we re-run?"):

| Change | Re-run cache? |
|---|---|
| `_compute_effective_bb` logic / abstain | **No** (it is downstream of the cache) |
| `nearest_depth` buckets / eval thresholds | **No** (applied in the harness) |
| Stack-number OCR (`table_parser` stacks) | **Yes** |
| Panel parse (`panel_parser`: action/size/pot) | **Yes** |
| Position/name assignment (`n8_parser._build_streets`, hero loc) | **Yes** |
| Card CNN / suit detection | **No** (effbb does not read cards) |

The harness warns automatically when the stored module hash no longer matches.
New corpus images are appended incrementally, not a full re-run.

### Component 3 — eval harness (`scripts/effbb_eval.py`, new)

Loads the cache, runs the current or a candidate `_compute_effective_bb` over
every cached hand, and reports in **seconds**:

- hero-active **coverage** and **bucket-precision** (the acceptance numbers),
- the **4-way fault breakdown** (overshoot / shorter-villain selection /
  over-compute-past-own / undershoot),
- a **precision/coverage curve** as the confidence threshold varies, so the
  operating point is chosen with eyes open,
- both populations side-by-side (hero-active headline; full-corpus for context).

This harness is the Phase-1 iteration loop and the Phase-2 acceptance gate.

## 4. Phases

- **Phase 0 — measure + infra.** Build `effbb_cache.py` (one 3h parse) and
  `effbb_eval.py`. Produce the precise hero-active baseline (firming the rough
  ~52%) and category sizes. Deliverable: baseline report.
- **Phase 1 — logic rewrite.** Implement the Component-1 algorithm. Iterate
  against the cache (seconds) to **precision ≥ 99.5%, coverage maximized**.
  Deliverable: the rewritten function + the precision/coverage curve + chosen
  operating point.
- **Phase 2 — validate + land.** One full re-parse to confirm cache-predicted
  numbers hold end-to-end. Wire `None` → soft fallback (already mostly present;
  ensure no hard validator warning). Add regression unit tests (multiway
  min-selection, over-compute bound, abstain-on-divergence) + snapshot
  re-locks for any shifted hands + corpus gate. Open PR.

## 5. Testing & Acceptance

- **Unit (`scripts/regression_test.py`):** one test per fault class —
  multiway `min`-with-shorter-villain; over-compute bound rejects
  `start > displayed + pot`; abstain when estimators diverge across a bucket.
- **Harness acceptance (`effbb_eval.py`):** hero-active emitted bucket-precision
  ≥ 99.5% at the chosen operating point; coverage reported.
- **Snapshot suite (`snapshot_test.py`):** no regression; re-lock hands whose
  `effective_bb` legitimately shifts (e.g. H3522 now emits 29.4 instead of
  None).
- **Corpus gate:** validator backlog does not grow; `EFFECTIVE_BB` flags that
  were deep-invested false-nulls clear.

## 6. Risks & Open Questions

- **99.5% may not be reachable at high coverage.** The undershoot class (8/39)
  may have an irreducible OCR-input floor. Mitigation: fix the systematic logic
  faults first, then abstain the residual to hold the precision bar; if coverage
  drops too far, revisit with a targeted OCR fix for the residual. The
  precision/coverage curve makes this tradeoff explicit before we commit.
- **Sample noise.** The ~52% baseline is n=110; Phase 0's full cache replaces it
  with the real number. Do not over-index on 52% until then.
- **GT edge cases.** A few GT rows are degenerate (`effective_bb < 1.0`, busted
  seats) and are excluded from the metric.
- **Multiway depth re-derivation.** `analyze_hand.py` also re-derives an
  SPR-based depth postflop (`:959-982`); the hand-level `effective_bb` still
  sets the preflop/base depth, so it remains the right optimization target, but
  Phase 2 should confirm no double-counting interaction.

## 7. Out of Scope

- Card / suit / board OCR (does not affect `effective_bb`).
- Hero-folded-preflop hands (excluded population).
- The text-input parse path (`effective_bb` is screenshot-specific here).
- A learned confidence calibrator (kept as fallback only; prior calibrator-wall
  evidence [[ocr-99-d-a-calibrator-wall]] argues against it as the primary
  mechanism).

## Related memories

[[effbb-gate-not-tweakable]], [[validation-backlog-mostly-stale]],
[[multiway-real-structure-hybrid]], [[multiway-preflop-reconcile]],
[[hand-rules-validator]].
