# effective_bb → 99.5% precision: revised architecture & phased plan

**Date:** 2026-06-11 · **Branch:** `feat/effective-bb-accuracy`
**Supersedes:** `2026-06-10-effective-bb-accuracy.md` (that plan assumed the fix
was pure reconstruction logic; corpus evidence + a gpt-5.4-pro design review
showed the real fix is position-first attribution + consensus emission + image
re-read).

## Goal (locked)
On **hero-active** hands (hero did not fold preflop — the only hands we coach),
**solver-depth-bucket precision ≥ 99.5% on emitted values**, coverage maximized.
Abstaining (`effective_bb=None`) is allowed and cheap (downstream degrades None
to a safe generic depth). Metric = `nearest_depth(parsed)==nearest_depth(gt)`.
Buckets: `[100,80,60,50,40,35,30,25,20,17,14,12,10,9,8]`.

## Why the old approach capped out (verified on the 7,183-hand cache)
Logic rewrite reached **66% precision @ 86% coverage** (from 47%). Oracle
analysis (`scripts/effbb_oracle.py`) on 1,805 hero-active hands:
- **89.4% recoverable** — the correct-bucket stack IS in the OCR inputs; the
  greedy attribution picks the wrong seat. **Only 21.9% of recoverable hands
  have a UNIQUE GT-bucket seat — 78% have ≥2 same-bucket candidate seats**, so
  attribution must be driven by *who is actually live at hero's decision*
  (betting logic), not by stack values.
- **10.6% input-bound** — correct value absent from OCR (digit misread). **59.4%
  of it is digit-slip explainable** (×10 / decimal / ±10 of an OCR'd seat) → a
  dedicated numeric re-reader recovers most; only ~2% have no usable stack.

So even a perfect attribution oracle caps at **~95.9% precision @ 86% coverage**
on current OCR. Reaching 99.5% requires: near-perfect attribution + image
re-read of the relevant numeric crops + consensus-based abstain. **Decision: full
stack (Phases 0–5), target 99.5% @ ~78–85% coverage.** No hand-start frame is
available (user confirmed) — we reconstruct from the single action-panel shot.

## Architecture (replaces greedy seat attribution)
1. **Position-first, seat-second.** A betting-state engine maps panel *actions →
   logical positions* from poker rules + action order (NOT names). Then map
   *positions → geometric seats* as a top-K assignment (Hungarian + Murty), names
   as weak confusable-normalized evidence only.
2. **Decision-local relevant-opponent set.** Track who is live at hero's
   decision; hard rules: uncalled-shove ceiling, walkover→BB, multiway-live.
3. **Interval / bucket-consensus emission.** Compute the bucket across all
   plausible (layout × OCR-read) hypotheses; emit iff the union stays inside one
   bucket cell, else abstain. This *is* the confidence signal (scalar
   "confidence" was just a provenance tag — its top band was only 75%).
4. **Pot conservation = hard rejector** of impossible arithmetic, not a seat
   selector.
5. **Numeric re-read** of only the 2–4 relevant crops: a dedicated CRNN/CTC
   digit reader (auto-labeled from HH) first; **targeted per-crop VLM** (one
   crop → one number, JSON) only on the residual.

Bucket cell boundaries (for interval emission):
`100:[90,∞) 80:[70,90) 60:[55,70) 50:[45,55) 40:[37.5,45) 35:[32.5,37.5)
30:[27.5,32.5) 25:[22.5,27.5) 20:[18.5,22.5) 17:[15.5,18.5) 14:[13,15.5)
12:[11,13) 10:[9.5,11) 9:[8.5,9.5) 8:[0,8.5)`.

## Infra already in place (keep)
- `data/effbb_cache/cache.jsonl` (7,183 hands, OCR inputs + GT; seconds to replay).
- `scripts/effbb_eval.py` (coverage/precision/faults + PR curve; replays cache).
- `scripts/effbb_metrics.py` (`depth_bucket`, `bucket_match`, fault classes).
- `scripts/effbb_oracle.py` (Phase-0 ceilings).
- `_compute_effective_bb` already returns a 3-tuple `(eff, hero_start, conf)`;
  has min-over-villains, pot-bound guard, uncalled-jam ceiling, geometry ring.
  Phases 1–2 rebuild attribution; the hard-rule pieces are a starting point.

## Phases (each: implement → validate on cache → commit; review per task)

### Phase 1 — betting-state engine + decision-local hard rules
- Preflop & postflop action-order engine for table sizes 2–9 (`POSITION_ORDERS`),
  folded/all-in skipping, raise-reopen, blinds/antes inferred from the preflop
  pot header. Map each panel row → actor position.
- Relevant-opponent set computed at **hero's decision** (not showdown).
- Hard rules: M1 uncalled-shove ceiling (effective ≤ shove size, read from
  panel All-In size); M2 walkover → min(hero, BB seat); M3 multiway live-set.
- **Validate:** unit tests on labeled M1/M2/M3 hands; `effbb_eval` precision up,
  selection+undershoot faults down. Expected ~70–76% @ ~86%.

**Phase 1 STATUS (shipped, `scripts/ocr/effbb_engine.py` + wiring):** the pure
engine is built (action-order assignment with `normalize_streets` hero-mislabel
scrubbing, blind/ante inference, per-position contribution, decision-local
relevant set, M1/M2/M3). Engine unit + M1/M2/M3 cached goldens green.
**Measured: 65.92% → 66.28% @ 86.1% hero-active — NOT the 70–76% the plan
projected.** Honest root cause: the engine selects the right *position*
reliably, but converting position → effective still needs a *seat-stack read*,
and with current attribution that read is noisy enough that letting the engine
*override* the legacy reconstruction is **net-negative** (76 regress / 26 gain
on the cache) — it corrects toward a misread seat. So the engine's
single-opponent value override is gated OFF by default (`OCR_EFFBB_ENGINE_OPP=1`
to A/B); only its **M1 uncalled-shove ceiling** (a pure panel read, no seat
dependency — and now using the shover's TOTAL contribution, not the bare shove
size) is applied, which is the small net win. This *confirms the oracle finding*
(78% of recoverable hands have ≥2 same-bucket candidate seats): the precision
ceiling is **seat attribution**, not position logic. The gain the plan budgeted
to Phase 1 actually lands in **Phase 2** (robust position→seat layout) — the
engine is the prerequisite that makes Phase 2's position choice trustworthy.
TM5863067607's final bucket is a concrete Phase-2 example: the engine picks the
SB limper correctly, but the SB sticker is OCR-misread (2.9 vs ~17.4), so every
seat-read path lands on 2.9 until Phase 2/3.

### Phase 2 — top-K layout + bucket-consensus emission
- Fit geometric ring templates `(μ_x,μ_y,Σ)` per (table_size, ring-slot) from a
  clean name-aligned subset (bootstrap + spot-verify).
- Top-K position→seat layouts (Hungarian + Murty); names as weak score.
- Pot-conservation hard reject.
- Emit iff all plausible hypotheses' effective intervals ⊆ one bucket cell, else
  abstain. **Validate:** PR curve with/without consensus gating, by decision
  class. Expected 99.5% @ ~40–65% (logic-only operating point).

### Phase 3 — dedicated numeric re-read OCR
- Extract seat-stack & action-size crops; auto-label from HH
  (`displayed = starting − invested`; action sizes aligned by street/order).
- Train a small CRNN/CTC (charset `0123456789.`) + preprocessing ensemble +
  a second cheap reader; accept on agreement (exact or same bucket-impact).
- Re-read only the 2–4 relevant crops (hero + candidate contestant seats +
  explicit shove/raise rows). **Validate:** held-out digit accuracy + downstream
  bucket precision/coverage. Expected 99.5% @ ~68–78%.

### Phase 4 — calibrated abstain (hard gates + selector)
- Hard gates: decision type resolved; position set unique; no arithmetic
  failure (pot residual < ε); layout ambiguity doesn't cross buckets; relevant
  OCR ambiguity doesn't cross buckets.
- Selector features: layout score margin / top-K entropy; relevant-field OCR
  confidence & OCR-vs-reread agreement; bucket-boundary distance; pot residual;
  decision class; None-name / geometry-only flags.
- **5-fold pooled CV** over 1,805 hero-active; pick threshold maximizing
  coverage s.t. empirical precision ≥ 99.5% (binomial lower-bound check). Lock.

### Phase 5 — targeted VLM fallback (residual only)
- Trigger only when one relevant crop is unreadable / bucket disagreement comes
  from a single numeric field. Crop+enlarge → VLM "read this one number, JSON
  only" → 2-pass/2-model consensus → accept only if agrees with OCR or across
  passes AND bucket-stable, else None. **Validate:** on cache. Expected 99.5% @
  ~78–85%.

### Final — validation + land
Full re-parse; snapshot relocks (verify the real screenshot first per
`validation-backlog-mostly-stale`); validator `None` = soft signal; regression
suite green; push + PR.

## Acceptance & honesty
The deliverable is the **precision/coverage frontier**, not a single number.
99.5% precision is the hard bar; report the coverage achieved at each phase on
the cache. Never fake the target with an absurd abstain floor — abstain only
where a hard gate or genuine bucket-straddle fires.

## Validation discipline (project rule)
Verify every "fix" against the REAL screenshot / cache, never the stored
expected_json alone ([[validation-backlog-mostly-stale]]). Snapshot suite flakes
under concurrent runs on shared `.gto_cache` — baseline sequentially
([[multiway-preflop-reconcile]]).
