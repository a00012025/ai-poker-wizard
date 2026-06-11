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

**Phase 2 STATUS (shipped, `_enumerate_layouts` + consensus orchestrator in
`_compute_effective_bb`):** built the top-K layout enumerator (both ring-walk
directions × `_candidate_rings` phantom-trim alternatives, weak confusable-name
score, score-margin keep), refactored the legacy core into
`_effective_bb_for_layout(_seat_map=, _disable_floors=)`, and made
`_compute_effective_bb` a consensus orchestrator. Gates: layout-bucket straddle
→ abstain (the engine breaks the tie when its relevant bucket matches one
layout); **engine-vs-legacy bucket consensus** is the primary discriminator (the
betting engine's decision-local relevant-seat bucket is an INDEPENDENT opinion —
corpus 76% precise when it agrees vs 52% on disagreement, so a geometry/heuristic
binding MUST earn engine confirmation or it abstains). Removed dead
`_seat_stack_for_position`; fixed HU postflop order (`['BB','SB']`).

**Measured hero-active depth-bucket frontier (1,805 cache, was 66.28% @ 86.1%):**
```
  conf>=0.0 : 69.95% @ 79.8%
  conf>=0.9 : 75.07% @ 60.0%
  conf>=1.0 : 76.44% @ 46.1%
```
NOT the 99.5% the plan projected. **Honest root cause (verified on the cache):**
bucket-consensus over the available inputs cannot exceed ~76% even at 46%
coverage. Of the 502 wrong emits at the 66% baseline, only **40 are
layout-DEPENDENT** (geometric direction changes the bucket — what consensus can
catch); **411 are layout-INDEPENDENT & recoverable** (the right number is in the
inputs but the binding value — hero's own displayed stack, the
uniquely-attributed contestant seat, or the panel all-in size — is itself wrong,
and EVERY hypothesis from those same inputs agrees on the same wrong bucket); 51
are input-bound. Even the conf=1.0 explicit-all-in tier is only 74.6% precise,
and 147/158 of its misses have the right number present elsewhere → the all-in
size/ceiling is misread or mis-attributed. An internally-consistent VALUE error
with no input redundancy to vote against → **a Phase-3 numeric-reread problem,
not an attribution one.** Phase 2 still delivers: (a) robust top-K attribution +
the abstain frontier Phase 4 calibrates on, (b) the engine-consensus
discriminator (+~5pp at 60% coverage), (c) the carry-over fixes. The 411
wrong-contestant / wrong-hero-stack residual is handed to Phase 3 (reread hero +
candidate-seat + shove crops) and Phase 5 (VLM on the residual).

### Phase 3 de-risking RESULT (2026-06-11) — RE-READ IS A DEAD LEVER
A whole-image VLM re-read experiment (`scripts/effbb_reread_probe.py`, Gemini
2.5-pro + gpt-5.4 vision, 60 wrong + 20 correct real hands) proved the re-read
premise WRONG: **the displayed numbers are already OCR'd correctly ~95% of the
time** (VLM agrees with our OCR; GT-bucket value present in VLM reads 63% ≈ our
OCR 62% — re-read surfaces nothing new). Whole-image VLM recovered only 8–35% of
wrong hands and *broke* up to 60% of correct ones. **35% of wrong hands have the
right value in NEITHER read** because the error is STRUCTURAL, not legibility:
(a) **start-vs-displayed** — the screenshot shows the *mid-hand* stack; the true
effective is the *starting* stack = displayed + invested (e.g. hero displayed
11.67 but GT 13.7); (b) **hero all-in shows stack=0 + shove badge** — effective =
the shove size, a reconstruction/attribution problem. The single action-panel
frame underdetermines the starting stack for deep-invested / all-in lines (no
hand-start frame available). **→ A trained CRNN and a VLM fallback are both
wasted effort. DROP Phase 5 (VLM) and the CRNN.** User-confirmed 2026-06-11.

### Phase 3 (REDEFINED) — reconstruction logic, not re-read
The residual is reconstruction logic + a hard single-frame information limit:
- **Starting-stack reconstruction:** ensure effective uses STARTING stacks
  (displayed + correctly-computed invested chips: posted blinds + antes + calls +
  bets), using the Phase-1 engine's per-position contribution model under the
  Phase-2 robust attribution + consensus gate. Fix the cases where invested chips
  are under/over-added (start-vs-displayed gap).
- **All-in ceiling / stack≈0 attribution:** when a contestant (hero OR a villain)
  is all-in showing ~0 + a shove badge, use the shove SIZE (read from the panel,
  it IS present) as that seat's committed/starting amount in min-over-contestants,
  correctly identifying whose shove it is.
- Turn the engine relevant-opponent value path ON (now that attribution is
  robust) gated by consensus; keep pot-conservation hard reject.
- **Validate** on the cache: precision up, the `selection`/`undershoot`/all-in
  residual down. Whatever cannot be confidently reconstructed → abstain (Phase 4).
  Realistic target after Phase 3+4: **99.5% precision @ ~50–65% coverage.**

**Phase 3 STATUS (shipped, 2026-06-11) — hero side recovered, villain side
confirmed unrecoverable.**

*Implemented (net win):* a **hero all-in / stack≈0 starting-stack
reconstruction** in `_effective_bb_for_layout`. When hero's displayed stack
reads ~0 (hero shoved, or called a villain shove for their whole stack), hero's
STARTING stack is exactly hero's permanent commitment. The betting engine's
decision-local per-position hero contribution is an INDEPENDENT reconstruction
of that amount; we prefer it when it agrees with the legacy displayed+walk
estimate on the depth bucket, and ABSTAIN (confidence cap) when they disagree
and the displayed read is uninformative (the partial/misread shove sticker is
single-frame unrecoverable). Measured effect on the 1,805 hero-active cache: **0
regressions, 0 wrong→right, 24 wrong→ABSTAIN** — purely an abstain-quality win.
Frontier moved **69.95% @ 79.8% → 70.94% @ 78.2%**; conf≥0.9 **75.07% @ 60.0% →
76.13% @ 58.7%**; conf=1.0 **76.44% @ 46.1% → 77.74% @ 45.0%**. The
hero-stack~0 sub-population (277 hands) emits at **~91.6% precision @ 77%
coverage** (was ~85% @ 90%).

*Confirmed not improvable (the single-frame VALUE limit):* turning the engine
relevant-opponent value override ON (`OCR_EFFBB_ENGINE_OPP=1`) is still
net-negative (69.63% vs 69.95%) — the engine reads a noisy seat, as Phase 1/2
found. Replacing the legacy villain invest with the engine's per-position
contribution (or with a panel-position contribution that DOES fix the canonical
start-vs-displayed hand TM5864261096 → 13.7) is net-NEGATIVE wholesale
(~107 better / ~510–539 worse): the same change that fixes one start-vs-displayed
villain over-adds invested chips on ~5× as many multiway hands where a villain
folded mid-street or the panel position drifts. So the **villain start-vs-displayed
class is layout-INDEPENDENT VALUE error with no input redundancy to vote against
it** — exactly the residual Phase 2 measured (411 wrong emits) and which the
Phase-3 de-risking proved is NOT a legibility problem. These stay OFF; the
residual is handed to **Phase 4 calibrated abstain**, not forced to a guess.

*Genuinely unrecoverable examples (→ Phase 4 abstain):* `TM5874977534` (hero
all-in sticker shows 1.24 for a true 11.2 stack — both reconstructions agree on
the wrong tiny value, no internal signal; stays a wrong emit); `TM5864261096`
(real start-vs-displayed villain under-add, but the engine MIS-assigns its
positions on panel-order divergence so neither path recovers 13.7);
`TM5863941899` (hero displayed 1.0 is itself an OCR corruption).

### ~~Phase 5 — targeted VLM fallback~~ DROPPED (de-risked ineffective 2026-06-11)

### Phase 4 — calibrated abstain (hard gates + selector)
- Hard gates: decision type resolved; position set unique; no arithmetic
  failure (pot residual < ε); layout ambiguity doesn't cross buckets; relevant
  OCR ambiguity doesn't cross buckets.
- Selector features: layout score margin / top-K entropy; relevant-field OCR
  confidence & OCR-vs-reread agreement; bucket-boundary distance; pot residual;
  decision class; None-name / geometry-only flags.
- **5-fold pooled CV** over 1,805 hero-active; pick threshold maximizing
  coverage s.t. empirical precision ≥ 99.5% (binomial lower-bound check). Lock.

**Phase 4 STATUS (shipped, 2026-06-11) — 99.5% PROVEN UNREACHABLE; shipped the
precision-maximizing structural gate honestly.**

*Built:* per-hand abstain-feature surfacing in `_compute_effective_bb`
(`_LAST_EFFBB_FEATURES` + `_effbb_last_features`, all returns routed through a
`_finish()` finalizer; pure calibration instrumentation, prod reads nothing) +
a bucket-cell boundary-distance fragility helper + the calibration harness
`scripts/effbb_calibrate.py` (5-fold pooled CV split by hand, Wilson 95% LB,
interpretable hard gates, no hand-ID literals).

*The honest frontier (1,805 hero-active cache, the deliverable, NOT a vanity
number):* **99.5% point-precision is UNREACHABLE at any usable coverage.** An
exhaustive small-combo search over every surfaced feature tops out at an
**absolute precision ceiling of ~86% @ ~10% coverage** (~78% @ 45%). The
residual wrong emits are layout-INDEPENDENT, internally-consistent VALUE errors
(hero/villain stack misread, start-vs-displayed) — EVERY reconstruction
hypothesis agrees on the same wrong bucket, so no consensus/ambiguity feature
separates them from correct emits (confirms the Phase-2/3 finding). The
calibration features (boundary_dist, pot_residual, engine-agree, etc.) measure
ambiguity/consistency, but these errors are confidently consistent. 5-fold CV
@ target 99.5% finds no train operating point → falls back to the
max-precision point (~77% @ ~27% held-out).

*Shipped (the defensible operating point):* a calibrated **structural abstain
gate** (`OCR_EFFBB_STRUCTURAL_GATE`, default ON; `=0` reverts to the bare conf
floor → 70.9% @ 78.2%). It abstains: geometry/heuristic binding the betting
engine didn't confirm; hero all-in (displayed ~0) the engine can't confirm;
independent engine bucket dissents; floors-on vs stack-only straddle — the
broad two SCOPED OFF strong M1/M2 panel reads (base_conf≥0.95) so correct
M1/M2 emits survive. **Measured (effbb_eval, default floor 0.7): 70.94% @ 78.2%
→ 74.21% @ 61.0%** emitted precision; 5-fold held-out CV is stable across seeds
(fixed interpretable gate, not per-fold-tuned). Abstaining is cheap (None →
safe generic solver depth). RECOMMENDATION: ship the gate (a real +3.3pp
precision win) and STOP — 99.5% is not attainable from the single action-panel
frame; the only path to it is a second frame (hand-start screenshot) the user
confirmed does not exist, or a fundamentally better numeric OCR (Phase-3
de-risking proved re-read does not help). Phase 5 (VLM) stays DROPPED.

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

---

## Phase 6 (2026-06-11 PM) — the "86% ceiling" challenge: VERDICT + logic fixes

**STATUS: DONE.** The user rejected the Phase-4 "absolute ceiling ~86%" claim
and asked for it to be dissected. The dissection (scripts/_tmp_diag*.py over
the 284 wrong emits) proved the claim WRONG in the way that matters:

* The 86% number was a ceiling on **abstain-calibration given the then-current
  reconstruction**, not a ceiling on accuracy. No feature separates the
  internally-consistent value errors — true — but the errors themselves were
  largely FIXABLE LOGIC, not input-bound noise:
  - boundary coin-flips: 18% of wrong emits (metric harshness, depth ~right)
  - GT noise: **0** programmatic inconsistencies (replayed GT vs hh_parser
    defs); 1 deep HH audit confirmed GT correct; 2 degenerate rows
  - truly input-bound: only **39/1101 = 3.5% of emitted** → in-principle
    ceiling on emitted ≈ 96%, far above 86%
  - the rest: attribution/selection logic vs the GT definition

**Fixes landed (all measured on effbb_eval, hero-active):**
1. *Hero-stream matched-shove floor*: hero's unnamed raise-then-call rows were
   keyed per-index so a called villain jam never registered as matched
   (TM5863575308); + blind credit for BB/SB callers when the Blinds column is
   empty (TM5875127705); + strong_panel_read extended to base_conf 1.0 panel
   all-in bindings so the gate doesn't method-straddle-abstain them.
2. *GT-aligned preflop-only relevant set* (engine + core behind-hero bound):
   a hand that truly ends preflop is bound by every seat acting after hero
   (incl. folders behind) + earlier voluntary entrants — matches hh_parser's
   in_pot definition. Guards: no preflop all-in (matched jam = board ran out →
   postflop def; uncalled jam = authoritative M1 ceiling), binder seat must be
   named or panel-present. M2 set widened [BB] → all seats behind.
3. *Poker-rules legality guard on all-in evidence*: an "All-In" row that does
   not exceed what a covering player already committed, followed by that
   player's fold, is a misparsed raise (TM5878838751: Bet 9.0 → "All-In 1.0" →
   Fold bound a 44bb spot at 1.0).
4. *Dropped the hero-near-zero structural-abstain clause*: post-fixes that
   slice measures 81% marginally precise (ABOVE the emitted average) — the
   clause was costing ~4pp coverage for negative precision value
   (scripts/_tmp_gate.py per-condition audit; engine-dissent stays at 48%
   marginal and keeps earning its abstains).

**Frontier: 74.21% @ 61.0% → 76.83% @ 71.2%** (correct emits 817 → 988, +21%).
Gate-off (ungated): 70.9%@78.2% → 74.07%@79.3%. conf≥0.9 band: 79.9%@60.9%.
Boundary-tolerant (±1 bucket when both values within max(1bb,4%) of the shared
cell edge): **81.4%**; any-adjacent: 86.6%.

**Residual error mass** (298 wrong emits): ~20% boundary coin-flips, ~17%
input-bound, the rest idiosyncratic seat-value misreads binding wrongly in
either direction (no separating feature; deep-undershoot/no-jam implausibility
probes measured ≈ average marginal precision — not actionable). Tested and
rejected: attribution-free named-min bound (−140 net), engine villain-invest
add (net-negative, unchanged), committed>60%-no-jam abstain (15 hands @ 47% —
too small/borderline).

**Open decisions for the user:**
* 99.5% point-precision from the single frame remains out of reach (best
  clean slice ~80% @ 61%); the honest routes stay (a) the boundary-tolerant
  metric — 81.4% today — if adjacent-near-boundary counts as correct for the
  product, and (b) HH-first for exactness (hh_parser computes effective_bb
  from the HH directly — the GT generator itself — so HH uploads are exact by
  construction; bot already ingests them).
* Snapshot suite: 16 pre-existing failures (hero_hand/board/action-size/
  Call→Limp formatter renames — stale expected, none effective_bb-related).

### Phase 6 addendum (same day, later)
Fifth fix landed: *behind-hero bound under an uncalled hero jam* (named seats
only; misread-folder golden TM5866594919 stays green). **Final frontier:
78.19% @ 70.4%** (conf≥0.9 band: 80.7% @ 60.3%). Correct emits 817 → 993
(+21.5%) vs the Phase-4 operating point at +9.4pp coverage.

Additionally tested and REJECTED (measured net-negative):
- per-villain engine-contribution invest in the name-matched branch (+44
  abstain→wrong: legacy values converge to the engine read, so the dissent
  gate loses its independence — keep the shared capped_invest there);
- named behind-seat bound under a villain's UNCALLED jam (fix 10 / break 27 —
  the M1 ceiling is authoritative);
- per-name own-rows invest as the villain estimate (fix 19 / break 203 —
  panel call sizes are increments with inconsistent semantics).

Suite: 625 passed / 1 pre-existing failure (H2494, unrelated). Snapshot suite:
16 pre-existing stale-expected failures, none effective_bb-related.
