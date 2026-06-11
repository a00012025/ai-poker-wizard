# Handoff — effective_bb → 99.5% precision (effbb accuracy project)

**Date:** 2026-06-11 · **Worktree:** `~/ai-poker-wizard-effbb-accuracy` ·
**Branch:** `feat/effective-bb-accuracy` (NOT pushed; all work committed locally)
**Read this whole doc before continuing. The user does NOT accept the "86%
ceiling" conclusion and wants the WHY investigated rigorously — that is the live
task.**

---

## 0. The goal (unchanged, locked by the user)
On **hero-active** hands (hero did not fold preflop — the only hands we coach),
get **solver-depth-bucket precision ≥ 99.5% on EMITTED values**, coverage
maximized. Abstaining (`effective_bb=None`) is allowed/cheap (downstream degrades
None to a safe generic depth). Metric: `nearest_depth(parsed)==nearest_depth(gt)`.
Buckets `[100,80,60,50,40,35,30,25,20,17,14,12,10,9,8]`.
**Why it matters:** effective_bb is the SOLE input to the solver depth bucket
(`analyze_hand.py: depth = nearest_depth(hand["effective_bb"])`). Wrong bucket →
wrong GTO node → meaningless coaching. The user is emphatic: "no excuses."

## 0.1 THE LIVE QUESTION (what to do next)
I told the user the absolute ceiling on the single screenshot is **~86%
precision (at ~10% coverage)** and that 99.5% is information-theoretically
impossible from one mid-hand frame. **The user replied: "為什麼 absolute ceiling
是 86%? 我不接受" — they reject this and want it dissected.** I was about to run
the diagnostic below when they asked for this handoff. DO THIS FIRST in the new
session:

**Dissect the residual wrong EMITTED hero-active hands and answer: is 86% a real
accuracy ceiling, or partly a metric/GT artifact, and does a genuinely-clean
≥99.5% subset exist?** Specifically categorize each wrong emit:
1. **Bucket-boundary coin-flip** — gt_eff and p_eff are in ADJACENT buckets and
   BOTH sit within ~0.5–1bb of the shared cell boundary (e.g. gt 22.4→b20 vs
   computed 22.8→b25). The depth is essentially right; the metric is harsh at the
   edge. Count these — if a big chunk of "errors" are these, the *effective*
   accuracy is much higher than 74/86% and the ceiling claim softens. (Solver
   strategy at adjacent short buckets is nearly identical.)
2. **GT noise** — is the ground-truth effective_bb itself wrong/degenerate for
   some? Spot-check a sample against the HH source. The spec flagged a few
   degenerate GT rows (effective<1, busted seats).
3. **Genuine reconstruction error, right value present elsewhere** — fixable
   logic (the Phase-1/2/3 residual; start-vs-displayed, multiway invest).
4. **Truly input-bound** — correct value absent from OCR (digit misread).
Report the histogram of (bb-distance, bucket-distance, boundary-distance) for
wrong emits, and the size+precision of the cleanest subset by robust signals
(far-from-boundary + strong attribution + low pot-residual + name-pinned). The
Phase-4 "exhaustive combo tops at 86%" claim was the SUBAGENT's; verify it
yourself — it may have used insufficient features, or counted boundary near-miss
as error.

Also seriously evaluate **whether the bucket metric should allow ±1 adjacent
bucket near a boundary** (or a small bb tolerance) — that may be the honest
"accuracy" the product needs, and could change the whole verdict. Discuss with
the user before/after measuring.

## 0.2 The OTHER real path to 99.5% (raised with user, awaiting their steer)
Ground-truth effbb is computed EXACTLY from the **hand-history file**. The bot
ALREADY ingests HH uploads (`.txt`/`.zip` → `scripts/hh_parser.py` →
`analyze_hand_full`). For HH-sourced hands effbb is exact (~100%) — no OCR. The
screenshot is inherently lossy. **The guaranteed-99.5% route is HH-first.** I
proposed: land the screenshot improvement + make HH the accuracy route. The user
pivoted to challenging the 86% number instead, so this is unresolved. Worth
verifying the HH path computes effbb exactly end-to-end and the bot prefers HH
when available / nudges users to upload HH.

---

## 1. Where the code is (worktree `~/ai-poker-wizard-effbb-accuracy`)
All NEW, all committed on `feat/effective-bb-accuracy`:
- `scripts/effbb_metrics.py` — `depth_bucket(bb)`, `bucket_match(a,b)`,
  `hero_folded_preflop(gt)`, `classify_fault(...)`. Pure helpers.
- `scripts/effbb_cache.py` — builds `data/effbb_cache/cache.jsonl` (one ~3h parse
  with `EFFBB_CAPTURE=1`; captures `_compute_effective_bb` INPUTS + GT + ocr-hash).
- `scripts/effbb_eval.py` — **the iteration loop.** Replays the cache through the
  CURRENT `_compute_effective_bb` IN SECONDS; prints hero-active coverage/precision
  + fault breakdown + precision/coverage curve. NOTE: it recomputes from raw cached
  inputs and does NOT pass num_players (no GT leak), so it reflects current code
  even though the cache file predates the commits. Run:
  `python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl`
- `scripts/effbb_oracle.py` — Phase-0 ceilings (recoverable vs input-bound,
  attribution ambiguity, input-bound severity).
- `scripts/effbb_calibrate.py` — Phase-4 5-fold CV abstain calibration harness.
- `scripts/effbb_reread_probe.py` — Phase-3 VLM re-read de-risking probe
  (`--build/--gemini/--gpt-overlap/--score`; artifacts under gitignored
  `data/effbb_reread/`).
- `scripts/ocr/effbb_engine.py` — pure betting-state engine: action-order
  position assignment (`assign_positions`, `normalize_streets`), blind/ante
  inference (`infer_blinds`), per-position contribution (`accumulate_contributions`),
  decision-local relevant-opponent set, hard rules M1/M2/M3. Has its own copy of
  `POSITION_ORDERS`.
- **`scripts/ocr/n8_parser.py` `_compute_effective_bb`** (~:901-2130) — the
  production function. Now a **consensus orchestrator**: `_enumerate_layouts`
  (top-K position→seat geometric layouts), `_effective_bb_for_layout`
  (per-layout legacy reconstruction core), engine-vs-legacy consensus gate,
  structural abstain gate, hero all-in/stack≈0 reconstruction. Returns 3-tuple
  `(effective_bb, hero_starting_stack, confidence)`. Module-level
  `_LAST_EFFBB_FEATURES` capture for calibration (prod ignores it).
  Call site ~:2539/2885 unpacks the 3-tuple.
- `scripts/effbb_*` scratch: `scripts/_tmp*.py` (gitignored) from diagnosis —
  reusable, e.g. `_tmp2.py` (recoverable vs input-bound), `_tmp3.py` (per-hand
  detail), `_tmp4.py` (confidence-band recoverability), `_tmp5.py` (oracle
  ceiling), `_tmp_gpt.py` (the gpt-5.4-pro consult).
- **Plans:** `docs/superpowers/plans/2026-06-11-effbb-99-5-architecture.md` (THE
  living plan, has per-phase STATUS notes), supersedes `2026-06-10-...md`.
  Baseline note: `docs/superpowers/plans/effbb-baseline-current.txt`.

### Env flags (all on `_compute_effective_bb` path)
- `EFFBB_CAPTURE=1` — makes `_assemble_hand` stash `__effbb_inputs__` (cache build).
- `OCR_EFFBB_CONF_FLOOR` (default 0.7) — abstain below this confidence.
- `OCR_EFFBB_STRUCTURAL_GATE` (default ON; `=0` reverts to bare conf floor) —
  the Phase-4 structural abstain gate.
- `OCR_EFFBB_ENGINE_OPP` (default OFF) — engine single-opponent value override;
  measured net-NEGATIVE, kept off.
- `OCR_EFFBB_LAYOUT_MARGIN` — top-K layout score-margin keep.

### Data
- `data/hand_images/img/*.png` (7,183) and
  `data/pokercraft_corpus/ground_truth/ground_truth.jsonl` are **symlinked** from
  the main repo (`~/ai-poker-wizard/data/...`) — gitignored. GT row: top-level
  `hand_id` + nested `ground_truth` (`effective_bb, stacks_bb, preflop_actions,
  num_players, table_size, hero_position, hero_chips, ...`).
- `data/effbb_cache/cache.jsonl` (gitignored) — 7,183 rows, 6,969 with `inputs`.
  Built 2026-06-10; eval recomputes from `inputs` so it's current. **Rebuild if
  OCR/panel/position parsing changes** (staleness contract in the plan); the
  effbb LOGIC changes do NOT need a rebuild.

---

## 2. Current measured state (hero-active, 1,805 hands)
```
EMITTED precision / coverage (production default, conf floor 0.7 + structural gate):
  74.21% @ 61.0%        (817/1101 correct)
  conf>=0.9: 75.98% @ 53.5%
Journey: 47% @ 96% (broken baseline) → 66% @ 86% (logic rewrite) →
         70% @ 80% (Phase-2 consensus) → 70.94% @ 78.2% (Phase-3) →
         74.21% @ 61.0% (Phase-4 structural gate)
Claimed absolute ceiling (Phase-4 exhaustive feature search): ~86% @ ~10% cov.
```
Full regression suite: **620 passed, 1 failed** — the 1 failure
(`test_snapshot_l2_gto_H2494`, GTO text 79 vs 80 lines) is **pre-existing &
unrelated** (present on main, not in the effbb/analyze path). Confirm it still is.

Golden hand IDs (must stay green): `TM5873208532`→b30, `TM5862907992`→b17,
`TM5896148353` (deep-invested), `TM5875533783` (over-compute), `TM5863941844`
(divergence), H3514 (all-in), M1 `TM5863067496`→b20, M2 `TM5863067852`/`TM5863068088`
(walkover→BB), M3 `TM5863067607`, Phase-3 `TM5875510185`/`TM5873208901` (hero
all-in), Phase-2 straddle `TM5864409682`/`TM5866699022`→None.

---

## 3. What was tried, in order (so you don't repeat it)
- **Phase 0 (oracle, `effbb_oracle.py`):** 89.4% of hero-active hands are
  "recoverable" (a seat displayed stack, ±small invest, lands in GT bucket) but
  only **21.9% have a UNIQUE GT-bucket seat — 78% have ≥2 same-bucket candidate
  seats** → attribution must be betting-logic-driven, not value-driven. 10.6%
  input-bound (59% digit-slip explainable). ⚠️ The 89.4% used a PERMISSIVE
  ±invest search (0–11bb across all seats) — the VLM probe later found raw-value
  presence is ~62%, so 89.4% was optimistic. Re-derive "recoverable" carefully.
- **Phase 1 (`effbb_engine.py` + wiring):** betting-state engine. Reliably picks
  the right POSITION, but converting position→effective needs a noisy seat-stack
  READ, so the engine's value override is net-NEGATIVE (gated OFF). Only the M1
  uncalled-shove ceiling (pure panel read) applied. 65.9→66.3%.
- **Phase 2 (top-K layouts + bucket-consensus):** `_enumerate_layouts` + consensus
  orchestrator. Consensus genuinely sheds 74% wrong (real discriminator, not a
  trick — verified). 66→70% @ 80%, ~76% @ 60%. **KEY FINDING: only 40/502 errors
  are layout-DEPENDENT; 411/502 are layout-INDEPENDENT VALUE errors** — every
  hypothesis agrees on the SAME WRONG bucket (high consensus, wrong answer), so
  consensus is BLIND to them.
- **Phase 3 de-risk (VLM re-read, `effbb_reread_probe.py`):** **re-read is a DEAD
  lever.** VLM (Gemini 2.5-pro + gpt-5.4 vision) agrees with our OCR on ~95% of
  seat values; GT-bucket value present in VLM reads 63% ≈ our OCR 62%. 35% of
  wrong hands have the right value in NEITHER read because the error is STRUCTURAL
  (start-vs-displayed: screenshot shows mid-hand stack, GT wants starting =
  displayed+invested; hero all-in shows stack=0 + shove badge). **Dropped the
  planned CRNN (Phase 3) and VLM fallback (Phase 5) — user confirmed.**
- **Phase 3 redefined (reconstruction logic):** hero all-in/stack≈0 → use
  shove/commitment as starting stack. Pure abstain-quality win: 24 wrong→abstain,
  0 regressions; hero-stack≈0 subpop 85→91.6% precise. 70.94% @ 78.2%. Villain
  start-vs-displayed invest add was net-NEGATIVE (over-counts multiway) → NOT
  applied, handed to abstain.
- **Phase 4 (calibrated abstain, `effbb_calibrate.py`):** surfaced per-hand
  features (`_LAST_EFFBB_FEATURES`), 5-fold pooled CV. Shipped a structural gate
  (abstain if: geometry binding the engine doesn't confirm / hero all-in engine
  can't confirm / engine bucket dissents / method straddle; scoped OFF strong
  M1/M2 panel reads). 70.94%@78.2% → 74.21%@61%. **Claim: exhaustive feature-combo
  search tops at ~86% precision @ ~10% cov; 99.5% unreachable at any coverage.**
  ← THIS is what the user rejects and what you must re-examine (§0.1).

---

## 4. The core difficulty (current understanding)
The single action-panel screenshot shows **mid-hand displayed stacks**, but the
solver depth needs **starting stacks** (= displayed + invested). For deep-invested
/ all-in / complex-multiway lines, the invested chips must be reconstructed and a
single frame genuinely underdetermines them. The dominant residual error is
**internally-consistent VALUE error**: the attribution is right, but the
reconstructed value is wrong and every hypothesis agrees on it, so NO
consensus/ambiguity feature flags it. That's why abstain calibration plateaus.

**BUT** — open question (§0.1): how much of the residual is actually
bucket-boundary near-misses (metric artifact) vs GT noise vs genuinely wrong?
That was about to be measured and is the crux of the user's challenge.

---

## 5. Concrete next steps (priority order)
1. **Answer the user's challenge (§0.1):** write a diagnostic over emitted-wrong
   hero-active hands: per hand record gt_eff, p_eff, |bb diff|, bucket distance,
   min distance to a bucket boundary for both. Categorize: boundary-coin-flip /
   GT-noise / fixable-logic / input-bound. Report the histogram + the size &
   precision of the cleanest subset by robust signals. Decide if 86% is real or
   inflated by boundary/GT artifacts. Bring numbers to the user.
2. **Evaluate a tolerance metric** (±1 adjacent bucket near a boundary, or ±X%
   bb): measure precision under it. May be the honest product accuracy. Discuss.
3. **Audit GT** on a sample of wrong emits vs the HH source — rule out GT errors.
4. **Re-verify the 86% "exhaustive" claim** independently (the Phase-4 subagent's
   feature set may be incomplete; try adding boundary-distance-as-hard-gate +
   invested/displayed ratio + relevant-seat-OCR-stack-conf if available).
5. **The HH-first path (§0.2):** verify `hh_parser`→`analyze_hand_full` yields
   exact effbb; consider making HH the accuracy route + bot nudging HH upload.
6. If real headroom remains in logic: per-seat invested reconstruction for
   multiway (the net-negative add needs to be made surgical, not wholesale).

## 6. How to run things
```bash
cd ~/ai-poker-wizard-effbb-accuracy
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl   # fast frontier
python scripts/effbb_oracle.py --cache data/effbb_cache/cache.jsonl # ceilings
python scripts/effbb_calibrate.py                                   # Phase-4 CV
python scripts/regression_test.py                                   # full suite (slow; 620 pass/1 known fail)
python scripts/regression_test.py 2>&1 | grep -iE "effbb|engine"    # just effbb tests
```
Scratch python → `scripts/_tmp*.py` (gitignored, per CLAUDE.md). Run plainly
(no `set -a && source .env`) unless the script needs API keys (then it loads
.env itself). Keys in `.env`: `OPENAI_API_KEY` (gpt-5.4 / gpt-5.4-pro available),
`GEMINI_API_KEY`/`GEMINI_MODEL`.

## 7. External design input already gathered
- **gpt-5.4-pro consultation** saved at `/tmp/effbb_gpt_advice.md` (regenerate via
  `scripts/_tmp_gpt.py`; needs `timeout=3000` on the client — pro is slow).
  It recommended: position-first attribution, bucket-consensus emission, dedicated
  numeric reread, targeted VLM. We BUILT the first two; the reread/VLM were
  de-risked as ineffective on THIS data (numbers already read correctly). gpt's
  strongest pushback: **capture a hand-start frame** (user says unavailable) — the
  HH file is the equivalent and IS available (§0.2).

## 8. Relevant memories (verify before trusting — some predate this work)
- `effbb-bottleneck-is-logic` — confirmed & sharpened: bottleneck is
  reconstruction LOGIC + single-frame limit, not stack OCR.
- `effbb-gate-not-tweakable` — the old displayed×5 gate; we REMOVED it (replaced
  by in-function confidence/abstain).
- `validation-backlog-mostly-stale` — verify against the REAL screenshot/cache,
  never stored expected_json alone. Applies to any snapshot relock at the end.
- `multiway-preflop-reconcile` — snapshot suite flakes under concurrent runs on
  shared `.gto_cache`; baseline sequentially.

## 9. Project workflow rules (CLAUDE.md)
Worktree dev + PR (don't push without user OK — branch is unpushed). Every bug
fix needs a regression test. Use `scripts/_tmp.py` for ad-hoc python. Snapshot &
regression suites must pass before commit. The FINAL step (Task "Final") is: full
re-parse, snapshot relocks (verify real screenshots), validator None=soft-signal,
suite green, push + PR — NOT done yet (blocked on the 99.5% decision).

## 10. Status of the plan's tasks
Phases 0-4 done/committed. Phase 5 (VLM) DROPPED. "Final: validation + PR" pending
the user's decision on the 86%-ceiling challenge (§0.1) and HH-path (§0.2). The
branch is in a landable state (a real 47→74% precision win with safe abstain) but
does NOT meet the 99.5% bar from screenshots — which §0.1/§0.2 are about resolving.
