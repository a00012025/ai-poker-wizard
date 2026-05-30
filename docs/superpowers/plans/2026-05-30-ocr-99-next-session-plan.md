# OCR 99%@95% — Next-session implementation plan (precision push)

> **Goal (unchanged): 99% precision @ 95% recall on the 718-hand PokerCraft test set.**
> Read this together with `2026-05-30-ocr-99-recall-precision-findings.md` (measured
> baseline + what's already ruled out) and the D-c handoff
> `2026-05-30-ocr-99-dc-handoff.md`. Work lives on branch
> `worktree-ocr-99-hero-recall` (PR #34). This doc is the actionable roadmap +
> the verbatim external-model consult that produced it.

---

## Where we are (measured, test set 718)

| pipeline stage | precision | recall |
|---|---|---|
| deterministic OCR alone (default gate) | **97.2%** | 79.8% (573/718, 557 correct) |
| calibrator forced to 95% coverage | 86.9% | 95% |
| full pipeline (OCR + Gemini fallback) @ ~100% recall | **84.3%** (naive) / **86.5%** (best) | ~100% |
| **TARGET** | **99%** | **95%** |

Gap ≈ **12–13 pp of precision** at high coverage. **Recall is solved** (the Gemini
fallback recovers a parsable hand for 81/81 parse_none). The bottleneck is
**precision** on a genuinely-hard tail (tiny WIN-overlay-occluded hero cards;
collapsed multiway all-in action rows). Error decomposition at full coverage:
hero_cards 28 (dominant; all RANK errors, 3 are `9c9c`-style localization dups),
preflop 6–9, position ~4, board ~1, parse_none 81.

### What's already RULED OUT this session (don't repeat — see findings doc)
1. **CardCNN cross-entropy retrain** — corpus saturated (`data/cards_v2`, 32079
   crops / 6633 hands incl. masked variants). 3 seeds all regressed below the
   committed v2 (rank 0.9719). The `retrain-card-classifier` skill is v1-stale
   (drives `data/cards` / `extract_crops`, NOT the v2 corpus that production uses).
2. **Broadening the VLM structural-recheck trigger** (`allin`→`reaction`/`all`) —
   MEASURED net-negative: abstained 16 previously-correct hands for 0 precision
   gain. Code exists (`OCR_VLM_RECHECK_TRIGGER=reaction`) but is experimentation-only.
3. **Full Gemini re-parse on the abstain set** — MEASURED net-negative: keeping
   the OCR answer is correct 40/64 but full re-parse only 24/64 (it flips OCR's
   correct position/board). Gemini only helps TRUE parse_none (0→24).

---

## Prioritized implementation plan

Ranked by impact × tractability. Both gpt-5.5-pro and gemini-pro-latest
independently converged on #1–#3 (high confidence). Full verbatim consult in the
appendix.

### #1 — Field-level micro-routing (replace "abstain → full Gemini") — START HERE
**Both models, low risk, fixes a measured regression.**
- **Mechanism:** when the calibrator abstains (or hero card-conf is low), DO NOT
  full-reparse. Keep OCR's structure (position/board/actions); crop ONLY the
  hero cards (padded bbox) and send that tiny image to `gemini-3.5-flash` with a
  constrained prompt: `識別這2張撲克牌，輸出 {"cards":["9c","Th"]}`. Merge back.
- **Why:** directly fixes finding #3 (full reparse destroys correct fields).
  Latency ~1–2s vs ~8s. The existing `_gemini_hero_hand_only` /
  `_merge_ocr_with_gemini_hero_hand` in `src/gemini_session.py` is exactly this
  shape — **the lever is widening its trigger** so it fires on ALL abstains-with-
  present-OCR, not just the current narrow `cards_need_fallback` condition.
- **Expected:** recover ~16 of the 40-correct-but-abstained hands; full-pipeline
  precision +~2pp; lower latency.
- **Files:** `src/gemini_session.py::_parse_hand_from_image` (the abstain branch),
  `_gemini_hero_hand_only`, `_merge_ocr_with_gemini_hero_hand`.
- **Validate:** extend `scripts/ocr_recall_eval.py` with a "cards-only" mode that
  keeps OCR structure + re-reads only hero cards, run on the 64 abstain hands,
  compare exact-rate vs the 24/64 full-reparse baseline (already measured).

### #2 — Deterministic poker-physics validation layer ("free" precision)
**Both models, low risk.**
- **Mechanism:** before emit, reject physically-impossible parses and route the
  offending field to micro-recheck (#1):
  - duplicate hero card (`9c9c`) or hero card ∈ board → impossible in one deck.
  - illegal preflop bet sequence (call amount ≠ outstanding bet; action after
    fold/all-in; non-monotone outstanding bet) → flag `action_conf=0`.
- **Why:** the parser currently emits impossible states. Fixes the 3 dup
  localizations instantly + catches 6–9 collapsed-row action errors.
- **Expected:** +2–3pp absolute precision (gemini est.).
- **Files:** new validator invoked in `scripts/ocr/n8_parser.py` before assembly
  return; reuse `POSITION_ORDERS`. Keep rules STRICTLY preflop bet-physics to
  avoid false rejects.
- **Validate:** re-dump test, check `hero_cards_wrong` dups → 0 and
  `preflop_action_types_wrong` drops, with no new abstains on correct hands.

### #3 — Hero-crop test-time augmentation (TTA) to punch through WIN overlay
**gemini, very low cost, no retrain.**
- **Mechanism:** on marginal hero card-conf, run the CardCNN on 3 variants of the
  crop (original, +20% contrast to cut the dark WIN overlay, ±2px translation)
  and average logits.
- **Why:** "saturation" is on CLEAN cards; failures are local pixel noise from the
  overlay/alignment. TTA smooths it without touching the (saturated) corpus.
- **Expected:** recover ~5–10 of the 28 rank misreads. ~50ms added latency.
- **Files:** `scripts/ocr/classifier/infer.py` (`CardClassifier` predict path);
  guard behind a card-conf threshold so it only fires on marginal crops.
- **Validate:** re-dump test; `hero_cards_wrong` should drop with zero correct-
  hand regressions (TTA on confident crops must be a no-op).

### #4 — Template / inverse-rendering hero-card reader (highest ceiling)
**gpt-5.5-pro's top pick. Medium effort, beats the saturated CNN.**
- **Mechanism:** the GGPoker UI is a FIXED asset set. Clone clean card glyphs from
  the corpus; estimate the hero-card homography from the full-res screenshot;
  render all 52 candidates through blur/JPEG/scale/gamma + WIN-overlay mask;
  score each by robust NCC / gradient loss over rank-corner + pips; pick the card
  PAIR jointly under no-duplicate + legal-deck constraints. Search small
  translation/scale offsets (also solves localization).
- **Why:** a renderer encodes a far stronger prior than a CNN at corpus capacity,
  and yields a clean "too-close-to-call" margin for abstaining.
- **Expected:** fix/abstain 15–22 of 28 hero-card errors.
- **Risk:** theme/asset mismatch; needs a solid overlay mask. Build the WIN-overlay
  mask first.
- **Validate:** unit-test the renderer on known crops; A/B vs CNN on the 28
  hero_cards_wrong hands.

### #5 — Contrastive / triplet fine-tune of the CNN heads on hard negatives
**gemini. Reframes "saturation" — UNTRIED (we only swept cross-entropy seeds).**
- **Mechanism:** freeze the CNN backbone; fine-tune the rank head with a
  supervised-contrastive / triplet loss, mining hard pairs `(rank X + overlay)`
  vs `(rank Y + overlay)`. Cross-entropy goes lazy at 97%; contrastive forces the
  embedding to separate occluded 8 vs 9.
- **Why:** our saturation finding only tested cross-entropy retrains. A different
  LOSS on the SAME data may still move the hard tail.
- **Expected:** medium; pushes the CNN asymptote.
- **Risk:** custom training loop + careful hard-negative mining. Keep the
  committed v2 as fallback; gate on the 44-hand regression before shipping.
- **Files:** `scripts/ocr/classifier/train.py` (new loss option),
  `scripts/ocr/classifier/model.py` (expose embeddings).

### #6 — Grammar/ILP preflop solver + verifier-feature rejector
**gpt-5.5-pro. Higher effort, structural.**
- **6a Grammar solver:** beam/ILP over action-row fragments with latent seat
  count + hero position; constraints = blinds, UTG-first order, monotone
  outstanding bet, all-in stack caps, no-action-after-fold. Token confidences as
  likelihoods. Fixes collapsed multiway rows (6–9 action + ~4 position errors).
- **6b New rejector:** retrain the emit gate on VERIFIER features (renderer margin,
  localization beam gap, duplicate/deck conflict, WIN-overlay coverage %, VLM
  disagree/UNKNOWN, grammar entropy, seat-count posterior, parse_none origin),
  OOF, thresholded for ≥95% recall. The old calibrator failed because its
  features couldn't see "confidently wrong." Abstain 25–35 residual bad hands
  while losing far fewer correct.

---

## Honest math / expectation setting

99%@95% = ≥683 emitted with ≤6 wrong. Full-coverage today has ~97 wrong, so we
need BOTH field repair (#1–#5) AND a much sharper rejector (#6b). #1–#3 are the
cheap, cross-validated wins (~86%→~90%+ plausible). #4–#6 are the path to the
last mile but are research-grade and may still not reach 99% — the hardest hands
are ambiguous to frontier VLMs too (parse_none: gemini-pro 30% exact). Treat 99%
as the direction; bank the measurable gains at each step.

## Verified run commands

```bash
# Recall / fallback eval (network-bound, no GPU)
python scripts/ocr_recall_eval.py --dump data/ocr_precision_phase11e_test \
  --workers 8 --out data/recall_eval_phase11e_test.json
# add --include-abstain to also score the 64 abstained hands

# Re-dump test set (current code + structural override). ~13 min real-flash.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/ocr_precision.py --bucket test --split data/splits/card_classifier_v2.json \
  --dump-all --workers 4 --out data/ocr_precision_<tag>_test

# Unit tests (23 green) + regression (423 green)
python -m pytest tests/ocr/test_vlm_recheck.py tests/ocr/test_recall_eval.py -q
python scripts/regression_test.py
```

Baseline dumps: `data/ocr_precision_phase11e_test` (current main code, allin
trigger). Eval outputs from this session: `data/recall_eval_phase11e_test.json`,
`data/recall_eval_abstain.json` (both gitignored under `data/`).

---

## Appendix A — gpt-5.5-pro (high reasoning effort), verbatim

First, the arithmetic is harsh: 99%@95% means **≥683 emitted with ≤6 wrong**. Current full-coverage has ~97 wrong, so you need both **field repair** and a **much sharper rejector**. I would prioritize:

1. **Replace final hero-card decision with fixed-asset inverse rendering, not another CNN.** Extract/clone clean GGPoker card glyph assets from corpus; estimate hero-card homography from full-res screenshot; render all 52 candidates through blur/JPEG/scale/gamma; mask detected WIN overlay; score by robust gradient/NCC loss over rank corners + pips. Search small translation/scale offsets. Choose card pair jointly with no-duplicate/deck constraints. UI is fixed; CNN is saturated but a renderer uses stronger prior and can produce reliable "too close to call" margins. Expected: fix/abstain **15–22/28 hero-card errors**. Risk: theme/asset mismatch; needs careful overlay mask.

2. **Joint hero-card re-localization with crop lattice.** For each hero card, generate 20–50 plausible boxes around detector output; score each via CNN+renderer; enforce two separate cards, non-overlap, sane geometry. Explicitly flag identical/correlated left/right crops to kill "9c9c" duplicate failures. Expected: **3–6 hands**. Risk: if search space too wide, can hallucinate; require large score margin.

3. **Use VLMs only as candidate-choice verifiers, never as whole-hand replacers.** For hard hero-card cases, send a montage of the original crop, nearest-neighbor 6× crop, rank-corner crop, overlay-masked crop, and top-2/top-3 candidates: "choose one or UNKNOWN." Run Gemini Flash/OpenAI small vision only on these ~50 cases. Accept override only if it agrees with renderer or two independent VLMs agree; otherwise abstain. Full Gemini is net-negative because it flips good fields; forced-choice field verification avoids that. Expected: **5–10 hero-card fixes** plus fewer destructive flips. Risk: VLM overconfidence; keep UNKNOWN easy.

4. **Turn preflop parsing into a weighted legal-action solver.** Build a beam/ILP over row fragments: latent seat count, hero position, row grouping, actor, action type, amount. Constraints: blinds, UTG-first order, monotone outstanding bet, all-in stack caps, no action after fold/all-in except calls, pot/stack arithmetic where visible. Use OCR/VLM row-token confidences as likelihoods; grammar chooses globally valid sequence. Collapsed multiway all-in rows are structural, not OCR-only. Expected: **6–9 action errors + ~4 position errors**. Risk: bad amount OCR can mislead; allow amount-optional action-type paths.

5. **Rebuild parse_none path as field-preserving assembly, not full Gemini JSON.** Keep deterministic/card-renderer/table fields; ask VLM only for missing row tokens or seat count; assemble through the grammar above. Never let Gemini overwrite high-confidence board/position/hero cards. Expected: **25–45 additional exact hands** from the 81 parse_none tail. Risk: engineering-heavy; gains depend on whether parse_none is mostly assembly failure.

6. **New hand-level rejector using verifier features, not old calibrator features.** Features: renderer rank margin, localization beam gap, duplicate/deck conflict, WIN-overlay coverage %, VLM disagreement/UNKNOWN, grammar beam entropy, collapsed-row count, seat-count posterior, parse_none origin. Train out-of-fold for exact-hand correctness; threshold for ≥95% recall. The old calibrator failed because candidates and uncertainty signals were weak. Expected: abstain **25–35 residual bad hands** while losing far fewer corrects. Risk: overfitting the 718; use split calibration.

7. **Mine systematic rank confusions into binary verifiers.** For each recurring confusion pair, train a tiny pairwise rank verifier on real hard crops plus synthetic rendered degradations/overlays. Use only when CNN top-2 contains that pair. A saturated 13-way CNN can still lose to specialized one-vs-one tests. Expected: **3–8 hero-card fixes**. Risk: only helps known confusions.

---

## Appendix B — gemini-pro-latest, verbatim

You are currently treating the pipeline as a monolith at the fallback stage (CNN fails -> full VLM re-parse). This is your biggest structural blind spot. Poker is a strictly constrained system; you need to decouple your fallback logic and enforce game physics.

### 1. Field-Level Micro-Routing (Fixes the VLM Fusion Regression)
Stop doing full-image Gemini re-parses. Your calibrator must output confidence *per-field* (hero_conf, board_conf, action_conf). If `hero_conf` is low, crop *only* the hero cards and send that tiny image to gemini-3.5-flash with a highly constrained prompt: `"Identify the 2 poker cards in this crop. Output JSON: {cards: ['9c', 'Th']}."` Keep the CNN's board and position. Directly addresses Finding 5. VLMs hallucinate and regress correct fields when forced to parse dense, full-screen UI. By feeding the VLM a tightly cropped image and a micro-task, you eliminate spatial confusion and drastically reduce latency (~1-2s vs 8s). Expected Gain: High. Recovers the 40/64 hands where the full VLM ruined good OCR. Main Risk: Misaligned crops; pad bounding boxes before cropping.

### 2. Deterministic Poker Physics & NMS (The "Free" Precision)
Implement a strict validation layer before emission. NMS/Duplicate Rule: if `hero_1 == hero_2` ("9c9c") or `hero_card in board_cards`, reject the parse and force a re-crop with tighter boxes. Action State Machine: pass the parsed action sequence through a lightweight poker logic validator; if a row violates rules (e.g., Player A calls $50 when max bet is $20 without a raise), flag `action_conf` as 0 and micro-route the action panel crop to the VLM. You are leaving precision on the table by allowing impossible game states. Fixes the 3 dups instantly + catches the 6-9 collapsed action errors. Expected Gain: ~2-3% absolute precision. Main Risk: false-positive rejections; keep logic strictly to preflop bet-sizing physics.

### 3. Test-Time Augmentation (TTA) for the WIN-Overlay
Your CNN corpus is saturated, but inference is brittle on the WIN-overlay. At inference, if `hero_conf` is marginal, run TTA: pass 3 variants of the hero crop (original, +20% contrast to punch through the dark overlay, and a slight 2px translation). Average the logits. Saturated corpora mean the model learned the global distribution but fails on local pixel noise. TTA smooths this without retraining. Expected Gain: Medium. Recover ~5-10 of the 28 rank-misread errors. Main Risk: marginal latency (tiny CNN 3x, <50ms).

### 4. Supervised Contrastive Learning (SCL) for Hard Negatives
I am skeptical that your CNN is truly "saturated" in capacity; rather, standard Cross-Entropy loss has saturated. Freeze your CNN backbone and fine-tune the classification heads using a Contrastive Loss (or Triplet Loss). Mine pairs of `(Rank X + WIN overlay)` vs `(Rank Y + WIN overlay)`. Cross-entropy gets lazy once it achieves 97% accuracy on clean cards; contrastive loss forces the embedding space to explicitly separate a heavily occluded "8" from an occluded "9". Expected Gain: Medium. Pushes the asymptotic limit. Main Risk: custom training loop + careful hard-negative mining.
