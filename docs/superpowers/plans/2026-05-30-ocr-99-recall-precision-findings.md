# OCR 99%@95% — Recall/Precision findings (continues the D-c handoff)

> Continues `2026-05-30-ocr-99-dc-handoff.md`. Worktree
> `.claude/worktrees/ocr-99-hero-recall`, branch `worktree-ocr-99-hero-recall`.
> **Headline: the goal is precision-bound, not recall-bound. Recall is already
> ~solved in production. The CardCNN retrain lever is saturated.**

---

## What was done this session

Three levers from the D-c handoff were implemented / investigated. Two shipped
as code; the third (the handoff's "next big lever") was investigated and
empirically closed.

### 1. Honest recall eval — `scripts/ocr_recall_eval.py` (SHIPPED)

Replays the *exact* production parse_none path (full `IMAGE_PARSE_PROMPT`
Gemini parse → `_normalize_cards` → `_fix_folded_players`) on a dump's
parse_none hands and scores them with the harness `compare()`. The
deterministic-only eval counts parse_none as misses, so it undercounts true
production recall — this measures the recovery honestly.

**Result (phase11e test set, 81 true parse_none, `gemini-pro-latest`):**
- **81/81 recovered** — the existing fallback ALWAYS returns a parsable hand.
- **24/81 fully exact (29.6%).**
- Field misses among the 57 wrong: hero_hand 56%, board 54%, preflop 47%,
  position 28%. (13 wrong on hero_hand only, 10 board only, 5 preflop only.)

**Conclusion — this reframes the whole goal:**
- **RECALL IS NOT THE BINDING CONSTRAINT.** Production recall ≈ 100%
  (deterministic 637/718 parsable + Gemini recovers the other 81). The D-c
  handoff's "recall ceiling 88.7%, need 95%" framing was about the
  *deterministic-only* path; production already clears 95% via the fallback.
- The real constraint is **PRECISION on the genuinely-hard tail.** Even a
  frontier VLM gets only ~30% of these screenshots exactly right (collapsed
  multiway rows, tiny WIN-overlay-occluded hero cards). These hands are
  ambiguous *from the image itself*.
- hero_hand is the dominant single-field miss — consistent across the emitted
  errors AND the recovered tail.

Artifacts: `data/recall_eval_phase11e_test.json` (gitignored; full per-hand
results). Re-run: `python scripts/ocr_recall_eval.py --dump <dir> --workers 8`.

### 2. VLM `reaction` trigger mode (SHIPPED)

`OCR_VLM_RECHECK_TRIGGER=reaction` — superset of `allin` that also re-checks
hands whose table size used the reaction signal (`estimate_used_reaction_signal`),
the residual ~18% of structural errors the all-in trigger misses. Wired
`diagnostics` into the `is_suspect` call in `n8_parser` so the mode sees the
signal. Override is safe on correct hands (30/30 preserved) → pure
coverage/latency knob. +5 unit tests (`test_vlm_recheck.py`), all green.

**MEASURED (real-flash dump, `data/ocr_precision_reaction_test` vs phase11e
`allin` baseline) — net-negative, so `allin` stays the default:**

| | allin (phase11e) | reaction |
|---|---|---|
| position_wrong | 4 | **4 (no change)** |
| true parse_none | 81 | **96 (+15)** |

Hand-level diff: **16 hands newly demoted to parse_none under `reaction` were
ALL emitted-CORRECT under `allin`** (0 were wrong). The flash re-check fired on
16 more correct hands, disagreed, and the re-derivation couldn't reconcile →
abstain. vlm_recheck outcomes: corrected 17, agree 229, **abstain 20** (16 of
them previously-correct). This **refutes the D-c handoff's "30/30 preserved,
pure latency knob" claim** — at scale the override is NOT 100% safe on correct
hands. In production the 16 go to the Gemini fallback (~30% recovered), so the
net is ~−11 correct hands for 0 precision gain.

⇒ The `reaction`/`all` triggers are net-negative on the current parser. Keep
`allin` as the default. The mode is retained as an experimentation knob (and
the diagnostics wiring is correct), but should not be enabled in production.

### 3. CardCNN hero-card retrain — INVESTIGATED, NO-SHIP (corpus saturated)

The handoff named this "the next big lever — start here." It is not a lever on
the current data.

- The committed `card_cnn_v2` is already trained on the full **25560-sample**
  `data/cards_v2` corpus (rank **0.9719**, suit 0.9757, card 0.9672).
- Three fresh retrains on the *identical* corpus all came out **worse**:

  | run | rank | suit | card |
  |---|---|---|---|
  | **committed v2** | **0.9719** | **0.9757** | **0.9672** |
  | seed 0 | 0.9694 | 0.9735 | 0.9621 |
  | seed 1 | 0.9668 | 0.9728 | 0.9599 |

  The committed v2 was simply a good seed; the corpus is at capacity.
- Even a hypothetical +0.003 rank gain ≈ **1 hand** of 718 — negligible for
  99%@95%.
- **The `retrain-card-classifier` skill is stale**: it drives the v1 pipeline
  (`extract_crops` → `data/cards`, ~1659 crops), but `train.py` defaults to the
  v2 corpus (`data/cards_v2`, split `card_classifier_v2.json`). Running the
  skill's flow does NOT touch the production v2 model's training data.
- Localization duplicates (`9c9c`, `6s6s`, `KcKc` — 3 cases) all read
  card_conf **0.0** → already abstained → Gemini-recovered. They hurt recall
  (already solved), not emitted precision. Not worth the risky `_locate_hero_cards`
  surgery.

**Real levers left for hero-card precision** (all research-grade, untried):
hard-negative mining on the confident-wrong hero cards, focal loss on confused
rank pairs, growing `data/cards_v2` with NEW hard hands (not reshuffling the
saturated set). A bigger/harder corpus is the only thing that moves rank
accuracy now.

---

## Full-pipeline precision/recall (test 718, all measured)

| segment | hands | correct via Gemini re-parse | correct if OCR kept |
|---|---|---|---|
| high-conf OCR emit | 573 | — (OCR used) | **557 (97.2%)** |
| confidence_abstain | 64 | **24 (38%)** | **40 (62%)** |
| true parse_none | 81 | **24 (30%)** | 0 (no OCR hand) |

- **Deterministic OCR alone:** 97.2% precision @ 79.8% recall (573/718 emitted).
- **Full pipeline @ ~100% recall:**
  - naive (route all low-conf → Gemini): (557+24+24)/718 = **84.3%**
  - best (keep OCR on abstain, Gemini only on parse_none): (557+40+24)/718 = **86.5%**
- vs target **99%@95% → ~12–13 pp precision short.**

### KEY NEW FINDING — the Gemini fallback is NET-NEGATIVE on the abstain set
On the 64 confidence-abstained hands, **keeping the OCR answer is correct 40×,
but full Gemini re-parse is correct only 24×** — the fallback DESTROYS 16 correct
hands by flipping OCR's right position/board fields. (Gemini only helps on true
parse_none, where OCR produced no hand: 0 → 24.) This is the same failure the
existing `cards-only fallback` mitigates (keep OCR structure, re-read only
hero_hand) but it doesn't fire on all abstains.

⇒ **Actionable, zero-cost precision lever:** production should default abstained-
but-present-OCR hands to *keep OCR / cards-only fallback*, NOT full Gemini
re-parse. That alone lifts full-pipeline precision 84.3% → 86.5%. This is more
real than broadening the VLM trigger (which was net-negative).

---

## Honest assessment of 99%@95%

The v4 calibrator coverage curve on the test set:

| target coverage | actual precision | correct/emitted |
|---|---|---|
| 0.645 | **0.989** | 458/463 |
| 0.75 | 0.968 | 521/538 |
| 0.85 | 0.918 | 560/610 |
| 0.95 | 0.869 | 566/651 |

To hit 99%@95% we'd need ≤7 wrong in ~682 emitted. Going from 65%→95% coverage
adds ~218 hands but only ~166 correct → ~52 genuinely-wrong hands the calibrator
*cannot* separate because the parser is confidently wrong on them and the
screenshots are ambiguous. The recall eval independently confirms this: the
hardest hands fool even `gemini-pro-latest` (30% exact).

**99%@95% is aspirational and not reachable by incremental parser/model fixes.**
The genuine path forward is data: a larger, harder-hand `data/cards_v2` corpus
(retrain becomes a lever again) plus broader structural override coverage
(`reaction`/`all` trigger) to shave the confident-error count. Recall recovery
is already done by the Gemini fallback.

---

## Concrete next steps (priority order)

1. **Grow the hard-hand corpus.** Mine confident-wrong hero cards from
   `analysis_snapshots` into `data/cards_v2`, re-split, retrain. This is the
   ONLY thing that lifts rank accuracy past 0.972.
2. **Improve the override's abstain logic, not the trigger.** `reaction`/`all`
   were measured net-negative (above) — broadening the trigger just abstains
   correct hands when flash disagrees. The lever is making the re-check
   *trust the parser more* on disagreement (e.g. require flash to be confident,
   or only override when the parser's own structural signals are weak) so it
   stops demoting correct hands. Keep `allin` default until then.
3. **Fix the `retrain-card-classifier` skill** to target the v2 pipeline
   (`data/cards_v2` + `card_classifier_v2.json`), or it will keep retraining a
   model that production doesn't use.

## How to run (verified this session)

```bash
# Honest recall eval (network-bound; no GPU)
python scripts/ocr_recall_eval.py --dump data/ocr_precision_phase11e_test \
  --workers 8 --out data/recall_eval_phase11e_test.json

# vlm_recheck + recall_eval unit tests (23, all green)
python -m pytest tests/ocr/test_vlm_recheck.py tests/ocr/test_recall_eval.py -q

# Retrain (saturated — for reference only; defaults to the v2 corpus)
python -m scripts.ocr.classifier.train   # writes card_cnn_v2.{pt,json}
```

Note: `tests/ocr` has 6 pre-existing failures on `main` (verified on d938274,
unrelated to this work) + `test_calibrate_eval.py` needs the uncommitted scratch
`scripts/_calibrate_v2.py`.
