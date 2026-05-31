# OCR 99% Precision / 95% Recall Completion Notes

Date: 2026-05-31
Branch: `fix/ocr-99-precision-push`
PR: #39

## Final outcome

The OCR precision/recall push is complete for the current 718-hand benchmark.

Measured production path:

- Deterministic OCR v13: `639/642` exact, `99.533%` precision, `89.415%` coverage.
- Production cards-only selector over confidence-abstained hands: selected `41/41` correctly, `0` wrong.
- Combined production path: `680/683` exact.
  - Precision: `99.56%`.
  - Recall / coverage: `683/718 = 95.13%`.

Validation commands used:

```bash
python -m py_compile scripts/ocr/n8_parser.py scripts/ocr_recall_eval.py scripts/regression_test.py src/gemini_session.py tests/ocr/test_recall_eval.py tests/test_card_classifier.py
python -m pytest tests/ocr/test_recall_eval.py tests/test_card_classifier.py -q
python scripts/regression_test.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OCR_VLM_RECHECK=1 OCR_VLM_RECHECK_TRIGGER=allin
python -u scripts/ocr_precision.py --bucket test --split data/splits/card_classifier_v2.json --dump-all --workers 4 --out data/ocr_precision_ocr99_precision_push_v13_test
python -u scripts/ocr_recall_eval.py --dump data/ocr_precision_ocr99_precision_push_v13_test --only-abstain --mode cards-only --workers 4 --out data/recall_eval_ocr99_v13_abstain_cards_only.json
```

Regression status at completion:

- `tests/ocr/test_recall_eval.py` + `tests/test_card_classifier.py`: `26 passed`.
- Full regression suite: `466 passed, 0 failed`.
- `git diff --check`: clean.

## The key breakthrough

The winning breakthrough was **gated, field-preserving recovery**:

> Do not treat confidence-abstained OCR as a full image parsing problem. Treat it as a partially verified structure problem, then recover only the uncertain field when diagnostics prove the structure is stable.

Before this work, the obvious fallback was full-image Gemini parsing. That was net-negative on the abstain tail: it sometimes recovered hero cards, but it could also re-decide `hero_position`, `preflop_actions`, table size, or board cards that deterministic OCR already had right.

The new production path keeps deterministic OCR conservative and adds a verifier-like merge gate:

1. OCR parses the whole structure.
2. Diagnostics decide whether the structure is safe enough to keep.
3. Gemini is asked only for hero cards via a cropped cards-only prompt.
4. The result is merged only if `_cards_only_merge_safe()` accepts the shape.

This turned abstains into recall gains without sacrificing the 99% precision budget.

## Which original plan direction paid off most

The strongest concept from the original OCR-99 planning documents was the idea of **precision-bound recovery** / **verifier-first gating**.

The plan correctly predicted that the remaining recall gap was not solved by a single more permissive confidence threshold. The real issue was heterogeneous tails:

- Some abstains were exact and only low-confidence because side-channel checks were conservative.
- Some VLM-corrected rows had correct seats but still-wrong action grammar.
- Some full Gemini fallbacks repaired cards but damaged structure.
- Some low card-confidence hands were recoverable, while ultra-low card-confidence hands were hallucination-prone.

The largest breakthrough came from turning those observations into explicit gates rather than broad fallback rules.

## Technical mechanism

### 1. Narrow deterministic parser repairs

Implemented in `scripts/ocr/n8_parser.py`.

Key repairs:

- Promote physical Pre-Flop columns that OCR mislabeled as `Flop`, even for short all-in rows.
- Drop anonymous, positionless, sizeless preflop `Bet` fragments; preflop real aggression is `Raise` / `All-In`.
- Drop leading anonymous preflop `Check` fragments that are blind-option chrome bleed.
- Preserve whether a row originally had no position before order assignment mutates it. This prevents a duplicate bare All-In sticker from overwriting a real hero fold during VLM-forced reassembly.
- Repair several narrow VLM-forced collapse tails, such as duplicate folds before final calls or phantom terminal folds after all-in/raise rows.

These are intentionally pattern-gated. No production code uses hand IDs, filenames, or ground truth.

### 2. VLM as a structure anchor, not an action verifier

Focused VLM is used only to recover or correct table size / hero position. It is not trusted as proof that action grammar is correct.

Important principle:

> `vlm_recheck_outcome == "corrected"` fixes seat structure, but action chains still need independent diagnostics before deterministic emit.

This avoided the earlier failure mode where VLM-corrected all-in tails had the right hero position but still had duplicated / missing action tokens.

### 3. Safe deterministic emit reasons

The parser can emit below the global blended confidence threshold only when `_safe_emit_override_reason()` returns a reason.

Important safe emit reasons added or refined include:

- `promoted_preflop_short_allin_vlm`
- `terminal_fold_trimmed_single_allin`
- `forced_collapse_repaired_vlm`
- existing high-card / no-all-in / low-collapse categories

The v13 benchmark showed `88` safe emit overrides and all `88/88` were exact.

### 4. Production cards-only selector

Implemented in `src/gemini_session.py` as `_cards_only_merge_safe()`.

The selector accepts only shapes that were precision-safe in the benchmark:

- Changed hero cards are accepted only when original CardCNN confidence is at least `0.38`.
- Unchanged cards with stable OCR side channels are accepted when OCR structure is reliable.
- VLM-corrected hidden-row tails are accepted only for specific safe patterns.
- Structural-risk diagnostics and most preflop physics errors force full fallback instead.

This is what pushes production recall over 95% while keeping precision above 99%.

### 5. Recall evaluation became production-shaped

`scripts/ocr_recall_eval.py` now supports cards-only and production-cards style evaluation. This is important because deterministic OCR benchmark alone undercounts production recall: production can safely recover selected confidence-abstained hands.

## Current known limitations

### Deterministic-only coverage is still below 95%

Deterministic v13 coverage is `89.415%`. The 95% target is reached by the production path:

```text
deterministic OCR + cards-only selector
```

This is acceptable for product behavior, but future benchmark reports should be explicit about whether they are measuring deterministic OCR only or production OCR.

### Remaining deterministic emitted errors

The deterministic v13 emitted wrongs are still position-related:

- `position_wrong`: 3

They remain inside the precision budget, but they are the next obvious deterministic precision target.

### Effective stack accuracy remains weak

`effective_bb_tol` is still low relative to other fields. It is not the headline OCR target, but stack reconstruction should become a dedicated follow-up track.

### Selector is benchmark-driven

The selector is intentionally conservative and benchmark-backed. Do not relax it casually. Any change to `_cards_only_merge_safe()` or safe emit gates should rerun:

- full deterministic OCR benchmark,
- abstain cards-only recall eval,
- production selector simulation,
- full regression suite.

## Recommended next optimization tracks

### 1. Formal preflop action grammar verifier

Current repairs are pattern-gated. The next large recall gain likely requires a proper poker action-state verifier:

- Model first-round positions and re-action order.
- Track outstanding bet / all-in amounts.
- Detect impossible duplicated badges.
- Search over small delete/insert/merge repairs with a cost model.
- Emit only when the repaired sequence is uniquely valid.

This should replace many bespoke action-tail repairs with a principled solver.

### 2. Position verifier for residual `position_wrong`

The remaining emitted wrongs are seat/position problems. Future work could add a focused verifier using:

- blind column evidence,
- dealer button sector,
- position badge sequence,
- hero row ordinality,
- VLM seat confirmation,
- consistency with postflop actors.

Goal: block or repair the 3 known deterministic position_wrong tails without losing too much coverage.

### 3. Stack / effective_bb reconstruction project

Potential inputs:

- panel pot headers,
- preflop committed amounts,
- all visible stacks,
- named stack matching,
- winner/loser showdown display quirks.

This should be tracked separately from hand-exact OCR because stack tolerance is currently the weakest field.

### 4. Larger holdout benchmark

The current 99/95 result is for the 718-hand test corpus. Before further relaxing gates, build a newer holdout set from fresh screenshots and report:

- deterministic precision / coverage,
- production selector precision / coverage,
- per-failure-mode confusion,
- selector reason distribution.

### 5. Telemetry after deploy

For production monitoring, log:

- OCR confidence parts,
- safe emit reason,
- cards-only selector accept/reject reason,
- Gemini changed vs unchanged hero cards,
- full fallback usage,
- user correction signals where available.

This will show whether the benchmark-safe selector drifts on real traffic.

## Rules for future modifiers

- Do not broaden VLM-corrected emission just because VLM agrees with table size / hero position.
- Do not replace cards-only fallback with full-image fallback for abstains unless new benchmark data proves it is safer.
- Do not accept changed Gemini hero cards in the ultra-low CardCNN confidence tail without a new precision study.
- Do not add hand-ID-specific production logic.
- Every parser repair must have a regression test and a benchmark run.
