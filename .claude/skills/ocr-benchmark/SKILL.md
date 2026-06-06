---
name: ocr-benchmark
description: Use to MEASURE OCR hero-card / parse accuracy against the 7k pokercraft ground-truth corpus — establishing a baseline, iterating on hero/board localization or CardCNN changes, or quantifying a fix's before/after. Triggers on "benchmark the OCR", "run the OCR benchmark", "hero localization accuracy", "train/test eval", "how accurate is the parse", "did this change help/hurt OCR". For the hard pre-commit no-regression GATE, use verify-ocr-no-regression instead.
---

# OCR Benchmark

Measure image→hand recognition accuracy against authoritative labels. This skill
is for **iteration and measurement**; the commit-blocking gate is the separate
`verify-ocr-no-regression` skill (run that before you ship).

## Ground truth corpus (7183 hands)

- Images: `data/hand_images/img/<hand_id>.png`
- Labels: `data/pokercraft_corpus/ground_truth/ground_truth.jsonl`
  (HH-derived, authoritative; each line `{hand_id, ground_truth:{hero_hand,
  hero_position, board, preflop_actions, ...}}`)
- Train/test split: `data/splits/production_v1.json`
  (`production_train` / `production_val` / `production_test`)

**Important corpus caveat:** the pokercraft replay images are CLEAN (~4.8%
degenerate hero crops). Live N8 *mobile* Telegram screenshots are much harder
(~26% blocking, WIN stickers / chips / flags / variable framing) and are NOT in
this corpus. Use the corpus for correctness (authoritative labels); use the
`live` mode below (DB snapshots) for the real-traffic speed/blocking metric.

## Fast hero-localization bench (minutes) — `scripts/ocr_hero_loc_bench.py`

Isolates `_locate_hero_cards` + CardCNN (skips panel EasyOCR and the Gemini
fallback). Use it to iterate on hero detection and prove a change is net-positive
*before* the full gate.

```bash
# 1. baseline on main (git stash / checkout the base first):
python scripts/ocr_hero_loc_bench.py run old 2000
# 2. candidate on your branch:
python scripts/ocr_hero_loc_bench.py run new 2000
# 3. diff — SHIP ONLY IF REGRESS == 0 and FIXED > 0:
python scripts/ocr_hero_loc_bench.py cmp old new
# real-traffic blocking-Gemini rate (label-free, from DB snapshots):
python scripts/ocr_hero_loc_bench.py live 200
```

`run` reports hero accuracy, local-emit rate (conf≥0.70), blocking rate
(conf<0.70). Results saved to `data/_ocr_loc_eval/<label>.jsonl` (gitignored).
Single-process by design — see the CUDA note below.

## Full accuracy benchmark (slower, full production parse incl. Gemini)

`scripts/ocr_precision.py` runs the real `parse_n8_screenshot` over the corpus
and reports per-field accuracy + a confidence/coverage curve, train/test aware.

```bash
# full 7k (no --split → all paired hands):
python scripts/ocr_precision.py --workers 12 --dump-all --out data/ocr_prec
# held-out test bucket only:
python scripts/ocr_precision.py --split data/splits/production_v1.json \
    --bucket production_test --out data/ocr_prec_test
```

Read `data/ocr_prec/all_records.jsonl` for per-hand `fields.hero_hand`,
`card_confidence`, etc. (`--dump-all` writes every record).

`scripts/ocr_benchmark.py` + `scripts/ocr_regression_gate.py` are the
base-vs-head gate used by `verify-ocr-no-regression`.

## CUDA gotcha (read before running with workers)

Multiprocess GPU dies: `ocr_precision.py --workers N` on the GPU throws
`CUBLAS_STATUS_NOT_INITIALIZED` (N CUDA contexts on one device). Either run it
**CPU-only** (`CUDA_VISIBLE_DEVICES="" python scripts/ocr_precision.py ...`) or
use the single-process `ocr_hero_loc_bench.py` for GPU runs.

## Decision rule

Net-positive is not enough for a localized fix: **any hand correct-before /
wrong-after is a regression — block it** (`cmp` REGRESS must be 0). Then run the
full `verify-ocr-no-regression` gate before commit/merge.

## Related

`verify-ocr-no-regression` (the commit gate — always run before shipping an
`scripts/ocr/**` change), `retrain-card-classifier`, `fix-hand`,
`snapshot_test.py` (L1/L2 deterministic layer).
