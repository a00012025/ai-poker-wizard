# OCR Card Classifier — Replace Heuristics with Small CNN

## Summary

Replace the hand-coded rank/suit identification in `scripts/ocr/table_parser.py`
with a small local CNN. Localization code stays; only the 52-class
"is this a 9♥ or 9♦?" decision moves to a learned model. One classifier
handles both hero and board cards — panel card reading is out of scope
because `panel_parser.py` does not currently extract villain reveals
(only actions/names/stacks). Target: kill the whac-a-mole bugfix pattern,
shrink `table_parser.py` by ~50%, and preserve the existing OCR fast-path
(no per-card LLM calls).

PyTorch is already transitively installed via `easyocr>=1.7` (torch 2.10) —
no new binary weight in the container.

## Motivation

The current pipeline has accumulated 93 OCR-related bugfix commits (60 of them
in the last two months). `table_parser.py` is 1574 lines, with 12 hand-IDs
(`H2554`, `H2587`, `H2659`, `H2660`, `H2668`, `H2758`, `H2759`, …) embedded as
code comments marking special-case branches. `_find_hero_cards` alone is 300
lines of stacked heuristics: template margin → hull defect → green channel →
width profile → "allow_override" flags. Each new failing hand adds another
if-branch that can break prior fixes.

The root cause is architectural: **we hand-engineered a 52-class visual
classifier**. That is the canonical thing ML solves, on a task of near-MNIST
difficulty (highly consistent glyphs, closed vocabulary). A small CNN trained
on ~1300 labeled crops from the snapshot corpus will reach 99%+ accuracy and
turn every new failure mode from "write new heuristic" into "add to training
set and retrain".

## Goals

- **Eliminate whac-a-mole**: no more per-hand special cases in OCR code
- **Preserve the OCR fast-path**: no API calls added per card; end-to-end
  latency stays at "no LLM call" when confidence is high
- **Shrink `table_parser.py`** by ~50% (target ≤800 lines) by deleting
  `_ocr_card_rank`, `_detect_suit_bgr`, `_hero_hull_norm`,
  `_suit_template_match`, width-profile / hull / template heuristics, and
  `card_matcher.py`
- **≥99% per-card accuracy** on a held-out split of the snapshot corpus;
  ≥95% end-to-end hand-level parse match on the 44 regression snapshots
- **Reproducible**: anyone can rebuild the dataset and retrain with one
  command

## Non-Goals

- Replace localization (blob detection, region splitting, panel parsing).
  Localization is not the source of bugs and stays as-is.
- Improve Gemini fallback path. It stays as today's "if OCR confidence < 0.85"
  backstop.
- Support new poker clients. N8 only, same as today.
- Train on augmented/synthetic cards. Real crops only.

## Architecture

```
screenshot bytes
    │
    ▼
┌─────────────────────────────────────┐
│ region_detector / table_parser      │  (unchanged)
│   → table_region, panel_region      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ localization (thinned)              │  (keep blob/contour finding,
│   _locate_hero_cards  → 2 crops      │   delete all rank/suit heuristics;
│   _locate_board_cards → N crops      │   functions return crops + metadata,
│                                      │   not card strings — Phase 0 refactor)
└─────────────────────────────────────┘
    │ list of np.ndarray card crops, batched
    ▼
┌─────────────────────────────────────┐
│ CardClassifier (NEW)                │
│   shared CNN backbone (1 model       │
│   trained on hero + board crops)     │
│   ├─ RankHead (13 classes)           │
│   └─ SuitHead  (4 classes)           │
│   → [(rank, suit, conf), ...]        │
│   missing/corrupt checkpoint →        │
│     returns (None, None, 0.0)         │
│     → flows naturally into Gemini     │
│     fallback via conf threshold       │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ hand-level confidence               │
│   = min(card_confs)                  │
│   > 0.85 → return hand directly      │
│   ≤ 0.85 → fall back to Gemini       │  (existing behavior, unchanged)
└─────────────────────────────────────┘
```

### Why decompose into two heads instead of one 52-way softmax

- Data efficiency: 13 + 4 = 17 class labels, ~100 samples/rank and ~325/suit
  on average, versus ~25/class for joint 52-way
- Orthogonal error modes: rank confusion (`Q` vs `O` from "10") and suit
  confusion (♥ vs ♦) have different visual causes; separating them simplifies
  eval and retraining
- Same backbone: two linear heads on a shared feature vector; ~100 extra
  params

## Module Structure

**New files:**

```
scripts/ocr/classifier/
  __init__.py
  model.py           # CardCNN (backbone + RankHead + SuitHead), ~80 lines
  infer.py           # CardClassifier: load checkpoint, classify(crop), classify_batch
  train.py           # training CLI: `python -m ocr.classifier.train`
  dataset.py         # CardDataset (PyTorch): loads from data/cards/{rank}/{suit}/*.png
  extract_crops.py   # one-time/refreshable: pulls snapshots from Supabase,
                     #   runs localization, writes labeled crops to data/cards/
  eval.py            # per-card accuracy on val split; hand-level pass rate on
                     #   regression snapshots

scripts/ocr/models/
  card_cnn_v1.pt     # checkpoint (committed; ~100KB)
  card_cnn_v1.json   # training metadata (see below)
```

**Training metadata (`card_cnn_v1.json`)** — the checkpoint's lineage in one file:

```json
{
  "version": "v1",
  "trained_at": "2026-04-24T12:34:56Z",
  "data_hash": "sha256(sorted tuples of (hand_id, slot, label))",
  "n_samples_train": 1040,
  "n_samples_val": 260,
  "val_accuracy_rank": 0.995,
  "val_accuracy_suit": 0.998,
  "val_per_class_f1": {"rank": {"2": 0.99, ...}, "suit": {"h": 0.99, ...}},
  "class_map": {"rank": ["2","3",...,"A"], "suit": ["c","d","h","s"]},
  "input_size": [48, 64],
  "torch_version": "2.10.0"
}
```

**Versioning rule**: bump `v1 → v2` on any checkpoint retrain, whether
data-only, hyperparam-only, or architecture change. Consumers pin to version
name in `OCR_CLASSIFIER=cnn_v1`; no silent swaps.

**Modified files:**

```
scripts/ocr/table_parser.py
  _find_hero_cards: strip to ~80 lines (localize → call classifier)
  _find_board_cards: strip to ~40 lines (localize → call classifier)
  _identify_cards: deleted (classification merges into call sites)
  DELETE: _ocr_card_rank, _detect_suit_bgr, _hero_hull_norm,
          _suit_template_match, _detect_suit_at, all width-profile /
          hull / green-channel helpers

scripts/ocr/panel_parser.py
  no changes (doesn't currently do card reading — actions/names/stacks only)

scripts/ocr/n8_parser.py
  no changes required (only glues upstream stages)
```

**Deleted files:**

```
scripts/ocr/card_matcher.py      # superseded
scripts/ocr/generate_templates.py
scripts/ocr/templates/           # rank_*.png / suit_*.png no longer used
```

## Data Pipeline

### Source of labels

`analysis_snapshots` table has 159 rows with `image_data IS NOT NULL`. Label
priority:

1. `expected_json.hero_hand` + `expected_json.streets[].board/card` when
   `is_regression = TRUE` (40 hands, human-verified)
2. `parsed_json` equivalents otherwise (Gemini-produced, treat as
   high-confidence label)

### Extraction (`extract_crops.py`)

For each snapshot row:

1. Decode `image_data` bytes → `np.ndarray`
2. Run the **Phase 0 refactored** localization APIs to get crops + metadata
   (hero crops, board crops per street). Localization must now return
   `(crops, boxes)` not `(card_strings, conf)` — see Phase 0 below.
3. Resolve labels per position:
   - hero: `expected.hero_hand or parsed.hero_hand` → two cards
   - board: traverse `streets[].board` (flop, 3 cards) and `streets[].card`
     (turn/river, 1 each) in order → flat list
4. Skip rows where count of localized crops ≠ label count (localization
   edge cases; write to `extract_crops.skipped.log` for later review;
   est. <5%). Never invent labels to fill mismatches.
5. Save crop as `data/cards/{rank}/{suit}/{hand_id}_{source}_{slot}.png`
   where `source ∈ {hero, board}` and `slot` is the ordinal within
   that source.

### Splits

- **Train / val split by `hand_id`**, not per-card. Cards from the same
  screenshot share OCR artifacts (glow, anti-aliasing) and must not straddle
  the split.
- 80% train / 20% val, seeded, stratified by rank.

### Expected counts (from probe)

- 159 hands × 2 hero = 318 hero crops
- Street entries with card(s): 398 → ~800–1000 board cards
- **Total ~1100–1300 labeled crops**
- Per rank: ~90–100 samples average
- Per suit: ~275–325 samples average

Enough for a small CNN to hit 99%+ on this tight visual domain.

### Gold-standard eval set

The 44 regression-flagged snapshots form the **eval set**. They are the
hands we have been getting wrong; if the new classifier handles them, it
beats the current system by definition.

## Model

### Backbone

Tiny CNN, ~50K params, CPU inference <5ms:

```
Input: 1×48×64 grayscale (or 3×48×64 if suit color is learned end-to-end)
  Conv(3×3, 16) → BN → ReLU → MaxPool
  Conv(3×3, 32) → BN → ReLU → MaxPool
  Conv(3×3, 64) → BN → ReLU → AdaptiveAvgPool → flatten
  → shared feature (64-d)
    ├─ Linear(64 → 13)  RankHead
    └─ Linear(64 → 4)   SuitHead
```

Use RGB input (3 channels), not grayscale. Suit color is the highest-signal
feature for red/black; forcing the model to learn it from morphology is
gratuitous.

### Training

- Optimizer: Adam lr=1e-3, weight_decay=1e-4
- Batch size: 64
- Epochs: 50 with early stopping on val loss
- Loss: sum of two cross-entropies (rank_ce + suit_ce)
- Augmentation (mild — the input is a tight crop from a deterministic
  renderer):
  - Random translate ±2px
  - Random brightness ±10%
  - Random Gaussian blur (prob 0.2, sigma ≤ 0.5)
  - **No horizontal flip** — suits are not flip-invariant
- Resize: letterbox crop to 64×48 (preserve aspect; cards are taller than
  wide)

### Inference API

```python
from ocr.classifier.infer import CardClassifier

clf = CardClassifier()          # lazy-loads checkpoint on first classify() call
rank, suit, conf = clf.classify(card_crop_bgr)
# or, batched (preferred when multiple cards are available):
results = clf.classify_batch([crop1, crop2, ...])
# results: list of (rank, suit, conf) in input order
# conf = min(rank_softmax_max, suit_softmax_max)
```

Behaviors and invariants:

- **Lazy import**: `import torch` happens inside `__init__`, not at module
  top. Importing `ocr.classifier.infer` stays cheap (~1ms); the torch cost
  (~500ms) is paid only when a classifier is instantiated.
- **Lazy checkpoint load**: checkpoint is read on first `classify` call, not
  at `__init__`, so constructing the singleton during startup doesn't block.
- **Missing / corrupt checkpoint**: `classify` returns `(None, None, 0.0)`
  for every input and logs `CLASSIFIER_CHECKPOINT_UNAVAILABLE` once per
  process. Zero-confidence flows through the existing hand-level
  `conf > 0.85` gate into the Gemini fallback — no new failure path.
- **torch.no_grad()** + **model.eval()** always
- **CPU only** — no GPU dependency in production container
- **Variable crop sizes**: `classify_batch` internally letterboxes to
  64×48 so callers may pass crops of any size.
- **Deterministic**: no dropout, no randomness at inference.

## Integration

### Wiring into `table_parser.py`

Phase 0 refactor first: split localization from classification. The localized
functions return crops + boxes; a thin wrapper at the call site runs the
classifier and assembles the final card strings. After refactor:

```python
# New localization API (Phase 0 — pure, testable, zero ML dependency):
def _locate_hero_cards(table_region) -> list[np.ndarray]: ...
def _locate_board_cards(table_region) -> list[np.ndarray]: ...

# New table_parser integration (one classifier call per image, batched
# across hero + board):
def _find_cards(table_region):
    hero_crops  = _locate_hero_cards(table_region)
    board_crops = _locate_board_cards(table_region)
    all_crops = hero_crops + board_crops
    if not all_crops:
        return {}, 0.0
    results = _classifier().classify_batch(all_crops)
    # unpack back into sources by ordinal
    ...
    conf = min(r[2] for r in results)
    return {"hero": ..., "board": ...}, conf
```

One batched forward pass for ~7 cards (~1ms) beats 7 separate calls. Batch
across sources at the call site.

### Startup pre-warm

Add a one-line pre-warm to `src/main_gemini.py` startup:

```python
from ocr.classifier.infer import CardClassifier
CardClassifier()._warm()  # loads checkpoint + runs 1 dummy forward pass
```

Pays the ~500ms torch+checkpoint cost once at boot instead of on first real
screenshot. Skippable if `OCR_ENABLED=false`.

### Feature flag and rollout

Add env var `OCR_CLASSIFIER=cnn_v1|legacy` (default `legacy` during Phase 1,
`cnn_v1` after Phase 2 flip).

**Phase 0 — localization refactor** (prerequisite; no behavior change):
- Split `_find_hero_cards` / `_find_board_cards` / panel card readers into
  `_locate_*` (returns crops) + heuristic classification (current behavior).
- Wire legacy classification on top of the new localization API.
- All 44 regression snapshots must still pass with current outputs — this
  is a pure refactor, zero observable change.
- Unlocks `extract_crops.py` and sets up a clean insertion point for the CNN.

**Phase 1 — train + shadow** (1 week in production):
- Run `extract_crops.py` → `train.py` → `eval.py`. Ship checkpoint only if
  val accuracy gate passes.
- In prod, run both classifiers on every hand.
- Log to a new `classifier_shadow_log` table: `(hand_id, source, slot,
  legacy_card, new_card, new_conf_rank, new_conf_suit)`.
- User-facing behavior still driven by legacy.
- **Threshold recalibration**: compute CNN confidence distribution from the
  shadow log. Pick a new `CNN_CONF_THRESHOLD` that matches the legacy
  `0.85` in terms of hand-level fast-path rate. The CNN's softmax is
  smoother than legacy heuristic confidence, so the same number means a
  different thing. Record the chosen threshold in `card_cnn_v1.json`
  metadata.

**Phase 2 — flip default**:
- Ship `OCR_CLASSIFIER=cnn_v1` with the recalibrated threshold.
- Keep `legacy` path for one release as instant rollback.

**Phase 3 — delete legacy** (after 2 weeks stable):
- Remove `card_matcher.py`, `templates/`, all deleted helpers from
  `table_parser.py` and `panel_parser.py`.
- Drop `classifier_shadow_log` table (archival export optional).

### OOD monitoring (post-Phase 3)

Once legacy is deleted, there is no built-in disagreement signal. If Natural8
re-skins cards or ships a UI update, the classifier may confidently
misclassify and nobody notices until users complain. To close that gap:

- Log hand-level `min(card_conf)` to an aggregate metric
  (Supabase `analysis_snapshots.classifier_conf` column, already cheap to
  add).
- Add a line to the existing `weekly_report.py` job: 7-day rolling mean
  hand-level confidence, plus count of hands with any card at `conf < 0.5`.
- Baseline: established during Phase 1 shadow (expect ≥0.95 mean, <2% of
  hands with any card < 0.5).
- Alert condition (manual review, not automated): rolling mean drops by
  more than 0.05, or low-conf share doubles — triggers retrain workflow.

### Runbook: a card is misclassified in production

1. Find the hand_id in snapshots (auto-captured on every analysis)
2. `python -m ocr.classifier.extract_crops --hand H2800` → pulls crops
3. `python scripts/snapshot_test.py --set-expected H2800 '{...}'` if label
   is wrong
4. `python -m ocr.classifier.train` → retrains, writes new checkpoint
5. `python -m ocr.classifier.eval` → confirms regression hands still pass
6. Commit checkpoint + metadata, deploy

No code change in the happy path.

## Testing Strategy

### Unit (`tests/test_card_classifier.py`)

- `CardClassifier.classify` returns `(str, str, float)`; ranks in
  `23456789TJQKA`, suits in `cdhs`, conf in `[0.0, 1.0]`
- `classify_batch` preserves input order (verify with permuted input)
- **[T1]** Inference speed: batch of 13 crops completes in <20ms p95 on
  CPU (<5ms per card). Soft gate — warn, don't block, to accommodate CI
  variance.
- **[T2]** Low-confidence fallback: mocked classifier returning conf=0.80
  causes `parse_n8_screenshot` to not early-return — existing fallback
  path invoked.
- **[T3]** Cross-source generalization: board crops (different size than
  hero) correctly classified via the same classifier.
- **[T4]** End-to-end retrain smoke test: `extract_crops.py --limit 10 →
  train.py --epochs 2 → eval.py` completes without crash on a 10-snapshot
  subset. No accuracy gate at this size; just a plumbing test.
- **[T5]** Crop size normalization: `classify_batch` accepts crops of
  size 30×20, 50×40, 100×75 in one batch and returns results for each.
- **[T-arch]** Missing checkpoint behavior: `CardClassifier` instantiated
  with nonexistent checkpoint path → `classify` returns
  `(None, None, 0.0)` and logs once.

### Accuracy gate (CI / `eval.py`)

- Per-card val accuracy ≥ **99%** (hard gate; release blocker)
- Per-class F1 ≥ **0.95** for every rank and suit (hard gate)
- Eval runs against held-out split by `hand_id` from Phase 1 training data

### Regression (`scripts/snapshot_test.py`)

- All 44 flagged snapshots must pass Layer 1 parse via the classifier
  (hero + board + panel cards all match `expected_json` exactly). Iron
  rule: no exceptions, no skips.
- Run in Phase 1 before shadow deploy, and in Phase 2 before flip.

### Shadow log review (manual, once at end of Phase 1)

- Audit every disagreement row. For each, label as "legacy correct",
  "new correct", or "both wrong".
- "new correct" rows: confirm the classifier is strictly better on these.
- "legacy correct" rows: add to training set, retrain, re-eval.
- "both wrong" rows: flag for human review via
  `snapshot_test.py --set-expected`.

## Deprecation & Cleanup

After Phase 3:

- Delete `scripts/ocr/card_matcher.py`
- Delete `scripts/ocr/generate_templates.py`
- Delete `scripts/ocr/templates/`
- Remove from `table_parser.py`: `_ocr_card_rank`, `_detect_suit_bgr`,
  `_hero_hull_norm`, `_suit_template_match`, `_detect_suit_at`, and all
  width-profile / hull / green-channel helpers
- Remove all H-numbered comments referencing specific hands
- Expected net change: `table_parser.py` 1574 → ≤800 lines

## Open Risks & Mitigations

1. **Hero vs board crops differ** in size/position/render (hero cards
   smaller, often overlap a tournament banner).
   - Mitigation: one shared classifier trained on both sources, resized
     to 64×48. Per-class F1 gate catches any source-specific failure
     mode. If eval reveals genuine distribution shift, we can split into
     two specialized heads — but start with one, because the visual
     content (rank glyph + suit symbol) is identical and learning a
     size-invariant feature is well within CNN reach.
2. **Label noise in the 119 hands without `expected_json`**.
   - Mitigation: during `extract_crops.py`, spot-check any crop whose
     legacy heuristic output disagrees with the label — manually resolve
     before training via `snapshot_test.py --set-expected`. Expected
     <5% of crops.
3. **Localization failures** (blob not found, wrong split).
   - Out of scope. Current rate is low; failures surface as "no hand"
     and fall through to Gemini. We log and leave for future work.
4. **N8 client re-skin** breaks everything.
   - Monitoring via post-Phase 3 OOD signal (rolling mean confidence).
     Mitigation is retrain, which is now a one-command step instead of
     a multi-day heuristic hunt.
5. **Threshold miscalibration after CNN replaces heuristics**.
   - Phase 1 shadow collects CNN softmax distribution vs. legacy heuristic
     conf. Threshold re-tuned before Phase 2 flip (recorded in
     `card_cnn_v1.json`). Without this step, Phase 2 could silently
     over-trust or under-trust the classifier.
6. **Checkpoint in git repo grows over time**.
   - At ~100KB per checkpoint and infrequent retraining, acceptable. Use
     Git LFS only if it becomes a problem.

## Success Criteria

- [ ] Phase 0 localization refactor ships with 44/44 regression pass (zero
      observable change)
- [ ] All 44 regression snapshots pass Layer 1 parse via OCR fast-path after
      Phase 2 flip (hero + board)
- [ ] `table_parser.py` ≤ 800 lines (from 1574)
- [ ] No hand-ID strings (`H\d{4}`) in `scripts/ocr/**` source
- [ ] Per-card val accuracy ≥ 99%, per-class F1 ≥ 0.95 for all 17 classes
- [ ] OOD monitoring line wired into `weekly_report.py` before Phase 3
- [ ] Zero new OCR bugfix commits needed to handle the next 10 reported
      misclassifications (i.e., retraining handles them)
- [ ] End-to-end hand parse latency unchanged (≤ current + 10ms) on the
      fast-path; first screenshot after boot amortized by pre-warm
