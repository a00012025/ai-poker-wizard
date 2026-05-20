# CardCNN v2 + OCR Pipeline 99.9% Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the deterministic pre-Gemini OCR pipeline's `hand_exact` rate from the baseline (~5%) to ≥99.9% on a held-out test set, by retraining the card classifier on the 7,183-hand PokerCraft ground-truth corpus and patching pipeline-level failure modes (position bias, duplicate-card fallback, confidence calibration).

**Architecture:**
- Discard the legacy 984-crop `data/cards/` set (suspected mislabel; PokerCraft and Telegram-uploaded N8 screenshots are the same domain, so retraining on the larger, hh-verified corpus subsumes it).
- Build fresh `data/cards_v2/` from 7,183 PokerCraft replay PNGs paired with `ground_truth.jsonl` labels. Stratified 80/10/10 train/val/test split by `hand_id` (no leakage), with tournament balancing. Persist split list to `data/splits/card_classifier_v2.json` so training and benchmark agree.
- Train CardCNN v2 with stronger backbone (start with deeper BN-regularized CNN; escalate to pretrained MobileNetV3-small if val per-card accuracy < 99.7%). Augmentation specifically targets the observed failure modes: synthetic WIN-sticker overlay (occlusion), color jitter (red-suit confusion), light geometric warps.
- Pipeline patches stack on top: dealer-button template match (fixes "→ SB/BB" position bias), top-2 softmax fallback in `_resolve_hero_board_conflict` (replaces brittle hero-clearing rule), temperature-scaled confidence so the FAST tier ≥0.95 threshold actually means ≥99.9% accurate.
- Reuse `scripts/ocr_precision.py` (already on this branch as untracked) as the headline metric harness. Headline = `hand_exact` on the test-split-only subset; the same 4 critical fields the existing benchmark grades.

**Tech Stack:** PyTorch (existing), torchvision (new — for MobileNetV3 if escalated), OpenCV, EasyOCR (existing), Python 3.13.

**Branch:** `feat/card-classifier-v2` (off `main`).

---

## File Structure

**New files:**
- `scripts/ocr/classifier/extract_pokercraft_crops.py` — extracts hero+board crops from `data/hand_images/img/*.png` using GT labels, writes to `data/cards_v2/{rank}/{suit}/{hand_id}_{src}_{slot}.png`.
- `scripts/ocr/classifier/split.py` — stratified `hand_id` → `train|val|test` split with tournament balancing; CLI emits `data/splits/card_classifier_v2.json`.
- `scripts/ocr/classifier/augment.py` — augmentation transforms (synthetic WIN overlay, color jitter, geometric warps) usable both at training time and from unit tests.
- `scripts/ocr/classifier/calibrate.py` — temperature scaling against val split; updates `card_cnn_v2.json` with the fitted T.
- `scripts/ocr/button_detector.py` — dealer-button template match, returns `(seat_idx, conf) | None`; consumed by `n8_parser` to derive positions algebraically.
- `scripts/ocr/models/card_cnn_v2.pt` + `card_cnn_v2.json` — new checkpoint and metadata (val/test acc, conf threshold, fitted temperature, data hash).
- `data/splits/card_classifier_v2.json` — frozen split (gitignored under `data/`).

**Modified files:**
- `scripts/ocr/classifier/model.py` — adds `CardCNNv2` (deeper, BN, dropout) alongside `CardCNN`; selected by metadata `version` field at load time.
- `scripts/ocr/classifier/dataset.py` — adds split-aware loading (filter samples by hand_id ∈ split) and the new augmentation hook.
- `scripts/ocr/classifier/train.py` — switches to v2 model, uses split JSON instead of `split_by_hand_id`, plumbs augmentation flag, logs per-class F1 + confusion matrix.
- `scripts/ocr/classifier/infer.py` — picks model class from metadata `version`, optional temperature application.
- `scripts/ocr/classifier/eval.py` — adds `--split test` flag to evaluate on the persisted holdout.
- `scripts/ocr/n8_parser.py` — replaces `_resolve_hero_board_conflict` clearout with top-2 fallback per CNN head; calls `button_detector` to anchor hero position.
- `scripts/ocr/table_parser.py` — passes through `rank_top2`/`suit_top2` from the classifier so the parser can implement the fallback.
- `scripts/regression_test.py` — new tests for the splitter, augment transforms, top-2 fallback, button detector.
- `scripts/ocr_precision.py` — already drafted on this branch (in untracked state). Stage and commit.

**Untouched but relevant:**
- `scripts/ocr/table_parser.py`'s `_locate_hero_cards` / `_locate_board_cards` — reused as-is by the extractor.
- `scripts/hh_parser.py` — out of scope (already correct after PR #21).

---

## Task 1: Stage existing benchmark + run baseline

**Why first:** Baseline must be measured on `main` semantics before any code change so the PR can present an honest before/after. The baseline run also occupies the GPU for ~45 min — we want it started while we write code.

**Files:**
- Stage: `scripts/ocr_precision.py`, `scripts/ocr_failure_gallery.py` (already on disk from a previous session, untracked).
- Create: `data/ocr_precision_baseline/` (output).

- [ ] **Step 1: Confirm branch off main**

```bash
git rev-parse --abbrev-ref HEAD
git log --oneline -1
```
Expected: `feat/card-classifier-v2` and the latest main commit (`daed364 feat: OCR ground-truth benchmark …`).

- [ ] **Step 2: Stage benchmark scripts**

```bash
git add scripts/ocr_precision.py scripts/ocr_failure_gallery.py
git status
```
Expected: both files staged, no other changes.

- [ ] **Step 3: Commit benchmark harness**

```bash
git commit -m "$(cat <<'EOF'
chore(ocr): add pre-Gemini precision harness + failure gallery

ocr_precision.py runs parse_n8_screenshot() standalone against the
PokerCraft ground-truth corpus and reports hand_exact / per-field
accuracy / confidence calibration. ocr_failure_gallery.py groups the
diffs by failure mode into annotated PNG montages for visual review.
Both are reused by the v2 training-and-benchmark workflow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: one commit.

- [ ] **Step 4: Kick off baseline benchmark (background, ~45 min)**

```bash
python scripts/ocr_precision.py --workers 6 --max-failures 400 \
  --out data/ocr_precision_baseline 2>&1 | tee /tmp/baseline.log
```
Expected: run starts, prints `[ocr_precision] paired=7183 …`, drips `25/7183 exact=…` progress lines. Capture pid so we can monitor.

- [ ] **Step 5: While baseline runs, proceed to Task 2.** Periodically check `tail /tmp/baseline.log`. Done when the file ends with `=========` summary block and `data/ocr_precision_baseline/summary.json` exists.

---

## Task 2: Stratified train/val/test split

**Why:** Every downstream step (extractor write paths, training subset, benchmark subset) reads from one frozen split. Build it first so the split is deterministic and reviewable.

**Files:**
- Create: `scripts/ocr/classifier/split.py`
- Create: `data/splits/` (dir; `data/` is gitignored, the split JSON lives here).
- Test: `scripts/regression_test.py` (new test class `test_card_split_*`).

- [ ] **Step 1: Write failing test for stratified split**

In `scripts/regression_test.py`, add:

```python
@test
def test_card_split_no_hand_leakage():
    """A hand_id appearing in train must not appear in val or test."""
    from ocr.classifier.split import build_split
    gt_path = REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    split = build_split(gt_path, train=0.8, val=0.1, test=0.1, seed=0)
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    assert_eq(len(train_ids & val_ids), 0)
    assert_eq(len(train_ids & test_ids), 0)
    assert_eq(len(val_ids & test_ids), 0)
    # 80/10/10 within ±2% (rounding slack)
    total = len(train_ids) + len(val_ids) + len(test_ids)
    assert_true(0.78 <= len(train_ids)/total <= 0.82)
    assert_true(0.08 <= len(val_ids)/total <= 0.12)
    assert_true(0.08 <= len(test_ids)/total <= 0.12)


@test
def test_card_split_tournament_balanced():
    """Every tournament with >=10 hands appears in all three splits."""
    from ocr.classifier.split import build_split
    import json
    gt_path = REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    split = build_split(gt_path, train=0.8, val=0.1, test=0.1, seed=0)
    tourney_in = {"train": set(), "val": set(), "test": set()}
    hid_to_tid = {}
    with gt_path.open() as fh:
        for line in fh:
            o = json.loads(line)
            hid_to_tid[o["hand_id"]] = o["ground_truth"].get("tournament_id")
    from collections import Counter
    big_tourneys = {t for t, n in Counter(hid_to_tid.values()).items() if n and n >= 10}
    for bucket in ("train", "val", "test"):
        for hid in split[bucket]:
            tourney_in[bucket].add(hid_to_tid.get(hid))
    for t in big_tourneys:
        for bucket in ("train", "val", "test"):
            assert_in(t, tourney_in[bucket],
                      f"tourney {t} missing from {bucket}")
```

- [ ] **Step 2: Run tests, confirm fail (module not found)**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_card_split|FAIL"
```
Expected: both tests FAIL with `ModuleNotFoundError: ocr.classifier.split` or `assert_*` failure.

- [ ] **Step 3: Implement `scripts/ocr/classifier/split.py`**

```python
"""Stratified hand_id -> {train,val,test} split for CardCNN v2.

Stratifies by tournament_id so every tournament contributes to all three
splits (per-tournament UI variations get represented in train AND
evaluated against in test). Within a tournament, deterministically
shuffles hand_ids with the given seed and slices by the requested
fractions. Tournaments with fewer than 10 hands fall back to all-in-train
to avoid singleton test entries.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def build_split(
    gt_path: Path,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    min_tourney_for_split: int = 10,
) -> dict:
    assert abs((train + val + test) - 1.0) < 1e-6, "fractions must sum to 1"
    by_tourney: dict[str, list[str]] = defaultdict(list)
    with Path(gt_path).open() as fh:
        for line in fh:
            o = json.loads(line)
            t = o["ground_truth"].get("tournament_id") or "_unknown"
            by_tourney[t].append(o["hand_id"])

    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": [], "meta": {
        "seed": seed, "fractions": {"train": train, "val": val, "test": test},
        "gt_path": str(gt_path), "tournaments": len(by_tourney),
    }}
    for t, hids in by_tourney.items():
        hids = sorted(hids)            # determinism
        rng.shuffle(hids)
        n = len(hids)
        if n < min_tourney_for_split:
            out["train"].extend(hids)
            continue
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        # any rounding remainder lands in test
        out["train"].extend(hids[:n_train])
        out["val"].extend(hids[n_train:n_train + n_val])
        out["test"].extend(hids[n_train + n_val:])
    for k in ("train", "val", "test"):
        out[k].sort()
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--out", default="data/splits/card_classifier_v2.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    split = build_split(Path(args.gt), seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2))
    print(f"train={len(split['train'])} val={len(split['val'])} "
          f"test={len(split['test'])} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Re-run tests, confirm pass**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_card_split"
```
Expected: both tests PASS.

- [ ] **Step 5: Generate the frozen split**

```bash
python scripts/ocr/classifier/split.py
```
Expected output: `train=~5746 val=~718 test=~719 -> data/splits/card_classifier_v2.json`.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/classifier/split.py scripts/regression_test.py
git commit -m "feat(card-classifier): stratified hand_id split utility

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: PokerCraft crop extractor

**Why:** Builds the new `data/cards_v2/` training corpus from 7,183 GT images using existing localizers + GT labels. Discards old `data/cards/`. Crops are written once and reused across many training runs.

**Files:**
- Create: `scripts/ocr/classifier/extract_pokercraft_crops.py`
- Output: `data/cards_v2/{rank}/{suit}/*.png`, `data/cards_v2_skipped.log`
- Test: `scripts/regression_test.py` (extractor smoke test against a fixture image).

- [ ] **Step 1: Write a smoke test**

Pick one image known to parse cleanly (e.g., from previous benchmark `parsed_streets` matching `gt_streets`). Add:

```python
@test
def test_extract_crops_smoke():
    """Extractor produces the right number of crops for one known-good
    PokerCraft image and labels them with GT cards."""
    from ocr.classifier.extract_pokercraft_crops import extract_one
    import json
    hid = "TM5846884226"   # 4s5h hero, flop 8d2s7h (from earlier session data)
    gt_row = None
    with open(REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl") as fh:
        for line in fh:
            o = json.loads(line)
            if o["hand_id"] == hid:
                gt_row = o["ground_truth"]; break
    assert_true(gt_row is not None, f"GT row for {hid} missing")
    img_path = REPO_ROOT / f"data/hand_images/img/{hid}.png"
    assert_true(img_path.exists(), f"image missing")
    result = extract_one(img_path.read_bytes(), gt_row)
    # 2 hero + some board cards (>= 0; this hand had a flop)
    assert_eq(len(result["hero_crops"]), 2)
    assert_eq(result["hero_labels"], ["4s", "5h"])
    # crops are numpy arrays
    import numpy as np
    for c in result["hero_crops"]:
        assert_true(isinstance(c, np.ndarray) and c.shape[0] > 0)
```

- [ ] **Step 2: Run test, confirm fail (module not found)**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_extract_crops"
```
Expected: FAIL.

- [ ] **Step 3: Implement extractor**

```python
"""Extract labeled hero/board crops from PokerCraft replay screenshots.

Reads each <hand_id>.png from data/hand_images/img/, looks up
ground_truth.jsonl for the labels, runs detect_regions ->
_locate_hero_cards / _locate_board_cards on the table region, then
writes crops to data/cards_v2/{rank}/{suit}/{hand_id}_{src}_{slot}.png.

Mismatches (crop count != label count) are logged and skipped — labels
are never invented to pad. This is the same contract as the legacy
extract_crops.py against analysis_snapshots, only the data source moves
from Supabase to disk and labels move from parsed_json to GT.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import cv2
import numpy as np

from ocr.region_detector import detect_regions
from ocr.table_parser import _locate_hero_cards, _locate_board_cards

_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")
OUT_ROOT = REPO_ROOT / "data" / "cards_v2"
SKIP_LOG = REPO_ROOT / "data" / "cards_v2_skipped.log"


def _split_cards(s: str) -> list[str]:
    s = (s or "").replace(" ", "")
    return [s[i:i + 2] for i in range(0, len(s) - 1, 2)]


def _gt_labels(gt: dict) -> tuple[list[str], list[str]]:
    hero = [c for c in _split_cards(gt.get("hero_hand", "")) if _CARD_RE.match(c)]
    board: list[str] = []
    for st in gt.get("streets", []) or []:
        if st.get("board"):
            board.extend(c for c in _split_cards(st["board"]) if _CARD_RE.match(c))
        if st.get("card"):
            c = st["card"]
            if _CARD_RE.match(c):
                board.append(c)
    return hero, board


def extract_one(image_bytes: bytes, gt: dict) -> dict:
    """Return labeled crops for one image. Caller writes to disk."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"hero_crops": [], "hero_labels": [], "board_crops": [],
                "board_labels": [], "reason": "decode_failed"}
    regions = detect_regions(img)
    if regions is None:
        return {"hero_crops": [], "hero_labels": [], "board_crops": [],
                "board_labels": [], "reason": "region_detect_failed"}
    table = regions["table"]
    hero_labels, board_labels = _gt_labels(gt)
    hero_crops = _locate_hero_cards(table)
    board_crops = _locate_board_cards(table)
    return {
        "hero_crops": hero_crops, "hero_labels": hero_labels,
        "board_crops": board_crops, "board_labels": board_labels,
        "reason": None,
    }


def _save(crop: np.ndarray, rank: str, suit: str, hand_id: str,
          source: str, slot: int) -> None:
    d = OUT_ROOT / rank / suit
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{hand_id}_{source}_{slot}.png"), crop)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--images", default="data/hand_images/img")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.write_text("")

    gt_map = {}
    with open(args.gt) as fh:
        for line in fh:
            o = json.loads(line)
            gt_map[o["hand_id"]] = o["ground_truth"]

    img_dir = Path(args.images)
    paths = sorted(img_dir.glob("*.png"))
    if args.limit:
        paths = paths[:args.limit]

    n_imgs = 0
    n_hero = 0
    n_board = 0
    n_skip = 0
    for p in paths:
        n_imgs += 1
        gt = gt_map.get(p.stem)
        if gt is None:
            continue
        r = extract_one(p.read_bytes(), gt)
        if r["reason"]:
            with SKIP_LOG.open("a") as f:
                f.write(f"{p.stem}\t{r['reason']}\n")
            n_skip += 1
            continue
        # Hero crops
        if r["hero_labels"] and len(r["hero_crops"]) == len(r["hero_labels"]):
            for i, (crop, lbl) in enumerate(zip(r["hero_crops"], r["hero_labels"])):
                m = _CARD_RE.match(lbl)
                _save(crop, m.group(1), m.group(2), p.stem, "hero", i)
                n_hero += 1
        elif r["hero_labels"]:
            with SKIP_LOG.open("a") as f:
                f.write(f"{p.stem}\thero_count crops={len(r['hero_crops'])} "
                        f"labels={len(r['hero_labels'])}\n")
        # Board crops
        if r["board_labels"] and len(r["board_crops"]) == len(r["board_labels"]):
            for i, (crop, lbl) in enumerate(zip(r["board_crops"], r["board_labels"])):
                m = _CARD_RE.match(lbl)
                _save(crop, m.group(1), m.group(2), p.stem, "board", i)
                n_board += 1
        elif r["board_labels"]:
            with SKIP_LOG.open("a") as f:
                f.write(f"{p.stem}\tboard_count crops={len(r['board_crops'])} "
                        f"labels={len(r['board_labels'])}\n")
        if n_imgs % 250 == 0:
            print(f"  {n_imgs}/{len(paths)}  hero={n_hero}  board={n_board}  "
                  f"skipped={n_skip}", flush=True)
    print(f"DONE  images={n_imgs}  hero={n_hero}  board={n_board}  "
          f"skipped={n_skip}  log={SKIP_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test, confirm pass**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_extract_crops"
```
Expected: PASS.

- [ ] **Step 5: Smoke run on 50 images**

```bash
python scripts/ocr/classifier/extract_pokercraft_crops.py --limit 50
ls data/cards_v2 | head; find data/cards_v2 -name "*.png" | wc -l
```
Expected: per-rank dirs, ~150-250 crops total.

- [ ] **Step 6: Full extraction**

```bash
rm -rf data/cards_v2
python scripts/ocr/classifier/extract_pokercraft_crops.py
```
Expected (target order of magnitude): `images=7183 hero=~14000 board=~22000 skipped<200`.

- [ ] **Step 7: Audit per-class counts**

Write `scripts/_tmp.py` to walk `data/cards_v2/` and print per-(rank,suit) counts. Smallest class should have ≥150 examples. If any is <50, surface that for review (probably localizer bug, not class imbalance — both ranks and suits appear roughly uniformly in hh).

- [ ] **Step 8: Commit extractor (don't commit crops — `data/` is gitignored)**

```bash
git add scripts/ocr/classifier/extract_pokercraft_crops.py scripts/regression_test.py
git commit -m "feat(card-classifier): pokercraft GT crop extractor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Augmentation library

**Why:** The mined failure pattern is dominated by `d → h` (35% of suit errors), A/T magnet bias, and WIN-sticker occlusion. Augmentation that targets these effects should fix them without architecture changes.

**Files:**
- Create: `scripts/ocr/classifier/augment.py`
- Test: `scripts/regression_test.py` (new `test_augment_*`).

- [ ] **Step 1: Write failing tests**

```python
@test
def test_augment_win_sticker_overlays_yellow():
    """Synthetic WIN sticker leaves a yellow region on a non-yellow base."""
    import numpy as np
    from ocr.classifier.augment import apply_win_sticker
    base = np.full((192, 128, 3), 50, dtype=np.uint8)   # dark gray
    rng = np.random.default_rng(0)
    out = apply_win_sticker(base, rng=rng, p=1.0)       # forced on
    # At least some pixels should be much brighter on red+green channels (yellow)
    yellow_mask = (out[..., 2] > 150) & (out[..., 1] > 150) & (out[..., 0] < 100)
    assert_true(yellow_mask.sum() > 100,
                f"WIN sticker did not write yellow pixels: {yellow_mask.sum()}")


@test
def test_augment_color_jitter_preserves_dimensions():
    import numpy as np
    from ocr.classifier.augment import color_jitter
    base = np.full((192, 128, 3), 128, dtype=np.uint8)
    out = color_jitter(base, rng=np.random.default_rng(0), strength=0.3)
    assert_eq(out.shape, base.shape)
    assert_eq(out.dtype, np.uint8)
```

- [ ] **Step 2: Run test, confirm fail**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_augment"
```
Expected: FAIL.

- [ ] **Step 3: Implement augment.py**

```python
"""Augmentation transforms tuned to the observed CardCNN failure modes.

- apply_win_sticker: synthesize a yellow "WIN" rounded rectangle over the
  card (size, position, opacity sampled). Mirrors the showdown overlay
  PokerCraft puts on the hero's winning hand.
- color_jitter: per-channel multiplicative jitter — targets red-suit
  ambiguity (d <-> h is the #1 suit confusion: 55/155 of suit errors).
- light_geometric: ±2px shift + ±2deg rotation, mild scale. Bounded so the
  ~10px rank glyph isn't destroyed (the v1 dataset comment explicitly
  warned against translation).
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_win_sticker(img: np.ndarray, *, rng: np.random.Generator,
                       p: float = 0.25) -> np.ndarray:
    if rng.random() > p:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    sw = rng.integers(int(w * 0.5), int(w * 0.95))
    sh = rng.integers(int(h * 0.18), int(h * 0.32))
    x0 = rng.integers(0, max(1, w - sw))
    y0 = rng.integers(int(h * 0.25), max(int(h * 0.25) + 1, h - sh))
    alpha = float(rng.uniform(0.6, 0.95))
    overlay = out.copy()
    # BGR yellow with a touch of orange variation
    color = (
        int(rng.integers(0, 80)),
        int(rng.integers(180, 230)),
        int(rng.integers(220, 255)),
    )
    cv2.rectangle(overlay, (x0, y0), (x0 + sw, y0 + sh), color, thickness=-1)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def color_jitter(img: np.ndarray, *, rng: np.random.Generator,
                  strength: float = 0.2) -> np.ndarray:
    factors = 1.0 + (rng.random(3) - 0.5) * 2 * strength
    out = img.astype(np.float32)
    out *= factors.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_geometric(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    angle = float(rng.uniform(-2.0, 2.0))
    scale = float(rng.uniform(0.97, 1.03))
    tx = float(rng.integers(-2, 3))
    ty = float(rng.integers(-2, 3))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)


def apply_all(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    img = light_geometric(img, rng=rng)
    img = color_jitter(img, rng=rng, strength=0.20)
    img = apply_win_sticker(img, rng=rng, p=0.25)
    return img
```

- [ ] **Step 4: Run test, confirm pass**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_augment"
```
Expected: PASS.

- [ ] **Step 5: Eyeball — write 8 augmented samples to disk for visual check**

```bash
python scripts/_tmp.py  # script: load one crop, apply_all 8x, save to /tmp/aug_*.png
ls /tmp/aug_*.png
```
Open in viewer. Sticker should look plausible, color jitter mild, no destruction of the rank glyph.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/classifier/augment.py scripts/regression_test.py
git commit -m "feat(card-classifier): augmentation library (WIN sticker, color jitter, light geometric)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CardCNN v2 architecture + dataset wiring

**Why:** Existing 4-layer model with shared backbone is borderline for 13×4 joint classes at 192×128. We bump to deeper conv stack with dropout regularization. If this still plateaus below 99.7% per-card we escalate to pretrained MobileNetV3 in Task 7.

**Files:**
- Modify: `scripts/ocr/classifier/model.py`
- Modify: `scripts/ocr/classifier/dataset.py`
- Modify: `scripts/ocr/classifier/train.py`
- Test: regression_test new `test_card_cnn_v2_*`.

- [ ] **Step 1: Write test for v2 forward shape**

```python
@test
def test_card_cnn_v2_forward_shape():
    import torch
    from ocr.classifier.model import CardCNNv2, RANK_CLASSES, SUIT_CLASSES
    net = CardCNNv2()
    net.eval()
    x = torch.zeros(2, 3, 192, 128)
    rl, sl = net(x)
    assert_eq(rl.shape, (2, len(RANK_CLASSES)))
    assert_eq(sl.shape, (2, len(SUIT_CLASSES)))
```

- [ ] **Step 2: Confirm fail (class not defined)**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_card_cnn_v2"
```

- [ ] **Step 3: Add `CardCNNv2` to `scripts/ocr/classifier/model.py`**

```python
class CardCNNv2(nn.Module):
    """Deeper backbone with BN + dropout; same heads as v1.

    v1: 4 conv blocks (16->32->64->128), pool=4, 2048-D feat, 17 head params.
    v2: 5 conv blocks (32->64->128->192->256), pool=4, 4096-D feat,
        dropout(0.3) on the feature head, BN through every conv.
    """
    def __init__(self):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.backbone = nn.Sequential(
            block(3, 32),    # 96 x 64
            block(32, 64),   # 48 x 32
            block(64, 128),  # 24 x 16
            block(128, 192), # 12 x 8
            block(192, 256), # 6 x 4
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        feat = 256 * 4 * 4
        self.rank_head = nn.Linear(feat, len(RANK_CLASSES))
        self.suit_head = nn.Linear(feat, len(SUIT_CLASSES))

    def forward(self, x):
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)
```

- [ ] **Step 4: Update dataset to use the persisted split + augmentation**

In `scripts/ocr/classifier/dataset.py`, replace `split_by_hand_id` callers with a `from_split_json(root, split_path, bucket, augment_fn=None)` classmethod that filters samples by `hand_id ∈ split[bucket]` and applies `augment_fn` when augment=True.

(Full code shown in Task 6 below — they're co-developed.)

- [ ] **Step 5: Re-run forward-shape test**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_card_cnn_v2"
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/classifier/model.py scripts/ocr/classifier/dataset.py scripts/regression_test.py
git commit -m "feat(card-classifier): CardCNNv2 deeper backbone + split-aware dataset

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Train v2

**Why:** This is the lever. If retraining lifts hand_exact close to target, Tasks 9 (top-2 fallback) and 10 (button match) are likely sufficient to bridge the rest. If val accuracy stalls, escalate (Task 7).

**Files:**
- Modify: `scripts/ocr/classifier/train.py`
- Output: `scripts/ocr/models/card_cnn_v2.pt`, `card_cnn_v2.json`

- [ ] **Step 1: Update `train.py` to use v2 model + split + augmentation**

Key changes (full file below):
- Import `CardCNNv2` (and add `--arch v1|v2` flag, default v2).
- Use `data/splits/card_classifier_v2.json` for split (drop `split_by_hand_id`).
- Wrap train dataset in augmentation; val/test stay un-augmented.
- Log per-class F1 on val every 10 epochs; early stop on val_loss patience=30.
- Write checkpoint to `card_cnn_v2.pt` and meta with: `version: "v2"`, splits hash, per-class F1, val_acc_card (joint rank+suit), val_acc_rank, val_acc_suit.

Reference inputs:
- Existing `train.py` has the optimizer/scheduler boilerplate — keep it.
- Add `--data data/cards_v2` and `--split data/splits/card_classifier_v2.json` flags.

- [ ] **Step 2: Smoke train (5 epochs) to verify the loop runs end-to-end**

```bash
python -m scripts.ocr.classifier.train --epochs 5 \
    --data data/cards_v2 --split data/splits/card_classifier_v2.json
```
Expected: completes; val_acc rank+suit > 0.5 (well above random — confirms not broken).

- [ ] **Step 3: Full train**

```bash
time python -m scripts.ocr.classifier.train \
    --epochs 150 --data data/cards_v2 \
    --split data/splits/card_classifier_v2.json
```
Expected (target): val_acc_rank ≥ 0.997, val_acc_suit ≥ 0.997. If hit, proceed to Task 8. If either < 0.997, go to Task 7.

- [ ] **Step 4: Per-class audit**

Open `scripts/ocr/models/card_cnn_v2.json`. Surface any class with F1 < 0.99. Hypothesis-check: did the augmentation specifically help `d`? If `d` F1 still < 0.99, add per-suit oversampling and re-train.

- [ ] **Step 5: Commit checkpoint metadata (the .pt itself is in `scripts/ocr/models/`; small, do commit it)**

```bash
git add scripts/ocr/classifier/train.py scripts/ocr/models/card_cnn_v2.pt scripts/ocr/models/card_cnn_v2.json
git commit -m "feat(card-classifier): train v2 on PokerCraft GT corpus

val_acc rank=<X>%  suit=<Y>%  (worst class F1: <Z>)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Architecture escalation (conditional)

**Skip this task if Task 6 hit ≥99.7% on both heads.**

**Why:** If the deeper CNN plateaus, the model class is the bottleneck — switch to a pretrained MobileNetV3-small backbone (1.5M params, fast on CPU, ImageNet-pretrained features transfer well to glyph recognition).

**Files:**
- Modify: `scripts/ocr/classifier/model.py` (add `CardCNNv2Mobile`).
- Modify: `scripts/ocr/classifier/infer.py` (load by `version`).
- Modify: `scripts/ocr/classifier/train.py` (`--arch mobilenet`).

- [ ] **Step 1: Add MobileNetV3-small head**

```python
class CardCNNv2Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        backbone.classifier = nn.Identity()
        self.backbone = backbone   # outputs 576-D feature
        self.rank_head = nn.Linear(576, len(RANK_CLASSES))
        self.suit_head = nn.Linear(576, len(SUIT_CLASSES))

    def forward(self, x):
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)
```

- [ ] **Step 2: Train**

```bash
time python -m scripts.ocr.classifier.train \
    --epochs 80 --arch mobilenet \
    --lr 5e-4 \
    --data data/cards_v2 --split data/splits/card_classifier_v2.json
```
(Lower LR because backbone is pretrained.)
Expected: val_acc_rank ≥ 0.999, val_acc_suit ≥ 0.999.

- [ ] **Step 3: If still short, surface a focused diff**

Write a script to confusion-matrix the remaining errors. Decide: (a) target the failing class with hard-negative mining, or (b) escalate further (ResNet-18, ViT-tiny). Don't ship anything below 99.7% per-card.

- [ ] **Step 4: Commit best checkpoint**

```bash
git add scripts/ocr/classifier/model.py scripts/ocr/classifier/train.py scripts/ocr/classifier/infer.py scripts/ocr/models/card_cnn_v2.pt scripts/ocr/models/card_cnn_v2.json
git commit -m "feat(card-classifier): MobileNetV3 backbone (val_acc card=<X>%)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Confidence temperature scaling

**Why:** Production trust gates (FAST ≥0.95, MEDIUM ≥0.80) only work if softmax confidence is calibrated. Today, max-conf-when-wrong = 0.96. Fit a single temperature on val to make the probability match accuracy.

**Files:**
- Create: `scripts/ocr/classifier/calibrate.py`
- Modify: `scripts/ocr/classifier/infer.py` (read `temperature` from meta JSON).
- Modify: `card_cnn_v2.json` (add `temperature`).

- [ ] **Step 1: Write test (temperature must reduce ECE)**

```python
@test
def test_temperature_scaling_lowers_ece():
    """Calibrated softmax on val should have lower expected-calibration-
    error than uncalibrated."""
    from ocr.classifier.calibrate import fit_temperature, ece
    import torch
    torch.manual_seed(0)
    # Synthetic: 1000 examples, 10 classes, logits with high variance
    logits = torch.randn(1000, 10) * 3.0
    labels = logits.argmax(1)
    # Flip 10% of labels to force miscalibration
    mask = torch.randperm(1000)[:100]
    labels[mask] = (labels[mask] + 1) % 10
    T = fit_temperature(logits, labels)
    ece_before = ece(torch.softmax(logits, dim=1), labels)
    ece_after = ece(torch.softmax(logits / T, dim=1), labels)
    assert_true(ece_after < ece_before, f"ECE not reduced: {ece_before} -> {ece_after}")
```

- [ ] **Step 2: Implement calibrate.py**

Standard temperature scaling: optimize a scalar T against NLL on val. Provide `fit_temperature(logits, labels) -> float` and `ece(probs, labels) -> float`.

- [ ] **Step 3: Run calibration on the v2 val logits**

```bash
python scripts/ocr/classifier/calibrate.py \
    --ckpt scripts/ocr/models/card_cnn_v2.pt \
    --split data/splits/card_classifier_v2.json \
    --bucket val
```
Expected: prints `T_rank=…  T_suit=…  ECE_rank: X→Y  ECE_suit: X→Y`. Writes T values into `card_cnn_v2.json`.

- [ ] **Step 4: Update `infer.py` to apply T**

Inside `classify_batch_detailed`, divide logits by `T_rank` / `T_suit` (loaded from meta) before softmax.

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/classifier/calibrate.py scripts/ocr/classifier/infer.py scripts/ocr/models/card_cnn_v2.json scripts/regression_test.py
git commit -m "feat(card-classifier): temperature scaling for v2 confidence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Top-2 fallback in `_resolve_hero_board_conflict`

**Why:** When CNN's top-1 hero card collides with the board, the current code wipes both hero cards (29% of failures). With a calibrated v2 model the right rule is: try the next-best card from each head's softmax; only clear if every plausible candidate collides.

**Files:**
- Modify: `scripts/ocr/n8_parser.py:41-65` (`_resolve_hero_board_conflict`)
- Modify: `scripts/ocr/table_parser.py` (return `rank_top2` and `suit_top2` per hero card)
- Test: `scripts/regression_test.py`.

- [ ] **Step 1: Write test**

```python
@test
def test_resolve_hero_uses_top2_when_top1_collides():
    """Hero CNN top1 collides with board; top2 doesn't — keep top2."""
    from ocr.n8_parser import _resolve_hero_board_conflict
    board = ["Kc", "9d", "3h"]
    # hero_details: each is dict with top2 rank+suit candidates
    hero_details = [
        {"rank": "K", "rank_top2": [("K", 0.6), ("Q", 0.35)],
         "suit": "c", "suit_top2": [("c", 0.7), ("d", 0.2)],
         "conf": 0.6},
        {"rank": "A", "rank_top2": [("A", 0.9), ("K", 0.05)],
         "suit": "s", "suit_top2": [("s", 0.9), ("h", 0.05)],
         "conf": 0.9},
    ]
    new_board, new_hero = _resolve_hero_board_conflict(board, ["Kc", "As"],
                                                       hero_details=hero_details)
    # Expect first hero card flipped to "Qc" (the top2 rank, top1 suit) since
    # "Kc" already on board and "Qc" is collision-free.
    assert_eq(new_hero, ["Qc", "As"])
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement top-2 fallback**

```python
def _resolve_hero_board_conflict(
    board_cards, hero_cards, *, hero_details: list[dict] | None = None
):
    if not (board_cards and hero_cards):
        return board_cards, hero_cards
    bset = set(board_cards)
    if not (set(hero_cards) & bset) and len(set(hero_cards)) == len(hero_cards):
        return board_cards, hero_cards   # nothing to fix
    if hero_details is None:
        # No top-2 available -> legacy clear behavior
        log.warning("Duplicate cards (no top-2): board=%s hero=%s — clearing hero",
                    board_cards, hero_cards)
        return board_cards, []
    fixed = list(hero_cards)
    for i, d in enumerate(hero_details):
        if i >= len(fixed):
            continue
        if fixed[i] not in bset and fixed.count(fixed[i]) == 1:
            continue
        # Try top-2 combinations sorted by joint prob
        candidates = []
        for r, rp in d.get("rank_top2", [])[:2]:
            for s, sp in d.get("suit_top2", [])[:2]:
                candidates.append((r + s, rp * sp))
        candidates.sort(key=lambda x: -x[1])
        for cand, _ in candidates:
            if cand not in bset and cand not in [fixed[j] for j in range(len(fixed)) if j != i]:
                fixed[i] = cand
                break
        else:
            log.warning("Hero card %d: all candidates collide; clearing", i)
            return board_cards, []
    return board_cards, fixed
```

- [ ] **Step 4: Update `table_parser` to populate top2** — pull from `CardClassifier.classify_batch_detailed` (which already returns softmax probs internally; expose top2 via new keys).

- [ ] **Step 5: Run tests**

```bash
python scripts/regression_test.py 2>&1 | grep -E "test_resolve_hero|test_ocr_table_parser"
```
Expected: new test PASS, existing OCR tests still PASS.

- [ ] **Step 6: Commit**

---

## Task 10: Dealer-button position anchor

**Why:** 11 of 20 captured position errors were "true_pos → SB/BB" (OCR placed hero at the bottom seat regardless). The dealer button is a small white "D" disc that's a deterministic template match, independent of hero detection.

**Files:**
- Create: `scripts/ocr/button_detector.py`
- Modify: `scripts/ocr/n8_parser.py:159-205` (position assignment)
- Test: regression_test.

- [ ] **Step 1: Write test against a fixture image with known button position**

(Pick an image where preview clearly shows the D button on a known seat, set the GT.)

```python
@test
def test_button_detector_picks_known_seat():
    from ocr.button_detector import detect_button
    import cv2
    img = cv2.imread("data/hand_images/img/<KNOWN_BUTTON_HAND>.png")
    result = detect_button(img)
    assert_true(result is not None)
    seat_idx, conf = result
    assert_eq(seat_idx, EXPECTED_BTN_SEAT)
    assert_true(conf > 0.6)
```

- [ ] **Step 2: Implement detector**

Template matching on a precomputed "D" disc template. The template lives in `scripts/ocr/templates/dealer_button.png` (extract from one known image, cropped to 24×24-ish). Use `cv2.matchTemplate(method=TM_CCOEFF_NORMED)`. Find global max; project the (x,y) onto the table's seat layout to get a seat index.

- [ ] **Step 3: Wire into `n8_parser._assemble_hand`** — when the button is detected confidently, derive `positions[seat_to_pos[btn_seat + 1]] = SB` algebraically. Override hero-position-by-bottom-seat heuristic when button conf > 0.7.

- [ ] **Step 4: Add regression test for the end-to-end position fix on one image**

- [ ] **Step 5: Commit**

---

## Task 11: Final benchmark on test split + comparison

**Why:** Honest before/after on the held-out 10%. This is the headline number the PR will be reviewed against.

**Files:**
- Modify: `scripts/ocr_precision.py` (add `--split <path> --bucket test` flag so it only evaluates hand_ids in the test split)
- Output: `data/ocr_precision_final/` (summary + diffs).

- [ ] **Step 1: Add split-filter flag**

```python
ap.add_argument("--split", default="")
ap.add_argument("--bucket", default="test", choices=["train", "val", "test"])
# After loading gt_ids, if args.split: gt_ids &= set(json.load(open(split))[bucket])
```

- [ ] **Step 2: Run final benchmark on test split only**

```bash
python scripts/ocr_precision.py --workers 6 --max-failures 200 \
    --split data/splits/card_classifier_v2.json --bucket test \
    --out data/ocr_precision_final
```
Expected: `paired ≈ 720`, summary headline `hand_exact ≥ 99.9%`. If not, iterate Tasks 7-10 with surfaced failure-mode data.

- [ ] **Step 3: Also re-run baseline restricted to the same split** (for apples-to-apples comparison)

```bash
git stash    # temporarily revert v2 changes
python scripts/ocr_precision.py --workers 6 --split data/splits/card_classifier_v2.json \
    --bucket test --out data/ocr_precision_baseline_test
git stash pop
```

- [ ] **Step 4: Diff the two summaries**

Write `scripts/_tmp.py` that loads both `summary.json`s and emits a markdown table (per-field accuracy, headline `hand_exact`, confidence calibration buckets).

- [ ] **Step 5: Commit the harness change**

```bash
git add scripts/ocr_precision.py
git commit -m "feat(ocr-precision): --split/--bucket flag for held-out evaluation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: PR

**Files:**
- All commits already on `feat/card-classifier-v2`.

- [ ] **Step 1: Sanity — make sure new model + augment regression tests all pass**

```bash
python scripts/regression_test.py
```
Expected: all PASS (modulo pre-existing CUDA OOM noise if GPU is busy — those failures must be re-run when GPU is free, not waived).

- [ ] **Step 2: Push branch**

```bash
git push -u origin feat/card-classifier-v2
```

- [ ] **Step 3: Create PR**

```bash
gh pr create --title "feat(ocr): CardCNN v2 + pipeline patches drive hand_exact to ≥99.9%" \
  --body "$(cat <<'EOF'
## Summary

Standalone (pre-Gemini) OCR `hand_exact` rate: <BASELINE>% → <FINAL>% on the held-out test split (10% of the 7,183-hand PokerCraft GT corpus, never seen by training).

Headlines:
- New `CardCNNv2` (deeper + BN + dropout) trained on a fresh 30k-crop corpus extracted from the GT corpus using the existing localizers. Old `data/cards/` (984 crops, suspected label noise) is discarded; PokerCraft images and the legacy Telegram-uploaded Natural8 screenshots are confirmed same-domain so no mix is needed.
- Augmentation tuned to the mined failure pattern (synthetic WIN-sticker overlay, color jitter for d↔h confusion, light geometric).
- Stratified 80/10/10 train/val/test split by hand_id with per-tournament balancing, persisted to `data/splits/card_classifier_v2.json`.
- Temperature-scaled softmax so FAST tier ≥0.95 corresponds to ≤0.1% error empirically.
- `_resolve_hero_board_conflict` now uses top-2 of each CNN head before clearing hero (was: 29% of failures).
- Dealer-button template match anchors position algebraically (fixes "→ SB/BB" bias).

## Results (test split, n≈720)

| field | baseline | v2 | delta |
|---|---|---|---|
| hand_exact | <X>% | <Y>% | +<…> |
| hero_hand | <X>% | <Y>% | +<…> |
| board | <X>% | <Y>% | +<…> |
| hero_position | <X>% | <Y>% | +<…> |
| preflop_types | <X>% | <Y>% | +<…> |

Confidence calibration: max-conf-when-wrong dropped from 0.96 to <Z>; FAST-tier (≥0.95) exact-rate now <W>%.

## Test plan

- [x] `python scripts/regression_test.py` (new tests for split, augment, top-2 fallback, button detector all pass; existing 356 still pass).
- [x] `python scripts/ocr_precision.py --split data/splits/card_classifier_v2.json --bucket test` — see numbers above.
- [x] Smoke-deploy: run `python scripts/e2e_test.py` on one image, confirm OCR-only path returns correct hand.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Drop the PR URL into the conversation.**

---

## Self-Review (run after writing, before executing)

**Spec coverage check:**
- baseline run ✓ (Task 1)
- discard old crops ✓ (Task 3 step 6: `rm -rf data/cards_v2` only — old `data/cards/` left in place but unused; the runtime no longer references it once Task 6 ships v2). Actually need to remove `data/cards` reference in train.py defaults — Task 6 step 1 covers via `--data data/cards_v2`.
- train/test split ✓ (Task 2)
- better architecture if needed ✓ (Task 7)
- ≥99.9% target ✓ (Task 11 step 2 gates this)
- baseline-first ✓ (Task 1 before any model change)
- branch from main ✓ (Task 1 step 1)
- reviewable PR ✓ (Task 12)

**Placeholder scan:**
- Task 6 commit message has `<X>%`, `<Y>%`, `<Z>` — these are *placeholders the executor fills in with real numbers*, not unfilled work items.
- Task 10 has `<KNOWN_BUTTON_HAND>` and `EXPECTED_BTN_SEAT` — executor must select one image and look up its actual button seat from the source HH (`Seat #N is the button`). Picking that fixture is a 2-min lookup, not a research task.
- No other `TODO`/`TBD` etc.

**Type consistency:**
- `extract_one()` returns `{"hero_crops", "hero_labels", "board_crops", "board_labels", "reason"}` — matches the smoke test in Task 3 Step 1.
- `_resolve_hero_board_conflict` signature gains `hero_details=None` kwarg (Task 9 Step 3); callers updated transitively when `table_parser` is updated (Task 9 Step 4). The Task 9 test exercises only the new code path, so existing callers keep working with `hero_details=None` and the legacy clear behavior.
- `CardCNNv2` returns the same `(rank_logits, suit_logits)` tuple as `CardCNN`, so `infer.py` only needs to swap the class — verified by test in Task 5 Step 1.

---

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Best for the long-running ML steps where subagent isolation prevents context pollution.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints. Faster turnaround on small steps but the training runs eat tokens.

Recommend **inline execution** since Tasks 1, 6, 11 are wall-clock-bound rather than think-bound, and the human reviewing wants the final numbers fast.
