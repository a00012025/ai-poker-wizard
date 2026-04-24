# OCR Card Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-coded rank/suit heuristics in `scripts/ocr/table_parser.py` with a small PyTorch CNN trained on labeled crops from the `analysis_snapshots` corpus.

**Architecture:** Phase 0 refactors localization/classification into separate functions (zero behavior change). Phase 1 builds the CNN (extract → train → eval) and deploys in shadow mode next to the legacy classifier. Phase 2 flips the default after threshold recalibration. Phase 3 deletes legacy code and wires OOD monitoring into the weekly report.

**Tech Stack:** Python 3.x · PyTorch 2.10 (already installed transitively via `easyocr`) · OpenCV · pytest · Supabase (asyncpg) · existing snapshot regression suite.

**Spec:** `docs/superpowers/specs/2026-04-24-ocr-card-classifier-design.md`.

---

## File Structure

**New files:**

| Path | Purpose |
|---|---|
| `scripts/ocr/classifier/__init__.py` | Package marker (empty) |
| `scripts/ocr/classifier/model.py` | `CardCNN` — shared backbone + `RankHead` (13) + `SuitHead` (4) |
| `scripts/ocr/classifier/dataset.py` | `CardDataset` — loads from `data/cards/{rank}/{suit}/*.png`, by-hand_id split |
| `scripts/ocr/classifier/extract_crops.py` | CLI: pull snapshots from Supabase → write labeled crops to `data/cards/` |
| `scripts/ocr/classifier/train.py` | CLI: load dataset → train CNN → write checkpoint + metadata JSON |
| `scripts/ocr/classifier/eval.py` | CLI: load checkpoint → compute per-card accuracy + per-class F1, run against 44 regression snapshots |
| `scripts/ocr/classifier/infer.py` | `CardClassifier` singleton — lazy-load, `classify(crop)`, `classify_batch(crops)`, `_warm()` |
| `scripts/ocr/models/card_cnn_v1.pt` | Trained weights checkpoint (~100KB, committed) |
| `scripts/ocr/models/card_cnn_v1.json` | Training metadata (data hash, val accuracy, class map, threshold) |
| `tests/test_card_classifier.py` | Unit tests: model forward, dataset, infer API, missing checkpoint, speed, crop sizes |
| `tests/test_localization.py` | Phase 0 tests — localization returns crops with expected counts |
| `supabase/migrations/20260424_classifier_shadow_log.sql` | `classifier_shadow_log` table |
| `supabase/migrations/20260501_analysis_snapshots_conf.sql` | Add `classifier_conf` column (Phase 3) |
| `data/cards/` | Extracted crop corpus (gitignored; rebuildable via `extract_crops.py`) |

**Modified files:**

| Path | Change |
|---|---|
| `scripts/ocr/table_parser.py` | Phase 0: split `_find_hero_cards` / `_find_board_cards` / `_identify_cards` into `_locate_*` + `_classify_*`. Phase 1: add CNN path behind `OCR_CLASSIFIER` flag. Phase 3: delete `_ocr_card_rank`, `_detect_suit_bgr`, `_hero_hull_norm`, `_suit_template_match`, width-profile/hull/green-channel helpers, and all H-numbered comment branches |
| `src/gemini_session.py` | No change to external API; reads `OCR_CLASSIFIER` env (forwarded to `n8_parser`) |
| `src/main_gemini.py` | Phase 1: call `CardClassifier()._warm()` at startup |
| `scripts/weekly_report.py` | Phase 3: add OOD monitoring line (rolling mean confidence + low-conf count) |
| `.gitignore` | Add `data/cards/` |

**Deleted files (Phase 3):**

| Path | Reason |
|---|---|
| `scripts/ocr/card_matcher.py` | Superseded by CardCNN |
| `scripts/ocr/generate_templates.py` | Template approach abandoned |
| `scripts/ocr/templates/` | Template PNGs no longer used |

---

# Phase 0 — Localization Refactor

Zero-behavior-change split: pull crop extraction out of the classification pipeline so `extract_crops.py` can reuse it. All 44 regression snapshots must still produce identical output at the end of Phase 0.

### Task 0.1: Add tests for new localization API

**Files:**
- Create: `tests/test_localization.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_localization.py
"""Phase 0 — verify localization functions expose crops without classification."""
import cv2
import numpy as np
import pytest
from pathlib import Path

from scripts.ocr.table_parser import _locate_hero_cards, _locate_board_cards


SNAPSHOT = Path(__file__).parent / "snapshots" / "H2491" / "input.jpeg"


@pytest.fixture
def table_region():
    img = cv2.imread(str(SNAPSHOT))
    assert img is not None, f"missing snapshot: {SNAPSHOT}"
    # table region is the top portion before the action panel
    h = img.shape[0]
    return img[0:int(h * 0.55)]


def test_locate_hero_cards_returns_crops(table_region):
    crops = _locate_hero_cards(table_region)
    assert isinstance(crops, list)
    assert len(crops) == 2
    for c in crops:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 3  # BGR
        assert c.shape[0] > 10 and c.shape[1] > 10


def test_locate_board_cards_returns_crops(table_region):
    crops = _locate_board_cards(table_region)
    assert isinstance(crops, list)
    # H2491 is a flop+turn+river — 5 cards; or hero folded — 0
    assert 0 <= len(crops) <= 5
    for c in crops:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_localization.py -v
```
Expected: FAIL — `ImportError: cannot import name '_locate_hero_cards'`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_localization.py
git commit -m "test: phase-0 localization API contract (failing)"
```

---

### Task 0.2: Extract `_locate_hero_cards` from `_find_hero_cards`

**Files:**
- Modify: `scripts/ocr/table_parser.py:304-602`

- [ ] **Step 1: Copy current `_find_hero_cards` blob-detection block into a new helper**

Extract lines 304–405 (the blob-finding, splitting, and tighter-blob logic that runs BEFORE the `for card, suit_card in ...` loop) into:

```python
def _locate_hero_cards(table_region: np.ndarray) -> list[np.ndarray]:
    """Return a list of [card1_crop, card2_crop] BGR ndarrays, or [] if none.

    Pure localization: no rank/suit detection. The returned crops are the
    same ones today's _find_hero_cards would pass to _ocr_card_rank /
    _detect_suit_bgr.
    """
    h, w = table_region.shape[:2]
    hero = table_region[int(h * 0.58):int(h * 0.85),
                        int(w * 0.28):int(w * 0.68)]
    ah, aw = hero.shape[:2]
    if ah < 20 or aw < 20:
        return []

    gray = cv2.cvtColor(hero, cv2.COLOR_BGR2GRAY)

    best_blob = None
    for tv in [200, 190, 180, 170, 160, 150, 140, 130, 120]:
        _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch_ = cv2.boundingRect(c)
            area = cw * ch_
            if (area > 1500 and ch_ > 25 and cw > 60
                    and 1.2 < cw / ch_ < 2.8):
                if best_blob is None or area > best_blob[4]:
                    best_blob = (x, y, cw, ch_, area)
        if best_blob and best_blob[4] > 2500:
            break

    if not best_blob:
        for tv in [160, 150, 140, 130, 120]:
            _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, cw, ch_ = cv2.boundingRect(c)
                area = cw * ch_
                if area > 1500 and ch_ > 25 and 0.7 < cw / ch_ < 2.8:
                    if best_blob is None or area > best_blob[4]:
                        best_blob = (x, y, cw, ch_, area)
            if best_blob and best_blob[4] > 2500:
                break

    if not best_blob:
        return []

    x, y, cw, ch_, _ = best_blob
    split = int(cw * 0.48)
    card1 = hero[y:y + ch_, x:x + split + 3]
    card2 = hero[y:y + ch_, x + split - 3:x + cw]
    return [card1, card2]
```

Place this function immediately ABOVE `_find_hero_cards` in `table_parser.py`.

- [ ] **Step 2: Refactor `_find_hero_cards` to call `_locate_hero_cards`**

Replace the first ~75 lines of `_find_hero_cards` body (the blob-finding/splitting) with:

```python
def _find_hero_cards(table_region: np.ndarray) -> tuple[list[str], float]:
    """(unchanged docstring)"""
    from .ocr_utils import ocr_full_image

    crops = _locate_hero_cards(table_region)
    if not crops:
        return [], 0.0
    card1, card2 = crops
    # (keep the rest of the function body starting from the tighter-blob
    # logic and the for-loop — unchanged)
```

Keep the `suit_card1`/`suit_card2` tighter-blob re-crop logic and everything after it exactly as-is. The only change is that `card1`/`card2` now come from the new helper.

- [ ] **Step 3: Run localization test**

```bash
python -m pytest tests/test_localization.py::test_locate_hero_cards_returns_crops -v
```
Expected: PASS.

- [ ] **Step 4: Run snapshot regression — must be identical**

```bash
python scripts/snapshot_test.py
```
Expected: same pass/fail set as before Phase 0 (no new failures, no newly passing — behavior is byte-identical).

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/table_parser.py tests/test_localization.py
git commit -m "refactor(ocr): extract _locate_hero_cards from _find_hero_cards (no behavior change)"
```

---

### Task 0.3: Extract `_locate_board_cards` from `_find_board_cards`

**Files:**
- Modify: `scripts/ocr/table_parser.py:192-242`

- [ ] **Step 1: Add `_locate_board_cards` above `_find_board_cards`**

```python
def _locate_board_cards(table_region: np.ndarray) -> list[np.ndarray]:
    """Return list of individual board card crops (BGR), or [] if none found.

    Pure localization — no rank/suit identification. Order is left-to-right.
    """
    h, w = table_region.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.55)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    center = table_region[y1:y2, x1:x2]

    rects = _find_individual_card_contours(center)
    if not (rects and len(rects) >= 3):
        row = _find_bright_row(center, thresh_val=160, min_height=30)
        if row is None:
            for tv in [140, 120]:
                row = _find_bright_row(center, thresh_val=tv, min_height=30)
                if row:
                    break
        if row is None:
            return []

        rx, ry, rw, rh = row
        if rw <= rh * 1.2:
            rects = [row]
        else:
            rects = _split_card_row(center, rx, ry, rw, rh)
            if not rects:
                return []
            if len(rects) >= 3:
                rects = [r for r in rects if r[2] / r[3] >= 0.55]
            if len(rects) > 5:
                rects.sort(key=lambda r: r[2], reverse=True)
                rects = rects[:5]
                rects.sort(key=lambda r: r[0])

    crops = []
    for (x, y, cw, ch) in rects:
        crop = center[y:y + ch, x:x + cw]
        if crop.size > 0:
            crops.append(crop)
    return crops
```

- [ ] **Step 2: Refactor `_find_board_cards` to call `_locate_board_cards`**

```python
def _find_board_cards(table_region: np.ndarray) -> list[str]:
    """Find and identify board cards in the center of the table."""
    from .ocr_utils import ocr_full_image

    crops = _locate_board_cards(table_region)
    if not crops:
        return []

    cards = []
    for crop in crops:
        rank, _ = _ocr_card_rank(crop, ocr_full_image)
        suit = _detect_suit_bgr(crop)
        if rank:
            cards.append(f"{rank}{suit}")
    return cards
```

Note: this replaces both the old `_find_board_cards` body AND makes `_identify_cards` dead (it's only called from the old `_find_board_cards`). Leave `_identify_cards` in place for now — Phase 3 deletes it.

- [ ] **Step 3: Run the board-localization test**

```bash
python -m pytest tests/test_localization.py::test_locate_board_cards_returns_crops -v
```
Expected: PASS.

- [ ] **Step 4: Run full snapshot regression**

```bash
python scripts/snapshot_test.py
```
Expected: identical results to pre-Phase-0 baseline.

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/table_parser.py
git commit -m "refactor(ocr): extract _locate_board_cards from _find_board_cards (no behavior change)"
```

---

### Task 0.4: Baseline the 44 regression snapshots

Establish a reproducible "Phase 0 baseline" against which Phase 1 is measured.

**Files:**
- Create: `scripts/_tmp.py` (ad-hoc)

- [ ] **Step 1: Write the baseline capture script**

```python
# scripts/_tmp.py
"""Capture Phase 0 snapshot baseline: per-hand pass/fail + hero/board parse."""
import json, subprocess, sys
out = subprocess.check_output(
    [sys.executable, "scripts/snapshot_test.py"], text=True)
print(out)
```

- [ ] **Step 2: Run and save baseline**

```bash
python scripts/_tmp.py > /tmp/phase0_baseline.txt
grep -E "^(PASS|FAIL|H[0-9]+)" /tmp/phase0_baseline.txt | head -60
```

- [ ] **Step 3: Commit the baseline into the plan directory (not source)**

```bash
mkdir -p docs/superpowers/plans/artifacts
cp /tmp/phase0_baseline.txt docs/superpowers/plans/artifacts/2026-04-24-phase0-baseline.txt
git add docs/superpowers/plans/artifacts/2026-04-24-phase0-baseline.txt
git commit -m "docs: phase-0 snapshot baseline for card-classifier plan"
```

---

# Phase 1 — Build, Train, Shadow-Deploy the Classifier

### Task 1.1: Classifier package skeleton

**Files:**
- Create: `scripts/ocr/classifier/__init__.py`
- Create: `scripts/ocr/classifier/model.py` (stub)
- Create: `scripts/ocr/classifier/dataset.py` (stub)
- Create: `scripts/ocr/classifier/extract_crops.py` (stub)
- Create: `scripts/ocr/classifier/train.py` (stub)
- Create: `scripts/ocr/classifier/eval.py` (stub)
- Create: `scripts/ocr/classifier/infer.py` (stub)
- Create: `scripts/ocr/models/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create the package layout**

```bash
mkdir -p scripts/ocr/classifier scripts/ocr/models
touch scripts/ocr/classifier/__init__.py
touch scripts/ocr/models/.gitkeep
for f in model dataset extract_crops train eval infer; do
  printf '"""Placeholder — see docs/superpowers/plans/2026-04-24-ocr-card-classifier.md"""\n' > scripts/ocr/classifier/$f.py
done
```

- [ ] **Step 2: Add `data/cards/` to .gitignore**

Append to `.gitignore`:

```
# Card classifier training crops (rebuildable via extract_crops.py)
data/cards/
```

- [ ] **Step 3: Commit**

```bash
git add scripts/ocr/classifier scripts/ocr/models .gitignore
git commit -m "chore(ocr-classifier): package skeleton"
```

---

### Task 1.2: `CardCNN` model with TDD

**Files:**
- Create: `tests/test_card_classifier.py`
- Modify: `scripts/ocr/classifier/model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_card_classifier.py
"""Unit tests for scripts/ocr/classifier/*"""
import pytest
import numpy as np
import torch

from scripts.ocr.classifier.model import CardCNN, RANK_CLASSES, SUIT_CLASSES


def test_rank_classes_are_13():
    assert RANK_CLASSES == ["2", "3", "4", "5", "6", "7", "8", "9",
                            "T", "J", "Q", "K", "A"]


def test_suit_classes_are_4():
    assert SUIT_CLASSES == ["c", "d", "h", "s"]


def test_forward_shapes_match_heads():
    net = CardCNN().eval()
    x = torch.randn(3, 3, 48, 64)
    rank_logits, suit_logits = net(x)
    assert rank_logits.shape == (3, 13)
    assert suit_logits.shape == (3, 4)


def test_forward_is_deterministic():
    net = CardCNN().eval()
    x = torch.randn(2, 3, 48, 64)
    with torch.no_grad():
        a = net(x)
        b = net(x)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])
```

- [ ] **Step 2: Run test — expect fail**

```bash
python -m pytest tests/test_card_classifier.py -v
```
Expected: FAIL — can't import `CardCNN`.

- [ ] **Step 3: Implement `CardCNN`**

```python
# scripts/ocr/classifier/model.py
"""CardCNN — shared backbone + RankHead (13) + SuitHead (4)."""
from __future__ import annotations

import torch
import torch.nn as nn

RANK_CLASSES = ["2", "3", "4", "5", "6", "7", "8", "9",
                "T", "J", "Q", "K", "A"]
SUIT_CLASSES = ["c", "d", "h", "s"]

# Input tensor shape: (B, 3, H=48, W=64)


class CardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                           # 24x32
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                           # 12x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                   # 1x1
            nn.Flatten(),
        )
        self.rank_head = nn.Linear(64, len(RANK_CLASSES))
        self.suit_head = nn.Linear(64, len(SUIT_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)
```

- [ ] **Step 4: Test passes**

```bash
python -m pytest tests/test_card_classifier.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_card_classifier.py scripts/ocr/classifier/model.py
git commit -m "feat(ocr-classifier): CardCNN with rank + suit heads"
```

---

### Task 1.3: `CardDataset` — load crops, by-hand_id split

**Files:**
- Modify: `tests/test_card_classifier.py`
- Modify: `scripts/ocr/classifier/dataset.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_card_classifier.py
import tempfile
from pathlib import Path
import cv2

from scripts.ocr.classifier.dataset import CardDataset, split_by_hand_id


def _write_dummy_crop(path: Path, rank_idx: int, suit_idx: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((40, 28, 3), 255, dtype=np.uint8)
    img[rank_idx % 40, suit_idx % 28] = 0  # make each label unique-ish
    cv2.imwrite(str(path), img)


def test_dataset_loads_labels_from_path():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_dummy_crop(root / "A" / "h" / "H100_hero_0.png", 12, 2)
        _write_dummy_crop(root / "2" / "c" / "H100_board_0.png", 0, 0)
        ds = CardDataset(root, augment=False)
        assert len(ds) == 2
        x, r, s = ds[0]
        assert x.shape == (3, 48, 64)
        assert 0 <= r < 13 and 0 <= s < 4


def test_split_by_hand_id_keeps_hands_together():
    samples = [
        ("H1", "hero_0", 0, 0),
        ("H1", "hero_1", 0, 1),
        ("H2", "board_0", 1, 2),
        ("H2", "board_1", 1, 3),
        ("H3", "hero_0", 2, 0),
    ]
    train, val = split_by_hand_id(samples, val_frac=0.4, seed=0)
    train_hands = {s[0] for s in train}
    val_hands = {s[0] for s in val}
    assert train_hands.isdisjoint(val_hands)
    assert train_hands | val_hands == {"H1", "H2", "H3"}
```

- [ ] **Step 2: Run — expect fail**

```bash
python -m pytest tests/test_card_classifier.py -v
```
Expected: FAIL on the two new tests.

- [ ] **Step 3: Implement `CardDataset` + `split_by_hand_id`**

```python
# scripts/ocr/classifier/dataset.py
"""CardDataset — loads labeled crops from data/cards/{rank}/{suit}/*.png."""
from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import RANK_CLASSES, SUIT_CLASSES

INPUT_H, INPUT_W = 48, 64
_RANK_TO_IDX = {r: i for i, r in enumerate(RANK_CLASSES)}
_SUIT_TO_IDX = {s: i for i, s in enumerate(SUIT_CLASSES)}
_FILENAME_RE = re.compile(r"^(?P<hand>[A-Za-z0-9]+)_(?P<src>hero|board)_(?P<slot>\d+)\.png$")


def _letterbox(img: np.ndarray, h: int = INPUT_H, w: int = INPUT_W) -> np.ndarray:
    """Resize + pad to (h, w) preserving aspect ratio, BGR."""
    ih, iw = img.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(ih * scale)), max(1, int(iw * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    top = (h - nh) // 2
    left = (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class CardDataset(Dataset):
    """Every crop under root/{rank}/{suit}/{hand}_{src}_{slot}.png."""

    def __init__(self, root: Path, augment: bool = True):
        self.root = Path(root)
        self.augment = augment
        self.samples: list[tuple[str, str, int, int]] = []  # (hand, slot, r_idx, s_idx)
        self.paths: list[Path] = []
        for rank_dir in self.root.iterdir():
            if not rank_dir.is_dir() or rank_dir.name not in _RANK_TO_IDX:
                continue
            for suit_dir in rank_dir.iterdir():
                if not suit_dir.is_dir() or suit_dir.name not in _SUIT_TO_IDX:
                    continue
                for png in suit_dir.glob("*.png"):
                    m = _FILENAME_RE.match(png.name)
                    if not m:
                        continue
                    self.paths.append(png)
                    self.samples.append((
                        m.group("hand"), f"{m.group('src')}_{m.group('slot')}",
                        _RANK_TO_IDX[rank_dir.name],
                        _SUIT_TO_IDX[suit_dir.name],
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        img = cv2.imread(str(self.paths[idx]))
        assert img is not None, f"unreadable: {self.paths[idx]}"
        img = _letterbox(img)
        if self.augment:
            img = _apply_aug(img)
        _, r_idx, s_idx = self.samples[idx][0], self.samples[idx][2], self.samples[idx][3]
        return _to_tensor(img), r_idx, s_idx


def _apply_aug(img: np.ndarray) -> np.ndarray:
    # translate ±2px
    dx, dy = random.randint(-2, 2), random.randint(-2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REPLICATE)
    # brightness ±10%
    scale = random.uniform(0.9, 1.1)
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    # blur with p=0.2
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.1, 0.5))
    return img


def split_by_hand_id(
    samples: list[tuple], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[tuple], list[tuple]]:
    """Split samples into train/val by hand_id (first element of each tuple)."""
    by_hand: dict[str, list[tuple]] = {}
    for s in samples:
        by_hand.setdefault(s[0], []).append(s)
    hands = sorted(by_hand.keys())
    rng = random.Random(seed)
    rng.shuffle(hands)
    n_val = max(1, int(len(hands) * val_frac))
    val_hands = set(hands[:n_val])
    train, val = [], []
    for h, rows in by_hand.items():
        (val if h in val_hands else train).extend(rows)
    return train, val
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_card_classifier.py -v
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_card_classifier.py scripts/ocr/classifier/dataset.py
git commit -m "feat(ocr-classifier): CardDataset with by-hand_id split"
```

---

### Task 1.4: `extract_crops.py` — build training corpus

**Files:**
- Modify: `scripts/ocr/classifier/extract_crops.py`

- [ ] **Step 1: Implement extraction CLI**

```python
# scripts/ocr/classifier/extract_crops.py
"""Pull snapshots from Supabase, run Phase-0 localization, write labeled
crops to data/cards/{rank}/{suit}/{hand_id}_{source}_{slot}.png.

Label priority: expected_json.hero_hand / expected_json.streets[].board|card
overrides parsed_json equivalents. Rows where crop count != label count are
written to extract_crops.skipped.log and skipped (never invent labels).

Usage:
    python -m scripts.ocr.classifier.extract_crops
    python -m scripts.ocr.classifier.extract_crops --limit 10  # smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocr.table_parser import _locate_hero_cards, _locate_board_cards  # noqa: E402

OUT_ROOT = REPO_ROOT / "data" / "cards"
SKIP_LOG = REPO_ROOT / "data" / "extract_crops.skipped.log"
_CARD_RE = re.compile(r"^([2-9TJQKA])([cdhs])$")


def _parse_hand_labels(parsed: dict, expected: dict | None) -> tuple[list[str], list[str]]:
    """Return (hero_cards, board_cards_in_order). Each card is 'Xy' like '9h'."""
    src = dict(parsed or {})
    if expected:
        for k, v in expected.items():
            if v is not None:
                src[k] = v

    hero_raw = src.get("hero_hand") or ""
    # hero_hand is like "Ac6c" -> ["Ac", "6c"]
    hero: list[str] = []
    if len(hero_raw) >= 2:
        for i in range(0, len(hero_raw) - 1, 2):
            pair = hero_raw[i:i + 2]
            if _CARD_RE.match(pair):
                hero.append(pair)

    board: list[str] = []
    for street in src.get("streets", []) or []:
        b = street.get("board")
        if b:
            for i in range(0, len(b) - 1, 2):
                pair = b[i:i + 2]
                if _CARD_RE.match(pair):
                    board.append(pair)
        c = street.get("card")
        if c and _CARD_RE.match(c):
            board.append(c)
    return hero, board


def _crop_table_region(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h = img.shape[0]
    return img[0:int(h * 0.55)]


def _save_crop(crop: np.ndarray, rank: str, suit: str, hand_id: str,
               source: str, slot: int) -> Path:
    out_dir = OUT_ROOT / rank / suit
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{hand_id}_{source}_{slot}.png"
    cv2.imwrite(str(out_path), crop)
    return out_path


async def main(limit: int | None):
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        q = ("SELECT hand_id, image_data, parsed_json, expected_json "
             "FROM analysis_snapshots "
             "WHERE image_data IS NOT NULL AND parsed_json IS NOT NULL "
             "ORDER BY hand_id")
        rows = await conn.fetch(q + (f" LIMIT {int(limit)}" if limit else ""))
    finally:
        await conn.close()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    SKIP_LOG.write_text("")

    total_saved = 0
    total_skipped = 0

    for r in rows:
        hand_id = r["hand_id"]
        parsed = r["parsed_json"]
        expected = r["expected_json"]
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(expected, str):
            expected = json.loads(expected)

        table_region = _crop_table_region(bytes(r["image_data"]))
        if table_region is None:
            _skip(hand_id, "image_decode_failed")
            total_skipped += 1
            continue

        hero_labels, board_labels = _parse_hand_labels(parsed, expected)
        hero_crops = _locate_hero_cards(table_region)
        board_crops = _locate_board_cards(table_region)

        if len(hero_crops) == len(hero_labels) and hero_labels:
            for i, (crop, lbl) in enumerate(zip(hero_crops, hero_labels)):
                m = _CARD_RE.match(lbl)
                _save_crop(crop, m.group(1), m.group(2), hand_id, "hero", i)
                total_saved += 1
        elif hero_labels:
            _skip(hand_id, f"hero_mismatch crops={len(hero_crops)} labels={len(hero_labels)}")
            total_skipped += 1

        if len(board_crops) == len(board_labels) and board_labels:
            for i, (crop, lbl) in enumerate(zip(board_crops, board_labels)):
                m = _CARD_RE.match(lbl)
                _save_crop(crop, m.group(1), m.group(2), hand_id, "board", i)
                total_saved += 1
        elif board_labels:
            _skip(hand_id, f"board_mismatch crops={len(board_crops)} labels={len(board_labels)}")
            total_skipped += 1

    print(f"saved {total_saved} crops, skipped {total_skipped} sources")
    print(f"skip log: {SKIP_LOG}")


def _skip(hand_id: str, reason: str):
    with SKIP_LOG.open("a") as f:
        f.write(f"{hand_id}\t{reason}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
```

- [ ] **Step 2: Smoke-test on 5 snapshots**

```bash
python -m scripts.ocr.classifier.extract_crops --limit 5
ls data/cards/*/*/ | head -20
```
Expected: ~10–30 PNG files appear under `data/cards/{rank}/{suit}/`.

- [ ] **Step 3: Commit**

```bash
git add scripts/ocr/classifier/extract_crops.py
git commit -m "feat(ocr-classifier): extract_crops.py — build labeled crop corpus"
```

---

### Task 1.5: Run full extraction, inspect distribution

**Files:**
- Create: `scripts/_tmp.py` (ad-hoc inspection)

- [ ] **Step 1: Run full extraction**

```bash
python -m scripts.ocr.classifier.extract_crops
```
Expected: `saved ~1100–1300 crops, skipped <~80 sources`.

- [ ] **Step 2: Inspect per-class distribution**

```python
# scripts/_tmp.py
from pathlib import Path
from collections import Counter

root = Path("data/cards")
by_rank: Counter = Counter()
by_suit: Counter = Counter()
total = 0
for r_dir in root.iterdir():
    if not r_dir.is_dir():
        continue
    for s_dir in r_dir.iterdir():
        if not s_dir.is_dir():
            continue
        n = sum(1 for _ in s_dir.glob("*.png"))
        by_rank[r_dir.name] += n
        by_suit[s_dir.name] += n
        total += n

print(f"total: {total}")
print("\nby rank:")
for r in "23456789TJQKA":
    print(f"  {r}: {by_rank[r]}")
print("\nby suit:")
for s in "cdhs":
    print(f"  {s}: {by_suit[s]}")
```

Run:
```bash
python scripts/_tmp.py
```

**Hard requirement:** every rank class has ≥20 samples and every suit has ≥100. If any class has <20, halt and investigate (likely label-noise issue — review `data/extract_crops.skipped.log` and corresponding `expected_json` entries, patch via `snapshot_test.py --set-expected` where needed, re-extract).

- [ ] **Step 3: Review skip log**

```bash
cat data/extract_crops.skipped.log | wc -l
head -20 data/extract_crops.skipped.log
```

If skip rate > 10% of total snapshots, investigate the most common skip reason before continuing.

---

### Task 1.6: `train.py` — TDD the training loop

**Files:**
- Modify: `scripts/ocr/classifier/train.py`
- Modify: `tests/test_card_classifier.py`

- [ ] **Step 1: Add failing test**

```python
# append to tests/test_card_classifier.py
from scripts.ocr.classifier.train import train


def test_train_smoke(tmp_path):
    # create 13 ranks × 4 suits = 52 tiny crops, 3 per class = 156 images
    for r in "23456789TJQKA":
        for s in "cdhs":
            for i in range(3):
                p = tmp_path / r / s / f"H{i}_hero_0.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                img = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(p), img)

    out_ckpt = tmp_path / "model.pt"
    out_meta = tmp_path / "model.json"
    train(
        data_root=tmp_path,
        out_ckpt=out_ckpt,
        out_meta=out_meta,
        epochs=2,
        batch_size=8,
        seed=0,
    )
    assert out_ckpt.exists()
    assert out_meta.exists()
    meta = json.loads(out_meta.read_text())
    assert "val_accuracy_rank" in meta
    assert "val_accuracy_suit" in meta
    assert "data_hash" in meta
```

- [ ] **Step 2: Run — expect fail**

```bash
python -m pytest tests/test_card_classifier.py::test_train_smoke -v
```
Expected: FAIL — `train` not importable.

- [ ] **Step 3: Implement `train.py`**

```python
# scripts/ocr/classifier/train.py
"""Train CardCNN on data/cards/. Writes checkpoint + metadata JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import CardDataset, split_by_hand_id
from .model import CardCNN, RANK_CLASSES, SUIT_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "cards"
DEFAULT_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
DEFAULT_META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.json"


def _data_hash(samples: list[tuple]) -> str:
    tuples = sorted((s[0], s[1], RANK_CLASSES[s[2]], SUIT_CLASSES[s[3]]) for s in samples)
    h = hashlib.sha256()
    for t in tuples:
        h.update("|".join(t).encode())
    return h.hexdigest()[:16]


def _per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    out = {}
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[str(c)] = f1
    return out


def train(
    data_root: Path,
    out_ckpt: Path,
    out_meta: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    seed: int = 0,
    patience: int = 10,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full = CardDataset(Path(data_root), augment=True)
    assert len(full) >= 52, f"dataset too small: {len(full)}"

    train_samples, val_samples = split_by_hand_id(full.samples, val_frac=val_frac, seed=seed)
    train_idx = [i for i, s in enumerate(full.samples) if s in train_samples]
    val_idx = [i for i, s in enumerate(full.samples) if s in val_samples]

    ds_train = Subset(full, train_idx)
    val_full = CardDataset(Path(data_root), augment=False)
    ds_val = Subset(val_full, val_idx)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    device = "cpu"  # production runs CPU-only; train on CPU for determinism
    net = CardCNN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    stale = 0
    for ep in range(epochs):
        net.train()
        for x, r, s in train_loader:
            x, r, s = x.to(device), r.to(device), s.to(device)
            opt.zero_grad()
            rl, sl = net(x)
            loss = F.cross_entropy(rl, r) + F.cross_entropy(sl, s)
            loss.backward()
            opt.step()

        net.eval()
        val_loss = 0.0
        r_true, r_pred, s_true, s_pred = [], [], [], []
        with torch.no_grad():
            for x, r, s in val_loader:
                x, r, s = x.to(device), r.to(device), s.to(device)
                rl, sl = net(x)
                val_loss += float(F.cross_entropy(rl, r) + F.cross_entropy(sl, s))
                r_true += r.tolist(); r_pred += rl.argmax(1).tolist()
                s_true += s.tolist(); s_pred += sl.argmax(1).tolist()
        print(f"epoch {ep}: val_loss={val_loss:.4f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep}")
                break

    assert best_state is not None, "training produced no best state"
    net.load_state_dict(best_state)
    net.eval()

    r_true, r_pred, s_true, s_pred = [], [], [], []
    with torch.no_grad():
        for x, r, s in val_loader:
            x, r, s = x.to(device), r.to(device), s.to(device)
            rl, sl = net(x)
            r_true += r.tolist(); r_pred += rl.argmax(1).tolist()
            s_true += s.tolist(); s_pred += sl.argmax(1).tolist()
    r_true_a = np.array(r_true); r_pred_a = np.array(r_pred)
    s_true_a = np.array(s_true); s_pred_a = np.array(s_pred)
    rank_acc = float((r_true_a == r_pred_a).mean())
    suit_acc = float((s_true_a == s_pred_a).mean())

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out_ckpt)

    raw_r_f1 = _per_class_f1(r_true_a, r_pred_a, len(RANK_CLASSES))
    raw_s_f1 = _per_class_f1(s_true_a, s_pred_a, len(SUIT_CLASSES))
    meta = {
        "version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_hash": _data_hash(train_samples + val_samples),
        "n_samples_train": len(train_samples),
        "n_samples_val": len(val_samples),
        "val_accuracy_rank": rank_acc,
        "val_accuracy_suit": suit_acc,
        "val_per_class_f1": {
            "rank": {RANK_CLASSES[int(c)]: f for c, f in raw_r_f1.items()},
            "suit": {SUIT_CLASSES[int(c)]: f for c, f in raw_s_f1.items()},
        },
        "class_map": {"rank": RANK_CLASSES, "suit": SUIT_CLASSES},
        "input_size": [48, 64],
        "torch_version": torch.__version__,
        "conf_threshold": 0.85,  # placeholder — recalibrated after Phase 1 shadow
    }

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"ckpt: {out_ckpt}")
    print(f"meta: {out_meta}")
    print(f"val accuracy: rank={rank_acc:.4f}  suit={suit_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        data_root=Path(args.data),
        out_ckpt=Path(args.ckpt),
        out_meta=Path(args.meta),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
```

- [ ] **Step 4: Test passes**

```bash
python -m pytest tests/test_card_classifier.py::test_train_smoke -v
```
Expected: PASS (runs in <10s).

- [ ] **Step 5: Commit**

```bash
git add tests/test_card_classifier.py scripts/ocr/classifier/train.py
git commit -m "feat(ocr-classifier): train.py with early-stopping + metadata"
```

---

### Task 1.7: Run full training, commit checkpoint

- [ ] **Step 1: Train on the full corpus**

```bash
python -m scripts.ocr.classifier.train
```
Expected: ~50 epochs, val rank acc ≥ 0.99, suit acc ≥ 0.99. Checkpoint at `scripts/ocr/models/card_cnn_v1.pt`.

- [ ] **Step 2: Review metadata**

```bash
cat scripts/ocr/models/card_cnn_v1.json | python -m json.tool
```
Check: every rank F1 ≥ 0.95, every suit F1 ≥ 0.95.

If any class < 0.95: inspect `data/cards/{class}/` — usually label noise. Fix labels via `snapshot_test.py --set-expected`, re-run `extract_crops`, re-train.

- [ ] **Step 3: Commit checkpoint**

```bash
git add scripts/ocr/models/card_cnn_v1.pt scripts/ocr/models/card_cnn_v1.json
git commit -m "feat(ocr-classifier): card_cnn_v1 checkpoint + metadata"
```

---

### Task 1.8: `eval.py` — accuracy + 44-snapshot regression

**Files:**
- Modify: `scripts/ocr/classifier/eval.py`

- [ ] **Step 1: Implement `eval.py`**

```python
# scripts/ocr/classifier/eval.py
"""Load card_cnn_v1.pt, compute val accuracy + per-class F1, run against
the 44 regression-flagged snapshots. Exits non-zero if any gate fails."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import cv2
import numpy as np
import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ocr.classifier.dataset import CardDataset, split_by_hand_id, _letterbox, _to_tensor
from ocr.classifier.model import CardCNN, RANK_CLASSES, SUIT_CLASSES
from ocr.table_parser import _locate_hero_cards, _locate_board_cards
from ocr.classifier.extract_crops import _parse_hand_labels, _crop_table_region, _CARD_RE

REQUIRE_VAL_ACCURACY = 0.99
REQUIRE_CLASS_F1 = 0.95

DATA_DIR = REPO_ROOT / "data" / "cards"
CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"
META = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.json"


def _predict_batch(net: CardCNN, crops: list[np.ndarray]) -> list[tuple[str, str, float]]:
    if not crops:
        return []
    x = torch.stack([_to_tensor(_letterbox(c)) for c in crops])
    with torch.no_grad():
        rl, sl = net(x)
        r_probs = torch.softmax(rl, dim=1)
        s_probs = torch.softmax(sl, dim=1)
    results = []
    for i in range(x.shape[0]):
        r_idx = int(r_probs[i].argmax()); r_c = float(r_probs[i, r_idx])
        s_idx = int(s_probs[i].argmax()); s_c = float(s_probs[i, s_idx])
        results.append((RANK_CLASSES[r_idx], SUIT_CLASSES[s_idx], min(r_c, s_c)))
    return results


async def _regression_check(net: CardCNN) -> tuple[int, int, list[str]]:
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        rows = await conn.fetch(
            """SELECT hand_id, image_data, parsed_json, expected_json
               FROM analysis_snapshots
               WHERE is_regression = TRUE AND image_data IS NOT NULL
               ORDER BY hand_id""")
    finally:
        await conn.close()

    passed = 0
    failed_hands: list[str] = []
    for r in rows:
        parsed = r["parsed_json"]; expected = r["expected_json"]
        if isinstance(parsed, str): parsed = json.loads(parsed)
        if isinstance(expected, str): expected = json.loads(expected)
        table_region = _crop_table_region(bytes(r["image_data"]))
        if table_region is None:
            failed_hands.append(f"{r['hand_id']}:decode")
            continue
        hero_labels, board_labels = _parse_hand_labels(parsed, expected)
        hero_crops = _locate_hero_cards(table_region)
        board_crops = _locate_board_cards(table_region)
        hero_preds = _predict_batch(net, hero_crops)
        board_preds = _predict_batch(net, board_crops)
        hero_strs = [f"{p[0]}{p[1]}" for p in hero_preds]
        board_strs = [f"{p[0]}{p[1]}" for p in board_preds]
        if hero_strs == hero_labels and board_strs == board_labels:
            passed += 1
        else:
            failed_hands.append(
                f"{r['hand_id']}: hero want={hero_labels} got={hero_strs} "
                f"board want={board_labels} got={board_strs}")
    return passed, len(rows), failed_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--meta", default=str(META))
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    net = CardCNN()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    print(f"metadata val_accuracy: rank={meta['val_accuracy_rank']:.4f} "
          f"suit={meta['val_accuracy_suit']:.4f}")

    failed_class = []
    for cls, f1 in meta["val_per_class_f1"]["rank"].items():
        if f1 < REQUIRE_CLASS_F1:
            failed_class.append(f"rank {cls} f1={f1:.3f}")
    for cls, f1 in meta["val_per_class_f1"]["suit"].items():
        if f1 < REQUIRE_CLASS_F1:
            failed_class.append(f"suit {cls} f1={f1:.3f}")
    if failed_class:
        print("F1 GATE FAILURE:", failed_class)
        sys.exit(2)
    if meta["val_accuracy_rank"] < REQUIRE_VAL_ACCURACY or \
       meta["val_accuracy_suit"] < REQUIRE_VAL_ACCURACY:
        print("ACCURACY GATE FAILURE")
        sys.exit(2)

    if args.skip_regression:
        print("OK — skipping regression check")
        return

    passed, total, failed = asyncio.run(_regression_check(net))
    print(f"regression: {passed}/{total} hands pass")
    for line in failed:
        print("  FAIL", line)
    if passed < total:
        print(f"REGRESSION GATE FAILURE: {total - passed} hands")
        sys.exit(3)
    print("OK — all gates passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run eval**

```bash
python -m scripts.ocr.classifier.eval
```
Expected: `OK — all gates passed`. If regression fails, inspect failed hands — typically means a label needs fixing (`snapshot_test.py --set-expected`) or augmentation is too weak.

- [ ] **Step 3: Commit**

```bash
git add scripts/ocr/classifier/eval.py
git commit -m "feat(ocr-classifier): eval.py with accuracy + F1 + regression gates"
```

---

### Task 1.9: `infer.py` — CardClassifier singleton

**Files:**
- Modify: `scripts/ocr/classifier/infer.py`
- Modify: `tests/test_card_classifier.py`

- [ ] **Step 1: Add failing tests**

```python
# append to tests/test_card_classifier.py
from scripts.ocr.classifier.infer import CardClassifier


def test_classify_returns_rank_suit_conf():
    clf = CardClassifier()
    crop = np.random.randint(0, 255, (40, 28, 3), dtype=np.uint8)
    rank, suit, conf = clf.classify(crop)
    assert rank in RANK_CLASSES
    assert suit in SUIT_CLASSES
    assert 0.0 <= conf <= 1.0


def test_classify_batch_preserves_order():
    clf = CardClassifier()
    crops = [np.random.randint(0, 255, (40, 28, 3), dtype=np.uint8) for _ in range(5)]
    results = clf.classify_batch(crops)
    assert len(results) == 5
    # deterministic: same input twice → same output, same order
    r2 = clf.classify_batch(crops)
    assert results == r2


def test_missing_checkpoint_returns_none_tuple(tmp_path):
    clf = CardClassifier(ckpt_path=tmp_path / "does_not_exist.pt")
    crop = np.random.randint(0, 255, (40, 28, 3), dtype=np.uint8)
    assert clf.classify(crop) == (None, None, 0.0)


def test_classify_batch_accepts_variable_sizes():
    clf = CardClassifier()
    crops = [
        np.random.randint(0, 255, (30, 20, 3), dtype=np.uint8),
        np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8),
        np.random.randint(0, 255, (100, 75, 3), dtype=np.uint8),
    ]
    results = clf.classify_batch(crops)
    assert len(results) == 3


def test_inference_speed_budget():
    import time
    clf = CardClassifier()
    crops = [np.random.randint(0, 255, (40, 28, 3), dtype=np.uint8) for _ in range(13)]
    clf.classify_batch(crops)  # warm
    t0 = time.perf_counter()
    for _ in range(5):
        clf.classify_batch(crops)
    elapsed = (time.perf_counter() - t0) / 5
    assert elapsed < 0.100, f"batch-of-13 p=5 mean {elapsed*1000:.1f}ms > 100ms"  # soft
```

- [ ] **Step 2: Run — expect fail**

```bash
python -m pytest tests/test_card_classifier.py -v -k infer
```
Expected: FAIL.

- [ ] **Step 3: Implement `infer.py`**

```python
# scripts/ocr/classifier/infer.py
"""CardClassifier — lazy load, batched inference, graceful-missing-checkpoint.

Used by scripts/ocr/table_parser.py when OCR_CLASSIFIER=cnn_v1."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = REPO_ROOT / "scripts" / "ocr" / "models" / "card_cnn_v1.pt"

_log = logging.getLogger(__name__)
_LOGGED_MISSING = False


class CardClassifier:
    """Thread-unsafe lazy singleton. Instantiate once (e.g., at startup)."""

    _instance: "CardClassifier | None" = None

    def __new__(cls, ckpt_path: Path | str | None = None):
        # Allow test-time instantiation with custom ckpt_path by bypassing
        # singleton; production code uses the default (no arg).
        if ckpt_path is not None:
            obj = super().__new__(cls)
            return obj
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, ckpt_path: Path | str | None = None):
        if getattr(self, "_initialized", False):
            return
        self._ckpt_path = Path(ckpt_path) if ckpt_path else _DEFAULT_CKPT
        self._net = None
        self._load_failed = False
        self._initialized = True

    def _ensure_loaded(self) -> bool:
        global _LOGGED_MISSING
        if self._net is not None:
            return True
        if self._load_failed:
            return False
        if not self._ckpt_path.exists():
            if not _LOGGED_MISSING:
                _log.error("CLASSIFIER_CHECKPOINT_UNAVAILABLE: %s", self._ckpt_path)
                _LOGGED_MISSING = True
            self._load_failed = True
            return False
        try:
            import torch
            from .model import CardCNN
            from .dataset import _letterbox, _to_tensor, INPUT_H, INPUT_W
            net = CardCNN()
            net.load_state_dict(torch.load(self._ckpt_path, map_location="cpu"))
            net.eval()
            self._net = net
            self._torch = torch
            self._letterbox = _letterbox
            self._to_tensor = _to_tensor
            return True
        except Exception as e:
            _log.error("CLASSIFIER_LOAD_FAILED: %s", e, exc_info=True)
            self._load_failed = True
            return False

    def _warm(self) -> None:
        """Force checkpoint load + one forward pass so first real call is cheap."""
        if not self._ensure_loaded():
            return
        import numpy as np
        dummy = np.zeros((48, 64, 3), dtype=np.uint8)
        self.classify_batch([dummy])

    def classify(self, crop: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        results = self.classify_batch([crop])
        return results[0] if results else (None, None, 0.0)

    def classify_batch(
        self, crops: list[np.ndarray]
    ) -> list[tuple[Optional[str], Optional[str], float]]:
        if not crops:
            return []
        if not self._ensure_loaded():
            return [(None, None, 0.0)] * len(crops)
        from .model import RANK_CLASSES, SUIT_CLASSES
        x = self._torch.stack([self._to_tensor(self._letterbox(c)) for c in crops])
        with self._torch.no_grad():
            rl, sl = self._net(x)
            r_probs = self._torch.softmax(rl, dim=1)
            s_probs = self._torch.softmax(sl, dim=1)
        out = []
        for i in range(x.shape[0]):
            r_idx = int(r_probs[i].argmax()); r_c = float(r_probs[i, r_idx])
            s_idx = int(s_probs[i].argmax()); s_c = float(s_probs[i, s_idx])
            out.append((RANK_CLASSES[r_idx], SUIT_CLASSES[s_idx], min(r_c, s_c)))
        return out
```

- [ ] **Step 4: All tests pass**

```bash
python -m pytest tests/test_card_classifier.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_card_classifier.py scripts/ocr/classifier/infer.py
git commit -m "feat(ocr-classifier): CardClassifier — lazy singleton, graceful fallback"
```

---

### Task 1.10: Supabase migration for shadow log

**Files:**
- Create: `supabase/migrations/20260424_classifier_shadow_log.sql`

- [ ] **Step 1: Write migration**

```sql
-- supabase/migrations/20260424_classifier_shadow_log.sql
CREATE TABLE IF NOT EXISTS classifier_shadow_log (
    id BIGSERIAL PRIMARY KEY,
    hand_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('hero', 'board')),
    slot INT NOT NULL,
    legacy_card TEXT,
    new_card TEXT,
    new_conf_rank REAL,
    new_conf_suit REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_shadow_log_hand ON classifier_shadow_log(hand_id);
CREATE INDEX IF NOT EXISTS idx_shadow_log_disagreement
    ON classifier_shadow_log(hand_id)
    WHERE legacy_card IS DISTINCT FROM new_card;
```

- [ ] **Step 2: Apply migration**

```bash
supabase db push
```
Expected: migration applied, table created.

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260424_classifier_shadow_log.sql
git commit -m "feat(ocr-classifier): classifier_shadow_log table"
```

---

### Task 1.11: Shadow-mode integration in `table_parser.py`

**Files:**
- Modify: `scripts/ocr/table_parser.py`

- [ ] **Step 1: Add classifier-backed classification next to legacy**

In `table_parser.py`, add at module level (near imports):

```python
import os
import logging

_CLASSIFIER_MODE = os.getenv("OCR_CLASSIFIER", "legacy")  # legacy | cnn_v1 | shadow
_log = logging.getLogger(__name__)


def _classify_crops_cnn(crops: list[np.ndarray]) -> list[tuple[str | None, str | None, float]]:
    from .classifier.infer import CardClassifier
    return CardClassifier().classify_batch(crops)


def _shadow_log(hand_id: str | None, source: str, slot: int,
                legacy_card: str | None, new: tuple[str | None, str | None, float]):
    """Fire-and-forget: push disagreements to classifier_shadow_log.

    Uses the same asyncio.create_task pattern as deviation extraction.
    hand_id may be None (screenshot not yet stored) — in that case we skip.
    """
    if not hand_id:
        return
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    async def _push():
        import asyncpg
        try:
            dsn = os.environ.get("SUPABASE_CONN")
            if not dsn:
                return
            conn = await asyncpg.connect(dsn, statement_cache_size=0)
            try:
                new_card = f"{new[0]}{new[1]}" if new[0] and new[1] else None
                await conn.execute(
                    """INSERT INTO classifier_shadow_log
                       (hand_id, source, slot, legacy_card, new_card,
                        new_conf_rank, new_conf_suit)
                       VALUES ($1,$2,$3,$4,$5,$6,$7)""",
                    hand_id, source, slot, legacy_card, new_card,
                    float(new[2]), float(new[2]))
            finally:
                await conn.close()
        except Exception:
            pass  # shadow logging must never break hand analysis
    loop.create_task(_push())
```

- [ ] **Step 2: Wire into `_find_hero_cards`**

Modify `_find_hero_cards` to optionally classify with CNN and log shadow:

```python
def _find_hero_cards(table_region: np.ndarray,
                     hand_id: str | None = None) -> tuple[list[str], float]:
    from .ocr_utils import ocr_full_image

    crops = _locate_hero_cards(table_region)
    if not crops:
        return [], 0.0

    # CNN pass (if configured)
    cnn_results = None
    if _CLASSIFIER_MODE in ("cnn_v1", "shadow"):
        cnn_results = _classify_crops_cnn(crops)

    # Legacy pass (always run for now; Phase 3 removes it)
    legacy_cards, legacy_conf = _legacy_classify_hero_crops(crops)

    # Shadow logging (disagreement-only is fine — write always, let the query filter)
    if _CLASSIFIER_MODE == "shadow" and cnn_results:
        for i, cnn in enumerate(cnn_results):
            legacy_card = legacy_cards[i] if i < len(legacy_cards) else None
            _shadow_log(hand_id, "hero", i, legacy_card, cnn)

    if _CLASSIFIER_MODE == "cnn_v1" and cnn_results:
        cards = [f"{r}{s}" for r, s, _ in cnn_results if r and s]
        conf = min((c for _, _, c in cnn_results), default=0.0)
        return cards, conf

    return legacy_cards, legacy_conf


def _legacy_classify_hero_crops(crops: list[np.ndarray]) -> tuple[list[str], float]:
    """Move the current (post-localization) body of _find_hero_cards here."""
    # <<< Paste the original code from _find_hero_cards starting at the line
    # `card1, card2 = crops` through the final `return results, card_conf`,
    # except replace the initial `card1, card2 = ...` with:
    #     if len(crops) < 2: return [], 0.0
    #     card1, card2 = crops[0], crops[1]
    # and keep everything else unchanged — including the tighter-blob
    # re-crop logic (which wants `table_region` — pass it in or refactor
    # to accept the already-located hero region).
    # >>>
```

> Note: because the legacy hero classifier needs the surrounding table_region
> to do the tighter-blob re-crop, either (a) pass `table_region` into
> `_legacy_classify_hero_crops` too, or (b) do the tighter-blob re-crop
> inside `_locate_hero_cards` and return a `(card, tight_card)` pair. Option
> (b) is cleaner — make that change in this task.

- [ ] **Step 3: Same wiring for `_find_board_cards`**

```python
def _find_board_cards(table_region: np.ndarray,
                      hand_id: str | None = None) -> list[str]:
    from .ocr_utils import ocr_full_image

    crops = _locate_board_cards(table_region)
    if not crops:
        return []

    cnn_results = None
    if _CLASSIFIER_MODE in ("cnn_v1", "shadow"):
        cnn_results = _classify_crops_cnn(crops)

    legacy_cards = []
    for crop in crops:
        rank, _ = _ocr_card_rank(crop, ocr_full_image)
        suit = _detect_suit_bgr(crop)
        if rank:
            legacy_cards.append(f"{rank}{suit}")

    if _CLASSIFIER_MODE == "shadow" and cnn_results:
        for i, cnn in enumerate(cnn_results):
            legacy_card = legacy_cards[i] if i < len(legacy_cards) else None
            _shadow_log(hand_id, "board", i, legacy_card, cnn)

    if _CLASSIFIER_MODE == "cnn_v1" and cnn_results:
        return [f"{r}{s}" for r, s, _ in cnn_results if r and s]
    return legacy_cards
```

- [ ] **Step 4: Callers may pass `hand_id`**

In `scripts/ocr/n8_parser.py`, find the call sites for `_find_hero_cards` and `_find_board_cards` and thread through `hand_id` if available in scope. If not available (screenshot not yet persisted), pass `None`.

- [ ] **Step 5: Snapshot regression still passes with default (legacy)**

```bash
OCR_CLASSIFIER=legacy python scripts/snapshot_test.py
```
Expected: same as Phase 0 baseline.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/table_parser.py scripts/ocr/n8_parser.py
git commit -m "feat(ocr-classifier): shadow-mode wiring + OCR_CLASSIFIER env flag"
```

---

### Task 1.12: Startup pre-warm in `main_gemini.py`

**Files:**
- Modify: `src/main_gemini.py`

- [ ] **Step 1: Add pre-warm call**

Find the startup block in `src/main_gemini.py` (after env setup, before bot.run). Add:

```python
if os.getenv("OCR_ENABLED", "false").lower() in ("true", "1", "yes"):
    try:
        from scripts.ocr.classifier.infer import CardClassifier
        CardClassifier()._warm()
        _log.info("card classifier pre-warmed")
    except Exception as e:
        _log.warning(f"card classifier pre-warm failed: {e}")
```

- [ ] **Step 2: Verify bot still starts**

```bash
timeout 15 python -c "from src.main_gemini import *" || true
```

- [ ] **Step 3: Commit**

```bash
git add src/main_gemini.py
git commit -m "perf(ocr-classifier): pre-warm CardClassifier at bot startup"
```

---

### Task 1.13: End-to-end shadow test

**Files:**
- Modify: `tests/test_card_classifier.py`

- [ ] **Step 1: Add integration test that exercises shadow mode**

```python
# append to tests/test_card_classifier.py
import subprocess
import os


def test_shadow_mode_returns_legacy_unchanged(monkeypatch, tmp_path):
    """With OCR_CLASSIFIER=shadow, output must equal legacy output."""
    snapshot_dir = Path(__file__).parent / "snapshots" / "H2491"
    if not (snapshot_dir / "input.jpeg").exists():
        pytest.skip("snapshot H2491 not available locally")

    env = os.environ.copy()
    env["OCR_CLASSIFIER"] = "legacy"
    legacy_out = subprocess.check_output(
        ["python", "-c",
         "import cv2, json;"
         "from scripts.ocr.n8_parser import parse_n8_screenshot;"
         f"r = parse_n8_screenshot(open('{snapshot_dir}/input.jpeg','rb').read());"
         "print(json.dumps(r.get('hand') or {}, sort_keys=True))"],
        env=env, text=True)

    env["OCR_CLASSIFIER"] = "shadow"
    shadow_out = subprocess.check_output(
        ["python", "-c",
         "import cv2, json;"
         "from scripts.ocr.n8_parser import parse_n8_screenshot;"
         f"r = parse_n8_screenshot(open('{snapshot_dir}/input.jpeg','rb').read());"
         "print(json.dumps(r.get('hand') or {}, sort_keys=True))"],
        env=env, text=True)

    assert legacy_out == shadow_out, "shadow mode must not change user-facing output"
```

- [ ] **Step 2: Run test**

```bash
python -m pytest tests/test_card_classifier.py::test_shadow_mode_returns_legacy_unchanged -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_card_classifier.py
git commit -m "test(ocr-classifier): shadow mode preserves legacy output"
```

---

### Task 1.14: Deploy Phase 1 to production in shadow mode

- [ ] **Step 1: Set env on prod**

Add to the deployed `.env` (or docker compose env):

```
OCR_CLASSIFIER=shadow
```

- [ ] **Step 2: Deploy**

```bash
bash scripts/deploy.sh
```

- [ ] **Step 3: Smoke-check via logs**

```bash
docker compose logs --tail 50 bot | grep -i "classifier\|OCR"
```
Expected: `card classifier pre-warmed`, no errors.

- [ ] **Step 4: Verify shadow rows appearing**

```bash
psql "$SUPABASE_CONN" -c "SELECT COUNT(*), COUNT(*) FILTER (WHERE legacy_card IS DISTINCT FROM new_card) AS disagreements FROM classifier_shadow_log;"
```

Let it soak for 3–7 days. Minimum: 50+ hands analyzed in shadow before Phase 2.

---

### Task 1.15: (HUMAN GATE) Shadow log analysis + threshold recalibration

After 3–7 days of production shadow traffic:

**Files:**
- Create: `scripts/_tmp.py` (ad-hoc analysis)
- Modify: `scripts/ocr/models/card_cnn_v1.json`

- [ ] **Step 1: Pull the shadow log**

```python
# scripts/_tmp.py
"""Analyze classifier_shadow_log for disagreements and conf distribution."""
import asyncio, json, os
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import asyncpg


async def main():
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
    try:
        rows = await conn.fetch(
            """SELECT hand_id, source, slot, legacy_card, new_card,
                      new_conf_rank, new_conf_suit
               FROM classifier_shadow_log
               WHERE created_at > NOW() - INTERVAL '14 days'""")
    finally:
        await conn.close()

    total = len(rows)
    disagreements = [r for r in rows if r["legacy_card"] != r["new_card"]]
    print(f"total shadow rows: {total}")
    print(f"disagreements: {len(disagreements)} ({len(disagreements)/max(total,1)*100:.1f}%)")

    confs = [min(r["new_conf_rank"] or 0, r["new_conf_suit"] or 0) for r in rows]
    confs.sort()
    if confs:
        print(f"CNN conf quantiles: p5={confs[len(confs)//20]:.3f} "
              f"p50={confs[len(confs)//2]:.3f} p95={confs[len(confs)*19//20]:.3f}")
    # Histogram of per-hand min conf bucketed at 0.05
    print("\ndisagreement examples (first 20):")
    for r in disagreements[:20]:
        print(f"  {r['hand_id']:7s} {r['source']}_{r['slot']}: "
              f"legacy={r['legacy_card']} cnn={r['new_card']} "
              f"conf={min(r['new_conf_rank'] or 0, r['new_conf_suit'] or 0):.3f}")


asyncio.run(main())
```

```bash
python scripts/_tmp.py
```

- [ ] **Step 2: Categorize every disagreement**

For each disagreement, manually label which is correct (look at the snapshot image). Write results into a spreadsheet or jsonl. Three buckets:
- **legacy_right**: add to training set, retrain (Task 1.7 again)
- **cnn_right**: classifier strictly wins — counts as expected improvement
- **both_wrong**: flag snapshot, `snapshot_test.py --set-expected` to fix `expected_json`

If `legacy_right` > 10% of disagreements: retrain before Phase 2.

- [ ] **Step 3: Choose `conf_threshold`**

Rule: pick the threshold where hand-level fast-path rate (hands with min card conf ≥ threshold) equals the legacy fast-path rate (the fraction of hands that today pass `conf > 0.85` on legacy). Record in `scripts/ocr/models/card_cnn_v1.json`:

```json
"conf_threshold": 0.87  // pick based on data
```

Commit the metadata update:

```bash
git add scripts/ocr/models/card_cnn_v1.json
git commit -m "chore(ocr-classifier): recalibrate conf_threshold from shadow data"
```

- [ ] **Step 4: Update `gemini_session.py` to read threshold from metadata**

Currently `gemini_session.py:1549` hard-codes `0.85`. Change to read the CNN-specific threshold when `OCR_CLASSIFIER=cnn_v1`:

```python
# src/gemini_session.py (near line 1549)
ocr_conf_threshold = 0.85
if os.getenv("OCR_CLASSIFIER", "legacy") == "cnn_v1":
    try:
        import json
        meta_path = Path(__file__).resolve().parents[1] / "scripts/ocr/models/card_cnn_v1.json"
        ocr_conf_threshold = json.loads(meta_path.read_text()).get("conf_threshold", 0.85)
    except Exception:
        pass

if ocr_conf > ocr_conf_threshold and ocr_result.get("hand"):
    ...
```

Commit:

```bash
git add src/gemini_session.py
git commit -m "feat(ocr-classifier): read conf threshold from card_cnn_v1.json"
```

---

# Phase 2 — Flip Default to CNN

### Task 2.1: Flip `OCR_CLASSIFIER` default to `cnn_v1`

**Files:**
- Modify: `scripts/ocr/table_parser.py`
- Modify: `.env.example` / production `.env`

- [ ] **Step 1: Change default**

```python
# scripts/ocr/table_parser.py
_CLASSIFIER_MODE = os.getenv("OCR_CLASSIFIER", "cnn_v1")  # was "legacy"
```

Also set `OCR_CLASSIFIER=cnn_v1` in production .env (remove from the example env if it was there, or explicitly document it).

- [ ] **Step 2: Full regression via CNN path**

```bash
OCR_CLASSIFIER=cnn_v1 python scripts/snapshot_test.py
```
Expected: all 44 regression snapshots pass. If any fail, halt — do not flip.

- [ ] **Step 3: Commit**

```bash
git add scripts/ocr/table_parser.py
git commit -m "feat(ocr-classifier): flip default OCR_CLASSIFIER to cnn_v1"
```

- [ ] **Step 4: Deploy**

```bash
bash scripts/deploy.sh
```

- [ ] **Step 5: Monitor for 48 hours**

```bash
docker compose logs --tail 200 bot | grep -iE "classifier|OCR|ERROR" | head -30
psql "$SUPABASE_CONN" -c "SELECT COUNT(*) FROM classifier_shadow_log WHERE legacy_card IS DISTINCT FROM new_card AND created_at > NOW() - INTERVAL '2 days';"
```

If disagreement rate or error rate spikes, roll back via `OCR_CLASSIFIER=legacy` env change — no redeploy needed.

---

### Task 2.2: 2-week soak → green-light Phase 3

No code changes. After 14 days at `OCR_CLASSIFIER=cnn_v1` with no user-reported OCR bugs and no regression failures, proceed to Phase 3.

---

# Phase 3 — Delete Legacy + OOD Monitoring

### Task 3.1: Add `classifier_conf` column for OOD monitoring

**Files:**
- Create: `supabase/migrations/20260501_analysis_snapshots_conf.sql`

- [ ] **Step 1: Migration**

```sql
-- supabase/migrations/20260501_analysis_snapshots_conf.sql
ALTER TABLE analysis_snapshots
  ADD COLUMN IF NOT EXISTS classifier_conf REAL;
CREATE INDEX IF NOT EXISTS idx_snapshots_classifier_conf
  ON analysis_snapshots(classifier_conf) WHERE classifier_conf IS NOT NULL;
```

- [ ] **Step 2: Apply**

```bash
supabase db push
```

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260501_analysis_snapshots_conf.sql
git commit -m "feat(ocr-classifier): add classifier_conf column for OOD monitoring"
```

---

### Task 3.2: Persist hand-level conf on every analysis

**Files:**
- Modify: `src/gemini_session.py` (or wherever `analysis_snapshots` rows are INSERTed)

- [ ] **Step 1: Find the snapshot-insert call**

```bash
grep -n "INSERT INTO analysis_snapshots\|INSERT.*analysis_snapshots" src/ scripts/ -r
```

- [ ] **Step 2: Thread `classifier_conf` through**

In the OCR path, capture `ocr_result["confidence"]` (already computed). In the INSERT call, include the new column. Example sketch:

```python
# where analysis_snapshots row is INSERTed
await conn.execute(
    """INSERT INTO analysis_snapshots
       (hand_id, chat_id, source_type, user_input, image_data, parsed_json,
        expected_json, gto_text, gto_compact, coaching_text, is_regression,
        classifier_conf)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""",
    hand_id, chat_id, source_type, user_input, image_data, parsed_json,
    expected_json, gto_text, gto_compact, coaching_text, is_regression,
    float(ocr_conf) if ocr_conf else None,
)
```

- [ ] **Step 3: Deploy + verify**

```bash
bash scripts/deploy.sh
sleep 300  # wait for some traffic
psql "$SUPABASE_CONN" -c "SELECT COUNT(*), AVG(classifier_conf), MIN(classifier_conf) FROM analysis_snapshots WHERE created_at > NOW() - INTERVAL '1 day';"
```

- [ ] **Step 4: Commit**

```bash
git add src/gemini_session.py
git commit -m "feat(ocr-classifier): persist classifier_conf per hand analysis"
```

---

### Task 3.3: OOD monitoring line in `weekly_report.py`

**Files:**
- Modify: `scripts/weekly_report.py`

- [ ] **Step 1: Add monitoring query**

Find the weekly summary block. Add a one-line stat:

```python
ood_row = await conn.fetchrow(
    """SELECT
         AVG(classifier_conf) AS mean_conf,
         COUNT(*) FILTER (WHERE classifier_conf < 0.5) AS low_conf_count,
         COUNT(*) AS total
       FROM analysis_snapshots
       WHERE classifier_conf IS NOT NULL
         AND created_at > NOW() - INTERVAL '7 days'""")
if ood_row and ood_row["total"]:
    mean = ood_row["mean_conf"] or 0.0
    low_pct = 100.0 * (ood_row["low_conf_count"] or 0) / ood_row["total"]
    body_lines.append(
        f"OCR 分類器信心: mean={mean:.3f} (baseline ≥0.95), "
        f"低信心手牌比例: {low_pct:.1f}% (baseline <2%)")
```

- [ ] **Step 2: Dry-run the report**

```bash
python scripts/weekly_report.py --dry-run 2>&1 | grep "OCR 分類器信心"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/weekly_report.py
git commit -m "feat(ocr-classifier): OOD monitoring line in weekly report"
```

---

### Task 3.4: Delete legacy modules

**Files:**
- Delete: `scripts/ocr/card_matcher.py`
- Delete: `scripts/ocr/generate_templates.py`
- Delete: `scripts/ocr/templates/` (directory)

- [ ] **Step 1: Confirm no internal callers remain**

```bash
grep -rn "card_matcher\|CardMatcher\|generate_templates" scripts/ src/ tests/
```
Expected: only the import inside `table_parser.py` (`_get_matcher`) and its callers in the heuristic helpers we're about to delete.

- [ ] **Step 2: Delete**

```bash
git rm scripts/ocr/card_matcher.py scripts/ocr/generate_templates.py
git rm -r scripts/ocr/templates
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(ocr-classifier): delete card_matcher + templates (superseded by CNN)"
```

---

### Task 3.5: Delete heuristic helpers from `table_parser.py`

**Files:**
- Modify: `scripts/ocr/table_parser.py`

- [ ] **Step 1: Identify delete targets**

Functions to delete entirely (they will be unused once shadow-mode and `_legacy_classify_hero_crops` are gone):
- `_get_matcher`
- `_ocr_card_rank`
- `_detect_suit_bgr`
- `_hero_hull_norm`
- `_suit_template_match`
- `_detect_suit_at`
- `_identify_cards`
- `_legacy_classify_hero_crops`
- all width-profile / hull / green-channel helpers within or near the above

- [ ] **Step 2: Simplify `_find_hero_cards` + `_find_board_cards`**

With no legacy pass, `_find_hero_cards` collapses to:

```python
def _find_hero_cards(table_region: np.ndarray,
                     hand_id: str | None = None) -> tuple[list[str], float]:
    """Find and identify hero's hole cards via CardClassifier."""
    crops = _locate_hero_cards(table_region)
    if not crops:
        return [], 0.0
    results = _classify_crops_cnn(crops)
    cards = [f"{r}{s}" for r, s, _ in results if r and s]
    conf = min((c for _, _, c in results), default=0.0)
    return cards, conf


def _find_board_cards(table_region: np.ndarray,
                      hand_id: str | None = None) -> list[str]:
    crops = _locate_board_cards(table_region)
    if not crops:
        return []
    results = _classify_crops_cnn(crops)
    return [f"{r}{s}" for r, s, _ in results if r and s]
```

Also delete: `_CLASSIFIER_MODE`, `_shadow_log`, and all shadow-mode branches (they've served their purpose).

- [ ] **Step 3: Verify line count + no H-numbered comments**

```bash
wc -l scripts/ocr/table_parser.py
grep -E "H[0-9]{4}" scripts/ocr/table_parser.py || echo "OK: no hand-IDs"
```
Expected: `table_parser.py` ≤ 800 lines; no `H\d{4}` matches.

- [ ] **Step 4: Full regression**

```bash
python scripts/snapshot_test.py
python scripts/regression_test.py
python -m pytest tests/ -v
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/table_parser.py
git commit -m "refactor(ocr): delete heuristic rank/suit helpers (replaced by CardCNN)"
```

---

### Task 3.6: (Optional) Drop shadow log table

After successful Phase 3, the `classifier_shadow_log` table has served its purpose. Either archive and drop, or leave it as historical audit (it's small).

**Recommendation:** leave it. Storage is cheap and it's useful historical data if we ever need to re-investigate a past production decision.

No action required unless table starts consuming non-trivial storage.

---

### Task 3.7: Final deploy

- [ ] **Step 1: Full integration run**

```bash
bash scripts/deploy.sh
```

- [ ] **Step 2: Smoke via real Telegram message**

Upload a known-good screenshot and verify the bot returns a correct analysis.

- [ ] **Step 3: Close out in project memory**

Update the project memory / CLAUDE.md if anything about the OCR pipeline has changed that future-you needs to know (e.g., "OCR rank/suit now uses CardCNN at `scripts/ocr/classifier/`; no more heuristic patches needed — retrain via `python -m scripts.ocr.classifier.train`").

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(ocr-classifier): phase-3 complete — legacy code gone, CNN in prod"
```

---

## Success Criteria Checklist

At the end of Phase 3:

- [ ] All 44 regression snapshots pass Layer 1 parse via `OCR_CLASSIFIER=cnn_v1`
- [ ] `scripts/ocr/table_parser.py` ≤ 800 lines (from 1574)
- [ ] `grep -rE "H[0-9]{4}" scripts/ocr/` returns no matches
- [ ] `scripts/ocr/card_matcher.py` no longer exists
- [ ] `scripts/ocr/templates/` no longer exists
- [ ] Per-card val accuracy ≥ 99%, per-class F1 ≥ 0.95 (all 17 classes)
- [ ] OOD monitoring line prints in weekly report
- [ ] End-to-end hand parse latency on the fast-path ≤ pre-refactor + 10ms
- [ ] Bot startup pre-warms classifier (one log line at boot)
- [ ] Runbook works: next reported misclass gets fixed by `extract_crops` → `train` → `eval` → commit checkpoint (no source changes)
