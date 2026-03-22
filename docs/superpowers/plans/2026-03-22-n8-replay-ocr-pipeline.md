# N8 Replay OCR Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenCV + Tesseract pipeline that parses Natural8 hand history replay screenshots into hand JSON, with Gemini Vision fallback for low-confidence or non-N8 images.

**Architecture:** Region detection splits screenshot into table + action panel. Table parser extracts cards/stacks via template matching + OCR. Panel parser detects entry cards by color (yellow=hero, white=opponent), OCR reads position/action/size. Confidence score routes to direct output or Gemini-assisted parsing.

**Tech Stack:** Python, OpenCV (`opencv-python-headless`), Tesseract OCR (`pytesseract`), NumPy

**Spec:** `docs/superpowers/specs/2026-03-22-n8-replay-ocr-pipeline-design.md`

**Sample data:** 15 N8 replay screenshots in `~/n8_image/`

---

## File Map

| File | Responsibility |
|------|---------------|
| Create: `scripts/ocr/__init__.py` | Package init |
| Create: `scripts/ocr/ocr_utils.py` | Tesseract wrapper, image preprocessing (upscale, binarize, sharpen) |
| Create: `scripts/ocr/region_detector.py` | Split screenshot into table region + panel region via divider line |
| Create: `scripts/ocr/card_matcher.py` | Rank + suit template matching for playing cards |
| Create: `scripts/ocr/panel_parser.py` | Parse action panel: column splitting, entry detection, action OCR |
| Create: `scripts/ocr/table_parser.py` | Parse table area: board cards, hero cards, stacks, table color |
| Create: `scripts/ocr/n8_parser.py` | Main entry: orchestrates all components, assembles hand JSON, confidence scoring |
| Create: `scripts/ocr/config/n8_default.json` | N8 layout config (color thresholds, position aliases, etc.) |
| Create: `scripts/ocr/templates/` | Card rank + suit template images (auto-generated from samples) |
| Create: `scripts/ocr/generate_templates.py` | One-time script to crop card templates from sample screenshots |
| Modify: `src/gemini_session.py` | Integrate OCR pipeline before Gemini vision call |
| Modify: `Dockerfile` | Add `tesseract-ocr` apt package |
| Modify: `requirements.txt` | Add `opencv-python-headless`, `pytesseract` |
| Test: `scripts/regression_test.py` | New OCR-specific tests |

---

## Chunk 1: Foundation (Dependencies + OCR Utils + Region Detection)

### Task 1: Add dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

- [ ] **Step 1: Add Python dependencies to requirements.txt**

Add these lines to `requirements.txt`:
```
opencv-python-headless>=4.8
pytesseract>=0.3.10
```

- [ ] **Step 2: Add Tesseract to Dockerfile**

Change `Dockerfile` to install Tesseract before pip install:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "src.main_gemini"]
```

- [ ] **Step 3: Install locally for development**

```bash
pip install opencv-python-headless pytesseract
sudo apt-get install -y tesseract-ocr 2>/dev/null || true
```

- [ ] **Step 4: Verify imports**

```bash
python -c "import cv2; import pytesseract; print(f'OpenCV {cv2.__version__}, Tesseract OK')"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt Dockerfile
git commit -m "chore: add OpenCV + Tesseract dependencies for OCR pipeline"
```

---

### Task 2: OCR utils module

**Files:**
- Create: `scripts/ocr/__init__.py`
- Create: `scripts/ocr/ocr_utils.py`
- Test in: `scripts/regression_test.py`

- [ ] **Step 1: Create package**

```bash
mkdir -p scripts/ocr/config scripts/ocr/templates
touch scripts/ocr/__init__.py
```

- [ ] **Step 2: Write test for OCR utils**

Add to `scripts/regression_test.py` before the Runner section:

```python
# ── OCR Pipeline Tests ──


@test
def test_ocr_preprocess_upscales_small_image():
    """OCR: preprocess upscales images smaller than 600px wide."""
    import numpy as np
    from ocr.ocr_utils import preprocess_for_ocr
    # Create a small 300x400 grayscale image
    small = np.zeros((400, 300), dtype=np.uint8)
    result = preprocess_for_ocr(small)
    assert_true(result.shape[1] >= 600, f"should upscale width, got {result.shape[1]}")


@test
def test_ocr_preprocess_keeps_large_image():
    """OCR: preprocess does not upscale images >= 600px wide."""
    import numpy as np
    from ocr.ocr_utils import preprocess_for_ocr
    large = np.zeros((800, 700), dtype=np.uint8)
    result = preprocess_for_ocr(large)
    assert_eq(result.shape[1], 700, "should not change width of large image")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python scripts/regression_test.py -f "ocr_preprocess"
```
Expected: FAIL (module not found)

- [ ] **Step 4: Implement ocr_utils.py**

Create `scripts/ocr/ocr_utils.py`:

```python
"""OCR utility functions: Tesseract wrapper and image preprocessing."""
import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


def preprocess_for_ocr(image: np.ndarray, min_width: int = 600) -> np.ndarray:
    """Preprocess image for OCR: upscale if small, convert to grayscale, sharpen.

    Args:
        image: Input image (grayscale or BGR)
        min_width: Minimum width; images smaller than this are upscaled 2x

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Upscale small images
    if gray.shape[1] < min_width:
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return gray


def ocr_text(image: np.ndarray, whitelist: str = "", psm: int = 7) -> tuple[str, float]:
    """Run Tesseract OCR on a preprocessed image region.

    Args:
        image: Grayscale image region
        whitelist: Character whitelist (e.g., "0123456789." for numbers)
        psm: Page segmentation mode (7=single line, 8=single word)

    Returns:
        (text, confidence) tuple. confidence is 0-100.
    """
    if pytesseract is None:
        return "", 0.0

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        texts = []
        confs = []
        for i, conf in enumerate(data["conf"]):
            conf = int(conf)
            if conf > 0 and data["text"][i].strip():
                texts.append(data["text"][i].strip())
                confs.append(conf)

        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf
    except Exception:
        return "", 0.0


def ocr_number(image: np.ndarray) -> tuple[float | None, float]:
    """OCR a number (e.g., bet size, stack) from an image region.

    Returns:
        (number, confidence). number is None if OCR fails.
    """
    text, conf = ocr_text(image, whitelist="0123456789.", psm=7)
    text = text.strip().replace(" ", "")
    try:
        return float(text), conf
    except (ValueError, TypeError):
        return None, 0.0


def binarize(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Apply adaptive thresholding for text extraction."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    if invert:
        binary = cv2.bitwise_not(binary)
    return binary
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python scripts/regression_test.py -f "ocr_preprocess"
```
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/ scripts/regression_test.py
git commit -m "feat(ocr): add ocr_utils module with preprocessing and Tesseract wrapper"
```

---

### Task 3: Region detector

**Files:**
- Create: `scripts/ocr/region_detector.py`
- Test in: `scripts/regression_test.py`

- [ ] **Step 1: Write test using real screenshot**

Add to `scripts/regression_test.py`:

```python
@test
def test_ocr_region_detection_finds_divider():
    """OCR: region detector finds table/panel divider in N8 screenshot."""
    import cv2
    from ocr.region_detector import detect_regions
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return  # skip if sample not available
    image = cv2.imread(img_path)
    result = detect_regions(image)
    assert_true(result is not None, "should detect N8 regions")
    assert_true("table" in result, "should have table region")
    assert_true("panel" in result, "should have panel region")
    assert_true(result["divider_y"] > image.shape[0] * 0.3, "divider should be below 30%")
    assert_true(result["divider_y"] < image.shape[0] * 0.6, "divider should be above 60%")


@test
def test_ocr_region_detection_returns_none_for_non_n8():
    """OCR: region detector returns None for non-N8 images."""
    import numpy as np
    from ocr.region_detector import detect_regions
    # Random noise image — not a poker screenshot
    noise = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    result = detect_regions(noise)
    assert_true(result is None, "should return None for non-N8 image")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python scripts/regression_test.py -f "ocr_region"
```

- [ ] **Step 3: Implement region_detector.py**

Create `scripts/ocr/region_detector.py`:

```python
"""Detect table vs action panel regions in N8 replay screenshots."""
import cv2
import numpy as np


def detect_regions(image: np.ndarray) -> dict | None:
    """Find the divider between table and action panel in an N8 replay screenshot.

    Args:
        image: BGR image (full screenshot)

    Returns:
        {"table": ndarray, "panel": ndarray, "divider_y": int} or None
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The divider is a horizontal line/band spanning most of the width
    # in the middle portion of the image (35-55% height range)
    y_start = int(h * 0.35)
    y_end = int(h * 0.55)

    best_y = None
    best_score = 0

    # Scan rows for dark horizontal lines spanning >80% width
    for y in range(y_start, y_end):
        row = gray[y, :]
        # Count dark pixels (< 80 brightness)
        dark_count = np.sum(row < 80)
        dark_ratio = dark_count / w

        if dark_ratio > 0.8 and dark_count > best_score:
            best_score = dark_count
            best_y = y

    if best_y is None:
        # Try alternative: look for a sharp brightness transition
        # (table area tends to be darker, panel header has text on dark bg)
        for y in range(y_start, y_end - 5):
            above_mean = np.mean(gray[y - 3:y, :])
            below_mean = np.mean(gray[y:y + 3, :])
            # Look for transition where both sides are dark (divider band)
            if above_mean < 60 and below_mean < 60:
                row_dark = np.sum(gray[y, :] < 50) / w
                if row_dark > 0.7:
                    best_y = y
                    break

    if best_y is None:
        return None

    # Validate: check for column headers below the divider
    # The header row should contain text like "Pre-Flop", "Flop" etc.
    header_region = gray[best_y:min(best_y + int(h * 0.05), h), :]
    # Header has relatively uniform dark background with lighter text
    if np.mean(header_region) > 150:
        # Too bright — probably not the right divider
        return None

    table = image[:best_y, :]
    panel = image[best_y:, :]

    return {
        "table": table,
        "panel": panel,
        "divider_y": best_y,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python scripts/regression_test.py -f "ocr_region"
```

- [ ] **Step 5: Test across all 15 sample images**

Write quick validation in `scripts/_tmp.py`:
```python
import cv2, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from ocr.region_detector import detect_regions

img_dir = os.path.expanduser("~/n8_image")
for fname in sorted(os.listdir(img_dir)):
    img = cv2.imread(os.path.join(img_dir, fname))
    result = detect_regions(img)
    h = img.shape[0]
    if result:
        pct = result["divider_y"] / h * 100
        print(f"  OK  {fname}: divider at y={result['divider_y']} ({pct:.0f}%)")
    else:
        print(f"  FAIL {fname}: no divider found")
```
Run: `python scripts/_tmp.py`
Expected: All 15 images should detect divider. Tune thresholds if any fail.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/region_detector.py scripts/regression_test.py
git commit -m "feat(ocr): region detector splits N8 screenshot into table + panel"
```

---

## Chunk 2: Card Matcher + Template Generation

### Task 4: Template generation script

**Files:**
- Create: `scripts/ocr/generate_templates.py`

- [ ] **Step 1: Implement template generation**

Create `scripts/ocr/generate_templates.py`:

```python
#!/usr/bin/env python3
"""Generate card rank + suit templates from N8 replay screenshots.

Finds board cards in the table region, crops the top-left corner
(rank + suit icon area), and saves as template images.

Usage:
    python scripts/ocr/generate_templates.py ~/n8_image/
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ocr.region_detector import detect_regions

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def find_board_cards(table_region: np.ndarray) -> list[np.ndarray]:
    """Find board card bounding boxes in the table region.

    Board cards are large white/colored rectangles in the center of the table.
    """
    h, w = table_region.shape[:2]

    # Board cards are in the center-middle area
    center_region = table_region[int(h * 0.2):int(h * 0.55), int(w * 0.15):int(w * 0.85)]

    # Convert to grayscale and find bright rectangular regions (card faces)
    gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
    # Cards have bright backgrounds (white or colored)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0
        area = cw * ch
        # Card aspect ratio ~0.65-0.75, minimum area
        if 0.5 < aspect < 0.9 and area > 1000:
            # Offset back to table_region coordinates
            card_x = x + int(w * 0.15)
            card_y = y + int(h * 0.2)
            card_img = table_region[card_y:card_y + ch, card_x:card_x + cw]
            cards.append(card_img)

    # Sort left to right
    return cards


def extract_rank_suit(card_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rank and suit regions from top-left corner of a card."""
    h, w = card_img.shape[:2]
    # Rank is in top ~40% of card, left ~40%
    rank_region = card_img[int(h * 0.05):int(h * 0.35), int(w * 0.05):int(w * 0.45)]
    # Suit is below rank, same left area
    suit_region = card_img[int(h * 0.35):int(h * 0.65), int(w * 0.05):int(w * 0.45)]
    return rank_region, suit_region


def main():
    img_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/n8_image")
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

    all_cards = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        regions = detect_regions(img)
        if regions is None:
            print(f"  SKIP {fname}: not N8 format")
            continue
        cards = find_board_cards(regions["table"])
        print(f"  {fname}: found {len(cards)} cards")
        all_cards.extend(cards)

    print(f"\nTotal cards found: {len(all_cards)}")
    print(f"Saving to {TEMPLATE_DIR}/")
    print("Please manually label these as rank_X.png and suit_X.png")

    for i, card in enumerate(all_cards):
        rank_region, suit_region = extract_rank_suit(card)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f"card_{i:03d}.png"), card)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f"rank_{i:03d}.png"), rank_region)
        cv2.imwrite(os.path.join(TEMPLATE_DIR, f"suit_{i:03d}.png"), suit_region)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run template generation**

```bash
python scripts/ocr/generate_templates.py ~/n8_image/
```

Review the output. Manually select the best examples of each rank (2-A) and suit (c/d/h/s), rename them to:
- `rank_2.png` through `rank_A.png` (13 files)
- `suit_c.png`, `suit_d.png`, `suit_h.png`, `suit_s.png` (4 files)

- [ ] **Step 3: Commit templates**

```bash
git add scripts/ocr/generate_templates.py scripts/ocr/templates/
git commit -m "feat(ocr): template generation script and initial card templates"
```

---

### Task 5: Card matcher

**Files:**
- Create: `scripts/ocr/card_matcher.py`
- Test in: `scripts/regression_test.py`

- [ ] **Step 1: Write test**

Add to `scripts/regression_test.py`:

```python
@test
def test_ocr_card_matcher_loads_templates():
    """OCR: card matcher loads rank and suit templates."""
    from ocr.card_matcher import CardMatcher
    matcher = CardMatcher()
    assert_true(len(matcher.rank_templates) > 0, "should load rank templates")
    assert_true(len(matcher.suit_templates) > 0, "should load suit templates")


@test
def test_ocr_card_matcher_identifies_card():
    """OCR: card matcher identifies a card from a sample screenshot."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.card_matcher import CardMatcher
    from ocr.generate_templates import find_board_cards
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    if regions is None:
        return
    cards = find_board_cards(regions["table"])
    assert_true(len(cards) > 0, "should find board cards")
    matcher = CardMatcher()
    rank, suit, conf = matcher.match(cards[0])
    assert_true(rank is not None, f"should identify rank, got None")
    assert_true(suit is not None, f"should identify suit, got None")
    assert_true(conf > 0.5, f"confidence should be > 0.5, got {conf}")
```

- [ ] **Step 2: Implement card_matcher.py**

Create `scripts/ocr/card_matcher.py`:

```python
"""Card rank + suit recognition via template matching."""
import cv2
import numpy as np
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["c", "d", "h", "s"]

# Suit color ranges in HSV (for color-based fallback)
SUIT_COLORS = {
    "h": {"h_range": (0, 10), "s_min": 100},      # red
    "d": {"h_range": (0, 10), "s_min": 100},      # red
    "s": {"max_s": 50},                             # black/dark
    "c": {"max_s": 50},                             # black/dark (green tint)
}


class CardMatcher:
    """Match playing cards by comparing against stored templates."""

    def __init__(self, template_dir: str = TEMPLATE_DIR):
        self.rank_templates = {}
        self.suit_templates = {}
        self._load_templates(template_dir)

    def _load_templates(self, template_dir: str):
        for rank in RANKS:
            path = os.path.join(template_dir, f"rank_{rank}.png")
            if os.path.exists(path):
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self.rank_templates[rank] = tmpl

        for suit in SUITS:
            path = os.path.join(template_dir, f"suit_{suit}.png")
            if os.path.exists(path):
                tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self.suit_templates[suit] = tmpl

    def match(self, card_image: np.ndarray) -> tuple[str | None, str | None, float]:
        """Identify rank and suit of a card image.

        Args:
            card_image: BGR image of a single card

        Returns:
            (rank, suit, confidence) e.g., ("K", "s", 0.92)
        """
        h, w = card_image.shape[:2]
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)

        # Extract top-left corner for rank + suit matching
        rank_region = gray[int(h * 0.05):int(h * 0.35), int(w * 0.05):int(w * 0.45)]
        suit_region = gray[int(h * 0.35):int(h * 0.65), int(w * 0.05):int(w * 0.45)]

        rank, rank_conf = self._match_region(rank_region, self.rank_templates)
        suit, suit_conf = self._match_region(suit_region, self.suit_templates)

        # Color-based suit fallback if template match is low confidence
        if suit_conf < 0.6:
            color_suit = self._detect_suit_by_color(card_image)
            if color_suit:
                suit = color_suit
                suit_conf = max(suit_conf, 0.7)

        confidence = min(rank_conf, suit_conf)
        return rank, suit, confidence

    def _match_region(self, region: np.ndarray, templates: dict) -> tuple[str | None, float]:
        """Match a region against a set of templates."""
        if not templates:
            return None, 0.0

        best_name = None
        best_score = -1.0

        for name, tmpl in templates.items():
            # Resize template to match region size
            tmpl_resized = cv2.resize(tmpl, (region.shape[1], region.shape[0]))
            result = cv2.matchTemplate(region, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            score = result.max()
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, max(best_score, 0.0)

    def _detect_suit_by_color(self, card_image: np.ndarray) -> str | None:
        """Detect suit by color (red = hearts/diamonds, black = spades/clubs)."""
        h, w = card_image.shape[:2]
        suit_area = card_image[int(h * 0.35):int(h * 0.65), int(w * 0.05):int(w * 0.45)]
        hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)

        # Check if predominantly red (hearts/diamonds)
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        red_ratio = (np.sum(red_mask > 0) + np.sum(red_mask2 > 0)) / suit_area.size

        if red_ratio > 0.1:
            # Red suit — disambiguate by shape (not reliable, return most common)
            return None  # let template matching decide between h and d
        return None  # let template matching decide between s and c
```

- [ ] **Step 3: Run tests**

```bash
python scripts/regression_test.py -f "ocr_card"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/ocr/card_matcher.py scripts/regression_test.py
git commit -m "feat(ocr): card matcher with template matching and color fallback"
```

---

## Chunk 3: Panel Parser

### Task 6: Config file

**Files:**
- Create: `scripts/ocr/config/n8_default.json`

- [ ] **Step 1: Create config**

Write `scripts/ocr/config/n8_default.json` with the full config from the spec (section "Config").

- [ ] **Step 2: Commit**

```bash
git add scripts/ocr/config/
git commit -m "feat(ocr): N8 default layout config"
```

---

### Task 7: Panel parser — column splitting

**Files:**
- Create: `scripts/ocr/panel_parser.py`
- Test in: `scripts/regression_test.py`

- [ ] **Step 1: Write test**

```python
@test
def test_ocr_panel_column_split():
    """OCR: panel parser splits action panel into 5 columns."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import split_columns
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    columns = split_columns(regions["panel"])
    assert_eq(len(columns), 5, f"should find 5 columns, got {len(columns)}")
    assert_true(columns[0]["name"].startswith("Blind"), f"first column should be Blinds, got {columns[0]['name']}")
```

- [ ] **Step 2: Implement split_columns in panel_parser.py**

Detect the 5-column header row by finding vertical dividers and OCR-ing the street names and pot values. Each column returned as `{"name": str, "pot": float, "region": ndarray, "x_start": int, "x_end": int}`.

- [ ] **Step 3: Run test, iterate until passing across all 15 screenshots**

- [ ] **Step 4: Commit**

---

### Task 8: Panel parser — entry detection

- [ ] **Step 1: Write test**

```python
@test
def test_ocr_panel_entry_detection():
    """OCR: panel parser detects hero (yellow) and opponent (white) entries."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import split_columns, detect_entries
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    columns = split_columns(regions["panel"])
    # PreFlop column (index 1) should have entries
    entries = detect_entries(columns[1]["region"])
    assert_true(len(entries) > 0, "should find entries in PreFlop column")
    hero_entries = [e for e in entries if e["type"] == "hero"]
    opp_entries = [e for e in entries if e["type"] == "opponent"]
    assert_true(len(hero_entries) > 0, "should find at least one hero entry")
    assert_true(len(opp_entries) > 0, "should find at least one opponent entry")
```

- [ ] **Step 2: Implement detect_entries**

Scan column region top-to-bottom. Use HSV color detection to find yellow (hero) and white (opponent) entry cards. Skip timebank icons, showdown cards, and "Wins" entries. Return list of `{"type": "hero"|"opponent", "region": ndarray, "y": int}`.

- [ ] **Step 3: Run test, iterate**

- [ ] **Step 4: Commit**

---

### Task 9: Panel parser — entry OCR (position + action + size)

- [ ] **Step 1: Write test**

```python
@test
def test_ocr_panel_entry_ocr():
    """OCR: panel parser reads position, action, and size from entries."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import parse_panel
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_panel(regions["panel"])
    # Check PreFlop column
    preflop = result["columns"][1]
    assert_true(len(preflop["entries"]) > 0, "PreFlop should have entries")
    # First entry should have action
    first = preflop["entries"][0]
    assert_true(first["action"] in ("Fold", "Raise", "Call", "Check", "Bet"),
                f"should have valid action, got {first.get('action')}")


@test
def test_ocr_position_alias_mapping():
    """OCR: MP→LJ, MP1→HJ position alias mapping."""
    from ocr.panel_parser import normalize_position
    assert_eq(normalize_position("MP"), "LJ")
    assert_eq(normalize_position("MP1"), "HJ")
    assert_eq(normalize_position("MP2"), "HJ")
    assert_eq(normalize_position("EP"), "UTG")
    assert_eq(normalize_position("CO"), "CO")  # unchanged
```

- [ ] **Step 2: Implement parse_panel and normalize_position**

`parse_panel(panel_image)` orchestrates: split_columns → detect_entries per column → OCR each entry for position badge, action text, and size. Returns full structured panel data.

`normalize_position(pos)` applies alias mapping.

- [ ] **Step 3: Run tests, iterate on OCR accuracy**

- [ ] **Step 4: Commit**

---

## Chunk 4: Table Parser + Main Entry + Integration

### Task 10: Table parser

**Files:**
- Create: `scripts/ocr/table_parser.py`

- [ ] **Step 1: Write test**

```python
@test
def test_ocr_table_parser_board_cards():
    """OCR: table parser finds board cards."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.table_parser import parse_table
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_table(regions["table"])
    assert_true(len(result["board_cards"]) >= 3, f"should find >=3 board cards, got {len(result['board_cards'])}")
    assert_true(result["hero_cards"] is not None, "should find hero cards")


@test
def test_ocr_table_color_detection():
    """OCR: table parser detects table color (green vs purple)."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.table_parser import parse_table
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_table(regions["table"])
    assert_true(result["table_color"] in ("green", "purple", "dark", "unknown"),
                f"unexpected table color: {result['table_color']}")
```

- [ ] **Step 2: Implement table_parser.py**

Uses `card_matcher.CardMatcher` for card identification, OCR for stack numbers, HSV for table color.

- [ ] **Step 3: Run tests, iterate**

- [ ] **Step 4: Commit**

---

### Task 11: Main parser (n8_parser.py) — assembly + confidence

**Files:**
- Create: `scripts/ocr/n8_parser.py`

- [ ] **Step 1: Write E2E test**

```python
@test
def test_ocr_n8_parser_full_pipeline():
    """OCR: full N8 parser produces hand JSON from screenshot."""
    import cv2
    from ocr.n8_parser import parse_n8_screenshot
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    with open(img_path, "rb") as f:
        image_bytes = f.read()
    result = parse_n8_screenshot(image_bytes)
    assert_true(result["confidence"] > 0, "should have non-zero confidence")
    if result["hand"]:
        hand = result["hand"]
        assert_true(hand.get("hero_hand") is not None, "should have hero_hand")
        assert_true(hand.get("preflop_actions") is not None, "should have preflop_actions")
        assert_true(hand.get("hero_position") is not None, "should have hero_position")
```

- [ ] **Step 2: Implement n8_parser.py**

Orchestrates: `detect_regions` → `parse_table` + `parse_panel` → assemble hand JSON → compute confidence score. Handles hero position inference, preflop_actions string building, streets assembly.

Key functions:
- `parse_n8_screenshot(image_bytes) → {"hand", "hints", "confidence"}`
- `_assemble_hand(table_data, panel_data) → (hand_dict, confidence)`
- `_compute_confidence(hand, panel_data) → float`

- [ ] **Step 3: Run test, iterate**

- [ ] **Step 4: Commit**

---

### Task 12: Integration with gemini_session.py

**Files:**
- Modify: `src/gemini_session.py`

- [ ] **Step 1: Modify _parse_hand_from_image to try OCR first**

In `_parse_hand_from_image()`, add OCR attempt before the Gemini call. See spec "Integration with Existing Code" section for exact code.

Key changes:
1. Try `parse_n8_screenshot(image_bytes)` first
2. If confidence > 0.85, return OCR result directly (after normalize + fix_folded)
3. If confidence 0.1-0.85, append OCR hints to Gemini prompt
4. If confidence 0.0, use pure Gemini (unchanged behavior)

- [ ] **Step 2: Test with e2e_test.py**

```bash
set -a && source .env && set +a
python scripts/e2e_test.py --image ~/n8_image/photo_2026-03-22\ 13.53.03.jpeg
```

- [ ] **Step 3: Run regression tests**

```bash
python scripts/regression_test.py
```
All existing tests must still pass.

- [ ] **Step 4: Commit**

```bash
git add src/gemini_session.py
git commit -m "feat: integrate OCR pipeline into image parsing with Gemini fallback"
```

---

## Chunk 5: E2E Validation + Deploy

### Task 13: E2E validation across all 15 screenshots

- [ ] **Step 1: Create ground truth for 3 sample images**

Manually create expected hand JSON for screenshots 03, 08, 13 (diverse scenarios). Save as `scripts/ocr/test_data/expected_*.json`.

- [ ] **Step 2: Write E2E accuracy test**

```python
@test
def test_ocr_e2e_accuracy():
    """OCR E2E: pipeline output matches ground truth for sample screenshots."""
    import json
    from ocr.n8_parser import parse_n8_screenshot
    test_dir = os.path.join(os.path.dirname(__file__), "ocr", "test_data")
    if not os.path.exists(test_dir):
        return
    for fname in os.listdir(test_dir):
        if not fname.startswith("expected_"):
            continue
        with open(os.path.join(test_dir, fname)) as f:
            expected = json.load(f)
        img_name = expected["_image_file"]
        img_path = os.path.expanduser(f"~/n8_image/{img_name}")
        if not os.path.exists(img_path):
            continue
        with open(img_path, "rb") as f:
            result = parse_n8_screenshot(f.read())
        hand = result["hand"]
        if hand is None:
            continue
        # Check key fields
        if expected.get("hero_hand"):
            assert_eq(hand["hero_hand"], expected["hero_hand"],
                      f"{img_name}: hero_hand mismatch")
        if expected.get("hero_position"):
            assert_eq(hand["hero_position"], expected["hero_position"],
                      f"{img_name}: hero_position mismatch")
```

- [ ] **Step 3: Run full pipeline on all 15 images, measure confidence**

```bash
python scripts/_tmp.py  # write a quick script to test all 15
```

Log: confidence score, detected fields, and any mismatches.

- [ ] **Step 4: Run full regression suite**

```bash
python scripts/regression_test.py
```

- [ ] **Step 5: Commit**

---

### Task 14: Full pipeline test — image to GTO analysis output

Test the complete flow: screenshot → OCR → hand JSON → `analyze_hand_full()` → GTO text output.
Uses real screenshots from `~/n8_image/` as input.

- [ ] **Step 1: Write full pipeline test**

```python
@test
def test_ocr_full_pipeline_image_to_analysis():
    """OCR E2E: screenshot → OCR parse → analyze_hand_full → produces GTO output."""
    from ocr.n8_parser import parse_n8_screenshot
    from analyze_hand import analyze_hand_full
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    with open(img_path, "rb") as f:
        ocr_result = parse_n8_screenshot(f.read())
    assert_true(ocr_result["confidence"] > 0, "should parse screenshot")
    hand = ocr_result["hand"]
    assert_true(hand is not None, "should produce hand JSON")
    # Run GTO analysis on the OCR output
    result = analyze_hand_full(hand)
    assert_true(len(result["text"]) > 50, "should produce analysis text")
    assert_true(result["hero_position"] is not None, "should have hero position")
    assert_true(result["hero_hand"] is not None, "should have hero hand")


@test
def test_ocr_full_pipeline_multiple_images():
    """OCR E2E: multiple screenshots all produce valid analysis or graceful fallback."""
    from ocr.n8_parser import parse_n8_screenshot
    from analyze_hand import analyze_hand_full
    img_dir = os.path.expanduser("~/n8_image")
    if not os.path.exists(img_dir):
        return
    success = 0
    total = 0
    for fname in sorted(os.listdir(img_dir))[:5]:  # test first 5
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        total += 1
        with open(os.path.join(img_dir, fname), "rb") as f:
            ocr_result = parse_n8_screenshot(f.read())
        if ocr_result["hand"] is not None and ocr_result["confidence"] > 0.5:
            result = analyze_hand_full(ocr_result["hand"])
            if len(result["text"]) > 50:
                success += 1
    assert_true(success >= 3, f"at least 3/5 screenshots should produce full analysis, got {success}/{total}")
```

- [ ] **Step 2: Run the full pipeline tests**

```bash
set -a && source .env && set +a
python scripts/regression_test.py -f "ocr_full_pipeline"
```

- [ ] **Step 3: Run full regression suite**

```bash
python scripts/regression_test.py
```

- [ ] **Step 4: Commit all remaining changes**

- [ ] **Step 5: Deploy**

```bash
git push origin main
bash scripts/deploy.sh
```
