# N8 Replay Screenshot OCR Pipeline

## Summary

Build an OpenCV + Tesseract OCR pipeline that parses Natural8 hand history replay screenshots into structured hand JSON. When confidence is high, bypass Gemini Vision entirely. When confidence is low, pass OCR-extracted hints alongside the image to Gemini for more accurate parsing. Non-N8 screenshots fall through to pure Gemini (current behavior unchanged).

## Goals

- **Accuracy**: Correctly extract bet sizes, card faces, action sequences (including multiway), player stacks, pot sizes, and position labels
- **Reliability**: Deterministic OCR for numbers and cards eliminates LLM hallucination of these values
- **Performance**: High-confidence parses skip Gemini API call entirely (~2-5s savings)
- **Extensibility**: Config-driven layout definitions allow adding new platforms later

## Architecture

```
截圖 (image_bytes)
  │
  ├─→ [1. n8_parser] ── 主入口
  │     │
  │     ├─→ [2. region_detector] ── 偵測桌面 vs 行動面板分隔線
  │     │     └─→ {table_region, panel_region, divider_y}
  │     │
  │     ├─→ [3. table_parser] ── 桌面區域
  │     │     ├─ Board cards (card_matcher 模板匹配)
  │     │     ├─ Hero cards (card_matcher 模板匹配)
  │     │     ├─ Player stacks (OCR 數字)
  │     │     └─ Table color (HSV → FT 紫色偵測)
  │     │
  │     ├─→ [4. panel_parser] ── 行動面板
  │     │     ├─ 5 欄分割 (Blinds/PreFlop/Flop/Turn/River)
  │     │     ├─ 欄標題 OCR (街名 + pot 數字)
  │     │     └─ Entry 偵測 + OCR (position, action, size)
  │     │
  │     ├─→ [5. card_matcher] ── 牌面辨識
  │     │     └─ Rank + Suit 模板匹配 (17 templates)
  │     │
  │     └─→ [6. ocr_utils] ── OCR 封裝
  │           └─ Tesseract + 預處理 (resize, 銳化, 二值化)
  │
  └─→ [7. 結構組裝 + 信心檢查]
        ├─ confidence > 0.85 → 直接輸出 hand JSON
        ├─ confidence 0.1-0.85 → OCR hints + 原圖 → Gemini → JSON
        └─ confidence 0.0 → 純 Gemini (current behavior)
```

## File Structure

```
scripts/
  ocr/
    __init__.py
    n8_parser.py         — 主入口: parse_n8_screenshot(image_bytes) → {hand, hints, confidence}
    region_detector.py   — 桌面 vs 面板分割 (水平分隔線偵測)
    table_parser.py      — 桌面: board cards, hero cards, stacks, table color
    panel_parser.py      — 面板: 分欄 → entry 偵測 → OCR 行動序列
    card_matcher.py      — 牌面 rank+suit 模板匹配
    ocr_utils.py         — Tesseract 封裝, 預處理 (upscale, sharpen, binarize)
    templates/           — 13 rank + 4 suit 模板圖片
    config/
      n8_default.json    — N8 佈局參數 (顏色閾值, 區域比例等)
```

## Integration with Existing Code

In `src/gemini_session.py`, `_parse_hand_from_image()`:

```python
async def _parse_hand_from_image(self, chat_id, image_bytes, mime_type, user_text="", ...):
    # Step 1: Try OCR pipeline
    from ocr.n8_parser import parse_n8_screenshot
    ocr_result = parse_n8_screenshot(image_bytes)

    if ocr_result["confidence"] > 0.85:
        hand = ocr_result["hand"]
        self._normalize_cards(hand)
        self._fix_folded_players(hand)
        return hand

    # Step 2: Low confidence or non-N8 → Gemini (with optional hints)
    prompt_text = IMAGE_PARSE_PROMPT
    if ocr_result["hints"]:
        prompt_text += f"\n\nOCR 預處理結果（供參考，可能有誤）：\n{json.dumps(ocr_result['hints'], ensure_ascii=False)}"
    if user_text.strip():
        prompt_text += f"\n\n用戶留言：{user_text.strip()}"

    # ... existing Gemini vision call ...
```

## Component Details

### 1. Region Detector (`region_detector.py`)

**Input**: Full screenshot image (numpy array)
**Output**: `{table_region, panel_region, divider_y}` or `None` if not N8

**Method**:
- Convert to grayscale
- Detect horizontal lines using HoughLinesP or row-wise pixel scanning
- The divider line spans >80% of image width, is dark colored, and sits roughly at 45-55% height
- Split image at divider_y into table (above) and panel (below)
- Validation: panel region should have the 5-column header row with "Blinds" / "Pre-Flop" / "Flop" / "Turn" / "River" text

### 2. Table Parser (`table_parser.py`)

**Input**: Table region crop
**Output**: `{board_cards, hero_cards, player_stacks, table_color}`

**Board cards**:
- Cards appear in center of table as large rectangles with white/colored backgrounds
- Use contour detection to find card bounding boxes
- Pass each to card_matcher for rank + suit identification

**Hero cards**:
- Located at bottom center of table region (hero position)
- Same card matching approach

**Player stacks**:
- Each player has "XX.X BB" text near their avatar
- Use OCR with digit whitelist on regions near each player seat
- Player seats are distributed around the table edge at known relative positions

**Table color (FT detection)**:
- Sample HSV values from the felt area (center of table, avoiding cards/chips)
- Purple hue (H: 260-290°) = Final Table
- Green hue (H: 90-150°) = Normal table

### 3. Panel Parser (`panel_parser.py`)

**Input**: Panel region crop
**Output**: `{columns: [{street_name, pot, entries: [{type, position, action, size}]}]}`

**Column splitting**:
- Header row contains 5 labels separated by vertical dividers
- OCR header row to get column boundaries and street names + pot values
- Pot format: "X.X BB" in colored text (cyan/yellow)

**Entry detection within each column**:
- Scan top to bottom
- Detect entry cards by background color:
  - **Hero entry**: Yellow/gold background (HSV H: 20-40, S > 100)
  - **Opponent entry**: White/light gray background (brightness > 200)
- Find bounding box of each entry card

**Entry OCR**:
- **Opponent entries** (white, arrow left):
  - Position badge: small colored rectangle at bottom-left of avatar area
    - Colors: UTG=red, UTG+1=red, LJ(MP)=green, HJ(MP1)=green, CO=yellow, BTN=gray, SB=cyan, BB=dark blue
    - OCR the badge text OR match by color
  - Action text: "Fold", "Check", "Call", "Bet", "Raise" — OCR with whitelist
  - Size: "X.X BB" — OCR with digit whitelist
- **Hero entries** (yellow, arrow right):
  - No position badge — position inferred from Blinds column or preflop action order
  - Action + size: same OCR approach

**Entries to skip**:
- Timebank icons (yellow alarm clock + "10s") — detect by icon shape or "10s" text
- Showdown cards / equity percentages — detect by card images in entry area
- "Wins XX BB" result entries — detect by "Wins" text

**Position alias mapping**:
```
MP → LJ, MP1 → HJ, MP2 → HJ
EP → UTG, EP1 → UTG+1
```

### 4. Card Matcher (`card_matcher.py`)

**Template library**: 13 rank templates + 4 suit templates = 17 images
- Generated by cropping the top-left corner of cards from sample screenshots
- Rank: 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A
- Suit: c(♣), d(♦), h(♥), s(♠)

**Matching method**:
- Crop top-left corner of detected card (rank + suit icon area)
- Split into rank region (top) and suit region (bottom)
- `cv2.matchTemplate()` against each template
- Return best match with confidence score
- Color can assist suit detection: red = hearts/diamonds, black = spades/clubs

### 5. OCR Utils (`ocr_utils.py`)

**Preprocessing pipeline for low-res images**:
1. Upscale 2x if image width < 600px (bicubic interpolation)
2. Convert to grayscale
3. Apply adaptive thresholding (for text on varied backgrounds)
4. Optional sharpening (unsharp mask)

**Tesseract configs**:
- Numbers: `--psm 7 -c tessedit_char_whitelist=0123456789.`
- Actions: `--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`
- Position badges: `--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+`

### 6. Structure Assembler (in `n8_parser.py`)

**Hero position inference**:
1. Check Blinds(Ante) column for hero SB/BB posting (yellow entry)
2. If hero posted SB → hero_position = "SB", if BB → "BB"
3. Otherwise: count preflop action order — hero's yellow entry position in the PreFlop column determines their seat

**Action sequence assembly**:
- Each column's entries from top to bottom = chronological action order for that street
- Map each entry to `{position, action, size}`:
  - Fold → "F"
  - Check → "X"
  - Call X BB → "C" (size for preflop)
  - Bet X BB → "R{size}" (postflop)
  - Raise X BB → "R{size}"
  - All Ante → skip (blinds column only)

**preflop_actions string**:
- Build full position list for table size
- Map each PreFlop entry to position + action code
- Fill unmentioned positions based on action logic

**Confidence scoring**:
- `pot_consistency`: each street's pot should = previous pot + actions invested (±10% tolerance)
- `player_tracking`: action count per street ≤ players still in hand
- `ocr_confidence`: average Tesseract confidence across all OCR calls
- `card_confidence`: minimum card match confidence
- Final score = weighted average: pot 30%, tracking 25%, ocr 25%, cards 20%

## Config (`n8_default.json`)

```json
{
  "platform": "natural8",
  "divider_detection": {
    "method": "horizontal_line",
    "min_width_ratio": 0.8,
    "y_range_ratio": [0.35, 0.55]
  },
  "panel_header": {
    "expected_columns": 5,
    "column_names": ["Blinds (Ante)", "Pre-Flop", "Flop", "Turn", "River"]
  },
  "entry_colors": {
    "hero_hsv": {"h_range": [20, 40], "s_min": 100, "v_min": 150},
    "opponent_brightness_min": 200
  },
  "position_badge_colors_rgb": {
    "UTG": [180, 60, 60],
    "UTG+1": [180, 60, 60],
    "LJ": [60, 140, 60],
    "HJ": [60, 140, 60],
    "CO": [200, 180, 50],
    "BTN": [140, 140, 140],
    "SB": [60, 180, 180],
    "BB": [60, 60, 160]
  },
  "position_aliases": {
    "MP": "LJ", "MP1": "HJ", "MP2": "HJ",
    "EP": "UTG", "EP1": "UTG+1"
  },
  "card_template_size": [40, 56],
  "confidence_weights": {
    "pot_consistency": 0.30,
    "player_tracking": 0.25,
    "ocr_confidence": 0.25,
    "card_confidence": 0.20
  },
  "confidence_threshold": 0.85
}
```

## Docker Changes

Add to `Dockerfile`:
```dockerfile
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
```

Add to `requirements.txt`:
```
opencv-python-headless>=4.8
pytesseract>=0.3.10
```

Estimated image size increase: ~150-200MB (Tesseract + OpenCV).

## Testing Strategy

**Unit tests** (in `scripts/regression_test.py`):
- `test_ocr_region_detection`: detect divider line in sample screenshots
- `test_ocr_card_matching`: match known cards from cropped regions
- `test_ocr_panel_entries`: parse entries from cropped panel columns
- `test_ocr_position_aliases`: MP→LJ, MP1→HJ mapping
- `test_ocr_fold_tracking`: verify folded players excluded from later streets
- `test_ocr_pot_consistency`: validate pot math across streets

**E2E tests**:
- Run OCR pipeline on all 15 sample screenshots in `~/n8_image/`
- Compare OCR output against manually verified ground truth
- Measure confidence scores and accuracy per field

**Template generation**:
- Auto-crop card faces from the 15 sample screenshots to build initial template library
- Validate templates by running card_matcher on all visible cards in the samples

## What Does NOT Change

- `scripts/analyze_hand.py` — receives the same hand JSON regardless of source
- `scripts/gto_api.py` — no changes
- `scripts/gto_formatter.py` — no changes
- `src/telegram_bot/bot.py` — no changes (calls same `send_image_message`)
- Non-N8 screenshots — fall through to Gemini as before
