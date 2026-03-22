"""Table region parser for Natural8 replay screenshots.

Extracts board cards, hero cards, player stacks, and table color
from the upper (table) region of an N8 replay screenshot.
"""

import cv2
import numpy as np

from .card_matcher import CardMatcher

_matcher = None


def _get_matcher() -> CardMatcher:
    global _matcher
    if _matcher is None:
        _matcher = CardMatcher()
    return _matcher


def _detect_table_color(table_region: np.ndarray) -> str:
    """Detect table felt color to distinguish normal vs Final Table.

    Samples HSV from the central felt area (avoiding cards/avatars).

    Returns:
        "green", "purple", "dark", or "unknown"
    """
    h, w = table_region.shape[:2]

    # Sample the felt from a ring around center, avoiding the board cards
    # Use left-center and right-center strips
    samples = []
    for (y1f, y2f, x1f, x2f) in [
        (0.30, 0.50, 0.02, 0.15),  # left of board
        (0.30, 0.50, 0.85, 0.98),  # right of board
        (0.55, 0.70, 0.30, 0.70),  # below board
    ]:
        y1, y2 = int(h * y1f), int(h * y2f)
        x1, x2 = int(w * x1f), int(w * x2f)
        if y2 > y1 and x2 > x1:
            samples.append(table_region[y1:y2, x1:x2])

    if not samples:
        return "unknown"

    # Collect all HSV pixels from all samples (can't vstack different widths)
    all_h, all_s, all_v = [], [], []
    for s in samples:
        hsv_s = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
        all_h.append(hsv_s[:, :, 0].ravel())
        all_s.append(hsv_s[:, :, 1].ravel())
        all_v.append(hsv_s[:, :, 2].ravel())

    # Compute median HSV values
    median_h = np.median(np.concatenate(all_h))
    median_s = np.median(np.concatenate(all_s))
    median_v = np.median(np.concatenate(all_v))

    # Classify based on hue
    if median_v < 40:
        return "dark"
    if median_s < 30:
        # Low saturation = gray/dark theme
        return "dark"

    # Green felt: H roughly 35-85 in OpenCV's 0-180 range
    if 35 <= median_h <= 85 and median_s > 40:
        return "green"

    # Purple felt: H roughly 120-150
    if 120 <= median_h <= 150 and median_s > 40:
        return "purple"

    return "unknown"


def _split_card_row(region: np.ndarray, row_x: int, row_y: int,
                    row_w: int, row_h: int,
                    thresh_val: int = 160) -> list[tuple]:
    """Split a bright card row into individual card rectangles.

    Uses vertical projection profile to find gaps between cards.

    Returns:
        List of (x, y, w, h) in region coordinates, sorted left-to-right.
    """
    card_row = region[row_y:row_y + row_h, row_x:row_x + row_w]
    if card_row.size == 0:
        return []

    gray = cv2.cvtColor(card_row, cv2.COLOR_BGR2GRAY) if len(card_row.shape) == 3 else card_row
    _, row_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Vertical projection: count bright pixels per column
    col_sum = np.sum(row_thresh > 0, axis=0)
    threshold_count = row_h * 0.25

    is_card = col_sum > threshold_count

    # Find transitions (card starts and ends)
    transitions = np.diff(is_card.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if is_card[0]:
        starts = np.concatenate([[0], starts])
    if is_card[-1]:
        ends = np.concatenate([ends, [len(is_card)]])

    if len(starts) != len(ends):
        return []

    rects = []
    for s, e in zip(starts, ends):
        card_w = e - s
        if card_w > 15:  # minimum reasonable card width
            rects.append((row_x + s, row_y, card_w, row_h))

    # Post-process: split segments that are too wide (merged cards)
    # A single card's width is roughly 0.5-0.85 of its height
    if rects:
        median_w = np.median([r[2] for r in rects]) if len(rects) > 1 else row_h * 0.65
        split_rects = []
        for (rx, ry, rw, rh) in rects:
            if rw > median_w * 1.6 and median_w > 20:
                # This segment likely has 2+ merged cards — split evenly
                n_cards = round(rw / median_w)
                n_cards = max(2, min(n_cards, 5))
                sub_w = rw // n_cards
                for k in range(n_cards):
                    sx = rx + k * sub_w
                    sw = sub_w if k < n_cards - 1 else (rw - k * sub_w)
                    split_rects.append((sx, ry, sw, rh))
            else:
                split_rects.append((rx, ry, rw, rh))
        rects = split_rects

    return rects


def _find_bright_row(region: np.ndarray, thresh_val: int = 160,
                     min_height: int = 30) -> tuple | None:
    """Find the largest bright rectangular row in a region.

    Returns (x, y, w, h) or None.
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    if h < min_height or w < 30:
        return None

    return (x, y, w, h)


def _identify_cards(region: np.ndarray, card_rects: list[tuple],
                    min_conf: float = 0.15) -> list[str]:
    """Run CardMatcher on detected card rectangles.

    Returns:
        List of card strings like ["Ks", "9d", "3d"].
    """
    matcher = _get_matcher()
    cards = []
    for (x, y, w, h) in card_rects:
        card_img = region[y:y + h, x:x + w]
        rank, suit, conf = matcher.match(card_img)
        if rank and suit and conf > min_conf:
            cards.append(f"{rank}{suit}")
        else:
            cards.append("??")
    return cards


def _find_board_cards(table_region: np.ndarray) -> list[str]:
    """Find and identify board cards in the center of the table.

    Board cards are displayed as a row of 3-5 cards in the center.
    Uses vertical projection profile to split the card row.
    """
    h, w = table_region.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.55)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    center = table_region[y1:y2, x1:x2]

    # Find the bright card row
    row = _find_bright_row(center, thresh_val=160, min_height=30)
    if row is None:
        return []

    rx, ry, rw, rh = row

    # If it looks like a single card (aspect <= 1.2), try as-is
    if rw <= rh * 1.2:
        cards = _identify_cards(center, [row])
        return [c for c in cards if c != "??"]

    # Split the card row into individual cards
    rects = _split_card_row(center, rx, ry, rw, rh)

    if not rects:
        return []

    # Board should have 3-5 cards
    if len(rects) > 5:
        # Keep the 5 widest
        rects.sort(key=lambda r: r[2], reverse=True)
        rects = rects[:5]
        rects.sort(key=lambda r: r[0])

    cards = _identify_cards(center, rects)
    return [c for c in cards if c != "??"]


def _find_hero_cards(table_region: np.ndarray) -> list[str]:
    """Find and identify hero's hole cards using EasyOCR.

    Uses EasyOCR to read the rank characters on hero's face-up cards
    at bottom center, then detects suit by color sampling.

    Strategy:
    1. Crop hero card area (bottom center of table)
    2. Upscale 2-3x for better OCR on small images
    3. Run EasyOCR to find rank characters (A, K, Q, J, T/10, 2-9)
    4. Filter by position: hero cards are two adjacent characters at similar Y
    5. Detect suit by color (red = h/d, black = s/c)
    """
    from .ocr_utils import ocr_full_image

    h, w = table_region.shape[:2]

    # Hero cards: bottom center. Tight crop to avoid board cards above.
    y1, y2 = int(h * 0.55), int(h * 0.88)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    hero_area = table_region[y1:y2, x1:x2]

    ah, aw = hero_area.shape[:2]
    if ah < 10 or aw < 10:
        return []

    # Upscale for better OCR (hero cards can be small in compressed screenshots)
    scale = max(2, 400 // max(ah, 1))
    scale = min(scale, 4)
    hero_upscaled = cv2.resize(hero_area, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    results = ocr_full_image(hero_upscaled)

    _RANK_CHARS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
    _RANK_MAP = {"10": "T", "1O": "T", "IO": "T", "l0": "T"}

    card_texts = []
    for r in results:
        text = r["text"].strip()
        # Also try uppercased
        text_up = text.upper()

        matched_rank = None
        if text in _RANK_CHARS:
            matched_rank = text
        elif text_up in _RANK_CHARS:
            matched_rank = text_up
        elif text in _RANK_MAP:
            matched_rank = _RANK_MAP[text]
        elif text_up in _RANK_MAP:
            matched_rank = _RANK_MAP[text_up]

        if matched_rank:
            rank = _RANK_MAP.get(matched_rank, matched_rank)
            # Map coordinates back to original hero_area space
            cx_orig = int(r["center_x"] / scale)
            cy_orig = int(r["center_y"] / scale)
            suit = _detect_suit_at(hero_area, cx_orig, cy_orig)
            card_texts.append((rank + suit, cx_orig, cy_orig, r["conf"]))

    if len(card_texts) < 2:
        # Fallback: try with contrast enhancement
        gray = cv2.cvtColor(hero_upscaled, cv2.COLOR_BGR2GRAY)
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results2 = ocr_full_image(enhanced)
        for r in results2:
            text = r["text"].strip()
            text_up = text.upper()
            matched_rank = None
            if text in _RANK_CHARS:
                matched_rank = text
            elif text_up in _RANK_CHARS:
                matched_rank = text_up
            elif text in _RANK_MAP:
                matched_rank = _RANK_MAP[text]
            elif text_up in _RANK_MAP:
                matched_rank = _RANK_MAP[text_up]

            if matched_rank:
                rank = _RANK_MAP.get(matched_rank, matched_rank)
                cx_orig = int(r["center_x"] / scale)
                cy_orig = int(r["center_y"] / scale)
                # Avoid duplicates (same position)
                is_dup = any(abs(cx_orig - c[1]) < 15 and abs(cy_orig - c[2]) < 15
                            for c in card_texts)
                if not is_dup:
                    suit = _detect_suit_at(hero_area, cx_orig, cy_orig)
                    card_texts.append((rank + suit, cx_orig, cy_orig, r["conf"]))

    if len(card_texts) < 2:
        return []

    # Pick the best 2 cards: similar Y (within 30% of card height), adjacent X
    # Sort by confidence descending
    card_texts.sort(key=lambda x: -x[3])

    best_pair = None
    best_score = float("inf")

    for i in range(len(card_texts)):
        for j in range(i + 1, len(card_texts)):
            ci, cj = card_texts[i], card_texts[j]
            y_diff = abs(ci[2] - cj[2])
            x_gap = abs(ci[1] - cj[1])
            # Cards should be at similar Y and close X
            if y_diff < ah * 0.3 and x_gap < aw * 0.5:
                score = y_diff + x_gap * 0.5
                if score < best_score:
                    best_score = score
                    best_pair = (ci, cj)

    if best_pair is None:
        # Just take first 2
        best_pair = (card_texts[0], card_texts[1])

    # Sort left to right
    pair = sorted([best_pair[0], best_pair[1]], key=lambda x: x[1])
    return [pair[0][0], pair[1][0]]


def _detect_suit_at(image: np.ndarray, x: int, y: int, radius: int = 15) -> str:
    """Detect card suit by sampling color near a position.

    Red = hearts(h) or diamonds(d), Black = spades(s) or clubs(c).
    """
    h, w = image.shape[:2]
    # Sample below the rank character (suit symbol is usually below)
    sy = min(y + radius, h - 1)
    y1 = max(0, sy - 5)
    y2 = min(h, sy + 15)
    x1 = max(0, x - 10)
    x2 = min(w, x + 10)

    sample = image[y1:y2, x1:x2]
    if sample.size == 0:
        return "s"  # default

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    # Red detection
    red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    red_ratio = (np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)) / max(sample.size // 3, 1)

    if red_ratio > 0.15:
        # Red suit — h or d. Check shape: heart is rounder, diamond is pointy.
        # For now, use 'h' as default red (more common)
        return "h"
    else:
        return "s"  # black suit default


def _find_player_stacks(table_region: np.ndarray) -> list[float]:
    """Best-effort OCR of player stack values (XX.X BB text).

    Not critical — returns whatever we can find.
    """
    from .ocr_utils import ocr_text, preprocess_for_ocr

    h, w = table_region.shape[:2]
    stacks = []

    # Player stacks appear as colored text (green/yellow) near avatars
    # Convert to HSV and look for green/yellow text regions
    hsv = cv2.cvtColor(table_region, cv2.COLOR_BGR2HSV)

    # Green text: H 35-85, S > 80, V > 100
    green_mask = cv2.inRange(hsv, np.array([35, 80, 100]), np.array([85, 255, 255]))
    # Yellow/gold text: H 15-35, S > 80, V > 100
    yellow_mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([35, 255, 255]))

    combined = cv2.bitwise_or(green_mask, yellow_mask)

    # Dilate to connect text fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 3))
    dilated = cv2.dilate(combined, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        # Stack text boxes are smallish and wide
        if bw < 30 or bh < 8 or bh > 40 or bw > 200:
            continue
        if bw / bh < 1.5:
            continue

        roi = table_region[y:y + bh, x:x + bw]
        prep = preprocess_for_ocr(roi)
        text, conf = ocr_text(prep, whitelist="0123456789.BBLR ", psm=7)
        if not text:
            inv = cv2.bitwise_not(prep)
            text, conf = ocr_text(inv, whitelist="0123456789.BBLR ", psm=7)

        if text and conf > 30:
            # Try to extract numeric BB value
            text_clean = text.upper().replace("BB", "").replace("B", "").strip()
            # Remove non-numeric except dot
            num_str = ""
            for ch in text_clean:
                if ch.isdigit() or ch == ".":
                    num_str += ch
                elif num_str:
                    break
            try:
                val = float(num_str)
                if 0 < val < 1000:
                    stacks.append(val)
            except (ValueError, TypeError):
                pass

    return stacks


def parse_table(table_region: np.ndarray) -> dict:
    """Parse the table region of an N8 replay screenshot.

    Args:
        table_region: BGR image of the table area (above the divider)

    Returns:
        {
            "board_cards": ["Ks", "9d", "3d", ...],
            "hero_cards": ["Ac", "Tc"],
            "player_stacks": [float, ...],
            "table_color": "green"|"purple"|"dark"|"unknown"
        }
    """
    if table_region is None or table_region.size == 0:
        return {
            "board_cards": [],
            "hero_cards": [],
            "player_stacks": [],
            "table_color": "unknown",
        }

    table_color = _detect_table_color(table_region)
    board_cards = _find_board_cards(table_region)
    hero_cards = _find_hero_cards(table_region)
    hero_stack = _find_hero_stack(table_region)

    return {
        "board_cards": board_cards,
        "hero_cards": hero_cards,
        "hero_stack": hero_stack,
        "player_stacks": [],  # skip full table stacks OCR (unreliable)
        "table_color": table_color,
    }


def _find_hero_stack(table_region: np.ndarray) -> float | None:
    """Find hero's stack (BB) from the colored text below hero's avatar.

    Hero is at bottom center. Stack is displayed as colored text like "18 BB".
    """
    from .ocr_utils import ocr_full_image

    h, w = table_region.shape[:2]
    # Hero stack text: bottom center, below the cards
    y1, y2 = int(h * 0.82), min(h, int(h * 0.98))
    x1, x2 = int(w * 0.25), int(w * 0.65)
    stack_area = table_region[y1:y2, x1:x2]

    if stack_area.size == 0:
        return None

    results = ocr_full_image(stack_area)
    for r in results:
        text = r["text"].strip().upper()
        # Look for "XX.X BB" or just a number near "BB"
        import re
        m = re.search(r"(\d+\.?\d*)\s*BB", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
        # Just a number
        m = re.search(r"(\d+\.?\d+)", text)
        if m and r["conf"] > 0.5:
            try:
                val = float(m.group(1))
                if 0.5 < val < 500:
                    return val
            except ValueError:
                continue

    return None
