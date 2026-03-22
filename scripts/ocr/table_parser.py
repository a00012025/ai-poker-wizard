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
    """Find and identify hero's hole cards at bottom center.

    Hero cards are displayed face-up near the hero's avatar (bottom center).
    They appear as a pair of bright card rectangles, sometimes merged into one blob.
    """
    h, w = table_region.shape[:2]

    # Hero is at bottom center — scan the area
    y1, y2 = int(h * 0.50), int(h * 0.95)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    bottom = table_region[y1:y2, x1:x2]

    # Find bright regions that could be cards
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY) if len(bottom.shape) == 3 else bottom

    # Try multiple thresholds — hero cards may have colored backgrounds
    for thresh_val in [140, 150, 160, 170]:
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        # Light dilation to connect card parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 400 or bh < 25 or bw < 15:
                continue
            aspect = bw / bh

            # Individual card: roughly 0.4-0.85 aspect (taller than wide)
            if 0.35 <= aspect <= 0.85:
                candidates.append((x, y, bw, bh))
            # Card pair merged: aspect 0.85-2.5, split them
            elif 0.85 < aspect <= 2.5 and bh > 30 and area > 1500:
                rects = _split_card_row(bottom, x, y, bw, bh, thresh_val=thresh_val)
                if len(rects) == 2:
                    candidates.extend(rects)
                else:
                    # Force split in half — hero cards often overlap
                    half_w = bw // 2
                    candidates.append((x, y, half_w, bh))
                    candidates.append((x + half_w, y, bw - half_w, bh))

        if len(candidates) >= 2:
            break

    if len(candidates) < 2:
        # Last resort: try to find the single brightest blob and force-split it
        for thresh_val in [130, 140]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                area = bw * bh
                aspect = bw / bh if bh > 0 else 0
                # Look for a merged pair: wider than tall, decent size
                if area > 2000 and bh > 30 and 0.85 < aspect < 3.0:
                    half_w = bw // 2
                    candidates = [(x, y, half_w, bh), (x + half_w, y, bw - half_w, bh)]
                    break
            if len(candidates) >= 2:
                break

    if len(candidates) < 2:
        return []

    # Find the best pair: two cards close together at similar y, similar size
    # Prefer larger cards closer to center-bottom (hero position)
    bh_center = bottom.shape[0]
    bw_center = bottom.shape[1] // 2
    candidates.sort(key=lambda r: r[0])
    best_pair = None
    best_score = float("inf")

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            xi, yi, wi, hi = candidates[i]
            xj, yj, wj, hj = candidates[j]
            y_diff = abs(yi - yj)
            gap = xj - (xi + wi)
            h_diff = abs(hi - hj)

            if y_diff < max(hi, hj) * 0.5 and -10 <= gap < wi * 2 and h_diff < max(hi, hj) * 0.5:
                # Base proximity score
                score = abs(gap) + y_diff * 2 + h_diff
                # Penalize small cards (hero cards should be among the largest)
                avg_area = (wi * hi + wj * hj) / 2
                size_penalty = max(0, 3000 - avg_area) * 0.05
                # Penalize cards far from horizontal center
                pair_cx = (xi + xj + wj) / 2
                center_dist = abs(pair_cx - bw_center)
                center_penalty = center_dist * 0.1
                score += size_penalty + center_penalty
                if score < best_score:
                    best_score = score
                    best_pair = [candidates[i], candidates[j]]

    if best_pair is None:
        return []

    cards = _identify_cards(bottom, best_pair, min_conf=0.10)
    return [c for c in cards if c != "??"]


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
    player_stacks = _find_player_stacks(table_region)

    return {
        "board_cards": board_cards,
        "hero_cards": hero_cards,
        "player_stacks": player_stacks,
        "table_color": table_color,
    }
