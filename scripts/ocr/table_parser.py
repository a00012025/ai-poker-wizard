"""Table region parser for Natural8 replay screenshots.

Extracts board cards, hero cards, player stacks, and table color
from the upper (table) region of an N8 replay screenshot.
"""

import cv2
import numpy as np


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


def _locate_board_cards(table_region: np.ndarray) -> list[np.ndarray]:
    """Return list of individual board card crops (BGR), left-to-right.

    Returns empty list if no board detected. Pure localization — no OCR,
    no suit detection, no classification.
    """
    h, w = table_region.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.55)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    center = table_region[y1:y2, x1:x2]

    # Strategy 1: Find individual card contours
    rects = _find_individual_card_contours(center)
    if rects and len(rects) >= 3:
        crops = []
        for (x, y, cw, ch) in rects:
            crop = center[y:y + ch, x:x + cw]
            if crop.size > 0:
                crops.append(crop)
        return crops

    # Strategy 2: Find merged bright row and split
    row = _find_bright_row(center, thresh_val=160, min_height=30)
    if row is None:
        # Try lower thresholds
        for tv in [140, 120]:
            row = _find_bright_row(center, thresh_val=tv, min_height=30)
            if row:
                break
    if row is None:
        return []

    rx, ry, rw, rh = row

    if rw <= rh * 1.2:
        crops = []
        for (x, y, cw, ch) in [row]:
            crop = center[y:y + ch, x:x + cw]
            if crop.size > 0:
                crops.append(crop)
        return crops

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


def _find_board_cards(table_region: np.ndarray) -> list[str]:
    """Find and identify board cards in the center of the table via CardCNN."""
    from .classifier.infer import CardClassifier

    crops = _locate_board_cards(table_region)
    if not crops:
        return []
    results = CardClassifier().classify_batch(crops)
    return [f"{r}{s}" for r, s, _ in results if r and s]


def _find_individual_card_contours(center: np.ndarray) -> list[tuple]:
    """Find individual card rectangles in the center region.

    Works when cards are visually separated (not touching).
    Each card is a bright rectangle with aspect ratio ~0.65-0.95.
    """
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    ch, cw = center.shape[:2]

    for tv in [180, 160, 140]:
        _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / h if h > 0 else 0
            # Individual card: roughly card-shaped, decent area, reasonable size.
            # Aspect 0.55-1.15: covers portrait cards (taller than wide)
            # and nearly-square cards from some table angles.
            # y > ch * 0.15: reject contours at top of center region where
            # player avatars are — board cards are in the lower portion.
            if (area > 800 and h > 25 and w > 20
                    and 0.55 < aspect < 1.15
                    and h < ch * 0.8  # not taller than 80% of center region
                    and y > ch * 0.15):  # not in top 15% (player avatars)
                candidates.append((x, y, w, h))

        # Need at least 3 cards at similar Y (same row)
        if len(candidates) >= 3:
            # Group by Y proximity — cards should be at roughly same Y
            candidates.sort(key=lambda r: r[1])
            best_cluster = []
            for i in range(len(candidates)):
                cluster = [candidates[i]]
                for j in range(i + 1, len(candidates)):
                    if abs(candidates[j][1] - candidates[i][1]) < candidates[i][3] * 0.5:
                        cluster.append(candidates[j])
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster

            if len(best_cluster) >= 3:
                best_cluster.sort(key=lambda r: r[0])
                # Verify cards are truly separated (gaps between them)
                # On green theme, cards touch → this strategy shouldn't fire
                has_gaps = True
                for k in range(len(best_cluster) - 1):
                    gap = best_cluster[k + 1][0] - (best_cluster[k][0] + best_cluster[k][2])
                    if gap < 2:  # cards touching or overlapping
                        has_gaps = False
                        break
                if has_gaps:
                    return best_cluster[:5]

    return []


def _locate_hero_cards(table_region: np.ndarray) -> list[np.ndarray]:
    """Return [card1_crop, card2_crop] (BGR ndarrays), or [] if no blob found.

    Pure localization — no rank/suit detection. Same blob logic currently
    used inside _find_hero_cards.
    """
    h, w = table_region.shape[:2]
    hero = table_region[int(h * 0.58):int(h * 0.85), int(w * 0.28):int(w * 0.68)]
    ah, aw = hero.shape[:2]
    if ah < 20 or aw < 20:
        return []

    gray = cv2.cvtColor(hero, cv2.COLOR_BGR2GRAY)

    # Find the card pair blob — try thresholds from high to low
    best_blob = None
    for tv in [200, 190, 180, 170, 160, 150, 140, 130, 120]:
        _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,
                                  iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, cw, ch_ = cv2.boundingRect(c)
            area = cw * ch_
            # Card pair: wider than tall, reasonable size
            if (area > 1500 and ch_ > 25 and cw > 60
                    and 1.2 < cw / ch_ < 2.8):
                if best_blob is None or area > best_blob[4]:
                    best_blob = (x, y, cw, ch_, area)
        if best_blob and best_blob[4] > 2500:
            break

    if not best_blob:
        # Fallback: accept wider aspect ratio range
        for tv in [160, 150, 140, 130, 120]:
            _, thresh = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,
                                      iterations=2)
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

    # Split at 48% (left card slightly narrower due to overlap rendering)
    split = int(cw * 0.48)
    card1 = hero[y:y + ch_, x:x + split + 3]
    card2 = hero[y:y + ch_, x + split - 3:x + cw]
    return [card1, card2]


def _find_hero_cards(table_region: np.ndarray) -> tuple[list[str], float]:
    """Find and identify hero's hole cards via CardCNN.

    Returns (cards, confidence) where confidence is min over all card
    predictions (min of rank_softmax_max and suit_softmax_max per card).
    Low confidence naturally triggers the Gemini fallback in gemini_session.
    """
    from .classifier.infer import CardClassifier

    crops = _locate_hero_cards(table_region)
    if not crops:
        return [], 0.0
    results = CardClassifier().classify_batch(crops)
    cards = [f"{r}{s}" for r, s, _ in results if r and s]
    conf = min((c for _, _, c in results), default=0.0)
    return cards, conf


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
    hero_cards, hero_card_conf = _find_hero_cards(table_region)
    all_stacks_named = _find_all_stacks(table_region)
    hero_stack = _find_hero_stack(table_region)

    # Flat list of stack values for backward compatibility
    all_stacks = [s["stack"] for s in all_stacks_named]

    return {
        "board_cards": board_cards,
        "hero_cards": hero_cards,
        "hero_card_conf": hero_card_conf,
        "hero_stack": hero_stack,
        "player_stacks": all_stacks,
        "named_stacks": all_stacks_named,
        "table_color": table_color,
    }


def _find_all_stacks(table_region: np.ndarray) -> list[dict]:
    """Find all player stacks with their names from the table region.

    Groups nearby name and "XX.X BB" text by proximity.  A name text and
    a BB text that are close vertically and horizontally belong to the
    same player.

    Returns:
        [{"name": str|None, "stack": float, "y": float, "x": float}, ...]
    """
    import re
    from .ocr_utils import ocr_full_image

    results = ocr_full_image(table_region)
    bb_pattern = re.compile(r'(\d+\.?\d*)\s*BB', re.IGNORECASE)

    # Collect BB entries
    bb_entries = []
    for r in results:
        m = bb_pattern.search(r["text"])
        if m:
            try:
                val = float(m.group(1))
                if 0.5 < val < 500:
                    bb_entries.append({
                        "value": val,
                        "y": r["center_y"],
                        "x": r["center_x"],
                    })
            except ValueError:
                pass

    # Collect name entries (non-numeric, not BB/WIN/action keywords)
    _SKIP_WORDS = {
        "BB", "SB", "WIN", "NATURAL8", "CHECK", "FOLD", "CALL",
        "BET", "RAISE", "WN",
    }
    name_entries = []
    for r in results:
        text = r["text"].strip()
        # Skip BB values, short text, pure numbers, skip words
        if len(text) < 2:
            continue
        if bb_pattern.match(text):
            continue
        if re.match(r'^[\d.]+$', text):
            continue
        if text.upper() in _SKIP_WORDS:
            continue
        # Skip pot-like numbers (standalone digits that aren't names)
        if re.match(r'^\d+$', text) and len(text) <= 3:
            continue
        name_entries.append({
            "name": text,
            "y": r["center_y"],
            "x": r["center_x"],
        })

    # Match names to stacks by proximity (name is usually ABOVE the stack)
    matched = []
    used_names = set()
    for bb in bb_entries:
        best_name = None
        best_dist = 999
        for i, nm in enumerate(name_entries):
            if i in used_names:
                continue
            dy = abs(nm["y"] - bb["y"])
            dx = abs(nm["x"] - bb["x"])
            # Name should be within ~60px vertically and ~100px horizontally
            dist = dy + dx * 0.5  # weight vertical proximity more
            if dy < 60 and dx < 100 and dist < best_dist:
                best_dist = dist
                best_name = (i, nm["name"])

        entry = {"stack": bb["value"], "y": bb["y"], "x": bb["x"]}
        if best_name:
            used_names.add(best_name[0])
            entry["name"] = best_name[1]
        else:
            entry["name"] = None
        matched.append(entry)

    return matched


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
