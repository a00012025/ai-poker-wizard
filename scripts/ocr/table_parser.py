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
    """Identify cards using multi-strategy OCR for rank + BGR for suit.

    Uses the same robust _ocr_card_rank pipeline as hero cards.

    Returns:
        List of card strings like ["Ks", "9d", "3d"].
    """
    from .ocr_utils import ocr_full_image

    cards = []
    for (x, y, w, h) in card_rects:
        card_img = region[y:y + h, x:x + w]
        if card_img.size == 0:
            cards.append("??")
            continue

        rank, _conf = _ocr_card_rank(card_img, ocr_full_image)
        suit = _detect_suit_bgr(card_img)
        if rank:
            cards.append(f"{rank}{suit}")
        else:
            cards.append("??")
    return cards


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


def _ocr_card_rank(card: np.ndarray, ocr_full_image) -> tuple[str | None, float]:
    """OCR a single card image for rank, with template matching fallback.

    Tries multiple strategies in order:
    1. OCR on top-left rank crop (avoids suit symbol misreads)
    2. OCR on upscaled full card
    3. OCR on inverted binary rank crop
    4. OCR on 2x further upscaled rank crop
    5. OCR on inverted binary full card
    6. OCR on sharpened image
    7. Template matching via CardMatcher

    Returns:
        (rank, confidence) — confidence decreases with later attempts.
    """
    _RANK_CHARS = {"2", "3", "4", "5", "6", "7", "8", "9", "10",
                   "J", "Q", "K", "A"}
    _RANK_MAP = {
        "10": "T", "1O": "T", "IO": "T", "l0": "T", "I0": "T",
    }
    # Characters that EasyOCR produces for Q (standalone, not part of "10")
    _Q_CHARS = {"0", "O"}

    if card is None or card.size == 0 or card.shape[0] < 10:
        return None, 0.0

    # Upscale aggressively for small cards
    target_h = 360
    scale = max(3, target_h // max(card.shape[0], 1))
    scale = min(scale, 8)
    card_up = cv2.resize(card, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    def _extract_rank(ocr_results, allow_q_from_zero=True):
        # First pass: look for definitive rank characters across ALL
        # results before considering the lossy "0"→Q fallback.  A crop
        # that overlaps a "$0.75" prize banner can produce multiple
        # detections like [('0', 0.95), ('75', 1.0), ('3', 1.0)] — the
        # actual rank (3) must win over the banner's "0" triggering the
        # 0→Q mapping.  Regression for H2758 where 3♠ was read as Q♠
        # because '0' appeared before '3' in the results list.
        for r in ocr_results:
            t = r["text"].strip()
            tu = t.upper()
            if t in _RANK_CHARS:
                return _RANK_MAP.get(t, t)
            if tu in _RANK_CHARS:
                return _RANK_MAP.get(tu, tu)
            if tu in _RANK_MAP:
                return _RANK_MAP[tu]
            if len(t) == 1 and tu in "23456789JQKA":
                return tu
        # Second pass: "0" / "O" → Q fallback (EasyOCR misreads Q).
        # Only reached when no definitive rank was found above.
        if allow_q_from_zero:
            for r in ocr_results:
                t = r["text"].strip()
                tu = t.upper()
                conf = r.get("conf", 0)
                if tu in _Q_CHARS and len(t) == 1 and conf > 0.7:
                    return "Q"
        return None

    # Attempt 1: OCR on top-left rank crop — highest confidence (0.9)
    ch, cw_up = card_up.shape[:2]
    rh = int(ch * 0.50)
    rw = int(cw_up * 0.60)
    rank_crop_rank = None
    rank_crop_via_zero = False  # True if rank was derived from "0"→Q mapping
    rank_crop_conf = 0.0
    rank_crop = None
    if rh > 10 and rw > 10:
        rank_crop = card_up[0:rh, 0:rw]
        rank_crop_ocr = ocr_full_image(rank_crop)
        rank_crop_conf = max((r.get("conf", 0) for r in rank_crop_ocr), default=0)
        # Check if we'd get Q only via 0→Q mapping
        rank_crop_rank = _extract_rank(rank_crop_ocr)
        rank_crop_strict = _extract_rank(rank_crop_ocr, allow_q_from_zero=False)
        if rank_crop_rank == "Q" and rank_crop_strict is None:
            rank_crop_via_zero = True
        # Also flag fragile if Q came only from lowercase 'q' (no uppercase
        # 'Q' in the OCR results). Lowercase 'q' is OCR-confusable with '9',
        # especially when overlays (e.g. WIN badge) obscure part of the card.
        if rank_crop_rank == "Q" and not rank_crop_via_zero:
            has_upper_q = any(
                r["text"].strip() == "Q" for r in rank_crop_ocr
            )
            if not has_upper_q:
                rank_crop_via_zero = True

    # Attempt 1b: on very small cards the rank digit extends past 50% of
    # the card height; a taller rank crop (0.70) recovers it when the
    # first attempt returned nothing.  Must not run when attempt 1 found
    # a rank — the tighter crop is preferred to avoid picking up the
    # suit symbol.
    if not rank_crop_rank:
        rh2 = int(ch * 0.70)
        rw2 = int(cw_up * 0.60)
        if rh2 > 10 and rw2 > 10:
            rank_crop2 = card_up[0:rh2, 0:rw2]
            rank_crop2_ocr = ocr_full_image(rank_crop2)
            conf2 = max((r.get("conf", 0) for r in rank_crop2_ocr), default=0)
            rank2 = _extract_rank(rank_crop2_ocr)
            rank2_strict = _extract_rank(rank_crop2_ocr, allow_q_from_zero=False)
            if rank2 and conf2 > 0.70:
                rank_crop_rank = rank2
                rank_crop_conf = conf2
                rank_crop = rank_crop2
                if rank2 == "Q" and rank2_strict is None:
                    rank_crop_via_zero = True

    # Attempt 1c: shifted-down rank crop for cards whose blob includes
    # a prize-money banner ("$0.75", "$1.12", …) overlapping the card
    # top.  The banner consumes the top ~20% of the blob, pushing the
    # rank digit into y[0.25:0.50] instead of y[0:0.45].  Standard
    # crops (attempts 1 and 1b) then see only the banner text, which
    # _extract_rank correctly rejects as non-rank.  A crop starting at
    # y=0.30 excludes the banner entirely and isolates the rank.
    # Regression for H2759 J♥ where "$0.75" masked the 'J' on every
    # top-anchored crop and attempt 2 also saw only "1.75".
    if not rank_crop_rank:
        rh3_lo = int(ch * 0.30)
        rh3_hi = int(ch * 0.80)
        rw3 = int(cw_up * 0.60)
        if rh3_hi - rh3_lo > 20 and rw3 > 10:
            rank_crop3 = card_up[rh3_lo:rh3_hi, 0:rw3]
            rank_crop3_ocr = ocr_full_image(rank_crop3)
            conf3 = max((r.get("conf", 0) for r in rank_crop3_ocr), default=0)
            rank3 = _extract_rank(rank_crop3_ocr)
            rank3_strict = _extract_rank(rank_crop3_ocr, allow_q_from_zero=False)
            if rank3 and conf3 > 0.80:
                rank_crop_rank = rank3
                rank_crop_conf = conf3
                rank_crop = rank_crop3
                if rank3 == "Q" and rank3_strict is None:
                    rank_crop_via_zero = True

    # Attempt 2: OCR on upscaled full card (0.85)
    full_card_ocr = ocr_full_image(card_up)
    full_card_rank = _extract_rank(full_card_ocr)
    full_card_conf = max((r.get("conf", 0) for r in full_card_ocr), default=0)

    if rank_crop_rank and full_card_rank:
        if rank_crop_rank == full_card_rank:
            return rank_crop_rank, 0.9
        # When rank_crop got Q via fragile "0"→Q mapping, check full_card.
        # For actual 9, full_card reads "9" with conf >0.45.
        # For actual Q, full_card reads "9" with conf <0.35.
        if rank_crop_via_zero and full_card_rank != "Q" and full_card_conf > 0.45:
            return full_card_rank, 0.85
        # When rank_crop has very low confidence (<0.3), trust full_card.
        # E.g., rank_crop misreads Q as "10"(conf 0.14) but full_card sees "Q".
        if rank_crop_conf < 0.3 and full_card_conf > rank_crop_conf:
            return full_card_rank, 0.85
        # For direct reads at decent confidence, only trust full card at
        # very high confidence (>0.999) to avoid false corrections.
        if full_card_conf > 0.999:
            return full_card_rank, 0.85
        return rank_crop_rank, 0.85
    if rank_crop_rank:
        return rank_crop_rank, 0.9
    if full_card_rank:
        return full_card_rank, 0.85

    # Attempt 3: Inverted binary on rank crop (0.7)
    if rh > 10 and rw > 10:
        gray_rc = cv2.cvtColor(rank_crop, cv2.COLOR_BGR2GRAY)
        _, bin_rc = cv2.threshold(gray_rc, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv_rc = cv2.bitwise_not(bin_rc)
        rank = _extract_rank(ocr_full_image(inv_rc))
        if rank:
            return rank, 0.7

    # Attempt 4: Further 2x upscale on rank crop (0.6)
    if rh > 10 and rw > 10:
        rank_crop_2x = cv2.resize(rank_crop, None, fx=2, fy=2,
                                  interpolation=cv2.INTER_CUBIC)
        rank = _extract_rank(ocr_full_image(rank_crop_2x))
        if rank:
            return rank, 0.6

    # Attempt 5: Inverted binary on full card (0.5)
    gray_card = cv2.cvtColor(card_up, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_card, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    rank = _extract_rank(ocr_full_image(inverted))
    if rank:
        return rank, 0.5

    # Attempt 6: Sharpened image (0.4)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(card_up, -1, sharpen_kernel)
    rank = _extract_rank(ocr_full_image(sharpened))
    if rank:
        return rank, 0.4

    # Attempt 7: CLAHE enhanced (0.35)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_card)
    rank = _extract_rank(ocr_full_image(enhanced))
    if rank:
        return rank, 0.35

    # Attempt 8: Red-minus-blue channel isolation (0.3)
    if card is not None and card.size > 0 and len(card.shape) == 3:
        rb_up = cv2.resize(card, None, fx=4, fy=4,
                           interpolation=cv2.INTER_CUBIC)
        rb_uh, rb_uw = rb_up.shape[:2]
        for (y1f, y2f, xf) in [(0.0, 0.50, 0.55),
                                (0.25, 0.60, 0.60)]:
            rb_y1 = int(rb_uh * y1f)
            rb_y2 = int(rb_uh * y2f)
            rb_xw = int(rb_uw * xf)
            if rb_y2 - rb_y1 > 10 and rb_xw > 10:
                rb_crop = rb_up[rb_y1:rb_y2, 0:rb_xw]
                b_ch, _g_ch, r_ch = cv2.split(rb_crop)
                rb_diff = cv2.subtract(r_ch, b_ch)
                _, rb_bin = cv2.threshold(rb_diff, 20, 255,
                                          cv2.THRESH_BINARY)
                rb_inv = cv2.bitwise_not(rb_bin)
                rank = _extract_rank(ocr_full_image(rb_inv))
                if rank:
                    return rank, 0.3

    # Attempt 9: Template matching as final fallback (0.2)
    matcher = _get_matcher()
    r, _s, conf = matcher.match(card)
    if r and conf > 0.05:
        return r, 0.2

    return None, 0.0


def _hero_hull_norm(card_img: np.ndarray) -> float:
    """Compute hull defect norm for the center red suit symbol.

    Used to verify heart vs diamond: hearts have deep concavity
    (norm > 70) from the top dip.  Returns 0 if no red contour found.
    """
    scale_up = 4
    up = cv2.resize(card_img, None, fx=scale_up, fy=scale_up,
                    interpolation=cv2.INTER_CUBIC)
    uh, uw = up.shape[:2]
    center = up[int(uh * 0.45):int(uh * 0.85),
                int(uw * 0.15):int(uw * 0.85)]
    if center.size == 0:
        return 0.0
    hsv_c = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    rm1 = cv2.inRange(hsv_c, np.array([0, 50, 50]),
                      np.array([15, 255, 255]))
    rm2 = cv2.inRange(hsv_c, np.array([165, 50, 50]),
                      np.array([180, 255, 255]))
    rmask = cv2.bitwise_or(rm1, rm2)
    rc, _ = cv2.findContours(rmask, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    rc = [c for c in rc if cv2.contourArea(c) > 20]
    if not rc:
        return 0.0
    big = max(rc, key=cv2.contourArea)
    ca = cv2.contourArea(big)
    hull_idx = cv2.convexHull(big, returnPoints=False)
    if len(hull_idx) <= 3:
        return 0.0
    defects = cv2.convexityDefects(big, hull_idx)
    if defects is None or len(defects) == 0:
        return 0.0
    max_defect = max(d[0][3] for d in defects)
    return max_defect / (ca ** 0.5) if ca > 0 else 0.0


def _suit_template_match(card_img: np.ndarray, is_red: bool,
                         min_margin: float = 0.19,
                         return_margin: bool = False,
                         ) -> str | None | tuple[str | None, float]:
    """Determine suit via template matching on the mini suit symbol.

    Matches the suit crop region (below rank, top-left of card) against
    suit templates, restricted to the correct color (red: h/d, black: s/c).
    Returns the best-matching suit or None if inconclusive.
    When return_margin=True, returns (suit, margin) tuple.
    """
    h, w = card_img.shape[:2]
    scale = max(3, 360 // max(h, 1))
    scale = min(scale, 8)
    up = cv2.resize(card_img, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
    uh, uw = up.shape[:2]
    # Suit symbol region: below rank character
    suit_crop = up[int(uh * 0.30):int(uh * 0.65), 0:int(uw * 0.55)]
    if suit_crop.size == 0:
        return None
    gray = cv2.cvtColor(suit_crop, cv2.COLOR_BGR2GRAY)
    sh, sw = suit_crop.shape[:2]
    if sh < 5 or sw < 5:
        return None

    matcher = _get_matcher()
    candidates = ("h", "d") if is_red else ("s", "c")
    scores = {}
    for sname in candidates:
        tmpl = matcher.suit_templates.get(sname)
        if tmpl is None:
            continue
        resized = cv2.resize(tmpl, (sw, sh))
        gt = (cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
              if len(resized.shape) == 3 else resized)
        result = cv2.matchTemplate(gray, gt, cv2.TM_CCOEFF_NORMED)
        scores[sname] = result.max()

    if not scores:
        return (None, 0.0) if return_margin else None
    best = max(scores, key=scores.get)
    second = min(scores, key=scores.get)
    margin = scores[best] - scores[second]
    # Require strong margin to avoid false positives.
    # Genuine corrections show large margin (e.g., d=0.35 vs h=-0.09 = 0.44).
    # False corrections show small margin (e.g., d=0.35 vs h=0.30 = 0.05).
    if margin < min_margin:
        return (None, margin) if return_margin else None
    if scores[best] < 0.10:
        return (None, margin) if return_margin else None
    return (best, margin) if return_margin else best


def _detect_suit_bgr(card_img: np.ndarray) -> str:
    """Detect suit by analyzing BGR color of dark (ink) pixels on the card.

    Step 1: Determine red vs black from ink pixel BGR values.
    Step 2a (red): Use convex hull defects of center suit contour — hearts
                   have a deep concavity (top dip), diamonds are fully convex.
                   Falls back to green channel when center crop is poor.
    Step 2b (black): Use solidity of center suit symbol — spades are more
                     solid (>0.88), clubs have lower solidity from lobes.
    """
    h, w = card_img.shape[:2]
    # Sample top portion where rank + suit symbol are
    sample = card_img[2:int(h * 0.75), 2:int(w * 0.6)]
    if sample.size == 0:
        return "s"

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Filter to card-ink pixels: dark pixels near bright card face.
    # On dark tables, table background pixels (dark green) contaminate
    # the sample and dilute red ink colors. Mask to dark pixels that
    # are adjacent to bright (card face) pixels.
    _, bright_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    card_ink_mask = cv2.bitwise_and(
        dark_mask, cv2.dilate(bright_mask, None, iterations=3))
    card_ink_pixels = sample[card_ink_mask > 0]

    # Use card-ink pixels if enough; fall back to all dark pixels
    if len(card_ink_pixels) >= 5:
        dark_pixels = card_ink_pixels
    else:
        dark_pixels = sample[dark_mask > 0]

    if len(dark_pixels) < 5:
        return "s"

    avg_r = float(np.mean(dark_pixels[:, 2]))  # BGR → R is index 2
    avg_g = float(np.mean(dark_pixels[:, 1]))
    avg_b = float(np.mean(dark_pixels[:, 0]))

    # Pre-check: when a tiny hero card crop includes green-felt bleed
    # at the overlap seam, ink-BGR averaging can be green-dominated
    # even on clearly red cards.  Count pure saturated-red HSV pixels
    # in an INNER region (y[0.25:0.72], x[0.10:0.55]) that excludes
    # (a) the top 25% where a "$0.75" prize-money banner overlays the
    # card and contaminates the ink-BGR mean, (b) the left seam to
    # adjacent cards (which leaks green felt on red-suit cards), and
    # (c) the right edge where a neighbor card's red can bleed onto a
    # black card (H2587 K♣ next to 9♥).  Tight H/S/V thresholds
    # (H≤10 / ≥170, S≥120, V≥80) exclude orange/yellow chip-glow and
    # the banner's own yellow pixels.  Threshold 0.05 gives a clean
    # gap: real red cards score ≥ 0.08 (H2759 J♥ at 0.088 is the
    # tightest), real black cards score ≤ 0.016.
    h_card, w_card = card_img.shape[:2]
    inner = card_img[int(h_card * 0.25):int(h_card * 0.72),
                     int(w_card * 0.10):int(w_card * 0.55)]
    if inner.size > 0:
        hsv_pre = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        rp_m1 = cv2.inRange(hsv_pre, np.array([0, 120, 80]),
                            np.array([10, 255, 255]))
        rp_m2 = cv2.inRange(hsv_pre, np.array([170, 120, 80]),
                            np.array([180, 255, 255]))
        red_pure_pct = float(np.sum(cv2.bitwise_or(rp_m1, rp_m2) > 0)) / max(
            inner.shape[0] * inner.shape[1], 1)
    else:
        red_pure_pct = 0.0

    is_red = (avg_r > avg_b + 20 and avg_r > avg_g + 20) or red_pure_pct > 0.05

    if is_red:
        # Hearts vs Diamonds: hull defects approach.
        # Heart ♥ has a concave dip at top → convex hull defect > 1500.
        # Diamond ♦ is fully convex → all defects < 1000.
        # Also use green channel as secondary signal.
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]),
                                np.array([15, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([165, 50, 50]),
                                np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = sample[red_mask > 0]

        green_says_heart = False
        if len(red_pixels) >= 5:
            red_avg_g = float(np.mean(red_pixels[:, 1]))
            green_says_heart = red_avg_g > 60

        # Analyze center suit symbol shape via hull defects
        scale_up = 4
        card_up = cv2.resize(card_img, None, fx=scale_up, fy=scale_up,
                             interpolation=cv2.INTER_CUBIC)
        uh, uw = card_up.shape[:2]
        center = card_up[int(uh * 0.45):int(uh * 0.85),
                         int(uw * 0.15):int(uw * 0.85)]
        if center.size > 0:
            hsv_c = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
            rm1 = cv2.inRange(hsv_c, np.array([0, 50, 50]),
                              np.array([15, 255, 255]))
            rm2 = cv2.inRange(hsv_c, np.array([165, 50, 50]),
                              np.array([180, 255, 255]))
            rmask = cv2.bitwise_or(rm1, rm2)
            rc, _ = cv2.findContours(rmask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            rc = [c for c in rc if cv2.contourArea(c) > 20]
            if rc:
                ch_c, cw_c = center.shape[:2]
                center_area = ch_c * cw_c
                big = max(rc, key=cv2.contourArea)
                big_area = cv2.contourArea(big)

                # On face cards (K/Q/J) the biggest contour can be
                # the card artwork, not the suit symbol.  When that
                # contour exceeds 25% of the center crop, look for a
                # smaller, suit-shaped contour instead.  The
                # replacement must be large enough to be a real suit
                # symbol (area > 800 after 4x upscale) to avoid
                # picking tiny noise contours.
                #
                # However, for Aces and number cards the center suit
                # symbol IS the biggest contour and naturally fills
                # >25% of the crop.  Check big contour's hull defects
                # first — if conclusive, use it directly.
                target = big
                if big_area / center_area > 0.25 and len(rc) > 1:
                    # Check if big contour's hull defects are conclusive
                    big_hull_idx = cv2.convexHull(big, returnPoints=False)
                    big_max_defect = 0
                    if len(big_hull_idx) > 3:
                        big_defects = cv2.convexityDefects(
                            big, big_hull_idx)
                        if big_defects is not None:
                            big_max_defect = max(
                                d[0][3] for d in big_defects)
                    big_norm = (big_max_defect / (big_area ** 0.5)
                                if big_area > 0 else 0)
                    # If big contour clearly heart or diamond, keep it
                    if big_norm > 22 or big_norm < 12:
                        pass  # target stays as big
                    else:
                        # Ambiguous — try a smaller contour
                        for c in sorted(rc, key=cv2.contourArea,
                                        reverse=True):
                            ca_c = cv2.contourArea(c)
                            if ca_c / center_area > 0.25:
                                continue
                            bx, by, bw_, bh_ = cv2.boundingRect(c)
                            asp = bw_ / bh_ if bh_ > 0 else 0
                            if 0.6 < asp < 1.4 and ca_c > 800:
                                target = c
                                break

                hll = cv2.convexHull(target)
                ha = cv2.contourArea(hll)
                ca = cv2.contourArea(target)
                csol = ca / ha if ha > 0 else 1.0

                # Hull defects normalized by sqrt(area) for scale
                # independence.  Hearts norm > 22, diamonds norm < 18.
                hull_idx = cv2.convexHull(target, returnPoints=False)
                max_defect = 0
                if len(hull_idx) > 3:
                    defects = cv2.convexityDefects(target, hull_idx)
                    if defects is not None:
                        max_defect = max(d[0][3] for d in defects)

                norm = max_defect / (ca ** 0.5) if ca > 0 else 0

                # Hull defects are only reliable when the contour is
                # well-formed (csol > 0.80).  Fragmented contours
                # (e.g. H2659: norm 117-178 on ♦ cards with
                # solidity 0.64-0.73) produce spurious high defects
                # that misclassify diamonds as hearts.  When solidity
                # is low, skip hull-based decisions entirely and fall
                # through to the G/R color ratio path.
                # Hull defects need cross-checking for small contours.
                # Three failure modes seen:
                #  1. Fragmented (csol < 0.80): spurious high norms
                #     (H2659: norm 117-178, csol 0.64-0.73 on ♦)
                #  2. Tiny with diamond color (H2660: norm 26.1,
                #     ca 1100, G/R 0.34 on A♦ misread as ♥)
                # For small contours (ca < 4000), cross-check hull
                # defects against G/R color.  If G/R > 0.28 (diamond
                # color) and green channel doesn't confirm heart,
                # don't trust the hull defect.
                if norm > 22 and csol > 0.80:
                    if ca >= 4000:
                        return "h"
                    # Small contour: cross-check with color
                    _cr_px = center[rmask > 0]
                    _skip = False
                    if len(_cr_px) >= 5:
                        _cg = float(np.mean(_cr_px[:, 1]))
                        _cr = float(np.mean(_cr_px[:, 2]))
                        _gr = _cg / _cr if _cr > 0 else 0
                        if _gr > 0.30 and not green_says_heart:
                            _skip = True
                    if not _skip:
                        return "h"

                # Very high hull-defect norm declares heart even when
                # solidity is low.  Face cards (J/Q/K) merge the rank
                # character into the red-mask center contour,
                # fragmenting it (csol drops to ~0.63) and blocking the
                # strict `norm > 22 and csol > 0.80` check above.
                # Threshold 150 cleanly separates observed samples:
                #   H2759 J♥ (real heart): norm=214, csol=0.63
                #   H2659 fragmented diamonds: norm=129-135, csol=0.70-0.73
                # Requires ca > 1500 to reject tiny noise contours.
                if norm >= 150 and ca > 1500 and csol <= 0.80:
                    return "h"

                if ca >= 4000 and csol > 0.80:
                    if norm < 18:
                        return "d"
                    # Ambiguous zone (18-22): prefer green channel,
                    # then solidity.
                    if green_says_heart:
                        return "h"
                    if csol >= 0.95:
                        return "d"
                    if csol < 0.90:
                        return "h"

                if ca < 4000 or csol <= 0.80:
                    # Small or fragmented contour: hull defects are
                    # unreliable at this scale/quality.  Use
                    # center suit pixel color instead.  In N8 rendering
                    # hearts are a purer red (lower green component)
                    # than diamonds.
                    # Two color regimes exist across screenshots:
                    #   Regime A (cool reds): hearts G/R < 0.093,
                    #                         diamonds G/R > 0.10
                    #   Regime B (warm reds): hearts G/R 0.186-0.198,
                    #                         diamonds G/R > 0.22
                    center_red_px = center[rmask > 0]
                    if len(center_red_px) >= 5:
                        cg = float(np.mean(center_red_px[:, 1]))
                        cr = float(np.mean(center_red_px[:, 2]))
                        gr = cg / cr if cr > 0 else 0
                        if gr < 0.15:
                            # Regime A: dark/cool reds
                            return "h" if gr < 0.093 else "d"
                        else:
                            # Regime B: warm reds.
                            # G/R 0.19-0.22 is ambiguous — consult
                            # template matching with relaxed margin to
                            # break the tie (e.g. diamond at G/R 0.207).
                            if 0.19 <= gr <= 0.22:
                                tmpl = _suit_template_match(
                                    card_img, is_red=True,
                                    min_margin=0.12)
                                if tmpl is not None:
                                    return tmpl
                            return "h" if gr < 0.21 else "d"

        if green_says_heart:
            return "h"
        return "d"
    else:
        # Spades vs Clubs: multi-feature voting on center suit symbol.
        # A single solidity threshold is fragile (spade hero cards can
        # have solidity 0.93-0.95 which overlaps with clubs).
        # Instead, use three features with weighted voting:
        #   1. top_ratio  — width of top 25% vs middle 40% of contour.
        #      Spade has a narrow point at top (ratio < 0.45),
        #      club has wide lobes (ratio > 0.55).  Best discriminator.
        #   2. solidity   — contour area / convex hull area.
        #      Spade > 0.93, club < 0.93 (with overlap zone).
        #   3. n_defects  — number of significant convex hull defects.
        #      Club lobes create more defects (>= 3), spade fewer.
        scale = 4
        card_up = cv2.resize(card_img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        uh, uw = card_up.shape[:2]

        center_suit = card_up[int(uh * 0.45):int(uh * 0.85),
                              int(uw * 0.15):int(uw * 0.85)]
        if center_suit.size == 0:
            return "s"

        gray_cs = cv2.cvtColor(center_suit, cv2.COLOR_BGR2GRAY)
        _, dark_cs = cv2.threshold(gray_cs, 130, 255,
                                   cv2.THRESH_BINARY_INV)
        cs_contours, _ = cv2.findContours(dark_cs, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        cs_contours = [c for c in cs_contours
                       if cv2.contourArea(c) > 20]

        if cs_contours:
            biggest_cs = max(cs_contours, key=cv2.contourArea)
            hull = cv2.convexHull(biggest_cs)
            hull_area = cv2.contourArea(hull)
            cont_area = cv2.contourArea(biggest_cs)
            sol = cont_area / hull_area if hull_area > 0 else 1.0
            bx, by, bw, bh = cv2.boundingRect(biggest_cs)

            # Feature 1: top_ratio — top width vs middle width.
            # Spade: narrow point at top -> ratio 0.27-0.44.
            # Club: wide lobes at top -> ratio 0.56-0.66.
            top_slice = dark_cs[by:by + max(1, int(bh * 0.25)),
                                bx:bx + bw]
            mid_slice = dark_cs[by + int(bh * 0.3):by + int(bh * 0.7),
                                bx:bx + bw]
            top_w = np.mean(np.sum(top_slice > 0, axis=1)) \
                if top_slice.size > 0 else 0
            mid_w = np.mean(np.sum(mid_slice > 0, axis=1)) \
                if mid_slice.size > 0 else 0
            top_ratio = top_w / mid_w if mid_w > 0 else 1.0

            # Feature 2: solidity (already computed above).
            # Feature 3: significant convex hull defects.
            hull_idx = cv2.convexHull(biggest_cs, returnPoints=False)
            n_sig_defects = 0
            if len(hull_idx) > 3:
                defects = cv2.convexityDefects(biggest_cs, hull_idx)
                if defects is not None:
                    thresh = (cont_area ** 0.5) * 5
                    n_sig_defects = sum(
                        1 for d in defects if d[0][3] > thresh)

            # Weighted vote: positive -> spade, negative -> club.
            # top_ratio is strongest signal (weight 3).
            score = 0.0
            if top_ratio < 0.45:
                score += 3.0   # strongly spade
            elif top_ratio > 0.55:
                score -= 3.0   # strongly club

            if sol > 0.96:
                score += 1.5   # lean spade
            elif sol < 0.92:
                score -= 1.5   # lean club
            elif sol > 0.94:
                score += 0.5
            elif sol < 0.94:
                score -= 0.5

            if n_sig_defects >= 3:
                score -= 1.0   # lean club
            elif n_sig_defects <= 1:
                score += 1.0   # lean spade

            return "s" if score >= 0 else "c"

        # No contours: default spade
        return "s"


def _detect_suit_at(image: np.ndarray, x: int, y: int, radius: int = 20) -> str:
    """Detect card suit by sampling color near and around the rank character.

    Samples the rank character itself AND the area below it (where suit symbol is).
    Red = hearts(h) or diamonds(d), Black = spades(s) or clubs(c).

    N8 uses red for hearts and diamonds, black for spades and clubs.
    The rank character itself is colored (red rank = red suit, black rank = black suit).
    """
    h, w = image.shape[:2]

    # Sample a larger area around the rank character
    y1 = max(0, y - 10)
    y2 = min(h, y + radius + 15)
    x1 = max(0, x - 15)
    x2 = min(w, x + 15)

    sample = image[y1:y2, x1:x2]
    if sample.size == 0:
        return "s"

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    # Red detection — generous range
    red_mask1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    red_pixels = np.sum(red_mask1 > 0) + np.sum(red_mask2 > 0)
    total_pixels = max(sample.shape[0] * sample.shape[1], 1)
    red_ratio = red_pixels / total_pixels

    if red_ratio > 0.08:
        # Red suit: distinguish heart vs diamond by suit symbol shape
        # Heart ♥ has a wider top, diamond ♦ is pointy
        # Simple heuristic: check the suit symbol area below rank
        suit_y1 = min(h, y + 10)
        suit_y2 = min(h, y + radius + 10)
        suit_x1 = max(0, x - 12)
        suit_x2 = min(w, x + 12)
        suit_area = image[suit_y1:suit_y2, suit_x1:suit_x2]
        if suit_area.size > 0:
            suit_hsv = cv2.cvtColor(suit_area, cv2.COLOR_BGR2HSV)
            red_s = cv2.inRange(suit_hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
            red_s2 = cv2.inRange(suit_hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
            red_suit_mask = cv2.bitwise_or(red_s, red_s2)
            # Find contour of suit symbol
            contours, _ = cv2.findContours(red_suit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                _, _, sw, sh = cv2.boundingRect(biggest)
                # Diamond is taller than wide, heart is wider than tall
                if sh > 0 and sw / sh < 0.85:
                    return "d"
                else:
                    return "h"
        # Default red: use green channel to decide heart vs diamond.
        # Hearts in N8 rendering have higher green component.
        red_combined = cv2.bitwise_or(red_mask1, red_mask2)
        rp = sample[red_combined > 0]
        if len(rp) >= 5 and float(np.mean(rp[:, 1])) > 60:
            return "h"
        return "d"
    else:
        # Black suit: distinguish spade vs club
        # Spade ♠ is pointy at top, club ♣ has round lobes
        # Simple: default to spade (more common)
        return "s"


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
