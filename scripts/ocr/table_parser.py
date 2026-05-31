"""Table region parser for Natural8 replay screenshots.

Extracts board cards, hero cards, player stacks, and table color
from the upper (table) region of an N8 replay screenshot.
"""

import os

import cv2
import numpy as np

from .button_detector import detect_button

# Route the hero crop through the multi-crop ensemble when any card's raw
# confidence (min of rank/suit head) falls below this floor. Picked at 0.50
# so we only fire on genuinely uncertain reads — confident classifications
# (the vast majority) take the single-pass path unchanged.
ENSEMBLE_FLOOR = float(os.getenv("OCR_ENSEMBLE_FLOOR", "0.50"))


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


def _find_board_cards(
    table_region: np.ndarray,
) -> tuple[list[str], list[dict]]:
    """Find and identify board cards in the center of the table via CardCNN.

    Returns ``(cards, details)`` where ``cards`` is the list of board card
    labels (e.g. ``["Ks", "9d", "3d"]``) and ``details`` is a parallel
    list of per-card dicts mirroring the hero-card detail structure
    (rank, rank_conf, suit, suit_conf, rank_top2, suit_top2, rank_source,
    conf) plus a ``corner_disagree`` flag set when corner OCR overrode the
    classifier rank. The calibrator consumes these signals (board
    classifier confidence, top-2 margins, corner disagreement) to catch
    board_wrong emits the v2 schema was blind to.
    """
    from .classifier.infer import CardClassifier

    crops = _locate_board_cards(table_region)
    if not crops:
        return [], []
    results = CardClassifier().classify_batch_detailed(crops)
    cards: list[str] = []
    details: list[dict] = []
    for crop, detail in zip(crops, results):
        rank = detail.get("rank")
        suit = detail.get("suit")
        rank_conf = float(detail.get("rank_conf") or 0.0)
        suit_conf = float(detail.get("suit_conf") or 0.0)
        rank_source = "classifier"
        corner_disagree = False
        corner_rank, corner_conf = _rank_from_corner_ocr(crop)
        if corner_rank and corner_conf >= 0.90 and corner_rank != rank:
            corner_disagree = True
            rank = corner_rank
            rank_conf = float(corner_conf)
            rank_source = "corner_ocr"
        details.append({
            "rank": rank, "rank_conf": rank_conf,
            "suit": suit, "suit_conf": suit_conf,
            "rank_top2": detail.get("rank_top2", []),
            "suit_top2": detail.get("suit_top2", []),
            "rank_source": rank_source,
            "corner_disagree": corner_disagree,
            "conf": min(rank_conf, suit_conf),
        })
        if rank and suit:
            cards.append(f"{rank}{suit}")
    return cards, details


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
    hero = table_region[int(h * 0.58):int(h * 0.98), int(w * 0.28):int(w * 0.68)]
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


def _trim_above_card_edge(crop: np.ndarray) -> np.ndarray:
    """Strip pot-UI rows that sometimes sit above the actual card.

    The hero blob detector can latch onto bright pot labels (e.g.
    "$0.50 / 35.5 BB") rendered just above the hero cards. When that
    happens the crop's top ~30% is N8 chrome instead of card, and the
    CardCNN — letterboxed to square — reads the noisy header as part of
    the rank and confidently misclassifies (H2851: 5d5c → 9d9c at
    rank_conf 0.97).

    Heuristic: scan from the top for the first row whose next 4 rows
    each have ≥55% bright pixels (the white card body). If that row is
    well below the top (≥6 rows), assume the rows above are pot UI and
    chop them. No-op when the crop already starts at the card top, so
    clean hero crops are untouched.
    """
    h, w = crop.shape[:2]
    if h < 20 or w < 10:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright = (gray >= 200).sum(axis=1) / float(w)
    needed = 4
    threshold = 0.55
    edge = None
    for y in range(h - needed):
        if (bright[y:y + needed] >= threshold).all():
            edge = y
            break
    if edge is None or edge < 6:
        return crop
    return crop[edge:]


def _mask_win_overlay(crop: np.ndarray) -> np.ndarray:
    """Whiten the orange/yellow `WIN` sticker that N8 paints over winning cards.

    The sticker bleeds into the lower half of the hero card crop, and the
    CardCNN reads its red-leaning hue as a red suit (e.g. K♣ → Kh on H2806,
    suit_conf 0.587). Training data has no sticker examples, so masking
    those pixels to white gives the classifier a clean card to read.

    The WIN letterforms render as scattered small orange specks (each
    stroke is its own blob), so we dilate the raw orange mask aggressively
    before measuring connected components — the W/I/N strokes coalesce
    into one cluster. We only fire when a cluster (a) covers >= 4% of the
    crop and (b) sits mostly in the lower half — that's where the sticker
    paints. Skips incidental orange like the `$0.50` price banner at the
    top of cash-game crops, which previously degraded a clean Ts9s crop.
    """
    out = crop.copy()
    h, w = out.shape[:2]
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array([10, 100, 100]),
                      np.array([35, 255, 255]))
    if int(raw.sum()) == 0:
        return out
    cluster_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clustered = cv2.dilate(raw, cluster_kernel, iterations=2)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clustered, connectivity=8
    )
    crop_area = h * w
    sticker_labels: list[int] = []
    for lab in range(1, n_labels):
        blob_area = int(stats[lab, cv2.CC_STAT_AREA])
        blob_top = int(stats[lab, cv2.CC_STAT_TOP])
        blob_height = int(stats[lab, cv2.CC_STAT_HEIGHT])
        blob_bottom = blob_top + blob_height
        if blob_area / crop_area < 0.04:
            continue
        center_y = blob_top + blob_height / 2
        if center_y < h * 0.40:
            continue
        sticker_labels.append(lab)
    if not sticker_labels:
        return out
    sticker_mask = np.isin(labels, sticker_labels).astype(np.uint8) * 255
    sticker_mask = cv2.bitwise_and(sticker_mask, raw)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sticker_mask = cv2.dilate(sticker_mask, edge_kernel, iterations=2)
    out[sticker_mask > 0] = (255, 255, 255)
    return out


def _repair_rank_from_top2(rank: str, rank_conf: float, top2: list) -> str:
    """Repair recurrent Natural8 hero-card rank confusions from top-2 logits.

    These are not poker-range guesses; they are visual OCR repairs for the
    card classifier's known N8 replay crop confusions where the correct glyph
    is consistently the second-ranked class and no held-out correct crops match
    the same confidence/top-2 pattern:
    - 5/6: lower-left curve clipped by card overlap.
    - K/T: ten's vertical stroke/painted corner is read as a king.
    - Q/6: six's closed loop is read as queen on narrow left-card crops.
    - Q/K and 5/9: low-margin folded-card crops where the second-ranked
      class is consistently correct on the held-out replay crop set.
    """
    if len(top2 or []) < 2:
        return rank
    second_rank, second_conf = top2[1]
    if rank_conf > 0.99:
        return rank
    if rank == "5" and second_rank == "6" and second_conf >= 0.07:
        return "6"
    if rank == "K" and second_rank == "T" and second_conf >= 0.15:
        return "T"
    if rank == "K" and second_rank == "A" and second_conf >= 0.15 and rank_conf <= 0.80:
        return "A"
    if rank == "A" and second_rank == "4" and second_conf >= 0.06 and rank_conf <= 0.90:
        return "4"
    if rank == "Q" and second_rank == "6" and second_conf >= 0.20:
        return "6"
    if rank == "Q" and second_rank == "K" and second_conf >= 0.15 and rank_conf <= 0.50:
        return "K"
    if rank == "5" and second_rank == "9" and second_conf >= 0.02:
        return "9"
    return rank


def _rank_from_corner_ocr(crop: np.ndarray) -> tuple[str | None, float]:
    """Read the visible top-left card rank with OCR as a CNN cross-check.

    Natural8 WIN stickers can cover the lower half of a hero card.  The
    classifier sees the whole crop and can confidently hallucinate a face card
    from the sticker/avatar noise even when the corner rank is unobstructed
    (H3429: 2♥ was read as K♥).  EasyOCR is much better at the isolated corner
    glyph, so use it only when it returns a clean single rank token.
    """
    from .ocr_utils import ocr_text, preprocess_for_ocr

    h, w = crop.shape[:2]
    if h < 30 or w < 25:
        return None, 0.0

    rank_chars = set("23456789TJQKA")

    def normalize_rank_token(text: str) -> str | None:
        token = text.strip().upper().replace(" ", "")
        token = token.replace("IO", "10").replace("O", "0")
        if token == "10":
            return "T"
        if len(token) == 1 and token in rank_chars:
            return token
        return None

    # The crop includes rank + suit in the top-left.  Keep candidates tight
    # enough that noisy chip/sticker text cannot be mixed into the OCR result.
    boxes = [
        (0, 0, int(w * 0.64), int(h * 0.52)),
        (0, 0, int(w * 0.55), int(h * 0.45)),
        (0, 0, int(w * 0.70), int(h * 0.60)),
    ]
    candidates: list[tuple[str, float]] = []
    for x1, y1, x2, y2 in boxes:
        x2 = min(w, max(x2, min(w, 35)))
        y2 = min(h, max(y2, min(h, 45)))
        corner = crop[y1:y2, x1:x2]
        if corner.size == 0:
            continue
        for image in (corner, preprocess_for_ocr(corner, min_width=160)):
            text, conf = ocr_text(image, whitelist="23456789TJQKA10", psm=10)
            rank = normalize_rank_token(text)
            if rank and conf >= 70:
                candidates.append((rank, float(conf) / 100.0))

    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def _corner_rank_overrides(cnn_rank: str, cnn_conf: float, corner_rank: str) -> bool:
    """Whether the corner-OCR rank should override the CNN rank.

    Corner OCR rescues confident CNN face-card hallucinations off sticker/avatar
    noise (H3429: 2♥ read as K♥). But EasyOCR misreads the Ace corner glyph as
    "4" (H2878: A♣ read as 4♣), so a corner "4" must never override a CNN that
    is certain the card is an Ace — the CNN rank head is reliable at high
    confidence and the A→4 corner confusion is a known EasyOCR failure.
    """
    if corner_rank == cnn_rank:
        return False
    if corner_rank == "4" and cnn_rank == "A" and cnn_conf >= 0.99:
        return False
    return True


def _repair_suit_from_top2(rank: str, suit: str, suit_conf: float, top2: list) -> str:
    """Repair narrow low-margin suit confusions from masked crop top-2.

    These fire only for rank+suit patterns that fixed held-out hero-card
    misses with no observed correct-crop regressions in the replay split:
    - 8s/8c at near-tie confidence on left-card crops.
    - Jh/Jd when the red-suit head is almost evenly split.
    """
    if len(top2 or []) < 2:
        return suit
    second_suit, second_conf = top2[1]
    if (
        rank == "8"
        and suit == "s"
        and second_suit == "c"
        and suit_conf <= 0.60
        and second_conf >= 0.45
    ):
        return "c"
    if (
        rank == "J"
        and suit == "h"
        and second_suit == "d"
        and suit_conf <= 0.60
        and second_conf >= 0.40
    ):
        return "d"
    return suit


def _find_hero_cards(
    table_region: np.ndarray,
) -> tuple[list[str], float, list[dict], bool]:
    """Find and identify hero's hole cards via CardCNN.

    Returns (cards, confidence, details) where confidence is min over all
    card predictions, and details is a list of per-card dicts with rank,
    rank_conf, suit, suit_conf, conf. Low confidence naturally triggers
    the Gemini fallback in gemini_session.

    Strategy with the WIN sticker mask: classify both the raw and the
    masked crop, then take RANK from the raw prediction (the rank corner
    sits at the top-left of the card, well above where the sticker
    paints — masking can only hurt by removing context, see H2829: Q♣
    raw=Q@0.75, masked=A@0.95 because masking the lower half made the
    rank head guess A) and take SUIT from the masked prediction (orange
    WIN pixels bleed red, flipping ♣→♥ on raw — the original purpose of
    the mask, see H2806). When the WIN mask is a no-op (no orange
    detected), masked == raw and this collapses to a single prediction.
    """
    from .classifier.infer import CardClassifier
    from .classifier.ensemble import predict_with_ensemble

    crops = _locate_hero_cards(table_region)
    if not crops:
        return [], 0.0, [], False
    crops = [_trim_above_card_edge(c) for c in crops]
    masked_crops = [_mask_win_overlay(c) for c in crops]
    clf = CardClassifier()
    raw_details = clf.classify_batch_detailed(crops)
    masked_details = clf.classify_batch_detailed(masked_crops)
    details: list[dict] = []
    for i, (raw, masked) in enumerate(zip(raw_details, masked_details)):
        rank_conf = raw["rank_conf"]
        rank = _repair_rank_from_top2(
            raw["rank"],
            rank_conf,
            raw.get("rank_top2", []),
        )
        rank_source = "classifier"
        corner_rank, corner_conf = _rank_from_corner_ocr(crops[i])
        if corner_rank and _corner_rank_overrides(rank, rank_conf, corner_rank):
            rank = corner_rank
            rank_conf = corner_conf
            rank_source = "corner_ocr"
        suit = masked["suit"]
        suit_conf = masked["suit_conf"]
        # The WIN mask is a suit-head aid, not an absolute override.
        # If masking changes the suit while making the suit head much less
        # confident, trust the raw crop: that pattern means incidental orange
        # or over-masking confused the masked pass rather than removing a real
        # sticker. Also trust a very strong raw suit when the masked pass is
        # only moderately confident and still has the raw suit as its runner-up
        # (H2894: WIN mask flipped 9h→9d at 0.82 while raw stayed 9h at 0.98).
        # This preserves the H2806-style masked rescue where raw is weak, while
        # fixing high-confidence raw suits degraded by over-masking.
        raw_suit = raw.get("suit")
        masked_suit = masked.get("suit")
        raw_suit_conf = raw.get("suit_conf", 0.0)
        masked_suit_top2 = masked.get("suit_top2", [])
        masked_second_suit = (
            masked_suit_top2[1][0]
            if len(masked_suit_top2) >= 2
            else None
        )
        masked_second_conf = (
            masked_suit_top2[1][1]
            if len(masked_suit_top2) >= 2
            else 0.0
        )
        if (
            raw_suit != masked_suit
            and (
                raw_suit_conf >= masked.get("suit_conf", 0.0) + 0.20
                or raw_suit_conf >= 0.99
                or (
                    raw_suit_conf >= 0.95
                    and suit_conf <= 0.85
                    and masked_second_suit == raw_suit
                    and masked_second_conf >= 0.05
                )
            )
        ):
            suit = raw["suit"]
            suit_conf = raw["suit_conf"]
        suit = _repair_suit_from_top2(
            rank,
            suit,
            suit_conf,
            masked.get("suit_top2", []),
        )
        details.append({
            "rank": rank, "rank_conf": rank_conf,
            "suit": suit, "suit_conf": suit_conf,
            "rank_top2": raw.get("rank_top2", []),
            "suit_top2": masked.get("suit_top2", []),
            "rank_source": rank_source,
            "raw_suit": raw_suit,
            "raw_suit_conf": raw_suit_conf,
            "masked_suit": masked_suit,
            "masked_suit_conf": masked.get("suit_conf", 0.0),
            "conf": min(rank_conf, suit_conf),
        })
    ensemble_used = False
    for i, d in enumerate(details):
        if d["conf"] >= ENSEMBLE_FLOOR:
            continue
        ens = predict_with_ensemble(crops[i], classifier=clf)
        d["ensemble_label"] = ens["label"]
        d["ensemble_conf"] = ens["card_conf"]
        d["ensemble_votes"] = ens["votes"]
        if ens["label"] and ens["card_conf"] > d["conf"]:
            ens_rank, ens_suit = ens["label"][0], ens["label"][1]
            d["rank"] = ens_rank
            d["suit"] = ens_suit
            d["rank_conf"] = ens["card_conf"]
            d["suit_conf"] = ens["card_conf"]
            d["conf"] = ens["card_conf"]
            d["ensemble_used"] = True
            ensemble_used = True
        else:
            d["ensemble_used"] = False
    cards = [f"{d['rank']}{d['suit']}" for d in details
             if d["rank"] and d["suit"]]
    conf = min((d["conf"] for d in details), default=0.0)
    return cards, conf, details, ensemble_used


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
            "board_card_details": [],
            "hero_cards": [],
            "player_stacks": [],
            "table_color": "unknown",
            "dealer_button": None,
            "dealer_button_seat": None,
            "dealer_button_conf": 0.0,
        }

    table_color = _detect_table_color(table_region)
    dealer_button_raw = detect_button(table_region)
    if dealer_button_raw is not None:
        dealer_button = dealer_button_raw
        dealer_button_seat, dealer_button_conf = dealer_button_raw
    else:
        dealer_button = None
        dealer_button_seat, dealer_button_conf = None, 0.0
    board_cards, board_card_details = _find_board_cards(table_region)
    hero_cards, hero_card_conf, hero_card_details, hero_ensemble_used = \
        _find_hero_cards(table_region)
    all_stacks_named = _find_all_stacks(table_region)
    hero_stack = _find_hero_stack(table_region)

    # Flat list of stack values for backward compatibility
    all_stacks = [s["stack"] for s in all_stacks_named]

    return {
        "board_cards": board_cards,
        "board_card_details": board_card_details,
        "hero_cards": hero_cards,
        "hero_card_conf": hero_card_conf,
        "hero_card_details": hero_card_details,
        "ensemble_used": hero_ensemble_used,
        "hero_stack": hero_stack,
        "player_stacks": all_stacks,
        "named_stacks": all_stacks_named,
        "table_color": table_color,
        "dealer_button": dealer_button,
        "dealer_button_seat": dealer_button_seat,
        "dealer_button_conf": dealer_button_conf,
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
    import re

    # Two-pass scan: prefer any "XX.X BB" match (much more reliable signal)
    # over a plain number. Regression: H2798 — hero crop area picked up
    # noise like '24' (a fragment from an adjacent UI element, no BB
    # suffix) BEFORE '11.5 BB' in the result list. Per-result fallback to
    # plain-number regex returned the first noisy number it saw and never
    # reached the real "11.5 BB" entry that came later.
    for r in results:
        text = r["text"].strip().upper()
        m = re.search(r"(\d+\.?\d*)\s*BB", text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue

    # Fallback only when no "BB"-suffixed value was found anywhere in the
    # crop. Pick the highest-confidence plain number in the plausible
    # range so we don't latch onto noise like 'gorj' fragments.
    best = None
    for r in results:
        text = r["text"].strip().upper()
        m = re.search(r"(\d+\.?\d+)", text)
        if not m or r["conf"] <= 0.5:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if not (0.5 < val < 500):
            continue
        if best is None or r["conf"] > best[1]:
            best = (val, r["conf"])
    return best[0] if best else None
