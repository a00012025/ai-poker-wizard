#!/usr/bin/env python3
"""Generate card rank and suit templates from N8 replay screenshots.

Reads screenshots from ~/n8_image/, detects table region, finds board cards
in the center area, and crops rank/suit regions for template matching.

Usage:
    python scripts/ocr/generate_templates.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.region_detector import detect_regions

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
IMAGE_DIR = Path.home() / "n8_image"


def find_board_cards(table: np.ndarray) -> list[dict]:
    """Find board card images in the center of the table region.

    Strategy: find the wide bright blob containing all cards, then split
    into individual cards by detecting vertical dark gaps between them.

    Returns list of dicts with keys: x, y, w, h, image (cropped card),
    sorted left to right.
    """
    h, w = table.shape[:2]

    # Board cards are in center area of table
    y_start = int(h * 0.15)
    y_end = int(h * 0.55)
    x_start = int(w * 0.15)
    x_end = int(w * 0.85)

    center = table[y_start:y_end, x_start:x_end]
    ch, cw = center.shape[:2]

    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the card row blob — the widest contour spanning >30% of center width
    best = None
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw > cw * 0.3 and bh > 30:
            if best is None or bw > best[2]:
                best = (x, y, bw, bh)

    if best is None:
        return []

    bx, by, bw, bh = best
    blob = center[by:by + bh, bx:bx + bw]
    gray_blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)

    # Find vertical gaps between cards using column-wise mean brightness
    col_means = np.mean(gray_blob, axis=0)
    dark_cols = col_means < 100

    # Find bright runs (card segments)
    in_card = False
    card_start = 0
    card_segments = []
    for i in range(len(dark_cols)):
        if not dark_cols[i] and not in_card:
            card_start = i
            in_card = True
        elif dark_cols[i] and in_card:
            if i - card_start > 20:
                card_segments.append((card_start, i))
            in_card = False
    if in_card and len(dark_cols) - card_start > 20:
        card_segments.append((card_start, len(dark_cols)))

    cards = []
    for cs, ce in card_segments:
        card_img = blob[:, cs:ce]

        # Trim top/bottom dark rows
        row_means = np.mean(cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY), axis=1)
        bright_rows = np.where(row_means > 100)[0]
        if len(bright_rows) > 0:
            top = max(0, bright_rows[0] - 2)
            bot = min(card_img.shape[0], bright_rows[-1] + 3)
            card_img = card_img[top:bot, :]

        abs_x = bx + cs + x_start
        abs_y = by + (top if len(bright_rows) > 0 else 0) + y_start
        cards.append({
            "x": abs_x,
            "y": abs_y,
            "w": card_img.shape[1],
            "h": card_img.shape[0],
            "image": card_img,
        })

    return cards


def crop_rank_region(card: np.ndarray) -> np.ndarray:
    """Crop the rank region from a card image (top ~35%, left ~50%)."""
    h, w = card.shape[:2]
    return card[0:int(h * 0.35), 0:int(w * 0.50)]


def crop_suit_region(card: np.ndarray) -> np.ndarray:
    """Crop the suit region from a card image (~30-65% height, left ~55%)."""
    h, w = card.shape[:2]
    return card[int(h * 0.30):int(h * 0.65), 0:int(w * 0.55)]


def main():
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = TEMPLATE_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.jpeg")) + sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return

    card_index = 0
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Could not read image")
            continue

        regions = detect_regions(image)
        if regions is None:
            print(f"  Could not detect regions")
            continue

        table = regions["table"]
        cards = find_board_cards(table)
        print(f"  Found {len(cards)} board cards")

        for i, card in enumerate(cards):
            # Save full card
            card_path = raw_dir / f"card_{card_index:03d}_{img_path.stem}_pos{i}.png"
            cv2.imwrite(str(card_path), card["image"])

            # Save rank crop
            rank_img = crop_rank_region(card["image"])
            rank_path = raw_dir / f"rank_{card_index:03d}_{img_path.stem}_pos{i}.png"
            cv2.imwrite(str(rank_path), rank_img)

            # Save suit crop
            suit_img = crop_suit_region(card["image"])
            suit_path = raw_dir / f"suit_{card_index:03d}_{img_path.stem}_pos{i}.png"
            cv2.imwrite(str(suit_path), suit_img)

            print(f"  Card {i}: {card['w']}x{card['h']} at ({card['x']},{card['y']})")
            card_index += 1

    print(f"\nTotal cards extracted: {card_index}")
    print(f"Raw crops saved to: {raw_dir}")


if __name__ == "__main__":
    main()
