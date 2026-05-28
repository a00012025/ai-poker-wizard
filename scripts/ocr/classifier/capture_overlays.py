"""Capture a WIN-sticker overlay from a hero-card crop as an RGBA template.

We isolate the yellow/orange overlay pixels (same HSV band as
table_parser._mask_win_overlay), dilate the strokes into one cluster,
and emit RGBA where:
  - opaque (alpha 255) = overlay pixel
  - transparent (alpha 0) = card background

The resulting template can be alpha-composited onto any clean card crop
to synthesise a realistic WIN-style training example.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..region_detector import detect_regions
from ..table_parser import _locate_hero_cards


def extract_overlay(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    regions = detect_regions(img)
    if not regions:
        return None
    table = regions.get("table")
    if table is None:
        return None
    crops = _locate_hero_cards(table)
    if len(crops) != 2:
        return None
    # Concatenate the two cards side-by-side to capture the full WIN sticker
    h = min(c.shape[0] for c in crops)
    a = cv2.resize(crops[0], (crops[0].shape[1], h))
    b = cv2.resize(crops[1], (crops[1].shape[1], h))
    pair = np.hstack([a, b])
    hsv = cv2.cvtColor(pair, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([35, 255, 255]))
    if int(raw.sum()) == 0:
        return None
    cluster = cv2.dilate(
        raw, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cluster, connectivity=8)
    crop_h, crop_w = pair.shape[:2]
    keep = []
    for lab in range(1, n_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        top = int(stats[lab, cv2.CC_STAT_TOP])
        height = int(stats[lab, cv2.CC_STAT_HEIGHT])
        if area / (crop_h * crop_w) < 0.02:
            continue
        if top + height / 2 < crop_h * 0.30:  # ignore chip-stack at top
            continue
        keep.append(lab)
    if not keep:
        return None
    mask = np.isin(labels, keep).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, raw)
    rgba = np.dstack([pair, mask])  # BGRA
    return rgba


def save_overlay(rgba: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rgba)
