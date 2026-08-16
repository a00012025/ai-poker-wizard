"""Dealer-button detection for N8 table regions."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from position_constants import BUTTON_FIRST_POSITION_ORDERS

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dealer_button.png"

SEAT_ANCHORS = {
    8: [
        (0.50, 0.94),
        (0.78, 0.82),
        (0.91, 0.50),
        (0.78, 0.18),
        (0.50, 0.08),
        (0.22, 0.18),
        (0.04, 0.94),
        (0.09, 0.50),
    ],
    7: [
        (0.50, 0.94),
        (0.82, 0.78),
        (0.91, 0.45),
        (0.68, 0.13),
        (0.32, 0.13),
        (0.07, 0.45),
        (0.18, 0.78),
    ],
    6: [
        (0.50, 0.94),
        (0.84, 0.76),
        (0.84, 0.24),
        (0.50, 0.08),
        (0.16, 0.24),
        (0.16, 0.76),
    ],
    5: [
        (0.50, 0.94),
        (0.86, 0.62),
        (0.72, 0.14),
        (0.28, 0.14),
        (0.14, 0.62),
    ],
    4: [(0.50, 0.94), (0.86, 0.50), (0.50, 0.08), (0.14, 0.50)],
    3: [(0.50, 0.94), (0.82, 0.28), (0.18, 0.28)],
    2: [(0.50, 0.94), (0.50, 0.08)],
}


def _load_template() -> np.ndarray | None:
    if not TEMPLATE_PATH.exists():
        return None
    template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if template is None or template.size == 0:
        return None
    return template


def _nearest_seat(x: float, y: float, table_shape: tuple[int, int], table_size: int) -> int:
    h, w = table_shape
    anchors = SEAT_ANCHORS.get(table_size) or SEAT_ANCHORS[8]
    best_idx = 0
    best_dist = float("inf")
    for idx, (ax, ay) in enumerate(anchors):
        dist = math.hypot(x - ax * w, y - ay * h)
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def detect_button(
    table_region: np.ndarray,
    *,
    table_size: int = 8,
    min_conf: float = 0.9,
) -> tuple[int, float] | None:
    """Return `(seat_idx, confidence)` in hero-relative table sectors."""
    template = _load_template()
    if template is None or table_region is None or table_region.size == 0:
        return None
    gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
        return None
    match = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(match)
    if max_val < min_conf:
        return None
    x = max_loc[0] + template.shape[1] / 2
    y = max_loc[1] + template.shape[0] / 2
    seat_idx = _nearest_seat(x, y, gray.shape[:2], table_size)
    return seat_idx, float(max_val)


def hero_position_from_button(
    button_seat_idx: int,
    *,
    table_size: int,
    hero_seat_idx: int = 0,
) -> str | None:
    order = BUTTON_FIRST_POSITION_ORDERS.get(table_size)
    if not order:
        return None
    offset = (hero_seat_idx - button_seat_idx) % len(order)
    return order[offset]
