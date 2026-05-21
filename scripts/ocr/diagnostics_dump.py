"""Per-hand visual diagnostics for OCR precision triage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .button_detector import SEAT_ANCHORS, detect_button
from .panel_parser import detect_entries, split_columns
from .region_detector import detect_regions


def _imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _draw_button_overlay(table_view: np.ndarray) -> None:
    button = detect_button(table_view)
    h, w = table_view.shape[:2]
    for idx, (ax, ay) in enumerate(SEAT_ANCHORS[8]):
        x = int(ax * w)
        y = int(ay * h)
        cv2.circle(table_view, (x, y), 8, (80, 80, 80), 1)
        cv2.putText(
            table_view,
            str(idx),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
    if button is None:
        cv2.putText(
            table_view,
            "button not detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return

    seat, conf = button
    ax, ay = SEAT_ANCHORS[8][seat]
    x = int(ax * w)
    y = int(ay * h)
    cv2.circle(table_view, (x, y), 18, (0, 255, 0), 2)
    cv2.putText(
        table_view,
        f"button seat={seat} conf={conf:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def _draw_entries_overlay(col_view: np.ndarray, entries: list[dict], label: str) -> None:
    cv2.putText(
        col_view,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    y = 50
    for entry in entries:
        text = " ".join(
            str(part)
            for part in (
                entry.get("type"),
                entry.get("position") or "-",
                entry.get("action") or "-",
                entry.get("size") if entry.get("size") is not None else "",
            )
            if part != ""
        )
        cv2.putText(
            col_view,
            text[:42],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )
        cv2.line(col_view, (0, y + 6), (col_view.shape[1], y + 6), (0, 120, 120), 1)
        y += 22


def dump_hand(image_bytes: bytes, *, out_dir: Path, hand_id: str) -> None:
    """Write original/table/panel diagnostic images for one hand."""
    out_dir = Path(out_dir)
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image bytes for {hand_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _imwrite(out_dir / "original.png", img)

    regions = detect_regions(img)
    if regions is None:
        raise ValueError(f"could not detect N8 regions for {hand_id}")

    table = regions.get("table")
    panel = regions.get("panel")

    if table is not None:
        table_view = table.copy()
        _draw_button_overlay(table_view)
        _imwrite(out_dir / "table_with_button.png", table_view)

    if panel is not None:
        for i, col in enumerate(split_columns(panel)):
            col_img = col["region"].copy()
            entries, pre_collapse_count = detect_entries(
                col_img,
                is_preflop=(col["name"] == "Pre-Flop"),
            )
            label = f"{col['name']} entries={len(entries)} pre={pre_collapse_count}"
            _draw_entries_overlay(col_img, entries, label)
            safe_name = col["name"].lower().replace("-", "_").replace(" ", "_")
            _imwrite(out_dir / f"col_{i}_{safe_name}.png", col_img)
