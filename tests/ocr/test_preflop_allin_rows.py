"""Preflop all-in rows must not collapse away later seats."""
from __future__ import annotations

from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.region_detector import detect_regions  # noqa: E402
from ocr.panel_parser import parse_panel  # noqa: E402


def test_preflop_allin_keeps_later_fold_rows():
    img = cv2.imread("data/hand_images/img/TM5863941940.png")
    panel = parse_panel(detect_regions(img)["panel"])
    preflop = next(col for col in panel["columns"] if col["name"] == "Pre-Flop")
    actions = [entry["action"] for entry in preflop["entries"]]

    assert "All-In" in actions
    assert len(preflop["entries"]) >= 8
    # Regression: the postflop all-in attribution collapse used to keep
    # only the first fold, the shove, and a synthetic responder.
    assert actions.count("Fold") >= 6
