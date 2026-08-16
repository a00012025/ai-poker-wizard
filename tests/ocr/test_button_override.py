"""Dealer-button blobs must not override action-panel hero seats."""
from __future__ import annotations

from pathlib import Path

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_high_confidence_button_detection_does_not_shift_hero_to_bb():
    # This screenshot has a high-confidence dealer-button blob, but the fixed
    # table-seat anchor maps it to BB.  The preflop action panel's ordered hero
    # row is CO and should remain authoritative.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5888079289.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "CO"
    assert result["hand"]["preflop_actions"] == "R2-F-F-C-F-F-C-C"
