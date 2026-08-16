"""Preflop all-in rows must not collapse away later seats."""
from __future__ import annotations

from pathlib import Path

import cv2

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


def test_duplicate_hero_allin_overlay_after_call_is_dropped():
    # The final anonymous hero All-In is just the red all-in overlay resolving
    # the opponent shove/call, not a new hero action. Keeping it used to append
    # a phantom AI12.83 and fail an otherwise exact preflop action sequence.
    from ocr.n8_parser import parse_n8_screenshot

    result = parse_n8_screenshot(Path("data/hand_images/img/TM5866698800.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-F-F-R2.2-C-F-F-AI12.83-AI37.82-C"


def test_duplicate_allin_overlay_does_not_hide_real_hero_fold():
    # The anonymous hero All-In overlay is false; the earlier anonymous hero
    # Fold row is the real local-player decision. Dropping the overlay before
    # fold-vs-nonfold hero cleanup keeps the hand parseable and exact.
    from ocr.n8_parser import parse_n8_screenshot

    result = parse_n8_screenshot(Path("data/hand_images/img/TM5846885345.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "LJ"
    assert result["hand"]["preflop_actions"] == "R1-F-F-F-C-F-F-AI12-C-F"


def test_true_short_allin_chain_is_not_dropped_as_duplicate_overlay():
    # Guardrail: this exact held-out hand legitimately has two All-In action
    # types in a 4-player chain. The duplicate-overlay filter must not remove
    # the anonymous hero all-in when there is no preceding caller/earlier hero
    # action proving it is a resolution sticker.
    from ocr.n8_parser import parse_n8_screenshot

    result = parse_n8_screenshot(Path("data/hand_images/img/TM5901336584.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "SB"
    assert result["hand"]["preflop_actions"] == "F-AI13.58-AI13.58-F"


def test_duplicate_sizeless_hero_allin_stickers_are_dropped():
    # N8 can paint repeated anonymous hero-colored All-In stickers after the
    # real hero shove. They are resolution overlays, not extra hero actions.
    from ocr.n8_parser import parse_n8_screenshot

    result = parse_n8_screenshot(Path("data/hand_images/img/TM5920473286.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "BTN"
    assert result["hand"]["preflop_actions"] == "F-C-F-AI-F-C-C"
