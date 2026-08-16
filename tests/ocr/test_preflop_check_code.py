"""Preflop BB option is encoded as X, not C."""
from __future__ import annotations

from pathlib import Path

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_preflop_check_uses_x_action_code():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5867250087.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["preflop_actions"] == "F-C-F-F-C-X"


def test_anonymous_duplicate_check_before_bb_call_is_dropped():
    # This row has both an anonymous Check fragment and an explicit BB Call at
    # the blind option. The Check is OCR duplication, not an X-C action pair.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5920751539.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_position"] == "SB"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-R2-F-F-C"
