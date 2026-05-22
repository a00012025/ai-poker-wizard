"""Natural8 preflop row overflow is a re-action, not a ninth seat."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_ninth_preflop_row_stays_reaction_for_eightmax_position_order():
    # OCR sees nine preflop rows, but the ninth is the opener's response to an
    # all-in, not a ninth player.  Using a 9-max order shifted hero HJ to LJ.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5866594919.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-R2-F-AI22.9-F-F-F-F-F"
