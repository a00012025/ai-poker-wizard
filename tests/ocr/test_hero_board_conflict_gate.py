"""Preflop false board detections must not rewrite correct hero cards."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_preflop_only_false_board_conflict_keeps_hero_cards():
    # Board detector sees a false 4h in the table region on this preflop
    # screenshot, while CardCNN correctly reads hero as 5h4h. The conflict
    # resolver used to swap the 4h to its top-2 rank (2h), corrupting a
    # correct hand even though no postflop street exists.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5846885329.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["hero_hand"] == "5h4h"
