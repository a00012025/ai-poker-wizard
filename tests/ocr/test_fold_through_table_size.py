"""Fold-through preflop columns keep the unseen BB seat for positions."""
from __future__ import annotations

from pathlib import Path

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_fold_through_keeps_full_table_size_without_phantom_bb_action():
    # Everyone before the big blind folds, so Natural8 shows seven fold rows
    # at an 8-max table and no BB decision row.  Counting visible rows as the
    # table size shifted HJ to CO and made hand_exact fail even though the
    # action string itself was correct.
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5874599522.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-F-F"
