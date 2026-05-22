"""Preflop re-action rows should not become extra seats."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_all_fold_column_ignores_false_reaction_signal_for_unseen_bb():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5878838656.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["players_at_table"] == 8
    assert result["hand"]["hero_position"] == "HJ"
    assert result["hand"]["preflop_actions"] == "F-F-F-F-F-F-F"
