"""N8 preflop hero-marker cleanup avoids adjacent opponent bleed."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import _filter_action_entries  # noqa: E402


def test_named_hero_markers_become_opponents_when_anonymous_hero_exists():
    entries = [
        {"type": "opponent", "position": "UTG", "action": "Fold", "size": None},
        {
            "type": "hero",
            "position": "UTG+1",
            "action": "Fold",
            "size": None,
            "player_name": "adjacent villain",
        },
        {"type": "hero", "position": None, "action": "Fold", "size": None},
    ]

    filtered = _filter_action_entries(entries)

    assert filtered[1]["type"] == "opponent"
    assert filtered[2]["type"] == "hero"


def test_named_non_fold_hero_bleed_does_not_hide_anonymous_real_hero_action():
    entries = [
        {
            "type": "hero",
            "position": "UTG",
            "action": "Raise",
            "size": 2.0,
            "player_name": "false yellow villain",
        },
        {"type": "opponent", "position": "BTN", "action": "Fold", "size": None},
        {"type": "hero", "position": "BB", "action": "Call", "size": 1.0},
    ]

    filtered = _filter_action_entries(entries)

    assert filtered[0]["type"] == "opponent"
    assert filtered[2]["type"] == "hero"
