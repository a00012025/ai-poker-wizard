"""Preflop BB option is encoded as X, not C."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_preflop_check_uses_x_action_code():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5867250087.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["preflop_actions"] == "F-C-F-F-C-X"
