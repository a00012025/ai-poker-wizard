"""Hero WIN overlay masking must not flip strong raw heart suits."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_h2894_second_card_stays_heart_after_win_mask():
    result = parse_n8_screenshot(
        Path("tests/fixtures/h2894_th9h_win_mask.png").read_bytes()
    )

    assert result["hand"] is not None
    assert result["hand"]["hero_hand"] == "Th9h"
    assert result["confidence"] >= 0.95
