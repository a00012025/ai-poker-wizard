"""Hero-card localization includes the full bottom card area."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_dim_folded_bottom_hero_cards_are_not_cropped_off():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5846885467.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i+2] for i in range(0, 4, 2)) == {"Kc", "2d"}
