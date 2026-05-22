"""Do not add turn/river cards when the panel has no street entries."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_empty_turn_river_columns_do_not_create_phantom_board_cards():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5863485128.png").read_bytes())

    assert result["hand"] is not None
    assert result["hand"]["streets"]
    assert [list(street.keys())[0] for street in result["hand"]["streets"]] == ["board"]
    assert set(result["hand"]["streets"][0]["board"][i:i+2] for i in range(0, 6, 2)) == {"Qh", "Th", "5s"}
