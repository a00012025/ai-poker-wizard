"""Hero-card rank repairs for recurrent Natural8 classifier confusions."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.n8_parser import parse_n8_screenshot  # noqa: E402


def test_top2_repairs_queen_to_six_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5864260800.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"6s", "5c"}


def test_top2_repairs_five_to_six_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5900728939.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"7c", "6s"}


def test_top2_repairs_king_to_ten_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5864409893.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"Td", "7c"}


def test_top2_repairs_queen_to_king_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5887576003.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"Kc", "6c"}


def test_top2_keeps_high_margin_queen_over_king():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5920382283.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"Qd", "Js"}


def test_top2_repairs_five_to_nine_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5920326093.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"Qc", "9h"}


def test_top2_repairs_spade_to_club_suit_tie_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5874042561.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"8c", "4c"}


def test_top2_repairs_heart_to_diamond_suit_tie_without_range_guessing():
    result = parse_n8_screenshot(Path("data/hand_images/img/TM5920381830.png").read_bytes())

    assert result["hand"] is not None
    assert set(result["hand"]["hero_hand"][i:i + 2] for i in (0, 2)) == {"Jd", "9h"}
