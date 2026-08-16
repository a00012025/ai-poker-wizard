"""Phase 11.A.2 — accept a snapshot into the corpus only when a fresh
Gemini reparse agrees with the stored OCR parse on hero_hand AND board
AND hero_position. Disagreement on ANY of the three rejects the record."""
from __future__ import annotations

from pathlib import Path

import pytest


from ocr.classifier.gemini_agreement_backfill import (
    _key_fields,
    backfill_one,
    is_agreement,
)


def _hand(hero, pos, flop):
    return {
        "hero_hand": hero,
        "hero_position": pos,
        "streets": [{"board": flop, "actions": []}],
    }


def test_key_fields_extracts_hero_board_position():
    assert _key_fields(_hand("AsKc", "BTN", "2d6cAd")) == ("AsKc", "2d6cAd", "BTN")


def test_key_fields_handles_missing_streets():
    assert _key_fields({"hero_hand": "AsKc", "hero_position": "BTN"}) == \
        ("AsKc", None, "BTN")


def test_is_agreement_all_three_match():
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6d5d", "BB", "2d6cAd")
    assert is_agreement(ocr, gemini) is True


def test_is_agreement_rejects_hero_disagreement():
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6h5d", "BB", "2d6cAd")
    assert is_agreement(ocr, gemini) is False


def test_is_agreement_rejects_board_disagreement():
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6d5d", "BB", "2d6cAh")
    assert is_agreement(ocr, gemini) is False


def test_is_agreement_rejects_position_disagreement():
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6d5d", "SB", "2d6cAd")
    assert is_agreement(ocr, gemini) is False


def _real_image_bytes():
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    return img.read_bytes()


def test_backfill_one_accepts_agreement(tmp_path):
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6d5d", "BB", "2d6cAd")
    result = backfill_one(
        hand_id="H_AGREE",
        image_bytes=_real_image_bytes(),
        ocr_parsed=ocr,
        gemini_parsed=gemini,
        out_dir=tmp_path,
    )
    assert result is not None
    assert result["source"] == "gemini_reparse_agreement"
    assert result["hand_id"] == "H_AGREE"
    # ground_truth carries OCR's parse
    assert result["ground_truth"]["hero_hand"] == "6d5d"


def test_backfill_one_rejects_hero_disagreement(tmp_path):
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6h5d", "BB", "2d6cAd")
    result = backfill_one(
        hand_id="H_HERO_DIFF",
        image_bytes=_real_image_bytes(),
        ocr_parsed=ocr,
        gemini_parsed=gemini,
        out_dir=tmp_path,
    )
    assert result is None


def test_backfill_one_rejects_board_disagreement(tmp_path):
    ocr = _hand("6d5d", "BB", "2d6cAd")
    gemini = _hand("6d5d", "BB", "2d6cAh")
    result = backfill_one(
        hand_id="H_BOARD_DIFF",
        image_bytes=_real_image_bytes(),
        ocr_parsed=ocr,
        gemini_parsed=gemini,
        out_dir=tmp_path,
    )
    assert result is None


def test_backfill_one_rejects_when_gemini_parse_missing(tmp_path):
    ocr = _hand("6d5d", "BB", "2d6cAd")
    result = backfill_one(
        hand_id="H_NONE",
        image_bytes=_real_image_bytes(),
        ocr_parsed=ocr,
        gemini_parsed=None,
        out_dir=tmp_path,
    )
    assert result is None
