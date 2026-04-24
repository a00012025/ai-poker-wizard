# tests/test_localization.py
"""Phase 0 — verify localization functions expose crops without classification."""
import cv2
import numpy as np
import pytest
from pathlib import Path

from scripts.ocr.region_detector import detect_regions
from scripts.ocr.table_parser import _locate_hero_cards, _locate_board_cards


SNAPSHOT = Path(__file__).parent / "snapshots" / "H2491" / "input.jpeg"


@pytest.fixture
def table_region():
    img = cv2.imread(str(SNAPSHOT))
    assert img is not None, f"missing snapshot: {SNAPSHOT}"
    regions = detect_regions(img)
    assert regions is not None, "region_detector failed on H2491"
    return regions["table"]


def test_locate_hero_cards_returns_crops(table_region):
    crops = _locate_hero_cards(table_region)
    assert isinstance(crops, list)
    assert len(crops) == 2
    for c in crops:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 3  # BGR
        assert c.shape[0] > 20 and c.shape[1] > 20


def test_locate_board_cards_returns_crops(table_region):
    # H2491 has a full flop+turn+river in expected.json → exactly 5 board cards
    crops = _locate_board_cards(table_region)
    assert isinstance(crops, list)
    assert len(crops) == 5
    for c in crops:
        assert isinstance(c, np.ndarray)
        assert c.ndim == 3
        assert c.shape[0] > 10 and c.shape[1] > 10
