"""harvest_snapshot extracts labeled hero crops from a snapshot using
the user-verified expected_json, not the (possibly wrong) parsed_json."""
from __future__ import annotations

from pathlib import Path

import pytest


from ocr.classifier.harvest_production import harvest_snapshot


def test_harvest_extracts_labeled_hero_crops(tmp_path):
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    expected = {"hero_hand": "6d5d"}
    n = harvest_snapshot(
        hand_id="H3433",
        image_bytes=img.read_bytes(),
        expected=expected,
        out_dir=tmp_path,
    )
    assert n >= 2
    labels = {p.parent.name for p in tmp_path.rglob("*.png")}
    assert "6d" in labels or "5d" in labels
