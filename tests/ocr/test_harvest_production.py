"""Harvest extracts hero/board crops from a snapshot's image_data and
labels them with the snapshot's expected_json (the user-verified truth)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.harvest_production import harvest_snapshot


def test_harvest_extracts_labeled_hero_crops(tmp_path):
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    expected = {"hero_hand": "6d5d", "streets": [{"board": "2d6cAd"}]}
    out = tmp_path / "out"
    n = harvest_snapshot(
        hand_id="H3433",
        image_bytes=img.read_bytes(),
        expected=expected,
        out_dir=out,
    )
    assert n >= 2, f"expected at least 2 hero crops, harvested {n}"
    files = list(out.rglob("*.png"))
    labels = {f.parent.name for f in files}
    assert "6d" in labels or "5d" in labels, f"got labels {labels}"
