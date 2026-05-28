"""The ensemble reads three overlay-disjoint crops and votes by confidence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.ensemble import predict_with_ensemble


def test_ensemble_returns_single_label_and_conf():
    crop = np.full((128, 96, 3), 255, dtype=np.uint8)
    result = predict_with_ensemble(crop)
    assert set(result.keys()) >= {"label", "card_conf", "votes"}
    assert isinstance(result["label"], str)
    assert 0.0 <= result["card_conf"] <= 1.0


def test_ensemble_three_votes_when_all_crops_valid():
    crop = np.full((128, 96, 3), 255, dtype=np.uint8)
    result = predict_with_ensemble(crop)
    assert len(result["votes"]) == 3
    for v in result["votes"]:
        assert {"crop", "label", "conf"} <= v.keys()
