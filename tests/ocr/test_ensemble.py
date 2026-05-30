"""Phase 11.B.1 — multi-crop ensemble must enforce hard majority (≥2/3).

Confidence-weighted voting was rejected in May 28's H3433 case: the bottom
crop's high-confidence wrong label (3c @ 0.27) overrode the full crop's
correct low-confidence label (5d @ 0.16). Hard majority disagreement →
empty label, force the caller to fall back to Gemini or abstain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.ensemble import predict_with_ensemble


class _StubClassifier:
    """Returns a scripted sequence of (rank, suit, conf) per call."""

    def __init__(self, scripted: list[tuple[str | None, str | None, float]]):
        self._scripted = list(scripted)
        self._calls = 0

    def classify(self, crop):
        result = self._scripted[self._calls]
        self._calls += 1
        return result


def _crop(h=40, w=30):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_empty_crop_returns_empty_label():
    tiny = np.zeros((5, 5, 3), dtype=np.uint8)
    out = predict_with_ensemble(tiny, classifier=_StubClassifier([]))
    assert out["label"] == ""
    assert out["card_conf"] == 0.0


def test_all_three_agree_boosts_confidence():
    stub = _StubClassifier([
        ("5", "d", 0.40),
        ("5", "d", 0.50),
        ("5", "d", 0.60),
    ])
    out = predict_with_ensemble(_crop(), classifier=stub)
    assert out["label"] == "5d"
    raw_mean = (0.40 + 0.50 + 0.60) / 3
    assert out["card_conf"] >= raw_mean + 0.1 - 1e-6
    assert out["card_conf"] <= 1.0


def test_two_of_three_agree_majority_no_boost():
    stub = _StubClassifier([
        ("5", "d", 0.40),
        ("5", "d", 0.50),
        ("3", "c", 0.90),  # outlier
    ])
    out = predict_with_ensemble(_crop(), classifier=stub)
    assert out["label"] == "5d"
    # Only the two agreeing votes contribute; no +0.1 because not unanimous.
    assert abs(out["card_conf"] - (0.40 + 0.50) / 2) < 1e-6


def test_three_way_disagreement_returns_empty():
    stub = _StubClassifier([
        ("5", "d", 0.16),
        ("Q", "h", 0.20),
        ("3", "c", 0.27),  # H3433-style: would win confidence vote
    ])
    out = predict_with_ensemble(_crop(), classifier=stub)
    assert out["label"] == ""
    assert out["card_conf"] == 0.0


def test_votes_field_records_all_three():
    stub = _StubClassifier([
        ("5", "d", 0.40),
        ("5", "d", 0.50),
        ("5", "d", 0.60),
    ])
    out = predict_with_ensemble(_crop(), classifier=stub)
    assert [v["crop"] for v in out["votes"]] == ["full", "top", "bottom"]
    assert all(v["label"] == "5d" for v in out["votes"])


def test_classifier_returning_none_is_skipped():
    stub = _StubClassifier([
        (None, None, 0.0),  # full crop classifier failed
        ("5", "d", 0.50),
        ("5", "d", 0.60),
    ])
    out = predict_with_ensemble(_crop(), classifier=stub)
    # Two valid votes both agree → majority of valid votes → label = 5d
    assert out["label"] == "5d"
