"""Calibration helpers: ECE, reliability bins, precision-coverage curve."""
from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.calibration import (
    expected_calibration_error,
    precision_coverage_curve,
    reliability_bins,
)


def test_ece_perfect_calibration():
    confs = [0.5] * 10
    correct = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ece = expected_calibration_error(confs, correct, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_completely_wrong():
    confs = [0.9] * 10
    correct = [0] * 10
    ece = expected_calibration_error(confs, correct, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-6)


def test_reliability_bins_shape():
    confs = [0.1, 0.2, 0.35, 0.55, 0.7, 0.95, 0.99]
    correct = [0, 0, 0, 1, 1, 1, 1]
    bins = reliability_bins(confs, correct, n_bins=5)
    assert len(bins) == 5
    for b in bins:
        assert {"lo", "hi", "n", "mean_conf", "accuracy"} <= b.keys()


def test_precision_coverage_curve_monotonic_thresholds():
    confs = [0.1, 0.5, 0.9, 0.95, 0.99]
    correct = [0, 0, 1, 1, 1]
    curve = precision_coverage_curve(confs, correct, n_points=5)
    coverages = [pt["coverage"] for pt in curve]
    assert coverages == sorted(coverages, reverse=True)


def test_precision_coverage_finds_high_precision_threshold():
    random.seed(0)
    confs = [0.99 - i * 0.01 for i in range(100)]
    correct = [1] * 20 + [random.randint(0, 1) for _ in range(80)]
    curve = precision_coverage_curve(confs, correct, n_points=50)
    qualifying = [
        pt for pt in curve
        if pt["precision"] >= 0.99 and pt["coverage"] >= 0.15
    ]
    assert qualifying, "no threshold achieves precision>=0.99 at coverage>=0.15"
