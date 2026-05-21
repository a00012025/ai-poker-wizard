"""Calibration metrics for the OCR confidence head."""
from __future__ import annotations

from typing import TypedDict


class Bin(TypedDict):
    lo: float
    hi: float
    n: int
    mean_conf: float
    accuracy: float


class CurvePoint(TypedDict):
    threshold: float
    coverage: float
    precision: float
    emitted: int
    correct: int
    total: int


def _validate(confs: list[float], correct: list[int]) -> None:
    if len(confs) != len(correct):
        raise ValueError("confs and correct must be same length")


def reliability_bins(confs: list[float], correct: list[int], n_bins: int = 10) -> list[Bin]:
    """Bucket confidence predictions and report observed accuracy per bin."""
    _validate(confs, correct)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bins: list[Bin] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            members = [(c, y) for c, y in zip(confs, correct) if lo <= c <= hi]
        else:
            members = [(c, y) for c, y in zip(confs, correct) if lo <= c < hi]
        n = len(members)
        mean_conf = sum(c for c, _ in members) / n if n else 0.0
        accuracy = sum(y for _, y in members) / n if n else 0.0
        bins.append({
            "lo": lo,
            "hi": hi,
            "n": n,
            "mean_conf": mean_conf,
            "accuracy": accuracy,
        })
    return bins


def expected_calibration_error(
    confs: list[float], correct: list[int], n_bins: int = 10,
) -> float:
    """Return expected calibration error over fixed-width confidence bins."""
    _validate(confs, correct)
    total = len(confs)
    if total == 0:
        return 0.0
    return sum(
        b["n"] / total * abs(b["mean_conf"] - b["accuracy"])
        for b in reliability_bins(confs, correct, n_bins=n_bins)
    )


def precision_coverage_curve(
    confs: list[float], correct: list[int], n_points: int = 50,
) -> list[CurvePoint]:
    """Sweep thresholds from 0 to 1 and report precision at each coverage."""
    _validate(confs, correct)
    if n_points <= 0:
        raise ValueError("n_points must be positive")

    total = len(confs)
    pairs = sorted(zip(confs, correct), key=lambda x: -x[0])
    points: list[CurvePoint] = []
    for i in range(n_points + 1):
        threshold = i / n_points
        kept = [(c, y) for c, y in pairs if c >= threshold]
        emitted = len(kept)
        correct_count = sum(y for _, y in kept)
        precision = correct_count / emitted if emitted else 0.0
        coverage = emitted / total if total else 0.0
        points.append({
            "threshold": threshold,
            "coverage": coverage,
            "precision": precision,
            "emitted": emitted,
            "correct": correct_count,
            "total": total,
        })
    return points
