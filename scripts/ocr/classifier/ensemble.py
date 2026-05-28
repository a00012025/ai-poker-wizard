"""Phase 11.B.1 — multi-crop ensemble with hard-majority safety.

Three sub-crops (full / top 45% / bottom 55%+) classified independently.
Label adopted only when ≥2 of 3 valid votes agree on the same rank+suit.
Disagreement → empty label (caller falls back to Gemini or abstains).

Confidence-weighted voting was rejected: H3433 case showed a high-conf
wrong minority can override a low-conf correct majority. See plan
docs/superpowers/plans/2026-05-28-ocr-production-precision.md Phase 11.B.1.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np


class _Classifier(Protocol):
    def classify(self, crop: np.ndarray) -> tuple[Optional[str], Optional[str], float]: ...


_DEFAULT_CLASSIFIER: _Classifier | None = None


def _default_classifier() -> _Classifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        from .infer import CardClassifier
        _DEFAULT_CLASSIFIER = CardClassifier()
    return _DEFAULT_CLASSIFIER


def _subcrops(crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
    h = crop.shape[0]
    return [
        ("full", crop),
        ("top", crop[: int(h * 0.45)]),
        ("bottom", crop[int(h * 0.55):]),
    ]


def predict_with_ensemble(
    crop: np.ndarray,
    *,
    classifier: _Classifier | None = None,
) -> dict[str, Any]:
    """Classify a card crop via 3-way ensemble with hard-majority safety.

    Returns ``{"label": str, "card_conf": float, "votes": list[dict]}``.
    Empty label when no rank+suit pair appears in ≥2 of the valid votes.
    """
    clf = classifier if classifier is not None else _default_classifier()
    votes: list[dict[str, Any]] = []
    for name, sub in _subcrops(crop):
        if sub.shape[0] < 10 or sub.shape[1] < 10:
            continue
        rank, suit, conf = clf.classify(sub)
        label = f"{rank}{suit}" if rank and suit else ""
        votes.append({"crop": name, "label": label, "conf": float(conf)})

    counts: dict[str, int] = {}
    for v in votes:
        if v["label"]:
            counts[v["label"]] = counts.get(v["label"], 0) + 1

    majority = next((lab for lab, n in counts.items() if n >= 2), None)
    if majority is None:
        return {"label": "", "card_conf": 0.0, "votes": votes}

    agreeing = [v for v in votes if v["label"] == majority]
    card_conf = sum(v["conf"] for v in agreeing) / len(agreeing)
    valid_votes = [v for v in votes if v["label"]]
    if len(agreeing) == len(valid_votes) and len(valid_votes) == 3:
        card_conf = min(1.0, card_conf + 0.1)
    return {"label": majority, "card_conf": float(card_conf), "votes": votes}
