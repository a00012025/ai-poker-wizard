"""Phase 11.B.1 — multi-crop CardCNN ensemble with hard-majority safety.

For a hero crop, run the classifier on three views (full / top 45% / bottom
from 55%) and require ≥2/3 to agree on the same label. Disagreement →
empty label, which forces the caller to fall back to Gemini or abstain.

Confidence-weighted voting was rejected by the H3433 case: the bottom
crop's 0.27 wrong "3c" overrode the full crop's 0.16 correct "5d".
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class Vote(TypedDict):
    crop: str
    label: str
    conf: float


class EnsembleResult(TypedDict):
    label: str
    card_conf: float
    votes: list[Vote]


_DEFAULT_CLASSIFIER = None


def _classifier():
    """Lazy-load the default CardClassifier singleton."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        from .infer import CardClassifier
        _DEFAULT_CLASSIFIER = CardClassifier()
    return _DEFAULT_CLASSIFIER


def predict_with_ensemble(
    crop: np.ndarray,
    *,
    classifier=None,
) -> EnsembleResult:
    """Three-view ensemble vote on a hero card crop.

    Returns label="" with card_conf=0.0 on either (a) a too-small crop or
    (b) no ≥2/3 majority among the three sub-views. Otherwise returns
    the agreed label with mean confidence of the agreeing votes, plus a
    +0.1 boost (clamped to 1.0) when the agreement is unanimous.
    """
    clf = classifier if classifier is not None else _classifier()
    votes: list[Vote] = []
    sub_views = (
        ("full", crop),
        ("top", crop[: int(crop.shape[0] * 0.45)] if crop.size else crop),
        ("bottom", crop[int(crop.shape[0] * 0.55):] if crop.size else crop),
    )
    for name, sub in sub_views:
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
    if len(agreeing) == len(valid_votes) == 3:
        card_conf = min(1.0, card_conf + 0.1)
    return {"label": majority, "card_conf": float(card_conf), "votes": votes}
