"""Multi-crop ensemble for hero card classification.

Reads three overlay-disjoint sub-crops of the same card and votes by
confidence. The WIN sticker covers the lower half — the top-third crop
isolates rank, and the bottom-third lets suit show through even when
the sticker bleeds vertically.

The single-crop classifier returns (rank, suit, conf). The ensemble
labels are the full card string `<rank><suit>` (e.g. ``"6d"``); a vote
with an unresolved rank or suit (``None``) contributes nothing.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np

from .infer import CardClassifier


class Vote(TypedDict):
    crop: str  # "full" | "top" | "bottom"
    label: str
    conf: float


class EnsembleResult(TypedDict):
    label: str
    card_conf: float
    votes: list[Vote]


_clf: CardClassifier | None = None


def _classifier() -> CardClassifier:
    global _clf
    if _clf is None:
        _clf = CardClassifier()
    return _clf


def _predict_one(crop: np.ndarray) -> tuple[str, float]:
    rank, suit, conf = _classifier().classify(crop)
    if rank is None or suit is None:
        return "", 0.0
    return f"{rank}{suit}", float(conf)


def _top_crop(crop: np.ndarray) -> np.ndarray:
    h = crop.shape[0]
    return crop[: int(h * 0.45)]


def _bottom_crop(crop: np.ndarray) -> np.ndarray:
    h = crop.shape[0]
    return crop[int(h * 0.55) :]


def predict_with_ensemble(crop: np.ndarray) -> EnsembleResult:
    votes: list[Vote] = []
    for name, sub in (
        ("full", crop),
        ("top", _top_crop(crop)),
        ("bottom", _bottom_crop(crop)),
    ):
        if sub.shape[0] < 10 or sub.shape[1] < 10:
            continue
        label, conf = _predict_one(sub)
        votes.append({"crop": name, "label": label, "conf": conf})

    # Require a hard majority — at least two of the three crops must agree
    # on the exact label — before we let the ensemble override the
    # single-pass read. Confidence-weighted voting alone is dangerous on
    # low-resolution hero crops: a single minority crop with the highest
    # raw conf can elect a label none of the others endorsed (H3433 card 1:
    # full=5d@0.16 (correct), top=7h@0.16, bottom=3c@0.27 → confidence-vote
    # picks 3c). Majority-agreement filters that case.
    counts: dict[str, int] = {}
    for v in votes:
        if v["label"]:
            counts[v["label"]] = counts.get(v["label"], 0) + 1
    majority = next(
        (lab for lab, n in counts.items() if n >= 2),
        None,
    )
    if majority is None:
        return {"label": "", "card_conf": 0.0, "votes": votes}
    agreeing = [v for v in votes if v["label"] == majority]
    card_conf = sum(v["conf"] for v in agreeing) / max(1, len(agreeing))
    # Boost when all three crops agree — strong signal.
    if len(agreeing) == len(votes):
        card_conf = min(1.0, card_conf + 0.1)
    return {"label": majority, "card_conf": float(card_conf), "votes": votes}
