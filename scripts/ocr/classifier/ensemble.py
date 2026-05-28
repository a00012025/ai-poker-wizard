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

    tallies: dict[str, float] = {}
    for v in votes:
        if not v["label"]:
            continue
        tallies[v["label"]] = tallies.get(v["label"], 0.0) + v["conf"]
    if not tallies:
        return {"label": "", "card_conf": 0.0, "votes": votes}
    label = max(tallies, key=tallies.get)
    total = sum(tallies.values())
    card_conf = tallies[label] / total if total > 0 else 0.0
    return {"label": label, "card_conf": float(card_conf), "votes": votes}
