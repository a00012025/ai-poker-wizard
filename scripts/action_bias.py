"""Sparse, EV-backed action-direction diagnosis for ledger decisions.

This is an explanation overlay on the EV-ranked leak board, not a second
ranking system.  It deliberately returns ``None`` unless one action error is
both repeated and robust enough to change the player's training prescription.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


LOSSY_MIN_BB = 0.10
DOMINANT_MIN_N = 5
DOMINANT_MIN_EV_BB = 3.0
DOMINANT_MIN_SHARE = 0.70
DOMINANT_LEAVE_ONE_OUT_SHARE = 0.60

BIAS_LABELS = {
    "overfold": "棄牌過多",
    "overcall": "跟注過多",
    "overraise": "加注過多",
    "too_passive": "該進攻時太被動",
}


def _action_kind(code) -> str | None:
    value = str(code or "").strip().upper()
    if not value:
        return None
    if value == "F" or value.startswith("FOLD"):
        return "fold"
    if value == "C" or value.startswith("CALL"):
        return "call"
    if value == "X" or value.startswith("CHECK"):
        return "check"
    if value.startswith("R") or value in {"AI", "ALLIN", "ALL-IN"}:
        return "raise"
    return None


def classify_action_bias(taken_code, best_code) -> str | None:
    """Map one material mistake to a player-readable action tendency.

    Raise-vs-raise differences are intentionally left unclassified: without
    resolved pot/stack context, calling them too large or too small would be a
    sizing guess rather than a ledger fact.
    """
    taken, best = _action_kind(taken_code), _action_kind(best_code)
    if not taken or not best or taken == best:
        return None
    if taken == "fold" and best != "fold":
        return "overfold"
    if taken == "call" and best == "fold":
        return "overcall"
    if taken == "raise" and best in {"fold", "call", "check"}:
        return "overraise"
    if taken in {"check", "call"} and best == "raise":
        return "too_passive"
    return None


def dominant_action_bias(
    decisions: Iterable[Mapping],
    *,
    lossy_min_bb: float = LOSSY_MIN_BB,
    min_n: int = DOMINANT_MIN_N,
    min_ev_bb: float = DOMINANT_MIN_EV_BB,
    min_share: float = DOMINANT_MIN_SHARE,
    leave_one_out_share: float = DOMINANT_LEAVE_ONE_OUT_SHARE,
) -> dict | None:
    """Return a dominant action tendency, or ``None`` when evidence is weak.

    The denominator is all material EV loss in the selected spot, including
    loss whose direction cannot be classified.  The leave-one-out gate removes
    the largest supporting hand so a single disaster cannot manufacture a
    user-visible tendency.
    """
    eligible = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        try:
            ev = float(row.get("ev_loss_bb") or 0.0)
        except (TypeError, ValueError):
            continue
        if ev + 1e-12 < lossy_min_bb:
            continue
        eligible.append(ev)
        direction = classify_action_bias(row.get("taken_code"), row.get("best_code"))
        if direction:
            grouped[direction].append(ev)

    total_ev = sum(eligible)
    if not grouped or total_ev <= 0:
        return None
    direction, losses = max(grouped.items(), key=lambda item: sum(item[1]))
    direction_ev = sum(losses)
    share = direction_ev / total_ev
    if len(losses) < min_n or direction_ev < min_ev_bb or share < min_share:
        return None

    largest = max(losses)
    remaining_total = total_ev - largest
    remaining_direction = direction_ev - largest
    if (remaining_total <= 0 or remaining_direction < min_ev_bb
            or remaining_direction / remaining_total < leave_one_out_share):
        return None

    return {
        "direction": direction,
        "label": BIAS_LABELS[direction],
        "n": len(losses),
        "ev_loss_bb": round(direction_ev, 4),
        "share": round(share, 4),
    }


def bias_suffix(bias: Mapping | None) -> str:
    """Compact queue-label suffix; absence is rendered as absence."""
    return f"｜{bias['label']}" if bias and bias.get("label") else ""
