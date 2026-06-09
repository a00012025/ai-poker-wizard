"""Pure metric helpers for effective_bb evaluation. No OCR, no I/O."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gto_api import nearest_depth

try:
    from analyze_hand import POSITION_ORDERS
except Exception:  # pragma: no cover - analyze_hand import is heavy
    POSITION_ORDERS = {}


def depth_bucket(bb):
    """Snap a bb value to its solver depth bucket (int). None on bad input."""
    try:
        return int(round(nearest_depth(float(bb))))
    except (TypeError, ValueError):
        return None


def bucket_match(a, b) -> bool:
    """True iff a and b snap to the same solver depth bucket."""
    ba, bb = depth_bucket(a), depth_bucket(b)
    return ba is not None and ba == bb


def hero_folded_preflop(gt: dict):
    """True/False if hero's preflop code is F, else None (unknown order)."""
    pa = (gt.get("preflop_actions") or "").split("-")
    hp = gt.get("hero_position")
    order = POSITION_ORDERS.get(gt.get("num_players")) or \
        POSITION_ORDERS.get(gt.get("table_size"))
    if not order or hp not in order:
        return None
    idx = order.index(hp)
    if idx < len(pa):
        return pa[idx] == "F"
    return None


def classify_fault(*, p_eff, gt_eff, hero_start, gt_max) -> str:
    """Bucket an emitted-but-wrong hand into one of 4 fault classes.

    impossible_over — p_eff exceeds the largest observed table stack (×1.1)
    selection       — the code returned hero's own start instead of a shorter
                      villain (p_eff ≈ hero_start, hero_start != gt_eff)
    undershoot      — p_eff is less than ~71% of ground-truth
    near            — adjacent-bucket near miss (everything else)
    """
    ratio = (p_eff / gt_eff) if gt_eff else 0.0
    if gt_max and p_eff > gt_max * 1.1:
        return "impossible_over"
    if hero_start and abs(p_eff - hero_start) < 0.5 and abs(hero_start - gt_eff) > 0.5:
        return "selection"      # returned hero's stack instead of a shorter villain
    if ratio <= 0.71:
        return "undershoot"
    return "near"               # adjacent bucket, < 1.4x off
