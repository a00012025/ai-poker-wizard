"""Pure metric helpers for effective_bb evaluation. No OCR, no I/O."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gto_api import nearest_depth, AVAILABLE_DEPTHS

try:
    from analyze_hand import POSITION_ORDERS
except Exception:  # pragma: no cover - analyze_hand import is heavy
    POSITION_ORDERS = {}

# classify_fault thresholds (ratio = p_eff / gt_eff)
_OVER_RATIO = 1.4        # well above GT before we call it an impossible over-add
_GT_MAX_MARGIN = 1.1     # 10% headroom above the largest observed seat stack
_SELECTION_RATIO = 1.2   # over-computed enough to be a min-over-villains miss
_UNDERSHOOT_RATIO = 0.71 # below ~71% of GT is a short reconstruction


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
    raw = gt.get("preflop_actions") or ""
    if not raw:
        return None  # empty/missing actions string — unknown
    pa = raw.split("-")
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

    - impossible_over: emitted a value above any seat's stack (over-add bug).
    - selection: over-computed but within table stacks (failed min-over-villains).
    - undershoot: emitted well below ground truth (dropped/short reconstruction).
    - near: adjacent-bucket miss, magnitude roughly right.
    """
    ratio = (p_eff / gt_eff) if gt_eff else 0.0
    # Above any plausible seat stack -> an over-add, not a selection error.
    if ratio >= _OVER_RATIO and gt_max and p_eff > gt_max * _GT_MAX_MARGIN:
        return "impossible_over"
    # Over-computed within table stacks: returned a too-large active stack.
    if ratio >= _SELECTION_RATIO:
        return "selection"
    if ratio <= _UNDERSHOOT_RATIO:
        return "undershoot"
    return "near"
