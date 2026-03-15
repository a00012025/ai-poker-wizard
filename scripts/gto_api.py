#!/usr/bin/env python3
"""GTO Wizard API client.

Pure HTTP calls — no browser needed.
"""
import threading
import time
import sys
from pathlib import Path

import requests

# Allow importing gto_token from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_token import get_access_token
from gto_cache import get as cache_get, put as cache_put, SENTINEL

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"
AVAILABLE_DEPTHS = [100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8]
AVAILABLE_CASH_DEPTHS = [300, 200, 100, 60, 50, 40, 30, 25, 20]
_TIMEOUT = 15
_MAX_RETRIES = 2

# Reuse TCP connection across requests
_session = requests.Session()
_session.headers.update({"origin": ORIGIN})

# Thread-local storage for per-user token override
_thread_local = threading.local()


def set_user_token(token: str):
    """Set a per-user access token for the current thread."""
    _thread_local.access_token = token


def clear_user_token():
    """Clear the per-user access token for the current thread."""
    _thread_local.access_token = None


def _ensure_auth():
    _session.headers["authorization"] = f"Bearer {get_access_token()}"


def _get_with_retry(url: str, params: dict, timeout: int = _TIMEOUT) -> requests.Response:
    """GET with automatic retry on timeout/connection errors.

    Uses per-user token from thread-local if set, otherwise falls back to
    global session token.
    """
    user_token = getattr(_thread_local, "access_token", None)
    if user_token:
        headers = {"authorization": f"Bearer {user_token}"}
    else:
        _ensure_auth()
        headers = None  # use session defaults

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _session.get(url, params=params, timeout=timeout, headers=headers)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(1 * (attempt + 1))


def nearest_depth(bb: float) -> float:
    """Find the nearest available depth and return as depth.125 format."""
    best = min(AVAILABLE_DEPTHS, key=lambda d: abs(d - bb))
    return best + 0.125


def nearest_cash_depth(bb: float) -> float:
    """Find the nearest available cash game depth (no .125 suffix)."""
    best = min(AVAILABLE_CASH_DEPTHS, key=lambda d: abs(d - bb))
    return float(best)


def get_next_actions(
    gametype: str = "MTTGeneral",
    depth: float = 30.125,
    stacks: str = "",
    preflop_actions: str = "",
    board: str = "",
    flop_actions: str = "",
    turn_actions: str = "",
    river_actions: str = "",
) -> dict:
    """Get available actions for a spot."""
    params = {
        "gametype": gametype, "depth": depth, "stacks": stacks,
        "preflop_actions": preflop_actions, "board": board,
        "flop_actions": flop_actions, "turn_actions": turn_actions,
        "river_actions": river_actions,
    }
    cached = cache_get("next_actions", params)
    if cached is not SENTINEL:
        return cached
    r = _get_with_retry(
        f"{API_BASE}/v1/poker/next-actions/",
        params=params,
    )
    r.raise_for_status()
    result = r.json()
    cache_put("next_actions", params, result)
    return result


def get_spot_solution(
    gametype: str = "MTTGeneral",
    depth: float = 30.125,
    stacks: str = "",
    preflop_actions: str = "",
    board: str = "",
    flop_actions: str = "",
    turn_actions: str = "",
    river_actions: str = "",
) -> dict | None:
    """Get full strategy solution for a spot. Returns None if no solution (204)."""
    params = {
        "gametype": gametype, "depth": depth, "stacks": stacks,
        "preflop_actions": preflop_actions, "board": board,
        "flop_actions": flop_actions, "turn_actions": turn_actions,
        "river_actions": river_actions,
    }
    cached = cache_get("spot_solution", params)
    if cached is not SENTINEL:
        return cached
    r = _get_with_retry(
        f"{API_BASE}/v4/solutions/spot-solution/",
        params=params,
    )
    if r.status_code in (204, 403):
        cache_put("spot_solution", params, None)
        return None
    r.raise_for_status()
    result = r.json()
    cache_put("spot_solution", params, result)
    return result


def find_closest_action_from_solutions(action_solutions: list[dict], target_size: float) -> str:
    """Find the closest action code from spot_solution's action_solutions.

    This avoids an extra next_actions API call.
    """
    best_code = None
    best_diff = float("inf")

    for sol in action_solutions:
        action = sol["action"]
        code = action["code"]
        if code == "X":
            if target_size == 0:
                return "X"
            continue
        if code in ("F", "C"):
            continue
        size = float(action["betsize"])
        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_code = code

    return best_code or "X"


def find_closest_action(available_actions: list[dict], target_size: float) -> str:
    """Find the action code closest to target bet size.

    Args:
        available_actions: list from next_actions response
        target_size: the bet size used in the actual hand

    Returns:
        action code string (e.g., "R1.9", "X", "RAI")
    """
    best_code = None
    best_diff = float("inf")

    for entry in available_actions:
        action = entry["action"]
        code = action["code"]
        if code == "X":
            if target_size == 0:
                return "X"
            continue
        if code == "F":
            continue
        size = float(action["betsize"])
        diff = abs(size - target_size)
        if diff < best_diff:
            best_diff = diff
            best_code = code

    return best_code or "X"


def find_closest_action_by_pot_pct(available_actions: list[dict], target_size: float) -> str:
    """Find closest action using pot-percentage matching.

    More robust than absolute matching when pot size is slightly off
    (e.g., LLM forgot antes). Computes the target's pot percentage and
    matches against each action's known pot percentage.

    Falls back to absolute matching if pot percentage can't be determined.
    """
    # Compute solver pot from any action with betsize_by_pot
    solver_pot = None
    for entry in available_actions:
        pct = entry["action"].get("betsize_by_pot")
        bs = entry["action"].get("betsize")
        if pct and float(pct) > 0 and bs:
            solver_pot = float(bs) / float(pct)
            break

    if not solver_pot or solver_pot <= 0:
        return find_closest_action(available_actions, target_size)

    target_pct = target_size / solver_pot

    best_code = None
    best_diff = float("inf")
    for entry in available_actions:
        action = entry["action"]
        code = action["code"]
        if code in ("X", "F"):
            continue
        action_pct = float(action.get("betsize_by_pot") or 0)
        diff = abs(action_pct - target_pct)
        if diff < best_diff:
            best_diff = diff
            best_code = code

    return best_code or find_closest_action(available_actions, target_size)


def find_closest_action_postflop(available_actions: list[dict], target_size: float) -> str:
    """Find closest postflop action, auto-detecting percentage-based sizes.

    LLM parsers sometimes output pot percentages (e.g. 40 for 40% pot)
    instead of absolute bb amounts. When the absolute match lands on
    all-in but the target doesn't look like an all-in, try interpreting
    it as a pot percentage and pick the better fit.
    """
    abs_code = find_closest_action(available_actions, target_size)

    # If absolute match is NOT all-in, it's almost certainly correct
    abs_action = next(
        (a for a in available_actions if a["action"]["code"] == abs_code), None
    )
    if not abs_action or not abs_action["action"].get("allin"):
        return abs_code

    # Matched all-in — check if target_size is actually a percentage
    # But if target is close to the all-in size, it's clearly an absolute amount
    allin_size = float(abs_action["action"]["betsize"])
    if allin_size > 0 and abs(allin_size - target_size) / max(target_size, 1) < 0.15:
        return abs_code

    # Compute solver pot from any action with betsize_by_pot
    solver_pot = None
    for entry in available_actions:
        pct = entry["action"].get("betsize_by_pot")
        if pct and float(pct) > 0:
            solver_pot = float(entry["action"]["betsize"]) / float(pct)
            break

    if not solver_pot or target_size < 5:
        return abs_code

    # Compare pot-percentage errors:
    # Absolute interpretation: target_size bb = what % of pot?
    abs_pct = target_size / solver_pot * 100
    abs_matched_pct = float(abs_action["action"].get("betsize_by_pot") or 0) * 100
    abs_err = abs(abs_pct - abs_matched_pct)

    # Percentage interpretation: target_size = X% of pot
    pct_bb = target_size / 100 * solver_pot
    pct_code = find_closest_action(available_actions, pct_bb)
    pct_action = next(
        (a for a in available_actions if a["action"]["code"] == pct_code), None
    )
    pct_matched_pct = float(pct_action["action"].get("betsize_by_pot") or 0) * 100 if pct_action else 0
    pct_err = abs(target_size - pct_matched_pct)

    if pct_err < abs_err:
        return pct_code
    return abs_code
