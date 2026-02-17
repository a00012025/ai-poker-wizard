#!/usr/bin/env python3
"""GTO Wizard API client.

Pure HTTP calls — no browser needed.
"""
import time
import sys
from pathlib import Path

import requests

# Allow importing gto_token from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_token import get_access_token

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"
AVAILABLE_DEPTHS = [100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8]
_TIMEOUT = 15
_MAX_RETRIES = 2

# Reuse TCP connection across requests
_session = requests.Session()
_session.headers.update({"origin": ORIGIN})


def _ensure_auth():
    _session.headers["authorization"] = f"Bearer {get_access_token()}"


def _get_with_retry(url: str, params: dict, timeout: int = _TIMEOUT) -> requests.Response:
    """GET with automatic retry on timeout/connection errors."""
    _ensure_auth()
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _session.get(url, params=params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(1 * (attempt + 1))


def nearest_depth(bb: float) -> float:
    """Find the nearest available depth and return as depth.125 format."""
    best = min(AVAILABLE_DEPTHS, key=lambda d: abs(d - bb))
    return best + 0.125


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
    r = _get_with_retry(
        f"{API_BASE}/v1/poker/next-actions/",
        params={
            "gametype": gametype,
            "depth": depth,
            "stacks": stacks,
            "preflop_actions": preflop_actions,
            "board": board,
            "flop_actions": flop_actions,
            "turn_actions": turn_actions,
            "river_actions": river_actions,
        },
    )
    r.raise_for_status()
    return r.json()


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
    r = _get_with_retry(
        f"{API_BASE}/v4/solutions/spot-solution/",
        params={
            "gametype": gametype,
            "depth": depth,
            "stacks": stacks,
            "preflop_actions": preflop_actions,
            "board": board,
            "flop_actions": flop_actions,
            "turn_actions": turn_actions,
            "river_actions": river_actions,
        },
    )
    if r.status_code in (204, 403):
        return None
    r.raise_for_status()
    return r.json()


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
    abs_matched_pct = float(abs_action["action"].get("betsize_by_pot", 0)) * 100
    abs_err = abs(abs_pct - abs_matched_pct)

    # Percentage interpretation: target_size = X% of pot
    pct_bb = target_size / 100 * solver_pot
    pct_code = find_closest_action(available_actions, pct_bb)
    pct_action = next(
        (a for a in available_actions if a["action"]["code"] == pct_code), None
    )
    pct_matched_pct = float(pct_action["action"].get("betsize_by_pot", 0)) * 100 if pct_action else 0
    pct_err = abs(target_size - pct_matched_pct)

    if pct_err < abs_err:
        return pct_code
    return abs_code
