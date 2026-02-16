#!/usr/bin/env python3
"""GTO Wizard API client.

Pure HTTP calls — no browser needed.
"""
import sys
from pathlib import Path

import requests

# Allow importing gto_token from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_token import get_access_token

API_BASE = "https://api.gtowizard.com"
ORIGIN = "https://app.gtowizard.com"
AVAILABLE_DEPTHS = [100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8]

# Reuse TCP connection across requests
_session = requests.Session()
_session.headers.update({"origin": ORIGIN})


def _ensure_auth():
    _session.headers["authorization"] = f"Bearer {get_access_token()}"


def nearest_depth(bb: float) -> float:
    """Find the nearest available depth and return as depth.125 format."""
    best = min(AVAILABLE_DEPTHS, key=lambda d: abs(d - bb))
    return best + 0.125


def get_next_actions(
    gametype: str = "MTTGeneral",
    depth: float = 30.125,
    preflop_actions: str = "",
    board: str = "",
    flop_actions: str = "",
    turn_actions: str = "",
    river_actions: str = "",
) -> dict:
    """Get available actions for a spot."""
    _ensure_auth()
    r = _session.get(
        f"{API_BASE}/v1/poker/next-actions/",
        params={
            "gametype": gametype,
            "depth": depth,
            "stacks": "",
            "preflop_actions": preflop_actions,
            "board": board,
            "flop_actions": flop_actions,
            "turn_actions": turn_actions,
            "river_actions": river_actions,
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def get_spot_solution(
    gametype: str = "MTTGeneral",
    depth: float = 30.125,
    preflop_actions: str = "",
    board: str = "",
    flop_actions: str = "",
    turn_actions: str = "",
    river_actions: str = "",
) -> dict | None:
    """Get full strategy solution for a spot. Returns None if no solution (204)."""
    _ensure_auth()
    r = _session.get(
        f"{API_BASE}/v4/solutions/spot-solution/",
        params={
            "gametype": gametype,
            "depth": depth,
            "stacks": "",
            "preflop_actions": preflop_actions,
            "board": board,
            "flop_actions": flop_actions,
            "turn_actions": turn_actions,
            "river_actions": river_actions,
        },
        timeout=10,
    )
    if r.status_code == 204:
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
