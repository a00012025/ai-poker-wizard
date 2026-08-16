"""Choose information-dense preflop hand classes for GTOW Trainer drills."""

from __future__ import annotations

import math

from gto_formatter import normalize_hand_name
from gtow_trainer_url import ALL_TRAINER_GROUPS
from hh_deviation_check import HANDS_169


CONTINUE_MIN_FREQUENCY = 0.01
MIN_BOUNDARY_HALO = 0.12
TARGET_MULTIPLIER = 3.0
MAX_NARROW_RANGE = 0.40
_TRAINER_ORDER = tuple(ALL_TRAINER_GROUPS.split(","))
_RANKS = "AKQJT98765432"


def _combo_count(hand: str) -> int:
    return 6 if len(hand) == 2 else 4 if hand.endswith("s") else 12


def _grid_point(hand: str) -> tuple[int, int, int]:
    hi, lo = _RANKS.index(hand[0]), _RANKS.index(hand[1])
    shape = 0 if len(hand) == 2 else 1 if hand.endswith("s") else 2
    return hi, lo, shape


def _grid_distance(hand: str, continues: set[str]) -> int:
    point = _grid_point(hand)
    return min(
        abs(point[0] - other[0]) + abs(point[1] - other[1])
        + (0 if point[2] == other[2] else 1)
        for other in map(_grid_point, continues)
    )


def _node_selection(solution: dict, hero_position: str) -> set[str] | None:
    player = next((info for info in solution.get("players_info", [])
                   if (info.get("player") or {}).get("position") == hero_position), None)
    actions = solution.get("action_solutions") or []
    ranges = (player or {}).get("range") or []
    if len(ranges) != 169 or not actions:
        return None

    fold = next((action for action in actions
                 if (action.get("action") or {}).get("code") == "F"), None)
    continues = [action for action in actions if action is not fold]
    if fold is None or not continues:
        return None
    if any(len(action.get("strategy") or []) != 169 for action in actions):
        return None

    weights = {
        hand: max(0.0, float(ranges[index])) * _combo_count(hand)
        for index, hand in enumerate(HANDS_169)
        if float(ranges[index]) >= 0.005
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return None

    mandatory: set[str] = set()
    action_mass = 0.0
    for index, hand in enumerate(HANDS_169):
        if hand not in weights:
            continue
        frequency = sum(max(0.0, float(action["strategy"][index]))
                        for action in continues)
        action_mass += weights[hand] * min(1.0, frequency)
        if frequency >= CONTINUE_MIN_FREQUENCY:
            mandatory.add(hand)
    if not mandatory:
        return None

    selected = set(mandatory)
    mandatory_mass = sum(weights[hand] for hand in mandatory) / total_weight
    continue_mass = action_mass / total_weight
    target_mass = max(
        min(1.0, mandatory_mass + MIN_BOUNDARY_HALO),
        min(MAX_NARROW_RANGE,
            max(continue_mass * TARGET_MULTIPLIER,
                continue_mass + MIN_BOUNDARY_HALO)),
    )

    fold_evs = fold.get("evs") or []

    def boundary_key(hand: str):
        index = HANDS_169.index(hand)
        regret = math.inf
        if len(fold_evs) == 169:
            candidates = [action.get("evs") or [] for action in continues]
            if candidates and all(len(values) == 169 for values in candidates):
                gap = float(fold_evs[index]) - max(float(values[index])
                                                       for values in candidates)
                if math.isfinite(gap) and gap >= -1e-6:
                    regret = max(0.0, gap)
        return regret, _grid_distance(hand, mandatory), hand

    candidates = sorted((hand for hand in weights if hand not in selected),
                         key=boundary_key)
    selected_weight = sum(weights[hand] for hand in selected)
    for hand in candidates:
        if selected_weight / total_weight >= target_mass:
            break
        selected.add(hand)
        selected_weight += weights[hand]
    return selected


def select_preflop_hand_groups(
    nodes: list[tuple[dict, str]],
    required_hands: list[str] | tuple[str, ...] = (),
) -> list[str] | None:
    """Return the union of each node's continue range plus boundary halo.

    ``None`` means the solver payload cannot support an honest restriction;
    callers must leave the Trainer on its full-range default.
    """
    if not nodes:
        return None
    selected: set[str] = set()
    for solution, hero_position in nodes:
        node = _node_selection(solution, hero_position)
        if node is None:
            return None
        selected.update(node)
    selected.update(
        hand for raw in required_hands
        if (hand := normalize_hand_name(raw)) in HANDS_169
    )
    return [hand for hand in _TRAINER_ORDER if hand in selected]
