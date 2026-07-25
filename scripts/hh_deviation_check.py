#!/usr/bin/env python3
"""Check hero's hand history actions against GTO Wizard solver.

Directly queries the API for each hero decision point, extracts hand-specific
frequencies from 169-element (preflop) or 1326-element (postflop) strategy
arrays, and flags deviations.

Usage:
    python scripts/hh_deviation_check.py 2026-02-17/
    python scripts/hh_deviation_check.py 2026-02-17/ --limit 5
    python scripts/hh_deviation_check.py 2026-02-17/ --threshold 20
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hh_parser import parse_directory, parse_file, POSITION_ORDERS
from gto_api import (
    get_spot_solution, get_next_actions,
    find_closest_action, find_closest_action_by_pot_fraction,
    find_closest_action_postflop, find_unique_nonallin_raise, nearest_depth,
)
from gto_formatter import (
    normalize_hand_name, _COMBO_INDEX, _COMBO_RANKS, _COMBO_SUITS, _RANK_ORDER,
    _get_board_cards, _combo_to_hand_name, combo_index_for_hand as _combo_index_for_hand,
)
from card_display import cards_to_emoji

# ── 169-element preflop hand index (ASCII-sorted hand names) ──

_RANKS_BY_VALUE = "23456789TJQKA"

def _build_hands_169() -> list[str]:
    """Build 169 hand names sorted by ASCII string comparison."""
    hands = []
    for i in range(13):
        for j in range(13):
            r1 = _RANKS_BY_VALUE[i]
            r2 = _RANKS_BY_VALUE[j]
            if i == j:
                hands.append(r1 + r2)
            elif i > j:
                hands.append(r1 + r2 + "o")
            else:
                hands.append(r2 + r1 + "s")
    hands.sort()
    return hands

HANDS_169 = _build_hands_169()
HAND_TO_169 = {h: i for i, h in enumerate(HANDS_169)}

POSITION_ORDER_8MAX = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def _convert_preflop_to_8max(preflop_actions: str, num_players: int) -> str:
    """Convert N-player preflop actions to 8-max by prepending folds.

    The API always uses 8-max position ordering. For a 6-player table,
    the first 2 positions (UTG, UTG+1) don't exist, so we prepend folds.
    """
    if num_players >= 8:
        return preflop_actions
    prefix_folds = 8 - num_players
    parts = preflop_actions.split("-")
    return "-".join(["F"] * prefix_folds + parts)


def _convert_hero_position_to_8max(hero_pos: str, num_players: int) -> str:
    """Hero position is already in standard names (LJ, HJ, CO, etc.)
    which are the same in both N-player and 8-max.
    """
    return hero_pos


def _hero_continuation_context(pf_parts: list[str], num_players: int,
                               hero_idx: int) -> tuple[list[str], str | None]:
    """Return actions before hero's next preflop turn and hero's action.

    Continuation tokens rotate through players who survived the first pass.
    Folded players are removed immediately; all-in players are never eligible
    to act again.  This is needed to query the solver at hero's actual node
    rather than at the first player still pending after the initial round.
    """
    active = [
        i for i in range(min(num_players, len(pf_parts)))
        if pf_parts[i] not in ("F", "") and not pf_parts[i].startswith("AI")
    ]
    cursor = 0
    before_hero: list[str] = []
    for raw in pf_parts[num_players:]:
        if not active:
            break
        cursor %= len(active)
        actor = active[cursor]
        if actor == hero_idx:
            return before_hero, raw
        before_hero.append(raw)
        if raw == "F" or raw.startswith("AI"):
            active.pop(cursor)
        else:
            cursor = (cursor + 1) % len(active)
    return before_hero, None


def _get_hand_ev(solution: dict, hero_hand: str, hero_pos: str, is_preflop: bool,
                 combo_idx: int | None = None) -> float | None:
    """Extract EV for hero's hand from a spot solution.

    Uses simple_hand_counters first (pre-computed per-hand EV), then falls back
    to the raw hand_evs array. For postflop with combo_idx, returns exact combo EV.

    Returns EV in bb, or None if unavailable.
    """
    for pi in solution.get("players_info", []):
        if pi["player"]["position"] != hero_pos:
            continue

        # For preflop or when no combo_idx, try simple_hand_counters first
        if is_preflop or combo_idx is None:
            shc = pi.get("simple_hand_counters", {})
            hand_data = shc.get(hero_hand)
            if hand_data and "hand_ev" in hand_data:
                return hand_data["hand_ev"]

        # Fallback to hand_evs array
        ev_arr = pi.get("hand_evs", [])

        if is_preflop and len(ev_arr) == 169:
            idx = HAND_TO_169.get(hero_hand)
            if idx is not None:
                return ev_arr[idx]
            return None

        if not is_preflop and len(ev_arr) == 1326:
            # Direct combo lookup
            if combo_idx is not None and combo_idx < len(ev_arr):
                range_arr = pi.get("range", [])
                if len(range_arr) == 1326 and range_arr[combo_idx] >= 0.005:
                    return ev_arr[combo_idx]
                return None

            # Fallback: average across all combos of the hand name
            range_arr = pi.get("range", [])
            if len(range_arr) != 1326:
                return None
            board_cards = _get_board_cards(solution["game"]["board"])
            total_weight = 0.0
            total_ev = 0.0
            for idx, (c1, c2) in enumerate(_COMBO_INDEX):
                if c1 in board_cards or c2 in board_cards:
                    continue
                if _combo_to_hand_name(c1, c2) != hero_hand:
                    continue
                rng = range_arr[idx]
                if rng < 0.005:
                    continue
                total_weight += rng
                total_ev += ev_arr[idx] * rng
            if total_weight > 0.005:
                return total_ev / total_weight
            return None

        return None
    return None


def _get_action_evs_preflop(solution: dict, hero_hand: str, hero_pos: str) -> dict[str, float] | None:
    """Extract per-action EVs for hero's hand from 169-element preflop arrays.

    Reads action_solutions[i].evs[idx] for each action.
    Returns {action_code: ev_in_bb} or None if data unavailable.
    """
    if not solution or "action_solutions" not in solution:
        return None

    player_info = None
    for pi in solution["players_info"]:
        if pi["player"]["position"] == hero_pos:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    if len(range_arr) != 169:
        return None

    idx = HAND_TO_169.get(hero_hand)
    if idx is None:
        return None

    if range_arr[idx] < 0.005:
        return None

    evs = {}
    for asol in solution["action_solutions"]:
        ev_arr = asol.get("evs")
        if not ev_arr or len(ev_arr) != 169:
            return None  # EVs not available for this solution
        code = asol["action"]["code"]
        evs[code] = ev_arr[idx]
    return evs if evs else None


def _get_action_evs_postflop(solution: dict, hero_hand: str, hero_pos: str,
                              combo_idx: int | None = None) -> dict[str, float] | None:
    """Extract per-action EVs for hero's hand from 1326-element postflop arrays.

    Reads action_solutions[i].evs[combo_idx] for each action.
    Returns {action_code: ev_in_bb} or None if data unavailable.
    """
    if not solution or "action_solutions" not in solution:
        return None

    player_info = None
    for pi in solution["players_info"]:
        if pi["player"]["position"] == hero_pos:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    if len(range_arr) != 1326:
        return None

    action_solutions = solution["action_solutions"]

    # Direct combo lookup
    if combo_idx is not None and combo_idx < len(range_arr):
        rng = range_arr[combo_idx]
        # Exact-combo decisions remain meaningful at very low but non-zero
        # reach frequencies. The old 0.5% display cutoff discarded GTOW-graded
        # rare branches (d8622ce7 had range 0.012%) and hid their EV entirely.
        if rng < 1e-12:
            return None
        evs = {}
        for asol in action_solutions:
            ev_arr = asol.get("evs")
            if not ev_arr or len(ev_arr) != 1326:
                return None
            code = asol["action"]["code"]
            evs[code] = ev_arr[combo_idx]
        return evs if evs else None

    # Fallback: weighted average across all combos of the hand name
    board_cards = _get_board_cards(solution["game"]["board"])

    total_weight = 0.0
    action_evs: dict[str, float] = {}

    for idx, (c1, c2) in enumerate(_COMBO_INDEX):
        if c1 in board_cards or c2 in board_cards:
            continue
        if _combo_to_hand_name(c1, c2) != hero_hand:
            continue
        rng = range_arr[idx]
        if rng < 0.005:
            continue

        # Check that evs arrays exist on first matching combo
        if total_weight == 0:
            for asol in action_solutions:
                if not asol.get("evs") or len(asol["evs"]) != 1326:
                    return None

        total_weight += rng
        for asol in action_solutions:
            code = asol["action"]["code"]
            action_evs[code] = action_evs.get(code, 0) + asol["evs"][idx] * rng

    if total_weight < 0.005:
        return None

    for code in action_evs:
        action_evs[code] /= total_weight

    return action_evs if action_evs else None


def _get_preflop_hand_freqs(solution: dict, hero_hand: str, hero_pos: str) -> dict[str, float] | None:
    """Extract per-action frequencies for hero's hand from 169-element preflop arrays.

    Returns {action_code: frequency} or None if data unavailable.
    """
    if not solution or "action_solutions" not in solution:
        return None

    # Find hero's player info
    player_info = None
    for pi in solution["players_info"]:
        if pi["player"]["position"] == hero_pos:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    if len(range_arr) != 169:
        return None

    idx = HAND_TO_169.get(hero_hand)
    if idx is None:
        return None

    if range_arr[idx] < 0.005:
        # Hand not in range
        return None

    freqs = {}
    for asol in solution["action_solutions"]:
        code = asol["action"]["code"]
        freq = asol["strategy"][idx]
        if freq > 0.005:
            freqs[code] = freq
    return freqs if freqs else None


def _get_postflop_hand_freqs(solution: dict, hero_hand: str, hero_pos: str,
                             combo_idx: int | None = None) -> dict[str, float] | None:
    """Extract per-action frequencies for hero's hand from 1326-element postflop arrays.

    If combo_idx is provided, looks up the exact combo (e.g. Ah6h) directly.
    Otherwise averages across all combos of the hand name (e.g. all A6s).
    Returns {action_code: frequency} or None if data unavailable.
    """
    if not solution or "action_solutions" not in solution:
        return None

    player_info = None
    for pi in solution["players_info"]:
        if pi["player"]["position"] == hero_pos:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    if len(range_arr) != 1326:
        return None

    action_solutions = solution["action_solutions"]

    # Direct combo lookup — use exact combo strategy instead of averaging
    if combo_idx is not None and combo_idx < len(range_arr):
        rng = range_arr[combo_idx]
        if rng < 0.005:
            return None
        freqs = {}
        for asol in action_solutions:
            code = asol["action"]["code"]
            freq = asol["strategy"][combo_idx]
            if freq > 0.005:
                freqs[code] = freq
        return freqs if freqs else None

    # Fallback: average across all combos of the hand name
    board_cards = _get_board_cards(solution["game"]["board"])

    total_weight = 0
    action_freqs = {}

    for idx, (c1, c2) in enumerate(_COMBO_INDEX):
        if c1 in board_cards or c2 in board_cards:
            continue
        if _combo_to_hand_name(c1, c2) != hero_hand:
            continue
        rng = range_arr[idx]
        if rng < 0.005:
            continue
        total_weight += rng
        for asol in action_solutions:
            code = asol["action"]["code"]
            freq = asol["strategy"][idx]
            action_freqs[code] = action_freqs.get(code, 0) + freq * rng

    if total_weight < 0.005:
        return None

    # Normalize
    for code in action_freqs:
        action_freqs[code] /= total_weight

    return {k: v for k, v in action_freqs.items() if v > 0.005} or None


def _fmt_betsize(v) -> str:
    """Format betsize, stripping unnecessary trailing zeros."""
    return f"{float(v):.3f}".rstrip("0").rstrip(".")


def _get_action_label(action_solutions: list[dict], code: str) -> str:
    """Get human-readable label for an action code."""
    labels = {"X": "Check", "C": "Call", "F": "Fold"}
    if code in labels:
        return labels[code]
    for asol in action_solutions:
        if asol["action"]["code"] == code:
            act = asol["action"]
            if act.get("allin"):
                return f"All-in {_fmt_betsize(act.get('betsize', 0))}bb"
            name = act.get("display_name", code)
            betsize = act.get("betsize")
            if betsize and name.upper() in ("RAISE", "BET"):
                return f"{name} {_fmt_betsize(betsize)}bb"
            return name
    return code


def _normalize_preflop_action(code: str, gametype: str, depth: float,
                               preflop_so_far: str, stacks: str = "") -> str:
    """Map a raw preflop action code to the solver's action code."""
    if code in ("F", "C", "X"):
        return code
    try:
        resp = get_next_actions(gametype=gametype, depth=depth,
                                stacks=stacks, preflop_actions=preflop_so_far)
        avail = resp["next_actions"]["available_actions"]
        if not avail:
            return code
        if code == "AI" or (code.startswith("AI") and code == "AI"):
            allin = next((a["action"]["code"] for a in avail if a["action"].get("allin")), code)
            return allin
        if code.startswith("AI"):
            target = float(code[2:])
            return find_closest_action(avail, target)
        if code.startswith("R"):
            if code == "R":
                return find_unique_nonallin_raise(avail) or code
            target = float(code[1:])
            return find_closest_action(avail, target)
    except Exception:
        pass
    return code


def check_hand(hand: dict, icm_params: dict | None = None,
               emit_ungraded: bool = False) -> list[dict]:
    """Check a single hand for GTO deviations.

    Args:
        hand: parsed hand dict from hh_parser
        icm_params: optional ICM params dict with gametype, depth, stacks
                    (from icm_modes.find_icm_params). None = chip EV only.
        emit_ungraded: also emit stub dicts {"street", "ungraded": True,
                    "reason": "offrange"|"no_solution"} for hero decisions the
                    solver could not grade. Keeps per-street node ordering
                    aligned for callers that zip decisions positionally
                    (live flow); default False preserves legacy output.

    Returns list of deviation dicts, each containing:
        street, hero_action, hero_freq, gto_action, gto_freq, actions_detail
    """
    gametype = "MTTGeneral"
    icm_gametype = None
    icm_depth = None
    icm_stacks = None

    hero_pos = hand["hero_position"]
    hero_hand_raw = hand["hero_hand"]
    hero_hand = normalize_hand_name(hero_hand_raw)
    hero_combo_idx = _combo_index_for_hand(hero_hand_raw)
    num_players = hand.get("num_players", hand.get("table_size", 8))
    depth = nearest_depth(hand["effective_bb"])
    streets = hand.get("streets", [])

    # Use provided ICM params for preflop
    if icm_params and icm_params.get("gametype", "MTTGeneral") != "MTTGeneral":
        icm_gametype = icm_params["gametype"]
        icm_depth = icm_params["depth"]
        icm_stacks = icm_params.get("stacks", "")

    # Convert to 8-max
    pf_8max = _convert_preflop_to_8max(hand["preflop_actions"], num_players)

    pos_order_n = POSITION_ORDERS.get(num_players, POSITION_ORDER_8MAX)

    deviations = []

    # ── Preflop: hero's first decision ──
    hero_idx_n = pos_order_n.index(hero_pos)
    hero_idx_8 = POSITION_ORDER_8MAX.index(hero_pos)

    # Build preflop actions before hero (in 8-max format)
    pf_parts_8 = pf_8max.split("-")
    pf_before_hero = "-".join(pf_parts_8[:hero_idx_8]) if hero_idx_8 > 0 else ""

    # Choose preflop gametype/depth/stacks (ICM if available, else chip EV)
    pf_gametype = icm_gametype or gametype
    pf_depth = icm_depth or depth
    pf_stacks = icm_stacks or ""

    # Normalize preflop actions up to hero
    normalized_parts = []
    for i in range(hero_idx_8):
        code = pf_parts_8[i]
        so_far = "-".join(normalized_parts) if normalized_parts else ""
        norm_code = _normalize_preflop_action(code, pf_gametype, pf_depth, so_far, pf_stacks)
        normalized_parts.append(norm_code)
    pf_before_hero_norm = "-".join(normalized_parts) if normalized_parts else ""

    # Get hero's actual first preflop action
    pf_parts_n = hand["preflop_actions"].split("-")
    if hero_idx_n < len(pf_parts_n):
        hero_pf_action_raw = pf_parts_n[hero_idx_n]
    else:
        return deviations  # hero never acted

    # Normalize hero's action too
    hero_pf_action = _normalize_preflop_action(
        hero_pf_action_raw, pf_gametype, pf_depth, pf_before_hero_norm, pf_stacks
    )

    # Query solver for preflop
    try:
        sol = get_spot_solution(gametype=pf_gametype, depth=pf_depth,
                                stacks=pf_stacks, preflop_actions=pf_before_hero_norm)
    except Exception:
        sol = None

    # ICM fallback: if ICM query returned None, retry with chip EV
    if sol is None and icm_gametype and pf_gametype == icm_gametype:
        try:
            sol = get_spot_solution(gametype=gametype, depth=depth,
                                    preflop_actions=pf_before_hero_norm)
        except Exception:
            sol = None

    if sol:
        freqs = _get_preflop_hand_freqs(sol, hero_hand, hero_pos)
        if freqs:
            hero_freq = freqs.get(hero_pf_action, 0)
            best_action = max(freqs, key=freqs.get)
            best_freq = freqs[best_action]

            hand_ev = _get_hand_ev(sol, hero_hand, hero_pos, is_preflop=True)
            action_evs = _get_action_evs_preflop(sol, hero_hand, hero_pos)
            ev_entry = {}
            if action_evs:
                hero_act_ev = action_evs.get(hero_pf_action)
                best_act_ev = max(action_evs.values()) if action_evs else None
                if hero_act_ev is not None and best_act_ev is not None:
                    ev_entry = {
                        "hero_action_ev": hero_act_ev,
                        "best_action_ev": best_act_ev,
                        "ev_loss": best_act_ev - hero_act_ev,
                        "action_evs": action_evs,
                    }
            deviations.append({
                "street": "preflop",
                "spot": "open/first action",
                "hero_action": hero_pf_action,
                "hero_action_label": _get_action_label(sol["action_solutions"], hero_pf_action),
                "hero_freq": hero_freq,
                "gto_action": best_action,
                "gto_action_label": _get_action_label(sol["action_solutions"], best_action),
                "gto_freq": best_freq,
                "all_freqs": freqs,
                "hero_ev": hand_ev,
                **ev_entry,
            })
        elif emit_ungraded:
            deviations.append({"street": "preflop", "ungraded": True,
                               "reason": "offrange"})
    elif emit_ungraded:
        deviations.append({"street": "preflop", "ungraded": True,
                           "reason": "no_solution"})

    # ── Preflop: hero's second decision (if facing re-raise) ──
    # Check if someone raised after hero
    has_reraise = False
    for i in range(hero_idx_n + 1, min(len(pf_parts_n), num_players)):
        if pf_parts_n[i].startswith("R") or pf_parts_n[i].startswith("AI"):
            has_reraise = True
            break

    if has_reraise and len(pf_parts_n) > num_players:
        # Normalize full first round in 8-max
        full_first_round = []
        for i in range(min(len(pf_parts_8), 8)):
            code = pf_parts_8[i]
            so_far = "-".join(full_first_round) if full_first_round else ""
            norm_code = _normalize_preflop_action(code, pf_gametype, pf_depth, so_far, pf_stacks)
            full_first_round.append(norm_code)

        # Include every intervening continuation action so the solver query
        # lands on hero's actual node (e.g. CO folds before BTN faces squeeze).
        before_hero_cont, hero_cont_raw = _hero_continuation_context(
            pf_parts_n, num_players, hero_idx_n)

        if hero_cont_raw:
            second_prefix_parts = list(full_first_round)
            for code in before_hero_cont:
                so_far = "-".join(second_prefix_parts)
                second_prefix_parts.append(_normalize_preflop_action(
                    code, pf_gametype, pf_depth, so_far, pf_stacks))
            second_prefix = "-".join(second_prefix_parts)
            hero_cont = _normalize_preflop_action(
                hero_cont_raw, pf_gametype, pf_depth, second_prefix, pf_stacks)
            try:
                sol2 = get_spot_solution(gametype=pf_gametype, depth=pf_depth,
                                          stacks=pf_stacks, preflop_actions=second_prefix)
            except Exception:
                sol2 = None

            # ICM fallback for re-raise spot
            if sol2 is None and icm_gametype and pf_gametype == icm_gametype:
                try:
                    sol2 = get_spot_solution(gametype=gametype, depth=depth,
                                              preflop_actions=second_prefix)
                except Exception:
                    sol2 = None

            if sol2:
                freqs2 = _get_preflop_hand_freqs(sol2, hero_hand, hero_pos)
                if freqs2:
                    hero_freq2 = freqs2.get(hero_cont, 0)
                    best2 = max(freqs2, key=freqs2.get)
                    best_freq2 = freqs2[best2]

                    hand_ev2 = _get_hand_ev(sol2, hero_hand, hero_pos, is_preflop=True)
                    action_evs2 = _get_action_evs_preflop(sol2, hero_hand, hero_pos)
                    ev_entry2 = {}
                    if action_evs2:
                        hero_act_ev2 = action_evs2.get(hero_cont)
                        best_act_ev2 = max(action_evs2.values()) if action_evs2 else None
                        if hero_act_ev2 is not None and best_act_ev2 is not None:
                            ev_entry2 = {
                                "hero_action_ev": hero_act_ev2,
                                "best_action_ev": best_act_ev2,
                                "ev_loss": best_act_ev2 - hero_act_ev2,
                                "action_evs": action_evs2,
                            }
                    deviations.append({
                        "street": "preflop",
                        "spot": "facing 3bet/4bet",
                        "hero_action": hero_cont,
                        "hero_action_label": _get_action_label(sol2["action_solutions"], hero_cont),
                        "hero_freq": hero_freq2,
                        "gto_action": best2,
                        "gto_action_label": _get_action_label(sol2["action_solutions"], best2),
                        "gto_freq": best_freq2,
                        "all_freqs": freqs2,
                        "hero_ev": hand_ev2,
                        **ev_entry2,
                    })
                elif emit_ungraded:
                    deviations.append({"street": "preflop", "ungraded": True,
                                       "reason": "offrange"})
            elif emit_ungraded:
                deviations.append({"street": "preflop", "ungraded": True,
                                   "reason": "no_solution"})

    # ── Postflop streets ──
    if not streets:
        return deviations

    # Build full normalized preflop for postflop queries
    all_norm = []
    for i in range(len(pf_parts_8)):
        code = pf_parts_8[i]
        so_far = "-".join(all_norm) if all_norm else ""
        norm_code = _normalize_preflop_action(code, gametype, depth, so_far)
        all_norm.append(norm_code)
    full_preflop_norm = "-".join(all_norm)

    board = ""
    flop_acts = ""
    turn_acts = ""
    river_acts = ""

    for street_idx, street in enumerate(streets):
        street_name = ["flop", "turn", "river"][street_idx]

        if street_idx == 0:
            board = street.get("board") or street.get("cards") or street.get("card", "")
        else:
            card = street.get("card") or street.get("cards", "")
            board += card

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)
            pot_fraction = act.get("pot_fraction")

            if pos == hero_pos:
                # Query solver at this point
                params = dict(
                    gametype=gametype, depth=depth,
                    preflop_actions=full_preflop_norm, board=board,
                    flop_actions=flop_acts, turn_actions=turn_acts,
                    river_actions=river_acts,
                )

                # Determine hero's taken action code
                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                elif action_type == "AI":
                    try:
                        next_resp = get_next_actions(**params)
                        avail = next_resp["next_actions"]["available_actions"]
                        allin = next((a["action"]["code"] for a in avail if a["action"].get("allin")), None)
                        taken_code = allin or find_closest_action_postflop(avail, target_size)
                    except Exception:
                        taken_code = action_type
                else:
                    try:
                        next_resp = get_next_actions(**params)
                        avail = next_resp["next_actions"]["available_actions"]
                        if pot_fraction is not None:
                            taken_code = find_closest_action_by_pot_fraction(
                                avail, pot_fraction)
                        else:
                            taken_code = find_closest_action_postflop(
                                avail, target_size)
                    except Exception:
                        taken_code = action_type

                try:
                    sol_post = get_spot_solution(**params)
                except Exception:
                    sol_post = None

                if sol_post:
                    # Remap "C" → all-in when solver has no Call but has all-in
                    # (happens when hero calls an all-in and is effectively all-in)
                    if taken_code == "C":
                        has_call = any(
                            a["action"]["code"] == "C"
                            for a in sol_post["action_solutions"]
                        )
                        if not has_call:
                            allin_code = next(
                                (a["action"]["code"] for a in sol_post["action_solutions"]
                                 if a["action"].get("allin")),
                                None,
                            )
                            if allin_code:
                                taken_code = allin_code

                    freqs_post = _get_postflop_hand_freqs(sol_post, hero_hand, hero_pos,
                                                          combo_idx=hero_combo_idx)
                    if freqs_post:
                        hero_freq_post = freqs_post.get(taken_code, 0)
                        best_post = max(freqs_post, key=freqs_post.get)
                        best_freq_post = freqs_post[best_post]

                        hand_ev_post = _get_hand_ev(sol_post, hero_hand, hero_pos,
                                                    is_preflop=False, combo_idx=hero_combo_idx)
                        action_evs_post = _get_action_evs_postflop(
                            sol_post, hero_hand, hero_pos, combo_idx=hero_combo_idx)
                        ev_entry_post = {}
                        if action_evs_post:
                            hero_act_ev_p = action_evs_post.get(taken_code)
                            best_act_ev_p = max(action_evs_post.values())
                            if hero_act_ev_p is not None and best_act_ev_p is not None:
                                ev_entry_post = {
                                    "hero_action_ev": hero_act_ev_p,
                                    "best_action_ev": best_act_ev_p,
                                    "ev_loss": best_act_ev_p - hero_act_ev_p,
                                    "action_evs": action_evs_post,
                                }
                        deviations.append({
                            "street": street_name,
                            "spot": f"board {cards_to_emoji(board)}",
                            "hero_action": taken_code,
                            "hero_action_label": _get_action_label(sol_post["action_solutions"], taken_code),
                            "hero_freq": hero_freq_post,
                            "gto_action": best_post,
                            "gto_action_label": _get_action_label(sol_post["action_solutions"], best_post),
                            "gto_freq": best_freq_post,
                            "all_freqs": freqs_post,
                            "hero_ev": hand_ev_post,
                            **ev_entry_post,
                        })
                    elif emit_ungraded:
                        deviations.append({"street": street_name, "ungraded": True,
                                           "reason": "offrange"})
                elif emit_ungraded:
                    deviations.append({"street": street_name, "ungraded": True,
                                       "reason": "no_solution"})

            # Advance action strings
            if action_type in ("X", "C", "F"):
                taken = action_type
            elif action_type == "AI":
                try:
                    params_adv = dict(
                        gametype=gametype, depth=depth,
                        preflop_actions=full_preflop_norm, board=board,
                        flop_actions=flop_acts, turn_actions=turn_acts,
                        river_actions=river_acts,
                    )
                    next_resp = get_next_actions(**params_adv)
                    avail = next_resp["next_actions"]["available_actions"]
                    allin = next((a["action"]["code"] for a in avail if a["action"].get("allin")), None)
                    taken = allin or find_closest_action_postflop(avail, target_size)
                except Exception:
                    taken = action_type
            else:
                try:
                    params_adv = dict(
                        gametype=gametype, depth=depth,
                        preflop_actions=full_preflop_norm, board=board,
                        flop_actions=flop_acts, turn_actions=turn_acts,
                        river_actions=river_acts,
                    )
                    next_resp = get_next_actions(**params_adv)
                    avail = next_resp["next_actions"]["available_actions"]
                    if pot_fraction is not None:
                        taken = find_closest_action_by_pot_fraction(
                            avail, pot_fraction)
                    else:
                        taken = find_closest_action_postflop(
                            avail, target_size)
                except Exception:
                    taken = action_type

            if street_idx == 0:
                flop_acts = f"{flop_acts}-{taken}" if flop_acts else taken
            elif street_idx == 1:
                turn_acts = f"{turn_acts}-{taken}" if turn_acts else taken
            elif street_idx == 2:
                river_acts = f"{river_acts}-{taken}" if river_acts else taken

    return deviations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check HH for GTO deviations")
    parser.add_argument("path", help="Directory or file with hand histories")
    parser.add_argument("--limit", "-n", type=int, default=0,
                       help="Limit number of hands (0 = all)")
    parser.add_argument("--threshold", "-t", type=float, default=10,
                       help="Minimum deviation %% to report (default: 10)")
    parser.add_argument("--output", "-o", default="hh_deviations.json",
                       help="Output JSON file")
    parser.add_argument("--delay", "-d", type=float, default=0.3,
                       help="Delay between hands in seconds")
    parser.add_argument("--starting-stack", "-s", type=int, default=0,
                       help="Starting stack in chips (enables ICM analysis)")
    parser.add_argument("--tournament-size", type=int, default=1000,
                       help="Tournament size: 1000 or 200 (default: 1000)")
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        hands = parse_directory(path, include_folds=True)
    else:
        hands = parse_file(path, include_folds=True)

    print(f"Parsed {len(hands)} hero hands (including preflop folds)\n")

    # Auto-detect starting_stack per tournament from earliest hand's hero chips
    # BEFORE applying --limit, so we see the full file
    # GGPoker HH files list newest first, so the last hand per tournament is the earliest
    starting_stack_by_tournament: dict[str, int] = {}
    if args.starting_stack == 0:
        for hand in reversed(hands):
            tid = hand.get("tournament_id", "")
            if tid and tid not in starting_stack_by_tournament and "hero_chips" in hand:
                starting_stack_by_tournament[tid] = hand["hero_chips"]

    if args.limit:
        hands = hands[:args.limit]

    if starting_stack_by_tournament:
        print("ICM mode (auto-detect starting stacks):")
        for tid, ss in starting_stack_by_tournament.items():
            print(f"  Tournament #{tid}: starting_stack={ss}")
        print()
    elif args.starting_stack > 0:
        print(f"ICM mode: starting_stack={args.starting_stack}, tournament_size={args.tournament_size}\n")

    all_results = []
    all_deviations = []
    max_ratio_by_tournament: dict[str, float] = {}

    for i, hand in enumerate(hands):
        hand_id = hand.get("hand_id", "?")
        hero_pos = hand["hero_position"]
        hero_hand = hand["hero_hand"]
        eff_bb = hand["effective_bb"]
        num_streets = len(hand.get("streets", []))

        # Compute ICM params
        icm_params = None
        tid = hand.get("tournament_id", "")
        effective_ss = args.starting_stack or starting_stack_by_tournament.get(tid, 0)
        if effective_ss > 0 and "avg_stack_chips" in hand and "stacks_bb" in hand:
            from icm_modes import find_icm_params
            raw_ratio = hand["avg_stack_chips"] / effective_ss
            if tid:
                prev_max = max_ratio_by_tournament.get(tid, 0)
                ratio = max(raw_ratio, prev_max)
                max_ratio_by_tournament[tid] = ratio
            else:
                ratio = raw_ratio
            table_size = hand.get("table_size", 8)
            est_remaining = max(table_size, min(args.tournament_size, args.tournament_size / ratio))
            # Pad stacks if short-handed
            stacks = list(hand["stacks_bb"])
            if len(stacks) < table_size:
                avg_bb = sum(stacks) / len(stacks) if stacks else 20
                stacks.extend([avg_bb] * (table_size - len(stacks)))
            icm_result = find_icm_params(
                player_stacks=stacks,
                tournament_size=args.tournament_size,
                players_remaining=int(round(est_remaining)),
            )
            if icm_result["gametype"] != "MTTGeneral":
                icm_params = icm_result

        icm_tag = f" ICM={icm_params['gametype'].split('PT')[-1]}" if icm_params else ""
        print(f"[{i+1}/{len(hands)}] {hand_id}: {hero_pos} {cards_to_emoji(hero_hand)} ({eff_bb:.1f}bb) "
              f"streets={num_streets}{icm_tag}", end=" ... ", flush=True)

        t0 = time.time()
        try:
            devs = check_hand(hand, icm_params=icm_params)
            elapsed = time.time() - t0

            # Identify significant deviations:
            # Only flag when hero's action differs from GTO dominant action
            # AND hero's action frequency is below threshold
            significant = []
            for d in devs:
                is_dominant = d["hero_action"] == d["gto_action"]
                hero_pct = d["hero_freq"] * 100
                if not is_dominant and hero_pct < (100 - args.threshold):
                    significant.append(d)

            result = {
                "hand_id": hand_id,
                "file": hand.get("file", ""),
                "hero_position": hero_pos,
                "hero_hand": hero_hand,
                "hero_hand_normalized": normalize_hand_name(hero_hand),
                "effective_bb": eff_bb,
                "num_players": hand.get("num_players", 8),
                "preflop_actions": hand["preflop_actions"],
                "spots_checked": len(devs),
                "deviations": devs,
                "elapsed_s": round(elapsed, 1),
            }
            all_results.append(result)

            # Collect significant deviations
            for d in significant:
                all_deviations.append({
                    **d,
                    "hand_id": hand_id,
                    "hero_position": hero_pos,
                    "hero_hand": normalize_hand_name(hero_hand),
                    "effective_bb": eff_bb,
                })

            dev_count = len(significant)
            spot_count = len(devs)
            status = f"ok ({spot_count} spots, {dev_count} devs, {elapsed:.1f}s)"
            print(status)

        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            print(f"ERROR ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            result = {
                "hand_id": hand_id,
                "hero_position": hero_pos,
                "hero_hand": hero_hand,
                "effective_bb": eff_bb,
                "error": str(e),
                "deviations": [],
                "elapsed_s": round(elapsed, 1),
            }
            all_results.append(result)

        if i < len(hands) - 1:
            time.sleep(args.delay)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("GTO Deviation Report")
    print("=" * 70)

    # Sort deviations by EV loss (largest first), then by hero_freq
    all_deviations.sort(key=lambda d: (-d.get("ev_loss", 0), d["hero_freq"]))

    total_ev_loss = sum(d.get("ev_loss", 0) for d in all_deviations)

    if not all_deviations:
        print(f"\nNo significant deviations found (threshold: {args.threshold}%)")
    else:
        print(f"\nFound {len(all_deviations)} spots where hero deviated from GTO:")
        print(f"(threshold: hero's action had <{100 - args.threshold:.0f}% GTO frequency)")
        if total_ev_loss > 0.005:
            print(f"Total EV lost: {total_ev_loss:.2f}bb")
        print()

        for d in all_deviations:
            hero_pct = d["hero_freq"] * 100
            gto_pct = d["gto_freq"] * 100
            ev_loss = d.get("ev_loss")
            ev_loss_str = f" EV loss: {ev_loss:.2f}bb" if ev_loss is not None else ""
            hero_ev_str = f" EV={d.get('hero_action_ev', d.get('hero_ev', 0)):.2f}bb" if d.get("hero_action_ev") is not None or d.get("hero_ev") is not None else ""
            print(f"  {d['hand_id']}: {d['hero_position']} {cards_to_emoji(d['hero_hand'])} "
                  f"({d['effective_bb']:.0f}bb) [{d['street']}]{hero_ev_str}"
                  f" →{ev_loss_str}")
            print(f"    Hero: {d['hero_action_label']} ({hero_pct:.0f}% GTO)")
            print(f"    GTO:  {d['gto_action_label']} ({gto_pct:.0f}%)")
            # Show all action frequencies
            freq_str = ", ".join(
                f"{k}={v*100:.0f}%"
                for k, v in sorted(d["all_freqs"].items(), key=lambda x: -x[1])
            )
            print(f"    All:  {freq_str}")
            print()

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
