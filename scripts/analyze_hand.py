#!/usr/bin/env python3
"""Analyze a poker hand against GTO Wizard solutions.

Usage:
    python scripts/analyze_hand.py --json '<hand_json>'

Hand JSON format:
{
    "gametype": "MTTGeneral",
    "effective_bb": 32,
    "hero_position": "CO",
    "hero_hand": "66",
    "preflop_actions": "F-F-F-F-R2-F-F-C",
    "streets": [
        {"board": "Js6h5s", "actions": [{"position": "BB", "action": "X"}, {"position": "CO", "action": "R2", "size": 2.0}]},
        {"card": "Kc", "actions": [{"position": "BB", "action": "X"}, {"position": "CO", "action": "R6.6", "size": 6.6}]},
        {"card": "2s", "actions": [{"position": "BB", "action": "X"}, {"position": "CO", "action": "X"}]}
    ]
}

Output: Natural language analysis for each street.
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_api import (
    get_spot_solution, get_next_actions,
    find_closest_action, find_closest_action_postflop, nearest_depth,
    nearest_cash_depth,
)
from gto_formatter import format_full_spot, format_ev_comparison, normalize_hand_name, combo_index_for_hand

POSITION_ORDER = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]

# Position orders by table size (GTO Wizard convention)
POSITION_ORDERS = {
    9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
    6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
    5: ["HJ", "CO", "BTN", "SB", "BB"],
    4: ["CO", "BTN", "SB", "BB"],
    3: ["BTN", "SB", "BB"],
    2: ["SB", "BB"],
}

CASH_GAMETYPES = {
    2: "CashHuGeneral_NL100R2",
    3: "Cash3mGeneral_3mGGAIorFoldcEV",
    4: "Cash4mGeneral_4mAnte075NL100R2",
    5: "Cash5mGeneral_5mWMXAIorFoldcEV",
    6: "Cash6mGeneral_6mNL100R2",
    7: "Cash7mGeneral_7mAnte01NL25R2",
    8: "Cash8mLiveGeneral_8mLIVER2",
    9: "Cash9mGGAnteGeneral_9mGGANTEcEVR2",
}


def _get_position_order(num_players: int = 8) -> list[str]:
    """Get position order for given table size."""
    return POSITION_ORDERS.get(num_players, POSITION_ORDER)


# Position alias mapping: common poker client names → GTO Wizard names
# MP (Middle Position) maps to LJ in most table sizes
POSITION_ALIASES = {
    "MP": "LJ",
    "MP1": "LJ",
    "MP2": "HJ",
    "EP": "UTG",
    "EP1": "UTG",
    "EP2": "UTG+1",
}


def _normalize_positions(hand: dict) -> dict:
    """Normalize position aliases (MP, EP, etc.) to GTO Wizard names."""
    aliases = POSITION_ALIASES
    hero_pos = hand.get("hero_position", "")
    if hero_pos in aliases:
        hand = dict(hand)
        hand["hero_position"] = aliases[hero_pos]
    # Normalize positions in street actions
    streets = hand.get("streets") or hand.get("postflop_actions", [])
    needs_fix = any(
        a.get("position") in aliases
        for st in streets
        for a in st.get("actions", [])
    )
    if needs_fix:
        hand = dict(hand)
        new_streets = []
        for st in streets:
            new_actions = []
            for a in st.get("actions", []):
                if a.get("position") in aliases:
                    a = dict(a, position=aliases[a["position"]])
                new_actions.append(a)
            new_streets.append(dict(st, actions=new_actions))
        if "streets" in hand:
            hand["streets"] = new_streets
        else:
            hand["postflop_actions"] = new_streets
    return hand


def _normalize_preflop_actions(preflop_actions: str, gametype: str, depth: float, stacks: str = "") -> str:
    """Validate and correct preflop action codes against the solver.

    LLM may output R2 but solver expects R2.1. Walk through each action,
    and for raises, discover the correct code via next-actions API.
    """
    parts = preflop_actions.split("-")
    corrected = []
    for i, code in enumerate(parts):
        if code == "F":
            corrected.append(code)
        elif code == "C":
            # BB checking option after SB limp requires "X", not "C".
            # Only check the LAST "C" in the sequence (BB's position) to avoid
            # unnecessary API calls — earlier C's are always genuine calls.
            is_last_c = all(p != "C" for p in parts[i + 1:])
            if is_last_c and i > 0:
                actions_so_far = "-".join(corrected) if corrected else ""
                try:
                    resp = get_next_actions(
                        gametype=gametype, depth=depth, stacks=stacks,
                        preflop_actions=actions_so_far,
                    )
                    avail = resp["next_actions"]["available_actions"]
                    avail_codes = {a["action"]["code"] for a in avail} if avail else set()
                    if "X" in avail_codes and "C" not in avail_codes:
                        corrected.append("X")
                    else:
                        corrected.append("C")
                except Exception:
                    corrected.append(code)
            else:
                corrected.append("C")
        elif code == "AI" or code.startswith("AI"):
            # AI = all-in (no size), AI10 = all-in for 10bb (treat as raise)
            actions_so_far = "-".join(corrected) if corrected else ""
            try:
                resp = get_next_actions(
                    gametype=gametype, depth=depth, stacks=stacks,
                    preflop_actions=actions_so_far,
                )
                avail = resp["next_actions"]["available_actions"]
                if not avail:
                    # No solver data (e.g. multiway) — use RAI for all-in
                    corrected.append("RAI")
                elif code == "AI":
                    allin_code = next(
                        (a["action"]["code"] for a in avail if a["action"].get("allin")),
                        code,
                    )
                    corrected.append(allin_code)
                else:
                    target = float(code[2:])  # AI10 → 10.0
                    correct_code = find_closest_action(avail, target)
                    corrected.append(correct_code)
            except Exception:
                corrected.append(code)
        elif code.startswith("R"):
            # Raise — discover correct code from solver
            actions_so_far = "-".join(corrected) if corrected else ""
            try:
                resp = get_next_actions(
                    gametype=gametype, depth=depth, stacks=stacks,
                    preflop_actions=actions_so_far,
                )
                avail = resp["next_actions"]["available_actions"]
                if not avail:
                    corrected.append(code)
                else:
                    target = float(code[1:])  # R2 → 2.0, R2.1 → 2.1
                    correct_code = find_closest_action(avail, target)
                    corrected.append(correct_code)
            except Exception:
                corrected.append(code)  # fallback to original
        else:
            corrected.append(code)
    return "-".join(corrected)


def _preflop_before_hero(preflop_actions: str, hero_position: str, position_order: list[str] | None = None) -> str:
    """Get preflop action string up to (but not including) hero's action."""
    pos_order = position_order or POSITION_ORDER
    parts = preflop_actions.split("-")
    hero_idx = pos_order.index(hero_position)
    before = parts[:hero_idx]
    return "-".join(before) if before else ""


STREET_NAMES = ["flop", "turn", "river"]

# Action display names for explanatory messages
_ACTION_LABELS = {
    "X": "Check", "C": "Call", "F": "Fold", "RAI": "All-in",
}


def _get_hero_action_freq(solution: dict, action_code: str, hero_hand: str, hero_pos: str,
                          combo_idx: int | None = None) -> tuple[float | None, str]:
    """Get hero's hand-specific frequency for a given action from a solution.

    If combo_idx is provided, looks up the exact combo (e.g. Ah6h) directly.
    Otherwise averages across all combos of the hand name (e.g. all A6s).
    Returns (frequency_0_to_1, gto_recommendation_str).
    frequency is None if data unavailable.
    """
    if not solution or "action_solutions" not in solution:
        return None, ""

    from gto_formatter import _get_combo_strategies, _COMBO_INDEX, _get_board_cards, _combo_to_hand_name

    action_solutions = solution["action_solutions"]

    # Find the target action's index
    target_asol = None
    for asol in action_solutions:
        if asol["action"]["code"] == action_code:
            target_asol = asol
            break
    if not target_asol:
        return None, ""

    # Try hand-specific frequency from strategy arrays
    player_info = None
    for pi in solution["players_info"]:
        if pi["player"]["position"] == hero_pos:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return target_asol.get("total_frequency"), ""

    range_arr = player_info["range"]
    if len(range_arr) != 1326:
        return target_asol.get("total_frequency"), ""

    action_freqs = {}  # code → freq

    # Direct combo lookup — use exact combo strategy
    if combo_idx is not None and combo_idx < len(range_arr):
        rng = range_arr[combo_idx]
        if rng < 0.005:
            return target_asol.get("total_frequency"), ""
        for asol in action_solutions:
            code = asol["action"]["code"]
            freq = asol["strategy"][combo_idx]
            if freq > 0.005:
                action_freqs[code] = freq
    else:
        # Fallback: average across all combos of the hand name
        board_cards = _get_board_cards(solution["game"]["board"])
        total_weight = 0

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
            return target_asol.get("total_frequency"), ""

        # Normalize
        for code in action_freqs:
            action_freqs[code] /= total_weight

    if not action_freqs:
        return target_asol.get("total_frequency"), ""

    hero_freq = action_freqs.get(action_code, 0)

    # Build GTO recommendation string (top action)
    best_code = max(action_freqs, key=action_freqs.get)
    best_freq = action_freqs[best_code]

    # Get display label
    best_asol = next((a for a in action_solutions if a["action"]["code"] == best_code), None)
    if best_asol:
        act = best_asol["action"]
        if act.get("allin"):
            label = "All-in"
        elif act["code"] in _ACTION_LABELS:
            label = _ACTION_LABELS[act["code"]]
        else:
            label = act.get("display_name", act["code"])
    else:
        label = best_code

    gto_rec = f"GTO 建議 {best_freq*100:.0f}% {label}"
    return hero_freq, gto_rec


def _compute_preflop_pot(preflop_actions: str, effective_bb: float) -> float:
    """Compute the pot at the start of the flop from original preflop actions."""
    parts = preflop_actions.split("-")

    # Initial: SB posts 0.5, BB posts 1.0
    investments = [0.0] * 8
    investments[6] = 0.5  # SB
    investments[7] = 1.0  # BB
    current_bet = 1.0  # BB is the initial bet to match

    for i in range(min(len(parts), 8)):
        code = parts[i]
        if code in ("F", ""):
            pass
        elif code == "C":
            investments[i] = current_bet
        elif code.startswith("R"):
            try:
                investments[i] = float(code[1:])
                current_bet = investments[i]
            except ValueError:
                pass
        elif code == "AI":
            investments[i] = effective_bb
            current_bet = effective_bb

    # Continuation actions (re-raises after initial 8 positions)
    if len(parts) > 8:
        active = [i for i in range(8) if parts[i] not in ("F", "")]
        cont_idx = 0
        for j in range(8, len(parts)):
            if cont_idx >= len(active):
                cont_idx = 0
            pos = active[cont_idx]
            code = parts[j]
            if code == "C":
                investments[pos] = current_bet
            elif code.startswith("R"):
                try:
                    investments[pos] = float(code[1:])
                    current_bet = investments[pos]
                except ValueError:
                    pass
            elif code == "AI":
                investments[pos] = effective_bb
                current_bet = effective_bb
            cont_idx += 1

    return sum(investments)


def _find_action_by_pot_pct(available_actions: list, bet_size: float, actual_pot: float) -> str:
    """Find closest action by pot percentage rather than absolute size.

    Computes the hero/villain bet as a fraction of the actual pot, then
    converts to the solver's pot context for matching.
    """
    target_pct = bet_size / actual_pot

    # Compute solver pot from any available raise action's betsize_by_pot
    solver_pot = None
    for entry in available_actions:
        action = entry["action"]
        pct = action.get("betsize_by_pot")
        if pct and float(pct) > 0:
            solver_pot = float(action["betsize"]) / float(pct)
            break

    if solver_pot:
        solver_bet = target_pct * solver_pot
        return find_closest_action(available_actions, solver_bet)

    # Fallback to absolute matching
    return find_closest_action(available_actions, bet_size)


def _simplify_multiway(hand: dict, hero_pos: str, gametype: str, depth: float) -> tuple[str, float, str, set[str] | None]:
    """Detect multiway pot and simplify to heads-up if needed.

    Returns (preflop_actions, adjusted_depth, simplification_note, active_positions).
    active_positions is the set of 2 positions in the simplified HU, or None if no simplification.
    If not multiway, returns (original_preflop, original_depth, "", None).
    """
    preflop = hand["preflop_actions"]
    streets = hand.get("streets") or hand.get("postflop_actions", [])
    parts = preflop.split("-")

    # Count non-fold actions in first 8 positions
    non_fold = [i for i in range(min(len(parts), 8)) if parts[i] not in ("F", "")]
    if len(non_fold) <= 2:
        return preflop, depth, "", None

    # Multiway — find earliest point where hand becomes HU involving hero.
    # Walk street by street: track who's still in. As soon as it drops to
    # 2 players including hero, simplify to that HU.
    if streets:
        active = set()
        folded = set()
        # Collect all positions that appear in any street
        for street in streets:
            for act in street.get("actions", []):
                active.add(act["position"])

        # Walk streets in order to find when it becomes HU
        villain_pos = None
        for street in streets:
            for act in street.get("actions", []):
                if act["action"] == "F":
                    folded.add(act["position"])

            remaining = [p for p in active if p not in folded]
            if len(remaining) == 2 and hero_pos in remaining:
                villain_pos = next(p for p in remaining if p != hero_pos)
                break
            elif len(remaining) == 1 and hero_pos in remaining:
                # Only hero remains — everyone else folded.
                # Find the last non-hero opponent who was active (even if they folded)
                for act in reversed(street.get("actions", [])):
                    if act["position"] != hero_pos and act["action"] not in ("X",):
                        villain_pos = act["position"]
                        break
                if villain_pos:
                    break

        if not villain_pos:
            # Still multiway at end of all streets, or no villain found
            return preflop, depth, "", None
    else:
        # Preflop-only: solver handles multiway preflop natively, no simplification needed
        return preflop, depth, "", None
    hero_idx = POSITION_ORDER.index(hero_pos)
    villain_idx = POSITION_ORDER.index(villain_pos)

    # Determine open/3bet structure by preflop position order
    if villain_idx < hero_idx:
        first_pos, first_idx = villain_pos, villain_idx
        second_pos, second_idx = hero_pos, hero_idx
    else:
        first_pos, first_idx = hero_pos, hero_idx
        second_pos, second_idx = villain_pos, villain_idx

    # Check pot type from original preflop
    second_action = parts[second_idx] if second_idx < len(parts) else "C"
    is_3bet = second_action.startswith("R") or second_action.startswith("AI")

    # Check for 4bet+ in continuation actions (parts[8:])
    fourbet_size = None
    if is_3bet:
        for p in parts[8:]:
            if p.startswith("R"):
                try:
                    fourbet_size = float(p[1:])
                except ValueError:
                    pass
                break

    # Estimate dead money from extra callers to adjust effective BB
    open_size = 2.0  # default
    for p in parts[:8]:
        if p.startswith("R"):
            try:
                open_size = float(p[1:])
            except ValueError:
                pass
            break
    extra_callers = len(non_fold) - 2  # hero + villain = 2
    # In 3bet/4bet pots, dead money is amplified: callers' money inflates the pot,
    # causing larger sizing and calls, roughly 3x the raw dead money
    amplifier = 3.0 if is_3bet else 1.0
    dead_money = extra_callers * open_size * amplifier
    adjusted_eff = hand["effective_bb"] - dead_money
    adjusted_depth = nearest_depth(adjusted_eff)

    # Build simplified preflop via API walk-through
    simplified = ["F"] * 8

    # First actor opens
    prefix = "-".join(simplified[:first_idx])
    try:
        resp = get_next_actions(gametype=gametype, depth=adjusted_depth,
                                preflop_actions=prefix if prefix else "")
        avail = resp["next_actions"]["available_actions"]
        raises = [a for a in avail
                  if a["action"]["code"].startswith("R") and not a["action"].get("allin")]
        first_code = raises[0]["action"]["code"] if raises else "R2"
    except Exception:
        first_code = "R2"
    simplified[first_idx] = first_code

    if is_3bet:
        # Second actor raised/all-in → 3bet pot
        prefix = "-".join(simplified[:second_idx])
        try:
            resp = get_next_actions(gametype=gametype, depth=adjusted_depth,
                                    preflop_actions=prefix)
            avail = resp["next_actions"]["available_actions"]
            if second_action.startswith("AI"):
                # All-in: find solver's all-in code, or closest raise to the size
                allin_code = next(
                    (a["action"]["code"] for a in avail if a["action"].get("allin")),
                    None,
                )
                if allin_code:
                    second_code = allin_code
                elif second_action != "AI":
                    second_code = find_closest_action(avail, float(second_action[2:]))
                else:
                    second_code = find_closest_action(avail, adjusted_depth - 0.125)
            else:
                second_size = float(second_action[1:])
                second_code = find_closest_action(avail, second_size)
        except Exception:
            second_code = second_action
        simplified[second_idx] = second_code

        if fourbet_size is not None:
            # 4bet pot: first actor re-raises, second calls
            base = "-".join(simplified)
            try:
                resp = get_next_actions(gametype=gametype, depth=adjusted_depth,
                                        preflop_actions=base)
                avail = resp["next_actions"]["available_actions"]
                fourbet_code = find_closest_action(avail, fourbet_size)
            except Exception:
                fourbet_code = f"R{fourbet_size}"
            full = base + f"-{fourbet_code}-C"
            pot_type = "4bet"
        else:
            # 3bet pot: first actor calls
            full = "-".join(simplified) + "-C"
            pot_type = "3bet"
    else:
        # Single raised pot
        simplified[second_idx] = "C"
        full = "-".join(simplified)
        pot_type = "call"

    note_parts = [
        f"⚠ 多人底池，簡化為 {first_pos} open vs {second_pos} {pot_type} 單挑分析",
    ]
    if adjusted_depth != depth:
        note_parts.append(
            f"有效籌碼調整: {hand['effective_bb']}bb → {adjusted_eff:.0f}bb"
            f"（扣除底池死錢 ~{dead_money:.0f}bb）"
        )

    return full, adjusted_depth, "\n".join(note_parts), {first_pos, second_pos}


def _explain_missing_solution(
    spot_idx: int, hero_spots: list, solutions: list,
    hero_hand: str, hero_pos: str, combo_idx: int | None = None,
) -> str | None:
    """Explain why a spot has no solver data by checking previous hero actions.

    When the solver doesn't have a solution, it's often because a previous
    hero action had very low GTO frequency (e.g., calling when should all-in).
    """
    # Search backwards for the most recent hero spot with a solution and a taken action
    for j in range(spot_idx - 1, -1, -1):
        prev_sol = solutions[j]
        prev_spot = hero_spots[j]
        taken_code = prev_spot.get("taken_code")
        if not prev_sol or not taken_code:
            continue

        freq, gto_rec = _get_hero_action_freq(prev_sol, taken_code, hero_hand, hero_pos,
                                                combo_idx=combo_idx)
        if freq is not None and freq < 0.10:
            # Hero's action was rare — explain it
            taken_label = _ACTION_LABELS.get(taken_code, taken_code)
            street_label = prev_spot["street"].capitalize()
            pct = freq * 100
            return (
                f"（{street_label} hero {taken_label} 頻率僅 {pct:.1f}%"
                f"（{gto_rec}），此後續節點 solver 未計算）"
            )
    return None


def _fix_collapsed_streets(streets: list) -> list:
    """Fix streets where LLM collapsed a check-check flop into the turn.

    Detects when the first street has a 4+ card board (e.g. "5s5h6c6d") and
    splits it into a proper flop (first 3 cards) + turn (4th card).
    Also handles a 5-card first street (flop+turn+river collapsed).
    """
    if not streets:
        return streets

    first = streets[0]
    board = first.get("board") or first.get("cards") or first.get("card", "")
    # Each card is 2 chars (rank+suit), so 4 cards = 8 chars, 3 cards = 6 chars
    if len(board) <= 6:
        return streets

    result = []
    flop_board = board[:6]  # first 3 cards
    turn_card = board[6:8]  # 4th card

    # Flop: both players checked (which is why LLM collapsed it)
    # Empty actions list — the check-through is inferred by _run_analysis
    result.append({"board": flop_board, "actions": []})

    # Turn: original actions belong here
    turn_entry = {"card": turn_card, "actions": first.get("actions", [])}
    result.append(turn_entry)

    # If board had 5 cards (10 chars), split river too
    if len(board) >= 10:
        river_card = board[8:10]
        # The turn entry gets no actions; move actions to river
        turn_entry["actions"] = []
        result.append({"card": river_card, "actions": first.get("actions", [])})

    # Append remaining streets (shift them forward)
    for s in streets[1:]:
        result.append(s)

    return result


def _run_analysis(hand: dict) -> dict:
    """Core analysis: walk hand, discover bet codes, fetch spot solutions.

    Returns structured data with both formatted text and raw solutions
    for caching and follow-up queries.
    """
    t0 = time.time()
    hand = _normalize_positions(hand)
    gametype = hand.get("gametype", "MTTGeneral")

    # Ensure effective_bb exists — estimate from preflop raises if missing
    if "effective_bb" not in hand or hand["effective_bb"] is None:
        # Estimate from largest preflop raise size × 10, or default 20bb
        parts = hand.get("preflop_actions", "").split("-")
        max_raise = 0
        for p in parts:
            if p.startswith("R") or p.startswith("AI"):
                try:
                    val = float(p.lstrip("RAI"))
                    max_raise = max(max_raise, val)
                except ValueError:
                    pass
        hand["effective_bb"] = max(max_raise * 10, 20) if max_raise > 0 else 20

    depth = nearest_depth(hand["effective_bb"])
    hero_pos = hand["hero_position"]
    hero_hand_raw = hand["hero_hand"]
    hero_hand = normalize_hand_name(hero_hand_raw)
    no_hero_hand = hand.get("no_hero_hand", False)
    # Compute 1326-combo index for exact postflop lookup (e.g. Ah6h vs generic A6s)
    hero_combo_idx = combo_index_for_hand(hero_hand_raw)
    streets = hand.get("streets") or hand.get("postflop_actions", [])

    # Fix malformed streets: if first street has 4+ card board, split into flop + turn
    # (LLM sometimes collapses check-check flop into the turn entry)
    streets = _fix_collapsed_streets(streets)

    # Determine position order based on number of players
    # Prefer players_at_table (explicitly set) over len(player_stacks)
    # (OCR may detect extra stacks from pot values or non-player text)
    num_players = hand.get("players_at_table", 0) or hand.get("num_players", 0)
    if not num_players:
        num_players = len(hand.get("player_stacks", []))
    if not num_players:
        # MTTGeneral always uses 8 positions in GTO Wizard API. Default to 8
        # to avoid position misalignment when preflop_actions are incomplete
        # (e.g. hero SB hasn't acted → only 6 actions for an 8-max table).
        # Genuine smaller tables should specify players_at_table explicitly.
        if gametype == "MTTGeneral":
            num_players = 8
        else:
            parts = hand["preflop_actions"].split("-")
            num_raises = sum(1 for p in parts if p.startswith("R") or p.startswith("AI"))
            num_players = len(parts) - max(0, num_raises - 1)
            num_players = max(2, min(9, num_players))

    # Cash game support: resolve gametype and depth
    is_cash = hand.get("game_format") == "cash"
    if is_cash:
        gametype = CASH_GAMETYPES.get(num_players, "Cash6mGeneral_6mNL100R2")
        depth = nearest_cash_depth(hand["effective_bb"])

    # Pad preflop actions and stacks to match target table size.
    # - MTTGeneral (chip EV): always pad to 8 positions
    # - ICM: pad to players_at_table (e.g., 5 given stacks at 8-max FT → pad to 8)
    is_icm = hand.get("tournament_type") == "icm"
    if is_icm:
        target_players = hand.get("players_at_table", num_players)
    elif gametype == "MTTGeneral":
        target_players = 8
    else:
        target_players = num_players

    if target_players > num_players:
        pad_count = target_players - num_players
        padding = "-".join(["F"] * pad_count)
        hand = dict(hand)  # shallow copy to avoid mutating original
        hand["preflop_actions"] = padding + "-" + hand["preflop_actions"]
        if hand.get("player_stacks"):
            hand["player_stacks"] = [0] * pad_count + hand["player_stacks"]
        num_players = target_players

    pos_order = _get_position_order(num_players)

    # ICM support: resolve gametype and stacks
    icm_stacks = ""
    icm_note = ""
    if is_icm:
        from icm_modes import find_icm_params
        player_stacks = hand.get("player_stacks")
        if player_stacks:
            icm_params = find_icm_params(
                player_stacks=player_stacks,
                pko=hand.get("pko", False),
                tournament_size=hand.get("tournament_size", 1000),
                players_remaining=hand.get("players_remaining"),
                phase=hand.get("phase"),
                players_at_table=num_players,
            )
            gametype = icm_params["gametype"]
            depth = icm_params["depth"]
            icm_stacks = icm_params["stacks"]
            icm_note = icm_params["approximation_note"]
        else:
            # No per-position stacks — use symmetric ICM with effective_bb
            from icm_modes import find_gametype
            gametype = find_gametype(
                players_at_table=num_players,
                pko=hand.get("pko", False),
                tournament_size=hand.get("tournament_size", 1000),
                players_remaining=hand.get("players_remaining"),
                phase=hand.get("phase"),
            )
            # Use symmetric stacks
            eff = hand["effective_bb"]
            depth = f"{eff + 0.125:.3f}"
            icm_stacks = "-".join(f"{eff + 0.125:.3f}" for _ in range(num_players))
            icm_note = f"ICM 模式: {gametype}\n對稱籌碼: {eff:.0f}bb"

    # For ICM preflop_only modes, postflop falls back to chip EV
    # For cash, use the same cash gametype throughout
    if is_cash:
        chipev_gametype = gametype
        chipev_depth = depth
    else:
        chipev_gametype = "MTTGeneral"
        chipev_depth = nearest_depth(hand["effective_bb"])

    # Detect multiway and simplify to heads-up if needed
    raw_preflop = hand["preflop_actions"]
    multiway_note = ""
    multiway_positions = None  # set of 2 positions if multiway simplified
    original_depth = depth  # preserve for preflop open spot (before multiway adjustment)
    simplified_preflop, adjusted_depth, multiway_note, multiway_positions = _simplify_multiway(
        hand, hero_pos, gametype if not is_icm else chipev_gametype,
        depth if not is_icm else chipev_depth,
    )
    if multiway_note:
        raw_preflop = simplified_preflop
        if not is_icm:
            depth = adjusted_depth
        chipev_depth = adjusted_depth

    # Compute actual pot from original preflop for pot-percentage bet matching
    # (only needed for multiway where simplified pot differs from actual pot)
    actual_pot = _compute_preflop_pot(hand["preflop_actions"], hand["effective_bb"]) if multiway_note else 0

    # Normalize preflop actions
    # For ICM, use ICM gametype for preflop normalization
    preflop_actions = _normalize_preflop_actions(
        raw_preflop, gametype, depth, stacks=icm_stacks,
    )

    # ── Phase 1: Walk hand, discover bet codes, collect hero spots ──
    hero_spots = []

    # Preflop hero spot (initial open/fold decision)
    # Use hero's OWN stack depth (not effective_bb) for the open decision when available.
    # Reason: effective_bb = min(hero, caller) is retroactive — hero doesn't know who'll call
    # when deciding to open. The solver models uniform stacks, so hero's stack is the best proxy.
    # Fall back to original_depth (from effective_bb) when player_stacks unavailable.
    preflop_before = _preflop_before_hero(preflop_actions, hero_pos, pos_order)
    preflop_depth = original_depth if multiway_note else depth
    if not is_icm:
        hero_stack_bb = hand.get("hero_starting_stack")
        if not hero_stack_bb and hand.get("player_stacks"):
            stacks = hand["player_stacks"]
            # Only use if stacks length matches padded table size (validates correct mapping)
            if len(stacks) == num_players:
                hero_idx_padded = pos_order.index(hero_pos)
                if hero_idx_padded < len(stacks) and stacks[hero_idx_padded] > 0:
                    hero_stack_bb = stacks[hero_idx_padded]
        if hero_stack_bb and hero_stack_bb > hand.get("effective_bb", 0):
            hero_depth = nearest_depth(hero_stack_bb)
            if hero_depth != preflop_depth:
                # Re-normalize preflop actions for the new depth — raise sizes
                # differ by depth (e.g., R2 at 20bb → R2.1 at 25bb).
                try:
                    renorm = _normalize_preflop_actions(
                        hand["preflop_actions"], gametype, hero_depth)
                    preflop_before = _preflop_before_hero(renorm, hero_pos, pos_order)
                except Exception:
                    pass  # fall back to original preflop_before
                preflop_depth = hero_depth
    hero_spots.append({
        "street": "preflop",
        "header": "【Preflop】",
        "params": dict(gametype=gametype, depth=preflop_depth, stacks=icm_stacks,
                       preflop_actions=preflop_before),
        "action_desc": None,
    })

    # Check if hero faces a re-raise (needs to act again preflop)
    pf_parts = preflop_actions.split("-")
    hero_idx = pos_order.index(hero_pos)
    has_reraise = any(
        pf_parts[i].startswith("R")
        for i in range(hero_idx + 1, min(len(pf_parts), num_players))
    )
    if has_reraise:
        # Query hero's second decision at the full N-position preflop
        full_n = "-".join(pf_parts[:num_players])

        # Build HU fallback: strip cold callers (keep only hero + 3bettor)
        # Solver often lacks 3-way cold call solutions; HU is a reasonable approximation
        reraise_idx = None
        for i in range(hero_idx + 1, num_players):
            if pf_parts[i].startswith("R") or pf_parts[i].startswith("AI"):
                reraise_idx = i
                break
        hu_fallback_n = None
        if reraise_idx is not None:
            cold_callers = [
                i for i in range(num_players)
                if i != hero_idx and i != reraise_idx and pf_parts[i] not in ("F", "")
            ]
            if cold_callers:
                hu_parts = list(pf_parts[:num_players])
                for ci in cold_callers:
                    hu_parts[ci] = "F"
                hu_fallback_n = "-".join(hu_parts)

        # Check if hero's continuation action is in the string (parts[N:])
        hero_cont_desc = None
        if len(pf_parts) > num_players:
            active = [i for i in range(num_players) if pf_parts[i] not in ("F", "")]
            cont_idx = 0
            for j in range(num_players, len(pf_parts)):
                if cont_idx >= len(active):
                    cont_idx = 0
                if active[cont_idx] == hero_idx:
                    code = pf_parts[j]
                    hero_cont_desc = f"  → 實際行動: {hero_pos} {code}（solver code: {code}）"
                    break
                cont_idx += 1

        hero_spots.append({
            "street": "preflop",
            "header": None,
            "params": dict(gametype=gametype, depth=depth, stacks=icm_stacks,
                           preflop_actions=full_n),
            "action_desc": hero_cont_desc,
            "hu_fallback_params": dict(gametype=gametype, depth=depth, stacks=icm_stacks,
                                       preflop_actions=hu_fallback_n) if hu_fallback_n else None,
        })

    board = ""
    flop_acts = ""
    turn_acts = ""
    river_acts = ""
    chipev_preflop = preflop_actions  # default; overridden for ICM on flop
    all_in_resolved = False  # True after an all-in is called — no further betting

    # Check if preflop ended with all-in called
    _raw_pf_parts = hand["preflop_actions"].split("-")
    if len(_raw_pf_parts) >= 2 and _raw_pf_parts[-1] == "C":
        prev = _raw_pf_parts[-2]
        if prev == "AI" or prev.startswith("AI") and prev[2:].replace(".", "", 1).isdigit():
            all_in_resolved = True

    # Track action strings at each street boundary (for hypothetical queries)
    street_states = {}

    for street_idx, street in enumerate(streets):
        street_name = STREET_NAMES[street_idx]

        # Always accumulate board cards, even for streets with no actions
        if street_idx == 0:
            board = street.get("board") or street.get("cards") or street.get("card", "")
            street_header = f"【Flop: {board}】"
        elif street_idx == 1:
            card = street.get("card") or street.get("cards", "")
            board += card
            street_header = f"【Turn: {card}（Board: {board}）】"
        elif street_idx == 2:
            card = street.get("card") or street.get("cards", "")
            board += card
            street_header = f"【River: {card}（Board: {board}）】"

        # Skip streets after all-in is resolved (no further betting possible)
        if all_in_resolved:
            continue

        # Skip streets with no actions (e.g. preflop all-in, board dealt but no play)
        # But if a later street has actions, infer check-through
        if not street.get("actions"):
            # If this is flop/turn with no actions but later streets exist,
            # both players checked through — record X-X for the API
            has_later = any(s.get("actions") for s in streets[street_idx + 1:])
            if has_later:
                if street_idx == 0:
                    flop_acts = "X-X"
                elif street_idx == 1:
                    turn_acts = "X-X"
            continue

        # Infer missing hero call: if an opponent bet/raised but hero didn't
        # respond, hero must have called (LLM sometimes omits hero's calls).
        # On non-final streets: hand continues → hero called.
        # On final street: opponent bet and hero is still in → assume call
        # to show GTO data for the decision point.
        acts = street["actions"]
        if acts:
            last_act = acts[-1]
            last_pos = last_act["position"]
            last_type = last_act["action"]
            hero_acted = any(a["position"] == hero_pos for a in acts)
            if (last_pos != hero_pos
                    and last_type not in ("X", "F")
                    and not hero_acted):
                inferred_size = last_act.get("size", 0)
                acts.append({
                    "position": hero_pos,
                    "action": "C",
                    "size": inferred_size,
                })

        # Snapshot state at start of this street (before actions)
        street_states[street_name] = {
            "board": board,
            "flop_actions": flop_acts,
            "turn_actions": turn_acts,
            "river_actions": river_acts,
        }

        street_first_hero = True
        outstanding_bet = 0
        street_investments = {}
        _prev_allin = False
        _acted_this_street = set()  # track who has acted (for misparsed dup detection)

        # Postflop uses chip EV for ICM modes (preflop_only)
        post_gametype = chipev_gametype if is_icm else gametype
        post_depth = chipev_depth if is_icm else depth
        # Normalize preflop for chip EV context (only once on flop)
        if is_icm and street_idx == 0:
            chipev_preflop = _normalize_preflop_actions(
                hand["preflop_actions"], chipev_gametype, chipev_depth,
            )

        # ── Postflop depth escalation ──
        # At very shallow depths (e.g. 8-9bb) the solver may not have
        # postflop solutions.  Detect this on the flop and bump up to the
        # next available depth that has data.
        if street_idx == 0 and street.get("actions"):
            post_preflop_check = chipev_preflop if is_icm else preflop_actions
            _probe = get_next_actions(
                gametype=post_gametype, depth=post_depth,
                preflop_actions=post_preflop_check, board=board,
            )
            if _probe and not _probe["next_actions"].get("available_actions"):
                from gto_api import AVAILABLE_DEPTHS
                cur_bb = float(post_depth) - 0.125
                higher = sorted(d for d in AVAILABLE_DEPTHS if d > cur_bb)
                for try_bb in higher[:3]:
                    try_depth = try_bb + 0.125
                    try_pf = _normalize_preflop_actions(
                        hand["preflop_actions"],
                        post_gametype, try_depth,
                    )
                    _probe2 = get_next_actions(
                        gametype=post_gametype, depth=try_depth,
                        preflop_actions=try_pf, board=board,
                    )
                    if _probe2 and _probe2["next_actions"].get("available_actions"):
                        post_depth = try_depth
                        if is_icm:
                            chipev_depth = try_depth
                            chipev_preflop = try_pf
                        else:
                            depth = try_depth
                            preflop_actions = try_pf
                        break

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)
            # Parse size from action string if not provided separately
            if not target_size and action_type.startswith("R"):
                try:
                    target_size = float(action_type[1:])
                except (ValueError, IndexError):
                    pass
            if not target_size and action_type.startswith("AI"):
                try:
                    target_size = float(action_type[2:]) if len(action_type) > 2 else 0
                except ValueError:
                    pass

            # Skip actions from positions not in simplified heads-up
            if multiway_positions and pos not in multiway_positions:
                # Still track pot changes from folded players
                if actual_pot > 0:
                    if action_type == "C":
                        prev = street_investments.get(pos, 0)
                        actual_pot += outstanding_bet - prev
                        street_investments[pos] = outstanding_bet
                    elif action_type not in ("X", "F"):
                        prev = street_investments.get(pos, 0)
                        actual_pot += target_size - prev
                        street_investments[pos] = target_size
                        outstanding_bet = target_size
                continue

            # Skip duplicate simple actions from the same position on the
            # same street (e.g., two SB checks).  In multiway pots the LLM
            # sometimes assigns another player's action to the wrong position;
            # the duplicate corrupts the solver action string.
            if pos != hero_pos and action_type == "X" and pos in _acted_this_street:
                continue
            _acted_this_street.add(pos)

            post_preflop = chipev_preflop if is_icm else preflop_actions

            if pos == hero_pos:
                params = dict(
                    gametype=post_gametype, depth=post_depth,
                    preflop_actions=post_preflop, board=board,
                    flop_actions=flop_acts, turn_actions=turn_acts,
                    river_actions=river_acts,
                )

                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                else:
                    next_resp = get_next_actions(**params)
                    avail = next_resp["next_actions"]["available_actions"]
                    if actual_pot > 0 and outstanding_bet == 0:
                        # First bet of street in multiway — use pot-pct matching
                        taken_code = _find_action_by_pot_pct(avail, target_size, actual_pot)
                    else:
                        # When action is a raise/bet but size is unknown (0), restrict
                        # matching to raise actions only — otherwise C/X wins by proximity
                        match_avail = avail
                        if not target_size and action_type.startswith(("R", "AI")):
                            raise_only = [a for a in avail if a["action"]["code"] not in ("X", "C", "F")]
                            if raise_only:
                                match_avail = raise_only
                        taken_code = find_closest_action_postflop(match_avail, target_size)

                size_str = f" {target_size}bb" if target_size else ""
                hero_spots.append({
                    "street": street_name,
                    "header": street_header if street_first_hero else None,
                    "params": params,
                    "action_desc": f"  → 實際行動: {pos} {action_type}{size_str}（solver code: {taken_code}）",
                    "taken_code": taken_code,
                })
                street_first_hero = False
            else:
                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                else:
                    params = dict(
                        gametype=post_gametype, depth=post_depth,
                        preflop_actions=post_preflop, board=board,
                        flop_actions=flop_acts, turn_actions=turn_acts,
                        river_actions=river_acts,
                    )
                    next_resp = get_next_actions(**params)
                    avail = next_resp["next_actions"]["available_actions"]
                    if actual_pot > 0 and outstanding_bet == 0:
                        # First bet of street in multiway — use pot-pct matching
                        taken_code = _find_action_by_pot_pct(avail, target_size, actual_pot)
                    else:
                        # When action is a raise/bet but size is unknown (0), restrict
                        # matching to raise actions only — otherwise C/X wins by proximity
                        match_avail = avail
                        if not target_size and action_type.startswith(("R", "AI")):
                            raise_only = [a for a in avail if a["action"]["code"] not in ("X", "C", "F")]
                            if raise_only:
                                match_avail = raise_only
                        taken_code = find_closest_action_postflop(match_avail, target_size)

            # Track actual pot through postflop (for multiway percentage matching)
            if actual_pot > 0:
                if action_type in ("X", "F"):
                    pass
                elif action_type == "C":
                    prev = street_investments.get(pos, 0)
                    actual_pot += outstanding_bet - prev
                    street_investments[pos] = outstanding_bet
                else:  # bet/raise
                    prev = street_investments.get(pos, 0)
                    actual_pot += target_size - prev
                    street_investments[pos] = target_size
                    outstanding_bet = target_size

            # Advance action string (only for positions in the simplified pair)
            if street_idx == 0:
                flop_acts = f"{flop_acts}-{taken_code}" if flop_acts else taken_code
            elif street_idx == 1:
                turn_acts = f"{turn_acts}-{taken_code}" if turn_acts else taken_code
            elif street_idx == 2:
                river_acts = f"{river_acts}-{taken_code}" if river_acts else taken_code

            # Detect all-in called — use normalized taken_code (RAI) since
            # the original action_type might be "R7" that got normalized to RAI
            if taken_code == "C" and _prev_allin:
                all_in_resolved = True
            _prev_allin = taken_code == "RAI" or action_type.startswith("AI")

        # After processing all actions on this street, check for incomplete
        # check-throughs.  When the parsed JSON only records one player's
        # check (e.g., only "BB X" on the turn with no "HJ X"), but later
        # streets have actions, both players must have checked through.
        # Append an implied opponent check so the action string becomes
        # "X-X" instead of just "X".
        if not all_in_resolved:
            cur_acts = (flop_acts if street_idx == 0
                        else turn_acts if street_idx == 1
                        else river_acts)
            # Only a single check was recorded and the street has later action
            if cur_acts == "X":
                has_later = any(s.get("actions") for s in streets[street_idx + 1:])
                if has_later:
                    if street_idx == 0:
                        flop_acts = "X-X"
                    elif street_idx == 1:
                        turn_acts = "X-X"
                    elif street_idx == 2:
                        river_acts = "X-X"

    # ── Phase 1.5: Create hero spot at current decision point ──
    # When the hand ends with opponent action and it's hero's turn to act
    # (e.g., BB checks on turn, hero hasn't acted yet), create a hero spot
    # so the user sees GTO strategy for the current decision.
    if streets and not all_in_resolved:
        last_street = streets[-1]
        last_acts = last_street.get("actions", [])
        if last_acts:
            last_pos = last_acts[-1]["position"]
            hero_acted_last_street = any(
                a["position"] == hero_pos for a in last_acts)
            if last_pos != hero_pos and not hero_acted_last_street:
                # Hero hasn't acted — create spot at current decision
                last_street_idx = len(streets) - 1
                street_name = (["flop", "turn", "river"]
                               [last_street_idx] if last_street_idx < 3
                               else f"street{last_street_idx}")
                post_gametype = chipev_gametype if is_icm else gametype
                post_depth = chipev_depth if is_icm else depth
                post_preflop = chipev_preflop if is_icm else preflop_actions
                params = dict(gametype=post_gametype, depth=post_depth,
                              preflop_actions=post_preflop)
                # Build full board: flop board + turn card + river card
                full_board = streets[0].get("board", "")
                for si in range(1, last_street_idx + 1):
                    full_board += streets[si].get("card", "")
                params["board"] = full_board
                params["flop_actions"] = flop_acts
                if last_street_idx >= 1:
                    params["turn_actions"] = turn_acts
                if last_street_idx >= 2:
                    params["river_actions"] = river_acts

                # Build display card for turn/river
                display_card = last_street.get("card", "")
                street_label = (f"{street_name.capitalize()}: "
                                f"{display_card}" if display_card
                                else f"{street_name.capitalize()}: {full_board}")
                hero_spots.append({
                    "street": street_name,
                    "header": street_label,
                    "params": params,
                    "action_desc": f"→ Hero 的決策點",
                    "taken_code": None,  # hero hasn't acted
                })

    t_phase1 = time.time()

    # ── Phase 2: Fetch all spot solutions in parallel ──
    # Propagate thread-local user token into executor threads
    from gto_api import _thread_local as _gto_tl, set_user_token, clear_user_token
    _parent_token = getattr(_gto_tl, "access_token", None)

    def _fetch_with_token(params):
        if _parent_token:
            set_user_token(_parent_token)
        try:
            return get_spot_solution(**params)
        finally:
            if _parent_token:
                clear_user_token()

    solutions = [None] * len(hero_spots)
    with ThreadPoolExecutor(max_workers=len(hero_spots)) as executor:
        future_to_idx = {
            executor.submit(_fetch_with_token, spot["params"]): i
            for i, spot in enumerate(hero_spots)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            solutions[idx] = future.result()

    # Retry with HU fallback for spots that returned no solution
    for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
        if sol is None and spot.get("hu_fallback_params"):
            solutions[i] = _fetch_with_token(spot["hu_fallback_params"])
            if solutions[i]:
                # Add multiway approximation note
                if not multiway_note:
                    multiway_note = "⚠ 多人底池，cold caller 已簡化為 heads-up 分析"

    # Retry ICM preflop spots with chip EV fallback (e.g. subscription insufficient)
    icm_fallback_note = ""
    if is_icm:
        for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
            if sol is None and spot["street"] == "preflop" and spot["params"].get("stacks"):
                chipev_params = dict(spot["params"])
                chipev_params["gametype"] = chipev_gametype
                chipev_params["depth"] = chipev_depth
                chipev_params["stacks"] = ""
                solutions[i] = _fetch_with_token(chipev_params)
                if solutions[i]:
                    icm_fallback_note = "⚠ ICM 模式不可用（可能需要更高等級的 GTO Wizard 訂閱），已自動改用 Chip EV"

    # Retry postflop spots with GTO-recommended action substitution.
    # When hero's actual bet maps to a low-frequency solver action (e.g. 33% pot
    # when GTO says 20% pot), the solver may not expand that line for later streets.
    # Fix: substitute hero's action with the GTO-recommended (highest freq) action
    # from the previous spot's solution, and retry.
    gto_line_note = ""
    for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
        if sol is not None or spot["street"] == "preflop":
            continue
        # Find the previous hero spot on the same or earlier street that has a solution
        for j in range(i - 1, -1, -1):
            prev_spot = hero_spots[j]
            prev_sol = solutions[j]
            prev_taken = prev_spot.get("taken_code")
            if not prev_sol or not prev_taken or prev_taken in ("X", "C", "F"):
                continue
            # Find the best action for hero's hand at that spot
            best_code = None
            best_freq = 0
            hn = normalize_hand_name(hero_hand)
            for pi in prev_sol.get("players_info", []):
                if pi["player"]["position"] != hero_pos:
                    continue
                shc = pi.get("simple_hand_counters", {})
                hd = shc.get(hn)
                if not hd:
                    break
                for code, freq in hd.get("actions_total_frequencies", {}).items():
                    if freq > best_freq:
                        best_freq = freq
                        best_code = code
                break
            if not best_code or best_code == prev_taken:
                continue
            # Only substitute when hero's action had very low frequency (<10%)
            hn2 = normalize_hand_name(hero_hand)
            hero_taken_freq = 0
            for pi in prev_sol.get("players_info", []):
                if pi["player"]["position"] != hero_pos:
                    continue
                shc2 = pi.get("simple_hand_counters", {})
                hd2 = shc2.get(hn2)
                if hd2:
                    hero_taken_freq = hd2.get("actions_total_frequencies", {}).get(prev_taken, 0)
                break
            if hero_taken_freq >= 0.10:
                continue
            # Substitute hero's action in the params
            retry_params = dict(spot["params"])
            prev_street = prev_spot["street"]
            action_key = f"{prev_street}_actions"
            if action_key in retry_params and prev_taken in retry_params[action_key]:
                retry_params[action_key] = retry_params[action_key].replace(
                    prev_taken, best_code, 1)
                retry_sol = _fetch_with_token(retry_params)
                if retry_sol:
                    solutions[i] = retry_sol
                    # Build descriptive labels for the note
                    from gto_formatter import _action_label
                    taken_label = _action_label(prev_taken, prev_sol)
                    best_label = _action_label(best_code, prev_sol)
                    prev_street_name = prev_street.capitalize()
                    gto_line_note = (
                        f"⚠ Hero 在 {prev_street_name} 的下注（{taken_label}）偏離 GTO 建議（{best_label}），"
                        f"{spot['street'].capitalize()} 的分析假設走 GTO 建議路線（{best_label}）作為參考"
                    )
                    # Update the spot params for display
                    hero_spots[i]["params"] = retry_params
            break

    # ── Phase 2.5: Preflop open depth correction ──
    # If hero raised preflop but the solver shows 0% raise for hero's hand at the
    # current depth, try the next higher depth. This handles depth quantization
    # boundary issues where effective_bb < hero's actual stack (e.g. 16bb effective
    # maps to 17bb solver = limp/fold, but hero's 21bb stack maps to 20bb = 100% raise).
    if not is_icm and solutions[0] is not None:
        pf_parts = preflop_actions.split("-")
        hero_pf_idx = pos_order.index(hero_pos)
        hero_pf_action = pf_parts[hero_pf_idx] if hero_pf_idx < len(pf_parts) else ""
        is_hero_open = (hero_pf_action.startswith("R") or hero_pf_action.startswith("AI"))
        all_fold_before = all(p == "F" for p in pf_parts[:hero_pf_idx])

        if is_hero_open and all_fold_before:
            # Check if hero's hand has any raise frequency at current depth
            sol0 = solutions[0]
            has_raise = False
            for pi in sol0.get("players_info", []):
                if pi["player"]["position"] == hero_pos and len(pi.get("range", [])) == 169:
                    hn = normalize_hand_name(hero_hand)
                    ranks = "23456789TJQKA"
                    all_hands = []
                    for _i, r1 in enumerate(ranks):
                        for _j, r2 in enumerate(ranks):
                            if _i == _j:
                                all_hands.append(f"{r1}{r2}")
                            elif _i > _j:
                                all_hands.append(f"{r1}{r2}o")
                                all_hands.append(f"{r1}{r2}s")
                    hand_names_sorted = sorted(all_hands)
                    if hn in hand_names_sorted:
                        hidx = hand_names_sorted.index(hn)
                        for asol in sol0.get("action_solutions", []):
                            code = asol["action"]["code"]
                            if code.startswith("R") or code == "RAI":
                                strat = asol.get("strategy", [])
                                if strat and hidx < len(strat) and strat[hidx] > 0.01:
                                    has_raise = True
                                    break
                    break

            if not has_raise:
                # Try next higher depth
                current_depth_bb = float(preflop_depth) - 0.125 if isinstance(preflop_depth, (int, float)) else 0
                from gto_api import AVAILABLE_DEPTHS
                higher = [d for d in AVAILABLE_DEPTHS if d > current_depth_bb]
                if higher:
                    next_depth = min(higher) + 0.125
                    retry_params = dict(hero_spots[0]["params"], depth=next_depth)
                    retry_sol = _fetch_with_token(retry_params)
                    if retry_sol:
                        # Check if hero's hand has raise at next depth
                        for pi in retry_sol.get("players_info", []):
                            if pi["player"]["position"] == hero_pos and len(pi.get("range", [])) == 169:
                                if hn in hand_names_sorted:
                                    for asol in retry_sol.get("action_solutions", []):
                                        code = asol["action"]["code"]
                                        if code.startswith("R") or code == "RAI":
                                            strat = asol.get("strategy", [])
                                            if strat and hidx < len(strat) and strat[hidx] > 0.01:
                                                solutions[0] = retry_sol
                                                hero_spots[0]["params"]["depth"] = next_depth
                                                preflop_depth = next_depth
                                                break
                                break

    # ── Phase 2.6: Postflop depth upgrade for off-range hero ──
    # When hero's preflop action (e.g., raise 2bb) differs from GTO (e.g.,
    # all-in), hero's combo has 0% postflop range at this depth.  Try 1-2
    # higher depths where the combo IS in the raise range.  This gives the
    # user useful postflop analysis despite the preflop deviation.
    offrange_note = ""
    if not is_icm and not no_hero_hand and hero_combo_idx is not None:
        # Check if hero should have gone all-in preflop (≥80% all-in freq)
        # AND hero combo has 0% postflop range as a result
        pf_allin_freq = 0
        if solutions[0]:
            hn = normalize_hand_name(hero_hand)
            for pi in solutions[0].get("players_info", []):
                if pi["player"]["position"] != hero_pos:
                    continue
                shc = pi.get("simple_hand_counters", {})
                hd = shc.get(hn)
                if hd:
                    af = hd.get("actions_total_frequencies", {})
                    pf_allin_freq = af.get("RAI", 0)
                break

        has_offrange = False
        # Trigger when hero combo is 0% at the FIRST postflop spot (flop start).
        # This means the hand truly shouldn't be in the postflop range
        # (e.g., GTO says all-in preflop but hero called).
        # Don't trigger for hands that are just 0% at a later node after
        # specific actions — those are legitimate range reductions.
        first_postflop = None
        for spot, sol in zip(hero_spots, solutions):
            if spot["street"] != "preflop" and sol is not None:
                first_postflop = (spot, sol)
                break
        if first_postflop:
            fp_spot, fp_sol = first_postflop
            # Check range at flop start (no actions = before any bets)
            flop_start_params = dict(fp_spot["params"])
            flop_start_params["flop_actions"] = ""
            flop_start_params["turn_actions"] = ""
            flop_start_params["river_actions"] = ""
            flop_start_sol = _fetch_with_token(flop_start_params)
            if flop_start_sol:
                for pi in flop_start_sol["players_info"]:
                    if pi["player"]["position"] == hero_pos:
                        rng = pi.get("range", [])
                        if len(rng) == 1326 and rng[hero_combo_idx] < 0.005:
                            has_offrange = True
                        break

        if has_offrange:
            from gto_api import AVAILABLE_DEPTHS
            current_bb = float(depth) - 0.125 if isinstance(depth, (int, float)) else 0
            higher = sorted(d for d in AVAILABLE_DEPTHS if d > current_bb)[:2]
            for try_bb in higher:
                try_depth = try_bb + 0.125
                # Re-normalize the simplified preflop for higher depth
                # (raise sizes differ — e.g., R2 at 20bb → R2.1 at 25bb)
                try:
                    renorm_pf = _normalize_preflop_actions(
                        preflop_actions, gametype, try_depth)
                except Exception:
                    renorm_pf = preflop_actions
                # Check if hero combo IS in range at higher depth.
                # Query flop start (no actions) to avoid action-code mismatch.
                first_board = streets[0].get("board", "") if streets else ""
                check_params = dict(gametype=gametype, depth=try_depth,
                                    preflop_actions=renorm_pf, board=first_board,
                                    flop_actions="")
                check_sol = _fetch_with_token(check_params)
                found = False
                if check_sol:
                    for pi in check_sol["players_info"]:
                        if pi["player"]["position"] == hero_pos:
                            rng = pi.get("range", [])
                            if len(rng) == 1326 and rng[hero_combo_idx] >= 0.005:
                                found = True
                            break
                if found:
                    # Re-run full analysis at higher depth by re-normalizing
                    # all postflop actions for the new depth.
                    for i, (spot, _) in enumerate(zip(hero_spots, solutions)):
                        if spot["street"] == "preflop":
                            continue
                        # Re-normalize action codes for new depth
                        retry_params = dict(spot["params"], depth=try_depth,
                                            preflop_actions=renorm_pf)
                        # Try with original action codes first
                        retry_sol = _fetch_with_token(retry_params)
                        if not retry_sol:
                            # Action codes might differ at new depth — try
                            # re-walking from flop start
                            retry_params2 = dict(retry_params)
                            retry_params2["flop_actions"] = ""
                            retry_params2["turn_actions"] = ""
                            retry_params2["river_actions"] = ""
                            retry_sol = _fetch_with_token(retry_params2)
                        if retry_sol:
                            solutions[i] = retry_sol
                            hero_spots[i]["params"] = retry_params
                    offrange_note = (
                        f"⚠ 此深度 {hero_hand} 不在 postflop range，"
                        f"postflop 使用 {try_bb:.0f}bb solver 近似（僅供參考）"
                    )
                    break

    t_phase2 = time.time()

    # ── Phase 3: Format results ──
    results = []
    results.append("=" * 50)
    hero_label = f"Hero: {hero_pos}" if no_hero_hand else f"Hero: {hero_pos} {hero_hand}"
    if is_icm:
        depth_display = depth if isinstance(depth, str) else f"{depth}"
        results.append(hero_label)
        if icm_note:
            results.append(icm_note)
        if streets:
            results.append(f"Postflop 使用 Chip EV {chipev_depth - 0.125:.0f}bb solver（ICM 僅支援 preflop）")
    elif is_cash:
        results.append(f"Cash Game {num_players}-max")
        results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth:.0f}bb solver）")
        results.append(hero_label)
    else:
        results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth - 0.125:.0f}bb solver）")
        results.append(hero_label)
    if icm_fallback_note:
        results.append(icm_fallback_note)
    if multiway_note:
        results.append(multiway_note)
    if gto_line_note:
        results.append(gto_line_note)
    if offrange_note:
        results.append(offrange_note)
    if raw_preflop != preflop_actions:
        # Generate detailed approximation notes for each corrected action
        raw_parts = raw_preflop.split("-")
        norm_parts = preflop_actions.split("-")
        corrections = []
        for idx, (raw_code, norm_code) in enumerate(zip(raw_parts, norm_parts)):
            if raw_code == norm_code:
                continue
            pos_name = pos_order[idx] if idx < len(pos_order) else f"pos{idx}"
            if raw_code.startswith("AI") and norm_code == "RAI":
                # Any all-in → solver all-in is the same thing, not a real correction
                continue
            if raw_code == "C" and norm_code == "X":
                # BB call closing preflop = check in solver — same thing
                continue
            elif raw_code.startswith("AI") and raw_code != "AI":
                size = raw_code[2:]
                corrections.append(f"{pos_name} all-in {size}bb → 近似為 raise {norm_code}")
            elif raw_code == "AI":
                corrections.append(f"{pos_name} all-in → {norm_code}")
            elif raw_code.startswith("R") and norm_code.startswith("R"):
                corrections.append(f"{pos_name} raise {raw_code[1:]}bb → 校正為 {norm_code}")
            else:
                corrections.append(f"{pos_name} {raw_code} → {norm_code}")
        if corrections:
            results.append(f"Preflop actions 校正: {raw_preflop} → {preflop_actions}")
            results.append(f"⚠ 近似說明: {'; '.join(corrections)}")
            results.append("  此場景無法被 solver 完全模擬，使用最接近的 solver 解作為參考")
    results.append("")

    from hand_eval import evaluate as _eval_hand

    for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
        if spot["header"]:
            results.append("")
            results.append("=" * 50)
            results.append(spot["header"])

            # Add deterministic hand type label for postflop streets
            # Use hero_hand_raw (with suits, e.g. "AcTh") so flush/flush-draw detection works
            if not no_hero_hand:
                spot_street = spot["street"]
                if spot_street != "preflop" and spot_street in street_states:
                    spot_board = street_states[spot_street].get("board", "")
                    if spot_board:
                        eval_input = hero_hand_raw if len(hero_hand_raw) == 4 else hero_hand
                        eval_result = _eval_hand(eval_input, spot_board)
                        if eval_result["full_label"]:
                            results.append(f"Hero {hero_hand} 牌型: {eval_result['full_label']}")

        if sol:
            # When no hero hand specified, show only range-level summary (no hero-specific detail)
            spot_text = format_full_spot(sol, None if no_hero_hand else hero_hand, hero_pos)
            results.append(spot_text)

            # Include full range breakdown when no hero hand specified or ICM
            # preflop — prevents Gemini from fabricating range compositions
            if (is_icm and spot["street"] == "preflop") or no_hero_hand:
                from gto_formatter import format_range_by_action
                range_text = format_range_by_action(sol, hero_pos)
                if range_text:
                    results.append("")
                    results.append(range_text)

            # Show EV loss if hero took a suboptimal action (skip when no hero hand)
            taken_code = spot.get("taken_code")
            if taken_code and not no_hero_hand:
                is_pf = spot["street"] == "preflop"
                ev_note = format_ev_comparison(
                    sol, taken_code, hero_hand, hero_pos,
                    is_preflop=is_pf, combo_idx=None if is_pf else hero_combo_idx,
                )
                if ev_note:
                    results.append(ev_note)
        else:
            # Check if a previous hero action explains the missing data
            explanation = _explain_missing_solution(i, hero_spots, solutions, hero_hand, hero_pos,
                                                      combo_idx=hero_combo_idx)
            if explanation:
                results.append(explanation)
            else:
                results.append("（無 solver 數據）")

        if spot["action_desc"]:
            results.append(spot["action_desc"])
            results.append("")

    results.append(f"⏱ Discovery: {t_phase1 - t0:.1f}s | Analysis: {t_phase2 - t_phase1:.1f}s | Total: {t_phase2 - t0:.1f}s")

    # ── Phase 3b: Build compact output for user-facing GTO summary ──
    from gto_formatter import format_spot_compact, _action_label, _action_label_short

    # Compact header
    eff = hand.get("effective_bb", "")
    eff_str = f"{eff:.0f}bb" if isinstance(eff, (int, float)) else f"{eff}bb"
    if is_icm:
        mode_str = "ICM"
    elif is_cash:
        mode_str = f"Cash {num_players}-max"
    else:
        mode_str = "MTT"
    compact_hero = f"♠ {hero_pos} | {eff_str} {mode_str}" if no_hero_hand else f"♠ {hero_pos} {hero_hand} | {eff_str} {mode_str}"
    compact = [compact_hero]
    if multiway_note:
        # multiway_note already starts with ⚠ — don't double it
        compact.append(multiway_note if multiway_note.startswith("⚠") else f"⚠ {multiway_note}")
    if gto_line_note:
        compact.append(gto_line_note)
    if offrange_note:
        compact.append(offrange_note)

    for i, (spot, sol) in enumerate(zip(hero_spots, solutions)):
        if spot["header"]:
            # Convert 【Preflop】 → ─── Preflop ───
            # Simplify 【Turn: Kc（Board: Js6h5sKc）】 → Turn: Kc
            raw_hdr = spot["header"].strip("【】")
            paren_idx = raw_hdr.find("（")
            if paren_idx > 0:
                raw_hdr = raw_hdr[:paren_idx].rstrip()
            compact.append(f"\n─── {raw_hdr} ───")

            # Hand type label (postflop) — skip when no hero hand
            if not no_hero_hand:
                spot_street = spot["street"]
                if spot_street != "preflop" and spot_street in street_states:
                    spot_board = street_states[spot_street].get("board", "")
                    if spot_board:
                        eval_input = hero_hand_raw if len(hero_hand_raw) == 4 else hero_hand
                        eval_result = _eval_hand(eval_input, spot_board)
                        if eval_result["full_label"]:
                            compact.append(f"🎯 {eval_result['full_label']}")

        if sol:
            # When no hero hand, use sentinel to force range-level frequencies
            # For postflop, pass combo_idx for combo-specific frequencies
            pf_spot = spot["street"] == "preflop"
            cidx = None if (pf_spot or no_hero_hand) else hero_combo_idx
            spot_compact = format_spot_compact(sol, "__RANGE__" if no_hero_hand else hero_hand, hero_pos,
                                               combo_idx=cidx)
            if spot_compact:
                compact.append(spot_compact)

            # Hero result line — skip when no hero hand
            taken_code = spot.get("taken_code")
            if taken_code and not no_hero_hand:
                # Build hero action label with sizing
                hero_action_short = taken_code
                hero_sizing_pct = ""
                for asol in sol.get("action_solutions", []):
                    if asol["action"]["code"] == taken_code:
                        act = asol["action"]
                        if taken_code == "X":
                            hero_action_short = "check"
                        elif taken_code == "F":
                            hero_action_short = "fold"
                        elif taken_code == "C":
                            hero_action_short = "call"
                        elif act.get("allin"):
                            hero_action_short = "all-in"
                        else:
                            # "raise" if preflop or facing a bet
                            has_fc = any(a["action"]["code"] in ("F", "C")
                                         for a in sol.get("action_solutions", []))
                            verb = "raise" if (spot["street"] == "preflop" or has_fc) else "bet"
                            pct = float(act.get("betsize_by_pot", 0)) * 100
                            if pct > 0:
                                hero_action_short = f"{verb} {pct:.0f}% pot"
                                hero_sizing_pct = f"{pct:.0f}%"
                            else:
                                hero_action_short = verb
                        break

                # Find GTO top action for sizing/deviation comparison.
                # Use combo-specific frequencies for postflop (not aggregate).
                gto_top_label = ""
                gto_top_freq = 0.0
                combo_freq = None
                if not is_pf and hero_combo_idx is not None:
                    action_sols = sol.get("action_solutions", [])
                    if action_sols and "strategy" in action_sols[0]:
                        combo_freq = {}
                        for asol in action_sols:
                            f = asol["strategy"][hero_combo_idx]
                            if f > 0.005:
                                combo_freq[asol["action"]["code"]] = f
                if combo_freq:
                    af = combo_freq
                else:
                    af = None
                    hand_name = normalize_hand_name(hero_hand)
                    for pi in sol.get("players_info", []):
                        if pi["player"]["position"] != hero_pos:
                            continue
                        shc = pi.get("simple_hand_counters", {})
                        hd = shc.get(hand_name)
                        if hd:
                            af = hd.get("actions_total_frequencies", {})
                        break
                if af:
                    top_code = max(af, key=af.get)
                    gto_top_freq = af[top_code]
                    if top_code != taken_code:
                        gto_top_label = _action_label_short(
                            top_code, sol, spot["street"])

                # Check EV loss
                is_pf = spot["street"] == "preflop"
                ev_note = format_ev_comparison(
                    sol, taken_code, hero_hand, hero_pos,
                    is_preflop=is_pf, combo_idx=None if is_pf else hero_combo_idx,
                )
                sizing_hint = f" (GTO建議 {gto_top_label})" if gto_top_label else ""
                # Determine ✅ vs ❌:
                #   ❌ if EV loss ≥ 0.5bb, OR
                #   ❌ if GTO top action ≥ 80% and hero took a different action
                is_bad = False
                ev_loss = 0.0
                if ev_note:
                    m = re.search(r"EV 損失 ([\d.]+)bb", ev_note)
                    ev_loss = float(m.group(1)) if m else 0
                    if ev_loss >= 0.5:
                        is_bad = True
                if gto_top_label and gto_top_freq >= 0.80:
                    is_bad = True

                if is_bad:
                    ev_part = f" EV損失 -{ev_loss:.1f}bb" if ev_loss >= 0.5 else ""
                    compact.append(f"→ Hero {hero_action_short} ❌{ev_part}{sizing_hint}")
                else:
                    compact.append(f"→ Hero {hero_action_short} ✅{sizing_hint}")
        else:
            compact.append("（無 solver 數據）")

    return {
        "text": "\n".join(results),
        "text_compact": "\n".join(compact),
        "hand": hand,
        "gametype": gametype,
        "depth": depth,
        "stacks": icm_stacks,
        "is_icm": is_icm,
        "hero_position": hero_pos,
        "hero_hand": hero_hand,
        "no_hero_hand": no_hero_hand,
        "preflop_actions": preflop_actions,
        "street_states": street_states,
        "final_actions": {
            "flop_actions": flop_acts,
            "turn_actions": turn_acts,
            "river_actions": river_acts,
        },
        "hero_spots": hero_spots,
        "solutions": solutions,
    }


def analyze_hand(hand: dict) -> str:
    """Run full multi-street analysis and return natural language summary."""
    return _run_analysis(hand)["text"]


def analyze_hand_full(hand: dict) -> dict:
    """Run full analysis and return structured data for caching.

    Returns dict with keys:
        text: formatted natural language summary
        gametype, depth, hero_position, hero_hand, preflop_actions
        street_states: {flop/turn/river: {board, flop_actions, turn_actions, river_actions}}
        hero_spots: list of {street, params, ...} per hero decision
        solutions: list of raw spot-solution API responses (parallel to hero_spots)
    """
    return _run_analysis(hand)


def main():
    parser = argparse.ArgumentParser(description="Analyze poker hand vs GTO")
    parser.add_argument("--json", required=True, help="Hand description as JSON string")
    args = parser.parse_args()

    hand = json.loads(args.json)
    result = analyze_hand(hand)
    print(result)


if __name__ == "__main__":
    main()
