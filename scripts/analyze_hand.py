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
from gto_formatter import format_full_spot, format_ev_comparison, normalize_hand_name

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


def _normalize_preflop_actions(preflop_actions: str, gametype: str, depth: float, stacks: str = "") -> str:
    """Validate and correct preflop action codes against the solver.

    LLM may output R2 but solver expects R2.1. Walk through each action,
    and for raises, discover the correct code via next-actions API.
    """
    parts = preflop_actions.split("-")
    corrected = []
    for i, code in enumerate(parts):
        if code in ("F", "C"):
            corrected.append(code)
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

    # Multiway — find who remains
    if streets:
        # Use flop actions to determine remaining players
        flop_positions = []
        seen = set()
        folded_on_flop = set()
        for act in streets[0]["actions"]:
            pos = act["position"]
            if pos not in seen:
                flop_positions.append(pos)
                seen.add(pos)
            if act["action"] == "F":
                folded_on_flop.add(pos)

        remaining = [p for p in flop_positions if p not in folded_on_flop]

        if len(remaining) == 2 and hero_pos in remaining:
            villain_pos = next(p for p in remaining if p != hero_pos)
        elif len(remaining) == 1 and remaining[0] == hero_pos:
            # Everyone folded to hero — find the last non-hero bettor as villain
            villain_pos = None
            for act in reversed(streets[0]["actions"]):
                if act["position"] != hero_pos and act["action"] not in ("X", "F"):
                    villain_pos = act["position"]
                    break
            if not villain_pos:
                return preflop, depth, "", None
        else:
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
    gametype = hand.get("gametype", "MTTGeneral")
    depth = nearest_depth(hand["effective_bb"])
    hero_pos = hand["hero_position"]
    hero_hand_raw = hand["hero_hand"]
    hero_hand = normalize_hand_name(hero_hand_raw)
    # Compute 1326-combo index for exact postflop lookup (e.g. Ah6h vs generic A6s)
    from gto_formatter import _COMBO_RANKS, _COMBO_SUITS
    hero_combo_idx = None
    if len(hero_hand_raw) == 4:
        try:
            _ci1 = _COMBO_RANKS.index(hero_hand_raw[0]) * 4 + _COMBO_SUITS.index(hero_hand_raw[1])
            _ci2 = _COMBO_RANKS.index(hero_hand_raw[2]) * 4 + _COMBO_SUITS.index(hero_hand_raw[3])
            _j, _i = max(_ci1, _ci2), min(_ci1, _ci2)
            if _j != _i:
                hero_combo_idx = _j * (_j - 1) // 2 + _i
        except (ValueError, IndexError):
            pass
    streets = hand.get("streets") or hand.get("postflop_actions", [])

    # Fix malformed streets: if first street has 4+ card board, split into flop + turn
    # (LLM sometimes collapses check-check flop into the turn entry)
    streets = _fix_collapsed_streets(streets)

    # Determine position order based on number of players
    num_players = len(hand.get("player_stacks", []))
    if not num_players:
        num_players = hand.get("players_at_table", 0) or hand.get("num_players", 0)
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

    # GTO Wizard's MTTGeneral API always expects 8 positions.
    # For tables < 8 players, pad preflop actions with folds for missing positions.
    if gametype == "MTTGeneral" and num_players < 8:
        pad_count = 8 - num_players
        padding = "-".join(["F"] * pad_count)
        hand = dict(hand)  # shallow copy to avoid mutating original
        hand["preflop_actions"] = padding + "-" + hand["preflop_actions"]
        if hand.get("player_stacks"):
            # Pad player_stacks too (shouldn't happen often for image-parsed hands)
            hand["player_stacks"] = [0] * pad_count + hand["player_stacks"]
        num_players = 8

    pos_order = _get_position_order(num_players)

    # ICM support: resolve gametype and stacks
    icm_stacks = ""
    icm_note = ""
    is_icm = hand.get("tournament_type") == "icm"
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
    preflop_before = _preflop_before_hero(preflop_actions, hero_pos, pos_order)
    hero_spots.append({
        "street": "preflop",
        "header": "【Preflop】",
        "params": dict(gametype=gametype, depth=depth, stacks=icm_stacks,
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

        # Postflop uses chip EV for ICM modes (preflop_only)
        post_gametype = chipev_gametype if is_icm else gametype
        post_depth = chipev_depth if is_icm else depth
        # Normalize preflop for chip EV context (only once on flop)
        if is_icm and street_idx == 0:
            chipev_preflop = _normalize_preflop_actions(
                hand["preflop_actions"], chipev_gametype, chipev_depth,
            )

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)

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
                        taken_code = find_closest_action_postflop(avail, target_size)

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
                        taken_code = find_closest_action_postflop(avail, target_size)

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

    t_phase2 = time.time()

    # ── Phase 3: Format results ──
    results = []
    results.append("=" * 50)
    if is_icm:
        depth_display = depth if isinstance(depth, str) else f"{depth}"
        results.append(f"Hero: {hero_pos} {hero_hand}")
        if icm_note:
            results.append(icm_note)
        if streets:
            results.append(f"Postflop 使用 Chip EV {chipev_depth - 0.125:.0f}bb solver（ICM 僅支援 preflop）")
    elif is_cash:
        results.append(f"Cash Game {num_players}-max")
        results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth:.0f}bb solver）")
        results.append(f"Hero: {hero_pos} {hero_hand}")
    else:
        results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth - 0.125:.0f}bb solver）")
        results.append(f"Hero: {hero_pos} {hero_hand}")
    if icm_fallback_note:
        results.append(icm_fallback_note)
    if multiway_note:
        results.append(multiway_note)
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
            spot_street = spot["street"]
            if spot_street != "preflop" and spot_street in street_states:
                spot_board = street_states[spot_street].get("board", "")
                if spot_board:
                    eval_result = _eval_hand(hero_hand, spot_board)
                    if eval_result["full_label"]:
                        results.append(f"Hero {hero_hand} 牌型: {eval_result['full_label']}")

        if sol:
            spot_text = format_full_spot(sol, hero_hand, hero_pos)
            results.append(spot_text)

            # Show EV loss if hero took a suboptimal action
            taken_code = spot.get("taken_code")
            if taken_code:
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

    return {
        "text": "\n".join(results),
        "hand": hand,
        "gametype": gametype,
        "depth": depth,
        "stacks": icm_stacks,
        "is_icm": is_icm,
        "hero_position": hero_pos,
        "hero_hand": hero_hand,
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
