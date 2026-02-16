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
    find_closest_action, nearest_depth,
)
from gto_formatter import format_full_spot, normalize_hand_name

POSITION_ORDER = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def _normalize_preflop_actions(preflop_actions: str, gametype: str, depth: float) -> str:
    """Validate and correct preflop action codes against the solver.

    LLM may output R2 but solver expects R2.1. Walk through each action,
    and for raises, discover the correct code via next-actions API.
    """
    parts = preflop_actions.split("-")
    corrected = []
    for i, code in enumerate(parts):
        if code in ("F", "C", "AI"):
            corrected.append(code)
        elif code.startswith("R"):
            # Raise — discover correct code from solver
            actions_so_far = "-".join(corrected) if corrected else ""
            try:
                resp = get_next_actions(
                    gametype=gametype, depth=depth,
                    preflop_actions=actions_so_far,
                )
                avail = resp["next_actions"]["available_actions"]
                target = float(code[1:])  # R2 → 2.0, R2.1 → 2.1
                correct_code = find_closest_action(avail, target)
                corrected.append(correct_code)
            except Exception:
                corrected.append(code)  # fallback to original
        else:
            corrected.append(code)
    return "-".join(corrected)


def _preflop_before_hero(preflop_actions: str, hero_position: str) -> str:
    """Get preflop action string up to (but not including) hero's action."""
    parts = preflop_actions.split("-")
    hero_idx = POSITION_ORDER.index(hero_position)
    before = parts[:hero_idx]
    return "-".join(before) if before else ""


STREET_NAMES = ["flop", "turn", "river"]


def _simplify_multiway(hand: dict, hero_pos: str, gametype: str, depth: float) -> tuple[str, float, str]:
    """Detect multiway pot and simplify to heads-up if needed.

    Returns (preflop_actions, adjusted_depth, simplification_note).
    If not multiway, returns (original_preflop, original_depth, "").
    """
    preflop = hand["preflop_actions"]
    streets = hand.get("streets", [])
    parts = preflop.split("-")

    # Count non-fold actions in first 8 positions
    non_fold = [i for i in range(min(len(parts), 8)) if parts[i] not in ("F", "")]
    if len(non_fold) <= 2:
        return preflop, depth, ""

    # Multiway — find who's on the flop
    if not streets:
        return preflop, depth, ""

    flop_positions = []
    seen = set()
    for act in streets[0]["actions"]:
        pos = act["position"]
        if pos not in seen:
            flop_positions.append(pos)
            seen.add(pos)

    if len(flop_positions) != 2 or hero_pos not in flop_positions:
        return preflop, depth, ""

    villain_pos = next(p for p in flop_positions if p != hero_pos)
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
    is_3bet = second_action.startswith("R")

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
    # In 3bet pots, dead money is amplified: callers' money inflates the pot,
    # causing larger 3bet sizing and call, roughly 3x the raw dead money
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
        # Second actor raised → 3bet pot
        prefix = "-".join(simplified[:second_idx])
        try:
            resp = get_next_actions(gametype=gametype, depth=adjusted_depth,
                                    preflop_actions=prefix)
            avail = resp["next_actions"]["available_actions"]
            second_size = float(second_action[1:])
            second_code = find_closest_action(avail, second_size)
        except Exception:
            second_code = second_action
        simplified[second_idx] = second_code

        # First actor calls the 3bet
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

    return full, adjusted_depth, "\n".join(note_parts)


def _run_analysis(hand: dict) -> dict:
    """Core analysis: walk hand, discover bet codes, fetch spot solutions.

    Returns structured data with both formatted text and raw solutions
    for caching and follow-up queries.
    """
    t0 = time.time()
    gametype = hand.get("gametype", "MTTGeneral")
    depth = nearest_depth(hand["effective_bb"])
    hero_pos = hand["hero_position"]
    hero_hand = normalize_hand_name(hand["hero_hand"])
    streets = hand.get("streets", [])

    # Detect multiway and simplify to heads-up if needed
    raw_preflop = hand["preflop_actions"]
    multiway_note = ""
    simplified_preflop, adjusted_depth, multiway_note = _simplify_multiway(
        hand, hero_pos, gametype, depth,
    )
    if multiway_note:
        raw_preflop = simplified_preflop
        depth = adjusted_depth

    # Normalize preflop actions (R2 → R2.1, etc.)
    preflop_actions = _normalize_preflop_actions(raw_preflop, gametype, depth)

    # ── Phase 1: Walk hand, discover bet codes, collect hero spots ──
    hero_spots = []

    # Preflop hero spot
    preflop_before = _preflop_before_hero(preflop_actions, hero_pos)
    hero_spots.append({
        "street": "preflop",
        "header": "【Preflop】",
        "params": dict(gametype=gametype, depth=depth, preflop_actions=preflop_before),
        "action_desc": None,
    })

    board = ""
    flop_acts = ""
    turn_acts = ""
    river_acts = ""

    # Track action strings at each street boundary (for hypothetical queries)
    street_states = {}

    for street_idx, street in enumerate(streets):
        street_name = STREET_NAMES[street_idx]

        if street_idx == 0:
            board = street["board"]
            street_header = f"【Flop: {board}】"
        elif street_idx == 1:
            board += street["card"]
            street_header = f"【Turn: {street['card']}（Board: {board}）】"
        elif street_idx == 2:
            board += street["card"]
            street_header = f"【River: {street['card']}（Board: {board}）】"

        # Snapshot state at start of this street (before actions)
        street_states[street_name] = {
            "board": board,
            "flop_actions": flop_acts,
            "turn_actions": turn_acts,
            "river_actions": river_acts,
        }

        street_first_hero = True

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)

            if pos == hero_pos:
                params = dict(
                    gametype=gametype, depth=depth,
                    preflop_actions=preflop_actions, board=board,
                    flop_actions=flop_acts, turn_actions=turn_acts,
                    river_actions=river_acts,
                )

                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                else:
                    next_resp = get_next_actions(**params)
                    avail = next_resp["next_actions"]["available_actions"]
                    taken_code = find_closest_action(avail, target_size)

                size_str = f" {target_size}bb" if target_size else ""
                hero_spots.append({
                    "street": street_name,
                    "header": street_header if street_first_hero else None,
                    "params": params,
                    "action_desc": f"  → 實際行動: {pos} {action_type}{size_str}（solver code: {taken_code}）",
                })
                street_first_hero = False
            else:
                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                else:
                    params = dict(
                        gametype=gametype, depth=depth,
                        preflop_actions=preflop_actions, board=board,
                        flop_actions=flop_acts, turn_actions=turn_acts,
                        river_actions=river_acts,
                    )
                    next_resp = get_next_actions(**params)
                    avail = next_resp["next_actions"]["available_actions"]
                    taken_code = find_closest_action(avail, target_size)

            # Advance action string
            if street_idx == 0:
                flop_acts = f"{flop_acts}-{taken_code}" if flop_acts else taken_code
            elif street_idx == 1:
                turn_acts = f"{turn_acts}-{taken_code}" if turn_acts else taken_code
            elif street_idx == 2:
                river_acts = f"{river_acts}-{taken_code}" if river_acts else taken_code

    t_phase1 = time.time()

    # ── Phase 2: Fetch all spot solutions in parallel ──
    solutions = [None] * len(hero_spots)
    with ThreadPoolExecutor(max_workers=len(hero_spots)) as executor:
        future_to_idx = {
            executor.submit(get_spot_solution, **spot["params"]): i
            for i, spot in enumerate(hero_spots)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            solutions[idx] = future.result()

    t_phase2 = time.time()

    # ── Phase 3: Format results ──
    results = []
    results.append("=" * 50)
    results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth - 0.125:.0f}bb solver）")
    results.append(f"Hero: {hero_pos} {hero_hand}")
    if multiway_note:
        results.append(multiway_note)
    if raw_preflop != preflop_actions:
        results.append(f"Preflop actions 校正: {raw_preflop} → {preflop_actions}")
    results.append("")

    for spot, sol in zip(hero_spots, solutions):
        if spot["header"]:
            results.append("")
            results.append("=" * 50)
            results.append(spot["header"])

        if sol:
            spot_text = format_full_spot(sol, hero_hand, hero_pos)
            results.append(spot_text)
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
