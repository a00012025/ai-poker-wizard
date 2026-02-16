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
    "preflop_actions": "F-F-F-R2.1-F-F-F-C",
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gto_api import get_spot_solution, find_closest_action_from_solutions, nearest_depth
from gto_formatter import format_full_spot

POSITION_ORDER = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]


def _preflop_before_hero(preflop_actions: str, hero_position: str) -> str:
    """Get preflop action string up to (but not including) hero's action.

    This gives us the decision point where hero needs to act.
    Example: hero=CO(idx 4), actions=F-F-F-R2.1-F-F-F-C
    → We want "F-F-F" (before CO's R2.1), but CO is idx 4, so 4 folds before.
    Wait — positions are UTG(0)..BB(7), but SB/BB post blinds first.
    In GTO Wizard preflop_actions, the actions go in position order: UTG, UTG+1, ..., BB.
    """
    parts = preflop_actions.split("-")
    hero_idx = POSITION_ORDER.index(hero_position)
    before = parts[:hero_idx]
    return "-".join(before) if before else ""


def analyze_hand(hand: dict) -> str:
    """Run full multi-street analysis and return natural language summary."""
    gametype = hand.get("gametype", "MTTGeneral")
    depth = nearest_depth(hand["effective_bb"])
    preflop_actions = hand["preflop_actions"]
    hero_pos = hand["hero_position"]
    hero_hand = hand["hero_hand"]
    streets = hand.get("streets", [])

    results = []

    # --- Preflop analysis ---
    results.append("=" * 50)
    results.append(f"籌碼深度: {hand['effective_bb']}bb（使用 {depth - 0.125:.0f}bb solver）")
    results.append(f"Hero: {hero_pos} {hero_hand}")
    results.append("")

    # Query hero's preflop decision point
    preflop_before = _preflop_before_hero(preflop_actions, hero_pos)
    preflop_sol = get_spot_solution(
        gametype=gametype, depth=depth,
        preflop_actions=preflop_before,
    )
    if preflop_sol:
        preflop_summary = format_full_spot(preflop_sol, hero_hand, hero_pos)
        results.append("【Preflop】")
        results.append(preflop_summary)
    else:
        results.append("【Preflop】（無 solver 數據）")

    if not streets:
        return "\n".join(results)

    # --- Postflop analysis ---
    board = ""
    flop_actions_str = ""
    turn_actions_str = ""
    river_actions_str = ""

    for street_idx, street in enumerate(streets):
        results.append("")
        results.append("=" * 50)

        if street_idx == 0:
            board = street["board"]
            results.append(f"【Flop: {board}】")
        elif street_idx == 1:
            board += street["card"]
            results.append(f"【Turn: {street['card']}（Board: {board}）】")
        elif street_idx == 2:
            board += street["card"]
            results.append(f"【River: {street['card']}（Board: {board}）】")

        current_flop = flop_actions_str
        current_turn = turn_actions_str
        current_river = river_actions_str

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)

            # For trivial actions (check/call/fold), skip spot_solution if it returns 204
            # to save API calls. Only fetch full solution when needed for analysis.
            sol = get_spot_solution(
                gametype=gametype, depth=depth,
                preflop_actions=preflop_actions, board=board,
                flop_actions=current_flop, turn_actions=current_turn,
                river_actions=current_river,
            )

            if sol:
                spot_text = format_full_spot(sol, hero_hand, hero_pos)
                results.append(spot_text)

            # Determine action code taken
            if action_type in ("X", "C", "F"):
                taken_code = action_type
            elif sol:
                # Use action_solutions from spot_solution (no extra API call)
                taken_code = find_closest_action_from_solutions(
                    sol["action_solutions"], target_size
                )
            else:
                taken_code = action_type  # fallback

            # Advance action string
            if street_idx == 0:
                current_flop = f"{current_flop}-{taken_code}" if current_flop else taken_code
            elif street_idx == 1:
                current_turn = f"{current_turn}-{taken_code}" if current_turn else taken_code
            elif street_idx == 2:
                current_river = f"{current_river}-{taken_code}" if current_river else taken_code

            size_str = f" {target_size}bb" if target_size else ""
            results.append(f"  → 實際行動: {pos} {action_type}{size_str}（solver code: {taken_code}）")
            results.append("")

        # Persist action strings for next street
        flop_actions_str = current_flop
        turn_actions_str = current_turn
        river_actions_str = current_river

    return "\n".join(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze poker hand vs GTO")
    parser.add_argument("--json", required=True, help="Hand description as JSON string")
    args = parser.parse_args()

    hand = json.loads(args.json)
    result = analyze_hand(hand)
    print(result)


if __name__ == "__main__":
    main()
