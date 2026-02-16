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
from gto_formatter import format_full_spot

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


def analyze_hand(hand: dict) -> str:
    """Run full multi-street analysis and return natural language summary.

    Two-phase approach for speed:
      Phase 1 (sequential, lightweight): Walk through actions, discover solver
              bet codes via next-actions API. Only called for bets/raises.
      Phase 2 (parallel): Fire all spot-solution calls simultaneously.
    """
    t0 = time.time()
    gametype = hand.get("gametype", "MTTGeneral")
    depth = nearest_depth(hand["effective_bb"])
    hero_pos = hand["hero_position"]
    hero_hand = hand["hero_hand"]
    streets = hand.get("streets", [])

    # Normalize preflop actions (R2 → R2.1, etc.)
    raw_preflop = hand["preflop_actions"]
    preflop_actions = _normalize_preflop_actions(raw_preflop, gametype, depth)

    # ── Phase 1: Walk hand, discover bet codes, collect hero spots ──
    hero_spots = []

    # Preflop hero spot
    preflop_before = _preflop_before_hero(preflop_actions, hero_pos)
    hero_spots.append({
        "header": "【Preflop】",
        "params": dict(gametype=gametype, depth=depth, preflop_actions=preflop_before),
        "action_desc": None,
    })

    board = ""
    flop_acts = ""
    turn_acts = ""
    river_acts = ""

    for street_idx, street in enumerate(streets):
        if street_idx == 0:
            board = street["board"]
            street_header = f"【Flop: {board}】"
        elif street_idx == 1:
            board += street["card"]
            street_header = f"【Turn: {street['card']}（Board: {board}）】"
        elif street_idx == 2:
            board += street["card"]
            street_header = f"【River: {street['card']}（Board: {board}）】"

        street_first_hero = True

        for act in street["actions"]:
            pos = act["position"]
            action_type = act["action"]
            target_size = act.get("size", 0)

            if pos == hero_pos:
                # Hero's decision — record for parallel analysis
                params = dict(
                    gametype=gametype, depth=depth,
                    preflop_actions=preflop_actions, board=board,
                    flop_actions=flop_acts, turn_actions=turn_acts,
                    river_actions=river_acts,
                )

                if action_type in ("X", "C", "F"):
                    taken_code = action_type
                else:
                    # Lightweight API call to discover closest solver bet code
                    next_resp = get_next_actions(**params)
                    avail = next_resp["next_actions"]["available_actions"]
                    taken_code = find_closest_action(avail, target_size)

                size_str = f" {target_size}bb" if target_size else ""
                hero_spots.append({
                    "header": street_header if street_first_hero else None,
                    "params": params,
                    "action_desc": f"  → 實際行動: {pos} {action_type}{size_str}（solver code: {taken_code}）",
                })
                street_first_hero = False
            else:
                # Opponent action — skip spot-solution, just discover code if needed
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
