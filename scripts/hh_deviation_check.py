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
    find_closest_action, find_closest_action_postflop, nearest_depth,
)
from gto_formatter import (
    normalize_hand_name, _COMBO_INDEX, _RANK_ORDER,
    _get_board_cards, _combo_to_hand_name,
)

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


def _get_hand_ev(solution: dict, hero_hand: str, hero_pos: str, is_preflop: bool) -> float | None:
    """Extract EV for hero's hand from a spot solution.

    Uses simple_hand_counters first (pre-computed per-hand EV), then falls back
    to the raw hand_evs array. For postflop, averages across in-range combos.

    Returns EV in bb, or None if unavailable.
    """
    for pi in solution.get("players_info", []):
        if pi["player"]["position"] != hero_pos:
            continue

        # Try simple_hand_counters first (has pre-computed per-hand EV)
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


def _get_postflop_hand_freqs(solution: dict, hero_hand: str, hero_pos: str) -> dict[str, float] | None:
    """Extract per-action frequencies for hero's hand from 1326-element postflop arrays.

    Averages across all combos of the hand that are in range.
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

    board_cards = _get_board_cards(solution["game"]["board"])
    action_solutions = solution["action_solutions"]

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


def _get_action_label(action_solutions: list[dict], code: str) -> str:
    """Get human-readable label for an action code."""
    labels = {"X": "Check", "C": "Call", "F": "Fold"}
    if code in labels:
        return labels[code]
    for asol in action_solutions:
        if asol["action"]["code"] == code:
            act = asol["action"]
            if act.get("allin"):
                return f"All-in {act.get('betsize', '')}bb"
            return act.get("display_name", code)
    return code


def _normalize_preflop_action(code: str, gametype: str, depth: float,
                               preflop_so_far: str) -> str:
    """Map a raw preflop action code to the solver's action code."""
    if code in ("F", "C", "X"):
        return code
    try:
        resp = get_next_actions(gametype=gametype, depth=depth,
                                preflop_actions=preflop_so_far)
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
            target = float(code[1:])
            return find_closest_action(avail, target)
    except Exception:
        pass
    return code


def check_hand(hand: dict) -> list[dict]:
    """Check a single hand for GTO deviations.

    Returns list of deviation dicts, each containing:
        street, hero_action, hero_freq, gto_action, gto_freq, actions_detail
    """
    gametype = "MTTGeneral"
    hero_pos = hand["hero_position"]
    hero_hand_raw = hand["hero_hand"]
    hero_hand = normalize_hand_name(hero_hand_raw)
    num_players = hand.get("num_players", hand.get("table_size", 8))
    depth = nearest_depth(hand["effective_bb"])
    streets = hand.get("streets", [])

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

    # Normalize preflop actions up to hero
    normalized_parts = []
    for i in range(hero_idx_8):
        code = pf_parts_8[i]
        so_far = "-".join(normalized_parts) if normalized_parts else ""
        norm_code = _normalize_preflop_action(code, gametype, depth, so_far)
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
        hero_pf_action_raw, gametype, depth, pf_before_hero_norm
    )

    # Query solver for preflop
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
            })

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
            norm_code = _normalize_preflop_action(code, gametype, depth, so_far)
            full_first_round.append(norm_code)
        full_first_pf = "-".join(full_first_round)

        # Hero's continuation action
        active = [i for i in range(num_players) if pf_parts_n[i] not in ("F", "")]
        cont_idx = 0
        hero_cont_raw = None
        for j in range(num_players, len(pf_parts_n)):
            if cont_idx >= len(active):
                cont_idx = 0
            if active[cont_idx] == hero_idx_n:
                hero_cont_raw = pf_parts_n[j]
                break
            cont_idx += 1

        if hero_cont_raw:
            hero_cont = _normalize_preflop_action(hero_cont_raw, gametype, depth, full_first_pf)
            try:
                sol2 = get_spot_solution(gametype=gametype, depth=depth,
                                          preflop_actions=full_first_pf)
            except Exception:
                sol2 = None

            if sol2:
                freqs2 = _get_preflop_hand_freqs(sol2, hero_hand, hero_pos)
                if freqs2:
                    hero_freq2 = freqs2.get(hero_cont, 0)
                    best2 = max(freqs2, key=freqs2.get)
                    best_freq2 = freqs2[best2]

                    hand_ev2 = _get_hand_ev(sol2, hero_hand, hero_pos, is_preflop=True)
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
                    })

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
                else:
                    try:
                        next_resp = get_next_actions(**params)
                        avail = next_resp["next_actions"]["available_actions"]
                        taken_code = find_closest_action_postflop(avail, target_size)
                    except Exception:
                        taken_code = action_type

                try:
                    sol_post = get_spot_solution(**params)
                except Exception:
                    sol_post = None

                if sol_post:
                    freqs_post = _get_postflop_hand_freqs(sol_post, hero_hand, hero_pos)
                    if freqs_post:
                        hero_freq_post = freqs_post.get(taken_code, 0)
                        best_post = max(freqs_post, key=freqs_post.get)
                        best_freq_post = freqs_post[best_post]

                        hand_ev_post = _get_hand_ev(sol_post, hero_hand, hero_pos, is_preflop=False)
                        deviations.append({
                            "street": street_name,
                            "spot": f"board {board}",
                            "hero_action": taken_code,
                            "hero_action_label": _get_action_label(sol_post["action_solutions"], taken_code),
                            "hero_freq": hero_freq_post,
                            "gto_action": best_post,
                            "gto_action_label": _get_action_label(sol_post["action_solutions"], best_post),
                            "gto_freq": best_freq_post,
                            "all_freqs": freqs_post,
                            "hero_ev": hand_ev_post,
                        })

            # Advance action strings
            if action_type in ("X", "C", "F"):
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
                    taken = find_closest_action_postflop(avail, target_size)
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
    args = parser.parse_args()

    path = Path(args.path)
    if path.is_dir():
        hands = parse_directory(path, include_folds=True)
    else:
        hands = parse_file(path, include_folds=True)

    print(f"Parsed {len(hands)} hero hands (including preflop folds)\n")

    if args.limit:
        hands = hands[:args.limit]

    all_results = []
    all_deviations = []

    for i, hand in enumerate(hands):
        hand_id = hand.get("hand_id", "?")
        hero_pos = hand["hero_position"]
        hero_hand = hand["hero_hand"]
        eff_bb = hand["effective_bb"]
        num_streets = len(hand.get("streets", []))

        print(f"[{i+1}/{len(hands)}] {hand_id}: {hero_pos} {hero_hand} ({eff_bb:.1f}bb) "
              f"streets={num_streets}", end=" ... ", flush=True)

        t0 = time.time()
        try:
            devs = check_hand(hand)
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

    # Sort deviations by severity (lowest hero_freq first)
    all_deviations.sort(key=lambda d: d["hero_freq"])

    if not all_deviations:
        print(f"\nNo significant deviations found (threshold: {args.threshold}%)")
    else:
        print(f"\nFound {len(all_deviations)} spots where hero deviated from GTO:")
        print(f"(threshold: hero's action had <{100 - args.threshold:.0f}% GTO frequency)\n")

        for d in all_deviations:
            hero_pct = d["hero_freq"] * 100
            gto_pct = d["gto_freq"] * 100
            ev_str = f" EV={d['hero_ev']:.2f}bb" if d.get("hero_ev") is not None else ""
            print(f"  {d['hand_id']}: {d['hero_position']} {d['hero_hand']} "
                  f"({d['effective_bb']:.0f}bb) [{d['street']}]{ev_str}")
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
