#!/usr/bin/env python3
"""Convert GTO Wizard API JSON into natural language summaries."""


def format_action_summary(spot_solution: dict) -> str:
    """Format overall action frequencies for the active position."""
    game = spot_solution["game"]
    position = game["active_position"]
    board = game["board"]
    street = game["current_street"]["type"].capitalize()
    pot = game["pot"]
    bet_name = game["bet_display_name"]

    lines = [f"【{position} 在 {street} {board}】底池 {pot}bb"]

    for sol in spot_solution["action_solutions"]:
        act = sol["action"]
        code = act["code"]
        freq = sol["total_frequency"]
        combos = sol["total_combos"]

        if freq < 0.001:
            continue

        if code == "X":
            label = "Check"
        elif code == "F":
            label = "Fold"
        elif code == "C":
            label = "Call"
        elif code.startswith("R"):
            if act.get("allin"):
                label = f"All-in {act['betsize']}"
            else:
                pct = float(act.get("betsize_by_pot", 0)) * 100
                label = f"{bet_name} {act['betsize']}（{pct:.0f}% pot）"
        else:
            label = code

        lines.append(f"  {label}: {freq*100:.1f}%（{combos:.0f} combos）")

    return "\n".join(lines)


def format_hand_detail(spot_solution: dict, hand_name: str, position: str) -> str:
    """Format detailed strategy for a specific hand at a specific position."""
    # Find the player info for the position
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break

    if not player_info:
        return f"找不到 {position} 的資料"

    shc = player_info.get("simple_hand_counters", {})
    hand_data = shc.get(hand_name)

    if not hand_data:
        return f"{position} range 中沒有 {hand_name}"

    combos_avail = hand_data["total_combos_available"]
    combos_in_range = hand_data["total_combos"]
    freq_in_range = hand_data["total_frequency"]
    ev = hand_data.get("hand_ev", 0)
    eq = hand_data.get("hand_eq", 0)

    lines = [
        f"【{position} {hand_name}】",
        f"  Range 頻率: {freq_in_range*100:.1f}%（{combos_in_range:.1f}/{combos_avail:.0f} combos）",
        f"  EV: {ev:.2f}bb | Equity: {eq*100:.1f}%",
    ]

    # Per-action breakdown
    actions_freq = hand_data.get("actions_total_frequencies", {})
    actions_combos = hand_data.get("actions_total_combos", {})
    if actions_freq:
        lines.append("  策略:")
        for action_code, freq in sorted(actions_freq.items(), key=lambda x: -x[1]):
            if freq < 0.001:
                continue
            combos = actions_combos.get(action_code, 0)
            action_label = _action_label(action_code, spot_solution)
            lines.append(f"    {action_label}: {freq*100:.1f}%（{combos:.1f} combos）")

    return "\n".join(lines)


def format_full_spot(spot_solution: dict, hero_hand: str = None, hero_position: str = None) -> str:
    """Format complete spot analysis including action summary and hero hand detail."""
    parts = [format_action_summary(spot_solution)]

    if hero_hand and hero_position:
        parts.append("")
        parts.append(format_hand_detail(spot_solution, hero_hand, hero_position))

    return "\n".join(parts)


def _action_label(code: str, spot_solution: dict) -> str:
    """Convert action code to readable label."""
    if code == "X":
        return "Check"
    if code == "F":
        return "Fold"
    if code == "C":
        return "Call"
    if code == "RAI":
        return "All-in"

    # Look up betsize from action_solutions
    for sol in spot_solution["action_solutions"]:
        if sol["action"]["code"] == code:
            act = sol["action"]
            if act.get("allin"):
                return f"All-in {act['betsize']}"
            pct = float(act.get("betsize_by_pot", 0)) * 100
            return f"Bet {act['betsize']}（{pct:.0f}% pot）"

    return code
