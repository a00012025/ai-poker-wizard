#!/usr/bin/env python3
"""Convert GTO Wizard API JSON into natural language summaries."""

_RANK_ORDER = "AKQJT98765432"


def normalize_hand_name(hand: str) -> str:
    """Convert specific combo (Qs6d) to simplified name (Q6o).

    GTO Wizard uses simplified names: 66, AKs, Q6o.
    LLM may output specific combos: 6h6s, AhKh, Qs6d.
    """
    if not hand:
        return ""
    if len(hand) <= 3:
        return hand  # "66", "AKs", "Q6o" — already simplified
    if len(hand) == 4:
        r1, s1, r2, s2 = hand[0], hand[1], hand[2], hand[3]
        # Higher rank first
        if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
            r1, s1, r2, s2 = r2, s2, r1, s1
        if r1 == r2:
            return r1 + r2  # Pair: "66"
        suffix = "s" if s1 == s2 else "o"
        return r1 + r2 + suffix
    return hand


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
    hand_name = normalize_hand_name(hand_name)
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


def format_range_overview(spot_solution: dict, position: str) -> str:
    """Format full range breakdown for a position at a spot.

    Shows all hands sorted by combos, with EV and equity.
    Used for follow-up questions like "what does BB have on the river?"
    """
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break

    if not player_info:
        return f"找不到 {position} 的資料"

    shc = player_info.get("simple_hand_counters", {})
    if not shc:
        return f"{position} 沒有 range 資料"

    game = spot_solution["game"]
    street = game["current_street"]["type"].capitalize()
    board = game["board"]

    # Collect hands with combos > 0
    hands = []
    total_combos = 0
    for hand_name, data in shc.items():
        combos = data.get("total_combos", 0)
        if combos < 0.01:
            continue
        hands.append({
            "name": hand_name,
            "combos": combos,
            "freq": data.get("total_frequency", 0),
            "ev": data.get("hand_ev", 0),
            "eq": data.get("hand_eq", 0),
        })
        total_combos += combos

    hands.sort(key=lambda h: -h["combos"])

    lines = [f"【{position} 在 {street} {board} 的範圍】共 {total_combos:.1f} combos"]

    for h in hands:
        pct = (h["combos"] / total_combos * 100) if total_combos else 0
        lines.append(
            f"  {h['name']}: {h['combos']:.1f} combos（{pct:.1f}%）"
            f" | EV {h['ev']:.2f}bb | Eq {h['eq']*100:.1f}%"
        )

    return "\n".join(lines)


def format_range_by_action(spot_solution: dict, position: str) -> str:
    """Format range grouped by action for a position.

    Shows which hands take each action, with mixed frequencies highlighted.
    Used for questions like "SB all-in / 3bet range 分別有哪些牌？"
    """
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break

    if not player_info:
        return f"找不到 {position} 的資料"

    shc = player_info.get("simple_hand_counters", {})
    if not shc:
        return f"{position} 沒有 range 資料"

    game = spot_solution["game"]
    street = game["current_street"]["type"].capitalize()
    board = game["board"]

    # Group hands by action
    action_groups: dict[str, list] = {}  # {action_code: [(hand, freq, combos)]}

    for hand_name, data in shc.items():
        actions_freq = data.get("actions_total_frequencies", {})
        actions_combos = data.get("actions_total_combos", {})

        if not actions_freq:
            continue

        for action_code, freq in actions_freq.items():
            if freq < 0.001:
                continue
            combos = actions_combos.get(action_code, 0)
            if combos < 0.01:
                continue
            if action_code not in action_groups:
                action_groups[action_code] = []
            action_groups[action_code].append((hand_name, freq, combos))

    if not action_groups:
        return f"{position} 沒有動作分佈資料"

    # Sort each group by combos descending
    for code in action_groups:
        action_groups[code].sort(key=lambda x: -x[2])

    # Order action groups by total combos descending, but put Fold last
    def sort_key(code):
        if code == "F":
            return (1, 0)
        return (0, -sum(x[2] for x in action_groups[code]))

    lines = [f"【{position} 在 {street} {board} 的策略分佈】"]

    for code in sorted(action_groups.keys(), key=sort_key):
        group = action_groups[code]
        total = sum(x[2] for x in group)
        label = _action_label(code, spot_solution)
        lines.append(f"\n{label}（{total:.0f} combos）:")

        for hand_name, freq, combos in group:
            if freq >= 0.995:
                lines.append(f"  {hand_name}: {combos:.1f}")
            else:
                lines.append(f"  {hand_name}: {combos:.1f}（{freq*100:.0f}%）")

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
