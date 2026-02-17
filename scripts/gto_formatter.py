#!/usr/bin/env python3
"""Convert GTO Wizard API JSON into natural language summaries."""

_RANK_ORDER = "AKQJT98765432"
_COMBO_RANKS = "23456789TJQKA"
_COMBO_SUITS = "cdhs"
_COMBO_CARDS = [r + s for r in _COMBO_RANKS for s in _COMBO_SUITS]

# Build 1326 combo index: outer j=1..51, inner i=0..j-1, combo=(cards[j], cards[i])
_COMBO_INDEX: list[tuple[str, str]] = []
for _j in range(1, 52):
    for _i in range(_j):
        _COMBO_INDEX.append((_COMBO_CARDS[_j], _COMBO_CARDS[_i]))


def _get_board_cards(board: str) -> set[str]:
    """Parse board string like 'KsKhQd7h' into a set of cards."""
    cards = set()
    for k in range(0, len(board), 2):
        cards.add(board[k:k + 2])
    return cards


def _combo_to_hand_name(c1: str, c2: str) -> str:
    """Convert combo pair to simplified hand name (e.g. ATo, ATs, AA)."""
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    i1 = _RANK_ORDER.index(r1)
    i2 = _RANK_ORDER.index(r2)
    if i1 > i2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1 + r2
    suffix = "s" if s1 == s2 else "o"
    return r1 + r2 + suffix


def _get_combo_strategies(spot_solution: dict, hand_name: str, position: str) -> list[dict] | None:
    """Extract per-combo strategies from the 1326-length strategy arrays.

    Returns list of {combo, range, ev, actions: {code: freq}} for each
    non-blocked combo belonging to hand_name, or None if data unavailable.
    """
    if "action_solutions" not in spot_solution:
        return None
    action_solutions = spot_solution["action_solutions"]
    if not action_solutions or "strategy" not in action_solutions[0]:
        return None

    # Find player range array
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    if len(range_arr) != 1326:
        return None
    ev_arr = player_info.get("hand_evs", [])
    board_cards = _get_board_cards(spot_solution["game"]["board"])

    results = []
    for idx, (c1, c2) in enumerate(_COMBO_INDEX):
        if c1 in board_cards or c2 in board_cards:
            continue
        if _combo_to_hand_name(c1, c2) != hand_name:
            continue
        rng = range_arr[idx]
        if rng < 0.005:
            continue
        actions = {}
        for asol in action_solutions:
            freq = asol["strategy"][idx]
            if freq > 0.005:
                actions[asol["action"]["code"]] = freq
        results.append({
            "combo": c1 + c2,
            "range": rng,
            "ev": ev_arr[idx] if idx < len(ev_arr) else 0,
            "actions": actions,
        })
    return results or None


def _has_significant_suit_diff(combo_strats: list[dict]) -> bool:
    """Check if combo strategies differ enough to be worth reporting.

    Returns True when BOTH conditions are met:
    1. The dominant action differs between at least two combos
    2. Some action's frequency varies by more than 35pp across combos
    """
    if len(combo_strats) < 2:
        return False

    # Check if dominant action differs
    dominants = set()
    for cs in combo_strats:
        if cs["actions"]:
            dominants.add(max(cs["actions"], key=cs["actions"].get))
    if len(dominants) <= 1:
        return False

    # Check max spread per action
    all_codes = set()
    for cs in combo_strats:
        all_codes.update(cs["actions"].keys())

    for code in all_codes:
        freqs = [cs["actions"].get(code, 0) for cs in combo_strats]
        if max(freqs) - min(freqs) > 0.35:
            return True

    return False


def _format_combo_breakdown(combo_strats: list[dict], spot_solution: dict) -> list[str]:
    """Format combo-level strategy lines, sorted by most aggressive first."""
    # Determine action ordering from action_solutions (skip check/fold)
    action_order = [asol["action"]["code"] for asol in spot_solution["action_solutions"]]

    def aggression_score(cs):
        """Higher score = more aggressive (weighted toward later actions)."""
        score = 0
        for i, code in enumerate(action_order):
            score += cs["actions"].get(code, 0) * i
        return score

    combo_strats.sort(key=lambda cs: -aggression_score(cs))

    lines = []
    for cs in combo_strats:
        parts = []
        for code in action_order:
            freq = cs["actions"].get(code, 0)
            if freq < 0.005:
                continue
            label = _action_label(code, spot_solution)
            parts.append(f"{label} {freq*100:.0f}%")
        actions_str = ", ".join(parts)
        lines.append(f"    {cs['combo']}: {actions_str}（EV {cs['ev']:.1f}bb）")
    return lines


def normalize_hand_name(hand: str) -> str:
    """Convert specific combo (Qs6d) to simplified name (Q6o).

    GTO Wizard uses simplified names: 66, AKs, Q6o.
    LLM may output specific combos: 6h6s, AhKh, Qs6d.
    """
    if not hand:
        return ""
    if len(hand) <= 3:
        # Fix case: "Kk" → "KK", "Aks" → "AKs"
        if len(hand) == 2:
            return hand[0].upper() + hand[1].upper()
        if len(hand) == 3:
            return hand[0].upper() + hand[1].upper() + hand[2].lower()
        return hand
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

    # Combo-level breakdown when suits matter
    combo_strats = _get_combo_strategies(spot_solution, hand_name, position)
    if combo_strats and _has_significant_suit_diff(combo_strats):
        lines.append("  花色差異:")
        lines.extend(_format_combo_breakdown(combo_strats, spot_solution))

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
