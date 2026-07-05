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


def combo_index_for_hand(hero_hand_raw: str) -> int | None:
    """Find the 1326-combo index for a specific hero hand like 'Ah6h'.

    Returns index into _COMBO_INDEX, or None if hand is not a 4-char specific combo.
    """
    if not hero_hand_raw or len(hero_hand_raw) != 4:
        return None

    card1 = hero_hand_raw[:2]  # e.g. "Ah"
    card2 = hero_hand_raw[2:]  # e.g. "6h"

    try:
        idx1 = _COMBO_RANKS.index(card1[0]) * 4 + _COMBO_SUITS.index(card1[1])
        idx2 = _COMBO_RANKS.index(card2[0]) * 4 + _COMBO_SUITS.index(card2[1])
    except (ValueError, IndexError):
        return None

    j = max(idx1, idx2)
    i = min(idx1, idx2)
    if j == i:
        return None
    return j * (j - 1) // 2 + i


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
    """Convert specific combo/class input to a canonical 169 hand name.

    GTO Wizard keys preflop strategy by the standard high-rank-first 169
    classes: ``66``, ``AKs``, ``Q6o``.  Users and parsers sometimes provide
    low-rank-first classes (``45o``) or specific combos (``Qs6d``); both must
    resolve to the same canonical key before looking up solver rows.
    """
    if not hand:
        return ""

    ranks = set(_RANK_ORDER)

    if len(hand) <= 3:
        # Fix case: "Kk" → "KK", "Aks" → "AKs", and canonicalize
        # low-rank-first non-pairs such as "45o" → "54o".
        if len(hand) == 2:
            r1, r2 = hand[0].upper(), hand[1].upper()
            if r1 in ranks and r2 in ranks:
                if r1 == r2:
                    return r1 + r2
                if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
                    r1, r2 = r2, r1
                return r1 + r2
            return r1 + r2
        if len(hand) == 3:
            r1, r2, suffix = hand[0].upper(), hand[1].upper(), hand[2].lower()
            if r1 in ranks and r2 in ranks and suffix in ("s", "o"):
                if r1 == r2:
                    return r1 + r2
                if _RANK_ORDER.index(r1) > _RANK_ORDER.index(r2):
                    r1, r2 = r2, r1
                return r1 + r2 + suffix
            return r1 + r2 + suffix
        return hand

    if len(hand) == 4:
        r1, s1, r2, s2 = hand[0].upper(), hand[1].lower(), hand[2].upper(), hand[3].lower()
        if r1 not in ranks or r2 not in ranks:
            return hand
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
            label = _action_label(code, spot_solution)
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
    original_hand = hand_name
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

    # Check if user queried a specific combo (e.g. "Ah8h" not "A8s")
    is_specific_combo = len(original_hand) == 4 and original_hand[1] in "cdhs" and original_hand[3] in "cdhs"

    combo_strats = _get_combo_strategies(spot_solution, hand_name, position)

    # If specific combo requested, show that combo's strategy first
    if is_specific_combo and combo_strats:
        # Normalize combo string to match format in combo_strats (e.g. "Ah8h")
        r1, s1, r2, s2 = original_hand[0].upper(), original_hand[1].lower(), original_hand[2].upper(), original_hand[3].lower()
        target_combos = [f"{r1}{s1}{r2}{s2}", f"{r2}{s2}{r1}{s1}"]
        target_cs = None
        for cs in combo_strats:
            if cs["combo"] in target_combos:
                target_cs = cs
                break

        if target_cs:
            action_order = [asol["action"]["code"] for asol in spot_solution["action_solutions"]]
            lines = [f"【{position} {target_cs['combo']}（{hand_name}）】"]
            parts = []
            for code in action_order:
                freq = target_cs["actions"].get(code, 0)
                if freq < 0.005:
                    continue
                label = _action_label(code, spot_solution)
                parts.append(f"{label} {freq*100:.0f}%")
            lines.append(f"  策略: {', '.join(parts)}（EV {target_cs['ev']:.2f}bb）")
            # Show all other combos for comparison
            lines.append(f"  {hand_name} 其他花色:")
            lines.extend(_format_combo_breakdown(
                [cs for cs in combo_strats if cs["combo"] not in target_combos],
                spot_solution,
            ))
            return "\n".join(lines)

    # Standard aggregated output
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

    # Per-action breakdown (with per-action EV if available)
    actions_freq = hand_data.get("actions_total_frequencies", {})
    actions_combos = hand_data.get("actions_total_combos", {})
    action_evs = _get_per_action_evs(spot_solution, hand_name, position)
    if actions_freq:
        lines.append("  策略:")
        for action_code, freq in sorted(actions_freq.items(), key=lambda x: -x[1]):
            if freq < 0.001:
                continue
            combos = actions_combos.get(action_code, 0)
            action_label = _action_label(action_code, spot_solution)
            ev_str = ""
            if action_evs and action_code in action_evs:
                ev_str = f" EV {action_evs[action_code]:.2f}bb"
            lines.append(f"    {action_label}: {freq*100:.1f}%（{combos:.1f} combos）{ev_str}")

    # Combo-level breakdown: always show when specific combo queried or when suits differ significantly
    if combo_strats and (is_specific_combo or _has_significant_suit_diff(combo_strats)):
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


# Range-compression frequency thresholds.
# Hands at or above _MERGE_FREQ are folded into the compact range notation
# (22+, AXs, K3o+, ...) instead of being broken out with an inline "(NN%)".
# Hands that merge but sit below _PURE_FREQ (i.e. high but not ~100%) get a
# trailing "~" approximate marker on their token so users know the simplified
# group still contains some high-frequency mixed hands.
_PURE_FREQ = 0.995
_MERGE_FREQ = 0.90


def _compress_range(
    hands: list[tuple[str, float, float]],
    merge_threshold: float = _MERGE_FREQ,
) -> str:
    """Compress hand list into standard poker notation.

    Input: [(hand_name, freq, combos), ...]
    Output: "22+, AXo, AXs, K3o+, K2s+, Q8o+, Q5s+, J9o+, J7s+, T9o, T8s+"

    Hands with freq >= ``merge_threshold`` are merged into the compact notation.
    Those that merge while still below ``_PURE_FREQ`` carry a trailing "~"
    marker, e.g. "22+~" (the run includes a high-but-not-pure pair like JJ@99%).
    """
    # Separate by type: pairs, suited, offsuit
    pairs = {}  # rank_idx -> (freq, combos)
    suited = {}  # (high_idx, low_idx) -> (freq, combos)
    offsuit = {}  # (high_idx, low_idx) -> (freq, combos)

    for hand_name, freq, combos in hands:
        if len(hand_name) == 2:
            # Pair: AA, KK, etc.
            idx = _RANK_ORDER.index(hand_name[0])
            pairs[idx] = (freq, combos)
        elif hand_name.endswith("s"):
            h, l = _RANK_ORDER.index(hand_name[0]), _RANK_ORDER.index(hand_name[1])
            suited[(h, l)] = (freq, combos)
        elif hand_name.endswith("o"):
            h, l = _RANK_ORDER.index(hand_name[0]), _RANK_ORDER.index(hand_name[1])
            offsuit[(h, l)] = (freq, combos)

    parts = []

    # Pairs: find consecutive runs (22+ means 22 through AA)
    if pairs:
        pair_ranks = sorted(pairs.keys())  # ascending by rank index (A=0, K=1, ..., 2=12)
        pure_pairs = [r for r in pair_ranks if pairs[r][0] >= merge_threshold]
        mixed_pairs = [(r, pairs[r][0]) for r in pair_ranks if pairs[r][0] < merge_threshold]
        # Ranks that merged but are below _PURE_FREQ → token needs "~" marker.
        soft_pairs = {r for r in pure_pairs if pairs[r][0] < _PURE_FREQ}

        if pure_pairs:
            # Check if consecutive from some rank down to 22 (index 12)
            # In notation: "55+" means 55, 66, 77, ..., AA (from 55 upward)
            lowest_pure = max(pure_pairs)  # highest index = lowest rank (e.g., 12=22)
            highest_pure = min(pure_pairs)  # lowest index = highest rank (e.g., 0=AA)
            expected = set(range(highest_pure, lowest_pure + 1))
            if set(pure_pairs) == expected and len(pure_pairs) > 1:
                mark = "~" if soft_pairs else ""
                parts.append(f"{_RANK_ORDER[lowest_pure]}{_RANK_ORDER[lowest_pure]}+{mark}")
            elif len(pure_pairs) == 1:
                r = pure_pairs[0]
                mark = "~" if r in soft_pairs else ""
                parts.append(f"{_RANK_ORDER[r]}{_RANK_ORDER[r]}{mark}")
            else:
                for r in pure_pairs:
                    mark = "~" if r in soft_pairs else ""
                    parts.append(f"{_RANK_ORDER[r]}{_RANK_ORDER[r]}{mark}")

        for r, freq in mixed_pairs:
            parts.append(f"{_RANK_ORDER[r]}{_RANK_ORDER[r]}({freq*100:.0f}%)")

    # Suited and offsuit: group by high card, find kicker ranges
    for category, label in [(suited, "s"), (offsuit, "o")]:
        if not category:
            continue
        by_high: dict[int, list] = {}
        for (h, l), (freq, combos) in category.items():
            if h not in by_high:
                by_high[h] = []
            by_high[h].append((l, freq, combos))

        for h in sorted(by_high.keys()):
            kickers = by_high[h]
            high_rank = _RANK_ORDER[h]

            pure = [(l, f, c) for l, f, c in kickers if f >= merge_threshold]
            mixed = [(l, f, c) for l, f, c in kickers if f < merge_threshold]
            # Kickers that merged but are below _PURE_FREQ → token needs "~".
            soft_lows = {l for l, f, _ in pure if f < _PURE_FREQ}

            if pure:
                pure_lows = sorted([l for l, _, _ in pure])
                # Check if "all kickers" (h+1 to 12)
                all_kickers = list(range(h + 1, 13))
                if set(pure_lows) == set(all_kickers):
                    mark = "~" if soft_lows else ""
                    parts.append(f"{high_rank}X{label}{mark}")
                elif len(pure) == 1:
                    l = pure[0][0]
                    mark = "~" if l in soft_lows else ""
                    parts.append(f"{high_rank}{_RANK_ORDER[l]}{label}{mark}")
                else:
                    lowest = max(pure_lows)  # highest index = lowest rank
                    highest = min(pure_lows)  # lowest index = highest rank
                    expected = set(range(highest, lowest + 1))
                    if set(pure_lows) == expected:
                        mark = "~" if soft_lows else ""
                        # "+" notation only valid if range reaches the top kicker
                        if highest == h + 1:
                            # Range goes all the way up: "K3o+" = K3o through KQo
                            parts.append(f"{high_rank}{_RANK_ORDER[lowest]}{label}+{mark}")
                        else:
                            # Range doesn't reach top: "Q2s-Q4s" not "Q2s+"
                            parts.append(
                                f"{high_rank}{_RANK_ORDER[lowest]}{label}"
                                f"-{high_rank}{_RANK_ORDER[highest]}{label}{mark}"
                            )
                    else:
                        for l in sorted(pure_lows):
                            mark = "~" if l in soft_lows else ""
                            parts.append(f"{high_rank}{_RANK_ORDER[l]}{label}{mark}")

            for l, f, c in sorted(mixed, key=lambda x: x[0]):
                parts.append(f"{high_rank}{_RANK_ORDER[l]}{label}({f*100:.0f}%)")

    return ", ".join(parts)


_HAND_CATEGORY_LABELS = {
    "straight_flush": "同花順", "quads": "四條", "fullhouse": "葫蘆",
    "flush": "同花", "straight": "順子", "set": "暗三條", "trips": "三條",
    "two_pair": "兩對", "overpair": "超對", "top_pair": "頂對",
    "second_pair": "中對", "underpair": "口袋對 < 牌面", "third_pair": "第三對",
    "low_pair": "底對", "ace_high": "Ace high", "king_high": "King high",
    "no_made_hand": "無成手牌",
}
_DRAW_CATEGORY_LABELS = {
    "combo_draw": "組合聽牌", "nut_flush_draw": "堅果花聽牌",
    "flush_draw": "花聽牌", "oesd": "兩頭順聽牌", "gutshot": "卡順聽牌",
}
# Display order: strongest made hand first
_HAND_CATEGORY_ORDER = [
    "straight_flush", "quads", "fullhouse", "flush", "straight",
    "set", "trips", "two_pair", "overpair", "top_pair", "second_pair",
    "underpair", "third_pair", "low_pair", "ace_high", "king_high",
    "no_made_hand",
]
_DRAW_CATEGORY_ORDER = [
    "combo_draw", "nut_flush_draw", "flush_draw", "oesd", "gutshot",
]


def _categorize_action_range(
    spot_solution: dict, position: str, action_code: str,
) -> tuple[dict[str, list], dict[str, float]]:
    """Group hands by category for a specific action.

    Returns (category_hands, draw_summary):
      category_hands: {category_name: [(hand_name, freq, combos), ...]}
      draw_summary: {draw_name: total_combos}
    """
    from collections import defaultdict

    hcr = spot_solution.get("hand_categories_range")
    dcr = spot_solution.get("draw_categories_range")
    if not hcr or len(hcr) != 1326:
        return {}, {}

    # Find player range
    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return {}, {}

    range_arr = player_info["range"]
    board_cards = _get_board_cards(spot_solution["game"]["board"])

    # Find the action's strategy array
    asol = None
    for a in spot_solution["action_solutions"]:
        if a["action"]["code"] == action_code:
            asol = a
            break
    if not asol or "strategy" not in asol:
        return {}, {}

    # Build category name mappings (use cat["index"] as key, not list position)
    hc_names = {cat["index"]: cat["name"] for cat in asol.get("hand_categories", [])}
    dc_names = {cat["index"]: cat["name"] for cat in asol.get("draw_categories", [])} if dcr else {}

    # Group combos by hand category
    cat_hands_raw = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # cat -> hand -> [combos, freq_sum]
    draw_combos = defaultdict(float)

    for idx, (c1, c2) in enumerate(_COMBO_INDEX):
        if c1 in board_cards or c2 in board_cards:
            continue
        rng = range_arr[idx]
        if rng < 0.005:
            continue
        freq = asol["strategy"][idx]
        if freq < 0.005:
            continue

        hand_name = _combo_to_hand_name(c1, c2)
        weighted = rng * freq
        cat_name = hc_names.get(hcr[idx], "no_made_hand")
        cat_hands_raw[cat_name][hand_name][0] += weighted
        cat_hands_raw[cat_name][hand_name][1] += freq  # for avg freq

        if dcr and idx < len(dcr):
            draw_name = dc_names.get(dcr[idx])
            if draw_name and draw_name not in ("no_draw", "onecard_bdfd", "twocards_bdfd"):
                draw_combos[draw_name] += weighted

    # Look up actual hand-level action frequencies from simple_hand_counters
    shc = player_info.get("simple_hand_counters", {})

    # Convert to list format compatible with _compress_range
    category_hands = {}
    for cat_name, hands in cat_hands_raw.items():
        hand_list = []
        for hand_name, (combos, freq_sum) in hands.items():
            # Use real freq so _compress_range shows mixed hands correctly
            # e.g. AA all-in at 5% → "AA(5%)" instead of being grouped into "TT+"
            freq = shc.get(hand_name, {}).get("actions_total_frequencies", {}).get(action_code, 1.0)
            hand_list.append((hand_name, freq, combos))
        hand_list.sort(key=lambda x: -x[2])
        category_hands[cat_name] = hand_list

    return category_hands, dict(draw_combos)


def format_range_by_action(spot_solution: dict, position: str) -> str:
    """Format range grouped by action for a position.

    Shows which hands take each action, categorized by hand type (top pair,
    draws, etc.) using the solver's pre-computed hand classifications.
    Used for questions like "BB raise range 有哪些？"
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

    # Group hands by action (for total counts and fallback)
    action_groups: dict[str, list] = {}
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

    for code in action_groups:
        action_groups[code].sort(key=lambda x: -x[2])

    def sort_key(code):
        if code == "F":
            return (1, 0)
        return (0, -sum(x[2] for x in action_groups[code]))

    lines = [f"【{position} 在 {street} {board} 的策略分佈】"]

    has_categories = bool(spot_solution.get("hand_categories_range"))

    for code in sorted(action_groups.keys(), key=sort_key):
        group = action_groups[code]
        total = sum(x[2] for x in group)
        label = _action_label(code, spot_solution)

        if has_categories and code != "F":
            # Categorized output
            cat_hands, draw_summary = _categorize_action_range(spot_solution, position, code)
            lines.append(f"\n{label}（{total:.0f} combos）:")

            if cat_hands:
                for cat_name in _HAND_CATEGORY_ORDER:
                    if cat_name not in cat_hands:
                        continue
                    hands = cat_hands[cat_name]
                    cat_total = sum(h[2] for h in hands)
                    if cat_total < 0.5:
                        continue
                    pct = cat_total / total * 100 if total else 0
                    cat_label = _HAND_CATEGORY_LABELS.get(cat_name, cat_name)
                    compressed = _compress_range(hands)
                    lines.append(f"  {cat_label} ({cat_total:.0f} combos, {pct:.0f}%): {compressed}")

                # Draw summary (one line)
                draw_parts = []
                for dn in _DRAW_CATEGORY_ORDER:
                    if dn in draw_summary and draw_summary[dn] > 0.5:
                        dl = _DRAW_CATEGORY_LABELS[dn]
                        draw_parts.append(f"{dl} {draw_summary[dn]:.0f}")
                if draw_parts:
                    lines.append(f"  (聽牌: {', '.join(draw_parts)} combos)")
            else:
                # Fallback to flat compressed range
                compressed = _compress_range(group)
                lines.append(f"  {compressed}")
        else:
            # Fold or no categories: flat compressed range
            compressed = _compress_range(group)
            lines.append(f"\n{label}（{total:.0f} combos）:")
            lines.append(f"  {compressed}")

    if any("~" in line for line in lines):
        lines.append("\n(~ = 該組已併入 >90% 高頻手牌，非 100% 純頻)")

    return "\n".join(lines)


def format_full_spot(spot_solution: dict, hero_hand: str = None, hero_position: str = None) -> str:
    """Format complete spot analysis including action summary and hero hand detail."""
    parts = [format_action_summary(spot_solution)]

    if hero_hand and hero_position:
        parts.append("")
        parts.append(format_hand_detail(spot_solution, hero_hand, hero_position))

    return "\n".join(parts)


def _combo_idx_in_player_range(
    spot_solution: dict,
    hero_position: str,
    combo_idx: int | None,
    min_range: float = 1e-12,
) -> bool:
    """Return True if an exact postflop combo is present at this node.

    Solver strategy/EV arrays often contain arbitrary-looking defaults for
    combos that have already taken a different earlier action and therefore
    have zero range at the current node.  Those off-node combo rows must not
    drive user-facing advice.

    Use a near-zero threshold instead of the display/action cutoff (0.5%):
    rare but non-zero combos still have meaningful exact-combo strategy rows
    and should be shown for later decision points.
    """
    if combo_idx is None:
        return False
    for pi in spot_solution.get("players_info", []):
        if pi.get("player", {}).get("position") != hero_position:
            continue
        range_arr = pi.get("range", [])
        return (
            len(range_arr) == 1326
            and combo_idx < len(range_arr)
            and range_arr[combo_idx] >= min_range
        )
    return False


def format_spot_compact(spot_solution: dict, hero_hand: str, hero_position: str,
                        min_freq: float = 0.05, combo_idx: int | None = None) -> str:
    """Format compact GTO line: one line with actions ≥ min_freq.

    For postflop, pass combo_idx to get combo-specific frequencies instead of
    aggregate hand frequencies (e.g., Ks9s vs K9s average which includes Kc9c).

    Output example:
        GTO: Raise 68% / Call 32%
        GTO: Bet 20% pot 97%
    """
    action_solutions = spot_solution.get("action_solutions", [])
    actions_freq = None

    # Postflop: use combo-specific frequencies from 1326-strategy arrays.
    # If the exact combo has zero range at this node, do not fall back to
    # same-hand aggregate counters. That node is unreachable for hero's
    # actual line (often because an earlier bet size was off-grid/off-mix),
    # so user-facing output should treat it as no solver data rather than
    # borrowing advice from different combos.
    if combo_idx is not None and action_solutions and "strategy" in action_solutions[0]:
        if not _combo_idx_in_player_range(spot_solution, hero_position, combo_idx):
            return ""
        combo_freq = {}
        for asol in action_solutions:
            freq = asol["strategy"][combo_idx]
            if freq > 0.005:
                combo_freq[asol["action"]["code"]] = freq
        if combo_freq:
            actions_freq = combo_freq

    # Fallback: aggregate hand-level frequencies
    if actions_freq is None:
        hand_name = normalize_hand_name(hero_hand)
        player_info = None
        for pi in spot_solution["players_info"]:
            if pi["player"]["position"] == hero_position:
                player_info = pi
                break
        if not player_info:
            return ""
        shc = player_info.get("simple_hand_counters", {})
        hand_data = shc.get(hand_name)
        if hand_data:
            actions_freq = hand_data.get("actions_total_frequencies", {})

    parts = []
    if actions_freq:
        sorted_actions = sorted(actions_freq.items(), key=lambda x: -x[1])
        for code, freq in sorted_actions:
            if freq < min_freq:
                continue
            label = _action_label(code, spot_solution)
            parts.append(f"{label} {freq*100:.0f}%")
    else:
        for sol in sorted(action_solutions, key=lambda s: -s["total_frequency"]):
            freq = sol["total_frequency"]
            if freq < min_freq:
                continue
            code = sol["action"]["code"]
            label = _action_label(code, spot_solution)
            parts.append(f"{label} {freq*100:.0f}%")

    if not parts:
        return ""
    return "GTO: " + " / ".join(parts)


def _get_per_action_evs(spot_solution: dict, hand_name: str, position: str) -> dict[str, float] | None:
    """Extract per-action EVs for a hand from action_solutions[i].evs arrays.

    Works for both preflop (169-element) and postflop (1326-element, averaged across combos).
    Returns {action_code: ev} or None if EV data unavailable.
    """
    if "action_solutions" not in spot_solution:
        return None

    player_info = None
    for pi in spot_solution["players_info"]:
        if pi["player"]["position"] == position:
            player_info = pi
            break
    if not player_info or "range" not in player_info:
        return None

    range_arr = player_info["range"]
    action_solutions = spot_solution["action_solutions"]

    # Preflop: 169-element arrays
    if len(range_arr) == 169:
        from hh_deviation_check import HAND_TO_169
        idx = HAND_TO_169.get(hand_name)
        if idx is None or range_arr[idx] < 0.005:
            return None
        evs = {}
        for asol in action_solutions:
            ev_arr = asol.get("evs")
            if not ev_arr or len(ev_arr) != 169:
                return None
            evs[asol["action"]["code"]] = ev_arr[idx]
        return evs if evs else None

    # Postflop: 1326-element arrays, average across combos
    if len(range_arr) == 1326:
        board_cards = _get_board_cards(spot_solution["game"]["board"])
        total_weight = 0.0
        action_evs: dict[str, float] = {}

        for idx, (c1, c2) in enumerate(_COMBO_INDEX):
            if c1 in board_cards or c2 in board_cards:
                continue
            if _combo_to_hand_name(c1, c2) != hand_name:
                continue
            rng = range_arr[idx]
            if rng < 0.005:
                continue
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

    return None


def _get_action_strategy_frequencies(
    spot_solution: dict,
    hero_hand: str,
    hero_pos: str,
    is_preflop: bool,
    combo_idx: int | None = None,
) -> dict[str, float] | None:
    """Return solver strategy frequencies for hero's hand/combo at a node.

    Per-action EV arrays can be noisy or use terminal-action accounting that
    is not directly comparable for every action.  Strategy frequencies are the
    guardrail for whether a taken action is actually solver-approved.
    """
    if not spot_solution or "action_solutions" not in spot_solution:
        return None

    player_info = None
    for pi in spot_solution.get("players_info", []):
        if pi.get("player", {}).get("position") == hero_pos:
            player_info = pi
            break
    if not player_info:
        return None

    if is_preflop:
        range_arr = player_info.get("range", [])
        if len(range_arr) == 169:
            from hh_deviation_check import HAND_TO_169
            idx = HAND_TO_169.get(normalize_hand_name(hero_hand))
            if idx is not None and idx < len(range_arr) and range_arr[idx] > 0:
                freqs: dict[str, float] = {}
                for asol in spot_solution["action_solutions"]:
                    strat = asol.get("strategy")
                    if strat and idx < len(strat):
                        freqs[asol.get("action", {}).get("code")] = float(strat[idx])
                return freqs or None

    elif combo_idx is not None and _combo_idx_in_player_range(
        spot_solution, hero_pos, combo_idx
    ):
        freqs = {}
        for asol in spot_solution["action_solutions"]:
            strat = asol.get("strategy")
            if strat and combo_idx < len(strat):
                freqs[asol.get("action", {}).get("code")] = float(strat[combo_idx])
        return freqs or None

    # Fallback to aggregate hand counters when exact strategy rows are not
    # available (or when caller intentionally did not pass a combo index).
    shc = player_info.get("simple_hand_counters", {})
    hand_data = shc.get(normalize_hand_name(hero_hand))
    if hand_data:
        actions_freq = hand_data.get("actions_total_frequencies", {})
        if actions_freq:
            return {code: float(freq) for code, freq in actions_freq.items()}

    return None


def format_ev_comparison(spot_solution: dict, taken_code: str, hero_hand: str,
                         hero_pos: str, is_preflop: bool, combo_idx: int | None = None) -> str | None:
    """Format EV comparison between hero's action and the best action.

    Returns e.g.: "⚠ EV 損失 3.71bb（All-in 7.04bb vs Raise 10.75bb）" or None if no loss or data unavailable.
    """
    if not spot_solution or "action_solutions" not in spot_solution:
        return None

    if is_preflop:
        from hh_deviation_check import _get_action_evs_preflop
        action_evs = _get_action_evs_preflop(spot_solution, hero_hand, hero_pos)
    else:
        from hh_deviation_check import _get_action_evs_postflop
        if combo_idx is not None and not _combo_idx_in_player_range(
            spot_solution, hero_pos, combo_idx
        ):
            return None
        action_evs = _get_action_evs_postflop(
            spot_solution, hero_hand, hero_pos, combo_idx=combo_idx)

    if not action_evs:
        return None

    strategy_freqs = _get_action_strategy_frequencies(
        spot_solution, hero_hand, hero_pos, is_preflop, combo_idx
    )
    taken_freq = strategy_freqs.get(taken_code) if strategy_freqs else None
    max_freq = max(strategy_freqs.values()) if strategy_freqs else None
    if (
        taken_code == "F"
        and taken_freq is not None
        and max_freq is not None
        and taken_freq >= 0.05
        and taken_freq >= max_freq - 0.005
    ):
        return None

    hero_ev = action_evs.get(taken_code)
    if hero_ev is None:
        return None

    best_code = max(action_evs, key=action_evs.get)
    best_ev = action_evs[best_code]
    ev_loss = best_ev - hero_ev

    if ev_loss < 0.005:
        return None  # No significant loss

    hero_label = _action_label(taken_code, spot_solution)
    best_label = _action_label(best_code, spot_solution)
    return f"⚠ EV 損失 {ev_loss:.2f}bb（{hero_label} {hero_ev:.2f}bb vs {best_label} {best_ev:.2f}bb）"


# --- EV-impact severity (preflop absolute bb, postflop pot-relative) --------
# Deviating from the GTO top action is only a real "mistake" if it costs
# meaningful EV. How much counts as negligible differs by street: preflop we
# judge in absolute bb (small pots, fixed sizing); postflop we normalise against
# the pot, because a 0.5bb loss is noise in an 80bb pot but huge in a 4bb pot.
# Below the negligible line the action is a frequency/mix preference, not an
# error — the coach must not call it a blunder.
EV_NEGLIGIBLE_PREFLOP_BB = 0.05
EV_NEGLIGIBLE_POSTFLOP_POT_FRAC = 0.005  # 0.5% of the pot


def classify_ev_impact(ev_loss: float, is_preflop: bool,
                       pot_bb: float | None = None) -> dict:
    """Classify how much an EV loss actually matters.

    Returns {"negligible": bool, "pot_frac": float|None}. Preflop uses the
    absolute bb threshold; postflop normalises against the node pot, falling
    back to the bb threshold when the pot is unknown.
    """
    if ev_loss < 0:
        ev_loss = 0.0
    if is_preflop:
        return {"negligible": ev_loss <= EV_NEGLIGIBLE_PREFLOP_BB,
                "pot_frac": None}
    if pot_bb and pot_bb > 0:
        pot_frac = ev_loss / pot_bb
        return {"negligible": pot_frac <= EV_NEGLIGIBLE_POSTFLOP_POT_FRAC,
                "pot_frac": pot_frac}
    return {"negligible": ev_loss <= EV_NEGLIGIBLE_PREFLOP_BB, "pot_frac": None}


def ev_loss_detail(spot_solution: dict, taken_code: str, hero_hand: str,
                   hero_pos: str, is_preflop: bool,
                   combo_idx: int | None = None) -> dict | None:
    """Structured EV-loss of hero's action vs the solver's best action.

    Returns None when there is no meaningful loss or data is unavailable
    (mirrors format_ev_comparison's gating, including the max-frequency-fold
    case). Otherwise returns
    {ev_loss, hero_ev, best_ev, best_code, pot_bb, pot_frac, negligible}.
    """
    if not spot_solution or "action_solutions" not in spot_solution:
        return None

    if is_preflop:
        from hh_deviation_check import _get_action_evs_preflop
        action_evs = _get_action_evs_preflop(spot_solution, hero_hand, hero_pos)
    else:
        from hh_deviation_check import _get_action_evs_postflop
        if combo_idx is not None and not _combo_idx_in_player_range(
            spot_solution, hero_pos, combo_idx
        ):
            return None
        action_evs = _get_action_evs_postflop(
            spot_solution, hero_hand, hero_pos, combo_idx=combo_idx)

    if not action_evs:
        return None

    # A high-frequency fold is not a loss even when its EV looks lower — the
    # solver mixes folds that are ~indifferent. Mirror format_ev_comparison.
    strategy_freqs = _get_action_strategy_frequencies(
        spot_solution, hero_hand, hero_pos, is_preflop, combo_idx
    )
    taken_freq = strategy_freqs.get(taken_code) if strategy_freqs else None
    max_freq = max(strategy_freqs.values()) if strategy_freqs else None
    if (
        taken_code == "F"
        and taken_freq is not None
        and max_freq is not None
        and taken_freq >= 0.05
        and taken_freq >= max_freq - 0.005
    ):
        return None

    hero_ev = action_evs.get(taken_code)
    if hero_ev is None:
        return None

    best_code = max(action_evs, key=action_evs.get)
    best_ev = action_evs[best_code]
    ev_loss = best_ev - hero_ev
    if ev_loss < 0:
        ev_loss = 0.0

    pot_bb = None
    if not is_preflop:
        raw_pot = (spot_solution.get("game") or {}).get("pot")
        try:
            pot_bb = float(raw_pot) if raw_pot is not None else None
        except (TypeError, ValueError):
            pot_bb = None

    impact = classify_ev_impact(ev_loss, is_preflop, pot_bb)
    return {
        "ev_loss": ev_loss,
        "hero_ev": hero_ev,
        "best_ev": best_ev,
        "best_code": best_code,
        "pot_bb": pot_bb,
        "pot_frac": impact["pot_frac"],
        "negligible": impact["negligible"],
    }


def format_ev_magnitude(detail: dict) -> str:
    """EV magnitude for display: '0.02bb' preflop, '0.30bb（0.4% pot）' postflop."""
    s = f"{detail['ev_loss']:.2f}bb"
    if detail.get("pot_frac") is not None:
        s += f"（{detail['pot_frac'] * 100:.1f}% pot）"
    return s


def _action_label(code: str, spot_solution: dict) -> str:
    """Convert action code to readable label."""
    if code == "X":
        return "Check"
    if code == "F":
        return "Fold"
    if code == "C":
        if _is_preflop_limp_call(code, spot_solution):
            return "Limp"
        return "Call"
    if code == "RAI":
        return "All-in"

    # Determine verb: "raise" if facing a bet (fold/call available), else "bet"
    street = spot_solution.get("game", {}).get("current_street", {}).get("type", "")
    is_preflop = street.lower() == "preflop"
    has_fold_or_call = any(
        s["action"]["code"] in ("F", "C")
        for s in spot_solution.get("action_solutions", []))
    is_raise = is_preflop or has_fold_or_call

    # Look up betsize from action_solutions
    for sol in spot_solution["action_solutions"]:
        if sol["action"]["code"] == code:
            act = sol["action"]
            if act.get("allin"):
                return f"All-in {act['betsize']}"
            pct = float(act.get("betsize_by_pot", 0)) * 100
            verb = "RAISE" if is_raise else "Bet"
            return f"{verb} {act['betsize']}（{pct:.0f}% pot）"

    return code


def _action_label_short(code: str, spot_solution: dict, street: str = "") -> str:
    """Short action label for compact hints, e.g. 'bet 20% pot', 'raise 55% pot'."""
    if code == "X":
        return "check"
    if code == "F":
        return "fold"
    if code == "C":
        if _is_preflop_limp_call(code, spot_solution):
            return "limp"
        return "call"
    if code == "RAI":
        return "all-in"
    # "raise" if preflop or facing a bet (fold/call available)
    has_fold_or_call = any(
        s["action"]["code"] in ("F", "C")
        for s in spot_solution.get("action_solutions", []))
    is_raise = street == "preflop" or has_fold_or_call
    for sol in spot_solution.get("action_solutions", []):
        if sol["action"]["code"] == code:
            act = sol["action"]
            if act.get("allin"):
                return "all-in"
            pct = float(act.get("betsize_by_pot", 0)) * 100
            verb = "raise" if is_raise else "bet"
            if pct > 0:
                return f"{verb} {pct:.0f}% pot"
            return verb
    return code


def _is_preflop_limp_call(code: str, spot_solution: dict) -> bool:
    """Return True when solver code C means an unopened preflop limp.

    GTO Wizard represents open-limps/completions with the same ``C`` code used
    for calling a raise.  In unopened preflop spots that "call" amount is the
    blind price (1bb total), so label it as Limp to keep the coaching layer
    from inventing a previous raiser.
    """
    if code != "C":
        return False
    street = spot_solution.get("game", {}).get("current_street", {}).get("type", "")
    if str(street).lower() != "preflop":
        return False
    call_action = next(
        (
            sol.get("action", {})
            for sol in spot_solution.get("action_solutions", [])
            if sol.get("action", {}).get("code") == "C"
        ),
        {},
    )
    try:
        return float(call_action.get("betsize", 0)) <= 1.0001
    except (TypeError, ValueError):
        return False
