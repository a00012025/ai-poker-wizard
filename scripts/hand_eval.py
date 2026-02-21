"""Deterministic hand evaluator — pure Python, no external dependencies.

Given hole cards and a board, returns the exact made hand and draws.
Labels are in 繁體中文 matching gto_formatter category names.

Usage:
    from hand_eval import evaluate
    result = evaluate("T8o", "8hTc2sAc")
    # → {"made_hand": "two_pair", "made_hand_label": "兩對 (T, 8)", ...}
"""

# ── Card parsing ──

_RANK_CHARS = "23456789TJQKA"
_RANK_VALUES = {c: i + 2 for i, c in enumerate(_RANK_CHARS)}  # '2'→2 .. 'A'→14
_SUITS = set("cdhs")


def _parse_cards(s: str) -> list[tuple[int, str]]:
    """Parse card string like '8hTc2s' into [(rank_val, suit), ...]."""
    cards = []
    i = 0
    while i < len(s):
        rank_ch = s[i].upper()
        if rank_ch == '1' and i + 1 < len(s) and s[i + 1] == '0':
            rank_ch = 'T'
            i += 2
        else:
            i += 1
        suit_ch = s[i].lower() if i < len(s) else ''
        if suit_ch in _SUITS:
            cards.append((_RANK_VALUES[rank_ch], suit_ch))
            i += 1
        else:
            # No suit specified — shouldn't happen for board cards
            cards.append((_RANK_VALUES[rank_ch], '?'))
    return cards


def _parse_hole_cards(hand: str) -> list[tuple[int, str]]:
    """Parse hole cards in various formats.

    Formats:
        - Specific: 'AhKs', 'Th8c' → [(14,'h'), (13,'s')]
        - Generic:  'AKo', 'AKs', '66' → [(14,'?'), (13,'?')]
    """
    hand = hand.strip()
    # Try specific format first (4 chars like AhKs or Th8c)
    if len(hand) == 4 and hand[1].lower() in _SUITS and hand[3].lower() in _SUITS:
        r1 = hand[0].upper()
        s1 = hand[1].lower()
        r2 = hand[2].upper()
        s2 = hand[3].lower()
        return [(_RANK_VALUES[r1], s1), (_RANK_VALUES[r2], s2)]
    # Generic format: AKo, AKs, 66
    r1 = hand[0].upper()
    r2 = hand[1].upper()
    return [(_RANK_VALUES[r1], '?'), (_RANK_VALUES[r2], '?')]


def _is_specific(hole_cards: list[tuple[int, str]]) -> bool:
    """Check if hole cards have specific suits (not generic)."""
    return all(s != '?' for _, s in hole_cards)


# ── Made hand evaluation ──

_MADE_HAND_LABELS = {
    "straight_flush": "同花順",
    "quads": "四條",
    "full_house": "葫蘆",
    "flush": "同花",
    "straight": "順子",
    "set": "暗三條",
    "trips": "三條",
    "two_pair": "兩對",
    "overpair": "超對",
    "top_pair": "頂對",
    "second_pair": "中對",
    "third_pair": "第三對",
    "low_pair": "底對",
    "ace_high": "Ace high",
    "king_high": "King high",
    "high_card": "高牌",
}

_DRAW_LABELS = {
    "nut_flush_draw": "堅果花聽牌",
    "flush_draw": "花聽牌",
    "oesd": "兩頭順聽牌",
    "gutshot": "卡順聽牌",
    "backdoor_flush_draw": "後門花聽牌",
}

_RANK_NAMES = {14: "A", 13: "K", 12: "Q", 11: "J", 10: "T",
               9: "9", 8: "8", 7: "7", 6: "6", 5: "5", 4: "4", 3: "3", 2: "2"}


def _find_straight(ranks: list[int]) -> int | None:
    """Find the highest straight top card from a set of ranks.

    Returns the top rank of the straight, or None.
    Handles A-2-3-4-5 (wheel) where A plays low.
    """
    unique = sorted(set(ranks), reverse=True)
    # Add ace-low (1) if ace present
    if 14 in unique:
        unique.append(1)

    for i in range(len(unique) - 4):
        if unique[i] - unique[i + 4] == 4:
            # Check all 5 are consecutive
            window = unique[i:i + 5]
            if all(window[j] - window[j + 1] == 1 for j in range(4)):
                return unique[i]
    return None


def _straight_name(top: int) -> str:
    """Format straight name like 'A-K-Q-J-T' or '5-4-3-2-A'."""
    if top == 5:
        # Wheel: 5-4-3-2-A
        return "-".join(_RANK_NAMES[r] for r in [5, 4, 3, 2, 14])
    return "-".join(_RANK_NAMES[r] for r in range(top, top - 5, -1))


def _evaluate_made_hand(hole: list[tuple[int, str]], board: list[tuple[int, str]]) -> tuple[str, str]:
    """Evaluate the best 5-card made hand.

    Returns (hand_type, label) where hand_type is the category key.
    """
    all_cards = hole + board
    all_ranks = [r for r, _ in all_cards]
    hole_ranks = [r for r, _ in hole]
    board_ranks = [r for r, _ in board]
    specific = _is_specific(hole)

    # --- Check flush & straight flush (only if specific suits) ---
    if specific:
        suit_cards: dict[str, list[int]] = {}
        for r, s in all_cards:
            suit_cards.setdefault(s, []).append(r)

        for suit, ranks_in_suit in suit_cards.items():
            if len(ranks_in_suit) >= 5:
                # Flush exists — check if straight flush
                sf_top = _find_straight(ranks_in_suit)
                if sf_top is not None:
                    return "straight_flush", f"同花順 ({_straight_name(sf_top)})"
                # Regular flush
                sorted_flush = sorted(ranks_in_suit, reverse=True)[:5]
                flush_label = "-".join(_RANK_NAMES[r] for r in sorted_flush)
                return "flush", f"同花 ({flush_label})"

    # --- Rank frequency analysis ---
    from collections import Counter
    rank_counts = Counter(all_ranks)

    # Quads
    for rank, cnt in rank_counts.items():
        if cnt == 4:
            kicker = max(r for r in all_ranks if r != rank)
            return "quads", f"四條 ({_RANK_NAMES[rank]})"

    # Full house
    trips_ranks = [r for r, c in rank_counts.items() if c >= 3]
    pairs_ranks = [r for r, c in rank_counts.items() if c >= 2]
    if trips_ranks:
        best_trip = max(trips_ranks)
        # Pair can come from another trips or a pair
        pair_options = [r for r in pairs_ranks if r != best_trip]
        if pair_options:
            best_pair = max(pair_options)
            return "full_house", f"葫蘆 ({_RANK_NAMES[best_trip]}滿{_RANK_NAMES[best_pair]})"

    # Straight (already checked straight flush above)
    straight_top = _find_straight(all_ranks)
    if straight_top is not None:
        return "straight", f"順子 ({_straight_name(straight_top)})"

    # Three of a kind (set vs trips)
    if trips_ranks:
        trip_rank = max(trips_ranks)
        hole_cnt = sum(1 for r in hole_ranks if r == trip_rank)
        if hole_cnt == 2:
            return "set", f"暗三條 ({_RANK_NAMES[trip_rank]})"
        else:
            return "trips", f"三條 ({_RANK_NAMES[trip_rank]})"

    # Two pair / one pair
    pair_list = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)
    board_sorted = sorted(set(board_ranks), reverse=True)

    if len(pair_list) >= 2:
        # Only count pairs hero contributes to (board-only pairs are shared by all)
        hero_pairs = [pr for pr in pair_list if any(r == pr for r in hole_ranks)]
        if len(hero_pairs) >= 2:
            label = f"兩對 ({_RANK_NAMES[hero_pairs[0]]}, {_RANK_NAMES[hero_pairs[1]]})"
            return "two_pair", label
        # Board pair present — classify by the pair(s) hero contributes to
        pair_list = hero_pairs

    if len(pair_list) == 1:
        pair_rank = pair_list[0]
        hole_cnt = sum(1 for r in hole_ranks if r == pair_rank)
        board_cnt = sum(1 for r in board_ranks if r == pair_rank)

        if hole_cnt == 2 and board_cnt == 0:
            # Pocket pair — classify relative to board
            if board_sorted and pair_rank > board_sorted[0]:
                return "overpair", f"超對 ({_RANK_NAMES[pair_rank]}{_RANK_NAMES[pair_rank]})"
            elif board_sorted and pair_rank < board_sorted[-1]:
                return "low_pair", f"口袋對 ({_RANK_NAMES[pair_rank]}{_RANK_NAMES[pair_rank]}) < 牌面"
            else:
                # Between board cards
                return "low_pair", f"口袋對 ({_RANK_NAMES[pair_rank]}{_RANK_NAMES[pair_rank]})"

        # One card from hole pairs with board
        if board_sorted:
            if pair_rank == board_sorted[0]:
                return "top_pair", f"頂對 ({_RANK_NAMES[pair_rank]})"
            elif len(board_sorted) >= 2 and pair_rank == board_sorted[1]:
                return "second_pair", f"中對 ({_RANK_NAMES[pair_rank]})"
            elif len(board_sorted) >= 3 and pair_rank == board_sorted[2]:
                return "third_pair", f"第三對 ({_RANK_NAMES[pair_rank]})"
            else:
                return "low_pair", f"底對 ({_RANK_NAMES[pair_rank]})"
        return "low_pair", f"一對 ({_RANK_NAMES[pair_rank]})"

    # High card
    if 14 in hole_ranks:
        return "ace_high", "Ace high"
    if 13 in hole_ranks:
        return "king_high", "King high"
    best = max(hole_ranks)
    return "high_card", f"高牌 ({_RANK_NAMES[best]})"


# ── Draw evaluation ──

def _count_straight_outs(all_ranks: list[int], hole_ranks: list[int]) -> tuple[str | None, int]:
    """Determine straight draw type by counting outs.

    Returns (draw_type, num_outs) or (None, 0).
    Only considers straights that use at least one hole card.
    """
    unique = set(all_ranks)
    # Also consider ace-low
    if 14 in unique:
        unique.add(1)
    hole_set = set(hole_ranks)
    if 14 in hole_set:
        hole_set.add(1)

    # Current count of cards (before drawing)
    current_count = len(set(all_ranks))

    # Try each possible missing rank (2-14 = ranks not yet on board/hand)
    all_possible = set(range(1, 15))
    missing = all_possible - unique
    outs = set()

    for missing_rank in missing:
        test_ranks = list(unique | {missing_rank})
        # Check if a straight exists using this new rank + at least one hole card
        test_sorted = sorted(test_ranks, reverse=True)
        if 14 in test_ranks:
            test_sorted.append(1)

        for i in range(len(test_sorted) - 4):
            window = test_sorted[i:i + 5]
            if all(window[j] - window[j + 1] == 1 for j in range(4)):
                # Found a straight — does it use a hole card?
                straight_ranks = set(window)
                if straight_ranks & hole_set:
                    outs.add(missing_rank if missing_rank != 1 else 14)
                    break

    num_rank_outs = len(outs)
    if num_rank_outs == 0:
        return None, 0

    # Already have a straight? Not a draw.
    if _find_straight(list(unique)) is not None:
        return None, 0

    # Each distinct rank out = 4 actual card outs
    # OESD: 2+ ranks (8+ outs), gutshot: 1 rank (4 outs)
    if num_rank_outs >= 2:
        return "oesd", num_rank_outs * 4
    else:
        return "gutshot", num_rank_outs * 4


def _evaluate_draws(hole: list[tuple[int, str]], board: list[tuple[int, str]]) -> list[str]:
    """Evaluate draws (flush draws and straight draws).

    Only applicable for flop (3 board cards) and turn (4 board cards).
    River (5 board cards) = no draws.
    Requires specific suits for flush draw detection.
    """
    if len(board) >= 5:
        return []

    draws = []
    specific = _is_specific(hole)

    # --- Flush draws (need specific suits) ---
    if specific:
        from collections import Counter
        all_cards = hole + board
        suit_counts = Counter(s for _, s in all_cards)
        hole_suits = [s for _, s in hole]

        for suit, count in suit_counts.items():
            if count == 4 and suit in hole_suits:
                # Flush draw — check if nut
                hole_rank_in_suit = max(
                    (r for r, s in hole if s == suit), default=0
                )
                # Nut flush draw = have the ace of that suit
                if hole_rank_in_suit == 14:
                    draws.append("nut_flush_draw")
                else:
                    draws.append("flush_draw")
            elif count == 3 and suit in hole_suits and len(board) == 3:
                # Backdoor flush draw (flop only, 3 suited with hole card)
                draws.append("backdoor_flush_draw")

    # --- Straight draws ---
    all_ranks = [r for r, _ in hole + board]
    hole_ranks = [r for r, _ in hole]
    draw_type, _ = _count_straight_outs(all_ranks, hole_ranks)
    if draw_type:
        draws.append(draw_type)

    return draws


# ── Main API ──

def evaluate(hole_cards: str, board: str = "") -> dict:
    """Evaluate hole cards on a board.

    Args:
        hole_cards: "T8o", "AhKh", "66", etc.
        board: "8hTc2s" (flop), "8hTc2sAc" (turn), "8hTc2sAcJd" (river)
               Empty string = preflop (no evaluation).

    Returns:
        {
            "made_hand": "two_pair",
            "made_hand_label": "兩對 (T, 8)",
            "draws": ["gutshot"],
            "draw_labels": ["卡順聽牌"],
            "full_label": "兩對 (T, 8) + 卡順聽牌"
        }
    """
    if not board:
        return {
            "made_hand": "",
            "made_hand_label": "",
            "draws": [],
            "draw_labels": [],
            "full_label": "",
        }

    hole = _parse_hole_cards(hole_cards)
    board_cards = _parse_cards(board)

    # Made hand
    made_hand, made_label = _evaluate_made_hand(hole, board_cards)

    # Draws
    draws = _evaluate_draws(hole, board_cards)
    draw_labels = [_DRAW_LABELS.get(d, d) for d in draws]

    # Full label
    parts = [made_label] + draw_labels
    full_label = " + ".join(parts)

    return {
        "made_hand": made_hand,
        "made_hand_label": made_label,
        "draws": draws,
        "draw_labels": draw_labels,
        "full_label": full_label,
    }
