#!/usr/bin/env python3
"""Spot categorizer — classify each hero decision point into ~15 spot buckets.

Pure function: no API calls, just parsing the action sequence and board.

Spot categories:
  Preflop:  open_raise, facing_open, facing_3bet, squeeze, facing_4bet, limp_pot
  Postflop: cbet_ip, cbet_oop, facing_cbet_ip, facing_cbet_oop,
            probe, facing_probe, donk, check_raise
"""

from __future__ import annotations

# Position orders by table size (same as analyze_hand.py)
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


# ── Board Texture Classification ──

_RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def classify_board_texture(board: str | None) -> str | None:
    """Classify board texture into: paired, monotone, wet, dry.

    Priority: paired > monotone > wet > dry.
    Returns None for empty/None boards (preflop).

    Board format: "Js6h5s" (rank+suit pairs).
    """
    if not board:
        return None

    cards = []
    for i in range(0, len(board) - 1, 2):
        rank = board[i]
        suit = board[i + 1] if i + 1 < len(board) else "?"
        if rank in _RANK_VALUES:
            cards.append((rank, suit))

    if len(cards) < 3:
        return None

    ranks = [_RANK_VALUES[c[0]] for c in cards]
    suits = [c[1] for c in cards]

    # Check paired: any two cards share the same rank
    if len(set(ranks)) < len(ranks):
        return "paired"

    # Check monotone: 3+ cards of same suit
    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    if any(v >= 3 for v in suit_counts.values()):
        return "monotone"

    # Check wet: flush draw possible (2+ same suit) OR 2+ connected cards within 3 ranks
    has_flush_draw = any(v >= 2 for v in suit_counts.values())

    # Check connectivity: sort ranks, check if any two are within 3 of each other
    sorted_ranks = sorted(ranks)
    has_connectivity = False
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] <= 3:
                has_connectivity = True
                break
        if has_connectivity:
            break

    if has_flush_draw or has_connectivity:
        return "wet"

    return "dry"


# ── Preflop Action Parsing ──

def _parse_preflop_context(preflop_actions: str, hero_pos: str,
                           num_players: int = 8) -> dict:
    """Parse preflop actions to determine context for hero's decision.

    Returns dict with keys:
        hero_idx: hero's position index
        first_raiser_idx: index of first raiser (None if no raises before hero)
        first_raiser_pos: position of first raiser
        num_raises_before: number of raises before hero
        num_calls_before: number of calls before hero (after first raiser)
        has_limp: whether there's a limp (call without a prior raise)
        hero_action: hero's raw action code
        num_raises_total: total raises in preflop (including hero's and after)
        hero_raised: whether hero raised
    """
    pos_order = POSITION_ORDERS.get(num_players, POSITION_ORDERS[8])
    parts = preflop_actions.split("-")
    hero_idx = pos_order.index(hero_pos) if hero_pos in pos_order else -1

    first_raiser_idx = None
    first_raiser_pos = None
    num_raises_before = 0
    num_calls_before_hero = 0
    has_limp = False

    # Scan positions before hero (within the N-player seats)
    for i in range(min(hero_idx, len(parts))):
        code = parts[i]
        if code.startswith("R") or code.startswith("AI"):
            if first_raiser_idx is None:
                first_raiser_idx = i
                first_raiser_pos = pos_order[i] if i < len(pos_order) else None
            num_raises_before += 1
        elif code == "C":
            if first_raiser_idx is None:
                # Call without a prior raise = limp
                has_limp = True
            else:
                num_calls_before_hero += 1

    hero_action = parts[hero_idx] if hero_idx < len(parts) else ""
    hero_raised = hero_action.startswith("R") or hero_action.startswith("AI")

    # Count total raises (for detecting 4bet)
    num_raises_total = sum(
        1 for p in parts if p.startswith("R") or p.startswith("AI")
    )

    return {
        "hero_idx": hero_idx,
        "first_raiser_idx": first_raiser_idx,
        "first_raiser_pos": first_raiser_pos,
        "num_raises_before": num_raises_before,
        "num_calls_before": num_calls_before_hero,
        "has_limp": has_limp,
        "hero_action": hero_action,
        "num_raises_total": num_raises_total,
        "hero_raised": hero_raised,
    }


def categorize_preflop(preflop_actions: str, hero_pos: str,
                       num_players: int = 8, action_index: int = 0) -> str:
    """Categorize a preflop hero decision into a spot bucket.

    action_index: 0 = first decision, 1 = second decision (facing 3bet/4bet).

    Returns one of:
        open_raise, facing_open, facing_3bet, squeeze, facing_4bet, limp_pot
    """
    ctx = _parse_preflop_context(preflop_actions, hero_pos, num_players)

    if action_index > 0:
        # Hero's second decision: they already acted, now facing a re-raise
        if ctx["num_raises_total"] >= 4:
            return "facing_4bet"
        return "facing_3bet"

    # Hero's first decision
    if ctx["has_limp"] and ctx["num_raises_before"] == 0:
        return "limp_pot"

    if ctx["num_raises_before"] == 0:
        # No raises before hero — this is an open spot
        return "open_raise"

    if ctx["num_raises_before"] == 1:
        # One raise before hero
        if ctx["num_calls_before"] > 0 and ctx["hero_raised"]:
            # Open + call(s) + hero raises = squeeze
            return "squeeze"
        return "facing_open"

    if ctx["num_raises_before"] == 2:
        return "facing_3bet"

    if ctx["num_raises_before"] >= 3:
        return "facing_4bet"

    return "facing_open"


# ── Postflop Spot Categorization ──

def _determine_postflop_position(hero_pos: str, villain_pos: str,
                                 num_players: int = 8) -> str:
    """Determine if hero is IP (in position) or OOP (out of position).

    Postflop order: SB first, then BB, then remaining positions in preflop order.
    The player who acts LAST is IP.
    """
    pos_order = POSITION_ORDERS.get(num_players, POSITION_ORDERS[8])

    # Postflop order: SB, BB, then the rest (UTG..BTN) in original order
    blinds = []
    non_blinds = []
    for p in pos_order:
        if p in ("SB", "BB"):
            blinds.append(p)
        else:
            non_blinds.append(p)
    # SB always first, BB second
    postflop_order = ["SB", "BB"] if "SB" in blinds and "BB" in blinds else blinds
    postflop_order += non_blinds

    hero_pf_idx = postflop_order.index(hero_pos) if hero_pos in postflop_order else -1
    villain_pf_idx = postflop_order.index(villain_pos) if villain_pos in postflop_order else -1

    return "ip" if hero_pf_idx > villain_pf_idx else "oop"


def _find_preflop_caller(preflop_actions: str, aggressor_pos: str,
                         num_players: int = 8) -> str | None:
    """Find the last caller in preflop (the opponent who called the aggressor)."""
    pos_order = POSITION_ORDERS.get(num_players, POSITION_ORDERS[8])
    parts = preflop_actions.split("-")

    # Check continuation actions first (parts after N positions)
    if len(parts) > num_players:
        active = [i for i in range(num_players) if parts[i] not in ("F", "")]
        cont_idx = 0
        last_caller_pos = None
        for j in range(num_players, len(parts)):
            if cont_idx >= len(active):
                cont_idx = 0
            pos_idx = active[cont_idx]
            if parts[j] == "C" and pos_order[pos_idx] != aggressor_pos:
                last_caller_pos = pos_order[pos_idx]
            cont_idx += 1
        if last_caller_pos:
            return last_caller_pos

    # Check within the N positions for callers
    for i in range(min(num_players, len(parts))):
        code = parts[i]
        if code == "C" and i < len(pos_order) and pos_order[i] != aggressor_pos:
            return pos_order[i]

    return None


def _identify_preflop_aggressor(preflop_actions: str, num_players: int = 8) -> str | None:
    """Identify the preflop aggressor (last raiser) position."""
    pos_order = POSITION_ORDERS.get(num_players, POSITION_ORDERS[8])
    parts = preflop_actions.split("-")

    last_raiser_pos = None
    for i, code in enumerate(parts):
        if i < len(pos_order) and (code.startswith("R") or code.startswith("AI")):
            last_raiser_pos = pos_order[i]

    # Also check continuation actions after the N positions
    # These are re-actions by earlier active players
    if len(parts) > num_players:
        active = [i for i in range(num_players) if parts[i] not in ("F", "")]
        cont_idx = 0
        for j in range(num_players, len(parts)):
            if cont_idx >= len(active):
                cont_idx = 0
            if parts[j].startswith("R") or parts[j].startswith("AI"):
                last_raiser_pos = pos_order[active[cont_idx]]
            cont_idx += 1

    return last_raiser_pos


def categorize_postflop_action(
    street: str,
    hero_pos: str,
    street_actions_before_hero: list[dict],
    preflop_actions: str,
    num_players: int = 8,
    board: str | None = None,
) -> str:
    """Categorize a postflop hero decision into a spot bucket.

    street: "flop", "turn", "river"
    hero_pos: hero's position
    street_actions_before_hero: list of {position, action} dicts on this street before hero acts
    preflop_actions: full preflop action string
    num_players: table size

    Returns one of:
        cbet_ip, cbet_oop, facing_cbet_ip, facing_cbet_oop,
        probe, facing_probe, donk, check_raise
    """
    pf_aggressor = _identify_preflop_aggressor(preflop_actions, num_players)
    hero_is_pf_aggressor = (pf_aggressor == hero_pos)

    # Determine what happened before hero on this street
    checks_before = []
    bets_before = []
    for act in street_actions_before_hero:
        code = act.get("action", "")
        pos = act.get("position", "")
        if code == "X":
            checks_before.append(pos)
        elif code.startswith("R") or code == "C":
            bets_before.append({"position": pos, "action": code})

    # Determine IP/OOP relative to the main opponent
    # Use the PF aggressor as villain if hero isn't the aggressor,
    # otherwise find the caller from preflop actions or street actions
    villain_pos = None
    if not hero_is_pf_aggressor and pf_aggressor:
        villain_pos = pf_aggressor
    elif street_actions_before_hero:
        villain_pos = street_actions_before_hero[0]["position"]
    else:
        # Hero is PF aggressor, find the caller from preflop
        villain_pos = _find_preflop_caller(preflop_actions, hero_pos, num_players)

    if villain_pos:
        ip_oop = _determine_postflop_position(hero_pos, villain_pos, num_players)
    else:
        ip_oop = "ip"  # default fallback

    has_bet_before = len(bets_before) > 0

    if not has_bet_before:
        # No bets before hero — hero is first to bet or checks
        if hero_is_pf_aggressor:
            # PF aggressor, no bets before = c-bet opportunity
            return f"cbet_{ip_oop}"
        else:
            # Not PF aggressor = probe / donk opportunity
            # Whether checks came before or hero acts first, if they're not
            # the PF aggressor and there's no bet, it's a probe spot
            return "probe"
    else:
        # There's a bet before hero
        first_bettor = bets_before[0]["position"]

        # Check if hero previously checked this street (check-raise)
        if hero_pos in checks_before:
            return "check_raise"

        if first_bettor == pf_aggressor:
            # PF aggressor bet — this is a c-bet we're facing
            return f"facing_cbet_{ip_oop}"
        else:
            # Non-aggressor bet (donk or probe)
            if hero_is_pf_aggressor:
                # We're the PF aggressor, villain bet first = facing donk/probe
                return "facing_probe"
            else:
                # Neither is PF aggressor, or villain is betting
                return f"facing_cbet_{ip_oop}"  # generic facing bet


def categorize_spot(
    hand: dict,
    street: str,
    action_index: int = 0,
    street_actions_before_hero: list[dict] | None = None,
) -> tuple[str, str | None]:
    """High-level spot categorizer. Returns (spot_category, board_texture).

    hand: the hand dict (with preflop_actions, hero_position, etc.)
    street: "preflop", "flop", "turn", "river"
    action_index: for preflop, 0=first decision, 1=second (facing 3bet/4bet)
    street_actions_before_hero: for postflop, actions before hero on this street
    """
    hero_pos = hand.get("hero_position", "")
    preflop_actions = hand.get("preflop_actions", "")
    num_players = hand.get("players_at_table", 8)

    if street == "preflop":
        category = categorize_preflop(preflop_actions, hero_pos, num_players, action_index)
        return category, None

    # Postflop
    board = _get_board_for_street(hand, street)
    texture = classify_board_texture(board)

    if street_actions_before_hero is None:
        street_actions_before_hero = []

    category = categorize_postflop_action(
        street=street,
        hero_pos=hero_pos,
        street_actions_before_hero=street_actions_before_hero,
        preflop_actions=preflop_actions,
        num_players=num_players,
        board=board,
    )
    return category, texture


def _get_board_for_street(hand: dict, street: str) -> str | None:
    """Extract board string for a given street from hand dict."""
    streets = hand.get("streets") or hand.get("postflop_actions", [])
    if not streets:
        return None

    board = ""
    street_idx = {"flop": 0, "turn": 1, "river": 2}.get(street, -1)

    for i, st in enumerate(streets):
        if i == 0:
            board = st.get("board", "")
        else:
            board += st.get("card", "")
        if i == street_idx:
            return board
    return board or None
