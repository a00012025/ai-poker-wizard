#!/usr/bin/env python3
"""Spot categorizer — classify each hero decision point into ~15 spot buckets.

Pure function: no API calls, just parsing the action sequence and board.

Spot categories:
  Preflop:  open_raise, facing_open, possible_squeeze, hero_3bet,
            facing_3bet, vs_squeeze, squeeze, facing_4bet, limp_pot
  Postflop: cbet_ip, cbet_oop, facing_cbet_ip, facing_cbet_oop,
            probe, facing_probe, donk, check_raise
"""

from __future__ import annotations

from position_constants import POSITION_ORDERS


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

    # Check wet: flush draw possible (≥2 same suit) OR straight draws live.
    # Straight-draw test mirrors GTOW's flop_connectedness vocab so the
    # cluster's texture label agrees with the practice-URL board params:
    #     gaps == [1,1]           → connected (e.g. 789)
    #     any gap of 1            → oesd_possible (e.g. 78T, 235)
    #     otherwise (all gaps ≥2) → disconnected
    # Both 'connected' and 'oesd_possible' classify as wet; only fully
    # disconnected rainbow boards (e.g. Q94r, K72r) are dry.
    has_flush_draw = any(v >= 2 for v in suit_counts.values())
    sorted_ranks_3 = sorted(ranks)[:3]
    gaps = [sorted_ranks_3[i + 1] - sorted_ranks_3[i]
            for i in range(len(sorted_ranks_3) - 1)]
    has_straight_potential = 1 in gaps

    if has_flush_draw or has_straight_potential:
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


def _iter_preflop_tokens(preflop_actions: str, num_players: int):
    """Yield (position, raw_token) pairs in ORDER of action.

    Handles both the seat-indexed section (first num_players entries)
    and the continuation section (subsequent entries cycle through
    still-active, non-folded players in seat order — same convention
    as _identify_preflop_aggressor).
    """
    pos_order = POSITION_ORDERS.get(num_players, POSITION_ORDERS[8])
    parts = preflop_actions.split("-")

    # Seat section: one token per seat, in seat order
    n_seat = min(num_players, len(parts))
    for i in range(n_seat):
        token = parts[i]
        if token == "":
            continue
        pos = pos_order[i] if i < len(pos_order) else f"P{i}"
        yield pos, token

    if len(parts) <= num_players:
        return

    # Continuation section: cycle through active (non-folded) seats.
    active = [i for i in range(num_players)
              if i < len(parts) and parts[i] not in ("F", "")]
    if not active:
        return
    cont_idx = 0
    for j in range(num_players, len(parts)):
        token = parts[j]
        if token == "":
            cont_idx = (cont_idx + 1) % len(active)
            continue
        pos_idx = active[cont_idx]
        pos = pos_order[pos_idx] if pos_idx < len(pos_order) else f"P{pos_idx}"
        yield pos, token
        cont_idx = (cont_idx + 1) % len(active)


def _classify_token(token: str, current_raise_level: int) -> tuple[str, int]:
    """Map a raw action token into (line_key_action, new_raise_level).

    Raise levels: 0 = no raise yet, 1 = R (open), 2 = RR (3bet), 3 = RRR (4bet), ...
    Returns (action_code, updated_raise_level). action_code in {F,C,R,RR,RRR,AI}.
    All-ins ('AI*') are treated as raises at the next level, but the
    returned action_code stays 'AI' (carries its own semantic).
    """
    if token == "F":
        return "F", current_raise_level
    if token == "C" or token == "X":
        # X (check) behaves like C in BB when BB can check through a limp pot.
        return "C", current_raise_level
    if token.startswith("AI"):
        new_level = current_raise_level + 1
        return "AI", new_level
    if token.startswith("R"):
        new_level = current_raise_level + 1
        code_map = {1: "R", 2: "RR", 3: "RRR", 4: "RRRR"}
        return code_map.get(new_level, "R" * new_level), new_level
    # Unknown token — skip.
    return "", current_raise_level


def _hero_faced_squeeze(preflop_actions: str, hero_pos: str,
                        num_players: int) -> bool:
    """True if hero opened, a caller came in, then a re-raise happened
    before hero's next decision (the vs_squeeze pattern)."""
    hero_opened = False
    saw_caller_after_hero = False
    for pos, token in _iter_preflop_tokens(preflop_actions, num_players):
        if not hero_opened:
            if pos == hero_pos and (token.startswith("R") or token.startswith("AI")):
                hero_opened = True
            continue
        # After hero opened
        if pos == hero_pos:
            # Hero's second action reached — stop scanning.
            return False
        if token == "C":
            saw_caller_after_hero = True
        elif token.startswith("R") or token.startswith("AI"):
            if saw_caller_after_hero:
                return True
            # A 3bet without a caller between = plain facing_3bet.
            return False
    return False


def compute_preflop_line_key(preflop_actions: str, hero_pos: str,
                             num_players: int = 8,
                             action_index: int | None = 0) -> str:
    """Build a compact signature of the preflop action sequence.

    Grammar:
        line_key := "-".join of POS-ACT tokens in action order
        POS       := position name (UTG, LJ, HJ, CO, BTN, SB, BB, ...)
        ACT       := F | C | R | RR | RRR | AI
    Rules:
        - Action order (not seat order).
        - Hero's own tokens are EXCLUDED.
        - Folds are elided unless they follow a re-raise (RR or higher).
          (Folds to a single open carry no range info; folds to a 3bet/4bet
          do.)
        - action_index semantics:
            * int (default 0): for a PREFLOP decision. Key captures what
              happened before hero's (action_index+1)-th preflop token.
              Use 0 for hero's first decision, 1 for facing-3bet, etc.
            * None: for a POSTFLOP decision. Consume the full preflop
              sequence (no stopping) — the key describes the pot type
              going into the flop.
    """
    tokens_out: list[str] = []
    raise_level = 0
    hero_action_count = 0
    # Stop once hero has been seen (action_index + 1) times: the key
    # captures everything strictly before hero's current decision point.
    # action_index=None means "never stop" — used for postflop line_keys
    # where we want the full preflop sequence.
    stop_after_hero_count = None if action_index is None else action_index + 1
    for pos, raw in _iter_preflop_tokens(preflop_actions, num_players):
        if pos == hero_pos:
            hero_action_count += 1
            if stop_after_hero_count is not None and hero_action_count >= stop_after_hero_count:
                break
            # Classify to update raise_level (so folds after a hero
            # re-raise are still kept) but do NOT emit hero's token.
            _code, raise_level = _classify_token(raw, raise_level)
            continue

        code, new_level = _classify_token(raw, raise_level)
        if code == "":
            continue
        if code == "F":
            # Keep folds only if they follow a re-raise (raise_level >= 2).
            if raise_level >= 2:
                tokens_out.append(f"{pos}-{code}")
            # else elide
        else:
            tokens_out.append(f"{pos}-{code}")
            raise_level = new_level

    return "-".join(tokens_out)


def compute_pot_type_from_preflop(preflop_actions: str,
                                  num_players: int = 8) -> str:
    """Classify pot type directly from the raw preflop_actions string.

    Preferred over compute_pot_type(line_key) because it doesn't depend on
    hero-exclusion semantics (when hero is the opener, their R is excluded
    from line_key, which can make an SRP look like a limp pot).

    Returns one of: SRP, 3bet, 4bet, squeezed, limp, unopened.
    """
    if not preflop_actions:
        return "unopened"
    max_level = 0
    saw_raise = False
    any_call_before_raise = False
    saw_call_after_first_raise = False
    squeeze = False
    for _pos, raw in _iter_preflop_tokens(preflop_actions, num_players):
        code, new_level = _classify_token(raw, max_level)
        if code == "C":
            if not saw_raise:
                any_call_before_raise = True
            elif max_level == 1:
                saw_call_after_first_raise = True
        elif code in ("R", "RR", "RRR", "RRRR", "AI"):
            if new_level == 2 and saw_call_after_first_raise:
                squeeze = True
            saw_raise = True
            if new_level > max_level:
                max_level = new_level

    if max_level >= 3:
        return "4bet"
    if max_level == 2:
        return "squeezed" if squeeze else "3bet"
    if max_level == 1:
        return "limp" if any_call_before_raise else "SRP"
    return "unopened"


def compute_pot_type(preflop_line_key: str) -> str:
    """Classify the pot type from a preflop line_key.

    NOTE: Prefer compute_pot_type_from_preflop() at call sites that have
    access to raw preflop_actions. This function is kept for cases where
    only the line_key is available, but beware: when hero is the opener
    the hero's raise is excluded from the line_key, which can cause
    false "limp" / "unopened" classifications.

    Returns one of: SRP, 3bet, 4bet, squeezed, limp, unopened.
    """
    if not preflop_line_key:
        return "unopened"
    tokens = preflop_line_key.split("-")
    # Extract just the action codes (every other token starting at index 1).
    actions = [tokens[i] for i in range(1, len(tokens), 2)]

    if "RRRR" in actions or "RRR" in actions:
        return "4bet"

    has_rr = "RR" in actions
    has_r = "R" in actions
    # Squeeze detection: a C appears before an RR.
    if has_rr:
        first_rr = actions.index("RR")
        earlier = actions[:first_rr]
        if "C" in earlier and "R" in earlier:
            return "squeezed"
        return "3bet"

    if has_r:
        # Limp vs SRP: if a C appears before the R, it's a limp pot (iso).
        first_r = actions.index("R")
        earlier = actions[:first_r]
        if "C" in earlier:
            return "limp"
        return "SRP"

    # No raise at all.
    if "C" in actions:
        return "limp"
    return "unopened"


def categorize_preflop(preflop_actions: str, hero_pos: str,
                       num_players: int = 8, action_index: int = 0) -> str:
    """Categorize a preflop hero decision into a spot bucket.

    action_index: 0 = first decision, 1 = second decision (facing 3bet/4bet).

    Returns one of:
        open_raise, facing_open, possible_squeeze, hero_3bet,
        facing_3bet, vs_squeeze, squeeze, facing_4bet, limp_pot
    """
    ctx = _parse_preflop_context(preflop_actions, hero_pos, num_players)

    if action_index > 0:
        # Hero's second decision: they already acted, now facing a re-raise
        if ctx["num_raises_total"] >= 4:
            return "facing_4bet"
        # vs_squeeze: hero opened, a caller came in, then a re-raise (squeeze).
        # Detect by scanning preflop_actions for pattern: hero-R, caller-C, then RR
        if _hero_faced_squeeze(preflop_actions, hero_pos, num_players):
            return "vs_squeeze"
        return "facing_3bet"

    # Hero's first decision
    if ctx["has_limp"] and ctx["num_raises_before"] == 0:
        return "limp_pot"

    if ctx["num_raises_before"] == 0:
        # No raises before hero — this is an open spot
        return "open_raise"

    if ctx["num_raises_before"] == 1:
        # One raise before hero
        if ctx["num_calls_before"] > 0:
            if ctx["hero_raised"]:
                # Open + call(s) + hero raises = squeeze
                return "squeeze"
            # Open + call(s), hero did not raise = possible_squeeze spot
            return "possible_squeeze"
        # No callers in front
        if ctx["hero_raised"]:
            # Hero is the one 3-betting facing an open
            return "hero_3bet"
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


# ── GTOW mapping + primary villain helpers ──

# Maps spot_category (preflop) → (gtow_type, gtow_hero_role).
# Postflop categories are handled separately via pot_type.
_PREFLOP_SPOT_TO_GTOW: dict[str, tuple[str, str]] = {
    "open_raise":       ("RFI",              "aggressor"),
    "facing_open":      ("vsSRP",            "caller_candidate"),
    "hero_3bet":        ("3bet",             "3bettor"),
    "facing_3bet":      ("vs3bet",           "opener"),
    "facing_4bet":      ("vs4bet",           "3bettor"),
    "squeeze":          ("Squeeze",          "squeezer"),
    "vs_squeeze":       ("vsSqueeze",        "opener"),
    "possible_squeeze": ("possibleSqueeze",  "caller_candidate"),
    "limp_pot":         ("vsLimp",           "iso_candidate"),
}

# Maps pot_type → postflop GTOW type (flop taxonomy).
_POT_TYPE_TO_FLOP_GTOW: dict[str, str] = {
    "SRP":       "SRP",
    "3bet":      "3bet",
    "4bet":      "3bet",      # GTOW has no 4bet flop; nearest is 3bet pot
    "squeezed":  "Squeeze",
    "limp":      "limp",
    "iso":       "iso",
    "unopened":  "SRP",       # best-effort fallback
}


def map_spot_to_gtow(
    spot_category: str,
    pot_type: str | None,
    street: str,
    hero_is_pf_aggressor: bool,
) -> tuple[str | None, str | None]:
    """Return (gtow_type, gtow_hero_role) for URL builder + mining.

    - Preflop uses spot_category directly.
    - Postflop uses pot_type for the type and hero_is_pf_aggressor for role.
    """
    if street == "preflop":
        return _PREFLOP_SPOT_TO_GTOW.get(spot_category, (None, None))

    gtow_type = _POT_TYPE_TO_FLOP_GTOW.get(pot_type or "", None)
    role = "aggressor" if hero_is_pf_aggressor else "caller"
    return gtow_type, role


def identify_primary_villain(
    hand: dict,
    hero_pos: str,
    street: str,
    street_actions_before_hero: list[dict] | None,
) -> str | None:
    """Pick a single primary villain for this decision point.

    - Preflop: last raiser (aggressor) that hero is facing.
    - Postflop: last bettor before hero on this street; else the preflop
      aggressor (the player hero most likely has in mind).
    """
    preflop_actions = hand.get("preflop_actions", "")
    num_players = hand.get("players_at_table", 8)

    if street == "preflop":
        agg = _identify_preflop_aggressor(preflop_actions, num_players)
        if agg and agg != hero_pos:
            return agg
        return None

    if street_actions_before_hero:
        last_bettor = None
        for act in street_actions_before_hero:
            code = act.get("action", "")
            pos = act.get("position", "")
            if not pos or pos == hero_pos:
                continue
            if code.startswith("R") or code.startswith("AI"):
                last_bettor = pos
        if last_bettor:
            return last_bettor

    agg = _identify_preflop_aggressor(preflop_actions, num_players)
    if agg and agg != hero_pos:
        return agg
    return None


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
