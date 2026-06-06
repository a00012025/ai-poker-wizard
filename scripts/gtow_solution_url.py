"""Build GTO Wizard `/solutions` strategy deep-links for a played hand.

Unlike gtow_custom_url (which builds a *practice trainer* drill) and
gtow_trainer_url (coarse bucket shortcuts), this module emits the
`/solutions?...&soltab=strategy` URL that lands the user on the exact
solver node for a specific decision in their hand — the same view the
bot's analysis is derived from.

Public entry points:
  - build_solution_url(resolved, board): pure URL assembly from a
    gtow_action_resolver result + a canonical board string. No I/O.
  - build_last_node_url(context): pick hero's LAST decision node in the
    analysed hand and return its solutions URL, walking backward to the
    nearest earlier buildable node on failure. Returns None if nothing
    builds. Calls the resolver (network) unless a stub is injected.

Reference URL (verified by hand, H3476 — BB 65s 40bb MTT, turn fold):

    https://app.gtowizard.com/solutions
        ?gametype=MTTGeneral&depth=40.125
        &gmfft_sort_key=0&gmfft_sort_order=desc
        &solution_type=gwiz&gmfs_solution_tab=ai_sols&soltab=strategy
        &preflop_actions=F-F-R2.3-F-F-F-F-C&history_spot=13
        &gmff_favorite=false&board=8h7d2hAh
        &flop_actions=X-R2-C&turn_actions=X-R8.4

Note the flop is reordered rank-descending (7d8h2h -> 8h7d2h) and the
board is truncated to the decision street (turn node -> flop+turn, no
river even when a river was dealt).
"""
from __future__ import annotations

import logging
from urllib.parse import quote, urlencode

_log = logging.getLogger(__name__)

_BASE_URL = "https://app.gtowizard.com/solutions"

# Static UI params, copied verbatim from the verified reference URL. Order
# is preserved for deterministic output (aids testing); the SPA does not
# depend on order.
_STATIC_UI: tuple[tuple[str, str], ...] = (
    ("gmfft_sort_key", "0"),
    ("gmfft_sort_order", "desc"),
    ("solution_type", "gwiz"),
    ("gmfs_solution_tab", "ai_sols"),
    ("soltab", "strategy"),
)

_RANK_ORDER = "23456789TJQKA"
# Secondary key for same-rank flop cards (paired flops). Verified by hand:
# a KhKd5c link (Kh before Kd) opens to the correct GTOW node. Distinct-rank
# flops never hit this path.
_SUIT_ORDER = "shdc"

# streets[] index count to include for a decision on each postflop street.
_STREET_BOARD_DEPTH = {"flop": 1, "turn": 2, "river": 3}
_STREET_ORDER = ("preflop", "flop", "turn", "river")


def _split_cards(s: str) -> list[str]:
    """'8h7d2h' -> ['8h','7d','2h']. Empty -> []."""
    s = (s or "").strip()
    if not s:
        return []
    if len(s) % 2 != 0:
        raise ValueError(f"card string length must be even, got {s!r}")
    return [s[i:i + 2] for i in range(0, len(s), 2)]


def _canonical_flop(flop: str) -> str:
    """Reorder a 3-card flop to GTOW canonical order: rank-descending.

    '7d8h2h' -> '8h7d2h'. Same-rank cards (paired flop) fall back to a
    fixed suit order so output stays deterministic.
    """
    cards = _split_cards(flop)
    cards.sort(key=lambda c: (-_RANK_ORDER.index(c[0]), _SUIT_ORDER.index(c[1])))
    return "".join(cards)


def canonical_board_through_street(hand_data: dict, street: str) -> str:
    """Board cards dealt through `street`, GTOW-canonical.

    Flop reordered rank-descending; turn/river appended in dealt order.
    Preflop -> "". A turn decision yields flop+turn even if a river card
    was later dealt (hero hadn't seen it at the decision).
    """
    if street == "preflop":
        return ""
    depth = _STREET_BOARD_DEPTH.get(street)
    if depth is None:
        raise ValueError(f"unknown street {street!r}")
    streets = hand_data.get("streets") or []
    if not streets:
        return ""
    parts: list[str] = []
    flop_raw = streets[0].get("board") or ""
    parts.append(_canonical_flop(flop_raw))
    for i in range(1, depth):
        if i < len(streets):
            parts.append((streets[i].get("card") or ""))
    return "".join(parts)


def build_solution_url(resolved: dict, board: str) -> str:
    """Assemble a /solutions strategy URL from a resolver result.

    Args:
        resolved: output of gtow_action_resolver.resolve_actions_for_deviation
            (keys: preflop_actions, flop/turn/river_actions, history_spot,
            depth, gametype).
        board: canonical board string for the decision street, or "" for
            a preflop node.

    Raises ValueError if resolved is missing the action line (no preflop
    actions means there is nothing to deep-link to).
    """
    preflop = resolved.get("preflop_actions") or ""
    if not preflop:
        raise ValueError("resolved result has no preflop_actions")

    gametype = resolved.get("gametype") or "MTTGeneral"
    depth_str = f"{resolved['depth']:g}"
    if gametype == "MTTGeneral" and not depth_str.endswith(".125"):
        depth_str = f"{int(resolved['depth'])}.125"

    params: list[tuple[str, str]] = []
    params.append(("gametype", gametype))
    params.append(("depth", depth_str))
    params.extend(_STATIC_UI)
    params.append(("preflop_actions", preflop))
    params.append(("history_spot", str(resolved.get("history_spot", 0))))
    params.append(("gmff_favorite", "false"))
    if board:
        params.append(("board", board))
    if resolved.get("flop_actions"):
        params.append(("flop_actions", resolved["flop_actions"]))
    if resolved.get("turn_actions"):
        params.append(("turn_actions", resolved["turn_actions"]))
    if resolved.get("river_actions"):
        params.append(("river_actions", resolved["river_actions"]))

    return f"{_BASE_URL}?{urlencode(params, quote_via=quote)}"


def enumerate_hero_decisions(context: dict) -> list[tuple[str, int]]:
    """List hero decision points as (street, action_index), in play order.

    action_index semantics match src/gemini_session._extract_deviations and
    gtow_action_resolver (preflop = hero's Nth preflop decision; postflop =
    hero's Nth decision on that street). Only spots with a real solution are
    included, so we never deep-link to a node the solver has no data for.
    """
    hero_spots = context.get("hero_spots") or []
    solutions = context.get("solutions") or []
    decisions: list[tuple[str, int]] = []
    preflop_idx = 0
    for i, spot in enumerate(hero_spots):
        sol = solutions[i] if i < len(solutions) else None
        if not sol or "action_solutions" not in sol:
            continue
        street = spot.get("street", "")
        if street == "preflop":
            decisions.append((street, preflop_idx))
            preflop_idx += 1
        else:
            action_idx = sum(
                1 for j in range(i)
                if hero_spots[j].get("street") == street and street != "preflop"
            )
            decisions.append((street, action_idx))
    return decisions


def build_last_node_url(context: dict, *, _resolver=None) -> str | None:
    """Build a /solutions URL for hero's last decision node in the hand.

    Walks decision points latest-first and returns the first that builds a
    valid URL — so an off-tree / unresolvable final node falls back to the
    nearest earlier node. Returns None if nothing builds.

    All resolver/builder failures are swallowed (logged at debug); this is a
    convenience link and must never break the surrounding message.
    """
    resolver = _resolver
    if resolver is None:
        from gtow_action_resolver import resolve_actions_for_deviation
        resolver = resolve_actions_for_deviation

    hand = context.get("hand") or {}
    # analyze_hand_full normalizes the preflop to the 8-max MTT tree in ctx hand
    # but leaves players_at_table at the physical count. The resolver pads to the
    # tree itself, so hand it the un-padded line + physical table size instead,
    # or it pads a SECOND time and misplaces every actor (H3490).
    raw_preflop = context.get("deeplink_raw_preflop")
    if raw_preflop is not None:
        hand = {**hand, "preflop_actions": raw_preflop,
                "players_at_table": context.get("deeplink_raw_players")}
    decisions = enumerate_hero_decisions(context)
    if not decisions:
        return None

    for street, action_index in reversed(decisions):
        try:
            resolved = resolver(hand, street, action_index)
            board = canonical_board_through_street(hand, street)
            return build_solution_url(resolved, board)
        except Exception as e:  # noqa: BLE001 — convenience link, never fatal
            _log.debug("solution URL build failed at %s[%d]: %s",
                       street, action_index, e)
            continue
    return None
