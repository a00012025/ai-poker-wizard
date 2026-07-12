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


# ── build straight from an archived GTOW Analyze hand detail ──────────────────
# The online ledger's raw archive already carries, per decision game_point, the
# exact solved action line (`solved_action_sequence`) + depth + board the solver
# was queried at. That maps 1:1 onto build_solution_url with no live token and
# no re-resolution — the most faithful node link for a played online hand.
_GA_STREET = {"PREFLOP": "preflop", "FLOP": "flop", "TURN": "turn", "RIVER": "river"}
_STREET_BOARD_CHARS = {"flop": 6, "turn": 8, "river": 10}


def _find_decision_game_point(detail: dict, hero_pos: str, street: str,
                              decision_idx: int) -> dict | None:
    """Locate the game_point for hero's Nth decision on `street`.

    Counts hero decisions per street exactly like ledger_distill.distill_hand
    (a hero decision = a game_point where the actor is hero and some available
    action is `selected`), so (street, decision_idx) match the ledger rows.
    """
    gps = ((detail.get("game_analysis") or {}).get("game_points")) or []
    counts: dict[str, int] = {}
    for gp in gps:
        pos = (gp.get("real_game_action") or {}).get("position", "")
        st = _GA_STREET.get(
            ((gp.get("real_game") or {}).get("current_street") or {}).get("type", ""), "")
        avail = (gp.get("analysis_solved") or {}).get("available_actions") or []
        if pos == hero_pos and any(a.get("selected") for a in avail):
            idx = counts.get(st, 0)
            if st == street and idx == decision_idx:
                return gp
            counts[st] = idx + 1
    return None


def _canonical_board_str(board_raw: str, street: str) -> str:
    """Canonical board (flop rank-descending) truncated to `street`'s depth."""
    if street == "preflop" or not board_raw:
        return ""
    n = _STREET_BOARD_CHARS.get(street)
    b = board_raw[:n] if n else board_raw
    return _canonical_flop(b[:6]) + b[6:] if len(b) >= 6 else b


def build_hand_solution_url(detail: dict, hero_pos: str, street: str,
                            decision_idx: int) -> str | None:
    """/solutions Study URL for hero's (street, decision_idx) decision, built
    from the archived hand detail's `solved_action_sequence`.

    Returns None when the node can't be located or the solver has no solution
    for it (caller falls back), so a review link never points at an empty node.
    """
    gp = _find_decision_game_point(detail, hero_pos, street, decision_idx)
    if gp is None or not gp.get("has_solution"):
        return None
    sas = gp.get("solved_action_sequence") or {}
    pre = sas.get("preflop_actions") or []
    if not pre:
        return None

    def _join(x):
        return "-".join(x) if x else ""

    def _cnt(x):
        return len(x or [])

    resolved = {
        "preflop_actions": _join(pre),
        "flop_actions": _join(sas.get("flop_actions")),
        "turn_actions": _join(sas.get("turn_actions")),
        "river_actions": _join(sas.get("river_actions")),
        "history_spot": (_cnt(pre) + _cnt(sas.get("flop_actions"))
                         + _cnt(sas.get("turn_actions")) + _cnt(sas.get("river_actions"))),
        "depth": float(gp.get("depth") or 0),
        "gametype": gp.get("gametype") or "MTTGeneral",
    }
    board = _canonical_board_str((gp.get("real_game") or {}).get("board") or "", street)
    try:
        return build_solution_url(resolved, board)
    except ValueError:
        return None


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


def _deeplink_hand(context: dict) -> dict:
    """Hand dict prepared for the resolver (un-padded preflop + physical seats).

    analyze_hand_full normalizes the preflop to the 8-max MTT tree in ctx hand
    but leaves players_at_table at the physical count. The resolver pads to the
    tree itself, so hand it the un-padded line + physical table size instead,
    or it pads a SECOND time and misplaces every actor (H3490).
    """
    hand = context.get("hand") or {}
    raw_preflop = context.get("deeplink_raw_preflop")
    if raw_preflop is not None:
        hand = {**hand, "preflop_actions": raw_preflop,
                "players_at_table": context.get("deeplink_raw_players")}
    return hand


def _resolved_from_spot(spot: dict) -> dict | None:
    """Resolver-shaped dict from a hero_spot's already-snapped params.

    ``analyze_hand_full`` snaps every action to its GTOW code while walking the
    hand and stores the exact line it queried the solver with in
    ``spot['params']`` — a node the API demonstrably returned data for.
    Reusing it makes the deep-link exact and API-free, instead of re-snapping
    via ``next_actions`` (which needs a live token; on failure that path falls
    back to raw ``'B'``/``'X'`` tokens GTOW can't parse, yielding a "something
    went wrong" page — H3639).

    Returns None if the spot has no usable params (e.g. a context rehydrated
    from the DB without them), so the caller can fall back to the resolver.
    """
    p = spot.get("params") or {}
    preflop = p.get("preflop_actions") or ""
    depth = p.get("depth")
    if not preflop or depth is None:
        return None

    flop = p.get("flop_actions") or ""
    turn = p.get("turn_actions") or ""
    river = p.get("river_actions") or ""

    def _count(s: str) -> int:
        return len([t for t in s.split("-") if t]) if s else 0

    return {
        "preflop_actions": preflop,
        "flop_actions": flop,
        "turn_actions": turn,
        "river_actions": river,
        "history_spot": _count(preflop) + _count(flop) + _count(turn) + _count(river),
        "depth": depth,
        "gametype": p.get("gametype") or "MTTGeneral",
    }


def _url_from_spot(context: dict, spot: dict) -> str | None:
    """Build a /solutions URL straight from a hero_spot's resolved params.

    The board is re-derived canonically (flop rank-descending) from the hand so
    the ordering matches GTOW, but the action codes come verbatim from the spot
    — no API call. Returns None if the spot lacks params.
    """
    resolved = _resolved_from_spot(spot)
    if resolved is None:
        return None
    try:
        board = canonical_board_through_street(_deeplink_hand(context), spot.get("street", ""))
        return build_solution_url(resolved, board)
    except Exception as e:  # noqa: BLE001 — fall back to the resolver path
        _log.debug("spot-params URL build failed at %s: %s", spot.get("street"), e)
        return None


def _solution_bearing_spots(context: dict) -> list[dict]:
    """Hero spots that actually have solver data, in play order."""
    hero_spots = context.get("hero_spots") or []
    solutions = context.get("solutions") or []
    out = []
    for i, spot in enumerate(hero_spots):
        sol = solutions[i] if i < len(solutions) else None
        if sol and "action_solutions" in sol:
            out.append(spot)
    return out


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

    hand = _deeplink_hand(context)
    decisions = enumerate_hero_decisions(context)
    if not decisions:
        return None

    # Prefer the already-resolved codes on each hero_spot (exact + API-free).
    # decisions and solution-bearing spots are both derived by filtering
    # solution-bearing spots in play order, so they align 1:1.
    spots = _solution_bearing_spots(context)
    for spot in reversed(spots):
        url = _url_from_spot(context, spot)
        if url:
            return url

    # Fallback: re-resolve via the next_actions API (needs a live token).
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


def build_node_url_for_street(context: dict, street: str,
                              *, _resolver=None) -> str | None:
    """Build a /solutions URL for hero's FIRST decision on `street`.

    A follow-up answer about an earlier street ("what's my turn betting
    range?") is grounded on hero's decision node for that street — which is
    NOT the played-line last node. Using build_last_node_url for that reply
    deep-links the user to the river while the coach quoted the turn, so the
    frequencies disagree (turn check 89% vs river check 23%, H3515). This
    builds the link to the exact node the answer describes so both sides match.

    Returns None (caller falls back to last-node) if the street has no hero
    decision or the node won't build.
    """
    resolver = _resolver
    if resolver is None:
        from gtow_action_resolver import resolve_actions_for_deviation
        resolver = resolve_actions_for_deviation

    hand = _deeplink_hand(context)

    # Prefer the already-resolved codes on hero's first solution-bearing spot
    # for this street (exact + API-free).
    for spot in _solution_bearing_spots(context):
        if spot.get("street") != street:
            continue
        url = _url_from_spot(context, spot)
        if url:
            return url
        break  # spot lacked params — fall through to the resolver below

    # Fallback: re-resolve via the next_actions API (needs a live token).
    for s, action_index in enumerate_hero_decisions(context):
        if s != street:
            continue
        try:
            resolved = resolver(hand, s, action_index)
            board = canonical_board_through_street(hand, s)
            return build_solution_url(resolved, board)
        except Exception as e:  # noqa: BLE001 — convenience link, never fatal
            _log.debug("node URL build failed at %s[%d]: %s", s, action_index, e)
            return None
    return None
