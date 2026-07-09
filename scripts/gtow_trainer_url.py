"""Pure URL builder for GTO Wizard practice trainer deep-links.

This module builds deep-link URLs to the GTOW practice trainer so users
can jump into targeted drills for their leaks. It has no I/O and no
dependencies beyond the standard library.

Option Y limitation (see design doc harry-main-design-20260413-215833.md):
    The builder maps a spot_category / pot_type to GTOW's coarse
    `fh_actions` shortcut taxonomy (RFI, vsSRP, vs3bet, ...). It does
    NOT pin the exact position pair (e.g. LJ-vs-HJ); the trainer will
    drop the user into the general shortcut and they may need to pick
    the exact spot within the UI.

    Option Z (precise reverse-engineering of position-pair params) is
    tracked as a future TODO — see TODOS.md.

Reference URL shape (verified working, preflop vs3bet at 20bb):

    https://app.gtowizard.com/practice/trainer
        ?solution_type=gwiz
        &gametype=MTTGeneral
        &depth=20.125
        &depth_list=20.125
        &...trainer UI flags...
        &fh_start_spot=preflop
        &fh_actions=vs3bet
        &dialogs=

Turn/river note: GTOW's flop shortcut taxonomy is pot-type based and has
no dedicated turn/river shortcuts. For street="turn"/"river" we reuse
the flop pot_type mapping but emit `fh_start_spot=<street>` so the
trainer starts from the requested street.
"""

from __future__ import annotations

from urllib.parse import quote, urlencode


# Available practice depths observed in GTOW trainer (verified empirically
# where possible). If a user's effective_bb doesn't match exactly, snap_depth
# rounds to the nearest.
AVAILABLE_DEPTHS_BB: tuple[int, ...] = (10, 15, 20, 25, 30, 40, 50, 75, 100)

# Spot category → GTOW shortcut type (preflop)
# Matches the /available-shortcuts?gametypes=MTTGeneral API taxonomy.
_PREFLOP_SPOT_TO_FH_ACTIONS: dict[str, str] = {
    "open_raise":       "RFI",
    "facing_open":      "vsSRP",
    "hero_3bet":        "3bet",
    "facing_3bet":      "vs3bet",
    "facing_4bet":      "vs4bet",
    "squeeze":          "Squeeze",
    "vs_squeeze":       "vsSqueeze",
    "possible_squeeze": "possibleSqueeze",
    "limp_pot":         "vsLimp",
}

# Pot type → GTOW flop shortcut type
# GTOW flop taxonomy is by pot type, not by action (cbet/probe/donk).
# 4bet pots have no dedicated GTOW flop shortcut; closest is 3bet pot.
_POT_TYPE_TO_FH_ACTIONS_FLOP: dict[str, str] = {
    "SRP":      "SRP",
    "3bet":     "3bet",
    "4bet":     "3bet",
    "squeezed": "Squeeze",
    "squeeze":  "Squeeze",
    "limp":     "limp",
    "iso":      "iso",
}

# Baseline trainer UI flags (copied from reference URL). Preserved as
# constants so every URL we build lands with the same UI affordances
# (ranges visible, equity chart, etc.).
_TRAINER_UI_DEFAULTS: dict[str, str] = {
    "solution_type":                "gwiz",
    "gmfft_sort_key":               "0",
    "gmfft_sort_order":             "desc",
    "fh_groups_selection":          "manual",
    "fh_trainer_tables":            "1",
    "fh_trainer_mode":              "stop_end_of_hand",
    "fh_trainer_game_mode":         "trainer_actions",
    "fh_trainer_grouping":          "swv_grouping_none",
    "fh_trainer_game_speed":        "normal",
    "fh_trainer_quick_result":      "on",
    "fh_trainer_hero_range":        "on",
    "fh_trainer_opponent_range":    "on",
    "fh_trainer_equity_chart":      "on",
    "fh_trainer_ranges_comparison": "on",
    "fh_trainer_hand_strength":     "on",
    "fh_trainer_total_frequency":   "on",
    "fh_trainer_category_hands":    "on",
    "fh_trainer_category_draws":    "on",
    "fh_trainer_hero_strategy":     "on",
    "fh_trainer_rng_visible":       "on",
    "fh_trainer_streak_visible":    "on",
    "dialogs":                      "",
}

_BASE_URL = "https://app.gtowizard.com/practice/trainer"

_VALID_STREETS = ("preflop", "flop", "turn", "river")


class SpotNotSupportedError(ValueError):
    """Raised when a spot_category / pot_type cannot be mapped to a GTOW shortcut."""


# ── Action-line taxonomy → GTOW drill params (verified 2026-07-09 via CDP,
#    see skill gtow-trainer-drill). fh_hero/fh_opponent/fh_rel_positions pin the
#    exact spot; comma-separated positions mean "any of". ──
_CAT_TO_FH_ACTIONS: dict[str, str] = {
    "RFI": "RFI",
    "vsOpen": "vsSRP",
    "vsRaiseCall": "vsRaiseCall",
    "vsSqueeze": "vsSqueeze",
    "vs3bet": "vs3bet",
    "vsCold3bet": "vs3bet",     # GTOW has no cold-3bet drill; nearest is vs3bet
    "vs4bet": "vs4bet",
    "vsCold4bet": "vs4bet",     # GTOW has no cold-4bet drill; nearest is vs4bet
}

CAT_POSITIONS: dict[str, list[str]] = {
    "EP": ["UTG", "UTG+1"], "MP": ["LJ", "HJ"], "LP": ["CO", "BTN"],
    "SB": ["SB"], "BB": ["BB"],
}

# MTT ChipEV trainer depth ladder (verified from a live drill URL 2026-07-09).
MTT_DEPTHS: tuple[int, ...] = (10, 12, 14, 17, 20, 25, 30, 35, 40)
# Stack bands (user's deep/mid/short) mapped onto the ladder. "deep" (>50bb) is
# rare in MTT ChipEV and snaps to the deepest available.
DEPTH_BAND_DEPTHS: dict[str, list[int]] = {
    "short": [10, 12, 14, 17, 20],   # <=20bb
    "medium": [25, 30, 35, 40],      # 20-50bb
    "large": [40],                    # >50bb -> deepest available
}


def _depth_params(depth_bb: float, depths: list[int] | None):
    """Return (depth, depth_list) strings. depths=None -> single snapped depth;
    a list -> multi-depth drill (spans all listed stacks)."""
    if depths:
        rep = depths[len(depths) // 2]
        return _format_depth(rep), ",".join(_format_depth(d) for d in depths)
    d = _format_depth(snap_depth(depth_bb))
    return d, d


def build_drill_url(
    category: str,
    street: str,
    depth_bb: float,
    hero_positions: list[str],
    opponent_positions: list[str] | None = None,
    rel_position: str | None = None,
    pot_type: str | None = None,
    depths: list[int] | None = None,
    gametype: str = "MTTGeneral",
) -> str:
    """Build a PRECISE GTOW trainer deep-link that pins the action-line spot.

    category: action-line spot_category (RFI/vsOpen/.../flop/turn/river).
    hero_positions: concrete GTOW positions (e.g. ["BTN"] or ["UTG","UTG+1"]).
    opponent_positions: villain positions, or None for "any".
    rel_position: "IP"/"OOP" or None. pot_type: for postflop fh_actions.
    depths: list of bb to span (multi-depth training); None -> single snapped
        depth_bb. Pass MTT_DEPTHS for "all stack depths".

    Raises SpotNotSupportedError if the category/pot_type has no GTOW shortcut.
    """
    if street not in _VALID_STREETS:
        raise ValueError(f"street must be one of {_VALID_STREETS}, got {street!r}")

    if street == "preflop":
        fh_actions = _CAT_TO_FH_ACTIONS.get(category)
        if fh_actions is None:
            raise SpotNotSupportedError(f"preflop category {category!r} has no GTOW shortcut")
        start_spot = "preflop"
    else:
        if pot_type is None:
            raise SpotNotSupportedError(f"postflop street {street!r} requires pot_type")
        fh_actions = _POT_TYPE_TO_FH_ACTIONS_FLOP.get(pot_type)
        if fh_actions is None:
            raise SpotNotSupportedError(f"postflop pot_type {pot_type!r} has no GTOW shortcut")
        start_spot = street  # trainer starts from the requested street

    depth_str, depth_list_str = _depth_params(depth_bb, depths)
    params: list[tuple[str, str]] = [
        ("solution_type", _TRAINER_UI_DEFAULTS["solution_type"]),
        ("gametype", gametype), ("depth", depth_str), ("depth_list", depth_list_str),
    ]
    for k, v in _TRAINER_UI_DEFAULTS.items():
        if k in ("solution_type", "dialogs"):
            continue
        params.append((k, v))
    params.append(("fh_start_spot", start_spot))
    params.append(("fh_actions", fh_actions))
    if hero_positions:
        params.append(("fh_hero", ",".join(hero_positions)))
    if opponent_positions:
        params.append(("fh_opponent", ",".join(opponent_positions)))
    if rel_position in ("IP", "OOP"):
        params.append(("fh_rel_positions", rel_position))
    params.append(("dialogs", _TRAINER_UI_DEFAULTS["dialogs"]))
    return f"{_BASE_URL}?{urlencode(params, quote_via=quote)}"


def snap_depth(effective_bb: float) -> int:
    """Snap to nearest available GTOW trainer depth.

    Clamps below min (10) to 10 and above max (100) to 100.
    For values between snap points, rounds to the nearest.
    Ties (equidistant) round DOWN (favoring the shallower stack which is
    usually more common in MTT late stages).

    Examples:
        snap_depth(22.4) -> 20
        snap_depth(22.6) -> 25
        snap_depth(17.5) -> 15  (tie rounds down)
        snap_depth(5)    -> 10  (clamped)
        snap_depth(150)  -> 100 (clamped)
        snap_depth(30.125) -> 30
    """
    lo = AVAILABLE_DEPTHS_BB[0]
    hi = AVAILABLE_DEPTHS_BB[-1]
    if effective_bb <= lo:
        return lo
    if effective_bb >= hi:
        return hi

    best = AVAILABLE_DEPTHS_BB[0]
    best_dist = abs(effective_bb - best)
    for d in AVAILABLE_DEPTHS_BB[1:]:
        dist = abs(effective_bb - d)
        # Strictly less → update. Ties (equal distance) do NOT update,
        # so we keep the earlier (shallower) candidate. This implements
        # "ties round down".
        if dist < best_dist:
            best = d
            best_dist = dist
    return best


def _format_depth(bb: int) -> str:
    """Format depth as GTOW's bb+0.125 string (e.g. 20 -> "20.125")."""
    return f"{bb}.125"


def build_trainer_url(
    spot_category: str,
    street: str,
    effective_bb: float,
    pot_type: str | None = None,
    gametype: str = "MTTGeneral",
) -> str:
    """Build a GTOW trainer deep-link URL.

    Args:
        spot_category: Leak detection spot bucket. For preflop, maps via
            _PREFLOP_SPOT_TO_FH_ACTIONS (open_raise, facing_3bet, ...).
            Ignored on postflop (pot_type drives the mapping instead).
        street: "preflop" | "flop" | "turn" | "river".
        effective_bb: Stack depth in bb; will be snapped to nearest
            GTOW-supported depth.
        pot_type: Required for postflop. One of SRP/3bet/4bet/squeezed/
            limp/iso. Maps via _POT_TYPE_TO_FH_ACTIONS_FLOP.
        gametype: GTOW gametype code, default MTTGeneral.

    Returns:
        Fully-formed URL string.

    Raises:
        SpotNotSupportedError: if the spot_category (preflop) or pot_type
            (postflop) cannot be mapped, or postflop pot_type is missing.
        ValueError: if street is not one of the four valid streets.

    Turn/river note: GTOW has no dedicated turn/river shortcuts, so the
    pot_type mapping falls back to flop taxonomy but fh_start_spot is
    still set to the requested street.
    """
    if street not in _VALID_STREETS:
        raise ValueError(
            f"street must be one of {_VALID_STREETS}, got {street!r}"
        )

    if street == "preflop":
        fh_actions = _PREFLOP_SPOT_TO_FH_ACTIONS.get(spot_category)
        if fh_actions is None:
            raise SpotNotSupportedError(
                f"preflop spot_category {spot_category!r} has no GTOW shortcut"
            )
    else:
        if pot_type is None:
            raise SpotNotSupportedError(
                f"postflop street {street!r} requires pot_type"
            )
        fh_actions = _POT_TYPE_TO_FH_ACTIONS_FLOP.get(pot_type)
        if fh_actions is None:
            raise SpotNotSupportedError(
                f"postflop pot_type {pot_type!r} has no GTOW shortcut"
            )

    depth_bb = snap_depth(effective_bb)
    depth_str = _format_depth(depth_bb)

    # Stable parameter order: solution_type, gametype, depth, depth_list,
    # trainer UI flags (in defaults dict order), fh_start_spot, fh_actions,
    # dialogs. Deterministic output aids testing + caching.
    params: list[tuple[str, str]] = []
    params.append(("solution_type", _TRAINER_UI_DEFAULTS["solution_type"]))
    params.append(("gametype", gametype))
    params.append(("depth", depth_str))
    params.append(("depth_list", depth_str))

    # Trainer UI flags (skip solution_type + dialogs which bookend the URL).
    for k, v in _TRAINER_UI_DEFAULTS.items():
        if k in ("solution_type", "dialogs"):
            continue
        params.append((k, v))

    params.append(("fh_start_spot", street))
    params.append(("fh_actions", fh_actions))
    params.append(("dialogs", _TRAINER_UI_DEFAULTS["dialogs"]))

    return f"{_BASE_URL}?{urlencode(params, quote_via=quote)}"
