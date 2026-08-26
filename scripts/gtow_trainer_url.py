"""Pure URL builder for GTO Wizard practice trainer deep-links.

This module builds deep-link URLs to the GTOW practice trainer so users
can jump into targeted drills for their leaks. It has no I/O and no
dependencies beyond the standard library.

Preflop shortcut links pin the supported GTOW action filter plus known seat
filters.  Exact postflop/cold-raise links are source-hand custom spots and are
built by ``gtow_custom_url``; this module refuses misleading approximations.

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

Postflop note: GTOW's public shortcuts only select a pot family.  They do not
pin the action history represented by an action-line leaf, and GTOW rewrites
turn/river starts back to preflop.  Queue links therefore build postflop spots
from a source hand via ``gtow_custom_url``; this module deliberately refuses
to present a coarse postflop shortcut as a precise drill.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


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

# Baseline trainer UI flags (copied from reference URL). Preserved as
# constants so every URL we build lands with the same UI affordances
# (ranges visible, equity chart, etc.).
ALL_TRAINER_GROUPS = (
    "22,33,44,55,66,77,88,99,AA,KK,QQ,JJ,TT,AKs,AKo,AQs,AJs,AQo,KQs,"
    "ATs,AJo,KJs,KQo,QJs,KTs,A9s,ATo,QTs,A8s,JTs,A7s,A5s,KJo,K9s,A4s,"
    "A6s,Q9s,A3s,T9s,QJo,J9s,A2s,KTo,A9o,K8s,QTo,K7s,JTo,Q8s,K6s,T8s,"
    "A8o,J8s,K5s,98s,A5o,A7o,K4s,K9o,Q7s,K3s,Q6s,A4o,T7s,A6o,K2s,J7s,"
    "Q9o,T9o,Q5s,97s,A3o,J9o,87s,Q4s,Q3s,A2o,K8o,76s,Q2s,J6s,T6s,J5s,"
    "65s,96s,86s,54s,K7o,J4s,Q8o,T8o,J3s,J8o,K6o,75s,98o,J2s,T5s,K5o,"
    "T4s,64s,53s,85s,95s,T3s,K4o,T2s,43s,Q7o,74s,K3o,Q6o,T7o,J7o,97o,"
    "63s,87o,K2o,52s,Q5o,93s,84s,94s,42s,92s,Q4o,32s,73s,76o,Q3o,65o,"
    "54o,86o,T6o,Q2o,J6o,96o,62s,J5o,83s,82s,J4o,75o,J3o,72s,64o,53o,"
    "J2o,85o,T5o,95o,T4o,43o,T3o,T2o,74o,63o,52o,42o,84o,93o,94o,92o,"
    "32o,73o,62o,82o,83o,72o"
)

_TRAINER_UI_DEFAULTS: dict[str, str] = {
    "solution_type":                "gwiz",
    "gmfft_sort_key":               "0",
    "gmfft_sort_order":             "desc",
    "fh_groups_selection":          "manual",
    # GTOW injects this complete 169-class selection into every session even
    # when the deep link omits it.  Pin it explicitly so the settings object
    # used for Drill creation is byte-for-byte matchable by the Trainer UI.
    "fh_groups":                    ALL_TRAINER_GROUPS,
    "fh_trainer_tables":            "1",
    "fh_trainer_mode":              "stop_end_of_hand",
    "fh_trainer_game_mode":         "trainer_actions",
    "fh_trainer_grouping":          "swv_grouping_none",
    "fh_trainer_game_speed":        "turbo",
    "fh_trainer_learning_mode":     "on",
    "fh_trainer_session":           "100",
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

# GTOW's official v4 game-mode metadata marks MTTGeneral as
# ``info.variant=with_limps`` (the no-limp family is ``no_limps``).  The
# Trainer persists that selection in ``gmff_variant``.  Pin it on every MTT
# link so a user's previous Game Formats filter cannot silently switch the
# exercise to a no-limp solution.
MTT_HAS_LIMP_VARIANT = "with_limps"

_VALID_STREETS = ("preflop", "flop", "turn", "river")


def trainer_solution_defaults(gametype: str | None) -> dict[str, str]:
    """Solution-family filters that must be part of Trainer/Drill identity."""
    if (gametype or "").upper().startswith("MTT"):
        return {"gmff_variant": MTT_HAS_LIMP_VARIANT}
    return {}


def apply_trainer_defaults(url: str | None) -> str | None:
    """Apply global session defaults to a previously persisted Trainer URL."""
    if not url:
        return url
    parts = urlsplit(url)
    if parts.path.rstrip("/") != "/practice/trainer":
        return url
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    selected_groups = params.get("fh_groups")
    params.update(_TRAINER_UI_DEFAULTS)
    if selected_groups:
        params["fh_groups"] = selected_groups
    params.update(trainer_solution_defaults(params.get("gametype")))
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(params), parts.fragment))


def with_trainer_hand_groups(url: str | None,
                             hand_groups: list[str] | None) -> str | None:
    """Apply a selected 169-class subset without changing spot identity."""
    if not url or not hand_groups:
        return url
    parts = urlsplit(url)
    if parts.path.rstrip("/") != "/practice/trainer":
        return url
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params["fh_groups_selection"] = "manual"
    params["fh_groups"] = ",".join(hand_groups)
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(params), parts.fragment))


class SpotNotSupportedError(ValueError):
    """Raised when a spot_category / pot_type cannot be mapped to a GTOW shortcut."""


# ── Action-line taxonomy → GTOW drill params (verified 2026-07-09 via CDP,
#    see skill gtow-trainer-drill). fh_hero/fh_opponent/fh_rel_positions pin the
#    exact spot; comma-separated positions mean "any of". ──
_CAT_TO_FH_ACTIONS: dict[str, str] = {
    "RFI": "RFI",
    "vsOpen": "vsSRP",
    # GTOW calls the raise+call state in which hero can squeeze
    # ``possibleSqueeze``. ``vsRaiseCall`` is only our taxonomy name.
    "vsRaiseCall": "possibleSqueeze",
    "vsSqueeze": "vsSqueeze",
    "vs3bet": "vs3bet",
    "vs4bet": "vs4bet",
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
        raise SpotNotSupportedError(
            "postflop action-line drills require an exact custom_spot source hand")

    depth_str, depth_list_str = _depth_params(depth_bb, depths)
    params: list[tuple[str, str]] = [
        ("solution_type", _TRAINER_UI_DEFAULTS["solution_type"]),
        ("gametype", gametype), ("depth", depth_str), ("depth_list", depth_list_str),
    ]
    params.extend(trainer_solution_defaults(gametype).items())
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
        spot_category: Preflop leak bucket mapped via
            _PREFLOP_SPOT_TO_FH_ACTIONS (open_raise, facing_3bet, ...).
        street: "preflop" | "flop" | "turn" | "river".
        effective_bb: Stack depth in bb; will be snapped to nearest
            GTOW-supported depth.
        pot_type: Retained for call compatibility; postflop shortcuts are no
            longer emitted because they do not identify an action line.
        gametype: GTOW gametype code, default MTTGeneral.

    Returns:
        Fully-formed URL string.

    Raises:
        SpotNotSupportedError: if the preflop category cannot be mapped or a
            postflop caller has not supplied an exact source-hand custom spot.
        ValueError: if street is not one of the four valid streets.

    Postflop links belong in ``gtow_custom_url.build_custom_spot_url``.
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
        raise SpotNotSupportedError(
            "postflop action-line drills require an exact custom_spot source hand")

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
    params.extend(trainer_solution_defaults(gametype).items())

    # Trainer UI flags (skip solution_type + dialogs which bookend the URL).
    for k, v in _TRAINER_UI_DEFAULTS.items():
        if k in ("solution_type", "dialogs"):
            continue
        params.append((k, v))

    params.append(("fh_start_spot", street))
    params.append(("fh_actions", fh_actions))
    params.append(("dialogs", _TRAINER_UI_DEFAULTS["dialogs"]))

    return f"{_BASE_URL}?{urlencode(params, quote_via=quote)}"


# ── shared spot→drill-URL policy ─────────────────────────────────────────────
# One place for the "classified spot → Trainer deep link" rules; the
# leaderboard rows and the live-flow queue items previously each had a copy.
PREFLOP_CATS = {"RFI", "vsOpen", "vsRaiseCall", "vsSqueeze", "vs3bet", "vs4bet"}


def drill_url_for_spot(category: str, *, hero_pos: str | None = None,
                       hero_cat: str | None = None, villain_cat: str | None = None,
                       ip_oop: str | None = None, pot_type: str | None = None,
                       depths: list[int] | None = None) -> str | None:
    """Trainer deep link for a classified spot, or None when unsupported.

    Standard preflop enums pin the classified hero/opponent scope. Postflop and
    cold-raise categories require an exact source-hand custom spot.
    """
    depths = list(MTT_DEPTHS) if depths is None else depths
    opp = CAT_POSITIONS.get(villain_cat) if villain_cat in CAT_POSITIONS else None
    try:
        if category in PREFLOP_CATS:
            hero = ([hero_pos] if category in ("RFI", "vsOpen") and hero_pos
                    else CAT_POSITIONS.get(hero_cat, []))
            return build_drill_url(
                category, "preflop", 20, hero, opponent_positions=opp,
                rel_position=ip_oop, depths=depths)
    except (SpotNotSupportedError, ValueError):
        return None
    return None
