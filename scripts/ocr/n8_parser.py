"""Main N8 replay screenshot parser.

Orchestrates region detection, table parsing, panel parsing, and hand
assembly into the JSON format expected by analyze_hand_full().
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

from .region_detector import detect_regions
from .table_parser import parse_table
from .panel_parser import parse_panel, normalize_position
from .button_detector import hero_position_from_button

# Load config for confidence weights/threshold
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "n8_default.json"
with open(_CONFIG_PATH) as f:
    _CONFIG = json.load(f)

_CONF_WEIGHTS = _CONFIG["confidence_weights"]
_CONF_THRESHOLD = _CONFIG["confidence_threshold"]

# Position orders by table size (must match analyze_hand.py)
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


def _resolve_hero_board_conflict(
    board_cards: list[str],
    hero_cards: list[str],
    *,
    hero_details: list[dict] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve duplicate hero/board cards using classifier alternates."""
    if not (board_cards and hero_cards):
        return board_cards, hero_cards

    board_set = set(board_cards)
    if not (set(hero_cards) & board_set) and len(set(hero_cards)) == len(hero_cards):
        return board_cards, hero_cards

    if hero_details is None:
        log.warning(
            "Duplicate cards detected without top2: board=%s hero=%s",
            board_cards,
            hero_cards,
        )
        return board_cards, []

    fixed = list(hero_cards)
    for idx, detail in enumerate(hero_details[:len(fixed)]):
        current = fixed[idx]
        if current not in board_set and fixed.count(current) == 1:
            continue

        candidates: list[tuple[str, float]] = []
        for rank, rank_prob in detail.get("rank_top2", [])[:2]:
            for suit, suit_prob in detail.get("suit_top2", [])[:2]:
                if rank and suit:
                    candidates.append((f"{rank}{suit}", rank_prob * suit_prob))
        candidates.sort(key=lambda item: item[1], reverse=True)

        for card, _ in candidates:
            others = [fixed[j] for j in range(len(fixed)) if j != idx]
            if card not in board_set and card not in others:
                fixed[idx] = card
                break
        else:
            log.warning(
                "Duplicate cards unresolved: board=%s hero=%s detail=%s",
                board_cards,
                hero_cards,
                detail,
            )
            return board_cards, []

    return board_cards, fixed


def parse_n8_screenshot(image_bytes: bytes) -> dict:
    """Parse N8 replay screenshot into hand JSON.

    Args:
        image_bytes: Raw image file bytes (JPEG/PNG)

    Returns:
        {
            "hand": dict|None,     # Hand JSON for analyze_hand_full()
            "hints": dict|None,    # Partial data for Gemini fallback
            "confidence": float    # 0.0 to 1.0
        }
    """
    # Decode image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "hand": None,
            "hints": None,
            "confidence": 0.0,
            "diagnostics": _build_diagnostics({}, []),
        }

    # Step 1: detect regions
    regions = detect_regions(image)
    if regions is None:
        return {
            "hand": None,
            "hints": None,
            "confidence": 0.0,
            "diagnostics": _build_diagnostics({}, []),
        }

    # Step 2: parse table
    table_result = parse_table(regions["table"])

    # Step 3: parse panel
    panel_result = parse_panel(regions["panel"])
    columns = panel_result.get("columns", [])

    # Step 4: assemble hand JSON
    hand, confidence_parts, diagnostics = _assemble_hand(table_result, columns)

    # Step 5: compute confidence
    confidence = _compute_confidence(confidence_parts)

    # Step 6: build hints for low confidence
    hints = None
    if confidence < _CONF_THRESHOLD or hand is None:
        hints = _build_hints(table_result, columns, hand)

    return {
        "hand": hand if confidence > 0.3 else None,
        "hints": hints,
        "confidence": confidence,
        "card_confidence": confidence_parts.get("card_confidence", 0.0),
        "confidence_parts": confidence_parts,
        "diagnostics": diagnostics,
    }


def _filter_action_entries(entries: list[dict]) -> list[dict]:
    """Filter preflop entries to keep only real action entries.

    Removes false hero detections (e.g., avatar markers that are yellow
    but don't contain action text).

    When multiple entries are detected as hero, only the one with a
    non-fold action is kept as hero; the others are reclassified as
    opponents (caused by yellow background bleeding into adjacent rows).
    """
    _ACTION_WORDS = {"fold", "call", "raise", "check", "bet", "all"}
    result = []
    for e in entries:
        if e["type"] == "hero":
            ocr = (e.get("action") or "").lower()
            if any(a in ocr for a in _ACTION_WORDS):
                result.append(e)
            # Skip hero entries without clear action text (avatar markers)
        else:
            result.append(e)

    # Disambiguate false hero detections: when yellow background bleeds
    # into adjacent rows, fold entries near the real hero get marked as
    # hero too.  Reclassify hero-Fold entries as opponents when there is
    # at least one hero with a non-fold action (the real hero).
    hero_indices = [i for i, e in enumerate(result) if e["type"] == "hero"]
    if len(hero_indices) > 1:
        has_non_fold_hero = any(
            (result[idx].get("action") or "").lower() in ("raise", "call", "bet", "all-in")
            for idx in hero_indices
        )
        if has_non_fold_hero:
            for idx in hero_indices:
                action = (result[idx].get("action") or "").lower()
                if action == "fold":
                    result[idx] = dict(result[idx], type="opponent")

    return result


def _estimate_table_size(action_entries: list[dict]) -> tuple[int, bool]:
    """Estimate table size from preflop action entries.

    In N8 PreFlop, entries appear in position order. The first round
    has exactly one entry per player. After a raise, some players may
    act again (re-actions).

    Strategy: find where the first round ends by looking for re-actions.
    A re-action happens when a player who already acted earlier in the
    round acts again (detected via duplicate player names or position
    badges).
    """
    n = len(action_entries)
    if n <= 2:
        return max(n, 2), False

    # Check for re-actions by looking for duplicate player names.
    # In the first round each player appears once.  If a name repeats,
    # the second occurrence is a re-action.  Uses fuzzy matching because
    # OCR may read the same name slightly differently in each row.
    seen_names: list[str] = []
    re_action_start = n  # index where re-actions begin
    for i, e in enumerate(action_entries):
        name = (e.get("player_name") or "").strip()
        if not name:
            continue
        # Check against all previously seen names using fuzzy match
        for prev in seen_names:
            if _fuzzy_name_match(name, prev):
                # This player already appeared — re-action detected
                re_action_start = min(re_action_start, i)
                break
        if re_action_start < n:
            break
        seen_names.append(name)

    # Also detect re-actions when hero acted twice (two hero entries)
    if re_action_start == n:
        hero_indices = [i for i, e in enumerate(action_entries) if e["type"] == "hero"]
        if len(hero_indices) >= 2:
            re_action_start = min(re_action_start, hero_indices[1])

    table_size = re_action_start if re_action_start < n else n
    used_reaction_signal = re_action_start < n

    if table_size > 9:
        return 9, used_reaction_signal
    return max(table_size, 2), used_reaction_signal


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching.

    Strips common OCR noise: spaces, underscores, dots, colons,
    trailing punctuation, quotes.  Case-insensitive.
    """
    import re
    s = name.lower()
    s = re.sub(r"[_. :;,'\"\-\[\](){}!]", "", s)
    return s


def _fuzzy_name_match(name1: str, name2: str) -> bool:
    """Fuzzy match two player names (case-insensitive, partial match).

    OCR may truncate or misread parts of names, so we use multiple
    strategies: exact, substring, common prefix, and simple edit
    distance for short names.
    """
    if not name1 or not name2:
        return False
    a = _normalize_name(name1)
    b = _normalize_name(name2)
    if not a or not b:
        return False
    # Exact match after normalization
    if a == b:
        return True
    # One contains the other
    if a in b or b in a:
        return True
    # Long common prefix (at least 5 chars or 70% of shorter name)
    min_len = min(len(a), len(b))
    prefix_len = 0
    for i in range(min_len):
        if a[i] == b[i]:
            prefix_len += 1
        else:
            break
    if prefix_len >= 5 or (min_len >= 3 and prefix_len >= min_len * 0.7):
        return True
    # Simple edit distance for names of similar length.
    # Allow up to 2 edits for names >= 6 chars, 1 edit for shorter.
    if abs(len(a) - len(b)) <= 2 and min_len >= 4:
        max_edits = 2 if min_len >= 6 else 1
        # Quick Levenshtein via two-row DP
        prev = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            curr = [i] + [0] * len(b)
            for j in range(1, len(b) + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        if prev[len(b)] <= max_edits:
            return True
    return False


def _compute_effective_bb(
    columns: list[dict],
    hero_stack_displayed: float | None,
    hero_position: str | None,
    all_stacks: list[float] | None,
    named_stacks: list[dict] | None = None,
) -> float | None:
    """Compute effective_bb from active players' starting stacks.

    In N8 replays, displayed stacks = starting - permanently_invested.
    The pot is shown separately in the table centre, so this equation
    holds for BOTH the winner and the loser(s).

    Uses player name matching between table stacks and panel entries
    to identify the correct opponent stack.  Falls back to heuristic
    stack selection when name matching fails.
    """
    if hero_stack_displayed is None:
        return None, None

    # ---- Locate columns ----
    blinds_col = None
    preflop_col = None
    street_cols = []

    for col in columns:
        name_lower = col["name"].lower()
        if "blind" in name_lower:
            blinds_col = col
        elif "pre" in name_lower:
            preflop_col = col
        elif name_lower in ("flop", "turn", "river"):
            street_cols.append(col)

    if preflop_col is None and street_cols:
        first_street = street_cols[0]
        first_entries = first_street.get("entries", [])
        if (first_street["name"].lower() == "flop"
                and len(first_entries) >= 5):
            preflop_col = first_street
            street_cols = street_cols[1:]

    # ---- Determine hero blind ----
    hero_blind = 0.0

    if blinds_col:
        for entry in blinds_col.get("entries", []):
            action_text = (entry.get("action") or "").lower()
            size = entry.get("size")
            if entry.get("type") == "hero":
                if "sb" in action_text or size == 0.5:
                    hero_blind = 0.5
                elif "bb" in action_text or size == 1.0:
                    hero_blind = 1.0

    if hero_blind == 0.0 and hero_position:
        if hero_position == "BB":
            hero_blind = 1.0
        elif hero_position == "SB":
            hero_blind = 0.5

    # ---- Collect pot headers ----
    # Pot headers = pot at START of each street (before that street's action).
    # preflop_pot = antes + blinds
    # flop_pot = pot after all preflop action
    # turn_pot = pot after all flop action
    # river_pot = pot after all turn action
    pot_by_street = {}
    for col in columns:
        if col.get("pot") is not None:
            pot_by_street[col["name"].lower()] = col["pot"]

    preflop_pot = pot_by_street.get("pre-flop") or pot_by_street.get("preflop")
    flop_pot = pot_by_street.get("flop")
    turn_pot = pot_by_street.get("turn")
    river_pot = pot_by_street.get("river")

    # ---- Walk preflop entries ----
    hero_perm = 0.0
    opp_perm = 0.0
    opp_entered = False
    first_hero_preflop_action = None

    # Track ALL opponents who enter preflop.  The one who stays longest
    # into postflop is the one whose starting stack determines eff_bb.
    # After preflop, we'll use pot headers to determine the continuing
    # opponent's preflop investment.
    n_opp_preflop = 0   # count of opponents who enter preflop
    hero_preflop_total = 0.0

    # Collect opponent names from panel entries (for name matching)
    opp_names_entered = []  # names of opponents who entered the pot

    if preflop_col:
        entries = preflop_col.get("entries", [])
        current_bet = 1.0  # BB level

        for entry in entries:
            action = (entry.get("action") or "").lower()
            size = entry.get("size") or 0.0
            is_hero = entry.get("type") == "hero"

            if action == "fold":
                continue

            if is_hero:
                if first_hero_preflop_action is None:
                    first_hero_preflop_action = action
                if action in ("raise", "all-in"):
                    hero_preflop_total = size
                    current_bet = size
                elif action == "call":
                    hero_preflop_total = hero_blind + size
                    if hero_preflop_total < current_bet:
                        hero_preflop_total = current_bet
            else:
                if action in ("call", "raise", "bet", "all-in"):
                    opp_entered = True
                    n_opp_preflop += 1
                    opp_name = entry.get("player_name")
                    if opp_name:
                        opp_names_entered.append(opp_name)
                    if action in ("raise", "all-in"):
                        current_bet = size

        if hero_preflop_total == 0.0 and hero_blind > 0:
            hero_preflop_total = hero_blind

    # Use pot headers to compute the continuing opponent's preflop total.
    # flop_pot = preflop_pot + hero_new + sum(all_opp_new)
    # For the continuing opponent: opp_pre_total = current_bet at end of
    # preflop (they called to this level).  This handles multi-way correctly
    # because each active caller matched current_bet.
    opp_preflop_total = current_bet if opp_entered else 0.0

    # Validate with pot headers if available
    if flop_pot is not None and preflop_pot is not None and opp_entered:
        hero_new = hero_preflop_total - hero_blind
        total_opp_new = flop_pot - preflop_pot - hero_new
        if n_opp_preflop == 1:
            # Heads-up: opp's new chips = total_opp_new
            opp_blind_inferred = opp_preflop_total - total_opp_new
            # Validate: blind should be 0, 0.5, or 1.0
            if opp_blind_inferred < -0.3:
                # Our opp_preflop_total is too low; adjust
                opp_preflop_total = total_opp_new
        # For multi-way: each caller put in current_bet total.
        # The pot confirms this indirectly.

    hero_perm += hero_preflop_total
    opp_perm += opp_preflop_total

    # ---- Walk postflop streets ----
    # Use pot-header progression where available to compute per-street
    # contributions.  For the last street (no next header), fall back
    # to entry-based computation.
    #
    # Calls are ADDITIVE: "call X" = add X more to street total.
    # Raises are REPLACE: "raise-to X" replaces the running total.

    # Build ordered pot sequence for delta computation:
    # [flop_pot, turn_pot, river_pot]
    pot_sequence = []
    for col in street_cols:
        nm = col["name"].lower()
        p = pot_by_street.get(nm)
        pot_sequence.append(p)

    # Also collect opponent names from postflop entries
    opp_names_postflop = []

    for idx, col in enumerate(street_cols):
        entries = col.get("entries", [])
        if not entries:
            continue

        hero_street = 0.0
        opp_street = 0.0

        for entry in entries:
            action = (entry.get("action") or "").lower()
            size = entry.get("size") or 0.0
            is_hero = entry.get("type") == "hero"

            # Track opponent names in postflop
            if not is_hero:
                pn = entry.get("player_name")
                if pn and pn not in opp_names_postflop:
                    opp_names_postflop.append(pn)

            if action in ("fold", "check"):
                continue

            if is_hero:
                if action == "bet":
                    hero_street += size
                elif action in ("raise", "all-in"):
                    hero_street = size  # raise-to replaces
                elif action == "call":
                    hero_street += size  # call is additive
            else:
                if action == "bet":
                    opp_street += size
                elif action in ("raise", "all-in"):
                    opp_street = size  # raise-to replaces
                elif action == "call":
                    opp_street += size  # call is additive

        last_entry = entries[-1]
        last_action = (last_entry.get("action") or "").lower()
        last_is_hero = last_entry.get("type") == "hero"

        # Try to use pot delta for this street if next header exists
        this_pot = pot_sequence[idx] if idx < len(pot_sequence) else None
        next_pot = pot_sequence[idx + 1] if idx + 1 < len(pot_sequence) else None

        if this_pot is not None and next_pot is not None:
            # Pot delta = total chips added this street by all players
            delta = next_pot - this_pot
            # Hero's matched contribution for this street
            hero_matched = min(hero_street, opp_street) if hero_street > 0 and opp_street > 0 else hero_street
            if last_action == "fold":
                if last_is_hero:
                    hero_matched = hero_street
                else:
                    hero_matched = min(hero_street, opp_street)
            elif last_action == "call" and not last_is_hero:
                hero_matched = min(hero_street, opp_street)
            else:
                hero_matched = hero_street

            # Derive opp's matched contribution from pot delta
            opp_matched = delta - hero_matched
            if opp_matched < 0:
                opp_matched = 0.0
            hero_perm += hero_matched
            opp_perm += opp_matched
        else:
            # Last street or no pot headers — use entry-based logic
            if last_action == "fold":
                if last_is_hero:
                    hero_perm += hero_street
                    opp_perm += min(opp_street, hero_street)
                else:
                    opp_perm += opp_street
                    hero_perm += min(hero_street, opp_street)
            elif last_action == "call":
                if last_is_hero:
                    hero_perm += hero_street
                    opp_perm += opp_street
                else:
                    # Opp's Call size sometimes can't be read when an
                    # "All-In" badge overlaps the size sticker. Without
                    # a size we'd count opp_street=0 and undercount hero
                    # via min(hero, opp). Per the call definition, a Call
                    # covers the outstanding bet, so assume opp matched
                    # hero when the call entry has no explicit size.
                    # Regression: H2852 river — hero jammed 11, OCR
                    # missed opp's call size, hero_perm dropped 11bb and
                    # effective_bb collapsed from 31 to 20.
                    last_entry_size = last_entry.get("size")
                    if last_entry_size is None and opp_street < hero_street:
                        opp_street = hero_street
                    opp_perm += opp_street
                    hero_perm += min(hero_street, opp_street)
            else:
                hero_perm += hero_street
                opp_perm += opp_street

    # ---- Detect opponent all-in ----
    # Case 1: Partial call — opponent's total street commitment < hero's.
    # Case 2: Opponent raises/bets and hero folds — if a non-hero stack
    #   matches the uncalled portion, opponent went all-in.
    opp_went_allin = False
    opp_allin_display = None  # display stack when opp went all-in

    for col in street_cols:
        entries = col.get("entries", [])
        if len(entries) < 2:
            continue
        last_entry = entries[-1]
        last_action = (last_entry.get("action") or "").lower()
        last_is_hero = last_entry.get("type") == "hero"

        # Compute total street commitment for each side
        hero_total = 0.0
        opp_total = 0.0
        for e in entries:
            ea = (e.get("action") or "").lower()
            es = e.get("size") or 0.0
            eh = e.get("type") == "hero"
            if ea in ("fold", "check"):
                continue
            if eh:
                if ea in ("raise", "all-in"):
                    hero_total = es
                else:
                    hero_total += es
            else:
                if ea in ("raise", "all-in"):
                    opp_total = es
                else:
                    opp_total += es

        if last_action == "call" and not last_is_hero:
            # Case 1: opp called for less (partial call)
            if opp_total < hero_total - 0.5:
                opp_went_allin = True

        elif last_action == "fold" and last_is_hero and opp_total > hero_total:
            # Case 2: opp raised/bet and hero folded.
            # Check if opp went all-in by looking for a non-hero stack
            # that matches the uncalled portion.
            uncalled = opp_total - hero_total
            if uncalled > 0 and all_stacks:
                non_hero = [s for s in (all_stacks or [])
                            if s != hero_stack_displayed]
                for s in non_hero:
                    if abs(s - uncalled) < 0.5:
                        opp_went_allin = True
                        opp_allin_display = s
                        break

    # ---- Compute starting stacks ----
    hero_starting = hero_stack_displayed + hero_perm

    hero_start_rounded = round(hero_starting, 1) if hero_starting >= 1.0 else None
    if not opp_entered:
        return (hero_start_rounded, hero_start_rounded)

    # ---- Determine opponent starting stack ----
    if opp_went_allin:
        if opp_allin_display is not None:
            # Opponent went all-in and we know their display (uncalled portion)
            opp_starting = opp_allin_display + opp_perm
        else:
            # Opponent went all-in: starting = total investment (display ≈ 0)
            opp_starting = opp_perm
    else:
        # Find opponent's displayed stack using name matching.
        # Prefer postflop names (the player who stayed), then preflop.
        active_opp_names = opp_names_postflop or opp_names_entered
        best_stack = None

        if active_opp_names and named_stacks:
            # Try to match the active opponent's name to a table stack
            for opp_name in active_opp_names:
                for ns in named_stacks:
                    if ns.get("name") and _fuzzy_name_match(opp_name, ns["name"]):
                        candidate = ns["stack"]
                        # Sanity: opp display + investment shouldn't wildly
                        # exceed hero starting (allow some tolerance)
                        if candidate + opp_perm <= hero_starting * 2.5:
                            best_stack = candidate
                            break
                if best_stack is not None:
                    break

        # Fallback: heuristic stack selection when name matching fails
        if best_stack is None:
            non_hero_stacks = list(all_stacks) if all_stacks else []
            if hero_stack_displayed is not None and hero_stack_displayed in non_hero_stacks:
                non_hero_stacks.remove(hero_stack_displayed)

            if non_hero_stacks:
                candidates = [
                    s for s in non_hero_stacks
                    if s + opp_perm <= hero_starting + 1.0
                ]
                if candidates:
                    best_stack = max(candidates)
                elif non_hero_stacks:
                    best_stack = min(non_hero_stacks)

        opp_starting = (best_stack + opp_perm) if best_stack is not None else hero_starting

    all_starting = [hero_starting, opp_starting]
    effective_bb = round(min(all_starting), 1)

    if effective_bb < 1.0:
        if all_stacks:
            return round(min(all_stacks), 1), round(hero_starting, 1)
        return None, round(hero_starting, 1) if hero_starting >= 1.0 else None

    return effective_bb, round(hero_starting, 1)


def _build_diagnostics(
    table_result: dict,
    columns: list[dict],
    *,
    preflop_col: dict | None = None,
    action_entries: list[dict] | None = None,
    players_at_table_raw: int | None = None,
    players_at_table_final: int | None = None,
    estimate_used_reaction_signal: bool = False,
) -> dict:
    street_entries_count = {}
    for col in columns:
        name = (col.get("street") or col.get("name") or "").lower()
        if name in ("flop", "turn", "river"):
            street_entries_count[name] = len(col.get("entries", []))

    return {
        "players_at_table_raw": players_at_table_raw,
        "players_at_table_final": players_at_table_final,
        "estimate_used_reaction_signal": estimate_used_reaction_signal,
        "dealer_button_seat": table_result.get("dealer_button_seat"),
        "dealer_button_conf": float(table_result.get("dealer_button_conf") or 0.0),
        "preflop_entries_count": len(action_entries or []),
        "preflop_entries_pre_collapse_count": (
            preflop_col.get("entries_pre_collapse_count") if preflop_col else None
        ),
        "street_entries_count": street_entries_count,
    }


def _assemble_hand(table_result: dict, columns: list[dict]) -> tuple[dict | None, dict, dict]:
    """Assemble hand JSON from parsed table and panel data.

    Uses position-order-based inference: in N8 PreFlop column, entries
    appear in strict position order (UTG first, BB last). Combined with
    entry count, we determine table size and assign positions.

    Returns:
        (hand_dict or None, confidence_parts dict, diagnostics dict)
    """
    conf_parts = {
        "pot_consistency": 0.0,
        "player_tracking": 0.0,
        "ocr_confidence": 0.5,
        "card_confidence": 0.0,
    }

    board_cards = table_result.get("board_cards", [])
    hero_cards = table_result.get("hero_cards", [])
    table_color = table_result.get("table_color", "unknown")
    diagnostics = _build_diagnostics(table_result, columns)

    board_cards, hero_cards = _resolve_hero_board_conflict(
        board_cards,
        hero_cards,
        hero_details=table_result.get("hero_card_details"),
    )

    # Card confidence — use actual hero detection quality from table parser.
    # Don't boost based on board legibility: CardCNN runs hero and board
    # crops independently, so board cards being clear says nothing about
    # hero rank reliability. Regression: H2822 — hero 8s/8d classified at
    # 0.611, +0.1 board boost pushed it to 0.711 (just above the 0.70
    # MIN_CARD_CONF gate), letting the wrong "9s8d" prediction ship.
    hero_card_conf = table_result.get("hero_card_conf", 0.0)
    if hero_cards and len(hero_cards) == 2:
        conf_parts["card_confidence"] = hero_card_conf

    # Find the PreFlop and Blinds columns
    blinds_col = None
    preflop_col = None
    street_cols = []  # Flop, Turn, River

    for col in columns:
        name_lower = col["name"].lower()
        if "blind" in name_lower:
            blinds_col = col
        elif "pre" in name_lower:
            preflop_col = col
        elif name_lower in ("flop", "turn", "river"):
            street_cols.append(col)

    # Fixup: if PreFlop wasn't found but the first Flop column has
    # many entries (5+), it's likely a misidentified PreFlop column
    # (OCR misread "PreFlop" as "Flop").
    if preflop_col is None and street_cols:
        first_street = street_cols[0]
        first_entries = first_street.get("entries", [])
        if (first_street["name"].lower() == "flop"
                and len(first_entries) >= 5):
            preflop_col = first_street
            street_cols = street_cols[1:]

    if preflop_col is None:
        return None, conf_parts, diagnostics

    preflop_entries = preflop_col.get("entries", [])

    # Filter out false hero entries (avatar markers without action text)
    action_entries = _filter_action_entries(preflop_entries)
    diagnostics = _build_diagnostics(
        table_result,
        columns,
        preflop_col=preflop_col,
        action_entries=action_entries,
    )

    if not action_entries:
        return None, conf_parts, diagnostics

    # Determine table size from entry count
    players_at_table_raw, estimate_used_reaction_signal = _estimate_table_size(action_entries)
    players_at_table = players_at_table_raw

    # Cross-column refinement: detect re-actions where the preflop name
    # duplicate can't be seen because one entry has no name.
    # Pattern: unnamed OPPONENT raise earlier + named call/fold at end
    # where the named entry matches a postflop opponent.  The unnamed
    # raiser IS the named caller (e.g., player opens, gets 3-bet, calls).
    if len(action_entries) == players_at_table and players_at_table >= 3:
        last_e = action_entries[-1]
        last_action = (last_e.get("action") or "").lower()
        last_name = (last_e.get("player_name") or "").strip()
        if last_action in ("call", "fold") and last_name and last_e.get("type") != "hero":
            # Check if this name matches an earlier unnamed OPPONENT raise
            has_unnamed_opp_raise = False
            for j, earlier in enumerate(action_entries[:-1]):
                if earlier.get("type") == "hero":
                    continue
                earlier_action = (earlier.get("action") or "").lower()
                earlier_name = (earlier.get("player_name") or "").strip()
                if earlier_action == "raise" and not earlier_name:
                    has_unnamed_opp_raise = True
                    break
            if has_unnamed_opp_raise:
                # Verify: this player appears in postflop (confirming
                # they entered the pot and are the same as the raiser)
                postflop_opp_names = []
                for col in street_cols:
                    for e in col.get("entries", []):
                        if e.get("type") != "hero":
                            pn = (e.get("player_name") or "").strip()
                            if pn:
                                postflop_opp_names.append(pn)
                in_postflop = any(
                    _fuzzy_name_match(last_name, pn) for pn in postflop_opp_names
                )
                if in_postflop:
                    players_at_table -= 1

    players_at_table = min(max(players_at_table, 2), 9)
    diagnostics = _build_diagnostics(
        table_result,
        columns,
        preflop_col=preflop_col,
        action_entries=action_entries,
        players_at_table_raw=players_at_table_raw,
        players_at_table_final=players_at_table,
        estimate_used_reaction_signal=estimate_used_reaction_signal,
    )
    pos_order = POSITION_ORDERS.get(players_at_table, POSITION_ORDERS[8])

    # Assign positions by entry order (first entry = first position, etc.)
    # Only the FIRST hero entry determines hero_position; later hero
    # entries are re-actions (hero acting again after being raised).
    hero_position = None
    hero_index = None
    for i, entry in enumerate(action_entries[:players_at_table]):
        if i < len(pos_order):
            entry["position"] = pos_order[i]
            if entry["type"] == "hero" and hero_position is None:
                hero_position = pos_order[i]
                hero_index = i

    # Mark re-action entries (beyond first round)
    for i, entry in enumerate(action_entries[players_at_table:], players_at_table):
        entry["_is_reaction"] = True

    # If hero was assigned to a FOLD position (false hero marker), look for
    # the actual hero entry — might be beyond [:players_at_table] due to
    # duplicate position entries pushing BB's check out of range.
    if hero_position and hero_index is not None:
        hero_entry = action_entries[hero_index]
        hero_action = (hero_entry.get("action") or "").lower()
        if hero_action == "fold":
            # Hero can't fold preflop and still appear in postflop — find
            # the real hero entry (non-fold, with hero marker)
            for j, entry in enumerate(action_entries):
                if j == hero_index:
                    continue
                if entry["type"] != "hero":
                    continue
                act = (entry.get("action") or "").lower()
                if act != "fold":
                    # This is the real hero — assign to last position (BB)
                    # since it's typically a BB check after limps
                    if j >= players_at_table:
                        hero_position = pos_order[-1] if pos_order else "BB"
                    elif j < len(pos_order):
                        hero_position = pos_order[j]
                    hero_index = j
                    break

    # Check blinds column for hero position override
    if blinds_col:
        blinds_entries = blinds_col.get("entries", [])
        for entry in blinds_entries:
            if entry["type"] == "hero":
                action_text = (entry.get("action") or "").lower()
                size = entry.get("size")
                if "sb" in action_text or size == 0.5:
                    hero_position = "SB"
                elif "bb" in action_text or size == 1.0:
                    hero_position = "BB"

    dealer_button = table_result.get("dealer_button")
    if dealer_button and players_at_table == 8:
        button_seat_idx, button_conf = dealer_button
        button_hero_position = hero_position_from_button(
            button_seat_idx,
            table_size=players_at_table,
        )
        if button_conf >= 0.9 and button_hero_position:
            hero_position = button_hero_position

    if not hero_position:
        return None, conf_parts, diagnostics

    # Build preflop_actions string using assigned positions
    preflop_actions = _build_preflop_actions_from_order(
        action_entries, pos_order, hero_position, players_at_table
    )

    if not preflop_actions:
        return None, conf_parts, diagnostics

    # Bail out when any raise/bet entry came back with no size — the
    # panel cell's "Raise N BB" text didn't OCR cleanly. We used to
    # silently substitute a min-raise placeholder ("R2") in
    # _action_to_code, which corrupts pot accounting and makes the
    # solver-side action mapping route the next bet to RAI. Better to
    # surface this as a structural failure so gemini_session falls back
    # to a full Gemini parse. Regression: H2823 — CO 3-bet "Raise 7 BB"
    # came back size=null, default R2 ate the rest of the analysis
    # (turn/river dropped because flop_actions resolved to X-RAI-C and
    # the API rejected anything beyond it).
    missing_raise_sizes = sum(
        1 for e in action_entries
        if (e.get("action") or "").lower() in ("raise", "bet")
        and e.get("size") is None
    )
    if missing_raise_sizes:
        conf_parts["ocr_confidence"] = 0.0
        return None, conf_parts, diagnostics

    # Build hero_hand — sort by rank (higher first), standard poker notation
    _RANK_ORDER = "23456789TJQKA"
    hero_hand = ""
    if hero_cards and len(hero_cards) == 2:
        c1, c2 = hero_cards[0], hero_cards[1]
        r1 = c1[0] if len(c1) >= 2 else ""
        r2 = c2[0] if len(c2) >= 2 else ""
        idx1 = _RANK_ORDER.index(r1) if r1 in _RANK_ORDER else -1
        idx2 = _RANK_ORDER.index(r2) if r2 in _RANK_ORDER else -1
        if idx1 >= idx2:
            hero_hand = c1 + c2
        else:
            hero_hand = c2 + c1

    if not hero_position:
        return None, conf_parts, diagnostics

    # Determine active players after preflop (didn't fold)
    active_positions = []
    for i, entry in enumerate(action_entries[:players_at_table]):
        pos = entry.get("position", pos_order[i] if i < len(pos_order) else None)
        action = (entry.get("action") or "").lower()
        if action != "fold" and pos:
            active_positions.append(pos)
    # Also check re-action entries (calls after 3bet etc.)
    for entry in action_entries[players_at_table:]:
        action = (entry.get("action") or "").lower()
        pos = entry.get("position")
        if action == "fold" and pos and pos in active_positions:
            active_positions.remove(pos)

    # Reconcile postflop entry positions with preflop's index-assigned ones.
    # Regression for H2810 (7-max): N8 used badges {UTG, UTG+1, MP, CO, BTN,
    # SB, BB} but our 7-max pos_order is [UTG, LJ, HJ, CO, BTN, SB, BB], so
    # the third entry's badge alias landed on LJ while the index assignment
    # promoted it to HJ. The flop column kept the LJ badge alias, but the
    # preflop_actions string treated LJ as folded — so _fix_folded_players
    # later stripped the opponent's flop bet/fold entries entirely, leaving
    # only the hero's two actions and producing nonsense GTO advice.
    #
    # Build a player_name → canonical-position map from the (already
    # reassigned) preflop entries and apply it to every postflop entry that
    # carries the same name. Names come from the panel's avatar text and
    # can drift slightly across columns, so use the existing fuzzy matcher.
    name_to_pos: list[tuple[str, str]] = []
    for entry in action_entries[:players_at_table]:
        if entry.get("type") == "hero":
            continue
        nm = (entry.get("player_name") or "").strip()
        pos = entry.get("position")
        if nm and pos:
            name_to_pos.append((nm, pos))
    if name_to_pos:
        for col in street_cols:
            for sub_entry in col.get("entries", []):
                if sub_entry.get("type") == "hero":
                    continue
                sub_name = (sub_entry.get("player_name") or "").strip()
                if not sub_name:
                    continue
                for ref_name, ref_pos in name_to_pos:
                    if _fuzzy_name_match(sub_name, ref_name):
                        sub_entry["position"] = ref_pos
                        break

    # Build streets with position context
    streets = _build_streets(street_cols, board_cards, pos_order,
                             hero_position, active_positions)

    # Effective BB: compute from hero's displayed stack + total investment.
    # Displayed stacks are end-of-hand remaining, so:
    #   hero_starting = hero_displayed + hero_invested  (conservative)
    # This gives hero's starting stack, which bounds effective_bb.
    hero_stack = table_result.get("hero_stack")
    stacks = table_result.get("player_stacks", [])
    named_stacks = table_result.get("named_stacks", [])
    effective_bb, hero_starting_stack = _compute_effective_bb(
        columns, hero_stack, hero_position, stacks, named_stacks,
    )

    hand = {
        "gametype": "MTTGeneral",
        "hero_hand": hero_hand,
        "hero_position": hero_position,
        "players_at_table": players_at_table,
        "preflop_actions": preflop_actions,
    }

    # Sanity check effective_bb — if unreasonable, leave it out for Gemini
    if effective_bb is not None:
        # Must be at least as large as the biggest preflop raise
        max_preflop_raise = 0
        for part in preflop_actions.split("-"):
            if part.startswith("R"):
                try:
                    max_preflop_raise = max(max_preflop_raise, float(part[1:]))
                except ValueError:
                    pass
            elif part.startswith("AI"):
                try:
                    max_preflop_raise = max(max_preflop_raise, float(part[2:]))
                except ValueError:
                    pass
        if effective_bb < max_preflop_raise:
            effective_bb = None  # unreasonable, let Gemini compute
        # Must be reasonable relative to hero's displayed stack
        elif hero_stack and effective_bb > hero_stack * 5:
            effective_bb = None  # likely stack matching error

    if effective_bb is not None:
        hand["effective_bb"] = effective_bb

    # Compute hero's starting stack from named_stacks (more reliable than table hero detection)
    # The table parser may misidentify the bottom-center player as hero.
    # Use the panel's hero name + named_stacks match for the correct displayed stack.
    if hero_starting_stack is not None:
        hero_name_from_panel = None
        if hero_index is not None and hero_index < len(action_entries):
            hero_name_from_panel = action_entries[hero_index].get("player_name")
        if hero_name_from_panel and named_stacks:
            for ns in named_stacks:
                if ns.get("name") and _fuzzy_name_match(hero_name_from_panel, ns["name"]):
                    # Recompute hero starting from correct displayed stack
                    hero_display = ns["stack"]
                    # hero_perm was computed in _compute_effective_bb; approximate from
                    # hero_starting_stack - hero_stack (table hero display)
                    # Instead, use: hero_starting = hero_display + total_invested
                    # total_invested = hero_starting_stack - (hero_stack or 0)
                    hero_invested = hero_starting_stack - (hero_stack or 0)
                    corrected = round(hero_display + hero_invested, 1)
                    if corrected > 0 and corrected != hero_starting_stack:
                        hero_starting_stack = corrected
                    break
        hand["hero_starting_stack"] = hero_starting_stack

    if streets:
        hand["streets"] = streets

    # Only include player_stacks if count matches players_at_table.
    # OCR stack detection is unreliable (includes pot values, wrong order).
    # Mismatched stacks cause position mapping errors downstream.
    if stacks and len(stacks) == players_at_table:
        hand["player_stacks"] = stacks

    # Final Table detection (temporarily disabled — purple-felt heuristic
    # was over-triggering ICM analysis. Users can still opt in via text
    # keywords like "FT/決賽桌" handled in gemini_session parsing.)
    # if table_color == "purple":
    #     hand["tournament_type"] = "icm"
    #     hand["phase"] = "FT"

    # Pot consistency check
    conf_parts["pot_consistency"] = _check_pot_consistency(columns)

    # Player tracking check
    conf_parts["player_tracking"] = _check_player_tracking(
        action_entries, street_cols
    )

    # OCR confidence from entries
    conf_parts["ocr_confidence"] = _avg_ocr_confidence(columns)

    return hand, conf_parts, diagnostics


def _extract_hero_hand_from_stack_text(table_result: dict) -> str:
    """Fallback: hero_hand might already be set by EasyOCR-based table parser."""
    # This is handled by table_parser._find_hero_cards now
    return ""


def _build_preflop_actions_from_order(
    action_entries: list[dict], pos_order: list[str],
    hero_position: str | None, table_size: int,
) -> str:
    """Build preflop_actions string from ordered action entries.

    Entries are already assigned positions by order. First `table_size`
    entries form the first round; remaining are re-actions.

    Format: "F-F-R2-F-F-F-C-F" (one action per position)
    """
    # First round: map position -> action code
    pos_actions: dict[str, str] = {}

    for i, entry in enumerate(action_entries[:table_size]):
        pos = entry.get("position")
        if entry["type"] == "hero":
            pos = hero_position
        if not pos:
            continue

        action = (entry.get("action") or "").lower()
        size = entry.get("size")
        code = _action_to_code(action, size)
        if code:
            pos_actions[pos] = code

    # Build first-round string in position order
    parts = []
    for pos in pos_order:
        parts.append(pos_actions.get(pos, "F"))

    result = "-".join(parts)

    # Re-actions (entries beyond first round)
    re_codes = []
    for entry in action_entries[table_size:]:
        if entry.get("_is_reaction"):
            action = (entry.get("action") or "").lower()
            size = entry.get("size")
            code = _action_to_code(action, size)
            if code:
                re_codes.append(code)

    if re_codes:
        result += "-" + "-".join(re_codes)

    return result


def _action_to_code(action: str, size: float | None) -> str | None:
    """Convert action name + size to preflop action code."""
    action = action.lower().strip()

    if action == "fold":
        return "F"
    elif action == "call":
        return "C"
    elif action == "check":
        return "C"  # preflop check = call (BB option)
    elif action in ("raise", "bet"):
        if size is not None:
            # Format: R{size} with no trailing zeros
            s = f"{size:g}"
            return f"R{s}"
        return "R2"  # default min raise
    elif action == "all-in":
        if size is not None:
            s = f"{size:g}"
            return f"AI{s}"
        return "AI"

    return None


def _build_streets(street_cols: list[dict], board_cards: list[str],
                   pos_order: list[str], hero_position: str = "",
                   active_positions: list[str] | None = None) -> list[dict]:
    """Build streets array from Flop/Turn/River columns.

    Uses hero_position and active_positions to correctly assign positions
    to postflop entries. Hero entries (type=hero) get hero_position.
    Opponent entries get their OCR-detected position, or are inferred
    from active_positions list.
    """
    streets = []

    # Map board cards to streets: first 3 = flop, 4th = turn, 5th = river
    flop_board = "".join(board_cards[:3]) if len(board_cards) >= 3 else ""
    turn_card = board_cards[3] if len(board_cards) >= 4 else ""
    river_card = board_cards[4] if len(board_cards) >= 5 else ""

    # Postflop action order: SB first, then BB, then other positions in order
    postflop_order = []
    if active_positions:
        for pos in ["SB", "BB"] + [p for p in pos_order if p not in ("SB", "BB")]:
            if pos in active_positions:
                postflop_order.append(pos)

    # Track who folds across streets
    folded_in_streets = set()

    for col in street_cols:
        name = col["name"].lower()
        entries = col.get("entries", [])

        # Include empty-entry columns only when a preceding street had
        # entries (the hand continued but no action was detected — e.g.,
        # check-check or went to showdown).
        if not entries and not streets:
            continue

        street = {}
        if name == "flop" and flop_board:
            street["board"] = flop_board
        elif name == "turn" and turn_card:
            street["card"] = turn_card
        elif name == "river" and river_card:
            street["card"] = river_card

        actions = []
        # Track position assignment for this street
        opp_positions_remaining = [p for p in postflop_order
                                   if p != hero_position and p not in folded_in_streets]
        opp_idx = 0

        for entry in entries:
            entry_type = entry.get("type", "opponent")
            action_text = (entry.get("action") or "").lower()
            size = entry.get("size")

            if not action_text or action_text in ("unknown", "skip"):
                continue

            # Assign position
            if entry_type == "hero":
                pos = hero_position
            else:
                # Use OCR-detected position if available
                ocr_pos = entry.get("position")
                if ocr_pos and ocr_pos != "BB":
                    # Trust OCR position if it's not the default
                    pos = ocr_pos
                elif opp_positions_remaining:
                    # Infer from postflop order
                    pos = opp_positions_remaining[opp_idx % len(opp_positions_remaining)]
                    opp_idx += 1
                else:
                    pos = ocr_pos or "?"

            act_code = _street_action_code(action_text, size)
            act_dict = {"position": pos, "action": act_code}
            if size is not None:
                act_dict["size"] = size
            actions.append(act_dict)

            # Track folds
            if action_text == "fold":
                folded_in_streets.add(pos)

        street["actions"] = actions
        if actions:
            streets.append(street)
        elif streets:
            # Include empty-action streets only when the prior street
            # did NOT end with a fold (hand continued to this street
            # but no entries were detected — e.g., went to showdown).
            prev_actions = streets[-1].get("actions", [])
            prev_ended_fold = (
                prev_actions
                and prev_actions[-1].get("action", "").upper() == "F"
            )
            if not prev_ended_fold:
                streets.append(street)

    return streets


def _street_action_code(action: str, size: float | None) -> str:
    """Convert postflop action to code for streets.

    Matches the format expected by analyze_hand.py:
    X=Check, C=Call, F=Fold, R{size}=Bet/Raise (absolute bb value)
    """
    action = action.lower().strip()
    if action == "fold":
        return "F"
    elif action == "check":
        return "X"
    elif action == "call":
        return "C"
    elif action in ("bet", "raise"):
        if size:
            return f"R{size:g}"
        return "R"
    elif "all" in action:
        if size:
            return f"R{size:g}"
        return "AI"
    return action.upper()


def _check_pot_consistency(columns: list[dict]) -> float:
    """Check if pot values are consistent across streets.

    Returns 0.0 to 1.0 confidence score.
    """
    pots = []
    for col in columns:
        if col.get("pot") is not None:
            pots.append(col["pot"])

    if len(pots) < 2:
        return 0.5  # Can't check with only 1 pot

    # Pots should be non-decreasing across streets
    increasing = all(pots[i] <= pots[i + 1] + 0.5 for i in range(len(pots) - 1))
    return 1.0 if increasing else 0.3


def _check_player_tracking(preflop_entries: list[dict],
                           street_cols: list[dict]) -> float:
    """Check that folded players don't reappear.

    Returns 0.0 to 1.0 confidence score.
    """
    folded = set()
    for entry in preflop_entries:
        pos = entry.get("position")
        if pos and (entry.get("action") or "").lower() == "fold":
            folded.add(pos)

    violations = 0
    total_checks = 0

    for col in street_cols:
        for entry in col.get("entries", []):
            pos = entry.get("position")
            if pos:
                total_checks += 1
                if pos in folded:
                    violations += 1
            # Track new folds
            if pos and (entry.get("action") or "").lower() == "fold":
                folded.add(pos)

    if total_checks == 0:
        return 0.5

    return max(0.0, 1.0 - violations / max(total_checks, 1))


def _avg_ocr_confidence(columns: list[dict]) -> float:
    """Average OCR confidence across all entries. Returns 0.0 to 1.0."""
    # We don't have direct access to OCR conf from entries,
    # so approximate: if entries exist with actions, confidence is decent
    total_entries = 0
    valid_entries = 0

    for col in columns:
        for entry in col.get("entries", []):
            total_entries += 1
            if entry.get("action") and entry["action"] != "Unknown":
                valid_entries += 1

    if total_entries == 0:
        return 0.3

    return min(1.0, valid_entries / max(total_entries, 1))


def _compute_confidence(parts: dict) -> float:
    """Compute weighted confidence score."""
    score = 0.0
    for key, weight in _CONF_WEIGHTS.items():
        score += parts.get(key, 0.0) * weight
    return min(1.0, max(0.0, score))


def _build_hints(table_result: dict, columns: list[dict],
                 hand: dict | None) -> dict:
    """Build hints dict with partial OCR data for Gemini fallback."""
    hints = {}

    board = table_result.get("board_cards", [])
    if board:
        hints["board_cards"] = board

    hero = table_result.get("hero_cards", [])
    if hero:
        hints["hero_cards"] = hero

    # Surface high-confidence suit predictions even when the rank head was
    # uncertain (or the cards were cleared as duplicates due to a rank
    # confusion). The CNN's suit head is far more reliable than the rank
    # head, so handing Gemini an authoritative suit list lets it focus
    # only on resolving the ranks.
    hero_details = table_result.get("hero_card_details") or []
    if len(hero_details) == 2 and all(
        d.get("suit") and d.get("suit_conf", 0.0) >= 0.90
        for d in hero_details
    ):
        hints["hero_card_suits"] = [d["suit"] for d in hero_details]

    color = table_result.get("table_color", "unknown")
    if color != "unknown":
        hints["table_color"] = color

    # Extract action summary from columns
    for col in columns:
        entries = col.get("entries", [])
        if entries:
            col_summary = []
            for e in entries:
                s = f"{e.get('position', '?')}: {e.get('action', '?')}"
                if e.get("size"):
                    s += f" {e['size']}"
                col_summary.append(s)
            if col_summary:
                hints[col["name"]] = col_summary

    if hand:
        hints["partial_hand"] = hand

    return hints if hints else None
