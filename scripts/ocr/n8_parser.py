"""Main N8 replay screenshot parser.

Orchestrates region detection, table parsing, panel parsing, and hand
assembly into the JSON format expected by analyze_hand_full().
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .region_detector import detect_regions
from .table_parser import parse_table
from .panel_parser import parse_panel, normalize_position

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
        return {"hand": None, "hints": None, "confidence": 0.0}

    # Step 1: detect regions
    regions = detect_regions(image)
    if regions is None:
        return {"hand": None, "hints": None, "confidence": 0.0}

    # Step 2: parse table
    table_result = parse_table(regions["table"])

    # Step 3: parse panel
    panel_result = parse_panel(regions["panel"])
    columns = panel_result.get("columns", [])

    # Step 4: assemble hand JSON
    hand, confidence_parts = _assemble_hand(table_result, columns)

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
    }


def _filter_action_entries(entries: list[dict]) -> list[dict]:
    """Filter preflop entries to keep only real action entries.

    Removes false hero detections (e.g., avatar markers that are yellow
    but don't contain action text).
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
    return result


def _estimate_table_size(action_entries: list[dict]) -> int:
    """Estimate table size from preflop action entries.

    In N8 PreFlop, entries appear in position order. The first round
    has exactly one entry per player. After a raise, some players may
    act again (re-actions).

    Strategy: find where the first round ends by looking for re-actions.
    A re-action happens when a raise is followed by calls/folds from
    positions that would have already acted in the first round.
    """
    n = len(action_entries)
    if n <= 2:
        return max(n, 2)
    if n <= 9:
        # Could be all first-round (no re-actions) or have re-actions
        # Check: if there's a raise, entries after it might be re-actions
        # In the first round, each player acts once in order
        # After a raise, the remaining first-round players still need to act
        # Re-actions only start AFTER all first-round players have acted

        # Heuristic: find the last raise index. If there are entries after
        # the raise that are calls/folds, check if they could be the remaining
        # first-round players or re-actions.

        # Simple approach: if total is <= 9, assume it's the table size
        # This works because N8 shows all players (folders included) in PreFlop
        return n

    # More than 9 entries — must have re-actions
    # Table size is at most 9
    return 9


def _assemble_hand(table_result: dict, columns: list[dict]) -> tuple[dict | None, dict]:
    """Assemble hand JSON from parsed table and panel data.

    Uses position-order-based inference: in N8 PreFlop column, entries
    appear in strict position order (UTG first, BB last). Combined with
    entry count, we determine table size and assign positions.

    Returns:
        (hand_dict or None, confidence_parts dict)
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

    # Card confidence
    if hero_cards and len(hero_cards) == 2:
        conf_parts["card_confidence"] = 0.8
    if board_cards and len(board_cards) >= 3:
        conf_parts["card_confidence"] = min(1.0, conf_parts["card_confidence"] + 0.2)

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
        return None, conf_parts

    preflop_entries = preflop_col.get("entries", [])

    # Filter out false hero entries (avatar markers without action text)
    action_entries = _filter_action_entries(preflop_entries)

    if not action_entries:
        return None, conf_parts

    # Determine table size from entry count
    players_at_table = _estimate_table_size(action_entries)
    players_at_table = min(max(players_at_table, 2), 9)
    pos_order = POSITION_ORDERS.get(players_at_table, POSITION_ORDERS[8])

    # Assign positions by entry order (first entry = first position, etc.)
    hero_position = None
    hero_index = None
    for i, entry in enumerate(action_entries[:players_at_table]):
        if i < len(pos_order):
            entry["position"] = pos_order[i]
            if entry["type"] == "hero":
                hero_position = pos_order[i]
                hero_index = i

    # Mark re-action entries (beyond first round)
    for i, entry in enumerate(action_entries[players_at_table:], players_at_table):
        entry["_is_reaction"] = True

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

    if not hero_position:
        return None, conf_parts

    # Build preflop_actions string using assigned positions
    preflop_actions = _build_preflop_actions_from_order(
        action_entries, pos_order, hero_position, players_at_table
    )

    if not preflop_actions:
        return None, conf_parts

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
        return None, conf_parts

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

    # Build streets with position context
    streets = _build_streets(street_cols, board_cards, pos_order,
                             hero_position, active_positions)

    # Effective BB: from hero stack or player stacks
    effective_bb = None
    hero_stack = table_result.get("hero_stack")
    stacks = table_result.get("player_stacks", [])
    if hero_stack:
        effective_bb = hero_stack
    elif stacks:
        effective_bb = min(stacks) if len(stacks) > 1 else stacks[0]

    hand = {
        "gametype": "MTTGeneral",
        "hero_hand": hero_hand,
        "hero_position": hero_position,
        "players_at_table": players_at_table,
        "preflop_actions": preflop_actions,
    }

    if effective_bb is not None:
        hand["effective_bb"] = effective_bb

    if streets:
        hand["streets"] = streets

    if stacks:
        hand["player_stacks"] = stacks

    # Final Table detection
    if table_color == "purple":
        hand["tournament_type"] = "icm"
        hand["phase"] = "FT"

    # Pot consistency check
    conf_parts["pot_consistency"] = _check_pot_consistency(columns)

    # Player tracking check
    conf_parts["player_tracking"] = _check_player_tracking(
        action_entries, street_cols
    )

    # OCR confidence from entries
    conf_parts["ocr_confidence"] = _avg_ocr_confidence(columns)

    return hand, conf_parts


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

        if not entries:
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

        if actions:
            street["actions"] = actions
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
