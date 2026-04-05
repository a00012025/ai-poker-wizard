"""Panel parser for Natural8 replay action panel.

Parses the action panel (below the divider) from N8 replay screenshots.
Uses EasyOCR to OCR each column body once, then groups text by Y-position
into action entries.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .ocr_utils import ocr_full_image

# Load config
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "n8_default.json"
with open(_CONFIG_PATH) as f:
    _CONFIG = json.load(f)

_POSITION_ALIASES = _CONFIG["position_aliases"]
_HERO_HSV = _CONFIG["entry_colors"]["hero_hsv"]

# Known positions and actions
_POSITIONS = {
    "UTG", "UTG1", "UTG+1", "UTG2", "UTG+2",
    "EP", "EP1", "MP", "MP1", "MP2",
    "LJ", "HJ", "CO", "BTN", "SB", "BB",
}
_ACTIONS = {"Fold", "Check", "Call", "Bet", "Raise", "All-In"}
_ACTION_PATTERNS = re.compile(
    r"(Fold|Check|Call|Bet|Raise|All.?In|All.?in|FOLD|CHECK|CALL|BET|RAISE)",
    re.IGNORECASE,
)
_BB_PATTERN = re.compile(r"(\d+\.?\d*)\s*BB", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"(\d+\.?\d*)")

# Skip patterns
_SKIP_PATTERNS = re.compile(r"(Wins|wins|10s|\d+\.\d+%|All Ante)", re.IGNORECASE)


def normalize_position(pos: str) -> str:
    """Apply position alias mapping (MP→LJ, MP1→HJ, etc.)."""
    if not pos:
        return pos
    pos = pos.strip()
    return _POSITION_ALIASES.get(pos, pos)


def split_columns(panel_image: np.ndarray) -> list[dict]:
    """Split the action panel into 5 street columns.

    Returns:
        List of 5 dicts: {"name": str, "pot": float|None, "region": ndarray}
    """
    h, w = panel_image.shape[:2]
    gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)

    # Find header end: scan for brightness transition
    header_end = _find_header_end(gray)
    col_w = w // 5

    # OCR the header row once
    header_region = panel_image[0:header_end, :]
    header_texts = ocr_full_image(header_region)

    _STREET_NAMES = ["Blinds", "Pre-Flop", "Flop", "Turn", "River"]

    columns = []
    for i in range(5):
        x1 = i * col_w
        x2 = (i + 1) * col_w if i < 4 else w

        # Find header texts in this column's x range
        col_header_texts = [
            t for t in header_texts
            if t["center_x"] >= x1 and t["center_x"] < x2
        ]

        # Determine street name
        street_name = _STREET_NAMES[i]  # default
        for t in col_header_texts:
            for sn in _STREET_NAMES:
                if sn.lower() in t["text"].lower():
                    street_name = sn
                    break

        # Extract pot value from header
        pot_value = None
        for t in col_header_texts:
            m = _BB_PATTERN.search(t["text"])
            if m:
                try:
                    pot_value = float(m.group(1))
                except ValueError:
                    pass
                break

        # When the panel has many entries (8-9 players), the first entry
        # can scroll up into the header area.  Detect this by checking if
        # any action keyword appears in the header for this column; if so,
        # extend the body start upward to capture the clipped entry.
        body_start = header_end
        action_header_texts = [
            t for t in col_header_texts
            if _ACTION_PATTERNS.search(t["text"])
        ]
        if action_header_texts:
            # Find topmost action-related text in the header
            min_action_y = min(t["center_y"] for t in action_header_texts)
            # Also check for player name text above the action
            name_texts = [
                t for t in col_header_texts
                if t["center_y"] < min_action_y
                and not _ACTION_PATTERNS.search(t["text"])
                and not any(sn.lower() in t["text"].lower() for sn in _STREET_NAMES)
                and not _BB_PATTERN.search(t["text"])
            ]
            if name_texts:
                body_start = max(0, int(min(t["center_y"] for t in name_texts)) - 10)
            else:
                body_start = max(0, int(min_action_y) - 30)

        body = panel_image[body_start:, x1:x2]
        columns.append({
            "name": street_name,
            "pot": pot_value,
            "region": body,
            "x_start": x1,
            "x_end": x2,
        })

    return columns


def _find_header_end(gray: np.ndarray) -> int:
    """Find where the header row ends."""
    h = gray.shape[0]
    row_means = np.mean(gray, axis=1)

    scan_start = int(h * 0.06)
    scan_end = int(h * 0.18)

    for y in range(scan_start, min(scan_end, h - 5)):
        if row_means[y] > 55 and row_means[max(0, y - 1)] < 40:
            return y

    return int(h * 0.10)


def detect_entries(column_region: np.ndarray) -> list[dict]:
    """Detect action entries in a column using full-column OCR.

    OCRs the entire column body once, then groups text results by
    Y-position into entries. Uses background color to classify hero/opponent.

    Returns:
        List of {"type", "position", "action", "size"} dicts
    """
    ch, cw = column_region.shape[:2]
    if ch < 20 or cw < 20:
        return []

    # OCR the entire column body at once
    ocr_results = ocr_full_image(column_region)

    if not ocr_results:
        return []

    # Group OCR results by Y proximity (texts within ~25px = same entry)
    groups = _group_by_y(ocr_results, y_threshold=25)

    # Split groups that contain multiple actions (merged entries)
    groups = _split_multi_action_groups(groups)

    # Classify each group into an action entry.
    # In N8, each entry has: name group (player name) then action group
    # (action text + position badge + size).  Pair them so we can extract
    # the player name for each action.
    entries = []
    pending_name = None
    for group in groups:
        entry = _classify_group(group, column_region)
        if entry is None:
            continue
        if entry["action"] == "Skip":
            continue
        if entry["action"] == "_name_only":
            # This group is just a player name — remember it for the next
            # action group.
            pending_name = entry.get("player_name")
            continue
        # Real action entry — attach pending name if available
        if pending_name:
            entry["player_name"] = pending_name
            pending_name = None
        entries.append(entry)

    return entries


def _split_multi_action_groups(groups: list[list[dict]]) -> list[list[dict]]:
    """Split groups that contain multiple action keywords.

    When two entries are very close vertically (gap < y_threshold), they
    get merged into one group. Detect this by counting action matches
    and split at the boundary between the last text of the first action
    entry and the first text of the second.
    """
    result = []
    for group in groups:
        # Count action matches in this group
        action_indices = []
        for i, t in enumerate(group):
            if _ACTION_PATTERNS.search(t["text"]):
                action_indices.append(i)

        if len(action_indices) <= 1:
            # 0 or 1 action — no split needed
            result.append(group)
            continue

        # Multiple actions found — split between them.
        # Each action belongs to its own entry. Split point: midway between
        # the last item before the next action's "name" line and that name.
        # Heuristic: split at the largest Y-gap between consecutive items
        # that falls between two action keywords.
        sorted_group = sorted(group, key=lambda t: t["center_y"])

        # Find split points: for each pair of consecutive actions,
        # find the largest Y-gap between them
        splits = []
        for ai in range(len(action_indices) - 1):
            # Items between action[ai] and action[ai+1]
            act1_y = group[action_indices[ai]]["center_y"]
            act2_y = group[action_indices[ai + 1]]["center_y"]

            # Find the best split point in sorted_group between these two actions
            best_gap = 0
            best_split_idx = None
            for si in range(len(sorted_group) - 1):
                y1 = sorted_group[si]["center_y"]
                y2 = sorted_group[si + 1]["center_y"]
                # Only consider gaps between the two actions
                if y1 >= act1_y and y2 <= act2_y:
                    gap = y2 - y1
                    if gap > best_gap:
                        best_gap = gap
                        best_split_idx = si + 1

            if best_split_idx is not None:
                splits.append(best_split_idx)

        if not splits:
            result.append(group)
            continue

        # Apply splits
        prev = 0
        for sp in splits:
            result.append(sorted_group[prev:sp])
            prev = sp
        result.append(sorted_group[prev:])

    return result


def _group_by_y(ocr_results: list[dict], y_threshold: int = 50) -> list[list[dict]]:
    """Group OCR text detections by Y proximity.

    Each action entry in N8 has: player name (~20px), action text (~30px below),
    and position badge (~15px below action). Total height ~60px.
    Use y_threshold=50 to keep all parts of one entry together.
    """
    if not ocr_results:
        return []

    sorted_results = sorted(ocr_results, key=lambda r: r["center_y"])
    groups = []
    current_group = [sorted_results[0]]

    for r in sorted_results[1:]:
        if r["center_y"] - current_group[-1]["center_y"] < y_threshold:
            current_group.append(r)
        else:
            groups.append(current_group)
            current_group = [r]
    groups.append(current_group)

    return groups


def _classify_group(group: list[dict], column_region: np.ndarray) -> dict | None:
    """Classify a group of OCR texts into an action entry.

    Returns:
        {"type": "hero"|"opponent", "position": str|None,
         "action": str, "size": float|None}
        or None if not an action entry
    """
    # Combine all text in group
    full_text = " ".join(t["text"] for t in group)

    # Skip non-action entries
    if _SKIP_PATTERNS.search(full_text):
        return None

    # Detect action
    action_match = _ACTION_PATTERNS.search(full_text)
    if not action_match:
        # No action found — this might be a player name group.
        # Name groups: 1-2 text items, no action keyword, not a position,
        # not a number.
        if len(group) <= 2:
            name_text = full_text.strip()
            # Skip if it's just a position badge or number
            if (name_text.upper() not in _POSITIONS
                    and not re.match(r'^[\d.]+\s*BB?$', name_text, re.I)
                    and len(name_text) >= 2):
                return {
                    "type": "opponent",
                    "position": None,
                    "action": "_name_only",
                    "size": None,
                    "player_name": name_text,
                }
        return None

    action_raw = action_match.group(1)
    action = _normalize_action(action_raw)

    # Detect size (BB amount)
    size = None
    bb_match = _BB_PATTERN.search(full_text)
    if bb_match:
        try:
            size = float(bb_match.group(1))
        except ValueError:
            pass
    elif action in ("Call", "Bet", "Raise"):
        # Look for standalone number after action
        after_action = full_text[action_match.end():]
        num_match = _NUMBER_PATTERN.search(after_action)
        if num_match:
            try:
                size = float(num_match.group(1))
            except ValueError:
                pass

    # Detect position — try exact match first, then longest substring
    # to avoid "UTG" matching "UTG1" before "UTG1" itself.
    position = None
    for t in group:
        text_upper = t["text"].strip().upper()
        # Exact match first
        for pos in _POSITIONS:
            if pos.upper() == text_upper:
                position = normalize_position(pos)
                break
        if not position:
            # Substring match, longest first (so "UTG1" beats "UTG")
            for pos in sorted(_POSITIONS, key=len, reverse=True):
                if pos.upper() in text_upper:
                    position = normalize_position(pos)
                    break
        if position:
            break

    # Extract player name: text items that appear ABOVE the action line,
    # are not action keywords, positions, or BB amounts.
    # Sort group by Y (top to bottom) and check items above the action.
    player_name = None
    sorted_group = sorted(group, key=lambda t: t["center_y"])
    action_y = None
    for t in sorted_group:
        if _ACTION_PATTERNS.search(t["text"]):
            action_y = t["center_y"]
            break
    if action_y is not None:
        for t in sorted_group:
            if t["center_y"] >= action_y:
                break  # past the action line
            text = t["text"].strip()
            text_upper = text.upper()
            # Skip positions, BB amounts, actions, short/numeric text
            if text_upper in _POSITIONS:
                continue
            if _BB_PATTERN.match(text):
                continue
            if _ACTION_PATTERNS.search(text):
                continue
            if len(text) < 2:
                continue
            if re.match(r'^\d+$', text):
                continue
            # This looks like a player name
            player_name = text
            break

    # Determine hero/opponent by checking background color at group center
    avg_y = int(sum(t["center_y"] for t in group) / len(group))
    entry_type = _detect_entry_type(column_region, avg_y)

    result = {
        "type": entry_type,
        "position": position,
        "action": action,
        "size": size,
    }
    if player_name:
        result["player_name"] = player_name
    return result


def _normalize_action(action_raw: str) -> str:
    """Normalize action text to standard form."""
    action = action_raw.strip()
    lower = action.lower()
    if "fold" in lower:
        return "Fold"
    elif "check" in lower:
        return "Check"
    elif "call" in lower:
        return "Call"
    elif "bet" in lower:
        return "Bet"
    elif "raise" in lower:
        return "Raise"
    elif "all" in lower:
        return "All-In"
    return action.capitalize()


def _detect_entry_type(column_region: np.ndarray, y: int) -> str:
    """Detect if entry at y-position is hero (yellow) or opponent (white).

    Samples a horizontal strip at y and checks for yellow hue.
    """
    ch, cw = column_region.shape[:2]
    y = max(0, min(y, ch - 1))

    # Sample a strip around y
    y1 = max(0, y - 10)
    y2 = min(ch, y + 10)
    strip = column_region[y1:y2, :]

    if strip.size == 0:
        return "opponent"

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h_lo, h_hi = _HERO_HSV["h_range"]
    s_min = _HERO_HSV["s_min"]
    v_min = _HERO_HSV["v_min"]

    hero_mask = cv2.inRange(
        hsv,
        np.array([h_lo, s_min, v_min]),
        np.array([h_hi, 255, 255]),
    )

    hero_ratio = np.sum(hero_mask > 0) / hero_mask.size
    return "hero" if hero_ratio > 0.05 else "opponent"


def parse_panel(panel_image: np.ndarray) -> dict:
    """Parse the entire action panel from an N8 replay screenshot.

    Returns:
        {"columns": [{"name": str, "pot": float|None,
                       "entries": [{"type", "position", "action", "size"}]}]}
    """
    columns = split_columns(panel_image)

    result_columns = []
    for col in columns:
        entries = detect_entries(col["region"])
        result_columns.append({
            "name": col["name"],
            "pot": col["pot"],
            "entries": entries,
        })

    return {"columns": result_columns}
