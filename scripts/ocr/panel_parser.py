"""Panel parser for Natural8 replay action panel.

Parses the action panel (below the divider) from N8 replay screenshots,
extracting street columns, pot sizes, and individual action entries.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .ocr_utils import ocr_text, preprocess_for_ocr

# Load config
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "n8_default.json"
with open(_CONFIG_PATH) as f:
    _CONFIG = json.load(f)

_POSITION_ALIASES = _CONFIG["position_aliases"]
_HERO_HSV = _CONFIG["entry_colors"]["hero_hsv"]

# Street names for header OCR matching
_STREET_NAMES = ["Blinds", "Pre-Flop", "Flop", "Turn", "River"]

# Known action words
_ACTIONS = {"Fold", "Check", "Call", "Bet", "Raise", "All-In"}


def normalize_position(pos: str) -> str:
    """Apply position alias mapping.

    Args:
        pos: Raw position string from OCR (e.g., "MP", "MP1")

    Returns:
        Normalized position (e.g., "LJ", "HJ")
    """
    if not pos:
        return pos
    pos = pos.strip()
    return _POSITION_ALIASES.get(pos, pos)


def _find_header_end(gray: np.ndarray) -> int:
    """Find the y-coordinate where the header row ends.

    The header has a dark background with text. Below it, entries start
    with brighter content. We detect the transition by looking for a row
    where brightness jumps after a dark region.
    """
    h = gray.shape[0]
    row_means = np.mean(gray, axis=1)

    # Scan from ~8% to ~16% of panel height for the header/body boundary
    scan_start = int(h * 0.08)
    scan_end = int(h * 0.18)

    for y in range(scan_start, min(scan_end, h - 5)):
        # Look for a bright row preceded by a dark row
        if row_means[y] > 55 and row_means[max(0, y - 1)] < 40:
            return y

    # Fallback: use ~12% of panel height
    return int(h * 0.12)


def split_columns(panel_image: np.ndarray) -> list[dict]:
    """Split the action panel into 5 street columns.

    The N8 panel has 5 equally-spaced columns:
    Blinds (Ante), Pre-Flop, Flop, Turn, River.

    Args:
        panel_image: BGR image of the action panel

    Returns:
        List of 5 dicts: {"name": str, "pot": float|None, "region": ndarray,
                          "x_start": int, "x_end": int, "header_end": int}
    """
    h, w = panel_image.shape[:2]
    gray = cv2.cvtColor(panel_image, cv2.COLOR_BGR2GRAY)

    header_end = _find_header_end(gray)
    col_w = w // 5

    columns = []
    for i in range(5):
        x1 = i * col_w
        x2 = (i + 1) * col_w if i < 4 else w

        # OCR the header: street name is in upper portion, pot in lower
        # The pot text is colored (yellow/green) on dark bg
        name_region = panel_image[2:int(header_end * 0.45), x1:x2]
        pot_region = panel_image[int(header_end * 0.45):header_end - 2, x1:x2]

        # OCR street name (light text on dark bg)
        name_prep = preprocess_for_ocr(name_region)
        name_inv = cv2.bitwise_not(name_prep)
        name_text, _ = ocr_text(name_inv, psm=7)

        # Match to known street names
        street_name = _STREET_NAMES[i]  # Default
        for sn in _STREET_NAMES:
            if sn.lower() in name_text.lower():
                street_name = sn
                break

        # OCR pot value (colored text on dark bg — normal OCR works)
        pot_prep = preprocess_for_ocr(pot_region)
        pot_text, _ = ocr_text(pot_prep, psm=7)
        pot_value = _parse_bb_amount(pot_text)
        if pot_value is None:
            # Try inverted
            pot_inv = cv2.bitwise_not(pot_prep)
            pot_text2, _ = ocr_text(pot_inv, psm=7)
            pot_value = _parse_bb_amount(pot_text2)

        # Extract column body region (below header)
        body = panel_image[header_end:, x1:x2]

        columns.append({
            "name": street_name,
            "pot": pot_value,
            "region": body,
            "x_start": x1,
            "x_end": x2,
            "header_end": header_end,
        })

    return columns


def _parse_bb_amount(text: str) -> float | None:
    """Parse a BB amount from text like '2.5 BB', '10 BB', '1.5BB'."""
    if not text:
        return None
    text = text.strip().upper().replace("BB", "").strip()
    # Remove any non-numeric chars except dot
    cleaned = ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            cleaned += ch
        elif cleaned:
            break  # Stop at first non-numeric after digits
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _parse_action_text(text: str) -> tuple[str | None, float | None]:
    """Parse action and optional size from OCR text.

    Args:
        text: OCR result like 'Fold', 'Call 2 BB', 'Raise 2.5 BB', 'Bet 4 BB'

    Returns:
        (action, size) e.g., ("Call", 2.0) or ("Fold", None)
    """
    if not text:
        return None, None

    # Clean common OCR artifacts
    text = text.strip().replace("}", "").replace("{", "").replace("|", "")
    text = text.replace("»", "").replace("«", "").strip()

    # Try to match known actions
    action = None
    remainder = text

    for act in sorted(_ACTIONS, key=len, reverse=True):
        # Case-insensitive match at start of text
        if text.lower().startswith(act.lower()):
            action = act
            remainder = text[len(act):].strip()
            break

    if action is None:
        # Try fuzzy: look for action words anywhere in text
        text_lower = text.lower()
        for act in sorted(_ACTIONS, key=len, reverse=True):
            if act.lower() in text_lower:
                action = act
                idx = text_lower.index(act.lower())
                remainder = text[idx + len(act):].strip()
                break

    if action is None:
        return None, None

    # Parse size from remainder
    size = _parse_bb_amount(remainder)

    return action, size


def detect_entries(column_region: np.ndarray) -> list[dict]:
    """Detect hero and opponent action entries in a column region.

    Detection strategy:
    - Hero entries: yellow/gold action bubbles (HSV H:15-45, S>60, V>130)
    - Opponent entries: white/light text boxes on dark background
      (found via contours in bright mask without dilation to avoid merging
       with player name text)

    Args:
        column_region: BGR image of the column body (below header)

    Returns:
        List of entry dicts sorted by y position:
        {"type": "hero"|"opponent", "y": int, "x": int, "w": int, "h": int,
         "region": ndarray}
    """
    ch, cw = column_region.shape[:2]
    if ch < 10 or cw < 10:
        return []

    hsv = cv2.cvtColor(column_region, cv2.COLOR_BGR2HSV)

    # Hero mask: yellow/gold action bubbles
    h_lo, h_hi = _HERO_HSV["h_range"]
    hero_mask = cv2.inRange(
        hsv,
        np.array([h_lo, 60, 130]),
        np.array([h_hi, 255, 255]),
    )
    # Light dilation for hero (connect text within the bubble)
    hero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    hero_dilated = cv2.dilate(hero_mask, hero_kernel, iterations=2)

    # Opponent mask: white/light text boxes (high V, low S)
    # Use NO dilation to keep action text boxes separate from name text
    opp_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    opp_mask[hero_mask > 0] = 0

    entries = []

    # Extract hero entries (with dilation)
    _extract_contour_entries(hero_dilated, column_region, "hero", entries,
                             min_area=600, min_h=12, min_w=30)

    # Extract opponent entries (without dilation, direct from mask)
    _extract_contour_entries(opp_mask, column_region, "opponent", entries,
                             min_area=400, min_h=12, min_w=30)

    # Sort by y position (chronological order)
    entries.sort(key=lambda e: e["y"])

    # Filter out entries that are likely player names (not action text)
    entries = _filter_action_entries(entries)

    return entries


def _extract_contour_entries(
    mask: np.ndarray, column_region: np.ndarray,
    entry_type: str, entries: list,
    min_area: int = 400, min_h: int = 12, min_w: int = 30,
):
    """Find contours in mask and add valid entries to the list."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area >= min_area and bh >= min_h and bw >= min_w:
            entries.append({
                "type": entry_type,
                "y": y,
                "x": x,
                "w": bw,
                "h": bh,
                "region": column_region[y:y + bh, x:x + bw],
            })


def _filter_action_entries(entries: list[dict]) -> list[dict]:
    """Filter entries to keep only those containing action text.

    OCR each entry to determine if it contains an action keyword.
    Entries that are just player names or avatars are filtered out.
    """
    filtered = []
    for entry in entries:
        region = entry["region"]
        if region is None or region.size == 0:
            continue

        eh = entry["h"]
        prep = preprocess_for_ocr(region)
        inv = cv2.bitwise_not(prep)

        # Choose PSM based on entry height: multi-line entries need psm=6
        psm_modes = [6, 7] if eh > 40 else [7, 6]

        best_text = ""
        best_conf = 0.0

        for psm in psm_modes:
            text, conf = ocr_text(prep, psm=psm)
            if conf > best_conf:
                best_text, best_conf = text, conf
            text2, conf2 = ocr_text(inv, psm=psm)
            if conf2 > best_conf:
                best_text, best_conf = text2, conf2

        # If OCR failed, try with padding (helps edge-clipped entries)
        if best_conf < 20:
            padded = cv2.copyMakeBorder(region, 8, 8, 8, 8,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            prep_p = preprocess_for_ocr(padded)
            inv_p = cv2.bitwise_not(prep_p)
            for psm in psm_modes:
                text, conf = ocr_text(prep_p, psm=psm)
                if conf > best_conf:
                    best_text, best_conf = text, conf
                text2, conf2 = ocr_text(inv_p, psm=psm)
                if conf2 > best_conf:
                    best_text, best_conf = text2, conf2

        # Check if text contains an action word
        if best_text:
            text_clean = best_text.replace("}", "").replace("{", "")
            text_clean = text_clean.replace("»", "").replace(">", "").strip()
            text_lower = text_clean.lower()
            has_action = any(act.lower() in text_lower for act in _ACTIONS)
            if has_action:
                entry["_ocr_text"] = text_clean
                entry["_ocr_conf"] = best_conf
                filtered.append(entry)
                continue

        # For hero entries, keep even without clear action text
        # (the yellow bubble usually IS an action)
        if entry["type"] == "hero":
            entry["_ocr_text"] = best_text.strip() if best_text else ""
            entry["_ocr_conf"] = best_conf
            filtered.append(entry)

    return filtered


def _is_skip_entry(text: str) -> bool:
    """Check if an entry should be skipped (timebank, wins, showdown)."""
    if not text:
        return False
    text_lower = text.lower()
    skip_patterns = ["wins", "10s"]
    return any(pat in text_lower for pat in skip_patterns)


def parse_entry(entry_region: np.ndarray, entry_type: str,
                ocr_text_cached: str | None = None) -> dict:
    """OCR a single entry to extract action and optional size.

    Args:
        entry_region: BGR image of the entry card
        entry_type: "hero" or "opponent"
        ocr_text_cached: Pre-computed OCR text (from _filter_action_entries)

    Returns:
        {"type": str, "position": str|None, "action": str, "size": float|None}
    """
    if entry_region is None or entry_region.size == 0:
        return {"type": entry_type, "position": None, "action": "Unknown", "size": None}

    # Use cached OCR text if available, otherwise OCR now
    if ocr_text_cached:
        text = ocr_text_cached
    else:
        prep = preprocess_for_ocr(entry_region)
        if entry_type == "hero":
            ocr_img = cv2.bitwise_not(prep)
        else:
            ocr_img = prep

        text, conf = ocr_text(ocr_img, psm=7)
        if conf < 50:
            alt_img = cv2.bitwise_not(prep) if entry_type != "hero" else prep
            text2, conf2 = ocr_text(alt_img, psm=7)
            if conf2 > conf:
                text = text2

    # Skip entries with "Wins" or timebank indicators
    if _is_skip_entry(text):
        return {"type": entry_type, "position": None, "action": "Skip", "size": None}

    # Parse action and size from OCR text
    action, size = _parse_action_text(text)

    # Detect position badge for opponent entries
    position = None
    if entry_type == "opponent":
        position = _detect_position_badge(entry_region)
        if position:
            position = normalize_position(position)

    if action is None:
        action = "Unknown"

    return {
        "type": entry_type,
        "position": position,
        "action": action,
        "size": size,
    }


def _detect_position_badge(entry_region: np.ndarray) -> str | None:
    """Detect and OCR the position badge on an opponent entry.

    Since we detect only the action text box (not the full entry card),
    the position badge is typically NOT inside our crop. Return None
    and let _detect_position_from_context handle it.
    """
    return None


def _detect_position_from_context(
    column_region: np.ndarray, entry_y: int, entry_x: int, entry_h: int
) -> str | None:
    """Detect position badge near the action text box in the column.

    Position badges are small colored rectangles (e.g., "UTG" on red, "SB" on cyan)
    that appear near opponent entries. They're typically found:
    - Below the action text box (within ~30px)
    - Above the action text box (within ~30px, near avatar area)
    - To the left of the action text box
    """
    ch, cw = column_region.shape[:2]

    known = [
        "UTG", "UTG1", "UTG+1", "UTG2", "UTG+2",
        "EP", "EP1", "MP", "MP1", "MP2",
        "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ]

    # Search areas: below and above the entry
    search_areas = []

    # Below the entry
    y1 = entry_y + entry_h
    y2 = min(ch, y1 + 30)
    if y2 > y1:
        search_areas.append(column_region[y1:y2, 0:min(cw, entry_x + 60)])

    # Above the entry (where avatar + position badge might be)
    y1_above = max(0, entry_y - 30)
    y2_above = entry_y
    if y2_above > y1_above:
        search_areas.append(column_region[y1_above:y2_above, 0:min(cw, entry_x + 60)])

    for badge_area in search_areas:
        if badge_area.size == 0:
            continue

        hsv = cv2.cvtColor(badge_area, cv2.COLOR_BGR2HSV)
        colored_mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([180, 255, 255]))

        contours, _ = cv2.findContours(
            colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 15 and bh > 8 and bw < 60 and bh < 25:
                badge_img = badge_area[y:y + bh, x:x + bw]
                prep = preprocess_for_ocr(badge_img)
                inv = cv2.bitwise_not(prep)
                text, conf = ocr_text(
                    inv,
                    whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+",
                    psm=7,
                )
                if text and conf > 30:
                    text = text.strip().upper()
                    for pos in known:
                        if pos in text:
                            return normalize_position(pos)

    return None


def parse_panel(panel_image: np.ndarray) -> dict:
    """Parse the entire action panel from an N8 replay screenshot.

    Args:
        panel_image: BGR image of the action panel (below divider)

    Returns:
        {"columns": [{"name": str, "pot": float|None,
                       "entries": [{"type": str, "position": str|None,
                                    "action": str, "size": float|None}]}]}
    """
    columns = split_columns(panel_image)

    result_columns = []
    for col in columns:
        col_region = col["region"]
        # Detect entry regions in this column
        raw_entries = detect_entries(col_region)

        # Parse each entry
        parsed_entries = []
        for entry in raw_entries:
            cached_text = entry.get("_ocr_text")
            parsed = parse_entry(entry["region"], entry["type"],
                                 ocr_text_cached=cached_text)

            if parsed["action"] == "Skip":
                continue

            # Try to detect position from column context for opponent entries
            if parsed["type"] == "opponent" and not parsed["position"]:
                pos = _detect_position_from_context(
                    col_region, entry["y"], entry["x"], entry["h"]
                )
                if pos:
                    parsed["position"] = pos

            if parsed["action"] != "Unknown":
                parsed_entries.append(parsed)

        result_columns.append({
            "name": col["name"],
            "pot": col["pot"],
            "entries": parsed_entries,
        })

    return {"columns": result_columns}
