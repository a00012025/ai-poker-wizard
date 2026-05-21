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
# All-In tolerates "ll" being misread as "II" / "1I" / "I1" / "11" — the
# Natural8 sticker uses a font where lowercase l and uppercase I are nearly
# identical. EasyOCR routinely returns "AII-In" / "AlI-In" / "AII-1n", so
# accept any 2-character mix of [lI1] in the middle and any of [InOoUu] for
# the trailing "in" (lowercase n sometimes lifts to "u").
# Regression: H2842 — hero flop all-in was OCR'd as "AII-In" (group fell
# through as a name-only entry, attached as player_name to the next "Call
# 7 BB" entry → final flop action mis-recorded as hero call instead of jam).
_ACTION_PATTERNS = re.compile(
    r"(Fold|Check|Call|Bet|Ra[iIl1]se|Raise|A[lI1]{1,2}.?[Ii1][nNuU]|FOLD|CHECK|CALL|BET|RA[IIL1]SE|RAISE)",
    re.IGNORECASE,
)
# Strips position badges, BB amounts, and stand-alone numbers so the
# residue check after an All-In match sees only alphabetic noise. Used by
# the username-vs-action disambiguation in _classify_group.
_ACTION_RESIDUE_STRIP_RE = re.compile(
    r"(\bUTG\+?\d?\b|\bMP\d?\b|\bEP\d?\b|\bLJ\b|\bHJ\b|\bCO\b|\bBTN\b|"
    r"\bSB\b|\bBB\b|\d+\.?\d*)",
    re.IGNORECASE,
)


def _looks_like_allin_match(matched: str) -> bool:
    """True when an action regex hit looks like the All-In branch.

    The Fold/Check/Call/Bet/Raise branches all start with distinct letters,
    so the All-In branch is the only one whose residue check we need.
    """
    return matched[:1].lower() == "a"


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


# EasyOCR frequently misreads the digit "1" as the letter "T" / "I" / "L"
# on small Natural8 position badges (confidence ~0.40-0.55). Map the
# corrupt reads back to the correct position before the substring matcher
# sees them — otherwise "UTGT" silently falls through to the substring
# rule and matches "UTG" (the shorter prefix), collapsing UTG+1 hero
# entries into UTG. Regression for H2766 where BBJordan's UTG1 badge
# was OCR'd as "UTGT" (conf 0.54) producing hero_position="UTG" in the
# parsed hand → wrong multiway simplification → missing turn solver data.
# Canonical form (whitespace-stripped, uppercased) → corrected badge text.
# EasyOCR frequently misreads the digit "1" as T / I / L on small badges.
_OCR_POSITION_CORRECTIONS = {
    "UTGT": "UTG1",  # digit 1 → letter T
    "UTGI": "UTG1",  # digit 1 → letter I
    "UTGL": "UTG1",  # digit 1 → letter L
    "UTGZ": "UTG2",  # digit 2 → letter Z
}


def _preprocess_ocr_position(text: str) -> str:
    """Map OCR-corrupted position reads back to canonical badge text.

    Whitespace-collapses and upper-cases the input before lookup, so "UTG 1",
    "utg1", "UTGt" all hit the same entries. Pure text→text; does not touch
    the _POSITION_ALIASES table. Returns the original (stripped) string
    unchanged when no known corruption matches.

    Regression for H2766 where BBJordan's UTG1 panel badge was OCR'd as
    "UTGT" (confidence 0.54) and the substring matcher silently fell
    through to "UTG".
    """
    if not text:
        return text
    stripped = text.strip()
    canon = re.sub(r"\s+", "", stripped).upper()
    if canon in _OCR_POSITION_CORRECTIONS:
        return _OCR_POSITION_CORRECTIONS[canon]
    # A stray space inside an otherwise-correct read ("UTG 1") — collapse it.
    if canon in ("UTG1", "UTG2") and " " in stripped:
        return canon.replace("UTG", "UTG")  # e.g. "UTG1"
    return stripped


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


def _resolve_allin_attribution(entries: list[dict]) -> list[dict]:
    """Disambiguate who shoved vs who called in an all-in street.

    N8 paints a red "All-In" badge on the all-in player's bet/raise
    sticker. In the showdown layout it re-renders that player's
    avatar+hole-cards directly below the responder's call sticker, so the
    full-column OCR can split the bare red badge into its own nameless
    entry — sometimes on the *other* player's side and with a garbled
    size (the sum of two adjacent "X BB" stickers). That fabricates a
    phantom raise/all-in for the player who really just called.

    Two invariants make this recoverable, regardless of which side is
    short and regardless of stack depths (the caller can cover the shover
    or be covered by it — the money is the same, but the attribution is
    not):

    1. A bare "All-In" entry (no ``player_name``) that sits on top of a
       *named* bet/raise is that bet's badge, not a separate action — it
       marks the bettor as all-in. Promote the bet to "All-In", drop the
       badge. (Generalises the same-side dedup pre-pass in
       ``detect_entries`` to the cross-side showdown layout.)
    2. Once a player is all-in they cannot act again on that street; the
       only remaining decision is a single Call or Fold by the *other*
       player. Collapse every post-shove fragment (split call stickers,
       a call mis-read as a raise) into one responder entry on the
       opposite side. A response that matches an all-in is a Call, never
       a Raise — even when the responder is themselves all-in for it
       ("call-while-all-in"); that is still a Call for solver purposes,
       and effective-stack accounting is handled downstream from the
       displayed stacks.

    Scoped to the hero-vs-one-opponent postflop line this codebase
    solves; multiway postflop already falls back elsewhere.
    """
    if not entries:
        return entries

    # 1. Fold a bare All-In badge into the adjacent prior shove (any side).
    collapsed: list[dict] = []
    for e in entries:
        is_bare_allin = (
            (e.get("action") or "") == "All-In"
            and not e.get("player_name")
        )
        if is_bare_allin and collapsed:
            prev = collapsed[-1]
            prev_act = (prev.get("action") or "").lower()
            if prev_act in ("bet", "raise", "all-in") and prev.get("size"):
                prev["action"] = "All-In"
                continue
        collapsed.append(e)
    entries = collapsed

    # 2. Find the all-in shove (last all-in carrying a real size).
    shove_idx = None
    for i, e in enumerate(entries):
        if (e.get("action") or "").lower() == "all-in" and e.get("size"):
            shove_idx = i
    if shove_idx is None:
        return entries

    shove = entries[shove_idx]
    post = entries[shove_idx + 1:]
    if not post:
        return entries

    # 3. Collapse everything after the shove into one responder decision
    #    by the opposite side.
    responder_type = "opponent" if shove.get("type") == "hero" else "hero"
    has_fold = any((p.get("action") or "").lower() == "fold" for p in post)
    has_match = any(
        (p.get("action") or "").lower() in ("call", "raise", "bet", "all-in")
        for p in post
    )
    if has_match or not has_fold:
        responder = {
            "type": responder_type,
            "position": None,
            "action": "Call",
            "size": shove.get("size"),
        }
    else:
        responder = {
            "type": responder_type,
            "position": None,
            "action": "Fold",
            "size": None,
        }
    # Carry an opponent responder's position from the fragments (hero
    # responders get hero_position assigned downstream regardless).
    if responder_type == "opponent":
        for p in post:
            pos = p.get("position")
            if pos and pos != shove.get("position"):
                responder["position"] = pos
                break

    return entries[: shove_idx + 1] + [responder]


def _collapse_preflop_raise_jam(entries: list[dict]) -> list[dict]:
    """Collapse N8's bare "All-In" overlay onto the preceding named raise.

    When a preflop raise is for all chips, N8 stamps a small red "All-In"
    badge on the raise sticker. Full-column OCR splits it into its own
    entry: action=All-In, no player_name, no position badge, no size (the
    badge carries no number). The red-sticker hero heuristic in
    `_classify_group` then mis-tags it `hero`, which (a) shifts index-based
    position assignment downstream and (b) trips the all-in post-pass into
    flipping the *real* hero's call to opponent.

    The badge always belongs to the immediately-preceding named raiser, so
    drop it and promote that raise to All-In (keeping its size). Side-
    agnostic — the badge's mis-detected hero/opponent type is irrelevant.

    Preflop-only (caller-gated): every N8 preflop sticker, including hero's
    own blind, carries a position badge, so a positionless/sizeless/nameless
    bare All-In can only be this overlay. Postflop hero jams legitimately
    have no badge and are handled by the H2842 passes in `detect_entries`.
    Regression: H2878 — CO raise-jam 3.5bb + SB raise-jam 11.1bb, hero BB
    call; was parsed as BTN A2o facing a CO open.
    """
    out: list[dict] = []
    for e in entries:
        if out:
            prev = out[-1]
            is_overlay = (
                (e.get("action") or "").lower() in ("all-in", "allin", "all in")
                and not e.get("player_name")
                and not e.get("position")
                and e.get("size") is None
            )
            prev_named_aggro = (
                bool(prev.get("player_name"))
                and (prev.get("action") or "").lower()
                in ("bet", "raise", "all-in")
            )
            if is_overlay and prev_named_aggro:
                if (prev.get("action") or "").lower() in ("bet", "raise"):
                    prev["action"] = "All-In"
                continue
        out.append(e)
    return out


def detect_entries(column_region: np.ndarray, is_preflop: bool = False) -> tuple[list[dict], int]:
    """Detect action entries in a column using full-column OCR.

    OCRs the entire column body once, then groups text results by
    Y-position into entries. Uses background color to classify hero/opponent.

    Args:
        column_region: cropped image of one street column body.
        is_preflop: True for the Pre-Flop column. Enables the raise-jam
            overlay collapse (see below) — preflop-only because N8 renders
            every preflop sticker with a position badge, so the bare red
            "All-In" overlay is unambiguous there; postflop hero jams have
            no badge and would be misread as an overlay.

    Returns:
        Tuple of (entries, pre-collapse group count). Entries are
        {"type", "position", "action", "size"} dicts.
    """
    ch, cw = column_region.shape[:2]
    if ch < 20 or cw < 20:
        return [], 0

    # OCR the entire column body at once
    ocr_results = ocr_full_image(column_region)

    if not ocr_results:
        return [], 0

    # Group OCR results by Y proximity (texts within ~25px = same entry)
    groups = _group_by_y(ocr_results, y_threshold=25)

    # Split groups that contain multiple actions (merged entries)
    groups = _split_multi_action_groups(groups)
    pre_collapse_count = len([g for g in groups if g])

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

    # Preflop raise-jam overlay collapse (see _collapse_preflop_raise_jam).
    if is_preflop:
        entries = _collapse_preflop_raise_jam(entries)

    # Pre-pass: drop a duplicate "All-In" entry that's just the sticker on
    # top of the same player's just-recorded bet/raise. N8 paints "All-In"
    # as a small red badge overlaid on the bet sticker when the wager is
    # for all chips. Two stickers, one action. Symptom (H2852 river):
    # the same `type` is hero/opponent on both consecutive entries; the
    # second has action=All-In, size=None, no player_name. Removing it lets
    # the bet-size carry through stack/EV accounting; without it the All-In
    # entry overwrites hero_street with size=0 and effective_bb collapses
    # from 31bb to 20bb, making the solver think the bet is the only chip
    # hero has left and labeling a 50% pot bet as all-in.
    cleaned: list[dict] = []
    for e in entries:
        if cleaned:
            prev = cleaned[-1]
            same_side = e.get("type") and e.get("type") == prev.get("type")
            is_dup_allin = (
                (e.get("action") or "") == "All-In"
                and e.get("size") is None
                and not e.get("player_name")
                and (prev.get("action") or "").lower() in ("bet", "raise", "all-in")
            )
            if same_side and is_dup_allin:
                # Promote the previous bet/raise to All-In so downstream
                # labeling (the "All-In" tag in summaries) is preserved.
                if (prev.get("action") or "").lower() in ("bet", "raise"):
                    prev["action"] = "All-In"
                continue
        cleaned.append(e)
    entries = cleaned

    # Post-pass: hero can't act twice after going all-in, so any subsequent
    # entry tagged "hero" must be the villain calling/folding to the jam.
    # The HSV detector latches onto the yellow Call sticker even when the
    # avatar sits below (showdown layout — N8 puts the calling player's
    # avatar under the sticker so the hole-card reveal lines up). Without
    # this flip, the all-in resolution gets walked as a phantom second
    # hero action. Regression: H2842.
    hero_allin_idx = None
    for i, entry in enumerate(entries):
        action = entry.get("action") or ""
        if entry.get("type") == "hero" and action == "All-In":
            hero_allin_idx = i
            continue
        if hero_allin_idx is not None and entry.get("type") == "hero":
            entry["type"] = "opponent"

    # The All-In sticker carries no number, so the entry leaves size=None.
    # Downstream stack inference (_calculate_starting_stacks) replaces
    # hero_street with the all-in size; without one, hero's full stack
    # contribution drops to 0 and effective_bb collapses (H2842 fell from
    # 30bb to 7.8bb). Recover the size from the called amount: villain's
    # last raise sets the price, and their subsequent call adds the
    # remainder = hero's jam total.
    if hero_allin_idx is not None and entries[hero_allin_idx].get("size") is None:
        last_villain_to = 0.0
        for prev in entries[:hero_allin_idx]:
            if prev.get("type") == "opponent":
                act = (prev.get("action") or "").lower()
                if act in ("raise", "bet", "all-in"):
                    last_villain_to = prev.get("size") or last_villain_to
        next_call_size = 0.0
        if hero_allin_idx + 1 < len(entries):
            nxt = entries[hero_allin_idx + 1]
            if (nxt.get("type") == "opponent"
                    and (nxt.get("action") or "").lower() == "call"):
                next_call_size = nxt.get("size") or 0.0
        inferred = last_villain_to + next_call_size
        if inferred > 0:
            entries[hero_allin_idx]["size"] = inferred

    # Final pass: settle who shoved vs who called. Runs after the
    # hero-specific dedup/flip/size-recovery above so it sees their
    # output and also catches the cross-side showdown layout they miss
    # (opponent shoves, hero calls deeper — H2881).
    entries = _resolve_allin_attribution(entries)

    return entries, pre_collapse_count


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

    # Guard against player usernames that embed "All-In" as a substring
    # (e.g. "All-In Steed" → OCR'd as "AIl-In Steed"). The OCR-tolerant
    # pattern A[lI1]{2}.?[Ii1][nNuU] matches such names too. A real action
    # sticker stands alone — once the matched text and any position/BB/
    # number tokens are stripped, no alphabetic remainder should be left.
    if action_match and _looks_like_allin_match(action_match.group(1)):
        residue = full_text.replace(action_match.group(0), " ", 1)
        residue = _ACTION_RESIDUE_STRIP_RE.sub(" ", residue)
        if re.search(r"[A-Za-z]{2,}", residue):
            action_match = None
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
    # Preprocess each OCR text through _preprocess_ocr_position so known
    # corrupt reads (UTGT → UTG1) don't fall through to the substring rule.
    position = None
    for t in group:
        text_upper = _preprocess_ocr_position(t["text"]).upper()
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

    # The All-In sticker is red, not yellow, so the HSV-based hero detector
    # always returns "opponent" for it. When the group has no opponent
    # markers (no position badge, no leading player name, no size number)
    # the sticker is the centered hero variant — flip back to hero.
    # Regression: H2842 — hero went all-in on the flop after a 3-bet line;
    # the red sticker was solo with no badge but was tagged opponent and
    # then dropped from the hero spot list.
    if action == "All-In" and entry_type == "opponent":
        if not position and not player_name and size is None:
            entry_type = "hero"

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
    elif "raise" in lower or re.match(r"^ra[iil1]se$", lower):
        return "Raise"
    elif "all" in lower:
        return "All-In"
    # Match the same shape the OCR-tolerant action regex catches: "AII-In",
    # "Al1-In", "AII-1n", etc. — same font-confusion pattern, lowercased.
    if re.match(r"^a[li1]{1,2}.?[i1][nu]$", lower):
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
        entries, pre_collapse_count = detect_entries(
            col["region"], is_preflop=(col["name"] == "Pre-Flop")
        )
        result_columns.append({
            "name": col["name"],
            "pot": col["pot"],
            "entries": entries,
            "entries_pre_collapse_count": pre_collapse_count,
        })

    return {"columns": result_columns}
