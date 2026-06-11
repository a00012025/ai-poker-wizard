"""Main N8 replay screenshot parser.

Orchestrates region detection, table parsing, panel parsing, and hand
assembly into the JSON format expected by analyze_hand_full().
"""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

from .region_detector import detect_regions
from .table_parser import parse_table
from .panel_parser import parse_panel, normalize_position

# Load config for confidence weights/threshold
_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "n8_default.json"
with open(_CONFIG_PATH) as f:
    _CONFIG = json.load(f)

_CONF_WEIGHTS = _CONFIG["confidence_weights"]
_CONF_THRESHOLD = _CONFIG["confidence_threshold"]

# effective_bb abstain floor. _compute_effective_bb returns None (abstain) when
# its confidence falls below this. Downstream degrades a None effective_bb to a
# safe solver-depth fallback, so abstaining is cheap. Tuned in a follow-up task.
_EFFBB_CONF_FLOOR = float(os.getenv("OCR_EFFBB_CONF_FLOOR", "0.7"))

# The engine's single-opponent value override corrects the legacy selection by
# reading the lone live opponent's seat — but with current (Phase-1) attribution
# the seat read is noisy enough that it is net-negative on the cache, so it is
# OFF by default. Phase 2 (robust position→seat) re-enables it. The engine's M1
# uncalled-shove ceiling (a pure panel read, no seat dependency) stays ON.
# Set OCR_EFFBB_ENGINE_OPP=1 to A/B the override back on.
_ENGINE_OPP_OVERRIDE_DISABLED = not bool(os.getenv("OCR_EFFBB_ENGINE_OPP"))

# Phase 4 — calibrated STRUCTURAL abstain. Beyond the scalar confidence floor,
# abstain whenever a structural error signal fires (calibrated on the 1,805-hand
# hero-active cache via scripts/effbb_calibrate.py, 5-fold pooled CV):
#   * geometry/heuristic binding the betting engine did NOT confirm,
#   * the engine's independent decision-local bucket DISAGREES with the emit,
#   * floors-on vs stack-only reconstructions land in different buckets,
#   * hero shoved / called all-in (displayed ~0) and the engine can't confirm.
# These isolate the layout-INDEPENDENT value errors the bucket-consensus signal
# is blind to. Held-out CV: lifts emitted precision 70.9%→76.5% at 48.8% coverage
# (from 78.2%). 99.5% is provably UNREACHABLE on single-frame inputs (the wrong
# emits are internally-consistent stack/start-vs-displayed misreads no feature
# separates; absolute ceiling ~86% @ ~10% cov — see the calibrate harness +
# docs plan Phase-4). Abstaining is cheap (None → safe generic solver depth), so
# the precision-maximizing structural gate is ON by default; set
# OCR_EFFBB_STRUCTURAL_GATE=0 to revert to the bare conf floor (70.9% @ 78.2%).
_EFFBB_STRUCTURAL_GATE = os.getenv("OCR_EFFBB_STRUCTURAL_GATE", "1") != "0"

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


def _promote_misnamed_preflop_column(
    preflop_col: dict | None,
    street_cols: list[dict],
) -> tuple[dict | None, list[dict]]:
    """Recover the physical Pre-Flop column when OCR names it ``Flop``.

    The action panel splitter returns fixed physical columns:
    ``Blinds, Pre-Flop, Flop, Turn, River``.  Header OCR sometimes labels the
    second physical column as ``Flop`` when compact all-in/fold rows overlap
    the header.  The old recovery only promoted columns with 5+ entries, which
    missed short all-in hands (2-4 visible rows) and forced destructive full
    Gemini fallback.  If no explicit Pre-Flop column exists, the first
    street-like column is still the physical Pre-Flop column, even when only a
    short shove/call sequence is visible.
    """
    if preflop_col is not None or not street_cols:
        return preflop_col, street_cols

    first_street = street_cols[0]
    first_entries = first_street.get("entries", [])
    if first_street.get("name", "").lower() == "flop" and len(first_entries) >= 2:
        return first_street, street_cols[1:]

    return preflop_col, street_cols


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


def _duplicate_known_cards(hand: dict | None) -> list[str]:
    """Return exact duplicate cards in a parsed hand.

    Natural8 screenshots can occasionally produce high-confidence but
    impossible board classifications (H2914: board contained Ks twice).  A
    duplicate exact card means the OCR parse is structurally unsafe: full
    Gemini must re-read the screenshot, not just hero cards.
    """
    if not hand:
        return []

    cards: list[str] = []
    hero_hand = hand.get("hero_hand") or ""
    if len(hero_hand) == 4:
        cards.extend([hero_hand[:2], hero_hand[2:]])

    for street in hand.get("streets") or hand.get("postflop_actions") or []:
        board = street.get("board") or street.get("cards") or street.get("card") or ""
        if isinstance(board, str):
            cards.extend(
                board[i:i + 2]
                for i in range(0, len(board) - 1, 2)
                if len(board[i:i + 2]) == 2
            )

    seen: set[str] = set()
    dupes: list[str] = []
    for card in cards:
        if card in seen and card not in dupes:
            dupes.append(card)
        seen.add(card)
    return dupes


def _board_cards_supported_by_panel(
    board_cards: list[str],
    street_cols: list[dict],
) -> list[str]:
    """Limit board-card evidence to streets that have parsed action rows.

    The table-region card detector often sees bright hero-card or table chrome
    shapes in the turn/river slots even when the action panel has no entries
    for those streets. Hero/board duplicate repair should only treat cards as
    real board blockers when the panel proves that street was dealt; otherwise
    a false extra board card can rewrite a correct high-confidence hero card.
    """
    if not board_cards:
        return []
    max_cards = 0
    for col in street_cols:
        if not col.get("entries"):
            continue
        name = str(col.get("name") or "").lower()
        if name == "flop":
            max_cards = max(max_cards, 3)
        elif name == "turn":
            max_cards = max(max_cards, 4)
        elif name == "river":
            max_cards = max(max_cards, 5)
    return board_cards[:max_cards]

def _vlm_recheck_enabled() -> bool:
    return os.environ.get("OCR_VLM_RECHECK", "").lower() in ("1", "true", "on")


def _maybe_vlm_recheck(
    image_bytes, hand, confidence_parts, diagnostics, table_result, columns,
    *, recheck_fn=None,
):
    """Apply the Phase 11.D-c VLM structural re-check when enabled.

    On a suspect hand, asks a clean VLM oracle (gemini-3.5-flash) for the
    true seat count + hero position. Three outcomes:
      - VLM agrees with the parser → keep the parse (just flag agreement).
      - VLM disagrees → re-derive via ``force_table_size``; if the re-derived
        hero_position now matches the VLM, use the corrected hand.
      - Still disagrees after re-derivation → ABSTAIN (return hand=None) so a
        confident-but-wrong structure never emits; production routes these to
        the full Gemini fallback instead.

    ``recheck_fn`` is injectable for tests; defaults to the live VLM call.
    """
    if not _vlm_recheck_enabled():
        return hand, confidence_parts, diagnostics
    if hand is None:
        if not _partial_columns_are_vlm_recoverable(columns):
            return hand, confidence_parts, diagnostics
        from .vlm_recheck import recheck_structure
        rc = (recheck_fn or recheck_structure)(image_bytes)
        if not rc:
            diagnostics["vlm_recheck"] = "no_result"
            return hand, confidence_parts, diagnostics
        hand2, cp2, diag2 = _assemble_hand(
            table_result,
            columns,
            force_table_size=rc["players_at_table"],
            force_hero_position=rc["hero_position"],
        )
        diag2["vlm_recheck"] = dict(rc)
        if hand2 is not None and hand2.get("hero_position") == rc["hero_position"]:
            diag2["vlm_recheck_outcome"] = "recovered"
            # Parse-none recovery is useful as a field-preserving fallback
            # artifact, but the 718-hand precision push showed the recovered
            # all-in tail is action-noisy (many exact cards/position but wrong
            # preflop token chains). Keep the hand attached for downstream
            # micro-routing / hints, while confidence-abstaining deterministic
            # emission until a grammar verifier can prove the action chain.
            cp2 = dict(cp2)
            cp2["ocr_confidence"] = 0.0
            return hand2, cp2, diag2
        diagnostics["vlm_recheck"] = dict(rc)
        diagnostics["vlm_recheck_outcome"] = "recover_failed"
        return hand, confidence_parts, diagnostics

    if hand is None:
        return hand, confidence_parts, diagnostics
    from .vlm_recheck import is_suspect, recheck_structure
    # Pass diagnostics so the ``reaction`` trigger mode can see
    # ``estimate_used_reaction_signal`` (the residual non-all-in structural
    # errors). The default ``allin`` mode ignores it.
    if not is_suspect({"hand": hand, "diagnostics": diagnostics}):
        return hand, confidence_parts, diagnostics
    rc = (recheck_fn or recheck_structure)(image_bytes)
    if not rc:
        diagnostics["vlm_recheck"] = "no_result"
        return hand, confidence_parts, diagnostics
    vlm_ts = rc["players_at_table"]
    vlm_pos = rc["hero_position"]
    diagnostics["vlm_recheck"] = dict(rc)
    if (hand.get("players_at_table") == vlm_ts
            and hand.get("hero_position") == vlm_pos):
        diagnostics["vlm_recheck_outcome"] = "agree"
        return hand, confidence_parts, diagnostics
    hand2, cp2, diag2 = _assemble_hand(
        table_result, columns, force_table_size=vlm_ts,
        force_hero_position=vlm_pos,
    )
    diag2["vlm_recheck"] = dict(rc)
    if hand2 is not None and hand2.get("hero_position") == vlm_pos:
        diag2["vlm_recheck_outcome"] = "corrected"
        # Focused VLM can correct seat count/position, but it does not verify
        # the action grammar.  On the precision-push test set, corrected
        # all-in tails were net-negative when emitted deterministically
        # (structure fixed, preflop tokens still wrong). Preserve the corrected
        # hand for downstream field-level fallback/hints, but confidence-
        # abstain deterministic emission until a grammar verifier accepts it.
        cp2 = dict(cp2)
        cp2["ocr_confidence"] = 0.0
        return hand2, cp2, diag2
    # Residual disagreement → confidence-abstain rather than emit a confident
    # wrong parse.  Keep the parser's hand attached so production can apply
    # field-level micro-routing (cards-only / keep OCR structure) instead of
    # destructive full-image Gemini reparse.  The VLM is a structural oracle,
    # not card evidence, so preserve card_confidence for downstream routing.
    diagnostics["vlm_recheck_outcome"] = "abstain"
    cp = dict(confidence_parts)
    cp["ocr_confidence"] = 0.0
    return hand, cp, diagnostics


def _partial_columns_are_vlm_recoverable(columns: list[dict]) -> bool:
    """Whether a parse_none has enough OCR panel signal to ask VLM structure.

    This targets the all-in/row-collapse tail where cards and action rows exist
    but assembly cannot identify hero_position.  Empty-panel parse_none remains
    a full-Gemini fallback problem.
    """
    for col in columns or []:
        if "pre" not in (col.get("name") or "").lower():
            continue
        entries = _filter_action_entries(col.get("entries") or [])
        if len(entries) < 4:
            return False
        return any(
            "all" in ((e.get("action") or "").lower())
            or (e.get("action") or "").lower() == "raise"
            for e in entries
        )
    return False


def _flag_possible_ft(hand: dict, table_color: str | None) -> None:
    """Flag a purple-felt table as a *possible* final table (ask, don't assume).

    N8 renders FT with a purple felt, but purple is not a guarantee — auto-
    committing to ICM/FT over-triggered tournament analysis (H3518). Set a
    soft ``possible_ft`` hint so the bot asks the user to confirm, unless the
    parse already resolved ICM from a stronger signal (user text keyword).
    """
    if not hand:
        return
    if table_color == "purple" and not hand.get("tournament_type"):
        hand["possible_ft"] = True


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

    # Step 2: parse table (pass full image + divider so hero localization can
    # search a band straddling the divider — hero cards are often clipped by it)
    table_result = parse_table(
        regions["table"],
        full_image=image,
        divider_y=regions.get("divider_y"),
    )

    # Step 3: parse panel
    panel_result = parse_panel(regions["panel"])
    columns = panel_result.get("columns", [])

    # Step 4: assemble hand JSON
    hand, confidence_parts, diagnostics = _assemble_hand(table_result, columns)

    # Step 4b (Phase 11.D-c): selective VLM structural re-check. Flag-gated;
    # only fires on suspect (all-in/multiway) hands. Fixes the confident
    # table-size/position errors the row-counting estimate can't catch.
    hand, confidence_parts, diagnostics = _maybe_vlm_recheck(
        image_bytes, hand, confidence_parts, diagnostics,
        table_result, columns,
    )
    _apply_structural_confidence_demotions(hand, confidence_parts, diagnostics)

    duplicate_cards = _duplicate_known_cards(hand)
    if duplicate_cards:
        diagnostics["duplicate_cards"] = duplicate_cards
        # Do not let an impossible exact-card duplicate pass the FAST or
        # cards-only Gemini paths.  This is a board/structure problem, so a
        # full image parse is required.
        confidence_parts["card_confidence"] = 0.0
        confidence_parts["ocr_confidence"] = 0.0

    # Step 5: compute confidence
    confidence = _compute_confidence(confidence_parts)
    safe_emit_reason = _safe_emit_override_reason(
        hand, confidence_parts, diagnostics
    )
    if safe_emit_reason:
        diagnostics["safe_emit_reason"] = safe_emit_reason

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
        "hero_card_details": table_result.get("hero_card_details") or [],
        "board_card_details": table_result.get("board_card_details") or [],
        "safe_emit_reason": safe_emit_reason,
    }


def _apply_structural_confidence_demotions(
    hand: dict | None,
    confidence_parts: dict,
    diagnostics: dict,
) -> None:
    """Demote known confidently-wrong structural tails before scoring.

    These are not parse failures: the parser often has a plausible hand, but
    benchmark inspection showed the shape is seat/board fragile.  Keeping the
    hand attached lets downstream field-level fallbacks use it as a hint while
    preventing the threshold gate from emitting it as deterministic OCR.
    """
    if not hand:
        return

    pre_count = diagnostics.get("preflop_entries_count")
    pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
    preloss = (
        (pre_collapse - pre_count)
        if isinstance(pre_collapse, int) and isinstance(pre_count, int)
        else 0
    )
    postflop_rows = sum(
        int(v or 0)
        for v in (diagnostics.get("street_entries_count") or {}).values()
    )
    risks: list[str] = []

    if (
        preloss >= 10
        and postflop_rows == 0
        and not diagnostics.get("estimate_used_reaction_signal")
        and not diagnostics.get("vlm_recheck_outcome")
    ):
        risks.append("large_preflop_collapse_no_postflop")

    street_counts = diagnostics.get("street_entries_count") or {}
    expected_board_streets = sum(
        1 for name in ("flop", "turn", "river")
        if int(street_counts.get(name) or 0) > 0
    )
    parsed_board_streets = sum(
        1
        for street in (hand.get("streets") or [])
        if street.get("board") or street.get("cards") or street.get("card")
    )
    if expected_board_streets > parsed_board_streets:
        risks.append("postflop_rows_without_matching_board_streets")

    if risks:
        diagnostics["structural_risk_issues"] = risks
        confidence_parts["ocr_confidence"] = 0.0


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

    # Full-column OCR can occasionally split table chrome/avatar text into a
    # nameless, positionless "Call" row near the top of otherwise fold-heavy
    # preflop columns.  In 8-max capped N8 rows this phantom limp shifts the
    # hero opener one seat toward the blinds.  Keep the guard narrow so real
    # limp pots with a readable player/name or size survive.
    if len(result) >= 8:
        cleaned: list[dict] = []
        for idx, entry in enumerate(result):
            action = (entry.get("action") or "").lower()
            is_phantom_early_call = (
                0 < idx <= 2
                and action == "call"
                and entry.get("size") is None
                and not (entry.get("position") or "").strip()
                and not (entry.get("player_name") or "").strip()
                and (result[idx - 1].get("action") or "").lower() == "fold"
                and idx + 1 < len(result)
                and (result[idx + 1].get("action") or "").lower() == "fold"
            )
            if is_phantom_early_call:
                continue
            is_phantom_late_call = (
                idx >= 3
                and action == "call"
                and entry.get("size") is None
                and not (entry.get("position") or "").strip()
                and not (entry.get("player_name") or "").strip()
                and (result[idx - 1].get("action") or "").lower() == "fold"
                and idx + 1 < len(result)
                and (result[idx + 1].get("action") or "").lower() in ("fold", "raise")
            )
            if is_phantom_late_call:
                continue
            cleaned.append(entry)
        result = cleaned

    # "Bet" is not a legal preflop action label in the action-history panel;
    # real voluntary preflop aggression is rendered as Raise/All-In.  EasyOCR
    # sometimes extracts a nameless, positionless "Bet" fragment from table
    # chrome or postflop text that bled into the physical Pre-Flop column.
    # If kept, the missing-size guard turns otherwise exact short histories
    # into parse_none.  Drop only the fully anonymous/sizeless form; sized
    # blind/ante fragments and named entries remain available to later guards.
    cleaned = []
    for idx, entry in enumerate(result):
        action = (entry.get("action") or "").lower()
        is_anonymous_sizeless_preflop_bet = (
            action == "bet"
            and entry.get("size") is None
            and not (entry.get("position") or "").strip()
            and not (entry.get("player_name") or "").strip()
        )
        is_leading_anonymous_preflop_check = (
            idx == 0
            and action == "check"
            and entry.get("size") is None
            and not (entry.get("position") or "").strip()
            and not (entry.get("player_name") or "").strip()
        )
        if is_anonymous_sizeless_preflop_bet or is_leading_anonymous_preflop_check:
            continue
        cleaned.append(entry)
    result = cleaned

    # Disambiguate false hero detections: when yellow background bleeds
    # into adjacent rows, fold entries near the real hero get marked as
    # hero too.  Reclassify hero-Fold entries as opponents when there is
    # at least one hero with a non-fold action (the real hero).
    #
    # A second, common bleed pattern in N8 all-in/fold-heavy rows is:
    # an opponent row with a readable avatar/name is tagged as hero while
    # the real hero row is an anonymous yellow action sticker.  The hero's
    # own panel row generally has no player_name OCR because the local
    # player's avatar/name is not rendered in the same way as opponents.
    # If an anonymous hero marker exists, named hero rows are much more
    # likely to be adjacent-opponent false positives.  Reclassifying them
    # before table-size estimation prevents those false markers from
    # triggering the "hero acted twice" re-action heuristic and shifting
    # hero_position toward the blinds.
    hero_indices = [i for i, e in enumerate(result) if e["type"] == "hero"]
    if len(hero_indices) > 1:
        has_anonymous_hero = any(
            not (result[idx].get("player_name") or "").strip()
            for idx in hero_indices
        )
        if has_anonymous_hero:
            for idx in hero_indices:
                if (result[idx].get("player_name") or "").strip():
                    result[idx] = dict(result[idx], type="opponent")

    hero_indices = [i for i, e in enumerate(result) if e["type"] == "hero"]
    if len(hero_indices) > 1 and all(
        (result[idx].get("action") or "").lower() == "fold"
        for idx in hero_indices
    ):
        # Multiple anonymous hero-colored Fold rows are marker bleed, not the
        # local player acting twice. Keep the first hero-fold as the real
        # seat and demote later fold markers so table-size estimation does not
        # mistake them for re-actions.
        for idx in hero_indices[1:]:
            result[idx] = dict(
                result[idx],
                type="opponent",
                _false_hero_marker=True,
            )

    # N8 paints an additional red All-In sticker over some preflop shove/call
    # resolutions. HSV sees the anonymous sticker as hero-colored even though
    # it duplicates an earlier opponent all-in amount, adding a phantom hero
    # re-action and shifting hero_position/action sequence.
    #
    # Drop this duplicate before the "non-fold hero beats fold hero" cleanup:
    # if the all-in sticker is the false hero marker, reclassifying the real
    # hero-fold row first leaves no hero row and turns otherwise parseable
    # fold-out hands into parse_none.
    cleaned: list[dict] = []
    for idx, entry in enumerate(result):
        action = (entry.get("action") or "").lower()
        is_duplicate_hero_allin_sticker = (
            entry.get("type") == "hero"
            and action == "all-in"
            and not entry.get("position")
            and not (entry.get("player_name") or "").strip()
            and entry.get("size") is None
            and any(
                prev.get("type") == "hero"
                and (prev.get("action") or "").lower() == "all-in"
                for prev in result[:idx]
            )
        )
        if is_duplicate_hero_allin_sticker:
            continue
        is_anonymous_hero_allin = (
            entry.get("type") == "hero"
            and action == "all-in"
            and not entry.get("position")
            and not (entry.get("player_name") or "").strip()
            and entry.get("size") is not None
        )
        if is_anonymous_hero_allin:
            size = float(entry.get("size") or 0.0)
            earlier_same_opponent_allin = any(
                prev.get("type") == "opponent"
                and (prev.get("action") or "").lower() == "all-in"
                and prev.get("size") is not None
                and abs(float(prev.get("size") or 0.0) - size) < 0.05
                for prev in result[:idx]
            )
            previous_is_opponent_call = (
                idx > 0
                and result[idx - 1].get("type") == "opponent"
                and (result[idx - 1].get("action") or "").lower() == "call"
            )
            earlier_nonfold_hero_action = any(
                prev.get("type") == "hero"
                and (prev.get("action") or "").lower() not in ("", "fold")
                for prev in result[:idx]
            )
            if earlier_same_opponent_allin and (
                previous_is_opponent_call or earlier_nonfold_hero_action
            ):
                continue
        cleaned.append(entry)
    result = cleaned

    # A nameless/positionless Check immediately before an explicit BB Call is
    # a duplicate blind-option OCR fragment, not a separate preflop action.
    # Keeping it creates an impossible X-C tail and shifts the action types.
    cleaned = []
    for idx, entry in enumerate(result):
        action = (entry.get("action") or "").lower()
        is_duplicate_check_before_bb_call = (
            action == "check"
            and not (entry.get("position") or "").strip()
            and not (entry.get("player_name") or "").strip()
            and entry.get("size") is None
            and idx + 1 < len(result)
            and (result[idx + 1].get("action") or "").lower() == "call"
            and (result[idx + 1].get("position") or "").strip().upper() == "BB"
        )
        if is_duplicate_check_before_bb_call:
            continue
        cleaned.append(entry)
    result = cleaned

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
    seen_names: list[tuple[str, str]] = []
    re_action_start = n  # index where re-actions begin
    for i, e in enumerate(action_entries):
        name = (e.get("player_name") or "").strip()
        if not name:
            continue
        action = (e.get("action") or "").lower()
        # Check against all previously seen names using fuzzy match
        for prev, prev_action in seen_names:
            if _fuzzy_name_match(name, prev):
                if prev_action == "fold":
                    # A folded player cannot re-act later in the same hand.
                    # Treat this as OCR name collision/bleed rather than a
                    # table-size boundary.
                    continue
                # This player already appeared — re-action detected
                re_action_start = min(re_action_start, i)
                break
        if re_action_start < n:
            break
        seen_names.append((name, action))

    # Six-handed replay columns sometimes show a seventh trailing fold after
    # an all-in/raise resolution, but OCR misses the repeated player name, so
    # the generic name-based re-action detector treats it as a seventh seat.
    # Keep this intentionally narrow: a lone explicit BB at row 5 also appears
    # in true 7-max steal/shove hands, so require either a repeated non-blind
    # trailing badge, adjacent duplicate BB badges, or a clean BTN->hero->BB
    # 6-max blind alignment.
    if re_action_start == n and n == 7:
        last = action_entries[-1]
        last_action = (last.get("action") or "").lower()
        if last_action == "fold" and not last.get("_false_hero_marker"):
            earlier_positions = {
                (e.get("position") or "").strip().upper()
                for e in action_entries[:-1]
                if (e.get("position") or "").strip()
            }
            last_pos = (last.get("position") or "").strip().upper()
            positions = [
                (e.get("position") or "").strip().upper()
                for e in action_entries
            ]
            duplicate_bb_blind_rows = (
                positions[4] == "BB" and positions[5] == "BB"
            )
            btn_hero_bb_alignment = (
                positions[3] == "BTN"
                and action_entries[4].get("type") == "hero"
                and positions[5] == "BB"
            )
            if (
                (last_pos and last_pos not in {"BTN", "SB", "BB"}
                 and last_pos in earlier_positions)
                or duplicate_bb_blind_rows
                or btn_hero_bb_alignment
            ):
                re_action_start = 6
        elif last_action != "fold":
            positions = [
                (e.get("position") or "").strip().upper()
                for e in action_entries
            ]
            actions = [(e.get("action") or "").lower() for e in action_entries]
            last_pos = positions[-1]
            hero_3bet_from_blind = (
                action_entries[5].get("type") == "hero"
                and actions[5] in {"raise", "all-in"}
                and any(a in {"raise", "all-in"} for a in actions[:5])
                and last_pos
                and last_pos not in {"BTN", "SB", "BB"}
            )
            if hero_3bet_from_blind:
                re_action_start = 6

    # Some N8 re-action rows carry the same position badge as the original
    # raiser/caller even when OCR misses the repeated player name. Treat a
    # later non-fold row with a repeated non-blind badge and a readable name as
    # the first re-action. The readable-name + non-blind guards avoid noisy
    # first-round badge bleed such as duplicated BB/SB labels on shove/call
    # rows.
    seen_positions: dict[str, list[str]] = {}
    for i, e in enumerate(action_entries):
        pos = (e.get("position") or "").strip().upper()
        action = (e.get("action") or "").lower()
        name = (e.get("player_name") or "").strip()
        if (
            i >= 6
            and pos
            and pos not in {"BTN", "SB", "BB"}
            and pos in seen_positions
            and any(prev_action != "fold" for prev_action in seen_positions[pos])
            and name
            and action != "fold"
        ):
            re_action_start = min(re_action_start, i)
            break
        if pos:
            seen_positions.setdefault(pos, []).append(action)

    # Also detect re-actions when hero acted twice (two hero entries).  This
    # can be earlier than a later repeated-name/badge signal, so always take
    # the earliest second hero row instead of only using it as a fallback.
    hero_indices = [i for i, e in enumerate(action_entries) if e["type"] == "hero"]
    if len(hero_indices) >= 2:
        re_action_start = min(re_action_start, hero_indices[1])

    # In 6-max shove trees the local player's re-action fold can be rendered
    # as an anonymous row immediately before a named caller's re-action. If
    # name matching finds the caller at row 7 of an 8-row sequence, row 6 is
    # often already the first re-action, not the final first-round seat.
    if n == 8 and re_action_start == 7:
        maybe_fold = action_entries[6]
        caller = action_entries[7]
        caller_name = (caller.get("player_name") or "").strip()
        caller_action = (caller.get("action") or "").lower()
        prior_nonfold_same_name = any(
            (prev.get("action") or "").lower() != "fold"
            and _fuzzy_name_match(caller_name, prev.get("player_name") or "")
            for prev in action_entries[:6]
        )
        if (
            (maybe_fold.get("action") or "").lower() == "fold"
            and not (maybe_fold.get("player_name") or "").strip()
            and not (maybe_fold.get("position") or "").strip()
            and caller_action != "fold"
            and caller_name
            and prior_nonfold_same_name
        ):
            re_action_start = 6

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
    # One contains the other. Very short OCR fragments (e.g. "ER") appear
    # inside many unrelated names and should not start a false re-action.
    min_len = min(len(a), len(b))
    if min_len >= 4 and (a in b or b in a):
        return True
    # Long common prefix (at least 5 chars or 70% of shorter name)
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


def _clean_seat_geometry(named_stacks: list[dict] | None) -> list[dict]:
    """Drop center-region (pot/street-header) phantom rows from named_stacks.

    N8 sometimes OCRs the central pot chip stack and the Pre-Flop/Flop/Turn/River
    column headers as ``named_stacks`` entries (e.g. ``{name:'Flop', stack:67.0}``
    sitting near the table centre, or a small pot value at mid-table). Real seats
    sit on the table perimeter. We keep only seats whose normalised distance from
    the bounding-box centre exceeds a threshold — those are the actual chairs.
    """
    import math as _math

    _HEADER_NAMES = {"blinds", "blind", "pre-flop", "preflop", "pre", "flop",
                     "turn", "river"}
    ns = [
        s for s in (named_stacks or [])
        if s.get("x") is not None and s.get("y") is not None
        and isinstance(s.get("stack"), (int, float)) and s.get("stack")
        # Panel column headers (Pre-Flop/Flop/Turn/River) leak into
        # named_stacks as a bottom row; drop them by name.
        and (s.get("name") or "").strip().lower() not in _HEADER_NAMES
    ]
    if len(ns) <= 2:
        return ns
    xs = [s["x"] for s in ns]
    ys = [s["y"] for s in ns]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    hx = (max(xs) - min(xs)) or 1.0
    hy = (max(ys) - min(ys)) or 1.0
    # Hero is the bottom-centre seat. The street-header phantom row (and the
    # central pot stack) sit near the centre OR below hero; drop both.
    hero_y = max(s["y"] for s in ns)
    kept = [
        s for s in ns
        if _math.hypot((s["x"] - cx) / hx, (s["y"] - cy) / hy) > 0.18
        and s["y"] <= hero_y + 1
    ]
    return kept or ns


def _seat_ring(named_stacks: list[dict] | None) -> list[dict]:
    """Order the physical seats clockwise starting at the hero seat.

    Hero is the bottom-centre seat (largest y in N8 layout). Seats are returned
    starting at hero and proceeding around the table by polar angle about the
    table centre. The caller resolves which angular direction matches the
    position-action order using the panel folder positions.
    """
    import math as _math

    seats = _clean_seat_geometry(named_stacks)
    if not seats:
        return []
    hero = max(seats, key=lambda s: s["y"])
    xs = [s["x"] for s in seats]
    ys = [s["y"] for s in seats]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    ordered = sorted(seats, key=lambda s: _math.atan2(s["y"] - cy, s["x"] - cx))
    hi = ordered.index(hero)
    return ordered[hi:] + ordered[:hi]


def _panel_distinct_positions(columns: list[dict]) -> set:
    """Set of distinct table positions that appear anywhere in the panel.

    N8 assigns every acting seat (incl. folders) a position, so the count of
    distinct positions is a reliable estimate of the physical player count —
    far more robust than counting OCR'd seat stickers (which include bet/pot
    chip phantoms). The BB is occasionally omitted when the open folds through.
    """
    out = set()
    for col in columns:
        for e in col.get("entries", []):
            if e.get("position"):
                out.add(e["position"])
    return out


def _reconcile_ring_to_count(ring: list[dict], target: int) -> list[dict]:
    """Trim phantom (bet/chip-sticker) seats so the ring matches ``target``.

    Extra rows are unnamed chip-amount stickers that sit INWARD of the true
    seat circle. When the ring is longer than the expected player count, drop
    the innermost name-less seats (closest to the table centre) first — those
    are the bet/pot stickers, never real chairs.
    """
    import math as _math

    if target is None or len(ring) <= target or target < 2:
        return ring
    xs = [s["x"] for s in ring]
    ys = [s["y"] for s in ring]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    hx = (max(xs) - min(xs)) or 1.0
    hy = (max(ys) - min(ys)) or 1.0

    def centrality(s):
        return _math.hypot((s["x"] - cx) / hx, (s["y"] - cy) / hy)

    extra = len(ring) - target
    # Candidate phantoms: name-less seats, most central first.
    nameless = sorted(
        [s for s in ring if not (s.get("name") or "").strip()],
        key=centrality,
    )
    drop = set(id(s) for s in nameless[:extra])
    trimmed = [s for s in ring if id(s) not in drop]
    if len(trimmed) == target:
        return trimmed
    return ring  # couldn't cleanly reconcile — leave as-is for caller to reject


def _candidate_rings(named_stacks: list[dict] | None, target: int) -> list[list[dict]]:
    """Enumerate the plausible cleaned seat-rings of length ``target``.

    Phantom-trimming is not unique: when the ring is one or two seats too long,
    several name-less central stickers are equally plausible drops. We enumerate
    the small set of plausible trims (the canonical innermost-first trim plus a
    couple of near-tie alternatives) so the top-K layout search can reason about
    them. Each returned ring is hero-anchored (``_seat_ring`` order) and exactly
    ``target`` long; an empty list means we could not reconcile.
    """
    import math as _math
    ring = _seat_ring(named_stacks)
    if not ring or target is None or target < 2:
        return []
    if len(ring) == target:
        return [ring]
    if len(ring) < target:
        return []  # can't invent seats
    xs = [s["x"] for s in ring]; ys = [s["y"] for s in ring]
    cx = (min(xs) + max(xs)) / 2.0; cy = (min(ys) + max(ys)) / 2.0
    hx = (max(xs) - min(xs)) or 1.0; hy = (max(ys) - min(ys)) or 1.0

    def centrality(s):
        return _math.hypot((s["x"] - cx) / hx, (s["y"] - cy) / hy)

    extra = len(ring) - target
    hero = max(ring, key=lambda s: s["y"])
    # Drop pool: name-less seats first (chip-stickers), then innermost named.
    nameless = [s for s in ring if not (s.get("name") or "").strip() and s is not hero]
    named_inner = [s for s in ring if (s.get("name") or "").strip() and s is not hero]
    pool = sorted(nameless, key=centrality) + sorted(named_inner, key=centrality)
    if len(pool) < extra:
        return []
    out: list[list[dict]] = []
    seen: set = set()
    # Canonical: drop the ``extra`` most central pool seats.
    # Alternatives: substitute the (extra)-th drop with the next 1-2 candidates,
    # capturing the near-tie ambiguity in which central sticker is the phantom.
    import itertools
    horizon = min(len(pool), extra + 2)
    for combo in itertools.combinations(range(horizon), extra):
        drop = set(id(pool[i]) for i in combo)
        trimmed = [s for s in ring if id(s) not in drop]
        if len(trimmed) != target:
            continue
        # re-anchor at hero (largest y) preserving angular order
        hi = trimmed.index(hero) if hero in trimmed else 0
        cand = trimmed[hi:] + trimmed[:hi]
        key = tuple(round(s["x"], 1) for s in cand) + tuple(round(s["y"], 1) for s in cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= 4:
            break
    return out


def _enumerate_layouts(
    named_stacks: list[dict] | None,
    panel_position_names: dict,
    hero_position: str | None,
    num_players: int | None,
    *,
    margin: int = 1,
    max_k: int = 6,
) -> list[dict]:
    """Top-K position→seat layouts, scored by panel-name agreement (weak).

    Enumerates over (candidate ring trim) × (both angular walk directions),
    scores each by confusable-normalized name agreement against the panel's
    position→name evidence, and returns the layouts within ``margin`` of the
    best score (capped at ``max_k``), best first. Names are WEAK evidence — the
    margin keeps near-tie layouts so the consensus gate can abstain when they
    straddle buckets. Each layout is ``{position: seat_dict}``.
    """
    order = POSITION_ORDERS.get(num_players)
    if not order or hero_position not in order:
        return []
    rings = _candidate_rings(named_stacks, len(order))
    if not rings:
        return []
    hi = order.index(hero_position)
    scored: list = []
    seen_maps: set = set()
    for ring in rings:
        if len(ring) != len(order):
            continue
        for direction in (1, -1):
            mapping = {
                order[(hi + direction * k) % len(order)]: ring[k]
                for k in range(len(ring))
            }
            sig = tuple(
                (p, round(mapping[p].get("stack") or 0, 2),
                 (mapping[p].get("name") or "")[:6])
                for p in order if p in mapping
            )
            if sig in seen_maps:
                continue
            seen_maps.add(sig)
            matches = mismatches = 0
            for pos, nm in (panel_position_names or {}).items():
                seat = mapping.get(pos)
                if seat and seat.get("name"):
                    if _fuzzy_name_match(nm, seat["name"]):
                        matches += 1
                    else:
                        mismatches += 1
            scored.append((matches - mismatches, mapping))
    if not scored:
        return []
    scored.sort(key=lambda t: -t[0])
    best = scored[0][0]
    kept = [m for s, m in scored if s >= best - margin][:max_k]
    return kept


def _map_positions_to_seats(
    named_stacks: list[dict] | None,
    panel_position_names: dict,
    hero_position: str | None,
    num_players: int | None,
) -> dict | None:
    """Map each table POSITION to its physical seat (a named_stacks dict).

    Builds a seat ring from geometry (``_seat_ring``), anchors hero at
    ``hero_position``, then walks the position-action order in BOTH angular
    directions, scoring each candidate by agreement with the panel's
    position→player-name evidence (folders carry a reliable position+name). The
    higher-scoring direction wins. Returns ``None`` when the seat count does not
    match the table size (ambiguous — caller should fall back / abstain).

    ``panel_position_names``: {position_str -> player_name} for NON-hero seats
    whose position the panel reported (folders, callers, raisers).

    Thin wrapper over ``_enumerate_layouts`` returning the single best layout;
    kept for the walkover / dead-code call sites. The consensus path in
    ``_compute_effective_bb`` calls ``_enumerate_layouts`` directly to reason
    over the top-K.
    """
    layouts = _enumerate_layouts(
        named_stacks, panel_position_names, hero_position, num_players,
        margin=0, max_k=1,
    )
    return layouts[0] if layouts else None


def _panel_position_names(columns: list[dict]) -> dict:
    """Extract {position -> player_name} for non-hero seats from the panel."""
    out = {}
    for col in columns:
        for e in col.get("entries", []):
            pos = e.get("position")
            nm = e.get("player_name")
            if pos and e.get("type") != "hero" and nm:
                out.setdefault(pos, nm)
    return out


def _engine_streets(columns: list[dict]) -> tuple:
    """Split panel columns into the engine's {street: entries} dict + pot map."""
    streets: dict = {}
    pot: dict = {}
    for col in columns:
        nm = (col.get("name") or "").lower()
        key = None
        if "blind" in nm:
            continue
        if "pre" in nm:
            key = "preflop"
        elif nm in ("flop", "turn", "river"):
            key = nm
        if key:
            streets[key] = col.get("entries", [])
            pot[key] = col.get("pot")
    # Recover a Pre-Flop column misnamed "Flop" (mirror of the main locator).
    if "preflop" not in streets and "flop" in streets:
        streets = {("preflop" if k == "flop" else k): v
                   for k, v in streets.items()}
        pot = {("preflop" if k == "flop" else k): v for k, v in pot.items()}
    return streets, pot


def _effective_bb_for_layout(
    columns: list[dict],
    hero_stack_displayed: float | None,
    hero_position: str | None,
    all_stacks: list[float] | None,
    named_stacks: list[dict] | None = None,
    num_players: int | None = None,
    _seat_map: dict | None = None,
    _disable_floors: bool = False,
) -> tuple:
    """Compute effective_bb under ONE fixed position→seat layout.

    Returns a 3-tuple ``(effective_bb, hero_starting_stack, confidence)``.
    ``effective_bb`` is ``None`` when we abstain (low confidence / no data);
    downstream degrades a ``None`` to a safe solver-depth fallback, so
    abstaining is cheap and correct.

    ``_seat_map`` (Phase 2): a fixed ``{position: seat_dict}`` layout to use for
    every geometry attribution in this call. When ``None`` the function derives
    the single best layout internally (legacy behaviour). The consensus
    orchestrator ``_compute_effective_bb`` calls this once per plausible layout
    and gates on whether the resulting depth buckets agree.

    In N8 replays, displayed stacks = starting - permanently_invested.
    The pot is shown separately in the table centre, so this equation
    holds for BOTH the winner and the loser(s).

    effective_bb = min(hero_starting, shortest_active_villain_starting),
    computed over ALL villains that entered preflop and did not fold
    preflop (so a folded short stack can't undershoot, and a deep seat
    can't mask the binding short villain).
    """
    if hero_stack_displayed is None:
        return None, None, 0.0

    # ---- Physical table size (for position/geometry seat attribution) ----
    # Prefer the caller's count; otherwise infer from the panel's distinct
    # position set (every acting seat — incl. folders — gets a position, so
    # this counts players far more reliably than seat stickers, which include
    # bet/pot chip phantoms). The BB is sometimes omitted on a fold-through, so
    # also consider the cleaned seat-ring length and take the larger plausible.
    if num_players is None:
        _panel_pos = _panel_distinct_positions(columns)
        if _panel_pos:
            # Smallest table size whose position set COVERS every observed
            # panel position. The panel omits some seats (a fold-through BB is
            # often not shown), so a raw distinct count under-counts; but the
            # SET of positions pins the table size because position names are
            # table-size-specific (e.g. an LJ rules out 5-max). This is far
            # more robust than counting seat stickers (bet/pot phantoms) or the
            # raw panel count. (TM5863068088: panel {LJ,HJ,CO,BTN,SB} → 6-max,
            # not 5; the unshown BB is the binding seat behind a HJ open.)
            for _sz in range(2, 10):
                _order = POSITION_ORDERS.get(_sz)
                if _order and _panel_pos.issubset(set(_order)):
                    num_players = _sz
                    break
        if num_players is None:
            _ring_seats = _seat_ring(named_stacks)
            if _ring_seats:
                num_players = min(max(len(_ring_seats), 2), 9)
                if num_players not in POSITION_ORDERS:
                    num_players = None

    # ---- Locate columns ----
    # ---- Betting-state engine (Phase 1) ----
    # Replay the panel as a real betting game to choose the RIGHT relevant
    # position(s) by action order, independent of name/geometry. The engine is
    # advisory: the legacy reconstruction below stays the safety net; when the
    # engine resolves a clean decision we let it pick the binding position(s),
    # then read those seats' stacks via name/geometry attribution.
    engine_result = None
    if hero_position and num_players and not globals().get("_DISABLE_ENGINE"):
        try:
            from . import effbb_engine as _eng
            _eng_streets, _eng_pot = _engine_streets(columns)
            engine_result = _eng.analyze(
                _eng_streets, num_players, hero_position,
                _eng_pot.get("preflop"),
            )
        except Exception:  # pragma: no cover - engine must never crash parsing
            engine_result = None

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

    preflop_col, street_cols = _promote_misnamed_preflop_column(
        preflop_col, street_cols
    )

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
    # Positions of opponents who entered AND did not later fold preflop — the
    # active villains whose stacks bind the effective. Used by the geometry
    # fallback when names are None/garbled. (preflop fold removes a seat.)
    opp_entered_positions = []   # all who voluntarily entered preflop
    opp_folded_positions = set()

    if preflop_col:
        entries = preflop_col.get("entries", [])
        current_bet = 1.0  # BB level

        for entry in entries:
            action = (entry.get("action") or "").lower()
            size = entry.get("size") or 0.0
            is_hero = entry.get("type") == "hero"

            if action == "fold":
                if not is_hero and entry.get("position"):
                    opp_folded_positions.add(entry["position"])
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
                    if entry.get("position"):
                        opp_entered_positions.append(entry["position"])
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

    # Per-villain investment to remain in the pot (for min-over-villains).
    # Starts at the preflop bet level a continuing villain matched (= the
    # final preflop bet level, e.g. a villain's 3bet size). Postflop we add
    # the SHARED matched level per street (hero_matched), NOT opp_matched —
    # opp_matched aggregates every caller in a multiway pot and double-counts
    # a single villain. This respects a villain who raised and hero folded
    # (their preflop level), and stays single-villain in multiway pots.
    opp_postflop_matched = 0.0

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
                    # Hero folded: an UNCALLED hero bet is returned, so hero
                    # only permanently invested what actually entered the pot
                    # this street (bounded by the pot delta). Without this,
                    # a hero "Bet X then Fold" mislabel (delta≈0) inflates
                    # hero_starting by X. (TM5862907992 turn 4.56 bet+fold.)
                    hero_matched = max(0.0, min(hero_street, delta))
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
            # Single continuing villain matched the shared street level.
            opp_postflop_matched += hero_matched
        else:
            # Last street or no pot headers — use entry-based logic
            if last_action == "fold":
                if last_is_hero:
                    hero_perm += hero_street
                    opp_perm += min(opp_street, hero_street)
                    opp_postflop_matched += min(opp_street, hero_street)
                else:
                    opp_perm += opp_street
                    hero_perm += min(hero_street, opp_street)
                    opp_postflop_matched += min(hero_street, opp_street)
            elif last_action == "call":
                if last_is_hero:
                    hero_perm += hero_street
                    opp_perm += opp_street
                    opp_postflop_matched += min(hero_street, opp_street)
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
                    opp_postflop_matched += min(hero_street, opp_street)
            else:
                hero_perm += hero_street
                opp_perm += opp_street
                opp_postflop_matched += min(hero_street, opp_street)

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

        elif last_action == "call" and last_is_hero and opp_total >= hero_total - 0.5:
            # Case 3: opp shoved (explicit All-In bet/raise) and hero CALLED
            # in full. Opp's displayed remaining stack is ~0, so their starting
            # stack must come from their total investment (opp_perm), not the
            # table stack — which the name/heuristic branch would otherwise
            # read (and N8 frequently misreads it as the pot chips, inflating
            # effective_bb wildly). H3514: hero called BB's 5.8bb river shove;
            # Gao zU's stack was misread as the 18.9bb pot, giving eff 29bb
            # instead of ~9bb. opp_allin_display stays None ⇒ opp_starting =
            # opp_perm.
            opp_has_allin = any(
                (e.get("action") or "").lower() == "all-in"
                and e.get("type") != "hero"
                for e in entries
            )
            if opp_has_allin:
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

    # ---- Matched all-in floor ----
    # When a player is all-in for S and another non-folding player matches it,
    # S caps the effective stack of that confrontation (once the short stack is
    # in, no more chips move between them). The smallest CALLED all-in size is
    # therefore an upper bound on effective_bb. This captures the large all-in
    # population (preflop shove wars, short-stack jams) that the displayed-stack
    # reconstruction otherwise misses (names often None ⇒ no name match).
    # We require the all-in to be MATCHED so an uncalled (folded-to) shove does
    # not pull effective_bb down to a stack that never went to showdown.
    matched_allin_floor = None
    _all_cols = ([preflop_col] if preflop_col else []) + list(street_cols)
    # A real all-in player cannot act again. Collect (type, name) that act on
    # a LATER street than their all-in, so a mislabeled "All-In" bet (the
    # player keeps acting) doesn't pull effective_bb down. (TM5863569047 flop
    # "All-In 12.27" then turn "All-In 23.09" — the flop one isn't a real shove.)
    _allin_actor_street = {}
    _acted_street = {}
    for sidx, col in enumerate(_all_cols):
        for i, e in enumerate(col.get("entries", []) if col else []):
            ea = (e.get("action") or "").lower()
            if ea in ("fold", "check"):
                continue
            key = (e.get("type"), e.get("player_name"))
            if e.get("player_name") is None:
                continue  # can't track unnamed across streets reliably
            _acted_street.setdefault(key, []).append(sidx)
            if ea == "all-in":
                _allin_actor_street[key] = sidx
    _false_allin = {
        k for k, s in _allin_actor_street.items()
        if any(st > s for st in _acted_street.get(k, []))
    }
    # Hero is a unique stream even when unnamed: a hero all-in followed by ANY
    # later hero action means the earlier "All-In" was a mislabeled bet.
    # (TM5863569047: flop hero "All-In 12.27" then turn hero "All-In 23.09".)
    _hero_streets = [sidx for sidx, col in enumerate(_all_cols)
                     if col and any(e.get("type") == "hero"
                                    and (e.get("action") or "").lower() not in ("fold", "check")
                                    for e in col.get("entries", []))]
    _hero_allin_streets = [sidx for sidx, col in enumerate(_all_cols)
                           if col and any(e.get("type") == "hero"
                                          and (e.get("action") or "").lower() == "all-in"
                                          for e in col.get("entries", []))]
    _false_hero_allin_streets = {
        s for s in _hero_allin_streets if any(hs > s for hs in _hero_streets)
    }
    # Prior cross-street investment per actor (named, + hero by type). A
    # shover's STARTING stack = shove size (their remaining) + what they
    # already committed on earlier streets. (H3514: BB shoves 5.8 on the
    # river but had invested ~3.0 before ⇒ starting 8.8, not 5.8.)
    _prior_invest = {}  # actor-key -> committed total before current street
    # Seed with posted blinds/antes so a blind player's shove floor includes
    # the chips they already had in. (H3514: BB posted 1.0 before calling.)
    if blinds_col:
        for e in blinds_col.get("entries", []):
            ea = (e.get("action") or "").lower()
            es = e.get("size") or 0.0
            if es and ("sb" in ea or "bb" in ea or "ante" in ea or "blind" in ea):
                actor = (e.get("type"), e.get("player_name"))
                # blind call entries in preflop also add the call increment;
                # the blind seeds the base level only.
                _prior_invest[actor] = max(_prior_invest.get(actor, 0.0), es)
    for _sidx, col in enumerate(_all_cols):
        entries = col.get("entries", []) if col else []
        # Highest committed amount on this street by any single player, and
        # the set of total commitments, to decide what got matched.
        running = {}       # id -> running total this street (calls additive)
        order = []
        for i, e in enumerate(entries):
            ea = (e.get("action") or "").lower()
            es = e.get("size") or 0.0
            if ea in ("fold", "check"):
                continue
            key = (e.get("type"), e.get("player_name"), i if e.get("player_name") is None else None)
            if ea in ("raise", "all-in", "bet"):
                running[key] = es  # raise/bet-to replaces; shove sets level
            elif ea == "call":
                running[key] = running.get(key, 0.0) + es
            order.append((key, ea, es, i))
        for key, ea, es, i in order:
            if ea == "all-in" and es and es > 0:
                actor = (key[0], key[1]) if len(key) >= 2 else key
                if actor in _false_allin:
                    continue  # player acted again later — not a real all-in
                if key[0] == "hero" and _sidx in _false_hero_allin_streets:
                    continue  # hero acted again on a later street — mislabel
                # Was this shove matched by another player to >= its size?
                shove = es
                matched = any(
                    k2 != key and v2 >= shove - 0.5
                    for k2, v2 in running.items()
                )
                if matched:
                    # starting = remaining (shove) + prior cross-street invest
                    floor = shove + _prior_invest.get(actor, 0.0)
                    if matched_allin_floor is None or floor < matched_allin_floor:
                        matched_allin_floor = floor
        # Accumulate this street's commitments into prior-investment totals.
        for key, total in running.items():
            actor = (key[0], key[1]) if len(key) >= 2 else key
            _prior_invest[actor] = _prior_invest.get(actor, 0.0) + total

    # ---- Uncalled preflop shove ceiling ----
    # When hero VOLUNTARILY invested preflop (raise/call, but did NOT shove)
    # and an opponent then jams all-in that hero folds to (the shove is
    # uncalled — everyone after folds), the shover's whole stack went in, so
    # the shove size is that villain's STARTING stack and an upper bound on the
    # effective stack of the spot. With several opponent jams, the SHORTEST one
    # binds (the smallest committed villain). The displayed-stack reconstruction
    # misses this because the shover's table stack reads ~0 and the names are
    # often None. (TM5863067496: hero SB R6.7, BB jams 20.4, hero folds — GT
    # eff 20.4; function returned 36.7.) This is NARROW on purpose: it does not
    # fire when hero himself shoved (that case is the matched_allin_floor).
    uncalled_shove_ceiling = None
    # When HERO jams preflop and everyone folds (uncalled), hero's shove size
    # IS hero's whole starting stack — a direct panel read, more reliable than
    # the displayed+reconstruction estimate (which can drift on a misread call
    # size). We fold it into hero_starting as an authoritative value.
    hero_uncalled_shove = None
    _has_postflop = any(
        (c.get("name") or "").lower() in ("flop", "turn", "river")
        and c.get("entries")
        for c in columns
    )
    if preflop_col:
        pf_entries = preflop_col.get("entries", [])
        _hero_jam = None
        for _i, _e in enumerate(pf_entries):
            if ((_e.get("action") or "").lower() == "all-in"
                    and _e.get("type") == "hero" and _e.get("size")):
                _hero_jam = (_i, _e)
        if _hero_jam and not _has_postflop:
            _ji, _je = _hero_jam
            if all((a.get("action") or "").lower() == "fold"
                   for a in pf_entries[_ji + 1:]):
                hero_uncalled_shove = _je["size"]
    if preflop_col:
        pf_entries = preflop_col.get("entries", [])
        hero_shoved_pf = any(
            (e.get("action") or "").lower() == "all-in"
            and e.get("type") == "hero"
            for e in pf_entries
        )
        hero_invested_pf = any(
            (e.get("action") or "").lower() in ("raise", "call", "bet")
            and e.get("type") == "hero"
            for e in pf_entries
        )
        if hero_invested_pf and not hero_shoved_pf:
            opp_jams = [
                (i, e) for i, e in enumerate(pf_entries)
                if (e.get("action") or "").lower() == "all-in"
                and e.get("type") != "hero"
                and e.get("size")
            ]
            if opp_jams:
                last_i = opp_jams[-1][0]
                after = pf_entries[last_i + 1:]
                # Uncalled: nobody called/raised after the final jam.
                if all((a.get("action") or "").lower() == "fold" for a in after):
                    uncalled_shove_ceiling = min(e["size"] for _, e in opp_jams)

    # ---- Compute starting stacks ----
    hero_starting = hero_stack_displayed + hero_perm
    # A hero uncalled-jam size is hero's authoritative starting stack.
    if hero_uncalled_shove is not None:
        hero_starting = hero_uncalled_shove

    # ---- Hero all-in / stack≈0 starting-stack reconstruction (Phase 3) ----
    # When hero's DISPLAYED stack reads ~0, hero is committed (all-in, or called
    # a villain's all-in for their whole stack): hero's STARTING stack is exactly
    # what hero permanently put in. The legacy displayed+walk estimate
    # (hero_perm) is noisy on these — a misread call/raise size over- or
    # under-adds (TM5875510185: hero SB called a shove for 13.19, legacy walk
    # over-computed hero_starting to 19.6 vs the true ~13.7; the engine's
    # decision-local hero contribution reads 13.69). The betting engine's
    # per-position contribution is an INDEPENDENT reconstruction of the same
    # amount; we prefer it when both agree on the depth bucket, and flag a
    # disagreement so the orchestrator can abstain (the input-bound residual:
    # TM5874977534 hero shows a partial 1.24 shove sticker for a true 11.2 stack
    # — neither read recovers it). ``hero_allin_recon_disagree`` is surfaced via
    # a confidence cap so the consensus gate drops the unrecoverable ones.
    hero_allin_recon = None
    hero_allin_recon_disagree = False
    if (hero_stack_displayed is not None and hero_stack_displayed <= 0.6
            and hero_uncalled_shove is None
            and engine_result is not None and hero_position):
        eng_hero_contrib = engine_result.contribution.get(hero_position)
        # Fire only when hero is genuinely committed: the engine saw hero shove,
        # or hero's last action was a Call that matched a villain all-in.
        hero_committed = engine_result.hero_all_in
        if not hero_committed:
            for col in ([preflop_col] if preflop_col else []) + list(street_cols):
                ents = col.get("entries", []) if col else []
                for k, e in enumerate(ents):
                    if e.get("type") == "hero" and (e.get("action") or "").lower() == "call":
                        # A call with a villain all-in anywhere this street = hero
                        # committed for the displayed-0 remaining.
                        if any((o.get("action") or "").lower() == "all-in"
                               and o.get("type") != "hero" for o in ents):
                            hero_committed = True
        if hero_committed and eng_hero_contrib and eng_hero_contrib >= 1.0:
            if _depth_bucket(eng_hero_contrib) == _depth_bucket(hero_starting):
                # Both reconstructions agree on the bucket → trust the engine's
                # exact contribution (less noisy than displayed≈0 + walk).
                hero_allin_recon = eng_hero_contrib
                hero_starting = eng_hero_contrib
            else:
                # The legacy walk and the engine disagree on hero's committed
                # amount and hero's displayed read is uninformative (~0): we
                # cannot reconstruct it from this frame → mark for abstain.
                hero_allin_recon_disagree = True

    # ---- Pot-bounded over-compute guard (computed early so walkover honours it) ----
    # A player's PERMANENT investment cannot exceed the chips that actually
    # entered the pot. The largest observed street-pot header bounds the total
    # contributions, so start = displayed + investment must satisfy
    #   start <= displayed + pot_bound.
    # OCR garbage (e.g. a "Call 77.0" misread) inflates hero_perm past any
    # physical pot; such an estimate is dropped. (TM5875583251: hero displayed
    # 56.97, action-walk adds ~80bb ⇒ 137.7, but the largest pot is ~8.9.)
    pot_values = [p for p in pot_by_street.values()
                  if isinstance(p, (int, float)) and p > 0]
    pot_bound = max(pot_values) if pot_values else None
    # Street-start headers EXCLUDE the final street's action (no header comes
    # after it), so add the final action street's MATCHED contribution on top.
    # Uncalled chips don't count (they're returned), so a lone uncalled shove
    # does NOT raise the bound. (H3514 river: shove 5.8 + call 5.8 ⇒ +11.6,
    # so hero's legit 8.8 invest is within bound; TM5875583251 river all-in is
    # uncalled ⇒ +0, leaving the garbled 80bb invest correctly out of bound.)
    if pot_bound is not None:
        last_matched = 0.0
        for col in reversed(_all_cols):
            ents = [e for e in (col.get("entries", []) if col else [])
                    if (e.get("action") or "").lower() not in ("fold", "check")]
            if not ents:
                continue
            hero_c = 0.0
            opp_c = 0.0
            for e in ents:
                ea = (e.get("action") or "").lower()
                es = e.get("size") or 0.0
                if e.get("type") == "hero":
                    hero_c = es if ea in ("raise", "all-in") else hero_c + es
                else:
                    opp_c = es if ea in ("raise", "all-in") else opp_c + es
            matched = min(hero_c, opp_c) if hero_c > 0 and opp_c > 0 else 0.0
            last_matched = matched * 2  # both sides' matched chips enter the pot
            break
        pot_bound = pot_bound + last_matched
    over_compute = (
        pot_bound is not None
        and (hero_starting - hero_stack_displayed) > pot_bound + 0.5
    )

    nameless_fallback = False
    geometry_pinned = False
    hero_start_rounded = round(hero_starting, 1) if hero_starting >= 1.0 else None
    if not opp_entered:
        # Walkover: hero opened (or limped) and everyone folded through. The
        # GTO spot is hero vs the players STILL TO ACT behind the opener, so the
        # effective stack is min(hero_start, the shortest seat from hero's
        # position through the BB). The old code returned hero's OWN stack,
        # which is almost always too deep (a short BB/blind behind binds the
        # spot). We resolve those seats by mapping table positions to physical
        # seats via geometry (names are often None), reading each seat's
        # displayed stack. (TM5863067852 HJ-open: GT 10.7 = a short seat behind,
        # not hero's 24.8; TM5863068088 GT 13.8.)
        if over_compute and not (
            matched_allin_floor is not None
            and matched_allin_floor <= (pot_bound or 0) + 0.5
        ):
            return (None, hero_start_rounded, 0.0)

        walkover_eff = hero_start_rounded
        walkover_conf = 0.55  # hero-own-stack fallback is usually too deep
        if hero_position and num_players:
            seat_map = _seat_map if _seat_map is not None else _map_positions_to_seats(
                named_stacks, _panel_position_names(columns),
                hero_position, num_players,
            )
            order = POSITION_ORDERS.get(num_players)
            if seat_map and order and hero_position in order:
                behind = order[order.index(hero_position):]  # hero + later seats
                behind_stacks = [
                    seat_map[p]["stack"]
                    for p in behind
                    if p in seat_map and seat_map[p].get("stack")
                ]
                # Need to actually see the seats behind hero to bind the spot;
                # require >=1 opponent seat resolved (hero alone learns nothing).
                if len(behind_stacks) >= 2 and hero_start_rounded:
                    cand = min(behind_stacks + [hero_start_rounded])
                    walkover_eff = round(cand, 1)
                    # Confident: geometry resolved all seats behind hero.
                    walkover_conf = (
                        1.0 if len(behind_stacks) >= len(behind) else 0.85
                    )

        if matched_allin_floor is not None and walkover_eff:
            return (round(min(walkover_eff, matched_allin_floor), 1),
                    hero_start_rounded, max(walkover_conf, 1.0))
        if walkover_eff:
            return (walkover_eff, hero_start_rounded, walkover_conf)
        return (hero_start_rounded, hero_start_rounded, 0.55)

    # ---- Determine opponent starting stack: min over ALL active villains ----
    # An "active villain" entered preflop (call/raise/bet/all-in) and did not
    # fold preflop. For each, estimate start = displayed + investment and take
    # the MIN, so a deep seat can't mask the binding short villain and a seat
    # that folded preflop can't undershoot. opp_perm is the modeled continuing
    # opponent's investment — an upper bound on any single active villain's
    # contribution, so displayed + opp_perm is a safe per-villain start.
    name_matched_villain = False
    engine_pinned = False
    engine_opp_candidate = None
    if opp_went_allin:
        if opp_allin_display is not None:
            # Opponent went all-in and we know their display (uncalled portion)
            opp_starting = opp_allin_display + opp_perm
        else:
            # Opponent went all-in: starting = total investment (display ≈ 0)
            opp_starting = opp_perm
    else:
        # Names of villains who entered and did not fold preflop. Prefer the
        # full entered set (every contesting villain bounds the effective
        # stack); postflop names alone would miss a villain who entered and
        # folded on a later street while still being the shortest stack.
        active_opp_names = list(opp_names_entered)

        # Per-villain investment to remain = preflop bet level a continuing
        # villain matched + the shared matched level each postflop street.
        # This respects a villain who 3bet and hero folded (preflop level),
        # and avoids the multiway double-count baked into opp_perm.
        per_villain_invest = opp_preflop_total + opp_postflop_matched

        # A villain's investment likewise cannot exceed the pot, so cap the
        # modeled per-villain investment by the pot bound (drops a runaway
        # per_villain_invest from OCR-garbled bet sizes).
        capped_invest = per_villain_invest
        if pot_bound is not None and capped_invest > pot_bound + 0.5:
            capped_invest = pot_bound

        # ---- Engine-driven relevant-opponent attribution (Phase 1) ----
        # The betting engine froze a decision-local live set by action order. We
        # use it as a SELECTION corrector for the multiway case the legacy
        # name-match gets wrong: when hero FOLDS facing a known live set and a
        # SINGLE relevant opponent survives, the legacy min-over-all-entered set
        # both over-includes folded villains AND can match a stale name to the
        # wrong seat. In that narrow, low-risk case we read the one relevant
        # seat by NAME (reliable) + its exact engine contribution. We do NOT
        # override the broad multiway value reconstruction (its tuned heuristics
        # beat a geometry-read seat — that is Phase 2's job). (TM5863067607:
        # relevant={SB}, the limper, name-resolved — not the wrong seat.)
        ppn = _panel_position_names(columns)
        if (engine_result is not None
                and engine_result.hero_folded
                and len(engine_result.relevant_opponents) == 1
                and not over_compute and not _ENGINE_OPP_OVERRIDE_DISABLED):
            pos = engine_result.relevant_opponents[0]
            nm = ppn.get(pos)
            seat = None
            if nm and named_stacks:
                for ns in named_stacks:
                    sn = ns.get("name")
                    if sn and _fuzzy_name_match(nm, sn) and ns.get("stack"):
                        seat = ns["stack"]
                        break
            if seat is not None:
                contrib = engine_result.contribution.get(pos, 0.0)
                if pot_bound is not None and contrib > pot_bound + 0.5:
                    contrib = pot_bound
                engine_opp_candidate = seat + contrib

        villain_starts = []
        matched_names = set()
        if not engine_pinned and active_opp_names and named_stacks:
            for opp_name in active_opp_names:
                for ns in named_stacks:
                    nm = ns.get("name")
                    if nm and _fuzzy_name_match(opp_name, nm):
                        villain_starts.append(ns["stack"] + capped_invest)
                        matched_names.add(nm)
                        break

        if engine_pinned:
            pass
        elif villain_starts:
            opp_starting = min(villain_starts)
            name_matched_villain = True
        else:
            # Name matching failed (names None/garbled). Before guessing the
            # shortest seat, try POSITION/GEOMETRY attribution: map the active
            # villains' positions (entered preflop, not folded preflop) to
            # physical seats and read each binding villain's DISPLAYED stack +
            # its modeled investment. This pins the RIGHT seat instead of the
            # shortest arbitrary one. (TM5863067607: the active villain is a
            # None-named 16.4 seat, not the misread 2.9 SebFerra seat.)
            # Prefer the engine's decision-local live set (excludes villains who
            # folded on a LATER street, which the preflop-only filter keeps and
            # which then undershoots). Fall back to the preflop-entered set.
            active_positions = [
                p for p in opp_entered_positions
                if p not in opp_folded_positions
            ]
            geo_positions = active_positions
            geo_starts = []
            if geo_positions and num_players and hero_position:
                seat_map = _seat_map if _seat_map is not None else _map_positions_to_seats(
                    named_stacks, _panel_position_names(columns),
                    hero_position, num_players,
                )
                if seat_map:
                    for p in set(geo_positions):
                        seat = seat_map.get(p)
                        if seat and seat.get("stack"):
                            # Use the engine's per-position contribution when we
                            # have it (exact), else the shared modeled invest.
                            inv = capped_invest
                            if (engine_result is not None
                                    and p in engine_result.contribution):
                                inv = engine_result.contribution[p]
                                if pot_bound is not None and inv > pot_bound + 0.5:
                                    inv = pot_bound
                            geo_starts.append(seat["stack"] + inv)
            if geo_starts:
                opp_starting = min(geo_starts)
                geometry_pinned = True
            else:
                # Last resort — shortest plausible non-hero seat. A single
                # caller is usually the short one; picking the largest (old
                # code) collapsed to hero's own stack. Unreliable → low conf.
                nameless_fallback = True
                non_hero_stacks = list(all_stacks) if all_stacks else []
                if hero_stack_displayed is not None and hero_stack_displayed in non_hero_stacks:
                    non_hero_stacks.remove(hero_stack_displayed)
                if non_hero_stacks:
                    opp_starting = min(s + capped_invest for s in non_hero_stacks)
                else:
                    opp_starting = hero_starting

    # When HERO jams uncalled, the opponents all FOLDED — none committed, so a
    # folder's reconstructed (and often misread-short) stack must not undercut
    # hero's shove. The effective is hero's shove size, capped only by a genuine
    # all-in floor. (TM5866594919: hero jams 22.9, opener folds; the opener's
    # seat misread to 17 wrongly bound it — GT is 22.9.)
    if hero_uncalled_shove is not None:
        opp_starting = hero_starting

    # ---- Engine single-opponent selection correction (downward-only) ----
    # When the engine name-resolved the lone live opponent at hero's fold, use
    # it ONLY if it does not INFLATE the legacy opp_starting. The legacy
    # min-over-all-entered set over-includes a deep seat or matches a wrong
    # (deeper) name; the engine's decision-local single seat corrects that
    # downward. We never let the engine raise opp_starting — an inflated engine
    # value means the engine's name→seat read landed on a deep seat (the
    # attribution noise Phase 2 fixes), so we don't trust it upward.
    if (engine_opp_candidate is not None
            and engine_opp_candidate <= opp_starting + 0.05):
        opp_starting = engine_opp_candidate
        engine_pinned = True
        name_matched_villain = True

    if _disable_floors:
        # Stack-only reconstruction cross-check (no panel-shove floors / M1
        # ceiling). The orchestrator compares this against the floor-inclusive
        # estimate: agreement on the depth bucket is the cross-method consensus
        # signal; disagreement means the all-in size / ceiling is misread or
        # mis-attributed → abstain.
        matched_allin_floor = None
        uncalled_shove_ceiling = None
    all_starting = [hero_starting, opp_starting]
    if matched_allin_floor is not None:
        # A matched all-in caps the effective stack of the confrontation.
        all_starting.append(matched_allin_floor)
    if uncalled_shove_ceiling is not None:
        # An uncalled villain jam hero folded to: the jam size is the villain's
        # whole (starting) stack and bounds the effective stack.
        all_starting.append(uncalled_shove_ceiling)
    # Engine M1: an uncalled-shove ceiling read straight off the panel. It is an
    # UPPER bound (the shover's whole stack), so it can only lower effective —
    # safe to add even when the legacy ceilings already fired. Suppress when
    # hero's own reconstruction over-computed (then the ceiling could be the
    # only sane value, handled by the over_compute recovery below).
    engine_m1_ceiling = (
        engine_result.rule_ceiling
        if (engine_result is not None and engine_result.rule == "M1")
        else None
    )
    if _disable_floors:
        engine_m1_ceiling = None
    if engine_m1_ceiling is not None and not over_compute:
        all_starting.append(engine_m1_ceiling)
    effective_bb = round(min(all_starting), 1)

    # Which source produced the binding (minimum) starting stack? Confidence is
    # driven by the CERTAINTY of that attribution, not by a second estimator
    # that eats the same inputs.
    _bind = round(min(all_starting), 1)
    binding_from_allin = (
        (matched_allin_floor is not None
         and round(matched_allin_floor, 1) <= _bind + 0.05)
        or (uncalled_shove_ceiling is not None
            and round(uncalled_shove_ceiling, 1) <= _bind + 0.05)
        or (engine_m1_ceiling is not None
            and round(engine_m1_ceiling, 1) <= _bind + 0.05)
    )
    binding_from_hero = round(hero_starting, 1) <= _bind + 0.05
    binding_from_opp = round(opp_starting, 1) <= _bind + 0.05

    # If hero's reconstruction over-computed, a hard physical ceiling (matched
    # all-in OR an uncalled-jam size, both bounded by real shove sizes) may
    # still be trustworthy and lower than the bogus hero/opp starts. Otherwise
    # we have no consistent estimate.
    _physical_floor = None
    for _f in (matched_allin_floor, uncalled_shove_ceiling):
        if _f is not None and (_physical_floor is None or _f < _physical_floor):
            _physical_floor = _f
    if over_compute:
        if (_physical_floor is not None
                and _physical_floor <= (pot_bound or 0) + 0.5
                and _physical_floor < hero_starting):
            effective_bb = round(_physical_floor, 1)
            over_compute = False  # recovered a physical estimate
        else:
            # Abstain: hero_starting is physically impossible and nothing else
            # constrains the effective stack.
            return (None, round(hero_starting, 1) if hero_starting >= 1.0 else None, 0.0)

    if effective_bb < 1.0:
        if all_stacks:
            return round(min(all_stacks), 1), round(hero_starting, 1), 0.6
        return (None,
                round(hero_starting, 1) if hero_starting >= 1.0 else None,
                0.0)

    # ---- Attribution-certainty confidence ----
    # Confidence reflects how firmly we identified the seat/size that BINDS the
    # effective stack — not a second estimator that eats the same OCR inputs
    # (those agree-or-disagree but don't track correctness). High when:
    #   * the binding came from an explicit panel all-in size (matched floor or
    #     uncalled-jam ceiling) — read directly off the screen;
    #   * the binding villain was pinned by a NAME match;
    #   * hero himself is the (shortest) binding stack — no villain attribution
    #     needed (hero's own displayed stack is the most reliable read);
    #   * there is a single unambiguous active villain (heads-up).
    # Medium when a villain was pinned only by POSITION/GEOMETRY (right seat,
    # no name confirmation). LOW (abstain) when the binding came from the
    # nameless shortest-seat GUESS — that's where attribution is genuinely
    # ambiguous and the input-bound residual lives.
    distinct_active = {p for p in opp_entered_positions
                       if p not in opp_folded_positions}
    single_active_villain = len(distinct_active) == 1 or n_opp_preflop == 1

    if binding_from_allin or (
        hero_uncalled_shove is not None and binding_from_hero
    ):
        confidence = 1.0          # explicit shove size off the panel
    elif binding_from_hero and not binding_from_opp:
        confidence = 0.95         # hero's own displayed stack binds
    elif binding_from_opp and engine_pinned and name_matched_villain:
        confidence = 0.95         # engine-chosen seat, name-confirmed
    elif binding_from_opp and engine_pinned:
        # The betting engine froze the live set and chose this position by
        # action order; the seat stack was read by geometry/name. Right
        # position, possibly geometry-read stack — solid but not name-certain.
        confidence = 0.85
    elif binding_from_opp and name_matched_villain:
        confidence = 0.95         # villain pinned by name
    elif binding_from_opp and geometry_pinned:
        confidence = 0.8          # villain pinned by position/geometry only
    elif binding_from_opp and single_active_villain:
        confidence = 0.8          # one obvious villain, even if unnamed
    elif nameless_fallback:
        confidence = 0.5          # shortest-seat guess — ambiguous, abstain
    else:
        confidence = 0.85

    # Internal-consistency abstain (narrow, NOT a precision lever): hero is the
    # BINDING (shortest) stack, hero barely invested (<3bb total), hero folded,
    # yet a still-live villain is >=3x deeper than hero's reconstructed start.
    # A folding player that short cannot define a deep spot — this is an
    # upstream hero-stack OCR corruption (displayed read far too small), which
    # both reconstruction estimators inherit (so the dual-estimator divergence
    # can't see it). Requiring hero_invest<3 avoids the deep-invested-hero
    # class (hero displayed tiny only because they invested a lot — a real,
    # correct binding). (TM5863941844: hero displayed 4.0, really ~20bb.)
    hero_invest = hero_starting - hero_stack_displayed
    hero_binds = abs(effective_bb - round(hero_starting, 1)) < 0.6
    hero_folded_anywhere = any(
        e.get("type") == "hero" and (e.get("action") or "").lower() == "fold"
        for c in (([preflop_col] if preflop_col else []) + list(street_cols))
        for e in (c.get("entries", []) if c else [])
    )
    live_villain_3x = any(
        s >= 3 * hero_starting for s in (all_stacks or [])
        if s != hero_stack_displayed
    )
    if (hero_binds and hero_invest < 3.0 and hero_folded_anywhere
            and live_villain_3x and hero_starting > 0):
        confidence = min(confidence, 0.3)

    # NOTE on the old "physical floor": the legacy gate nulled effective_bb when
    # it fell below the largest preflop raise. That is INVALID here — when hero
    # opens and a short stack calls all-in for less, the effective stack is
    # legitimately below the open size. Folding that into confidence
    # false-abstains a large, correct short-stack population, so it is dropped.
    # The only physically sound abstention is the pot-bounded over-compute guard
    # above (an UPPER bound, which has a valid physical basis).

    # Hero displayed ~0 (all-in / called-all-in) but the two independent
    # reconstructions of hero's committed amount disagree on the bucket — the
    # single-frame input does not determine hero's starting stack (the shove
    # sticker is partial/misread). Abstain. (TM5874977534: 1.24 shown vs 11.2.)
    if hero_allin_recon_disagree:
        confidence = min(confidence, 0.3)

    if confidence < _EFFBB_CONF_FLOOR:
        return None, round(hero_starting, 1), confidence
    return effective_bb, round(hero_starting, 1), confidence


# Solver depth buckets (must match gto_api.AVAILABLE_DEPTHS / effbb_metrics).
_DEPTH_BUCKETS = [100, 80, 60, 50, 40, 35, 30, 25, 20, 17, 14, 12, 10, 9, 8]


def _depth_bucket(bb) -> int | None:
    """Snap a bb value to its nearest solver depth bucket (matches the metric)."""
    try:
        bb = float(bb)
    except (TypeError, ValueError):
        return None
    return min(_DEPTH_BUCKETS, key=lambda d: abs(d - bb))


# Bucket cell boundaries (the midpoints between adjacent solver depths, the
# decision surfaces of _depth_bucket). A value near one of these edges flips its
# bucket under a tiny OCR error → a Phase-4 abstain risk signal. Edges, high→low:
# 90, 70, 55, 45, 37.5, 32.5, 27.5, 22.5, 18.5, 15.5, 13, 11, 9.5, 8.5.
_BUCKET_EDGES = sorted(
    (_DEPTH_BUCKETS[i] + _DEPTH_BUCKETS[i + 1]) / 2.0
    for i in range(len(_DEPTH_BUCKETS) - 1)
)


def _bucket_boundary_distance(bb) -> float | None:
    """Relative distance from ``bb`` to the nearest bucket-cell edge.

    Returns ``abs(bb - nearest_edge) / bb`` (a small value = the emitted depth is
    fragile: a few-percent OCR error in the binding stack would flip its solver
    bucket). ``None`` on bad input. Values above the top edge (no upper bound on
    the 100bb cell) return a large sentinel so deep stacks aren't flagged fragile.
    """
    try:
        bb = float(bb)
    except (TypeError, ValueError):
        return None
    if bb <= 0:
        return None
    nearest = min(_BUCKET_EDGES, key=lambda e: abs(e - bb))
    return abs(bb - nearest) / bb


# --- Phase 4: per-hand abstain-feature capture ---------------------------------
# _compute_effective_bb stashes the abstain-signal features for the LAST hand it
# scored here, so the calibration harness (scripts/effbb_calibrate.py) can pull
# them WITHOUT re-OCR. Production reads nothing from this; it is pure debug/calib
# instrumentation (a single dict, overwritten each call — no memory growth).
_LAST_EFFBB_FEATURES: dict = {}


def _effbb_last_features() -> dict:
    """Return the abstain-signal features captured for the last scored hand."""
    return dict(_LAST_EFFBB_FEATURES)


def _engine_relevant_bucket(
    columns, hero_position, num_players, named_stacks, hero_start, seat_map,
):
    """Depth bucket of the engine's decision-local relevant-opponent estimate.

    The betting engine froze the live contestant set at hero's decision by
    action order (independent of geometry/names). We map those positions to
    seats through ``seat_map``, read ``start = displayed + engine_contribution``,
    min with hero_start (and an M1 ceiling), and return that estimate's depth
    bucket — an INDEPENDENT second opinion on the binding stack. The orchestrator
    compares it to the legacy reconstruction's bucket: agreement is a strong
    correctness signal (corpus: 76% vs 52% on disagreement), so disagreement is
    a confidence penalty (abstain-eligible). Returns ``(bucket, is_singleton)``;
    ``is_singleton`` flags a single-opponent relevant set (the cleanest engine
    signal). Returns ``(None, False)`` when the engine has no usable seat.
    """
    if not (hero_position and num_players and hero_start and seat_map):
        return None, False
    try:
        from . import effbb_engine as _eng
        e_streets, e_pot = _engine_streets(columns)
        er = _eng.analyze(e_streets, num_players, hero_position,
                          e_pot.get("preflop"))
    except Exception:
        return None, False
    if er is None or not er.relevant_opponents:
        return None, False
    pot_vals = [c.get("pot") for c in columns
                if isinstance(c.get("pot"), (int, float)) and c.get("pot") > 0]
    pot_bound = max(pot_vals) if pot_vals else None
    vals = [hero_start]
    found_seat = False
    for pos in er.relevant_opponents:
        seat = seat_map.get(pos)
        if seat and seat.get("stack"):
            inv = er.contribution.get(pos, 0.0)
            if pot_bound is not None and inv > pot_bound + 0.5:
                inv = pot_bound
            vals.append(seat["stack"] + inv)
            found_seat = True
    if er.rule == "M1" and er.rule_ceiling:
        vals.append(er.rule_ceiling)
    if not found_seat and not (er.rule == "M1" and er.rule_ceiling):
        return None, False
    singleton = len(er.relevant_opponents) == 1
    return _depth_bucket(min(vals)), singleton


def _infer_num_players(columns, named_stacks):
    """Physical table size from panel positions (robust) → seat-ring fallback."""
    panel_pos = _panel_distinct_positions(columns)
    if panel_pos:
        for sz in range(2, 10):
            order = POSITION_ORDERS.get(sz)
            if order and panel_pos.issubset(set(order)):
                return sz
    ring = _seat_ring(named_stacks)
    if ring:
        n = min(max(len(ring), 2), 9)
        return n if n in POSITION_ORDERS else None
    return None


def _compute_effective_bb(
    columns: list[dict],
    hero_stack_displayed: float | None,
    hero_position: str | None,
    all_stacks: list[float] | None,
    named_stacks: list[dict] | None = None,
    num_players: int | None = None,
) -> tuple:
    """Bucket-consensus orchestrator over the top-K position→seat layouts.

    Phase 2. The single-layout reconstruction (``_effective_bb_for_layout``) is
    run once per *plausible* layout (top-K by weak name agreement, within a
    score margin, over both ring-walk directions and phantom-trim alternatives).
    The emitted depth bucket is the CONSENSUS signal:

      * All plausible layouts land in the SAME solver-depth bucket  → emit
        (confidence = consensus strength: 1.0 unanimous, lower with abstainers).
      * They straddle buckets                                       → abstain
        (return ``None`` — the attribution is genuinely ambiguous; Phase 3's
        re-read is what resolves the underlying misread seat).

    Layouts only change the GEOMETRY attribution branch, so for name-pinned /
    hero-binds / explicit-all-in hands every layout returns the identical value
    and consensus holds trivially at full confidence — the gate bites only where
    seat attribution is actually ambiguous, which is precisely the 78% of
    recoverable hands with ≥2 same-bucket candidate seats.

    Pot conservation is enforced inside the core as a hard reject (the
    over-compute guard nulls a layout whose investment exceeds the pot bound).
    Returns the legacy 3-tuple ``(effective_bb, hero_starting, confidence)``.
    """
    # --- Phase 4 feature capture (debug/calibration only; prod ignores it) ---
    # Accumulate the candidate abstain signals as we compute them, and flush to
    # the module-level store at every return via _finish(). No GT here.
    global _LAST_EFFBB_FEATURES
    feat: dict = {
        "hero_stack_displayed": hero_stack_displayed,
        "hero_position": hero_position,
        "num_players": num_players,
        "n_layouts": 0,
        "layout_buckets": [],
        "layout_straddle": False,
        "rep_eff": None,
        "rep_bucket": None,
        "base_conf": None,
        "emit_frac": None,
        "eng_bucket": None,
        "eng_singleton": None,
        "engine_agrees": None,
        "engine_disagrees": None,
        "engine_eligible": None,
        "x_agree": None,
        "boundary_dist": None,
        "hero_stack_near_zero": (hero_stack_displayed is not None
                                 and hero_stack_displayed <= 1.5),
        "decision_class": None,
        "n_relevant_opp": None,
        "rule_ceiling": None,
        "pot_residual": None,
        "binding_geometry_only": None,
        "stackonly_buckets": [],
        "method_straddle": False,
        "confidence": 0.0,
        "effective_bb": None,
    }

    def _finish(eff, hero_start_, conf):
        global _LAST_EFFBB_FEATURES
        feat["effective_bb"] = eff
        feat["confidence"] = conf
        if eff is not None:
            feat["boundary_dist"] = _bucket_boundary_distance(eff)
            feat["rep_bucket"] = _depth_bucket(eff)
        _LAST_EFFBB_FEATURES = feat
        return eff, hero_start_, conf

    if hero_stack_displayed is None:
        return _finish(None, None, 0.0)

    if num_players is None:
        num_players = _infer_num_players(columns, named_stacks)
    feat["num_players"] = num_players

    import os as _os
    _margin = int(_os.getenv("OCR_EFFBB_LAYOUT_MARGIN", "1"))
    layouts = []
    if hero_position and num_players:
        layouts = _enumerate_layouts(
            named_stacks, _panel_position_names(columns),
            hero_position, num_players, margin=_margin, max_k=8,
        )

    # Even with no geometric ambiguity (0 or 1 layout) we still run the
    # cross-method consensus (floors-on vs stack-only), so a single fixed seat
    # map gets the same abstain discipline. ``[None]`` = "let the core derive
    # its own seat map".
    if not layouts:
        layouts = [None]

    # Run the reconstruction under each plausible layout, BOTH with the panel
    # all-in floors / M1 ceiling on (the default estimate) and off (a stack-only
    # cross-check). Two independent consensus axes:
    #   * layout consensus   — does the geometric seat-direction matter?
    #   * cross-method consensus — does the panel-shove-size estimate agree with
    #     the pure stack reconstruction? (A misread/mis-attributed all-in size
    #     diverges here — the dominant residual error, NOT seat direction.)
    # Emit iff EVERY (layout × method) hypothesis lands in the same depth bucket.
    results = []          # floor-inclusive (default) per layout
    stack_only = []       # floors-off cross-check per layout
    for sm in layouts:
        try:
            r1 = _effective_bb_for_layout(
                columns, hero_stack_displayed, hero_position, all_stacks,
                named_stacks, num_players, _seat_map=sm,
            )
            r2 = _effective_bb_for_layout(
                columns, hero_stack_displayed, hero_position, all_stacks,
                named_stacks, num_players, _seat_map=sm, _disable_floors=True,
            )
        except Exception:  # pragma: no cover - core must never crash parsing
            continue
        results.append(r1)
        stack_only.append(r2)

    feat["n_layouts"] = len(layouts)
    feat["layout_buckets"] = sorted(
        {b for b in (_depth_bucket(eff) for eff, _, _ in results
                     if eff is not None) if b is not None})
    feat["stackonly_buckets"] = sorted(
        {b for b in (_depth_bucket(eff) for eff, _, _ in stack_only
                     if eff is not None) if b is not None})

    if not results:
        return _finish(None, None, 0.0)

    # hero_starting is layout-independent (hero's own seat).
    hero_start = next((hs for _, hs, _ in results if hs is not None), None)

    emitted = [(eff, conf) for eff, _, conf in results if eff is not None]
    n_total = len(results)
    n_emit = len(emitted)

    if not emitted:
        return _finish(None, hero_start, 0.3)

    # Representative value = highest internal-confidence layout (floors on).
    emitted.sort(key=lambda t: -t[1])
    rep_eff = emitted[0][0]
    base_conf = emitted[0][1]
    emit_frac = n_emit / n_total

    # --- Engine-vs-legacy consensus (the strongest discriminator) ---
    # The betting engine's decision-local relevant seat gives an INDEPENDENT
    # bucket (action-order logic, not geometry). On the corpus it agrees with a
    # correct legacy value 76% of the time but only 52% when it disagrees — the
    # single best abstain/tiebreak signal Phase 2 has. Read through the rep
    # layout (falling back to the single best map when rep is the [None] slot).
    best_idx = results.index(max(results, key=lambda r: r[2] if r[0] is not None else -1))
    rep_seat_map = layouts[best_idx]
    if rep_seat_map is None:
        rep_seat_map = _map_positions_to_seats(
            named_stacks, _panel_position_names(columns),
            hero_position, num_players)
    eng_bucket, eng_singleton = _engine_relevant_bucket(
        columns, hero_position, num_players, named_stacks, hero_start,
        rep_seat_map)
    feat["base_conf"] = base_conf
    feat["emit_frac"] = emit_frac
    feat["eng_bucket"] = eng_bucket
    feat["eng_singleton"] = eng_singleton

    # --- Engine decision class / pot-conservation residual (Phase-4 features) ---
    # One extra engine read for the calibration features (decision class M1/M2/M3
    # or standard, relevant-opponent count, M1 ceiling, and the preflop
    # pot-conservation residual: |inferred contributions − pot header|). Wrapped
    # so a feature-only failure never affects emission.
    try:
        from . import effbb_engine as _eng_f
        _es, _ep = _engine_streets(columns)
        _er = _eng_f.analyze(_es, num_players, hero_position, _ep.get("preflop"))
        if _er is not None:
            feat["decision_class"] = _er.rule or "standard"
            feat["n_relevant_opp"] = len(_er.relevant_opponents or [])
            feat["rule_ceiling"] = _er.rule_ceiling
            pot_hdr = _ep.get("preflop")
            if pot_hdr and _er.contribution:
                recon = sum(_er.contribution.values()) + (_er.ante_total or 0.0)
                feat["pot_residual"] = abs(recon - pot_hdr) / pot_hdr \
                    if pot_hdr > 0 else None
            feat["blinds_ok"] = bool(_er.blinds_ok)
    except Exception:
        pass

    # --- Layout consensus (geometric seat-direction ambiguity) ---
    # Do the floor-inclusive estimates agree on the depth bucket across all
    # plausible layouts? A straddle is usually genuine ambiguity → abstain,
    # UNLESS the independent engine bucket matches exactly one straddling
    # layout — then the engine breaks the tie toward that layout's value.
    layout_buckets = {_depth_bucket(eff) for eff, _, _ in results if eff is not None}
    if len(layout_buckets) != 1 or None in layout_buckets:
        feat["layout_straddle"] = True
        if eng_bucket is not None and eng_bucket in layout_buckets:
            # Engine resolves the direction: keep the layout whose bucket the
            # engine confirms.
            tie = [(e, c) for e, _, c in results
                   if e is not None and _depth_bucket(e) == eng_bucket]
            if tie:
                tie.sort(key=lambda t: -t[1])
                rep_eff = tie[0][0]
                base_conf = max(tie[0][1], 0.9)
            else:
                return _finish(None, hero_start, 0.4)
        else:
            return _finish(None, hero_start, 0.4)

    rep_bucket = _depth_bucket(rep_eff)
    engine_disagrees = eng_bucket is not None and eng_bucket != rep_bucket
    engine_agrees = eng_bucket is not None and eng_bucket == rep_bucket
    feat["rep_eff"] = rep_eff
    feat["engine_disagrees"] = engine_disagrees
    feat["engine_agrees"] = engine_agrees
    # method straddle: floors-on vs stack-only disagree on bucket across layouts
    feat["method_straddle"] = (
        bool(feat["stackonly_buckets"])
        and set(feat["layout_buckets"]) != set(feat["stackonly_buckets"]))

    # --- Cross-method agreement (panel-shove vs pure-stack) ---
    # NOT a hard gate (corpus evidence: a floor/stack disagreement is right as
    # often as wrong, so abstaining on it sheds correct hands). It is a soft
    # confidence input: agreement nudges confidence up.
    x_agree = False
    for (eff, _, _), (s_eff, _, _) in zip(results, stack_only):
        if (eff is not None and s_eff is not None
                and _depth_bucket(eff) == _depth_bucket(s_eff)):
            x_agree = True
            break

    consensus_conf = base_conf * (0.85 + 0.15 * emit_frac)
    if x_agree:
        consensus_conf += 0.03
    # The engine-vs-legacy disagreement penalty applies ONLY where the binding
    # was a GEOMETRY/heuristic read (base_conf <= ~0.85). The strong-evidence
    # bindings — an explicit panel all-in size, hero's own displayed stack, a
    # name-matched villain, a walkover BB read (all base_conf >= 0.95) — are
    # already reliable and an engine that reads a wrong seat must not abstain
    # them (corpus: disagreement is a coin-flip overall, but on the geometry
    # tier it is the strongest abstain signal we have).
    engine_eligible = base_conf <= 0.86
    feat["x_agree"] = x_agree
    feat["engine_eligible"] = engine_eligible
    feat["binding_geometry_only"] = engine_eligible
    if engine_eligible:
        # A geometry/heuristic binding (base_conf <= ~0.85) is only ~28% precise
        # on its own (corpus) — it MUST earn independent betting-logic
        # confirmation to be emitted. If the engine disagrees OR can't supply a
        # relevant seat to vouch for it, abstain.
        if engine_agrees:
            # Confirmed. A SINGLETON relevant set is the engine's cleanest
            # signal — lift just over the abstain floor; multiway earns a nudge.
            consensus_conf = max(consensus_conf, 0.72) if eng_singleton \
                else consensus_conf + 0.03
        else:
            consensus_conf = min(consensus_conf, 0.45)
    elif engine_disagrees:
        # Strong-evidence binding (all-in size / hero / name) but the engine
        # dissents — moderate penalty, not a full abstain (that tier is ~75%
        # precise and includes the explicit-shove goldens the engine misreads).
        # Held below the 0.9 top band so the top band stays the cleanest slice.
        consensus_conf = min(consensus_conf, 0.85)
    elif engine_agrees:
        consensus_conf += 0.03
    consensus_conf = min(1.0, round(consensus_conf, 3))

    if consensus_conf < _EFFBB_CONF_FLOOR:
        return _finish(None, hero_start, consensus_conf)

    # --- Phase 4: calibrated structural abstain (precision-maximizing) ---
    # The conf floor catches AMBIGUITY; these catch internally-consistent VALUE
    # errors the consensus signal is blind to. Any firing → abstain (cap conf so
    # downstream sees a None). Calibrated on the 1,805-hand hero-active cache
    # (scripts/effbb_calibrate.py, 5-fold pooled CV). The broad engine-disagree /
    # method-straddle signals are SCOPED OFF the strong panel-read bindings
    # (M1 uncalled-shove ceiling / M2 walkover at base_conf>=0.95) — on those
    # the engine reads a noisy seat and falsely dissents, so applying them there
    # is net-negative AND would abstain correct M1/M2 emits. Held-out: lifts
    # emitted precision ~70.9%→~73-75% at the cost of ~25pp coverage (cheap:
    # None → safe generic solver depth). 99.5% is NOT reachable on single-frame
    # inputs (the wrong residual is internally-consistent stack misreads no
    # feature separates — ceiling ~86% @ ~10% cov; see the plan Phase-4).
    if _EFFBB_STRUCTURAL_GATE:
        hero_near_zero = (hero_stack_displayed is not None
                          and hero_stack_displayed <= 1.5)
        strong_panel_read = (
            feat.get("decision_class") in ("M1", "M2") and base_conf >= 0.95)
        structural_abstain = (
            (engine_eligible and not engine_agrees)      # geometry binding unconfirmed
            or (hero_near_zero and not engine_agrees)    # all-in shove unconfirmed
            or (engine_disagrees and not strong_panel_read)   # independent engine dissents
            or (feat["method_straddle"] and not strong_panel_read)  # floors↔stack straddle
        )
        if structural_abstain:
            return _finish(None, hero_start, min(consensus_conf, 0.69))

    return _finish(round(rep_eff, 1), hero_start, consensus_conf)


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
    street_entries_pre_collapse_count = {}
    for col in columns:
        name = (col.get("street") or col.get("name") or "").lower()
        if name in ("flop", "turn", "river"):
            street_entries_count[name] = len(col.get("entries", []))
            pre = col.get("entries_pre_collapse_count")
            if pre is not None:
                street_entries_pre_collapse_count[name] = pre

    return {
        "players_at_table_raw": players_at_table_raw,
        "players_at_table_final": players_at_table_final,
        "estimate_used_reaction_signal": estimate_used_reaction_signal,
        "dealer_button_seat": table_result.get("dealer_button_seat"),
        "dealer_button_conf": float(table_result.get("dealer_button_conf") or 0.0),
        "ensemble_used": bool(table_result.get("ensemble_used")),
        "preflop_entries_count": len(action_entries or []),
        "preflop_entries_pre_collapse_count": (
            preflop_col.get("entries_pre_collapse_count") if preflop_col else None
        ),
        "street_entries_count": street_entries_count,
        "street_entries_pre_collapse_count": street_entries_pre_collapse_count,
    }


def _assemble_hand(
    table_result: dict,
    columns: list[dict],
    *,
    force_table_size: int | None = None,
    force_hero_position: str | None = None,
) -> tuple[dict | None, dict, dict]:
    """Assemble hand JSON from parsed table and panel data.

    ``force_table_size`` / ``force_hero_position`` (Phase 11.D-c/precision
    push) override row-counting with trusted focused-VLM structure, allowing
    row-collapse parse_none hands to keep deterministic card/action evidence
    instead of falling back to a destructive full-image reparse.

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
    promoted_misnamed_preflop = False
    forced_structure_reassembly = (
        force_table_size is not None or force_hero_position is not None
    )

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

    # Fixup: if PreFlop wasn't found but the second physical column was named
    # Flop, it is a header OCR error.  Promote it even for short 2-4 row shove
    # sequences so focused structure re-check can recover the hidden folds.
    had_preflop_col = preflop_col is not None
    preflop_col, street_cols = _promote_misnamed_preflop_column(
        preflop_col, street_cols
    )
    promoted_misnamed_preflop = (not had_preflop_col and preflop_col is not None)
    if promoted_misnamed_preflop:
        diagnostics["promoted_misnamed_preflop"] = True
    if forced_structure_reassembly:
        diagnostics["forced_structure_reassembly"] = True

    conflict_board_cards = _board_cards_supported_by_panel(
        board_cards, street_cols
    )
    if conflict_board_cards:
        _, hero_cards = _resolve_hero_board_conflict(
            conflict_board_cards,
            hero_cards,
            hero_details=table_result.get("hero_card_details"),
        )

    # Card confidence — use actual hero detection quality from table parser.
    # Don't boost based on board legibility: CardCNN runs hero and board
    # crops independently, so board cards being clear says nothing about
    # hero rank reliability. Regression: H2822 — hero 8s/8d classified at
    # 0.611, +0.1 board boost pushed it to 0.711 (just above the 0.70
    # MIN_CARD_CONF gate), letting the wrong "9s8d" prediction ship.
    #
    # Also do not let false board-card detections rewrite correct hero
    # cards on preflop-only hands. Natural8 screenshots can show bright
    # hero-card blobs in the table center detector's search window even
    # when no flop exists; conflict resolution is only useful when the
    # action panel proves a postflop street happened.
    hero_card_conf = table_result.get("hero_card_conf", 0.0)
    if hero_cards and len(hero_cards) == 2:
        conf_parts["card_confidence"] = hero_card_conf

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
    if promoted_misnamed_preflop:
        diagnostics["promoted_misnamed_preflop"] = True
    if forced_structure_reassembly:
        diagnostics["forced_structure_reassembly"] = True

    if not action_entries:
        return None, conf_parts, diagnostics

    for entry in action_entries:
        if "_position_missing_before_order" not in entry:
            entry["_position_missing_before_order"] = not (
                (entry.get("position") or "").strip()
            )

    # Determine table size from entry count
    players_at_table_raw, estimate_used_reaction_signal = _estimate_table_size(action_entries)

    # VLM structural override (Phase 11.D-c): the row-counting estimate fails
    # confidently on all-in/multiway hands. When a gemini-3.5-flash re-check
    # supplies the true seat count we trust it (validated clean oracle) and let
    # the existing, well-tested position-assignment path re-derive everything
    # from the corrected table size — no bespoke re-alignment needed.
    if force_table_size is not None:
        players_at_table_raw = force_table_size
        estimate_used_reaction_signal = False
    players_at_table = players_at_table_raw

    # Natural8 tournament replays in the paired PokerCraft corpus are 8-max.
    # A ninth visible preflop row has consistently been a duplicate/re-action
    # fragment (often an all-in overlay or caller response), not a real ninth
    # seat.  Cap the N8 parser at eight seats so those fragments stay in the
    # re-action tail instead of shifting all positions through a 9-max order.
    # A trusted VLM override may legitimately report 9-max, so only clamp the
    # heuristic estimate; respect an explicit override up to 9.
    if force_table_size is not None:
        players_at_table = min(max(players_at_table, 2), 9)
    else:
        players_at_table = min(max(players_at_table, 2), 8)
    first_round_count = players_at_table

    # If every visible preflop entry is a fold, action stops once the last
    # player before the big blind folds, so the BB never receives a decision
    # row.  The visible action count is
    # therefore one smaller than the number of players at the table.  Use
    # the larger table size for position assignment, but keep the action
    # string limited to the rows that actually exist so we do not append a
    # phantom BB fold.  Without this, 8-max fold-through screenshots were
    # treated as 7-max, shifting hero_position one seat toward the blinds.
    if (
        2 <= len(action_entries) < 9
        and all((e.get("action") or "").lower() == "fold" for e in action_entries)
    ):
        # A forced (VLM) seat count is authoritative; only adjust the action
        # string length (first_round_count), not the table size itself.
        if force_table_size is None:
            players_at_table = min(len(action_entries) + 1, 8)
        first_round_count = len(action_entries)

    diagnostics = _build_diagnostics(
        table_result,
        columns,
        preflop_col=preflop_col,
        action_entries=action_entries,
        players_at_table_raw=players_at_table_raw,
        players_at_table_final=players_at_table,
        estimate_used_reaction_signal=estimate_used_reaction_signal,
    )
    if promoted_misnamed_preflop:
        diagnostics["promoted_misnamed_preflop"] = True
    if forced_structure_reassembly:
        diagnostics["forced_structure_reassembly"] = True
    pos_order = POSITION_ORDERS.get(players_at_table, POSITION_ORDERS[8])
    # Assign positions by entry order (first entry = first position, etc.)
    # Only the FIRST hero entry determines hero_position; later hero
    # entries are re-actions (hero acting again after being raised).
    hero_position = None
    hero_index = None
    hero_position_source = None
    for i, entry in enumerate(action_entries[:first_round_count]):
        if i < len(pos_order):
            entry["position"] = pos_order[i]
            if entry["type"] == "hero" and hero_position is None:
                hero_position = pos_order[i]
                hero_index = i
                hero_position_source = "preflop_index_order"

    # Mark re-action entries (beyond first round)
    for i, entry in enumerate(action_entries[first_round_count:], first_round_count):
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
                    hero_position_source = "hero_fold_recovery"
                    break

    # Check blinds column for hero position override
    hero_blind_detected = 0.0
    if blinds_col:
        blinds_entries = blinds_col.get("entries", [])
        for entry in blinds_entries:
            if entry["type"] == "hero":
                action_text = (entry.get("action") or "").lower()
                size = entry.get("size")
                if "sb" in action_text or size == 0.5:
                    hero_position = "SB"
                    hero_blind_detected = 0.5
                    hero_position_source = "blind_column"
                elif "bb" in action_text or size == 1.0:
                    hero_position = "BB"
                    hero_blind_detected = 1.0
                    hero_position_source = "blind_column"

    # Do not override the action-panel hero position from table dealer-button
    # detection.  The button detector uses fixed seat anchors for table
    # geometry, but replay screenshots include crop/scale variants where a
    # high-confidence button blob maps to the wrong seat.  The action panel is
    # already ordered by preflop seat and gives the direct hero row; overriding
    # it shifted otherwise exact 8-max hands to BB.

    # Surface how hero_position was derived + blind-consistency for the
    # calibrator. position_wrong is 39% of wrong emits and the v2 schema is
    # blind to it (position was only "assigned or not"). The derivation
    # source, hero's seat ordinality, and whether a detected blind agrees
    # with the assigned position give the calibrator something to reject on.
    if force_hero_position:
        forced = normalize_position(str(force_hero_position))
        if forced in pos_order:
            hero_position = forced
            hero_index = pos_order.index(forced)
            hero_position_source = (
                "vlm_force_position"
                if hero_position_source is None
                else f"{hero_position_source}+vlm_force_position"
            )

    diagnostics["hero_position_source"] = hero_position_source
    diagnostics["hero_seat_index"] = hero_index
    diagnostics["hero_blind_detected"] = hero_blind_detected
    expected_blind = (
        1.0 if hero_position == "BB"
        else 0.5 if hero_position == "SB"
        else 0.0
    )
    # Consistent when no blind was detected (can't contradict) or the
    # detected blind matches the position's expected blind. A detected
    # blind that disagrees with the assigned position is the tell.
    diagnostics["hero_blind_consistent"] = (
        hero_blind_detected == 0.0 or hero_blind_detected == expected_blind
    )

    if not hero_position:
        return None, conf_parts, diagnostics

    # Build preflop_actions string using assigned positions
    preflop_actions = _build_preflop_actions_from_order(
        action_entries, pos_order, hero_position, players_at_table,
        first_round_count=first_round_count,
    )

    if not preflop_actions:
        return None, conf_parts, diagnostics

    # A raise/bet with no size is not solver-ready: using a placeholder raise
    # corrupts pot reconstruction and can map later street actions to the
    # wrong solver node.  Return no deterministic hand so production falls
    # back to Gemini vision with the partial OCR hints.
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
    for i, entry in enumerate(action_entries[:first_round_count]):
        pos = entry.get("position", pos_order[i] if i < len(pos_order) else None)
        action = (entry.get("action") or "").lower()
        if action != "fold" and pos:
            active_positions.append(pos)
    # Also check re-action entries (calls after 3bet etc.)
    for entry in action_entries[first_round_count:]:
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
    for entry in action_entries[:first_round_count]:
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
    effective_bb, hero_starting_stack, _effbb_conf = _compute_effective_bb(
        columns, hero_stack, hero_position, stacks, named_stacks,
    )

    # Phase-0 effbb cache: stash the raw inputs so effbb_eval can replay
    # _compute_effective_bb without re-OCR. Gated by env var; no prod cost.
    if os.getenv("EFFBB_CAPTURE"):
        hand_capture = {
            "columns": columns,
            "hero_stack": hero_stack,
            "hero_position": hero_position,
            "stacks": stacks,
            "named_stacks": named_stacks,
        }
    else:
        hand_capture = None

    preflop_actions, preflop_size_repairs = _repair_implausible_open_raise_sizes(
        preflop_actions
    )
    if preflop_size_repairs:
        diagnostics["preflop_size_repairs"] = preflop_size_repairs
    pre_count = diagnostics.get("preflop_entries_count")
    pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
    preloss = (
        pre_collapse - pre_count
        if isinstance(pre_collapse, int) and isinstance(pre_count, int)
        else 99
    )
    preflop_actions, allin_reaction_repairs = _repair_missing_allin_reaction_folds(
        preflop_actions,
        players_at_table,
        max_raw_loss=7,
        raw_loss=preloss,
    )
    if allin_reaction_repairs:
        diagnostics["preflop_allin_reaction_repairs"] = allin_reaction_repairs

    preflop_actions, terminal_fold_repairs = _repair_terminal_fold_after_vlm_allin_call(
        preflop_actions,
        diagnostics,
    )
    if terminal_fold_repairs:
        diagnostics["preflop_terminal_fold_repairs"] = terminal_fold_repairs

    preflop_actions, forced_collapse_repairs = _repair_forced_collapse_action_tail(
        preflop_actions,
        diagnostics,
    )
    if forced_collapse_repairs:
        diagnostics["preflop_forced_collapse_repairs"] = forced_collapse_repairs

    hand = {
        "gametype": "MTTGeneral",
        "hero_hand": hero_hand,
        "hero_position": hero_position,
        "players_at_table": players_at_table,
        "preflop_actions": preflop_actions,
    }

    # effective_bb abstention now lives inside _compute_effective_bb (the
    # dual-estimator confidence gate). The old external sanity gate — the
    # `< max_preflop_raise` null AND the `> hero_stack*5` displayed×5 null —
    # is removed: the physical-floor half is folded into the function's
    # confidence, and the displayed×5 half false-nulled deep-invested hero
    # hands (hero displayed ~3-5bb after heavy action, true effective ~30-60bb).
    if effective_bb is not None:
        hand["effective_bb"] = effective_bb

    physics_issues = _validate_preflop_bet_physics(
        preflop_actions,
        players_at_table,
        effective_bb=effective_bb,
    )
    if (
        hero_index == 0
        and first_round_count == players_at_table - 1
        and set(preflop_actions.split("-")) == {"F"}
        and not streets
    ):
        physics_issues.append("all_fold_walk_hero_first")
    if (
        players_at_table == 7
        and hero_position == "SB"
        and preflop_actions.startswith("R")
        and streets
    ):
        physics_issues.append("seven_max_sb_open_postflop_ambiguous")
    if physics_issues:
        diagnostics["preflop_physics_issues"] = physics_issues
        # Keep the partial structure for hints / field-level repair, but do
        # not let an impossible action chain pass confidence gates.
        conf_parts["ocr_confidence"] = 0.0

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

    if hand_capture is not None:
        hand["__effbb_inputs__"] = hand_capture

    # Purple felt is a final-table SIGNAL on N8, not a guarantee — auto-setting
    # ICM/FT from it over-triggered ICM analysis (H3518: an 8-handed purple
    # table judged FT). Don't commit; flag it so the bot ASKS the user to
    # confirm (chip-EV analysis runs meanwhile). User text keywords like
    # "FT/決賽桌" still opt in directly via gemini_session parsing.
    _flag_possible_ft(hand, table_color)

    # Pot consistency check
    conf_parts["pot_consistency"] = _check_pot_consistency(columns)

    # Player tracking check
    conf_parts["player_tracking"] = _check_player_tracking(
        action_entries, street_cols
    )

    # OCR confidence from entries.  Keep the hard penalty when an aggressive
    # action is missing its size; the structure can still be useful, but it
    # should not pass production confidence gates as solver-ready.
    if missing_raise_sizes or physics_issues:
        conf_parts["ocr_confidence"] = 0.0
    else:
        conf_parts["ocr_confidence"] = _avg_ocr_confidence(columns)

    return hand, conf_parts, diagnostics


def _extract_hero_hand_from_stack_text(table_result: dict) -> str:
    """Fallback: hero_hand might already be set by EasyOCR-based table parser."""
    # This is handled by table_parser._find_hero_cards now
    return ""


def _build_preflop_actions_from_order(
    action_entries: list[dict], pos_order: list[str],
    hero_position: str | None, table_size: int,
    *, first_round_count: int | None = None,
) -> str:
    """Build preflop_actions string from ordered action entries.

    Entries are already assigned positions by order.  Usually the first
    `table_size` entries form the first round; fold-through hands are the
    exception because the BB has no visible decision row, so callers can pass
    `first_round_count=table_size-1` to keep the action string faithful while
    still using the full table size for position assignment.

    Format: "F-F-R2-F-F-F-C-F" (one action per position)
    """
    # First round: map position -> action code
    pos_actions: dict[str, str] = {}
    if first_round_count is None:
        first_round_count = table_size

    for i, entry in enumerate(action_entries[:first_round_count]):
        pos = entry.get("position")
        if entry["type"] == "hero":
            pos = hero_position
        if not pos:
            continue

        action = (entry.get("action") or "").lower()
        size = entry.get("size")
        code = _action_to_code(action, size)
        if code:
            # A late anonymous/sizeless hero All-In sticker can be a duplicate
            # red all-in badge, not the hero's first-round action.  When a
            # forced/fallback hero position already has a concrete first-round
            # fold from the ordered row, do not let that bare sticker overwrite
            # it.  Real all-ins with a size stay authoritative.
            if (
                pos == hero_position
                and pos in pos_actions
                and pos_actions[pos] == "F"
                and code == "AI"
                and size is None
                and i > 0
                and bool(entry.get("_position_missing_before_order"))
                and not (entry.get("player_name") or "").strip()
            ):
                continue
            pos_actions[pos] = code

    # Build first-round string in position order
    parts = []
    for pos in pos_order[:first_round_count]:
        parts.append(pos_actions.get(pos, "F"))

    result = "-".join(parts)

    # Re-actions (entries beyond first round)
    re_codes = []
    for entry in action_entries[first_round_count:]:
        if entry.get("_is_reaction"):
            action = (entry.get("action") or "").lower()
            size = entry.get("size")
            code = _action_to_code(action, size)
            if code:
                re_codes.append(code)

    if re_codes:
        result += "-" + "-".join(re_codes)

    return _drop_duplicate_bare_allin_after_call(result)


def _drop_duplicate_bare_allin_after_call(preflop_actions: str) -> str:
    """Remove sizeless all-in badges duplicated after a call token.

    Natural8 showdown/reaction tails can split a red All-In badge into a bare
    ``AI`` token immediately after the real caller's ``C``.  With no amount,
    actor, or position, that badge is not a solver action; leaving it in
    creates impossible preflop chains such as ``...-C-AI-F-F``.  Sized all-ins
    stay intact.
    """
    toks = [t for t in (preflop_actions or "").split("-") if t]
    out: list[str] = []
    for tok in toks:
        if tok == "AI" and out and out[-1] == "C":
            continue
        out.append(tok)
    return "-".join(out)


def _repair_forced_collapse_action_tail(
    preflop_actions: str,
    diagnostics: dict,
) -> tuple[str, list[str]]:
    """Repair narrow VLM-forced row-collapse tails with duplicated folds.

    These patterns come from N8 replay columns where an all-in/call resolution
    is visually compressed into the physical preflop column while river
    runout chrome contributes hidden row fragments.  The trusted VLM pass
    fixes seat count/hero position, but action rows can still contain a single
    duplicated fold around the all-in sticker.  Keep the repairs intentionally
    shape-gated so normal ``VLM agree`` multi-all-in sequences are untouched.
    """
    if not diagnostics.get("forced_structure_reassembly"):
        return preflop_actions, []

    postflop_rows = sum(
        int(v or 0)
        for v in (diagnostics.get("street_entries_count") or {}).values()
    )
    if postflop_rows != 0:
        return preflop_actions, []

    pre_count = diagnostics.get("preflop_entries_count")
    pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
    preloss = (
        pre_collapse - pre_count
        if isinstance(pre_collapse, int) and isinstance(pre_count, int)
        else 99
    )
    street_counts = diagnostics.get("street_entries_count") or {}
    street_pre_counts = diagnostics.get("street_entries_pre_collapse_count") or {}
    hidden_street_fragments = sum(
        max(0, int(street_pre_counts.get(name) or 0) - int(street_counts.get(name) or 0))
        for name in ("flop", "turn", "river")
    )

    toks = [t for t in (preflop_actions or "").split("-") if t.strip()]
    parsed = [_parse_preflop_token(t) for t in toks]
    repairs: list[str] = []

    # promoted short column: C-F-F-AI-F → C-F-AI-F.  The extra fold is the
    # hidden row that was also responsible for the physical Pre-Flop header
    # being OCR'd as Flop.
    if (
        diagnostics.get("promoted_misnamed_preflop")
        and hidden_street_fragments >= 6
        and len(toks) >= 7
    ):
        for i in range(2, len(toks) - 1):
            if (
                parsed[i - 2][0] == "C"
                and parsed[i - 1][0] == "F"
                and parsed[i][0] == "F"
                and parsed[i + 1][0] == "AI"
                and parsed[i + 1][1] is not None
            ):
                del toks[i]
                repairs.append("drop_duplicate_fold_before_promoted_allin")
                break

    parsed = [_parse_preflop_token(t) for t in toks]

    # Forced all-in then raise tail: AI-F-F-R-F-F → AI-F-F-R-F.  The final
    # default fold is a phantom seat after VLM expands the hidden player count.
    if (
        not repairs
        and len(toks) == 6
        and [typ for typ, _amt in parsed] == ["AI", "F", "F", "R", "F", "F"]
        and parsed[0][1] is not None
        and parsed[3][1] is not None
        and hidden_street_fragments == 3
        and preloss == 7
    ):
        toks.pop()
        repairs.append("drop_terminal_phantom_fold_after_allin_raise")

    parsed = [_parse_preflop_token(t) for t in toks]

    # VLM-corrected two-shove tail where both middle folds are duplicated
    # collapse artifacts: F-AI-F-F-AI-F → F-AI-AI-F.
    if (
        not repairs
        and len(toks) == 6
        and [typ for typ, _amt in parsed] == ["F", "AI", "F", "F", "AI", "F"]
        and parsed[1][1] is not None
        and parsed[4][1] is not None
        and hidden_street_fragments == 3
        and preloss >= 10
    ):
        toks = [toks[0], toks[1], toks[4], toks[5]]
        repairs.append("drop_middle_folds_between_forced_allins")

    parsed = [_parse_preflop_token(t) for t in toks]

    # Squeeze/call tail with a duplicated fold before the final caller:
    # R-F-AI-F-C-F-F-C → R-F-AI-F-C-F-C.
    if (
        not repairs
        and len(toks) == 8
        and [typ for typ, _amt in parsed] == ["R", "F", "AI", "F", "C", "F", "F", "C"]
        and parsed[0][1] is not None
        and parsed[2][1] is not None
        and hidden_street_fragments >= 6
        and preloss >= 10
    ):
        del toks[6]
        repairs.append("drop_duplicate_fold_before_final_call")

    parsed = [_parse_preflop_token(t) for t in toks]

    # Short BB fold-out after one shove: F-F-F-AI-F-F-F → F-F-F-AI-F-F.
    # Limit to the low-collapse VLM-corrected variant; adjacent preloss=4
    # examples on the benchmark are valid seven-token hands.
    if (
        not repairs
        and len(toks) == 7
        and [typ for typ, _amt in parsed] == ["F", "F", "F", "AI", "F", "F", "F"]
        and parsed[3][1] is not None
        and hidden_street_fragments == 0
        and preloss == 3
    ):
        toks.pop()
        repairs.append("drop_low_collapse_terminal_phantom_fold")

    if repairs:
        return "-".join(toks), repairs
    return preflop_actions, []


def _action_to_code(action: str, size: float | None) -> str | None:
    """Convert action name + size to preflop action code."""
    action = action.lower().strip()

    if action == "fold":
        return "F"
    elif action == "call":
        return "C"
    elif action == "check":
        return "X"  # BB option is a check in PokerCraft/GTO action strings
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

    # Heads-up pots: exactly one non-hero player reaches postflop.  N8's
    # per-row position OCR/reconciliation can mislabel that lone villain
    # (H3517: a BB 3-bettor's flop bet + turn shove tagged LJ), and
    # _fix_folded_players then strips the mislabeled rows as "folded player"
    # actions — leaving an orphan hero Call with nothing to call and no solver
    # node on that street.  When only one opponent is live, every opponent
    # action is theirs; trust that over the noisy per-row label.
    _nonhero_active = [p for p in (active_positions or []) if p != hero_position]
    sole_villain = _nonhero_active[0] if len(_nonhero_active) == 1 else None

    # Track who folds across streets
    folded_in_streets = set()

    runout_after_allin_call = False
    allin_closed = False
    for col in street_cols:
        # Once a postflop all-in resolves a street, every later street is a
        # physical runout with no decisions. The solver-relevant board stops at
        # the all-in street (matches the hand-history ground truth), so drop the
        # runout turn/river — the dominant board_wrong cause (27/28) was the
        # parser appending these visible-but-irrelevant runout cards.
        if allin_closed:
            break

        name = col["name"].lower()
        entries = col.get("entries", [])

        # Do not infer a later street from board-card pixels alone.  N8 keeps
        # bright table/hero-card shapes in the board detector's search window,
        # so false turn/river cards are common when the action panel has no
        # entries for that street.  A dealt street should have at least one
        # parsed panel row; otherwise leave it for Gemini fallback/hints rather
        # than corrupting deterministic hand_exact with phantom runout cards.
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
        street_name_positions: list[tuple[str, str]] = []
        pending_all_in = False

        for entry in entries:
            entry_type = entry.get("type", "opponent")
            action_text = (entry.get("action") or "").lower()
            size = entry.get("size")

            if not action_text or action_text in ("unknown", "skip"):
                continue

            # Assign position
            if entry_type == "hero":
                pos = hero_position
            elif sole_villain is not None:
                # Heads-up: the only live opponent owns every opponent action,
                # regardless of a noisy per-row position label (H3517).
                pos = sole_villain
            else:
                # Use OCR-detected position if available
                ocr_pos = entry.get("position")
                player_name = (entry.get("player_name") or "").strip()
                prior_name_pos = next(
                    (
                        prev_pos
                        for prev_name, prev_pos in street_name_positions
                        if player_name and _fuzzy_name_match(player_name, prev_name)
                    ),
                    None,
                )
                if ocr_pos and ocr_pos != "BB":
                    # Trust OCR position if it's not the default
                    pos = ocr_pos
                elif ocr_pos == "BB" and prior_name_pos == "BB":
                    # BB badges are common OCR defaults, so the generic path
                    # usually infers order instead of trusting them.  But when
                    # the same named player was already assigned BB on this
                    # street (check → raise), keep BB; otherwise a folded BTN
                    # can be incorrectly resurrected as the raiser. H2896.
                    pos = "BB"
                elif opp_positions_remaining:
                    # Infer from postflop order
                    pos = opp_positions_remaining[opp_idx % len(opp_positions_remaining)]
                    opp_idx += 1
                else:
                    pos = ocr_pos or "?"
                # A fold can never belong to a seat that already acted
                # (bet/raised/called) on this street.  In multiway pots a
                # cold-caller's fold is sometimes mapped — by a misread per-row
                # badge or by order inference — onto an already-acted seat, most
                # often the bettor (H3531: BB's flop fold tagged SB, the SB
                # bettor).  That leaves an impossible self-fold that breaks the
                # multiway→HU collapse and drops every post-flop solver node.
                # Reassign the fold to the first live opponent who has not yet
                # acted this street (the real cold-caller).
                if action_text == "fold":
                    acted = {a["position"] for a in actions if a["action"] != "F"}
                    if pos in acted:
                        unacted = [
                            p for p in opp_positions_remaining
                            if p not in acted and p not in folded_in_streets
                        ]
                        if unacted:
                            pos = unacted[0]
                if player_name and pos and pos != "?":
                    street_name_positions.append((player_name, pos))

            act_code = _street_action_code(action_text, size)
            act_dict = {"position": pos, "action": act_code}
            if size is not None:
                act_dict["size"] = size
            # A sized all-in keeps the absolute R{size} code (so solver
            # action-matching and golden snapshots are unchanged), but we tag
            # it explicitly so downstream analysis knows the bettor is committed
            # — a player who calls this is calling an all-in, not facing a bet
            # they could still raise. H3459 (SB turn shove "Bet 17.1 / All-In").
            if action_text == "all-in":
                act_dict["allin"] = True
            actions.append(act_dict)

            # Track folds
            if action_text == "fold":
                folded_in_streets.add(pos)
            if action_text == "all-in":
                pending_all_in = True
            elif pending_all_in and action_text == "call":
                runout_after_allin_call = True

        street["actions"] = actions
        if actions:
            streets.append(street)
            # A called or uncalled all-in closes the decision tree; subsequent
            # streets are runout only and must not contribute board cards.
            if pending_all_in:
                allin_closed = True

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


def _preflop_type_tokens(hand: dict | None) -> list[str]:
    """Return preflop action type tokens, ignoring numeric sizes."""
    if not hand:
        return []
    tokens: list[str] = []
    for tok in (hand.get("preflop_actions") or "").split("-"):
        tok = tok.strip().upper()
        if not tok:
            continue
        if tok.startswith("AI"):
            tokens.append("AI")
        elif tok.startswith("R"):
            tokens.append("R")
        else:
            tokens.append(tok)
    return tokens


def _parse_preflop_token(tok: str) -> tuple[str, float | None]:
    tok = (tok or "").strip().upper()
    if not tok:
        return "", None
    if tok.startswith("AI"):
        try:
            return "AI", float(tok[2:]) if tok[2:] else None
        except ValueError:
            return "AI", None
    if tok.startswith("R"):
        try:
            return "R", float(tok[1:]) if tok[1:] else None
        except ValueError:
            return "R", None
    if tok in {"F", "C", "X"}:
        return tok, None
    return tok, None


def _repair_implausible_open_raise_sizes(preflop_actions: str) -> tuple[str, list[str]]:
    """Snap impossible ``Raise 1 BB`` open-size OCR to the legal min-open.

    Natural8 displays a limp as ``Call 1 BB``.  When the panel parser produces
    ``R1`` before any previous raise, it is almost always a dropped digit from
    ``Raise 2 BB`` (or a nearby 2.x BB open), and leaving it as-is triggers the
    physics rejector plus gives downstream solver matching an impossible open
    size.  Keep this deliberately narrow: once an explicit raise/all-in has
    occurred, an ``R1`` token is a malformed re-raise and must remain rejected.
    """
    toks = [t for t in (preflop_actions or "").split("-") if t.strip()]
    if not toks:
        return preflop_actions, []

    repaired: list[str] = []
    repairs: list[str] = []
    explicit_raise_seen = False
    for idx, tok in enumerate(toks):
        typ, amt = _parse_preflop_token(tok)
        if (
            typ == "R"
            and amt is not None
            and amt <= 1.05
            and not explicit_raise_seen
        ):
            repaired.append("R2")
            repairs.append(f"open_raise_min_snap@{idx}:{amt:g}->2")
            explicit_raise_seen = True
            continue
        repaired.append(tok)
        if typ == "R" and amt is not None:
            explicit_raise_seen = True
        elif typ == "AI" and amt is not None and amt > 1.0:
            explicit_raise_seen = True
    return "-".join(repaired), repairs


def _repair_missing_allin_reaction_folds(
    preflop_actions: str,
    players_at_table: int | None,
    *,
    raw_loss: int,
    max_raw_loss: int = 7,
) -> tuple[str, list[str]]:
    """Insert conservative missing folds after a preflop all-in raise.

    Natural8 can hide the original opener/limper's final fold behind the
    all-in sticker.  When the row-collapse loss is small, a single missing
    fold after an all-in is usually an OCR omission, not a genuinely ambiguous
    action-chain rewrite.  Avoid very high-collapse VLM-corrected tails where
    the same shape is unsafe.
    """
    toks = [t for t in (preflop_actions or "").split("-") if t.strip()]
    n = int(players_at_table or 0)
    if not toks or not n or raw_loss > max_raw_loss:
        return preflop_actions, []
    if sum(1 for t in toks if _parse_preflop_token(t)[0] == "AI") != 1:
        return preflop_actions, []

    repairs: list[str] = []
    for idx, tok in enumerate(toks[:n]):
        typ, _amt = _parse_preflop_token(tok)
        if typ != "AI":
            continue
        prior_active = sum(
            1
            for t in toks[:idx]
            if _parse_preflop_token(t)[0] in {"C", "R"}
        )
        if prior_active <= 0:
            return preflop_actions, []
        remaining_seats = max(0, n - idx - 1)
        expected_after = remaining_seats + prior_active
        after = toks[idx + 1 :]
        if len(after) >= expected_after:
            return preflop_actions, []

        missing = expected_after - len(after)
        if missing != 1:
            return preflop_actions, []

        # If the first post-AI token is a call/raise/all-in, it is likely a
        # wrap-around response; insert the skipped remaining-seat fold before
        # it.  Otherwise append the opener/limper's missing final fold.
        if remaining_seats and after and _parse_preflop_token(after[0])[0] in {"C", "R", "AI"}:
            repaired = toks[: idx + 1] + ["F"] + toks[idx + 1 :]
            repairs.append(f"insert_remaining_fold_after_ai@{idx}")
        else:
            repaired = toks + ["F"]
            repairs.append(f"append_prior_fold_after_ai@{idx}")
        return "-".join(repaired), repairs
    return preflop_actions, []


def _repair_terminal_fold_after_vlm_allin_call(
    preflop_actions: str,
    diagnostics: dict,
) -> tuple[str, list[str]]:
    """Trim a duplicate terminal fold in narrow VLM-corrected all-in tails.

    Some collapsed all-in rows include an extra final Fold sticker after a
    called all-in has already closed the preflop action.  Broad trimming is
    very unsafe (many exact hands legitimately end with folds), so this only
    applies to the measured low-collapse VLM-corrected shape where the hidden
    street-fragment count is exactly three and no postflop action rows exist.
    """
    toks = [t for t in (preflop_actions or "").split("-") if t.strip()]
    if len(toks) < 3 or toks[-1] != "F":
        return preflop_actions, []
    if not diagnostics.get("forced_structure_reassembly"):
        return preflop_actions, []
    if not any(_parse_preflop_token(t)[0] == "AI" for t in toks):
        return preflop_actions, []
    if "C" not in [_parse_preflop_token(t)[0] for t in toks]:
        return preflop_actions, []

    pre_count = diagnostics.get("preflop_entries_count")
    pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
    preloss = (
        pre_collapse - pre_count
        if isinstance(pre_collapse, int) and isinstance(pre_count, int)
        else 99
    )
    postflop_rows = sum(
        int(v or 0)
        for v in (diagnostics.get("street_entries_count") or {}).values()
    )
    street_counts = diagnostics.get("street_entries_count") or {}
    street_pre_counts = diagnostics.get("street_entries_pre_collapse_count") or {}
    hidden_street_fragments = sum(
        max(0, int(street_pre_counts.get(name) or 0) - int(street_counts.get(name) or 0))
        for name in ("flop", "turn", "river")
    )
    if postflop_rows != 0 or hidden_street_fragments != 3:
        return preflop_actions, []
    if not (5 <= preloss <= 8):
        return preflop_actions, []

    return "-".join(toks[:-1]), ["trim_duplicate_terminal_fold_after_allin_call"]


def _validate_preflop_bet_physics(
    preflop_actions: str,
    players_at_table: int | None,
    *,
    effective_bb: float | None = None,
) -> list[str]:
    """Return strict, low-false-positive preflop impossibility flags.

    This is intentionally narrower than a full poker action solver: it only
    rejects states that cannot be valid regardless of hidden stack details.
    The aim is precision protection, not speculative repair.
    """
    toks = [t for t in (preflop_actions or "").split("-") if t.strip()]
    if not toks:
        return ["empty_preflop"]
    n = int(players_at_table or 0)
    if n and len(toks) < max(1, n - 1):
        return [f"too_few_initial_actions:{len(toks)}<{n - 1}"]

    issues: list[str] = []
    outstanding = 1.0
    explicit_raise_seen = False
    for idx, tok in enumerate(toks):
        typ, amt = _parse_preflop_token(tok)
        is_bb_option = bool(n and idx == n - 1 and not explicit_raise_seen)

        if typ == "R":
            if amt is None:
                issues.append(f"raise_missing_size@{idx}")
            elif amt <= outstanding + 0.05:
                issues.append(f"non_monotone_raise@{idx}:{amt:g}<={outstanding:g}")
            elif effective_bb is not None and amt > effective_bb + 0.5:
                issues.append(f"raise_exceeds_effective@{idx}:{amt:g}>{effective_bb:g}")
            if amt is not None:
                outstanding = max(outstanding, amt)
                explicit_raise_seen = True
        elif typ == "AI":
            # All-in may be a short all-in call, so only reject impossible stack
            # sizes when effective stack is known.
            if (amt is not None and effective_bb is not None
                    and amt > effective_bb + 0.5):
                issues.append(f"allin_exceeds_effective@{idx}:{amt:g}>{effective_bb:g}")
            if amt is not None and amt > outstanding:
                outstanding = amt
                explicit_raise_seen = True
        elif typ == "X":
            if not is_bb_option:
                # No one except BB can check preflop before a voluntary bet,
                # and after a raise/call sequence an arbitrary X token is a
                # duplicated blind-option fragment.
                issues.append(f"illegal_preflop_check@{idx}")
        elif typ in {"F", "C"}:
            pass
        else:
            issues.append(f"unknown_preflop_token@{idx}:{tok}")
    return issues


def _safe_emit_override_reason(
    hand: dict | None,
    confidence_parts: dict,
    diagnostics: dict,
) -> str | None:
    """Return a guarded reason for emitting below the global confidence gate.

    The blended confidence score is intentionally conservative: missing raise
    sizes and weak postflop player tracking can push otherwise exact parses
    below the benchmark's 0.88 emission threshold.  These predicates recover
    only shapes whose risk is bounded by strong card confidence plus stable
    action/table diagnostics.  Known danger shapes stay abstained: low card
    confidence, ambiguous all-in grammar, missing effective stack on fragile
    all-in rows, and reaction/table-size mismatches.
    """
    if not hand:
        return None
    physics_issues = diagnostics.get("preflop_physics_issues") or []
    all_fold_hero_first_only = bool(physics_issues) and all(
        str(issue).startswith("all_fold_walk_hero_first")
        for issue in physics_issues
    )
    if physics_issues and not all_fold_hero_first_only:
        return None
    if diagnostics.get("structural_risk_issues"):
        return None
    vlm_outcome = diagnostics.get("vlm_recheck_outcome")
    if vlm_outcome == "abstain":
        return None
    if (
        vlm_outcome == "corrected"
        and not diagnostics.get("promoted_misnamed_preflop")
        and not diagnostics.get("preflop_terminal_fold_repairs")
        and not diagnostics.get("preflop_forced_collapse_repairs")
    ):
        return None

    card_conf = float(confidence_parts.get("card_confidence") or 0.0)
    pot_conf = float(confidence_parts.get("pot_consistency") or 0.0)
    player_conf = float(confidence_parts.get("player_tracking") or 0.0)
    ocr_conf = float(confidence_parts.get("ocr_confidence") or 0.0)
    pre_count = diagnostics.get("preflop_entries_count")
    pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
    preloss = (
        (pre_collapse - pre_count)
        if isinstance(pre_collapse, int) and isinstance(pre_count, int)
        else 99
    )
    raw_players = diagnostics.get("players_at_table_raw")
    final_players = diagnostics.get("players_at_table_final")
    postflop_rows = sum(
        int(v or 0)
        for v in (diagnostics.get("street_entries_count") or {}).values()
    )
    street_counts = diagnostics.get("street_entries_count") or {}
    street_pre_counts = diagnostics.get("street_entries_pre_collapse_count") or {}
    max_street_loss = max(
        (
            int(street_pre_counts.get(name) or 0)
            - int(street_counts.get(name) or 0)
        )
        for name in ("flop", "turn", "river")
    )
    hidden_street_fragments = sum(
        max(0, int(street_pre_counts.get(name) or 0) - int(street_counts.get(name) or 0))
        for name in ("flop", "turn", "river")
    )
    used_reaction_signal = bool(
        diagnostics.get("estimate_used_reaction_signal")
    )
    tokens = _preflop_type_tokens(hand)
    has_allin = "AI" in tokens
    raise_count = tokens.count("R")
    effective_missing = hand.get("effective_bb") is None

    if vlm_outcome == "recovered":
        if (
            card_conf >= 0.99
            and pot_conf == 1.0
            and player_conf >= 0.5
            and preloss <= 9
            and postflop_rows == 0
        ):
            return "vlm_recovered_stable_preflop"
        if (
            card_conf >= 0.996
            and pot_conf >= 0.3
            and player_conf >= 0.5
            and postflop_rows == 0
            and (preloss <= 9 or hidden_street_fragments > 0)
        ):
            return "vlm_recovered_preflop_high_card"
        if (
            card_conf >= 0.99
            and pot_conf >= 0.5
            and player_conf >= 0.5
            and not has_allin
            and max_street_loss <= 2
        ):
            return "vlm_recovered_no_allin_low_collapse"
        return None

    if (
        vlm_outcome == "corrected"
        and diagnostics.get("promoted_misnamed_preflop")
        and isinstance(pre_count, int)
        and 3 <= pre_count <= 4
        and postflop_rows == 0
        and has_allin
        and card_conf >= 0.98
        and pot_conf >= 0.5
        and player_conf >= 0.5
    ):
        return "promoted_preflop_short_allin_vlm"
    if diagnostics.get("preflop_terminal_fold_repairs"):
        raw_toks = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
        parsed_toks = [_parse_preflop_token(t) for t in raw_toks]
        if (
            parsed_toks
            and (
                (
                    parsed_toks[0][0] == "AI"
                    and parsed_toks[0][1] is not None
                )
                or parsed_toks[0][0] == "F"
            )
            and sum(1 for typ, _amt in parsed_toks if typ == "AI") == 1
            and card_conf >= 0.95
            and pot_conf >= 1.0
            and player_conf >= 0.5
        ):
            return "terminal_fold_trimmed_single_allin"
    if diagnostics.get("preflop_forced_collapse_repairs"):
        raw_toks = [t for t in (hand.get("preflop_actions") or "").split("-") if t]
        parsed_toks = [_parse_preflop_token(t) for t in raw_toks]
        if (
            parsed_toks
            and card_conf >= 0.99
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and any(typ in {"AI", "R"} and amt is not None for typ, amt in parsed_toks)
        ):
            return "forced_collapse_repaired_vlm"
    if vlm_outcome == "corrected":
        return None

    high_card_base = card_conf >= 0.998 and pot_conf >= 0.5 and player_conf >= 0.5

    if (
        preloss >= 10
        and postflop_rows == 0
        and not diagnostics.get("vlm_recheck_outcome")
    ):
        return None

    if (
        all_fold_hero_first_only
        and tokens
        and all(tok == "F" for tok in tokens)
        and postflop_rows == 0
        and int(final_players or 0) >= 7
        and card_conf >= 0.95
        and player_conf >= 0.5
    ):
        return "all_fold_hero_first_large_table"
    if all_fold_hero_first_only:
        return None

    if high_card_base and not has_allin and raise_count <= 1:
        return "simple_preflop_high_card"

    if (
        tokens
        and all(tok == "F" for tok in tokens)
        and postflop_rows == 0
        and card_conf >= 0.95
        and player_conf >= 0.5
        and ocr_conf == 1.0
    ):
        return "all_fold_high_card"

    if (
        card_conf >= 0.60
        and pot_conf >= 0.3
        and player_conf >= 0.5
        and ocr_conf == 1.0
        and not has_allin
    ):
        return "no_allin_structural_high_card"

    if (
        card_conf >= 0.80
        and not has_allin
        and not diagnostics.get("structural_risk_issues")
        and max_street_loss <= 2
    ):
        return "no_allin_low_street_collapse"

    danger_complex = (
        bool(tokens and tokens[-1] == "AI")
        or bool(tokens and tokens[0] == "R")
        or (effective_missing and preloss <= 2)
        or (effective_missing and raw_players != final_players)
        or (
            postflop_rows > 0
            and raw_players != final_players
            and ocr_conf == 0.0
            and raise_count >= 2
            and preloss <= 7
        )
    )
    if (
        card_conf >= 0.999
        and pot_conf >= 0.5
        and player_conf >= 0.5
        and not danger_complex
    ):
        return "high_card_complex_non_danger"

    if (
        card_conf >= 0.999
        and ocr_conf == 1.0
        and pot_conf == 1.0
        and preloss <= 1
        and not has_allin
        and postflop_rows > 0
        and not used_reaction_signal
    ):
        return "stable_postflop_high_card"

    return None


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
