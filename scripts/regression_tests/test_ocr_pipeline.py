"""Regression tests extracted from the legacy monolithic suite."""

import json
import logging
import os
import sys
from pathlib import Path

from regression_tests.harness import (
    REPO_ROOT,
    SCRIPTS_DIR,
    _tests,
    _verbose,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
    test,
)

# ── OCR Pipeline Tests ──


@test
def test_ocr_preprocess_upscales_small_image():
    """OCR: preprocess upscales images smaller than 600px wide."""
    import numpy as np
    from ocr.ocr_utils import preprocess_for_ocr
    small = np.zeros((400, 300), dtype=np.uint8)
    result = preprocess_for_ocr(small)
    assert_true(result.shape[1] >= 600, f"should upscale width, got {result.shape[1]}")


@test
def test_ocr_preprocess_keeps_large_image():
    """OCR: preprocess does not upscale images >= 600px wide."""
    import numpy as np
    from ocr.ocr_utils import preprocess_for_ocr
    large = np.zeros((800, 700), dtype=np.uint8)
    result = preprocess_for_ocr(large)
    assert_eq(result.shape[1], 700, "should not change width of large image")


@test
def test_ocr_region_detection_finds_divider():
    """OCR: region detector finds table/panel divider in N8 screenshot."""
    import cv2
    from ocr.region_detector import detect_regions
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    result = detect_regions(image)
    assert_true(result is not None, "should detect N8 regions")
    assert_true("table" in result, "should have table region")
    assert_true("panel" in result, "should have panel region")
    assert_true(result["divider_y"] > image.shape[0] * 0.3, "divider should be below 30%")
    assert_true(result["divider_y"] < image.shape[0] * 0.6, "divider should be above 60%")


@test
def test_ocr_region_detection_returns_none_for_non_n8():
    """OCR: region detector returns None for non-N8 images."""
    import numpy as np
    from ocr.region_detector import detect_regions
    noise = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    result = detect_regions(noise)
    assert_true(result is None, "should return None for non-N8 image")


@test
def test_ocr_panel_column_split():
    """OCR: panel parser splits action panel into 5 columns."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import split_columns
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    columns = split_columns(regions["panel"])
    assert_eq(len(columns), 5, f"should find 5 columns, got {len(columns)}")


@test
def test_ocr_panel_entry_detection():
    """OCR: panel parser detects hero and opponent entries."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import parse_panel
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_panel(regions["panel"])
    preflop = result["columns"][1]
    assert_true(len(preflop["entries"]) > 0, "PreFlop should have entries")
    hero_entries = [e for e in preflop["entries"] if e["type"] == "hero"]
    assert_true(len(hero_entries) > 0, "should find at least one hero entry")


@test
def test_ocr_position_alias_mapping():
    """OCR: MP→LJ, MP1→HJ position alias mapping."""
    from ocr.panel_parser import normalize_position
    assert_eq(normalize_position("MP"), "LJ")
    assert_eq(normalize_position("MP1"), "HJ")
    assert_eq(normalize_position("MP2"), "HJ")
    assert_eq(normalize_position("EP"), "UTG")
    assert_eq(normalize_position("CO"), "CO")


@test
def test_ocr_position_corrupt_digit_to_letter():
    """OCR: UTG1 badge misread as UTGT/UTGI/UTGL should still resolve to UTG+1.

    Regression for H2766 where BBJordan's UTG1 panel badge was OCR'd
    as 'UTGT' (digit 1 misread as letter T, conf=0.54). The substring
    matcher used to fall through to 'UTG', collapsing hero's position
    to UTG+0 and cascading into a wrong multiway simplification that
    dropped turn solver data entirely.
    """
    from ocr.panel_parser import _preprocess_ocr_position, normalize_position
    # Digit 1 misread variants → canonical UTG1
    assert_eq(_preprocess_ocr_position("UTGT"), "UTG1")
    assert_eq(_preprocess_ocr_position("UTGI"), "UTG1")
    assert_eq(_preprocess_ocr_position("UTGL"), "UTG1")
    assert_eq(_preprocess_ocr_position("UTGt"), "UTG1")
    assert_eq(_preprocess_ocr_position("UTG 1"), "UTG1")
    # UTG2 corrupt reads
    assert_eq(_preprocess_ocr_position("UTGZ"), "UTG2")
    assert_eq(_preprocess_ocr_position("UTG 2"), "UTG2")
    # Untouched when the text is already correct
    assert_eq(_preprocess_ocr_position("UTG"), "UTG")
    assert_eq(_preprocess_ocr_position("UTG1"), "UTG1")
    assert_eq(_preprocess_ocr_position("CO"), "CO")
    # End-to-end: corrupt badge → canonical → aliased position
    assert_eq(normalize_position(_preprocess_ocr_position("UTGT")), "UTG+1")


@test
def test_ocr_action_pattern_allin_misread():
    """OCR: All-In sticker tolerates 'll'→'II' / '1l' / 'lI' misreads but
    rejects player usernames that embed 'All-In' as a substring.

    Regression for H2842 where the hero's flop all-in sticker was OCR'd as
    'AII-In' and dropped (silently treated as a player_name on the next
    Call entry, mis-recording the final action as a hero call). The fix
    broadens the action regex to accept 'A[lI1]{2}.?[Ii1][nNuU]', then
    guards against false positives like H2774's 'AIl-In Steed' username
    by checking that no extra alphabetic word remains after stripping the
    matched action and standard position/BB/number tokens.
    """
    from ocr.panel_parser import (
        _ACTION_PATTERNS, _ACTION_RESIDUE_STRIP_RE, _looks_like_allin_match,
        _normalize_action,
    )
    import re

    def is_real(text: str) -> bool:
        m = _ACTION_PATTERNS.search(text)
        if not m:
            return False
        if not _looks_like_allin_match(m.group(1)):
            return True
        residue = text.replace(m.group(0), " ", 1)
        residue = _ACTION_RESIDUE_STRIP_RE.sub(" ", residue)
        return not re.search(r"[A-Za-z]{2,}", residue)

    # Real action stickers
    assert_true(is_real("All-In"), "All-In should match")
    assert_true(is_real("AII-In"), "AII-In (OCR ll→II) should match")
    assert_true(is_real("AIl-In"), "AIl-In (OCR ll→Il) should match")
    assert_true(is_real("All-in"), "All-in (lowercase n) should match")
    # Player names that contain All-In as a substring must NOT match
    assert_true(not is_real("AIl-In Steed"),
                "username 'AIl-In Steed' must not match")
    assert_true(not is_real("All-In Cowboy"),
                "username 'All-In Cowboy' must not match")
    assert_true(not is_real("AllInHero"),
                "no-hyphen camel-case username must not match (no boundary)")
    # _normalize_action recovers the canonical label even from corrupt reads
    assert_eq(_normalize_action("AII-In"), "All-In")
    assert_eq(_normalize_action("AIl-In"), "All-In")
    assert_eq(_normalize_action("Al-In"), "All-In")
    assert_eq(_normalize_action("All-In"), "All-In")


@test
def test_ocr_action_pattern_raise_misread_as_ralse():
    """OCR: Raise sticker tolerates i/l/I/1 confusion.

    Phase-1 OCR-99 inspection found multiple position_wrong hands where
    EasyOCR read a preflop Raise row as "Ralse". The panel parser then
    treated the group as a player name, dropping the raise row and
    undercounting table size, which shifted hero_position.
    """
    from ocr.panel_parser import _ACTION_PATTERNS, _classify_group, _normalize_action
    import numpy as np

    for text in ("Ralse", "RaIse", "Ra1se", "Raise"):
        assert_true(_ACTION_PATTERNS.search(text) is not None, f"{text} should match")
        assert_eq(_normalize_action(text), "Raise")

    column_region = np.zeros((140, 240, 3), dtype=np.uint8)
    group = [
        {"text": "Ralse", "center_y": 70, "center_x": 80},
        {"text": "2.4 BB", "center_y": 88, "center_x": 80},
    ]
    entry = _classify_group(group, column_region)
    assert_true(entry is not None, "Ralse group should classify as an action")
    assert_eq(entry["action"], "Raise")
    assert_eq(entry["size"], 2.4)


@test
def test_ocr_focused_crop_recovers_missing_bb_amount():
    """OCR: focused action-sticker re-read recovers a digit lost in full-column OCR.

    TM5867249527's white BB 5-bet sticker was read as ``Ralse BB`` in the
    full-column pass; a tight 2x crop reads ``Raise 5 BB``.  Recovering the
    size keeps the all-in preflop chain assemblable instead of forcing a
    parse-none/full-Gemini fallback.
    """
    import cv2
    from pathlib import Path
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import (
        _classify_group,
        _group_by_y,
        _split_multi_action_groups,
        split_columns,
    )
    from ocr.ocr_utils import ocr_full_image

    img_path = REPO_ROOT / "data" / "hand_images" / "img" / "TM5867249527.png"
    if not img_path.exists():
        return
    image = cv2.imread(str(img_path))
    regions = detect_regions(image)
    preflop = split_columns(regions["panel"])[1]
    groups = _split_multi_action_groups(
        _group_by_y(ocr_full_image(preflop["region"]), y_threshold=25)
    )
    target = next(
        g for g in groups
        if "Ralse" in " ".join(t["text"] for t in g)
        and "BB" in " ".join(t["text"] for t in g)
        and "29" not in " ".join(t["text"] for t in g)
    )
    entry = _classify_group(target, preflop["region"])
    assert_true(entry is not None, "target group should classify")
    assert_eq(entry["action"], "Raise")
    assert_eq(entry["size"], 5.0, "focused crop should recover the missing 5 BB")


@test
def test_ocr_split_amount_group_attaches_to_previous_action():
    """OCR: a standalone ``2 BB`` group belongs to the previous Raise sticker.

    On tall screenshots EasyOCR can put the yellow ``Raise`` text and its
    ``2 BB`` amount more than the y-group threshold apart.  The amount-only
    group must be attached back to the preceding action instead of leaving the
    raise sizeless and parse-none.
    """
    import cv2
    from pathlib import Path
    from ocr.region_detector import detect_regions
    from ocr.panel_parser import parse_panel

    img_path = REPO_ROOT / "data" / "hand_images" / "img" / "TM5901972230.png"
    if not img_path.exists():
        return
    image = cv2.imread(str(img_path))
    regions = detect_regions(image)
    preflop = parse_panel(regions["panel"])["columns"][1]["entries"]
    hero_raise = next(e for e in preflop if e.get("type") == "hero")
    assert_eq(hero_raise["action"], "Raise")
    assert_eq(hero_raise["size"], 2.0)


@test
def test_resolve_allin_attribution_opp_shoves_hero_calls_deeper():
    """panel_parser: opponent donk-shoves all-in, hero calls with the
    deeper stack — hero must be the CALLER, never re-classified as the
    raiser/all-in aggressor.

    Regression for H2881 (river). N8's showdown layout stacks the
    short-stack's "Bet 11 / All-In" sticker, then the hero's "Call 11"
    sticker, then the all-in player's avatar+cards reveal. OCR splits
    the bare red All-In badge into its own nameless entry (with a
    garbled size = 11+11 = 22) sitting between the real shove and the
    real call, fabricating a phantom "hero All-In 22". The bot then
    told the coach hero RAISED all-in (a "serious mistake") when hero
    in fact just called the shove with a much bigger stack. The two
    money outcomes are equivalent because hero covers villain, but the
    action attribution — and therefore the coaching narrative — must
    distinguish who shoved vs who called.
    """
    from ocr.panel_parser import _resolve_allin_attribution

    raw = [
        {"type": "opponent", "position": "SB", "action": "Bet",
         "size": 11.0, "player_name": "Ciulo84"},
        {"type": "hero", "position": None, "action": "All-In",
         "size": 22.0},
        {"type": "opponent", "position": "BB", "action": "Call",
         "size": 11.0},
    ]
    out = _resolve_allin_attribution(raw)

    assert_eq(len(out), 2,
              "phantom All-In + split Call must collapse to shove + 1 call")
    shove, resp = out
    # The short stack (SB) is the one who is all-in.
    assert_eq(shove["type"], "opponent", "SB is the shover")
    assert_eq(shove["position"], "SB", "shover position preserved")
    assert_eq((shove["action"] or "").lower(), "all-in",
              "the donk bet that carried the red badge IS the all-in")
    assert_eq(shove["size"], 11.0, "shove size is the real 11bb, not 22")
    # Hero is the caller — NOT a raiser, NOT all-in (hero covers villain).
    assert_eq(resp["type"], "hero", "hero is the responder")
    assert_eq(resp["action"], "Call",
              "hero called the shove; must never be Raise/All-In")
    assert_eq(resp["size"], 11.0, "hero call matches the 11bb shove")


@test
def test_resolve_allin_attribution_hero_shoves_opp_calls_unchanged():
    """panel_parser: hero shoves all-in and opponent calls — the canonical
    [shover All-In, responder Call] shape must survive unchanged (guards
    the H2842/H2852 hero-all-in path against the new resolver)."""
    from ocr.panel_parser import _resolve_allin_attribution

    raw = [
        {"type": "hero", "position": None, "action": "All-In", "size": 11.0},
        {"type": "opponent", "position": "SB", "action": "Call",
         "size": 11.0, "player_name": "Villain"},
    ]
    out = _resolve_allin_attribution(raw)
    assert_eq(len(out), 2, "shape preserved")
    assert_eq((out[0]["action"] or "").lower(), "all-in", "hero still all-in")
    assert_eq(out[0]["type"], "hero")
    assert_eq(out[1]["action"], "Call", "opponent still calling")
    assert_eq(out[1]["type"], "opponent")
    assert_eq(out[1]["size"], 11.0)


@test
def test_resolve_allin_attribution_short_hero_calls_opp_shove():
    """panel_parser: when hero calls all-in for less after an opponent
    shove, N8 may OCR a trailing hero All-In badge from the showdown reveal.
    That badge is not a raise; collapse to opponent All-In + hero Call.

    Regression for H2896 turn: BB shoves 23.2bb, HJ calls remaining
    17.5bb all-in. The OCR fragments were
    [BB All-In 23.2, BB Call 17.5, hero All-In 23.2], which made the
    solver walk an impossible RAI-C-X turn node and print "no solver data".
    """
    from ocr.panel_parser import _resolve_allin_attribution

    raw = [
        {"type": "opponent", "position": "BB", "action": "All-In",
         "size": 23.2, "player_name": "HiagoS"},
        {"type": "opponent", "position": "BB", "action": "Call",
         "size": 17.5},
        {"type": "hero", "position": None, "action": "All-In",
         "size": 23.2},
    ]
    out = _resolve_allin_attribution(raw)

    assert_eq(len(out), 2, "trailing all-in badge must be dropped")
    shove, resp = out
    assert_eq(shove["type"], "opponent", "BB is the shover")
    assert_eq(shove["position"], "BB", "shover position preserved")
    assert_eq((shove["action"] or "").lower(), "all-in", "BB shove preserved")
    assert_eq(shove["size"], 23.2, "shove size preserved")
    assert_eq(resp["type"], "hero", "hero is the responder")
    assert_eq(resp["action"], "Call", "hero called; must not become all-in raise")
    assert_eq(resp["size"], 17.5, "hero call size comes from the call sticker")


@test
def test_resolve_allin_attribution_opp_shoves_hero_folds():
    """panel_parser: opponent bet carries the All-In badge, hero folds —
    collapse to [opponent All-In, hero Fold] (no phantom call)."""
    from ocr.panel_parser import _resolve_allin_attribution

    raw = [
        {"type": "opponent", "position": "BTN", "action": "Bet",
         "size": 8.0, "player_name": "Shover"},
        {"type": "opponent", "position": None, "action": "All-In",
         "size": None},
        {"type": "hero", "position": None, "action": "Fold", "size": None},
    ]
    out = _resolve_allin_attribution(raw)
    assert_eq(len(out), 2, "bare badge collapses into the bet")
    assert_eq((out[0]["action"] or "").lower(), "all-in",
              "opponent bet promoted to all-in by its badge")
    assert_eq(out[0]["type"], "opponent")
    assert_eq(out[0]["size"], 8.0)
    assert_eq(out[1]["action"], "Fold", "hero folded to the shove")
    assert_eq(out[1]["type"], "hero")


@test
def test_resolve_allin_attribution_normal_line_untouched():
    """panel_parser: a normal bet/call line with no all-in must pass
    through the resolver completely unchanged (no over-collapsing)."""
    from ocr.panel_parser import _resolve_allin_attribution

    raw = [
        {"type": "opponent", "position": "SB", "action": "Check",
         "size": None, "player_name": "V"},
        {"type": "hero", "position": "BB", "action": "Bet", "size": 5.0},
        {"type": "opponent", "position": "SB", "action": "Call",
         "size": 5.0, "player_name": "V"},
    ]
    out = _resolve_allin_attribution(raw)
    assert_eq(out, raw, "no all-in → resolver is a no-op")


@test
def test_ocr_collapse_preflop_raise_jam():
    """OCR: bare preflop All-In overlay collapses onto the preceding raise.

    Regression for H2878. N8 stamps a small red "All-In" badge on a
    preflop raise sticker when the raise is for all chips. Full-column
    OCR splits it into a separate entry (no name, no position, no size)
    that the red-sticker heuristic mis-tags `hero`. Left alone it shifts
    index-based position assignment (hero parsed as BTN instead of BB)
    and trips the all-in post-pass into flipping the real hero's call to
    opponent. The overlay must fold into the raiser, promoting it to
    All-In and keeping its size. Genuine jams (which carry a position
    badge) and standalone jams (no preceding raise) must be left intact.
    """
    from ocr.panel_parser import _collapse_preflop_raise_jam

    # H2878 preflop entries as produced just before the collapse:
    # CO raise-jam 3.5, SB raise-jam 11.1, hero (BB) calls. Both bare
    # All-In overlays were mis-tagged hero by the red-sticker heuristic.
    entries = [
        {"type": "opponent", "player_name": "Papito alva .", "action": "Fold", "position": "UTG", "size": None},
        {"type": "opponent", "player_name": "AKSyang8899", "action": "Fold", "position": "LJ", "size": None},
        {"type": "opponent", "player_name": "bronice", "action": "Raise", "position": "CO", "size": 3.5},
        {"type": "hero", "player_name": None, "action": "All-In", "position": None, "size": None},
        {"type": "opponent", "player_name": "Robl297", "action": "Fold", "position": "BTN", "size": None},
        {"type": "opponent", "player_name": "DCP1975", "action": "Raise", "position": "SB", "size": 11.1},
        {"type": "hero", "player_name": None, "action": "All-In", "position": None, "size": None},
        {"type": "hero", "player_name": None, "action": "Call", "position": "BB", "size": 10.1},
    ]
    out = _collapse_preflop_raise_jam(entries)
    assert_eq(len(out), 6, "two overlay badges dropped")
    assert_eq(out[2]["action"], "All-In", "CO raise promoted to All-In")
    assert_eq(out[2]["size"], 3.5, "CO all-in size preserved")
    assert_eq(out[4]["action"], "All-In", "SB raise promoted to All-In")
    assert_eq(out[4]["size"], 11.1, "SB all-in size preserved")
    last = out[-1]
    assert_eq(last["type"], "hero", "real hero call survives, still hero")
    assert_eq(last["action"], "Call", "real hero action unchanged")
    assert_eq(last["position"], "BB", "real hero position unchanged")
    assert_true(
        not any(e.get("action") == "All-In" and not e.get("player_name")
                for e in out),
        "no nameless All-In overlay remains",
    )

    # Negative: a genuine jam-over-raise carries a position badge — the
    # raiser must NOT be collapsed (villain 3-bet jam stays a distinct
    # action).
    villain_jam = [
        {"type": "opponent", "player_name": "opener", "action": "Raise", "position": "CO", "size": 2.0},
        {"type": "opponent", "player_name": None, "action": "All-In", "position": "BTN", "size": None},
    ]
    out2 = _collapse_preflop_raise_jam(villain_jam)
    assert_eq(len(out2), 2, "positioned jam is not an overlay — kept")
    assert_eq(out2[0]["action"], "Raise", "opener raise left intact")

    # Negative: a standalone jam with no preceding raise (first aggressor)
    # must not be folded into a fold entry.
    standalone = [
        {"type": "opponent", "player_name": "u", "action": "Fold", "position": "UTG", "size": None},
        {"type": "hero", "player_name": None, "action": "All-In", "position": None, "size": None},
    ]
    out3 = _collapse_preflop_raise_jam(standalone)
    assert_eq(len(out3), 2, "no preceding raise — jam kept")
    assert_eq(out3[1]["action"], "All-In", "standalone jam preserved")


@test
def test_ocr_table_parser_board_cards():
    """OCR: table parser finds board cards."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.table_parser import parse_table
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_table(regions["table"])
    assert_true(len(result["board_cards"]) >= 3, f"should find >=3 board cards, got {len(result['board_cards'])}")


@test
def test_ocr_h3429_win_sticker_corner_rank_reads_pocket_twos():
    """OCR: WIN sticker noise must not turn a visible 2h corner into Kh."""
    from ocr.n8_parser import parse_n8_screenshot

    img_path = REPO_ROOT / "tests" / "fixtures" / "ocr" / "H3429.jpeg"
    result = parse_n8_screenshot(img_path.read_bytes())
    assert_true(result.get("hand"), "H3429 should parse into a hand")
    assert_eq(result["hand"].get("hero_hand"), "2h2c")


@test
def test_ocr_card_confidence_surfaced_separately():
    """OCR: parse_n8_screenshot exposes card_confidence on the result so
    the gemini_session tiered gate can apply a hard card-conf floor.

    Regression for H2772: card_confidence=0.66 (CardCNN classified hero K
    as 8 with rank conf 0.56) but the blended overall confidence reached
    0.86 thanks to good action tracking, slipping through the MEDIUM
    gate. We need card_confidence to be visible to the gate so it can be
    treated as a hard floor independent of action-tracking quality.
    """
    # Synthetic check: the field is wired through. Real CardCNN
    # behavior is exercised via the snapshot tests.
    from ocr.n8_parser import _compute_confidence
    parts = {
        "pot_consistency": 1.0, "player_tracking": 1.0,
        "ocr_confidence": 1.0, "card_confidence": 0.55,
    }
    blended = _compute_confidence(parts)
    # Sanity: blended can mask a weak card_confidence.
    assert_true(blended > 0.80,
                f"action-tracking should mask weak card_conf; got {blended}")
    # The fix is gemini_session checking card_confidence directly, so the
    # parser must surface it on its return dict.
    import inspect
    src = inspect.getsource(__import__("ocr.n8_parser", fromlist=["_dummy"]))
    assert_in('"card_confidence":', src)


@test
def test_ocr_bails_when_raise_size_missing():
    """OCR: _assemble_hand returns hand=None when any preflop raise/bet
    entry has size=None.

    Regression for H2823: panel cell "Raise 7 BB" had its size lost in
    OCR. _action_to_code silently substituted the "R2" min-raise default,
    which corrupted _compute_preflop_pot (5.5bb instead of 15.5bb), and
    _find_action_by_pot_pct mapped the next 8bb flop bet to RAI (145%
    of the fake pot). flop_actions ended up "X-RAI-C" — the solver tree
    treated that as terminal so turn/river dropped out and the API
    rejected the spot-solution call. Returning None forces full Gemini
    fallback which can re-read the panel.
    """
    from ocr.n8_parser import _assemble_hand
    table_result = {
        "board_cards": ["9s", "Ad", "7s"],
        "hero_cards": ["Ac", "4c"],
        "hero_card_conf": 0.95,
        "hero_card_details": [],
        "table_color": "green",
        "action_entries": [
            {"type": "opponent", "position": "UTG", "action": "Fold", "size": None},
            {"type": "opponent", "position": "UTG+1", "action": "Fold", "size": None},
            {"type": "hero", "position": "HJ", "action": "Raise", "size": 2.2},
            {"type": "opponent", "position": "CO", "action": "Raise", "size": None},  # missing
            {"type": "opponent", "position": "BTN", "action": "Fold", "size": None},
        ],
    }
    columns = [
        {"name": "Pre-Flop", "pot": 2.6, "entries": table_result["action_entries"]},
        {"name": "Flop", "pot": 16.6, "entries": []},
    ]
    hand, conf_parts, _diagnostics = _assemble_hand(table_result, columns)
    assert_true(hand is None,
                f"_assemble_hand should return None when a raise has no size; got {hand}")
    assert_eq(conf_parts["ocr_confidence"], 0.0,
              "ocr_confidence should be zeroed when a raise size is missing")


@test
def test_multiway_simplification_remaps_dropped_opponent_bets():
    """Multiway HU simplification: when the postflop bettor is the dropped
    third player (not in {hero, kept_villain}), remap their bet/raise onto
    the kept villain so hero's response matches a real solver spot.

    Regression for H2830: 6-max SB ATo, HJ opens, SB+BB cold-call. Flop
    is SB X, BB X, HJ R2.3, SB C, HJ C. The simplifier kept SB+BB and
    dropped HJ. Without remapping, the action loop produced
    flop_actions="X-X-C" — hero "calling" a non-existent bet — and
    every hero spot from the call onward returned no solver data.
    """
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "hero_hand": "AsTc",
        "effective_bb": 52.5,
        "hero_position": "SB",
        "preflop_actions": "F-R2-F-F-C-C",
        "players_at_table": 6,
        "hero_starting_stack": 72.3,
        "streets": [
            {"board": "5d6cAd", "actions": [
                {"action": "X", "position": "SB"},
                {"action": "X", "position": "BB"},
                {"size": 2.3, "action": "R2.3", "position": "HJ"},
                {"size": 2.3, "action": "C", "position": "SB"},
                {"size": 2.3, "action": "C", "position": "HJ"},
            ]},
            {"card": "5s", "actions": [
                {"action": "X", "position": "SB"},
                {"action": "X", "position": "BB"},
                {"size": 10.3, "action": "R10.3", "position": "HJ"},
                {"size": 10.3, "action": "C", "position": "SB"},
                {"action": "F", "position": "HJ"},
            ]},
            {"card": "9s", "actions": [
                {"action": "X", "position": "SB"},
                {"action": "F", "position": "SB"},
            ]},
        ],
    }
    text = analyze_hand_full(hand)["text"]
    flop_section = text.split("【Flop:")[1].split("==")[0]
    assert_true("無 solver 數據" not in flop_section,
                "Flop should have solver data after multiway remap")
    turn_section = text.split("【Turn:")[1].split("==")[0]
    assert_true("無 solver 數據" not in turn_section,
                "Turn should have solver data after multiway remap")


@test
def test_hero_pair_healthy_rejects_degenerate_geometry():
    """OCR: _hero_pair_healthy gates the whiteness-localizer retry.

    The bright-blob localizer fails on WIN-sticker / window-clipped / merged
    flag-badge cases by latching onto a ~40px-tall sliver or a >2.5-aspect
    merged blob. Those degenerate crops collapse CardCNN to ~0.13 noise and
    force the Gemini cards-only fallback (~43% of live screenshots). The
    geometry check must accept a square-ish ~120px pair and reject the
    degenerate shapes so the retry path can fire.
    """
    import numpy as np
    from ocr import table_parser
    healthy = [np.zeros((120, 58, 3), np.uint8), np.zeros((120, 58, 3), np.uint8)]
    sliver = [np.zeros((41, 34, 3), np.uint8), np.zeros((41, 33, 3), np.uint8)]
    too_wide = [np.zeros((85, 140, 3), np.uint8), np.zeros((85, 80, 3), np.uint8)]
    assert_true(table_parser._hero_pair_healthy(healthy),
                "~120px square-ish pair must be healthy")
    assert_true(not table_parser._hero_pair_healthy(sliver),
                "41px sliver (WIN-sticker fragment / window clip) must be rejected")
    assert_true(not table_parser._hero_pair_healthy(too_wide),
                "merged flag/badge wide blob (ar>2.5) must be rejected")
    assert_true(not table_parser._hero_pair_healthy([]), "empty pair not healthy")


@test
def test_find_hero_cards_confidence_gated_three_stage():
    """OCR: _find_hero_cards is a confidence-gated 3-stage localizer — bright,
    then whiteness on the table region, then whiteness on a divider-spanning
    band of the full image — each adopted only if strictly more confident.

    This cuts the cards-only Gemini fallback rate (raw-CNN TM corpus 4.8% →
    0.3%, 83 hands fixed / 0 regressed) without disturbing confident reads:
    every retry is gated on `< HERO_RELOCATE_CONF` and `cand[1] > result[1]`,
    so already-correct high-confidence hands are untouched. Stage 3 needs the
    full image + divider_y because hero pairs are often clipped by the divider.
    """
    import inspect
    from ocr import table_parser
    src = inspect.getsource(table_parser._find_hero_cards)
    assert_in("_locate_hero_bright", src, "default pass is the bright localizer")
    assert_in("HERO_RELOCATE_CONF", src, "retries must be confidence-gated")
    assert_in("_locate_hero_white(table_region)", src,
              "stage 2 retries whiteness on the table region")
    assert_in("divider_y=divider_y", src,
              "stage 3 retries whiteness on a divider-spanning full-image band")
    assert_in("cand[1] > result[1]", src,
              "a retry is adopted only if strictly more confident")


@test
def test_locate_hero_white_recovers_win_sticker_pair():
    """OCR: _locate_hero_white isolates the white card bodies past a saturated
    WIN sticker that fragments the bright-blob localizer.

    Synthetic table: two white cards low-center with an orange sticker over
    their lower half (the live failure mode, e.g. H3436/H3454). The card body
    is high-value/low-saturation; the sticker is saturated, so whiteness
    masking + an aggressive close rebuilds the full pair rectangle.
    """
    import numpy as np
    from ocr import table_parser
    table = np.full((400, 300, 3), 45, np.uint8)  # dark felt
    # pair low-center, inside the [0.55:1.0, 0.24:0.72] whiteness window
    table[248:356, 100:148] = (255, 255, 255)      # left card (white)
    table[248:356, 152:200] = (255, 255, 255)      # right card (white)
    table[330:356, 100:200] = (0, 140, 255)        # orange WIN sticker (BGR)
    crops = table_parser._locate_hero_white(table)
    assert_eq(len(crops), 2, "must locate a two-card pair past the sticker")
    assert_true(table_parser._hero_pair_healthy(crops),
                "recovered pair must have healthy geometry")


@test
def test_locate_hero_white_divider_mode_picks_bottom_clipped_pair():
    """OCR: in divider mode _locate_hero_white searches a band straddling the
    divider and picks the BOTTOM-most pair — recovering hero cards clipped by
    the table/panel split while rejecting the board cards that sit higher.

    Dominant residual cause (TM5863068198/TM5866746802): hero pair clipped by
    the divider (lower half in the panel) renders ~69px tall; the table-only
    search saw only the board. Regression for the floor (these real pairs are
    ~69px, just under the old 70px floor) and for bottom-most selection.
    """
    import numpy as np
    from ocr import table_parser
    H, W, divider_y = 500, 300, 330
    img = np.full((H, W, 3), 45, np.uint8)
    # decoy "board" pair-shaped blob higher up (wider), inside the band:
    img[200:266, 75:205] = (255, 255, 255)              # w130 h66, bottom=266
    # hero pair lower, small (~68px) and straddling the divider:
    img[300:368, 105:150] = (255, 255, 255)             # left card
    img[300:368, 155:195] = (255, 255, 255)             # right card  -> w90 h68
    crops = table_parser._locate_hero_white(img, divider_y=divider_y)
    assert_eq(len(crops), 2, "divider mode must locate the clipped pair")
    assert_true(table_parser._hero_pair_healthy(crops),
                "~68px clipped pair must pass (floor 60, not 70)")
    total_w = crops[0].shape[1] + crops[1].shape[1]
    assert_true(total_w < 115,
                f"must pick the bottom hero pair (~96px) not the wider board "
                f"decoy (~136px); got width {total_w}")


@test
def test_parse_table_plumbs_full_image_for_hero_localization():
    """OCR: parse_table forwards the full image + divider_y to hero
    localization so stage-3 (divider-spanning) can fire; n8_parser supplies
    them. Without this plumbing the clipped-hero recovery is dead code."""
    import inspect
    from ocr import table_parser, n8_parser
    pt = inspect.getsource(table_parser.parse_table)
    assert_in("full_image=full_image", pt, "parse_table must pass full_image on")
    assert_in("divider_y=divider_y", pt, "parse_table must pass divider_y on")
    caller = inspect.getsource(n8_parser.parse_n8_screenshot)
    assert_in("full_image=image", caller, "n8_parser must supply the full image")


@test
def test_find_hero_cards_takes_rank_from_raw_suit_from_masked():
    """OCR: _find_hero_cards classifies both raw and masked crops, taking
    rank from the raw prediction (rank corner sits at the top — masking
    the bottom WIN sticker can only confuse the rank head) and suit from
    the masked prediction (orange WIN pixels bleed red, flipping ♣→♥).

    Regression for H2829: Q♣ was misread as A at rank_conf 0.95 because
    the WIN mask whitened the bottom half of the crop, removing the Q's
    distinctive lower-right tail. Raw rank head correctly read Q at 0.75.
    The mask still helps suit, so we keep it for that head only.
    """
    import inspect
    from ocr import table_parser
    # The raw+masked classification body lives in _classify_hero_crops, shared
    # by both localizer passes in _find_hero_cards (bright + whiteness retry).
    src = inspect.getsource(table_parser._classify_hero_crops)
    assert_in("classify_batch_detailed_tta(crops)", src,
              "_classify_hero_crops should classify the raw crops too")
    assert_in("classify_batch_detailed_tta(masked_crops)", src,
              "_classify_hero_crops should classify the masked crops too")
    # Sanity: rank starts from raw, can be repaired by raw top-2/corner OCR,
    # and suit comes from the masked crop.
    assert_in('raw["rank"]', src)
    assert_in("_rank_from_corner_ocr(crops[i])", src)
    assert_in('suit = masked["suit"]', src)


@test
def test_ocr_card_confidence_not_boosted_by_board():
    """OCR: card_confidence in _assemble_hand reflects raw hero CardCNN
    confidence — no synthetic boost from board legibility.

    Regression for H2822: hero 8s8d misclassified as 9s8d at 0.611. A
    legacy +0.1 board-cards boost lifted card_confidence to 0.711, just
    above the 0.70 MIN_CARD_CONF gate in gemini_session, so the
    cards-only Gemini fallback never fired and the wrong hand shipped.
    Board CardCNN predictions are independent of hero predictions, so
    boosting hero confidence based on board legibility is invalid.
    """
    from ocr.n8_parser import _assemble_hand
    table_result = {
        "board_cards": ["6d", "Td", "5c", "3c", "5h"],  # full 5-card board
        "hero_cards": ["9s", "8d"],
        "hero_card_conf": 0.611,                          # weak hero CNN
        "hero_card_details": [],
        "table_color": "green",
    }
    _hand, conf_parts, _diagnostics = _assemble_hand(table_result, columns=[])
    assert_eq(conf_parts["card_confidence"], 0.611,
              "card_confidence should equal raw hero_card_conf, not get a "
              "+0.1 boost from board-cards being legible")


@test
def test_ocr_hero_card_suits_hint_emitted():
    """OCR: high-conf suit predictions are surfaced as hero_card_suits hint
    even when ranks are uncertain or hero_cards got cleared.

    Regression for H2768: CardCNN predicted (9h, 9h) — same rank twice due
    to 8↔9 confusion — but suit-head conf was 0.97 for both. The duplicate
    triggered hero_cards clearing, which dropped the only suit signal
    Gemini had. After the fix, _build_hints emits hero_card_suits=['h', 'h']
    so Gemini's prompt can fix the rank without re-guessing the suit.
    """
    from ocr.n8_parser import _build_hints
    table_result = {
        "board_cards": ["6d", "Qh", "5d", "Jd", "Qd"],
        "hero_cards": [],   # cleared by hero/board duplicate resolution
        "hero_card_details": [
            {"rank": "9", "rank_conf": 0.62, "suit": "h", "suit_conf": 0.97,
             "conf": 0.62},
            {"rank": "9", "rank_conf": 0.51, "suit": "h", "suit_conf": 0.97,
             "conf": 0.51},
        ],
    }
    hints = _build_hints(table_result, [], None)
    assert_eq(hints.get("hero_card_suits"), ["h", "h"])

    # Sanity: when suit confidence is below threshold, no hint is emitted.
    table_result["hero_card_details"][0]["suit_conf"] = 0.55
    hints2 = _build_hints(table_result, [], None)
    assert_true(
        "hero_card_suits" not in (hints2 or {}),
        "low-conf suits should NOT emit hero_card_suits hint",
    )


@test
def test_find_hero_stack_prefers_bb_suffix():
    """Two-pass scan: prefer any 'XX.X BB' match over a plain number.

    Regression: H2798 — hero crop OCR returned 5 text regions:
      ['gorj', '24', 'B', 'cbd191320', '11.5 BB']
    Per-result fallback latched onto '24' (a fragment from an adjacent UI
    element) at conf 0.87 because it matched the plain-number regex,
    returning 24.0 and never seeing the real '11.5 BB' entry that came
    later. Effective_bb cascaded to 26.0 instead of 13.5.
    """
    import numpy as np
    sys.path.insert(0, str(SCRIPTS_DIR))
    import ocr.table_parser as _tp

    fake_results = [
        {"text": "gorj",       "conf": 1.00},
        {"text": "24",         "conf": 0.87},
        {"text": "B",          "conf": 1.00},
        {"text": "cbd191320",  "conf": 1.00},
        {"text": "11.5 BB",    "conf": 1.00},
    ]
    orig = _tp.ocr_full_image if hasattr(_tp, "ocr_full_image") else None
    # The function imports ocr_full_image lazily, so patch the source module.
    import ocr.ocr_utils as _ou
    orig = _ou.ocr_full_image
    _ou.ocr_full_image = lambda img: fake_results
    try:
        # Any non-empty image will do; ocr_full_image is mocked.
        fake_img = np.zeros((100, 200, 3), dtype=np.uint8) + 1
        got = _tp._find_hero_stack(fake_img)
    finally:
        _ou.ocr_full_image = orig
    assert_eq(got, 11.5,
              "should prefer '11.5 BB' over the plain '24' fragment")


@test
def test_find_hero_stack_falls_back_to_plain_number():
    """When NO 'XX.X BB' string is present, fall back to the highest-conf
    plain number in the plausible range — not the FIRST plain number, which
    can be noise like a name fragment.
    """
    import numpy as np
    sys.path.insert(0, str(SCRIPTS_DIR))
    import ocr.table_parser as _tp
    import ocr.ocr_utils as _ou

    fake_results = [
        {"text": "gorj",   "conf": 1.00},   # not numeric
        {"text": "24",     "conf": 0.60},   # plausible number, lower conf
        {"text": "12.5",   "conf": 0.95},   # plausible number, higher conf
    ]
    orig = _ou.ocr_full_image
    _ou.ocr_full_image = lambda img: fake_results
    try:
        fake_img = np.zeros((100, 200, 3), dtype=np.uint8) + 1
        got = _tp._find_hero_stack(fake_img)
    finally:
        _ou.ocr_full_image = orig
    # Highest-conf plain number wins.
    assert_eq(got, 12.5)


@test
def test_ocr_confidence_parts_exposed():
    """OCR: parse_n8_screenshot exposes confidence_parts so callers can read
    structural confidence (pot/player/ocr) separately from card_confidence.

    Required by the field-level Gemini fallback: when card_conf is below
    threshold but the structural components are strong, we want to do a
    cards-only Gemini call instead of letting the full IMAGE_PARSE_PROMPT
    re-decide hero_position/stacks/actions.
    """
    import inspect
    src = inspect.getsource(__import__("ocr.n8_parser", fromlist=["_dummy"]))
    assert_in('"confidence_parts":', src)


@test
def test_merge_ocr_with_gemini_hero_hand_keeps_structural():
    """Field-level merge replaces ONLY hero_hand and leaves every structural
    field (hero_position, stacks, actions, streets) intact.

    Regression: H2790 — when card_conf < MIN_CARD_CONF the full Gemini
    fallback was used, and Gemini's IMAGE_PARSE_PROMPT let it re-decide
    hero_position visually. It flipped the correct OCR-detected SB to BB.
    The field-level merge keeps OCR's blind-based position read.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager

    ocr_hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 6,
        "hero_position": "SB",
        "hero_hand": "Th4s",
        "effective_bb": 63,
        "player_stacks": [71.5, 90.9, 77.1, 76.5, 62.9, 84.4],
        "preflop_actions": "F-F-F-F-C-X",
        "streets": [{"board": "8cQs9c", "actions": [
            {"size": 1.0, "action": "R1", "position": "SB"},
            {"action": "C", "position": "BB"},
        ]}],
    }
    merged = GeminiSessionManager._merge_ocr_with_gemini_hero_hand(
        ocr_hand, "Th2s"
    )
    assert_eq(merged["hero_hand"], "Th2s")
    assert_eq(merged["hero_position"], "SB")
    assert_eq(merged["effective_bb"], 63)
    assert_eq(merged["player_stacks"], [71.5, 90.9, 77.1, 76.5, 62.9, 84.4])
    assert_eq(merged["preflop_actions"], "F-F-F-F-C-X")
    assert_eq(merged["streets"], ocr_hand["streets"])
    assert_eq(merged["players_at_table"], 6)
    # OCR hand must NOT be mutated.
    assert_eq(ocr_hand["hero_hand"], "Th4s")


@test
def test_field_level_fallback_used_when_structural_high():
    """When card_conf < MIN_CARD_CONF but structural_conf >= STRUCTURAL_MIN,
    _parse_hand_from_image should call _gemini_hero_hand_only and merge the
    result — never reaching the full Gemini parse path that would override
    hero_position.
    """
    import asyncio as _aio
    import logging as _l
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager

    ocr_hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 6,
        "hero_position": "SB",
        "hero_hand": "Th4s",
        "effective_bb": 63,
        "player_stacks": [71.5, 90.9, 77.1, 76.5, 62.9, 84.4],
        "preflop_actions": "F-F-F-F-C-X",
        "streets": [],
    }
    fake_ocr_result = {
        "hand": ocr_hand,
        "hints": None,
        "confidence": 0.72,
        "card_confidence": 0.40,
        "confidence_parts": {
            "pot_consistency": 1.0,
            "player_tracking": 1.0,
            "ocr_confidence": 0.95,
            "card_confidence": 0.40,
        },
    }

    import ocr.n8_parser as _np
    orig_parse = _np.parse_n8_screenshot
    _np.parse_n8_screenshot = lambda b: fake_ocr_result

    cards_only_calls = []
    async def _fake_cards_only(self, *a, **k):
        cards_only_calls.append(k)
        return "Th2s"
    orig_cards_only = GeminiSessionManager._gemini_hero_hand_only
    GeminiSessionManager._gemini_hero_hand_only = _fake_cards_only

    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session._logger = _l.getLogger("test_field_level_fallback")
    session._logger.setLevel(_l.WARNING)
    # client=None makes any full-Gemini path explode — proves we never get there
    session.client = None
    session.image_parse_model = "fake-model"

    prev_enabled = os.environ.get("OCR_ENABLED")
    prev_struct = os.environ.get("OCR_STRUCTURAL_MIN")
    os.environ["OCR_ENABLED"] = "true"
    os.environ.pop("OCR_STRUCTURAL_MIN", None)
    try:
        result = _aio.run(session._parse_hand_from_image(
            chat_id=1, image_bytes=b"\x00", mime_type="image/jpeg"
        ))
    finally:
        _np.parse_n8_screenshot = orig_parse
        GeminiSessionManager._gemini_hero_hand_only = orig_cards_only
        if prev_enabled is None:
            os.environ.pop("OCR_ENABLED", None)
        else:
            os.environ["OCR_ENABLED"] = prev_enabled
        if prev_struct is not None:
            os.environ["OCR_STRUCTURAL_MIN"] = prev_struct

    assert_true(result is not None, "should return a merged hand, not None")
    assert_eq(result["hero_position"], "SB")
    assert_eq(result["hero_hand"], "Th2s")
    assert_eq(result["effective_bb"], 63)
    assert_eq(len(cards_only_calls), 1)


@test
def test_field_level_fallback_skipped_when_structural_low():
    """When BOTH card_conf and structural_conf are below threshold,
    _parse_hand_from_image must NOT take the cards-only branch (the
    structural fields aren't trustworthy). Should fall through to the
    existing full Gemini parse path.
    """
    import asyncio as _aio
    import logging as _l
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager

    ocr_hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 6,
        "hero_position": "SB",
        "hero_hand": "Th4s",
        "preflop_actions": "F-F-F-F-C-X",
    }
    fake_ocr_result = {
        "hand": ocr_hand,
        "hints": None,
        "confidence": 0.40,
        "card_confidence": 0.30,
        "confidence_parts": {
            "pot_consistency": 0.30,
            "player_tracking": 0.40,
            "ocr_confidence": 0.50,
            "card_confidence": 0.30,
        },
    }

    import ocr.n8_parser as _np
    orig_parse = _np.parse_n8_screenshot
    _np.parse_n8_screenshot = lambda b: fake_ocr_result

    cards_only_calls = []
    async def _fake_cards_only(self, *a, **k):
        cards_only_calls.append(k)
        return "Th2s"
    orig_cards_only = GeminiSessionManager._gemini_hero_hand_only
    GeminiSessionManager._gemini_hero_hand_only = _fake_cards_only

    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session._logger = _l.getLogger("test_field_level_skipped")
    session._logger.setLevel(_l.CRITICAL)
    # Patch the full-Gemini path: client.aio.models.generate_content must be
    # reached. We make it raise a sentinel so the test knows the full path
    # was hit instead of the cards-only branch.
    class _Sentinel(Exception): pass
    class _FakeModels:
        async def generate_content(self, **kw):
            raise _Sentinel("full Gemini path reached as expected")
    class _FakeAio:
        models = _FakeModels()
    class _FakeClient:
        aio = _FakeAio()
    session.client = _FakeClient()
    session.image_parse_model = "fake-model"

    prev_enabled = os.environ.get("OCR_ENABLED")
    os.environ["OCR_ENABLED"] = "true"
    sentinel_hit = False
    try:
        try:
            _aio.run(session._parse_hand_from_image(
                chat_id=1, image_bytes=b"\x00", mime_type="image/jpeg"
            ))
        except _Sentinel:
            sentinel_hit = True
    finally:
        _np.parse_n8_screenshot = orig_parse
        GeminiSessionManager._gemini_hero_hand_only = orig_cards_only
        if prev_enabled is None:
            os.environ.pop("OCR_ENABLED", None)
        else:
            os.environ["OCR_ENABLED"] = prev_enabled

    assert_eq(len(cards_only_calls), 0,
              "cards-only fallback must NOT fire when structural_conf is low")
    assert_true(sentinel_hit,
                "full Gemini path should be reached when structural_conf is low")


@test
def test_field_level_fallback_used_for_confidence_abstain_with_ocr():
    """gemini_session: confidence-abstained OCR hands with usable structure
    should use the cards-only micro-route instead of full Gemini reparse.

    The 718-hand precision study found full-image Gemini is net-negative on
    confidence-abstained-but-present OCR parses: it often flips correct
    structure.  This locks the intended routing: keep OCR structure and only
    re-read hero cards.
    """
    import asyncio as _aio
    import logging as _l
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from gemini_session import GeminiSessionManager

    ocr_hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 7,
        "hero_position": "SB",
        "hero_hand": "8h7c",
        "preflop_actions": "F-F-F-F-R500-F-AI485-F",
    }
    fake_ocr_result = {
        "hand": ocr_hand,
        "hints": None,
        "confidence": 0.79,
        "card_confidence": 0.99,
        "confidence_parts": {
            "pot_consistency": 0.5,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
            "card_confidence": 0.99,
        },
    }

    import ocr.n8_parser as _np
    orig_parse = _np.parse_n8_screenshot
    _np.parse_n8_screenshot = lambda b: fake_ocr_result

    cards_only_calls = []
    async def _fake_cards_only(self, *a, **k):
        cards_only_calls.append(k)
        return "8h7c"
    orig_cards_only = GeminiSessionManager._gemini_hero_hand_only
    GeminiSessionManager._gemini_hero_hand_only = _fake_cards_only

    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session._logger = _l.getLogger("test_field_level_abstain")
    session._logger.setLevel(_l.WARNING)
    session.client = None
    session.image_parse_model = "fake-model"

    prev_enabled = os.environ.get("OCR_ENABLED")
    prev_abstain_struct = os.environ.get("OCR_ABSTAIN_STRUCTURAL_MIN")
    os.environ["OCR_ENABLED"] = "true"
    os.environ.pop("OCR_ABSTAIN_STRUCTURAL_MIN", None)
    try:
        result = _aio.run(session._parse_hand_from_image(
            chat_id=1, image_bytes=b"\x00", mime_type="image/jpeg"
        ))
    finally:
        _np.parse_n8_screenshot = orig_parse
        GeminiSessionManager._gemini_hero_hand_only = orig_cards_only
        if prev_enabled is None:
            os.environ.pop("OCR_ENABLED", None)
        else:
            os.environ["OCR_ENABLED"] = prev_enabled
        if prev_abstain_struct is not None:
            os.environ["OCR_ABSTAIN_STRUCTURAL_MIN"] = prev_abstain_struct

    assert_eq(len(cards_only_calls), 1)
    assert_eq(result["hero_position"], "SB")
    assert_eq(result["preflop_actions"], "F-F-F-F-R500-F-AI485-F")


@test
def test_cards_only_merge_selector_rejects_low_conf_changed_hero():
    """gemini_session: a changed cards-only hero read is accepted only when
    CardCNN was not in the ultra-low-confidence tail.

    This prevents the micro-route from replacing one bad hero read with a
    second hallucinated one while still allowing the 0.38+ confidence hero-fix
    cluster recovered in the 718-hand recall pass.
    """
    from gemini_session import GeminiSessionManager

    base = {
        "hand": {
            "hero_hand": "Ah9d",
            "hero_position": "CO",
            "preflop_actions": "R2-F-F-F-F",
        },
        "diagnostics": {},
        "confidence_parts": {
            "pot_consistency": 1.0,
            "player_tracking": 0.5,
            "ocr_confidence": 1.0,
        },
    }
    low = dict(base, card_confidence=0.30)
    high = dict(base, card_confidence=0.39)

    assert_eq(
        GeminiSessionManager._cards_only_merge_safe(low, "AdAd"),
        False,
    )
    assert_eq(
        GeminiSessionManager._cards_only_merge_safe(high, "Ad9d"),
        True,
    )


@test
def test_cards_only_merge_selector_accepts_vlm_hidden_three_single_allin_raise():
    """gemini_session: VLM-corrected hidden-three all-in/raise tails can keep
    OCR structure when Gemini confirms hero cards unchanged.

    TM5873873878/TM5875585050-like shapes were exact OCR abstains: the VLM
    corrected seat structure, cards are high confidence, and the action tail
    has one all-in plus one raise ending in a call.
    """
    from gemini_session import GeminiSessionManager

    ocr_result = {
        "hand": {
            "hero_hand": "AhAc",
            "hero_position": "BB",
            "preflop_actions": "F-F-F-F-F-C-R3-AI52-C",
        },
        "card_confidence": 0.999,
        "confidence_parts": {
            "pot_consistency": 1.0,
            "player_tracking": 0.5,
            "ocr_confidence": 0.0,
        },
        "diagnostics": {
            "vlm_recheck_outcome": "corrected",
            "preflop_entries_count": 9,
            "preflop_entries_pre_collapse_count": 16,
            "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
            "street_entries_pre_collapse_count": {
                "flop": 0,
                "turn": 0,
                "river": 3,
            },
        },
    }

    assert_eq(
        GeminiSessionManager._cards_only_merge_safe(ocr_result, "AhAc"),
        True,
    )


@test
def test_ocr_table_color_detection():
    """OCR: table parser detects table color."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.table_parser import parse_table
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    result = parse_table(regions["table"])
    assert_true(result["table_color"] in ("green", "purple", "dark", "unknown"), f"unexpected: {result['table_color']}")


@test
def test_ocr_n8_parser_full_pipeline():
    """OCR: full N8 parser produces hand JSON from screenshot."""
    from ocr.n8_parser import parse_n8_screenshot
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    with open(img_path, "rb") as f:
        result = parse_n8_screenshot(f.read())
    assert_true(result["confidence"] > 0, "should have non-zero confidence")
    if result["hand"]:
        hand = result["hand"]
        assert_true(hand.get("hero_position") is not None, "should have hero_position")
        assert_true(hand.get("preflop_actions") is not None, "should have preflop_actions")


@test
def test_ocr_table_size_from_entry_count():
    """OCR: table size inferred from preflop entry count."""
    from ocr.n8_parser import _estimate_table_size
    # 8 entries = 8 players
    entries = [{"type": "opponent"}] * 7 + [{"type": "hero"}]
    assert_eq(_estimate_table_size(entries)[0], 8)
    # 6 entries = 6 players
    entries = [{"type": "opponent"}] * 5 + [{"type": "hero"}]
    assert_eq(_estimate_table_size(entries)[0], 6)
    # 2 entries = 2 players (min)
    entries = [{"type": "hero"}, {"type": "opponent"}]
    assert_eq(_estimate_table_size(entries)[0], 2)


@test
def test_ocr_filter_false_hero_entries():
    """OCR: false hero entries (avatar markers) are filtered out."""
    from ocr.n8_parser import _filter_action_entries
    entries = [
        {"type": "opponent", "action": "Fold"},
        {"type": "hero", "action": ", 3"},       # false — no action word
        {"type": "hero", "action": "Raise"},      # real action
        {"type": "opponent", "action": "Fold"},
    ]
    filtered = _filter_action_entries(entries)
    assert_eq(len(filtered), 3, f"expected 3, got {len(filtered)}")
    assert_eq(filtered[1]["action"], "Raise")


# ── Padding + Multiway Tests ──


@test
def test_6max_lj_open_qjo_is_raise():
    """QJo E2E: 6-player LJ open QJo at 33bb must show RAISE 100%, not fold."""
    from analyze_hand import analyze_hand_full
    # Exact scenario from OCR: 6-player table, OCR detected 7 stacks (noise)
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "QsJd",
        "hero_position": "LJ",
        "players_at_table": 6,
        "effective_bb": 33,
        "preflop_actions": "R2.2-F-C-F-F-C",
        "player_stacks": [66.5, 31.0, 107.5, 48.0, 36.9, 10.8, 25.3],
        "streets": [
            {"board": "6c2dTs", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
            {"card": "Ad", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R4", "size": 4.0},
                {"position": "CO", "action": "F"},
                {"position": "BB", "action": "C", "size": 4.0},
            ]},
        ],
    })
    # QJo at LJ open = 100% RAISE, not fold
    assert_in("RAISE", result["text"], "QJo should show RAISE in solver data")
    assert_true(
        "Fold: 100.0%" not in result["text"] or "【LJ QJo】" not in result["text"],
        "QJo must NOT show Fold 100%"
    )
    # Verify padding: preflop should start with F-F (2 pads for 6→8)
    assert_true(
        result["preflop_actions"].startswith("F-F-R"),
        f"Should pad 2 folds, got: {result['preflop_actions']}"
    )
    # After CO folds on turn, should simplify to LJ vs BB HU
    # Turn/River should attempt solver data (not all "無 solver 數據")
    assert_in("LJ", result["text"])
    assert_in("BB", result["text"])


@test
def test_6max_padding_uses_players_at_table():
    """Padding: 6-player table pads to 8 even if player_stacks has 7 elements."""
    from analyze_hand import analyze_hand_full
    # OCR may detect 7 stacks for a 6-player table (noise).
    # players_at_table=6 must take priority, padding 2 folds.
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "QJo",
        "hero_position": "LJ",
        "players_at_table": 6,
        "effective_bb": 33,
        "preflop_actions": "R2-F-C-F-F-C",
        "player_stacks": [66.5, 31.0, 107.5, 48.0, 36.9, 10.8, 25.3],
    })
    # LJ open QJo at 33bb should be ~100% raise, NOT fold
    assert_in("RAISE", result["text"], "LJ open QJo should show RAISE in solver data")
    # The preflop_actions used should have F-F prefix (2 pads for 6→8)
    assert_true(
        result["preflop_actions"].startswith("F-F-R"),
        f"Should pad 2 folds, got: {result['preflop_actions']}"
    )


@test
def test_multiway_simplifies_after_flop_fold():
    """Multiway: 3-way pot where one folds on turn simplifies to HU."""
    from analyze_hand import _simplify_multiway, POSITION_ORDER
    from gto_api import nearest_depth
    hand = {
        "preflop_actions": "F-F-R2.2-F-C-F-F-C",
        "effective_bb": 33,
        "streets": [
            {"board": "6c2dTs", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
            {"card": "Ad", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R4", "size": 4.0},
                {"position": "CO", "action": "F"},
                {"position": "BB", "action": "C", "size": 4.0},
            ]},
        ],
    }
    depth = nearest_depth(33)
    simplified, adj_depth, note, positions = _simplify_multiway(
        hand, "LJ", "MTTGeneral", depth
    )
    # Should simplify to LJ vs BB (CO folds on turn)
    assert_true(note != "", "should produce a simplification note")
    assert_true(positions is not None, "should have active positions")
    assert_in("LJ", positions, "LJ should be in active positions")
    assert_in("BB", positions, "BB should be in active positions")


@test
def test_multiway_simplifies_when_hero_folds_same_street_as_hu():
    """H3506: 3-way pot, checked-down flop; on the turn BTN bets, BB folds,
    THEN hero folds — both folds in the same street.

    The HU node hero actually faced (HJ vs BTN) exists for the instant between
    BB's fold and hero's fold. The street walk must evaluate folds action-by-
    action: batching the whole turn's folds collapsed the pot straight to {BTN},
    dropped hero, and skipped simplification, leaving flop+turn with no solver
    data ("（無 solver 數據）"). Action-by-action catches HJ-vs-BTN at BB's fold.
    """
    from analyze_hand import _simplify_multiway
    from gto_api import nearest_depth
    hand = {
        "preflop_actions": "F-F-F-R2-F-C-F-C",  # HJ open, BTN call, BB call
        "effective_bb": 25,
        "players_at_table": 8,
        "streets": [
            {"board": "TdJhQc", "street": "flop", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "X"},
                {"position": "BTN", "action": "X"}]},
            {"card": "7c", "street": "turn", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "X"},
                {"position": "BTN", "action": "R", "size": 2.5},
                {"position": "BB", "action": "F"},
                {"position": "HJ", "action": "F"}]},
        ],
    }
    simplified, adj_depth, note, positions = _simplify_multiway(
        hand, "HJ", "MTTGeneral", nearest_depth(25)
    )
    assert_true(note != "", "should produce a simplification note (not skip)")
    assert_eq(positions, {"HJ", "BTN"},
              "HU villain must be BTN (the player still in when hero folded)")
    assert_eq(simplified, "F-F-F-R2-F-C-F-F",
              "BB cold-caller folded, hero open + BTN call kept")


@test
def test_simplify_multiway_spr_depth_floor():
    """Real-structure simplification compresses the effective stack to match the
    multiway SPR, but floors the compression so a shallow stack isn't pushed into
    preflop jam/fold (which would distort the range reaching the flop).

    LJ opens, HJ calls, CO cold-calls (3-way); CO folds the flop → HU LJ vs HJ.
    CO's dead call shrinks the solver pot, so a deep stack is compressed below its
    real depth; a shallow stack stays at its real depth (floored).
    """
    from analyze_hand import _simplify_multiway, MULTIWAY_SPR_DEPTH_FLOOR
    from gto_api import nearest_depth

    def hand(eff):
        return {
            "preflop_actions": "F-F-R2-C-C-F-F-F",  # LJ open, HJ call, CO cold-call
            "effective_bb": eff,
            "players_at_table": 8,
            "streets": [
                {"board": "Js7d2c", "actions": [
                    {"position": "LJ", "action": "R2", "size": 2.0},
                    {"position": "HJ", "action": "C"},
                    {"position": "CO", "action": "F"}]},
            ],
        }

    # Deep: CO's dead money drops the pot → SPR-compressed below the real depth.
    pf, d_deep, note, pos = _simplify_multiway(
        hand(40), "LJ", "MTTGeneral", nearest_depth(40))
    assert_eq(pos, {"LJ", "HJ"}, "HU = hero + villain")
    assert_eq(pf, "F-F-R2-C-F-F-F-F", "CO cold-caller folded; real structure kept")
    assert_true(d_deep < nearest_depth(40),
                f"deep stack must be SPR-compressed (got {d_deep})")
    assert_true(d_deep >= MULTIWAY_SPR_DEPTH_FLOOR, "but not below the floor")

    # Shallow: compression would breach the floor → keep the real depth.
    _, d_shallow, _, _ = _simplify_multiway(
        hand(12), "LJ", "MTTGeneral", nearest_depth(12))
    assert_eq(d_shallow, nearest_depth(12),
              "shallow stack keeps real depth (floored, no preflop-jam distortion)")


@test
def test_preflop_open_uses_hero_stack():
    """Preflop open: uses hero's own stack (not effective) when player_stacks available."""
    from analyze_hand import analyze_hand_full
    # Hero LJ has 21bb, BB has 18bb → effective_bb=18.
    # At effective 18bb (solver 17bb): A3s is limp/fold (no raise).
    # At hero's 21bb (solver 20bb): A3s is 100% raise.
    # Preflop open should use hero's stack since hero doesn't know who'll call.
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 18,
        "players_at_table": 7,
        "hero_position": "LJ",
        "hero_hand": "Ac3c",
        "player_stacks": [14, 21, 36, 20, 16, 16, 18],
        "preflop_actions": "F-R2-F-F-F-F-C",
        "streets": [],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # A3s should show RAISE in the preflop strategy, not just Call/Fold
    assert_in("RAISE", text, "A3s from LJ at hero's 21bb depth should show RAISE")
    assert_true("Call" not in text.split("【LJ A3s】")[1].split("==")[0],
                "A3s should NOT show Call (limp) when hero stack maps to raise depth")


@test
def test_preflop_facing_open_uses_effective_stack_not_hero_stack():
    """A call/3bet decision is already opponent-bound, unlike an unopened RFI."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral", "effective_bb": 30,
        "players_at_table": 8, "hero_position": "BTN", "hero_hand": "AsKs",
        "hero_starting_stack": 50,
        "player_stacks": [30, 40, 40, 40, 40, 50, 40, 40],
        "preflop_actions": "R2-F-F-F-F-R7-F-F-C", "streets": [],
    }
    result = analyze_hand_full(hand)
    assert_eq(result["hero_spots"][0]["params"]["depth"], 30.125,
              "facing an open uses the 30bb opponent-bound node, not hero's 50bb stack")


@test
def test_preflop_threebet_then_faces_shove_uses_shover_depth_for_both_decisions():
    """Hero's first action is not an RFI; a later covering shove binds the line."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral", "effective_bb": 59.357,
        "players_at_table": 8, "hero_position": "LJ", "hero_hand": "AsKd",
        "hero_starting_stack": 59.357,
        "player_stacks": [11.862, 8.805, 59.357, 72.083, 8.91, 50.824, 31.346, 66.13],
        "preflop_actions": "F-R2-R6-F-F-AI50.681-F-F-C-C", "streets": [],
    }
    result = analyze_hand_full(hand)
    depths = [s["params"]["depth"] for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_eq(depths, [50.125, 50.125])


@test
def test_preflop_open_depth_correction_no_stacks():
    """Preflop open: depth auto-corrects to next higher when hero raised but solver shows 0% raise."""
    from analyze_hand import analyze_hand_full
    # Same scenario as above but WITHOUT player_stacks — depth correction kicks in.
    # Hero raised A3s from LJ at effective 16bb (solver 17bb = 0% raise).
    # Phase 2.5 should detect this and try 20bb solver (100% raise).
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 16,
        "players_at_table": 6,
        "hero_position": "LJ",
        "hero_hand": "Ac3c",
        "preflop_actions": "R2-F-F-F-F-C",
        "streets": [],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    assert_in("RAISE", text, "A3s should show RAISE after depth correction (no player_stacks)")
    assert_true("Call" not in text.split("【LJ A3s】")[1].split("==")[0],
                "A3s should NOT show Call after depth auto-correction")


@test
def test_bb_check_option_normalized_to_x():
    """Preflop: BB check option after SB limp uses X not C, enabling postflop solver data."""
    from analyze_hand import analyze_hand_full
    # SB limps, BB checks → preflop "F-F-F-F-C-C" should normalize to "F-F-F-F-F-F-C-X"
    # Without this, postflop solver returns None (board query fails with C-C).
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 58,
        "players_at_table": 6,
        "hero_position": "SB",
        "hero_hand": "Kh2s",
        "preflop_actions": "F-F-F-F-C-C",
        "streets": [
            {"board": "4sTcJs", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "R2", "size": 2.0},
                {"position": "SB", "action": "C", "size": 2.0},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Flop must have solver data (not "無 solver 數據")
    assert_true("無 solver 數據" not in text.split("Flop")[1].split("==")[0],
                "Flop should have solver data after BB check option normalized to X")
    # Verify the preflop was normalized to include X
    assert_eq(result["preflop_actions"].split("-")[-1], "X",
              "BB check option should be X not C")


@test
def test_postflop_size_parsed_from_action_string():
    """Postflop: bet size parsed from action string when 'size' field missing."""
    from analyze_hand import analyze_hand_full
    # 3-way pot: UTG opens, SB+BB call. Flop actions have no "size" field.
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 14.9,
        "hero_position": "UTG",
        "hero_hand": "KQo",
        "preflop_actions": "R2-F-F-F-F-F-C-C",
        "streets": [
            {"board": "8s7dAh", "actions": [
                {"action": "X", "position": "SB"},
                {"action": "X", "position": "BB"},
                {"action": "R2.4", "position": "UTG"},  # no "size" field
                {"action": "C", "position": "SB"},
                {"action": "F", "position": "BB"},
            ]},
            {"card": "5h", "actions": [
                {"action": "X", "position": "SB"},
                {"action": "R10.5", "position": "UTG"},  # no "size" field
                {"action": "C", "position": "SB"},
            ]},
        ]
    })
    # Flop hero action should be a bet (R*), not check (X)
    flop_spot = [s for s in result["hero_spots"] if s["street"] == "flop"][0]
    assert_true(flop_spot["taken_code"].startswith("R"),
                f"Flop taken_code should be R* not {flop_spot['taken_code']}")
    # Turn should have solver data (not "無 solver 數據")
    turn_sols = [sol for spot, sol in zip(result["hero_spots"], result["solutions"])
                 if spot["street"] == "turn"]
    assert_true(turn_sols and turn_sols[0] is not None,
                "Turn should have solver data when flop bet size parsed from action string")


@test
def test_gto_line_fallback_when_sizing_off_tree():
    """GTO line fallback: turn gets solver data when flop bet was off-tree sizing."""
    from analyze_hand import analyze_hand_full
    # CO opens, BB calls — standard HU postflop
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 15,
        "hero_position": "CO",
        "hero_hand": "KQo",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "8s7dAh", "actions": [
                {"position": "BB", "action": "X"},
                # Hero bets 2.4bb (~37% pot), off-GTO sizing
                {"position": "CO", "action": "R2.4", "size": 2.4},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "5h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R10", "size": 10},
            ]},
        ]
    })
    # Turn should have solver data
    turn_has_data = False
    for spot, sol in zip(result["hero_spots"], result["solutions"]):
        if spot["street"] == "turn" and sol is not None:
            turn_has_data = True
    assert_true(turn_has_data, "Turn should have solver data")


@test
def test_raise_without_size_maps_to_raise_not_call():
    """Action matching: raise with no size maps to smallest raise, not call."""
    from analyze_hand import analyze_hand_full
    # H2506: BB check-raises HJ's cbet but parsed without a size ("R" not "R4.15")
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 20,
        "players_at_table": 6,
        "hero_position": "HJ",
        "hero_hand": "Th9h",
        "preflop_actions": "F-R2-F-F-F-C",
        "streets": [
            {"board": "Jc6d5d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R1.4", "size": 1.4},
                {"position": "BB", "action": "R"},  # check-raise, no size
                {"position": "HJ", "action": "F"},
            ]},
        ],
    })
    # Hero's second flop spot (facing check-raise) must have solver data
    flop_spots = [(spot, sol) for spot, sol in zip(result["hero_spots"], result["solutions"])
                  if spot["street"] == "flop"]
    assert_true(len(flop_spots) >= 2, f"Expected 2+ flop spots, got {len(flop_spots)}")
    facing_xr_sol = flop_spots[1][1]
    assert_true(facing_xr_sol is not None,
                "Facing check-raise spot must have solver data (raise without size should not match to Call)")


@test
def test_duplicate_opponent_check_skipped_in_multiway():
    """Multiway: duplicate opponent check (misparsed position) is skipped."""
    from analyze_hand import analyze_hand_full
    # H2508: 3-way pot, BB's flop check mislabeled as SB → two SB checks.
    # Without fix, flop_actions="X-X" (invalid), solver returns 204.
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 14.9,
        "players_at_table": 8,
        "hero_position": "UTG",
        "hero_hand": "KdQs",
        "preflop_actions": "R2-F-F-F-F-F-C-C",
        "streets": [
            {"board": "8s7dAd", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "SB", "action": "X"},  # misparsed BB check
                {"position": "UTG", "action": "R2.4", "size": 2.4},
                {"position": "SB", "action": "C", "size": 2.4},
                {"position": "BB", "action": "F"},
            ]},
            {"card": "5d", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "UTG", "action": "R10.5", "size": 10.5},
                {"position": "SB", "action": "C", "size": 10.5},
            ]},
        ],
    })
    flop_spots = [(spot, sol) for spot, sol in zip(result["hero_spots"], result["solutions"])
                  if spot["street"] == "flop"]
    assert_true(len(flop_spots) >= 1, f"Expected flop spot, got {len(flop_spots)}")
    assert_true(flop_spots[0][1] is not None,
                "Flop must have solver data (duplicate SB check should be skipped)")
    turn_spots = [(spot, sol) for spot, sol in zip(result["hero_spots"], result["solutions"])
                  if spot["street"] == "turn"]
    assert_true(len(turn_spots) >= 1, f"Expected turn spot, got {len(turn_spots)}")
    assert_true(turn_spots[0][1] is not None,
                "Turn must have solver data")


@test
def test_infer_missing_hero_call():
    """Multiway: missing hero call inferred when opponent bets and hand continues."""
    from analyze_hand import analyze_hand_full
    # H2517: SB bets on turn/river but hero (CO) call actions are missing.
    # Analysis should infer hero called and produce solver data for all streets.
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 116.1,
        "players_at_table": 6,
        "hero_position": "CO",
        "hero_hand": "Jd8d",
        "preflop_actions": "F-F-R2.2-F-C-C",
        "streets": [
            {"board": "9cAsJc", "actions": [
                {"position": "SB", "action": "R3.2", "size": 3.2},
                {"position": "BB", "action": "F"},
                {"position": "CO", "action": "C"},
            ]},
            {"card": "Ts", "actions": [
                {"position": "SB", "action": "R9.2", "size": 9.2},
                # hero call MISSING — should be inferred
            ]},
            {"card": "8c", "actions": [
                {"position": "SB", "action": "R40", "size": 40},
                # hero call MISSING — should be inferred (last street)
            ]},
        ],
    })
    turn_spots = [(s, sol) for s, sol in zip(result["hero_spots"], result["solutions"])
                  if s["street"] == "turn"]
    assert_true(len(turn_spots) >= 1, "Should have turn hero spot")
    assert_true(turn_spots[0][1] is not None, "Turn must have solver data (inferred hero call)")
    river_spots = [(s, sol) for s, sol in zip(result["hero_spots"], result["solutions"])
                   if s["street"] == "river"]
    assert_true(len(river_spots) >= 1, "Should have river hero spot")
    assert_true(river_spots[0][1] is not None, "River must have solver data (inferred hero call)")


@test
def test_compact_format_preflop():
    """Compact: preflop output has header, emoji markers, and hero result."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "66",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
    })
    compact = result["text_compact"]
    assert_in("♠ CO 66", compact, "compact should have header with position and hand")
    assert_in("30bb", compact, "compact should show effective bb")
    assert_in("─── Preflop ───", compact, "compact should have street separator")
    assert_in("GTO:", compact, "compact should have GTO action line")
    assert_true("combos" not in compact.lower(), "compact should not show combos")
    assert_true("底池" not in compact, "compact should not show pot size")


@test
def test_compact_format_multi_street():
    """Compact: multi-street output includes hand type labels and hero results."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "66",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "Js6h5s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
            {"card": "Kc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R6.6", "size": 6.6},
            ]},
        ]
    })
    compact = result["text_compact"]
    assert_in("─── Flop:", compact, "compact should have flop section")
    assert_in("─── Turn:", compact, "compact should have turn section")
    assert_in("🎯", compact, "compact should have hand type emoji on postflop")
    # Also verify detailed text still exists for coaching
    assert_in("Preflop", result["text"])
    assert_in("Flop", result["text"])


@test
def test_compact_format_shows_gto_for_later_decision_points():
    """Compact: later same-street decision points still show a GTO line.

    Regression for H3416: after hero check-raised flop, the exact JTo combo
    had a very small but non-zero range at the turn call and river fold nodes.
    The compact formatter treated that as off-range and printed only
    "→ Hero call/fold", hiding the solver frequencies for those later
    decisions.
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 99.9,
        "hero_position": "BB",
        "hero_hand": "JsTc",
        "preflop_actions": "F-F-F-R2-F-F-C",
        "players_at_table": 7,
        "hero_starting_stack": 99.9,
        "streets": [
            {"board": "6d8hJd", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R1.7", "size": 1.7},
                {"position": "BB", "action": "R5.2", "size": 5.2},
                {"position": "CO", "action": "C", "size": 3.5},
            ]},
            {"card": "8d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R7.8", "size": 7.8},
                {"position": "BB", "action": "C", "size": 7.8},
            ]},
            {"card": "Qc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R23.4", "size": 23.4},
                {"position": "BB", "action": "F"},
            ]},
        ],
    })

    compact = result["text_compact"]
    turn_section = compact.split("─── Turn: 8♦️ ───", 1)[1].split("─── River:", 1)[0]
    river_section = compact.split("─── River: Q♣️ ───", 1)[1]

    assert_in("→ Hero check", turn_section)
    assert_in("GTO: Call", turn_section,
              "turn facing-bet decision should show solver frequencies")
    assert_in("→ Hero call", turn_section)
    assert_true(
        turn_section.index("GTO: Call") < turn_section.index("→ Hero call"),
        "turn call should be immediately explained by a preceding GTO line",
    )

    assert_in("→ Hero check", river_section)
    assert_in("GTO: Fold", river_section,
              "river facing-bet decision should show solver frequencies")
    assert_in("→ Hero fold", river_section)
    assert_true(
        river_section.index("GTO: Fold") < river_section.index("→ Hero fold"),
        "river fold should be immediately explained by a preceding GTO line",
    )


@test
def test_compact_format_spot_compact():
    """Compact: format_spot_compact produces emoji-marked action lines."""
    from gto_formatter import format_spot_compact
    from gto_api import get_spot_solution
    sol = get_spot_solution(gametype="MTTGeneral", depth="30.125",
                            preflop_actions="F-F-F-F-R2-F-F-C")
    if sol is None:
        return  # API unavailable, skip
    compact = format_spot_compact(sol, "66", "CO")
    assert_in("GTO:", compact, "should start with GTO: prefix")
    assert_in("%", compact, "should show frequency percentage")
    assert_true("combos" not in compact.lower(), "should not show combos count")


@test
def test_compact_offrange_exact_combo_returns_no_data():
    """Compact formatter: if the exact combo has zero range at a later
    node, do not use either its solver-default row or same-hand aggregate
    counters.

    Regression for H2902 river: Qh9d bet an off-grid/off-mix river size.
    The facing-raise node was unreachable for that exact combo. GTO Wizard
    still returned a misleading raw row ("Call 100%") and aggregate Q9o
    counters ("Fold 99%"), but the user-facing result should be no solver
    data for hero's actual combo/line.
    """
    from gto_formatter import combo_index_for_hand, format_ev_comparison, format_spot_compact

    off_idx = combo_index_for_hand("Qh9d")
    in_idx = combo_index_for_hand("Qs9d")
    assert_true(off_idx is not None and in_idx is not None, "fixture combos must index")

    n = 1326
    range_arr = [0.0] * n
    range_arr[in_idx] = 1.0

    fold_strategy = [0.0] * n
    call_strategy = [0.0] * n
    fold_strategy[in_idx] = 0.991
    call_strategy[in_idx] = 0.004
    # Off-range exact combo row is misleading solver noise and must be ignored.
    call_strategy[off_idx] = 1.0

    fold_evs = [0.0] * n
    call_evs = [0.0] * n
    call_evs[in_idx] = -3.5
    call_evs[off_idx] = 9.9  # would hide the mistake if exact off-range row is used

    sol = {
        "game": {
            "board": "Jd7d4dTd4c",
            "current_street": {"type": "river"},
        },
        "players_info": [{
            "player": {"position": "BB"},
            "range": range_arr,
            "simple_hand_counters": {
                "Q9o": {
                    "actions_total_frequencies": {
                        "F": 0.991,
                        "C": 0.004,
                    }
                }
            },
        }],
        "action_solutions": [
            {
                "action": {"code": "F"},
                "strategy": fold_strategy,
                "evs": fold_evs,
                "total_frequency": 0.5,
            },
            {
                "action": {"code": "C"},
                "strategy": call_strategy,
                "evs": call_evs,
                "total_frequency": 0.5,
            },
        ],
    }

    compact = format_spot_compact(sol, "Q9o", "BB", combo_idx=off_idx)
    assert_eq(compact, "",
              "off-range exact combo should format as no solver data")

    ev = format_ev_comparison(
        sol, "C", "Q9o", "BB", is_preflop=False, combo_idx=off_idx)
    assert_true(ev is None,
                f"off-range exact combo should not produce EV advice, got {ev!r}")


@test
def test_h2902_river_offrange_shows_no_solver_and_actual_bet_pct():
    """H2902: river facing-raise node is off-range after hero's 1.8bb
    lead, so compact output should show no solver data for the call. The
    hero lead label must use the actual pot percentage (~33%), not the
    nearest solver bucket (~45%).
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 19.8,
        "hero_position": "BB",
        "hero_hand": "Qh9d",
        "preflop_actions": "R2-F-F-F-F-C",
        "players_at_table": 6,
        "hero_starting_stack": 19.8,
        "streets": [
            {"board": "4dJd7d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "X"},
            ]},
            {"card": "Td", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "X"},
            ]},
            {"card": "4c", "actions": [
                {"position": "BB", "action": "R1.8", "size": 1.8},
                {"position": "LJ", "action": "R"},
                {"position": "BB", "action": "C", "size": 2.2},
            ]},
        ],
    })

    compact = result["text_compact"]
    assert_in("→ Hero bet 33% pot", compact,
              "river lead should display actual 1/3-pot size")
    assert_not_in("→ Hero bet 45% pot", compact,
                  "compact label must not display the nearest solver bucket")
    assert_in("（無 solver 數據）", compact,
              "off-range facing-raise node should show no solver data")
    river_section = compact.split("─── River: 4♣️ ───", 1)[1]
    assert_not_in("GTO: Call 100%", river_section,
                  "must not use zero-range exact combo strategy row")
    assert_not_in("GTO: Fold 99%", river_section,
                  "must not borrow same-hand aggregate data for off-range node")


@test
def test_h2905_threeway_overcall_gets_preflop_and_hu_postflop_data():
    """H2905: HJ open, CO overcall, BB call is a 3-way pot, not 4-way.
    Reduce to HJ-vs-CO heads-up. With real-structure simplification the BB
    cold-caller (folds the flop) collapses to a single pre-flop fold and hero
    CO keeps his TRUE role — a flat-caller facing HJ's open — so the preflop
    node is CO's call/jam range, not a recast opener range. Every street must
    still have solver data.
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 8.6,
        "hero_position": "CO",
        "hero_hand": "As8s",
        "preflop_actions": "F-F-R2-C-F-F-C",
        "players_at_table": 7,
        "hero_starting_stack": 18.6,
        "streets": [
            {"board": "JhKs4h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "X"},
                {"position": "CO", "action": "R2.4", "size": 2.4},
                {"position": "BB", "action": "F"},
                {"position": "HJ", "action": "C", "size": 2.4},
            ]},
            {"card": "5h", "actions": [
                {"position": "HJ", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
            {"card": "6d", "actions": [
                {"position": "HJ", "action": "R15", "size": 15.0},
                {"position": "CO", "action": "F"},
            ]},
        ],
    })

    compact = result["text_compact"]
    assert_in("多人底池", compact, "must note the multiway simplification")
    assert_in("CO vs HJ", compact,
              "must simplify the 3-way HJ+CO+BB pot to the real CO-vs-HJ HU")
    assert_not_in("4-way", compact, "must not describe this hand as 4-way")
    assert_in("─── Preflop ───\nGTO:", compact,
              "CO preflop facing HJ open must have solver data")
    flop_section = compact.split("─── Flop: J♥️K♠️4♥️ ───", 1)[1].split("─── Turn:", 1)[0]
    turn_section = compact.split("─── Turn: 5♥️ ───", 1)[1].split("─── River:", 1)[0]
    assert_in("GTO:", flop_section, "flop should use HU approximation data")
    assert_in("GTO:", turn_section, "turn should use HU approximation data")


@test
def test_h2915_turn_ends_after_hero_call_without_extra_no_solver_node():
    """H2915: OCR split a terminal turn call into BB call + duplicate BB bet.

    After hero calls CO's turn shove-sized bet, there is no further decision
    point.  The compact output should stop after "Hero call" and must not add
    a trailing same-street "（無 solver 數據）" line.
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 12.4,
        "hero_position": "BB",
        "hero_hand": "QcJc",
        "preflop_actions": "F-F-F-F-R2-C-F-C",
        "players_at_table": 8,
        "hero_starting_stack": 12.4,
        "streets": [
            {"board": "6d4s3c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2.6", "size": 2.6},
                {"position": "BB", "action": "C", "size": 2.6},
            ]},
            {"card": "Kc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R7.8", "size": 7.8},
                {"position": "BB", "action": "C", "size": 7.8},
                # Phantom duplicate: same BB cannot call then immediately
                # raise the same amount without CO acting again.
                {"position": "BB", "action": "R7.8", "size": 7.8},
            ]},
        ],
    })

    turn_section = result["text_compact"].split("─── Turn: K♣️ ───", 1)[1]
    assert_in("→ Hero call", turn_section, "turn call should still be shown")
    assert_not_in("（無 solver 數據）", turn_section,
                  "terminal turn call must not be followed by extra no-data node")


@test
def test_no_hero_hand_flag():
    """No hero hand: output omits hero-specific sections when no_hero_hand=True."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "LJ",
        "hero_hand": "AA",
        "no_hero_hand": True,
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "streets": [
            {"board": "Th6c2d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R2", "size": 2.0},
            ]},
        ]
    })
    text = result["text"]
    compact = result["text_compact"]
    # Header should show position without hand
    assert_in("Hero: LJ", text, "detailed text should show hero position")
    assert_true("Hero: LJ AA" not in text, "detailed text should NOT show AA as hero hand")
    # Compact header should not show AA
    assert_in("♠ LJ |", compact, "compact should show position without hand")
    assert_true("♠ LJ AA" not in compact, "compact should NOT show AA")
    # Should not show hand type eval for AA (no 🎯 overpair)
    assert_true("牌型" not in text, "should not show hand type when no hero hand")
    assert_true("🎯" not in compact, "compact should not show hand type emoji")
    # Return dict should carry the flag
    assert_true(result["no_hero_hand"], "result should carry no_hero_hand flag")


# ── Snapshot E2E tests (image → OCR parse → GTO analysis) ──

def _load_snapshots():
    """Load regression snapshots from tests/snapshots/ directory."""
    snapshots_dir = REPO_ROOT / "tests" / "snapshots"
    manifest_path = snapshots_dir / "manifest.json"
    if not manifest_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text())
    snapshots = []
    for entry in manifest:
        hid = entry["hand_id"]
        hand_dir = snapshots_dir / hid
        if not hand_dir.exists():
            continue
        snap = {"hand_id": hid, "source_type": entry["source_type"]}

        img_path = hand_dir / "input.jpeg"
        if img_path.exists():
            snap["image_data"] = img_path.read_bytes()

        expected_path = hand_dir / "expected.json"
        if expected_path.exists():
            snap["expected_json"] = expected_path.read_text()

        gto_path = hand_dir / "gto_text.txt"
        if gto_path.exists():
            snap["gto_text"] = gto_path.read_text()

        gto_compact_path = hand_dir / "gto_compact.txt"
        if gto_compact_path.exists():
            snap["gto_compact"] = gto_compact_path.read_text()

        snapshots.append(snap)
    return snapshots


_SNAPSHOTS_DIR = REPO_ROOT / "tests" / "snapshots"


def _register_snapshot_tests():
    """Dynamically register snapshot E2E tests from files."""
    import re as _re

    snapshots = _load_snapshots()
    if not snapshots:
        return

    strip_timing = lambda s: _re.sub(r"⏱ Discovery:.*$", "", s, flags=_re.MULTILINE).rstrip()

    for snap in snapshots:
        hid = snap["hand_id"]
        source = snap["source_type"]

        # Layer 1: OCR parse test (image snapshots only)
        if source == "image" and snap.get("image_data"):
            def make_l1(s=snap, h=hid):
                def _test():
                    expected = json.loads(s["expected_json"]) if s.get("expected_json") else json.loads(s["parsed_json"])
                    from ocr.n8_parser import parse_n8_screenshot
                    result = parse_n8_screenshot(bytes(s["image_data"]))
                    conf = float(result.get("confidence", 0.0))
                    # Mirror production's tiered gate: anything under the
                    # medium-tier floor (default 0.80) would fall back to
                    # Gemini in the real bot. A low-conf wrong parse is not
                    # a regression — it's the system correctly signalling
                    # uncertainty. The medium-tier band (0.80..0.95) still
                    # surfaces OCR to the user so mismatches there are real.
                    MEDIUM_TIER_MIN = float(os.getenv("OCR_MEDIUM_TIER_MIN", "0.80"))
                    if not result.get("hand"):
                        if conf < MEDIUM_TIER_MIN:
                            return  # low-conf no-hand → fallback territory, OK
                        assert_true(False,
                                    f"OCR returned no hand (confidence={conf:.2f})")
                    parsed = result["hand"]
                    try:
                        for key in ["hero_hand", "hero_position", "preflop_actions",
                                    "players_at_table", "tournament_type"]:
                            p_val = parsed.get(key)
                            e_val = expected.get(key)
                            if e_val is not None:
                                assert_eq(p_val, e_val, f"{key} mismatch")
                        p_streets = parsed.get("streets") or []
                        e_streets = expected.get("streets") or []
                        assert_eq(len(p_streets), len(e_streets), "streets count mismatch")
                        for i, (ps, es) in enumerate(zip(p_streets, e_streets)):
                            p_board = ps.get("board", ps.get("card", ""))
                            e_board = es.get("board", es.get("card", ""))
                            assert_eq(p_board, e_board, f"street[{i}] board mismatch")
                    except AssertionError:
                        if conf < MEDIUM_TIER_MIN:
                            return  # low-conf mismatch → fallback territory, OK
                        raise
                _test.__name__ = f"test_snapshot_l1_ocr_{h}"
                _test.__doc__ = f"Snapshot L1-OCR: {h} image → OCR parse matches expected."
                return _test
            _tests.append(make_l1())

        # Layer 2: GTO output test (all snapshots)
        # Deterministic on same machine — uses local .gto_cache.
        # On first run (no gto_text.txt), generates the golden file.
        # Subsequent runs compare against it to catch formatting regressions.
        def make_l2(s=snap, h=hid):
            def _test():
                expected_json_str = s.get("expected_json")
                hand_json = json.loads(expected_json_str) if isinstance(expected_json_str, str) else expected_json_str
                # Use an isolated cache dir for snapshot tests to avoid
                # cross-contamination with non-snapshot regression tests.
                # Golden files are generated on first run using this isolated
                # cache; subsequent runs read from the same cache → deterministic.
                import gto_cache
                snapshot_cache = _SNAPSHOTS_DIR / ".gto_cache"
                snapshot_cache.mkdir(exist_ok=True)
                orig_cache_dir = gto_cache._CACHE_DIR
                gto_cache._CACHE_DIR = snapshot_cache
                gto_cache._mem.clear()
                try:
                    from analyze_hand import analyze_hand_full
                    result = analyze_hand_full(hand_json)
                finally:
                    gto_cache._CACHE_DIR = orig_cache_dir
                    gto_cache._mem.clear()
                actual = strip_timing(result["text"])

                gto_path = _SNAPSHOTS_DIR / h / "gto_text.txt"
                if not gto_path.exists():
                    # First run: generate golden file
                    gto_path.write_text(result["text"])
                    compact_path = _SNAPSHOTS_DIR / h / "gto_compact.txt"
                    if result.get("text_compact"):
                        compact_path.write_text(result["text_compact"])
                    return  # pass on first run (nothing to compare yet)

                expected = strip_timing(gto_path.read_text())
                if actual != expected:
                    # Tolerate tiny solver drift in EV (bb) / frequency (%);
                    # combos counts, action sequences, ranges and line count
                    # are still compared exactly. A fresh worktree that misses
                    # the snapshot .gto_cache re-fetches live and wobbles the
                    # last digit (±0.01bb / ±0.2pp); that is not a regression.
                    from gto_text_compare import gto_text_matches
                    ok, msg = gto_text_matches(expected, actual)
                    if not ok:
                        raise AssertionError(f"GTO text mismatch: {msg}")
            _test.__name__ = f"test_snapshot_l2_gto_{h}"
            _test.__doc__ = f"Snapshot L2-GTO: {h} analyze_hand_full() matches stored output."
            return _test
        _tests.append(make_l2())


_register_snapshot_tests()
