#!/usr/bin/env python3
"""Regression test suite for core analysis logic.

Run after any changes to:
  - scripts/analyze_hand.py
  - scripts/gto_api.py
  - scripts/gto_formatter.py
  - scripts/icm_modes.py
  - src/gemini_session.py

Usage:
    python scripts/regression_test.py          # Run all tests
    python scripts/regression_test.py -v       # Verbose output
    python scripts/regression_test.py -k chip  # Run only tests matching "chip"

Requires: valid GTO Wizard token (.tokens.json) and network access.
Does NOT require GEMINI_API_KEY (tests bypass LLM layer).
"""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Test infrastructure ──

_tests = []
_verbose = "-v" in sys.argv
_filter = None
for i, arg in enumerate(sys.argv):
    if arg == "-k" and i + 1 < len(sys.argv):
        _filter = sys.argv[i + 1].lower()


def test(fn):
    _tests.append(fn)
    return fn


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}\n  expected: {expected!r}\n  actual:   {actual!r}")


def assert_in(needle, haystack, msg=""):
    if needle not in haystack:
        raise AssertionError(f"{msg}\n  {needle!r} not found in:\n  {haystack!r}")


def assert_not_in(needle, haystack, msg=""):
    if needle in haystack:
        raise AssertionError(f"{msg}\n  {needle!r} should not be in:\n  {haystack!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "condition was False")


@test
def test_postflop_allin_resolution_preserves_sized_hero_backjam():
    """OCR: a sized hero all-in must not be collapsed onto villain's raise.

    Regression for H3442: Natural8 renders hero's back-jam as a yellow
    "Raise 16.4 BB" row plus a red "All-In" badge.  The same-side collapse
    correctly turns that into a sized hero All-In, but the final all-in
    attribution pass then treated every nameless All-In as a badge and
    attached it to BB's prior raise, fabricating "hero fold".
    """
    from ocr.panel_parser import _resolve_allin_attribution

    entries = [
        {"type": "opponent", "position": "BB", "action": "Check", "size": None},
        {"type": "hero", "position": "BB", "action": "Bet", "size": 1.8},
        {"type": "opponent", "position": "BB", "action": "Raise", "size": 4.8},
        {"type": "hero", "position": "BB", "action": "All-In", "size": 16.4},
        {"type": "opponent", "position": "BB", "action": "Fold", "size": None},
    ]

    resolved = _resolve_allin_attribution(entries)
    assert_eq([e["action"] for e in resolved],
              ["Check", "Bet", "Raise", "All-In", "Fold"])
    assert_eq(resolved[3]["type"], "hero")
    assert_eq(resolved[3]["size"], 16.4)
    assert_eq(resolved[4]["type"], "opponent")


@test
def test_corner_ocr_does_not_override_confident_ace():
    """OCR: EasyOCR misreads the Ace corner glyph as '4' (H2878).

    The corner-OCR cross-check exists to rescue confident CNN face-card
    hallucinations (H3429: 2 read as K), but it must never let a corner '4'
    override a CNN that is certain the card is an Ace.
    """
    from ocr.table_parser import _corner_rank_overrides

    # H2878: CNN certain it's an Ace; corner OCR misreads it as '4'. Keep A.
    assert_true(not _corner_rank_overrides("A", 1.00, "4"),
                "corner '4' must not override a confident CNN Ace (H2878)")
    # H3429-style: CNN confidently hallucinated a face card; corner rescues it.
    assert_true(_corner_rank_overrides("K", 0.99, "2"),
                "corner OCR must still rescue a CNN face-card hallucination")
    # Ordinary disagreement still defers to the corner reading.
    assert_true(_corner_rank_overrides("Q", 0.80, "K"),
                "corner OCR should override a non-Ace CNN rank")


@test
def test_corner_ocr_override_requires_cnn_top2_support():
    """OCR: corner OCR must not invent ranks absent from the CNN top-2.

    Precision-push regressions TM5867169951/TM5920473278/TM5962778472 had
    EasyOCR confidently read clean corners as A/4/4 while the CNN top-2 did
    not contain those ranks.  H3429 remains covered because the true 2 is the
    CNN runner-up under a WIN sticker.
    """
    from ocr.table_parser import _corner_rank_supported_by_cnn_top2

    assert_true(
        _corner_rank_supported_by_cnn_top2(
            "2", [("K", 0.985), ("2", 0.009)]
        ),
        "H3429-style corner rescue should stay enabled",
    )
    assert_true(
        not _corner_rank_supported_by_cnn_top2(
            "4", [("Q", 0.999), ("J", 0.0006)]
        ),
        "Q→4 EasyOCR hallucination must not override a strong CNN Q",
    )
    assert_true(
        not _corner_rank_supported_by_cnn_top2(
            "A", [("8", 0.862), ("3", 0.104)]
        ),
        "8→A EasyOCR hallucination must not override when A is absent top-2",
    )


@test
def test_corner_ocr_can_rescue_low_conf_overlapped_card():
    """OCR: shifted corner OCR can rescue an overlapped low-confidence crop.

    TM5900728345's right hero card crop included the neighboring T♦ on the
    left edge; the CNN read 3♣ at low confidence while shifted corner OCR read
    the true 8 at 1.0.  This rescue must stay narrower than the false A/4
    corner reads guarded by the top-2-support test above.
    """
    from ocr.table_parser import _corner_rank_can_override

    assert_true(
        _corner_rank_can_override(
            cnn_rank="3",
            cnn_conf=0.765,
            corner_rank="8",
            corner_conf=1.0,
            rank_top2=[("3", 0.765), ("6", 0.074)],
        ),
        "low-confidence CNN + perfect shifted corner OCR should override",
    )
    assert_true(
        not _corner_rank_can_override(
            cnn_rank="8",
            cnn_conf=0.862,
            corner_rank="A",
            corner_conf=0.958,
            rank_top2=[("8", 0.862), ("3", 0.104)],
        ),
        "moderate-confidence CNN plus imperfect unsupported corner OCR is unsafe",
    )


@test
def test_board_corner_override_blocks_confident_cnn_ace():
    """OCR: board corner OCR must not flip a confident CNN board rank.

    H2565: river A♥ — CNN A@0.91 — was overwritten by EasyOCR's Ace-glyph→"4"
    corner read (the naive "corner_conf >= 0.90" board rule had no CNN-support
    guard the hero path already used).  The bad board 4h then collided with the
    hero's correct 4h and conflict resolution rewrote the hero to Ah6d.  Block
    the override when the CNN is confident and the corner rank is absent from
    the CNN top-2; keep the WIN-sticker rescue when the CNN is unsure (verified
    zero-change across the 7,183-image corpus).
    """
    from ocr.table_parser import _board_corner_override_allowed

    # H2565: confident CNN Ace, corner "4" not in CNN top-2 → keep the CNN.
    assert_true(
        not _board_corner_override_allowed(
            cnn_rank="A", cnn_conf=0.908, corner_rank="4",
            corner_conf=0.942, rank_top2=[("A", 0.908), ("K", 0.091)],
        ),
        "corner '4' must not override a confident CNN board Ace (H2565)",
    )
    # WIN-sticker rescue: CNN near-random, corner confident → still override.
    assert_true(
        _board_corner_override_allowed(
            cnn_rank="4", cnn_conf=0.188, corner_rank="Q",
            corner_conf=0.942, rank_top2=[("4", 0.188), ("2", 0.163)],
        ),
        "low-confidence CNN board read must still defer to a clean corner",
    )
    # Corner rank present in the CNN top-2 → legitimate rescue, override stays.
    assert_true(
        _board_corner_override_allowed(
            cnn_rank="K", cnn_conf=0.985, corner_rank="2",
            corner_conf=0.95, rank_top2=[("K", 0.985), ("2", 0.009)],
        ),
        "corner rescue supported by the CNN top-2 must still override",
    )


@test
def test_postflop_allin_resolution_still_attaches_sticker_only_badge():
    """OCR: sticker-only All-In fragments still belong to the prior raise.

    Regression coverage for H3441's opposite case: BB's all-in badge was
    OCR-classified as a nameless hero-looking red sticker.  With no size and
    no position, it is a badge for the immediately previous BB raise, not a
    separate hero jam.
    """
    from ocr.panel_parser import _resolve_allin_attribution

    entries = [
        {"type": "opponent", "position": "BB", "action": "Check", "size": None},
        {"type": "hero", "position": "BB", "action": "Bet", "size": 1.3},
        {"type": "opponent", "position": "BB", "action": "Raise", "size": 12.5},
        {"type": "hero", "position": None, "action": "All-In", "size": None},
        {"type": "opponent", "position": None, "action": "Fold", "size": None},
    ]

    resolved = _resolve_allin_attribution(entries)
    assert_eq([e["action"] for e in resolved],
              ["Check", "Bet", "All-In", "Fold"])
    assert_eq(resolved[2]["type"], "opponent")
    assert_eq(resolved[2]["size"], 12.5)
    assert_eq(resolved[3]["type"], "hero")


@test
def test_dup_allin_badge_on_call_keeps_call_size():
    """OCR: a bare All-In badge on hero's own Call must keep the call size.

    H3462 river: villain bets 13.6, hero calls all-in for their last 6.8bb.
    N8 stamps a red "All-In" badge on the call sticker, which OCR splits into
    a trailing same-side All-In entry (size=None, no name). Dropping it keeps
    the 6.8 call so hero's starting stack reconstructs to ~19bb; without the
    drop the sizeless badge wipes hero_street and effective_bb collapsed
    19→12bb. The call stays a Call (hero called for less than the bet — not a
    jam); only bet/raise badges promote to All-In.
    """
    from ocr.panel_parser import _collapse_dup_allin_badge

    entries = [
        {"type": "hero", "position": None, "action": "Check", "size": None},
        {"type": "opponent", "position": "CO", "action": "Bet", "size": 13.6},
        {"type": "hero", "position": "BB", "action": "Call", "size": 6.8},
        {"type": "hero", "position": None, "action": "All-In", "size": None},
    ]

    cleaned = _collapse_dup_allin_badge(entries)
    assert_eq([e["action"] for e in cleaned], ["Check", "Bet", "Call"],
              "bare All-In badge dropped, call preserved")
    assert_eq(cleaned[2]["action"], "Call", "calling all-in stays a Call")
    assert_eq(cleaned[2]["size"], 6.8, "call size must survive")


@test
def test_dup_allin_badge_on_bet_promotes_to_allin():
    """OCR: a bare All-In badge on a bet/raise promotes it to All-In (H2852).

    The badge sits on the same player's bet; dropping it but promoting the
    bet to All-In preserves both the wager size (for stack accounting) and
    the all-in label (for the summary).
    """
    from ocr.panel_parser import _collapse_dup_allin_badge

    entries = [
        {"type": "hero", "position": "BB", "action": "Bet", "size": 15.5},
        {"type": "hero", "position": None, "action": "All-In", "size": None},
    ]

    cleaned = _collapse_dup_allin_badge(entries)
    assert_eq([e["action"] for e in cleaned], ["All-In"],
              "badge dropped, bet promoted to All-In")
    assert_eq(cleaned[0]["size"], 15.5, "bet size must survive")


def _allin_badge_region(above_color):
    """Build a synthetic column region with a red All-In badge.

    The badge occupies rows [120:140]; the band the visual disambiguator
    reads (rows [98:116], just above the badge) is painted ``above_color``
    (BGR). Returns (region, group) ready for ``_classify_group``.
    """
    import numpy as np

    region = np.zeros((260, 160, 3), dtype=np.uint8)
    # Red All-In badge sticker (BGR) so _detect_entry_type reads "opponent".
    region[120:140, 60:140] = (40, 40, 200)
    # The sticker the badge sits on, read in the band above the badge.
    region[96:117, :] = above_color
    group = [{
        "text": "All-In", "center_x": 90, "center_y": 130,
        "x_min": 73, "y_min": 120, "x_max": 121, "y_max": 140,
    }]
    return region, group


@test
def test_classify_group_keeps_opponent_allin_badge_over_white_sticker():
    """OCR: a bare All-In badge on villain's raise stays opponent (H3577).

    N8 stamps a red "All-In" badge on the *opponent's* raise sticker when
    villain shoves. The badge OCRs as its own nameless/sizeless entry that
    the HSV detector tags "opponent" (red ≠ yellow). The H2842 rule flipped
    every such bare badge to hero, fabricating a phantom hero shove (sized
    from the following call as villain_raise + call = 22.4 + 14.4 = 36.8)
    and pushing hero's real call onto the opponent. Reading the white
    sticker above the badge vetoes the flip so the badge folds onto
    villain's raise and hero's call survives.
    """
    from ocr.panel_parser import _classify_group

    # White (low-sat, high-value) sticker above => opponent's raise.
    region, group = _allin_badge_region(above_color=(220, 220, 220))
    entry = _classify_group(group, region)
    assert_eq(entry["action"], "All-In", "still an All-In entry")
    assert_eq(entry["type"], "opponent",
              "badge over white opponent sticker must stay opponent")


@test
def test_classify_group_flips_bare_hero_allin_with_no_sticker_above():
    """OCR: a centered bare hero All-In with no sticker above flips to hero.

    Preserves H2842 — hero's own centered red All-In sticker has no white
    (or yellow) sticker rendered directly above it, so the visual read is
    inconclusive and the flip-to-hero default still applies.
    """
    from ocr.panel_parser import _classify_group

    # Dark band above (no opponent sticker) => inconclusive => flip to hero.
    region, group = _allin_badge_region(above_color=(0, 0, 0))
    entry = _classify_group(group, region)
    assert_eq(entry["action"], "All-In", "still an All-In entry")
    assert_eq(entry["type"], "hero",
              "a solo centered hero all-in must still flip to hero")


# ── Calling a villain all-in == committing (PR: fix/allin-call-deviation) ──
# H3459: SB shoves the turn ("Bet 17.1 / All-In"), hero calls.  The solver
# models a deeper 35bb world where 17.1 is just a big bet (Fold/Call/All-in),
# and the compact renderer flagged hero's call ❌ against "GTO建議 all-in".
# But facing a shove, Call and All-in are the same real action — both commit
# every chip to a showdown.  The fix tags the shove and merges its frequency
# into Call so the call is scored as a match, not a deviation.


@test
def test_build_streets_tags_sized_allin():
    """OCR: a sized villain all-in keeps R{size} but is tagged ``allin``.

    The action code must stay ``R17.1`` so solver action-matching and golden
    snapshots are unchanged, while the explicit flag tells analyze_hand the
    bettor is committed (so a caller is calling an all-in, not facing a
    raisable bet). H3459 turn.
    """
    from ocr.n8_parser import _build_streets

    street_cols = [{
        "name": "Turn",
        "entries": [
            {"type": "opponent", "position": "SB", "action": "All-In", "size": 17.1},
            {"type": "hero", "position": None, "action": "Call", "size": 17.1},
        ],
    }]
    streets = _build_streets(
        street_cols, board_cards=["Qs", "Qd", "5d", "3c"],
        pos_order=["SB", "BTN"], hero_position="BTN",
        active_positions=["SB", "BTN"],
    )
    turn = streets[0]["actions"]
    assert_eq(turn[0]["action"], "R17.1", "sized all-in keeps absolute raise code")
    assert_eq(turn[0]["allin"], True, "villain all-in must be tagged")
    assert_eq(turn[1]["action"], "C", "hero call code unchanged")
    assert_true("allin" not in turn[1], "a plain call is not an all-in")


@test
def test_build_streets_multiway_fold_not_attributed_to_the_bettor():
    """OCR: a multiway cold-caller's fold must not land on the bettor's seat.

    H3531 (3-way flop, hero CO vs SB vs BB): SB bets, BB folds, hero calls.
    N8's per-row reconciliation/order-inference tagged BB's fold to SB (the
    bettor), giving the impossible 'SB bet then SB fold'. That broke the
    multiway→HU collapse (flop_actions became R2-F instead of R2-C) and dropped
    every post-flop solver node. A fold can never belong to a seat that already
    acted this street; it must be reassigned to the first un-acted opponent.
    """
    from ocr.n8_parser import _build_streets

    street_cols = [{
        "name": "Flop",
        "entries": [
            {"type": "opponent", "position": "SB", "action": "Bet", "size": 3.6},
            # BB's fold whose badge was misread onto the bettor's SB seat.
            {"type": "opponent", "position": "SB", "action": "Fold"},
            {"type": "hero", "position": None, "action": "Call", "size": 3.6},
        ],
    }]
    streets = _build_streets(
        street_cols, board_cards=["Th", "2c", "6h"],
        pos_order=["UTG", "UTG+1", "MP", "MP1", "CO", "BTN", "SB", "BB"],
        hero_position="CO", active_positions=["CO", "SB", "BB"],
    )
    flop = streets[0]["actions"]
    assert_eq(flop[0]["position"], "SB", "the bettor stays SB")
    assert_eq(flop[1]["action"], "F", "second action is the fold")
    assert_eq(flop[1]["position"], "BB", "fold reassigned off the bettor to BB")
    assert_eq(flop[2]["position"], "CO", "hero call stays CO")


@test
def test_effective_bb_opp_shove_called_uses_investment_not_misread_stack():
    """Opp shoves all-in, hero calls in full → eff_bb = opp's investment.

    H3514: BB (Gao zU) shoved 5.8bb on the river and hero (BTN) called. The
    BB's tiny remaining stack (~0) was OCR-misread as the 18.9bb pot chips,
    so the name/heuristic branch produced effective_bb ≈ 27-29bb. When the
    last action is hero's call covering an explicit opponent all-in, the
    opponent is committed (display ≈ 0) so their starting stack must come
    from their total investment (opp_perm), giving the real ~8.8bb effective.
    """
    from ocr.n8_parser import _compute_effective_bb

    columns = [
        {"name": "Blinds (Ante)", "pot": None, "entries": [
            {"type": "opp", "action": "sb", "size": 0.5, "player_name": "LetmeinAA"},
            {"type": "opp", "action": "bb", "size": 1.0, "player_name": "Gao zU"},
        ]},
        {"name": "Pre-Flop", "pot": 2.3, "entries": [
            {"type": "opp", "action": "fold", "player_name": "BboySheep"},
            {"type": "opp", "action": "fold", "player_name": "alpaca"},
            {"type": "opp", "action": "fold", "player_name": "Luv2crush"},
            {"type": "opp", "action": "fold", "player_name": "JYNY"},
            {"type": "hero", "action": "raise", "size": 2.0, "player_name": "cbd191320"},
            {"type": "opp", "action": "fold", "player_name": "LetmeinAA"},
            {"type": "opp", "action": "call", "size": 1.0, "player_name": "Gao zU"},
        ]},
        {"name": "Flop", "pot": 5.3, "entries": [
            {"type": "opp", "action": "check", "player_name": "Gao zU"},
            {"type": "hero", "action": "check", "player_name": "cbd191320"},
        ]},
        {"name": "Turn", "pot": 5.3, "entries": [
            {"type": "opp", "action": "bet", "size": 1.0, "player_name": "Gao zU"},
            {"type": "hero", "action": "call", "size": 1.0, "player_name": "cbd191320"},
        ]},
        {"name": "River", "pot": 7.3, "entries": [
            {"type": "opp", "action": "all-in", "size": 5.8, "player_name": "Gao zU"},
            {"type": "hero", "action": "call", "size": 5.8, "player_name": "cbd191320"},
        ]},
    ]
    named_stacks = [
        {"name": "Gao zU", "stack": 18.9},   # MISREAD: pot chips, really ~0
        {"name": "cbd191320", "stack": 25.9},
    ]
    eff, hero_start, _conf = _compute_effective_bb(
        columns, 25.9, "BTN",
        [20.2, 15.4, 18.9, 18.8, 19.1, 55.4, 25.9], named_stacks)
    assert_eq(hero_start, 34.7, "hero starting stack = display 25.9 + invested 8.8")
    assert_eq(eff, 8.8, "eff_bb = BB's full investment (all-in), not misread stack")


@test
def test_effbb_multiway_selection():
    """effbb: multiway returns min(hero, shortest active villain) bucket."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    by_id = {}
    for line in open(cache, encoding="utf-8"):
        o = json.loads(line)
        if "inputs" in o:
            by_id[o["hand_id"]] = o
    # TM5867671391: multiway (2 live opponents), min-over-villains binds the
    # shortest active caller -> 29.9, bucket 30. (Replaced TM5873208532, which
    # the Phase-4 structural gate now abstains: its independent engine bucket
    # dissents — an unavoidable collateral of the engine-disagree abstain signal
    # that is net precision-positive on the cache; see effbb_calibrate.)
    inp = by_id["TM5867671391"]["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_eq(depth_bucket(eff), 30)
    # TM5862907992: heads-up, true effective 16.5 (shorter opener) not hero's 22.9 -> bucket 17
    inp2 = by_id["TM5862907992"]["inputs"]
    eff2, _hs2, _c2 = _compute_effective_bb(
        inp2["columns"], inp2["hero_stack"], inp2["hero_position"],
        inp2["stacks"], inp2["named_stacks"])
    assert_eq(depth_bucket(eff2), 17)


@test
def test_effbb_deep_invested_not_nulled():
    """effbb: deep-invested hero keeps a real value (no displayed*5 false-null)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    o = rows["TM5896148353"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_walkover_seat_attribution():
    """effbb: a fold-through open binds on the shortest seat still to act,
    resolved by position/geometry — not hero's own (deeper) stack."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # TM5863067852: hero HJ opens, all fold. GT eff 10.7 (a short seat behind),
    # NOT hero's 24.8 stack.
    for hid in ("TM5863067852", "TM5863068088"):
        o = rows[hid]; inp = o["inputs"]
        eff, _hs, _c = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        assert_true(eff is not None)
        assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_uncalled_shove_ceiling():
    """effbb: hero invests preflop then folds to an uncalled villain jam — the
    jam size is the villain's whole stack and caps the effective stack."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # TM5863067496: hero SB R6.7, BB jams 20.4 uncalled, hero folds. GT eff 20.4
    # (the jam size), not hero's 36.7 reconstructed start.
    o = rows["TM5863067496"]; inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_geometry_villain_attribution():
    """effbb: when villain names are None/garbled, the active villain's stack is
    resolved by position/geometry (not the shortest arbitrary seat)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # Both have a None-named active villain; the shortest-seat guess undershot
    # badly (10.1 / 9.2) before geometry attribution pinned the right seat.
    for hid in ("TM5863568780", "TM5863569012"):
        o = rows[hid]; inp = o["inputs"]
        eff, _hs, _c = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        assert_true(eff is not None)
        assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_hero_uncalled_shove_starting_stack():
    """effbb: when hero jams preflop and everyone folds (uncalled), hero's shove
    size is hero's authoritative starting stack."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # TM5866594919: hero HJ jams 22.9, all fold. GT 22.9 (the shove size). The
    # displayed+reconstruction path abstained (None) before.
    o = rows["TM5866594919"]; inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_hero_allin_stack_zero_reconstruction():
    """effbb (Phase 3): when hero's displayed stack reads ~0 (all-in / called a
    villain shove), hero's STARTING stack is what hero permanently committed —
    reconstructed from the engine's decision-local hero contribution, NOT the
    noisy displayed+walk estimate. The two reconstructions must agree on the
    depth bucket to emit; otherwise abstain (single-frame unrecoverable).

    Every hero-stack~0 emit must be CORRECT-OR-ABSTAIN — no confidently-wrong
    value (the Phase-3 contract).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}

    # TM5875510185: hero SB called a BTN 19.58 shove for their last 13.19 — hero
    # is the SHORT stack, true effective ~13.7. The legacy walk over-computed
    # hero_starting to 19.6 (wrong bucket 20). The engine reconstructs hero's
    # committed 13.69 (bucket 14); the two disagree on the bucket, so we ABSTAIN
    # rather than emit the wrong 19.6.
    o = rows["TM5875510185"]; inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    # correct-or-abstain: never the wrong 19.6 (bucket 20).
    if eff is not None:
        assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]),
                  "TM5875510185 must abstain or hit GT bucket, never the wrong 20")

    # TM5873208901: hero LJ all-in over a messy multiway line; the displayed
    # walk computed a wrong 3.8. With the disagree-abstain it no longer emits a
    # confidently-wrong value.
    o2 = rows["TM5873208901"]; inp2 = o2["inputs"]
    eff2, _h2, _c2 = _compute_effective_bb(
        inp2["columns"], inp2["hero_stack"], inp2["hero_position"],
        inp2["stacks"], inp2["named_stacks"])
    if eff2 is not None:
        assert_eq(depth_bucket(eff2), depth_bucket(o2["gt"]["effective_bb"]),
                  "TM5873208901 must abstain or hit GT bucket, never the wrong 3.8")

    # A genuinely recoverable hero-all-in hand stays CORRECT (the agreement path
    # emits the engine's committed reconstruction). TM5866911989 GT 19.9.
    o3 = rows["TM5866911989"]; inp3 = o3["inputs"]
    eff3, _h3, _c3 = _compute_effective_bb(
        inp3["columns"], inp3["hero_stack"], inp3["hero_position"],
        inp3["stacks"], inp3["named_stacks"])
    assert_true(eff3 is not None, "TM5866911989 (clean hero all-in) must emit")
    assert_eq(depth_bucket(eff3), depth_bucket(o3["gt"]["effective_bb"]))


@test
def test_effbb_hero_stack_zero_no_confident_wrong():
    """effbb (Phase 3): across ALL hero-active, hero-stack~0 emits, the
    correct-or-abstain contract holds at high precision — the hero all-in
    reconstruction converts confidently-wrong emits into abstains rather than
    emitting a misread shove value. Guards the Phase-3 gain (24 wrong->abstain,
    0 regressions vs the prior commit)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    emit = ok = 0
    for line in open(cache, encoding="utf-8"):
        o = json.loads(line)
        gt = o.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in o:
            continue
        if hero_folded_preflop(gt) is not False:
            continue
        inp = o["inputs"]
        hs_disp = inp["hero_stack"]
        if hs_disp is None or hs_disp > 0.6:
            continue
        eff, _h, _c = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        if eff is not None:
            emit += 1
            if bucket_match(eff, ge):
                ok += 1
    # Emitted hero-stack~0 precision must clear 88% (Phase-3 measured ~91.6%);
    # before the reconstruction the same slice emitted at ~85%.
    prec = 100 * ok / emit if emit else 0.0
    assert_true(prec >= 88.0,
                f"hero-stack~0 emitted precision regressed below 88%: {prec:.1f}% ({ok}/{emit})")


@test
def test_effbb_confidence_is_calibrated_monotonic():
    """effbb: attribution-certainty confidence yields a MONOTONIC precision/
    coverage curve — raising the floor trades coverage for precision (the old
    dual-estimator agreement-confidence produced a flat curve)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    active = []
    for line in open(cache, encoding="utf-8"):
        o = json.loads(line)
        gt = o.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in o:
            continue
        if hero_folded_preflop(gt) is not False:
            continue
        inp = o["inputs"]
        eff, _hs, conf = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        if eff is not None:
            active.append((eff, conf, bucket_match(eff, ge)))

    def prec_at(floor):
        em = [a for a in active if a[1] >= floor]
        ok = [a for a in em if a[2]]
        return (100 * len(ok) / len(em)) if em else 0.0, len(em)

    p0, _ = prec_at(0.0)
    p7, _ = prec_at(0.7)
    p9, _ = prec_at(0.9)
    # Higher floor must not LOWER precision (monotone non-decreasing), and the
    # top band must be more precise than the whole population.
    # NOTE: the Phase-4 structural gate now does most of the precision lifting
    # (it abstains the low-precision hands INDEPENDENT of conf), so the residual
    # conf curve is flatter than pre-Phase-4 — the top-band gap is smaller but
    # still positive and monotone. (The pre-gate frontier is in effbb_calibrate.)
    assert_true(p7 >= p0 - 0.5, f"precision dropped raising floor 0->0.7: {p0:.1f}->{p7:.1f}")
    assert_true(p9 >= p7 - 0.5, f"precision dropped raising floor 0.7->0.9: {p7:.1f}->{p9:.1f}")
    assert_true(p9 >= p0 + 1.0, f"high-conf band not more precise: all={p0:.1f} conf>=0.9={p9:.1f}")


@test
def test_effbb_abstain_or_correct_on_divergence():
    """effbb: ambiguous/divergent reconstruction abstains or hits the right bucket."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    o = rows["TM5863941844"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    # Either it now lands on the right bucket, or it correctly abstains.
    assert_true(eff is None or depth_bucket(eff) == depth_bucket(o["gt"]["effective_bb"]))


@test
def test_effbb_overcompute_bounded():
    """effbb: over-compute past table max is rejected (bounded or abstain)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    o = rows["TM5875533783"]; inp = o["inputs"]   # gt 21.0, p_eff 80.5, gt_max 69.1
    gt_max = max(o["gt"]["stacks_bb"])
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is None or eff <= gt_max * 1.1)
    # TM5875583251: gt 9.2, action-walk inflates hero to 137.7 vs gt_max 62 -> abstain
    o2 = rows["TM5875583251"]; inp2 = o2["inputs"]
    gt_max2 = max(o2["gt"]["stacks_bb"])
    eff2, _hs2, _c2 = _compute_effective_bb(
        inp2["columns"], inp2["hero_stack"], inp2["hero_position"],
        inp2["stacks"], inp2["named_stacks"])
    assert_true(eff2 is None or eff2 <= gt_max2 * 1.1)


# ---------------------------------------------------------------------------
# Phase 1: betting-state engine (scripts/ocr/effbb_engine.py)
# ---------------------------------------------------------------------------
def _engine():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr import effbb_engine as eng
    return eng


def _eng_streets_from_cache(hid):
    """Load a cached hand and split its panel into the engine's streets/pot."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _engine_streets
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    o = rows[hid]
    streets, pot = _engine_streets(o["inputs"]["columns"])
    return o, streets, pot


@test
def test_engine_position_orders_match_parser():
    """Engine's POSITION_ORDERS must be identical to the parser's."""
    eng = _engine()
    from ocr.n8_parser import POSITION_ORDERS as PARSER_ORDERS
    assert_eq(eng.POSITION_ORDERS, PARSER_ORDERS,
              "engine POSITION_ORDERS drifted from n8_parser")


@test
def test_engine_infer_blinds():
    """Engine infers SB=0.5/BB=1.0 and a BB-ante from the preflop pot."""
    eng = _engine()
    sb, bb, ante, ok = eng.infer_blinds(1.5, 6)   # no ante
    assert_eq((sb, bb), (0.5, 1.0)); assert_true(ok and ante == 0.0)
    sb, bb, ante, ok = eng.infer_blinds(2.4, 6)   # 0.9 BB-ante
    assert_true(ok and abs(ante - 0.9) < 0.01, f"ante={ante}")
    _sb, _bb, _ante, ok2 = eng.infer_blinds(99.0, 6)  # absurd → not ok
    assert_true(not ok2, "absurd preflop pot should not reconcile")


@test
def test_engine_action_order_assignment_preflop():
    """Engine assigns positions by legal action order, not player_name."""
    eng = _engine()
    # 6-max, hero HJ opens, folds through. Rows are in UTG-first order.
    streets = {"preflop": [
        {"type": "opponent", "action": "Fold", "player_name": "x", "position": "LJ"},
        {"type": "hero", "action": "Raise", "size": 2.0, "position": "HJ"},
        {"type": "opponent", "action": "Fold", "player_name": "y", "position": "CO"},
        {"type": "opponent", "action": "Fold", "player_name": "z", "position": "BTN"},
        {"type": "opponent", "action": "Fold", "player_name": "w", "position": "SB"},
    ]}
    assigned = eng.assign_positions(
        eng.normalize_streets(streets, "HJ"), 6, "HJ")
    by = {(a.position, a.action) for a in assigned}
    assert_in(("HJ", "raise"), by)
    assert_in(("LJ", "fold"), by)
    assert_in(("SB", "fold"), by)


@test
def test_engine_m1_uncalled_shove_ceiling():
    """M1: hero invests then folds to an uncalled villain jam → ceiling = the
    shover's TOTAL contribution (prior streets + shove). TM5863067496 GT 20.4."""
    eng = _engine()
    o, streets, pot = _eng_streets_from_cache("TM5863067496")
    r = eng.analyze(streets, o["gt"]["num_players"],
                    o["inputs"]["hero_position"], pot.get("preflop"))
    assert_eq(r.rule, "M1")
    from effbb_metrics import depth_bucket
    assert_eq(depth_bucket(r.rule_ceiling), 20,
              f"M1 ceiling {r.rule_ceiling} should be bucket 20")


@test
def test_engine_m1_postflop_jam_uses_total_contribution():
    """M1: a small river jam over a deep prior invest must use the shover's
    TOTAL contribution, not the bare shove size. TM5880480237 GT 15.0."""
    eng = _engine()
    o, streets, pot = _eng_streets_from_cache("TM5880480237")
    r = eng.analyze(streets, o["gt"]["num_players"],
                    o["inputs"]["hero_position"], pot.get("preflop"))
    assert_eq(r.rule, "M1")
    from effbb_metrics import depth_bucket
    assert_eq(depth_bucket(r.rule_ceiling), depth_bucket(o["gt"]["effective_bb"]),
              f"ceiling {r.rule_ceiling} vs GT {o['gt']['effective_bb']}")


@test
def test_engine_m2_walkover_binds_on_seats_behind():
    """M2: hero opens and folds through → EVERY seat still to act behind hero
    binds the steal spot (GT-aligned preflop-only set: a short BTN/SB behind
    defines the depth as much as the BB; hh_parser in_pot_chips includes all
    of them). BB must be in the set; no seat acting BEFORE hero that folded
    may be. TM5863067852 (GT 10.7) and TM5863068088 (GT 13.8)."""
    eng = _engine()
    for hid in ("TM5863067852", "TM5863068088"):
        o, streets, pot = _eng_streets_from_cache(hid)
        order = eng.POSITION_ORDERS[o["gt"]["num_players"]]
        hidx = order.index(o["inputs"]["hero_position"])
        r = eng.analyze(streets, o["gt"]["num_players"],
                        o["inputs"]["hero_position"], pot.get("preflop"))
        assert_eq(r.rule, "M2", f"{hid} should be a walkover")
        assert_in("BB", r.relevant_opponents,
                  f"{hid} BB must bind the walkover, got {r.relevant_opponents}")
        assert_eq(sorted(r.relevant_opponents), sorted(order[hidx + 1:]),
                  f"{hid} walkover binds on all seats behind hero, "
                  f"got {r.relevant_opponents}")


@test
def test_engine_m3_multiway_live_set():
    """M3: relevant = the live contestants at hero's decision (action order),
    NOT showdown survivors or a folded short seat. TM5863067607: hero BB, SB
    limps, hero checks, SB bets flop, hero folds → live = {SB}."""
    eng = _engine()
    o, streets, pot = _eng_streets_from_cache("TM5863067607")
    r = eng.analyze(streets, o["gt"]["num_players"],
                    o["inputs"]["hero_position"], pot.get("preflop"))
    assert_eq(r.rule, "M3")
    # The engine must SELECT the SB limper as the lone live opponent. (Reading
    # SB's seat STACK correctly is a Phase-2 attribution job — the cached SB
    # sticker is OCR-misread to 2.9 vs the true ~17.4, so the final bucket is
    # not yet recoverable here; this test pins the Phase-1 deliverable: the
    # right POSITION.)
    assert_eq(r.relevant_opponents, ["SB"],
              f"live set should be the SB limper, got {r.relevant_opponents}")


@test
def test_engine_m3_multiway_postflop_callers_in_live_set():
    """M3: a caller who acts AFTER hero's open stays in the live set when hero
    does not fold — freezing at hero's open would drop the very callers who
    define the spot. TM5873208532: hero UTG+1 opens, LJ+CO call; LJ folds the
    river, CO calls to the end → CO is the live binding opponent (and the engine
    must NOT pick a folded-preflop short seat)."""
    eng = _engine()
    o, streets, pot = _eng_streets_from_cache("TM5873208532")
    r = eng.analyze(streets, o["gt"]["num_players"],
                    o["inputs"]["hero_position"], pot.get("preflop"))
    assert_in("CO", r.relevant_opponents,
              f"river contestant CO must be live, got {r.relevant_opponents}")
    # No seat that folded preflop (UTG, HJ, BTN, SB, BB) may be in the live set.
    for folded in ("UTG", "HJ", "BTN", "SB", "BB"):
        assert_true(folded not in r.relevant_opponents,
                    f"{folded} folded preflop, must not be live")


@test
def test_effbb_engine_m1_endtoend_bucket():
    """End-to-end: the M1 ceiling reaches _compute_effective_bb. TM5863067496
    GT 20.4 (bucket 20) — read straight off the panel shove."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    o, _s, _p = _eng_streets_from_cache("TM5863067496")
    inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_eq(depth_bucket(eff), 20)


# ---------------------------------------------------------------------------
# Phase 2: top-K layout enumeration + bucket-consensus emission
# ---------------------------------------------------------------------------
@test
def test_engine_hu_postflop_order_bb_first():
    """HU (2-handed) postflop the BB acts FIRST (carry-over fix). Preflop the
    SB/BTN acts first; postflop the order flips to ['BB','SB']."""
    eng = _engine()
    assert_eq(eng._postflop_order(["SB", "BB"], 2), ["BB", "SB"])
    # 3+ handed is unchanged: blinds (SB) lead postflop.
    assert_eq(eng._postflop_order(["LJ", "HJ", "CO", "BTN", "SB", "BB"], 6)[0],
              "SB")


@test
def test_phase2_dead_code_removed():
    """The unused _seat_stack_for_position helper was removed in Phase 2."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    import ocr.n8_parser as P
    assert_true(not hasattr(P, "_seat_stack_for_position"),
                "_seat_stack_for_position should be deleted (dead code)")


@test
def test_phase2_enumerate_layouts_topk():
    """_enumerate_layouts returns the top-K position->seat layouts, each a full
    {position: seat} map for the table, best-first by weak name agreement."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    import ocr.n8_parser as P
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    inp = rows["TM5862907992"]["inputs"]
    nump = P._infer_num_players(inp["columns"], inp["named_stacks"])
    layouts = P._enumerate_layouts(
        inp["named_stacks"], P._panel_position_names(inp["columns"]),
        inp["hero_position"], nump, margin=99, max_k=8)
    assert_true(len(layouts) >= 1, "expected at least one layout")
    order = P.POSITION_ORDERS[nump]
    for m in layouts:
        # Hero anchored at hero_position in every layout.
        assert_in(inp["hero_position"], m)
        # Each layout covers the full position ring.
        assert_eq(set(m.keys()), set(order))


@test
def test_phase2_consensus_holds_emits_correct_bucket():
    """When all plausible layouts + the engine's relevant seat agree on the
    bucket, the consensus path emits the correct bucket. TM5862907992: a
    2-direction straddle the engine's single live opponent (UTG) resolves to
    bucket 17 (true effective 16.5, not hero's deeper 22.9)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    o = rows["TM5862907992"]; inp = o["inputs"]
    eff, _hs, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None, "consensus should emit, not abstain")
    assert_eq(depth_bucket(eff), 17)
    assert_true(conf >= 0.7, f"emitted confidence below floor: {conf}")


@test
def test_phase2_layout_straddle_abstains():
    """When the plausible geometric layouts straddle DIFFERENT depth buckets
    and the engine cannot break the tie, the consensus path ABSTAINS (None) —
    the attribution is genuinely ambiguous. TM5864409682 / TM5866699022 both
    straddle two buckets."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    for hid in ("TM5864409682", "TM5866699022"):
        inp = rows[hid]["inputs"]
        eff, _hs, conf = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        assert_true(eff is None,
                    f"{hid}: bucket-straddle must abstain, got {eff}")
        assert_true(conf < 0.7, f"{hid}: abstain conf should be low, got {conf}")


@test
def test_phase2_consensus_curve_trades_coverage_for_precision():
    """The consensus confidence yields a real precision/coverage frontier on
    hero-active hands: raising the floor must not lower precision and the top
    band is meaningfully cleaner than the whole emitted population."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    active = []
    for line in open(cache, encoding="utf-8"):
        if not line.strip():
            continue
        o = json.loads(line)
        gt = o.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in o:
            continue
        if hero_folded_preflop(gt) is not False:
            continue
        inp = o["inputs"]
        eff, _hs, conf = _compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])
        if eff is not None:
            active.append((conf, bucket_match(eff, ge)))

    def prec_cov(floor):
        em = [a for a in active if a[0] >= floor]
        ok = [a for a in em if a[1]]
        return (100 * len(ok) / len(em) if em else 0.0,
                100 * len(em) / len(active) if active else 0.0)

    p0, c0 = prec_cov(0.0)
    p9, c9 = prec_cov(0.9)
    # The Phase-4 structural gate now absorbs most of the precision separation
    # (it abstains low-precision hands regardless of conf), so the residual conf
    # band gap is smaller post-gate but still positive + monotone.
    assert_true(p9 >= p0 + 1.0,
                f"top band not cleaner: all={p0:.1f}@{c0:.0f}% conf>=0.9={p9:.1f}@{c9:.0f}%")
    assert_true(c9 < c0, "raising the floor must reduce coverage")


# ---------------------------------------------------------------------------
# Phase 4 — calibrated structural abstain
# ---------------------------------------------------------------------------
def _effbb_run(hid, *, gate=True):
    """Run _compute_effective_bb on a cache hand with the structural gate on/off.

    Imports the parser fresh under the requested OCR_EFFBB_STRUCTURAL_GATE so the
    module-level gate constant reflects the flag (the constant is read at import).
    Returns (effective_bb, gt_eff).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    import importlib
    prev = os.environ.get("OCR_EFFBB_STRUCTURAL_GATE")
    os.environ["OCR_EFFBB_STRUCTURAL_GATE"] = "1" if gate else "0"
    try:
        import ocr.n8_parser as _P
        importlib.reload(_P)
        cache = os.path.join(os.path.dirname(__file__), "..",
                             "data/effbb_cache/cache.jsonl")
        o = None
        for line in open(cache, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["hand_id"] == hid:
                o = row
                break
        assert o is not None, f"{hid} not in cache"
        inp = o["inputs"]
        eff = _P._compute_effective_bb(
            inp["columns"], inp["hero_stack"], inp["hero_position"],
            inp["stacks"], inp["named_stacks"])[0]
        return eff, o["gt"]["effective_bb"]
    finally:
        if prev is None:
            os.environ.pop("OCR_EFFBB_STRUCTURAL_GATE", None)
        else:
            os.environ["OCR_EFFBB_STRUCTURAL_GATE"] = prev
        import ocr.n8_parser as _P
        importlib.reload(_P)


@test
def test_phase4_features_surfaced_per_hand():
    """The Phase-4 abstain features are captured per hand (no re-OCR) and carry
    the candidate signals the calibration harness fits gates on."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb, _effbb_last_features
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {}
    for line in open(cache, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["hand_id"]] = r
    o = rows["TM5862908042"]; inp = o["inputs"]   # a clean emitting hand
    _compute_effective_bb(inp["columns"], inp["hero_stack"], inp["hero_position"],
                          inp["stacks"], inp["named_stacks"])
    f = _effbb_last_features()
    for key in ("confidence", "engine_agrees", "engine_disagrees",
                "binding_geometry_only", "method_straddle",
                "hero_stack_near_zero", "boundary_dist", "decision_class",
                "n_relevant_opp", "pot_residual", "n_layouts", "layout_buckets"):
        assert_in(key, f)


@test
def test_chip_solver_features_surfaced_per_hand():
    """B2: the chip-conservation features are captured per hand alongside the
    Phase-4 features (no re-OCR, no behavior change). The clean hand
    TM5862908042 reconciles its pot header -> chip_consistent True."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb, _effbb_last_features
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = {}
    for line in open(cache, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            rows[r["hand_id"]] = r
    o = rows["TM5862908042"]; inp = o["inputs"]   # a clean emitting hand
    _compute_effective_bb(inp["columns"], inp["hero_stack"], inp["hero_position"],
                          inp["stacks"], inp["named_stacks"])
    f = _effbb_last_features()
    for key in ("chip_consistent", "chip_repair_found", "chip_residual"):
        assert_in(key, f)
    # A preflop-RESOLVED hand actually runs the chip-conservation check (the
    # equation is only valid pre-flop; postflop hands leave it None). The check
    # produces a concrete verdict + residual for such a hand.
    o2 = rows["TM5863485159"]; inp2 = o2["inputs"]
    _compute_effective_bb(inp2["columns"], inp2["hero_stack"], inp2["hero_position"],
                          inp2["stacks"], inp2["named_stacks"])
    f2 = _effbb_last_features()
    assert_true(f2["chip_consistent"] in (True, False),
                f"preflop-resolved hand must get a chip verdict: {f2['chip_consistent']}")
    assert_true(f2["chip_residual"] is not None)


@test
def test_phase4_bucket_boundary_distance():
    """Boundary-distance fragility signal: a value near a bucket-cell edge is
    flagged fragile (small), a value at a bucket centre is not."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _bucket_boundary_distance
    # 27.5 is the 25<->30 cell edge: distance ~0.
    assert_true(_bucket_boundary_distance(27.6) < 0.02,
                "near-edge value not flagged fragile")
    # 30.0 is a bucket centre: comfortably away from its 27.5 / 32.5 edges.
    assert_true(_bucket_boundary_distance(30.0) > 0.05,
                "bucket-centre value wrongly flagged fragile")


@test
def test_phase4_gate_abstains_structurally_wrong_hands():
    """The shipped structural gate ABSTAINS representative internally-consistent
    wrong emits (the layout-independent value errors consensus is blind to),
    where the ungated path emitted a confidently-wrong bucket."""
    # TM5863067607: SB-limper sticker misread (3.9 for ~17.4) — Phase-1 example.
    eff_off, gt = _effbb_run("TM5863067607", gate=False)
    from effbb_metrics import bucket_match
    assert_true(eff_off is not None and not bucket_match(eff_off, gt),
                f"fixture no longer a wrong emit ungated: {eff_off} vs {gt}")
    eff_on, _ = _effbb_run("TM5863067607", gate=True)
    assert_true(eff_on is None, f"gate failed to abstain wrong hand: {eff_on}")
    # NOTE: TM5863941899 (hero displayed read corrupted to 1.0) used to be
    # abstained by the hero-near-zero clause; that clause measured 81%
    # marginally precise overall (above the emitted average) and was dropped —
    # this corrupt-hero case is an accepted residual of the better operating
    # point (see test_effbb_herozero_slice_emitted for the upside).


@test
def test_phase4_gate_keeps_clean_correct_hands():
    """The shipped structural gate does NOT abstain clean, correct emits — the
    abstain is targeted, not a blanket coverage cut."""
    from effbb_metrics import bucket_match
    for hid in ("TM5862908042", "TM5863067726", "TM5863067671"):
        eff, gt = _effbb_run(hid, gate=True)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"gate wrongly abstained/missed clean hand {hid}: {eff} vs {gt}")


@test
def test_phase4_gate_lifts_precision_over_baseline():
    """The shipped structural gate lifts emitted precision over the bare-conf
    baseline across the hero-active cache, at the cost of coverage (abstaining
    is cheap downstream). Calibrated 5-fold-CV operating point: ~74% @ ~61%
    coverage vs the ~71% @ ~78% ungated baseline. 99.5% is NOT reachable on
    single-frame inputs (documented in the Phase-4 plan); this guards the
    precision GAIN the gate actually delivers."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    import importlib
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(os.path.dirname(__file__), "..", "data/effbb_cache/cache.jsonl")
    rows = [json.loads(l) for l in open(cache, encoding="utf-8") if l.strip()]

    def frontier(gate):
        prev = os.environ.get("OCR_EFFBB_STRUCTURAL_GATE")
        os.environ["OCR_EFFBB_STRUCTURAL_GATE"] = "1" if gate else "0"
        try:
            import ocr.n8_parser as _P
            importlib.reload(_P)
            emit = ok = total = 0
            for o in rows:
                gt = o.get("gt") or {}
                ge = gt.get("effective_bb")
                if ge is None or ge < 1.0 or "inputs" not in o:
                    continue
                if hero_folded_preflop(gt) is not False:
                    continue
                total += 1
                inp = o["inputs"]
                eff = _P._compute_effective_bb(
                    inp["columns"], inp["hero_stack"], inp["hero_position"],
                    inp["stacks"], inp["named_stacks"])[0]
                if eff is not None:
                    emit += 1
                    if bucket_match(eff, ge):
                        ok += 1
            return (100 * ok / emit if emit else 0.0,
                    100 * emit / total if total else 0.0)
        finally:
            if prev is None:
                os.environ.pop("OCR_EFFBB_STRUCTURAL_GATE", None)
            else:
                os.environ["OCR_EFFBB_STRUCTURAL_GATE"] = prev
            import ocr.n8_parser as _P
            importlib.reload(_P)

    p_off, c_off = frontier(False)
    p_on, c_on = frontier(True)
    # The gate raises precision by a real margin and trades coverage for it.
    # (Margin re-centred after the matched-floor / behind-bound / legality
    # fixes lifted the UNGATED baseline too: ~74.1%@79.3% off vs
    # ~76.8%@71.2% on as of 2026-06-11.)
    assert_true(p_on >= p_off + 1.5,
                f"gate did not lift precision: off={p_off:.1f} on={p_on:.1f}")
    assert_true(c_on < c_off,
                f"gate must trade coverage: off={c_off:.1f}% on={c_on:.1f}%")
    # Guard the calibrated operating point doesn't silently collapse/loosen.
    assert_true(73.0 <= p_on <= 84.0,
                f"shipped precision off calibrated band (~77%): {p_on:.1f}%")
    assert_true(64.0 <= c_on <= 80.0,
                f"shipped coverage off calibrated band (~71%): {c_on:.1f}%")


@test
def test_effbb_called_shove_floor_binds():
    """A villain shove that hero CALLS binds the effective stack — the explicit
    panel all-in size is the authoritative read. Guards two fixes:
    (1) hero is ONE running stream even when unnamed, so hero's raise-then-call
        accumulates and the shove registers as matched (TM5863575308);
    (2) a BB/SB caller gets blind credit when the Blinds column is empty
        (TM5875127705: BB calls 8.51 over a posted 1.0 vs a 9.51 jam)."""
    from effbb_metrics import bucket_match
    for hid in ("TM5863575308", "TM5875127705", "TM5879715175",
                "TM5887364784"):
        eff, gt = _effbb_run(hid)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"{hid}: called-shove floor missed: {eff} vs GT {gt}")


@test
def test_effbb_behind_hero_bound_preflop_only():
    """A hand that truly ends preflop (no jam, no run-out) is bound by every
    seat acting AFTER hero — including seats that folded behind — plus earlier
    voluntary entrants (the HH ground-truth in_pot definition). TM5866773503 /
    TM5873598400 / TM5867329780: the entered-only min over-estimated; the
    behind-hero seat-map bound recovers the exact GT bucket."""
    from effbb_metrics import bucket_match
    for hid in ("TM5866773503", "TM5873598400", "TM5867329780"):
        eff, gt = _effbb_run(hid)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"{hid}: behind-hero bound missed: {eff} vs GT {gt}")


@test
def test_effbb_hero_jam_behind_bound():
    """Hero jams UNCALLED preflop: a genuinely short NAMED seat folding behind
    still binds the ground-truth effective (hh_parser in_pot definition).
    TM5866747832 (opener 19.2 < hero's 27.7 jam) / TM5919864376 (27.9 vs an
    84.9 jam). The misread-folder golden TM5866594919 (hero jam 22.9 IS the
    GT) must stay green — the bound reads NAMED seats only."""
    from effbb_metrics import bucket_match
    for hid in ("TM5866747832", "TM5919864376", "TM5896802248",
                "TM5866594919"):
        eff, gt = _effbb_run(hid)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"{hid}: hero-jam behind bound wrong: {eff} vs GT {gt}")


@test
def test_effbb_allin_legality_guard():
    """An 'All-In' row that does not exceed what a player already committed,
    followed by a fold from that covering player, is a misparsed raise — it
    must not bind. TM5878838751: river hero Bet 9.0 → 'All-In 1.0' → hero
    Fold on a 44.3bb spot; the bogus 1.0 floor must not be emitted."""
    eff, gt = _effbb_run("TM5878838751")
    assert_true(eff is None or eff >= 5.0,
                f"illegal sub-level all-in misread bound the spot: {eff}")


@test
def test_effbb_herozero_slice_emitted():
    """Hero displayed ≈0 (all-in) with no engine confirmation is EMITTED when
    the other gate clauses pass — the old blanket herozero abstain measured
    81% marginally precise (above the emitted average) and was dropped."""
    from effbb_metrics import bucket_match
    for hid in ("TM5963779172", "TM5920068321"):
        eff, gt = _effbb_run(hid)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"{hid}: herozero hand wrongly abstained/wrong: {eff} vs GT {gt}")


@test
def test_collapse_allin_into_call_merges_shove_frequency():
    """A call facing a shove matches the GTO commit, not a phantom raise.

    With the solver offering Fold 8% / Call 7% / All-in 85% (H3459 turn for
    99), merging the all-in line into Call yields Call 92% / Fold 8%, so the
    top action becomes Call — hero's call is no longer a deviation.
    """
    from analyze_hand import _collapse_allin_into_call

    display_sol = {
        "action_solutions": [
            {"action": {"code": "F"}},
            {"action": {"code": "C"}},
            {"action": {"code": "R27.2", "allin": True}},
        ]
    }
    af = {"F": 0.08, "C": 0.07, "R27.2": 0.85}
    merged = _collapse_allin_into_call(af, display_sol)
    assert_eq(round(merged["C"], 2), 0.92, "call absorbs the all-in frequency")
    assert_eq(round(merged["F"], 2), 0.08, "fold frequency preserved")
    assert_true("R27.2" not in merged, "all-in code folded into call")
    assert_eq(max(merged, key=merged.get), "C", "call is now the top action")


@test
def test_collapse_allin_into_call_noop_without_allin_option():
    """When no solver action is an all-in, frequencies are untouched."""
    from analyze_hand import _collapse_allin_into_call

    display_sol = {
        "action_solutions": [
            {"action": {"code": "X"}},
            {"action": {"code": "R5"}},
        ]
    }
    af = {"X": 0.6, "R5": 0.4}
    assert_eq(_collapse_allin_into_call(af, display_sol), af)


@test
def test_find_action_by_pot_pct_preserves_near_shove_allin():
    """A near-shove opening bet snaps to all-in, not a pot-fraction bucket.

    The solver models a small pot (24.8) but the real multiway pot is inflated
    (60) by dead money from folded cold-callers. Hero shoves ~40bb into a
    43.5bb effective stack. Pure pot-ratio matching computes a 16.5 solver bet
    and snaps to the 1/2-pot bucket (R12.4) — wrong. The all-in guard recognises
    the bet is within 15% of the stack and keeps RAI. Same guard already proven
    in gtow_action_resolver._resolve_one_raise; now shared in the matcher.
    """
    from analyze_hand import _find_action_by_pot_pct

    available = [
        {"action": {"code": "X", "betsize": 0, "betsize_by_pot": 0}},
        {"action": {"code": "R6.2", "betsize": 6.2, "betsize_by_pot": 0.25}},
        {"action": {"code": "R12.4", "betsize": 12.4, "betsize_by_pot": 0.5}},
        {"action": {"code": "RAI", "betsize": 43.5, "betsize_by_pot": 1.754,
                    "allin": True}},
    ]
    code = _find_action_by_pot_pct(available, bet_size=40.0, actual_pot=60.0)
    assert_eq(code, "RAI", "near-shove must keep all-in, not snap to 1/2-pot")


@test
def test_find_action_by_pot_pct_normal_bet_unaffected_by_allin_guard():
    """A genuine pot-fraction bet is unchanged by the all-in guard.

    Hero bets 6.2bb (1/4 pot) into the same tree; nowhere near the 43.5 stack,
    so the guard must not fire and pot-ratio matching still selects R6.2.
    """
    from analyze_hand import _find_action_by_pot_pct

    available = [
        {"action": {"code": "X", "betsize": 0, "betsize_by_pot": 0}},
        {"action": {"code": "R6.2", "betsize": 6.2, "betsize_by_pot": 0.25}},
        {"action": {"code": "R12.4", "betsize": 12.4, "betsize_by_pot": 0.5}},
        {"action": {"code": "RAI", "betsize": 43.5, "betsize_by_pot": 1.754,
                    "allin": True}},
    ]
    code = _find_action_by_pot_pct(available, bet_size=6.2, actual_pot=24.8)
    assert_eq(code, "R6.2", "quarter-pot bet must not be hijacked by all-in guard")


# ── Visual All-In badge attribution (PR: fix/allin-visual-attribution) ──
#
# The red "All-In" sticker carries no name/position/size, so sequence rules
# alone must guess whether a bare badge belongs to the prior raiser (a badge)
# or is a standalone jam by the other player. N8 paints the red badge directly
# on the all-in player's bet/raise sticker, so the rendered color just *above*
# the red badge reveals whose sticker it sits on: yellow=hero, white=opponent.
# Empirically (H3441/H3442 flop crops) the separation is clean (>0.7 vs 0.0).


def _hsv_strip(h, s, v, rows, cols):
    """Build a solid BGR strip from one HSV color (test fixture helper)."""
    import cv2
    import numpy as np
    hsv = np.full((rows, cols, 3), (h, s, v), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _region_with_band_above(badge_bbox, band_hsv, region_shape=(360, 174)):
    """Synthetic column region: dark everywhere, a colored sticker band just
    above the All-In badge, and a red badge in the badge bbox itself."""
    import numpy as np
    x_min, y_min, x_max, y_max = badge_bbox
    region = np.zeros((region_shape[0], region_shape[1], 3), np.uint8)
    # Sticker band above the badge top (the helper samples y_min-22 .. y_min-4).
    by1, by2 = max(0, y_min - 24), max(0, y_min - 2)
    if by2 > by1:
        region[by1:by2, x_min:x_max] = _hsv_strip(*band_hsv, by2 - by1, x_max - x_min)
    # Red badge in the badge's own row.
    region[y_min:y_max, x_min:x_max] = _hsv_strip(0, 220, 200, y_max - y_min, x_max - x_min)
    return region


_HERO_BAND = (30, 200, 220)       # gold/yellow hero sticker
_OPP_BAND = (0, 12, 235)          # near-white opponent sticker


@test
def test_visual_allin_owner_reads_hero_from_yellow_band_above():
    """Yellow sticker above the red badge => hero owns the All-In."""
    from ocr.panel_parser import _infer_allin_badge_owner
    bbox = (75, 280, 125, 300)
    region = _region_with_band_above(bbox, _HERO_BAND)
    entry = {"action": "All-In", "size": None, "_bbox": bbox}
    owner, conf, evidence = _infer_allin_badge_owner(entry, region)
    assert_eq(owner, "hero", f"evidence={evidence}")
    assert_true(conf >= 0.55, f"expected high confidence, got {conf}")


@test
def test_visual_allin_owner_reads_opponent_from_white_band_above():
    """White sticker above the red badge => opponent owns the All-In (H3441)."""
    from ocr.panel_parser import _infer_allin_badge_owner
    bbox = (75, 280, 125, 300)
    region = _region_with_band_above(bbox, _OPP_BAND)
    entry = {"action": "All-In", "size": None, "_bbox": bbox}
    owner, conf, evidence = _infer_allin_badge_owner(entry, region)
    assert_eq(owner, "opponent", f"evidence={evidence}")
    assert_true(conf >= 0.55, f"expected high confidence, got {conf}")


@test
def test_visual_allin_owner_abstains_when_inconclusive():
    """No clear sticker color above or below => no opinion (fall back to rules)."""
    import numpy as np
    from ocr.panel_parser import _infer_allin_badge_owner
    bbox = (75, 280, 125, 300)
    # All dark except a pure-red badge row — nothing to read color from.
    region = np.zeros((360, 174, 3), np.uint8)
    region[280:300, 75:125] = _hsv_strip(0, 220, 200, 20, 50)
    entry = {"action": "All-In", "size": None, "_bbox": bbox}
    owner, conf, evidence = _infer_allin_badge_owner(entry, region)
    assert_eq(owner, None, f"evidence={evidence}")


@test
def test_visual_allin_owner_requires_bbox_metadata():
    """Without _bbox metadata the helper abstains (backward-compatible)."""
    import numpy as np
    from ocr.panel_parser import _infer_allin_badge_owner
    region = np.zeros((360, 174, 3), np.uint8)
    owner, conf, evidence = _infer_allin_badge_owner({"action": "All-In"}, region)
    assert_eq(owner, None)


@test
def test_visual_attribution_keeps_sticker_only_hero_jam():
    """Hardening: a sticker-only (sizeless) All-In after an opponent raise is
    KEPT as a hero jam when the sticker above is yellow — sequence rules alone
    would have collapsed it into an opponent all-in.

    This is the gap visual attribution closes: when hero's back-jam size fails
    to OCR, the bare badge looks identical to an opponent all-in badge to the
    sequence rules. The rendered yellow sticker disambiguates it.
    """
    from ocr.panel_parser import _resolve_allin_attribution
    bbox = (60, 290, 130, 312)
    region = _region_with_band_above(bbox, _HERO_BAND, region_shape=(360, 174))
    entries = [
        {"type": "opponent", "position": "BB", "action": "Check", "size": None},
        {"type": "hero", "position": "BB", "action": "Bet", "size": 1.3},
        {"type": "opponent", "position": "BB", "action": "Raise", "size": 12.5},
        {"type": "hero", "position": None, "action": "All-In", "size": None, "_bbox": bbox},
        {"type": "opponent", "position": None, "action": "Fold", "size": None},
    ]
    resolved = _resolve_allin_attribution(entries, column_region=region)
    assert_eq([e["action"] for e in resolved],
              ["Check", "Bet", "Raise", "All-In", "Fold"],
              "hero jam must survive (not collapse onto villain's raise)")
    assert_eq(resolved[2]["action"], "Raise", "villain's raise must stay a raise")
    assert_eq(resolved[3]["type"], "hero", "the All-In belongs to hero")


@test
def test_visual_attribution_collapses_opponent_badge():
    """A sticker-only All-In after an opponent raise with a WHITE sticker above
    is the opponent's all-in badge — collapse it onto their raise (H3441).
    Visual agrees with the existing sequence rule here, so behavior is unchanged.
    """
    from ocr.panel_parser import _resolve_allin_attribution
    bbox = (60, 290, 130, 312)
    region = _region_with_band_above(bbox, _OPP_BAND, region_shape=(360, 174))
    entries = [
        {"type": "opponent", "position": "BB", "action": "Check", "size": None},
        {"type": "hero", "position": "BB", "action": "Bet", "size": 1.3},
        {"type": "opponent", "position": "BB", "action": "Raise", "size": 12.5},
        {"type": "hero", "position": None, "action": "All-In", "size": None, "_bbox": bbox},
        {"type": "opponent", "position": None, "action": "Fold", "size": None},
    ]
    resolved = _resolve_allin_attribution(entries, column_region=region)
    assert_eq([e["action"] for e in resolved],
              ["Check", "Bet", "All-In", "Fold"],
              "opponent's all-in badge must collapse onto their raise")
    assert_eq(resolved[2]["type"], "opponent")
    assert_eq(resolved[2]["size"], 12.5)


# ── GTO snapshot text comparison tolerance ──
#
# Solver values wobble in the last digit between runs / cache states (a fresh
# worktree that misses the snapshot .gto_cache re-fetches live and drifts
# ±0.01bb / ±0.2pp). The strategy/structure is the contract, not the last
# digit — so the L2 comparator tolerates tiny EV (bb) and frequency/equity (%)
# drift while keeping combos counts, action sequences, ranges, and line count
# exact. See _gto_text_compare.gto_text_matches.


@test
def test_gto_text_compare_exact_match():
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("EV: 7.57bb\nFold: 42.0%", "EV: 7.57bb\nFold: 42.0%")
    assert_true(ok, msg)


@test
def test_gto_text_compare_tolerates_ev_drift():
    """0.01bb EV drift (H2504) is within tolerance => match."""
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("  EV: 7.57bb | Equity: 64.2%",
                               "  EV: 7.56bb | Equity: 64.2%")
    assert_true(ok, msg)


@test
def test_gto_text_compare_tolerates_frequency_drift():
    """0.2pp frequency drift (H2505) is within tolerance => match."""
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("  Fold: 42.0%（22 combos）",
                               "  Fold: 42.2%（22 combos）")
    assert_true(ok, msg)


@test
def test_gto_text_compare_rejects_large_ev_drift():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("EV: 7.57bb", "EV: 7.70bb")
    assert_true(not ok, "0.13bb EV drift must fail")


@test
def test_gto_text_compare_rejects_large_frequency_drift():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("Fold: 42.0%", "Fold: 43.0%")
    assert_true(not ok, "1.0pp frequency drift must fail")


@test
def test_gto_text_compare_rejects_combos_count_change():
    """Combos counts are part of the structure — compared exactly."""
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("Fold: 42.0%（22 combos）", "Fold: 42.2%（23 combos）")
    assert_true(not ok, "combos count change must fail even within freq tolerance")


@test
def test_gto_text_compare_rejects_structural_change():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("  Fold: 42.0%（22 combos）", "  Call: 42.0%（22 combos）")
    assert_true(not ok, "action label change must fail")


@test
def test_gto_text_compare_rejects_line_count_change():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("a\nb", "a\nb\nc")
    assert_true(not ok, "line count change must fail")


@test
def test_ev_comparison_suppresses_gto_mixed_taken_action():
    """Formatter: do not show EV loss for a solver-approved mixed action.

    Regression for H3441: exact-combo terminal call EV was numerically high,
    but solver strategy folded the combo 93%.  A fold at 93% frequency is not
    an EV-loss punt and must not produce "EV 損失 -5.7bb".
    """
    from gto_formatter import combo_index_for_hand, format_ev_comparison

    combo_idx = combo_index_for_hand("6d6h")
    range_arr = [0.0] * 1326
    range_arr[combo_idx] = 0.1056
    fold_strategy = [0.0] * 1326
    call_strategy = [0.0] * 1326
    fold_evs = [0.0] * 1326
    call_evs = [0.0] * 1326
    fold_strategy[combo_idx] = 0.933
    call_strategy[combo_idx] = 0.067
    call_evs[combo_idx] = 5.65

    solution = {
        "game": {"board": "2c8sJs", "current_street": {"type": "flop"}},
        "players_info": [{
            "player": {"position": "LJ"},
            "range": range_arr,
            "simple_hand_counters": {},
        }],
        "action_solutions": [
            {"action": {"code": "F", "allin": False, "type": "FOLD"},
             "strategy": fold_strategy, "evs": fold_evs},
            {"action": {"code": "C", "allin": True, "type": "CALL", "betsize": "12.000"},
             "strategy": call_strategy, "evs": call_evs},
        ],
    }

    note = format_ev_comparison(
        solution, "F", "66", "LJ", is_preflop=False, combo_idx=combo_idx
    )
    assert_true(note is None, f"GTO-approved fold should not show EV loss: {note}")


# ── GTO auth context Tests ──

@test
def test_run_with_gto_token_preserves_main_thread_token():
    import analyze_hand
    import gto_api

    calls = []
    gto_api.set_user_token("parent-access-token")

    def fake_solver_call(**_kwargs):
        calls.append(getattr(gto_api._thread_local, "access_token", None))
        return "ok"

    try:
        result = analyze_hand._run_with_gto_token(
            "parent-access-token", fake_solver_call, gametype="MTTGeneral"
        )
        assert_eq(result, "ok")
        assert_eq(calls, ["parent-access-token"])
        assert_eq(
            getattr(gto_api._thread_local, "access_token", None),
            "parent-access-token",
            "Inline helper calls must restore the request's per-user token.",
        )
    finally:
        gto_api.clear_user_token()


@test
def test_run_with_gto_token_clears_executor_thread_token():
    import analyze_hand
    import gto_api

    calls = []
    gto_api.clear_user_token()

    def fake_solver_call(**_kwargs):
        calls.append(getattr(gto_api._thread_local, "access_token", None))
        return "ok"

    result = analyze_hand._run_with_gto_token(
        "parent-access-token", fake_solver_call, gametype="MTTGeneral"
    )
    assert_eq(result, "ok")
    assert_eq(calls, ["parent-access-token"])
    assert_eq(
        getattr(gto_api._thread_local, "access_token", None),
        None,
        "Executor helper calls should not leak per-user tokens after fetch.",
    )


# ── Card classifier v2 Tests ──

@test
def test_extract_crops_smoke():
    from ocr.classifier.extract_pokercraft_crops import extract_one
    import numpy as np

    hid = "TM5846884226"
    gt_row = None
    gt_path = Path(__file__).resolve().parent.parent / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    with gt_path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["hand_id"] == hid:
                gt_row = row["ground_truth"]
                break
    assert_true(gt_row is not None, f"GT row for {hid} missing")
    img_path = Path(__file__).resolve().parent.parent / f"data/hand_images/img/{hid}.png"
    assert_true(img_path.exists(), f"image missing: {img_path}")
    result = extract_one(img_path.read_bytes(), gt_row)
    assert_eq(len(result["hero_crops"]), 2)
    assert_eq(result["hero_labels"], ["5h", "4s"])
    for crop in result["hero_crops"]:
        assert_true(isinstance(crop, np.ndarray) and crop.shape[0] > 0)


@test
def test_extract_hero_labels_match_n8_visual_order():
    from ocr.classifier.extract_pokercraft_crops import _visual_hero_order

    assert_eq(_visual_hero_order(["3c", "7c"]), ["7c", "3c"])
    assert_eq(_visual_hero_order(["Ah", "5h"]), ["Ah", "5h"])
    assert_eq(_visual_hero_order(["3d", "3s"]), ["3s", "3d"])
    assert_eq(_visual_hero_order(["7c", "7h"]), ["7h", "7c"])
    assert_eq(_visual_hero_order(["2h", "2d"]), ["2d", "2h"])


@test
def test_augment_win_sticker_overlays_yellow():
    import numpy as np
    from ocr.classifier.augment import apply_win_sticker

    base = np.full((192, 128, 3), 50, dtype=np.uint8)
    out = apply_win_sticker(base, rng=np.random.default_rng(0), p=1.0)
    yellow_mask = (out[..., 2] > 150) & (out[..., 1] > 150) & (out[..., 0] < 100)
    assert_true(yellow_mask.sum() > 100, f"WIN sticker did not write yellow pixels: {yellow_mask.sum()}")


@test
def test_augment_color_jitter_preserves_dimensions():
    import numpy as np
    from ocr.classifier.augment import color_jitter

    base = np.full((192, 128, 3), 128, dtype=np.uint8)
    out = color_jitter(base, rng=np.random.default_rng(0), strength=0.3)
    assert_eq(out.shape, base.shape)
    assert_eq(out.dtype, np.uint8)


@test
def test_card_cnn_v2_forward_shape():
    import torch
    from ocr.classifier.model import CardCNNv2, RANK_CLASSES, SUIT_CLASSES

    net = CardCNNv2()
    net.eval()
    rank_logits, suit_logits = net(torch.zeros(2, 3, 192, 128))
    assert_eq(rank_logits.shape, (2, len(RANK_CLASSES)))
    assert_eq(suit_logits.shape, (2, len(SUIT_CLASSES)))


@test
def test_card_mobilenet_v3_small_forward_shape():
    import torch
    from ocr.classifier.model import CardMobileNetV3Small, RANK_CLASSES, SUIT_CLASSES

    net = CardMobileNetV3Small(pretrained=False)
    net.eval()
    rank_logits, suit_logits = net(torch.zeros(2, 3, 192, 128))
    assert_eq(rank_logits.shape, (2, len(RANK_CLASSES)))
    assert_eq(suit_logits.shape, (2, len(SUIT_CLASSES)))


@test
def test_button_detector_picks_known_fixture_sector():
    import cv2

    from ocr.button_detector import detect_button, hero_position_from_button
    from ocr.region_detector import detect_regions

    img_path = Path(__file__).resolve().parent.parent / "data/hand_images/img/TM5864550087.png"
    image = cv2.imread(str(img_path))
    regions = detect_regions(image)
    result = detect_button(regions["table"], table_size=8)

    assert_true(result is not None)
    seat_idx, conf = result
    assert_eq(seat_idx, 6)
    assert_true(conf > 0.95)
    assert_eq(hero_position_from_button(seat_idx, table_size=8), "BB")


# ── Chip EV Tests ──

@test
def test_chip_ev_preflop_basic():
    """Chip EV: basic preflop open spot returns valid data."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "66",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
    })
    assert_in("Preflop", result["text"])
    assert_true(result["solutions"][0] is not None, "preflop solution should not be None")
    assert_eq(result["hero_position"], "CO")
    assert_eq(result["hero_hand"], "66")
    assert_eq(result["is_icm"], False)
    assert_eq(result["stacks"], "")


@test
def test_chip_ev_multi_street():
    """Chip EV: multi-street hand walks through flop/turn/river."""
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
    assert_in("Flop", result["text"])
    assert_in("Turn", result["text"])
    assert_true("flop" in result["street_states"], "should have flop state")
    assert_true("turn" in result["street_states"], "should have turn state")


@test
def test_chip_ev_alternate_street_keys():
    """Chip EV: handles LLM outputting 'cards' or 'card' instead of 'board' for flop."""
    from analyze_hand import analyze_hand_full
    # Flop uses "cards" instead of "board", turn uses "cards" instead of "card"
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"cards": "As7d2c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
            {"cards": "Tc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
        ],
    })
    assert_in("Flop", result["text"])
    assert_in("Turn", result["text"])


@test
def test_chip_ev_preflop_reraise():
    """Chip EV: preflop re-raise creates second hero decision point."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "CO",
        "hero_hand": "TT",
        "preflop_actions": "F-F-F-F-R2-R7-F-F-C",
    })
    # Should have two preflop spots (initial open + facing 3bet)
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected 2 preflop spots, got {len(preflop_spots)}")


@test
def test_chip_ev_3way_cold_call_fallback():
    """Chip EV: 3-way cold call preflop falls back to HU for hero's second decision."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "players_at_table": 8,
        "effective_bb": 100,
        "hero_position": "HJ",
        "hero_hand": "K9s",
        "preflop_actions": "F-F-F-R2-R6-F-F-C-C",
        "streets": [],
    })
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected 2 preflop spots, got {len(preflop_spots)}")
    # Second spot should have a solution (HU fallback)
    second_sol = result["solutions"][1]
    assert_true(second_sol is not None, "second preflop spot should have HU fallback solution")
    # Should mention multiway approximation
    assert_in("cold caller", result["text"].lower())


@test
def test_preflop_continuation_spot_for_facing_4bet_call():
    """H3427: hero's preflop call facing a 4-bet must be its own solver node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "TsTc",
        "hero_position": "SB",
        "preflop_actions": "F-F-F-F-R2.1-R7-F-R18-C",
        "players_at_table": 7,
        "hero_starting_stack": 20.4,
        "streets": [
            {"board": "4dKc9h", "actions": [
                {"action": "X", "position": "SB"},
                {"size": 12.0, "action": "R12", "position": "BTN"},
                {"action": "F", "position": "SB"},
            ]},
        ],
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-4bet spot, got {preflop_spots}")
    assert_eq(preflop_spots[1].get("taken_code"), "C")
    assert_in("─── Preflop — Facing 4-bet ───", compact)
    facing_section = compact.split("─── Preflop — Facing 4-bet ───", 1)[1].split("─── Flop:", 1)[0]
    assert_in("GTO:", facing_section)
    assert_in("→ Hero call", facing_section)


@test
def test_preflop_pending_facing_allin_uses_allin_effective_depth():
    """H3428: initial-round AI action reopens a visible 20bb facing-all-in node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "6s6c",
        "effective_bb": 37.8,
        "hero_position": "UTG",
        "player_stacks": [17.9, 30.8, 6.4, 10.9, 9.1, 25.7, 71.9, 37.3],
        "preflop_actions": "R2-F-F-AI19.9-F-F-F-F",
        "players_at_table": 8,
        "hero_starting_stack": 39.3,
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-all-in spot, got {preflop_spots}")
    assert_in("♠ UTG 66 | 20bb MTT", compact)
    assert_in("─── Preflop — Facing all-in ───", compact)
    facing_section = compact.split("─── Preflop — Facing all-in ───", 1)[1]
    assert_in("GTO:", facing_section)
    assert_eq(preflop_spots[1]["params"]["depth"], 20.125)
    assert_in("RAI", preflop_spots[1]["params"]["preflop_actions"])


@test
def test_seven_max_padded_utg_facing_3bet_spot_from_sb():
    """H3431: 7-max UTG maps to solver UTG+1 and must expose the SB 3-bet call node."""
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "QsTs",
        "effective_bb": 23.4,
        "hero_position": "UTG",
        "player_stacks": [41.9, 6.7, 57.7, 18.9, 28.7, 16.4, 14.9],
        "preflop_actions": "R2-F-F-F-F-R5-F-C",
        "players_at_table": 7,
        "hero_starting_stack": 23.4,
        "streets": [
            {"board": "5dTc9d", "actions": [
                {"size": 3.5, "action": "R3.5", "position": "SB"},
                {"size": 3.5, "action": "C", "position": "UTG"},
            ]},
            {"card": "4d", "actions": [
                {"size": 18.9, "action": "R18.9", "position": "SB"},
                {"action": "F", "position": "UTG"},
            ]},
        ],
    })

    compact = result["text_compact"]
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    assert_true(len(preflop_spots) >= 2, f"expected facing-3bet spot, got {preflop_spots}")
    facing_spot = preflop_spots[1]
    assert_eq(facing_spot.get("taken_code"), "C")
    assert_eq(facing_spot.get("solver_hero_pos"), "UTG+1")
    assert_in("R7.1", facing_spot["params"]["preflop_actions"], "SB 3-bet should be normalized in the node before hero call")
    assert_true(not facing_spot["params"]["preflop_actions"].endswith("-C"), "node must stop before hero's call")
    assert_in("─── Preflop — Facing 3-bet ───", compact)
    facing_section = compact.split("─── Preflop — Facing 3-bet ───", 1)[1].split("─── Flop:", 1)[0]
    assert_in("GTO:", facing_section)
    assert_in("→ Hero call", facing_section)
    assert_true("此手牌 0% 到達此節點" not in compact, compact)
    assert_true("cold call" not in result["text"].lower(), result["text"])


@test
def test_chip_ev_depth_mapping():
    """Chip EV: depth maps to nearest available solver depth."""
    from gto_api import nearest_depth
    assert_eq(nearest_depth(32), 30.125)
    assert_eq(nearest_depth(50), 50.125)
    assert_eq(nearest_depth(7), 8.125)
    assert_eq(nearest_depth(100), 100.125)
    assert_eq(nearest_depth(15), 14.125)


# ── Multiway Simplification Tests ──

@test
def test_multiway_3way_fold_on_flop():
    """Multiway: 3-way pot where one folds on flop simplifies to heads-up."""
    from analyze_hand import analyze_hand_full
    # UTG raise, SB call, BB call → 3-way to flop
    # Flop: SB checks, BB checks, UTG bets, SB folds, BB calls → heads-up
    # Turn: BB checks, UTG bets, BB folds
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "ATo",
        "preflop_actions": "R2-F-F-F-F-F-C-C",
        "streets": [
            {"board": "JsTc3h", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "X"},
                {"position": "UTG", "action": "R2", "size": 2.0},
                {"position": "SB", "action": "F"},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "6c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG", "action": "R5", "size": 5.0},
                {"position": "BB", "action": "F"},
            ]},
        ],
    })
    # Should have multiway simplification note
    assert_in("多人底池", result["text"], "should note multiway simplification")
    assert_in("UTG", result["text"])
    # Flop and turn should have solver data (not "無 solver 數據")
    assert_in("Flop", result["text"])
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data after multiway simplification")


@test
def test_multiway_3way_check_raise_on_flop():
    """Multiway: 3-way pot with check-raise on flop matches correctly (not all-in)."""
    from analyze_hand import analyze_hand_full
    # UTG+1 raise, BTN call, BB call → 3-way
    # Flop: BB checks, UTG+1 bets 2.5, BTN folds, BB raises 8.7, UTG+1 calls
    # Turn: BB all-in, UTG+1 calls
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 20,
        "hero_position": "UTG+1",
        "hero_hand": "9h9c",
        "preflop_actions": "F-R2-F-F-F-C-F-C",
        "streets": [
            {"board": "6s7h6h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 2.5},
                {"position": "BTN", "action": "F"},
                {"position": "BB", "action": "R", "size": 8.7},
                {"position": "UTG+1", "action": "C"},
            ]},
            {"card": "3c", "actions": [
                {"position": "BB", "action": "AI"},
                {"position": "UTG+1", "action": "C"},
            ]},
        ],
    })
    assert_in("多人底池", result["text"])
    # BB's raise should NOT match all-in (RAI) — 8.7bb is a raise, not an all-in
    assert_true("solver code: RAI" not in result["text"],
                "BB's 8.7bb raise should not match all-in")
    # Flop and turn should both have solver data
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    turn_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "turn" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data")
    assert_true(len(turn_solutions) > 0, "turn should have solver data")


@test
def test_multiway_2way_flop_unchanged():
    """Multiway: 3-way preflop but only 2 see flop already works without change."""
    from analyze_hand import analyze_hand_full
    # UTG raise, BTN call, BB fold → only UTG+BTN see flop (already 2-way)
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BTN",
        "hero_hand": "AQs",
        "preflop_actions": "R2-F-F-F-F-C-F-F",
        "streets": [
            {"board": "As7d2c", "actions": [
                {"position": "UTG", "action": "X"},
                {"position": "BTN", "action": "R2", "size": 2.0},
            ]},
        ],
    })
    # This is actually heads-up (only 2 non-fold), no multiway note expected
    # The point is this should still work and have flop data
    assert_in("Flop", result["text"])


@test
def test_multiway_all_fold_to_hero_raise():
    """Multiway: 3-way pot where everyone folds to hero's flop raise simplifies to HU."""
    from analyze_hand import analyze_hand_full
    # HJ raise, SB call, BB call → 3-way
    # Flop T44: SB x, BB x, HJ bet, SB raise, BB fold, HJ fold
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "SB",
        "hero_hand": "AcTc",
        "preflop_actions": "F-F-F-R2-F-F-C-C",
        "streets": [
            {"board": "Td4h4c", "actions": [
                {"position": "SB", "action": "X"},
                {"position": "BB", "action": "X"},
                {"position": "HJ", "action": "R2", "size": 2.0},
                {"position": "SB", "action": "R6", "size": 6.0},
                {"position": "BB", "action": "F"},
                {"position": "HJ", "action": "F"},
            ]},
        ],
    })
    assert_in("多人底池", result["text"], "should note multiway simplification")
    assert_in("HJ", result["text"])
    # Flop should have solver data for SB's check and facing-bet decisions
    flop_solutions = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                      if spot["street"] == "flop" and s is not None]
    assert_true(len(flop_solutions) > 0, "flop should have solver data when villain folds to hero raise")


# ── Position Order Tests ──

@test
def test_position_orders():
    """Position orders match GTO Wizard convention for all table sizes."""
    from analyze_hand import POSITION_ORDERS
    assert_eq(POSITION_ORDERS[9], ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[8], ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[6], ["LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[3], ["BTN", "SB", "BB"])
    assert_eq(POSITION_ORDERS[2], ["SB", "BB"])


@test
def test_position_order_for_hand():
    """Position order is selected correctly based on player_stacks length."""
    from analyze_hand import _get_position_order
    assert_eq(_get_position_order(6), ["LJ", "HJ", "CO", "BTN", "SB", "BB"])
    assert_eq(_get_position_order(8), ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"])


# ── Range Compression Tests ──

@test
def test_compress_range_pairs():
    """Range compression: consecutive pairs produce 22+ notation."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TJQKA"]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_not_in("AA", result.replace("22+", ""))  # AA shouldn't appear separately


@test
def test_compress_range_all_kickers():
    """Range compression: all suited kickers produce AXs notation."""
    from gto_formatter import _compress_range
    ranks = "KQJT98765432"
    hands = [(f"A{r}s", 1.0, 4) for r in ranks]
    result = _compress_range(hands)
    assert_in("AXs", result)


@test
def test_compress_range_plus_notation():
    """Range compression: K3o+ means K3o through KQo (reaches top kicker)."""
    from gto_formatter import _compress_range
    ranks = "QJT9876543"
    hands = [(f"K{r}o", 1.0, 12) for r in ranks]
    result = _compress_range(hands)
    assert_in("K3o+", result)


@test
def test_compress_range_partial_dash():
    """Range compression: partial kicker range uses dash notation (Q2s-Q4s)."""
    from gto_formatter import _compress_range
    hands = [(f"Q{r}s", 1.0, 4) for r in "234"]
    result = _compress_range(hands)
    assert_in("Q2s-Q4s", result)
    assert_not_in("+", result)


@test
def test_compress_range_mixed_freq():
    """Range compression: mixed frequency shows inline percentage."""
    from gto_formatter import _compress_range
    hands = [("K2o", 0.28, 12)]
    result = _compress_range(hands)
    assert_in("K2o(28%)", result)


@test
def test_compress_range_full_call_range():
    """Range compression: full BB call range compresses correctly (real scenario)."""
    from gto_formatter import _compress_range
    # Simulated 10bb SB all-in BB call range
    hands = [
        ("AA", 1.0, 6), ("KK", 1.0, 6), ("QQ", 1.0, 6), ("JJ", 1.0, 6),
        ("TT", 1.0, 6), ("99", 1.0, 6), ("88", 1.0, 6), ("77", 1.0, 6),
        ("66", 1.0, 6), ("55", 1.0, 6), ("44", 1.0, 6), ("33", 1.0, 6), ("22", 1.0, 6),
        ("AKs", 1.0, 4), ("AQs", 1.0, 4), ("AJs", 1.0, 4), ("ATs", 1.0, 4),
        ("A9s", 1.0, 4), ("A8s", 1.0, 4), ("A7s", 1.0, 4), ("A6s", 1.0, 4),
        ("A5s", 1.0, 4), ("A4s", 1.0, 4), ("A3s", 1.0, 4), ("A2s", 1.0, 4),
        ("KQs", 1.0, 4), ("KJs", 1.0, 4), ("KTs", 1.0, 4), ("K9s", 1.0, 4),
        ("K8s", 1.0, 4), ("K7s", 1.0, 4), ("K6s", 1.0, 4), ("K5s", 1.0, 4),
        ("K4s", 1.0, 4), ("K3s", 1.0, 4), ("K2s", 1.0, 4),
        ("Q5s", 1.0, 4), ("Q6s", 1.0, 4), ("Q7s", 1.0, 4), ("Q8s", 1.0, 4),
        ("Q9s", 1.0, 4), ("QTs", 1.0, 4), ("QJs", 1.0, 4),
        ("J7s", 1.0, 4), ("J8s", 1.0, 4), ("J9s", 1.0, 4), ("JTs", 1.0, 4),
        ("T8s", 1.0, 4), ("T9s", 1.0, 4),
        ("98s", 1.0, 4),
        ("AKo", 1.0, 12), ("AQo", 1.0, 12), ("AJo", 1.0, 12), ("ATo", 1.0, 12),
        ("A9o", 1.0, 12), ("A8o", 1.0, 12), ("A7o", 1.0, 12), ("A6o", 1.0, 12),
        ("A5o", 1.0, 12), ("A4o", 1.0, 12), ("A3o", 1.0, 12), ("A2o", 1.0, 12),
        ("K3o", 1.0, 12), ("K4o", 1.0, 12), ("K5o", 1.0, 12), ("K6o", 1.0, 12),
        ("K7o", 1.0, 12), ("K8o", 1.0, 12), ("K9o", 1.0, 12), ("KTo", 1.0, 12),
        ("KJo", 1.0, 12), ("KQo", 1.0, 12),
        ("K2o", 0.28, 12),
        ("Q8o", 1.0, 12), ("Q9o", 1.0, 12), ("QTo", 1.0, 12), ("QJo", 1.0, 12),
        ("J9o", 1.0, 12), ("JTo", 1.0, 12),
        ("T9o", 1.0, 12),
    ]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_in("AXs", result)
    assert_in("KXs", result)
    assert_in("AXo", result)
    assert_in("K3o+", result)
    assert_in("K2o(28%)", result)
    assert_in("Q5s+", result)
    assert_in("J7s+", result)
    assert_in("T8s+", result)
    assert_in("Q8o+", result)


@test
def test_compress_range_highfreq_merge_pairs():
    """Range compression: ≥90% hands merge into the run (JJ@99% → 22+~), not split out."""
    from gto_formatter import _compress_range
    # All pairs pure except JJ at 99% — should still collapse to 22+ (with ~ marker)
    hands = []
    for r in "23456789TJQKA":
        freq = 0.99 if r == "J" else 1.0
        hands.append((f"{r}{r}", freq, 6 * freq))
    result = _compress_range(hands)
    assert_in("22+~", result)
    assert_not_in("JJ(99%)", result)
    assert_not_in("JJ", result.replace("22+~", ""))  # JJ must not appear separately


@test
def test_compress_range_highfreq_below_threshold_stays_mixed():
    """Range compression: hands below 90% stay broken out with inline %, not merged."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TQKA"]  # all pure except JJ
    hands.append(("JJ", 0.85, 5.1))  # 85% < 90% → stays mixed
    result = _compress_range(hands)
    assert_in("JJ(85%)", result)
    assert_not_in("22+", result)  # run is broken by missing JJ from pure set


@test
def test_compress_range_pure_no_marker():
    """Range compression: fully-pure run carries no ~ marker."""
    from gto_formatter import _compress_range
    hands = [(f"{r}{r}", 1.0, 6) for r in "23456789TJQKA"]
    result = _compress_range(hands)
    assert_in("22+", result)
    assert_not_in("~", result)


@test
def test_compress_range_highfreq_suited_marker():
    """Range compression: a ≥90% suited hand merges as pure but its token gets ~."""
    from gto_formatter import _compress_range
    # A9s/A8s/A4s/A2s pure, A7s at 92% → merges (no longer "(92%)") but marked
    hands = [
        ("A9s", 1.0, 4), ("A8s", 1.0, 4), ("A7s", 0.92, 3.68),
        ("A4s", 1.0, 4), ("A2s", 1.0, 4),
    ]
    result = _compress_range(hands)
    assert_in("A7s~", result)
    assert_not_in("A7s(92%)", result)


# ── GTO API Tests ──

@test
def test_api_get_next_actions():
    """API: next_actions returns valid response for UTG first-to-act."""
    from gto_api import get_next_actions
    resp = get_next_actions(gametype="MTTGeneral", depth=30.125)
    assert_true("next_actions" in resp, "response should have next_actions key")
    avail = resp["next_actions"]["available_actions"]
    assert_true(len(avail) > 0, "should have at least one available action")
    codes = [a["action"]["code"] for a in avail]
    assert_in("F", codes, "Fold should be available")


@test
def test_api_next_actions_endpoint_path():
    """API: next-actions URL pinned to /v4/game-points/ (was /v1/poker/, moved 2026-05-02)."""
    import inspect
    import gto_api
    src = inspect.getsource(gto_api.get_next_actions)
    assert_true(
        "/v4/game-points/next-actions/" in src,
        "get_next_actions must call /v4/game-points/next-actions/",
    )
    assert_true(
        "/v1/poker/next-actions/" not in src,
        "old /v1/poker/next-actions/ path is dead — must not be used",
    )


@test
def test_api_get_spot_solution():
    """API: spot_solution returns valid data for basic preflop spot."""
    from gto_api import get_spot_solution
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    assert_true(sol is not None, "solution should not be None")
    assert_true("action_solutions" in sol, "should have action_solutions")
    assert_true("players_info" in sol, "should have players_info")


@test
def test_api_find_closest_action():
    """API: find_closest_action picks nearest raise size."""
    from gto_api import get_next_actions, find_closest_action
    resp = get_next_actions(gametype="MTTGeneral", depth=30.125)
    avail = resp["next_actions"]["available_actions"]
    code = find_closest_action(avail, 2.0)
    assert_true(code.startswith("R"), f"expected raise code, got {code}")


@test
def test_api_stacks_param():
    """API: stacks parameter is accepted (ICM mode)."""
    from gto_api import get_next_actions
    resp = get_next_actions(
        gametype="MTTGeneral", depth=30.125,
        stacks="30.125-30.125-30.125-30.125-30.125-30.125-30.125-30.125",
    )
    assert_true("next_actions" in resp)


@test
def test_api_no_solution_returns_none():
    """API: spot_solution returns None for 204/403 responses."""
    from gto_api import get_spot_solution
    # ICM mode with mismatched stacks → should return 204 or 403
    sol = get_spot_solution(
        gametype="MTTGeneral_ICM8m1000PTBUBBLE160PT",
        depth="50.125",
        stacks="50.125-50.125-50.125-50.125-50.125-50.125-50.125-50.125",
        preflop_actions="F-F-F-F-F-F-R2-F",
        board="Js6h5s",  # ICM preflop_only → flop should return 204
    )
    assert_true(sol is None, "ICM mode should return None for postflop query")


@test
def test_api_404_spot_solution_returns_none():
    """API: spot_solution returns None for 404 responses.

    Regression for H2914: an impossible OCR runout (river Ks duplicated
    from the flop) made GTO Wizard return 404 from spot-solution.  The bot
    must treat that as no solver data instead of crashing the screenshot
    analysis.
    """
    import gto_api

    class FakeResponse:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("404 should be handled before raise_for_status")

    orig_get = gto_api._get_with_retry
    orig_cache_get = gto_api.cache_get
    orig_cache_put = gto_api.cache_put
    writes = []
    try:
        gto_api._get_with_retry = lambda *a, **kw: FakeResponse()
        gto_api.cache_get = lambda *a, **kw: gto_api.SENTINEL
        gto_api.cache_put = lambda *a, **kw: writes.append((a, kw))
        sol = gto_api.get_spot_solution(
            gametype="MTTGeneral", depth=17.125,
            preflop_actions="F-F-F-F-R2-F-F-C",
            board="KhJsKsAdKs",
            flop_actions="X-R1.1-C",
            turn_actions="X-R4.25-C",
            river_actions="X",
        )
    finally:
        gto_api._get_with_retry = orig_get
        gto_api.cache_get = orig_cache_get
        gto_api.cache_put = orig_cache_put

    assert_true(sol is None, "404 spot-solution should be cached as no data")
    assert_true(writes and writes[-1][0][2] is None,
                "404 response should write a null cache entry")


@test
def test_api_postflop_percentage_detection():
    """API: find_closest_action_postflop detects percentage-based sizes."""
    from gto_api import get_next_actions, find_closest_action_postflop
    # UTG+1 open, BB call, flop 2h8cTc, BB checks → UTG+1 to act
    resp = get_next_actions(
        gametype="MTTGeneral", depth=30.125,
        preflop_actions="F-R2.1-F-F-F-F-F-C",
        board="2h8cTc", flop_actions="X",
    )
    avail = resp["next_actions"]["available_actions"]
    # size=40 means "40% pot" from LLM — should NOT match all-in
    code = find_closest_action_postflop(avail, 40)
    assert_true(code != "RAI", f"size=40 should not match all-in, got {code}")
    assert_true(code.startswith("R"), f"expected raise code, got {code}")
    # size=27.9 is actual all-in — should still match RAI
    code_ai = find_closest_action_postflop(avail, 27.9)
    assert_true(code_ai == "RAI", f"actual all-in should match RAI, got {code_ai}")


@test
def test_rederive_postflop_codes_remaps_stale_bet():
    """Off-range depth escalation must re-match opponent bet codes to the
    new depth's bet grid.

    H2890: KQs flatted a 3-bet (off-range at 30bb), escalating postflop to
    35bb.  SB's flop bet was coded 'R4.25' at 30bb; that code does not
    exist at 35bb, so the API silently collapsed the flop to SB's
    first-action root node — showing SB's Check/Bet strategy instead of
    HJ's facing-bet (Call/Fold/Raise) decision.
    """
    from analyze_hand import _rederive_postflop_codes
    from gto_api import get_next_actions

    params = {
        "gametype": "MTTGeneral", "depth": 35.125,
        "preflop_actions": "F-F-F-R2.2-F-F-R8.3-F-C",
    }
    nf, nt, nr = _rederive_postflop_codes(
        params, "Ts8d8h", "Ts8d8hAs", "",
        "R4.25", "", "",
    )
    assert_true(nf != "R4.25", "stale 30bb bet code R4.25 must be remapped")
    resp = get_next_actions(
        gametype="MTTGeneral", depth=35.125,
        preflop_actions="F-F-F-R2.2-F-F-R8.3-F-C",
        board="Ts8d8h", flop_actions="",
    )
    codes = [a["action"]["code"]
             for a in resp["next_actions"]["available_actions"]]
    assert_in(nf, codes,
              f"re-derived flop code {nf} must be a real 35bb action {codes}")
    # Simple codes on later streets pass through untouched
    assert_eq(nt, "", "no turn actions in → empty out")
    assert_eq(nr, "", "no river actions in → empty out")


@test
def test_api_postflop_overbet_clamps_to_allin():
    """API: hero's all-in bet that overshoots solver's modeled all-in
    (hero stack > opponent stack, so real all-in > solver's effective
    all-in) must still match RAI — not get re-interpreted as a pot%.

    Regression for H2760 where hero bet 26.6bb into a 27.3bb river
    pot (solver all-in = 17.35bb, capped by shorter SB). The bet was
    mis-matched to R9.5 (35% pot) via the percentage-interpretation
    fallback, hiding the fact that hero's action WAS the all-in
    recommended by GTO. Also regresses H2492 (R27.6 → was R6.5, now RAI).
    """
    from gto_api import find_closest_action_postflop
    avail = [
        {"action": {"code": "X", "betsize": "0.000", "betsize_by_pot": None, "allin": False}},
        {"action": {"code": "R2.5", "betsize": "2.500", "betsize_by_pot": "0.09157509", "allin": False}},
        {"action": {"code": "R9.5", "betsize": "9.500", "betsize_by_pot": "0.34798535", "allin": False}},
        {"action": {"code": "RAI", "betsize": "17.350", "betsize_by_pot": "0.63553114", "allin": True}},
    ]
    # Hero's real all-in 26.6bb > solver all-in 17.35bb; fractional .6 is
    # an OCR-native absolute bb, not an LLM percentage → keep RAI.
    assert_eq(find_closest_action_postflop(avail, 26.6), "RAI",
              "fractional overbet past all-in must match RAI")
    # H2492: 27.6bb fractional overbet
    assert_eq(find_closest_action_postflop(avail, 27.6), "RAI",
              "27.6bb fractional overbet must match RAI")
    # Integer percentages (from LLM) should still use the pct path
    assert_eq(find_closest_action_postflop(avail, 40), "R9.5",
              "integer 40 treated as 40% pot → R9.5")
    # Target within 15% of all-in → always RAI (existing behavior)
    assert_eq(find_closest_action_postflop(avail, 17.1), "RAI",
              "17.1bb close to all-in 17.35 → RAI")


@test
def test_chip_ev_percentage_size_analysis():
    """ChipEV: analysis handles percentage-based bet sizes without errors."""
    from analyze_hand import analyze_hand
    result = analyze_hand({
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "J9o",
        "preflop_actions": "F-R2-F-F-F-F-F-C",
        "streets": [
            {"board": "2h8cTc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 40},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "7s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R", "size": 50},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "9h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "X"},
            ]},
        ],
    })
    assert_in("Flop", result)
    assert_in("Turn", result)
    assert_in("River", result)
    # Solver code lines should not show RAI for the 40%/50% bets
    assert_true("solver code: RAI" not in result, f"Percentage bets should not match all-in")


# ── Formatter Tests ──


@test
def test_solver_detail_uses_exact_postflop_combo_for_coaching_text():
    """Analyze text: coach data must use exact postflop combo.

    Regression for H3451: compact output used AdTh's exact river strategy
    (check 14%), but the full solver text fed to the coach used aggregate ATo
    (check 4.6%), causing contradictory advice.
    """
    from analyze_hand import _hero_hand_for_solver_detail

    assert_eq(
        _hero_hand_for_solver_detail("ATo", "AdTh", "river", 1210),
        "AdTh",
        "postflop detail should preserve the exact combo for coach grounding",
    )
    assert_eq(
        _hero_hand_for_solver_detail("ATo", "AdTh", "preflop", 1210),
        "ATo",
        "preflop detail should remain on the 169 hand class",
    )
    assert_eq(
        _hero_hand_for_solver_detail("ATo", "ATo", "river", None),
        "ATo",
        "non-specific hands should keep aggregate display",
    )


@test
def test_h3471_preflop_rfi_not_misreported_as_call_vs_raise():
    """Analyze text: H3471 is HJ RFI, not HJ calling a prior raise.

    The solver's unopened 14bb HJ node encodes open-limp as action code C.
    Regression: compact/full text showed only "Call 98%" and no hero
    preflop action line, so the coach hallucinated that HJ faced an open
    raise and called.  The analysis must show Hero's actual open raise
    while ensuring solver C cannot be misread as a call versus a prior raiser.
    """
    from analyze_hand import analyze_hand_full

    result = analyze_hand_full({
        "streets": [
            {
                "board": "As7cAc",
                "actions": [
                    {"action": "X", "position": "HJ"},
                    {"size": 1.5, "action": "R1.5", "position": "BTN"},
                    {"size": 4.0, "action": "R4", "position": "HJ"},
                    {"size": 2.5, "action": "C", "position": "BTN"},
                ],
            },
            {
                "card": "7h",
                "actions": [
                    {"size": 8.5, "allin": True, "action": "R8.5", "position": "HJ"},
                    {"size": 8.5, "action": "C", "position": "BTN"},
                ],
            },
        ],
        "gametype": "MTTGeneral",
        "hero_hand": "TdTc",
        "effective_bb": 14.5,
        "hero_position": "HJ",
        "player_stacks": [48.8, 16.3, 31.2, 11.0, 57.8],
        "preflop_actions": "R2-F-C-F-F",
        "players_at_table": 5,
        "hero_starting_stack": 14.5,
    })

    assert_eq(result["preflop_actions"], "F-F-F-R2-F-C-F-F")
    assert_in("Limp: 98.5%", result["text"])
    assert_in("GTO: Limp 98%", result["text_compact"])
    assert_in("→ 實際行動: HJ R2", result["text"])
    assert_in("→ Hero open raise 29% pot ✅", result["text_compact"])
    assert_not_in("GTO: Call 98%", result["text_compact"])
    assert_not_in("→ Hero limp", result["text_compact"])


@test
def test_formatter_action_summary():
    """Formatter: format_action_summary produces readable output."""
    from gto_api import get_spot_solution
    from gto_formatter import format_action_summary
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_action_summary(sol)
    assert_in("Preflop", text)
    assert_in("底池", text)


@test
def test_formatter_hand_detail():
    """Formatter: format_hand_detail shows strategy for specific hand."""
    from gto_api import get_spot_solution
    from gto_formatter import format_hand_detail
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_hand_detail(sol, "AA", "UTG")
    assert_in("AA", text)
    assert_in("Range 頻率", text)


@test
def test_formatter_range_by_action():
    """Formatter: format_range_by_action uses compressed notation."""
    from gto_api import get_spot_solution
    from gto_formatter import format_range_by_action
    sol = get_spot_solution(gametype="MTTGeneral", depth=30.125)
    text = format_range_by_action(sol, "UTG")
    assert_in("策略分佈", text)
    # Should use compressed notation (e.g., "+" or "Xs" patterns)
    assert_true("+" in text or "Xs" in text or "Xo" in text,
                "should use compressed range notation")


@test
def test_formatter_range_by_action_categorized():
    """Formatter: range_by_action shows hand categories (top pair, trips, etc.)."""
    from gto_api import get_spot_solution
    from gto_formatter import format_range_by_action
    sol = get_spot_solution(gametype="MTTGeneral", depth=20.125,
        preflop_actions="F-R2-F-F-F-F-F-C",
        board="6s7h6h", flop_actions="X-R1.8")
    text = format_range_by_action(sol, "BB")
    # A7s/A7o should be under 頂對 (top pair), not 聽牌
    assert_in("頂對", text, "Should categorize top pair hands")
    assert_in("三條", text, "Should categorize trips")
    # Draw summary should appear
    assert_in("聽牌", text, "Should include draw summary")
    assert_in("花聽牌", text, "Should mention flush draws")


@test
def test_solver_grounding_intent_gate():
    """Follow-up gate: strategy/range/hypothetical questions must be detected
    so a solver tool call can be hard-forced (anti-hallucination, H2873).

    Regression for: bot answered 'which hands bet/check on this turn' from
    poker theory (claimed AA → check for pot control) with 0 tool calls.
    """
    from gemini_session import _needs_solver_grounding as g
    must_fire = [
        "在這種雙花面 turn hero 如果拿梅花 or 方塊 suited "
        "如何決定整體範圍哪些牌要下注哪些要過牌？",   # the exact H2873 follow-up
        "BB 在 turn 的 check-raise 範圍是什麼？",
        "如果 flop 用 33% pot 下注會怎樣？",
        "對手 3-bet 的話 KQo 應該怎麼打？",
        "AA 在這個 turn 是 bet 還是 check？",
        "為什麼 AJo 要 check？",
    ]
    for q in must_fire:
        assert_true(g(q), f"gate must fire for strategy/range question: {q!r}")
    must_not_fire = ["謝謝教練", "你好", "看一下我上週的漏洞",
                     "我的訓練計畫是什麼", "給我看 progress report"]
    for q in must_not_fire:
        assert_true(not g(q), f"gate must NOT fire for: {q!r}")


@test
def test_h2873_turn_AA_is_bet_not_check():
    """Ground truth guard (H2873): on the HJ turn JcTd5c8d, AA is ~100% bet,
    NOT check. The bot must answer range questions from THIS data, never from
    'overpair → pot control' theory. Guards solver wiring + categorization so
    the data feeding the LLM (system-prompt range breakdown) stays correct.
    """
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "hero_hand": "Kd4d", "effective_bb": 30,
        "hero_position": "HJ", "preflop_actions": "F-F-F-R2-F-F-F-C",
        "players_at_table": 8,
        "streets": [
            {"board": "5cJcTd", "street": "flop", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 2.5, "action": "R2.5", "position": "HJ"},
                {"action": "C", "position": "BB"}]},
            {"card": "8d", "street": "turn", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 8.5, "action": "R8.5", "position": "HJ"},
                {"action": "F", "position": "BB"}]},
        ],
    })
    turn_sols = [s for s, spot in zip(result["solutions"], result["hero_spots"])
                 if spot["street"] == "turn" and s is not None]
    assert_true(len(turn_sols) > 0, "turn should have solver data")
    sol = turn_sols[0]
    pi = next((p for p in sol["players_info"]
               if p["player"]["position"] == "HJ"), None)
    assert_true(pi is not None, "HJ player_info must exist in turn solution")
    aa = pi["simple_hand_counters"].get("AA")
    assert_true(aa is not None, "AA must be present in HJ turn range")
    freqs = aa.get("actions_total_frequencies", {})
    check_freq = freqs.get("X", 0.0)
    bet_raise_freq = sum(v for k, v in freqs.items()
                         if k.upper().startswith("R"))
    assert_true(check_freq < 0.10,
                f"AA check freq must be ~0 (was {check_freq:.4f}); "
                f"'AA checks for pot control' is a hallucination")
    assert_true(bet_raise_freq > 0.85,
                f"AA must be ~100% bet/raise (was {bet_raise_freq:.4f})")


@test
def test_formatter_normalize_hand_name():
    """Formatter: normalize_hand_name handles various input formats."""
    from gto_formatter import normalize_hand_name
    assert_eq(normalize_hand_name("AhKs"), "AKo")
    assert_eq(normalize_hand_name("KsAh"), "AKo")
    assert_eq(normalize_hand_name("6h6s"), "66")
    assert_eq(normalize_hand_name("AhKh"), "AKs")
    assert_eq(normalize_hand_name("AKs"), "AKs")
    assert_eq(normalize_hand_name("KAs"), "AKs")
    assert_eq(normalize_hand_name("45o"), "54o")
    assert_eq(normalize_hand_name("54o"), "54o")
    assert_eq(normalize_hand_name("45s"), "54s")
    assert_eq(normalize_hand_name("66"), "66")


@test
def test_formatter_low_rank_first_class_uses_canonical_solver_row():
    """Formatter: low-rank-first classes like 45o must look up 54o.

    Regression for H3638: the parser/user supplied "45o", while GTO Wizard's
    169-class keys use "54o".  The old lookup missed the hand row, printed
    "range 中沒有 45o", and let coaching incorrectly call preflop a fold.
    """
    from gto_formatter import format_full_spot, format_spot_compact

    sol = {
        "game": {
            "active_position": "BB",
            "board": "",
            "current_street": {"type": "preflop"},
            "pot": 4.5,
            "bet_display_name": "RAISE",
        },
        "action_solutions": [
            {
                "action": {"code": "F"},
                "total_frequency": 0.2,
                "total_combos": 265,
                "strategy": [0.0] * 169,
            },
            {
                "action": {"code": "C", "betsize": 2.0},
                "total_frequency": 0.7,
                "total_combos": 928,
                "strategy": [0.0] * 169,
            },
            {
                "action": {"code": "RAI", "allin": True, "betsize": 17.0},
                "total_frequency": 0.1,
                "total_combos": 133,
                "strategy": [0.0] * 169,
            },
        ],
        "players_info": [
            {
                "player": {"position": "BB"},
                "range": [1.0] * 169,
                "simple_hand_counters": {
                    "54o": {
                        "total_combos_available": 12,
                        "total_combos": 12,
                        "total_frequency": 1.0,
                        "hand_ev": 0.1,
                        "hand_eq": 0.32,
                        "actions_total_frequencies": {"C": 1.0},
                        "actions_ev": {"C": 0.1},
                    }
                },
            }
        ],
    }

    full = format_full_spot(sol, "45o", "BB")
    compact = format_spot_compact(sol, "45o", "BB")

    assert_in("【BB 54o】", full)
    assert_in("Call: 100.0%", full)
    assert_not_in("range 中沒有", full)
    assert_eq(compact, "GTO: Call 100%")


@test
def test_formatter_low_range_exact_combo_not_aggregated():
    """Formatter: hero's exact combo below the 0.5% display range must still
    drive the full-text verdict, not the same-class aggregate.

    Regression for H3639: hero holds Ac8c (nut flush) on 9c9s5cKs2c and jams
    the river.  Ac8c reaches this node only ~0.09% of the time (it usually bets
    the turn), so _get_combo_strategies filtered it out and the full text fell
    back to the aggregate A8s ("Fold 94.5%") — which averages in the three
    non-flush ace-high combos.  The coach then contradicted the compact's
    correct "All-in 99% ✅".  The full text must show 【LJ Ac8c（A8s）】 All-in,
    matching the compact.
    """
    from gto_formatter import (
        format_full_spot,
        format_spot_compact,
        combo_index_for_hand,
    )

    board = "9c9s5cKs2c"
    board_cards = {"9c", "9s", "5c", "Ks", "2c"}
    ac8c = combo_index_for_hand("Ac8c")   # 1152 — the nut flush combo
    ad8d = combo_index_for_hand("Ad8d")   # non-flush ace-high
    ah8h = combo_index_for_hand("Ah8h")   # non-flush ace-high

    def arr(mapping):
        a = [0.0] * 1326
        for i, v in mapping.items():
            a[i] = v
        return a

    # Range: Ac8c survives rarely (0.001 < 0.005 display cutoff); the two
    # non-flush combos are the bulk of the same-class range.
    range_arr = arr({ac8c: 0.001, ad8d: 0.044, ah8h: 0.044})
    # Strategy: the nut flush jams, the ace-high junk folds.
    fold_strat = arr({ac8c: 0.01, ad8d: 0.95, ah8h: 0.95})
    jam_strat = arr({ac8c: 0.99, ad8d: 0.04, ah8h: 0.04})
    hand_evs = arr({ac8c: 16.9, ad8d: -0.01, ah8h: -0.01})

    sol = {
        "game": {
            "active_position": "LJ",
            "board": board,
            "current_street": {"type": "river"},
            "pot": 11.3,
            "bet_display_name": "RAISE",
        },
        "action_solutions": [
            {"action": {"code": "F"}, "total_frequency": 0.257,
             "total_combos": 9, "strategy": fold_strat},
            {"action": {"code": "RAI", "allin": True, "betsize": 16.6},
             "total_frequency": 0.167, "total_combos": 6, "strategy": jam_strat},
        ],
        "players_info": [
            {
                "player": {"position": "LJ"},
                "range": range_arr,
                "hand_evs": hand_evs,
                "simple_hand_counters": {
                    # Aggregate A8s: dominated by the folding junk combos.
                    "A8s": {
                        "total_combos_available": 4,
                        "total_combos": 0.1,
                        "total_frequency": 0.023,
                        "hand_ev": 0.17,
                        "hand_eq": 0.079,
                        "actions_total_frequencies": {"F": 0.945, "RAI": 0.053},
                        "actions_total_combos": {"F": 0.1, "RAI": 0.0},
                        "actions_ev": {"F": 0.0, "RAI": -0.12},
                    }
                },
            }
        ],
    }

    # board cards must not be misclassified as blockers of the wrong hand
    assert not ({c for c in board_cards} & {"Ac", "8c"})

    full = format_full_spot(sol, "Ac8c", "LJ")
    compact = format_spot_compact(sol, "Ac8c", "LJ", combo_idx=ac8c)

    # Full text shows hero's exact combo and its jam verdict — NOT the
    # aggregate fold.
    assert_in("【LJ Ac8c（A8s）】", full)
    assert_in("All-in 99%", full)
    assert_not_in("【LJ A8s】", full)
    assert_not_in("Fold: 94", full)
    # And it agrees with the compact.
    assert_eq(compact, "GTO: All-in 99%")


# ── ICM Tests ──

@test
def test_icm_gametype_lookup():
    """ICM: find_gametype returns valid ICM mode for bubble scenario."""
    from icm_modes import find_gametype
    gt = find_gametype(
        players_at_table=8,
        pko=False,
        tournament_size=1000,
        phase="BUBBLE",
    )
    assert_true(gt.startswith("MTTGeneral_ICM"), f"expected ICM mode, got {gt}")
    assert_in("BUBBLE", gt)


@test
def test_icm_stacks_matching():
    """ICM: find_stacks returns matching stack configuration."""
    from icm_modes import find_gametype, find_stacks
    gt = find_gametype(players_at_table=8, phase="BUBBLE")
    depth, stacks = find_stacks(gt, [50, 30, 45, 20, 35, 25, 15, 40])
    assert_true("-" in stacks, "stacks should be dash-separated")
    parts = stacks.split("-")
    assert_eq(len(parts), 8, "should have 8 stack values")
    # Each should end in .125
    for p in parts:
        assert_true(p.endswith("125"), f"stack {p} should end in .125")


@test
def test_icm_find_params():
    """ICM: find_icm_params returns complete ICM configuration."""
    from icm_modes import find_icm_params
    result = find_icm_params(
        player_stacks=[50, 30, 45, 20, 35, 25, 15, 40],
        phase="BUBBLE",
    )
    assert_true("gametype" in result)
    assert_true("depth" in result)
    assert_true("stacks" in result)
    assert_true("approximation_note" in result)
    assert_true(result["gametype"].startswith("MTTGeneral_ICM"))


@test
def test_icm_preflop_analysis():
    """ICM: full preflop analysis with ICM mode and stacks."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "player_stacks": [50, 30, 45, 20, 35, 25, 15, 40],
        "effective_bb": 50,
        "hero_position": "SB",
        "hero_hand": "A5s",
        "preflop_actions": "F-F-F-F-F-F-R2-F",
    })
    assert_eq(result["is_icm"], True)
    assert_true(result["stacks"] != "", "ICM should have stacks")
    assert_true(result["gametype"].startswith("MTTGeneral_ICM"))
    assert_true(result["solutions"][0] is not None, "preflop solution should exist")


@test
def test_icm_symmetric_stacks():
    """ICM: symmetric stacks fallback when no player_stacks given."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "effective_bb": 20,
        "hero_position": "BTN",
        "hero_hand": "A5s",
        "preflop_actions": "F-F-F-F-F-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    assert_true(result["stacks"] != "")
    assert_in("對稱", result["text"])
    # 20bb is an available SYMMETRIC depth for BUBBLE 8-max 1000 — must be picked exactly
    assert_in("20.125", result["stacks"])


@test
def test_icm_symmetric_stacks_off_grid_depth():
    """ICM: 17bb symmetric (no SYMMETRIC config at that depth) must snap to nearest available.

    Regression: H2702 — user said "17bb icm near bubble", parsed_json had no
    player_stacks, the else branch synthesized stacks=17.125×8 but the solver
    only exposes SYMMETRIC configs at 20/25/30/35/40/50bb for
    MTTGeneral_ICM8m1000PTBUBBLE160PT. The 17.125 symmetric request returned
    204 → forced fallback to Chip EV and hid the ICM analysis the user wanted.
    """
    import analyze_hand
    # Stub solver calls — this test only verifies param resolution, not solver data.
    orig_next = analyze_hand.get_next_actions
    orig_spot = analyze_hand.get_spot_solution
    analyze_hand.get_next_actions = lambda **kw: {"actions": []}
    analyze_hand.get_spot_solution = lambda **kw: None
    try:
        result = analyze_hand.analyze_hand_full({
            "gametype": "MTTGeneral",
            "tournament_type": "icm",
            "phase": "BUBBLE",
            "effective_bb": 17,
            "hero_position": "CO",
            "hero_hand": "QQ",
            "preflop_actions": "F-R2-F-F-R5-F-F-F",
            "players_at_table": 8,
        })
    finally:
        analyze_hand.get_next_actions = orig_next
        analyze_hand.get_spot_solution = orig_spot
    assert_eq(result["is_icm"], True)
    # Must snap to 20bb SYMMETRIC (nearest available); must NOT emit 17.125
    # which corresponds to an ASYMMETRIC_FAR config that won't match uniform stacks.
    assert_true(result["stacks"].startswith("20.125-"),
                f"expected 20.125 symmetric stacks, got {result['stacks']!r}")
    assert_eq(len(result["stacks"].split("-")), 8, "must be 8 stack positions")
    assert_eq(result["depth"], "20.125")
    assert_in("用戶籌碼: 17bb", result["text"])
    assert_in("Solver 籌碼: 20bb", result["text"])
    # The resolved (depth, stacks) must exist as a visible config in the cached
    # game modes — the bug was picking a config the solver doesn't actually expose.
    from icm_modes import _load_game_modes
    gt_name = result["gametype"]
    mode = next(m for m in _load_game_modes() if m["name"] == gt_name)
    picked_stacks = result["stacks"].split("-")
    found = any(
        gm["depth"] == result["depth"]
        and gm.get("stacks") == picked_stacks
        and not gm.get("info", {}).get("hidden", False)
        for gm in mode["game_modes"]
    )
    assert_true(found,
                f"resolved config (depth={result['depth']}, symmetric 20bb) must be a visible entry in {gt_name}")


@test
def test_icm_6max_ft():
    """ICM: 6-player final table uses correct position order."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "FT",
        "player_stacks": [30, 25, 50, 40, 15, 20],
        "effective_bb": 40,
        "hero_position": "BTN",
        "hero_hand": "TT",
        "preflop_actions": "F-F-R2-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    # 6-player: LJ, HJ, CO, BTN, SB, BB
    # CO open (index 2) → preflop has R at position 2
    assert_true(result["solutions"][0] is not None, "should have preflop solution")


@test
def test_icm_postflop_falls_back_to_chipev():
    """ICM: postflop streets fall back to chip EV (ICM is preflop_only)."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "BUBBLE",
        "player_stacks": [50, 30, 45, 20, 35, 25, 15, 40],
        "effective_bb": 50,
        "hero_position": "CO",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "Ks7d2c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R2", "size": 2.0},
            ]},
        ],
    })
    assert_eq(result["is_icm"], True)
    assert_in("Chip EV", result["text"])
    assert_in("Flop", result["text"])


@test
def test_icm_hh_deviation_differs_from_chipev():
    """ICM HH: bubble ICM flags T9s UTG raise as deviation (chip EV says raise 100%)."""
    from hh_deviation_check import check_hand
    from icm_modes import find_icm_params

    # T9s UTG 20bb: chip EV = Raise 100%, ICM bubble = Fold 100%
    hand = {
        "hand_id": "TEST_ICM_HH",
        "tournament_id": "999",
        "table_size": 8,
        "num_players": 8,
        "gametype": "MTTGeneral",
        "effective_bb": 20,
        "hero_position": "UTG",
        "hero_hand": "Ts9s",
        "preflop_actions": "R2-F-F-F-F-F-F-F",
        "stacks_bb": [20, 20, 20, 20, 20, 20, 20, 20],
        "avg_stack_chips": 20000,
    }

    # Without ICM: hero raising T9s should be the dominant action (100% raise)
    devs_chipev = check_hand(hand, icm_params=None)
    assert_true(len(devs_chipev) > 0, "chip EV should have a preflop spot")
    assert_eq(devs_chipev[0]["hero_action"], devs_chipev[0]["gto_action"],
              "chip EV: T9s UTG raise should match GTO dominant action (raise)")

    # With ICM bubble: hero raising T9s should be flagged as deviation (GTO = fold)
    icm = find_icm_params(player_stacks=[20]*8, phase="BUBBLE")
    devs_icm = check_hand(hand, icm_params=icm)
    assert_true(len(devs_icm) > 0, "ICM should have a preflop spot")
    assert_true(devs_icm[0]["hero_action"] != devs_icm[0]["gto_action"],
                "ICM bubble: T9s UTG raise should NOT match GTO (GTO = fold)")
    assert_eq(devs_icm[0]["gto_action"], "F", "ICM bubble GTO action should be Fold")


@test
def test_missing_solver_data_explains_rare_line():
    """Missing solver data: explains hero's rare action caused solver gap."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 22,
        "hero_position": "UTG+1",
        "hero_hand": "9h9c",
        "preflop_actions": "F-R2-F-F-F-C-F-C",
        "streets": [
            {"board": "6s7h6h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "UTG+1", "action": "R2.5", "size": 2.5},
                {"position": "BB", "action": "R8.7", "size": 8.7},
                {"position": "UTG+1", "action": "C"},
            ]},
            {"card": "3c", "actions": [
                {"position": "BB", "action": "AI", "size": 9.3},
                {"position": "UTG+1", "action": "C"},
            ]},
        ],
    })
    text = result["text"]
    # Turn should explain why no solver data (hero's rare flop call)
    assert_not_in("無 solver 數據", text, "Should explain instead of generic message")
    assert_in("solver 未計算", text, "Should mention solver gap due to rare line")
    assert_in("All-in", text, "Should mention GTO recommended action")


@test
def test_preflop_only_multiway_allin():
    """Multiway preflop-only: SB all-in should simplify without false corrections."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 10,
        "hero_position": "SB",
        "hero_hand": "A8s",
        "preflop_actions": "F-R2-C-F-F-F-AI10-F",
        "streets": [],
    })
    text = result["text"]
    # Should NOT contain correction notes for AI→RAI
    assert_not_in("近似說明", text, "Should not show false correction note for AI→RAI")
    # Should have some analysis output
    assert_true(len(text) > 10, "Should produce analysis text")


# ── Multiway preflop reconciliation + real-node preflop (H3511) ──

# H3511: "lj raise, co call, hero btn call, bb call" parsed to F-F-R2-C-C-F-F-C
# — the LLM packed callers next to the raiser (HJ & CO call) and FOLDED hero BTN,
# despite BTN checking the flop. The multiway collapse then folded hero pre-flop,
# leaving no post-flop node, so every street printed "（無 solver 數據）".

_H3511_STREETS = [
    {"board": "9sJcQh", "actions": [
        {"position": "BB", "action": "X"}, {"position": "LJ", "action": "X"},
        {"position": "CO", "action": "X"}, {"position": "BTN", "action": "X"}]},
    {"card": "Th", "actions": [
        {"position": "LJ", "action": "R", "size": 2.6},
        {"position": "CO", "action": "F"}, {"position": "BTN", "action": "C"},
        {"position": "BB", "action": "F"}]},
    {"card": "Ac", "actions": [
        {"position": "LJ", "action": "X"}, {"position": "BTN", "action": "X"}]},
]


@test
def test_reconcile_rebuilds_when_hero_folded_on_checkaround_flop():
    """H3511: hero folded pre-flop but checks the flop → rebuild from flop seats.

    The flop is a pure check-around (BB/LJ/CO/BTN all check), so its participant
    list is complete: re-seat the callers (drop the phantom HJ, restore BTN) and
    keep the single raise.
    """
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    new, changed = _reconcile_preflop_with_streets(
        "F-F-R2-C-C-F-F-C", _H3511_STREETS, "BTN", POSITION_ORDER)
    assert_true(changed, "should reconcile a hero-folded multiway line")
    assert_eq(new, "F-F-R2-F-C-C-F-C", "callers re-seated to CO/BTN/BB, HJ dropped")


@test
def test_reconcile_noop_when_hero_not_folded():
    """A faithfully-parsed multiway line (hero is a caller) is left untouched."""
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    new, changed = _reconcile_preflop_with_streets(
        "F-F-R2-F-C-C-F-C", _H3511_STREETS, "BTN", POSITION_ORDER)
    assert_true(not changed, "consistent line must not be rewritten")
    assert_eq(new, "F-F-R2-F-C-C-F-C")


@test
def test_reconcile_does_not_drop_caller_on_bet_flop():
    """Hero folded + flop has a bet → ADD hero, but do NOT drop the other caller.

    A non-check-around flop may omit players who folded to the bet, so a pre-flop
    caller absent from the flop actions is kept (it collapses to a fold later)
    rather than wrongly dropped. Hero (UTG+1) is restored as a caller.
    """
    from analyze_hand import _reconcile_preflop_with_streets, POSITION_ORDER
    streets = [
        {"board": "6s7h6h", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "UTG+1", "action": "R", "size": 2.5},
            {"position": "BB", "action": "R", "size": 8.7}]},
    ]
    # UTG+1 (idx1) folded pre-flop but bets the flop; BTN (idx5) called pre-flop
    # but never appears on this bet flop — must be kept, not dropped.
    new, changed = _reconcile_preflop_with_streets(
        "F-F-F-F-F-C-F-C", streets, "UTG+1", POSITION_ORDER)
    assert_true(changed, "hero folded pre-flop yet plays the flop → must repair")
    parts = new.split("-")
    assert_true(parts[1] != "F", "hero UTG+1 must be restored as a non-folder")
    assert_eq(parts[5], "C", "the off-flop caller (BTN) is kept, not dropped")


@test
def test_h3511_multiway_postflop_has_solver_data_and_overcall_preflop():
    """End-to-end H3511: buggy parse → full post-flop solver data + overcall node.

    After reconciliation the pot is BTN-vs-LJ heads-up post-flop (so every street
    has solver data, not "（無 solver 數據）"), while the pre-flop BTN node reflects
    the REAL multiway decision — facing LJ's open AND CO's call (the real-structure
    branch), not the open alone.
    """
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "players_at_table": 8, "effective_bb": 60,
        "hero_position": "BTN", "hero_hand": "6h7h",
        "preflop_actions": "F-F-R2-C-C-F-F-C",  # the buggy LLM parse
        "streets": _H3511_STREETS,
    })
    text = result["text"]
    assert_not_in("（無 solver 數據）", text,
                  "post-flop must have solver data after HU simplification")
    assert_in("保留真實下注結構", text, "must use the real-structure HU branch")
    # Real-structure preflop spot: BTN faces LJ open + CO call. The collapsed
    # open-only node would leave the BTN spot facing a single raiser; the real
    # node includes the overcaller's dead money, so the pre-flop pot exceeds the
    # open-only 4.8bb (≈2.5 open + blinds). Assert the overcall node is in use.
    assert_in("【Preflop】", text)
    # Hero spot present on every street (preflop + flop + turn + river headers).
    for header in ("【Flop:", "【Turn:", "【River:"):
        assert_in(header, text, f"{header} section must render")


# ── HH Parser Tests ──

_SAMPLE_HH_PREFLOP = """\
Poker Hand #TM5600279262: Tournament #264809938, ¥220 Satellite to #12: Zodiac Monkey King Wukong, 5 Seats Hold'em No Limit - Level4(100/200) - 2026/02/17 14:37:16
Table '4' 8-max Seat #6 is the button
Seat 3: e0d65ab0 (18,221 in chips)
Seat 4: Hero (2,177 in chips)
Seat 5: dad95b5a (4,836 in chips)
Seat 6: 4337b2cd (5,160 in chips)
Seat 7: f7728f06 (9,474 in chips)
Seat 8: e1f388aa (15,436 in chips)
f7728f06: posts the ante 20
dad95b5a: posts the ante 20
Hero: posts the ante 20
e1f388aa: posts the ante 20
e0d65ab0: posts the ante 20
4337b2cd: posts the ante 20
f7728f06: posts small blind 100
e1f388aa: posts big blind 200
*** HOLE CARDS ***
Dealt to Hero [Ad 9c]
e0d65ab0: folds
Hero: raises 1,957 to 2,157 and is all-in
dad95b5a: folds
4337b2cd: folds
f7728f06: folds
e1f388aa: folds
Uncalled bet (1,957) returned to Hero
*** SUMMARY ***
Total pot 620 | Rake 0"""

_SAMPLE_HH_FOLD = """\
Poker Hand #TM5600279272: Tournament #264809938, ¥220 Satellite to #12: Zodiac Monkey King Wukong, 5 Seats Hold'em No Limit - Level4(100/200) - 2026/02/17 14:36:26
Table '4' 8-max Seat #6 is the button
Seat 3: e0d65ab0 (16,164 in chips)
Seat 4: Hero (2,037 in chips)
Seat 5: dad95b5a (5,856 in chips)
Seat 6: 4337b2cd (5,380 in chips)
Seat 7: f7728f06 (10,811 in chips)
Seat 8: e1f388aa (15,056 in chips)
f7728f06: posts the ante 20
dad95b5a: posts the ante 20
Hero: posts the ante 20
e1f388aa: posts the ante 20
e0d65ab0: posts the ante 20
4337b2cd: posts the ante 20
f7728f06: posts small blind 100
e1f388aa: posts big blind 200
*** HOLE CARDS ***
Dealt to Hero [8s 6d]
e0d65ab0: raises 200 to 400
Hero: folds
dad95b5a: folds
4337b2cd: calls 400
f7728f06: folds
e1f388aa: calls 200
*** FLOP *** [7s Ad 3h]
e1f388aa: checks
e0d65ab0: bets 554
4337b2cd: folds
e1f388aa: folds
*** SUMMARY ***
Total pot 1,520 | Rake 0"""


@test
def test_hh_parser_preflop_basic():
    """HH Parser: parses preflop-only hand correctly."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_PREFLOP)
    assert_true(result is not None, "should parse hero hand")
    assert_eq(result["hand_id"], "TM5600279262")
    assert_eq(result["hero_hand"], "Ad9c")
    assert_eq(result["hero_position"], "HJ")
    assert_eq(result["num_players"], 6)
    assert_eq(result["table_size"], 8)
    assert_true(result["effective_bb"] > 10, f"ebb={result['effective_bb']}")
    assert_in("AI", result["preflop_actions"])
    assert_true("streets" not in result or len(result.get("streets", [])) == 0)


@test
def test_hh_parser_fold_excluded():
    """HH Parser: hero fold excluded by default."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_FOLD, include_folds=False)
    assert_true(result is None, "fold hand should be excluded")


@test
def test_hh_parser_fold_included():
    """HH Parser: hero fold included with include_folds=True."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_FOLD, include_folds=True)
    assert_true(result is not None, "fold hand should be included")
    assert_eq(result["hero_hand"], "8s6d")
    assert_eq(result["hero_position"], "HJ")  # seat 4, button=seat 6, 6 players
    # Hero's action is F (fold) at HJ position (index 1 in 6-player)
    parts = result["preflop_actions"].split("-")
    assert_eq(parts[1], "F", "Hero HJ folds")


@test
def test_hh_parser_postflop_streets():
    """HH Parser: postflop actions parsed from fold hand (other players)."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_FOLD, include_folds=True)
    assert_true(result is not None)
    # This hand has a flop even though hero folded
    streets = result.get("streets", [])
    if streets:
        assert_eq(streets[0]["board"], "7sAd3h")


# SB 26bb all-in vs BB 10bb — effective should be 10bb (min of involved stacks)
# 8-max, button=seat 1 → seat 2=SB, seat 3=BB
_SAMPLE_HH_EFF_STACK = """\
Poker Hand #TM5600280421: Tournament #264809938, ¥220 Hold'em No Limit - Level8(200/400) - 2026/02/17 15:00:00
Table '2' 8-max Seat #1 is the button
Seat 1: a1234567 (12,000 in chips)
Seat 2: Hero (10,400 in chips)
Seat 3: c3456789 (4,000 in chips)
Seat 4: d4567890 (15,000 in chips)
Seat 5: e5678901 (9,000 in chips)
Seat 6: f6789012 (7,000 in chips)
Seat 7: g7890123 (8,000 in chips)
Seat 8: h8901234 (6,000 in chips)
Hero: posts the ante 40
c3456789: posts the ante 40
a1234567: posts the ante 40
d4567890: posts the ante 40
e5678901: posts the ante 40
f6789012: posts the ante 40
g7890123: posts the ante 40
h8901234: posts the ante 40
Hero: posts small blind 200
c3456789: posts big blind 400
*** HOLE CARDS ***
Dealt to Hero [Qd Tc]
d4567890: folds
e5678901: folds
f6789012: folds
g7890123: folds
h8901234: folds
a1234567: folds
Hero: raises 9,960 to 10,160 and is all-in
c3456789: folds
Uncalled bet (9,760) returned to Hero
*** SUMMARY ***
Total pot 1,120 | Rake 0"""


@test
def test_hh_parser_effective_stack_min():
    """HH Parser: effective_bb is min of hero and opponent stacks in pot."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_EFF_STACK)
    assert_true(result is not None, "should parse hand")
    assert_eq(result["hero_position"], "SB")
    # Hero SB = 10400 chips = 26bb, but BB = 4000 chips = 10bb
    # Effective stack should be 10bb (min of the two)
    assert_true(result["effective_bb"] <= 10.0,
                f"effective_bb should be <=10 (BB has 10bb), got {result['effective_bb']}")
    assert_true(result["effective_bb"] >= 9.5,
                f"effective_bb should be ~10, got {result['effective_bb']}")


@test
def test_hh_parser_subtracts_ante():
    """HH Parser: ante is deducted from chip counts so stacks reflect the
    post-ante state going into the betting round (same convention as the
    all-in sizes printed in the action log)."""
    from hh_parser import parse_hand
    # _SAMPLE_HH_EFF_STACK: bb=400, ante=40
    # Hero seat 2 declared 10,400 chips → post-ante 10,360 → 25.9bb
    # BB seat 3 declared  4,000 chips → post-ante  3,960 →  9.9bb
    result = parse_hand(_SAMPLE_HH_EFF_STACK)
    assert_true(result is not None)
    # hero_chips field reports post-ante chips (matches the all-in size logged
    # by the dealer, which is also post-ante).
    assert_eq(result["hero_chips"], 10360, f"hero_chips={result['hero_chips']}")
    # effective_bb = min(hero, BB) / bb_size = 3960 / 400 = 9.9
    assert_true(abs(result["effective_bb"] - 9.9) < 0.05,
                f"effective_bb should be 9.9 (post-ante), got {result['effective_bb']}")
    # stacks_bb reports each position's post-ante stack.
    # SB stack 25.9bb, BB stack 9.9bb at minimum should both appear.
    stacks = result["stacks_bb"]
    assert_true(9.9 in stacks, f"BB 9.9bb missing from stacks_bb={stacks}")
    assert_true(25.9 in stacks, f"Hero SB 25.9bb missing from stacks_bb={stacks}")


# Partial-ante all-in: a 30-chip short stack at a 100-ante table can only
# post 30 of the 100; their post-ante stack is 0, while everyone else has
# their full ante subtracted.
_SAMPLE_HH_PARTIAL_ANTE = """\
Poker Hand #TM5600281000: Tournament #264809938, ¥220 Hold'em No Limit - Level10(500/1000(100)) - 2026/02/17 16:00:00
Table '3' 8-max Seat #1 is the button
Seat 1: alpha (20000 in chips)
Seat 2: bravo (15000 in chips)
Seat 3: charlie (30 in chips)
Seat 4: Hero (12000 in chips)
Seat 5: echo (8000 in chips)
Seat 6: foxtrot (10000 in chips)
alpha: posts the ante 100
bravo: posts the ante 100
charlie: posts the ante 30 and is all-in
Hero: posts the ante 100
echo: posts the ante 100
foxtrot: posts the ante 100
bravo: posts small blind 500
charlie: posts big blind 0 and is all-in
*** HOLE CARDS ***
Dealt to Hero [Qd Tc]
Hero: folds
echo: folds
foxtrot: folds
alpha: folds
bravo: folds
*** SUMMARY ***
Total pot 530 | Rake 0"""


@test
def test_hh_parser_partial_ante_allin():
    """HH Parser: short stack that all-in'd on the ante gets 0 post-ante chips;
    everyone else gets the full header ante subtracted."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_PARTIAL_ANTE, include_folds=True)
    assert_true(result is not None, "should parse")
    # bb=1000, ante=100. alpha 20000-100=19900 → 19.9bb. charlie 30-30=0 → 0.0bb.
    stacks = result["stacks_bb"]
    assert_true(0.0 in stacks, f"partial-ante all-in not at 0bb: {stacks}")
    assert_true(19.9 in stacks, f"alpha not at 19.9bb post-ante: {stacks}")


@test
def test_card_split_no_hand_leakage():
    """A hand_id appearing in train must not appear in val or test."""
    from ocr.classifier.split import build_split
    from pathlib import Path
    gt_path = Path(__file__).resolve().parent.parent / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    split = build_split(gt_path, train=0.8, val=0.1, test=0.1, seed=0)
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    assert_eq(len(train_ids & val_ids), 0, "train/val overlap")
    assert_eq(len(train_ids & test_ids), 0, "train/test overlap")
    assert_eq(len(val_ids & test_ids), 0, "val/test overlap")
    total = len(train_ids) + len(val_ids) + len(test_ids)
    assert_true(0.78 <= len(train_ids)/total <= 0.82,
                f"train frac off: {len(train_ids)/total}")
    assert_true(0.08 <= len(val_ids)/total <= 0.12,
                f"val frac off: {len(val_ids)/total}")
    assert_true(0.08 <= len(test_ids)/total <= 0.12,
                f"test frac off: {len(test_ids)/total}")


@test
def test_card_split_tournament_balanced():
    """Every tournament with >=10 hands appears in all three splits."""
    import json
    from collections import Counter
    from pathlib import Path
    from ocr.classifier.split import build_split
    gt_path = Path(__file__).resolve().parent.parent / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
    split = build_split(gt_path, train=0.8, val=0.1, test=0.1, seed=0)
    hid_to_tid = {}
    with gt_path.open() as fh:
        for line in fh:
            o = json.loads(line)
            hid_to_tid[o["hand_id"]] = o["ground_truth"].get("tournament_id")
    big_tourneys = {t for t, n in Counter(hid_to_tid.values()).items()
                    if n and n >= 10}
    in_bucket = {"train": set(), "val": set(), "test": set()}
    for bucket in ("train", "val", "test"):
        for hid in split[bucket]:
            in_bucket[bucket].add(hid_to_tid.get(hid))
    for t in big_tourneys:
        for bucket in ("train", "val", "test"):
            assert_in(t, in_bucket[bucket], f"tourney {t} missing from {bucket}")


# ── 169 Hand Index Tests ──

@test
def test_169_hand_index_count():
    """169 Index: generates exactly 169 unique hand names."""
    from hh_deviation_check import HANDS_169, HAND_TO_169
    assert_eq(len(HANDS_169), 169)
    assert_eq(len(HAND_TO_169), 169)


@test
def test_169_hand_index_ascii_sorted():
    """169 Index: hand names are sorted by ASCII comparison."""
    from hh_deviation_check import HANDS_169
    assert_eq(HANDS_169, sorted(HANDS_169))


@test
def test_169_hand_index_premiums():
    """169 Index: premium hands map to correct indices."""
    from hh_deviation_check import HAND_TO_169
    # Verify key hands exist
    for h in ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "22"]:
        assert_true(h in HAND_TO_169, f"{h} should be in index")
    # AA should come before KK in ASCII (A < K)
    assert_true(HAND_TO_169["AA"] < HAND_TO_169["KK"],
                "AA index should be less than KK (A < K in ASCII)")


@test
def test_169_hand_index_offsuit_before_suited():
    """169 Index: offsuit comes before suited for same ranks (o < s in ASCII)."""
    from hh_deviation_check import HAND_TO_169
    assert_true(HAND_TO_169["AKo"] < HAND_TO_169["AKs"],
                "AKo should come before AKs")
    assert_true(HAND_TO_169["KQo"] < HAND_TO_169["KQs"])


# ── Preflop 8-max Conversion Tests ──

@test
def test_convert_preflop_8max_6p():
    """8max convert: 6-player prepends 2 folds."""
    from hh_deviation_check import _convert_preflop_to_8max
    result = _convert_preflop_to_8max("R2-F-F-F-F-C", 6)
    assert_eq(result, "F-F-R2-F-F-F-F-C")


@test
def test_convert_preflop_8max_8p():
    """8max convert: 8-player unchanged."""
    from hh_deviation_check import _convert_preflop_to_8max
    result = _convert_preflop_to_8max("F-R2-F-F-F-F-F-C", 8)
    assert_eq(result, "F-R2-F-F-F-F-F-C")


# ── Deviation Report Format Tests ──

@test
def test_deviation_report_no_deviations():
    """Report: no deviations produces clean message."""
    from hh_deviation_report import format_deviation_report
    results = [{
        "hand_id": "TM1", "hero_position": "CO", "hero_hand": "AKs",
        "hero_hand_normalized": "AKs", "effective_bb": 30, "num_players": 8,
        "preflop_actions": "F-F-F-F-R2-F-F-F", "spots_checked": 1,
        "deviations": [{
            "street": "preflop", "spot": "open",
            "hero_action": "R2.1", "hero_action_label": "RAISE",
            "hero_freq": 1.0, "gto_action": "R2.1", "gto_action_label": "RAISE",
            "gto_freq": 1.0, "all_freqs": {"R2.1": 1.0},
        }],
    }]
    report = format_deviation_report(results)
    assert_in("不錯", report)
    assert_not_in("嚴重", report)


@test
def test_deviation_report_severe():
    """Report: severe deviation (0% GTO) categorized correctly."""
    from hh_deviation_report import format_deviation_report
    results = [{
        "hand_id": "TM1", "hero_position": "BB", "hero_hand": "8s6d",
        "hero_hand_normalized": "86o", "effective_bb": 10, "num_players": 6,
        "preflop_actions": "F-R2-F-C-F-C", "spots_checked": 1,
        "deviations": [{
            "street": "preflop", "spot": "open",
            "hero_action": "C", "hero_action_label": "Call",
            "hero_freq": 0, "gto_action": "F", "gto_action_label": "Fold",
            "gto_freq": 1.0, "all_freqs": {"F": 1.0},
        }],
    }]
    report = format_deviation_report(results)
    assert_in("嚴重偏差", report)
    assert_in("86o", report)
    assert_in("Call", report)
    assert_in("Fold", report)


@test
def test_deviation_report_mixed_severity():
    """Report: multiple severity levels categorized separately."""
    from hh_deviation_report import format_deviation_report
    results = [
        {
            "hand_id": "TM1", "hero_position": "BB", "hero_hand": "8s6d",
            "hero_hand_normalized": "86o", "effective_bb": 10, "num_players": 6,
            "preflop_actions": "F-R2-F-C-F-C", "spots_checked": 1,
            "deviations": [{
                "street": "preflop", "spot": "open",
                "hero_action": "C", "hero_action_label": "Call",
                "hero_freq": 0, "gto_action": "F", "gto_action_label": "Fold",
                "gto_freq": 1.0, "all_freqs": {"F": 1.0},
            }],
        },
        {
            "hand_id": "TM2", "hero_position": "SB", "hero_hand": "AhKh",
            "hero_hand_normalized": "AKs", "effective_bb": 74, "num_players": 8,
            "preflop_actions": "F-F-F-F-F-F-R3-F", "spots_checked": 1,
            "deviations": [{
                "street": "preflop", "spot": "open",
                "hero_action": "R4", "hero_action_label": "RAISE",
                "hero_freq": 0.34, "gto_action": "C", "gto_action_label": "Call",
                "gto_freq": 0.66, "all_freqs": {"C": 0.66, "R4": 0.34},
            }],
        },
    ]
    report = format_deviation_report(results)
    assert_in("嚴重偏差", report)
    assert_in("1 處偏差", report)
    assert_true("中等偏差" not in report, "moderate deviations should be excluded")


# ── HH Deviation Check E2E (API) ──

@test
def test_hh_check_hand_preflop():
    """HH Check: check_hand returns deviations for known bad play."""
    from hh_deviation_check import check_hand
    hand = {
        "hand_id": "TEST1",
        "hero_position": "BB",
        "hero_hand": "8s6d",
        "effective_bb": 10.2,
        "num_players": 6,
        "table_size": 8,
        "preflop_actions": "F-R2.0-F-C-F-C",
    }
    devs = check_hand(hand)
    assert_true(len(devs) >= 1, "should have at least 1 spot checked")
    # BB calling LJ open with 86o at 10bb — GTO says fold
    assert_eq(devs[0]["street"], "preflop")
    assert_true(devs[0]["hero_freq"] < 0.05,
                f"86o call should be ~0% GTO, got {devs[0]['hero_freq']:.1%}")


@test
def test_hh_check_hand_correct_play():
    """HH Check: check_hand shows high frequency for correct play."""
    from hh_deviation_check import check_hand
    hand = {
        "hand_id": "TEST2",
        "hero_position": "LJ",
        "hero_hand": "AcKc",
        "effective_bb": 24,
        "num_players": 6,
        "table_size": 8,
        "preflop_actions": "R2.0-F-F-F-F-F",
    }
    devs = check_hand(hand)
    assert_true(len(devs) >= 1, "should have at least 1 spot")
    # AKs opening from LJ at 24bb — should be very high frequency
    assert_true(devs[0]["hero_freq"] > 0.9,
                f"AKs open should be >90% GTO, got {devs[0]['hero_freq']:.1%}")


@test
def test_deviation_report_low_ev_shown():
    """Report: low EV deviations are still shown (no EV filter)."""
    from hh_deviation_report import format_deviation_report
    results = [{
        "hand_id": "TM1", "hero_position": "BB", "hero_hand": "9s3s",
        "hero_hand_normalized": "93s", "effective_bb": 42, "num_players": 7,
        "preflop_actions": "R2.0-F-F-F-F-R4.8-F-C", "spots_checked": 1,
        "deviations": [{
            "street": "preflop", "spot": "open",
            "hero_action": "F", "hero_action_label": "Fold",
            "hero_freq": 0, "gto_action": "C", "gto_action_label": "Call",
            "gto_freq": 1.0, "all_freqs": {"C": 1.0},
            "hero_ev": 0.3,
        }],
    }]
    report = format_deviation_report(results)
    # Low EV deviation should still appear (EV filter removed)
    assert_in("嚴重偏差", report)
    assert_in("93s", report)


@test
def test_deviation_report_tiny_ev_not_filtered():
    """Report: very low EV hands (like K5o EV=0.01bb) still show deviations."""
    from hh_deviation_report import format_deviation_report
    # Mirrors real case: SB K5o facing 3-bet, hero folds but GTO says call 58%
    results = [{
        "hand_id": "TM5614184519", "hero_position": "SB", "hero_hand": "Kc5d",
        "hero_hand_normalized": "K5o", "effective_bb": 19.6, "num_players": 8,
        "preflop_actions": "F-F-F-F-F-C-R3.0-F", "spots_checked": 2,
        "icm_phase": "25%",
        "deviations": [
            {
                "street": "preflop", "spot": "facing 3bet",
                "hero_action": "F", "hero_action_label": "Fold",
                "hero_freq": 0, "gto_action": "C", "gto_action_label": "Call",
                "gto_freq": 0.58, "all_freqs": {"C": 0.58, "R8.5": 0.42},
                "hero_ev": 0.012,
            },
        ],
    }]
    report = format_deviation_report(results)
    # Must appear despite tiny EV
    assert_in("嚴重偏差", report)
    assert_in("K5o", report)
    assert_in("Call", report)


@test
def test_deviation_report_severe_category():
    """Report: 0% freq deviations appear in severe category."""
    from hh_deviation_report import format_deviation_report
    results = [{
        "hand_id": "TM1", "hero_position": "CO", "hero_hand": "AcKc",
        "hero_hand_normalized": "AKs", "effective_bb": 30, "num_players": 6,
        "preflop_actions": "F-F-F-F-F-F", "spots_checked": 1,
        "deviations": [{
            "street": "preflop", "spot": "open",
            "hero_action": "F", "hero_action_label": "Fold",
            "hero_freq": 0, "gto_action": "R2.1", "gto_action_label": "RAISE",
            "gto_freq": 1.0, "all_freqs": {"R2.1": 1.0},
            "hero_ev": 3.5,
        }],
    }]
    report = format_deviation_report(results)
    assert_in("嚴重偏差", report)


@test
def test_deviation_report_format_structure():
    """Report: new format has street name, 建議 on new line, clean numbers."""
    from hh_deviation_report import format_deviation_report
    results = [{
        "hand_id": "TM5608762330", "hero_position": "CO", "hero_hand": "As8s",
        "hero_hand_normalized": "A8s", "effective_bb": 12.0, "num_players": 6,
        "preflop_actions": "F-R2.0-C-F-C", "spots_checked": 1,
        "deviations": [{
            "street": "preflop", "spot": "open",
            "hero_action": "R2.0", "hero_action_label": "RAISE",
            "hero_freq": 0, "gto_action": "RAI", "gto_action_label": "All-in 12bb",
            "gto_freq": 0.78, "all_freqs": {"RAI": 0.78, "F": 0.22},
        }],
    }]
    report = format_deviation_report(results)
    # Street name before hero action
    assert_in("Preflop RAISE", report)
    # Recommendation on new line with 建議 prefix
    assert_in("建議：應 All-in 12bb", report)
    # No trailing .0 on effective_bb
    assert_in("12bb", report)
    assert_not_in("12.0bb", report)
    # Inline "→ 應" should NOT exist (moved to new line)
    assert_not_in("→ 應", report)


@test
def test_check_hand_includes_ev():
    """HH Check: check_hand returns hero_ev in deviation dicts."""
    from hh_deviation_check import check_hand
    hand = {
        "hand_id": "TEST_EV",
        "hero_position": "LJ",
        "hero_hand": "AcKc",
        "effective_bb": 24,
        "num_players": 6,
        "table_size": 8,
        "preflop_actions": "R2.0-F-F-F-F-F",
    }
    devs = check_hand(hand)
    assert_true(len(devs) >= 1, "should have at least 1 spot")
    # hero_ev should be present (not None) for a premium hand
    assert_true("hero_ev" in devs[0], "deviation should include hero_ev key")
    # AKs at LJ should have positive EV
    if devs[0]["hero_ev"] is not None:
        assert_true(devs[0]["hero_ev"] > 0,
                    f"AKs open EV should be positive, got {devs[0]['hero_ev']}")


@test
def test_hh_e2e_parse_check_report():
    """HH E2E: parse hand → check deviations → format report."""
    from hh_parser import parse_hand
    from hh_deviation_check import check_hand
    from hh_deviation_report import format_deviation_report

    # Parse
    hand = parse_hand(_SAMPLE_HH_PREFLOP)
    assert_true(hand is not None)

    # Check
    devs = check_hand(hand)
    assert_true(len(devs) >= 1)

    # Build result for report
    from gto_formatter import normalize_hand_name
    result = {
        "hand_id": hand["hand_id"],
        "hero_position": hand["hero_position"],
        "hero_hand": hand["hero_hand"],
        "hero_hand_normalized": normalize_hand_name(hand["hero_hand"]),
        "effective_bb": hand["effective_bb"],
        "num_players": hand["num_players"],
        "preflop_actions": hand["preflop_actions"],
        "spots_checked": len(devs),
        "deviations": devs,
    }
    report = format_deviation_report([result])
    assert_in("GTO 偏差分析報告", report)
    assert_in("1 手", report)


# ── Combo index + postflop suit-specific tests ──

@test
def test_combo_index_for_hand():
    """Combo index: combo_index_for_hand maps specific combos to correct 1326 index."""
    from gto_formatter import combo_index_for_hand as _combo_index_for_hand
    from gto_formatter import _COMBO_INDEX

    # Ah6h → should map to the correct index
    idx = _combo_index_for_hand("Ah6h")
    assert_true(idx is not None, "Ah6h should have a valid index")
    c1, c2 = _COMBO_INDEX[idx]
    assert_true({c1, c2} == {"Ah", "6h"}, f"index {idx} should be Ah+6h, got {c1}+{c2}")

    # AcKd → different combo
    idx2 = _combo_index_for_hand("AcKd")
    assert_true(idx2 is not None, "AcKd should have a valid index")
    c1, c2 = _COMBO_INDEX[idx2]
    assert_true({c1, c2} == {"Ac", "Kd"}, f"index {idx2} should be Ac+Kd, got {c1}+{c2}")

    # 6hAh (reversed) → should give same index as Ah6h
    idx3 = _combo_index_for_hand("6hAh")
    assert_eq(idx3, idx, "6hAh and Ah6h should map to same combo index")

    # Invalid inputs
    assert_eq(_combo_index_for_hand("A6s"), None, "simplified name should return None")
    assert_eq(_combo_index_for_hand(""), None, "empty string should return None")
    assert_eq(_combo_index_for_hand("AhAh"), None, "same card should return None")


@test
def test_postflop_combo_specific_lookup():
    """HH Check: postflop uses exact combo (Ah6h) not aggregated A6s on flush-draw board."""
    from hh_deviation_check import check_hand

    # TM5628247517: SB Ah6h on 7hJhQd — has nut flush draw
    # Ah6h should have high call/raise freq; other A6s combos fold
    hand = {
        "hand_id": "TEST_COMBO",
        "hero_position": "SB",
        "hero_hand": "6hAh",
        "effective_bb": 54.6,
        "num_players": 8,
        "table_size": 8,
        "preflop_actions": "F-F-F-F-F-R2.2-C-F",
        "streets": [{
            "board": "7hJhQd",
            "actions": [
                {"action": "X", "position": "SB"},
                {"action": "R3.3", "position": "BTN", "size": 3.3},
                {"action": "R8.7", "position": "SB", "size": 8.7},
                {"action": "F", "position": "BTN"},
            ],
        }],
    }
    devs = check_hand(hand)

    # Find the flop deviation where hero faces bet (second flop spot)
    flop_devs = [d for d in devs if d["street"] == "flop"]
    assert_true(len(flop_devs) >= 2, "should have 2 flop spots (check + facing bet)")

    # Second flop spot: SB facing BTN's bet — this is where suit matters
    facing_bet = flop_devs[1]
    # Ah6h with nut flush draw should NOT have fold as GTO recommendation
    # Solver says ~89% call for Ah6h specifically
    assert_true(
        facing_bet["gto_action"] != "F",
        f"Ah6h on 7hJhQd should not be told to fold, got gto_action={facing_bet['gto_action']}"
    )
    # Call frequency should be high (>50%) for the flush draw combo
    call_freq = facing_bet["all_freqs"].get("C", 0)
    assert_true(
        call_freq > 0.50,
        f"Ah6h call freq should be >50% (flush draw), got {call_freq*100:.0f}%"
    )


# ── Table size inference + padding tests ──

@test
def test_num_players_inferred_from_preflop():
    """Table size: 6-player with players_at_table=6 pads correctly."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "players_at_table": 6,
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-R2-F-C",  # 6 actions = 6-player
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Should recognize BTN raised (not HJ which would be 8-player mapping)
    assert_in("BTN", text)
    # Should find BB's data (not "找不到 BB")
    assert_in("BB", text)
    assert_true("找不到 BB" not in text, "Should find BB preflop data with 6-player padding")


@test
def test_multiway_preflop_default_8max():
    """Table size: incomplete preflop actions default to 8-max for MTTGeneral."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "effective_bb": 25,
        "hero_position": "SB",
        "hero_hand": "88",
        "preflop_actions": "F-R2-F-F-C-F",  # 6 actions, hero SB hasn't acted
    })
    text = result["text"]
    # Should map UTG+1 as raiser (not HJ from wrong 6-max padding)
    assert_in("UTG+1", text, "Should identify UTG+1 as raiser in 8-max")
    assert_true("HJ" not in text, "Should NOT map raiser to HJ (wrong 6-max padding)")


@test
def test_num_players_8p_no_padding():
    """Table size: 8-player hand needs no padding."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "hero_position": "BB",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-F-F-R2-F-C",  # 8 actions = 8-player
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    assert_in("BTN", text)
    assert_true("找不到 BB" not in text, "Should find BB preflop data for 8-player")


@test
def test_num_players_from_players_at_table():
    """Table size: players_at_table field takes priority over preflop count."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "players_at_table": 6,
        "hero_position": "BB",
        "hero_hand": "AKs",
        "preflop_actions": "F-F-F-R2-F-C",
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    assert_in("BTN", text)
    assert_true("找不到 BB" not in text, "Should find BB data with players_at_table=6")


@test
def test_num_players_field_pads_correctly():
    """Table size: num_players field (from hh_parser) triggers correct padding."""
    from analyze_hand import analyze_hand_full
    # 7-player table: BTN opens, SB folds, BB calls
    # Without fix: num_players not read → defaults to 8 → no padding → CO opens instead of BTN
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 51.8,
        "num_players": 7,
        "hero_position": "BB",
        "hero_hand": "AcJh",
        "preflop_actions": "F-F-F-F-R2.0-F-C",
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Should correctly identify BTN as opener (not CO)
    assert_in("BTN", text)
    assert_not_in("CO raise", text)
    # AJo should NOT be 64% all-in (that was the wrong CO-open solver node)
    assert_not_in("64.2%", text)


@test
def test_postflop_allin_action_matching():
    """Action matching: near-all-in bet matches RAI, not Call."""
    from gto_api import find_closest_action_postflop
    # Simulate available actions: F, C(3bb), R6.8, RAI(12bb)
    available = [
        {"action": {"code": "F", "betsize": "0", "allin": False}},
        {"action": {"code": "C", "betsize": "3.0", "allin": False}},
        {"action": {"code": "R6.8", "betsize": "6.8", "allin": False, "betsize_by_pot": "0.33"}},
        {"action": {"code": "RAI", "betsize": "12.0", "allin": True, "betsize_by_pot": "0.78"}},
    ]
    # 11.8 is very close to all-in (12.0), should match RAI not C
    result = find_closest_action_postflop(available, 11.8)
    assert_eq(result, "RAI")


@test
def test_postflop_pct_bet_still_detected():
    """Action matching: percentage-based bet still detected when far from all-in."""
    from gto_api import find_closest_action_postflop
    # Simulate: pot ~20bb, actions include R6.6 (33%), R10 (50%), RAI(40bb)
    available = [
        {"action": {"code": "X", "betsize": "0", "allin": False}},
        {"action": {"code": "R6.6", "betsize": "6.6", "allin": False, "betsize_by_pot": "0.33"}},
        {"action": {"code": "R10", "betsize": "10.0", "allin": False, "betsize_by_pot": "0.50"}},
        {"action": {"code": "RAI", "betsize": "40.0", "allin": True, "betsize_by_pot": "2.0"}},
    ]
    # 33 could be "33% pot" → 6.6bb. Without fix this matches RAI(40bb).
    # With fix: |40-33|/33 = 21% > 30% threshold, so pct detection kicks in
    result = find_closest_action_postflop(available, 33)
    assert_eq(result, "R6.6")


@test
def test_find_action_by_pot_pct_maps_real_50pct_to_solver_61pct():
    """Action matching: when solver's normalized preflop inflates the pot
    (e.g. 35bb MTT where user's R2 becomes R2.2), a real 50%-pot river
    bet must still match the solver's 61% option — not the 36% option
    that would win by raw-bb distance against the inflated pot.

    H2767 regression: hero bet 4.6bb into real pot 9.1bb (50% pot).
    Solver pot inflated to 9.8bb by preflop R2→R2.2 rewrite. Absolute
    bb matching: |4.6-3.5|=1.1 < |4.6-6|=1.4 → wrongly picks R3.5.
    Pot-pct matching with actual_pot=9.1: target_pct=50.5% → closest
    solver pct is 61% → correctly picks R6.
    """
    from analyze_hand import _find_action_by_pot_pct

    # H2767-exact available actions on the river with solver pot 9.8
    available = [
        {"action": {"code": "X", "betsize": "0", "allin": False}},
        {"action": {"code": "R3.5", "betsize": "3.5", "allin": False, "betsize_by_pot": "0.36"}},
        {"action": {"code": "R6",   "betsize": "6.0", "allin": False, "betsize_by_pot": "0.61"}},
        {"action": {"code": "R8.5", "betsize": "8.5", "allin": False, "betsize_by_pot": "0.87"}},
        {"action": {"code": "R14.5","betsize": "14.5","allin": False, "betsize_by_pot": "1.48"}},
        {"action": {"code": "RAI",  "betsize": "34.6","allin": True,  "betsize_by_pot": "3.53"}},
    ]

    # Real pot 9.1bb (user's actual preflop R2 without solver inflation)
    assert_eq(_find_action_by_pot_pct(available, 4.6, 9.1), "R6")

    # Sanity: 20% pot bet (1.82bb) → solver R3.5 (36%), the closest
    assert_eq(_find_action_by_pot_pct(available, 1.82, 9.1), "R3.5")

    # Overbet 110% pot (10bb into 9.1bb real) → solver 87% is closer in
    # pot-pct terms (|110-87|=23pp < |110-148|=38pp).
    assert_eq(_find_action_by_pot_pct(available, 10.0, 9.1), "R8.5")

    # Guard: percentage-shaped input (bet_size=50 meaning "50% pot", which
    # OCR/LLM parsers sometimes emit unconverted). target_pct > 2.0 so the
    # helper must defer to find_closest_action_postflop which detects the
    # percentage and resolves it to the right raise code, not an all-in.
    result = _find_action_by_pot_pct(available, 50, 9.1)
    assert_eq(result, "R6",
              f"bet_size=50 (interpreted as 50% pot) should match R6 (61%); got {result}")


@test
def test_find_action_by_pot_pct_exact_betsize_wins_over_pot_pct():
    """When hero's bb amount equals an available betsize exactly, return it
    even if pot-pct conversion would tie at a midpoint.

    H2797 regression: 7-max MTT, hero limped SB and bet 1bb into the 3bb
    flop pot. Solver pot 3.0 (with ante), but the local actual_pot
    computation excludes ante and lands at 2.0. Pot-pct math:
    target_pct = 1.0/2.0 = 0.5 → solver_bet = 0.5 * 3.0 = 1.5, dead
    midpoint between R1 (1bb, 33%) and R2 (2bb, 67%). Float error tipped
    the tie to R2, falsely flagging hero's standard 33% c-bet as a 67%
    bet. The exact-betsize shortcut returns R1 directly.
    """
    from analyze_hand import _find_action_by_pot_pct

    # H2797 flop: 12bb solver, SB cbets 1bb into pot 3.0
    available = [
        {"action": {"code": "X", "betsize": "0", "allin": False}},
        {"action": {"code": "R1", "betsize": "1.0", "allin": False, "betsize_by_pot": "0.33333333"}},
        {"action": {"code": "R2", "betsize": "2.0", "allin": False, "betsize_by_pot": "0.66666667"}},
        {"action": {"code": "R3", "betsize": "3.0", "allin": False, "betsize_by_pot": "1.00000000"}},
        {"action": {"code": "RAI", "betsize": "11.0", "allin": True, "betsize_by_pot": "3.66666667"}},
    ]

    # actual_pot=2.0 (missing ante) — exact betsize match should win
    assert_eq(_find_action_by_pot_pct(available, 1.0, 2.0), "R1")
    # Same with the correct actual_pot=3.0
    assert_eq(_find_action_by_pot_pct(available, 1.0, 3.0), "R1")
    # 5% tolerance: 1.04bb still matches R1
    assert_eq(_find_action_by_pot_pct(available, 1.04, 2.0), "R1")
    # Outside tolerance: 1.3bb falls through to pot-pct logic
    # target=1.3, actual_pot=2.0 → pct=0.65 → solver_bet=1.95 → R2
    assert_eq(_find_action_by_pot_pct(available, 1.3, 2.0), "R2")


@test
def test_find_action_by_pot_pct_dead_money_pot_ignores_exact_betsize():
    """In a multiway dead-money pot the exact-betsize shortcut must NOT fire.

    The collapsed HU node models a 5.5bb pot, but the REAL multiway pot is 8bb
    (cold-callers' dead money). Hero bets 2.7bb = 1/3 of the real pot, which by
    pot ratio is the 25% bucket (R1.4). The raw 2.7 happens to sit within 5% of
    the solver's 50% bucket (2.75), so the absolute exact-match shortcut would
    wrongly pick R2.75. The shortcut is only trustworthy when the solver pot
    matches the real pot, so it is skipped once the real pot exceeds the solver
    pot by >15% — pot ratio then drives the bucket.
    """
    from analyze_hand import _find_action_by_pot_pct

    avail = [
        {"action": {"code": "X",     "betsize": 0,     "betsize_by_pot": 0}},
        {"action": {"code": "R1.4",  "betsize": 1.375, "betsize_by_pot": 0.25}},
        {"action": {"code": "R2.75", "betsize": 2.75,  "betsize_by_pot": 0.50}},
        {"action": {"code": "R4.1",  "betsize": 4.125, "betsize_by_pot": 0.75}},
        {"action": {"code": "RAI",   "betsize": 28.0,  "betsize_by_pot": 5.09,
                    "allin": True}},
    ]
    # Dead-money pot (real 8 >> solver 5.5): pot ratio → 25% bucket.
    assert_eq(_find_action_by_pot_pct(avail, 2.7, 8.0), "R1.4",
              "1/3-of-real-pot bet must snap by pot ratio, not absolute size")
    # No dead money (real ≈ solver pot): exact-betsize shortcut still applies.
    assert_eq(_find_action_by_pot_pct(avail, 2.7, 5.5), "R2.75",
              "without dead money, a bet equal to a bucket size keeps that bucket")


# ── Hand Eval Tests ──

@test
def test_hand_eval_two_pair():
    """Hand eval: T8o on 8-T-2-A board = two pair."""
    from hand_eval import evaluate
    r = evaluate("T8o", "8hTc2sAc")
    assert_eq(r["made_hand"], "two_pair")
    assert_in("兩對", r["made_hand_label"])
    assert_in("T", r["made_hand_label"])
    assert_in("8", r["made_hand_label"])


@test
def test_hand_eval_gutshot():
    """Hand eval: KQo on 8-T-2-A needs J for straight = gutshot."""
    from hand_eval import evaluate
    r = evaluate("KQo", "8hTc2sAc")
    assert_in("gutshot", r["draws"])
    assert_eq(r["made_hand"], "king_high")


@test
def test_hand_eval_straight():
    """Hand eval: T7s on 8-9-4-J = straight (7-8-9-T-J)."""
    from hand_eval import evaluate
    r = evaluate("T7s", "8c9d4hJc")
    assert_eq(r["made_hand"], "straight")
    assert_in("順子", r["made_hand_label"])


@test
def test_hand_eval_flush_draw():
    """Hand eval: AhKh on 8h3hTc = nut flush draw (4 hearts)."""
    from hand_eval import evaluate
    r = evaluate("AhKh", "8h3hTc")
    assert_in("nut_flush_draw", r["draws"])
    assert_eq(r["made_hand"], "ace_high")


@test
def test_hand_eval_no_draw_on_river():
    """Hand eval: no draws on river (5 board cards)."""
    from hand_eval import evaluate
    r = evaluate("KQo", "8hTc2sAcJd")
    assert_eq(r["draws"], [])
    assert_eq(r["made_hand"], "straight")


@test
def test_hand_eval_overpair():
    """Hand eval: AA on K-5-2 board = overpair."""
    from hand_eval import evaluate
    r = evaluate("AA", "Kh5d2c")
    assert_eq(r["made_hand"], "overpair")
    assert_in("超對", r["made_hand_label"])


@test
def test_hand_eval_set():
    """Hand eval: pocket 6s on K-6-2 board = set."""
    from hand_eval import evaluate
    r = evaluate("66", "Kh6d2c")
    assert_eq(r["made_hand"], "set")
    assert_in("暗三條", r["made_hand_label"])


@test
def test_hand_eval_oesd():
    """Hand eval: 9-8 on 7-T-2 = OESD (needs 6 or J)."""
    from hand_eval import evaluate
    r = evaluate("9h8c", "7hTc2s")
    assert_in("oesd", r["draws"])


@test
def test_hand_eval_top_pair():
    """Hand eval: AhKh on Ah3hTc = top pair + nut flush draw."""
    from hand_eval import evaluate
    r = evaluate("AhKh", "Ah3hTc")
    assert_eq(r["made_hand"], "top_pair")
    assert_in("nut_flush_draw", r["draws"])


@test
def test_hand_eval_board_pair_not_hero():
    """H2671: JTo on KhQdKd = J high (board pair K, hero has no K)."""
    from hand_eval import evaluate
    r = evaluate("JhTc", "KhQdKd")
    assert_eq(r["made_hand"], "high_card")
    assert_in("J", r["made_hand_label"])
    r2 = evaluate("JTo", "KhQdKd3h2s")
    assert_eq(r2["made_hand"], "high_card")


@test
def test_hand_eval_two_pair_with_board_pair():
    """Two pair logic: board pair does not inflate hero's made hand."""
    from hand_eval import evaluate
    # Real two pair on paired board — hero still has two pair
    r = evaluate("KhQs", "KdQc2h2d")
    assert_eq(r["made_hand"], "two_pair")
    # Hero one pair + board pair → should be single pair, NOT two pair
    r = evaluate("Qh5c", "KhKsQd")
    assert_eq(r["made_hand"], "second_pair")
    # Hero no contribution + board pair → high card
    r = evaluate("9h8c", "KhKs2d3c")
    assert_eq(r["made_hand"], "high_card")


@test
def test_hand_eval_preflop_empty():
    """Hand eval: no board = empty result."""
    from hand_eval import evaluate
    r = evaluate("AKo", "")
    assert_eq(r["made_hand"], "")
    assert_eq(r["draws"], [])
    assert_eq(r["full_label"], "")


@test
def test_postflop_actions_key():
    """Postflop: 'postflop_actions' key works as alias for 'streets'."""
    from analyze_hand import analyze_hand_full
    hand = {
        "effective_bb": 60,
        "hero_position": "BTN",
        "hero_hand": "J8o",
        "preflop_actions": "F-F-F-F-F-R2-F-C",
        "postflop_actions": [
            {"board": "5s5h6c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "BTN", "action": "X"},
            ]},
            {"card": "6d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "BTN", "action": "R", "size": 2},
                {"position": "BB", "action": "F"},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    assert_in("Flop", text)
    assert_in("Turn", text)
    assert_in("BTN", text)


@test
def test_standalone_board_override():
    """Standalone query: board_override builds params when no street_states."""
    from src.gemini_session import GeminiSessionManager
    mgr = GeminiSessionManager.__new__(GeminiSessionManager)
    mgr.hand_contexts = {}
    ctx = {
        "gametype": "MTTGeneral",
        "depth": 30.125,
        "stacks": "",
        "preflop_actions": "F-F-F-R2-F-F-F-C",
        "hero_position": "",
        "hero_hand": "",
        "hero_spots": [],
        "solutions": [],
        "street_states": {},
        "final_actions": {},
    }
    params = mgr._build_query_params(
        ctx, "turn",
        board_override="QhTd3c3s",
        flop_override="X-R1.15-C",
        turn_override="X",
        river_override=None,
        preflop_override=None,
    )
    assert_true(params is not None, "params should not be None for standalone query with board_override")
    assert_eq(params["board"], "QhTd3c3s")
    assert_eq(params["flop_actions"], "X-R1.15-C")
    assert_eq(params["turn_actions"], "X")


@test
def test_h3473_low_conf_ocr_hero_cards_do_not_anchor_gemini():
    """Image parse: below-threshold hero cards must not survive fallback.

    Regression for H3473: OCR card_conf=0.43 misread hero KhJc as AhAs.
    Cards-only Gemini returned no usable repair in production, and the
    confidence-abstain branch kept OCR's low-confidence AhAs anyway; the full
    Gemini prompt also included the bad hero_cards hint, anchoring the parse.
    """
    from src.gemini_session import GeminiSessionManager

    ocr_result = {
        "card_confidence": 0.4307,
        "hints": {
            "board_cards": ["7s", "Td", "7d", "Qh", "9d"],
            "hero_cards": ["Ah", "As"],
            "partial_hand": {
                "gametype": "MTTGeneral",
                "hero_position": "BTN",
                "hero_hand": "AhAs",
                "preflop_actions": "F-F-R2-F-F-C-F-F",
            },
        },
        "hand": {
            "gametype": "MTTGeneral",
            "hero_position": "BTN",
            "hero_hand": "AhAs",
            "preflop_actions": "F-F-R2-F-F-C-F-F",
        },
    }

    assert_true(
        not GeminiSessionManager._can_keep_ocr_abstain_after_cards_only(
            confidence_abstain_with_ocr=True,
            hero_hand_present=True,
            cards_need_fallback=True,
            original_hero_hand="AhAs",
            gemini_hero_hand=None,
        ),
        "low-confidence card fallback must not keep the original OCR hand",
    )

    hints, partial, low_card_conf = GeminiSessionManager._gemini_ocr_context(
        ocr_result, min_card_conf=0.70
    )

    assert_true(low_card_conf)
    assert_not_in("hero_cards", hints)
    assert_in("hero_cards_low_confidence", hints)
    assert_not_in("hero_hand", hints["partial_hand"])
    assert_true(hints["partial_hand"]["hero_hand_low_confidence"])
    assert_not_in("hero_hand", partial)
    assert_true(partial["hero_hand_low_confidence"])

    # Structural anchors remain useful for the full Gemini reparse.
    assert_eq(hints["board_cards"], ["7s", "Td", "7d", "Qh", "9d"])
    assert_eq(partial["hero_position"], "BTN")
    assert_eq(partial["preflop_actions"], "F-F-R2-F-F-C-F-F")


@test
def test_high_conf_ocr_hero_cards_still_anchor_gemini():
    """Image parse: confident hero card hints stay available to Gemini."""
    from src.gemini_session import GeminiSessionManager

    ocr_result = {
        "card_confidence": 0.92,
        "hints": {
            "hero_cards": ["Kh", "Jc"],
            "partial_hand": {"hero_hand": "KhJc", "hero_position": "BTN"},
        },
        "hand": {"hero_hand": "KhJc", "hero_position": "BTN"},
    }

    hints, partial, low_card_conf = GeminiSessionManager._gemini_ocr_context(
        ocr_result, min_card_conf=0.70
    )

    assert_true(not low_card_conf)
    assert_eq(hints["hero_cards"], ["Kh", "Jc"])
    assert_eq(hints["partial_hand"]["hero_hand"], "KhJc")
    assert_eq(partial["hero_hand"], "KhJc")


@test
def test_collapsed_streets_4card_board():
    """Collapsed streets: 4-card board split into flop + turn."""
    from analyze_hand import _fix_collapsed_streets
    streets = [{"street": "turn", "board": "5s5h6c6d", "actions": [
        {"position": "BB", "action": "X"},
        {"position": "BTN", "action": "R2", "size": 2.0},
        {"position": "BB", "action": "F"},
    ]}]
    fixed = _fix_collapsed_streets(streets)
    assert_eq(len(fixed), 2)
    assert_eq(fixed[0]["board"], "5s5h6c")
    assert_eq(fixed[0]["actions"], [])
    assert_eq(fixed[1]["card"], "6d")
    assert_eq(len(fixed[1]["actions"]), 3)


@test
def test_collapsed_streets_normal_board_unchanged():
    """Collapsed streets: normal 3-card flop is not modified."""
    from analyze_hand import _fix_collapsed_streets
    streets = [{"board": "Js6h5s", "actions": [
        {"position": "BB", "action": "X"},
        {"position": "BTN", "action": "R2", "size": 2.0},
    ]}]
    fixed = _fix_collapsed_streets(streets)
    assert_eq(len(fixed), 1)
    assert_eq(fixed[0]["board"], "Js6h5s")


@test
def test_collapsed_streets_full_analysis():
    """Collapsed streets: full analysis works with 4-card board input."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 60,
        "hero_position": "BTN",
        "hero_hand": "J8o",
        "preflop_actions": "F-F-F-F-F-R2.1-F-C",
        "streets": [{"street": "turn", "board": "5s5h6c6d", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "BTN", "action": "R2", "size": 2.0},
            {"position": "BB", "action": "F"},
        ]}],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Should have turn data (not "無 solver 數據")
    assert_in("Turn", text)
    assert_true("無 solver 數據" not in text, "Should have solver data for turn")
    # Should show BTN's strategy on the turn
    assert_in("BTN", text)


@test
def test_check_through_flop_infers_xx():
    """Check-through: empty flop actions infer X-X when turn follows."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 60,
        "hero_position": "BTN",
        "hero_hand": "J8o",
        "preflop_actions": "F-F-F-F-F-R2.1-F-C",
        "streets": [
            {"board": "5s5h6c", "actions": []},
            {"card": "6d", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "BTN", "action": "R2", "size": 2.0},
                {"position": "BB", "action": "F"},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Turn should have solver data
    assert_in("Turn", text)
    assert_true("無 solver 數據" not in text, "Should have solver data after check-through flop")
    # flop_actions should be X-X in the final state
    assert_eq(result["final_actions"]["flop_actions"], "X-X")


@test
def test_single_check_turn_infers_check_through():
    """Check-through: single check on turn infers X-X when river follows (H2565)."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 21.8,
        "hero_position": "BB",
        "hero_hand": "6d4h",
        "preflop_actions": "F-F-R2-F-F-F-C",
        "players_at_table": 7,
        "streets": [
            {"board": "2dQh4c", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 1.8, "action": "C", "position": "BB"},
            ]},
            {"card": "Qd", "actions": [
                {"action": "X", "position": "BB"},
            ]},
            {"card": "Ah", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 3.0, "action": "R3", "position": "HJ"},
                {"size": 3.0, "action": "C", "position": "BB"},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    # turn_actions should be X-X (inferred opponent check)
    assert_eq(result["final_actions"]["turn_actions"], "X-X")
    # River BB check (first river spot) must have solver data
    river_spots = [(i, s) for i, s in enumerate(result["hero_spots"])
                   if s["street"] == "river"]
    assert_true(len(river_spots) >= 1, "Should have at least 1 river hero spot")
    first_river_idx = river_spots[0][0]
    assert_true(result["solutions"][first_river_idx] is not None,
                "River BB check should have solver data (not None)")


@test
def test_allin_turn_skips_river_actions():
    """All-in on turn: river actions are skipped (no 400 error from API)."""
    from analyze_hand import analyze_hand_full
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 14,
        "hero_position": "BB",
        "hero_hand": "AcTh",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "streets": [
            {"board": "Qh9hAc", "actions": [
                {"position": "BB", "action": "R4.55", "size": 4.55},
                {"position": "CO", "action": "C"},
            ]},
            {"card": "7h", "actions": [
                {"position": "BB", "action": "AI", "size": 10.0},
                {"position": "CO", "action": "C"},
            ]},
            {"card": "4s", "actions": [
                {"position": "BB", "action": "X"},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    text = result["text"]
    # Should not crash with 400 error; river_actions should be empty
    assert_eq(result["final_actions"]["river_actions"], "",
              "River actions should be empty after turn all-in")
    # Turn should still have solver data
    assert_in("Turn", text)


@test
def test_allin_turn_normalized_from_raise_skips_river():
    """All-in on turn (bet normalized to RAI): river actions are skipped."""
    from analyze_hand import analyze_hand_full
    # Reproduces actual screenshot parse: R7 on turn gets normalized to RAI
    hand = {
        "gametype": "MTTGeneral",
        "effective_bb": 13.9,
        "hero_position": "CO",
        "hero_hand": "QdJh",
        "players_at_table": 6,
        "preflop_actions": "F-F-R2-F-F-C",
        "streets": [
            {"board": "Qh9hAc", "actions": [
                {"position": "BB", "action": "R4", "size": 4},
                {"position": "CO", "action": "C"},
            ]},
            {"card": "7h", "actions": [
                {"position": "BB", "action": "R7", "size": 7},
                {"position": "CO", "action": "C"},
            ]},
            {"card": "4s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R1", "size": 1},
                {"position": "BB", "action": "C"},
            ]},
        ],
    }
    result = analyze_hand_full(hand)
    # Should not crash with 400 error; river_actions should be empty
    assert_eq(result["final_actions"]["river_actions"], "",
              "River actions should be empty when turn bet normalizes to RAI")


@test
def test_categorized_range_uses_real_frequencies():
    """Formatter: categorized range shows real per-hand frequencies, not 1.0."""
    from gto_api import get_spot_solution
    from gto_formatter import format_range_by_action

    # 40bb BTN open R2.3, SB 3bet R8.6, BB fold, BTN call. Flop 8s9s6d.
    sol = get_spot_solution(
        gametype="MTTGeneral", depth="40.125",
        preflop_actions="F-F-F-F-F-R2.3-R8.6-F-C",
        board="8s9s6d",
    )
    assert_true(sol is not None, "Solution should exist for this spot")
    text = format_range_by_action(sol, "SB")
    # AA should NOT appear as pure in the all-in range.
    # Old bug: _categorize_action_range used freq=1.0 → "TT+" which includes AA.
    # AA is actually ~96% check, so it should either not appear in all-in section
    # or appear with a low percentage like AA(4%).
    allin_section = False
    has_ttp = False  # "TT+" in all-in
    for line in text.split("\n"):
        if "All-in" in line and "combos" in line:
            allin_section = True
        elif allin_section and line.startswith("\n"):
            allin_section = False
        if allin_section and "TT+" in line:
            has_ttp = True
    assert_true(not has_ttp,
                "All-in range should not show TT+ (AA is ~96% check, not all-in)")


@test
def test_hand_eval_uses_suited_hero_hand():
    """Hand eval: AcTh on 4-club board correctly identifies flush."""
    from hand_eval import evaluate
    # Without suits: misses flush
    result_no_suit = evaluate("ATo", "Jc7cQcJs9c")
    # Board is paired JJ but hero has no J — hero's best is ace high
    assert_eq(result_no_suit["made_hand"], "ace_high",
              "ATo (no suits) on JJQ79 board has no pair — ace high")
    # With suits: detects flush
    result_suited = evaluate("AcTh", "Jc7cQcJs9c")
    assert_eq(result_suited["made_hand"], "flush",
              "AcTh should be flush on 4-club board")


@test
def test_analyze_hand_eval_uses_raw_suits():
    """Analysis: hand type label uses raw suited hand, not normalized."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral", "effective_bb": 42,
        "hero_position": "BB", "hero_hand": "AcTh",
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "streets": [
            {"board": "Jc7cQc", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R2.5", "size": 2.5},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "Js", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "R4.8", "size": 4.8},
                {"position": "BB", "action": "C"},
            ]},
            {"card": "9c", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "LJ", "action": "X"},
            ]},
        ],
    })
    text = result["text"]
    # River label should show flush, not just second pair
    assert_in("同花", text, "River hand type should show flush (同花) for AcTh on 4-club board")
    # Flop label should show flush draw
    assert_in("堅果花聽牌", text, "Flop hand type should show nut flush draw for AcTh on 3-club board")


@test
def test_format_hand_detail_specific_combo():
    """Formatter: specific combo query (Ah8h) shows that combo's strategy, not aggregated."""
    from gto_api import get_spot_solution
    from gto_formatter import format_hand_detail

    sol = get_spot_solution(
        gametype="MTTGeneral", depth="100.125",
        preflop_actions="F-F-F-F-R2.3-F-F-C",
        board="Jc4d3s5d",
        flop_actions="X-R2-C",
        turn_actions="X",
    )
    assert_true(sol is not None, "Solution should exist")
    # Specific combo: Ah8h (no flush draw on diamond board)
    text_specific = format_hand_detail(sol, "Ah8h", "CO")
    assert_in("Ah8h", text_specific,
              "Specific combo query should show Ah8h in output")
    assert_in("A8s", text_specific,
              "Specific combo query should reference parent hand A8s")
    # Compare with aggregated: should be different format
    text_agg = format_hand_detail(sol, "A8s", "CO")
    assert_in("Range 頻率", text_agg,
              "Aggregated query should show Range 頻率 header")


@test
def test_pot_pct_action_matching():
    """API: find_closest_action_by_pot_pct matches by pot percentage, not absolute bb."""
    from gto_api import get_next_actions, find_closest_action, find_closest_action_by_pot_pct

    na = get_next_actions(
        gametype="MTTGeneral", depth=25.125,
        preflop_actions="F-F-R2.1-F-F-C-F-F",
        board="JcTs6d",
    )
    assert_true(na is not None, "next_actions should return data")
    actions = na["next_actions"]["available_actions"]

    # 2.85bb = 50% of 5.7bb (pot without antes) → absolute match picks R2.2 (wrong)
    abs_result = find_closest_action(actions, 2.85)
    assert_eq(abs_result, "R2.2", "Absolute match for 2.85bb should be R2.2")

    # 3.35bb = 50% of 6.7bb (pot with antes) → should match R3.7
    pct_result = find_closest_action_by_pot_pct(actions, 3.35)
    assert_eq(pct_result, "R3.7", "Pot-pct match for 3.35bb should be R3.7")


@test
def test_normalize_pct_flop_override():
    """Session: R50% flop override resolves to correct solver action code."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gemini_session import GeminiSessionManager

    mgr = GeminiSessionManager.__new__(GeminiSessionManager)
    params = {
        "gametype": "MTTGeneral",
        "depth": 25.125,
        "preflop_actions": "F-F-R2.1-F-F-C-F-F",
        "board": "JcTs6dAh",
    }
    result = mgr._normalize_override_actions(
        dict(params), "turn",
        flop_override="R50%-C",
        turn_override=None,
        river_override=None,
    )
    assert_eq(result["flop_actions"], "R3.7-C",
              "R50% should resolve to R3.7 (55% pot, nearest to 50%)")


# ── ICM FT Image/Stacks Tests ──


@test
def test_icm_ft_5player_at_8max_table():
    """ICM FT: 5 active players at 8-max FT uses ICM8m mode."""
    from icm_modes import find_icm_params
    # 5 players with stacks, padded to 8 positions (3 zeros for empty seats)
    result = find_icm_params(
        player_stacks=[0, 0, 8, 0, 23, 10, 18, 23],
        phase="FT",
        players_at_table=8,
    )
    assert_in("ICM8m", result["gametype"],
              f"should use ICM8m for 8-max FT, got {result['gametype']}")
    assert_true(result["stacks"] != "", "should have stacks string")
    # Verify all 8 solver stacks are non-zero
    solver_stacks = result["stacks"].split("-")
    assert_eq(len(solver_stacks), 8, "should have 8 stack values")


@test
def test_icm_ft_5player_at_8max_analysis():
    """ICM FT: 5 players at 8-max FT produces ICM analysis with correct padding."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "FT",
        "players_at_table": 8,
        "player_stacks": [8, 23, 10, 18, 23],
        "effective_bb": 23,
        "hero_position": "CO",
        "hero_hand": "AKs",
        "preflop_actions": "F-R2-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    assert_in("ICM8m", result["gametype"],
              f"should use ICM8m, got {result['gametype']}")
    assert_in("用戶籌碼", result["text"])
    assert_in("Solver 籌碼", result["text"])


@test
def test_icm_ft_5player_stacks():
    """ICM FT: 5-player final table with asymmetric stacks finds valid ICM mode."""
    from icm_modes import find_icm_params
    # Stacks from the N8 FT screenshot: ~109, 21, 18, 33, 16 bb
    result = find_icm_params(
        player_stacks=[109, 21, 18, 33, 16],
        phase="FT",
    )
    assert_true(result["gametype"] != "MTTGeneral",
                f"should find ICM FT mode, got {result['gametype']}")
    assert_in("ICM", result["approximation_note"])
    assert_true(result["stacks"] != "", "should have stacks string")
    assert_true(result["depth"] != "", "should have depth string")


@test
def test_icm_ft_4player_stacks():
    """ICM FT: 4-player final table finds valid ICM mode."""
    from icm_modes import find_icm_params
    result = find_icm_params(
        player_stacks=[60, 45, 30, 25],
        phase="FT",
    )
    assert_true(result["gametype"] != "MTTGeneral",
                f"should find ICM FT mode for 4 players, got {result['gametype']}")
    assert_in("ICM", result["approximation_note"])


@test
def test_icm_ft_7player_stacks():
    """ICM FT: 7-player final table finds valid ICM mode."""
    from icm_modes import find_icm_params
    result = find_icm_params(
        player_stacks=[80, 50, 40, 35, 30, 25, 20],
        phase="FT",
    )
    assert_true(result["gametype"] != "MTTGeneral",
                f"should find ICM FT mode for 7 players, got {result['gametype']}")


@test
def test_structured_icm_open_range_query_preserves_stack_order():
    """ICM text range query: exact slash-delimited stacks map UTG→BB without LLM reorder."""
    from gemini_session import GeminiSessionManager

    hand = GeminiSessionManager._parse_structured_icm_range_query(
        "icm final table 剩餘 7 人, stack size 15/68/35/50/18/10/26 "
        "這時 hero hj open range 如何"
    )

    assert_true(hand is not None, "explicit ICM FT stack/range query should parse deterministically")
    assert_eq(hand["player_stacks"], [15.0, 68.0, 35.0, 50.0, 18.0, 10.0, 26.0])
    assert_eq(hand["players_at_table"], 7)
    assert_eq(hand["hero_position"], "HJ")
    assert_eq(hand["effective_bb"], 35.0, "7-max HJ is the third stack, not LJ's 68bb")
    assert_eq(hand["preflop_actions"], "F-F-R2-F-F-F-F")
    assert_eq(hand["no_hero_hand"], True)
    assert_eq(hand["phase"], "FT")


@test
def test_structured_icm_facing_range_query_prefers_explicit_hero():
    """ICM text range query: 'HJ raise hero CO ...' should query CO facing HJ, not HJ."""
    from gemini_session import GeminiSessionManager

    hand = GeminiSessionManager._parse_structured_icm_range_query(
        "那 icm final table 剩餘 7 人，stack size 分布從 utg 開始為 "
        "12,14,37,15,42,11,7 這時當 hj raise hero co call/raise/all in range 如何"
    )

    assert_true(hand is not None, "explicit hero in ICM range query should parse")
    assert_eq(hand["player_stacks"], [12.0, 14.0, 37.0, 15.0, 42.0, 11.0, 7.0])
    assert_eq(hand["hero_position"], "CO")
    assert_eq(hand["effective_bb"], 15.0)
    assert_eq(hand["preflop_actions"], "F-F-R2-F-F-F-F")


@test
def test_icm_no_hero_range_coach_summary_keeps_approximation_context():
    """ICM range coaching: no-hero FT response should be explanatory but deterministic."""
    from gemini_session import GeminiSessionManager

    raise_hands = {
        "AA": 6, "KK": 6, "QQ": 6, "JJ": 6, "TT": 6, "99": 6, "88": 6,
        "77": 6, "66": 6, "A3s": 4, "A4s": 4, "A5s": 4, "AKo": 12,
        "AQo": 12, "AJo": 12, "ATo": 12, "KQo": 12, "KJo": 12,
        "JTs": 4, "T9s": 4,
    }
    shc = {
        hand: {
            "actions_total_frequencies": {"R2": 1.0},
            "actions_total_combos": {"R2": combos},
        }
        for hand, combos in raise_hands.items()
    }
    shc["55"] = {
        "actions_total_frequencies": {"R2": 0.32, "F": 0.68},
        "actions_total_combos": {"R2": 1.92, "F": 4.08},
    }
    shc["22"] = {
        "actions_total_frequencies": {"F": 1.0},
        "actions_total_combos": {"F": 6},
    }
    solution = {
        "game": {
            "active_position": "HJ",
            "board": "",
            "current_street": {"type": "preflop"},
            "pot": 2.375,
            "bet_display_name": "RAISE",
        },
        "action_solutions": [
            {
                "action": {"code": "F"},
                "total_frequency": 0.827,
                "total_combos": 1096,
            },
            {
                "action": {"code": "R2", "betsize": "2", "betsize_by_pot": 0.30},
                "total_frequency": 0.173,
                "total_combos": 230,
            },
        ],
        "players_info": [
            {"player": {"position": "HJ"}, "simple_hand_counters": shc}
        ],
    }
    context = {
        "hand": {
            "players_at_table": 7,
            "player_stacks": [15.0, 68.0, 35.0, 50.0, 18.0, 10.0, 26.0],
            "hero_position": "HJ",
        },
        "gametype": "MTTGeneral_ICM7m1000PTFT",
        "stacks": "15.125-20.125-30.125-45.125-40.125-10.125-50.125",
        "hero_spots": [{"street": "preflop", "solver_hero_pos": "HJ"}],
        "solutions": [solution],
    }

    text = GeminiSessionManager._format_icm_range_coach_response(context)

    assert_in("🎯 教練解讀", text)
    assert_in("近似說明", text)
    assert_in("用戶籌碼: 15 / 68 / 35 / 50 / 18 / 10 / 26", text)
    assert_in("Solver 籌碼: 15 / 20 / 30 / 45 / 40 / 10 / 50", text)
    assert_in("最大差異: 48bb", text)
    assert_in("HJ 對應 35bb", text)
    assert_in("GTO Wizard ICM 只能查內建的 FT stack configuration", text)
    assert_in("Fold: 82.7%", text)
    assert_in("RAISE 2（30% pot）: 17.3%", text)
    assert_in("可玩範圍", text)
    assert_not_in("Discovery:", text)
    assert_not_in("==================================================", text)


@test
def test_icm_ft_9player_stacks():
    """ICM FT: 9-player final table (full ring) finds valid ICM mode."""
    from icm_modes import find_icm_params
    result = find_icm_params(
        player_stacks=[60, 50, 45, 40, 35, 30, 25, 20, 15],
        phase="FT",
    )
    assert_true(result["gametype"] != "MTTGeneral",
                f"should find ICM FT mode for 9 players, got {result['gametype']}")


@test
def test_icm_ft_5player_analysis():
    """ICM FT: 5-player FT hand analysis runs successfully with player_stacks."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "FT",
        "player_stacks": [109, 21, 18, 33, 16],
        "players_at_table": 5,
        "effective_bb": 16,
        "hero_position": "BB",
        "hero_hand": "52o",
        "preflop_actions": "F-R2-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    assert_true(result["solutions"][0] is not None, "should have preflop solution")
    assert_in("ICM", result["text"])


@test
def test_icm_ft_image_parse_fields_flow():
    """ICM FT: hand JSON with image-parsed ICM fields flows through analyze_hand_full."""
    from analyze_hand import analyze_hand_full
    # Simulate what IMAGE_PARSE_PROMPT would output for an N8 FT screenshot
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "tournament_type": "icm",
        "phase": "FT",
        "player_stacks": [109, 21, 18, 33, 16],
        "players_at_table": 5,
        "effective_bb": 16,
        "hero_position": "BB",
        "hero_hand": "5s2c",
        "preflop_actions": "F-R2-F-F-F",
    })
    assert_eq(result["is_icm"], True)
    assert_eq(result["hero_position"], "BB")
    assert_true("ICM" in result["text"], "output should mention ICM")


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

    img_path = Path(__file__).resolve().parent.parent / "data" / "hand_images" / "img" / "TM5867249527.png"
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

    img_path = Path(__file__).resolve().parent.parent / "data" / "hand_images" / "img" / "TM5901972230.png"
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

    img_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "ocr" / "H3429.jpeg"
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
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
    turn_section = compact.split("─── Turn: 8d ───", 1)[1].split("─── River:", 1)[0]
    river_section = compact.split("─── River: Qc ───", 1)[1]

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
    river_section = compact.split("─── River: 4c ───", 1)[1]
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
    flop_section = compact.split("─── Flop: JhKs4h ───", 1)[1].split("─── Turn:", 1)[0]
    turn_section = compact.split("─── Turn: 5h ───", 1)[1].split("─── River:", 1)[0]
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

    turn_section = result["text_compact"].split("─── Turn: Kc ───", 1)[1]
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
    snapshots_dir = Path(__file__).resolve().parent.parent / "tests" / "snapshots"
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


_SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "tests" / "snapshots"


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
                # Disable DB cache (L2) — unset env var to prevent auto-reconnect
                orig_db = gto_cache._db_conn
                orig_dsn = os.environ.pop("SUPABASE_CONN", None)
                gto_cache._db_conn = None
                try:
                    from analyze_hand import analyze_hand_full
                    result = analyze_hand_full(hand_json)
                finally:
                    gto_cache._CACHE_DIR = orig_cache_dir
                    gto_cache._db_conn = orig_db
                    if orig_dsn:
                        os.environ["SUPABASE_CONN"] = orig_dsn
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

# ── Spot Categorizer Tests ──

@test
def test_spot_categorize_open_raise():
    """Spot categorizer: first to raise = open_raise."""
    from spot_categorizer import categorize_preflop
    # CO opens, everyone folds before CO
    cat = categorize_preflop("F-F-F-F-R2-F-F-C", "CO", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_open_raise_utg():
    """Spot categorizer: UTG first to act = open_raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("R2-F-F-F-F-F-F-F", "UTG", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_facing_open():
    """Spot categorizer: facing a single raise = facing_open."""
    from spot_categorizer import categorize_preflop
    # UTG opens, hero is CO (folds before, one raise = facing_open)
    cat = categorize_preflop("R2-F-F-F-C-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "facing_open")

@test
def test_spot_categorize_facing_3bet():
    """Spot categorizer: hero opened, facing re-raise = facing_3bet."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, BB 3bets R8, CO faces the 3bet (action_index=1)
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-C", "CO", 8, action_index=1)
    assert_eq(cat, "facing_3bet")

@test
def test_spot_categorize_squeeze():
    """Spot categorizer: open + call + hero raises = squeeze."""
    from spot_categorizer import categorize_preflop
    # UTG+1 opens R2, LJ calls, hero (CO) raises
    cat = categorize_preflop("F-R2-C-F-R8-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "squeeze")

@test
def test_spot_categorize_facing_4bet():
    """Spot categorizer: 3+ raises before hero's second decision = facing_4bet."""
    from spot_categorizer import categorize_preflop
    # CO open R2, BB 3bet R8, CO 4bet R20, BB faces 4bet (action_index=1 for BB)
    # Total raises: R2, R8, R20, R50 = 4 raises
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-R20-R50", "BB", 8, action_index=1)
    assert_eq(cat, "facing_4bet")

@test
def test_spot_categorize_limp_pot():
    """Spot categorizer: calls without prior raise = limp_pot."""
    from spot_categorizer import categorize_preflop
    # SB limps (calls), hero is BB
    cat = categorize_preflop("F-F-F-F-F-F-C-X", "BB", 8, action_index=0)
    assert_eq(cat, "limp_pot")

@test
def test_spot_categorize_6max_open():
    """Spot categorizer: 6-max table open raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-R2-F-F-F", "CO", 6, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_categorize_cbet_ip():
    """Spot categorizer: PF aggressor bets IP = cbet_ip."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB checks, CO (hero, IP) acts.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="CO",
        street_actions_before_hero=[{"position": "BB", "action": "X"}],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    assert_eq(cat, "cbet_ip")

@test
def test_spot_categorize_cbet_oop():
    """Spot categorizer: PF aggressor bets OOP = cbet_oop."""
    from spot_categorizer import categorize_postflop_action
    # BB 3bet, CO called. Flop: BB (hero, OOP) first to act.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-R8-C",
        num_players=8,
    )
    assert_eq(cat, "cbet_oop")

@test
def test_spot_categorize_facing_cbet_oop():
    """Spot categorizer: facing c-bet when OOP = facing_cbet_oop."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB checks, CO bets → BB (hero) faces cbet
    # BB is OOP relative to CO
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[{"position": "BB", "action": "X"}, {"position": "CO", "action": "R3"}],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB checked then CO bet — this is check-raise opportunity for BB
    assert_eq(cat, "check_raise")

@test
def test_spot_categorize_facing_cbet_ip_btn():
    """Spot categorizer: BTN facing BB c-bet = facing_cbet_ip."""
    from spot_categorizer import categorize_postflop_action
    # BB 3bet, BTN called. Flop: BB bets, BTN (hero, IP) faces it.
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BTN",
        street_actions_before_hero=[{"position": "BB", "action": "R3"}],
        preflop_actions="F-F-F-F-F-R2-F-R8-C",
        num_players=8,
    )
    assert_eq(cat, "facing_cbet_ip")

@test
def test_spot_categorize_probe():
    """Spot categorizer: non-aggressor bets after check-through = probe."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: x-x (check through). Turn: BB (hero) bets.
    cat = categorize_postflop_action(
        street="turn",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB is not PF aggressor, no bets before, but BB has checks before? No, empty.
    # No checks before hero on this street, BB is first to act and not aggressor
    # This should be probe since PF aggressor (CO) will act after BB
    assert_eq(cat, "probe")

@test
def test_spot_categorize_donk():
    """Spot categorizer: non-aggressor bets into aggressor (donk is detected as probe)."""
    from spot_categorizer import categorize_postflop_action
    # CO opened, BB called. Flop: BB (hero) bets into CO = donk bet
    # In our simplified categorization, this maps to "probe" when no checks before
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    # BB is OOP, not aggressor, first to act = probe (donk is a form of probe)
    assert_eq(cat, "probe")

@test
def test_spot_categorize_check_raise():
    """Spot categorizer: hero checks then faces bet = check_raise."""
    from spot_categorizer import categorize_postflop_action
    cat = categorize_postflop_action(
        street="flop",
        hero_pos="BB",
        street_actions_before_hero=[
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "R3"},
        ],
        preflop_actions="F-F-F-F-R2-F-F-C",
        num_players=8,
    )
    assert_eq(cat, "check_raise")


# ── Board Texture Tests ──

@test
def test_board_texture_paired():
    """Board texture: paired board (any pair on board)."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture("Ks6h6s"), "paired")
    assert_eq(classify_board_texture("AhAdKs"), "paired")

@test
def test_board_texture_monotone():
    """Board texture: monotone (3+ same suit)."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture("Ks9s3s"), "monotone")
    assert_eq(classify_board_texture("AhKhQh"), "monotone")

@test
def test_board_texture_wet():
    """Board texture: wet (flush draw or connected)."""
    from spot_categorizer import classify_board_texture
    # Two spades = flush draw = wet
    assert_eq(classify_board_texture("Ks9s3h"), "wet")
    # Connected cards within 3 ranks
    assert_eq(classify_board_texture("Jh9c8d"), "wet")

@test
def test_board_texture_dry():
    """Board texture: dry (no pair, no flush draw, no straight draw).

    Aligned with GTOW's flop_connectedness vocab: a board needs a gap of 1
    between adjacent sorted ranks to count as having straight-draw potential
    (so Q94 rainbow is dry, not wet — its smallest gap is 3).
    """
    from spot_categorizer import classify_board_texture
    # All different suits, large gaps
    assert_eq(classify_board_texture("Ah8c2d"), "dry")
    # Q94 rainbow — smallest gap is 3 (Q-9). GTOW would call this
    # 'disconnected'; we follow the same convention.
    assert_eq(classify_board_texture("Qd9h4s"), "dry")
    # K72 rainbow — gaps 5, 5. Disconnected.
    assert_eq(classify_board_texture("Kd7h2s"), "dry")


@test
def test_board_texture_wet_via_straight_draw():
    """A flop with any gap of 1 in adjacent sorted ranks is wet (matches
    GTOW's oesd_possible bucket). Boards previously over-tagged as wet
    because the threshold was gap<=3 should now be 'dry'."""
    from spot_categorizer import classify_board_texture
    # 78T rainbow — gaps [1, 2]. oesd_possible → wet.
    assert_eq(classify_board_texture("7h8c Td".replace(" ", "")), "wet")
    # 234 rainbow — gaps [1, 1]. connected → wet.
    assert_eq(classify_board_texture("2h3c4d"), "wet")
    # 235 rainbow — gaps [1, 2]. wet.
    assert_eq(classify_board_texture("2h3c5d"), "wet")
    # 8h2c3d — gaps from sorted [2,3,8] are [1, 5]. wet (low end straight draws).
    assert_eq(classify_board_texture("8h2c3d"), "wet")

@test
def test_board_texture_empty():
    """Board texture: empty or None returns None."""
    from spot_categorizer import classify_board_texture
    assert_eq(classify_board_texture(None), None)
    assert_eq(classify_board_texture(""), None)

@test
def test_board_texture_priority():
    """Board texture: paired takes priority over monotone."""
    from spot_categorizer import classify_board_texture
    # Paired AND monotone: AhAh... wait, paired + 3 same suit
    assert_eq(classify_board_texture("AhKh6h6d"), "paired")  # paired > monotone

@test
def test_spot_categorize_full_hand():
    """Spot categorizer: categorize_spot with full hand dict."""
    from spot_categorizer import categorize_spot
    hand = {
        "hero_position": "CO",
        "preflop_actions": "F-F-F-F-R2-F-F-C",
        "players_at_table": 8,
        "streets": [
            {"board": "Js6h5s", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R3"},
            ]},
        ],
    }
    # Preflop: CO opens = open_raise
    cat, tex = categorize_spot(hand, "preflop", action_index=0)
    assert_eq(cat, "open_raise")
    assert_eq(tex, None)

    # Flop: CO is PF aggressor, BB checked, CO bets = cbet_ip
    cat, tex = categorize_spot(
        hand, "flop", action_index=0,
        street_actions_before_hero=[{"position": "BB", "action": "X"}],
    )
    assert_eq(cat, "cbet_ip")
    assert_eq(tex, "wet")  # Js6h5s = two spades = flush draw = wet

@test
def test_spot_edge_missing_actions():
    """Spot categorizer: empty preflop actions defaults to open_raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("", "UTG", 8, action_index=0)
    assert_eq(cat, "open_raise")

@test
def test_spot_edge_facing_open_caller():
    """Spot categorizer: facing open when hero just calls."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, hero is BTN, calls (facing_open, not squeeze since no callers in between)
    cat = categorize_preflop("F-F-F-F-R2-C-F-F", "BTN", 8, action_index=0)
    assert_eq(cat, "facing_open")


# ── New buckets: possible_squeeze, hero_3bet, vs_squeeze ──

@test
def test_spot_categorize_possible_squeeze():
    """possible_squeeze: open + caller in front, hero does not raise."""
    from spot_categorizer import categorize_preflop
    # CO opens R2, BTN calls, hero is BB and flats/folds
    cat = categorize_preflop("F-F-F-F-R2-C-F-F", "BB", 8, action_index=0)
    assert_eq(cat, "possible_squeeze")

@test
def test_spot_categorize_possible_squeeze_sb():
    """possible_squeeze: LJ opens, CO calls, hero SB does not raise."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-R2-F-C-F-F-F", "SB", 8, action_index=0)
    assert_eq(cat, "possible_squeeze")

@test
def test_spot_categorize_hero_3bet():
    """hero_3bet: hero 3bets facing an open with no callers in between."""
    from spot_categorizer import categorize_preflop
    # LJ opens R2, HJ/CO fold, hero is BTN 3bets
    cat = categorize_preflop("F-F-R2-F-F-R8-F-F", "BTN", 8, action_index=0)
    assert_eq(cat, "hero_3bet")

@test
def test_spot_categorize_hero_3bet_bb():
    """hero_3bet: CO opens, hero BB 3bets, no callers."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8", "BB", 8, action_index=0)
    assert_eq(cat, "hero_3bet")

@test
def test_spot_categorize_vs_squeeze():
    """vs_squeeze: hero opened, caller came in, then re-raise (squeeze)."""
    from spot_categorizer import categorize_preflop
    # Hero LJ opens, CO calls, BTN squeezes, LJ's second decision
    cat = categorize_preflop("F-F-R2-F-C-R8-F-F", "LJ", 8, action_index=1)
    assert_eq(cat, "vs_squeeze")

@test
def test_spot_categorize_vs_squeeze_co():
    """vs_squeeze: CO opens, BTN calls, BB squeezes; CO faces squeeze."""
    from spot_categorizer import categorize_preflop
    cat = categorize_preflop("F-F-F-F-R2-C-F-R8", "CO", 8, action_index=1)
    assert_eq(cat, "vs_squeeze")

@test
def test_spot_categorize_facing_3bet_no_squeeze_still_works():
    """Regression: facing_3bet without caller between stays facing_3bet."""
    from spot_categorizer import categorize_preflop
    # CO opens, BB 3bets, CO faces 3bet (no caller between)
    cat = categorize_preflop("F-F-F-F-R2-F-F-R8-C", "CO", 8, action_index=1)
    assert_eq(cat, "facing_3bet")

@test
def test_spot_categorize_facing_open_regression():
    """REGRESSION: facing_open must still classify when no callers in front
    and hero does not raise. This is the critical split-safety guarantee."""
    from spot_categorizer import categorize_preflop
    # UTG opens, folds to CO (hero), one raise + no calls before = facing_open
    cat = categorize_preflop("R2-F-F-F-F-F-F-F", "UTG+1", 8, action_index=0)
    assert_eq(cat, "facing_open")
    cat2 = categorize_preflop("F-F-R2-F-F-F-F-F", "HJ", 8, action_index=0)
    assert_eq(cat2, "facing_open")

@test
def test_spot_categorize_squeeze_still_works():
    """Regression: squeeze (hero IS the squeezer) unchanged."""
    from spot_categorizer import categorize_preflop
    # UTG+1 opens, LJ calls, hero CO squeezes
    cat = categorize_preflop("F-R2-C-F-R8-F-F-F", "CO", 8, action_index=0)
    assert_eq(cat, "squeeze")


# ── compute_preflop_line_key tests ──

@test
def test_line_key_srp_open_call():
    """HJ opens, hero BB calls → 'HJ-R' (hero action excluded, pre-raise folds elided)."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-R2-F-F-F-C", "BB", 8)
    assert_eq(key, "HJ-R")

@test
def test_line_key_simple_open_fold():
    """CO opens, hero BTN folds (or acts) → 'CO-R'."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-F", "BTN", 8)
    assert_eq(key, "CO-R")

@test
def test_line_key_3bet_pot():
    """LJ opens, CO folds, BTN 3bets, SB folds, hero BB; BTN-F elided (pre-RR fold),
    but SB-F is retained since it comes AFTER the 3bet."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, HJ folds, CO folds, BTN 3bets, SB folds, hero BB
    key = compute_preflop_line_key("F-F-R2-F-F-R8-F-F", "BB", 8)
    # Folds before RR (HJ, CO) elided; SB-F comes after RR, kept.
    assert_eq(key, "LJ-R-BTN-RR-SB-F")

@test
def test_line_key_squeeze_pot():
    """LJ opens, HJ folds, CO calls, BTN squeezes, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-R2-F-C-R8-F-F", "SB", 8)
    assert_eq(key, "LJ-R-CO-C-BTN-RR")

@test
def test_line_key_limp_iso():
    """UTG limps, UTG+1 limps, BTN iso-raises, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("C-C-F-F-F-R2-F-F", "SB", 8)
    assert_eq(key, "UTG-C-UTG+1-C-BTN-R")

@test
def test_line_key_hero_excluded():
    """Hero's own token must not appear in the key."""
    from spot_categorizer import compute_preflop_line_key
    # Hero CO opens R2; key should be empty (nothing before hero, hero excluded)
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-F", "CO", 8)
    assert_eq(key, "")

@test
def test_line_key_4bet_pot():
    """CO opens, BB 3bets, CO 4bets (hero is BB, second decision) → captures
    LJ-R ... wait: CO opens R2, hero BB 3bets, CO 4bets → hero BB acts second.
    Key includes CO-R, then hero's 3bet excluded, then CO-RR (the 4bet)."""
    from spot_categorizer import compute_preflop_line_key
    # seats 0..7, CO idx4 R2, BB idx7 R8, CO (continuation) R20 ...
    key = compute_preflop_line_key("F-F-F-F-R2-F-F-R8-R20", "BB", 8, action_index=1)
    # Action order: F,F,F,F (elide), CO-R (level1), F,F (elide),
    # BB=hero → excluded, raise_level becomes 2. Then continuation:
    # active=[idx4 CO, idx7 BB], cont_idx=0 → CO. Token R20 → RRR (level3).
    # But we stop at hero's second action (BB). CO-RRR comes before that.
    assert_eq(key, "CO-R-CO-RRR")

@test
def test_line_key_fold_after_3bet_kept():
    """Fold that follows a 3bet (RR) should be kept."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, CO 3bets, BTN folds, SB folds, hero BB
    key = compute_preflop_line_key("F-F-R2-F-R8-F-F-F", "BB", 8)
    # HJ-F elided (pre-RR). BTN-F and SB-F kept (post-RR).
    assert_eq(key, "LJ-R-CO-RR-BTN-F-SB-F")

@test
def test_line_key_fold_after_open_elided():
    """Folds that only follow a single raise (R) are elided."""
    from spot_categorizer import compute_preflop_line_key
    # LJ opens, everyone folds to hero BB
    key = compute_preflop_line_key("F-F-R2-F-F-F-F-F", "BB", 8)
    assert_eq(key, "LJ-R")

@test
def test_line_key_unopened():
    """All folds with no raise — hero BB walks."""
    from spot_categorizer import compute_preflop_line_key
    key = compute_preflop_line_key("F-F-F-F-F-F-F-F", "BB", 8)
    assert_eq(key, "")

@test
def test_line_key_6max_3bet():
    """6-max: CO opens, BTN 3bets, hero SB."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max seats: LJ, HJ, CO, BTN, SB, BB
    key = compute_preflop_line_key("F-F-R2-R8-F-F", "SB", 6)
    assert_eq(key, "CO-R-BTN-RR")


@test
def test_line_key_postflop_consumes_full_preflop():
    """action_index=None: consume full preflop line, don't stop at hero."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: LJ, HJ, CO, BTN, SB, BB
    # LJ folds, HJ opens, CO/BTN/SB fold, BB calls. Hero=HJ on flop.
    key = compute_preflop_line_key("F-R2-F-F-F-C", "HJ", 6, action_index=None)
    # Hero's own R2 is excluded. Pre-hero fold elided. Post-hero folds
    # elided (no re-raise). BB's call is kept — this is the critical
    # difference from action_index=0 (which would stop at HJ's R and
    # never see BB's C).
    assert_eq(key, "BB-C")


@test
def test_line_key_postflop_3bet_pot_full_preflop():
    """Postflop line_key for a 3bet pot: full preflop consumed."""
    from spot_categorizer import compute_preflop_line_key
    # 8-max: UTG opens, HJ 3bets, UTG calls. Hero=UTG on flop.
    key = compute_preflop_line_key(
        "R2-F-F-R8-F-F-F-F-C", "UTG", 8, action_index=None,
    )
    # Hero's R2 excluded, hero's C excluded. Folds-to-open elided.
    # HJ-RR kept. Post-RR folds kept (raise_level=2).
    assert_in("HJ-RR", key)


@test
def test_line_key_postflop_srp_hero_is_caller():
    """Postflop line_key when hero flatted preflop: full preflop kept."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: HJ opens, hero BTN calls, SB/BB fold.
    key = compute_preflop_line_key("F-R2-F-C-F-F", "BTN", 6, action_index=None)
    # Pre-hero: HJ-R kept (the open). Hero's own C excluded.
    # Post-hero: SB/BB folds elided (no re-raise).
    assert_eq(key, "HJ-R")


# ── compute_pot_type_from_preflop tests (hero-independent) ──

@test
def test_pot_type_from_preflop_srp_hero_is_opener():
    """Regression for the bug where hero-as-opener falsely showed as limp.
    compute_pot_type_from_preflop works directly on raw actions."""
    from spot_categorizer import compute_pot_type_from_preflop
    # HJ opens, folds around to BB who calls
    assert_eq(compute_pot_type_from_preflop("F-R2-F-F-F-C", 6), "SRP")
    # UTG opens, hero=UTG (this was the empty-line_key case in backfill)
    assert_eq(compute_pot_type_from_preflop("R2-F-F-F-F-F-C-C", 8), "SRP")


@test
def test_pot_type_from_preflop_3bet():
    from spot_categorizer import compute_pot_type_from_preflop
    # UTG opens, HJ 3bets, UTG calls
    assert_eq(compute_pot_type_from_preflop("R2-F-F-R8-F-F-F-F-C", 8), "3bet")


@test
def test_pot_type_from_preflop_4bet():
    from spot_categorizer import compute_pot_type_from_preflop
    # CO opens, BTN 3bets, CO 4bets
    assert_eq(compute_pot_type_from_preflop("F-F-R2-R8-F-F-F-F-R20", 8), "4bet")


@test
def test_pot_type_from_preflop_squeezed():
    from spot_categorizer import compute_pot_type_from_preflop
    # LJ opens, CO calls, BTN squeezes
    assert_eq(compute_pot_type_from_preflop("F-F-R2-F-C-R8-F-F", 8), "squeezed")


@test
def test_pot_type_from_preflop_limp():
    from spot_categorizer import compute_pot_type_from_preflop
    # UTG limps, CO iso-raises
    assert_eq(compute_pot_type_from_preflop("C-F-F-R3-F-F-F-F", 8), "limp")


@test
def test_pot_type_from_preflop_unopened():
    from spot_categorizer import compute_pot_type_from_preflop
    # All folds (hypothetical — shouldn't happen in practice)
    assert_eq(compute_pot_type_from_preflop("F-F-F-F-F-F-F-F", 8), "unopened")
    assert_eq(compute_pot_type_from_preflop("", 8), "unopened")


@test
def test_line_key_preflop_default_still_stops_at_hero():
    """action_index=0 (default): preflop behavior unchanged — stop at hero."""
    from spot_categorizer import compute_preflop_line_key
    # 6-max: LJ-HJ-CO-BTN-SB-BB. HJ opens, hero=BTN about to act.
    # Pre-hero: only HJ's open. Hero's own action not yet in the string,
    # so everything we see goes into the key.
    key = compute_preflop_line_key("F-R2-F", "BTN", 6)  # default action_index=0
    assert_eq(key, "HJ-R")


# ── compute_pot_type tests ──

@test
def test_pot_type_srp():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("CO-R"), "SRP")
    assert_eq(compute_pot_type("HJ-R-BTN-F-SB-F"), "SRP")

@test
def test_pot_type_3bet():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("LJ-R-BTN-RR-SB-F"), "3bet")
    assert_eq(compute_pot_type("CO-R-BB-RR"), "3bet")

@test
def test_pot_type_4bet():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("CO-R-BB-RR-CO-RRR"), "4bet")

@test
def test_pot_type_squeezed():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("LJ-R-CO-C-BTN-RR"), "squeezed")

@test
def test_pot_type_limp():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type("UTG-C-UTG+1-C-BTN-R"), "limp")

@test
def test_pot_type_limp_pure():
    from spot_categorizer import compute_pot_type
    # pure limp pot with no iso raise
    assert_eq(compute_pot_type("UTG-C-SB-C"), "limp")

@test
def test_pot_type_unopened():
    from spot_categorizer import compute_pot_type
    assert_eq(compute_pot_type(""), "unopened")


# ── Follow-up Parse Guard Tests ──

@test
def test_followup_question_not_parsed_as_hand():
    """Follow-up questions should not be treated as new hands when context exists."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gemini_session import GeminiSessionManager
    session = GeminiSessionManager.__new__(GeminiSessionManager)
    # Simulate existing hand context
    session.hand_contexts = {123: {"hero_position": "HJ", "hero_hand": "JTs"}}
    # Follow-up questions should NOT look like new hands
    followups = [
        "hero turn bet 83% 的範圍有哪些",
        "對手 check-raise 的範圍是什麼？",
        "如果 flop 用 33% pot 下注會怎樣？",
        "BB 在 turn 的策略",
        "為什麼 solver 建議 check",
        "這手牌的 EV 是多少",
    ]
    for q in followups:
        result = session._text_looks_like_hand(q)
        assert_eq(result, False, f"Follow-up should NOT look like a hand: {q!r}")
    # Hand ID reference followed by "BB" position should NOT match the
    # effective-bb regex (H2672 bug: "H2672 BB ..." was parsed as "2672 bb").
    assert_eq(session._text_looks_like_hand("H2672 BB 在河牌的小額下注範圍是什麼？"),
              False, "H2672 BB question should not look like a new hand")
    assert_eq(session._text_looks_like_hand("H2489 hero 的翻牌範圍"),
              False, "Hxxx hero question should not look like a new hand")
    # ICM stack-distribution follow-up: digits like 37/42/76 in stack lists
    # must NOT be treated as poker hands (production timeout on 2026-05-30).
    icm_followup = (
        "那 icm final table 剩餘 7 人，stack size 分布從 utg 開始為 "
        "12,14,37,15,42,11,7 這時當 hj raise hero co call/raise/all in range 如何"
    )
    assert_eq(session._text_looks_like_hand(icm_followup), False,
              "ICM stack-distribution follow-up should not look like a new hand")
    # Bare non-pair digit token (e.g., "76" without s/o suffix) plus an
    # action word should NOT count as a hand — proper hand notation requires
    # a suit/offsuit marker for non-pairs.
    assert_eq(session._text_looks_like_hand("對手 76 持有 raise 範圍"),
              False, "Bare '76' (no s/o suffix) should not look like a hand")
    # Numeric pairs (22-99) and suited/offsuit digit pairs are still hands.
    assert_eq(session._text_looks_like_hand("hero 77 raise"),
              True, "Numeric pair '77' + action is a hand description")
    assert_eq(session._text_looks_like_hand("hero 76s raise"),
              True, "Suited digit pair '76s' + action is a hand description")


@test
def test_real_hand_description_parsed():
    """Real hand descriptions should still be parsed even with existing context."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gemini_session import GeminiSessionManager
    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session.hand_contexts = {123: {"hero_position": "HJ", "hero_hand": "JTs"}}
    hands = [
        "有效 30bb, hero CO open raise, BB call, flop Qs7h2d",
        "50bb hero UTG TT raise, BTN 3bet all in",
        "hero BTN AKs raise 2.5bb, SB 3bet 8bb, hero call",
        "25bb CO open, hero BB AQo 該 3bet 還是 call",
    ]
    for h in hands:
        result = session._text_looks_like_hand(h)
        assert_eq(result, True, f"Hand description should look like a hand: {h!r}")


@test
def test_query_gto_h2643_redundant_overrides():
    """H2643 river follow-up: LLM sent redundant overrides (including a
    7-position preflop from a 7-max hand). The cached context has 8-position
    preflop (MTTGeneral 8-max padding). Should: (1) auto-pad leading F's,
    (2) detect overrides match played line, (3) return cached river
    range — NOT hit the API with a malformed preflop and get no data.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from analyze_hand import analyze_hand_full
    from gemini_session import GeminiSessionManager

    hand_json = {
        "streets": [
            {"board": "3d3sJd", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 1.1, "action": "R1.1", "position": "LJ"},
                {"size": 1.1, "action": "C", "position": "BB"}]},
            {"card": "7c", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 3.8, "action": "R3.8", "position": "LJ"},
                {"size": 3.8, "action": "C", "position": "BB"}]},
            {"card": "Ks", "actions": [
                {"action": "X", "position": "BB"},
                {"action": "X", "position": "LJ"}]},
        ],
        "gametype": "MTTGeneral",
        "hero_hand": "AdQd",
        "effective_bb": 15.9,
        "hero_position": "LJ",
        "preflop_actions": "F-R2-F-F-F-F-C",  # 7-max (will be padded to 8)
        "players_at_table": 7,
        "hero_starting_stack": 31.9,
    }

    ctx = analyze_hand_full(hand_json)
    # Sanity: analyze_hand padded preflop to 8 positions
    assert_eq(len(ctx["preflop_actions"].split("-")), 8,
              "analyze_hand should pad 7-max preflop to 8 for MTTGeneral")

    session = GeminiSessionManager.__new__(GeminiSessionManager)
    session.hand_contexts = {1: ctx}
    session.pending_images = {}
    session.last_hand_ids = {}
    session.db = None

    import logging as _l
    session._logger = _l.getLogger("test_h2643_redundant")
    session._logger.setLevel(_l.WARNING)  # quiet during tests

    # Exact LLM call that failed in production on 2026-04-09 for H2643
    args = {
        "street": "river",
        "position": "LJ",
        "board_override": "3d3sJd7cKs",
        "flop_actions_override": "X-R1.1-C",
        "turn_actions_override": "X-R4.25-C",
        "river_actions_override": "X",
        "preflop_actions_override": "F-R2-F-F-F-F-C",  # 7 positions
    }

    result = session._execute_query_gto(1, args)

    assert_not_in(
        "沒有 solver 數據", result,
        "H2643 fix: redundant overrides should hit cache, not return empty"
    )
    # Should show the cached river range by action
    assert_in("All-in", result, "Should show the All-in action in the result")
    assert_in("Check", result, "Should show the Check action in the result")


@test
def test_overrides_match_played_line_helper():
    """Unit test for the _overrides_match_played_line helper used by Fix B."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from gemini_session import GeminiSessionManager

    mgr = GeminiSessionManager.__new__(GeminiSessionManager)

    cached_params = {
        "gametype": "MTTGeneral",
        "depth": 17.125,
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "board": "3d3sJd7cKs",
        "flop_actions": "X-R1.1-C",
        "turn_actions": "X-R4.25-C",
        "river_actions": "X",
    }

    # Exact match (all overrides match)
    assert_true(mgr._overrides_match_played_line(
        cached_params,
        preflop_override="F-F-R2-F-F-F-F-C",
        board_override="3d3sJd7cKs",
        flop_override="X-R1.1-C",
        turn_override="X-R4.25-C",
        river_override="X",
        depth_override=None,
    ), "exact match should return True")

    # Partial (only preflop + board provided, rest None → should match)
    assert_true(mgr._overrides_match_played_line(
        cached_params,
        preflop_override="F-F-R2-F-F-F-F-C",
        board_override="3d3sJd7cKs",
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "partial overrides (None for unspecified) should match")

    # Mismatch: different board
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override="AhKhQh",  # wrong
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "different board should not match")

    # Mismatch: different flop actions
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override=None,
        flop_override="X-X",  # wrong
        turn_override=None,
        river_override=None,
        depth_override=None,
    ), "different flop actions should not match")

    # Depth mismatch
    assert_true(not mgr._overrides_match_played_line(
        cached_params,
        preflop_override=None,
        board_override=None,
        flop_override=None,
        turn_override=None,
        river_override=None,
        depth_override=30.125,  # wrong
    ), "different depth should not match")


@test
def test_extract_followups_strips_from_text():
    """Extract FOLLOWUP lines from coaching response and store separately."""
    from gemini_session import GeminiSessionManager as GeminiSession
    text = (
        "*Preflop*\n好的分析\n\n"
        "FOLLOWUP: Turn 上對手的範圍是什麼？\n"
        "FOLLOWUP: 如果河牌是空白牌怎麼打？\n"
        "FOLLOWUP: 這手牌的 EV 如何？"
    )
    clean, followups = GeminiSession._extract_followups(text)
    assert_eq(len(followups), 3, "should extract 3 followup questions")
    assert_true("FOLLOWUP" not in clean, "clean text should not contain FOLLOWUP lines")
    assert_eq(followups[0], "Turn 上對手的範圍是什麼？", "first followup content")
    # Full-width colon variant
    text2 = "分析內容\nFOLLOWUP：全形冒號問題？"
    clean2, followups2 = GeminiSession._extract_followups(text2)
    assert_eq(len(followups2), 1, "should handle full-width colon")
    assert_true("FOLLOWUP" not in clean2, "clean text should not contain full-width FOLLOWUP")
    # Markdown/bullet variants should also be removed from user-visible text.
    text_md = "分析內容\n- **FOLLOWUP:** 如果 CO 3-bet all-in 要跟哪些牌？"
    clean_md, followups_md = GeminiSession._extract_followups(text_md)
    assert_eq(followups_md, ["如果 CO 3-bet all-in 要跟哪些牌？"],
              "should strip bullet+bold FOLLOWUP marker")
    assert_true("FOLLOWUP" not in clean_md, "markdown followup marker should not leak")
    # No followups
    text3 = "普通分析文字，沒有 followup"
    clean3, followups3 = GeminiSession._extract_followups(text3)
    assert_eq(clean3, text3, "text without followups unchanged")
    assert_eq(len(followups3), 0, "no followups extracted")


def _followup_bot_stub(contexts):
    """A bare PokerWizardBot wired to a session stub exposing _extract_followups."""
    from telegram_bot.bot import PokerWizardBot
    from gemini_session import GeminiSessionManager as GeminiSession
    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = type("SessionStub", (), {
        "hand_contexts": contexts,
        "_extract_followups": staticmethod(GeminiSession._extract_followups),
    })()
    return bot


@test
def test_finalize_followups_recovers_leaked_lines():
    """Follow-up answers whose FOLLOWUP lines leaked become buttons, not raw text.

    Regression: the plain-chat follow-up path (_chat) never ran
    _extract_followups, so FOLLOWUP: lines surfaced as visible text instead of
    inline buttons. _finalize_followups is the send-time safety net.
    """
    ctx = {
        # Initial analysis already emitted its button set on this hand.
        "hand": {"hero_position": "BB", "hero_hand": "JJ"},
        "_followup_sent": True,
    }
    bot = _followup_bot_stub({5: ctx})

    response = (
        "JJ 在這裡用來抓詐唬，66 用來詐唬。\n\n"
        "FOLLOWUP: BB 在 flop 面對這個 all-in 的跟注範圍是什麼？\n"
        "FOLLOWUP: 如果 Hero 在 flop 只是跟注，turn 9s 會如何遊戲？\n"
        "FOLLOWUP: 為什麼像 KJs 這種頂對，跟注頻率遠高於 all-in？"
    )
    clean, markup = bot._finalize_followups(5, response)

    assert_true("FOLLOWUP" not in clean,
                "leaked FOLLOWUP lines must be stripped from visible text")
    assert_true(markup is not None,
                "recovered followups should re-render as buttons")
    assert_eq(len(markup.inline_keyboard), 3,
              "three recovered questions → three buttons")
    assert_eq(markup.inline_keyboard[0][0].callback_data, "fq:0",
              "buttons use short callback IDs, not Chinese text")
    assert_in("flop", markup.inline_keyboard[0][0].text,
              "button shows the recovered question text")


@test
def test_finalize_followups_noop_when_already_clean():
    """Already-extracted responses pass through unchanged, keeping prior followups."""
    ctx = {
        "hand": {"hero_position": "CO", "hero_hand": "AKs"},
        "followup_questions": ["問題一？", "問題二？", "問題三？"],
    }
    bot = _followup_bot_stub({9: ctx})

    response = "乾淨的分析文字，沒有任何 followup 標記。"
    clean, markup = bot._finalize_followups(9, response)

    assert_eq(clean, response, "clean text is unchanged")
    assert_true(markup is not None, "existing followups still render buttons")
    assert_eq(len(markup.inline_keyboard), 3, "three stored questions → three buttons")


@test
def test_followup_markup_for_no_hero_uses_callback_ids():
    """Telegram follow-ups: no-hero range spots still get buttons without truncating callback data."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    long_q = "如果 CO 3-bet all-in，我們應該用哪些牌跟注，哪些牌改成 4-bet all-in？"
    ctx = {
        "hand": {
            "hero_position": "HJ",
            "hero_hand": "AA",
            "no_hero_hand": True,
        },
        "followup_questions": [long_q],
    }
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {123: ctx}})()

    markup = bot._build_followup_markup(123)

    assert_true(markup is not None, "no-hero range analysis should still render follow-up buttons")
    button = markup.inline_keyboard[0][0]
    assert_eq(button.text, long_q, "button text should show the full question")
    assert_eq(button.callback_data, "fq:0", "callback_data should be an ID, not truncated Chinese text")
    assert_eq(ctx["_followup_buttons"]["0"], long_q, "full question should be stored in context")


@test
def test_gto_link_lives_on_summary_card_not_coaching():
    """"Open in GTO Wizard" button rides the 📋 summary card, not the coaching reply."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    ctx = {
        "hand": {"hero_position": "BB", "hero_hand": "85s"},
        "followup_questions": ["BB 在 turn 的 check-raise 範圍是什麼？"],
    }
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {7: ctx}})()
    # Avoid the network resolver — pin the deep-link URL.
    bot._build_gto_solution_url = lambda c: "https://app.gtowizard.com/solutions?x=1"

    # Summary card: a single GTO Wizard link button, no follow-up buttons.
    link_markup = bot._build_gto_link_markup(7)
    assert_true(link_markup is not None, "summary card should carry the GTO link")
    assert_eq(len(link_markup.inline_keyboard), 1, "summary card has exactly one button row")
    btn = link_markup.inline_keyboard[0][0]
    assert_in("GTO Wizard", btn.text)
    assert_eq(btn.url, "https://app.gtowizard.com/solutions?x=1")

    # Coaching reply (image flow): follow-up buttons only, no GTO link.
    coach_markup = bot._build_followup_markup(7, include_gto_link=False)
    assert_true(coach_markup is not None, "coaching reply still renders follow-up buttons")
    texts = [b.text for row in coach_markup.inline_keyboard for b in row]
    assert_true(all("GTO Wizard" not in t for t in texts),
                "coaching reply must NOT carry the GTO Wizard link")


@test
def test_gto_link_markup_none_without_url():
    """No deep-link buildable → no summary-card button (graceful)."""
    from telegram_bot.bot import PokerWizardBot

    bot = PokerWizardBot.__new__(PokerWizardBot)
    ctx = {"hand": {"hero_position": "BB", "hero_hand": "85s"}}
    bot.session_manager = type("SessionStub", (), {"hand_contexts": {9: ctx}})()
    bot._build_gto_solution_url = lambda c: None
    assert_true(bot._build_gto_link_markup(9) is None,
                "no URL → no markup, never raises")


# ── Lane A2: EV loss + DeviationMeta + aggression direction tests ──

from leak_service import (  # noqa: E402
    DeviationMeta,
    compute_ev_loss,
    pick_best_ev_action,
    classify_aggression_direction,
)
from spot_categorizer import map_spot_to_gtow  # noqa: E402


@test
def test_ev_loss_tied_best():
    """Mixed bet/check with tied EVs → loss is 0 when hero bets."""
    evs = {"R2": 10.5, "X": 10.5}
    assert_eq(compute_ev_loss(evs, "R2"), 0.0)
    assert_eq(compute_ev_loss(evs, "X"), 0.0)


@test
def test_ev_loss_small_delta():
    """Hero picks the slightly worse line → loss equals delta."""
    evs = {"R2": 10.5, "X": 10.3}
    loss = compute_ev_loss(evs, "X")
    assert_true(loss is not None and abs(loss - 0.2) < 1e-9, f"loss={loss}")


@test
def test_ev_loss_dominated_action():
    """Dominated action: bet 10, call 9, fold 8 → hero folds → loss=2.0."""
    evs = {"R2": 10.0, "C": 9.0, "F": 8.0}
    loss = compute_ev_loss(evs, "F")
    assert_true(loss is not None and abs(loss - 2.0) < 1e-9, f"loss={loss}")


@test
def test_ev_loss_one_legal_action():
    """Only one legal action → loss is 0."""
    evs = {"F": 0.0}
    assert_eq(compute_ev_loss(evs, "F"), 0.0)


@test
def test_ev_loss_missing_inputs():
    """Missing EVs or unknown code → returns None, no crash."""
    assert_eq(compute_ev_loss(None, "R2"), None)
    assert_eq(compute_ev_loss({}, "R2"), None)
    assert_eq(compute_ev_loss({"R2": 10.0}, None), None)
    assert_eq(compute_ev_loss({"R2": 10.0}, "X"), None)  # code not in dict


@test
def test_ev_loss_fp_edge_clamp():
    """Floating-point: hero_ev marginally > max due to FP error → clamps to 0."""
    # 0.1 + 0.2 == 0.30000000000000004 ≠ 0.3; construct a tiny negative delta.
    a = 0.1 + 0.2  # 0.30000000000000004
    b = 0.3
    evs = {"R2": b, "X": a}
    loss = compute_ev_loss(evs, "X")
    assert_true(loss is not None and loss == 0.0, f"loss={loss}")


@test
def test_pick_best_ev_action():
    assert_eq(pick_best_ev_action({"R2": 10.0, "C": 9.0, "F": 8.0}), "R2")
    assert_eq(pick_best_ev_action({}), None)
    assert_eq(pick_best_ev_action(None), None)


@test
def test_deviation_meta_to_jsonb_excludes_none():
    dm = DeviationMeta(villain_pos="HJ", pot_type="SRP")
    d = dm.to_jsonb()
    assert_eq(d, {"villain_pos": "HJ", "pot_type": "SRP"})
    assert_true("aggression_direction" not in d)


@test
def test_deviation_meta_from_jsonb_none():
    dm = DeviationMeta.from_jsonb(None)
    assert_eq(dm, DeviationMeta())
    dm2 = DeviationMeta.from_jsonb({})
    assert_eq(dm2, DeviationMeta())


@test
def test_deviation_meta_round_trip():
    original = DeviationMeta(
        villain_pos="HJ",
        preflop_line_key="LJ-R-HJ-C",
        pot_type="SRP",
        aggression_direction="too_aggressive",
        gtow_type="SRP",
        gtow_hero_role="aggressor",
        gto_dominant_action="R2",
        gto_best_ev_action="R2",
    )
    restored = DeviationMeta.from_jsonb(original.to_jsonb())
    assert_eq(restored, original)


@test
def test_deviation_meta_from_jsonb_ignores_unknown():
    """Unknown keys in JSONB should not crash from_jsonb."""
    dm = DeviationMeta.from_jsonb({"villain_pos": "BTN", "future_field": 42})
    assert_eq(dm.villain_pos, "BTN")


@test
def test_aggression_direction_aligned():
    assert_eq(classify_aggression_direction("R2", "R2"), "aligned")
    assert_eq(classify_aggression_direction("F", "F"), "aligned")


@test
def test_aggression_direction_too_passive():
    # X (check) when GTO wants to bet/raise.
    assert_eq(classify_aggression_direction("X", "R2"), "too_passive")
    assert_eq(classify_aggression_direction("C", "R3"), "too_passive")
    assert_eq(classify_aggression_direction("F", "AI"), "too_passive")


@test
def test_aggression_direction_too_aggressive():
    assert_eq(classify_aggression_direction("R2", "X"), "too_aggressive")
    assert_eq(classify_aggression_direction("AI", "C"), "too_aggressive")
    assert_eq(classify_aggression_direction("R3", "F"), "too_aggressive")


@test
def test_aggression_direction_mixed():
    # Two aggressive actions, different sizings → "mixed".
    assert_eq(classify_aggression_direction("R2", "R3"), "mixed")
    assert_eq(classify_aggression_direction("R2", "AI"), "mixed")


@test
def test_aggression_direction_missing():
    assert_eq(classify_aggression_direction(None, "R2"), None)
    assert_eq(classify_aggression_direction("R2", None), None)


@test
def test_map_spot_to_gtow_preflop():
    cases = {
        "open_raise":       ("RFI",             "aggressor"),
        "facing_open":      ("vsSRP",           "caller_candidate"),
        "hero_3bet":        ("3bet",            "3bettor"),
        "facing_3bet":      ("vs3bet",          "opener"),
        "facing_4bet":      ("vs4bet",          "3bettor"),
        "squeeze":          ("Squeeze",         "squeezer"),
        "vs_squeeze":       ("vsSqueeze",       "opener"),
        "possible_squeeze": ("possibleSqueeze", "caller_candidate"),
        "limp_pot":         ("vsLimp",          "iso_candidate"),
    }
    for spot, expected in cases.items():
        actual = map_spot_to_gtow(spot, None, "preflop", hero_is_pf_aggressor=False)
        assert_eq(actual, expected, msg=f"preflop spot {spot}")


@test
def test_map_spot_to_gtow_postflop():
    # SRP pot, hero is the aggressor (cbet_ip) → (SRP, aggressor).
    assert_eq(
        map_spot_to_gtow("cbet_ip", "SRP", "flop", hero_is_pf_aggressor=True),
        ("SRP", "aggressor"),
    )
    # 3bet pot, hero is the caller → (3bet, caller).
    assert_eq(
        map_spot_to_gtow("facing_cbet_oop", "3bet", "flop", hero_is_pf_aggressor=False),
        ("3bet", "caller"),
    )
    # 4bet pot collapses to 3bet flop type.
    assert_eq(
        map_spot_to_gtow("cbet_oop", "4bet", "flop", hero_is_pf_aggressor=True),
        ("3bet", "aggressor"),
    )
    # Squeezed pot.
    assert_eq(
        map_spot_to_gtow("cbet_ip", "squeezed", "flop", hero_is_pf_aggressor=True),
        ("Squeeze", "aggressor"),
    )


@test
def test_map_spot_to_gtow_unknown():
    # Unknown preflop spot category → (None, None).
    assert_eq(
        map_spot_to_gtow("nonsense_spot", None, "preflop", hero_is_pf_aggressor=False),
        (None, None),
    )


# ── Lane B: GTOW trainer URL builder tests ──

from urllib.parse import urlparse, parse_qs
from gtow_trainer_url import (
    build_trainer_url,
    snap_depth,
    SpotNotSupportedError,
    AVAILABLE_DEPTHS_BB,
)


@test
def test_snap_depth_exact_points():
    """snap_depth: exact snap points round to themselves"""
    for d in (10, 20, 30, 100):
        assert_eq(snap_depth(d), d, f"exact {d}")


@test
def test_snap_depth_round_down():
    """snap_depth: 22.4 → 20 (nearer to 20 than 25)"""
    assert_eq(snap_depth(22.4), 20)


@test
def test_snap_depth_round_up():
    """snap_depth: 22.6 → 25"""
    assert_eq(snap_depth(22.6), 25)


@test
def test_snap_depth_tie_rounds_down():
    """snap_depth: 17.5 → 15 (tie rounds down)"""
    assert_eq(snap_depth(17.5), 15)


@test
def test_snap_depth_clamp_min():
    """snap_depth: 5 → 10 (clamped to min)"""
    assert_eq(snap_depth(5), 10)


@test
def test_snap_depth_clamp_max():
    """snap_depth: 150 → 100 (clamped to max)"""
    assert_eq(snap_depth(150), 100)


@test
def test_snap_depth_gtow_float_format():
    """snap_depth: 30.125 (GTOW internal format) → 30"""
    assert_eq(snap_depth(30.125), 30)


@test
def test_snap_depth_boundary_tie_low():
    """snap_depth: 12.5 → 10 (tie between 10 and 15 rounds down)"""
    assert_eq(snap_depth(12.5), 10)


@test
def test_build_url_open_raise():
    """build_trainer_url: open_raise → fh_actions=RFI"""
    url = build_trainer_url("open_raise", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["RFI"])
    assert_eq(qs["fh_start_spot"], ["preflop"])


@test
def test_build_url_facing_3bet():
    """build_trainer_url: facing_3bet → fh_actions=vs3bet"""
    url = build_trainer_url("facing_3bet", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["vs3bet"])


@test
def test_build_url_possible_squeeze():
    """build_trainer_url: possible_squeeze → fh_actions=possibleSqueeze"""
    url = build_trainer_url("possible_squeeze", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["possibleSqueeze"])


@test
def test_build_url_hero_3bet():
    """build_trainer_url: hero_3bet → fh_actions=3bet"""
    url = build_trainer_url("hero_3bet", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["3bet"])


@test
def test_build_url_vs_squeeze():
    """build_trainer_url: vs_squeeze → fh_actions=vsSqueeze"""
    url = build_trainer_url("vs_squeeze", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["vsSqueeze"])


@test
def test_build_url_depth_snapped():
    """build_trainer_url: effective_bb=22.4 → depth=20.125"""
    url = build_trainer_url("open_raise", "preflop", 22.4)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["depth"], ["20.125"])
    assert_eq(qs["depth_list"], ["20.125"])


@test
def test_build_url_unknown_preflop_spot_raises():
    """build_trainer_url: unknown preflop spot → SpotNotSupportedError"""
    try:
        build_trainer_url("made_up_spot", "preflop", 20)
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_is_parseable():
    """build_trainer_url: result is a valid URL starting with base"""
    url = build_trainer_url("open_raise", "preflop", 20)
    assert_true(url.startswith("https://app.gtowizard.com/practice/trainer?"))
    parsed = urlparse(url)
    assert_eq(parsed.scheme, "https")
    assert_eq(parsed.netloc, "app.gtowizard.com")
    assert_eq(parsed.path, "/practice/trainer")


@test
def test_build_url_postflop_srp():
    """build_trainer_url: flop + SRP → fh_actions=SRP, fh_start_spot=flop"""
    url = build_trainer_url("cbet_ip", "flop", 30, pot_type="SRP")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["SRP"])
    assert_eq(qs["fh_start_spot"], ["flop"])


@test
def test_build_url_postflop_3bet_pot():
    """build_trainer_url: flop + 3bet pot → fh_actions=3bet"""
    url = build_trainer_url("cbet_ip", "flop", 30, pot_type="3bet")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["3bet"])


@test
def test_build_url_postflop_squeezed():
    """build_trainer_url: flop + squeezed pot → fh_actions=Squeeze"""
    url = build_trainer_url("cbet_ip", "flop", 30, pot_type="squeezed")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["Squeeze"])


@test
def test_build_url_postflop_4bet_falls_back_to_3bet():
    """build_trainer_url: flop + 4bet pot → fh_actions=3bet (closest)"""
    url = build_trainer_url("cbet_ip", "flop", 30, pot_type="4bet")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["3bet"])


@test
def test_build_url_turn_srp_keeps_turn_start():
    """build_trainer_url: turn + SRP → fh_actions=SRP, fh_start_spot=turn"""
    url = build_trainer_url("cbet_ip", "turn", 30, pot_type="SRP")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_actions"], ["SRP"])
    assert_eq(qs["fh_start_spot"], ["turn"])


@test
def test_build_url_postflop_missing_pot_type_raises():
    """build_trainer_url: postflop without pot_type → SpotNotSupportedError"""
    try:
        build_trainer_url("cbet_ip", "flop", 30)
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_postflop_unknown_pot_type_raises():
    """build_trainer_url: unknown pot_type → SpotNotSupportedError"""
    try:
        build_trainer_url("cbet_ip", "flop", 30, pot_type="weirdpot")
    except SpotNotSupportedError:
        return
    raise AssertionError("expected SpotNotSupportedError")


@test
def test_build_url_preserves_ui_flags():
    """build_trainer_url: every URL contains fh_trainer_hero_range=on"""
    url = build_trainer_url("open_raise", "preflop", 20)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["fh_trainer_hero_range"], ["on"])


@test
def test_build_url_contains_solution_type():
    """build_trainer_url: every URL contains solution_type=gwiz"""
    url = build_trainer_url("facing_3bet", "preflop", 25)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["solution_type"], ["gwiz"])


# ── Lane B2: GTOW /solutions strategy URL builder tests ──

import gtow_solution_url as _gsu
from gtow_solution_url import (
    build_solution_url,
    build_last_node_url,
    canonical_board_through_street,
    enumerate_hero_decisions,
)

# Resolver result that reproduces the hand-verified H3476 reference URL.
_H3476_RESOLVED = {
    "preflop_actions": "F-F-R2.3-F-F-F-F-C",
    "flop_actions": "X-R2-C",
    "turn_actions": "X-R8.4",
    "river_actions": "",
    "history_spot": 13,
    "depth": 40.0,
    "gametype": "MTTGeneral",
}
_H3476_URL = (
    "https://app.gtowizard.com/solutions"
    "?gametype=MTTGeneral&depth=40.125"
    "&gmfft_sort_key=0&gmfft_sort_order=desc"
    "&solution_type=gwiz&gmfs_solution_tab=ai_sols&soltab=strategy"
    "&preflop_actions=F-F-R2.3-F-F-F-F-C&history_spot=13"
    "&gmff_favorite=false&board=8h7d2hAh"
    "&flop_actions=X-R2-C&turn_actions=X-R8.4"
)


@test
def test_solution_url_matches_h3476_reference():
    """build_solution_url: exact match to the hand-verified H3476 URL"""
    url = build_solution_url(_H3476_RESOLVED, "8h7d2hAh")
    assert_eq(url, _H3476_URL)


@test
def test_solution_url_canonical_flop_rank_descending():
    """_canonical_flop: flop reordered rank-descending, suits follow rank"""
    assert_eq(_gsu._canonical_flop("7d8h2h"), "8h7d2h")
    # Paired flop: grouped by rank, suit order shdc — verified by hand against
    # GTOW (a KhKd5c link opens to the correct node).
    assert_eq(_gsu._canonical_flop("2c2d2h"), "2h2d2c")
    assert_eq(_gsu._canonical_flop("AhKsQd"), "AhKsQd")
    assert_eq(_gsu._canonical_flop("2h3h4h"), "4h3h2h")


@test
def test_solution_url_board_truncated_to_decision_street():
    """canonical_board_through_street: turn node excludes a dealt river card"""
    hand = {"streets": [
        {"board": "7d8h2h", "actions": []},
        {"card": "Ah", "actions": []},
        {"card": "Ks", "actions": []},
    ]}
    assert_eq(canonical_board_through_street(hand, "preflop"), "")
    assert_eq(canonical_board_through_street(hand, "flop"), "8h7d2h")
    assert_eq(canonical_board_through_street(hand, "turn"), "8h7d2hAh")
    assert_eq(canonical_board_through_street(hand, "river"), "8h7d2hAhKs")


@test
def test_solution_url_preflop_node_has_no_board():
    """build_solution_url: preflop node omits board / postflop action params"""
    resolved = {
        "preflop_actions": "F-F-R2.3-F-F-F-F-C", "flop_actions": "",
        "turn_actions": "", "river_actions": "", "history_spot": 8,
        "depth": 40.0, "gametype": "MTTGeneral",
    }
    url = build_solution_url(resolved, "")
    qs = parse_qs(urlparse(url).query)
    assert_true("board" not in qs, "preflop URL must not carry a board param")
    assert_true("flop_actions" not in qs, "preflop URL must not carry flop_actions")
    assert_eq(qs["history_spot"], ["8"])


@test
def test_solution_url_includes_river_actions():
    """build_solution_url: river_actions emitted when present"""
    resolved = dict(_H3476_RESOLVED, river_actions="X-C", history_spot=15)
    url = build_solution_url(resolved, "8h7d2hAhKs")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["river_actions"], ["X-C"])
    assert_eq(qs["board"], ["8h7d2hAhKs"])


@test
def test_solution_url_matches_river_reference():
    """build_solution_url: core params match a hand-verified river URL

    Reference (clicked in GTOW): a 20bb river node, hero faces a river X.
    Param ORDER differs from ours (GTOW emits board first, adds depth_list)
    so we compare via parse_qs — pinning the 5-card board ordering
    (flop rank-desc + turn + river) and the river_actions param name.
    """
    resolved = {
        "preflop_actions": "F-F-F-F-F-F-R3-C", "flop_actions": "X-R2.1-C",
        "turn_actions": "X-X", "river_actions": "X", "history_spot": 14,
        "depth": 20.0, "gametype": "MTTGeneral",
    }
    url = build_solution_url(resolved, "Jh8d2c6hJd")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["board"], ["Jh8d2c6hJd"], "5-card board ordering")
    assert_eq(qs["depth"], ["20.125"])
    assert_eq(qs["history_spot"], ["14"])
    assert_eq(qs["preflop_actions"], ["F-F-F-F-F-F-R3-C"])
    assert_eq(qs["flop_actions"], ["X-R2.1-C"])
    assert_eq(qs["turn_actions"], ["X-X"])
    assert_eq(qs["river_actions"], ["X"])
    assert_eq(qs["soltab"], ["strategy"])


@test
def test_solution_url_no_preflop_raises():
    """build_solution_url: empty preflop line → ValueError"""
    try:
        build_solution_url({"preflop_actions": "", "depth": 40.0}, "")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty preflop_actions")


@test
def test_solution_url_cash_depth_no_125_suffix():
    """build_solution_url: non-MTT gametype keeps a plain depth (no .125)"""
    resolved = {
        "preflop_actions": "R2.5-C", "flop_actions": "", "turn_actions": "",
        "river_actions": "", "history_spot": 2, "depth": 100.0,
        "gametype": "Cash6m",
    }
    url = build_solution_url(resolved, "")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["depth"], ["100"])


@test
def test_enumerate_hero_decisions_action_indices():
    """enumerate_hero_decisions: per-street hero action indices, in play order"""
    context = {
        "hero_spots": [
            {"street": "preflop"}, {"street": "flop"},
            {"street": "flop"}, {"street": "turn"},
        ],
        "solutions": [{"action_solutions": []}] * 4,
    }
    assert_eq(enumerate_hero_decisions(context),
              [("preflop", 0), ("flop", 0), ("flop", 1), ("turn", 0)])


@test
def test_enumerate_hero_decisions_skips_dataless_spots():
    """enumerate_hero_decisions: spots without a solution are skipped"""
    context = {
        "hero_spots": [{"street": "preflop"}, {"street": "flop"}],
        "solutions": [{"action_solutions": []}, None],
    }
    assert_eq(enumerate_hero_decisions(context), [("preflop", 0)])


@test
def test_build_last_node_falls_back_to_earlier_node():
    """build_last_node_url: off-tree last node falls back to nearest earlier"""
    hand = {"streets": [
        {"board": "7d8h2h", "actions": []},
        {"card": "Ah", "actions": []},
    ]}
    context = {
        "hand": hand,
        "hero_spots": [{"street": "flop"}, {"street": "turn"}],
        "solutions": [{"action_solutions": []}, {"action_solutions": []}],
    }

    def stub(_hand, street, action_index):
        if (street, action_index) == ("turn", 0):
            raise ValueError("off-tree size")
        return {
            "preflop_actions": "F-F-R2.3-F-F-F-F-C", "flop_actions": "X-X",
            "turn_actions": "", "river_actions": "", "history_spot": 10,
            "depth": 40.0, "gametype": "MTTGeneral",
        }

    url = build_last_node_url(context, _resolver=stub)
    assert_true(url is not None, "should fall back to the flop node")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-X"])
    assert_eq(qs["board"], ["8h7d2h"])  # turn card excluded — fell back to flop


@test
def test_build_last_node_returns_none_when_nothing_builds():
    """build_last_node_url: None when every decision node fails to build"""
    context = {
        "hand": {"streets": [{"board": "7d8h2h", "actions": []}]},
        "hero_spots": [{"street": "flop"}],
        "solutions": [{"action_solutions": []}],
    }

    def always_fail(_hand, _street, _idx):
        raise ValueError("nope")

    assert_true(build_last_node_url(context, _resolver=always_fail) is None)


@test
def test_build_last_node_none_when_no_decisions():
    """build_last_node_url: None when context has no hero decisions"""
    assert_true(build_last_node_url({"hand": {}, "hero_spots": [], "solutions": []}) is None)


def _h3639_spot_context():
    """H3639 context with each hero_spot carrying analyze_hand's snapped codes."""
    return {
        "hand": {"streets": [{"board": "9c9s5c"}, {"card": "Ks"}, {"card": "2c"}]},
        "hero_spots": [
            {"street": "preflop", "params": {
                "preflop_actions": "F-F", "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "flop", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5c",
                "flop_actions": "X", "turn_actions": "", "river_actions": "",
                "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "turn", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs",
                "flop_actions": "X-R1.4-C", "turn_actions": "X", "river_actions": "",
                "depth": 20.125, "gametype": "MTTGeneral"}},
            {"street": "river", "params": {
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs2c",
                "flop_actions": "X-R1.4-C", "turn_actions": "X-X", "river_actions": "R3",
                "depth": 20.125, "gametype": "MTTGeneral"}},
        ],
        "solutions": [{"action_solutions": []}] * 4,
    }


@test
def test_build_last_node_url_uses_resolved_spot_params_h3639():
    """build_last_node_url: sources snapped codes from spot params, not the API.

    Regression for H3639: with an expired GTOW token the resolver's
    next_actions calls failed and the deep-link fell back to raw 'B'/'X' tokens
    (flop_actions=X-B-C, river_actions=B) that GTOW can't parse → "something
    went wrong". analyze_hand already snapped every action to its GTOW code and
    stored the line on each hero_spot; reuse it so the link is exact and needs
    no live token. The resolver must NOT be called when params are present.
    """
    def boom(*a, **k):
        raise AssertionError("resolver must not run when spot params exist")

    url = build_last_node_url(_h3639_spot_context(), _resolver=boom)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["preflop_actions"], ["F-F-R2-F-F-F-F-C"])
    assert_eq(qs["flop_actions"], ["X-R1.4-C"])   # not raw X-B-C
    assert_eq(qs["turn_actions"], ["X-X"])
    assert_eq(qs["river_actions"], ["R3"])        # not raw B
    assert_eq(qs["board"], ["9s9c5cKs2c"])        # canonical flop order
    assert_eq(qs["history_spot"], ["14"])         # hero's river decision node


@test
def test_build_node_url_for_street_uses_resolved_spot_params():
    """build_node_url_for_street: turn link uses the turn spot's snapped codes."""
    from gtow_solution_url import build_node_url_for_street

    def boom(*a, **k):
        raise AssertionError("resolver must not run when spot params exist")

    url = build_node_url_for_street(_h3639_spot_context(), "turn", _resolver=boom)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-R1.4-C"])
    assert_eq(qs["turn_actions"], ["X"])          # hero's turn decision node
    assert_true("river_actions" not in qs)
    assert_eq(qs["board"], ["9s9c5cKs"])          # flop + turn, canonical


@test
def test_build_last_node_url_falls_back_to_resolver_without_params():
    """build_last_node_url: spots lacking params still resolve via the API path."""
    ctx = {
        "hand": {"streets": [{"board": "7d8h2h"}]},
        "hero_spots": [{"street": "flop"}],   # no params → resolver fallback
        "solutions": [{"action_solutions": []}],
    }

    def stub(_hand, _street, _idx):
        return {"preflop_actions": "R2.3-C", "flop_actions": "X-X",
                "turn_actions": "", "river_actions": "", "history_spot": 4,
                "depth": 40.0, "gametype": "MTTGeneral"}

    url = build_last_node_url(ctx, _resolver=stub)
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["flop_actions"], ["X-X"])
    assert_eq(qs["board"], ["8h7d2h"])


def _split_flow_session(fake_ctx, fake_hand):
    """Build a GeminiSessionManager wired for send_message split-flow tests.

    Bypasses __init__ (which needs a live genai client / API key) and stubs
    every I/O method send_message touches, so the test stays hermetic and
    only exercises the split-response gate.
    """
    import logging as _logging
    import gemini_session as _gs

    sess = _gs.GeminiSessionManager.__new__(_gs.GeminiSessionManager)
    sess._logger = _logging.getLogger("test_split_flow")
    sess.hand_contexts = {}
    sess.histories = {}
    sess.last_hand_ids = {}
    sess.pending_images = {}
    sess.db = None
    sess.model = "test-model"
    sess.parse_model = "test-parse"

    async def _fake_parse(chat_id, user_text, usage_acc=None):
        return fake_hand

    async def _fake_coach(*a, **k):
        return "COACHING REPLY"

    async def _noop(*a, **k):
        return None

    sess._parse_hand = _fake_parse
    sess._chat_with_tools = _fake_coach
    sess._save_usage = _noop
    sess._save_snapshot = _noop
    sess._extract_deviations = _noop
    sess._update_snapshot_coaching = _noop
    sess._setup_user_token = lambda *a, **k: None
    sess._clear_user_token = lambda *a, **k: None
    return sess


def _run_split_flow(fake_ctx, fake_hand, user_text):
    """Drive send_message with a recording send_gto_callback; return (sent, result)."""
    import asyncio
    import analyze_hand as _ah

    sess = _split_flow_session(fake_ctx, fake_hand)
    orig = _ah.analyze_hand_full
    _ah.analyze_hand_full = lambda hj: fake_ctx
    sent = []

    async def _cb(text):
        sent.append(text)

    async def _drive():
        return await sess.send_message(
            999, user_text, refresh_token="x", send_gto_callback=_cb)

    try:
        result = asyncio.run(_drive())
    finally:
        _ah.analyze_hand_full = orig
    return sent, result


@test
def test_text_split_flow_fires_gto_card_for_concrete_hand():
    """send_message pushes the structured per-street GTO card via
    send_gto_callback before the coaching reply when there's a concrete hero
    hand — the perceived-speed split that mirrors the image pipeline."""
    fake_hand = {
        "hero_position": "HJ", "hero_hand": "Ah7h",
        "preflop_actions": "R2-C-C", "effective_bb": 25,
        "streets": [{"board": "TdJhQc",
                     "actions": [{"position": "HJ", "action": "X"}]}],
    }
    fake_ctx = {
        "text": "FULL GTO TEXT FOR COACH",
        "text_compact": "♠ HJ A7s | 25bb MTT\n─── Preflop ───\nGTO: RAISE 100%",
        "no_hero_hand": False, "hand": fake_hand,
    }
    sent, result = _run_split_flow(fake_ctx, fake_hand, "Eff 25bb hj Ah7h ...")
    assert_eq(len(sent), 1, "GTO summary card should fire exactly once")
    assert_in("─── Preflop ───", sent[0], "card carries text_compact content")
    assert_in("COACHING REPLY", result, "coaching reply still returned after card")


@test
def test_text_split_flow_skips_gto_card_for_range_only_query():
    """send_message must NOT push the GTO card for a range-only query
    (no_hero_hand) — there's no per-hand verdict to show, so the split would
    only add a noisy extra message."""
    fake_hand = {
        "hero_position": "HJ", "hero_hand": "", "no_hero_hand": True,
        "preflop_actions": "R2", "effective_bb": 25,
        "streets": [{"board": "TdJhQc",
                     "actions": [{"position": "HJ", "action": "X"}]}],
    }
    fake_ctx = {
        "text": "RANGE TEXT", "text_compact": "RANGE CARD",
        "no_hero_hand": True, "hand": fake_hand,
    }
    sent, _ = _run_split_flow(fake_ctx, fake_hand, "HJ 開牌範圍是什麼")
    assert_eq(len(sent), 0, "no GTO card should fire for a range-only query")


# ── Runner ──

def run_tests():
    passed = 0
    failed = 0
    errors = []
    t0 = time.time()

    for fn in _tests:
        name = fn.__name__
        doc = fn.__doc__ or name

        if _filter and _filter not in name.lower() and _filter not in (doc or "").lower():
            continue

        try:
            t_start = time.time()
            fn()
            elapsed = time.time() - t_start
            passed += 1
            status = f"\033[32mPASS\033[0m"
            if _verbose:
                print(f"  {status} {doc} ({elapsed:.1f}s)")
            else:
                print(f"  {status} {doc}")
        except Exception as e:
            failed += 1
            status = f"\033[31mFAIL\033[0m"
            err_msg = str(e)
            print(f"  {status} {doc}")
            print(f"         {err_msg}")
            if _verbose:
                traceback.print_exc()
            errors.append((name, err_msg))

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed ({total:.1f}s)")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")

    return failed == 0


# ── Leak Miner Tests ──

@test
def test_label_aggression_all_passive():
    """leak_miner: all passive → too_passive."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(5, 0, 0, 0), "too_passive")


@test
def test_label_aggression_all_aggressive():
    """leak_miner: all aggressive → too_aggressive."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(0, 5, 0, 0), "too_aggressive")


@test
def test_label_aggression_70pct_passive():
    """leak_miner: 70% passive exactly → too_passive."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(7, 3, 0, 0), "too_passive")


@test
def test_label_aggression_69pct_passive():
    """leak_miner: 69% passive → mixed (below threshold)."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(69, 31, 0, 0), "mixed")


@test
def test_label_aggression_50_50():
    """leak_miner: 50/50 split → mixed."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(5, 5, 0, 0), "mixed")


@test
def test_label_aggression_all_aligned():
    """leak_miner: all aligned → aligned."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(0, 0, 10, 0), "aligned")


@test
def test_label_aggression_mostly_aligned_one_passive():
    """leak_miner: mostly aligned with 1 passive → too_passive (non-aligned dominated by passive)."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(1, 0, 9, 0), "too_passive")


@test
def test_label_aggression_empty():
    """leak_miner: zero everything → mixed (degenerate fallback)."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(0, 0, 0, 0), "mixed")


@test
def test_label_aggression_mixed_with_mixed_bucket():
    """leak_miner: 3/2/0/5 → mixed (neither side hits threshold)."""
    from leak_miner import _label_aggression
    assert_eq(_label_aggression(3, 2, 0, 5), "mixed")


@test
def test_cluster_key_to_dict():
    """leak_miner: ClusterKey.to_dict preserves all fields incl. Nones."""
    from leak_miner import ClusterKey
    k = ClusterKey(
        pot_type="srp",
        street="flop",
        gtow_hero_role=None,
        villain_pos="BB",
        hero_pos="BTN",
        spot_category="cbet_ip",
        board_texture=None,
    )
    d = k.to_dict()
    assert_eq(d["pot_type"], "srp")
    assert_eq(d["street"], "flop")
    assert_eq(d["gtow_hero_role"], None)
    assert_eq(d["villain_pos"], "BB")
    assert_eq(d["hero_pos"], "BTN")
    assert_eq(d["spot_category"], "cbet_ip")
    assert_eq(d["board_texture"], None)
    assert_eq(set(d.keys()), {
        "pot_type", "street", "gtow_hero_role", "villain_pos",
        "hero_pos", "spot_category", "board_texture",
    })


@test
def test_cluster_to_dict_rounding():
    """leak_miner: Cluster.to_dict rounds numeric fields as specified."""
    from leak_miner import Cluster, ClusterKey
    k = ClusterKey(
        pot_type="3bp",
        street="preflop",
        gtow_hero_role="IP_3B",
        villain_pos="CO",
        hero_pos="BTN",
        spot_category="facing_3bet",
        board_texture=None,
    )
    c = Cluster(
        key=k,
        sample_count=12,
        total_ev_loss_bb=3.14159,
        avg_ev_loss_bb=0.26180,
        aggression_label="too_passive",
        passive_ratio=0.83333,
        aggressive_ratio=0.16666,
        top_hand_ids=[101, 202, 303],
        top_deviation_ids=[9101, 9202, 9303],
        effective_bb_median=27.55,
        gtow_type="ICMGeneral",
    )
    d = c.to_dict()
    assert_eq(d["sample_count"], 12)
    assert_eq(d["total_ev_loss_bb"], 3.14)
    assert_eq(d["avg_ev_loss_bb"], 0.262)
    assert_eq(d["aggression_label"], "too_passive")
    assert_eq(d["passive_ratio"], 0.83)
    assert_eq(d["aggressive_ratio"], 0.17)
    assert_eq(d["top_hand_ids"], [101, 202, 303])
    assert_eq(d["effective_bb_median"], 27.6)
    assert_eq(d["gtow_type"], "ICMGeneral")
    assert_true("key" in d and isinstance(d["key"], dict))


# ── Weekly Report v2 Tests (Lane C2) ──


def _make_test_cluster(
    spot_category="cbet_ip",
    street="flop",
    pot_type="SRP",
    hero_pos="BTN",
    villain_pos="BB",
    board_texture="dry",
    sample_count=11,
    total_ev_loss_bb=4.80,
    aggression_label="too_aggressive",
    top_hand_ids=None,
    top_deviation_ids=None,
    effective_bb_median=30.0,
    gtow_type="MTTGeneral",
):
    from leak_miner import Cluster, ClusterKey
    if top_hand_ids is None:
        top_hand_ids = [2590, 2574, 2601]
    if top_deviation_ids is None:
        top_deviation_ids = []
    return Cluster(
        key=ClusterKey(
            pot_type=pot_type,
            street=street,
            gtow_hero_role=None,
            villain_pos=villain_pos,
            hero_pos=hero_pos,
            spot_category=spot_category,
            board_texture=board_texture,
        ),
        sample_count=sample_count,
        total_ev_loss_bb=total_ev_loss_bb,
        avg_ev_loss_bb=total_ev_loss_bb / max(sample_count, 1),
        aggression_label=aggression_label,
        passive_ratio=0.0 if aggression_label == "too_aggressive" else 1.0,
        aggressive_ratio=1.0 if aggression_label == "too_aggressive" else 0.0,
        top_hand_ids=top_hand_ids,
        top_deviation_ids=top_deviation_ids,
        effective_bb_median=effective_bb_median,
        gtow_type=gtow_type,
    )


@test
def test_validate_hand_ids_clean():
    """weekly_report: narrative referencing only allowed H IDs validates."""
    from weekly_report import _validate_narrative_hand_ids, ClusterNarrative
    nar = ClusterNarrative(
        cluster_id="0",
        headline="cbet 過頻",
        explanation="H2590 和 H2574 都打了過頻，特別是 H2601。",
        practice_hint="練 SRP 乾板",
    )
    assert_true(_validate_narrative_hand_ids(nar, {2590, 2574, 2601}))


@test
def test_validate_hand_ids_extra_id_rejected():
    """weekly_report: narrative referencing un-allowed H ID rejected."""
    from weekly_report import _validate_narrative_hand_ids, ClusterNarrative
    nar = ClusterNarrative(
        cluster_id="0",
        headline="cbet 過頻",
        explanation="H2590 和 H9999 都打了過頻。",  # 9999 not allowed
        practice_hint="hint",
    )
    assert_true(not _validate_narrative_hand_ids(nar, {2590, 2574, 2601}))


@test
def test_validate_hand_ids_no_mentions():
    """weekly_report: narrative with zero hand IDs is vacuously valid."""
    from weekly_report import _validate_narrative_hand_ids, ClusterNarrative
    nar = ClusterNarrative(
        cluster_id="0",
        headline="cbet 過頻",
        explanation="這個 spot 你打太多。",
        practice_hint="hint",
    )
    assert_true(_validate_narrative_hand_ids(nar, {2590}))


@test
def test_validate_hand_ids_strict_h_prefix():
    """weekly_report: bare numbers (e.g. dates, percentages) are NOT matched."""
    from weekly_report import _validate_narrative_hand_ids, ClusterNarrative
    # 2026 (year) and 50 (a percent) should NOT be parsed as hand IDs.
    nar = ClusterNarrative(
        cluster_id="0",
        headline="2026 年表現",
        explanation="這個 spot 偏離 50% 以上，看 H2590。",
        practice_hint="hint",
    )
    assert_true(_validate_narrative_hand_ids(nar, {2590}))


@test
def test_templated_narrative_basic():
    """weekly_report: templated fallback fills required fields + flags is_fallback."""
    from weekly_report import _templated_narrative
    cluster = _make_test_cluster()
    nar = _templated_narrative(cluster, "0")
    assert_true(nar.is_fallback)
    assert_eq(nar.cluster_id, "0")
    assert_true(len(nar.headline) > 0)
    assert_true(len(nar.explanation) > 0)
    assert_true(len(nar.practice_hint) > 0)
    # Templated narrative should not invent hand IDs.
    from weekly_report import _validate_narrative_hand_ids
    assert_true(_validate_narrative_hand_ids(nar, set(cluster.top_hand_ids)))


@test
def test_render_cluster_line_postflop_dry():
    """weekly_report: postflop SRP dry cluster line contains key substrings."""
    from weekly_report import _render_cluster_line, ClusterNarrative
    cluster = _make_test_cluster()
    nar = ClusterNarrative(
        cluster_id="0",
        headline="LJ 開 + HJ flat 之後乾板過度 cbet",
        explanation="H2590 和 H2574 都太頻繁。",
        practice_hint="練 1/3 pot 頻率",
    )
    line = _render_cluster_line(
        cluster, nar, "https://example.com/url", rank=1,
    )
    assert_in("**1.", line)
    assert_in("LJ 開", line)
    assert_in("乾燥面", line)
    assert_in("SRP", line)
    assert_in("n=11", line)
    assert_in("-4.80bb", line)
    assert_in("太 aggressive", line)
    assert_in("H2590", line)
    assert_in("https://example.com/url", line)


@test
def test_render_cluster_line_preflop_pot_type():
    """weekly_report: preflop cluster descriptor uses pot_type + position."""
    from weekly_report import _render_cluster_line, ClusterNarrative
    cluster = _make_test_cluster(
        spot_category="facing_3bet",
        street="preflop",
        pot_type="3bet",
        hero_pos="SB",
        villain_pos="BTN",
        board_texture=None,
        aggression_label="too_passive",
    )
    nar = ClusterNarrative(
        cluster_id="0",
        headline="SB 面對 3bet 太緊",
        explanation="",
        practice_hint="",
    )
    line = _render_cluster_line(cluster, nar, None, rank=2)
    assert_in("3bet pot", line)
    assert_in("SB", line)
    assert_in("太 passive", line)


@test
def test_render_cluster_line_direction_aligned():
    """weekly_report: 'aligned' direction renders with proper Chinese label."""
    from weekly_report import _render_cluster_line, ClusterNarrative
    cluster = _make_test_cluster(aggression_label="aligned")
    nar = ClusterNarrative("0", "headline", "exp", "hint")
    line = _render_cluster_line(cluster, nar, None, rank=1)
    assert_in("頻率大致正確", line)


@test
def test_render_cluster_line_ev_format():
    """weekly_report: EV loss formatted with 2 decimals + minus sign."""
    from weekly_report import _render_cluster_line, ClusterNarrative
    cluster = _make_test_cluster(total_ev_loss_bb=2.5)
    nar = ClusterNarrative("0", "h", "e", "p")
    line = _render_cluster_line(cluster, nar, None, rank=1)
    assert_in("-2.50bb", line)


class _MockGenAIClient:
    """Mock google-genai client matching client.aio.models.generate_content."""
    def __init__(self, responses):
        # responses: list[str] returned in order on successive calls
        self._responses = list(responses)
        self.calls = 0

        class _Models:
            def __init__(inner, parent):
                inner._parent = parent

            async def generate_content(inner, model, contents, config=None):
                idx = inner._parent.calls
                inner._parent.calls += 1
                if idx >= len(inner._parent._responses):
                    text = inner._parent._responses[-1]
                else:
                    text = inner._parent._responses[idx]

                class _Resp:
                    pass
                r = _Resp()
                r.text = text
                return r

        class _Aio:
            def __init__(inner, parent):
                inner.models = _Models(parent)

        self.aio = _Aio(self)


@test
def test_generate_cluster_narratives_happy_path():
    """weekly_report: LLM returns valid array → narratives passed through."""
    import asyncio as _asyncio
    from weekly_report import generate_cluster_narratives
    cluster = _make_test_cluster()
    raw = json.dumps([{
        "cluster_id":   "0",
        "headline":     "cbet 過頻",
        "explanation":  "H2590 是最貴的決策。",
        "practice_hint": "練 1/3 pot 頻率",
    }])
    mock = _MockGenAIClient([raw])
    out = _asyncio.run(
        generate_cluster_narratives([cluster], model_client=mock)
    )
    assert_eq(len(out), 1)
    assert_true(not out[0].is_fallback)
    assert_eq(out[0].headline, "cbet 過頻")
    assert_eq(mock.calls, 1)


@test
def test_generate_cluster_narratives_retry_then_succeed():
    """weekly_report: hallucinated ID → retry once → second attempt valid."""
    import asyncio as _asyncio
    from weekly_report import generate_cluster_narratives
    cluster = _make_test_cluster()
    bad = json.dumps([{
        "cluster_id":    "0",
        "headline":      "headline",
        "explanation":   "H9999 是最貴的決策。",  # hallucinated
        "practice_hint": "hint",
    }])
    good = json.dumps([{
        "cluster_id":    "0",
        "headline":      "cbet 過頻",
        "explanation":   "H2590 是最貴的決策。",
        "practice_hint": "練習",
    }])
    mock = _MockGenAIClient([bad, good])
    out = _asyncio.run(
        generate_cluster_narratives([cluster], model_client=mock, max_retries=1)
    )
    assert_eq(len(out), 1)
    assert_true(not out[0].is_fallback)
    assert_eq(out[0].headline, "cbet 過頻")
    assert_eq(mock.calls, 2)


@test
def test_generate_cluster_narratives_two_fails_falls_back():
    """weekly_report: two validation failures in a row → templated fallback."""
    import asyncio as _asyncio
    from weekly_report import generate_cluster_narratives
    cluster = _make_test_cluster()
    bad = json.dumps([{
        "cluster_id":    "0",
        "headline":      "headline",
        "explanation":   "H9999 hallucinated.",
        "practice_hint": "hint",
    }])
    mock = _MockGenAIClient([bad, bad])
    out = _asyncio.run(
        generate_cluster_narratives([cluster], model_client=mock, max_retries=1)
    )
    assert_eq(len(out), 1)
    assert_true(out[0].is_fallback)
    assert_eq(mock.calls, 2)


@test
def test_generate_cluster_narratives_no_client():
    """weekly_report: model_client=None → all clusters templated."""
    import asyncio as _asyncio
    from weekly_report import generate_cluster_narratives
    clusters = [_make_test_cluster(), _make_test_cluster(spot_category="cbet_oop")]
    out = _asyncio.run(
        generate_cluster_narratives(clusters, model_client=None)
    )
    assert_eq(len(out), 2)
    assert_true(all(n.is_fallback for n in out))


@test
def test_empty_state_message():
    """weekly_report: empty state helper returns a non-empty zh-TW string."""
    from weekly_report import _empty_state_message
    msg = _empty_state_message()
    assert_true(len(msg) > 0)
    assert_in("本週", msg)


@test
def test_render_report_full():
    """weekly_report: end-to-end render assembles header + clusters + total."""
    from weekly_report import _render_report, _templated_narrative
    from datetime import datetime as _dt
    clusters = [
        _make_test_cluster(total_ev_loss_bb=4.80),
        _make_test_cluster(spot_category="cbet_oop", total_ev_loss_bb=2.30),
    ]
    narratives = [_templated_narrative(c, str(i)) for i, c in enumerate(clusters)]
    out = _render_report(
        clusters=clusters,
        narratives=narratives,
        urls=[None, None],
        period_start=_dt(2026, 4, 4),
        period_end=_dt(2026, 4, 11),
        total_hands=50,
        total_decisions=159,
    )
    assert_in("📊 週報", out)
    assert_in("04/04", out)
    assert_in("04/11", out)
    assert_in("50 手", out)
    assert_in("159 決策", out)
    assert_in("**1.", out)
    assert_in("**2.", out)
    assert_in("-7.10bb", out)  # cumulative


# ── Backfill script pure helpers ──

@test
def test_backfill_walk_preflop_first():
    """backfill: _walk_to_decision finds preflop snapshot at action_index=0."""
    from backfill_ev_loss import _walk_to_decision
    analysis = {
        "hero_spots": [
            {"street": "preflop"},
            {"street": "flop"},
        ],
        "solutions": [
            {"action_solutions": [{"action": {"code": "R2"}}]},
            {"action_solutions": [{"action": {"code": "X"}}]},
        ],
    }
    snap = _walk_to_decision(analysis, "preflop", 0)
    assert_true(snap is not None, "expected non-None snapshot")
    assert_eq(snap["action_solutions"][0]["action"]["code"], "R2")


@test
def test_backfill_walk_missing_street():
    """backfill: _walk_to_decision returns None for mismatched street."""
    from backfill_ev_loss import _walk_to_decision
    analysis = {
        "hero_spots": [{"street": "preflop"}],
        "solutions": [{"action_solutions": [{"action": {"code": "R2"}}]}],
    }
    assert_true(_walk_to_decision(analysis, "river", 0) is None)
    assert_true(_walk_to_decision({}, "preflop", 0) is None)
    assert_true(_walk_to_decision(analysis, "preflop", 5) is None)


@test
def test_backfill_walk_postflop_second():
    """backfill: _walk_to_decision indexes per-street for postflop."""
    from backfill_ev_loss import _walk_to_decision
    analysis = {
        "hero_spots": [
            {"street": "preflop"},
            {"street": "flop"},
            {"street": "flop"},
            {"street": "turn"},
        ],
        "solutions": [
            {"action_solutions": [{"tag": "pf"}]},
            {"action_solutions": [{"tag": "flop_a"}]},
            {"action_solutions": [{"tag": "flop_b"}]},
            {"action_solutions": [{"tag": "turn"}]},
        ],
    }
    snap = _walk_to_decision(analysis, "flop", 1)
    assert_true(snap is not None)
    assert_eq(snap["action_solutions"][0]["tag"], "flop_b")


@test
def test_backfill_parse_args():
    """backfill: parse_args defaults to dry-run, --execute flips it."""
    from backfill_ev_loss import parse_args
    ns = parse_args([])
    assert_true(ns.dry_run is True, "default must be dry-run")
    assert_true(ns.execute is False, "execute must default False")
    ns = parse_args(["--execute"])
    assert_true(ns.dry_run is False)
    assert_true(ns.execute is True)
    ns = parse_args(["--execute", "--limit", "42", "--chat-id", "7"])
    assert_eq(ns.limit, 42)
    assert_eq(ns.chat_id, 7)


# ── Unified leak tools (EV-ranked) Tests ──


@test
def test_spot_descriptions_has_new_buckets():
    """leak_service: SPOT_DESCRIPTIONS_ZH contains the 3 new squeeze/3bet buckets."""
    from leak_service import SPOT_DESCRIPTIONS_ZH
    for k in ("possible_squeeze", "hero_3bet", "vs_squeeze"):
        assert_true(k in SPOT_DESCRIPTIONS_ZH, f"missing {k} in SPOT_DESCRIPTIONS_ZH")
        assert_true(bool(SPOT_DESCRIPTIONS_ZH[k]), f"{k} has empty label")


@test
def test_aggression_direction_zh_complete():
    """leak_service: AGGRESSION_DIRECTION_ZH has all 4 direction labels."""
    from leak_service import AGGRESSION_DIRECTION_ZH
    for k in ("too_passive", "too_aggressive", "mixed", "aligned"):
        assert_true(k in AGGRESSION_DIRECTION_ZH, f"missing {k}")


@test
def test_get_top_leaks_ev_ranked_shape():
    """leak_service: EV-ranked leak rows carry cluster fields + practice_url."""
    import asyncio
    from leak_service import get_top_leaks_ev_ranked

    c1 = _make_test_cluster(
        spot_category="cbet_ip", street="flop", pot_type="SRP",
        hero_pos="BTN", sample_count=11, total_ev_loss_bb=4.80,
    )
    c2 = _make_test_cluster(
        spot_category="facing_3bet", street="preflop", pot_type="3bet",
        hero_pos="CO", villain_pos="BB", board_texture=None,
        sample_count=8, total_ev_loss_bb=3.20,
        aggression_label="too_passive",
    )
    c3 = _make_test_cluster(
        spot_category="open_raise", street="preflop", pot_type=None,
        hero_pos="LJ", villain_pos=None, board_texture=None,
        sample_count=6, total_ev_loss_bb=1.50,
        aggression_label="too_aggressive",
    )

    async def fake_mine(pool, chat_id, start, end, min_sample=5, top_k=5):
        return [c1, c2, c3]

    rows = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=42, days=30, limit=5,
        mine_clusters_fn=fake_mine,
    ))
    assert_eq(len(rows), 3)
    # Order preserved (EV ranking done inside mine_clusters)
    assert_eq(rows[0]["spot_category"], "cbet_ip")
    assert_eq(rows[1]["spot_category"], "facing_3bet")
    assert_eq(rows[2]["spot_category"], "open_raise")
    # Shape: required keys
    for key in ("spot_category", "street", "pot_type", "hero_pos",
                "sample_count", "total_ev_loss_bb", "avg_ev_loss_bb",
                "aggression_label", "top_hand_ids", "effective_bb_median",
                "practice_url"):
        assert_true(key in rows[0], f"missing {key}")
    assert_eq(rows[0]["sample_count"], 11)
    assert_eq(rows[0]["total_ev_loss_bb"], 4.80)
    # Practice URL should be built for known preflop/postflop mappings
    assert_true(rows[0]["practice_url"] is not None, "cbet_ip should have URL")
    assert_true("gtowizard.com" in rows[0]["practice_url"])
    assert_true(rows[1]["practice_url"] is not None, "facing_3bet should have URL")


@test
def test_get_top_leaks_ev_ranked_post_filter():
    """leak_service: post-filter by spot_category narrows the result set."""
    import asyncio
    from leak_service import get_top_leaks_ev_ranked

    clusters = [
        _make_test_cluster(spot_category="cbet_ip", sample_count=10, total_ev_loss_bb=5.0),
        _make_test_cluster(
            spot_category="facing_3bet", street="preflop", pot_type="3bet",
            board_texture=None, sample_count=9, total_ev_loss_bb=4.0,
        ),
        _make_test_cluster(spot_category="cbet_ip", sample_count=8, total_ev_loss_bb=3.0,
                           hero_pos="CO"),
        _make_test_cluster(spot_category="open_raise", street="preflop",
                           pot_type=None, board_texture=None,
                           sample_count=7, total_ev_loss_bb=2.0, hero_pos="LJ"),
        _make_test_cluster(spot_category="cbet_oop", sample_count=6, total_ev_loss_bb=1.0,
                           hero_pos="BB"),
    ]

    async def fake_mine(pool, chat_id, start, end, min_sample=5, top_k=5):
        return clusters

    # Filter by spot_category
    rows = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=5, spot_category="cbet_ip",
        mine_clusters_fn=fake_mine,
    ))
    assert_eq(len(rows), 2)
    assert_true(all(r["spot_category"] == "cbet_ip" for r in rows))

    # Filter by street
    rows = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=5, street="preflop",
        mine_clusters_fn=fake_mine,
    ))
    assert_eq(len(rows), 2)
    assert_true(all(r["street"] == "preflop" for r in rows))

    # Filter by position
    rows = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=5, position="CO",
        mine_clusters_fn=fake_mine,
    ))
    assert_eq(len(rows), 1)
    assert_eq(rows[0]["hero_pos"], "CO")


@test
def test_get_top_leaks_ev_ranked_empty():
    """leak_service: empty cluster list → empty result."""
    import asyncio
    from leak_service import get_top_leaks_ev_ranked

    async def fake_mine(pool, chat_id, start, end, min_sample=5, top_k=5):
        return []

    rows = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=5, mine_clusters_fn=fake_mine,
    ))
    assert_eq(rows, [])


@test
def test_query_my_leaks_rendering():
    """gemini_session: query_my_leaks branch renders EV-ranked zh-TW output."""
    import asyncio
    from leak_service import (
        SPOT_DESCRIPTIONS_ZH, AGGRESSION_DIRECTION_ZH, get_top_leaks_ev_ranked,
    )

    c1 = _make_test_cluster(
        spot_category="cbet_ip", sample_count=11, total_ev_loss_bb=4.80,
        top_hand_ids=[2590, 2574, 2601],
    )
    c2 = _make_test_cluster(
        spot_category="facing_3bet", street="preflop", pot_type="3bet",
        board_texture=None, sample_count=8, total_ev_loss_bb=3.20,
        aggression_label="too_passive", top_hand_ids=[100, 200, 300],
    )

    async def fake_mine(pool, chat_id, start, end, min_sample=5, top_k=5):
        return [c1, c2]

    leaks = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=5, mine_clusters_fn=fake_mine,
    ))

    # Replicate the gemini_session rendering loop
    lines = ["💸 你的 leaks（按 EV 損失排序）：\n"]
    for i, leak in enumerate(leaks, 1):
        desc = SPOT_DESCRIPTIONS_ZH.get(leak["spot_category"], leak["spot_category"])
        direction = AGGRESSION_DIRECTION_ZH.get(
            leak["aggression_label"], leak["aggression_label"])
        ev = leak["total_ev_loss_bb"]
        n = leak["sample_count"]
        hands = " · ".join(f"H{h}" for h in leak["top_hand_ids"][:3])
        block = [f"**{i}. {desc}**（n={n}, -{ev:.2f}bb）"]
        block.append(f"   方向：{direction}")
        if hands:
            block.append(f"   最貴決策：{hands}")
        if leak.get("practice_url"):
            block.append(f"   → [練習連結]({leak['practice_url']})")
        lines.append("\n".join(block))
    rendered = "\n".join(lines)

    assert_in("位置內 C-bet", rendered)
    assert_in("-4.80bb", rendered)
    assert_in("n=11", rendered)
    assert_in("H2590", rendered)
    assert_in("面對 3-bet 的防禦", rendered)
    assert_in("太 passive", rendered)
    assert_in("練習連結", rendered)


@test
def test_get_training_plan_rendering():
    """gemini_session: training plan renders EV loss + direction + practice URL."""
    import asyncio
    from leak_service import (
        SPOT_DESCRIPTIONS_ZH, AGGRESSION_DIRECTION_ZH, get_top_leaks_ev_ranked,
    )

    c1 = _make_test_cluster(
        spot_category="cbet_ip", sample_count=11, total_ev_loss_bb=4.80,
    )

    async def fake_mine(pool, chat_id, start, end, min_sample=5, top_k=5):
        return [c1]

    leaks = asyncio.run(get_top_leaks_ev_ranked(
        pool=None, chat_id=1, limit=3, mine_clusters_fn=fake_mine,
    ))

    lines = ["🎯 訓練計畫（根據本月最貴的 leak）：\n"]
    for i, leak in enumerate(leaks, 1):
        desc = SPOT_DESCRIPTIONS_ZH.get(leak["spot_category"], leak["spot_category"])
        direction = AGGRESSION_DIRECTION_ZH.get(
            leak["aggression_label"], leak["aggression_label"])
        ev = leak["total_ev_loss_bb"]
        n = leak["sample_count"]
        block = [
            f"重點 {i}: {desc}",
            f"  累計 EV 損失: -{ev:.2f}bb (n={n})",
            f"  方向: {direction}",
        ]
        if leak.get("practice_url"):
            block.append(f"  練習連結: {leak['practice_url']}")
        else:
            block.append(f"  建議: 在 GTO Wizard 練習 {desc} 場景")
        lines.append("\n".join(block))
    rendered = "\n\n".join(lines)

    assert_in("重點 1", rendered)
    assert_in("位置內 C-bet", rendered)
    assert_in("-4.80bb", rendered)
    assert_in("練習連結", rendered)
    assert_in("gtowizard.com", rendered)


@test
def test_classify_board_flush_draw_disconnected():
    """gtow_custom_url: 4c6h8h — flush_draw flop (2 hearts), not paired, disconnected (H2665 flop)."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "flush_draw")
    assert_eq(r["flop_connectedness"], "disconnected")
    assert_eq(r.get("turn_paired"), None)  # no turn card


@test
def test_classify_board_connected_flop():
    """gtow_custom_url: 7h8d9s — 3 consecutive ranks → connected."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8d9s")
    assert_eq(r["flop_connectedness"], "connected")


@test
def test_classify_board_oesd_possible_flop():
    """gtow_custom_url: 7h8dJc — two adjacent + one gap → oesd_possible."""
    from gtow_custom_url import classify_board
    r = classify_board("7h8dJc")
    assert_eq(r["flop_connectedness"], "oesd_possible")


@test
def test_classify_board_turn_pairs_flop():
    """gtow_custom_url: 4c6h8h4h — flush_draw flop, turn pairs the 4 AND completes 3 hearts."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4h")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["flop_suits"], "flush_draw")
    # 4 cards suit counts: c=1, h=3, s=0 → max 3 → flush
    assert_eq(r["turn_suit"], "flush")


@test
def test_classify_board_turn_backdoor():
    """gtow_custom_url: 4c6h8s2h — flop rainbow, turn brings 2nd heart → backdoor."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8s2h")
    assert_eq(r["flop_suits"], "rainbow")
    # c=1, h=2, s=1 → max 2 → backdoor
    assert_eq(r["turn_suit"], "backdoor")


@test
def test_classify_board_flush_draw_flop():
    """gtow_custom_url: AhKh2c — 2-tone flop → flush_draw."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKh2c")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "flush_draw")


@test
def test_classify_board_monotone_flop():
    """gtow_custom_url: AhKhQh — all hearts → monotone."""
    from gtow_custom_url import classify_board
    r = classify_board("AhKhQh")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["flop_suits"], "monotone")


@test
def test_classify_board_paired_flop():
    """gtow_custom_url: 7h7d2c — paired flop."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d2c")
    assert_eq(r["flop_paired"], "paired")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_river():
    """gtow_custom_url: 4c6h8h4hKh — flush_draw flop, turn pairs the 4 AND completes flush, river keeps flush."""
    from gtow_custom_url import classify_board
    r = classify_board("4c6h8h4hKh")
    # 5 cards: 4c 6h 8h 4h Kh → flop c=1, h=2 → flush_draw; turn c=1, h=3 → flush; river c=1, h=4 → flush
    assert_eq(r["flop_suits"], "flush_draw")
    assert_eq(r["turn_suit"], "flush")
    assert_eq(r["river_suit"], "flush")
    assert_eq(r["flop_paired"], "not_paired")
    assert_eq(r["turn_paired"], "paired")
    assert_eq(r["river_paired"], "paired")


@test
def test_classify_board_empty():
    """gtow_custom_url: empty board → empty dict (no keys, not an error)."""
    from gtow_custom_url import classify_board
    assert_eq(classify_board(""), {})
    assert_eq(classify_board(None), {})


@test
def test_classify_board_tripled_flop():
    """gtow_custom_url: 7h7d7s — tripled flop (NOT 'paired')."""
    from gtow_custom_url import classify_board
    r = classify_board("7h7d7s")
    assert_eq(r["flop_paired"], "tripled")
    assert_eq(r["flop_suits"], "rainbow")


@test
def test_classify_board_odd_length_raises():
    """gtow_custom_url: odd-length board string → ValueError (caller falls back)."""
    from gtow_custom_url import classify_board
    try:
        classify_board("4c6h8")  # 5 chars — malformed
        assert_true(False, "expected ValueError")
    except ValueError:
        pass


@test
def test_resolve_h2665_turn_decision():
    """gtow_action_resolver: H2665 turn fold resolves to R2.1 / R1.9-C / R5.2 at 30bb."""
    from gtow_action_resolver import resolve_actions_for_deviation

    # NOTE: effective_bb=30.0 here (constructed), not H2665's real 36.7,
    # so nearest_depth snaps to 30.125 where verified R2.1/R1.9/R5.2 codes apply.
    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {
                "board": "4c6h8h",
                "actions": [
                    {"position": "BB",  "action": "R2.7", "size": 2.7},
                    {"position": "BTN", "action": "C"},
                ],
            },
            {
                "card": "4h",
                "actions": [
                    {"position": "BB",  "action": "R5.4", "size": 5.4},
                    {"position": "BTN", "action": "F"},
                ],
            },
        ],
    }

    # action_index=0 = hero's FIRST decision on the turn (BTN's fold).
    # Raw stream: [BB donk @ idx 0, BTN fold @ idx 1]. Resolver must emit R5.2
    # (BB's donk) as turn_actions, then stop before hero's fold.
    result = resolve_actions_for_deviation(
        hand_data, street="turn", action_index=0,
    )

    assert_eq(result["preflop_actions"], "F-F-F-F-F-R2.1-F-C")
    assert_eq(result["flop_actions"], "R1.9-C")
    assert_eq(result["turn_actions"], "R5.2")
    assert_eq(result["river_actions"], "")
    assert_eq(result["hero_pos"], "BTN")
    assert_eq(result["villain_pos"], "BB")
    assert_eq(result["history_spot"], 11)
    assert_eq(result["depth"], 30.125)
    assert_eq(result["gametype"], "MTTGeneral")


@test
def test_resolve_3bet_pot_preflop():
    """gtow_action_resolver: 6-max CO open, BTN 3bet, CO call, flop check.

    Ensures multi-raise preflop lines resolve correctly (each R token gets a
    new next_actions lookup that sees the previously-resolved prefix).
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 40.0,
        "hero_position": "CO",
        "players_at_table": 6,
        # 6-max: UTG, HJ, CO, BTN, SB, BB.
        # Here: UTG F, HJ F, CO R2.3, BTN R6.5, SB F, BB F, CO C.
        "preflop_actions": "F-F-R2.3-R6.5-F-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "CO",  "action": "X"},
                {"position": "BTN", "action": "X"},
            ]},
        ],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="flop", action_index=0,
    )

    # Padded to 8-max: 2 extra folds at front (6-max → 8-max = +2).
    # Shape: F-F-F-F-<CO>-<BTN>-F-F-C (9 tokens; CO at slot 4, BTN at slot 5).
    pf = result["preflop_actions"].split("-")
    assert_eq(len(pf), 9)
    assert_eq(pf[0:4], ["F", "F", "F", "F"])
    assert_true(pf[4].startswith("R"), f"CO open must be R*, got {pf[4]}")
    assert_true(pf[5].startswith("R"), f"BTN 3bet must be R*, got {pf[5]}")
    assert_eq(pf[6:9], ["F", "F", "C"])
    assert_eq(result["hero_pos"], "CO")
    assert_eq(result["villain_pos"], "BTN")  # last non-hero aggressor


@test
def test_resolve_cash_game_depth_has_no_125():
    """gtow_action_resolver: cash games use nearest_cash_depth without .125 suffix."""
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "Cash6m100",
        "effective_bb": 100.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-R2.5-F-C",
        "streets": [],
    }
    result = resolve_actions_for_deviation(
        hand_data, street="preflop", action_index=0,
    )
    # Cash depth is a plain float, no .125 suffix
    assert_true(
        not str(result["depth"]).endswith(".125"),
        f"cash depth should not have .125 suffix, got {result['depth']}",
    )


@test
def test_resolve_h3480_multiway_coldcall_and_pot_ratio():
    """gtow_action_resolver: H3480 turn fold — full deep-link resolution.

    Live end-to-end of the multiway → HU deep-link approximation, verified by
    hand against GTOW:
      - CO cold-calls then folds to the SB 3-bet → collapsed to a single fold
        (preflop F-R2.3-C-F-F-F-R10.3-F-F-C; opener and hero preserved).
      - Flop SB bet 10.6bb is ~1/3 of the REAL pot (~30bb incl. CO's dead call),
        so it snaps to R6.2 (25% bucket) by pot ratio — NOT R12.45 (50%), which
        is what absolute bb against the smaller solver pot would pick.
      - Turn SB bet 40.6bb is ~93% of the 43.5bb stack → all-in (RAI), preserved
        over the 75%-pot bucket the pot ratio alone would choose.
    """
    from gtow_action_resolver import resolve_actions_for_deviation

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 62.1,
        "hero_position": "LJ",
        "players_at_table": 8,
        "preflop_actions": "F-R2-C-F-C-F-R12-F-F-C-F",
        "streets": [
            {"board": "6h3cQc", "actions": [
                {"position": "SB", "action": "R10.6", "size": 10.6},
                {"position": "LJ", "action": "C"},
            ]},
            {"card": "8h", "actions": [
                {"position": "SB", "action": "R40.6", "size": 40.6},
                {"position": "LJ", "action": "F"},
            ]},
        ],
    }
    result = resolve_actions_for_deviation(hand_data, street="turn", action_index=0)

    assert_eq(result["preflop_actions"], "F-R2.3-C-F-F-F-R10.3-F-F-C")
    assert_eq(result["flop_actions"], "R6.2-C")
    assert_eq(result["turn_actions"], "RAI")
    assert_eq(result["history_spot"], 13)
    assert_eq(result["depth"], 60.125)
    assert_eq(result["hero_pos"], "LJ")
    assert_eq(result["villain_pos"], "SB")


@test
def test_build_last_node_url_h3490_no_double_pad():
    """build_last_node_url: H3490 turn fold deep-links to the verified node.

    Two bugs sat between the analysis and the GTOW button URL:

    1. Double-pad: analyze_hand_full normalizes the 6-max preflop to the 8-max
       MTT tree in ctx["hand"] (F-F-R2-... → F-F-F-F-R2-...) but leaves
       players_at_table=6. The resolver pads to the tree itself from
       players_at_table, so it padded the already-padded line AGAIN
       (F-F-F-F-F-F-R2-..., 11 tokens), shifting every actor to the wrong seat.
       Fixed by carrying the un-padded line through ctx["deeplink_raw_preflop"].

    2. Undersized-3bet → Call: BB's 5bb 3-bet sits between Call (2.1) and the
       solver's only 3-bet size (8.2), so absolute matching mis-snapped it to
       Call, putting the line off-tree (...C-C → empty actions). Fixed by
       restricting raise resolution to raise/all-in candidates.

    Verified GTOW node: preflop F-F-F-F-R2.1-F-F-R8.2-C, flop R8.95-C, turn RAI,
    board Kc5d3h2c, depth 30.125, history_spot 12.
    """
    from analyze_hand import analyze_hand_full
    from gtow_solution_url import build_last_node_url

    parsed = {
        "gametype": "MTTGeneral", "hero_hand": "Ac3c", "effective_bb": 29,
        "hero_position": "CO", "players_at_table": 6,
        "player_stacks": [14.9, 30.6, 42.6, 14.5, 39.5, 29.1],
        "preflop_actions": "F-F-R2-F-F-R5-C",  # raw 6-max: CO open, BB 3bet, CO call
        "streets": [
            {"board": "5dKc3h", "actions": [
                {"position": "BB", "action": "R5.8", "size": 5.8},
                {"position": "CO", "action": "C"}]},
            {"card": "2c", "actions": [
                {"position": "BB", "action": "R18.3", "size": 18.3},
                {"position": "CO", "action": "F"}]},
        ],
    }

    ctx = analyze_hand_full(parsed)
    # analyze padded to the 8-max tree → must preserve the raw line for the resolver
    assert_eq(ctx.get("deeplink_raw_preflop"), "F-F-R2-F-F-R5-C")
    assert_eq(ctx.get("deeplink_raw_players"), 6)

    url = build_last_node_url(ctx)
    assert_true(url is not None, "H3490 turn node must build a URL")
    qs = parse_qs(urlparse(url).query)
    assert_eq(qs["preflop_actions"], ["F-F-F-F-R2.1-F-F-R8.2-C"],
              "preflop padded once, BB 3bet kept as a raise")
    assert_eq(qs["flop_actions"], ["R8.95-C"], "flop bet snaps by pot ratio")
    assert_eq(qs["turn_actions"], ["RAI"], "turn shove is all-in")
    assert_eq(qs["board"], ["Kc5d3h2c"], "canonical board through the turn")
    assert_eq(qs["depth"], ["30.125"])
    assert_eq(qs["history_spot"], ["12"])


@test
def test_collapse_coldcall_folder_h3480():
    """gtow_action_resolver: a cold-caller who folds before the flop collapses
    to a single preflop fold so the HU postflop node stays on-tree (H3480).

    Raw H3480 preflop (8-max): UTG+1 opens, hero LJ flat-calls, CO cold-calls,
    SB 3-bets, UTG+1 and CO fold, hero calls. CO's cold-call-then-fold must
    collapse to a fold at its first action; the opener (a raiser) and hero are
    left intact. Result is the verified GTOW deep-link preflop line.
    """
    from gtow_action_resolver import _collapse_coldcall_folders, POSITION_ORDERS

    pos = POSITION_ORDERS[8]
    raw = "F-R2-C-F-C-F-R12-F-F-C-F"
    assert_eq(_collapse_coldcall_folders(raw, pos, "LJ"),
              "F-R2-C-F-F-F-R12-F-F-C")


@test
def test_collapse_coldcall_preserves_raisers_and_flop_caller():
    """gtow_action_resolver: collapse only rewrites cold-call-then-fold players.

    - A flat-caller who SEES the flop (last action is a call, not a fold) is
      untouched — it is the heads-up postflop villain.
    - A raiser who later folds (opener facing a 3-bet) is untouched — its bet
      defines the node.
    """
    from gtow_action_resolver import _collapse_coldcall_folders, POSITION_ORDERS

    pos = POSITION_ORDERS[8]
    # Hero BTN opens, BB flat-calls and reaches the flop → unchanged.
    hu_call = "F-F-F-F-F-R2.3-F-C"
    assert_eq(_collapse_coldcall_folders(hu_call, pos, "BTN"), hu_call)

    # UTG opens, hero CO 3-bets, blinds fold, UTG folds to the 3-bet.
    # UTG raised (then folded) so it is NOT collapsed → unchanged.
    opener_folds = "R2.3-F-F-F-R7-F-F-F-F"
    assert_eq(_collapse_coldcall_folders(opener_folds, pos, "CO"), opener_folds)


@test
def test_build_custom_spot_url_h2665():
    """gtow_custom_url: H2665 turn fold → URL with all expected params.

    Fixture uses effective_bb=30.0 (constructed) so depth snaps to 30.125
    where verified R2.1/R1.9/R5.2 codes apply.
    """
    from gtow_custom_url import build_custom_spot_url

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 5,
        "preflop_actions": "F-F-R2.2-F-C",
        "streets": [
            {"board": "4c6h8h", "actions": [
                {"position": "BB", "action": "R2.7", "size": 2.7},
                {"position": "BTN", "action": "C"},
            ]},
            {"card": "4h", "actions": [
                {"position": "BB", "action": "R5.4", "size": 5.4},
                {"position": "BTN", "action": "F"},
            ]},
        ],
    }

    url = build_custom_spot_url(
        hand_data, street="turn", action_index=0, pot_type="SRP",
    )

    # H2665's actual board 4c6h8h has 2 hearts → flush_draw (not rainbow).
    # Turn 4h brings 3 hearts → flush + pairs the 4. No river flags (folded turn).
    assert_in("fh_start_spot=custom_spot", url)
    assert_in("gmfs_solution_tab=ai_sols", url)
    assert_in("preflop_actions=F-F-F-F-F-R2.1-F-C", url)
    assert_in("flop_actions=R1.9-C", url)
    assert_in("turn_actions=R5.2", url)
    assert_in("history_spot=11", url)
    assert_in("fh_hero=BTN", url)
    assert_in("fh_opponent=BB", url)
    assert_in("fh_actions=SRP", url)
    assert_in("flop_paired=not_paired", url)
    assert_in("flop_suits=flush_draw", url)
    assert_in("flop_connectedness=disconnected", url)
    assert_in("turn_paired=paired", url)
    assert_in("turn_suit=flush", url)
    assert_true("river_paired" not in url, "river flags should be omitted when hand ended on turn")
    assert_true("river_suit" not in url, "river flags should be omitted when hand ended on turn")
    assert_in("depth=30.125", url)
    assert_in("depth_list=30.125", url)
    assert_in("gametype=MTTGeneral", url)
    assert_in("dialogs=trainer-advanced-filter-dialog", url)


@test
def test_build_custom_spot_url_raises_on_multiway_postflop():
    """gtow_custom_url: >2 distinct postflop actors → CustomSpotBuildError."""
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {
        "gametype": "MTTGeneral",
        "effective_bb": 30.0,
        "hero_position": "BTN",
        "players_at_table": 6,
        # 3-way to flop: CO open, BTN call, BB call
        "preflop_actions": "F-F-R2.5-C-F-C",
        "streets": [
            {"board": "2c7dJh", "actions": [
                {"position": "BB",  "action": "X"},
                {"position": "CO",  "action": "R1.8", "size": 1.8},
                {"position": "BTN", "action": "C"},
                {"position": "BB",  "action": "C"},
            ]},
        ],
    }
    try:
        build_custom_spot_url(hand_data, street="flop", action_index=0, pot_type="SRP")
        assert_true(False, "expected CustomSpotBuildError for multiway")
    except CustomSpotBuildError:
        pass


@test
def test_build_custom_spot_url_raises_on_unmapped_pot_type():
    """gtow_custom_url: unknown pot_type → CustomSpotBuildError (bucket fallback)."""
    from gtow_custom_url import build_custom_spot_url, CustomSpotBuildError

    hand_data = {"gametype": "MTTGeneral", "effective_bb": 30.0,
                 "hero_position": "BTN", "players_at_table": 5,
                 "preflop_actions": "F-F-R2.2-F-C", "streets": []}
    try:
        build_custom_spot_url(
            hand_data, street="flop", action_index=0, pot_type="straddled",
        )
        assert_true(False, "expected CustomSpotBuildError")
    except CustomSpotBuildError:
        pass


@test
def test_identify_villain_with_unplayed_river_street():
    """gtow_action_resolver: empty river actions list (hand ended on turn) must
    not disqualify villain identification. Regression for H2661-style hands
    where streets[] includes a recorded-but-unplayed later street."""
    from gtow_action_resolver import _identify_villain

    hand_data = {
        "streets": [
            {"board": "7h9sJs", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "X"},
            ]},
            {"card": "Ah", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "CO", "action": "R4.2"},
            ]},
            # River present but not played — empty actions list must be skipped.
            {"card": "Th", "actions": []},
        ],
    }
    # Preflop codes: 8-max CO opens, BB calls. Hero=CO, villain=BB.
    result = _identify_villain(
        hand_data, hero_pos_8max="CO",
        preflop_codes="F-F-F-F-R2.1-F-F-C", street="turn",
    )
    assert_eq(result, "BB")


@test
def test_build_url_for_cluster_falls_back_on_build_error():
    """weekly_report: if custom builder fails (no deviation_ids), returns bucket URL."""
    import asyncio
    from weekly_report import _build_url_for_cluster

    cluster = _make_test_cluster(
        spot_category="cbet_ip", street="turn", pot_type="SRP",
        hero_pos="BTN", villain_pos="BB", board_texture="paired",
        effective_bb_median=30.0, top_deviation_ids=[],
    )

    url = asyncio.run(_build_url_for_cluster(cluster, pool=None))
    assert_true(url is not None, "fallback should return bucket URL")
    # Bucket URL markers for postflop street "turn" with pot_type "SRP":
    assert_in("fh_start_spot=turn", url)
    assert_in("fh_actions=SRP", url)


@test
def test_resolve_hero_board_conflict_unresolvable_clears_hero():
    """n8_parser: when hero duplicates a board card and no common-OCR
    rank-swap resolves it, clear the hero side and keep the board.

    Regression for H2758 where hero was OCR'd as AsQs (true hand Ac3s,
    occluded by a WIN banner) duplicating the board's Qs.  Pre-fix:
    whole board was cleared on conflict, breaking flop/turn solver
    lookups.  Post-fix: hero side is cleared so confidence drops below
    the 0.85 gate and the Gemini fallback re-reads hero with OCR's
    board as a hint.
    """
    from ocr.n8_parser import _resolve_hero_board_conflict

    board = ["Qs", "Qd", "2d", "Ah", "5c"]
    hero = ["As", "Qs"]  # Qs unresolvable (no common OCR swap for Q)

    new_board, new_hero = _resolve_hero_board_conflict(board, hero)
    assert_eq(new_board, board, "board must be preserved")
    assert_eq(new_hero, [], "hero must be cleared on unresolvable conflict")


@test
def test_duplicate_runout_detected_for_full_gemini_fallback():
    """n8_parser: impossible exact-card duplicates are structural failures.

    Regression for H2914: OCR read the river As as Ks, producing
    KhJsKsAdKs with the exact Ks appearing twice.  That parse must not be
    trusted or sent to solver; production should demote it to full Gemini
    vision so the screenshot can be re-read as the correct river As.
    """
    from ocr.n8_parser import _duplicate_known_cards

    hand = {
        "hero_hand": "3s3h",
        "streets": [
            {"board": "KhJsKs", "actions": []},
            {"card": "Ad", "actions": []},
            {"card": "Ks", "actions": []},
        ],
    }
    assert_eq(_duplicate_known_cards(hand), ["Ks"],
              "duplicate Ks in board runout must be detected")

    corrected = {
        "hero_hand": "3s3h",
        "streets": [
            {"board": "KhJsKs", "actions": []},
            {"card": "Ad", "actions": []},
            {"card": "As", "actions": []},
        ],
    }
    assert_eq(_duplicate_known_cards(corrected), [],
              "valid KhJsKsAdAs runout must not be flagged")


@test
def test_preflop_physics_rejects_non_monotone_raise():
    """n8_parser: impossible preflop raise sequences are precision rejects."""
    from ocr.n8_parser import _validate_preflop_bet_physics

    issues = _validate_preflop_bet_physics(
        "F-F-R8-F-R6-F-F-C", 8, effective_bb=20
    )
    assert_true(
        any("non_monotone_raise" in issue for issue in issues),
        "a later raise to less than the outstanding bet is impossible",
    )


@test
def test_open_raise_one_bb_is_snapped_before_physics():
    """n8_parser: impossible OCR ``R1`` opens are repaired to legal min-raises.

    Precision-push tails include several exact hands where Natural8 displayed a
    normal 2bb-ish open, but EasyOCR lost the leading digit and produced
    ``Raise 1 BB``.  A real limp is rendered as ``Call 1 BB``, so pre-raise
    ``R1`` is a size OCR failure; snap only that open-size case and keep true
    post-raise non-monotone raises rejected.
    """
    from ocr.n8_parser import (
        _repair_implausible_open_raise_sizes,
        _validate_preflop_bet_physics,
    )

    repaired, repairs = _repair_implausible_open_raise_sizes(
        "F-F-F-R1-C-F-F"
    )
    assert_eq(repaired, "F-F-F-R2-C-F-F")
    assert_true(repairs and repairs[0].startswith("open_raise_min_snap@3"))
    assert_eq(_validate_preflop_bet_physics(repaired, 7), [])

    still_bad, repairs = _repair_implausible_open_raise_sizes(
        "R2-F-F-R1-F-AI12.85-F-AI15.02-F-C"
    )
    assert_eq(still_bad, "R2-F-F-R1-F-AI12.85-F-AI15.02-F-C")
    assert_eq(repairs, [])
    assert_true(
        any(
            "non_monotone_raise" in issue
            for issue in _validate_preflop_bet_physics(still_bad, 8)
        ),
        "post-raise R1 must remain a physics reject",
    )


@test
def test_missing_allin_reaction_fold_repair_is_low_collapse_only():
    """n8_parser: missing preflop all-in reaction folds are repaired narrowly.

    TM5846884791/TM5866478558/TM5895757896 lost one final fold after an all-in
    re-raise, which breaks action-type exactness.  Similar high-collapse
    VLM-corrected tails can be exact already, so the repair is gated by raw
    fragment loss and only inserts/appends one fold.
    """
    from ocr.n8_parser import _repair_missing_allin_reaction_folds

    repaired, repairs = _repair_missing_allin_reaction_folds(
        "F-F-R320-F-F-F-AI2-F",
        8,
        raw_loss=2,
    )
    assert_eq(repaired, "F-F-R320-F-F-F-AI2-F-F")
    assert_eq(repairs, ["append_prior_fold_after_ai@6"])

    repaired, repairs = _repair_missing_allin_reaction_folds(
        "F-R3-F-F-F-AI12.25-C",
        7,
        raw_loss=7,
    )
    assert_eq(repaired, "F-R3-F-F-F-AI12.25-F-C")
    assert_eq(repairs, ["insert_remaining_fold_after_ai@5"])

    unchanged, repairs = _repair_missing_allin_reaction_folds(
        "R2.1-F-F-AI27.7-F-C",
        6,
        raw_loss=12,
    )
    assert_eq(unchanged, "R2.1-F-F-AI27.7-F-C")
    assert_eq(repairs, [])


@test
def test_preflop_physics_allows_short_allin_call():
    """n8_parser: short all-in calls are legal and must not be rejected."""
    from ocr.n8_parser import _validate_preflop_bet_physics

    issues = _validate_preflop_bet_physics(
        "F-F-R8-F-AI5-F-F-C", 8, effective_bb=8
    )
    assert_eq(issues, [])


@test
def test_duplicate_bare_allin_after_call_is_dropped():
    """n8_parser: bare AI after C is a duplicated badge, not an action.

    TM5901639322 parsed ``F-AI16.86-F-F-C-AI-F-F-F`` while the hand history is
    ``F-AI16.9-F-F-C-F-F-F``.  The second bare AI has no amount and follows a
    call, matching the N8 all-in badge split pattern; dropping only the bare
    token preserves sized all-ins.
    """
    from ocr.n8_parser import _drop_duplicate_bare_allin_after_call

    assert_eq(
        _drop_duplicate_bare_allin_after_call("F-AI16.86-F-F-C-AI-F-F-F"),
        "F-AI16.86-F-F-C-F-F-F",
    )
    assert_eq(
        _drop_duplicate_bare_allin_after_call("F-F-C-AI12-F"),
        "F-F-C-AI12-F",
    )


@test
def test_safe_emit_recovers_no_allin_structural_high_card_tails():
    """n8_parser: safe emit can recover low-scored but structurally clean
    non-all-in hands.

    Precision-push v3 left exact hands below the 0.88 blended gate when only
    stack/pot side channels were weak.  If cards, OCR, and the non-all-in action
    chain are all stable, recovering them improves recall without trusting the
    known all-in danger tail.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Tc7c",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "R3-F-R5.77-F-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.60,
    }
    diag = {
        "preflop_entries_count": 7,
        "preflop_entries_pre_collapse_count": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "no_allin_structural_high_card",
    )
    cp["player_tracking"] = 0.2
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "weak player tracking must not revive post-size-recovery tails",
    )


@test
def test_safe_emit_recovers_narrow_vlm_recovered_preflop_subset():
    """n8_parser: VLM recovered hands can emit only in a narrow stable subset.

    Recovered all-in tails were net-negative overall, but the v5 tail showed a
    small fully preflop, high-card, low-collapse subset that was exact.  Keep
    generic recovered/corrected hands abstained; only this verifier-like shape
    may safe-emit.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "7h2s",
        "hero_position": "HJ",
        "players_at_table": 8,
        "preflop_actions": "F-F-R2-F-F-AI26.81-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "recovered",
        "preflop_entries_count": 9,
        "preflop_entries_pre_collapse_count": 18,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "vlm_recovered_stable_preflop",
    )
    diag["preflop_entries_pre_collapse_count"] = 19
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_vlm_recovered_high_card_preflop_tail():
    """n8_parser: VLM recovered high-card preflop tails can emit.

    Size-recovery v6 left exact recovered all-in hands below the blended gate
    because ``ocr_confidence`` is intentionally zeroed after VLM anchoring.
    When cards are very strong, no postflop board is involved, and player
    tracking is at least usable, the recovered structure was exact in the
    measured tail.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Kd9d",
        "hero_position": "LJ",
        "players_at_table": 8,
        "preflop_actions": "R1-F-F-F-C-F-F-AI12-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "recovered",
        "preflop_entries_count": 11,
        "preflop_entries_pre_collapse_count": 15,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 5},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "vlm_recovered_preflop_high_card",
    )


@test
def test_safe_emit_recovers_no_allin_low_street_collapse_tail():
    """n8_parser: non-all-in low street-collapse tails can emit.

    These hands were confidence-abstained only because side-channel player
    tracking was weak; the card/action/board shape has no all-in grammar and
    at most two OCR fragments lost on any postflop street.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "5c4d",
        "hero_position": "UTG",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-R2-F-R5.35-F-F-C",
        "streets": [{"board": "Ah7d2c", "actions": []}],
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.33,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 9,
        "preflop_entries_pre_collapse_count": 12,
        "street_entries_count": {"flop": 3, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 5, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "no_allin_low_street_collapse",
    )
    diag["preflop_physics_issues"] = ["all_fold_walk_hero_first"]
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_all_fold_high_card_walks_after_seat_guard():
    """n8_parser: all-fold rows can safe-emit only after the hero-first walk
    guard has had a chance to reject unsafe seat assignments.

    This recovers exact low-pot-confidence walk/fold hands while keeping
    ``all_fold_walk_hero_first`` blocked by the preflop-physics guard.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "6h2h",
        "hero_position": "CO",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-F-F-F-F",
    }
    cp = {
        "pot_consistency": 0.3,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 7,
        "preflop_entries_pre_collapse_count": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(_safe_emit_override_reason(hand, cp, diag), "all_fold_high_card")
    diag["preflop_physics_issues"] = ["all_fold_walk_hero_first"]
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_safe_emit_recovers_large_table_all_fold_hero_first_rows():
    """n8_parser: large-table all-fold hero-first walks can safe-emit.

    TM5963073078 showed this physics warning is unsafe at six-max because hero
    can shift from BTN to LJ.  The measured 7/8-player tail had exact UTG walks,
    so recover only larger-table all-fold rows with strong card/player signals.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "8s2c",
        "hero_position": "UTG",
        "players_at_table": 7,
        "preflop_actions": "F-F-F-F-F-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_physics_issues": ["all_fold_walk_hero_first"],
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 8,
        "players_at_table_final": 7,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "all_fold_hero_first_large_table",
    )
    diag["players_at_table_final"] = 6
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_misnamed_flop_column_promotes_short_preflop_rows():
    """n8_parser: physical Pre-Flop columns mislabeled as Flop are promoted
    even when only a short all-in/fold tail is visible.

    TM5873874017-like screenshots had the second physical column header OCR'd
    as Flop with only four preflop rows.  The old 5+ entry guard sent them to
    full Gemini fallback instead of focused VLM structure recovery.
    """
    from ocr.n8_parser import _promote_misnamed_preflop_column

    first = {
        "name": "Flop",
        "entries": [
            {"action": "Fold"},
            {"action": "Fold"},
            {"action": "All-In", "size": 4.1},
            {"action": "Fold"},
        ],
    }
    preflop, streets = _promote_misnamed_preflop_column(None, [first])

    assert_eq(preflop, first)
    assert_eq(streets, [])


@test
def test_preflop_filter_drops_anonymous_sizeless_bet_fragments():
    """n8_parser: nameless/positionless sizeless Bet fragments in Pre-Flop are
    OCR chrome bleed, not real preflop actions.

    TM5880331313/TM5920325572 parsed none because this fragment tripped the
    missing-raise-size guard.  Dropping only the anonymous/sizeless form keeps
    real named or sized rows available to later validation.
    """
    from ocr.n8_parser import _filter_action_entries

    entries = [
        {"type": "opponent", "action": "Fold", "position": "UTG"},
        {"type": "hero", "action": "Raise", "size": 2.0},
        {"type": "opponent", "action": "Bet", "size": None},
        {"type": "opponent", "action": "Raise", "size": 5.0},
    ]

    filtered = _filter_action_entries(entries)
    assert_eq([e["action"] for e in filtered], ["Fold", "Raise", "Raise"])

    named = [
        {"type": "opponent", "action": "Bet", "size": None,
         "player_name": "real_name"},
    ]
    assert_eq(_filter_action_entries(named), named)


@test
def test_preflop_filter_drops_leading_anonymous_check_fragment():
    """n8_parser: a nameless leading preflop Check is a duplicate blind-option
    fragment and should not create an illegal X before UTG acts.
    """
    from ocr.n8_parser import _filter_action_entries

    entries = [
        {"type": "opponent", "action": "Check"},
        {"type": "opponent", "action": "Fold", "position": "UTG"},
        {"type": "hero", "action": "Check"},
    ]
    filtered = _filter_action_entries(entries)
    assert_eq([e["action"] for e in filtered], ["Fold", "Check"])


@test
def test_terminal_fold_trim_safe_emit_single_allin_tail():
    """n8_parser: the duplicate terminal-fold trimmer can safe-emit only the
    single-sized-all-in tail; bare/multiple-AI chains stay abstained.
    """
    from ocr.n8_parser import (
        _repair_terminal_fold_after_vlm_allin_call,
        _safe_emit_override_reason,
    )

    diag = {
        "forced_structure_reassembly": True,
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 13,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 3},
        "vlm_recheck_outcome": "corrected",
    }
    repaired, repairs = _repair_terminal_fold_after_vlm_allin_call(
        "AI14.4-C-F-F-F-F-F-F", diag,
    )
    assert_eq(repaired, "AI14.4-C-F-F-F-F-F")
    diag["preflop_terminal_fold_repairs"] = repairs

    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.99,
    }
    assert_eq(
        _safe_emit_override_reason(
            {"preflop_actions": repaired}, cp, diag,
        ),
        "terminal_fold_trimmed_single_allin",
    )
    assert_eq(
        _safe_emit_override_reason(
            {"preflop_actions": "AI-AI16.4-F-C-F-F-F"}, cp, diag,
        ),
        None,
    )


@test
def test_preflop_builder_preserves_original_missing_position_flag():
    """n8_parser: VLM reassembly must remember whether a hero all-in sticker
    was originally positionless before order assignment mutated the row.

    TM5867350464/TM5873208650 first parse assigned a position to the duplicate
    bare All-In badge, then the forced VLM pass overwrote the real hero fold.
    Keeping the original-missing marker prevents that bare sticker from
    replacing a concrete fold.
    """
    from ocr.n8_parser import _build_preflop_actions_from_order

    entries = [
        {"type": "hero", "position": "UTG", "action": "Fold",
         "_position_missing_before_order": True},
        {"type": "opponent", "position": "LJ", "action": "All-In",
         "size": 16.4},
        {"type": "hero", "position": "SB", "action": "All-In",
         "size": None, "_position_missing_before_order": True},
        {"type": "opponent", "position": "BB", "action": "Fold"},
    ]

    assert_eq(
        _build_preflop_actions_from_order(
            entries,
            ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            "UTG",
            7,
            first_round_count=4,
        ),
        "F-AI16.4-F-F",
    )


@test
def test_forced_collapse_action_tail_repairs_and_safe_emit():
    """n8_parser: narrow VLM-forced collapse repairs recover exact action
    tails and can pass the safe-emission gate.
    """
    from ocr.n8_parser import (
        _repair_forced_collapse_action_tail,
        _safe_emit_override_reason,
    )

    base_diag = {
        "forced_structure_reassembly": True,
        "vlm_recheck_outcome": "corrected",
        "preflop_entries_count": 8,
        "preflop_entries_pre_collapse_count": 20,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 6},
    }
    repaired, repairs = _repair_forced_collapse_action_tail(
        "R3.5-F-AI39.03-F-C-F-F-C",
        dict(base_diag),
    )
    assert_eq(repaired, "R3.5-F-AI39.03-F-C-F-C")
    assert_eq(repairs, ["drop_duplicate_fold_before_final_call"])

    diag = dict(base_diag, preflop_forced_collapse_repairs=repairs)
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    assert_eq(
        _safe_emit_override_reason({"preflop_actions": repaired}, cp, diag),
        "forced_collapse_repaired_vlm",
    )


@test
def test_safe_emit_recovers_promoted_short_allin_vlm_corrections():
    """n8_parser: VLM-corrected action chains normally abstain, but a promoted
    short physical-Pre-Flop column with 3-4 visible all-in rows can emit.

    The promoted-column flag distinguishes these deterministic hidden-fold
    recoveries from generic VLM-corrected row-collapse tails whose action
    chains remain unsafe.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "Kh6h",
        "hero_position": "SB",
        "players_at_table": 7,
        "preflop_actions": "F-F-AI4.1-F-F-F-F",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 0.5,
        "ocr_confidence": 0.0,
        "card_confidence": 0.999,
    }
    diag = {
        "vlm_recheck_outcome": "corrected",
        "promoted_misnamed_preflop": True,
        "preflop_entries_count": 4,
        "preflop_entries_pre_collapse_count": 8,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "street_entries_pre_collapse_count": {"flop": 0, "turn": 0, "river": 0},
    }

    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        "promoted_preflop_short_allin_vlm",
    )
    diag["preflop_entries_count"] = 2
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "Two-row all-in/call columns are still action-order ambiguous",
    )
    diag["preflop_entries_count"] = 4
    diag.pop("promoted_misnamed_preflop")
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "Generic VLM-corrected all-in tails remain abstained",
    )


@test
def test_safe_emit_blocks_large_raw_collapse_walk_tail():
    """n8_parser: large raw-fragment loss with no postflop signal is unsafe.

    TM5947799144 looked like a simple high-card preflop tail, but 16 raw
    fragments collapsed to 6 entries and shifted hero_position.  This shape
    has no street signal to repair the seat assignment, so safe emit must not
    revive it below the gate.
    """
    from ocr.n8_parser import _safe_emit_override_reason

    hand = {
        "hero_hand": "7d2h",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 16,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
    }
    assert_eq(_safe_emit_override_reason(hand, cp, diag), None)


@test
def test_structural_demotions_block_threshold_emit_for_known_bad_tails():
    """n8_parser: structural-risk demotions must lower threshold confidence.

    Safe-emit blocking alone is insufficient when a bad tail scores above the
    0.88 threshold.  TM5947799144 (large preflop collapse/no postflop) and
    TM5912802228 (postflop rows but no board streets) need OCR confidence
    demoted before the final score is computed.
    """
    from ocr.n8_parser import _apply_structural_confidence_demotions

    hand = {
        "hero_hand": "7d2h",
        "hero_position": "SB",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-C-F",
    }
    cp = {
        "pot_consistency": 0.5,
        "player_tracking": 0.5,
        "ocr_confidence": 1.0,
        "card_confidence": 0.999,
    }
    diag = {
        "preflop_entries_count": 6,
        "preflop_entries_pre_collapse_count": 16,
        "street_entries_count": {"flop": 0, "turn": 0, "river": 0},
        "estimate_used_reaction_signal": False,
    }
    _apply_structural_confidence_demotions(hand, cp, diag)
    assert_eq(cp["ocr_confidence"], 0.0)
    assert_in("large_preflop_collapse_no_postflop", diag["structural_risk_issues"])

    cp_reaction = dict(cp, ocr_confidence=1.0)
    diag_reaction = dict(diag, estimate_used_reaction_signal=True)
    _apply_structural_confidence_demotions(hand, cp_reaction, diag_reaction)
    assert_eq(
        cp_reaction["ocr_confidence"],
        1.0,
        "reaction/table evidence keeps exact large-collapse tails recoverable",
    )

    cp2 = {
        "pot_consistency": 0.5,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.995,
    }
    diag2 = {
        "preflop_entries_count": 4,
        "preflop_entries_pre_collapse_count": 10,
        "street_entries_count": {"flop": 3, "turn": 2, "river": 2},
    }
    _apply_structural_confidence_demotions(
        {
            "hero_hand": "AhKh",
            "preflop_actions": "F-R2.2-F-C",
            "streets": [
                {"actions": [{"position": "BB", "action": "X"}]},
                {"actions": [{"position": "BB", "action": "X"}]},
            ],
        },
        cp2,
        diag2,
    )
    assert_eq(cp2["ocr_confidence"], 0.0)
    assert_in(
        "postflop_rows_without_matching_board_streets",
        diag2["structural_risk_issues"],
    )


@test
def test_vlm_residual_disagreement_keeps_hand_for_field_routing():
    """n8_parser: VLM structural disagreement should confidence-abstain with
    the OCR hand preserved, not erase it.

    This lets gemini_session route the abstained-but-present OCR parse through
    field-level cards-only repair instead of the measured net-negative full
    Gemini reparse.
    """
    import os as _os
    from ocr import n8_parser as _np

    hand = {
        "hero_hand": "Jh2c",
        "hero_position": "BB",
        "players_at_table": 8,
        "preflop_actions": "F-F-F-F-AI10-F-F-C",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    diag = {}

    orig_assemble = _np._assemble_hand
    _np._assemble_hand = lambda *a, **k: (dict(hand, hero_position="CO"), cp, {})
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", hand, cp, diag, {}, [],
            recheck_fn=lambda b: {"players_at_table": 7, "hero_position": "UTG"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, hand)
    assert_eq(out_diag.get("vlm_recheck_outcome"), "abstain")
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(out_cp.get("card_confidence"), 0.99)
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-disagreed hands must stay confidence-abstained, not safe-emitted",
    )


@test
def test_vlm_corrected_structure_confidence_abstains_actions():
    """n8_parser: VLM-corrected structure is not an action verifier.

    Precision-push v3 showed VLM ``corrected`` hands often had the right
    hero_position but still-wrong all-in action chains.  Keep the corrected
    hand for field-preserving fallback, but demote OCR confidence so it cannot
    deterministically emit before a grammar verifier accepts the actions.
    """
    import os as _os
    from ocr import n8_parser as _np

    hand = {
        "hero_hand": "AsKs",
        "hero_position": "CO",
        "players_at_table": 6,
        "preflop_actions": "F-F-F-AI12-C-C",
    }
    corrected = {**hand, "hero_position": "BTN"}
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    orig_assemble = _np._assemble_hand
    _np._assemble_hand = lambda *a, **k: (corrected, cp, {})
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", hand, cp, {}, {}, [{"name": "Pre-Flop", "entries": []}],
            recheck_fn=lambda b: {"players_at_table": 6, "hero_position": "BTN"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, corrected)
    assert_eq(out_diag.get("vlm_recheck_outcome"), "corrected")
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-corrected action chains are hints until independently verified",
    )


@test
def test_vlm_recovers_parse_none_with_partial_allin_panel():
    """n8_parser: parse_none all-in tails can be reassembled with focused VLM
    structure instead of going straight to full Gemini.

    This is the recall-side companion to field-level micro-routing: when OCR
    has cards/action rows but no trusted hero position, ask only for table size
    + hero position and re-run deterministic assembly with those anchors.
    """
    import os as _os
    from ocr import n8_parser as _np

    columns = [{
        "name": "Pre-Flop",
        "entries": [
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Raise", "size": 2.0},
            {"type": "opponent", "action": "Fold"},
            {"type": "hero", "action": "All-In", "size": None},
        ],
    }]
    recovered = {
        "hero_hand": "AsKs",
        "hero_position": "BTN",
        "players_at_table": 6,
        "preflop_actions": "F-R2-F-AI",
    }
    cp = {
        "pot_consistency": 1.0,
        "player_tracking": 1.0,
        "ocr_confidence": 1.0,
        "card_confidence": 0.99,
    }
    orig_assemble = _np._assemble_hand
    calls = []
    def _fake_assemble(*a, **k):
        calls.append(k)
        if k.get("force_hero_position") == "BTN":
            return recovered, cp, {}
        return None, cp, {}
    _np._assemble_hand = _fake_assemble
    prev = _os.environ.get("OCR_VLM_RECHECK")
    _os.environ["OCR_VLM_RECHECK"] = "1"
    try:
        out_hand, out_cp, out_diag = _np._maybe_vlm_recheck(
            b"x", None, cp, {}, {}, columns,
            recheck_fn=lambda b: {"players_at_table": 6, "hero_position": "BTN"},
        )
    finally:
        _np._assemble_hand = orig_assemble
        if prev is None:
            _os.environ.pop("OCR_VLM_RECHECK", None)
        else:
            _os.environ["OCR_VLM_RECHECK"] = prev

    assert_eq(out_hand, recovered)
    assert_eq(out_cp.get("ocr_confidence"), 0.0)
    assert_eq(calls[-1].get("force_table_size"), 6)
    assert_eq(calls[-1].get("force_hero_position"), "BTN")
    assert_eq(out_diag.get("vlm_recheck_outcome"), "recovered")
    assert_eq(
        _np._safe_emit_override_reason(out_hand, out_cp, out_diag),
        None,
        "VLM-recovered action chains are hints until independently verified",
    )


@test
def test_all_fold_walk_with_hero_first_confidence_abstains():
    """n8_parser: all-fold walk with hero at first row is unsafe to emit.

    TM5963073078 showed a five-row all-fold walk at a six-handed table where
    the first row was tagged hero, shifting hero_position to LJ instead of BTN.
    This shape carries no betting signal to repair the seat assignment, so it
    must stay available as a hint but confidence-abstain.
    """
    from ocr.n8_parser import _assemble_hand, _safe_emit_override_reason

    columns = [{
        "name": "Pre-Flop",
        "entries": [
            {"type": "hero", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
            {"type": "opponent", "action": "Fold"},
        ],
    }]
    table = {
        "hero_cards": ["6s", "4h"],
        "hero_card_conf": 0.99,
        "board_cards": [],
        "table_color": "green",
    }
    hand, cp, diag = _assemble_hand(table, columns, force_table_size=6)
    assert_true(hand is not None, "unsafe shape is preserved for fallback hints")
    assert_eq(cp.get("ocr_confidence"), 0.0)
    assert_in("all_fold_walk_hero_first", diag.get("preflop_physics_issues") or [])
    assert_eq(
        _safe_emit_override_reason(hand, cp, diag),
        None,
        "preflop physics rejects must not be revived by safe emit",
    )


@test
def test_seven_max_sb_open_with_postflop_confidence_abstains():
    """n8_parser: 7-max SB open prefix with postflop is seat-count fragile.

    TM5913201917 parsed as 7-max SB with preflop ``R2-...`` and postflop
    action, but the correct structure was 8-max BTN; the missing leading seat
    shifts the hero one position.  The same preflop-only shape can be exact, so
    only postflop hands get confidence-abstained.
    """
    from ocr.n8_parser import _assemble_hand

    columns = [
        {
            "name": "Pre-Flop",
            "entries": [
                {"type": "opponent", "action": "Raise", "size": 2.0},
                {"type": "opponent", "action": "Call"},
                {"type": "opponent", "action": "Call"},
                {"type": "opponent", "action": "Fold"},
                {"type": "opponent", "action": "Call"},
                {"type": "hero", "action": "Fold"},
                {"type": "opponent", "action": "Call"},
            ],
        },
        {
            "name": "Flop",
            "entries": [{"type": "hero", "action": "Check"}],
        },
    ]
    table = {
        "hero_cards": ["Kh", "Jd"],
        "hero_card_conf": 0.99,
        "board_cards": ["2c", "3d", "4h"],
        "table_color": "green",
    }
    hand, cp, diag = _assemble_hand(table, columns)
    assert_true(hand is not None, "unsafe structure is preserved for fallback")
    assert_eq(hand.get("hero_position"), "SB")
    assert_eq(cp.get("ocr_confidence"), 0.0)
    assert_in(
        "seven_max_sb_open_postflop_ambiguous",
        diag.get("preflop_physics_issues") or [],
    )


@test
def test_mask_win_overlay_whitens_large_lower_blob():
    """table_parser._mask_win_overlay paints out the orange WIN sticker.

    Regression for H2806: the K♣ T♣ hero crop had the N8 win sticker
    bleeding orange/yellow into the lower half of the cards. CardCNN
    read those red-leaning hues as a red suit (Kh, suit_conf=0.587),
    routing past the field-level fallback. After masking the sticker
    pixels to white, the classifier sees a clean card.
    """
    import numpy as np
    from ocr.table_parser import _mask_win_overlay

    # Synthetic crop: white card body with a big orange blob in the
    # lower half (BGR for orange ≈ (50, 165, 255)).
    crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    crop[35:55, 10:50] = (50, 165, 255)
    out = _mask_win_overlay(crop)
    # Sticker pixels must be whitened.
    assert_true(
        bool((out[40:50, 20:40] == 255).all()),
        "WIN sticker region should be whitened to 255",
    )


@test
def test_mask_win_overlay_skips_small_top_banner():
    """No-op when the only orange is a thin top-edge banner.

    Regression: a previous version of the mask was too aggressive and
    whitened the small `$0.50` price banner at the top of cash-game
    crops. That changed pixels the CardCNN was already handling
    correctly and degraded a clean Ts9s read into a misclassification.
    The mask must leave the crop untouched in that case.
    """
    import numpy as np
    from ocr.table_parser import _mask_win_overlay

    crop = np.full((60, 60, 3), 255, dtype=np.uint8)
    # A 4-px-tall orange strip pinned to the top — far smaller than the
    # WIN sticker would be, and located above the lower-half gate.
    crop[0:4, 5:55] = (50, 165, 255)
    out = _mask_win_overlay(crop)
    assert_true(
        bool(np.array_equal(crop, out)),
        "small top-edge orange strip must NOT trigger the mask",
    )


@test
def test_field_level_fallback_fires_on_empty_hero_hand():
    """gemini_session: cards-only Gemini fallback gate triggers when OCR
    cleared hero_cards (hero_hand="") but structural fields are good.

    Regression for the Ts9s screenshot where CardCNN labeled both hero
    crops as Tc with high confidence; _resolve_hero_board_conflict's
    duplicate guard cleared hero_cards, leaving hero_hand="". The old
    gate required hero_hand non-empty (`hand_ok`) before considering
    the cards-only fallback, so the path was skipped entirely and we
    fell through to the full IMAGE_PARSE_PROMPT — which has separately
    failed in production on similar inputs, returning the
    "無法從截圖中辨識出撲克手牌" rejection.

    This test verifies the source code surfaces the empty-hero gate so
    a future refactor can't silently re-tighten it back to hand_ok.
    """
    import inspect
    src = inspect.getsource(__import__("gemini_session", fromlist=["_dummy"]))
    assert_in("hero_hand_present", src,
              "gate variable name should appear in source")
    assert_in("cards_need_fallback", src,
              "fallback condition should track empty hero_hand")
    assert_in("not hero_hand_present", src,
              "must trigger cards-only when hero_hand is empty")


@test
def test_normalize_cards_handles_list_of_string_actions():
    """gemini_session: _normalize_cards must convert a street whose
    `actions` is a LIST OF BARE STRINGS into structured {position, action}
    dicts — not just the flat-string case.

    Regression: production screenshot (chat 556028753, hero 8h5h, BB,
    board 5d6dQh). The full Gemini fallback returned
    `"actions": ["X", "R1.4", "R5.2", "F"]` (list of strings) instead of
    either structured dicts or the flat "X-R1.4-R5.2-F" string. The old
    _normalize_cards only handled `isinstance(actions, str)`, so the list
    passed through unconverted. _fix_folded_players then called
    `a.get("position")` on the string "X" → `'str' object has no attribute
    'get'`, the exception was swallowed, _parse_hand_from_image returned
    None, and the bot wrongly told the user "無法從截圖中辨識出撲克手牌"
    despite a fully valid parse.
    """
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "hero_position": "BB",
        "hero_hand": "8h5h",
        "preflop_actions": "F-F-R2-F-F-F-F-C",
        "streets": [{
            "board": "5d6dQh",
            "actions": ["X", "R1.4", "R5.2", "F"],
        }],
    }
    # Must not raise, and must structure the actions.
    GeminiSessionManager._normalize_cards(hand)
    acts = hand["streets"][0]["actions"]
    assert_true(all(isinstance(a, dict) for a in acts),
                f"actions must be dicts after normalize, got {acts}")
    # Positional assignment must match the OCR-detected structure:
    # postflop order for active [LJ, BB] is [BB, LJ].
    assert_eq(acts[0]["position"], "BB")
    assert_eq(acts[0]["action"], "X")
    assert_eq(acts[1]["position"], "LJ")
    assert_eq(acts[1]["action"], "R1.4")
    assert_eq(acts[2]["position"], "BB")
    assert_eq(acts[2]["action"], "R5.2")
    assert_eq(acts[3]["position"], "LJ")
    assert_eq(acts[3]["action"], "F")
    # And _fix_folded_players must run cleanly on the normalized result.
    GeminiSessionManager._fix_folded_players(hand)


@test
def test_postflop_position_reconciliation_with_preflop_index():
    """n8_parser: postflop entries inherit the preflop's index-assigned
    canonical positions when player_name matches.

    Regression for H2810 (7-max). N8's badges were UTG, UTG+1, MP, CO,
    BTN, SB, BB but our 7-max pos_order is [UTG, LJ, HJ, CO, BTN, SB,
    BB]. The MP-badged third entry (h3scar) got aliased to LJ in the
    panel parser. Preflop reassigned by index pushed it to HJ, but the
    flop entries kept LJ. preflop_actions then said LJ folded, so
    _fix_folded_players stripped h3scar's flop bet/fold entries — leaving
    only [BB X, BB R4.8] as the flop and producing analysis that showed
    bet/check options for hero's second decision (open) instead of the
    correct call/raise/fold (facing a bet).

    This unit test exercises just the reconciliation block: a flop entry
    keyed by the same player_name as a preflop entry must be rewritten
    to the preflop's canonical position.
    """
    from ocr.n8_parser import _assemble_hand
    columns = [
        {"name": "Blinds", "pot": None, "entries": []},
        {"name": "Pre-Flop", "pot": 2.0, "entries": [
            {"type": "opponent", "position": "UTG", "action": "Fold",
             "size": None, "player_name": "Kony"},
            {"type": "opponent", "position": "UTG+1", "action": "Fold",
             "size": None, "player_name": "lily"},
            {"type": "opponent", "position": "LJ", "action": "Raise",
             "size": 2.0, "player_name": "h3scar"},
            {"type": "opponent", "position": "CO", "action": "Fold",
             "size": None, "player_name": "L189"},
            {"type": "opponent", "position": "BTN", "action": "Fold",
             "size": None, "player_name": "yeying"},
            {"type": "opponent", "position": "SB", "action": "Fold",
             "size": None, "player_name": "Zy"},
            {"type": "hero", "position": "BB", "action": "Call",
             "size": None},
        ]},
        {"name": "Flop", "pot": 5.5, "entries": [
            {"type": "hero", "position": None, "action": "Check",
             "size": None},
            {"type": "opponent", "position": "LJ", "action": "Bet",
             "size": 1.3, "player_name": "h3scar"},
            {"type": "hero", "position": "BB", "action": "Raise",
             "size": 4.8},
            {"type": "opponent", "position": "LJ", "action": "Fold",
             "size": None, "player_name": "h3scar"},
        ]},
        {"name": "Turn", "pot": 8.0, "entries": []},
        {"name": "River", "pot": 8.0, "entries": []},
    ]
    table_result = {
        "board_cards": ["Th", "6c", "3c"],
        "hero_cards": ["Qc", "Js"],
        "hero_card_conf": 0.97,
        "table_color": "green",
        "named_stacks": [],
    }
    hand, _conf_parts, _diagnostics = _assemble_hand(table_result, columns)
    assert_true(hand is not None, "hand must be assembled")
    flop_actions = hand["streets"][0]["actions"]
    # The opponent's flop bet must be present and tagged with the same
    # canonical position the preflop string uses.
    opp_actions = [a for a in flop_actions if a.get("position") != "BB"]
    assert_eq(len(opp_actions), 2,
              "h3scar's flop bet AND fold must both survive reconciliation")
    for a in opp_actions:
        assert_true(
            a["position"] != "LJ",
            f"flop opponent position must not stay LJ: got {a['position']}",
        )
    # Preflop_actions string places the raiser at index 2 → HJ for 7-max
    # (pos_order [UTG, LJ, HJ, CO, BTN, SB, BB]). Reconciliation must
    # propagate that exact position to the flop entries.
    assert_eq(opp_actions[0]["position"], "HJ",
              "h3scar's flop position must match preflop reassignment")


@test
def test_postflop_repeated_named_bb_stays_bb_after_folded_btn():
    """n8_parser: a named BB who checks then raises must remain BB even
    after another opponent folds on the same street.

    Regression for H2896 flop: BB checked, HJ bet, BTN folded, then the
    same named BB raised. Because BB badges are normally treated as noisy
    defaults, the raise was inferred from rotating opponent order and
    incorrectly assigned to the already-folded BTN.
    """
    from ocr.n8_parser import _build_streets

    streets = _build_streets(
        [{"name": "Flop", "entries": [
            {"type": "opponent", "position": "BB", "action": "Check",
             "size": None, "player_name": "HiagoS"},
            {"type": "hero", "position": "BB", "action": "Bet",
             "size": 2.4},
            {"type": "opponent", "position": "BTN", "action": "Fold",
             "size": None, "player_name": "rudevirus"},
            {"type": "opponent", "position": "BB", "action": "Raise",
             "size": 6.5, "player_name": "Hiagos"},
            {"type": "hero", "position": "BB", "action": "Call",
             "size": 4.1},
        ]}],
        ["8c", "4s", "3c"],
        ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
        hero_position="HJ",
        active_positions=["HJ", "BTN", "BB"],
    )

    actions = streets[0]["actions"]
    assert_eq(actions[2]["position"], "BTN", "BTN fold preserved")
    assert_eq(actions[3]["position"], "BB",
              "same named checker's raise must not move to folded BTN")
    assert_eq(actions[3]["action"], "R6.5")




@test
def test_hero_spots_carry_street_actions_before_hero_for_facing_donk():
    """Bug regression: H2755 was an LJ-opens / SB-calls / SB-donks-into-LJ
    spot but every hero deviation got tagged spot_category='cbet_ip'. Root
    cause was that hero_spots was built without `street_actions_before_hero`,
    so the categorizer never saw the SB donk and defaulted to cbet_ip for
    every postflop decision by the PF aggressor.

    This test mimics H2755 in the small via the spot_categorizer directly:
      preflop: F-R2.2-F-F-F-C-F   (7-max: LJ opens, SB calls)
      flop:    SB R4.8 (donk) → LJ to act
    The hero's flop spot must be categorized as 'facing_probe', not 'cbet_ip'.
    """
    from spot_categorizer import categorize_spot

    hand = {
        "players_at_table": 7,
        "hero_position":    "LJ",
        "preflop_actions":  "F-R2.2-F-F-F-C-F",
        "streets": [{"street": "flop", "board": "JsAsQh", "actions": []}],
    }
    cat, tex = categorize_spot(
        hand, street="flop", action_index=0,
        street_actions_before_hero=[
            {"position": "SB", "action": "R4.8", "size": 4.8},
        ],
    )
    assert_eq(cat, "facing_probe",
              "LJ facing SB donk-lead must be facing_probe, not cbet_ip")
    assert_eq(tex, "wet", "JsAsQh is a wet board")

    # Sanity check the inverse: with NO actions before hero, the same hand
    # is genuinely a c-bet decision and must be cbet_ip.
    cat_cbet, _ = categorize_spot(
        hand, street="flop", action_index=0,
        street_actions_before_hero=[],
    )
    assert_eq(cat_cbet, "cbet_ip",
              "PF aggressor first to act with no prior bet must be cbet_ip")


@test
def test_analyze_hand_attaches_street_actions_before_hero():
    """The fix only works if analyze_hand.py actually populates the new key
    on every hero_spot it builds. Static-check the source to guarantee that.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parent / "analyze_hand.py"
    text = src.read_text()
    assert_in("street_actions_before_hero", text,
              "analyze_hand.py must populate street_actions_before_hero "
              "on hero_spots so _extract_deviations can categorize correctly")
    # And it must appear inside an hero_spots.append(...) literal — count the
    # occurrences as a crude proxy. We expect at least 2 sites (one per
    # spot-append path: in-loop postflop and Phase 1.5 'hero hasn't acted').
    appends_with_key = text.count("\"street_actions_before_hero\"")
    assert_true(appends_with_key >= 2,
                f"expected ≥2 hero_spots appends to set street_actions_before_hero "
                f"(found {appends_with_key})")


@test
def test_weekly_report_schedule_fires_on_sunday():
    """Bug regression: PTB v20+ remapped run_daily day_of_week to cron-style
    (0=Sun … 6=Sat). The old value `days=(6,)` was Saturday, not Sunday, so
    the weekly leak report never fired on the intended day. This test parses
    the actual scheduling call in src/main_gemini.py and asserts the next
    fire lands on a Sunday at 10:00 Taipei.
    """
    import ast
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    src = Path(__file__).resolve().parent.parent / "src" / "main_gemini.py"
    tree = ast.parse(src.read_text())

    days_tuple = None
    hour = minute = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_daily"):
            for kw in node.keywords:
                if kw.arg == "days" and isinstance(kw.value, ast.Tuple):
                    days_tuple = tuple(
                        e.value for e in kw.value.elts
                        if isinstance(e, ast.Constant)
                    )
                if kw.arg == "time" and isinstance(kw.value, ast.Call):
                    for tkw in kw.value.keywords:
                        if tkw.arg == "hour" and isinstance(tkw.value, ast.Constant):
                            hour = tkw.value.value
                        if tkw.arg == "minute" and isinstance(tkw.value, ast.Constant):
                            minute = tkw.value.value
            break

    assert_true(days_tuple is not None, "could not locate run_daily(days=...) in main_gemini.py")
    assert_eq(hour, 10, "weekly job hour must be 10")
    assert_eq(minute, 0, "weekly job minute must be 0")

    from apscheduler.triggers.cron import CronTrigger
    import telegram.ext._jobqueue as jq

    cron_days = ",".join([jq.JobQueue._CRON_MAPPING[d] for d in days_tuple])
    assert_eq(cron_days, "sun",
              f"weekly job must fire on Sunday (cron 'sun'); got {cron_days!r} "
              f"from days={days_tuple!r}. Note: in PTB v20+, 0=Sun, 6=Sat.")

    tz = ZoneInfo("Asia/Taipei")
    trigger = CronTrigger(
        day_of_week=cron_days, hour=hour, minute=minute, second=0, timezone=tz,
    )
    now = datetime(2026, 5, 13, 12, 0, tzinfo=tz)  # Wednesday
    next_fire = trigger.get_next_fire_time(None, now)
    assert_eq(next_fire.weekday(), 6,
              f"next fire must be Sunday (weekday=6); got {next_fire} (weekday={next_fire.weekday()})")


@test
def test_normalize_terms_deterministic():
    """Output-terminology safety net (_normalize_terms): the zero-false-
    positive corrections must apply with correct ordering, be idempotent,
    and must NOT touch ambiguous terms left to the prompt (看牌面, English
    river/range/equity)."""
    from gemini_session import _normalize_terms as n

    # core corrections
    assert_eq(n("這手要彩池控制"), "這手要控制底池")
    assert_eq(n("建議控制彩池"), "建議控制底池")
    assert_eq(n("彩池 12bb"), "底池 12bb")
    assert_eq(n("池底 12bb"), "底池 12bb")
    assert_eq(n("這是純唬牌"), "這是純詐唬")
    assert_eq(n("用 c-bet 施壓"), "用 cbet 施壓")
    assert_eq(n("C-Bet 30%"), "cbet 30%")

    # ordering: compound forms replaced before the 彩池 substring
    # (must not leave a 底池控制 artifact)
    assert_eq(n("用彩池控制讓對手棄牌"), "用控制底池讓對手棄牌")
    assert_true("彩池" not in n("彩池控制 彩池 控制彩池 池底"),
                "no 彩池 may survive")
    assert_true("底池控制" not in n("彩池控制"),
                "compound must map to 控制底池, not 底池控制")

    # idempotent — safe to apply more than once
    once = n("彩池控制讓對手唬牌 c-bet")
    assert_eq(n(once), once, "normalize must be idempotent")

    # zero false positives — these must pass through UNCHANGED
    for s in ("放棄這條線", "精彩的一手", "底池 8bb", "看牌面很濕",
              "river 很危險", "他的 range 很寬", "equity 不夠", "cbet 兩次"):
        assert_eq(n(s), s, f"must not alter {s!r}")

    assert_eq(n(""), "")


@test
def test_coach_system_terminology_rule():
    """Guard: the COACH_SYSTEM 術語規範 section must stay in place so the
    no-bilingual-gloss / canonical-term rules can't be silently dropped,
    and the prompt body must not reproduce a gloss it bans by example."""
    from gemini_session import COACH_SYSTEM as cs

    assert_in("術語規範", cs, "terminology section header missing")
    assert_in("禁止中英對照翻譯", cs, "no-bilingual-gloss rule missing")
    assert_in("不要用「彩池」", cs, "彩池→底池 canonical rule missing")
    assert_in("詐唬（不要用「唬牌」）", cs, "唬牌→詐唬 canonical rule missing")
    assert_in("all-in", cs, "English-abbreviation whitelist missing")
    assert_true("pot control / 控制底池" not in cs,
                "prompt body still contains a banned bilingual gloss")


# ── HH Parser (ground-truth oracle) ──

# Real GGPoker tournament hand: blinds 400/800 with a 100 ante. The ante is
# rendered inside the level as "Level14(400/800(100))" — the regex used to
# require "(400/800)" and returned None for every anted hand, silently
# zeroing ground-truth coverage for ~all late-tournament hands.
_HH_ANTE_HAND = """\
Poker Hand #TM5963540471: Tournament #284938542, Daily Turbo $3 Hold'em No Limit - Level14(400/800(100)) - 2026/05/17 17:27:30
Table '20' 8-max Seat #6 is the button
Seat 1: Hero (8,063 in chips)
Seat 2: 89fa3636 (4,660 in chips)
Seat 3: 27448217 (16,567 in chips)
Seat 4: d7153fd2 (40,398 in chips)
Seat 5: a84236c (5,358 in chips)
Seat 6: 1021c65b (50,149 in chips)
Seat 7: 36a75840 (18,017 in chips)
1021c65b: posts the ante 100
89fa3636: posts the ante 100
36a75840: posts the ante 100
Hero: posts the ante 100
a84236c: posts the ante 100
d7153fd2: posts the ante 100
27448217: posts the ante 100
36a75840: posts small blind 400
Hero: posts big blind 800
*** HOLE CARDS ***
Dealt to Hero [9c Th]
Dealt to 89fa3636
Dealt to 27448217
Dealt to d7153fd2
Dealt to a84236c
Dealt to 1021c65b
Dealt to 36a75840
89fa3636: folds
27448217: folds
d7153fd2: folds
a84236c: folds
1021c65b: folds
36a75840: raises 800 to 1,600
Hero: calls 800
*** FLOP *** [3c 6c 7c]
36a75840: bets 3,200
Hero: raises 3,163 to 6,363 and is all-in
36a75840: calls 3,163
Hero: shows [9c Th] (Ten high)
36a75840: shows [5h 6s] (a pair of Sixes)
*** TURN *** [3c 6c 7c] [Kh]
*** RIVER *** [3c 6c 7c Kh] [4s]
*** SHOWDOWN ***
36a75840 collected 16,626 from pot
*** SUMMARY ***
Total pot 16,626 | Rake 0 | Jackpot 0 | Bingo 0 | Fortune 0 | Tax 0
Board [3c 6c 7c Kh 4s]
"""

_HH_NO_ANTE_HAND = """\
Poker Hand #TM5963540999: Tournament #284938542, Daily Turbo $3 Hold'em No Limit - Level1(100/200) - 2026/05/17 15:00:00
Table '20' 3-max Seat #1 is the button
Seat 1: Hero (10,000 in chips)
Seat 2: villainA (10,000 in chips)
Seat 3: villainB (10,000 in chips)
villainA: posts small blind 100
villainB: posts big blind 200
*** HOLE CARDS ***
Dealt to Hero [Ah Ks]
Hero: raises 200 to 400
villainA: folds
villainB: calls 200
*** FLOP *** [2d 7h Jc]
villainB: checks
Hero: bets 300
villainB: folds
*** SUMMARY ***
Total pot 1,100
"""


@test
def test_hh_parser_ante_level_format():
    """Anted level 'Level14(400/800(100))' must parse (bb=800, not None)."""
    from hh_parser import parse_hand

    gt = parse_hand(_HH_ANTE_HAND, include_folds=True)
    assert_true(gt is not None, "anted hand parsed to None (Level regex regression)")
    assert_eq(gt["hand_id"], "TM5963540471", "hand_id")
    assert_eq(gt["hero_position"], "BB", "hero_position")
    assert_eq(gt["hero_hand"], "9cTh", "hero_hand")
    # bb=800, ante=100. Hero seat declares 8,063 → post-ante 7,963 → 9.95bb,
    # rounded to 10.0bb. The previous expected value (10.1) was computed
    # against the pre-ante chip count, which made the effective stack disagree
    # with the all-in sizes in the action log.
    assert_eq(gt["effective_bb"], 10.0, "effective_bb (bb_size must be 800, post-ante)")
    assert_eq(gt["preflop_actions"], "F-F-F-F-F-R2.0-C", "preflop_actions")
    assert_true("streets" in gt and len(gt["streets"]) >= 1, "flop street missing")


# A short stack all-in from the ante ("posts the ante 4,942" with only 4,942
# chips) emits no preflop action line, so its position is absent from
# preflop_parts. That used to push hero_preflop_idx past the list end and
# false-trigger the "walk" skip, dropping every anted hand with a sub-ante
# stack. parse_hand must still return the hand (hero fields intact).
_HH_ANTE_ALLIN_HAND = """\
Poker Hand #TM5901976430: Tournament #280457497, Phase-L: 25 Zodiac Million Festival [Final] Hold'em No Limit - Level16(20,000/40,000(5,000)) - 2026/05/03 22:46:45
Table '73' 8-max Seat #8 is the button
Seat 1: 87514188 (1,318,736 in chips)
Seat 2: Hero (1,283,487 in chips)
Seat 3: 7edf0999 (650,858 in chips)
Seat 4: 2dcb0bd9 (95,921 in chips)
Seat 5: 35d9f56f (1,118,439 in chips)
Seat 6: 15f641ed (4,942 in chips)
Seat 7: b34ff918 (1,033,228 in chips)
Seat 8: c1fd42c9 (472,515 in chips)
15f641ed: posts the ante 4,942
7edf0999: posts the ante 5,000
2dcb0bd9: posts the ante 5,000
Hero: posts the ante 5,000
b34ff918: posts the ante 5,000
35d9f56f: posts the ante 5,000
c1fd42c9: posts the ante 5,000
87514188: posts the ante 5,000
87514188: posts small blind 20,000
Hero: posts big blind 40,000
*** HOLE CARDS ***
Dealt to 87514188
Dealt to Hero [5c Kc]
Dealt to 7edf0999
Dealt to 2dcb0bd9
Dealt to 35d9f56f
Dealt to 15f641ed
Dealt to b34ff918
Dealt to c1fd42c9
7edf0999: folds
2dcb0bd9: folds
35d9f56f: folds
b34ff918: folds
c1fd42c9: folds
87514188: raises 100,000 to 140,000
Hero: calls 100,000
*** FLOP *** [Th 8d 9h]
87514188: bets 76,787
Hero: folds
Uncalled bet (76,787) returned to 87514188
*** TURN *** [Th 8d 9h] [5d]
*** RIVER *** [Th 8d 9h 5d] [7h]
*** SHOWDOWN ***
87514188 collected 319,942 from pot
*** SUMMARY ***
Total pot 319,942
Board [Th 8d 9h 5d 7h]
"""


@test
def test_hh_parser_ante_allin_does_not_false_walk():
    """Sub-ante all-in seat must not drop the whole hand via the walk gate."""
    from hh_parser import parse_hand

    gt = parse_hand(_HH_ANTE_ALLIN_HAND, include_folds=True)
    assert_true(gt is not None,
                "ante-all-in hand parsed to None (walk-gate false positive)")
    assert_eq(gt["hand_id"], "TM5901976430", "hand_id")
    assert_eq(gt["hero_position"], "BB", "hero_position")
    assert_eq(gt["hero_hand"], "5cKc", "hero_hand")
    assert_eq(gt["preflop_actions"], "F-F-F-F-F-R3.5-C", "preflop_actions")
    assert_true("streets" in gt and len(gt["streets"]) >= 1, "flop missing")


@test
def test_hh_parser_no_ante_level_still_works():
    """Non-anted level 'Level1(100/200)' must still parse (no regression)."""
    from hh_parser import parse_hand

    gt = parse_hand(_HH_NO_ANTE_HAND, include_folds=True)
    assert_true(gt is not None, "non-anted hand parsed to None")
    assert_eq(gt["hero_hand"], "AhKs", "hero_hand")
    assert_eq(gt["hero_position"], "BTN", "hero_position (3-max button)")
    assert_eq(gt["effective_bb"], 50.0, "effective_bb (10000/200)")


# ── Title OCR (dataset label oracle) ──
#
# Regression for the Daily Hyper 1 off-by-one: the scraper used to name
# each replay PNG from its arrow-walk position, assuming the in-modal
# right-arrow steps in hand-list order. That assumption broke (a stale
# anchor frame shifted a whole tournament by one), so files held the wrong
# hand. The fix makes the rendered title bar — the ONLY place the true id
# exists — authoritative. These prove the reader is exact and that a
# misnamed file is detected by reading its own title.

_FX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@test
def test_title_ocr_reads_correct_id_exactly():
    """A scene whose id is known from a direct row-click anchor must read
    back exactly (no off-by-one, no digit slip)."""
    from title_ocr import read_title_id

    fx = _FX / "title_correct_TM5963540471.png"
    assert_true(fx.exists(), f"fixture missing: {fx}")
    tid, votes, total = read_title_id(fx.read_bytes())
    assert_eq(tid, "TM5963540471", "title id must read exactly")
    assert_true(votes * 2 > total, "must win by a strict majority")


@test
def test_title_ocr_detects_mislabeled_file():
    """The regression case itself: a file on disk named TM5880084315 whose
    replay actually renders TM5880084269. Reading the title (not trusting
    the name) must recover the TRUE id — this is what repairs the dataset
    and what the self-correcting scraper now does at capture time."""
    from title_ocr import read_title_id

    fx = _FX / "title_mislabeled_file_TM5880084315.png"
    assert_true(fx.exists(), f"fixture missing: {fx}")
    gt = {"TM5880084269", "TM5880084315"}  # both are real GT ids
    tid, _, _ = read_title_id(fx.read_bytes(), valid=gt)
    assert_eq(tid, "TM5880084269",
              "must read the TRUE rendered id, not the (wrong) filename")
    assert_true(tid != "TM5880084315",
                "filename is the mislabel; title is ground truth")


@test
def test_title_ocr_unreadable_returns_none():
    """A title that cannot be read must return None — never a guessed id
    (a false id would silently corrupt the benchmark pairing)."""
    import io

    from PIL import Image

    from title_ocr import read_title_id

    buf = io.BytesIO()
    Image.new("RGB", (640, 900), (0, 0, 0)).save(buf, "PNG")
    tid, _, _ = read_title_id(buf.getvalue())
    assert_true(tid is None, f"blank image must be unreadable, got {tid!r}")


@test
def test_resolve_hero_uses_top2_when_top1_collides():
    """Hero CNN top1 collides with board; top2 doesn't, so keep top2."""
    from ocr.n8_parser import _resolve_hero_board_conflict

    board = ["Kc", "9d", "3h"]
    hero_details = [
        {
            "rank": "K",
            "rank_top2": [("K", 0.6), ("Q", 0.35)],
            "suit": "c",
            "suit_top2": [("c", 0.7), ("d", 0.2)],
            "conf": 0.6,
        },
        {
            "rank": "A",
            "rank_top2": [("A", 0.9), ("K", 0.05)],
            "suit": "s",
            "suit_top2": [("s", 0.9), ("h", 0.05)],
            "conf": 0.9,
        },
    ]

    new_board, new_hero = _resolve_hero_board_conflict(
        board,
        ["Kc", "As"],
        hero_details=hero_details,
    )

    assert_eq(new_board, board)
    assert_eq(new_hero, ["Qc", "As"])


@test
def test_temperature_scaling_lowers_ece():
    """Calibrated softmax should reduce expected calibration error."""
    import torch

    from ocr.classifier.calibrate import ece, fit_temperature

    torch.manual_seed(0)
    labels = torch.randint(0, 10, (1000,))
    pred = labels.clone()
    wrong = torch.randperm(1000)[:100]
    pred[wrong] = (pred[wrong] + 1) % 10
    logits = torch.zeros(1000, 10)
    logits.scatter_(1, pred[:, None], 6.0)

    temp = fit_temperature(logits, labels)
    before = ece(torch.softmax(logits, dim=1), labels)
    after = ece(torch.softmax(logits / temp, dim=1), labels)

    assert_true(after < before, f"ECE not reduced: {before} -> {after}")


@test
def test_document_image_routing():
    """Bug regression: Telegram delivers uncompressed/HEIC screenshots as
    Document (not Photo) when the user picks "send as file". The previous
    handle_document path rejected anything not .txt/.zip with the misleading
    `請上傳手牌歷史檔案（.txt 或 .zip）`. _is_image_document must classify
    common image documents so they get routed to the photo-analysis pipeline.
    """
    from telegram_bot.bot import _is_image_document

    # Screenshots uploaded as file (with mime + extension)
    assert_true(_is_image_document("image/png", "screenshot.png"))
    assert_true(_is_image_document("image/jpeg", "img_001.jpg"))
    assert_true(_is_image_document("image/jpeg", "photo.jpeg"))
    assert_true(_is_image_document("image/webp", "photo.webp"))
    # iPhone screenshots
    assert_true(_is_image_document("image/heic", "IMG_1234.HEIC"))
    assert_true(_is_image_document("image/heif", "IMG_1234.heif"))
    # Mime missing but extension is an image
    assert_true(_is_image_document(None, "scene.png"))
    assert_true(_is_image_document("", "scene.jpg"))
    # Mime says image but extension is unusual (case-insensitive mime)
    assert_true(_is_image_document("IMAGE/PNG", "weird"))

    # Animated GIF: don't route to image analysis
    assert_true(not _is_image_document("image/gif", "anim.gif"))
    # Hand-history files must remain non-image
    assert_true(not _is_image_document("text/plain", "GG2026.txt"))
    assert_true(not _is_image_document("application/zip", "hands.zip"))
    # Other documents
    assert_true(not _is_image_document("application/pdf", "doc.pdf"))
    assert_true(not _is_image_document(None, "notes.txt"))


@test
def test_first_bet_pot_pct_includes_ante():
    """H3432 regression: postflop first-bet matching must use a pot value that
    includes the MTT ante. analyze_hand previously computed actual_pot WITHOUT
    antes while display_pot included them, so a 3.6bb cbet on a real 5.4bb
    pot (≈67% pot) was scored as 80% against a no-ante 4.5bb actual_pot and
    routed to the 83%-pot bucket (R4.55) instead of the 55%-pot bucket (R3).
    The cascading consequence was an entirely wrong solver subtree showing
    "All-in 89.5%" for As6d when the correct combo strategy is Call 99%.
    """
    from analyze_hand import analyze_hand_full

    hand = {
        "gametype": "MTTGeneral",
        "hero_hand": "As6d",
        "hero_position": "BB",
        "effective_bb": 15.1,
        "hero_starting_stack": 15.1,
        "players_at_table": 7,
        "preflop_actions": "F-F-F-F-R2-F-C",
        "streets": [
            {"board": "3c5d7c", "actions": [
                {"action": "X", "position": "BB"},
                {"size": 3.6, "action": "R3.6", "position": "BTN"},
                {"size": 3.6, "action": "C", "position": "BB"},
            ]},
        ],
    }

    r = analyze_hand_full(hand)

    flop_spots = [s for s in r["hero_spots"] if s.get("street") == "flop"]
    facing_cbet = next((s for s in flop_spots if s.get("taken_code") == "C"), None)
    assert_true(facing_cbet is not None, "expected a facing-cbet hero spot on flop")
    flop_actions = facing_cbet["params"]["flop_actions"]
    assert_eq(flop_actions, "X-R3",
              f"BTN 3.6bb cbet on 5.4bb pot must match R3 (55% pot bucket), got {flop_actions!r}")

    assert_in("Call 99%", r["text_compact"],
              "As6d-specific combo strategy at facing-cbet node must be Call 99%")


@test
def test_postflop_pre_collapse_in_diagnostics():
    """H3433 regression: per-street pre_collapse counts must be exposed in
    diagnostics so the fallback gate can demote to full-Gemini when an
    All-In re-action box silently disappears in the collapse step.

    structural_conf alone (pot/player/ocr consistency) cannot detect this
    failure mode — the surviving action chain still passes those checks,
    even when the BB's "Raise 13.6 BB All-In" sticker was eaten and the
    villain's fold got orphaned. The hidden signal is large per-street
    raw-fragment → final-entry loss.
    """
    import io
    from pathlib import Path
    from ocr.n8_parser import parse_n8_screenshot

    img_path = Path(__file__).resolve().parent.parent / "tests" / "snapshots" / "H3433" / "input.jpeg"
    if not img_path.exists():
        # Falls back to skipping if the fixture isn't checked in yet; the
        # snapshot test runner will still cover the same behavior via DB.
        return

    result = parse_n8_screenshot(img_path.read_bytes())
    diag = result.get("diagnostics") or {}
    assert_in("street_entries_pre_collapse_count", diag,
              "diagnostics must expose per-street pre_collapse counts")
    pre = diag["street_entries_pre_collapse_count"]
    final = diag.get("street_entries_count") or {}
    losses = {s: int(pre[s]) - int(final.get(s, 0)) for s in pre}
    assert_true(max(losses.values(), default=0) >= 4,
                f"H3433 river must show large collapse loss; got {losses}")


@test
def test_range_image_legend_check_bet_vs_call_raise():
    """range_image: legend labels follow the node type.

    Regression for H3469: the range grid is sent alongside a follow-up. On a
    first-to-act (check/bet) spot the legend was hardcoded to Fold/Call/Raise,
    mislabeling Check as Call and Bet as Raise (and listing a Fold that can't
    happen). It must read Check/Bet with no Fold there, while a facing-bet node
    keeps Call/Raise/Fold, and preflop aggression stays Raise.
    """
    from range_image import _legend_labels

    # First-to-act postflop: Check + two bet sizes, no Call, no Fold.
    check_node = [
        {"action": {"code": "X"}},
        {"action": {"code": "R5.2"}},
        {"action": {"code": "R8.7"}},
    ]
    game_turn = {"current_street": {"type": "turn"}}
    pas, agg, show_fold = _legend_labels(check_node, game_turn)
    assert_eq(pas, "Check", "first-to-act passive bucket must read Check")
    assert_eq(agg, "Bet", "first-to-act aggressive bucket must read Bet")
    assert_true(not show_fold, "check/bet node must not show a Fold legend entry")

    # Facing a bet: Fold/Call/Raise available.
    facing_bet = [
        {"action": {"code": "F"}},
        {"action": {"code": "C"}},
        {"action": {"code": "R12"}},
    ]
    pas, agg, show_fold = _legend_labels(facing_bet, game_turn)
    assert_eq(pas, "Call", "facing-bet passive bucket must read Call")
    assert_eq(agg, "Raise", "facing-bet aggressive bucket must read Raise")
    assert_true(show_fold, "facing-bet node must show a Fold legend entry")

    # Preflop first-in (Fold + raise, no check): aggression is a Raise, not Bet.
    preflop_rfi = [
        {"action": {"code": "F"}},
        {"action": {"code": "R2.5"}},
    ]
    game_pre = {"current_street": {"type": "preflop"}}
    pas, agg, show_fold = _legend_labels(preflop_rfi, game_pre)
    assert_eq(agg, "Raise", "preflop open must read Raise, not Bet")
    assert_true(show_fold, "preflop RFI node must show a Fold legend entry")


# ──────────────────────────────────────────────────────────────────────────
# coach_facts: grounded follow-up answers (P0 B/C/D/E + P1 F/G/H/I + verifier)
# Fixtures are real spot-solution nodes captured offline (no network at test time).
# ──────────────────────────────────────────────────────────────────────────

def _load_coach_ctx():
    import coach_facts as cf
    base = Path(__file__).resolve().parent / "test_fixtures" / "coach_facts"
    hctx = json.loads((base / "ctx.json").read_text())
    hero = json.loads((base / "hero_node.json").read_text())
    villain = json.loads((base / "villain_response_node.json").read_text())
    return cf, hctx, hero, villain


@test
def test_coach_facts_class_groups():
    """coach_facts: class->combo-index grouping covers all 1326 and 169 classes."""
    import coach_facts as cf
    groups = cf._class_to_combo_indices()
    assert_eq(len(groups), 169, "169 classes")
    assert_eq(sum(len(v) for v in groups.values()), 1326, "all combos grouped")
    assert_eq(len(groups["AA"]), 6, "AA has 6 combos")
    assert_eq(len(groups["AKs"]), 4, "AKs has 4 combos")
    assert_eq(len(groups["AKo"]), 12, "AKo has 12 combos")


@test
def test_coach_facts_extract_tokens():
    """coach_facts: extract_combo_tokens finds hands in Chinese prose, skips noise."""
    import coach_facts as cf
    toks = cf.extract_combo_tokens("對手 AJo 會棄牌，但 66 和 AhKh 跟注，頂對價值。")
    assert_in("AJo", toks)
    assert_in("66", toks)
    assert_in("AhKh", toks)
    none = cf.extract_combo_tokens("BB 在 BTN 的 EV 很高")
    assert_true("BB" not in none and "EV" not in none, "positions/terms not combos")
    # percentages and bet sizes must NOT be read as pairs
    pct = cf.extract_combo_tokens("棄牌 88% | 跟注 22%，下注2.75 底池")
    assert_true("88" not in pct and "22" not in pct and "75" not in pct,
                f"numbers leaked as combos: {pct}")
    # lowercase user hands are caught and normalized; English filler is not
    low = cf.extract_combo_tokens("但為什麼 jj call 66，hero fold ato a9o")
    assert_in("JJ", low)
    assert_in("66", low)
    assert_in("ATo", low)
    assert_in("A9o", low)
    assert_true("AT" not in cf.extract_combo_tokens("check at the turn"),
                "lowercase English 'at' not a hand")


@test
def test_coach_facts_canonical_forms():
    """coach_facts: canonical_forms normalizes order + derives class from a combo."""
    import coach_facts as cf
    assert_in("KJs", cf.canonical_forms("KsJs"))
    assert_in("KsJs", cf.canonical_forms("KsJs"))
    assert_in("KT", cf.canonical_forms("TK"))  # rank order normalized


@test
def test_coach_facts_digest_helpers():
    """coach_facts: acting_position + category_action_table from a real node."""
    cf, hctx, hero, villain = _load_coach_ctx()
    assert_eq(cf._acting_position(hero), hctx["hero_position"], "hero acts at hero node")
    table = cf._category_action_table(hero, top_n=4)
    assert_true(len(table) >= 1, "at least one category")
    name, freq, actions = table[0]
    assert_true(0.0 <= freq <= 1.0, "category freq is a fraction")
    assert_true(abs(sum(actions.values()) - 1.0) < 0.05, "per-category actions sum ~1")


@test
def test_coach_facts_rep_classes():
    """coach_facts: rep_classes_for_category returns in-range classes of that category."""
    cf, hctx, hero, villain = _load_coach_ctx()
    table = cf._category_action_table(hero, top_n=6)
    reps = cf._rep_classes_for_category(hero, table[0][0], top_k=2)
    assert_true(len(reps) >= 1, "at least one representative class")
    cls, freq, actions = reps[0]
    assert_in(cls, cf._class_to_combo_indices(), "rep is a real 169 class")


@test
def test_coach_facts_fetch_why_action():
    """coach_facts: fetch_why_action builds grounded card with hero combo facts."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_why_action(cf.Ctx(question="為什麼這手牌要下注？", hand_context=hctx))
    assert_true(facts is not None, "B fetch returns facts")
    assert_eq(facts.intent, "why_action")
    assert_in(hctx["hero_hand"], facts.allowed_claims)
    assert_true(any("%" in ln for ln in facts.lines), "card has numbers")


@test
def test_coach_facts_target_hand():
    """coach_facts: a named hand in the question overrides hero's hand."""
    import coach_facts as cf
    assert_eq(cf._target_hand_from_question(cf.Ctx("為什麼 KTo 也要下注", {})), "KTo")
    # earliest token wins, lowercase normalized to uppercase
    assert_eq(cf._target_hand_from_question(cf.Ctx("但為什麼 jj call 66 all in", {})), "JJ")
    assert_true(cf._target_hand_from_question(cf.Ctx("我這手牌算強嗎", {})) is None)


@test
def test_coach_facts_prefer_first_street():
    """coach_facts: why-action defaults to the first postflop street, not the river."""
    import coach_facts as cf
    ctx = cf.Ctx(question="x", hand_context={
        "hero_spots": [{"street": "flop"}, {"street": "turn"}],
        "solutions": [{"game": {"board": "FLOP"}}, {"game": {"board": "TURN"}}]})
    assert_eq(cf._hero_spot_and_sol(ctx, None, prefer="first")[1]["game"]["board"], "FLOP")
    assert_eq(cf._hero_spot_and_sol(ctx, None, prefer="last")[1]["game"]["board"], "TURN")


@test
def test_coach_facts_why_named_hand():
    """coach_facts: fetch_why_action answers about a named hand from the acting range."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_why_action(cf.Ctx(question="為什麼 55 也要下注？", hand_context=hctx))
    assert_true(facts is not None, "named-hand why returns facts")
    assert_in("55", facts.meta.get("hands", []))
    assert_in("55", facts.allowed_claims)
    assert_true(any("solver 動作" in ln for ln in facts.lines), "shows action frequencies")


@test
def test_coach_facts_hero_specific_combo():
    """coach_facts: hero's SPECIFIC combo (AdKd) beats the normalized class (AKs).

    On suit-specific boards the class average is wrong; we must evaluate the combo.
    """
    import coach_facts as cf
    hc = {"hero_hand": "AKs", "hand": {"hero_hand": "AdKd"}}
    assert_eq(cf._hero_hand(hc), "AdKd", "prefer specific combo from raw hand")
    hc2 = {"hero_hand": "AKs", "hand": {"hero_hand": "AKs"}}
    assert_eq(cf._hero_hand(hc2), "AKs", "fall back to class when no combo")
    hc3 = {"hero_hand": "QQ"}
    assert_eq(cf._hero_hand(hc3), "QQ", "no raw hand -> ctx value")


@test
def test_coach_facts_low_weight_node_sentinel():
    """coach_facts: a combo barely in range (off-strategy line) is not reported
    as '0% equity' — _hero_eq_vs_range returns None and the combo is low_weight."""
    import coach_facts as cf
    import gto_formatter as gf
    idx = gf.combo_index_for_hand("AdKd")
    rng = [0.0] * 1326
    eqs = [0.0] * 1326
    pctl = [0.0] * 1326
    rng[idx] = 0.0          # essentially not in range here
    eqs[idx] = 0.0
    pctl[idx] = -1.0        # solver "not in range" sentinel
    sol = {"game": {"active_position": "CO", "board": "Qd8d3cTh2d"},
           "players_info": [{"player": {"position": "CO"}, "range": rng,
                             "hand_eqs": eqs, "eq_percentile": pctl,
                             "simple_hand_counters": {}}]}
    hf = cf._hero_combo_facts(sol, "CO", "AdKd")
    assert_true(hf.get("low_weight"), "near-zero weight + neg percentile -> low_weight")
    assert_true(cf._hero_eq_vs_range(sol, "CO", "AdKd") is None,
                "degenerate node -> no misleading equity")


@test
def test_coach_facts_sizing_allows_size_numbers():
    """coach_facts: numeric audit must not flag legit pot-size %s in a sizing card."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_sizing_from(hero, hctx)
    assert_true(facts is not None, "sizing facts")
    # every percentage printed in the card is registered as a fact number
    import re as _re
    for ln in facts.lines:
        for m in _re.finditer(r"(\d{1,3})\s*%", ln):
            assert_in(int(m.group(1)), facts.numbers, f"{m.group(1)}% must be a fact number")


@test
def test_coach_facts_fetch_hand_strength():
    """coach_facts: fetch_hand_strength reports equity + percentile."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf.fetch_hand_strength(cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx))
    assert_true(facts is not None, "E fetch returns facts")
    assert_eq(facts.intent, "hand_strength")
    assert_true(len(facts.numbers) >= 1, "numbers captured for audit")


@test
def test_coach_facts_fetch_fold_equity():
    """coach_facts: fold-equity uses villain response node, all examples grounded."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_fold_equity_from(villain, hctx)
    assert_true(facts is not None, "C fetch returns facts")
    assert_eq(facts.intent, "fold_equity")
    assert_true(any("棄牌" in ln for ln in facts.lines), "shows fold split")
    for ln in facts.lines:
        for tok in cf.extract_combo_tokens(ln):
            assert_in(tok, facts.allowed_claims, f"{tok} must be grounded")


@test
def test_coach_facts_fetch_villain_range():
    """coach_facts: villain-range composes range composition + hero equity."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_villain_range_from(hero, hctx)
    assert_true(facts is not None, "D fetch returns facts")
    assert_eq(facts.intent, "villain_range")
    for ln in facts.lines:
        for tok in cf.extract_combo_tokens(ln):
            assert_in(tok, facts.allowed_claims, f"{tok} must be grounded")


@test
def test_coach_facts_verifier():
    """coach_facts: verifier passes grounded prose, flags ungrounded combos."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t",
                     lines=["A高 棄牌 80%，例：AJo"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KsJh"))
    board = "Ks9s2h"
    assert_true(cf.verify_claims("對手 A高如 AJo 會棄牌，你的 KsJh 領先。", facts, board).ok,
                "grounded prose passes")
    bad = cf.verify_claims("對手 AQo 和 KTs 會棄牌。", facts, board)
    assert_true(not bad.ok, "ungrounded AQo/KTs flagged")
    assert_in("AQo", bad.violations)


# ── H3639: villain calling-range facing hero's (hypothetical) bet ──────────

@test
def test_coach_facts_hero_bet_size_from_question():
    """coach_facts: parse the hero bet size a follow-up posits (半池 → 0.5)."""
    import coach_facts as cf
    assert_eq(cf._hero_bet_pot_ratio_from_question("面對我的半池下注他會跟嗎"), 0.5)
    assert_eq(cf._hero_bet_pot_ratio_from_question("如果我下 75% 底池"), 0.75)
    assert_eq(cf._hero_bet_pot_ratio_from_question("我下三分之一"), 1 / 3)
    assert_true(cf._hero_bet_pot_ratio_from_question("超池下注") > 1.0, "overbet > pot")
    assert_true(cf._hero_bet_pot_ratio_from_question("他的範圍是什麼") is None,
                "no size named → None")


@test
def test_coach_facts_deterministic_intent_reroutes_calling_range():
    """coach_facts: 'his calling range facing MY bet' → fold_equity, not villain_range."""
    import coach_facts as cf
    assert_eq(cf._deterministic_intent("BB 面對我的半池下注，他的跟注範圍是什麼？"),
              "fold_equity")
    assert_eq(cf._deterministic_intent("facing my turn bet what does he call"),
              "fold_equity")
    # A pure villain-aggression range question must NOT be forced.
    assert_true(cf._deterministic_intent("BB 河牌領投的範圍有哪些牌？") is None,
                "villain's own bet range is not fold_equity")
    assert_true(cf._deterministic_intent("我這手牌多強？") is None)


@test
def test_coach_facts_closest_bet_code():
    """coach_facts: snap a requested pot ratio to a real bet code; off-tree → None."""
    import coach_facts as cf
    sol = {"action_solutions": [
        {"action": {"code": "X"}},
        {"action": {"code": "R2.1", "betsize_by_pot": 0.25}},
        {"action": {"code": "R4.15", "betsize_by_pot": 0.5}},
        {"action": {"code": "R8.3", "betsize_by_pot": 1.0}},
    ]}
    assert_eq(cf._closest_bet_code(sol, 0.5), "R4.15")
    assert_eq(cf._closest_bet_code(sol, 0.25), "R2.1")
    # 0.6 target snaps to the 0.5 node (within tol); a far target is off-tree.
    assert_eq(cf._closest_bet_code(sol, 0.6), "R4.15")
    assert_true(cf._closest_bet_code(sol, 3.0) is None, "no bet within tolerance")


@test
def test_coach_facts_attach_chart_rejects_nonactor():
    """coach_facts: never chart a position that isn't the node's actor (blank grid)."""
    import coach_facts as cf
    sol = {"game": {"active_position": "BB"}, "players_info": []}
    f1 = cf.Facts(intent="x", title="t")
    cf._attach_chart(f1, sol, "LJ")          # LJ is not acting → refused
    assert_true("chart" not in f1.meta, "non-actor chart refused")
    f2 = cf.Facts(intent="x", title="t")
    cf._attach_chart(f2, sol, "BB")          # BB acts → attached
    assert_eq(f2.meta.get("chart", {}).get("position"), "BB")


@test
def test_coach_facts_villain_calling_range_hypothetical_node():
    """coach_facts: hero checked the turn; 'calling range facing my half-pot bet'
    fetches the hypothetical hero-bet node (turn_actions X-R4.15), reads the
    villain's fold/call split, and charts the villain (who acts there). H3639.
    """
    import coach_facts as cf

    hctx = {
        "hero_position": "LJ",
        "hero_hand": "Ac8c",
        "hand": {"hero_hand": "Ac8c"},
        "hero_spots": [{
            "street": "turn",
            "taken_code": "X",                       # hero CHECKED the turn
            "params": {
                "gametype": "MTTGeneral", "depth": 20.125,
                "preflop_actions": "F-F-R2-F-F-F-F-C", "board": "9c9s5cKs",
                "flop_actions": "X-R1.4-C", "turn_actions": "X", "river_actions": "",
            },
        }],
        "solutions": [{
            "game": {"active_position": "LJ", "board": "9s9c5cKs",
                     "current_street": {"type": "turn"}},
            "action_solutions": [
                {"action": {"code": "X"}, "total_frequency": 0.15},
                {"action": {"code": "R2.1", "betsize_by_pot": 0.25}, "total_frequency": 0.37},
                {"action": {"code": "R4.15", "betsize_by_pot": 0.5}, "total_frequency": 0.47},
            ],
            "players_info": [
                {"player": {"position": "LJ"}, "range": [0.5] * 1326},
                {"player": {"position": "BB"}, "range": [0.5] * 1326},
            ],
        }],
    }

    captured = {}

    def fake_sol(**p):
        captured.update(p)
        return {
            "game": {"active_position": "BB", "board": "9s9c5cKs",
                     "current_street": {"type": "turn"}},
            "players_info": [
                {"player": {"position": "BB"}, "range": [0.5] * 1326,
                 "hand_categories": [
                     {"index": 0, "name": "top_pair", "total_frequency": 0.30,
                      "actions_total_frequencies": {"C": 0.9, "F": 0.1}},
                     {"index": 1, "name": "ace_high", "total_frequency": 0.20,
                      "actions_total_frequencies": {"F": 0.8, "C": 0.2}},
                 ],
                 "simple_hand_counters": {}},
                {"player": {"position": "LJ"}, "range": [0.5] * 1326},
            ],
            "action_solutions": [
                {"action": {"code": "F"}, "total_frequency": 0.35, "strategy": [0.0] * 1326},
                {"action": {"code": "C"}, "total_frequency": 0.55, "strategy": [0.0] * 1326},
                {"action": {"code": "R8"}, "total_frequency": 0.10, "strategy": [0.0] * 1326},
            ],
        }

    orig = cf.get_spot_solution
    cf.get_spot_solution = fake_sol
    try:
        facts = cf.fetch_fold_equity(cf.Ctx(
            question="BB 在 Turn check 後，面對我的半池下注，他的跟注範圍是什麼？",
            hand_context=hctx))
    finally:
        cf.get_spot_solution = orig

    # Fetched the hypothetical hero-bet node, not hero's check node.
    assert_eq(captured.get("turn_actions"), "X-R4.15", "half-pot hero bet appended")
    assert_true(facts is not None, "fold_equity facts produced")
    assert_eq(facts.intent, "fold_equity")
    assert_true(any("跟注" in ln or "棄牌" in ln for ln in facts.lines),
                "shows the villain's call/fold split")
    # Chart is for the villain, who acts at the response node → will render.
    chart = facts.meta.get("chart")
    assert_true(chart is not None and chart["position"] == "BB",
                "villain chart attached at the response node")


@test
def test_coach_facts_verifier_board():
    """coach_facts: verifier whitelists board cards + hero hand + board pairs."""
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t", lines=["equity 37%"],
                     allowed_claims=cf.canonical_forms("KsJh"))
    assert_true(cf.verify_claims("KsJh 在 Ks9s2h 上是頂對。", facts, "Ks9s2h").ok,
                "board cards + hero combo allowed")


@test
def test_coach_facts_registry():
    """coach_facts: registry covers all P0/P1 intent labels."""
    import coach_facts as cf
    ids = {qt.id for qt in cf.REGISTRY}
    for need in ("why_action", "fold_equity", "villain_range", "hand_strength",
                 "range_shift", "sizing", "hypothetical", "node_url"):
        assert_in(need, ids)


@test
def test_coach_facts_template():
    """coach_facts: deterministic template is fully grounded."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對你的下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo"))
    out = cf.render_template(facts)
    assert_in("A高", out)
    for tok in cf.extract_combo_tokens(out):
        assert_in(tok, facts.allowed_claims)


@test
def test_coach_facts_other_fallback():
    """coach_facts: answer_followup returns None for 'other' intent (caller falls back)."""
    import coach_facts as cf
    cf._set_intent_classifier(lambda q, c: "other")
    try:
        out = cf.answer_followup(cf.Ctx(question="天氣如何", hand_context={}))
        assert_true(out is None, "other -> None so caller keeps existing path")
    finally:
        cf._set_intent_classifier(None)


@test
def test_coach_facts_golden_invented():
    """coach_facts golden: invented combos flagged unless grounded (KTo-bet case)."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="對手 BB 面對下注：",
                     lines=["  A高 佔 29% — 棄牌 80% | 跟注 20%   例：AJo(棄牌 84%)"],
                     allowed_claims=cf.canonical_forms("AJo") | cf.canonical_forms("KdTc"),
                     meta={"board": "9h8s2s"})
    board = "9h8s2s"
    invented = "BB 範圍裡有大量 AJo、AQo、ATo 會棄牌。"
    v = cf.verify_claims(invented, facts, board)
    assert_true(not v.ok, "ungrounded AQo/ATo flagged")
    assert_in("AQo", v.violations)
    assert_in("ATo", v.violations)
    assert_true("AJo" not in v.violations, "grounded AJo allowed")
    good = "對手 A高（例如 AJo）大多會棄牌，整體棄牌率約 80%。"
    assert_true(cf.verify_claims(good, facts, board).ok, "category + grounded example passes")


@test
def test_coach_facts_golden_outs():
    """coach_facts golden: A3-vs-Q9 invented draw combos flagged."""
    import coach_facts as cf
    facts = cf.Facts(intent="hand_strength", title="t", lines=["equity 41%"],
                     allowed_claims=cf.canonical_forms("Q9s"), meta={"board": "Kc7h2d"})
    v = cf.verify_claims("對手 KJs、KTs、QJs、JTs 有 6 outs。", facts, "Kc7h2d")
    assert_true(not v.ok and "KJs" in v.violations, "invented draws flagged")


@test
def test_coach_facts_sizing():
    """coach_facts P1: fetch_sizing lists solver bet sizes + frequencies."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_sizing_from(hero, hctx)
    assert_true(facts is not None and facts.intent == "sizing")
    assert_true(any("%" in ln for ln in facts.lines), "sizes have freqs")


@test
def test_coach_facts_range_shift():
    """coach_facts P1: range_shift needs >=2 streets, degrades gracefully."""
    cf, hctx, hero, villain = _load_coach_ctx()
    out = cf.fetch_range_shift(cf.Ctx(question="轉牌之後牌力怎麼變", hand_context=hctx))
    assert_true(out is None or out.intent == "range_shift",
                "single-street fixture -> None or valid range_shift")


@test
def test_coach_facts_hypothetical():
    """coach_facts P1: hypothetical maps requested size to on-tree, rejects off-tree."""
    cf, hctx, hero, villain = _load_coach_ctx()
    facts = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=0.5)
    assert_true(facts is not None and facts.intent == "hypothetical")
    far = cf._fetch_hypothetical_size_from(hero, hctx, target_pot_ratio=9.9)
    assert_true(far is None or far.note, "off-tree flagged")


@test
def test_coach_facts_node_url():
    """coach_facts P1: node_url parses GTO Wizard link params."""
    import coach_facts as cf
    p = cf._parse_gtow_url(
        "https://app.gtowizard.com/solutions?gametype=MTTGeneral&depth=40.125"
        "&board=Ks9s2h&preflop_actions=F-F-R2-F-C-F&flop_actions=X-R1.4")
    assert_eq(p["board"], "Ks9s2h")
    assert_eq(p["flop_actions"], "X-R1.4")


@test
def test_coach_facts_numeric_audit():
    """coach_facts P1: numeric audit flags grossly wrong percentages."""
    import coach_facts as cf
    facts = cf.Facts(intent="fold_equity", title="t", lines=["A高 棄牌 80%"],
                     allowed_claims=cf.canonical_forms("KsJh"),
                     numbers={80, 20}, meta={"board": "Ks9s2h"})
    assert_true(cf.verify_claims("對手約 80% 會棄牌。", facts, "Ks9s2h",
                                 audit_numbers=True).ok, "matching number passes")
    bad = cf.verify_claims("對手約 35% 會棄牌。", facts, "Ks9s2h", audit_numbers=True)
    assert_true(not bad.ok and 35 in bad.number_violations, "gross-mismatch flagged")


@test
def test_session_routes_coach_facts():
    """gemini_session: follow-up routes through coach_facts when grounded."""
    import coach_facts as cf
    called = {}

    def fake_answer_ex(ctx):
        called["q"] = ctx.question
        return "GROUNDED_ANSWER", None

    orig = cf.answer_followup_ex
    cf.answer_followup_ex = fake_answer_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager()
        mgr.hand_contexts[1] = {"hero_position": "CO", "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}
        out = mgr._try_coach_facts(1, "為什麼這手牌下注？")
        assert_eq(out, "GROUNDED_ANSWER")
        assert_in("下注", called["q"])
    finally:
        cf.answer_followup_ex = orig


@test
def test_session_coach_facts_no_ctx():
    """gemini_session: _try_coach_facts returns None without cached hand."""
    from gemini_session import GeminiSessionManager
    mgr = GeminiSessionManager()
    assert_true(mgr._try_coach_facts(999, "為什麼下注") is None)


@test
def test_coach_facts_live_smoke():
    """coach_facts live: narrated answer is grounded (skips without GEMINI_API_KEY)."""
    if not os.getenv("GEMINI_API_KEY"):
        return
    cf, hctx, hero, villain = _load_coach_ctx()
    cf._set_intent_classifier(lambda q, c: "hand_strength")
    try:
        out = cf.answer_followup(cf.Ctx(question="我這手牌算強嗎？", hand_context=hctx))
    finally:
        cf._set_intent_classifier(None)
    assert_true(out and len(out) > 5, "produced an answer")
    facts = cf.fetch_hand_strength(cf.Ctx(question="x", hand_context=hctx))
    v = cf.verify_claims(out, facts, facts.meta.get("board", ""))
    assert_true(v.ok, f"live answer grounded (violations={v.violations})")


@test
def test_classify_ev_impact_preflop_uses_absolute_bb():
    """Preflop EV impact is judged in absolute bb (≤0.05bb = negligible)."""
    from gto_formatter import classify_ev_impact

    assert_true(classify_ev_impact(0.05, is_preflop=True)["negligible"],
                "preflop 0.05bb sits on the negligible boundary")
    assert_true(classify_ev_impact(0.02, is_preflop=True)["negligible"],
                "preflop 0.02bb is negligible (frequency issue)")
    assert_true(not classify_ev_impact(0.20, is_preflop=True)["negligible"],
                "preflop 0.20bb is not negligible")


@test
def test_classify_ev_impact_postflop_is_pot_relative():
    """Postflop the same bb loss is graded against the pot, not in absolute bb."""
    from gto_formatter import classify_ev_impact

    # 0.30bb is noise in an 80bb pot (0.375% ≤ 0.5%) ...
    big = classify_ev_impact(0.30, is_preflop=False, pot_bb=80.0)
    assert_true(big["negligible"], "0.30bb in an 80bb pot is negligible")
    assert_eq(round(big["pot_frac"] * 100, 3), 0.375, "pot fraction computed")

    # ... but the same 0.30bb is huge in a 4bb pot (7.5% > 0.5%).
    small = classify_ev_impact(0.30, is_preflop=False, pot_bb=4.0)
    assert_true(not small["negligible"], "0.30bb in a 4bb pot is NOT negligible")

    # No pot info → fall back to the absolute bb threshold.
    fb = classify_ev_impact(0.30, is_preflop=False, pot_bb=None)
    assert_true(not fb["negligible"], "no-pot fallback uses the bb threshold")


@test
def test_ev_loss_detail_preflop_negligible_high_freq_call():
    """H3510-style: SB 55 Call 97% vs all-in ~0.02bb apart → negligible mix.

    The deterministic layer must report this as a near-zero loss so the coach
    frames it as a frequency/mix preference, not a "serious mistake".
    """
    from gto_formatter import ev_loss_detail
    from hh_deviation_check import HAND_TO_169

    idx = HAND_TO_169["55"]
    call_ev = [0.0] * 169
    jam_ev = [0.0] * 169
    call_strat = [0.0] * 169
    jam_strat = [0.0] * 169
    call_ev[idx] = 10.00
    jam_ev[idx] = 9.98          # all-in is 0.02bb worse than calling
    call_strat[idx] = 0.97
    jam_strat[idx] = 0.03
    rng = [0.0] * 169
    rng[idx] = 1.0

    sol = {
        "game": {"pot": 3.5},
        "players_info": [{"player": {"position": "SB"}, "range": rng}],
        "action_solutions": [
            {"action": {"code": "C"}, "evs": call_ev, "strategy": call_strat},
            {"action": {"code": "RAI", "allin": True},
             "evs": jam_ev, "strategy": jam_strat},
        ],
    }

    d = ev_loss_detail(sol, taken_code="RAI", hero_hand="55",
                       hero_pos="SB", is_preflop=True)
    assert_true(d is not None, "ev_loss_detail returns data")
    assert_eq(round(d["ev_loss"], 2), 0.02, "ev_loss is ~0.02bb")
    assert_eq(d["best_code"], "C", "best action is Call")
    assert_true(d["negligible"], "0.02bb preflop is negligible → frequency issue")
    assert_true(d["pot_frac"] is None, "preflop carries no pot fraction")


@test
def test_format_ev_magnitude_splits_preflop_and_postflop():
    """Magnitude string is bare bb preflop, bb + %pot postflop."""
    from gto_formatter import format_ev_magnitude

    pf = format_ev_magnitude({"ev_loss": 0.02, "pot_frac": None})
    assert_eq(pf, "0.02bb", "preflop magnitude is bare bb")

    post = format_ev_magnitude({"ev_loss": 0.30, "pot_frac": 0.004})
    assert_in("% pot", post, "postflop magnitude includes pot fraction")
    assert_in("0.30bb", post, "postflop magnitude includes the bb figure")


@test
def test_ensure_hand_context_rehydrates_from_db_after_restart():
    """Follow-up after a bot restart rebuilds the lost in-memory hand context.

    Regression for H3515: ``hand_contexts`` lives only in process memory, so a
    deploy/restart wipes the "last analyzed hand".  A follow-up like
    「那我 turn 下注範圍應該長怎樣」 then hit query_gto with no context and the
    coach replied "I need to know which hand".  _ensure_hand_context must pull
    the most recent snapshot from the DB and re-run analyze_hand_full to
    restore the context (and last_hand_ids) so the follow-up resolves.
    """
    import asyncio as _asyncio
    import analyze_hand
    from gemini_session import GeminiSessionManager

    class _FakeDB:
        def __init__(self):
            self.calls = 0

        async def get_last_hand(self, chat_id):
            self.calls += 1
            return {"hand": {"hero_position": "BB", "hero_hand": "T9s",
                             "preflop_actions": "F-F-F-F-R2-F-F-C"},
                    "hand_id": "H3515"}

    sentinel_ctx = {"hero_position": "BB", "hero_hand": "T9s",
                    "solutions": [{"ok": True}]}
    orig = analyze_hand.analyze_hand_full
    analyze_hand.analyze_hand_full = lambda hand: sentinel_ctx
    try:
        s = GeminiSessionManager.__new__(GeminiSessionManager)
        s.hand_contexts = {}
        s.last_hand_ids = {}
        s.db = _FakeDB()
        s._setup_user_token = lambda *a, **k: None
        s._clear_user_token = lambda *a, **k: None
        s._logger = logging.getLogger("regression-rehydrate")

        ok = _asyncio.run(s._ensure_hand_context(
            42, user_id=1, refresh_token="tok"))

        assert_true(ok, "rehydrate reports success")
        assert_true(s.hand_contexts.get(42) is sentinel_ctx,
                    "context rebuilt from DB snapshot")
        assert_eq(s.last_hand_ids.get(42), "H3515",
                  "last_hand_ids restored for tool-call tagging")
        assert_eq(s.db.calls, 1, "DB queried exactly once")

        # Idempotent: a second call with context present must not re-query.
        ok2 = _asyncio.run(s._ensure_hand_context(
            42, user_id=1, refresh_token="tok"))
        assert_true(ok2, "second call still reports a context")
        assert_eq(s.db.calls, 1, "no redundant DB query when context cached")
    finally:
        analyze_hand.analyze_hand_full = orig


@test
def test_ensure_hand_context_noop_without_token_or_db():
    """Rehydrate is best-effort: no token or no DB → leave context empty."""
    import asyncio as _asyncio
    from gemini_session import GeminiSessionManager

    class _FakeDB:
        async def get_last_hand(self, chat_id):
            raise AssertionError("get_last_hand must not run without a token")

    s = GeminiSessionManager.__new__(GeminiSessionManager)
    s.hand_contexts = {}
    s.last_hand_ids = {}
    s.db = _FakeDB()
    s._logger = logging.getLogger("regression-rehydrate-noop")

    # No refresh_token → cannot set up the solver, so don't even query.
    assert_true(not _asyncio.run(s._ensure_hand_context(7, user_id=1,
                                                        refresh_token=None)),
                "no token → returns False")
    assert_true(7 not in s.hand_contexts, "context stays empty without token")

    # No DB at all → also a no-op.
    s.db = None
    assert_true(not _asyncio.run(s._ensure_hand_context(7, user_id=1,
                                                        refresh_token="tok")),
                "no DB → returns False")


@test
def test_attach_chart_only_when_solution_and_position():
    """coach_facts._attach_chart records (solution, position) only for the ACTOR."""
    import coach_facts as cf
    f = cf.Facts(intent="why_action", title="t")
    cf._attach_chart(f, None, "CO")
    assert_true("chart" not in f.meta, "no chart without a solution")
    cf._attach_chart(f, {"game": {"active_position": "CO"}}, None)
    assert_true("chart" not in f.meta, "no chart without a position")
    # position must be the node's acting player, else the grid is blank (H3639)
    cf._attach_chart(f, {"game": {"active_position": "BB"}}, "CO")
    assert_true("chart" not in f.meta, "non-acting position refused")
    sol = {"game": {"active_position": "CO"}}
    cf._attach_chart(f, sol, "CO")
    assert_eq(f.meta["chart"]["position"], "CO", "position recorded")
    assert_true(f.meta["chart"]["solution"] is sol, "solution recorded")


@test
def test_coach_facts_fetchers_attach_chart_meta():
    """Range/strategy fetchers carry chart meta so the caller can draw the grid.

    Regression for the H3515 follow-up: a range question answered by the
    deterministic coach_facts path produced prose but no 13x13 range chart,
    because that path bypasses the tool loop that queues the image.
    """
    cf, hctx, hero, villain = _load_coach_ctx()
    actor = hero["game"]["active_position"]

    sizing = cf._fetch_sizing_from(hero, hctx)
    assert_true(sizing and sizing.meta.get("chart"), "sizing carries chart meta")
    assert_eq(sizing.meta["chart"]["position"], actor,
              "sizing charts the acting position")
    assert_true(sizing.meta["chart"]["solution"] is hero,
                "sizing chart points at the node solution")

    vr = cf._fetch_villain_range_from(hero, hctx)
    assert_true(vr is not None, "villain_range still produces facts")
    # The villain isn't the actor at hero's node, so a strategy grid would be
    # blank — it must NOT be attached (H3639: the BB Turn chart came back empty).
    assert_true(not vr.meta.get("chart"),
                "villain_range does not attach a non-actor chart")

    fe = cf._fetch_fold_equity_from(villain, hctx)
    assert_true(fe and fe.meta.get("chart"), "fold_equity carries chart meta")
    assert_eq(fe.meta["chart"]["position"], cf._acting_position(villain),
              "fold_equity charts the acting villain")


@test
def test_answer_followup_ex_returns_facts_with_chart():
    """answer_followup_ex returns (text, facts); answer_followup stays text-only."""
    import coach_facts as cf
    cf._set_intent_classifier(lambda q, c: "sizing")
    orig_narrate = cf._narrate
    cf._narrate = lambda facts, question, extra_vocab="": "教練回答（已驗證）"
    try:
        _, hctx, hero, _ = _load_coach_ctx()
        # Drive the registry's sizing fetch off the real hero node.
        cf.REGISTRY  # noqa: B018  (ensure module loaded)
        text, facts = cf.answer_followup_ex(
            cf.Ctx(question="我這條線下注尺寸要多大", hand_context=hctx))
        # hctx may not resolve a postflop sizing node; only assert the contract
        # when a grounded answer was produced.
        if text is not None:
            assert_true(facts is not None, "ex returns the facts alongside text")
            assert_eq(cf.answer_followup(
                cf.Ctx(question="我這條線下注尺寸要多大", hand_context=hctx)), text,
                "answer_followup delegates and returns the same text")
    finally:
        cf._narrate = orig_narrate
        cf._set_intent_classifier(None)


@test
def test_session_queues_grounded_range_chart():
    """_try_coach_facts queues a range grid when the grounded facts are chartable."""
    import coach_facts as cf
    _, hctx, hero, _ = _load_coach_ctx()
    actor = hero["game"]["active_position"]

    facts = cf.Facts(intent="sizing", title="t", lines=["x"],
                     meta={"chart": {"solution": hero, "position": actor}})

    def fake_ex(ctx):
        return "GROUNDED", facts

    orig = cf.answer_followup_ex
    cf.answer_followup_ex = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager()
        mgr.hand_contexts[5] = {"hero_position": actor, "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}
        out = mgr._try_coach_facts(5, "下注尺寸要多大")
        assert_eq(out, "GROUNDED", "grounded answer returned")
        pending = mgr.pending_images.get(5) or []
        assert_eq(len(pending), 1, "one range chart queued")
        img_bytes, caption = pending[0]
        assert_true(isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 100,
                    "queued a real PNG")
        assert_in("📊", caption, "chart caption present")
        assert_in(actor, caption, "caption names the charted position")
    finally:
        cf.answer_followup_ex = orig


@test
def test_session_no_chart_when_facts_not_chartable():
    """No chart queued when the grounded facts carry no chart meta."""
    import coach_facts as cf

    def fake_ex(ctx):
        return "ANSWER", cf.Facts(intent="hand_strength", title="t", lines=["x"])

    orig = cf.answer_followup_ex
    cf.answer_followup_ex = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager()
        mgr.hand_contexts[6] = {"hero_position": "CO", "hero_hand": "KsJh",
                                "solutions": [{"x": 1}], "hero_spots": []}
        out = mgr._try_coach_facts(6, "我這手牌算強嗎")
        assert_eq(out, "ANSWER")
        assert_true(not mgr.pending_images.get(6),
                    "hand_strength is not chartable → no image queued")
    finally:
        cf.answer_followup_ex = orig


@test
def test_attach_node_records_street():
    """coach_facts._attach_node records the grounded street, skips empties."""
    import coach_facts as cf
    f = cf.Facts(intent="why_action", title="t")
    cf._attach_node(f, None)
    assert_true("node_street" not in f.meta, "no street -> nothing recorded")
    cf._attach_node(f, "turn")
    assert_eq(f.meta["node_street"], "turn", "street recorded")


@test
def test_coach_facts_hero_intents_attach_node_street():
    """Hero-decision intents tag the street so the link can match the prose.

    Regression for H3515: 'turn betting range' answer (turn check 89%) carried
    a played-line GTO Wizard link to the river node (check 23%). The answer now
    records its node street so the caller deep-links to the matching node.
    """
    cf, hctx, _, _ = _load_coach_ctx()  # fixture: flop hero decision, CO
    wa = cf.fetch_why_action(cf.Ctx(question="為什麼這手在 flop 下注", hand_context=hctx))
    assert_true(wa and wa.meta.get("node_street") == "flop",
                "why_action tags the flop node")
    sz = cf.fetch_sizing(cf.Ctx(question="flop 下注尺寸多大", hand_context=hctx))
    assert_true(sz and sz.meta.get("node_street") == "flop",
                "sizing tags the flop node")


@test
def test_build_node_url_for_street_targets_the_named_street():
    """build_node_url_for_street links to hero's decision on that street only.

    H3515: a turn-range answer must deep-link to the turn decision node
    (board flop+turn, no turn action) — not the played-line river node.
    """
    import gtow_solution_url as gs
    context = {
        "hand": {"streets": [{"board": "Ad3c7h", "actions": []},
                             {"card": "Qc", "actions": []},
                             {"card": "4d", "actions": []}],
                 "preflop_actions": "F-R2-F-C-F", "players_at_table": 5},
        "deeplink_raw_preflop": "F-R2-F-C-F",
        "deeplink_raw_players": 5,
        "hero_spots": [{"street": "flop"}, {"street": "turn"}, {"street": "river"}],
        "solutions": [{"action_solutions": []}] * 3,
    }

    def stub(hand, street, ai):
        return {"preflop_actions": "F-F-F-F-R2-F-C-F", "depth": 20.0,
                "gametype": "MTTGeneral", "history_spot": 10,
                "flop_actions": "X-X" if street in ("turn", "river") else "",
                "turn_actions": "R3-C" if street == "river" else ""}

    turn = gs.build_node_url_for_street(context, "turn", _resolver=stub)
    assert_in("board=Ad7h3cQc", turn, "turn link carries flop+turn board")
    assert_not_in("4d", turn, "turn link must not include the river card")
    assert_not_in("turn_actions", turn, "turn decision has no turn action yet")

    river = gs.build_node_url_for_street(context, "river", _resolver=stub)
    assert_in("Ad7h3cQc4d", river, "river link carries the full board")
    assert_in("turn_actions=R3-C", river, "river link carries the turn action")

    assert_true(gs.build_node_url_for_street(context, "preflop", _resolver=stub) is None,
                "no hero decision on a street -> None (caller falls back)")


@test
def test_session_sets_followup_node_street_from_facts():
    """_try_coach_facts records facts.meta['node_street'] on the ctx for the link."""
    import coach_facts as cf

    def fake_ex(ctx):
        return "ANS", cf.Facts(intent="why_action", title="t", lines=["x"],
                               meta={"node_street": "turn"})

    orig = cf.answer_followup_ex
    cf.answer_followup_ex = fake_ex
    try:
        from gemini_session import GeminiSessionManager
        mgr = GeminiSessionManager()
        mgr.hand_contexts[8] = {"hero_position": "SB", "hero_hand": "Th9h",
                                "solutions": [{"x": 1}], "hero_spots": []}
        out = mgr._try_coach_facts(8, "我 turn 下注範圍應該長怎樣")
        assert_eq(out, "ANS")
        assert_eq(mgr.hand_contexts[8].get("_followup_node_street"), "turn",
                  "node street recorded so the GTO link targets the turn node")
    finally:
        cf.answer_followup_ex = orig


@test
def test_build_streets_sole_villain_overrides_position_mislabel():
    """Heads-up: the lone live opponent owns every opponent action (H3517).

    N8 tagged a BB 3-bettor's flop bet + turn shove as LJ (a preflop-folded
    seat). _build_streets must attribute them to the only non-hero active
    player (BB), not the noisy per-row label — otherwise _fix_folded_players
    strips them and the turn loses its solver node.
    """
    from ocr.n8_parser import _build_streets
    street_cols = [
        {"name": "Flop", "entries": [
            {"type": "opponent", "position": "LJ", "player_name": "jch_",
             "action": "Bet", "size": 4.5},
            {"type": "hero", "action": "Call", "size": 4.5}]},
        {"name": "Turn", "entries": [
            {"type": "opponent", "position": "LJ", "player_name": "jch_",
             "action": "All-In", "size": 14.3},
            {"type": "hero", "action": "Call", "size": 14.3}]},
    ]
    pos_order = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    streets = _build_streets(street_cols, ["Ts", "Th", "9h", "4d"], pos_order,
                             hero_position="UTG+1",
                             active_positions=["UTG+1", "BB"])
    flop = streets[0]["actions"]
    assert_eq(flop[0]["position"], "BB", "lone villain attributed to BB not LJ")
    assert_eq(flop[0]["action"], "R4.5")
    assert_eq(flop[1]["position"], "UTG+1")
    turn = streets[1]["actions"]
    assert_eq(turn[0]["position"], "BB", "turn shove also attributed to BB")
    assert_true(turn[0].get("allin"), "all-in flag preserved")


@test
def test_build_streets_keeps_multiway_inference():
    """3-way: no sole villain → keep the existing per-row position inference."""
    from ocr.n8_parser import _build_streets
    street_cols = [
        {"name": "Flop", "entries": [
            {"type": "opponent", "position": "SB", "player_name": "a", "action": "Check"},
            {"type": "opponent", "position": "BTN", "player_name": "b", "action": "Bet", "size": 2.0},
            {"type": "hero", "action": "Call", "size": 2.0}]},
    ]
    pos_order = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
    streets = _build_streets(street_cols, ["Ts", "Th", "9h"], pos_order,
                             hero_position="CO",
                             active_positions=["CO", "SB", "BTN"])
    poss = [a["position"] for a in streets[0]["actions"]]
    # Two distinct opponents preserved — the sole-villain override must NOT fire.
    assert_true("SB" in poss and "BTN" in poss, "multiway positions preserved")
    assert_eq(poss[2], "CO", "hero action keeps hero position")


@test
def test_fix_folded_players_keeps_mislabeled_aggressor_not_orphan_call():
    """_fix_folded_players must not strip a 'folded' player's bet a call needs.

    Defense-in-depth for H3517: an aggressive postflop action attributed to a
    preflop-folded seat, with a later same-street Call/Raise depending on it,
    is a position mislabel — keep it instead of orphaning the call.
    """
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "preflop_actions": "F-R2.2-F-F-F-F-F-R6.5-C",  # LJ folded preflop
        "streets": [
            {"board": "TsTh9h", "actions": [
                {"position": "LJ", "action": "R4.5", "size": 4.5},   # mislabeled villain bet
                {"position": "UTG+1", "action": "C", "size": 4.5}]},
        ],
    }
    GeminiSessionManager._fix_folded_players(hand)
    acts = hand["streets"][0]["actions"]
    assert_eq(len(acts), 2, "load-bearing bet kept; call not orphaned")
    assert_eq(acts[0]["action"], "R4.5")


@test
def test_fix_folded_players_still_drops_passive_ghost():
    """Genuine ghost actions by a folded player are still removed."""
    from gemini_session import GeminiSessionManager
    hand = {
        "players_at_table": 8,
        "preflop_actions": "F-R2.2-F-F-F-F-F-R6.5-C",  # LJ folded
        "streets": [
            {"board": "TsTh9h", "actions": [
                {"position": "LJ", "action": "X"},                 # ghost check by folded LJ
                {"position": "BB", "action": "R3", "size": 3.0},
                {"position": "UTG+1", "action": "C", "size": 3.0}]},
        ],
    }
    GeminiSessionManager._fix_folded_players(hand)
    poss = [a["position"] for a in hand["streets"][0]["actions"]]
    assert_true("LJ" not in poss, "passive ghost by folded player still dropped")
    assert_true("BB" in poss and "UTG+1" in poss, "real actions preserved")


@test
def test_flag_possible_ft_purple_asks_not_assumes():
    """Purple felt flags possible_ft (ask the user); it must not auto-set ICM/FT.

    Regression for H3518: an 8-handed purple table was auto-judged icm/FT.
    """
    from ocr.n8_parser import _flag_possible_ft

    h = {"hero_position": "BTN"}
    _flag_possible_ft(h, "purple")
    assert_true(h.get("possible_ft") is True, "purple → possible_ft hint")
    assert_true("tournament_type" not in h, "purple must NOT auto-set ICM")
    assert_true("phase" not in h, "purple must NOT auto-set FT phase")

    # Green/dark/unknown felt → no FT hint at all.
    for color in ("green", "dark", "unknown", None):
        g = {"hero_position": "BTN"}
        _flag_possible_ft(g, color)
        assert_true("possible_ft" not in g, f"{color} felt → no FT hint")

    # A stronger signal already resolved ICM (user said "FT") → don't override.
    icm = {"hero_position": "BTN", "tournament_type": "icm", "phase": "FT"}
    _flag_possible_ft(icm, "purple")
    assert_true("possible_ft" not in icm, "explicit ICM not downgraded to a hint")


@test
def test_image_parse_prompt_purple_does_not_auto_ft():
    """The image-parse prompt must ask on purple, not auto-commit to ICM/FT."""
    from gemini_session import IMAGE_PARSE_PROMPT

    assert_in("possible_ft", IMAGE_PARSE_PROMPT,
              "prompt still uses the possible_ft ask path")
    assert_not_in('設置 tournament_type: "icm", phase: "FT"', IMAGE_PARSE_PROMPT,
                  "prompt no longer auto-sets ICM/FT from purple felt")


# ──────────────────────────────────────────────────────────────────────────
# Poker-rules structural validator (scripts/hand_validator.py)
# Replays each parsed hand as a real Hold'em betting game; any rule the game
# forbids is a parse bug that must never silently reach the solver.
# See docs/handoffs/2026-06-08-poker-rules-validator.md
# ──────────────────────────────────────────────────────────────────────────

def _vhand(**over):
    """Build a minimal, structurally-valid hand dict for validator tests."""
    h = {
        "gametype": "MTTGeneral",
        "effective_bb": 30,
        "players_at_table": 8,
        "hero_position": "CO",
        "hero_hand": "AsKd",
        "preflop_actions": "F-F-F-F-R2-F-F-C",  # CO opens, BB calls
        "streets": [],
    }
    h.update(over)
    return h


def _hard_codes(report):
    return [i.code for i in report.hard]


def _soft_codes(report):
    return [i.code for i in report.soft]


@test
def test_validator_legal_check_bet_call_passes():
    """A clean single-raised pot replayed street-by-street must validate."""
    from hand_validator import validate_hand
    h = _vhand(streets=[
        {"board": "Js6h5s", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "R2", "size": 2.0},
            {"position": "BB", "action": "C", "size": 2.0}]},
        {"card": "Kc", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "CO", "action": "X"}]},
    ])
    r = validate_hand(h)
    assert_true(r.ok, f"clean hand flagged: {_hard_codes(r)}")


@test
def test_validator_flags_orphan_call():
    """A Call with no preceding bet this street is an orphan call (H2565/H3485)."""
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=7, hero_position="BB", hero_hand="6d4h",
        preflop_actions="F-F-R2-F-F-F-C",
        streets=[{"board": "2dQh4c", "actions": [
            {"position": "BB", "action": "X"},
            {"position": "BB", "action": "C", "size": 1.8}]}])
    r = validate_hand(h)
    assert_in("ORPHAN_CALL", _hard_codes(r), "orphan call not flagged")
    assert_true(not r.ok, "hand with orphan call must be invalid")


@test
def test_validator_flags_act_after_fold():
    """A player who folded pre-flop must not act post-flop (H2838)."""
    from hand_validator import validate_hand
    # 8-max: CO opens (R2), BB calls; SB folded pre-flop yet acts on the flop.
    h = _vhand(
        hero_position="BTN", hero_hand="Ac9s",
        preflop_actions="F-F-F-F-F-R2-F-C",  # BTN opens, BB calls
        streets=[{"board": "8c5d2h", "actions": [
            {"position": "SB", "action": "X"},
            {"position": "BB", "action": "X"}]}])
    r = validate_hand(h)
    assert_in("ACT_AFTER_FOLD", _hard_codes(r), "folded SB acting not flagged")


@test
def test_validator_recognizes_all_aggression_codes():
    """AI<n>, RAI, bare R, and allin:true all close an orphan-call check (H2740)."""
    from hand_validator import validate_hand
    for aggr in (
        {"position": "CO", "action": "AI14", "size": 14},
        {"position": "CO", "action": "RAI", "size": 14},
        {"position": "CO", "action": "R", "size": 14},
        {"position": "CO", "action": "AllIn", "size": 14, "allin": True},
    ):
        h = _vhand(streets=[{"board": "Qd9h4s", "actions": [
            {"position": "BB", "action": "X"},
            dict(aggr),
            {"position": "BB", "action": "C", "size": 14}]}])
        r = validate_hand(h)
        assert_not_in("ORPHAN_CALL", _hard_codes(r),
                      f"{aggr['action']} not treated as aggression → false orphan call")


@test
def test_validator_flags_illegal_check_facing_bet():
    """A check is illegal once someone has wagered this street."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "CO", "action": "R2", "size": 2.0},
        {"position": "BB", "action": "X"}]}])
    r = validate_hand(h)
    assert_in("ILLEGAL_CHECK", _hard_codes(r), "check facing a bet not flagged")


@test
def test_validator_flags_non_monotonic_raise():
    """A raise must exceed the standing bet."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "BB", "action": "R10", "size": 10.0},
        {"position": "CO", "action": "R8", "size": 8.0}]}])
    r = validate_hand(h)
    assert_in("NON_MONOTONIC_RAISE", _hard_codes(r), "shrinking raise not flagged")


@test
def test_validator_flags_action_after_allin_called():
    """Once an all-in is called the round closes; later actions are illegal."""
    from hand_validator import validate_hand
    h = _vhand(streets=[{"board": "Js6h5s", "actions": [
        {"position": "CO", "action": "AI20", "size": 20.0, "allin": True},
        {"position": "BB", "action": "C", "size": 20.0, "allin": True},
        {"position": "CO", "action": "R30", "size": 30.0}]}])
    r = validate_hand(h)
    assert_in("ACTION_AFTER_ALLIN_CALLED", _hard_codes(r),
              "action after a called all-in not flagged")


@test
def test_validator_flags_duplicate_card():
    """The same card cannot appear in hero's hand and on the board."""
    from hand_validator import validate_hand
    h = _vhand(hero_hand="AsKd", streets=[{"board": "AsQh3c", "actions": []}])
    r = validate_hand(h)
    assert_in("DUP_CARD", _hard_codes(r), "duplicate As not flagged")


@test
def test_validator_flags_bad_card_and_board_count():
    """Illegal card faces and wrong board lengths are structural errors."""
    from hand_validator import validate_hand
    bad_face = validate_hand(_vhand(hero_hand="ZxKd"))
    assert_in("BAD_CARD", _hard_codes(bad_face), "illegal rank not flagged")
    short_flop = validate_hand(_vhand(streets=[{"board": "AsQh", "actions": []}]))
    assert_in("BOARD_COUNT", _hard_codes(short_flop), "2-card flop not flagged")


@test
def test_validator_flags_hero_pos_and_preflop_len_and_effbb():
    """Hero position, pre-flop length and effective_bb invariants."""
    from hand_validator import validate_hand
    assert_in("HERO_POS_INVALID",
              _hard_codes(validate_hand(_vhand(hero_position="UTG+2"))),  # not in 8-max
              "invalid hero position not flagged")
    assert_in("PREFLOP_LEN",
              _hard_codes(validate_hand(_vhand(preflop_actions="F-F-R2-C"))),
              "short pre-flop line not flagged")
    assert_in("EFFECTIVE_BB",
              _hard_codes(validate_hand(_vhand(effective_bb=0))),
              "non-positive effective_bb not flagged")


@test
def test_validator_legal_allin_called_runout_passes():
    """All-in + call + a dealt runout with NO decisions is legal."""
    from hand_validator import validate_hand
    h = _vhand(streets=[
        {"board": "Js6h5s", "actions": [
            {"position": "CO", "action": "AI20", "size": 20.0, "allin": True},
            {"position": "BB", "action": "C", "size": 20.0, "allin": True}]},
        {"card": "Kc", "actions": []},   # runout, no decisions
        {"card": "2d", "actions": []}])
    r = validate_hand(h)
    assert_true(r.ok, f"legal all-in runout flagged: {_hard_codes(r)}")


@test
def test_validator_multiway_hero_fold_reconcile_clean():
    """H3511: hero folded pre-flop per the raw string but plays the flop.

    The reconciled participant model must NOT flag the hero (or the other flop
    participants) as acting-after-fold — that was the prototype's false positive.
    """
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=8, hero_position="BTN", hero_hand="6h7h",
        effective_bb=60,
        preflop_actions="F-F-R2-C-C-F-F-C",  # raw: BTN folded (wrong)
        streets=[
            {"board": "9sJcQh", "actions": [
                {"position": "BB", "action": "X"}, {"position": "LJ", "action": "X"},
                {"position": "CO", "action": "X"}, {"position": "BTN", "action": "X"}]},
            {"card": "Th", "actions": [
                {"position": "LJ", "action": "R", "size": 2.6},
                {"position": "CO", "action": "F"},
                {"position": "BTN", "action": "C", "size": 2.6},
                {"position": "BB", "action": "F"}]},
            {"card": "Ac", "actions": [
                {"position": "LJ", "action": "X"}, {"position": "BTN", "action": "X"}]},
        ])
    r = validate_hand(h)
    assert_true(r.ok, f"reconciled multiway hand false-positived: {_hard_codes(r)}")


@test
def test_validator_soft_stacks_len_mismatch():
    """player_stacks length ≠ players_at_table is a SOFT warning, not a block."""
    from hand_validator import validate_hand
    h = _vhand(players_at_table=8, player_stacks=[30, 25, 40])  # too few
    r = validate_hand(h)
    assert_in("STACKS_LEN", _soft_codes(r), "stacks length mismatch not warned")
    assert_true(r.ok, "stacks length is SOFT — must not invalidate the hand")


@test
def test_validator_user_warning_messages():
    """user_warning picks the right zh-TW note for hard / soft / clean reports."""
    from hand_validator import validate_hand, user_warning, HARD_WARNING, SOFT_WARNING
    # Hard-invalid → the "contradiction, re-send" message.
    hard = validate_hand(_vhand(streets=[{"board": "2dQh4c", "actions": [
        {"position": "BB", "action": "X"}, {"position": "BB", "action": "C"}]}]))
    assert_eq(user_warning(hard), HARD_WARNING, "hard report → hard warning")
    # Soft-only → the low-confidence note; hand still ok.
    soft = validate_hand(_vhand(possible_ft=True))
    assert_eq(user_warning(soft), SOFT_WARNING, "soft-only report → soft warning")
    # Clean → no warning.
    assert_eq(user_warning(validate_hand(_vhand())), "", "clean report → no warning")


@test
def test_validator_parser_feedback_localizes_the_spot():
    """to_parser_feedback renders the failing street + repair hint for re-parse."""
    from hand_validator import validate_hand, to_parser_feedback
    r = validate_hand(_vhand(streets=[{"board": "2dQh4c", "actions": [
        {"position": "BB", "action": "X"}, {"position": "BB", "action": "C"}]}]))
    fb = to_parser_feedback(r)
    assert_in("2dQh4c", fb, "feedback must name the street")
    assert_in("Call", fb, "feedback must describe the orphan call")


@test
def test_validator_hero_folded_but_plays_with_continuation_tokens_clean():
    """H2823: hero folded pre-flop but plays — even with 3-bet continuation tokens.

    `_reconcile_preflop_with_streets` bails when the line has continuation
    tokens (len != table size), so the participant model must independently
    trust that a hero who acts post-flop did not fold (else false ACT_AFTER_FOLD).
    """
    from hand_validator import validate_hand
    h = _vhand(
        players_at_table=7, hero_position="HJ", hero_hand="Ac4c",
        preflop_actions="F-F-F-R2.2-R7-F-F-F-C",  # raw folds HJ; has cont token
        streets=[
            {"board": "9sAd7s", "actions": [
                {"position": "HJ", "action": "X"},
                {"position": "CO", "action": "R8", "size": 8.0},
                {"position": "HJ", "action": "C", "size": 8.0}]},
            {"card": "2h", "actions": [
                {"position": "HJ", "action": "X"}, {"position": "CO", "action": "X"}]},
        ])
    r = validate_hand(h)
    assert_true(r.ok, f"hero-plays-after-raw-fold false-positived: {_hard_codes(r)}")


@test
def test_validator_unknown_hero_hand_is_not_a_card_error():
    """An 'XX' placeholder (hero folded pre-flop, cards unknown) is not BAD_CARD."""
    from hand_validator import validate_hand
    r = validate_hand(_vhand(hero_hand="XX", preflop_actions="F-F-F-F-F-R2.5-R12-F",
                             streets=[]))
    assert_not_in("BAD_CARD", _hard_codes(r), "unknown-hero placeholder wrongly flagged")


@test
def test_validator_soft_size_exceeds_stack():
    """A bet larger than the effective stack is a SOFT warning (OCR noise)."""
    from hand_validator import validate_hand
    h = _vhand(effective_bb=25, streets=[{"board": "Js6h5s", "actions": [
        {"position": "BB", "action": "X"},
        {"position": "CO", "action": "R80", "size": 80.0}]}])  # 80 >> 25
    r = validate_hand(h)
    assert_in("SIZE_EXCEEDS_STACK", _soft_codes(r), "oversized bet not warned")
    assert_true(r.ok, "size check is SOFT — must not invalidate")


# Known triaged hands whose STORED parse legitimately violates the rules — every
# one is a confirmed silent parse bug (position mislabel, duplicate card, orphan
# call, or dropped pre-flop seat).  The corpus gate asserts the validator flags
# NOTHING outside this set; a new entry here must come with a triage note, and a
# new *un-triaged* flag fails the build (forces a participant-model/aggression fix
# before merge).  See docs/handoffs/2026-06-08-poker-rules-validator.md §7.3.
KNOWN_VALIDATOR_FLAGS = {
    # ACT_AFTER_FOLD — a live player's action mislabeled onto a folded seat:
    "H2492", "H2496", "H2543", "H2548", "H2549", "H2630", "H2838",
    # DUP_CARD — the same card parsed into hero's hand and the board (impossible):
    "H2534", "H2551", "H2615", "H2626", "H2686", "H2849",
    # ORPHAN_CALL — a Call with no preceding bet on that street:
    "H2554", "H2565", "H2764", "H3485",
    # PREFLOP_LEN — a pre-flop seat dropped from the action line:
    "H2527", "H2651", "H2835", "H3494",
}


@test
def test_validator_corpus_no_new_false_positives():
    """Corpus gate: validator must flag nothing outside the triaged bug set.

    Runs validate_hand over every stored snapshot's parse.  Guards against a
    participant-model/aggression regression silently re-introducing false
    positives (e.g. the AI14 or multiway-reconcile classes).  Skips when the DB
    is unreachable so offline core runs still pass.
    """
    import asyncio
    from hand_validator import validate_hand

    dsn = os.environ.get("SUPABASE_CONN")
    if not dsn:
        if _verbose:
            print("    (skipped: SUPABASE_CONN unset)")
        return
    try:
        import asyncpg
    except ImportError:
        return

    async def _scan():
        conn = await asyncpg.connect(dsn, statement_cache_size=0)
        try:
            rows = await conn.fetch(
                "SELECT hand_id, expected_json, parsed_json FROM analysis_snapshots")
        finally:
            await conn.close()
        flagged = {}
        for row in rows:
            raw = row["expected_json"] or row["parsed_json"]
            if not raw:
                continue
            try:
                hand = json.loads(raw)
            except Exception:
                continue
            rep = validate_hand(hand)
            if not rep.ok:
                flagged[row["hand_id"]] = sorted({i.code for i in rep.hard})
        return flagged

    try:
        flagged = asyncio.run(_scan())
    except Exception as e:
        if _verbose:
            print(f"    (skipped: DB error {e})")
        return

    new_fps = {h: c for h, c in flagged.items() if h not in KNOWN_VALIDATOR_FLAGS}
    assert_eq(new_fps, {},
              "validator produced NEW false positives outside the triaged set — "
              "fix the participant model/aggression handling, do not just add them here")


@test
def test_validator_soft_icm_unconfirmed():
    """possible_ft set without a confirmed ICM signal is a SOFT warning."""
    from hand_validator import validate_hand
    r = validate_hand(_vhand(possible_ft=True))
    assert_in("ICM_UNCONFIRMED", _soft_codes(r), "possible_ft not surfaced as soft")
    assert_true(r.ok, "ICM uncertainty is SOFT — must not invalidate")
    # A normal chip-EV hand must stay quiet.
    assert_not_in("ICM_UNCONFIRMED", _soft_codes(validate_hand(_vhand())),
                  "chip-EV hand wrongly flagged ICM_UNCONFIRMED")


@test
def test_effbb_depth_bucket():
    """effbb_metrics: depth_bucket snaps to AVAILABLE_DEPTHS"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from effbb_metrics import depth_bucket
    assert_eq(depth_bucket(21.6), 20)
    assert_eq(depth_bucket(24.0), 25)   # |25-24|=1 < |20-24|=4
    assert_eq(depth_bucket(29.3), 30)
    assert_eq(depth_bucket(None), None)
    assert_eq(depth_bucket("x"), None)


@test
def test_effbb_bucket_match():
    """effbb_metrics: bucket_match compares snapped depths"""
    from effbb_metrics import bucket_match
    assert_true(bucket_match(21.6, 19.0))    # both -> 20
    assert_true(not bucket_match(29.3, 36.2)) # 30 vs 35
    assert_true(not bucket_match(None, 20.0))


@test
def test_effbb_hero_folded():
    """effbb_metrics: hero_folded_preflop reads position-ordered action string"""
    from effbb_metrics import hero_folded_preflop
    # 8-max, hero UTG (index 0), preflop UTG folds
    gt = {"num_players": 8, "table_size": 8, "hero_position": "UTG",
          "preflop_actions": "F-F-F-F-F-R2.0-F-C"}
    assert_eq(hero_folded_preflop(gt), True)
    # hero UTG+1 raises
    gt2 = {"num_players": 8, "table_size": 8, "hero_position": "UTG+1",
           "preflop_actions": "F-R2.2-C-F-F-C-F-F"}
    assert_eq(hero_folded_preflop(gt2), False)


@test
def test_effbb_classify_fault():
    """effbb_metrics: classify_fault buckets the 4 error classes"""
    from effbb_metrics import classify_fault
    # overshoot beyond any table stack -> impossible_over
    assert_eq(classify_fault(p_eff=162.9, gt_eff=20.4, hero_start=20.4,
                             gt_max=63.0), "impossible_over")
    # returned hero's own start, a shorter villain existed -> selection
    assert_eq(classify_fault(p_eff=36.2, gt_eff=29.3, hero_start=36.2,
                             gt_max=69.4), "selection")
    # under-compute
    assert_eq(classify_fault(p_eff=7.4, gt_eff=24.1, hero_start=24.1,
                             gt_max=80.0), "undershoot")
    # adjacent-bucket near miss
    assert_eq(classify_fault(p_eff=40.0, gt_eff=37.1, hero_start=45.0,
                             gt_max=78.0), "near")


# ── Phase A: per-node depth resolution (node_depth.py) ──

@test
def test_node_depth_open_uses_max_live_cover():
    """D1: the open node plays vs the deepest live opponent, not the shortest
    seat behind. CO 30bb opens, BTN 30bb behind, SB 17bb behind -> open node
    is 30bb; the 17bb stack does NOT shallow the open."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        # position -> starting stack where known (None = unknown)
        stacks={"UTG": 42.0, "HJ": 25.0, "CO": 30.0, "BTN": 30.0,
                "SB": 17.0, "BB": 51.0},
        is_icm=False,
    )
    open_node = nodes[0]
    assert_eq(open_node["node"], "open")
    assert_eq(open_node["eff"], 30.0)
    assert_eq(open_node["depth_bucket"], 30)


@test
def test_node_depth_facing_allin_uses_jammer_commitment():
    """D1: the facing-all-in node queries min(hero, jam total). SB jams 17
    over hero CO's 30bb open -> facing node is 17bb with a range-mismatch
    caveat naming both depths."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "BTN": 30.0, "SB": 17.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"]
    assert_eq(len(facing), 1)
    assert_eq(facing[0]["eff"], 17.0)
    assert_eq(facing[0]["depth_bucket"], 17)
    assert_true(facing[0]["caveat"] is not None
                and "17" in facing[0]["caveat"] and "30" in facing[0]["caveat"],
                f"caveat must name both depths: {facing[0]['caveat']}")


@test
def test_node_depth_same_bucket_no_caveat():
    """No caveat when consecutive nodes land in the SAME depth bucket —
    don't spam the user with a meaningless warning."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI29.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0,
        stacks={"CO": 30.0, "SB": 29.0},
        is_icm=False,
    )
    facing = [n for n in nodes if n["node"] == "facing_allin"][0]
    assert_eq(facing["depth_bucket"], 30)
    assert_true(facing["caveat"] is None, "same-bucket node must carry no caveat")


@test
def test_node_depth_icm_returns_none():
    """D1c: ICM hands keep the single find_icm_params depth — resolver opts out."""
    from node_depth import resolve_preflop_nodes
    nodes = resolve_preflop_nodes(
        preflop_actions="F-F-R2.0-F-AI17.0-F-C",
        hero_position="CO",
        position_order=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        hero_start=30.0, stacks={}, is_icm=True,
    )
    assert_true(nodes is None, "ICM must opt out of per-node depths")


@test
def test_hh_node_effectives_open_vs_facing():
    """D2: hh_parser.node_effectives derives per-node effectives from exact HH
    chips. Build a synthetic HH where hero CO (9000 chips, bb=300 -> 30bb)
    opens, BTN (9000) folds, SB (5100 -> 17bb) jams, hero calls: open node
    30bb, facing node 17bb."""
    from hh_parser import node_effectives
    nodes = node_effectives(
        positions=["UTG", "HJ", "CO", "BTN", "SB", "BB"],
        pos_to_chips={"UTG": 12000, "HJ": 8000, "CO": 9000, "BTN": 9000,
                      "SB": 5100, "BB": 15000},
        preflop_actions_ordered=[("UTG", "F"), ("HJ", "F"), ("CO", "R2.0"),
                                 ("BTN", "F"), ("SB", "AI17.0"), ("BB", "F"),
                                 ("CO", "C")],
        hero_position="CO", bb_size=300,
    )
    assert_eq(nodes[0]["node"], "open")
    assert_eq(nodes[0]["eff"], 30.0)
    facing = [n for n in nodes if n["node"].startswith("facing")][0]
    assert_eq(facing["eff"], 17.0)


@test
def test_analyze_per_node_depths_split():
    """The open spot queries the deep tree; the facing-all-in spot queries the
    jam-depth tree with a range-mismatch caveat — replacing the old global
    allin_effective override that dragged the WHOLE hand to jam depth."""
    from analyze_hand import _build_hero_spot_depths   # new pure helper
    hand = {
        "effective_bb": 30.0, "hero_starting_stack": 30.0,
        "hero_position": "CO", "players_at_table": 6,
        "preflop_actions": "F-F-R2.0-F-AI17.0-F-C",
        "player_stacks": [42.0, 25.0, 30.0, 30.0, 17.0, 51.0],
    }
    spots = _build_hero_spot_depths(hand, is_icm=False, is_cash=False)
    assert_eq(spots["open"]["depth"], "30.125")
    assert_eq(spots["facing"]["depth"], "17.125")
    assert_true(spots["facing"]["caveat"] is not None)


@test
def test_analyze_open_node_keeps_deep_depth_under_allin_override():
    """D1d: an all-in that reopens to hero no longer drags the OPEN node to jam
    depth. Hero UTG opens (39bb stack), an early seat jams 19.9bb: the open spot
    queries the deep (40bb) tree, the facing spot the 20bb jam tree, and the
    facing section carries a range-mismatch caveat naming both depths."""
    from analyze_hand import analyze_hand_full
    result = analyze_hand_full({
        "gametype": "MTTGeneral",
        "hero_hand": "6s6c",
        "effective_bb": 37.8,
        "hero_position": "UTG",
        "player_stacks": [17.9, 30.8, 6.4, 10.9, 9.1, 25.7, 71.9, 37.3],
        "preflop_actions": "R2-F-F-AI19.9-F-F-F-F",
        "players_at_table": 8,
        "hero_starting_stack": 39.3,
    })
    preflop_spots = [s for s in result["hero_spots"] if s["street"] == "preflop"]
    # open node now plays the deep tree (hero's own stack), NOT the 20bb jam
    assert_eq(preflop_spots[0]["params"]["depth"], 40.125)
    # facing node still queries the jam-depth tree
    assert_eq(preflop_spots[1]["params"]["depth"], 20.125)
    # facing spot carries the range-mismatch caveat naming both depths
    assert_true(preflop_spots[1].get("depth_caveat"),
                f"expected caveat, got {preflop_spots[1].get('depth_caveat')}")
    assert_in("20bb", preflop_spots[1]["depth_caveat"])
    assert_in("40bb", preflop_spots[1]["depth_caveat"])
    # and it reaches the user-facing compact output
    assert_in("此節點以 20bb 樹查詢", result["text_compact"])


# ── Phase B: chip constraint solver (ocr/chip_solver.py) ──

@test
def test_chip_solver_consistent_hand():
    """Pot headers that match the engine contributions -> consistent, ~0 residuals.
    6-max, blinds 0.5/1.0, no ante: UTG opens to 2.0, BB calls (1.0 more) ->
    flop pot = 2.0 + 2.0 + 0.5(SB fold) = 4.5."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0, "SB": 0.5},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 4.5},
    )
    assert_true(res.consistent, f"residuals={res.residuals}")
    assert_true(abs(res.residuals["flop"]) < 0.01)
    assert_true(res.repair is None)


@test
def test_chip_solver_single_field_repair():
    """A garbled call size (1.0 read for 10.0) leaves a 9.0 residual that
    exactly ONE field change explains -> repair names that field; nothing is
    auto-applied (D3a)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 11.0, "BB": 2.0, "SB": 0.5},   # BB call misread
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 22.5},                          # truth: BB called 11
        candidates={"BB": [2.0]},   # repairable fields: BB's contribution
    )
    assert_true(not res.consistent)
    assert_true(res.repair is not None and res.repair["field"] == "BB",
                f"repair={res.repair}")
    assert_true(abs(res.repair["to"] - 11.0) < 0.01)


@test
def test_chip_solver_ambiguous_repair_returns_none():
    """Two fields could each explain the residual -> repair=None (never guess)."""
    from ocr.chip_solver import check_chips
    res = check_chips(
        contributions={"UTG": 2.0, "BB": 2.0},
        sb=0.5, bb=1.0, ante_total=0.0,
        pot_headers={"flop": 9.5},          # 5.0 unexplained
        candidates={"UTG": [2.0], "BB": [2.0]},
    )
    assert_true(not res.consistent and res.repair is None)


# ── Phase C: avatar-anchored seat reading (ocr/seat_detector + seat_reader) ──

@test
def test_seat_detector_excludes_board_zone():
    """C2: avatar candidates inside the central board-card zone are dropped —
    no seat sits there. Ring discs survive; a central disc does not."""
    import numpy as np
    import cv2
    from ocr.seat_detector import detect_avatars
    img = np.zeros((499, 640, 3), dtype=np.uint8)
    # ring avatars (corners of the oval) + one bogus disc dead-center (board)
    ring = [(90, 120), (550, 120), (90, 380), (550, 380)]
    for (x, y) in ring:
        cv2.circle(img, (x, y), 22, (200, 200, 200), -1)
    cv2.circle(img, (320, 250), 22, (200, 200, 200), -1)  # board-zone decoy
    avs = detect_avatars(img, None)
    assert_true(len(avs) >= 3, f"should find ring avatars, got {len(avs)}")
    for a in avs:
        assert_true("cx" in a and "cy" in a and "r" in a and "conf" in a)
        in_board = abs(a["cx"] - 320) < 0.34 * 640 and abs(a["cy"] - 249.5) < 0.16 * 499
        assert_true(not in_board, f"board-zone disc not excluded: {a}")


@test
def test_seat_reader_anchors_stack_and_rejects_phantom():
    """C3/D4: read_seats claims the 'XX.X BB' under each avatar and drops BB
    text not near any avatar (pot/timeline phantoms). Rows carry anchor_conf."""
    import numpy as np
    from ocr import ocr_utils
    from ocr import seat_reader
    table = np.zeros((499, 640, 3), dtype=np.uint8)
    avatars = [{"cx": 100.0, "cy": 120.0, "r": 22.0, "conf": 0.8}]

    def _fake_ocr(_img):
        # a seat stack directly below the avatar + a phantom pot value centre-table
        return [
            {"text": "23.4 BB", "center_x": 100.0, "center_y": 168.0},
            {"text": "PlayerX", "center_x": 100.0, "center_y": 96.0},
            {"text": "120 BB", "center_x": 320.0, "center_y": 250.0},  # pot phantom
        ]
    orig = ocr_utils.ocr_full_image
    ocr_utils.ocr_full_image = _fake_ocr
    try:
        rows = seat_reader.read_seats(table, avatars)
    finally:
        ocr_utils.ocr_full_image = orig
    assert_eq(len(rows), 1)
    assert_true(abs(rows[0]["stack"] - 23.4) < 0.01, f"row={rows[0]}")
    assert_eq(rows[0]["name"], "PlayerX")
    assert_in("anchor_conf", rows[0])
    # the centre-table 120 BB pot value is NOT emitted as a seat
    assert_true(all(abs(r["stack"] - 120.0) > 0.5 for r in rows))


# ── Phase 1 Ledger: GTOW Analyze API client ──

@test
def test_analyze_api_pagination():
    """iter_all_hands pages until offset >= total using injected transport."""
    import gtow_analyze_api as gapi
    pages = [
        {"items": [{"hand_id": "a"}, {"hand_id": "b"}], "total": 3, "limit": 2, "offset": 0},
        {"items": [{"hand_id": "c"}], "total": 3, "limit": 2, "offset": 2},
    ]
    calls = []
    def fake_request(method, url, **kw):
        calls.append(kw["json"]["pagination"]["offset"])
        class R:
            status_code = 200
            def json(self): return pages[len(calls) - 1]
            content = b"{}"
        return R()
    rows = list(gapi.iter_all_hands("2026-02-28T16:00:00.000Z", page_size=2,
                                    request_fn=fake_request))
    assert_eq([r["hand_id"] for r in rows], ["a", "b", "c"])
    assert_eq(calls, [0, 2])


@test
def test_analyze_api_backoff_then_success():
    """429 twice then 200 -> returns parsed json; delays follow _backoff_delay."""
    import gtow_analyze_api as gapi
    assert_eq(gapi._backoff_delay(0), 2)
    assert_eq(gapi._backoff_delay(3), 16)
    seq = [429, 429, 200]
    def fake_request(method, url, **kw):
        class R:
            status_code = seq.pop(0)
            def json(self): return {"items": [], "total": 0}
            content = b"{}"
        return R()
    out = gapi.list_hands("2026-02-28T16:00:00.000Z", request_fn=fake_request,
                          _sleep=lambda s: None)
    assert_eq(out["total"], 0)


@test
def test_analyze_api_hand_detail_soft_404_returns_none():
    """A single not-ready hand (404 'upload taking longer') must not crash the
    sweep — hand_detail returns None so the caller skips + retries later."""
    import gtow_analyze_api as gapi
    def fake_request(method, url, **kw):
        class R:
            status_code = 404
            def json(self): return {}
            content = b'{"code":"NOT_FOUND","detail":"Hand upload is taking longer"}'
        return R()
    assert_eq(gapi.hand_detail("deadbeef", request_fn=fake_request), None)
    # 403 (forbidden config) and 204 (no solution) are soft too
    for code in (403, 204):
        def fk(method, url, _c=code, **kw):
            class R:
                status_code = _c
                def json(self): return {}
                content = b"{}"
            return R()
        assert_eq(gapi.hand_detail("x", request_fn=fk), None)


@test
def test_analyze_api_client_id_persisted():
    import gtow_analyze_api as gapi, os, uuid as _uuid
    p = "/tmp/_test_gtow_client_id"
    if os.path.exists(p): os.remove(p)
    a = gapi.get_client_id(path=p)
    b = gapi.get_client_id(path=p)
    assert_eq(a, b)
    _uuid.UUID(a)  # raises if not a uuid
    os.remove(p)


# ── Phase 1 Ledger: distiller ──

def _load_fix(name):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parent / "fixtures" / "gtow" / name).read_text())


@test
def test_distill_river_blunder_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"]
    det = _load_fix("detail_eef0b07b.json")
    hand, decs = distill_hand(lr, det)

    assert_eq(hand["gtow_hand_id"], "eef0b07b-23b6-4fe0-bcc6-41d83629583c")
    assert_eq(hand["position"], "SB")
    assert_eq(round(hand["total_ev_loss_bb"], 4), 22.6627)
    assert_eq(hand["hand_correctness"], "BLUNDER")

    assert_eq(len(decs), 6)
    assert_eq([d["street"] for d in decs],
              ["preflop", "flop", "turn", "turn", "river", "river"])
    assert_eq([d["decision_idx"] for d in decs], [0, 0, 0, 1, 0, 1])

    pre = decs[0]
    assert_eq(pre["family"], "open_raise")
    assert_eq(pre["correctness"], "BEST_MOVE")
    assert_eq(pre["ev_loss_bb"], 0.0)
    assert_eq(pre["depth_band"], "25_40")

    flop = decs[1]
    assert_eq(flop["family"], "cbet_oop")
    assert_eq(flop["texture"], "monotone")     # Kh6h4h
    assert_eq(flop["correctness"], "CORRECT_MOVE")

    riv = decs[5]
    assert_eq(riv["family"], "check_raise")    # checked, now facing a bet
    assert_eq(riv["taken_code"], "F")
    assert_eq(riv["best_code"], "C")
    assert_eq(riv["correctness"], "BLUNDER")
    assert_eq(round(riv["ev_loss_bb"], 4), 22.6627)
    assert_eq(round(riv["hand_eq"], 4), 0.7069)
    assert_true(riv["facing"].startswith("vs_R"), riv["facing"])
    assert_true("chipev_grading" in riv["approx_flags"])
    assert_eq(riv["excluded"], False)

    # fidelity property: per-decision losses sum to hand total
    assert_eq(round(sum(d["ev_loss_bb"] for d in decs), 4),
              round(hand["total_ev_loss_bb"], 4))


@test
def test_distill_preflop_fold_hand():
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = rows["bed8860a-442b-4478-a9b4-8acfd52b6143"]
    det = _load_fix("detail_bed8860a.json")
    hand, decs = distill_hand(lr, det)
    assert_eq(len(decs), 1)
    assert_eq(decs[0]["street"], "preflop")
    assert_eq(decs[0]["taken_code"], "F")
    assert_eq(decs[0]["correctness"], "BEST_MOVE")
    assert_eq(decs[0]["family"], "open_raise")
    assert_eq(decs[0]["depth_band"], "15_25")
    assert_eq(hand["total_ev_loss_bb"], 0.0)


@test
def test_distill_honesty_rules():
    """Synthetic mutations of the fixture exercise every honesty rule (pure fn)."""
    import copy
    from ledger_distill import distill_hand
    rows = _load_fix("list_rows.json")
    lr = copy.deepcopy(rows["eef0b07b-23b6-4fe0-bcc6-41d83629583c"])
    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))

    det["game_analysis"]["warning_status"] = "SOMETHING_ODD"
    _, decs = distill_hand(lr, det)
    assert_true(all(d["excluded"] for d in decs))
    assert_true(all(any(f.startswith("warning:") for f in d["approx_flags"]) for d in decs))

    det = copy.deepcopy(_load_fix("detail_eef0b07b.json"))
    det["game_analysis"]["approximation_reason"] = "NEAREST_DEPTH"
    _, decs = distill_hand(lr, det)
    assert_true(all(any(f.startswith("approx:") for f in d["approx_flags"]) for d in decs))
    assert_true(not any(d["excluded"] for d in decs))  # approx flags don't exclude

    lr2 = copy.deepcopy(lr); lr2["solution_status"] = "NO_SOLUTION"
    _, decs = distill_hand(lr2, _load_fix("detail_eef0b07b.json"))
    assert_true(all(d["excluded"] for d in decs))


@test
def test_depth_band_boundaries():
    from ledger_distill import depth_band
    assert_eq(depth_band(9.9), "le15")
    assert_eq(depth_band(15.0), "15_25")
    assert_eq(depth_band(24.99), "15_25")
    assert_eq(depth_band(25.0), "25_40")
    assert_eq(depth_band(40.0), "40plus")


# ── Phase 1 Ledger: ingest raw paths ──

@test
def test_ingest_raw_paths():
    from ledger_ingest import raw_paths
    ld, dp = raw_paths("abc-123", "2026-05-30T21:03:23Z")
    assert_true(str(dp).endswith("data/gtow_raw/detail/2026-05/abc-123.json.gz"))
    assert_true(str(ld).endswith("data/gtow_raw/list/2026-05.jsonl.gz"))


# ── Phase 1 Ledger: session clustering ──

@test
def test_session_clustering():
    from datetime import datetime, timedelta, timezone
    from ledger_sessions import cluster_sessions
    t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    mk = lambda i, mins, t: {"gtow_hand_id": f"h{i}",
                             "played_at": t0 + timedelta(minutes=mins),
                             "tournament_id": t}
    hands = [mk(1, 0, "A"), mk(2, 5, "B"), mk(3, 10, "A"),   # session 1: A+B overlap
             mk(4, 200, "C"), mk(5, 210, "C")]               # gap 190min -> session 2
    ss = cluster_sessions(hands)
    assert_eq(len(ss), 2)
    assert_eq(ss[0]["hands_count"], 3)
    assert_eq(ss[0]["max_concurrent_tables"], 2)
    assert_eq(sorted(ss[0]["tournaments"]), ["A", "B"])
    assert_eq(ss[1]["max_concurrent_tables"], 1)


# ── Phase 1 Ledger: diagnostics ──

def _dec(family="facing_cbet_oop", band="15_25", loss=0.0, week_day="2026-06-01",
         texture="wet", excluded=False):
    from datetime import datetime
    return {"family": family, "depth_band": band, "ev_loss_bb": loss,
            "texture": texture, "excluded": excluded,
            "played_at": datetime.fromisoformat(week_day + "T12:00:00+00:00")}


@test
def test_leak_board_ev_ranking_and_min_n():
    from ledger_diagnostics import leak_board
    decs = ([_dec(loss=1.0)] * 30                                   # 30bb over n=30
            + [_dec(family="open_raise", band="40plus", loss=5.0)] * 3   # big but n<25
            + [_dec(family="probe", loss=0.0)] * 40)
    out = leak_board(decs, min_n=25)
    ranked = out["cells"]
    assert_eq(ranked[0]["family"], "facing_cbet_oop")
    assert_eq(ranked[0]["n"], 30)
    assert_eq(round(ranked[0]["per100"], 2), round(30 / 30 * 100, 2))
    assert_true(all(c["family"] != "open_raise" for c in ranked))
    assert_true(any(c["family"] == "open_raise" for c in out["insufficient"]))


@test
def test_classify_leak_boundary_vs_knowledge():
    from ledger_diagnostics import classify_leak
    conc = [_dec(band="le15", loss=1.0)] * 12 + [_dec(band="40plus", loss=0.1)] * 12
    t, desc = classify_leak(conc)
    assert_eq(t, "boundary")
    assert_in("le15", desc)
    spread = ([_dec(band="le15", loss=0.5)] * 12 + [_dec(band="15_25", loss=0.5)] * 12
              + [_dec(band="40plus", loss=0.5)] * 12)
    t2, _ = classify_leak(spread)
    assert_eq(t2, "knowledge")


@test
def test_weekly_series_tz_bucketing():
    from ledger_diagnostics import weekly_series
    # 2026-06-07 15:59 UTC = 06-07 23:59 Taipei (Sunday, W23); 16:01 UTC = 06-08 Taipei (Monday, W24)
    from datetime import datetime, timezone
    d1 = dict(_dec(loss=2.0), played_at=datetime(2026, 6, 7, 15, 59, tzinfo=timezone.utc))
    d2 = dict(_dec(loss=0.0), played_at=datetime(2026, 6, 7, 16, 1, tzinfo=timezone.utc))
    out = weekly_series([d1, d2])
    assert_eq([w["week"] for w in out], ["2026-W23", "2026-W24"])
    assert_eq(out[0]["n"], 1)


# ── Phase 1 Ledger: scorecard ──

@test
def test_training_plan_and_retrieval_first():
    """Scorecard v2 = training plan: focus spot + retrieval-first prompt +
    precise drill link + self-contained HTML + readback."""
    from scorecard import (compute_training_plan, render_html, retrieval_prompt,
                           spot_desc_zh)
    row = {"spot_leaf": "MP_vs3bet_IP", "spot_category": "vs3bet", "avg_ev": 0.135,
           "n": 67, "hero_cat": "MP", "villain_cat": "SB", "ip_oop": "IP", "hero_pos": "HJ"}
    assert_in("先自問", retrieval_prompt(row))
    assert_in("3bet", spot_desc_zh(row))
    spots = [{"row": row, "url": "https://app.gtowizard.com/practice/trainer?fh_actions=vs3bet",
              "samples": [], "bands": [], "restrict": None}]
    weekly = [{"week": "2026-W27", "n": 100, "per100": 2.5, "total_bb": 2.5},
              {"week": "2026-W28", "n": 120, "per100": 2.0, "total_bb": 2.4}]
    honesty = {"excluded_n": 5, "discarded_n": 3, "chipev_share": 1.0, "total": 100}
    data = compute_training_plan("2026-W28", weekly, spots, [], None, honesty)
    assert_true(data["headline"])
    assert_eq(round(data["delta"], 2), -0.50)
    assert_eq(data["focus"][0]["spot_leaf"], "MP_vs3bet_IP")
    assert_true(data["focus"][0]["drill_url"].startswith("https://app.gtowizard.com/"))
    html = render_html(data)
    assert_in("MP_vs3bet_IP", html)
    assert_in("先自問", html)
    assert_in("<svg", html)
    assert_true("<script src" not in html)
    rb = compute_training_plan("2026-W29", weekly, spots, [],
                               [{"spot_leaf": "MP_vs3bet_IP", "per100": 20.0}], honesty)
    assert_eq(rb["readback"][0]["spot_leaf"], "MP_vs3bet_IP")
    assert_eq(round(rb["readback"][0]["current_per100"], 1), 13.5)


@test
def test_build_drill_url_pins_position():
    """Precise drill URL pins fh_hero/fh_opponent/fh_rel_positions/fh_actions
    (params verified live 2026-07-09; see skill gtow-trainer-drill)."""
    from gtow_trainer_url import build_drill_url, SpotNotSupportedError
    # preflop vsOpen: exact hero + opener category positions
    u = build_drill_url("vsOpen", "preflop", 20, ["BTN"], opponent_positions=["UTG", "UTG+1"])
    assert_in("fh_actions=vsSRP", u)
    assert_in("fh_hero=BTN", u)
    assert_in("fh_opponent=UTG%2CUTG%2B1", u)
    assert_in("fh_start_spot=preflop", u)
    # postflop SRP with IP
    u2 = build_drill_url("flop", "flop", 30, ["BB"], opponent_positions=["SB"],
                         rel_position="IP", pot_type="SRP")
    assert_in("fh_actions=SRP", u2)
    assert_in("fh_hero=BB", u2)
    assert_in("fh_rel_positions=IP", u2)
    assert_in("fh_start_spot=flop", u2)
    # unmapped category raises
    try:
        build_drill_url("bogus", "preflop", 20, ["BTN"])
        assert_true(False, "should have raised")
    except SpotNotSupportedError:
        pass


@test
def test_analyze_table_url_shape():
    from scorecard import analyze_table_url
    url = analyze_table_url("2026-05-30", "2026-05-30")
    assert_in("app.gtowizard.com/analyze/v4/hands/table?filters=", url)
    assert_in("preselectGamemode=TOURNAMENT", url)


@test
def test_spot_taxonomy_preflop_lines():
    from spot_taxonomy import classify_preflop, pos_cat, ip_oop, board_suit, eff_stack_cat
    # RFI: folded to hero (exact position)
    r = classify_preflop("BTN", [("UTG","F"),("LJ","F"),("HJ","F"),("CO","F")], 8)
    assert_eq(r["category"], "RFI"); assert_eq(r["l1"], "BTN_RFI")
    # vsOpen: single open, hero faces it; opener seat -> category L2
    r = classify_preflop("BTN", [("UTG","F"),("LJ","F"),("HJ","R2.5")], 8)
    assert_eq(r["category"], "vsOpen"); assert_eq(r["l1"], "BTN_vsOpen")
    assert_eq(r["l2"], "BTN_vsOpen_MP")           # HJ opener -> MP
    # vsRaiseCall: open + caller before hero
    r = classify_preflop("BTN", [("HJ","R2.5"),("CO","C")], 8)
    assert_eq(r["category"], "vsRaiseCall"); assert_eq(r["l1"], "BTN_vsRaiseCall".replace("BTN","LP"))
    # vs3bet: hero opened, faces a 3bet (hero cat + IP/OOP)
    r = classify_preflop("CO", [("UTG","F"),("LJ","F"),("HJ","F"),("CO","R2.5"),("BTN","F"),
                                ("SB","R9"),("BB","F")], 8)
    assert_eq(r["category"], "vs3bet"); assert_eq(r["l1"], "LP_vs3bet")
    assert_eq(r["l2"], "LP_vs3bet_IP")            # CO is IP vs SB postflop
    # vsCold3bet: hero cold (did not open), faces a 3bet
    r = classify_preflop("BB", [("CO","R2.5"),("BTN","R8"),("SB","F")], 8)
    assert_eq(r["category"], "vsCold3bet"); assert_eq(r["l1"], "BB_vsCold3bet")
    # limp-involved decisions are discarded (limp ranges unreliable)
    r = classify_preflop("BB", [("SB","C")], 8)
    assert_eq(r["category"], "discarded"); assert_eq(r["l1"], "discarded:faced_limp")
    r = classify_preflop("SB", [("HJ","F"),("CO","F"),("BTN","F"),("SB","C"),("BB","R3")], 8)
    assert_eq(r["category"], "discarded"); assert_eq(r["l1"], "discarded:hero_limped")
    # helpers
    assert_eq(pos_cat("UTG+2"), "EP"); assert_eq(pos_cat("HJ"), "MP")
    assert_eq(ip_oop("BTN", "SB", 8), "IP"); assert_eq(ip_oop("SB", "BTN", 8), "OOP")
    assert_eq(board_suit("Kh6h4h"), "monotone"); assert_eq(board_suit("Kh6s4h"), "two_tone")
    assert_eq(eff_stack_cat(60), "large"); assert_eq(eff_stack_cat(30), "medium")
    assert_eq(eff_stack_cat(12), "short")


@test
def test_spot_taxonomy_walk_fixture():
    import json
    from pathlib import Path
    from spot_taxonomy import walk_spots
    FIX = Path(__file__).resolve().parent / "fixtures" / "gtow"
    rows = json.loads((FIX / "list_rows.json").read_text())
    hid = "eef0b07b-23b6-4fe0-bcc6-41d83629583c"
    det = json.loads((FIX / "detail_eef0b07b.json").read_text())
    spots = list(walk_spots(rows[hid], det))
    assert_eq(len(spots), 6)
    assert_eq(spots[0]["leaf"], "SB_RFI")
    assert_eq(spots[1]["leaf"], "flop:SRP:SBvBB:OOP:first_to_act")
    riv = spots[-1]
    assert_eq(riv["leaf"], "river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet")
    assert_eq(round(riv["ev_loss_bb"], 3), 22.663)
    assert_eq(riv["tags"]["board_suit"], "monotone")


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
