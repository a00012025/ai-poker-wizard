"""Regression tests extracted from the legacy monolithic suite."""

import json
import logging
import os
import sys
from pathlib import Path

from regression_tests.harness import (
    REPO_ROOT,
    SCRIPTS_DIR,
    assert_eq,
    assert_in,
    assert_not_in,
    assert_true,
)

import pytest

pytestmark = pytest.mark.ocr

def test_board_unknown_suits_are_canonicalized_before_api():
    """Text parses must not leak unknown board suits like 5x to GTOW.



    Regression for a 422 on:
      board=5c7c9c5x after user wrote "579r ... turn 5".
    """
    from analyze_hand import _canonicalize_board_streets

    streets, notes = _canonicalize_board_streets([
        {"board": "579r", "actions": []},
        {"card": "5x", "actions": []},
    ])

    assert_eq(streets[0]["board"], "5c7d9h", "579r becomes legal rainbow flop")
    assert_eq(streets[1]["card"], "5s", "bare/unknown paired turn gets legal unused suit")
    full_board = streets[0]["board"] + streets[1]["card"]
    assert_not_in("x", full_board.lower(), "GTOW board params must never contain x")
    assert_in("579r → 5☘️7🔷9♥️", "; ".join(notes))
    assert_in("5x → 5♠️", "; ".join(notes))


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


def test_effbb_multiway_selection():
    """effbb: multiway returns min(hero, shortest active villain) bucket."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_deep_invested_not_nulled():
    """effbb: deep-invested hero keeps a real value (no displayed*5 false-null)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    o = rows["TM5896148353"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


def test_effbb_walkover_seat_attribution():
    """effbb: a fold-through open binds on the shortest seat still to act,
    resolved by position/geometry — not hero's own (deeper) stack."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_uncalled_shove_ceiling():
    """effbb: hero invests preflop then folds to an uncalled villain jam — the
    jam size is the villain's whole stack and caps the effective stack."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # TM5863067496: hero SB R6.7, BB jams 20.4 uncalled, hero folds. GT eff 20.4
    # (the jam size), not hero's 36.7 reconstructed start.
    o = rows["TM5863067496"]; inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


def test_effbb_geometry_villain_attribution():
    """effbb: when villain names are None/garbled, the active villain's stack is
    resolved by position/geometry (not the shortest arbitrary seat)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_hero_uncalled_shove_starting_stack():
    """effbb: when hero jams preflop and everyone folds (uncalled), hero's shove
    size is hero's authoritative starting stack."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    # TM5866594919: hero HJ jams 22.9, all fold. GT 22.9 (the shove size). The
    # displayed+reconstruction path abstained (None) before.
    o = rows["TM5866594919"]; inp = o["inputs"]
    eff, _hs, _c = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


def test_effbb_hero_allin_stack_zero_reconstruction():
    """effbb (Phase 3): when hero's displayed stack reads ~0 (all-in / called a
    villain shove), hero's STARTING stack is what hero permanently committed —
    reconstructed from the engine's decision-local hero contribution, NOT the
    noisy displayed+walk estimate. The two reconstructions must agree on the
    depth bucket to emit; otherwise abstain (single-frame unrecoverable).

    Every hero-stack~0 emit must be CORRECT-OR-ABSTAIN — no confidently-wrong
    value (the Phase-3 contract).
    """
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_hero_stack_zero_no_confident_wrong():
    """effbb (Phase 3): across ALL hero-active, hero-stack~0 emits, the
    correct-or-abstain contract holds at high precision — the hero all-in
    reconstruction converts confidently-wrong emits into abstains rather than
    emitting a misread shove value. Guards the Phase-3 gain (24 wrong->abstain,
    0 regressions vs the prior commit)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_confidence_is_calibrated_monotonic():
    """effbb: attribution-certainty confidence yields a MONOTONIC precision/
    coverage curve — raising the floor trades coverage for precision (the old
    dual-estimator agreement-confidence produced a flat curve)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_abstain_or_correct_on_divergence():
    """effbb: ambiguous/divergent reconstruction abstains or hits the right bucket."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l) for l in open(cache, encoding="utf-8")}
    o = rows["TM5863941844"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    # Either it now lands on the right bucket, or it correctly abstains.
    assert_true(eff is None or depth_bucket(eff) == depth_bucket(o["gt"]["effective_bb"]))


def test_effbb_overcompute_bounded():
    """effbb: over-compute past table max is rejected (bounded or abstain)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr import effbb_engine as eng
    return eng


def _eng_streets_from_cache(hid):
    """Load a cached hand and split its panel into the engine's streets/pot."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _engine_streets
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    o = rows[hid]
    streets, pot = _engine_streets(o["inputs"]["columns"])
    return o, streets, pot


def test_engine_position_orders_match_parser():
    """Engine's POSITION_ORDERS must be identical to the parser's."""
    eng = _engine()
    from ocr.n8_parser import POSITION_ORDERS as PARSER_ORDERS
    assert_eq(eng.POSITION_ORDERS, PARSER_ORDERS,
              "engine POSITION_ORDERS drifted from n8_parser")


def test_engine_infer_blinds():
    """Engine infers SB=0.5/BB=1.0 and a BB-ante from the preflop pot."""
    eng = _engine()
    sb, bb, ante, ok = eng.infer_blinds(1.5, 6)   # no ante
    assert_eq((sb, bb), (0.5, 1.0)); assert_true(ok and ante == 0.0)
    sb, bb, ante, ok = eng.infer_blinds(2.4, 6)   # 0.9 BB-ante
    assert_true(ok and abs(ante - 0.9) < 0.01, f"ante={ante}")
    _sb, _bb, _ante, ok2 = eng.infer_blinds(99.0, 6)  # absurd → not ok
    assert_true(not ok2, "absurd preflop pot should not reconcile")


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


def test_effbb_engine_m1_endtoend_bucket():
    """End-to-end: the M1 ceiling reaches _compute_effective_bb. TM5863067496
    GT 20.4 (bucket 20) — read straight off the panel shove."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
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
def test_engine_hu_postflop_order_bb_first():
    """HU (2-handed) postflop the BB acts FIRST (carry-over fix). Preflop the
    SB/BTN acts first; postflop the order flips to ['BB','SB']."""
    eng = _engine()
    assert_eq(eng._postflop_order(["SB", "BB"], 2), ["BB", "SB"])
    # 3+ handed is unchanged: blinds (SB) lead postflop.
    assert_eq(eng._postflop_order(["LJ", "HJ", "CO", "BTN", "SB", "BB"], 6)[0],
              "SB")


def test_phase2_dead_code_removed():
    """The unused _seat_stack_for_position helper was removed in Phase 2."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    import ocr.n8_parser as P
    assert_true(not hasattr(P, "_seat_stack_for_position"),
                "_seat_stack_for_position should be deleted (dead code)")


def test_phase2_enumerate_layouts_topk():
    """_enumerate_layouts returns the top-K position->seat layouts, each a full
    {position: seat} map for the table, best-first by weak name agreement."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    import ocr.n8_parser as P
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_phase2_consensus_holds_emits_correct_bucket():
    """When all plausible layouts + the engine's relevant seat agree on the
    bucket, the consensus path emits the correct bucket. TM5862907992: a
    2-direction straddle the engine's single live opponent (UTG) resolves to
    bucket 17 (true effective 16.5, not hero's deeper 22.9)."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8") if l.strip()}
    o = rows["TM5862907992"]; inp = o["inputs"]
    eff, _hs, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None, "consensus should emit, not abstain")
    assert_eq(depth_bucket(eff), 17)
    assert_true(conf >= 0.7, f"emitted confidence below floor: {conf}")


def test_phase2_layout_straddle_abstains():
    """When the plausible geometric layouts straddle DIFFERENT depth buckets
    and the engine cannot break the tie, the consensus path ABSTAINS (None) —
    the attribution is genuinely ambiguous. TM5864409682 / TM5866699022 both
    straddle two buckets."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_phase2_consensus_curve_trades_coverage_for_precision():
    """The consensus confidence yields a real precision/coverage frontier on
    hero-active hands: raising the floor must not lower precision and the top
    band is meaningfully cleaner than the whole emitted population."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    import importlib
    prev = os.environ.get("OCR_EFFBB_STRUCTURAL_GATE")
    os.environ["OCR_EFFBB_STRUCTURAL_GATE"] = "1" if gate else "0"
    try:
        import ocr.n8_parser as _P
        importlib.reload(_P)
        cache = os.path.join(str(SCRIPTS_DIR), "..",
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


def test_phase4_features_surfaced_per_hand():
    """The Phase-4 abstain features are captured per hand (no re-OCR) and carry
    the candidate signals the calibration harness fits gates on."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb, _effbb_last_features
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_chip_solver_features_surfaced_per_hand():
    """B2: the chip-conservation features are captured per hand alongside the
    Phase-4 features (no re-OCR, no behavior change). The clean hand
    TM5862908042 reconciles its pot header -> chip_consistent True."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _compute_effective_bb, _effbb_last_features
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_phase4_bucket_boundary_distance():
    """Boundary-distance fragility signal: a value near a bucket-cell edge is
    flagged fragile (small), a value at a bucket centre is not."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    from ocr.n8_parser import _bucket_boundary_distance
    # 27.5 is the 25<->30 cell edge: distance ~0.
    assert_true(_bucket_boundary_distance(27.6) < 0.02,
                "near-edge value not flagged fragile")
    # 30.0 is a bucket centre: comfortably away from its 27.5 / 32.5 edges.
    assert_true(_bucket_boundary_distance(30.0) > 0.05,
                "bucket-centre value wrongly flagged fragile")


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


def test_phase4_gate_keeps_clean_correct_hands():
    """The shipped structural gate does NOT abstain clean, correct emits — the
    abstain is targeted, not a blanket coverage cut."""
    from effbb_metrics import bucket_match
    for hid in ("TM5862908042", "TM5863067726", "TM5863067671"):
        eff, gt = _effbb_run(hid, gate=True)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"gate wrongly abstained/missed clean hand {hid}: {eff} vs {gt}")


def test_phase4_gate_lifts_precision_over_baseline():
    """The shipped structural gate lifts emitted precision over the bare-conf
    baseline across the hero-active cache, at the cost of coverage (abstaining
    is cheap downstream). Calibrated 5-fold-CV operating point: ~74% @ ~61%
    coverage vs the ~71% @ ~78% ungated baseline. 99.5% is NOT reachable on
    single-frame inputs (documented in the Phase-4 plan); this guards the
    precision GAIN the gate actually delivers."""
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR)))
    sys.path.insert(0, os.path.join(str(SCRIPTS_DIR), "ocr"))
    import importlib
    from effbb_metrics import hero_folded_preflop, bucket_match
    cache = os.path.join(str(SCRIPTS_DIR), "..", "data/effbb_cache/cache.jsonl")
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


def test_effbb_allin_legality_guard():
    """An 'All-In' row that does not exceed what a player already committed,
    followed by a fold from that covering player, is a misparsed raise — it
    must not bind. TM5878838751: river hero Bet 9.0 → 'All-In 1.0' → hero
    Fold on a 44.3bb spot; the bogus 1.0 floor must not be emitted."""
    eff, gt = _effbb_run("TM5878838751")
    assert_true(eff is None or eff >= 5.0,
                f"illegal sub-level all-in misread bound the spot: {eff}")


def test_effbb_herozero_slice_emitted():
    """Hero displayed ≈0 (all-in) with no engine confirmation is EMITTED when
    the other gate clauses pass — the old blanket herozero abstain measured
    81% marginally precise (above the emitted average) and was dropped."""
    from effbb_metrics import bucket_match
    for hid in ("TM5963779172", "TM5920068321"):
        eff, gt = _effbb_run(hid)
        assert_true(eff is not None and bucket_match(eff, gt),
                    f"{hid}: herozero hand wrongly abstained/wrong: {eff} vs GT {gt}")


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


def test_visual_allin_owner_reads_hero_from_yellow_band_above():
    """Yellow sticker above the red badge => hero owns the All-In."""
    from ocr.panel_parser import _infer_allin_badge_owner
    bbox = (75, 280, 125, 300)
    region = _region_with_band_above(bbox, _HERO_BAND)
    entry = {"action": "All-In", "size": None, "_bbox": bbox}
    owner, conf, evidence = _infer_allin_badge_owner(entry, region)
    assert_eq(owner, "hero", f"evidence={evidence}")
    assert_true(conf >= 0.55, f"expected high confidence, got {conf}")


def test_visual_allin_owner_reads_opponent_from_white_band_above():
    """White sticker above the red badge => opponent owns the All-In (H3441)."""
    from ocr.panel_parser import _infer_allin_badge_owner
    bbox = (75, 280, 125, 300)
    region = _region_with_band_above(bbox, _OPP_BAND)
    entry = {"action": "All-In", "size": None, "_bbox": bbox}
    owner, conf, evidence = _infer_allin_badge_owner(entry, region)
    assert_eq(owner, "opponent", f"evidence={evidence}")
    assert_true(conf >= 0.55, f"expected high confidence, got {conf}")


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


def test_visual_allin_owner_requires_bbox_metadata():
    """Without _bbox metadata the helper abstains (backward-compatible)."""
    import numpy as np
    from ocr.panel_parser import _infer_allin_badge_owner
    region = np.zeros((360, 174, 3), np.uint8)
    owner, conf, evidence = _infer_allin_badge_owner({"action": "All-In"}, region)
    assert_eq(owner, None)


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


def test_gto_text_compare_exact_match():
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("EV: 7.57bb\nFold: 42.0%", "EV: 7.57bb\nFold: 42.0%")
    assert_true(ok, msg)


def test_gto_text_compare_tolerates_ev_drift():
    """0.01bb EV drift (H2504) is within tolerance => match."""
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("  EV: 7.57bb | Equity: 64.2%",
                               "  EV: 7.56bb | Equity: 64.2%")
    assert_true(ok, msg)


def test_gto_text_compare_tolerates_frequency_drift():
    """0.2pp frequency drift (H2505) is within tolerance => match."""
    from gto_text_compare import gto_text_matches
    ok, msg = gto_text_matches("  Fold: 42.0%（22 combos）",
                               "  Fold: 42.2%（22 combos）")
    assert_true(ok, msg)


def test_gto_text_compare_rejects_large_ev_drift():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("EV: 7.57bb", "EV: 7.70bb")
    assert_true(not ok, "0.13bb EV drift must fail")


def test_gto_text_compare_rejects_large_frequency_drift():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("Fold: 42.0%", "Fold: 43.0%")
    assert_true(not ok, "1.0pp frequency drift must fail")


def test_gto_text_compare_rejects_combos_count_change():
    """Combos counts are part of the structure — compared exactly."""
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("Fold: 42.0%（22 combos）", "Fold: 42.2%（23 combos）")
    assert_true(not ok, "combos count change must fail even within freq tolerance")


def test_gto_text_compare_rejects_structural_change():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("  Fold: 42.0%（22 combos）", "  Call: 42.0%（22 combos）")
    assert_true(not ok, "action label change must fail")


def test_gto_text_compare_rejects_line_count_change():
    from gto_text_compare import gto_text_matches
    ok, _ = gto_text_matches("a\nb", "a\nb\nc")
    assert_true(not ok, "line count change must fail")


def test_snapshot_cli_uses_the_same_hermetic_cache_as_regressions():
    """H2494 regression: snapshot golden updates used the root ``.gto_cache``
    while the regression harness read ``tests/snapshots/.gto_cache``. The two
    caches held different solver responses (57% vs 68%), so a freshly updated
    golden failed deterministically in the full suite."""
    import analyze_hand
    import gto_cache
    import snapshot_test

    original = analyze_hand.analyze_hand_full
    original_dir = gto_cache._CACHE_DIR
    try:
        analyze_hand.analyze_hand_full = lambda _hand: {
            "cache_dir": str(gto_cache._CACHE_DIR)
        }
        result = snapshot_test._analyze_snapshot_hand({})
        expected = REPO_ROOT / "tests" / "snapshots" / ".gto_cache"
        assert_eq(result["cache_dir"], str(expected))
        assert_eq(gto_cache._CACHE_DIR, original_dir,
                  "snapshot cache override must be restored")
    finally:
        analyze_hand.analyze_hand_full = original
        gto_cache._CACHE_DIR = original_dir
        gto_cache._mem.clear()


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
