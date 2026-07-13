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

_FX = REPO_ROOT / "tests" / "fixtures"


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

    img_path = REPO_ROOT / "tests" / "snapshots" / "H3433" / "input.jpeg"
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
