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


def test_hh_parser_fold_excluded():
    """HH Parser: hero fold excluded by default."""
    from hh_parser import parse_hand
    result = parse_hand(_SAMPLE_HH_FOLD, include_folds=False)
    assert_true(result is None, "fold hand should be excluded")


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


def test_card_split_no_hand_leakage():
    """A hand_id appearing in train must not appear in val or test."""
    from ocr.classifier.split import build_split
    from pathlib import Path
    gt_path = REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
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


def test_card_split_tournament_balanced():
    """Every tournament with >=10 hands appears in all three splits."""
    import json
    from collections import Counter
    from pathlib import Path
    from ocr.classifier.split import build_split
    gt_path = REPO_ROOT / "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
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

def test_169_hand_index_count():
    """169 Index: generates exactly 169 unique hand names."""
    from hh_deviation_check import HANDS_169, HAND_TO_169
    assert_eq(len(HANDS_169), 169)
    assert_eq(len(HAND_TO_169), 169)


def test_169_hand_index_ascii_sorted():
    """169 Index: hand names are sorted by ASCII comparison."""
    from hh_deviation_check import HANDS_169
    assert_eq(HANDS_169, sorted(HANDS_169))


def test_169_hand_index_premiums():
    """169 Index: premium hands map to correct indices."""
    from hh_deviation_check import HAND_TO_169
    # Verify key hands exist
    for h in ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "22"]:
        assert_true(h in HAND_TO_169, f"{h} should be in index")
    # AA should come before KK in ASCII (A < K)
    assert_true(HAND_TO_169["AA"] < HAND_TO_169["KK"],
                "AA index should be less than KK (A < K in ASCII)")


def test_169_hand_index_offsuit_before_suited():
    """169 Index: offsuit comes before suited for same ranks (o < s in ASCII)."""
    from hh_deviation_check import HAND_TO_169
    assert_true(HAND_TO_169["AKo"] < HAND_TO_169["AKs"],
                "AKo should come before AKs")
    assert_true(HAND_TO_169["KQo"] < HAND_TO_169["KQs"])


# ── Preflop 8-max Conversion Tests ──

def test_convert_preflop_8max_6p():
    """8max convert: 6-player prepends 2 folds."""
    from hh_deviation_check import _convert_preflop_to_8max
    result = _convert_preflop_to_8max("R2-F-F-F-F-C", 6)
    assert_eq(result, "F-F-R2-F-F-F-F-C")


def test_convert_preflop_8max_8p():
    """8max convert: 8-player unchanged."""
    from hh_deviation_check import _convert_preflop_to_8max
    result = _convert_preflop_to_8max("F-R2-F-F-F-F-F-C", 8)
    assert_eq(result, "F-R2-F-F-F-F-F-C")


# ── Deviation Report Format Tests ──

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


def test_hh_check_hand_second_decision_queries_after_intervening_fold():
    """BTN's squeeze response must be queried after CO folds, not at CO's node."""
    import hh_deviation_check as hdc

    calls = []
    originals = {
        "get_spot_solution": hdc.get_spot_solution,
        "_normalize_preflop_action": hdc._normalize_preflop_action,
        "_get_preflop_hand_freqs": hdc._get_preflop_hand_freqs,
        "_get_hand_ev": hdc._get_hand_ev,
        "_get_action_evs_preflop": hdc._get_action_evs_preflop,
    }

    def fake_solution(**kwargs):
        prefix = kwargs["preflop_actions"]
        calls.append(prefix)
        return {"node": prefix, "action_solutions": []}

    hdc.get_spot_solution = fake_solution
    hdc._normalize_preflop_action = lambda code, *_args, **_kwargs: code
    hdc._get_preflop_hand_freqs = lambda sol, *_args: (
        {"C": 1.0} if sol["node"] == "F-F-F-F-R2-C-R7-F-F" else None)
    hdc._get_hand_ev = lambda *_args, **_kwargs: 0.0
    hdc._get_action_evs_preflop = lambda *_args, **_kwargs: {"F": 0.0, "C": -1.0}
    try:
        devs = hdc.check_hand({
            "hand_id": "LIVE-HAND-2",
            "hero_position": "BTN",
            "hero_hand": "7s8s",
            "effective_bb": 35,
            "num_players": 8,
            "table_size": 8,
            "preflop_actions": "F-F-F-F-R2-C-R7-F-F-C",
        }, emit_ungraded=True)
    finally:
        for name, value in originals.items():
            setattr(hdc, name, value)

    assert_in("F-F-F-F-R2-C-R7-F-F", calls)
    assert_eq(devs[1]["spot"], "facing 3bet/4bet")
    assert_eq(devs[1]["hero_action"], "C")
    assert_eq(
        devs[1]["ev_loss"], 0.0,
        "a pure solver call is equilibrium-approved despite inconsistent raw EVs",
    )


def test_hh_check_hand_second_decision_detects_earlier_seat_fourbet():
    """UTG's continuation 4-bet must create LJ's second decision node."""
    import hh_deviation_check as hdc

    calls = []
    originals = {
        "get_spot_solution": hdc.get_spot_solution,
        "_normalize_preflop_action": hdc._normalize_preflop_action,
        "_get_preflop_hand_freqs": hdc._get_preflop_hand_freqs,
        "_get_hand_ev": hdc._get_hand_ev,
        "_get_action_evs_preflop": hdc._get_action_evs_preflop,
    }

    exact_prefix = "R2-F-R5-F-F-F-F-C-R15"
    hu_prefix = "R2-F-R5-F-F-F-F-F-R15"

    def fake_solution(**kwargs):
        prefix = kwargs["preflop_actions"]
        calls.append(prefix)
        if prefix == exact_prefix:
            return None
        return {"node": prefix, "action_solutions": []}

    hdc.get_spot_solution = fake_solution
    hdc._normalize_preflop_action = lambda code, *_args, **_kwargs: code
    hdc._get_preflop_hand_freqs = lambda sol, *_args: (
        {"F": 1.0} if sol["node"] == hu_prefix else {"R5": 1.0})
    hdc._get_hand_ev = lambda *_args, **_kwargs: 0.0
    hdc._get_action_evs_preflop = lambda *_args, **_kwargs: {"F": 0.0, "R5": 0.0}
    try:
        devs = hdc.check_hand({
            "hero_position": "LJ",
            "hero_hand": "JJ",
            "effective_bb": 28,
            "num_players": 8,
            "preflop_actions": "R2-F-R5-F-F-F-F-C-R15-F",
        }, emit_ungraded=True)
    finally:
        for name, value in originals.items():
            setattr(hdc, name, value)

    assert_in(exact_prefix, calls)
    assert_in(hu_prefix, calls)
    assert_eq(devs[1]["spot"], "facing 3bet/4bet")
    assert_eq(devs[1]["hero_action"], "F")
    assert_eq(devs[1]["approximation"], "cold_callers_folded_hu")


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


def test_find_closest_action_by_explicit_pot_fraction():
    """An explicit live ``25%``/``50%`` matches GTOW's nearest
    ``betsize_by_pot`` branch without requiring a reconstructed BB pot."""
    from gto_api import find_closest_action_by_pot_fraction

    available = [
        {"action": {"code": "X", "betsize": "0",
                    "betsize_by_pot": None, "allin": False}},
        {"action": {"code": "R1.2", "betsize": "1.2",
                    "betsize_by_pot": "0.125", "allin": False}},
        {"action": {"code": "R3.2", "betsize": "3.2",
                    "betsize_by_pot": "0.33", "allin": False}},
        {"action": {"code": "R4.8", "betsize": "4.8",
                    "betsize_by_pot": "0.50", "allin": False}},
        {"action": {"code": "RAI", "betsize": "30",
                    "betsize_by_pot": "3.1", "allin": True}},
    ]
    assert_eq(find_closest_action_by_pot_fraction(available, 0.25), "R3.2")
    assert_eq(find_closest_action_by_pot_fraction(available, 0.50), "R4.8")


def test_unsized_preflop_raise_uses_only_unambiguous_solver_branch():
    """A bare preflop ``R`` can advance only when GTOW offers exactly one
    non-all-in raise; multiple raise sizes remain unresolved."""
    import analyze_hand as ah
    import hh_deviation_check as hdc

    one_raise = [
        {"action": {"code": "C", "betsize": "2", "allin": False}},
        {"action": {"code": "R10", "betsize": "10", "allin": False}},
        {"action": {"code": "RAI", "betsize": "100", "allin": True}},
    ]
    two_raises = one_raise + [
        {"action": {"code": "R12", "betsize": "12", "allin": False}}]
    old_ah = ah.get_next_actions
    old_hdc = hdc.get_next_actions
    try:
        def one_by_node(**kwargs):
            available = (
                [{"action": {"code": "R2", "betsize": "2",
                             "allin": False}}]
                if kwargs.get("preflop_actions") == "F-F-F-F"
                else one_raise
            )
            return {"next_actions": {"available_actions": available}}

        ah.get_next_actions = one_by_node
        hdc.get_next_actions = ah.get_next_actions
        assert_eq(ah._normalize_preflop_actions(
            "F-F-F-F-R2-C-R", "MTTGeneral", 100.125),
            "F-F-F-F-R2-C-R10")
        assert_eq(hdc._normalize_preflop_action(
            "R", "MTTGeneral", 100.125, "F-F-F-F-R2-C"), "R10")

        def two_by_node(**kwargs):
            available = (
                [{"action": {"code": "R2", "betsize": "2",
                             "allin": False}}]
                if kwargs.get("preflop_actions") == "F-F-F-F"
                else two_raises
            )
            return {"next_actions": {"available_actions": available}}

        ah.get_next_actions = two_by_node
        hdc.get_next_actions = ah.get_next_actions
        assert_eq(ah._normalize_preflop_actions(
            "F-F-F-F-R2-C-R", "MTTGeneral", 100.125),
            "F-F-F-F-R2-C-R")
        assert_eq(hdc._normalize_preflop_action(
            "R", "MTTGeneral", 100.125, "F-F-F-F-R2-C"), "R")
    finally:
        ah.get_next_actions = old_ah
        hdc.get_next_actions = old_hdc


def test_hh_check_hand_advances_explicit_pot_fraction_on_solver_line():
    """Live grading must advance a villain's 25%-pot action through the
    nearest GTOW branch even when the action JSON has no absolute BB size."""
    import hh_deviation_check as hdc

    solution_calls = []
    originals = {
        "get_spot_solution": hdc.get_spot_solution,
        "get_next_actions": hdc.get_next_actions,
        "_normalize_preflop_action": hdc._normalize_preflop_action,
    }
    available = [
        {"action": {"code": "X", "betsize": "0",
                    "betsize_by_pot": None, "allin": False}},
        {"action": {"code": "R1.2", "betsize": "1.2",
                    "betsize_by_pot": "0.125", "allin": False}},
        {"action": {"code": "R3.2", "betsize": "3.2",
                    "betsize_by_pot": "0.33", "allin": False}},
        {"action": {"code": "R4.8", "betsize": "4.8",
                    "betsize_by_pot": "0.50", "allin": False}},
    ]

    def fake_solution(**kwargs):
        solution_calls.append(dict(kwargs))
        return None

    hdc.get_spot_solution = fake_solution
    hdc.get_next_actions = lambda **_kwargs: {
        "next_actions": {"available_actions": available}}
    hdc._normalize_preflop_action = lambda code, *_args, **_kwargs: code
    try:
        hdc.check_hand({
            "hand_id": "LIVE-PCT",
            "hero_position": "BB",
            "hero_hand": "AJo",
            "effective_bb": 30,
            "num_players": 8,
            "table_size": 8,
            "preflop_actions": "F-F-F-F-F-R2-F-C",
            "streets": [{"street": "flop", "board": "Kc7d2h", "actions": [
                {"position": "BB", "action": "X"},
                {"position": "BTN", "action": "R",
                 "pot_fraction": 0.25},
                {"position": "BB", "action": "F"},
            ]}],
        }, emit_ungraded=True)
    finally:
        for name, value in originals.items():
            setattr(hdc, name, value)

    assert_true(any(
        call.get("flop_actions") == "X-R3.2"
        for call in solution_calls
    ), solution_calls)


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
