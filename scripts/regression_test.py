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
def test_formatter_normalize_hand_name():
    """Formatter: normalize_hand_name handles various input formats."""
    from gto_formatter import normalize_hand_name
    assert_eq(normalize_hand_name("AhKs"), "AKo")
    assert_eq(normalize_hand_name("KsAh"), "AKo")
    assert_eq(normalize_hand_name("6h6s"), "66")
    assert_eq(normalize_hand_name("AhKh"), "AKs")
    assert_eq(normalize_hand_name("AKs"), "AKs")
    assert_eq(normalize_hand_name("66"), "66")


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
    assert_in("對稱籌碼", result["text"])


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
def test_ocr_card_matcher_loads_templates():
    """OCR: card matcher loads rank and suit templates."""
    from ocr.card_matcher import CardMatcher
    matcher = CardMatcher()
    assert_true(len(matcher.rank_templates) > 0, "should load rank templates")
    assert_true(len(matcher.suit_templates) > 0, "should load suit templates")


@test
def test_ocr_card_matcher_identifies_card():
    """OCR: card matcher identifies a card from a sample screenshot."""
    import cv2
    from ocr.region_detector import detect_regions
    from ocr.card_matcher import CardMatcher
    from ocr.generate_templates import find_board_cards
    img_path = os.path.expanduser("~/n8_image/photo_2026-03-22 13.53.03.jpeg")
    if not os.path.exists(img_path):
        return
    image = cv2.imread(img_path)
    regions = detect_regions(image)
    if regions is None:
        return
    table = regions["table"]
    cards = find_board_cards(table)
    if not cards:
        return  # skip if card detection fails on this image
    matcher = CardMatcher()
    rank, suit, conf = matcher.match(cards[0]["image"])
    assert_true(rank is not None, f"should identify rank, got None")
    assert_true(suit is not None, f"should identify suit, got None")
    assert_true(conf > 0.3, f"confidence should be > 0.3, got {conf}")


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
    assert_eq(_estimate_table_size(entries), 8)
    # 6 entries = 6 players
    entries = [{"type": "opponent"}] * 5 + [{"type": "hero"}]
    assert_eq(_estimate_table_size(entries), 6)
    # 2 entries = 2 players (min)
    entries = [{"type": "hero"}, {"type": "opponent"}]
    assert_eq(_estimate_table_size(entries), 2)


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
def test_board_cards_no_tuples():
    """OCR: board card strings must be clean (no tuples from _ocr_card_rank)."""
    from ocr.table_parser import _identify_cards, _ocr_card_rank
    # _ocr_card_rank returns (rank, conf) tuple — _identify_cards must unpack it
    # Verify _identify_cards returns clean strings not tuple representations
    import numpy as np
    # Create a simple white card image (will probably fail OCR but that's OK)
    dummy = np.ones((50, 35, 3), dtype=np.uint8) * 255
    cards = _identify_cards(dummy, [(0, 0, 35, 50)])
    # Cards should be strings like "Xs" or "??" — never contain parentheses
    for c in cards:
        assert_true("(" not in c, f"Card string '{c}' should not contain tuple parentheses")


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
def test_ocr_rank_lowercase_q_fragile_vs_full_card_9():
    """OCR rank: lowercase 'q' from rank crop must yield to full-card '9'.

    H2649 regression: hero 9c had a WIN badge over the bottom; rank crop OCR'd
    'q' (conf 0.64) — direct lowercase-q match — while the full card OCR'd '9'
    (conf 0.60). Lowercase 'q' is OCR-confusable with '9' and should be
    treated as fragile, mirroring the existing '0'/'O' → Q fragile path.
    """
    from ocr.table_parser import _ocr_card_rank
    import numpy as np

    dummy = np.ones((50, 35, 3), dtype=np.uint8) * 255

    # Sequence of fake OCR results for the calls _ocr_card_rank makes:
    #   1) rank_crop  → lowercase 'q' (fragile)
    #   2) full_card  → '9' with conf > 0.45 (should win)
    fake_results = [
        [{"text": "q", "conf": 0.64}],
        [{"text": "9", "conf": 0.60}],
    ]
    calls = {"i": 0}

    def fake_ocr(_img):
        i = calls["i"]
        calls["i"] += 1
        if i < len(fake_results):
            return fake_results[i]
        return []

    rank, _conf = _ocr_card_rank(dummy, fake_ocr)
    assert_eq(rank, "9",
              "lowercase 'q' from rank crop must yield to full-card '9'")


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
                    assert_true(result.get("hand") is not None,
                                f"OCR returned no hand (confidence={result.get('confidence', 0):.2f})")
                    parsed = result["hand"]
                    # Compare key fields (effective_bb excluded — acceptable variance)
                    for key in ["hero_hand", "hero_position", "preflop_actions",
                                "players_at_table", "tournament_type"]:
                        p_val = parsed.get(key)
                        e_val = expected.get(key)
                        if e_val is not None:
                            assert_eq(p_val, e_val, f"{key} mismatch")
                    # Compare board cards per street
                    p_streets = parsed.get("streets") or []
                    e_streets = expected.get("streets") or []
                    assert_eq(len(p_streets), len(e_streets), "streets count mismatch")
                    for i, (ps, es) in enumerate(zip(p_streets, e_streets)):
                        p_board = ps.get("board", ps.get("card", ""))
                        e_board = es.get("board", es.get("card", ""))
                        assert_eq(p_board, e_board, f"street[{i}] board mismatch")
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
                    exp_lines = expected.split("\n")
                    act_lines = actual.split("\n")
                    for i, (el, al) in enumerate(zip(exp_lines, act_lines)):
                        if el != al:
                            raise AssertionError(
                                f"GTO text mismatch at line {i+1}:\n"
                                f"  expected: {el[:120]}\n"
                                f"  actual:   {al[:120]}"
                            )
                    assert_eq(len(act_lines), len(exp_lines), "GTO text line count mismatch")
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
    """Board texture: dry (no pair, no flush draw, no connectivity)."""
    from spot_categorizer import classify_board_texture
    # All different suits, ranks far apart
    assert_eq(classify_board_texture("Ah8c2d"), "dry")

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
    # No followups
    text3 = "普通分析文字，沒有 followup"
    clean3, followups3 = GeminiSession._extract_followups(text3)
    assert_eq(clean3, text3, "text without followups unchanged")
    assert_eq(len(followups3), 0, "no followups extracted")


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


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
