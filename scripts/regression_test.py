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
    assert_eq(_normalize_action("All-In"), "All-In")


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
    hand, conf_parts = _assemble_hand(table_result, columns)
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
    src = inspect.getsource(table_parser._find_hero_cards)
    assert_in("classify_batch_detailed(crops)", src,
              "_find_hero_cards should classify the raw crops too")
    assert_in("classify_batch_detailed(masked_crops)", src,
              "_find_hero_cards should classify the masked crops too")
    # Sanity: rank pulled from raw, suit from masked.
    assert_in('rank = raw["rank"]', src)
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
    _hand, conf_parts = _assemble_hand(table_result, columns=[])
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
        "card_confidence": 0.30,
        "confidence_parts": {
            "pot_consistency": 1.0,
            "player_tracking": 1.0,
            "ocr_confidence": 0.95,
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
    hand, _ = _assemble_hand(table_result, columns)
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


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
