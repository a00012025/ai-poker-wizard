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
def test_hand_eval_ac7c_on_7h5c4c_is_nut_flush_draw():
    """Exact real-hand regression used by grounded coaching."""
    from hand_eval import evaluate

    result = evaluate("Ac7c", "7h5c4c")
    assert_eq(result["made_hand"], "top_pair")
    assert_in("nut_flush_draw", result["draws"])


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
    assert_in("A♥️8♥️", text_specific,
              "Specific combo query should show A♥️8♥️ in output")
    assert_in("A8s", text_specific,
              "Specific combo query should reference parent hand A8s")
    # Compare with aggregated: should be different format
    text_agg = format_hand_detail(sol, "A8s", "CO")
    assert_in("Range 頻率", text_agg,
              "Aggregated query should show Range 頻率 header")


@test
def test_format_hand_detail_omits_class_fallback_for_zero_reach_exact_combo():
    """An absent exact suit is labelled unavailable without class advice."""
    from gto_formatter import format_hand_detail

    zeroes = [0.0] * 1326
    sol = {
        "game": {"board": "9c7h4c2sKh"},
        "players_info": [{
            "player": {"position": "SB"},
            "range": zeroes,
            "simple_hand_counters": {
                "32s": {
                    "total_combos_available": 3.0,
                    "total_combos": 0.1,
                    "total_frequency": 0.041,
                    "hand_ev": 0.0,
                    "hand_eq": 0.306,
                    "actions_total_frequencies": {"F": 0.833, "C": 0.024},
                    "actions_total_combos": {"F": 0.08, "C": 0.002},
                },
            },
        }],
        "action_solutions": [
            {"action": {"code": code}, "strategy": zeroes}
            for code in ("F", "C")
        ],
    }
    text = format_hand_detail(sol, "3h2h", "SB")
    assert_in("3♥️2♥️", text)
    assert_in("Exact combo 在此 solver node 沒有可用", text)
    assert_in("不可用 hand-class 平均替代", text)
    assert_not_in("Range 頻率", text)


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
    sys.path.insert(0, str(REPO_ROOT / "src"))
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
