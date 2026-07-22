"""Regression tests for the session 復盤 renderer (scripts/session_review.py).

Pure-function tests on render_tg — no DB, no network. Guards the North Star
invariants that live in the message shape: EV-weighted single-session facts,
no trend verdict / no percentile, deliberate per-item enqueue (no 全部排入),
and Telegram-safe callback_data.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from regression_tests.harness import assert_eq, assert_in, assert_not_in, assert_true, test

import session_review as sr

TPE = ZoneInfo("Asia/Taipei")


def _sample(empty: bool = False) -> dict:
    return {
        "session_id": 42,
        "started_at": datetime(2026, 7, 14, 20, 14, tzinfo=TPE),
        "ended_at": datetime(2026, 7, 14, 23, 47, tzinfo=TPE),
        "n_hands": 283,
        "n_decisions": 95,
        "per100": 1.9,
        "total_bb": 14.2,
        "n_lossy": 0 if empty else 11,
        "top_spots": [] if empty else [
            {"desc": "turn OOP 面對下注", "total_ev": 6.1, "n": 3,
             "drill_url": "https://app.gtowizard.com/practice?x=1", "enqueue_item": {}},
            {"desc": "翻前 vsOpen 3bet 位", "total_ev": 4.4, "n": 2,
             "drill_url": None, "enqueue_item": {}},
        ],
        "top_decisions": [] if empty else [
            {"combo": "Q♣️J♣️", "position": "HJ", "depth": 30.125,
             "boards": "", "desc": "MP flat 後面對 squeeze（對手 BB，你 IP）",
             "street_lines": [
                 "翻前: LJ Raise, HJ Call, BB Raise",
             ],
             "action_line": "Call→應Fold", "ev_loss": 0.76,
             "exact_url": "https://app.gtowizard.com/analyze/v4/hands/table?filters=x",
             "drill_url": "https://app.gtowizard.com/practice?d=1", "enqueue_item": {}},
            {"combo": "T♠️9♠️", "position": "CO", "depth": 25.0,
             "boards": "8h7c2d5sQc", "desc": "SRP 底池（HU），Hero CO 對 BB、處於 IP，轉牌首動",
             "street_lines": [
                 "Flop 8♥️7♣️2♦️: BB Check, Hero Bet 33%, BB Call",
                 "Turn 5♠️: BB Check",
             ],
             "action_line": "Raise→應Call", "ev_loss": 3.4,
             "exact_url": "https://app.gtowizard.com/a", "drill_url": None, "enqueue_item": {}},
            {"combo": "A♥️Q♦️", "position": "UTG+1", "depth": 40.0,
             "boards": "KsJd4c9h", "desc": "面對 3bet fold",
             "action_line": "Fold→應Call", "ev_loss": 2.6,
             "exact_url": "https://app.gtowizard.com/b", "drill_url": None, "enqueue_item": {}},
        ] + [
            {"combo": f"A{i}♠", "position": "BTN", "depth": 20.0,
             "boards": "", "desc": "vsOpen", "action_line": "Fold→應Raise",
             "ev_loss": 1.0 + i / 10, "exact_url": f"https://app.gtowizard.com/{i}",
             "drill_url": None, "enqueue_item": {}}
            for i in range(4, 9)
        ],
        "honesty": {"discarded_n": 6, "low_conf_n": 3},
        "empty": empty,
    }


def _all_buttons(rows):
    return [b for row in rows for b in row]


@test
def test_session_review_uses_ledger_wall_clock_no_tz_shift():
    d = _sample()
    d["started_at"] = datetime(2026, 7, 20, 19, 0, tzinfo=timezone.utc)
    d["ended_at"] = datetime(2026, 7, 20, 21, 4, tzinfo=timezone.utc)
    html = sr.render_tg(d)["html"]
    assert_in("7/20 19:00–21:04", html)
    assert_not_in("7/21 03:00", html)


@test
def test_session_review_action_line_shows_recommended_action():
    assert_eq(sr.action_line("C", "F"), "Call→應Fold")
    assert_eq(sr.action_line("R2.5", "C"), "Raise→應Call")


@test
def test_session_review_postflop_bet_not_raise_with_size():
    detail = {
        "game_analysis": {"game_points": [
            {"real_game": {"current_street": {"type": "FLOP"}},
             "real_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "solved_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "analysis_solved": {"available_actions": []}},
            {"real_game": {"current_street": {"type": "FLOP"}},
             "real_game_action": {"position": "HJ", "code": "R1.8", "display_name": "BET",
                                  "betsize_by_pot": "0.333"},
             "solved_game_action": {"position": "HJ", "code": "R1.8", "display_name": "BET",
                                    "betsize_by_pot": "0.333"},
             "analysis_solved": {"available_actions": [
                 {"selected": True, "action": {"position": "HJ", "code": "R1.8",
                                               "display_name": "BET", "betsize_by_pot": "0.333"},
                  "correctness": "WRONG_MOVE", "ev": "1"},
                 {"selected": False, "action": {"position": "HJ", "code": "X",
                                                "display_name": "CHECK"},
                  "correctness": "BEST_MOVE", "ev": "2"},
             ]}},
        ]}
    }
    old_loader = sr._load_detail
    try:
        sr._load_detail = lambda _p: detail
        ctx = sr.decision_action_context({
            "raw_path": "unused", "street": "flop", "decision_idx": 0,
            "hero_pos": "HJ", "boards": "Qs9s8c"})
    finally:
        sr._load_detail = old_loader
    assert_in("Flop Q♠️9♠️8♣️: BB Check", ctx["street_line"])
    assert_eq(ctx["action_line"], "Bet 33%→應Check")


@test
def test_session_review_turn_first_to_act_keeps_turn_card_and_action_line():
    detail = {
        "game_analysis": {"game_points": [
            {"real_game": {"current_street": {"type": "FLOP"}},
             "real_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "solved_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "analysis_solved": {"available_actions": []}},
            {"real_game": {"current_street": {"type": "FLOP"}},
             "real_game_action": {"position": "LP", "code": "R1.1", "display_name": "BET",
                                  "betsize_by_pot": "0.333"},
             "solved_game_action": {"position": "LP", "code": "R1.1", "display_name": "BET",
                                    "betsize_by_pot": "0.333"},
             "analysis_solved": {"available_actions": []}},
            {"real_game": {"current_street": {"type": "FLOP"}},
             "real_game_action": {"position": "BB", "code": "C", "display_name": "CALL"},
             "solved_game_action": {"position": "BB", "code": "C", "display_name": "CALL"},
             "analysis_solved": {"available_actions": []}},
            {"real_game": {"current_street": {"type": "TURN"}},
             "real_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "solved_game_action": {"position": "BB", "code": "X", "display_name": "CHECK"},
             "analysis_solved": {"available_actions": [
                 {"selected": True, "action": {"position": "BB", "code": "R2.5",
                                               "display_name": "RAISE"},
                  "correctness": "WRONG_MOVE", "ev": "1"},
                 {"selected": False, "action": {"position": "BB", "code": "X",
                                                "display_name": "CHECK"},
                  "correctness": "BEST_MOVE", "ev": "2"},
             ]}},
        ]}
    }
    old_loader = sr._load_detail
    try:
        sr._load_detail = lambda _p: detail
        ctx = sr.decision_action_context({
            "raw_path": "unused", "street": "turn", "decision_idx": 0,
            "hero_pos": "BB", "boards": "6s5d4h5h"})
    finally:
        sr._load_detail = old_loader
    assert_eq(ctx["street_lines"], [
        "Flop 6♠️5♦️4♥️: Hero Check, LP Bet 33%, Hero Call",
        "Turn 5♥️: Hero 首動",
    ])
    assert_eq(ctx["action_line"], "Raise→應Check")


@test
def test_session_review_decision_depth_prefers_solver_effective_stack():
    assert_eq(sr._decision_display_depth({
        "preflop_depth_bb": 50.832, "played_depth_bb": 50.832, "solver_depth_bb": 12.0,
    }), 12.0)
    assert_eq(sr.depth_label(sr._decision_display_depth({
        "preflop_depth_bb": 50.832, "played_depth_bb": 50.832, "solver_depth_bb": 12.0,
    })), "有效 12bb")


@test
def test_session_review_compacts_preflop_history():
    assert_eq(
        sr._format_street_history("preflop", [
            {"position": "UTG", "code": "F"},
            {"position": "LJ", "code": "F"},
        ], "CO", ""),
        "翻前: Fold to Hero")
    assert_eq(
        sr._format_street_history("preflop", [
            {"position": "UTG", "code": "F"},
            {"position": "UTG+1", "code": "F"},
            {"position": "LJ", "code": "F"},
            {"position": "HJ", "code": "R2"},
            {"position": "CO", "code": "C"},
            {"position": "BTN", "code": "F"},
            {"position": "SB", "code": "F"},
        ], "BB", ""),
        "翻前: HJ Raise, CO Call")


@test
def test_session_review_full_message():
    out = sr.render_tg(_sample())
    html = out["html"]
    # header + span + core numbers
    assert_in("這場復盤", html)
    assert_in("7/14 20:14–23:47", html)
    assert_in("283", html)          # hands
    assert_in("1.9 bb/100", html)   # per100
    assert_in("14.2 bb", html)      # total loss
    # top spot + concrete decision rows surfaced
    assert_in("EV Loss 最多的情境", html)
    assert_in("turn OOP 面對下注", html)
    assert_in("6.1 bb", html)
    assert_in("最值得回看的 8 個決策", html)
    assert_in("Q♣️J♣️", html)
    assert_in("HJ 有效 30bb", html)
    assert_in("MP flat 後面對 squeeze", html)
    assert_in("1️⃣ Q♣️J♣️ HJ 有效 30bb｜MP flat 後面對 squeeze", html)
    assert_in("翻前: LJ Raise, HJ Call, BB Raise｜<b>Call→應Fold</b>", html)
    assert_in("Call→應Fold", html)
    assert_in("−<b>0.76bb</b>", html)
    assert_in("T♠️9♠️", html)
    assert_in("Flop 8♥️7♣️2♦️: BB Check, Hero Bet 33%, BB Call", html)
    assert_in("Turn 5♠️: BB Check｜<b>Raise→應Call</b>", html)
    # honesty caveat with session-scoped counts
    assert_in("limp 6 手未計", html)
    assert_in("3 個低信心未計", html)


@test
def test_session_review_no_trend_no_percentile():
    """North Star §2.1/§7-490: single session is descriptive, never a verdict,
    never a percentile baseline."""
    html = sr.render_tg(_sample())["html"]
    assert_not_in("進步/退步", html)
    assert_not_in("百分位", html)          # no percentile machinery
    assert_not_in("比上週", html)          # no weekly trend comparison


@test
def test_session_review_deliberate_enqueue_only():
    """No 全部排入 batch shortcut, no skip button — adds are per-item deliberate
    (§5.10); review link points at the exact hand (owner request)."""
    out = sr.render_tg(_sample())
    labels = [b["text"] for b in _all_buttons(out["buttons"])]
    assert_true(not any("全部" in t for t in labels), f"unexpected batch button: {labels}")
    assert_true(not any("略過" in t for t in labels), f"skip button must be gone: {labels}")
    assert_true(any("排入佇列" in t for t in labels), "missing spot enqueue button")
    assert_true(any("復盤" in t for t in labels), "missing decision review button")
    assert_true(sum("排入佇列" in t for t in labels) >= 2, "missing enqueue buttons")
    assert_true(not any("queue" in t for t in labels), "enqueue copy should be 統一中文")
    assert_true(not any("🎯 練" in t for t in labels),
                "decision drill buttons make Telegram reply_markup too large")
    # every review button links to the exact hand_id__in filter, not a day range
    review_btns = [b for b in _all_buttons(out["buttons"]) if "復盤" in b["text"]]
    assert_true(review_btns and all("url" in b for b in review_btns),
                "review buttons must be exact-hand URL links")


@test
def test_session_review_callback_data_telegram_safe():
    out = sr.render_tg(_sample())
    for b in _all_buttons(out["buttons"]):
        cb = b.get("callback_data")
        if cb is not None:
            assert_true(len(cb.encode()) <= 64, f"callback_data too long: {cb}")
            assert_true(cb.split(":")[0] in {"srd2", "srv2"}, f"bad callback: {cb}")


@test
def test_session_review_callback_key_survives_session_id_rebuild():
    """Callback identity must not depend on ledger_sessions.id, which is
    delete/reinserted by every ingest session rebuild."""
    before = _sample()
    after = dict(before)
    after["session_id"] = 1409
    assert_eq(sr.session_callback_key(before), sr.session_callback_key(after))
    out = sr.render_tg(before)
    callbacks = [b.get("callback_data") for b in _all_buttons(out["buttons"])
                 if b.get("callback_data")]
    assert_true(callbacks, "session review has enqueue callbacks")
    assert_true(all(":42:" not in cb for cb in callbacks),
                f"volatile session id leaked into callbacks: {callbacks}")


@test
def test_session_review_decision_enqueue_persists_review_url_as_drill_url():
    """Decision-level queue rows use drill_queue.drill_url for their Analyze
    review link; a dead review_url key is ignored by enqueue_one."""
    import inspect
    src = inspect.getsource(sr._decision_items)
    assert_in('"drill_url": exact_url', src)
    assert_not_in('"review_url": exact_url', src)


@test
def test_session_review_auto_send_skips_clean_session():
    """Sync auto-append fires only when there's something to review (§7-11 依從):
    non-empty → push, clean/empty → stay silent (manual /review still works)."""
    assert_true(sr.should_auto_send(_sample()), "non-empty session should auto-push")
    assert_true(not sr.should_auto_send(_sample(empty=True)),
                "clean session must NOT auto-push")


@test
def test_session_review_empty_session():
    out = sr.render_tg(_sample(empty=True))
    assert_in("沒有值得復盤的漏損", out["html"])
    # nothing to enqueue, no skip button → no buttons at all
    assert_eq(len(_all_buttons(out["buttons"])), 0)

@test
def test_session_review_marks_hu_only_for_real_sb_bb_heads_up_pots():
    """HU means an actual SB-vs-BB heads-up spot, not merely two players left.

    CO/SB or BTN/BB postflop pots can have exactly two un-folded players, but
    they are still non-HU table spots and must keep the neutral label.
    """
    def gp(active_positions, hero_pos="SB"):
        return {
            "real_game_action": {"position": hero_pos, "code": "C"},
            "solved_game_action": {"position": hero_pos, "code": "C"},
            "analysis_solved": {"available_actions": [
                {"selected": True, "action": {"code": "C"}},
                {"correctness": "BEST_MOVE", "action": {"code": "F"}, "ev": 0},
            ]},
            "real_game": {
                "current_street": {"type": "FLOP"},
                "players": [
                    {"position": p, "is_folded": p not in active_positions}
                    for p in ["CO", "BTN", "SB", "BB"]
                ],
            },
        }

    sb_bb = {"game_analysis": {"game_points": [gp({"SB", "BB"}, "SB")]}}
    co_sb = {"game_analysis": {"game_points": [gp({"CO", "SB"}, "CO")]}}
    btn_bb = {"game_analysis": {"game_points": [gp({"BTN", "BB"}, "BTN")]}}
    three_way = {"game_analysis": {"game_points": [gp({"CO", "BTN", "SB"}, "CO")]}}

    assert_true(sr._is_real_hu_decision(
        sb_bb, hero_pos="SB", target_street="flop", target_idx=0))
    assert_true(not sr._is_real_hu_decision(
        co_sb, hero_pos="CO", target_street="flop", target_idx=0))
    assert_true(not sr._is_real_hu_decision(
        btn_bb, hero_pos="BTN", target_street="flop", target_idx=0))
    assert_true(not sr._is_real_hu_decision(
        three_way, hero_pos="CO", target_street="flop", target_idx=0))
    assert_eq(sr.hand_desc({
        "spot_category": "flop", "spot_leaf": "flop:SRP:SBvBB:IP:vs_raise",
        "hero_cat": "SB", "villain_cat": "BB", "ip_oop": "IP", "hero_pos": "SB",
    }, is_real_hu=True), "SRP 底池（HU），Hero SB 對 BB、處於 IP，翻牌面對加注")
    assert_eq(sr.hand_desc({
        "spot_category": "flop", "spot_leaf": "flop:SRP:COvSB:IP:vs_raise",
        "hero_cat": "CO", "villain_cat": "SB", "ip_oop": "IP", "hero_pos": "CO",
    }, is_real_hu=False), "SRP 底池，Hero CO 對 SB、處於 IP，翻牌面對加注")
