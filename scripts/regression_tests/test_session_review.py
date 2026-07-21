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
            {"combo": "Q♣J♣", "position": "HJ", "depth": 30.125,
             "boards": "", "desc": "MP flat 後面對 squeeze（對手 BB，你 IP）",
             "street_lines": [
                 "PF: LJ Raise, HJ Call, BB Raise",
             ],
             "action_line": "Call→應Fold", "ev_loss": 0.76,
             "exact_url": "https://app.gtowizard.com/analyze/v4/hands/table?filters=x",
             "drill_url": "https://app.gtowizard.com/practice?d=1", "enqueue_item": {}},
            {"combo": "T♠9♠", "position": "CO", "depth": 25.0,
             "boards": "8h7c2d5sQc", "desc": "turn barrel",
             "street_lines": [
                 "Flop 8h7c2d: BB Check, Hero Bet 33%, BB Call",
                 "Turn 5s: BB Check",
             ],
             "action_line": "Raise→應Call", "ev_loss": 3.4,
             "exact_url": "https://app.gtowizard.com/a", "drill_url": None, "enqueue_item": {}},
            {"combo": "A♥Q♦", "position": "UTG+1", "depth": 40.0,
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
    assert_in("Flop Qs9s8c: BB Check", ctx["street_line"])
    assert_eq(ctx["action_line"], "Bet 33%→應Check")


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
    assert_in("Q♣J♣", html)
    assert_in("HJ 30bb", html)
    assert_in("MP flat 後面對 squeeze", html)
    assert_in("1️⃣ Q♣J♣ HJ 30bb｜MP flat 後面對 squeeze", html)
    assert_in("PF: LJ Raise, HJ Call, BB Raise｜<b>Call→應Fold</b>", html)
    assert_in("Call→應Fold", html)
    assert_in("−<b>0.76bb</b>", html)
    assert_in("T♠9♠", html)
    assert_in("Flop 8h7c2d: BB Check, Hero Bet 33%, BB Call", html)
    assert_in("Turn 5s: BB Check｜<b>Raise→應Call</b>", html)
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
    assert_true(any("入 queue" in t for t in labels), "missing decision enqueue button")
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
            assert_true(cb.split(":")[0] in {"srd", "srv"}, f"bad callback: {cb}")


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
