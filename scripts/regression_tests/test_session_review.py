"""Regression tests for the session 復盤 renderer (scripts/session_review.py).

Pure-function tests on render_tg — no DB, no network. Guards the North Star
invariants that live in the message shape: EV-weighted single-session facts,
no trend verdict / no percentile, deliberate per-item enqueue (no 全部排入),
and Telegram-safe callback_data.
"""
from datetime import datetime
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
        "top_hands": [] if empty else [
            {"combo": "T♠9♠", "boards": "8h7c2d5sQc", "desc": "turn barrel",
             "max_ev": 3.4, "total_ev": 3.4,
             "exact_url": "https://app.gtowizard.com/analyze/v4/hands/table?filters=x",
             "enqueue_item": {}},
            {"combo": "A♥Q♦", "boards": "KsJd4c9h", "desc": "面對 3bet fold",
             "max_ev": 2.6, "total_ev": 2.6,
             "exact_url": "https://app.gtowizard.com/a", "enqueue_item": {}},
            {"combo": "K♦K♥", "boards": "Ac8s3s7d2h", "desc": "river call 太寬",
             "max_ev": 1.8, "total_ev": 1.8,
             "exact_url": "https://app.gtowizard.com/c", "enqueue_item": {}},
        ],
        "honesty": {"discarded_n": 6, "low_conf_n": 3},
        "empty": empty,
    }


def _all_buttons(rows):
    return [b for row in rows for b in row]


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
    # top spot + top hand surfaced
    assert_in("turn OOP 面對下注", html)
    assert_in("6.1 bb", html)
    assert_in("T♠9♠", html)
    assert_in("8h7c2d5sQc", html)
    # honesty caveat with session-scoped counts
    assert_in("limp 6 手未計", html)
    assert_in("3 個低信心未計", html)


@test
def test_session_review_no_trend_no_percentile():
    """North Star §2.1/§7-490: single session is descriptive, never a verdict,
    never a percentile baseline."""
    html = sr.render_tg(_sample())["html"]
    assert_in("不是進步/退步結論", html)   # the variance guard line is present
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
    assert_true(any("復盤" in t for t in labels), "missing hand review button")
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
    assert_in("不是進步/退步結論", out["html"])
    # nothing to enqueue, no skip button → no buttons at all
    assert_eq(len(_all_buttons(out["buttons"])), 0)
