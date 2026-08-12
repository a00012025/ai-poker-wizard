"""Regression tests for the session 復盤 renderer (scripts/session_review.py).

Pure-function tests on render_tg — no DB, no network. Guards the North Star
invariants that live in the message shape: EV-weighted single-session facts,
no trend verdict / no percentile, deliberate per-item enqueue (no 全部排入),
and Telegram-safe callback_data.
"""
import asyncio
import inspect
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
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
            {"combo": "Q☘️J☘️", "position": "HJ", "depth": 30.125,
             "boards": "", "desc": "MP flat 後面對 squeeze（對手 BB，你 IP）",
             "street_lines": [
                 "翻前: LJ Raise, HJ Call, BB Raise",
             ],
             "action_line": "Call→應Fold", "ev_loss": 0.76,
             "line_frequency": 0.0087, "rare_line": True,
             "ref_hand_id": "online-hand-1",
             "exact_url": "https://app.gtowizard.com/analyze/v4/hands/table?filters=x",
             "study_url": "https://app.gtowizard.com/solutions?history_spot=3",
             "drill_url": "https://app.gtowizard.com/practice?d=1", "enqueue_item": {}},
            {"combo": "T♠️9♠️", "position": "CO", "depth": 25.0,
             "boards": "8h7c2d5sQc", "desc": "SRP 底池（HU），Hero CO 對 BB、處於 IP，轉牌首動",
             "street_lines": [
                 "Flop 8♥️7☘️2🔷: BB Check, Hero Bet 33%, BB Call",
                 "Turn 5♠️: BB Check",
             ],
             "action_line": "Raise→應Call", "ev_loss": 3.4,
             "ref_hand_id": "online-hand-2",
             "exact_url": "https://app.gtowizard.com/a",
             "study_url": "https://app.gtowizard.com/solutions?history_spot=7",
             "drill_url": None, "enqueue_item": {}},
            {"combo": "A♥️Q🔷", "position": "UTG+1", "depth": 40.0,
             "boards": "KsJd4c9h", "desc": "面對 3bet fold",
             "action_line": "Fold→應Call", "ev_loss": 2.6,
             "ref_hand_id": "online-hand-3",
             "exact_url": "https://app.gtowizard.com/b", "study_url": None,
             "drill_url": None, "enqueue_item": {}},
        ] + [
            {"combo": f"A{i}♠", "position": "BTN", "depth": 20.0,
             "boards": "", "desc": "vsOpen", "action_line": "Fold→應Raise",
             "ev_loss": 1.0 + i / 10, "ref_hand_id": f"online-hand-{i}",
             "exact_url": f"https://app.gtowizard.com/{i}", "study_url": None,
             "drill_url": None, "enqueue_item": {}}
            for i in range(4, 11)
        ],
        "honesty": {"discarded_n": 6, "low_conf_n": 3},
        "empty": empty,
    }


def _all_buttons(rows):
    return [b for row in rows for b in row]


@test
def test_session_review_renders_normalized_utc_in_taipei_time():
    d = _sample()
    d["started_at"] = datetime(2026, 7, 20, 11, 0, tzinfo=timezone.utc)
    d["ended_at"] = datetime(2026, 7, 20, 13, 4, tzinfo=timezone.utc)
    html = sr.render_tg(d)["html"]
    assert_in("7/20 19:00–21:04", html)
    assert_not_in("7/20 11:00", html)


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
    assert_in("Flop Q♠️9♠️8☘️: BB Check", ctx["street_line"])
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
        "Flop 6♠️5🔷4♥️: Hero Check, LP Bet 33%, Hero Call",
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
def test_session_review_study_url_binds_user_token_and_fails_closed():
    """Study links use per-user auth and never fall back to approximate nodes."""
    import gto_api
    import gto_credentials

    calls = []
    old_builder = sr.qf._study_solution_link
    old_credentials = gto_credentials.get_user_credentials
    old_set = gto_api.set_user_token
    old_clear = gto_api.clear_user_token
    try:
        gto_credentials.get_user_credentials = lambda user_id: SimpleNamespace(
            access_token="access", client_id="client")
        gto_api.set_user_token = lambda token, client_id, user_id: calls.append(
            ("set", token, client_id, user_id))
        gto_api.clear_user_token = lambda: calls.append(("clear",))
        sr.qf._study_solution_link = lambda _row: {
            "url": "https://app.gtowizard.com/solutions?history_spot=3",
            "line_frequency": 0.0087, "rare_line": True,
        }
        link = sr._decision_study_link({}, user_id=123)
        sr.qf._study_solution_link = lambda _row: None
        missing = sr._decision_study_link({}, user_id=None)
    finally:
        sr.qf._study_solution_link = old_builder
        gto_credentials.get_user_credentials = old_credentials
        gto_api.set_user_token = old_set
        gto_api.clear_user_token = old_clear
    assert_in("/solutions?", link["url"])
    assert_true(link["rare_line"])
    assert_eq(calls, [("set", "access", "client", 123), ("clear",)])
    assert_eq(missing, None)


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
    assert_eq(sr.TOP_DECISIONS, 10)
    assert_in("LIMIT 10", sr._TOP_DECISIONS_SQL)
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
    assert_in("最值得回看的 10 個決策", html)
    assert_in("🔟 A10♠ BTN 有效 20bb｜vsOpen", html)
    assert_in("Q☘️J☘️", html)
    assert_in("HJ 有效 30bb", html)
    assert_in("MP flat 後面對 squeeze", html)
    assert_in("1️⃣ Q☘️J☘️ HJ 有效 30bb｜MP flat 後面對 squeeze", html)
    assert_in("翻前: LJ Raise, HJ Call, BB Raise｜<b>Call→應Fold</b>", html)
    assert_in("Call→應Fold", html)
    assert_in("−<b>0.76bb</b>", html)
    assert_in("GTO 只走 <b>0.87%</b>", html)
    assert_in("EV loss 仍是這手的實際條件式損失", html)
    assert_in("T♠️9♠️", html)
    assert_in("Flop 8♥️7☘️2🔷: BB Check, Hero Bet 33%, BB Call", html)
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
    assert_true(any("手牌" in t for t in labels), "missing exact-hand button")
    assert_true(any("復盤" in t for t in labels), "missing GTOW Study review button")
    assert_true(any("教練" in t for t in labels), "missing coach button")
    assert_true(sum("排入佇列" in t for t in labels) >= 2, "missing enqueue buttons")
    assert_true(not any("queue" in t for t in labels), "enqueue copy should be 統一中文")
    assert_true(not any("🎯 練" in t for t in labels),
                "decision drill buttons make Telegram reply_markup too large")
    hand_btns = [b for b in _all_buttons(out["buttons"]) if "手牌" in b["text"]]
    assert_true(hand_btns and all("/analyze/" in b["url"] or b["url"].endswith(("/a", "/b", "/4", "/5", "/6", "/7", "/8", "/9", "/10"))
                                  for b in hand_btns),
                "hand buttons must retain exact Analyzer hand links")
    review_btns = [b for b in _all_buttons(out["buttons"]) if "復盤" in b["text"]]
    assert_true(review_btns and all("/solutions?" in b["url"] for b in review_btns),
                "review buttons must link directly to GTOW Study solutions")
    coach_btns = [b for b in _all_buttons(out["buttons"]) if "教練" in b["text"]]
    assert_true(coach_btns and all(b.get("callback_data", "").startswith("src2:")
                                   for b in coach_btns),
                "coach buttons must use stable online-session callbacks")


@test
def test_session_review_callback_data_telegram_safe():
    out = sr.render_tg(_sample())
    for b in _all_buttons(out["buttons"]):
        cb = b.get("callback_data")
        if cb is not None:
            assert_true(len(cb.encode()) <= 64, f"callback_data too long: {cb}")
            assert_true(cb.split(":")[0] in {"srd2", "srv2", "src2"},
                        f"bad callback: {cb}")


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
def test_online_session_coach_uses_grounded_solver_context():
    from telegram_bot.bot import PokerWizardBot
    import analyze_hand

    captured = {}

    class SessionManager:
        def __init__(self):
            self.hand_contexts = {}
            self.pending_images = {}

        async def _chat_with_tools(self, chat_id, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return "教練文字\nFOLLOWUP: 為什麼？"

        @staticmethod
        def _extract_followups(_response):
            return "教練文字", ["為什麼？"]

    bot = PokerWizardBot.__new__(PokerWizardBot)
    bot.session_manager = SessionManager()
    main_thread = threading.get_ident()
    worker_threads = []
    bot._setup_user_token = lambda *_args: worker_threads.append(threading.get_ident())
    bot._clear_user_token = lambda: worker_threads.append(threading.get_ident())
    old_analyze = analyze_hand.analyze_hand_full
    try:
        def fake_analyze(hand):
            worker_threads.append(threading.get_ident())
            return {"text": "已驗證 solver 事實", "hand": hand,
                    "validation": {}}
        analyze_hand.analyze_hand_full = fake_analyze
        result = asyncio.run(bot._analyze_online_parsed_hand(
            10, 20, "online-hand-1",
            {"hero_position": "CO", "hero_hand": "AsKs",
             "effective_bb": 30, "players_at_table": 8,
             "preflop_actions": "F-F-R2", "streets": []},
            None, "refresh-token"))
    finally:
        analyze_hand.analyze_hand_full = old_analyze

    assert_in("教練文字", result)
    assert_in("已驗證 solver 事實", captured["prompt"])
    assert_in("不要重新解析或改寫動作", captured["prompt"])
    assert_eq(bot.session_manager.hand_contexts[10]["followup_questions"], ["為什麼？"])
    assert_true(worker_threads and all(tid != main_thread for tid in worker_threads),
                "token binding and solver analysis must stay off the event-loop thread")


@test
def test_online_session_coach_callback_is_registered_and_routed():
    from telegram_bot.bot import PokerWizardBot

    handler_source = inspect.getsource(PokerWizardBot.handle_live_button)
    setup_source = inspect.getsource(PokerWizardBot.setup_handlers)
    assert_in('data.startswith("src2:")', handler_source)
    assert_in("|src2):", setup_source)


@test
def test_recent_online_sessions_are_newest_first_and_bounded():
    class Conn:
        async def fetch(self, sql, *args):
            assert_in("FROM ledger_sessions", sql)
            assert_in("ORDER BY ended_at DESC", sql)
            assert_in("LIMIT $1", sql)
            assert_eq(args, (8,))
            return [{
                "id": 42,
                "started_at": datetime(2026, 7, 14, 12, 14, tzinfo=timezone.utc),
                "ended_at": datetime(2026, 7, 14, 15, 47, tzinfo=timezone.utc),
                "duration_min": 213,
                "tournaments": ["t1", "t2"],
                "max_concurrent_tables": 2,
                "hands_count": 283,
            }]

    sessions = asyncio.run(sr.list_recent_sessions(Conn(), 8))
    assert_eq(len(sessions), 1)
    assert_eq(sessions[0]["id"], 42)
    assert_eq(sessions[0]["hands_count"], 283)


@test
def test_recent_online_sessions_payload_uses_stable_resend_keys():
    from telegram_bot.bot import _recent_online_sessions_payload

    session = {
        "id": 42,
        "started_at": datetime(2026, 7, 14, 12, 14, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 7, 14, 15, 47, tzinfo=timezone.utc),
        "duration_min": 213,
        "tournaments": ["t1", "t2"],
        "max_concurrent_tables": 2,
        "hands_count": 283,
    }
    html, buttons = _recent_online_sessions_payload([session])
    stable_key = sr.session_callback_key(session)

    assert_in("最近線上 Sessions", html)
    assert_in("7/14 20:14–23:47", html)
    assert_in("283 手", html)
    assert_in("2 桌", html)
    assert_eq(buttons[0][0]["callback_data"], f"ors:{stable_key}")
    assert_not_in(":42", buttons[0][0]["callback_data"])


@test
def test_online_sessions_command_lists_resend_buttons():
    from telegram_bot.bot import PokerWizardBot

    class Pool:
        async def fetch(self, _sql, *_args):
            return [{
                "id": 42,
                "started_at": datetime(2026, 7, 14, 12, 14,
                                       tzinfo=timezone.utc),
                "ended_at": datetime(2026, 7, 14, 15, 47,
                                     tzinfo=timezone.utc),
                "duration_min": 213,
                "tournaments": ["t1", "t2"],
                "max_concurrent_tables": 2,
                "hands_count": 283,
            }]

    class Message:
        text = "/sessions"
        sent = []

        async def reply_text(self, *args, **kwargs):
            self.sent.append((args, kwargs))

    bot = object.__new__(PokerWizardBot)
    bot.admin_chat_id = 556028753
    bot.db = SimpleNamespace(pool=Pool())
    bot.log = logging.getLogger("test-online-sessions-command")
    bot._user_label = lambda _update: "owner"
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=556028753),
        effective_chat=SimpleNamespace(id=99),
        message=Message(),
    )
    asyncio.run(bot.online_sessions_command(update, SimpleNamespace()))

    html = update.message.sent[0][0][0]
    assert_in("最近線上 Sessions", html)
    assert_in("283 手", html)
    markup = update.message.sent[0][1]["reply_markup"].to_dict()
    buttons = [b for row in markup["inline_keyboard"] for b in row]
    assert_true(any(b.get("callback_data", "").startswith("ors:")
                    for b in buttons))


@test
def test_online_sessions_command_and_resend_callback_are_registered():
    from src.telegram_bot.bot import PokerWizardBot

    handlers = inspect.getsource(PokerWizardBot.setup_handlers)
    assert_in('["sessions", "online_sessions"]', handlers)
    assert_in("|ors|", handlers)
    menu = (sr.ROOT / "src/main_gemini.py").read_text()
    assert_in('BotCommand("sessions", "最近線上 sessions／重傳復盤")', menu)


@test
def test_recent_online_session_button_sends_fresh_summary():
    from telegram_bot.bot import PokerWizardBot

    session = {
        "id": 42,
        "started_at": datetime(2026, 7, 14, 12, 14, tzinfo=timezone.utc),
        "ended_at": datetime(2026, 7, 14, 15, 47, tzinfo=timezone.utc),
        "duration_min": 213,
        "tournaments": ["t1"],
        "max_concurrent_tables": 1,
        "hands_count": 283,
    }
    data = _sample()
    stable_key = sr.session_callback_key(session)
    captured = {}

    async def fake_resolve(_conn, key):
        assert_eq(key, stable_key)
        return session

    async def fake_compute(_conn, resolved, user_id=None):
        assert_eq(resolved, session)
        captured["compute_user_id"] = user_id
        return data

    class Query:
        data = f"ors:{stable_key}"
        answers = []

        async def answer(self, text=None):
            self.answers.append(text)

    class TgBot:
        async def send_message(self, *args, **kwargs):
            captured["send"] = (args, kwargs)

    old_resolve, old_compute = sr.resolve_session_key, sr.compute
    sr.resolve_session_key, sr.compute = fake_resolve, fake_compute
    try:
        bot = object.__new__(PokerWizardBot)
        bot.admin_chat_id = 556028753
        bot.db = SimpleNamespace(pool=object())
        bot.log = logging.getLogger("test-online-session-resend")
        context = SimpleNamespace(
            bot=TgBot(), application=SimpleNamespace(bot_data={}))
        update = SimpleNamespace(
            callback_query=Query(),
            effective_user=SimpleNamespace(id=556028753),
            effective_chat=SimpleNamespace(id=99),
        )
        asyncio.run(bot.handle_live_button(update, context))
    finally:
        sr.resolve_session_key, sr.compute = old_resolve, old_compute

    assert_eq(captured["compute_user_id"], 556028753)
    assert_eq(update.callback_query.answers, [None])
    assert_eq(captured["send"][0][0], 99)
    assert_in("這場復盤", captured["send"][0][1])
    assert_true(stable_key in context.application.bot_data["srev"])


@test
def test_session_review_decision_enqueue_persists_review_url_as_drill_url():
    """Decision-level queue rows use drill_queue.drill_url for their Analyze
    review link; a dead review_url key is ignored by enqueue_one."""
    import inspect
    src = inspect.getsource(sr._decision_items)
    assert_in('"drill_url": exact_url', src)
    assert_not_in('"review_url": exact_url', src)


@test
def test_session_review_top_decisions_use_one_joined_query_without_dead_drill_work():
    """Top-N rendering must not issue metadata/source N+1 queries.

    Hand metadata is joined into the ranked query, and decision rows only need
    Analyze/Study links; their queue payload already stores the Analyze URL.
    """
    calls = []

    class Conn:
        async def fetch(self, sql, *args):
            calls.append((sql, args))
            return [{
                "ref_hand_id": "hand-1", "street": "preflop", "decision_idx": 0,
                "spot_leaf": "BB_vsRaiseCall_OOP", "spot_category": "vsOpen",
                "hero_cat": "BB", "villain_cat": "BTN", "ip_oop": "OOP",
                "hero_pos": "BB", "ev_loss_bb": 0.419, "approx_flags": [],
                "played_at": datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
                "taken_code": "AI", "best_code": "C", "correctness": "BLUNDER",
                "pot_type": "Preflop", "eff_stack": "15_25", "gametype": "MTTGeneral",
                "played_depth_bb": 20.0, "solver_depth_bb": 20.0,
                "hero_hand": "7d7c", "boards": "", "raw_path": "detail.json.gz",
                "preflop_depth_bb": 20.0,
                "source_hands": [{"hand_id": "hand-1", "street": "preflop",
                                  "decision_idx": 0, "ev_loss_bb": 0.419,
                                  "taken_code": "AI", "best_code": "C",
                                  "src": "online"}],
            }]

        async def fetchrow(self, *_args):
            raise AssertionError("top decisions must not issue per-row fetchrow queries")

    async def dead_drill_builder(*_args, **_kwargs):
        raise AssertionError("decision Trainer URL is unused and must not be built")

    old_drill = sr.qf.queue_drill_url_from_sources
    old_urls = sr.qf.gtow_analyze_hands_urls
    old_label = sr.qf.review_label
    old_action = sr.decision_action_context
    old_study = sr._decision_study_link
    try:
        sr.qf.queue_drill_url_from_sources = dead_drill_builder
        sr.qf.gtow_analyze_hands_urls = lambda _ids: [("https://example/analyze", [])]
        sr.qf.review_label = lambda _row: "review"
        sr.decision_action_context = lambda _row: {
            "action_line": "All-in→應Call", "street_lines": [], "is_real_hu": False}
        sr._decision_study_link = lambda _row, _user_id=None: {
            "url": "https://example/study", "rare_line": False,
            "line_frequency": 0.25,
        }
        out = asyncio.run(sr._decision_items(Conn(), 42, user_id=7))
    finally:
        sr.qf.queue_drill_url_from_sources = old_drill
        sr.qf.gtow_analyze_hands_urls = old_urls
        sr.qf.review_label = old_label
        sr.decision_action_context = old_action
        sr._decision_study_link = old_study

    assert_eq(len(calls), 1)
    assert_eq(calls[0][1], (42,))
    assert_in("JOIN ledger_hands h", calls[0][0])
    assert_in("h.hero_hand", calls[0][0])
    assert_eq(out[0]["combo"], "7🔷7☘️")
    assert_eq(out[0]["drill_url"], None)
    assert_eq(out[0]["enqueue_item"]["drill_url"], "https://example/analyze")


@test
def test_session_review_session_membership_index_migration_exists():
    migration = (sr.ROOT / "supabase/migrations" /
                 "20260809130000_ledger_hands_session_index.sql").read_text()
    assert_in("idx_ledger_hands_session", migration)
    assert_in("ledger_hands(session_id)", migration)


@test
def test_session_review_parallelizes_independent_reads_for_runtime_pool():
    """The Telegram path passes a pool, so independent summary branches overlap."""
    started = 0
    all_started = asyncio.Event()

    async def rendezvous(result):
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.5)
        return result

    class Pool:
        acquire = object()  # asyncpg Pool marker used by compute()

        async def fetchrow(self, sql, *_args):
            if "discarded_n" in sql:
                return await rendezvous({"discarded_n": 0, "low_conf_n": 0})
            return await rendezvous({"n": 1, "per100": 0.0, "total_bb": 0.0,
                                     "n_lossy": 0})

    async def spots(_conn, _sid):
        return await rendezvous([])

    async def decisions(_conn, _sid, user_id=None):
        assert_eq(user_id, 7)
        return await rendezvous([])

    old_spots, old_decisions = sr._spot_items, sr._decision_items
    sr._spot_items, sr._decision_items = spots, decisions
    try:
        data = asyncio.run(sr.compute(Pool(), {
            "id": 42,
            "started_at": datetime(2026, 8, 8, tzinfo=timezone.utc),
            "ended_at": datetime(2026, 8, 8, 1, tzinfo=timezone.utc),
            "hands_count": 1,
        }, user_id=7))
    finally:
        sr._spot_items, sr._decision_items = old_spots, old_decisions

    assert_eq(started, 4)
    assert_eq(data["n_decisions"], 1)


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
    }, is_real_hu=True), "SRP（HU）｜Hero SB 對 BB，IP；翻牌 Hero 面對加注")
    assert_eq(sr.hand_desc({
        "spot_category": "flop", "spot_leaf": "flop:SRP:COvSB:IP:vs_raise",
        "hero_cat": "CO", "villain_cat": "SB", "ip_oop": "IP", "hero_pos": "CO",
    }, is_real_hu=False), "SRP｜Hero CO 對 SB，IP；翻牌 Hero 面對加注")
