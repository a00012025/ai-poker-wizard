"""Live ingest progress-bar rendering + debounce tests.

Pure formatting helpers and the _LiveStatus debounce/no-op/error handling are
tested without a bot or network (fake clock + fake bot recording edits).
"""

import asyncio

from regression_tests.harness import assert_eq, assert_in, assert_true, test


# ── Pure formatters ────────────────────────────────────────────────────────

@test
def test_parse_progress_extracts_done_total():
    from src.ingest_runner import parse_progress
    assert_eq(parse_progress("  detail sweep: 126/241"), (126, 241),
              "detail sweep x/total")
    assert_eq(parse_progress("  list-only sweep: 500/1700"), (500, 1700),
              "list-only sweep x/total")


@test
def test_parse_progress_none_without_denominator():
    from src.ingest_runner import parse_progress
    assert_eq(parse_progress("  list sweep: 320 new..."), None,
              "list sweep has no denominator")
    assert_eq(parse_progress("攝取中…"), None, "stage label is not progress")
    assert_eq(parse_progress("  detail sweep: skipped 4 hands"), None,
              "non-fraction sweep line")


@test
def test_render_bar_boundaries():
    from src.ingest_runner import render_bar
    assert_eq(render_bar(0, 10, width=10), "░" * 10, "0%")
    assert_eq(render_bar(10, 10, width=10), "▓" * 10, "100%")
    assert_eq(render_bar(5, 10, width=10), "▓" * 5 + "░" * 5, "50%")
    # never overflow / underflow the width
    assert_eq(len(render_bar(999, 10, width=8)), 8, "clamped high")
    assert_eq(len(render_bar(0, 0, width=8)), 8, "zero total is safe")


@test
def test_format_eta_none_until_data():
    from src.ingest_runner import format_eta
    assert_eq(format_eta(0, 241, 0.0), None, "no elapsed, no rate")
    assert_eq(format_eta(0, 241, 5.0), None, "nothing done yet")


@test
def test_format_eta_seconds_and_minutes():
    from src.ingest_runner import format_eta
    # 100 done in 50s -> 2/s; 100 remaining -> ~50s
    eta = format_eta(100, 200, 50.0)
    assert_true(eta is not None and "秒" in eta, f"seconds bucket: {eta}")
    # 20 done in 60s -> 1/3 per s; 220 remaining -> ~660s -> ~11 分
    eta = format_eta(20, 240, 60.0)
    assert_true(eta is not None and "分" in eta, f"minutes bucket: {eta}")


@test
def test_render_status_denominatorless_stage_has_no_bar():
    from src.ingest_runner import render_status
    text = render_status("攝取中", None, 3.0, {}, running_count=320)
    assert_in("攝取中", text, "stage label present")
    assert_true("▓" not in text and "░" not in text, "no bar without denominator")
    assert_in("320", text, "running count shown")


@test
def test_render_status_detail_stage_has_bar_and_eta():
    from src.ingest_runner import render_status
    text = render_status("攝取中", (120, 240), 60.0, {})
    assert_true("▓" in text and "░" in text, "bar present")
    assert_in("50%", text, "percentage")
    assert_in("120/240", text, "fraction")
    assert_true("剩約" in text, "eta present")


# ── _LiveStatus debounce / no-op / error handling ──────────────────────────

class _FakeBot:
    def __init__(self, fail_not_modified=False):
        self.edits = []
        self.fail_not_modified = fail_not_modified

    async def edit_message_text(self, *, chat_id, message_id, text, **kw):
        if self.fail_not_modified:
            from telegram.error import BadRequest
            raise BadRequest("Message is not modified")
        self.edits.append(text)


@test
def test_live_status_debounces_rapid_same_stage_updates():
    from src.ingest_runner import _LiveStatus

    async def run():
        clock = {"t": 1000.0}
        bot = _FakeBot()
        live = _LiveStatus(bot, chat_id=1, message_id=2,
                           now=lambda: clock["t"])
        await live.update("攝取中", "  detail sweep: 10/240")   # first edit always fires
        await live.update("攝取中", "  detail sweep: 11/240")   # <4s later, same stage
        clock["t"] += 5.0
        await live.update("攝取中", "  detail sweep: 60/240")   # now allowed
        return bot.edits

    edits = asyncio.run(run())
    assert_eq(len(edits), 2, f"debounced to 2 edits, got {len(edits)}: {edits}")


@test
def test_live_status_stage_change_bypasses_debounce():
    from src.ingest_runner import _LiveStatus

    async def run():
        clock = {"t": 1000.0}
        bot = _FakeBot()
        live = _LiveStatus(bot, chat_id=1, message_id=2, now=lambda: clock["t"])
        await live.update("攝取中", "  detail sweep: 10/240")
        # immediate stage change must not be swallowed by the 4s debounce
        await live.update("補 spot 分類", None)
        return bot.edits

    edits = asyncio.run(run())
    assert_eq(len(edits), 2, f"stage change bypasses debounce: {edits}")


@test
def test_live_status_swallows_not_modified():
    from src.ingest_runner import _LiveStatus

    async def run():
        bot = _FakeBot(fail_not_modified=True)
        live = _LiveStatus(bot, chat_id=1, message_id=2, now=lambda: 1000.0)
        # must not raise
        await live.update("攝取中", "  detail sweep: 10/240")
        await live.settle("✅ 完成")
        return True

    ok = asyncio.run(run())
    assert_true(ok, "BadRequest not-modified swallowed")


# ── process_next wiring (fresh-send path, bar edit, settle) ─────────────────

class _FakeSentMsg:
    def __init__(self, chat_id, message_id):
        self.chat_id = chat_id
        self.message_id = message_id


class _RecordingBot:
    def __init__(self):
        self.sent = []
        self.edits = []

    async def send_message(self, chat_id, text=None, **kw):
        self.sent.append(text)
        return _FakeSentMsg(chat_id, 999)

    async def edit_message_text(self, *, chat_id, message_id, text, **kw):
        self.edits.append(text)


class _FakeDB:
    pool = object()

    async def get_user_gto_token(self, user_id):
        return "tok"


@test
def test_process_next_sends_live_bar_and_settles():
    """Extension-path run: no pre-registered message, so the runner sends its
    own, edits it with a real bar during the detail sweep, then settles it."""
    import ledger_service
    import src.ingest_runner as ir

    OWNER = 556028753
    saved = {name: getattr(ir, name) for name in
             ("_expire_stale", "_claim_next", "_recent_permanent_mismatch",
              "_set", "_send_session_review", "run_pipeline", "_EDIT_DEBOUNCE_S")}
    saved_resolve = ledger_service.resolve_owner_chat_id

    async def _run():
        async def fake_claim(pool):
            return {"id": "req-1", "user_id": OWNER}

        async def fake_run_pipeline(token, progress, *, mode="incremental",
                                    allow_full_sweep=True):
            await progress("攝取中…", stage="攝取中")
            await progress("攝取中：detail sweep: 120/240", stage="攝取中",
                           raw="detail sweep: 120/240")
            await progress("重建 sessions…", stage="重建 sessions")
            return ("本次同步結果：\n• 新增手牌：413\n• 完整分析：241\n"
                    "• 決策紀錄：490")

        async def anoop(*a, **k):
            return None

        async def afalse(*a, **k):
            return False

        async def fake_resolve(pool):
            return OWNER

        ir._expire_stale = anoop
        ir._claim_next = fake_claim
        ir._recent_permanent_mismatch = afalse
        ir._set = anoop
        ir._send_session_review = anoop
        ir.run_pipeline = fake_run_pipeline
        ir._EDIT_DEBOUNCE_S = 0.0            # fire every edit in fast test
        ledger_service.resolve_owner_chat_id = fake_resolve

        bot = _RecordingBot()
        ran = await ir.process_next(ir_pool := object(), bot, _FakeDB())
        return ran, bot

    try:
        ran, bot = asyncio.run(_run())
    finally:
        for name, val in saved.items():
            setattr(ir, name, val)
        ledger_service.resolve_owner_chat_id = saved_resolve

    assert_true(ran, "process_next reports it ran")
    # Runner sent its own live message (extension path) + the final result.
    assert_true(any("開始同步" in s for s in bot.sent), f"opening msg: {bot.sent}")
    assert_true(any("413" in s for s in bot.sent), f"final result msg: {bot.sent}")
    # The detail-sweep edit rendered a real bar + percentage.
    assert_true(any("▓" in e and "50%" in e for e in bot.edits),
                f"bar edit present: {bot.edits}")
    # The bar was settled to a terminal state pointing at the result.
    assert_true(any("結果見下方" in e for e in bot.edits),
                f"settle edit present: {bot.edits}")


# ── full-history import mode ────────────────────────────────────────────────

@test
def test_run_pipeline_full_mode_backfills_directly():
    """mode='full' runs --backfill straight away (no --incremental first) and
    marks the result as a full import."""
    import src.ingest_runner as ir
    from regression_tests.test_ledger import _fake_ingest_env

    fake_run, calls = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (0, "VERIFY OK api=413 db=413")),
        (lambda a, c: "--backfill" in a,
         (0, "INGEST list=413 detail=241 decisions=490 skipped=170")),
    ])

    async def progress(t, **kw):
        pass

    async def _run():
        orig = ir._run_script
        ir._run_script = fake_run
        try:
            return await ir.run_pipeline("tok-r", progress, mode="full")
        finally:
            ir._run_script = orig

    result = asyncio.run(_run())
    assert_in("新增手牌：413", result)
    assert_in("全量匯入", result)
    assert_true(not any("--incremental" in a for a in calls),
                f"full mode must not run --incremental: {calls}")
    assert_true(any("--backfill" in a for a in calls), "backfill ran")


@test
def test_run_pipeline_full_mode_no_new_hands_message():
    """A full import that finds nothing new says so (not the incremental
    'GTOW still processing' hint)."""
    import src.ingest_runner as ir
    from regression_tests.test_ledger import _fake_ingest_env

    fake_run, _ = _fake_ingest_env([
        (lambda a, c: "--verify" in a, (0, "VERIFY OK api=413 db=413")),
        (lambda a, c: "--backfill" in a,
         (0, "INGEST list=0 detail=0 decisions=0 skipped=413")),
    ])

    async def progress(t, **kw):
        pass

    async def _run():
        orig = ir._run_script
        ir._run_script = fake_run
        try:
            return await ir.run_pipeline("tok-r", progress, mode="full")
        finally:
            ir._run_script = orig

    result = asyncio.run(_run())
    assert_in("歷史手牌都已在資料庫", result)
    assert_true("稍後再點一次" not in result, "no incremental-only hint in full mode")


@test
def test_process_next_threads_mode_to_pipeline():
    """process_next passes the claimed row's mode through to run_pipeline, and a
    full import bypasses the 24h already-swept guard."""
    import ledger_service
    import src.ingest_runner as ir

    OWNER = 556028753
    saved = {name: getattr(ir, name) for name in
             ("_expire_stale", "_claim_next", "_recent_permanent_mismatch",
              "_set", "_send_session_review", "_init_live_status", "run_pipeline")}
    saved_resolve = ledger_service.resolve_owner_chat_id
    seen = {}

    async def _run():
        async def fake_claim(pool):
            return {"id": "req-1", "user_id": OWNER, "mode": "full"}

        async def fake_run_pipeline(token, progress, *, mode="incremental",
                                    allow_full_sweep=True):
            seen["mode"] = mode
            seen["allow_full_sweep"] = allow_full_sweep
            return "本次同步結果：\n• 新增手牌：413\n• 完整分析：241\n• 決策紀錄：490"

        async def anoop(*a, **k):
            return None

        async def afalse(*a, **k):
            return False

        async def fake_resolve(pool):
            return OWNER

        ir._expire_stale = anoop
        ir._claim_next = fake_claim
        ir._recent_permanent_mismatch = afalse
        ir._set = anoop
        ir._send_session_review = anoop
        ir._init_live_status = anoop        # no live bar needed for this assertion
        ir.run_pipeline = fake_run_pipeline
        ledger_service.resolve_owner_chat_id = fake_resolve
        return await ir.process_next(object(), _RecordingBot(), _FakeDB())

    try:
        asyncio.run(_run())
    finally:
        for name, val in saved.items():
            setattr(ir, name, val)
        ledger_service.resolve_owner_chat_id = saved_resolve

    assert_eq(seen.get("mode"), "full", "mode threaded to run_pipeline")
    assert_true(seen.get("allow_full_sweep") is True, "full bypasses 24h guard")
