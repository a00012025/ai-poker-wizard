# src/ingest_runner.py
"""Extension-triggered hand-ingest queue runner.

The Chrome extension enqueues `gtow_ingest_requests` rows through the
gtow-sync Edge Function (device-authenticated); this module polls the queue
every 5s, runs the ingest pipeline with the requesting user's own GTOW
refresh token (subprocess env GTOW_REFRESH_TOKEN), streams subprocess progress
back onto the row (which also
serves as the liveness heartbeat), and sends the final result over Telegram.

The daily 05:00 scheduled ingest and /ingest both enqueue a row here instead
of running their own pipeline, so every ingest goes through the same
single-flight claim + per-user token path.
"""
import asyncio
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("poker_bot")

ROOT = Path(__file__).resolve().parent.parent
# Expiry keys on the last heartbeat (progress write), not run start — the
# ingest subprocess prints sweep progress lines, so a healthy long run
# heartbeats continuously while a dead one goes silent.
STALE_RUNNING_MINUTES = 45
_EXPIRE_CHECK_INTERVAL = 60          # seconds between stale-row scans

# One ingest at a time per process; ticks that find the lock held just skip.
_run_lock = asyncio.Lock()
_last_expire_check = 0.0

# Bridges the /ingest command's immediate reply to the poller that later claims
# the row (same process). Keyed by user_id; the one-open-request-per-user unique
# index guarantees at most one live status message per user at a time.
_PENDING_STATUS: dict[int, tuple[int, int]] = {}

# Minimum seconds between live message edits within a stage (Telegram rate
# limits edits; a stage change always edits immediately regardless).
_EDIT_DEBOUNCE_S = 4.0


def register_status_message(user_id: int, chat_id: int, message_id: int) -> None:
    """Record the /ingest reply so the poller edits it in place on claim."""
    _PENDING_STATUS[user_id] = (chat_id, message_id)


# ── Live progress rendering (pure helpers, unit-tested without a bot) ────────

# `x/total` appears in "list scan: 100/241", "detail prep: 100/241",
# "detail sweep: 126/241", "detail write: 126/241", and "list-only sweep: 500/1700";
# the plain "list sweep: 320 new..." has no denominator (streaming paginator).
_FRACTION_RE = re.compile(r"(?:scan|sweep|write):\s*(\d+)\s*/\s*(\d+)")
_COUNT_RE = re.compile(r"list (?:sweep|write):\s*(\d+)\s+new")


def progress_stage_label(stage_label: str, raw_line: str | None = None) -> str:
    """Translate machine progress lines into user-facing sub-stages."""
    line = (raw_line or "").strip()
    if line.startswith("list scan:"):
        return "掃描 GTOW 手牌清單"
    if line.startswith("list write:"):
        return "寫入新手牌清單"
    if line.startswith("list-only sweep:"):
        return "建立零損失摘要"
    if line.startswith("detail prep:"):
        return "準備完整分析清單"
    if line.startswith("detail sweep:"):
        return "下載完整分析"
    if line.startswith("detail write:"):
        return "寫入完整分析到 DB"
    return stage_label


def parse_progress(line: str) -> tuple[int, int] | None:
    """Extract (done, total) from a progress line, or None without a denominator."""
    m = _FRACTION_RE.search(line or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_running_count(line: str) -> int | None:
    """Extract the running 'N new' count from a denominator-less list line."""
    m = _COUNT_RE.search(line or "")
    return int(m.group(1)) if m else None


def _fmt_dur(seconds: float) -> str:
    s = max(0, round(seconds))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def render_bar(done: int, total: int, width: int = 10) -> str:
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, done / total))
    filled = int(round(frac * width))
    filled = max(0, min(width, filled))
    return "▓" * filled + "░" * (width - filled)


def format_eta(done: int, total: int, elapsed_s: float) -> str | None:
    """Rough ETA from the observed rate; None until we have a datapoint."""
    if done <= 0 or elapsed_s <= 0 or total <= done:
        return None
    rate = done / elapsed_s                 # items per second
    if rate <= 0:
        return None
    remaining_s = (total - done) / rate
    if remaining_s < 90:
        return f"剩約 {max(1, round(remaining_s))} 秒"
    return f"剩約 {round(remaining_s / 60)} 分"


def render_status(stage_label: str, parsed: tuple[int, int] | None,
                  elapsed_s: float, stage_times: dict, *,
                  running_count: int | None = None) -> str:
    """Assemble the live status message body."""
    lines = ["⏳ GTOW 手牌同步"]
    if parsed:
        done, total = parsed
        pct = 0 if total <= 0 else round(100 * done / total)
        bar = render_bar(done, total)
        eta = format_eta(done, total, elapsed_s)
        tail = f" · {eta}" if eta else ""
        lines.append(f"{stage_label} · {bar} {pct}%{tail}（{done}/{total}）")
    elif running_count is not None:
        lines.append(f"{stage_label} · 已發現 {running_count:,} 筆新手牌…")
    else:
        lines.append(f"{stage_label}…")
    done_stages = [f"{k} {v}" for k, v in stage_times.items()]
    if done_stages:
        lines.append(f"（{' · '.join(done_stages)}）")
    return "\n".join(lines)


class _LiveStatus:
    """Owns one Telegram message for a run; debounced, error-swallowing edits.

    `now` is injectable so tests drive a fake clock. Formatting is delegated to
    the module-level pure helpers above.
    """

    def __init__(self, bot, chat_id: int, message_id: int, *, now=time.monotonic):
        self.bot = bot
        self.chat_id = chat_id
        self.message_id = message_id
        self._now = now
        self.started_at = now()
        self.detail_started_at: float | None = None
        self.stage_times: dict[str, str] = {}
        self._stage: str | None = None
        self._stage_started_at: float | None = None
        self._last_edit_at = 0.0
        self._last_text: str | None = None

    async def update(self, stage_label: str, raw_line: str | None = None) -> None:
        display_stage = progress_stage_label(stage_label, raw_line)
        parsed = parse_progress(raw_line) if raw_line else None
        running = parse_running_count(raw_line) if raw_line else None
        stage_changed = display_stage != self._stage
        if stage_changed:
            # Book the finished stage's duration so the message shows where the
            # time actually went (instruments the pipeline for later perf work).
            if self._stage is not None and self._stage_started_at is not None:
                self.stage_times[self._stage] = _fmt_dur(
                    self._now() - self._stage_started_at)
            self._stage = display_stage
            self._stage_started_at = self._now()
            self.detail_started_at = None
        # Anchor the ETA clock at the first detail-sweep datapoint.
        if parsed and self.detail_started_at is None:
            self.detail_started_at = self._now()
        anchor = self.detail_started_at or self.started_at
        elapsed = self._now() - anchor
        text = render_status(display_stage, parsed, elapsed, self.stage_times,
                             running_count=running)
        now = self._now()
        if not stage_changed and now - self._last_edit_at < _EDIT_DEBOUNCE_S:
            return
        if text == self._last_text:
            return
        await self._edit(text)

    async def settle(self, final_text: str) -> None:
        await self._edit(final_text)

    async def _edit(self, text: str) -> None:
        try:
            await self.bot.edit_message_text(
                chat_id=self.chat_id, message_id=self.message_id, text=text)
            self._last_edit_at = self._now()
            self._last_text = text
        except Exception as e:  # not-modified, flood-control, deleted msg — never fatal
            logger.debug(f"Live status edit skipped: {e}")


async def _run_script(env: dict, *script_args, on_line=None) -> tuple[int, str]:
    """Run a repo script, streaming stdout lines to on_line; return (rc, tail)."""
    env = dict(env)
    env.pop("POKER_BOT_PROCESS", None)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, *script_args, cwd=str(ROOT), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    lines: list[str] = []
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        line = raw.decode(errors="replace")
        lines.append(line)
        if on_line:
            await on_line(line.rstrip())
    rc = await proc.wait()
    return rc, "".join(lines)[-3000:]


def _summary_line(out: str) -> str | None:
    return next((l for l in out.splitlines() if l.startswith("INGEST")), None)


def _format_summary(summary: str) -> str:
    """Turn the ingest CLI's machine summary into a user-facing explanation.

    Extra counters remain machine-readable in the stored CLI output, while the
    Telegram summary only includes counts that are useful to the user.
    """
    counts = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", summary)}
    required = ("list", "detail", "decisions")
    if not all(key in counts for key in required):
        return summary

    lines = [
        "本次同步結果：",
        f"• 新增手牌：{counts['list']:,}",
        f"• 完整分析：{counts['detail']:,}",
        f"• 決策紀錄：{counts['decisions']:,}",
    ]
    if counts.get("skipped_zeroloss"):
        lines.append(f"• 零損失摘要建檔：{counts['skipped_zeroloss']:,}")
    return "\n".join(lines)


def _tail(out: str) -> str:
    return out.strip().splitlines()[-1] if out.strip() else "(無輸出)"


async def _pass(env: dict, progress, ingest_args: tuple, label: str):
    """One full ingest pass: ledger_ingest → spots → sessions → verify.

    Returns (summary, verify_rc, verify_tail); raises RuntimeError with a
    user-facing message when any stage fails.
    """
    await progress(f"{label}…", stage=label)

    async def heartbeat(line):
        # ledger_ingest prints periodic "list scan/list write/detail sweep"
        # progress and "detail write" during serial DB writes; surface all of
        # them in the toast and refresh the liveness heartbeat.
        if any(marker in line for marker in ("scan:", "sweep:", "write:")):
            await progress(f"{label}：{line.strip()}", stage=label,
                           raw=line.strip())

    rc, out = await _run_script(env, "scripts/ledger_ingest.py", *ingest_args,
                                on_line=heartbeat)
    summary = _summary_line(out)
    if rc != 0 or not summary:
        raise RuntimeError(f"{label}失敗 (rc={rc}): {_tail(out)}")
    for args, stage in ((("scripts/backfill_spots.py",), "補 spot 分類"),
                        (("scripts/ledger_sessions.py", "--rebuild"), "重建 sessions")):
        await progress(f"{stage}…", stage=stage)
        rc, out = await _run_script(env, *args)
        if rc != 0:
            raise RuntimeError(f"{stage}失敗 (rc={rc}): {_tail(out)}")
    await progress("對數中…", stage="對數中")
    rc_v, out_v = await _run_script(env, "scripts/ledger_ingest.py", "--verify")
    if rc_v not in (0, 2):
        raise RuntimeError(f"對數檢查失敗 (rc={rc_v}): {_tail(out_v)}")
    return summary, rc_v, _tail(out_v)


async def run_pipeline(refresh_token: str, progress, *, mode: str = "incremental",
                       allow_full_sweep: bool = True) -> str:
    """incremental ingest → verify; on mismatch escalate to a full sweep.

    mode='full' skips the incremental-first pass and backfills the whole history
    from the ledger epoch directly (the /fullingest path) — backfill already
    covers everything, so there is no escalation branch.

    `progress` is an async callable taking the current stage text. When
    allow_full_sweep is False (a recent full sweep already proved the mismatch
    is unfixable — GTOW-side deletions / pre-epoch hands), the escalation is
    skipped so we don't re-run the ~350-request sweep every day for nothing.
    Returns the final result text; raises RuntimeError with a user-facing
    message on failure.
    """
    env = {**os.environ, "GTOW_REFRESH_TOKEN": refresh_token}
    escalated = False
    guard_skipped = False
    full_import = mode == "full"
    if full_import:
        summary, rc_v, verify_tail = await _pass(
            env, progress, ("--backfill",), "全量攝取中")
    else:
        summary, rc_v, verify_tail = await _pass(
            env, progress, ("--incremental",), "攝取中")
        if rc_v == 2:
            if allow_full_sweep:
                # Hands played outside the 30d incremental window (late uploads
                # of old sessions) only surface in a full list sweep.
                # --backfill's --since defaults to the ledger epoch (2026-03-01).
                escalated = True
                summary, rc_v, verify_tail = await _pass(
                    env, progress, ("--backfill",), "窗外手牌全量補齊中")
            else:
                guard_skipped = True
    result = _format_summary(summary)
    if full_import:
        result += "\n• 範圍：全量匯入（自 ledger epoch）"
    elif escalated:
        result += "\n• 範圍：已執行全量補齊"
    if rc_v == 2:
        # A full sweep structurally cannot repair this (GTOW-side deletions
        # or hands played before the epoch) — report it, don't hard-fail.
        result += f"\n⚠️ 對數仍不符（{verify_tail}）— 可能有 GTOW 端刪除或 epoch 前的手牌"
        if guard_skipped:
            result += "（24h 內已全量補齊仍不符，本次略過全量 sweep）"
    if re.search(r"\blist=0 detail=0\b", summary):
        result += ("\n（歷史手牌都已在資料庫，沒有新增）" if full_import
                   else "\n（沒有新手牌 — 若剛上傳，GTOW 可能還在處理，稍後再點一次）")
    return result


async def enqueue_request(pool, user_id: int, mode: str = "incremental") -> bool:
    """Enqueue an ingest request; returns True if an open one already existed.

    Atomic via the partial unique index (one pending/running row per user,
    mode-agnostic) — the targetless ON CONFLICT catches its violation without a
    check-then-insert race, so a full request is blocked while an incremental
    one is open and vice versa.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO gtow_ingest_requests (user_id, mode) VALUES ($1, $2) "
            "ON CONFLICT DO NOTHING RETURNING id", user_id, mode)
    return row is None


async def _recent_permanent_mismatch(pool, user_id: int) -> bool:
    """True if this user has a done request in the last 24h whose result shows
    a full sweep ran AND still couldn't reconcile — the signal that another
    full sweep would be wasted effort (deferred #9)."""
    async with pool.acquire() as conn:
        return bool(await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM gtow_ingest_requests "
            "  WHERE user_id=$1 AND status='done' "
            "    AND finished_at > now() - interval '24 hours' "
            "    AND result LIKE '%全量補齊%' AND result LIKE '%對數仍不符%')",
            user_id))


async def _set(pool, req_id, **fields):
    sets = ", ".join(f"{k}=${i+2}" for i, k in enumerate(fields))
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE gtow_ingest_requests SET {sets} WHERE id=$1",
            req_id, *fields.values())


async def _claim_next(pool):
    async with pool.acquire() as conn:
        async with conn.transaction():
            return await conn.fetchrow(
                "UPDATE gtow_ingest_requests SET status='running', "
                "started_at=now(), heartbeat_at=now() "
                "WHERE id = (SELECT id FROM gtow_ingest_requests WHERE status='pending' "
                "            ORDER BY requested_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id, mode")


async def _expire_stale(pool, bot):
    """Mark heartbeat-silent running rows as error and tell their user."""
    global _last_expire_check
    if time.monotonic() - _last_expire_check < _EXPIRE_CHECK_INTERVAL:
        return
    _last_expire_check = time.monotonic()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "UPDATE gtow_ingest_requests SET status='error', finished_at=now(), "
            "result='中斷（bot 重啟或逾時），請重新觸發' "
            "WHERE status='running' "
            f"  AND coalesce(heartbeat_at, started_at) < now() - interval '{STALE_RUNNING_MINUTES} minutes' "
            "RETURNING user_id")
    for r in rows:
        try:
            await bot.send_message(
                r["user_id"], "❌ GTOW 手牌同步中斷（bot 重啟或逾時），請重新觸發")
        except Exception as e:
            logger.warning(f"Ingest expiry notify failed for user {r['user_id']}: {e}")


async def _finish(pool, bot, req_id, user_id, *, ok: bool, text: str) -> bool:
    """Record the terminal state; send a fresh notification unless idle.

    Returns True if a Telegram message was sent (so the caller can settle the
    live progress bar to a matching terminal state)."""
    await _set(pool, req_id, status="done" if ok else "error", result=text,
               progress=None, finished_at=datetime.now(timezone.utc))
    # An idle sync is still recorded as successfully completed, but does not
    # need a Telegram notification. Errors must always remain visible.
    if ok and re.search(r"(?m)^• 新增手牌：0$", text):
        return False
    try:
        icon = "✅" if ok else "❌"
        await bot.send_message(user_id, f"{icon} GTOW 手牌同步\n{text}")
    except Exception as e:
        logger.warning(f"Ingest notify failed for user {user_id}: {e}")
    return True


async def _send_session_review(pool, bot, user_id, application=None):
    """After a successful sync, auto-append the latest online session's 復盤
    digest — but only when there's something worth reviewing (skip clean/empty
    sessions, §7-11 依從). Best-effort: never blocks or fails the sync result.
    """
    try:
        from session_review import compute, render_tg, resolve_session, should_auto_send
        session = await resolve_session(pool, None)
        if not session:
            return
        data = await compute(pool, session)
        if not should_auto_send(data):
            return
        if application is not None:   # warm the callback cache (recompute fallback exists)
            from session_review import session_callback_key
            cache = application.bot_data.setdefault("srev", {})
            cache[data["session_id"]] = data
            cache[session_callback_key(data)] = data
        out = render_tg(data)
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        kb = [[InlineKeyboardButton(b["text"], url=b.get("url"),
                                    callback_data=b.get("callback_data"))
               for b in row] for row in out["buttons"]]
        await bot.send_message(
            user_id, out["html"], parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(kb) if kb else None)
    except Exception as e:
        logger.warning(f"Auto session-review failed for user {user_id}: {e}")


async def process_next(pool, bot, db, application=None) -> bool:
    """Claim and run at most one queued request. Returns True if one ran."""
    from ledger_service import resolve_owner_chat_id

    await _expire_stale(pool, bot)
    row = await _claim_next(pool)
    if not row:
        return False
    req_id, user_id = row["id"], row["user_id"]
    mode = row["mode"] if "mode" in row else "incremental"
    logger.info(f"Ingest request {req_id} claimed (user {user_id}, mode={mode})")

    owner = await resolve_owner_chat_id(pool)
    if user_id != owner:
        await _finish(pool, bot, req_id, user_id, ok=False,
                      text="ledger 目前僅支援 owner 帳號")
        return True
    token = await db.get_user_gto_token(user_id)
    if not token:
        await _finish(pool, bot, req_id, user_id, ok=False,
                      text="找不到有效的 GTOW token，請在 GTOW 頁面重新登入後再點一次")
        return True

    live = await _init_live_status(bot, user_id)

    async def progress(text, stage=None, raw=None):
        await _set(pool, req_id, progress=text,
                   heartbeat_at=datetime.now(timezone.utc))
        if live is not None:
            await live.update(stage or text, raw)

    # The 24h "already fully swept" guard only gates the *incremental* path's
    # escalation; a full import always runs the backfill by definition.
    allow_full_sweep = (mode == "full"
                        or not await _recent_permanent_mismatch(pool, user_id))
    try:
        result = await run_pipeline(token, progress, mode=mode,
                                    allow_full_sweep=allow_full_sweep)
        notified = await _finish(pool, bot, req_id, user_id, ok=True, text=result)
        if live is not None:
            await live.settle("✅ 同步完成 · 結果見下方 ↓" if notified
                              else "✅ 已是最新，沒有新手牌進來")
        # Skip the auto-review when the sync added nothing new — re-pushing the
        # same session's digest on every idle sync tap is noise (§7-11 依從).
        if "沒有新手牌" not in result:
            await _send_session_review(pool, bot, user_id, application)
    except RuntimeError as e:
        await _finish(pool, bot, req_id, user_id, ok=False, text=str(e))
        if live is not None:
            await live.settle("❌ 同步失敗 · 詳情見下方 ↓")
    except Exception as e:
        logger.error(f"Ingest request {req_id} crashed: {e}", exc_info=True)
        await _finish(pool, bot, req_id, user_id, ok=False, text=f"內部錯誤：{e}")
        if live is not None:
            await live.settle("❌ 同步失敗 · 詳情見下方 ↓")
    return True


async def _init_live_status(bot, user_id: int) -> "_LiveStatus | None":
    """Build the live status message: reuse the /ingest reply if the command
    registered one, else send a fresh message (extension ♠-sync path). Popped so
    a stale message id can never leak into a later run. Best-effort — a failure
    here just means no live bar, never a failed sync."""
    pending = _PENDING_STATUS.pop(user_id, None)
    try:
        if pending:
            return _LiveStatus(bot, pending[0], pending[1])
        msg = await bot.send_message(user_id, "⏳ 開始同步…")
        return _LiveStatus(bot, msg.chat_id, msg.message_id)
    except Exception as e:
        logger.warning(f"Live status init failed for user {user_id}: {e}")
        return None


async def poll_job(context, db):
    """JobQueue every 5s: run at most one queued ingest request."""
    if not db.pool or _run_lock.locked():
        return
    async with _run_lock:
        try:
            await process_next(db.pool, context.bot, db,
                               application=getattr(context, "application", None))
        except Exception as e:
            logger.error(f"Ingest poll job failed: {e}", exc_info=True)
