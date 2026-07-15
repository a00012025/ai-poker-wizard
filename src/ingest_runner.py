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

    Older deployments called already-known hands ``skipped``.  Accept both
    names so queued/running jobs from either CLI version cannot make normal
    deduplication look like a failure.
    """
    counts = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", summary)}
    required = ("list", "detail", "decisions")
    if not all(key in counts for key in required):
        return summary

    known = counts.get("known", counts.get("skipped", 0))
    lines = [
        "本次同步結果：",
        f"• 新增手牌：{counts['list']:,}",
        f"• 完整分析：{counts['detail']:,}",
        f"• 決策紀錄：{counts['decisions']:,}",
        f"• 已在資料庫：{known:,}（不是失敗）",
    ]
    if counts.get("skipped_zeroloss"):
        lines.append(f"• 零損失摘要建檔：{counts['skipped_zeroloss']:,}")
    if counts.get("reconstruct_fallback"):
        lines.append(
            f"• 摘要不足、改抓完整分析：{counts['reconstruct_fallback']:,}")
    return "\n".join(lines)


def _tail(out: str) -> str:
    return out.strip().splitlines()[-1] if out.strip() else "(無輸出)"


async def _pass(env: dict, progress, ingest_args: tuple, label: str):
    """One full ingest pass: ledger_ingest → spots → sessions → verify.

    Returns (summary, verify_rc, verify_tail); raises RuntimeError with a
    user-facing message when any stage fails.
    """
    await progress(f"{label}…")

    async def heartbeat(line):
        # ledger_ingest prints periodic "  list/detail sweep: ..." progress;
        # surface it in the toast and refresh the row's liveness heartbeat.
        if "sweep:" in line:
            await progress(f"{label}：{line.strip()}")

    rc, out = await _run_script(env, "scripts/ledger_ingest.py", *ingest_args,
                                on_line=heartbeat)
    summary = _summary_line(out)
    if rc != 0 or not summary:
        raise RuntimeError(f"{label}失敗 (rc={rc}): {_tail(out)}")
    for args, stage in ((("scripts/backfill_spots.py",), "補 spot 分類"),
                        (("scripts/ledger_sessions.py", "--rebuild"), "重建 sessions")):
        await progress(f"{stage}…")
        rc, out = await _run_script(env, *args)
        if rc != 0:
            raise RuntimeError(f"{stage}失敗 (rc={rc}): {_tail(out)}")
    await progress("對數中…")
    rc_v, out_v = await _run_script(env, "scripts/ledger_ingest.py", "--verify")
    if rc_v not in (0, 2):
        raise RuntimeError(f"對數檢查失敗 (rc={rc_v}): {_tail(out_v)}")
    return summary, rc_v, _tail(out_v)


async def run_pipeline(refresh_token: str, progress, *, allow_full_sweep: bool = True) -> str:
    """incremental ingest → verify; on mismatch escalate to a full sweep.

    `progress` is an async callable taking the current stage text. When
    allow_full_sweep is False (a recent full sweep already proved the mismatch
    is unfixable — GTOW-side deletions / pre-epoch hands), the escalation is
    skipped so we don't re-run the ~350-request sweep every day for nothing.
    Returns the final result text; raises RuntimeError with a user-facing
    message on failure.
    """
    env = {**os.environ, "GTOW_REFRESH_TOKEN": refresh_token}
    summary, rc_v, verify_tail = await _pass(env, progress, ("--incremental",), "攝取中")
    escalated = False
    guard_skipped = False
    if rc_v == 2:
        if allow_full_sweep:
            # Hands played outside the 30d incremental window (late uploads of
            # old sessions) only surface in a full list sweep. --backfill's
            # --since defaults to the ledger epoch (2026-03-01).
            escalated = True
            summary, rc_v, verify_tail = await _pass(
                env, progress, ("--backfill",), "窗外手牌全量補齊中")
        else:
            guard_skipped = True
    result = _format_summary(summary)
    if escalated:
        result += "\n• 範圍：已執行全量補齊"
    if rc_v == 2:
        # A full sweep structurally cannot repair this (GTOW-side deletions
        # or hands played before the epoch) — report it, don't hard-fail.
        result += f"\n⚠️ 對數仍不符（{verify_tail}）— 可能有 GTOW 端刪除或 epoch 前的手牌"
        if guard_skipped:
            result += "（24h 內已全量補齊仍不符，本次略過全量 sweep）"
    if re.search(r"\blist=0 detail=0\b", summary):
        result += "\n（沒有新手牌 — 若剛上傳，GTOW 可能還在處理，稍後再點一次）"
    return result


async def enqueue_request(pool, user_id: int) -> bool:
    """Enqueue an ingest request; returns True if an open one already existed.

    Atomic via the partial unique index (one pending/running row per user) —
    the targetless ON CONFLICT catches its violation without a check-then-
    insert race.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO gtow_ingest_requests (user_id) VALUES ($1) "
            "ON CONFLICT DO NOTHING RETURNING id", user_id)
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
                "RETURNING id, user_id")


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


async def _finish(pool, bot, req_id, user_id, *, ok: bool, text: str):
    await _set(pool, req_id, status="done" if ok else "error", result=text,
               progress=None, finished_at=datetime.now(timezone.utc))
    try:
        icon = "✅" if ok else "❌"
        await bot.send_message(user_id, f"{icon} GTOW 手牌同步\n{text}")
    except Exception as e:
        logger.warning(f"Ingest notify failed for user {user_id}: {e}")


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
            application.bot_data.setdefault("srev", {})[data["session_id"]] = data
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
    logger.info(f"Ingest request {req_id} claimed (user {user_id})")

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

    async def progress(text):
        await _set(pool, req_id, progress=text,
                   heartbeat_at=datetime.now(timezone.utc))

    allow_full_sweep = not await _recent_permanent_mismatch(pool, user_id)
    try:
        result = await run_pipeline(token, progress, allow_full_sweep=allow_full_sweep)
        await _finish(pool, bot, req_id, user_id, ok=True, text=result)
        # Skip the auto-review when the sync added nothing new — re-pushing the
        # same session's digest on every idle sync tap is noise (§7-11 依從).
        if "沒有新手牌" not in result:
            await _send_session_review(pool, bot, user_id, application)
    except RuntimeError as e:
        await _finish(pool, bot, req_id, user_id, ok=False, text=str(e))
    except Exception as e:
        logger.error(f"Ingest request {req_id} crashed: {e}", exc_info=True)
        await _finish(pool, bot, req_id, user_id, ok=False, text=f"內部錯誤：{e}")
    return True


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
