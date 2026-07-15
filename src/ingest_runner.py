# src/ingest_runner.py
"""Extension-triggered hand-ingest queue runner.

The Chrome extension enqueues `gtow_ingest_requests` rows through the
gtow-sync Edge Function (device-authenticated); this module polls the queue
every 5s, runs the ingest pipeline with the requesting user's own GTOW
refresh token (subprocess env GTOW_REFRESH_TOKEN — `.tokens.json` is never
read or written), streams stage progress back onto the row for the
extension's toast, and sends the final result over Telegram.

The daily 05:00 scheduled ingest reuses run_pipeline() with the owner's
stored token, so it survives FORCED_LOGOUT as long as the extension keeps
users.gto_refresh_token fresh.
"""
import asyncio
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("poker_bot")

ROOT = Path(__file__).resolve().parent.parent
STALE_RUNNING_MINUTES = 45

# One ingest at a time per process; ticks that find the lock held just skip.
_run_lock = asyncio.Lock()


async def _run_script(env: dict, *script_args) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, *script_args, cwd=str(ROOT), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")[-3000:]


def _summary_line(out: str) -> str | None:
    return next((l for l in out.splitlines() if l.startswith("INGEST")), None)


async def run_pipeline(refresh_token: str, progress) -> str:
    """incremental ingest → spots → sessions → verify (mismatch: full sweep).

    `progress` is an async callable taking the current stage text. Returns
    the final result text; raises RuntimeError with a user-facing message on
    failure.
    """
    env = {**os.environ, "GTOW_REFRESH_TOKEN": refresh_token}
    await progress("攝取中…")
    rc, out = await _run_script(env, "scripts/ledger_ingest.py", "--incremental")
    summary = _summary_line(out)
    if rc != 0 or not summary:
        tail = out.strip().splitlines()[-1] if out.strip() else "(無輸出)"
        raise RuntimeError(f"攝取失敗 (rc={rc}): {tail}")
    await progress("補 spot 分類…")
    await _run_script(env, "scripts/backfill_spots.py")
    await progress("重建 sessions…")
    await _run_script(env, "scripts/ledger_sessions.py", "--rebuild")
    await progress("對數中…")
    rc_v, _ = await _run_script(env, "scripts/ledger_ingest.py", "--verify")
    escalated = False
    if rc_v == 2:
        # Hands played outside the 30d incremental window (late uploads of
        # old sessions) only surface in a full list sweep.
        escalated = True
        await progress("窗外手牌補齊中（全量 sweep）…")
        rc, out = await _run_script(
            env, "scripts/ledger_ingest.py", "--backfill", "--since", "2026-03-01")
        summary = _summary_line(out) or summary
        if rc != 0:
            tail = out.strip().splitlines()[-1] if out.strip() else "(無輸出)"
            raise RuntimeError(f"全量補齊失敗 (rc={rc}): {tail}")
        await progress("補 spot 分類…")
        await _run_script(env, "scripts/backfill_spots.py")
        await progress("重建 sessions…")
        await _run_script(env, "scripts/ledger_sessions.py", "--rebuild")
        rc_v, out_v = await _run_script(env, "scripts/ledger_ingest.py", "--verify")
        if rc_v == 2:
            raise RuntimeError(f"全量補齊後仍對數不符：{out_v.strip().splitlines()[-1]}")
    result = summary + (" · 全量補齊" if escalated else "")
    if re.search(r"\blist=0 detail=0\b", summary):
        result += "\n（沒有新手牌 — 若剛上傳，GTOW 可能還在處理，稍後再點一次）"
    return result


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
                "UPDATE gtow_ingest_requests SET status='running', started_at=now() "
                "WHERE id = (SELECT id FROM gtow_ingest_requests WHERE status='pending' "
                "            ORDER BY requested_at LIMIT 1 FOR UPDATE SKIP LOCKED) "
                "RETURNING id, user_id")


async def _expire_stale(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE gtow_ingest_requests SET status='error', finished_at=now(), "
            "result='中斷（bot 重啟或逾時），請重新觸發' "
            "WHERE status='running' AND started_at < now() - interval '%s minutes'"
            % STALE_RUNNING_MINUTES)


async def _finish(pool, bot, req_id, user_id, *, ok: bool, text: str):
    await _set(pool, req_id, status="done" if ok else "error", result=text,
               progress=None, finished_at=datetime.now(timezone.utc))
    try:
        icon = "✅" if ok else "❌"
        await bot.send_message(user_id, f"{icon} GTOW 手牌同步\n{text}")
    except Exception as e:
        logger.warning(f"Ingest notify failed for user {user_id}: {e}")


async def process_next(pool, bot, db) -> bool:
    """Claim and run at most one queued request. Returns True if one ran."""
    from ledger_service import resolve_owner_chat_id

    await _expire_stale(pool)
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
        await _set(pool, req_id, progress=text)

    try:
        result = await run_pipeline(token, progress)
        await _finish(pool, bot, req_id, user_id, ok=True, text=result)
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
            await process_next(db.pool, context.bot, db)
        except Exception as e:
            logger.error(f"Ingest poll job failed: {e}", exc_info=True)
