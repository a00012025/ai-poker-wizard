# src/main_gemini.py
"""Entry point using Gemini API (fast, no Claude CLI subprocess)."""
import logging
import os
import sys
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.database import Database
from src.gemini_session import GeminiSessionManager
from src.telegram_bot.bot import PokerWizardBot

logger = logging.getLogger("poker_bot")

db = Database()

# Weekly report timezone (Taiwan = UTC+8)
TZ_TAIPEI = ZoneInfo("Asia/Taipei")


async def _weekly_report_job(context):
    """PTB JobQueue callback: generate and send weekly reports."""
    try:
        if not db.pool:
            logger.warning("Weekly report: DB pool not available, skipping")
            return
        from weekly_report import send_weekly_reports
        sent = await send_weekly_reports(db.pool, context.bot)
        logger.info(f"Weekly report job completed: {sent} reports sent")
    except Exception as e:
        logger.error(f"Weekly report job failed: {e}")


async def _run_script(*script_args) -> tuple[int, str]:
    """Run a repo script as a subprocess from the repo root; return (rc, tail)."""
    import asyncio
    proc = await asyncio.create_subprocess_exec(
        sys.executable, *script_args,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace")[-3000:]


async def _daily_ledger_ingest_job(context):
    """Daily 05:00 Taipei: incremental ingest (30d re-sweep) + sessions + verify."""
    from ledger_service import resolve_owner_chat_id
    try:
        rc, out = await _run_script("scripts/ledger_ingest.py", "--incremental")
        logger.info(f"Daily ingest rc={rc}: {out.splitlines()[-1] if out.strip() else ''}")
        await _run_script("scripts/backfill_spots.py")
        await _run_script("scripts/ledger_sessions.py", "--rebuild")
        rc_v, out_v = await _run_script("scripts/ledger_ingest.py", "--verify")
        if rc_v == 2 and db.pool:
            owner = await resolve_owner_chat_id(db.pool)
            if owner:
                await context.bot.send_message(owner, f"⚠️ Ledger 對數不符\n{out_v.strip()}")
    except Exception as e:
        logger.error(f"Daily ledger ingest job failed: {e}")


async def _weekly_scorecard_job(context):
    """Sunday 21:00 Taipei: build training-plan scorecard + push to owner."""
    import json
    from ledger_service import resolve_owner_chat_id
    try:
        rc, out = await _run_script("scripts/scorecard.py", "--weekly")
        if rc != 0:
            logger.error(f"Scorecard failed: {out}"); return
        if not db.pool:
            return
        owner = await resolve_owner_chat_id(db.pool)
        if not owner:
            logger.warning("Scorecard: no owner chat id"); return
        async with db.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT week, data_json FROM scorecards ORDER BY created_at DESC LIMIT 1")
        data = json.loads(row["data_json"]) if isinstance(row["data_json"], str) else row["data_json"]
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from scorecard import weekly_tg_html
        msg = weekly_tg_html(row["week"], data)
        await context.bot.send_message(
            owner, msg, parse_mode="HTML", disable_web_page_preview=True)
        path = Path(__file__).resolve().parent.parent / "data/scorecards" / f"{row['week']}.html"
        if path.exists():
            with open(path, "rb") as fh:
                await context.bot.send_document(owner, document=fh,
                                                filename=f"training-plan-{row['week']}.html")
        async with db.pool.acquire() as conn:
            await conn.execute("UPDATE scorecards SET pushed_at=NOW() WHERE week=$1", row["week"])
    except Exception as e:
        logger.error(f"Weekly scorecard job failed: {e}")


async def post_init(application):
    """Called after Application.initialize() — connect DB + preload OCR."""
    dsn = os.getenv("SUPABASE_CONN")
    if dsn:
        await db.connect(dsn)
        await db.check_tables()
        logger.info("Database ready")
    else:
        logger.warning("SUPABASE_CONN not set — running without database")

    # Preload EasyOCR model + CardClassifier to avoid cold start on first image
    if os.getenv("OCR_ENABLED", "false").lower() in ("true", "1", "yes"):
        try:
            from ocr.ocr_utils import _get_reader
            _get_reader()
            logger.info("OCR model preloaded")
        except Exception as e:
            logger.warning(f"OCR model preload failed: {e}")
        try:
            from ocr.classifier.infer import CardClassifier
            CardClassifier()._warm()
            logger.info("CardClassifier preloaded")
        except Exception as e:
            logger.warning(f"CardClassifier preload failed: {e}")

    # Schedule weekly leak report (Sunday 10:00 AM Taipei time).
    # PTB v20+ uses cron-style day numbering: 0=Sun, 1=Mon, ..., 6=Sat.
    if db.pool and application.job_queue:
        application.job_queue.run_daily(
            _weekly_report_job,
            time=dt_time(hour=10, minute=0, tzinfo=TZ_TAIPEI),
            days=(0,),
            name="weekly_leak_report",
        )
        logger.info("Weekly leak report job scheduled (Sunday 10:00 AM Taipei)")

        # Phase 1 ledger loop: daily incremental ingest + Sunday training-plan push.
        application.job_queue.run_daily(
            _daily_ledger_ingest_job,
            time=dt_time(hour=5, minute=0, tzinfo=TZ_TAIPEI),
            days=tuple(range(7)), name="daily_ledger_ingest")
        application.job_queue.run_daily(
            _weekly_scorecard_job,
            time=dt_time(hour=21, minute=0, tzinfo=TZ_TAIPEI),
            days=(0,), name="weekly_scorecard")
        logger.info("Ledger ingest (daily 05:00) + scorecard (Sun 21:00) jobs scheduled")


async def post_shutdown(application):
    """Called after Application.shutdown() — close DB pool."""
    await db.close()


def main():
    print("AI Poker Wizard (Gemini) starting...")

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        print("ERROR: BOT_TOKEN not set")
        return

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return

    session_manager = GeminiSessionManager(db=db)
    bot = PokerWizardBot(token=bot_token, session_manager=session_manager, db=db)

    print(f"Model: {session_manager.model}")
    print("Starting Telegram Bot...")
    bot.run(post_init=post_init, post_shutdown=post_shutdown)


if __name__ == "__main__":
    main()
