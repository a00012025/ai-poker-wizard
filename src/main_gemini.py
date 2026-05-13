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
