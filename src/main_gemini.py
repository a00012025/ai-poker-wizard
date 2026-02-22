# src/main_gemini.py
"""Entry point using Gemini API (fast, no Claude CLI subprocess)."""
import datetime
import logging
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

from src.database import Database
from src.gemini_session import GeminiSessionManager
from src.telegram_bot.bot import PokerWizardBot

logger = logging.getLogger("poker_bot")

db = Database()
bot: PokerWizardBot | None = None


async def post_init(application):
    """Called after Application.initialize() — connect DB and schedule jobs."""
    dsn = os.getenv("SUPABASE_CONN")
    if dsn:
        await db.connect(dsn)
        await db.check_tables()
        logger.info("Database ready")
    else:
        logger.warning("SUPABASE_CONN not set — running without database")

    # Schedule daily analytics report at 09:00 Asia/Taipei
    tz = ZoneInfo("Asia/Taipei")
    application.job_queue.run_daily(
        bot.send_daily_report,
        time=datetime.time(hour=9, tzinfo=tz),
        name="daily_report",
    )
    logger.info("Scheduled daily report at 09:00 Asia/Taipei")

    # One-time test — remove after verifying
    application.job_queue.run_once(bot.send_daily_report, when=5)


async def post_shutdown(application):
    """Called after Application.shutdown() — close DB pool."""
    await db.close()


def main():
    global bot
    print("AI Poker Wizard (Gemini) starting...")

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        print("ERROR: BOT_TOKEN not set")
        return

    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return

    session_manager = GeminiSessionManager()
    bot = PokerWizardBot(token=bot_token, session_manager=session_manager, db=db)

    print(f"Model: {session_manager.model}")
    print("Starting Telegram Bot...")
    bot.run(post_init=post_init, post_shutdown=post_shutdown)


if __name__ == "__main__":
    main()
