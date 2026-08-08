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

# Scheduled-job timezone (Taiwan = UTC+8)
TZ_TAIPEI = ZoneInfo("Asia/Taipei")


async def _run_script(*script_args) -> tuple[int, str]:
    """Run a repo script as a subprocess from the repo root; return (rc, tail)."""
    from src.ingest_runner import _run_script as _run
    env = dict(os.environ)
    env.pop("POKER_BOT_PROCESS", None)  # child is owner-run tooling, not a request
    return await _run(env, *script_args)


async def _daily_ledger_ingest_job(context):
    """Daily 05:00 Taipei: enqueue an ingest request for the owner.

    The 5s queue poller runs it with the owner's extension-synced token
    (users.gto_refresh_token) — same single-flight path as the extension
    button and /ingest, so it can't race a user-triggered ingest.
    """
    from ledger_service import resolve_owner_chat_id
    from src.ingest_runner import enqueue_request
    try:
        owner = await resolve_owner_chat_id(db.pool) if db.pool else None
        if not owner:
            raise RuntimeError("no owner chat id (OWNER_CHAT_ID unset / multiple active users)")
        reused = await enqueue_request(db.pool, owner)
        logger.info(f"Daily ingest enqueued for owner {owner} (reused_open_request={reused})")
    except Exception as e:
        logger.error(f"Daily ledger ingest job failed: {e}")
        admin = os.getenv("ADMIN_CHAT_ID")
        if admin:
            try:
                await context.bot.send_message(int(admin), f"⚠️ 每日手牌攝取排程失敗\n{e}")
            except Exception as notify_err:
                logger.error(f"Daily ingest failure notify failed: {notify_err}")


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
        from scorecard import weekly_tg_payload
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        payload = weekly_tg_payload(row["week"], data)
        markup = None
        if payload["buttons"]:
            # rows may carry url (drills / 復盤) OR callback_data (✔ 完成 / ➕ 加練)
            markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton(b["text"], url=b.get("url"),
                                       callback_data=b.get("callback_data")) for b in r]
                 for r in payload["buttons"]])
        await context.bot.send_message(
            owner, payload["html"], parse_mode="HTML",
            disable_web_page_preview=True, reply_markup=markup)
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

    # PTB v20+ uses cron-style day numbering: 0=Sun, 1=Mon, ..., 6=Sat.
    # The legacy frequency-era weekly leak report is retired — the ledger
    # scorecard (Sun 21:00) is the single weekly surface (North Star §12).
    if db.pool and application.job_queue:
        # Phase 1 ledger loop: daily incremental ingest + Sunday training-plan push.
        application.job_queue.run_daily(
            _daily_ledger_ingest_job,
            time=dt_time(hour=5, minute=0, tzinfo=TZ_TAIPEI),
            days=tuple(range(7)), name="daily_ledger_ingest")
        application.job_queue.run_daily(
            _weekly_scorecard_job,
            time=dt_time(hour=21, minute=0, tzinfo=TZ_TAIPEI),
            days=(0,), name="weekly_scorecard")
        # Extension-triggered ingest queue (gtow_ingest_requests via gtow-sync).
        from src.ingest_runner import poll_job
        application.job_queue.run_repeating(
            lambda ctx: poll_job(ctx, db), interval=5, first=10,
            name="ingest_request_poller")
        logger.info("Ledger ingest (daily 05:00) + scorecard (Sun 21:00) "
                    "+ ingest poller (5s) jobs scheduled")

    # Command menu ("/"): public users get the basics; the owner additionally
    # sees the training-loop commands (live import / practice queue / plan).
    try:
        from telegram import BotCommand, BotCommandScopeChat
        base = [
            BotCommand("help", "使用說明"),
            BotCommand("clear", "清除對話上下文"),
            BotCommand("pair", "配對 Chrome Extension"),
            BotCommand("devices", "查看同步裝置"),
            BotCommand("revoke", "撤銷同步裝置"),
            BotCommand("settoken", "手動設定 GTOW token（備援）"),
            BotCommand("logout", "解除 GTOW 綁定"),
        ]
        await application.bot.set_my_commands(base)
        owner = os.getenv("ADMIN_CHAT_ID")
        if owner:
            await application.bot.set_my_commands(
                base + [
                    BotCommand("live", "導入現場手牌（批次評分入帳）"),
                    BotCommand("lives", "最近線下 sessions／重傳復盤"),
                    BotCommand("sessions", "最近線上 sessions／重傳復盤"),
                    BotCommand("queue", "練習佇列"),
                    BotCommand("plan", "本週訓練計畫"),
                    BotCommand("review", "這場復盤（最近一個 session）"),
                    BotCommand("ingest", "手動攝取 GTOW Analyze"),
                    BotCommand("report", "使用量報告"),
                ],
                scope=BotCommandScopeChat(chat_id=int(owner)))
        logger.info("Bot command menu registered")
    except Exception as e:
        logger.warning(f"set_my_commands failed: {e}")


async def post_shutdown(application):
    """Called after Application.shutdown() — close DB pool."""
    await db.close()


def main():
    print("AI Poker Wizard (Gemini) starting...")

    # Production requests must carry the requesting user's DB token. This flag
    # makes any missed wiring path fail closed instead of borrowing owner auth.
    os.environ["POKER_BOT_PROCESS"] = "1"

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
