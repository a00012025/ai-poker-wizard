# src/telegram_bot/bot.py
import asyncio
import logging
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.claude_session import ClaudeSessionManager

# Allow importing from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# Telegram message limit
MAX_MESSAGE_LENGTH = 4096

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def _setup_logger() -> logging.Logger:
    _LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("poker_bot")
    if not logger.handlers:
        handler = logging.FileHandler(_LOG_DIR / "bot.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        # Also log to console
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(console)
        logger.setLevel(logging.DEBUG)
    return logger


class PokerWizardBot:
    def __init__(self, token: str = None, session_manager: ClaudeSessionManager = None):
        self.token = token or os.getenv('BOT_TOKEN')
        self.session_manager = session_manager or ClaudeSessionManager()
        self.application = None
        self.log = _setup_logger()

    def _user_label(self, update: Update) -> str:
        u = update.effective_user
        chat = update.effective_chat
        name = u.username or u.first_name or str(u.id) if u else "?"
        return f"user=@{name} chat={chat.id}"

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        self.log.info(f"[{self._user_label(update)}] /start")
        welcome_msg = """🃏 **歡迎使用 AI Poker Wizard！**

我是你的專業撲克教練，可以幫你：
• 分析手牌歷史
• 提供 GTO 策略建議
• 復盤錦標賽手牌
• 改進決策技巧

📝 **使用方法：**
直接發送手牌描述，例如：
"Hero 42bb effective, UTG+1 raise 2bb, hero SB all-in A9s"

📁 **或上傳 GGPoker 手牌歷史**（.txt 或 .zip），自動比對 GTO 策略

🔄 /clear — 清除對話紀錄，開始新對話

讓我們開始提升你的撲克技巧！💪"""

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        self.log.info(f"[{self._user_label(update)}] /help")
        help_msg = """🆘 **使用說明**

**指令：**
/start - 開始使用
/help - 顯示說明
/clear - 清除對話紀錄

**手牌分析：**
直接描述你的手牌情況，我會提供專業 GTO 分析和教練建議。
支援多輪對話 — 你可以追問細節或討論不同打法。

**範例格式：**
```
Hero 17bb effective
UTG raise 2bb, BTN call, Hero SB all-in A9s
UTG fold, BTN call
```

有問題隨時問我！"""

        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command — reset Claude session"""
        chat_id = update.effective_chat.id
        self.log.info(f"[{self._user_label(update)}] /clear")
        self.session_manager.clear_session(chat_id)
        await update.message.reply_text("🔄 對話紀錄已清除，開始新的對話！")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text messages via Claude session"""
        chat_id = update.effective_chat.id
        user_text = update.message.text
        label = self._user_label(update)

        self.log.info(f"[{label}] Message: {user_text[:300]}")

        await update.message.chat.send_action(action="typing")

        t0 = time.time()
        try:
            response = await self.session_manager.send_message(chat_id, user_text)
            elapsed = time.time() - t0
            self.log.info(f"[{label}] Response OK ({elapsed:.1f}s, {len(response)} chars)")
            formatted = _format_for_telegram(response)
            if not formatted.strip():
                self.log.warning(f"[{label}] Empty response from session manager")
                await update.message.reply_text("抱歉，分析過程中出現問題，請重新傳送手牌。")
                return
            for chunk in _split_message(formatted):
                if not chunk.strip():
                    continue
                try:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
                except Exception:
                    # Markdown parse error — retry without parse_mode
                    self.log.warning(f"[{label}] Markdown parse failed, retrying as plain text")
                    await update.message.reply_text(chunk)
        except Exception as e:
            elapsed = time.time() - t0
            self.log.error(f"[{label}] Error after {elapsed:.1f}s: {e}", exc_info=True)
            error_msg = f"❌ 分析時發生錯誤：{str(e)}\n\n請稍後再試，或使用 /clear 重新開始。"
            await update.message.reply_text(error_msg)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded hand history files (.txt or .zip)."""
        label = self._user_label(update)
        doc = update.message.document

        if not doc:
            return

        fname = doc.file_name or ""
        fsize = doc.file_size or 0
        self.log.info(f"[{label}] Document: {fname} ({fsize} bytes)")

        # Validate file type
        fname_lower = fname.lower()
        if not (fname_lower.endswith(".txt") or fname_lower.endswith(".zip")):
            await update.message.reply_text(
                "請上傳手牌歷史檔案（.txt 或 .zip）"
            )
            return

        # Validate file size (5MB max)
        if fsize > 5 * 1024 * 1024:
            await update.message.reply_text("檔案太大（上限 5MB）")
            return

        # Send initial processing message
        status_msg = await update.message.reply_text(
            "📥 下載檔案中..."
        )

        t0 = time.time()
        try:
            # Download file
            tg_file = await doc.get_file()
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                download_path = tmpdir_path / fname
                await tg_file.download_to_drive(str(download_path))

                # Extract txt files
                txt_files = []
                if fname_lower.endswith(".zip"):
                    with zipfile.ZipFile(download_path) as zf:
                        for name in zf.namelist():
                            if name.lower().endswith(".txt") and not name.startswith("__"):
                                zf.extract(name, tmpdir_path)
                                txt_files.append(tmpdir_path / name)
                else:
                    txt_files = [download_path]

                if not txt_files:
                    await status_msg.edit_text("ZIP 中沒有找到 .txt 檔案")
                    return

                # Parse hands
                await status_msg.edit_text(
                    f"📂 解析 {len(txt_files)} 個檔案..."
                )

                from hh_parser import parse_file
                all_hands = []
                for tf in sorted(txt_files):
                    hands = parse_file(tf, include_folds=True)
                    all_hands.extend(hands)

                if not all_hands:
                    await status_msg.edit_text(
                        "沒有解析到 hero 的手牌。請確認檔案是 GGPoker 手牌歷史格式，且包含 Hero。"
                    )
                    return

                await status_msg.edit_text(
                    f"🔍 解析到 {len(all_hands)} 手，開始 GTO 分析..."
                )

                # Ensure GTO Wizard session
                from gto_token import ensure_session, capture_browser_token
                if not ensure_session():
                    self.log.warning(f"[{label}] GTO session expired for HH upload")
                    await status_msg.edit_text(
                        "GTO Wizard session 已過期，正在等待登入..."
                    )
                    for _ in range(24):
                        await asyncio.sleep(5)
                        if capture_browser_token():
                            break
                    else:
                        await status_msg.edit_text(
                            "GTO Wizard session 過期，請登入後重新上傳。"
                        )
                        return

                # Run deviation analysis in thread to not block event loop
                last_update_time = [time.time()]

                async def progress_callback(current, total, hand_info):
                    now = time.time()
                    # Update every 10 hands or every 5 seconds
                    if current == total or current % 10 == 0 or now - last_update_time[0] > 5:
                        last_update_time[0] = now
                        elapsed = now - t0
                        try:
                            await status_msg.edit_text(
                                f"🔍 分析中 {current}/{total}... ({elapsed:.0f}s)\n"
                                f"目前: {hand_info}"
                            )
                        except Exception:
                            pass  # Telegram rate limit

                from hh_deviation_report import analyze_hands, format_deviation_report

                # Run blocking analysis in executor with async progress
                loop = asyncio.get_event_loop()
                progress_queue = asyncio.Queue()

                def sync_progress(current, total, info):
                    loop.call_soon_threadsafe(progress_queue.put_nowait, (current, total, info))

                async def drain_progress():
                    while True:
                        try:
                            item = progress_queue.get_nowait()
                            await progress_callback(*item)
                        except asyncio.QueueEmpty:
                            break

                async def run_analysis():
                    results = await loop.run_in_executor(
                        None, lambda: analyze_hands(all_hands, delay=0.3, on_progress=sync_progress)
                    )
                    return results

                # Run analysis with periodic progress drain
                analysis_task = asyncio.create_task(run_analysis())
                while not analysis_task.done():
                    await asyncio.sleep(2)
                    await drain_progress()
                await drain_progress()
                results = await analysis_task

                elapsed = time.time() - t0
                self.log.info(
                    f"[{label}] HH analysis done: {len(all_hands)} hands in {elapsed:.1f}s"
                )

                # Format report
                report = format_deviation_report(results)
                report += f"\n⏱ 分析耗時 {elapsed:.0f} 秒"

                # Send report
                await status_msg.delete()
                formatted = _format_for_telegram(report)
                for chunk in _split_message(formatted):
                    if not chunk.strip():
                        continue
                    try:
                        await update.message.reply_text(chunk, parse_mode='Markdown')
                    except Exception:
                        self.log.warning(f"[{label}] Markdown parse failed, retrying as plain text")
                        await update.message.reply_text(chunk)

        except Exception as e:
            elapsed = time.time() - t0
            self.log.error(f"[{label}] HH upload error after {elapsed:.1f}s: {e}", exc_info=True)
            try:
                await status_msg.edit_text(f"❌ 分析時發生錯誤：{e}")
            except Exception:
                await update.message.reply_text(f"❌ 分析時發生錯誤：{e}")

    def setup_handlers(self):
        """Setup bot handlers"""
        if not self.application:
            self.application = Application.builder().token(self.token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

        return self.application

    def run(self):
        """Run the bot (blocking — manages its own event loop)."""
        app = self.setup_handlers()
        self.log.info(f"Bot starting — model={self.session_manager.model}, max_turns={self.session_manager.max_turns}")
        app.run_polling(drop_pending_updates=True)


def _format_for_telegram(text: str) -> str:
    """Convert LLM output to Telegram-compatible Markdown.

    Telegram Markdown supports: *bold*, _italic_, `code`, ```pre```, [link](url)
    Does NOT support: **bold**, # headers, tables, * bullets.
    """
    import re
    # * bullet points → • (must do BEFORE bold processing)
    # Matches: "* text" or "*   text" at start of line
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    # **bold** → *bold*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # # headers → *bold*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    return text


def _split_message(text: str) -> list[str]:
    """Split text into chunks that fit Telegram's message limit."""
    if len(text) <= MAX_MESSAGE_LENGTH:
        return [text]

    chunks = []
    while text:
        if len(text) <= MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        # Try to split at last newline within limit
        split_at = text.rfind('\n', 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks
