# src/telegram_bot/bot.py
import logging
import os
import time
from pathlib import Path
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.claude_session import ClaudeSessionManager

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

📁 **或上傳 Natural8 檔案** 並告訴我要分析哪一手牌

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

    def setup_handlers(self):
        """Setup bot handlers"""
        if not self.application:
            self.application = Application.builder().token(self.token).build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

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
