# src/telegram_bot/bot.py
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.claude_session import ClaudeSessionManager

if TYPE_CHECKING:
    from src.database import Database

# Allow importing from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# Upload analysis timeout (seconds)
ANALYSIS_TIMEOUT = 600

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
    def __init__(self, token: str = None, session_manager: ClaudeSessionManager = None,
                 db: Database | None = None):
        self.token = token or os.getenv('BOT_TOKEN')
        self.session_manager = session_manager or ClaudeSessionManager()
        self.db = db
        self.application = None
        self.log = _setup_logger()
        # Store uploaded HH hands per chat for follow-up queries
        self.hh_hands: dict[int, list[dict]] = {}  # chat_id -> parsed hands
        self.admin_chat_id = int(os.getenv("ADMIN_CHAT_ID", "0")) or None
        # Per-user lock to serialize messages from the same user
        self._user_locks: dict[int, asyncio.Lock] = {}

    def _user_lock(self, chat_id: int) -> asyncio.Lock:
        """Get or create a per-user lock to serialize message handling."""
        if chat_id not in self._user_locks:
            self._user_locks[chat_id] = asyncio.Lock()
        return self._user_locks[chat_id]

    def _user_label(self, update: Update) -> str:
        u = update.effective_user
        chat = update.effective_chat
        name = u.username or u.first_name or str(u.id) if u else "?"
        return f"user=@{name} chat={chat.id}"

    async def _check_user(self, update: Update) -> bool:
        """Check if user is allowed. Returns True if allowed, False if rejected."""
        if not self.db:
            return True  # No DB = no whitelist enforcement
        user_id = update.effective_user.id
        if await self.db.is_user_allowed(user_id):
            return True
        self.log.warning(f"[{self._user_label(update)}] Rejected — not in whitelist")
        await update.message.reply_text("Sorry, access is restricted.")
        return False

    async def _notify_admin(self, text: str):
        """Send a notification to the admin chat if configured."""
        if self.admin_chat_id and self.application:
            try:
                await self.application.bot.send_message(self.admin_chat_id, text)
            except Exception as e:
                self.log.error(f"Failed to notify admin: {e}")

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

📸 **或上傳手牌回放截圖**，自動辨識並分析 GTO 策略
📁 **或上傳 GGPoker 手牌歷史**（.txt 或 .zip），批次比對 GTO 策略

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
        if not await self._check_user(update):
            return
        chat_id = update.effective_chat.id
        self.log.info(f"[{self._user_label(update)}] /clear")
        self.session_manager.clear_session(chat_id)
        await update.message.reply_text("🔄 對話紀錄已清除，開始新的對話！")

    async def login_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /login — send bookmarklet setup instructions."""
        if not await self._check_user(update):
            return
        self.log.info(f"[{self._user_label(update)}] /login")

        bookmarklet = (
            "javascript:void(navigator.clipboard.writeText("
            "'/settoken '+localStorage.getItem('user_refresh'))"
            ".then(()=>alert('已複製！請回 Telegram 貼上')))"
        )

        msg = (
            "*綁定 GTO Wizard 帳號*\n\n"
            "*一次性設定（建立書籤工具）：*\n"
            "1. 在瀏覽器新增一個書籤（任意網頁按 Ctrl+D）\n"
            "2. 編輯書籤，名稱改為 `GTO Token`\n"
            "3. 網址欄貼上以下代碼：\n\n"
            f"`{bookmarklet}`\n\n"
            "*每次綁定 token：*\n"
            "1. 登入 app.gtowizard.com\n"
            "2. 點擊書籤工具 → 自動複製指令\n"
            "3. 回到 Telegram 貼上即可\n\n"
            "綁定後，你的查詢會使用自己的 GTO Wizard 帳號。"
        )
        try:
            await update.message.reply_text(msg, parse_mode='Markdown')
        except Exception:
            await update.message.reply_text(msg)

    async def settoken_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settoken <token> — validate and store user's GTO Wizard token."""
        if not await self._check_user(update):
            return
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /settoken")

        # Extract token from command args
        token = " ".join(context.args) if context.args else ""
        if not token or not token.startswith("eyJ"):
            await update.message.reply_text(
                "格式錯誤。請使用書籤工具複製指令，或手動輸入：\n"
                "`/settoken eyJhbG...`",
                parse_mode='Markdown',
            )
            return

        # Validate: try refreshing the token
        from gto_token import _refresh_access, _jwt_exp
        try:
            exp = _jwt_exp(token)
            if exp < time.time():
                await update.message.reply_text("Token 已過期，請重新登入 GTO Wizard 後再試。")
                return
        except Exception:
            await update.message.reply_text("Token 格式無效。")
            return

        access = _refresh_access(token)
        if not access:
            await update.message.reply_text(
                "Token 無法刷新，請確認你已登入 GTO Wizard 且 token 有效。"
            )
            return

        # Store in DB
        if self.db:
            await self.db.save_user_gto_token(user_id, token)

        self.log.info(f"[{label}] GTO token bound successfully")
        await update.message.reply_text("GTO Wizard 帳號綁定成功！之後的查詢會使用你的帳號。")

    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout — remove user's GTO Wizard token."""
        if not await self._check_user(update):
            return
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /logout")

        if self.db:
            await self.db.delete_user_gto_token(user_id)

        from gto_token import invalidate_user_token
        invalidate_user_token(user_id)

        await update.message.reply_text("已解除 GTO Wizard 帳號綁定。將使用共用帳號。")

    async def _get_user_refresh_token(self, user_id: int) -> str | None:
        """Look up user's GTO Wizard refresh token from DB."""
        if not self.db:
            return None
        return await self.db.get_user_gto_token(user_id)

    @staticmethod
    def _setup_user_token(user_id: int, refresh_token: str):
        """Set thread-local GTO token for the current thread."""
        from gto_token import get_user_access_token
        from gto_api import set_user_token
        access = get_user_access_token(user_id, refresh_token)
        set_user_token(access)

    @staticmethod
    def _clear_user_token():
        """Clear thread-local GTO token."""
        from gto_api import clear_user_token
        clear_user_token()

    async def _find_hh_hand(self, chat_id: int, text: str) -> dict | None:
        """Try to find a referenced hand from uploaded HH by hand_id suffix."""
        import re
        # Check in-memory cache first
        hands = self.hh_hands.get(chat_id, [])
        # Match TM followed by digits, or just last 4+ digits of a hand_id
        m = re.search(r'(TM\d+)', text)
        if m:
            full_id = m.group(1)
            for h in hands:
                if h.get("hand_id") == full_id:
                    return h
        m = re.search(r'\b(\d{4,})\b', text)
        if m:
            suffix = m.group(1)
            for h in hands:
                if h.get("hand_id", "").endswith(suffix):
                    return h
            # Fall back to DB
            if self.db:
                return await self.db.find_hand(chat_id, suffix)
        return None

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle all text messages via Claude session"""
        if not await self._check_user(update):
            return
        chat_id = update.effective_chat.id
        async with self._user_lock(chat_id):
            await self._handle_message_inner(update, context)

    async def _handle_message_inner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        user_text = update.message.text
        label = self._user_label(update)

        self.log.info(f"[{label}] Message: {user_text[:300]}")

        # Check if user is referencing a hand from uploaded HH
        hh_hand = await self._find_hh_hand(chat_id, user_text)
        if hh_hand:
            self.log.info(f"[{label}] HH follow-up: {hh_hand['hand_id']} "
                          f"{hh_hand['hero_position']} {hh_hand['hero_hand']}")
            await self._analyze_hh_hand(update, hh_hand, user_text)
            return

        status_msg = await update.message.reply_text("🔍 分析中...")

        async def _on_status(msg: str):
            try:
                await status_msg.edit_text(f"⏳ {msg}")
            except Exception:
                pass  # message already deleted or unchanged

        # Look up user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)

        t0 = time.time()
        try:
            response = await self.session_manager.send_message(
                chat_id, user_text, on_status=_on_status,
                user_id=user_id, refresh_token=refresh_token,
            )
            elapsed = time.time() - t0
            self.log.info(f"[{label}] Response OK ({elapsed:.1f}s, {len(response)} chars)")

            await status_msg.delete()

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
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError) and refresh_token:
                # Per-user token expired — tell the user to re-bind
                self.log.warning(f"[{label}] User token expired during message handling")
                try:
                    await status_msg.edit_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                except Exception:
                    await update.message.reply_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                return
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Global token expired during message handling")
                try:
                    await status_msg.edit_text("GTO Wizard token 過期，請管理員更新 token。")
                except Exception:
                    await update.message.reply_text("GTO Wizard token 過期，請管理員更新 token。")
                await self._notify_admin(f"GTO Wizard token 過期！{label} 發送訊息時觸發。")
                return
            try:
                await status_msg.edit_text(f"❌ {str(e)}")
            except Exception:
                await update.message.reply_text(f"❌ 分析時發生錯誤：{str(e)}\n\n請稍後再試，或使用 /clear 重新開始。")

    async def _analyze_hh_hand(self, update: Update, hand: dict, user_text: str):
        """Run full GTO analysis on a specific HH hand and coach via LLM."""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        label = self._user_label(update)
        hand_id = hand["hand_id"]

        await update.message.chat.send_action(action="typing")

        # Look up user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)

        t0 = time.time()
        try:
            # Build analyze_hand_full input from parsed HH hand
            analysis_input = {
                "gametype": hand.get("gametype", "MTTGeneral"),
                "effective_bb": hand["effective_bb"],
                "hero_position": hand["hero_position"],
                "hero_hand": hand["hero_hand"],
                "preflop_actions": hand["preflop_actions"],
            }
            if hand.get("streets"):
                analysis_input["streets"] = hand["streets"]

            # Set user token for GTO API calls
            if refresh_token:
                self._setup_user_token(user_id, refresh_token)
            try:
                from analyze_hand import analyze_hand_full
                context = analyze_hand_full(analysis_input)
            finally:
                if refresh_token:
                    self._clear_user_token()
            gto_data = context["text"]
            self.session_manager.hand_contexts[chat_id] = context

            t_analyze = time.time()
            self.log.info(
                f"[{label}] HH hand {hand_id} analyzed in {t_analyze - t0:.1f}s"
            )

            # Coach with LLM
            hand_desc = (
                f"Hand ID: {hand_id}\n"
                f"Hero {hand['hero_position']} {hand['hero_hand']} "
                f"({hand['effective_bb']:.0f}bb, {hand.get('num_players', 8)}人)\n"
                f"Preflop: {hand['preflop_actions']}"
            )
            if hand.get("streets"):
                for s in hand["streets"]:
                    board = s.get("board", s.get("card", ""))
                    acts = " ".join(
                        f"{a['position']}:{a['action']}" for a in s["actions"]
                    )
                    hand_desc += f"\n{board} → {acts}"

            coaching_prompt = (
                f"用戶上傳了手牌歷史檔案，想分析這手牌：\n{hand_desc}\n\n"
                f"用戶問題：{user_text}\n\n"
                f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
                f"請根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
            )
            response = await self.session_manager._chat_with_tools(
                chat_id, coaching_prompt,
                user_id=user_id, refresh_token=refresh_token,
            )

            elapsed = time.time() - t0
            self.log.info(f"[{label}] HH follow-up done ({elapsed:.1f}s)")

            formatted = _format_for_telegram(response)
            for chunk in _split_message(formatted):
                if not chunk.strip():
                    continue
                try:
                    await update.message.reply_text(chunk, parse_mode='Markdown')
                except Exception:
                    await update.message.reply_text(chunk)

        except Exception as e:
            elapsed = time.time() - t0
            # Handle token expiry gracefully
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError) and refresh_token:
                self.log.warning(f"[{label}] User token expired during HH follow-up")
                await update.message.reply_text(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Global token expired during HH follow-up")
                await update.message.reply_text("GTO Wizard token 過期，請管理員更新 token。")
                await self._notify_admin(f"GTO Wizard token 過期！{label} 查詢 {hand_id} 時觸發。")
                return
            self.log.error(f"[{label}] HH follow-up error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ 分析 {hand_id} 時發生錯誤：{e}")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded photos (poker screenshots for GTO analysis)."""
        if not await self._check_user(update):
            return
        chat_id = update.effective_chat.id
        async with self._user_lock(chat_id):
            await self._handle_photo_inner(update, context)

    async def _handle_photo_inner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        label = self._user_label(update)
        user_id = update.effective_user.id
        caption = update.message.caption or ""

        self.log.info(f"[{label}] Photo received, caption: {caption[:200]}")

        # Get the largest photo resolution
        photo = update.message.photo[-1]

        status_msg = await update.message.reply_text("🔍 正在下載圖片...")

        # Look up user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)

        t0 = time.time()
        try:
            # Download photo
            tg_file = await photo.get_file()
            image_bytes = bytes(await tg_file.download_as_bytearray())

            response = await self.session_manager.send_image_message(
                chat_id=update.effective_chat.id,
                image_bytes=image_bytes,
                mime_type="image/jpeg",
                user_text=caption,
                status_callback=status_msg.edit_text,
                user_id=user_id,
                refresh_token=refresh_token,
            )

            elapsed = time.time() - t0
            self.log.info(f"[{label}] Photo response OK ({elapsed:.1f}s)")

            await status_msg.delete()

            formatted = _format_for_telegram(response)
            if not formatted.strip():
                await update.message.reply_text("抱歉，無法分析截圖，請重新發送。")
                return
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
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError) and refresh_token:
                self.log.warning(f"[{label}] User token expired during photo analysis")
                try:
                    await status_msg.edit_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                except Exception:
                    await update.message.reply_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                return
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Global token expired during photo analysis")
                try:
                    await status_msg.edit_text("GTO Wizard token 過期，請管理員更新 token。")
                except Exception:
                    await update.message.reply_text("GTO Wizard token 過期，請管理員更新 token。")
                await self._notify_admin(f"GTO Wizard token 過期！{label} 發送截圖時觸發。")
                return
            self.log.error(f"[{label}] Photo error after {elapsed:.1f}s: {e}", exc_info=True)
            try:
                await status_msg.edit_text(f"❌ 分析截圖時發生錯誤：{e}")
            except Exception:
                await update.message.reply_text(f"❌ 分析截圖時發生錯誤：{e}")

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded hand history files (.txt or .zip)."""
        if not await self._check_user(update):
            return
        chat_id = update.effective_chat.id
        async with self._user_lock(chat_id):
            await self._handle_document_inner(update, context)

    async def _handle_document_inner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        label = self._user_label(update)
        user_id = update.effective_user.id
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

        refresh_token = None  # set later inside try block
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

                # Look up user's GTO token
                refresh_token = await self._get_user_refresh_token(user_id)

                # Ensure GTO Wizard session (user token or global)
                if not refresh_token:
                    from gto_token import ensure_session, TokenExpiredError
                    if not ensure_session():
                        self.log.warning(f"[{label}] GTO session expired for HH upload")
                        await status_msg.edit_text(
                            "GTO Wizard session 過期，請管理員更新 token。"
                        )
                        await self._notify_admin(
                            f"GTO Wizard token 過期！用戶 {label} 上傳 HH 時觸發。\n"
                            f"請更新 .tokens.json"
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

                # Capture token setup for executor thread
                _user_id = user_id
                _refresh_token = refresh_token
                _setup = self._setup_user_token
                _clear = self._clear_user_token

                def _run_in_thread():
                    if _refresh_token:
                        _setup(_user_id, _refresh_token)
                    try:
                        return analyze_hands(all_hands, delay=0.3, on_progress=sync_progress)
                    finally:
                        if _refresh_token:
                            _clear()

                async def run_analysis():
                    return await loop.run_in_executor(None, _run_in_thread)

                # Run analysis with periodic progress drain + overall timeout
                analysis_task = asyncio.create_task(run_analysis())
                try:
                    deadline = time.time() + ANALYSIS_TIMEOUT
                    while not analysis_task.done():
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            analysis_task.cancel()
                            raise asyncio.TimeoutError()
                        await asyncio.sleep(min(2, remaining))
                        await drain_progress()
                    await drain_progress()
                    results = await analysis_task
                except asyncio.TimeoutError:
                    await drain_progress()
                    self.log.warning(f"[{label}] HH analysis timed out after {ANALYSIS_TIMEOUT}s")
                    await status_msg.edit_text(
                        f"分析超時（{ANALYSIS_TIMEOUT // 60} 分鐘），請減少手牌數量後重試。"
                    )
                    return

                elapsed = time.time() - t0
                self.log.info(
                    f"[{label}] HH analysis done: {len(all_hands)} hands in {elapsed:.1f}s"
                )

                # Store hands for follow-up queries
                chat_id = update.effective_chat.id
                self.hh_hands[chat_id] = all_hands
                if self.db:
                    try:
                        await self.db.save_hands(chat_id, all_hands)
                    except Exception as e:
                        self.log.error(f"[{label}] Failed to save hands to DB: {e}")

                # Format report
                report = format_deviation_report(results)
                report += "\n\n💬 回覆 hand ID（如 `TM5600279272`）可查看該手詳細 GTO 分析"
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
            # Handle token expiry gracefully
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError) and refresh_token:
                self.log.warning(f"[{label}] User token expired during HH upload")
                try:
                    await status_msg.edit_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                except Exception:
                    await update.message.reply_text(
                        "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                    )
                return
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Global token expired during HH upload")
                try:
                    await status_msg.edit_text("GTO Wizard token 過期，請管理員更新 token。")
                except Exception:
                    await update.message.reply_text("GTO Wizard token 過期，請管理員更新 token。")
                await self._notify_admin(f"GTO Wizard token 過期！{label} 上傳 HH 時觸發。")
                return
            self.log.error(f"[{label}] HH upload error after {elapsed:.1f}s: {e}", exc_info=True)
            try:
                await status_msg.edit_text(f"❌ 分析時發生錯誤：{e}")
            except Exception:
                await update.message.reply_text(f"❌ 分析時發生錯誤：{e}")

    def setup_handlers(self, post_init=None, post_shutdown=None):
        """Setup bot handlers"""
        if not self.application:
            builder = Application.builder().token(self.token)
            if post_init:
                builder = builder.post_init(post_init)
            if post_shutdown:
                builder = builder.post_shutdown(post_shutdown)
            self.application = builder.build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("login", self.login_command))
        self.application.add_handler(CommandHandler("settoken", self.settoken_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))

        return self.application

    def run(self, post_init=None, post_shutdown=None):
        """Run the bot (blocking — manages its own event loop)."""
        app = self.setup_handlers(post_init=post_init, post_shutdown=post_shutdown)
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
