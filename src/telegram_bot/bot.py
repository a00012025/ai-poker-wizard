# src/telegram_bot/bot.py
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          MessageHandler, filters, ContextTypes)
from telegram.request import HTTPXRequest
from src.claude_session import ClaudeSessionManager

if TYPE_CHECKING:
    from src.database import Database

# Allow importing from scripts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

# Upload analysis timeout (seconds)
ANALYSIS_TIMEOUT = 1800

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


_IMAGE_DOC_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")


def _is_image_document(mime_type: str | None, fname_lower: str) -> bool:
    """True when a Telegram Document is actually an image.

    Telegram delivers images as Document (not Photo) when the user sends
    "as file" or when the client uploads an unsupported-for-compression
    format (HEIC, large PNG, etc.). Animated GIFs are excluded — they
    arrive via ANIMATION and aren't poker screenshots.
    """
    if mime_type:
        mt = mime_type.lower()
        if mt.startswith("image/") and not mt.endswith("/gif"):
            return True
    return fname_lower.endswith(_IMAGE_DOC_EXTS)


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

    async def _touch_user(self, update: Update):
        """Upsert user row with latest username/name from Telegram."""
        if not self.db:
            return
        u = update.effective_user
        if not u:
            return
        try:
            await self.db.upsert_user(u.id, u.username, u.first_name)
        except Exception:
            pass

    async def _has_gto_token(self, user_id: int) -> bool:
        """Check if user has a GTO Wizard token stored."""
        if not self.db:
            return False
        token = await self.db.get_user_gto_token(user_id)
        return token is not None

    async def _send_token_gate(self, update: Update):
        """Reply with setup instructions when user has no GTO token."""
        msg = (
            "請先綁定 GTO Wizard 帳號才能使用。\n\n"
            "安裝 Extension 自動取得 token：\n"
            "→ github.com/a00012025/ai-poker-wizard/releases\n\n"
            "或手動：登入 app.gtowizard.com → F12 Console → "
            "copy(localStorage.getItem('user_refresh')) → /settoken <貼上>"
        )
        await update.message.reply_text(msg)

    async def _notify_admin(self, text: str):
        """Send a notification to the admin chat if configured."""
        if self.admin_chat_id and self.application:
            try:
                await self.application.bot.send_message(self.admin_chat_id, text)
            except Exception as e:
                self.log.error(f"Failed to notify admin: {e}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await self._touch_user(update)
        self.log.info(f"[{self._user_label(update)}] /start")
        welcome_msg = (
            "🃏 歡迎使用 AI Poker Wizard！\n\n"
            "GTO 撲克教練，可以幫你分析手牌、比對 GTO 策略。\n\n"
            "⚡ 開始前，請先綁定 GTO Wizard 帳號：\n\n"
            "方法一：安裝 Chrome Extension（推薦）\n"
            "→ github.com/a00012025/ai-poker-wizard/releases\n"
            "安裝後登入 GTO Wizard，自動複製 token\n\n"
            "方法二：手動取得\n"
            "1. 登入 app.gtowizard.com\n"
            "2. F12 開啟 Console\n"
            "3. 貼上: copy(localStorage.getItem('user_refresh'))\n"
            "4. 回到這裡輸入: /settoken <貼上>\n\n"
            "📝 使用方式：\n"
            "• 發送手牌描述或截圖，自動 GTO 分析\n"
            "• 上傳 GGPoker .txt/.zip，批次比對偏差\n\n"
            "🎯 遊戲格式判斷：\n"
            "• 預設 = MTT（錦標賽）\n"
            "• 提到「cash」「現金桌」「ring game」= 現金桌分析\n"
            "• 例：cash 6max 100bb, CO raise 2.5bb, BTN 3bet 8bb AKs\n\n"
            "/settoken — 綁定 token\n"
            "/logout — 解除綁定\n"
            "/clear — 清除對話紀錄"
        )
        await update.message.reply_text(welcome_msg)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        self.log.info(f"[{self._user_label(update)}] /help")
        help_msg = """🆘 **使用說明**

**指令：**
/start - 開始使用
/help - 顯示說明
/clear - 清除對話紀錄

**手牌分析：**
直接描述手牌情況，自動 GTO 分析。支援多輪追問。

**MTT（預設）：**
`Hero 17bb, UTG raise, Hero SB all-in A9s`

**現金桌：**
`cash 6max 100bb, CO raise 2.5bb, BTN 3bet 8bb AKs`
提到「cash」「現金桌」「ring game」會自動切換

有問題隨時問我！"""

        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command — reset Claude session"""
        chat_id = update.effective_chat.id
        self.log.info(f"[{self._user_label(update)}] /clear")
        self.session_manager.clear_session(chat_id)
        await update.message.reply_text("🔄 對話紀錄已清除，開始新的對話！")

    async def settoken_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settoken <token> — validate and store user's GTO Wizard token."""
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /settoken")

        # Extract token from command args
        raw_text = update.message.text or ""
        token = " ".join(context.args) if context.args else ""
        self.log.info(f"[{label}] /settoken raw len={len(raw_text)}, token len={len(token)}, prefix={token[:20]}...")
        if not token or not token.startswith("eyJ"):
            self.log.warning(f"[{label}] /settoken bad format: token_empty={not token}, raw='{raw_text[:60]}'")
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
            remaining = exp - time.time()
            self.log.info(f"[{label}] /settoken JWT exp in {remaining:.0f}s")
            if exp < time.time():
                self.log.warning(f"[{label}] /settoken token expired {-remaining:.0f}s ago")
                await update.message.reply_text("Token 已過期，請重新登入 GTO Wizard 後再試。")
                return
        except Exception as e:
            self.log.warning(f"[{label}] /settoken JWT decode failed: {e}")
            await update.message.reply_text("Token 格式無效。")
            return

        access = _refresh_access(token)
        if not access:
            # Fetch error detail for user-facing message
            reason = ""
            code = ""
            try:
                import requests as _req
                from gto_token import API_BASE, ORIGIN
                r = _req.post(
                    f"{API_BASE}/v1/token/refresh/",
                    json={"refresh": token},
                    headers={"origin": ORIGIN, "content-type": "application/json"},
                    timeout=10,
                )
                if r.headers.get("content-type", "").startswith("application/json"):
                    body = r.json()
                    reason = body.get("error", "")
                    code = body.get("code", "")
            except Exception:
                pass
            self.log.warning(f"[{label}] /settoken refresh failed (token prefix: {token[:20]}...) code={code} reason={reason}")

            ERROR_HINTS = {
                "FORCED_LOGOUT": (
                    "GTO Wizard 回報「同時登入裝置過多」，你的 token 已被強制登出。\n\n"
                    "請在 GTO Wizard 網站重新登入，取得新的 token 後再試一次。"
                ),
            }
            hint = ERROR_HINTS.get(code)
            if hint:
                await update.message.reply_text(hint)
            else:
                msg = "Token 無法刷新，請確認你已登入 GTO Wizard 且 token 有效。"
                if reason:
                    msg += f"\n\n原因：{reason}"
                await update.message.reply_text(msg)
            return

        # Store in DB
        if self.db:
            await self.db.save_user_gto_token(user_id, token)

        self.log.info(f"[{label}] GTO token bound successfully")
        await update.message.reply_text("GTO Wizard 帳號綁定成功！之後的查詢會使用你的帳號。")

    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout — remove user's GTO Wizard token."""
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /logout")

        if self.db:
            await self.db.delete_user_gto_token(user_id)

        from gto_token import invalidate_user_token
        invalidate_user_token(user_id)

        await update.message.reply_text("已解除 GTO Wizard 帳號綁定。")

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
            # Fall back to DB
            if self.db:
                return await self.db.find_hand(chat_id, full_id)
            return None
        # Match "H2672", "h2672", or a bare "2672" (4+ digits).
        # Can't use \b before \d because "H" + "2" has no word boundary
        # (both are word chars).
        m = re.search(r'(?:^|[^A-Za-z0-9])[Hh]?(\d{4,})\b', text)
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
        await self._touch_user(update)
        user_id = update.effective_user.id
        if not await self._has_gto_token(user_id):
            await self._send_token_gate(update)
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

        if self.db:
            try:
                await self.db.log_message(chat_id, "text")
            except Exception:
                pass

        # Check if user is referencing a hand from uploaded HH
        hh_hand = await self._find_hh_hand(chat_id, user_text)
        if hh_hand:
            self.log.info(f"[{label}] HH follow-up: {hh_hand['hand_id']} "
                          f"{hh_hand['hero_position']} {hh_hand['hero_hand']}")
            await self._analyze_hh_hand(update, hh_hand, user_text)
            return

        raw_status = await _send_status(update.message, "🔍 分析中...")
        status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

        async def _on_status(msg: str):
            await status_msg.edit_text(f"⏳ {msg}")

        # Look up user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)

        t0 = time.time()
        try:
            async with _TypingLoop(update.message.chat):
                response = await self.session_manager.send_message(
                    chat_id, user_text, on_status=_on_status,
                    user_id=user_id, refresh_token=refresh_token,
                )
            elapsed = time.time() - t0
            self.log.info(f"[{label}] Response OK ({elapsed:.1f}s, {len(response)} chars)")

            await status_msg.delete()

            if not response or not response.strip():
                self.log.warning(f"[{label}] Empty response from session manager")
                await update.message.reply_text("抱歉，分析過程中出現問題，請重新傳送手牌。")
                return
            markup = self._build_followup_markup(chat_id, include_gto_link=True)
            await _send_reply(update.message, response, self.log, label,
                              reply_markup=markup)

            # Send range grid images
            await self._send_pending_range_images(update, chat_id, label)

        except Exception as e:
            elapsed = time.time() - t0
            self.log.error(f"[{label}] Error after {elapsed:.1f}s: {e}", exc_info=True)
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Token expired during message handling")
                await status_msg.edit_text(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            await status_msg.edit_text(f"❌ {str(e)}")

    async def _analyze_hh_hand(self, update: Update, hand: dict, user_text: str):
        """Run full GTO analysis on a specific HH hand and coach via LLM."""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        label = self._user_label(update)
        hand_id = hand["hand_id"]

        # Require user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)
        if not refresh_token:
            await update.message.reply_text("請先使用 /settoken 綁定你的 GTO Wizard 帳號。")
            return

        t0 = time.time()
        _typing = _TypingLoop(update.message.chat)
        await _typing.__aenter__()
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
            self._setup_user_token(user_id, refresh_token)
            try:
                from analyze_hand import analyze_hand_full
                context = analyze_hand_full(analysis_input)
            finally:
                self._clear_user_token()
            gto_data = context["text"]
            self.session_manager.hand_contexts[chat_id] = context

            # Extract deviations for leak detection (fire-and-forget)
            import asyncio as _aio
            if self.session_manager.db and self.session_manager.db.pool:
                _aio.create_task(self.session_manager._extract_deviations(
                    chat_id, hand_id, analysis_input, context))

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
                "\n\n在回覆的最後，用以下格式輸出 3 個值得深入的 follow-up 問題（用戶可以點擊按鈕直接發送）：\n"
                "FOLLOWUP: 問題一\n"
                "FOLLOWUP: 問題二\n"
                "FOLLOWUP: 問題三\n"
            )
            response = await self.session_manager._chat_with_tools(
                chat_id, coaching_prompt,
                user_id=user_id, refresh_token=refresh_token,
            )
            response, followups = self.session_manager._extract_followups(response)
            if followups:
                ctx = self.session_manager.hand_contexts.get(chat_id)
                if ctx is not None:
                    ctx["followup_questions"] = followups

            elapsed = time.time() - t0
            self.log.info(f"[{label}] HH follow-up done ({elapsed:.1f}s)")

            formatted = _format_for_telegram(response)
            markup = self._build_followup_markup(chat_id)
            sent_markup = False
            for chunk in _split_message(formatted):
                if not chunk.strip():
                    continue
                try:
                    await update.message.reply_text(
                        chunk, parse_mode='Markdown',
                        reply_markup=markup if not sent_markup else None)
                except Exception:
                    await update.message.reply_text(
                        chunk, reply_markup=markup if not sent_markup else None)
                sent_markup = True

            # Flush any range images queued by tool calls during this turn.
            await self._send_pending_range_images(update, chat_id, label)

        except Exception as e:
            elapsed = time.time() - t0
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Token expired during HH follow-up")
                await update.message.reply_text(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            self.log.error(f"[{label}] HH follow-up error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ 分析 {hand_id} 時發生錯誤：{e}")
        finally:
            await _typing.__aexit__(None, None, None)

    async def _send_pending_range_images(self, update: Update, chat_id: int, label: str):
        """Send at most one range grid image (last street with data).

        Uses update.effective_chat so it works for both messages and
        callback queries (where update.message is None).
        """
        chat = update.effective_chat
        try:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if ctx:
                hand = ctx.get("hand", {})
                no_hero = hand.get("no_hero_hand")
                is_icm = ctx.get("is_icm", False)
                # Send range image once after initial analysis:
                # 1. no_hero_hand queries (asking about a position's range)
                # 2. ICM preflop-only spots (push/fold ranges are critical)
                # Flag prevents re-sending on follow-up messages.
                if (no_hero or is_icm) and not ctx.get("_range_img_sent"):
                    hero_pos = hand.get("hero_position", "")
                    # Find the LAST street with solver data
                    last_spot, last_sol = None, None
                    for spot, sol in zip(ctx.get("hero_spots", []),
                                         ctx.get("solutions", [])):
                        if sol is not None:
                            last_spot, last_sol = spot, sol
                    if last_spot and last_sol:
                        from range_image import generate_range_grid
                        st = last_spot.get("street", "").capitalize()
                        board = last_spot.get("params", {}).get("board", "")
                        title = f"{hero_pos} {st}"
                        if board:
                            title += f" | {board}"
                        solver_pos = last_spot.get("solver_hero_pos", hero_pos)
                        img = generate_range_grid(
                            last_sol, solver_pos, title=title)
                        if img:
                            await chat.send_photo(
                                photo=img, caption=f"📊 {title}")
                            ctx["_range_img_sent"] = True

            # Send queued images from tool calls (query_gto with position, no hand)
            pending = self.session_manager.pending_images.pop(chat_id, [])
            if pending:
                # Only send the last one
                img_bytes, caption = pending[-1]
                await chat.send_photo(photo=img_bytes, caption=caption)
        except Exception as e:
            self.log.warning(f"[{label}] Range image failed: {e}")

    def _build_followup_markup(self, chat_id: int,
                               include_gto_link: bool = False) -> InlineKeyboardMarkup | None:
        """Build follow-up question buttons from hand analysis context.

        When include_gto_link is True (image/text analysis flow only — not HH
        batch follow-ups), append a URL button deep-linking to hero's last
        decision node in GTO Wizard's /solutions strategy view.
        """
        try:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if not ctx or ctx.get("_followup_sent"):
                return None
            hand = ctx.get("hand", {})
            hero_hand = hand.get("hero_hand", "")
            hero_pos = hand.get("hero_position", "")
            if not hero_pos:
                return None

            # Use LLM-generated follow-up questions if available
            followups = ctx.get("followup_questions")
            if followups and isinstance(followups, list):
                questions = [q for q in followups if isinstance(q, str)][:3]
            else:
                # Fallback: build from context
                streets = hand.get("streets", [])
                is_icm = ctx.get("is_icm", False)
                questions = []

                opp_pos = None
                for s in streets:
                    for a in s.get("actions", []):
                        if a["position"] != hero_pos:
                            opp_pos = a["position"]
                if opp_pos:
                    if len(streets) > 0:
                        last_street = ["flop", "turn", "river"][
                            min(len(streets) - 1, 2)]
                        questions.append(
                            f"{opp_pos} 在 {last_street} 的範圍是什麼？")
                    else:
                        questions.append(f"{opp_pos} 的範圍是什麼？")

                if hand.get("no_hero_hand"):
                    questions.append(f"{hero_pos} 這個 range 面對 3-bet 要怎麼防守？")
                elif len(streets) > 1:
                    questions.append(
                        f"{hero_hand} 在這裡應該用什麼 size？")
                elif is_icm:
                    questions.append("這個位置的 push 範圍有多寬？")
                else:
                    questions.append(
                        f"{hero_hand} 的 EV 跟其他手牌比如何？")

                if is_icm:
                    questions.append("如果對手更短碼，策略會怎麼變？")
                elif len(streets) >= 2:
                    questions.append("對手 raise 的話應該怎麼打？")
                else:
                    questions.append("這手牌有哪些常見的錯誤打法？")

            questions = questions[:3]

            # Store full questions in context and use short callback IDs.
            # callback_data has a 64-byte limit; Chinese strategy questions
            # often exceed it and used to be truncated before execution.
            button_map = {str(i): q for i, q in enumerate(questions)}
            ctx["_followup_buttons"] = button_map

            keyboard = [
                [InlineKeyboardButton(q, callback_data=f"fq:{i}")]
                for i, q in button_map.items()
            ]

            if include_gto_link:
                gto_url = self._build_gto_solution_url(ctx)
                if gto_url:
                    keyboard.append([InlineKeyboardButton(
                        "🧙 在 GTO Wizard 開啟", url=gto_url)])

            if not keyboard:
                return None

            ctx["_followup_sent"] = True
            return InlineKeyboardMarkup(keyboard)

        except Exception:
            return None

    def _build_gto_link_markup(self, chat_id: int) -> InlineKeyboardMarkup | None:
        """Build a markup with ONLY the "Open in GTO Wizard" deep-link button.

        Attached to the compact GTO summary (📋) message in the image flow so
        the link sits on the analysis card, not the coaching reply.
        """
        try:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if not ctx:
                return None
            gto_url = self._build_gto_solution_url(ctx)
            if not gto_url:
                return None
            return InlineKeyboardMarkup(
                [[InlineKeyboardButton("🧙 在 GTO Wizard 開啟", url=gto_url)]])
        except Exception:
            return None

    def _build_gto_solution_url(self, ctx: dict) -> str | None:
        """Deep-link to hero's last decision node in GTOW /solutions.

        Never raises — a failed build just means no button.
        """
        try:
            from gtow_solution_url import build_last_node_url
            return build_last_node_url(ctx)
        except Exception:
            self.log.debug("GTOW solution URL build failed", exc_info=True)
            return None

    async def handle_followup_button(self, update: Update,
                                     context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button press.

        1. Remove buttons from the original message
        2. Send the question as a visible user message (for history)
        3. Process the question and send the response
        """
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        if not data.startswith("fq:"):
            return
        chat_id = update.effective_chat.id
        key_or_question = data[3:]
        ctx = self.session_manager.hand_contexts.get(chat_id, {})
        question = (ctx.get("_followup_buttons", {}) or {}).get(key_or_question, key_or_question)
        user_id = update.effective_user.id
        label = f"followup-{chat_id}"
        self.log.info(f"[{label}] Button: {question}")

        # Remove buttons from original message
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        # Send question as a visible "user" message so it appears in chat
        await context.bot.send_message(
            chat_id, f"💬 {question}",
            read_timeout=10, write_timeout=10, connect_timeout=10)

        # Process the question
        raw_status = await context.bot.send_message(
            chat_id, "⏳ 查詢中...",
            read_timeout=10, write_timeout=10, connect_timeout=10)
        status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

        async def _on_status(msg: str):
            await status_msg.edit_text(f"⏳ {msg}")

        refresh_token = await self._get_user_refresh_token(user_id)
        try:
            response = await self.session_manager.send_message(
                chat_id, question, on_status=_on_status,
                user_id=user_id, refresh_token=refresh_token,
            )
            await status_msg.delete()
            if response and response.strip():
                formatted = _format_for_telegram(response)
                for chunk in _split_message(formatted):
                    if not chunk.strip():
                        continue
                    try:
                        await context.bot.send_message(
                            chat_id, chunk, parse_mode='Markdown',
                            read_timeout=30, write_timeout=30,
                            connect_timeout=30)
                    except Exception:
                        await context.bot.send_message(
                            chat_id, _strip_markdown(chunk),
                            read_timeout=30, write_timeout=30,
                            connect_timeout=30)

            # Flush any range images queued by tool calls during this turn.
            # Without this, images get stranded and bleed into the next message.
            await self._send_pending_range_images(update, chat_id, label)
        except Exception as e:
            self.log.error(f"[{label}] Error: {e}", exc_info=True)
            try:
                await status_msg.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id, "抱歉，處理問題時出錯了。")

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded photos (poker screenshots for GTO analysis)."""
        await self._touch_user(update)
        user_id = update.effective_user.id
        if not await self._has_gto_token(user_id):
            await self._send_token_gate(update)
            return
        chat_id = update.effective_chat.id
        async with self._user_lock(chat_id):
            await self._handle_photo_inner(update, context)

    async def _handle_photo_inner(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        label = self._user_label(update)
        user_id = update.effective_user.id
        caption = update.message.caption or ""

        self.log.info(f"[{label}] Photo received, caption: {caption[:200]}")

        if self.db:
            try:
                await self.db.log_message(update.effective_chat.id, "photo")
            except Exception:
                pass

        # Get the largest photo resolution
        photo = update.message.photo[-1]

        raw_status = await _send_status(update.message, "🔍 正在下載圖片...")
        status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

        # Look up user's GTO token
        refresh_token = await self._get_user_refresh_token(user_id)

        t0 = time.time()
        image_bytes = await self._download_telegram_image(photo, label, status_msg)
        if image_bytes is None:
            return

        await self._run_image_analysis(
            update,
            label=label, user_id=user_id, caption=caption,
            image_bytes=image_bytes, mime_type="image/jpeg",
            status_msg=status_msg, refresh_token=refresh_token, t0=t0,
        )

    async def _download_telegram_image(self, source, label: str, status_msg) -> bytes | None:
        """Download bytes from a Telegram PhotoSize or Document with retry.

        Returns None when all attempts timed out (status message updated).
        """
        for attempt in range(3):
            try:
                tg_file = await source.get_file(
                    read_timeout=30, write_timeout=30, connect_timeout=30,
                )
                return bytes(await tg_file.download_as_bytearray(
                    read_timeout=30, write_timeout=30, connect_timeout=30,
                ))
            except telegram.error.TimedOut:
                delay = 3 * (attempt + 1)
                self.log.warning(
                    f"[{label}] Image download timed out (attempt {attempt+1}/3), "
                    f"retry in {delay}s"
                )
                if attempt < 2:
                    await asyncio.sleep(delay)
        await status_msg.edit_text("❌ 圖片下載失敗（Telegram 超時），請稍後再試。")
        return None

    async def _run_image_analysis(
        self, update: Update, *,
        label: str, user_id: int, caption: str,
        image_bytes: bytes, mime_type: str,
        status_msg, refresh_token, t0: float,
    ):
        """Run the shared image-analysis pipeline.

        Used by both _handle_photo_inner (image-as-photo) and the image
        branch of _handle_document_inner (image-as-file).
        """
        gto_sent = False

        async def send_gto_summary(text: str):
            nonlocal gto_sent
            await status_msg.delete()
            # Put the "Open in GTO Wizard" deep-link on the analysis card (📋),
            # not the coaching reply that follows.
            gto_markup = self._build_gto_link_markup(update.effective_chat.id)
            await _send_reply(update.message, text, self.log, label,
                              reply_markup=gto_markup)
            gto_sent = True

        try:
            async with _TypingLoop(update.message.chat):
                response = await self.session_manager.send_image_message(
                    chat_id=update.effective_chat.id,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    user_text=caption,
                    status_callback=status_msg.edit_text,
                    send_gto_callback=send_gto_summary,
                    user_id=user_id,
                    refresh_token=refresh_token,
                )

            elapsed = time.time() - t0
            self.log.info(f"[{label}] Photo response OK ({elapsed:.1f}s)")

            if not gto_sent:
                await status_msg.delete()

            if not response or not response.strip():
                if not gto_sent:
                    await update.message.reply_text("抱歉，無法分析截圖，請重新發送。")
                return
            # GTO Wizard link rides on the 📋 summary card; coaching reply only
            # carries the follow-up question buttons. Fall back to putting the
            # link on the coaching reply if the summary card never went out.
            markup = self._build_followup_markup(update.effective_chat.id,
                                                 include_gto_link=not gto_sent)
            await _send_reply(update.message, response, self.log, label,
                              reply_markup=markup)

            # Send range grid images
            await self._send_pending_range_images(update, update.effective_chat.id, label)

        except Exception as e:
            elapsed = time.time() - t0
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Token expired during photo analysis")
                await status_msg.edit_text(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            self.log.error(f"[{label}] Photo error after {elapsed:.1f}s: {e}", exc_info=True)
            err_msg = f"❌ 分析截圖時發生錯誤：{e}"
            if gto_sent:
                # status_msg was deleted after sending GTO summary;
                # send a new reply so the user sees the error.
                await update.message.reply_text(err_msg)
            else:
                await status_msg.edit_text(err_msg)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle uploaded hand history files (.txt or .zip)."""
        await self._touch_user(update)
        user_id = update.effective_user.id
        if not await self._has_gto_token(user_id):
            await self._send_token_gate(update)
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
        caption = update.message.caption or ""
        self.log.info(f"[{label}] Document: {fname} ({fsize} bytes), caption: {caption[:100]}")

        if self.db:
            try:
                await self.db.log_message(update.effective_chat.id, "document")
            except Exception:
                pass

        fname_lower = fname.lower()

        # Image-as-file: Telegram delivers uncompressed/HEIC screenshots as
        # Document, not Photo. Route to the same analysis pipeline as photos
        # instead of rejecting with the hand-history prompt.
        if _is_image_document(doc.mime_type, fname_lower):
            raw_status = await _send_status(update.message, "🔍 正在下載圖片...")
            status_msg = _ResilientStatus(raw_status, log=self.log, label=label)
            refresh_token = await self._get_user_refresh_token(user_id)
            t0 = time.time()
            image_bytes = await self._download_telegram_image(doc, label, status_msg)
            if image_bytes is None:
                return
            await self._run_image_analysis(
                update,
                label=label, user_id=user_id, caption=caption,
                image_bytes=image_bytes,
                mime_type=doc.mime_type or "image/jpeg",
                status_msg=status_msg, refresh_token=refresh_token, t0=t0,
            )
            return

        # Parse optional ICM override from caption (e.g., "10000" or "10000 200")
        # When starting_stack=0, analyze_hands auto-detects from first hand's hero chips
        starting_stack = 0
        tournament_size = 1000
        caption_numbers = re.findall(r'\d+', caption)
        if caption_numbers:
            starting_stack = int(caption_numbers[0])
            if len(caption_numbers) >= 2:
                ts = int(caption_numbers[1])
                if ts in (200, 1000):
                    tournament_size = ts

        # Validate file type
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
        raw_status = await _send_status(update.message, "📥 下載檔案中...")
        status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

        refresh_token = None  # set later inside try block
        t0 = time.time()
        try:
            # Download file (retry on Telegram timeout)
            tg_file = await _tg_retry(
                lambda: doc.get_file(
                    read_timeout=30, write_timeout=30, connect_timeout=30,
                ),
                label=label, log=self.log,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                download_path = tmpdir_path / fname
                await _tg_retry(
                    lambda: tg_file.download_to_drive(
                        str(download_path),
                        read_timeout=30, write_timeout=30, connect_timeout=30,
                    ),
                    label=label, log=self.log,
                )

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
                    f"🔍 解析到 {len(all_hands)} 手，開始 GTO 分析（含 ICM）..."
                )

                # Require user's GTO token
                refresh_token = await self._get_user_refresh_token(user_id)
                if not refresh_token:
                    await status_msg.edit_text("請先使用 /settoken 綁定你的 GTO Wizard 帳號。")
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

                _starting_stack = starting_stack
                _tournament_size = tournament_size

                def _run_in_thread():
                    _setup(_user_id, _refresh_token)
                    try:
                        return analyze_hands(all_hands, delay=0.3, on_progress=sync_progress,
                                             starting_stack=_starting_stack,
                                             tournament_size=_tournament_size)
                    finally:
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
                await _send_reply(update.message, report, self.log, label)

        except Exception as e:
            elapsed = time.time() - t0
            # Handle token expiry gracefully
            from gto_token import TokenExpiredError
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Token expired during HH upload")
                await status_msg.edit_text(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            self.log.error(f"[{label}] HH upload error after {elapsed:.1f}s: {e}", exc_info=True)
            await status_msg.edit_text(f"❌ 分析時發生錯誤：{e}")

    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report — admin-only analytics report."""
        user_id = update.effective_user.id
        if not self.admin_chat_id or user_id != self.admin_chat_id:
            return
        self.log.info(f"[{self._user_label(update)}] /report")
        if not self.db:
            await update.message.reply_text("Database not connected.")
            return
        try:
            m = await self.db.get_analytics_metrics()
            def _fmt_tokens(n):
                """Format token count: 1234567 → 1.2M, 12345 → 12.3K."""
                if n >= 1_000_000:
                    return f"{n / 1_000_000:.1f}M"
                if n >= 1_000:
                    return f"{n / 1_000:.1f}K"
                return str(n)

            text = (
                "📊 *數據報告*\n"
                "\n"
                f"*用戶*：共 {m['users_total']} 人，"
                f"{m['users_with_token']} 人已綁定 token\n"
                f"  新增：今日 {m['users_new_today']}，"
                f"本週 {m['users_new_week']}\n"
                "\n"
                f"*活躍*：今日 {m['active_today']}，"
                f"本週 {m['active_week']}\n"
                "\n"
                f"*對話*：今日 {m['messages_today']}，"
                f"本週 {m['messages_week']}，"
                f"累計 {m['messages_total']}\n"
                "\n"
                f"*手牌*：今日 {m['hands_today']}，"
                f"本週 {m['hands_week']}，"
                f"累計 {m['hands_total']}\n"
                "\n"
                f"*Gemini Token*：今日 {_fmt_tokens(m['tokens_today'])}，"
                f"本週 {_fmt_tokens(m['tokens_week'])}\n"
                f"  API 呼叫：今日 {m['api_calls_today']}，"
                f"本週 {m['api_calls_week']}\n"
                f"  輸入 {_fmt_tokens(m['prompt_tokens_today'])} / "
                f"輸出 {_fmt_tokens(m['completion_tokens_today'])} / "
                f"快取 {_fmt_tokens(m['cached_tokens_today'])} / "
                f"思考 {_fmt_tokens(m['thinking_tokens_today'])}\n"
                "\n"
                f"*GTO 快取*：{m['cache_total']} 筆"
            )
            await update.message.reply_text(text, parse_mode="Markdown")
        except Exception as e:
            self.log.error(f"Failed to generate report: {e}", exc_info=True)
            await update.message.reply_text(f"❌ 報告產生失敗：{e}")

    def setup_handlers(self, post_init=None, post_shutdown=None):
        """Setup bot handlers"""
        if not self.application:
            request = HTTPXRequest(read_timeout=30, write_timeout=30, connect_timeout=30)
            get_updates_request = HTTPXRequest(read_timeout=60, write_timeout=30, connect_timeout=30)
            builder = (
                Application.builder()
                .token(self.token)
                .request(request)
                .get_updates_request(get_updates_request)
            )
            if post_init:
                builder = builder.post_init(post_init)
            if post_shutdown:
                builder = builder.post_shutdown(post_shutdown)
            self.application = builder.build()

        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("settoken", self.settoken_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        self.application.add_handler(CallbackQueryHandler(self.handle_followup_button))

        return self.application

    def run(self, post_init=None, post_shutdown=None):
        """Run the bot (blocking — manages its own event loop)."""
        app = self.setup_handlers(post_init=post_init, post_shutdown=post_shutdown)
        self.log.info(f"Bot starting — model={self.session_manager.model}, max_turns={self.session_manager.max_turns}")
        app.run_polling(drop_pending_updates=False)


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
    # Sanitize unmatched markdown markers that Telegram would reject.
    # For each marker char, if count is odd, escape the last occurrence.
    for ch in ('*', '_', '`'):
        # Count occurrences outside of paired markers
        count = text.count(ch)
        if count % 2 == 1:
            # Escape the last lone occurrence
            idx = text.rfind(ch)
            text = text[:idx] + '\\' + text[idx:]
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


async def _tg_retry(coro_fn, retries=3, label="tg_retry", log=None):
    """Retry a Telegram API call on TimedOut/NetworkError.

    coro_fn: zero-arg callable returning an awaitable (called fresh each attempt).
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_fn()
        except (telegram.error.TimedOut, telegram.error.NetworkError) as e:
            last_exc = e
            delay = 3 * (attempt + 1)
            if log:
                log.warning(f"[{label}] Telegram call failed (attempt {attempt+1}/{retries}): {e}, retry in {delay}s")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    raise last_exc


class _TypingLoop:
    """Send 'typing' action every 4s until stopped. Use as async context manager."""

    def __init__(self, chat):
        self._chat = chat
        self._task = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            try:
                await self._chat.send_action(action="typing")
            except Exception:
                pass
            await asyncio.sleep(4)


class _ResilientStatus:
    """Wraps a Telegram status message so edit_text never kills the caller."""

    def __init__(self, msg, log=None, label="status"):
        self._msg = msg
        self._log = log
        self._label = label

    async def edit_text(self, text, **kwargs):
        try:
            await _tg_retry(
                lambda: self._msg.edit_text(
                    text, read_timeout=15, write_timeout=15, connect_timeout=15,
                    **kwargs,
                ),
                retries=2, label=self._label, log=self._log,
            )
        except Exception:
            if self._log:
                self._log.debug(f"[{self._label}] Status edit failed (non-fatal): {text[:60]}")

    async def delete(self):
        try:
            await self._msg.delete()
        except Exception:
            pass


async def _send_status(message, text: str):
    """Send a status message with retry on timeout."""
    for attempt in range(3):
        try:
            return await message.reply_text(text,
                                            read_timeout=15, write_timeout=15, connect_timeout=15)
        except Exception:
            if attempt < 2:
                await asyncio.sleep(2)
    # Last resort — try once more, let exception propagate if it fails
    return await message.reply_text(text,
                                    read_timeout=30, write_timeout=30, connect_timeout=30)


async def _send_reply(message, text: str, log: logging.Logger, label: str,
                      reply_markup=None) -> None:
    """Send a formatted reply with Markdown fallback and timeout retry.

    1. Try Markdown parse_mode
    2. If Markdown fails, strip formatting and send plain text
    3. If any send times out, retry up to 3 times with increasing delays

    If reply_markup is provided, it's attached to the LAST chunk only.
    """
    formatted = _format_for_telegram(text)
    chunks = [c for c in _split_message(formatted) if c.strip()]
    for i, chunk in enumerate(chunks):
        is_last = (i == len(chunks) - 1)
        markup = reply_markup if is_last else None
        sent = False
        # Try Markdown first
        try:
            await message.reply_text(chunk, parse_mode='Markdown',
                                     reply_markup=markup,
                                     read_timeout=30, write_timeout=30, connect_timeout=30)
            sent = True
        except telegram.error.TimedOut:
            log.warning(f"[{label}] Markdown send timed out")
        except Exception:
            log.warning(f"[{label}] Markdown parse failed")
        # Fallback to plain text with retries
        if not sent:
            plain = _strip_markdown(chunk)
            for attempt in range(3):
                try:
                    await message.reply_text(plain, reply_markup=markup,
                                             read_timeout=30, write_timeout=30, connect_timeout=30)
                    sent = True
                    break
                except telegram.error.TimedOut:
                    delay = 2 * (attempt + 1)
                    log.warning(f"[{label}] Plain text send timed out (attempt {attempt+1}/3), retry in {delay}s")
                    await asyncio.sleep(delay)
        if not sent:
            log.error(f"[{label}] Failed to send message after all retries")


def _strip_markdown(text: str) -> str:
    """Remove Markdown formatting characters for plain-text fallback."""
    import re
    text = re.sub(r'\\([*_`])', r'\1', text)  # unescape first
    text = text.replace('*', '').replace('_', '').replace('`', '')
    return text
