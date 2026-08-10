# src/telegram_bot/bot.py
from __future__ import annotations

import asyncio
import json
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

from card_display import cards_to_emoji

# Upload analysis timeout (seconds)
ANALYSIS_TIMEOUT = 1800

# Telegram message limit
MAX_MESSAGE_LENGTH = 4096
QUEUE_PAGE_SIZE = 10
QUEUE_SOURCE_PAGE_SIZE = 8
LIVE_SESSION_LIST_LIMIT = 8
ONLINE_SESSION_LIST_LIMIT = 8

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def _estimate_live_batch_minutes(hand_count: int) -> tuple[int, int]:
    """Return a conservative ETA range for /live batch solver grading.

    Observed throughput is roughly 6-12 hands/minute for typical live
    shorthand batches, so 12 hands should read as about 1-2 minutes rather
    than the old per-hand minute estimate.
    """
    n = max(1, hand_count)
    low = max(1, (n + 11) // 12)
    high = max(2, (n + 5) // 6)
    return low, high


def _queue_payload(rows, *, page: int = 0,
                   total: int | None = None) -> tuple[str, list[list[dict]]]:
    """Render the current practice queue for both /queue and qcl refresh."""
    total = len(rows) if total is None else int(total)
    if total <= 0:
        return "📥 練習佇列已清空 — 沒有待練的行動線。", []
    from html import escape as _esc
    pages = max(1, (total + QUEUE_PAGE_SIZE - 1) // QUEUE_PAGE_SIZE)
    page = max(0, min(int(page), pages - 1))
    if pages > 1:
        heading = f"📥 <b>練習佇列</b>（{total} 項，第 {page + 1}/{pages} 頁）"
    else:
        heading = f"📥 <b>練習佇列</b>（{total} 項）"
    L = [heading, ""]
    buttons: list[list[dict]] = []
    start = page * QUEUE_PAGE_SIZE
    for local_i, r in enumerate(rows, 1):
        i = start + local_i
        if r["kind"] == "review":
            lbl = r["label"] or r["spot_leaf"]
        else:
            from spot_naming import compact_spot_name
            lbl = compact_spot_name(r)
        st = "（本週課表內）" if r["status"] == "prescribed" else ""
        if r["kind"] == "review":
            L.append(f"🔍 {i}. {_esc(lbl)}{st}")
            row_btns = []
            anchor = r.get("review_anchor_url")
            anchor_street = r.get("review_anchor_street")
            if anchor:
                row_btns.append({"text": f"↩ {i} {(anchor_street or '上游').title()}",
                                 "url": anchor})
            if r["drill_url"]:
                text = f"💥 {i} 損失" if anchor else f"🔗 復盤 {i}"
                from gtow_trainer_url import apply_trainer_defaults
                row_btns.append({"text": text,
                                 "url": apply_trainer_defaults(r["drill_url"])})
            actions = [
                {"text": f"📚 {i} 來源", "callback_data": f"qsrc:{r['id']}"},
                {"text": f"✔ {i} 完成", "callback_data": f"qcl:{r['id']}:{page}"},
                {"text": f"➕ {i} 加練", "callback_data": f"qex:{r['id']}"},
            ]
            if anchor and row_btns:
                buttons.append(row_btns)
                row_btns = actions
            else:
                row_btns.extend(actions)
        else:
            from spot_naming import telegram_bias_summary
            ev = r["total_ev_loss_bb"] or 0
            # 手動加練（➕加練 / 復盤排入）不設 EV 門檻，可能是 0bb 的
            # non-leak spot；標「（手動加入）」讓 0.0bb 不被誤讀為算錯。
            manual = "（手動加入）" if r.get("added_by") == "manual" else ""
            L.append(f"🎯 {i}. {_esc(lbl)} — 來自 {r['n_sources']} 手，累計損失 {ev:.1f}bb{st}{manual}")
            bias = telegram_bias_summary(r)
            if bias:
                L.append(f"   ↳ {_esc(bias)}")
            row_btns = [{"text": f"🎯 詳細／練習 {i}",
                         "callback_data": f"qdet:{r['id']}:{page}"}]
            row_btns.append({"text": f"📚 {i} 來源",
                             "callback_data": f"qsrc:{r['id']}"})
            row_btns.append({"text": f"✔ {i} 完成",
                             "callback_data":
                                 f"qcl:{r['id']}:{page}:completed"})
        buttons.append(row_btns)
    if pages > 1:
        nav = []
        if page > 0:
            nav.append({"text": "⬅ 上一頁", "callback_data": f"qpg:{page - 1}"})
        if page + 1 < pages:
            nav.append({"text": "下一頁 ➡", "callback_data": f"qpg:{page + 1}"})
        if nav:
            buttons.append(nav)
    return "\n".join(L), buttons


def _recent_live_sessions_payload(sessions: list[dict]
                                  ) -> tuple[str, list[list[dict]]]:
    """Render the persisted /live report index and resend callbacks."""
    if not sessions:
        return "🃏 還沒有線下 session — 先用 /live 匯入現場手牌。", []

    from html import escape as _esc
    lines = ["🃏 <b>最近線下 Sessions</b>", "點按鈕可重新傳送該場復盤列表。", ""]
    buttons: list[list[dict]] = []
    for i, session in enumerate(sessions, 1):
        result = session.get("result") or {}
        totals = result.get("totals") or {}
        hands = int(totals.get("hands") or len(result.get("hands") or []))
        mistakes = int(totals.get("mistakes") or 0)
        created_at = session.get("created_at")
        created_label = ""
        if created_at:
            from zoneinfo import ZoneInfo
            local_created = created_at.astimezone(ZoneInfo("Asia/Taipei"))
            created_label = local_created.strftime("%H:%M")
        raw_date = str(result.get("date") or "")
        date_match = re.fullmatch(r"\d{4}-(\d{1,2})-(\d{1,2})", raw_date)
        if date_match:
            date_label = f"{int(date_match.group(1))}/{int(date_match.group(2))}"
        elif raw_date:
            date_label = raw_date
        elif created_at:
            date_label = local_created.strftime("%-m/%-d")
        else:
            date_label = "日期未知"
        stamp = f"{date_label} {created_label}".strip()
        lines.append(
            f"{i}. <b>{_esc(stamp)}</b> · {hands} 手 · {mistakes} 個偏差")
        buttons.append([{
            "text": f"↩ {i}　{stamp}（{hands} 手）",
            "callback_data": f"lvs:{session['id']}",
        }])
    return "\n".join(lines), buttons


def _recent_online_sessions_payload(sessions: list[dict]
                                    ) -> tuple[str, list[list[dict]]]:
    """Render recent online ledger sessions with stable summary callbacks."""
    if not sessions:
        return ("♠️ 還沒有線上 session — 先用 ♠ 同步手牌或 /ingest。", [])

    from html import escape as _esc
    from session_review import _session_span, session_callback_key

    lines = ["♠️ <b>最近線上 Sessions</b>",
             "點按鈕可重新產生並傳送該場復盤 summary。", ""]
    buttons: list[list[dict]] = []
    for i, session in enumerate(sessions, 1):
        span = _session_span(session["started_at"], session["ended_at"])
        hands = int(session.get("hands_count") or 0)
        tables = int(session.get("max_concurrent_tables") or 1)
        lines.append(
            f"{i}. <b>{_esc(span)}</b> · {hands} 手 · 最多 {tables} 桌")
        buttons.append([{
            "text": f"↩ {i}　{span}（{hands} 手）",
            "callback_data": f"ors:{session_callback_key(session)}",
        }])
    return "\n".join(lines), buttons


def _queue_source_payload(queue_id: int, label: str, sources: list[dict],
                          *, page: int = 0, queue_page: int = 0,
                          kind: str = "drill"
                          ) -> tuple[str, list[list[dict]]]:
    """Render exact online links + lightweight live-hand callbacks."""
    from html import escape as _esc
    from queue_feed import (QUEUE_SOURCE_HANDS_PER_LINK,
                            gtow_analyze_hands_urls)

    online = [source for source in sources if source.get("source") == "online"]
    live = [source for source in sources if source.get("source") == "live"]
    missing = [source for source in sources if source.get("source") == "missing"]
    action_rows: list[list[dict]] = []

    online_ids = [source["hand_id"] for source in online]
    online_urls = gtow_analyze_hands_urls(online_ids)
    for index, (url, chunk) in enumerate(online_urls):
        start = index * QUEUE_SOURCE_HANDS_PER_LINK + 1
        end = start + len(chunk) - 1
        if len(online_urls) == 1:
            text = f"🌐 線上實際牌局（{len(chunk)}）"
        else:
            text = f"🌐 線上實際牌局 {start}–{end} / {len(online_ids)}"
        action_rows.append([{"text": text, "url": url}])

    for source in live:
        played_at = source.get("played_at")
        if hasattr(played_at, "astimezone"):
            from queue_feed import TPE
            date_text = played_at.astimezone(TPE).strftime("%-m/%-d")
        elif hasattr(played_at, "strftime"):
            date_text = played_at.strftime("%-m/%-d")
        else:
            date_text = "線下"
        detail = " ".join(part for part in (
            date_text, source.get("position"), source.get("hero_hand")) if part)
        ev = float(source.get("ev_loss_bb") or 0.0)
        text = f"🎴 {detail} · 損失 {ev:.1f}bb"[:60]
        action_rows.append([{
            "text": text,
            "callback_data":
                f"qraw:{queue_id}:{page}:{queue_page}:{source['hand_id']}",
        }])

    pages = max(1, (len(action_rows) + QUEUE_SOURCE_PAGE_SIZE - 1)
                // QUEUE_SOURCE_PAGE_SIZE)
    page = max(0, min(int(page), pages - 1))
    heading = "📚 <b>來源牌局</b>"
    if pages > 1:
        heading += f"（第 {page + 1}/{pages} 頁）"
    counts = f"線上 {len(online)} 手、線下 {len(live)} 手"
    if missing:
        counts += f"、缺資料 {len(missing)} 手"
    html = (f"{heading}\n{_esc(label or str(queue_id))}\n"
            f"{counts}\n線上以實際來源牌局分組；線下保留原始紀錄。")
    if online_ids:
        html += ("\n線上只列這個項目的實際來源牌局，"
                 f"每組最多 {QUEUE_SOURCE_HANDS_PER_LINK} 手。")
    buttons = action_rows[page * QUEUE_SOURCE_PAGE_SIZE:
                          (page + 1) * QUEUE_SOURCE_PAGE_SIZE]
    if pages > 1:
        nav = []
        if page > 0:
            nav.append({"text": "⬅ 上一頁",
                        "callback_data":
                            f"qsrc:{queue_id}:{page - 1}:{queue_page}"})
        if page + 1 < pages:
            nav.append({"text": "下一頁 ➡",
                        "callback_data":
                            f"qsrc:{queue_id}:{page + 1}:{queue_page}"})
        buttons.append(nav)
    if kind == "drill":
        buttons.append([{
            "text": "⬅ 返回練習詳情",
            "callback_data": f"qdet:{queue_id}:{queue_page}",
        }])
    else:
        buttons.append([{
            "text": "⬅ 返回 Queue",
            "callback_data": f"qpg:{queue_page}",
        }])
    return html, buttons


def _queue_drill_detail_payload(item: dict, binding, lifetime, attempt,
                                *, page: int = 0) -> tuple[str, list[list[dict]]]:
    """Render one queue prescription after its GTOW Drill is ensured."""
    from html import escape as _esc
    from spot_naming import compact_spot_name, telegram_bias_summary

    label = compact_spot_name(item)
    bias = telegram_bias_summary(item)
    bias_line = f"• {_esc(bias)}\n" if bias else ""
    target_hands = int(item.get("gtow_target_hands") or 30)
    target_score = float(item.get("gtow_target_score") or 0.90)
    passed = (attempt.total_hands >= target_hands
              and attempt.gto_score >= target_score)
    link_state = "剛建立" if binding.created else "已連結既有"
    lifetime_score = (f"{lifetime.gto_score * 100:.1f}%"
                      if lifetime.total_hands else "—")
    attempt_score = (f"{attempt.gto_score * 100:.1f}%"
                     if attempt.total_hands else "—")
    status = "✅ 本次 Drill 已達標" if passed else "⏳ 尚未達標"
    html = (
        f"🎯 <b>{_esc(label)}</b>\n\n"
        f"<b>處方來源</b>\n"
        f"• 來自 {int(item.get('n_sources') or 0)} 手真實對局\n"
        f"• 累計 EV loss：{float(item.get('total_ev_loss_bb') or 0):.1f}bb\n\n"
        f"{bias_line}"
        f"<b>GTOW Drill</b>\n"
        f"• {link_state}：{_esc(binding.name)}\n"
        f"• 歷史累計：{lifetime.total_hands} hands / "
        f"{lifetime.played_moves} decisions\n"
        f"• 歷史 Score：{lifetime_score}\n"
        f"• 歷史 EV loss：{lifetime.total_ev_loss_bb:.2f}bb\n\n"
        f"<b>本次處方</b>\n"
        f"• {attempt.sessions} sessions · {attempt.total_hands}/{target_hands} hands"
        f" · {attempt.played_moves} decisions\n"
        f"• Score：{attempt_score}（目標 ≥{target_score * 100:.0f}%）\n"
        f"• EV loss：{attempt.total_ev_loss_bb:.2f}bb\n"
        f"• {status}\n\n"
        "門檻只用來標示是否達標；即使未達標，也可以隨時完成。"
    )
    qid = int(item["id"])
    buttons = [
        [{"text": "🎯 開始練習", "url": item["drill_url"]}],
        [
            {"text": "🔄 更新成績", "callback_data": f"qdst:{qid}:{page}"},
            {"text": "📚 來源牌局",
             "callback_data": f"qsrc:{qid}:0:{page}"},
        ],
        [
            {"text": "✔ 完成",
             "callback_data": f"qcl:{qid}:{page}:completed"},
            {"text": "⬅ 返回 Queue", "callback_data": f"qpg:{page}"},
        ],
    ]
    return html, buttons


async def _present_queue_detail(query, context, chat_id: int, html: str,
                                markup, *, new_message: bool = False):
    """Keep the weekly plan immutable while retaining in-place queue detail."""
    kwargs = {"parse_mode": "HTML", "disable_web_page_preview": True,
              "reply_markup": markup}
    if new_message:
        await context.bot.send_message(chat_id, html, **kwargs)
    else:
        try:
            await query.edit_message_text(html, **kwargs)
        except telegram.error.BadRequest as exc:
            if "Message is not modified" not in str(exc):
                raise


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
        # chats whose NEXT text message is a /live hand batch
        self._live_pending: set[int] = set()
        # chats whose NEXT text message replaces one hand in a live session:
        # chat_id -> (owner_user_id, session_id, hand_idx)
        self._live_resend_pending: dict[int, tuple[int, int, int]] = {}

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
            "推薦：安裝 Chrome Extension 自動同步。\n"
            "1. 下載最新版：github.com/a00012025/ai-poker-wizard/releases\n"
            "2. 回到 Telegram 輸入 /pair 取得五分鐘配對碼\n"
            "3. 在 Extension popup 輸入配對碼並登入 GTOW\n\n"
            "手動備援：登入 app.gtowizard.com → F12 Console → "
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
            "安裝後輸入 /pair，把配對碼貼到 Extension popup。\n"
            "之後每次登入 GTO Wizard 都會自動同步，不必再貼 token。\n\n"
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
            "/pair — 配對 Chrome Extension\n"
            "/devices — 查看已配對裝置\n"
            "/revoke — 撤銷指定裝置\n"
            "/settoken — 手動綁定 token（備援）\n"
            "/logout — 解除 token 並撤銷所有裝置\n"
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
/pair - 產生 Chrome Extension 配對碼
/devices - 查看已配對裝置
/revoke 裝置ID - 撤銷裝置
/settoken - 手動 token 備援
/logout - 解除 token 並撤銷同步裝置
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

    async def pair_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create a short-lived code that pairs the Chrome Extension."""
        from datetime import datetime, timedelta, timezone
        from src.token_sync import (
            PAIR_TTL_MINUTES,
            generate_pair_code,
            hash_pair_code,
        )

        if update.effective_chat.type != "private":
            await update.message.reply_text("為保護配對碼，請私訊 Bot 後再輸入 /pair。")
            return
        await self._touch_user(update)
        if not self.db:
            await update.message.reply_text("配對服務目前不可用，請稍後再試。")
            return
        try:
            code = generate_pair_code()
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=PAIR_TTL_MINUTES)
            await self.db.create_gtow_device_pairing(
                update.effective_user.id,
                hash_pair_code(code),
                expires_at,
            )
        except Exception:
            self.log.exception("Failed to create GTOW device pairing")
            await update.message.reply_text("配對服務設定尚未完成，請聯絡管理員。")
            return
        await update.message.reply_text(
            "🔗 Chrome Extension 配對碼\n\n"
            f"`{code}`\n\n"
            f"請在 {PAIR_TTL_MINUTES} 分鐘內貼到 Extension popup。"
            "配對碼只能使用一次；之後 GTOW token 會自動同步。",
            parse_mode="Markdown",
        )

    async def devices_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List active Chrome Extension pairings."""
        from src.token_sync import short_device_id
        from telegram.helpers import escape_markdown

        if update.effective_chat.type != "private":
            await update.message.reply_text("裝置管理僅限私訊 Bot 使用。")
            return

        if not self.db:
            await update.message.reply_text("裝置服務目前不可用。")
            return
        rows = await self.db.list_gtow_sync_devices(update.effective_user.id)
        if not rows:
            await update.message.reply_text("目前沒有已配對的 Chrome Extension。輸入 /pair 開始配對。")
            return
        lines = ["🧩 已配對裝置", ""]
        for row in rows:
            last_sync = row["last_sync_at"]
            sync_text = last_sync.strftime("%Y-%m-%d %H:%M UTC") if last_sync else "尚未同步"
            lines.append(
                f"• {escape_markdown(row['name'], version=1)}\n"
                f"  ID: `{short_device_id(row['id'])}` · {sync_text}"
            )
        lines.append("\n撤銷：`/revoke 裝置ID`")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def revoke_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Revoke one Chrome Extension device by its displayed UUID prefix."""
        if update.effective_chat.type != "private":
            await update.message.reply_text("裝置管理僅限私訊 Bot 使用。")
            return
        if not self.db:
            await update.message.reply_text("裝置服務目前不可用。")
            return
        prefix = "".join(context.args).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8,32}", prefix):
            await update.message.reply_text("請使用 `/revoke 裝置ID`。裝置 ID 可從 /devices 查看。", parse_mode="Markdown")
            return
        row = await self.db.revoke_gtow_sync_device(update.effective_user.id, prefix)
        if not row:
            await update.message.reply_text("找不到唯一對應的有效裝置，請重新查看 /devices。")
            return
        await update.message.reply_text(f"已撤銷裝置：{row['name']}")

    async def settoken_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settoken <token> — validate and store user's GTO Wizard token."""
        if update.effective_chat.type != "private":
            await update.message.reply_text("Token 綁定僅限私訊 Bot 使用，請勿在群組貼上 token。")
            return
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /settoken")

        # Extract token from command args
        raw_text = update.message.text or ""
        token = " ".join(context.args) if context.args else ""
        self.log.info(f"[{label}] /settoken raw len={len(raw_text)}, token len={len(token)}")
        if not token or not token.startswith("eyJ"):
            self.log.warning(f"[{label}] /settoken bad format: token_empty={not token}, raw='{raw_text[:60]}'")
            await update.message.reply_text(
                "格式錯誤。請使用書籤工具複製指令，或手動輸入：\n"
                "`/settoken eyJhbG...`",
                parse_mode='Markdown',
            )
            return

        # Validate: try refreshing the token
        from gto_signing import generate_keypair_jwk
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

        signing_keypair = generate_keypair_jwk()
        access = _refresh_access(token, signing_keypair)
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
            self.log.warning(f"[{label}] /settoken refresh failed code={code} reason={reason}")

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

        # Store in DB (manual /settoken force-overrides any stored token —
        # it was just validated against GTOW above)
        if self.db:
            await self.db.save_user_gto_token(
                user_id,
                token,
                access_token=access,
                signing_keypair=signing_keypair,
            )

        self.log.info(f"[{label}] GTO token bound successfully")
        await update.message.reply_text("GTO Wizard 帳號綁定成功！之後的查詢會使用你的帳號。")

    async def logout_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logout — remove user's GTO Wizard token."""
        label = self._user_label(update)
        user_id = update.effective_user.id
        self.log.info(f"[{label}] /logout")

        if self.db:
            await self.db.logout_gtow_user(user_id)

        from gto_credentials import invalidate_synced_credentials
        from gto_token import invalidate_user_token
        invalidate_user_token(user_id)
        invalidate_synced_credentials(user_id)

        await update.message.reply_text("已解除 GTO Wizard token，並撤銷所有 Extension 同步裝置。")

    async def _get_user_refresh_token(self, user_id: int) -> str | None:
        """Look up user's GTO Wizard refresh token from DB."""
        if not self.db:
            return None
        return await self.db.get_user_gto_token(user_id)

    @staticmethod
    def _setup_user_token(user_id: int, refresh_token: str):
        """Set thread-local GTO token for the current thread."""
        from gto_credentials import get_user_credentials
        from gto_api import set_user_token
        credentials = get_user_credentials(
            user_id, fallback_refresh=refresh_token)
        set_user_token(
            credentials.access_token,
            credentials.client_id,
            user_id,
        )

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

        # /live resend mode: only the owner who tapped 🔁 may satisfy the
        # pending correction. In shared chats, other users must not consume it.
        resend_pending = self._live_resend_pending.get(chat_id)
        if resend_pending and resend_pending[0] == user_id:
            _owner_id, sid, hand_idx = self._live_resend_pending.pop(chat_id)
            await self._apply_live_resend(
                update, context, sid, hand_idx, user_text or "")
            return

        # /live capture mode: this message is the live-hand batch
        if chat_id in self._live_pending:
            self._live_pending.discard(chat_id)
            await self._process_live_batch(update, user_text)
            return

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

        # Split response: send the structured per-street GTO summary card the
        # moment analysis finishes, before the slow coaching reply.  Matches the
        # image flow so text hands with a concrete hero hand feel responsive.
        gto_sent = False

        async def send_gto_summary(text: str):
            nonlocal gto_sent
            await status_msg.delete()
            # "Open in GTO Wizard" deep-link rides on the 📋 summary card.
            gto_markup = self._build_gto_link_markup(chat_id)
            await _send_reply(update.message, text, self.log, label,
                              reply_markup=gto_markup)
            gto_sent = True

        t0 = time.time()
        try:
            async with _TypingLoop(update.message.chat):
                response = await self.session_manager.send_message(
                    chat_id, user_text, on_status=_on_status,
                    user_id=user_id, refresh_token=refresh_token,
                    send_gto_callback=send_gto_summary,
                )
            elapsed = time.time() - t0
            self.log.info(f"[{label}] Response OK ({elapsed:.1f}s, {len(response)} chars)")

            if not gto_sent:
                await status_msg.delete()

            if not response or not response.strip():
                self.log.warning(f"[{label}] Empty response from session manager")
                if not gto_sent:
                    await update.message.reply_text("抱歉，分析過程中出現問題，請重新傳送手牌。")
                return
            # When the summary card already carries the GTO Wizard link, the
            # coaching reply only needs follow-up buttons; otherwise fall back
            # to putting the link on the coaching reply.
            response, markup = self._finalize_followups(
                chat_id, response, include_gto_link=not gto_sent)
            await _send_reply(update.message, response, self.log, label,
                              reply_markup=markup)

            # Send range grid images
            await self._send_pending_range_images(update, chat_id, label)

        except Exception as e:
            elapsed = time.time() - t0
            self.log.error(f"[{label}] Error after {elapsed:.1f}s: {e}", exc_info=True)
            from gto_token import TokenExpiredError
            # Once the summary card went out the status message is gone, so a
            # failed coaching call must surface the error as a fresh reply.
            async def _show_err(text: str):
                if gto_sent:
                    await update.message.reply_text(text)
                else:
                    await status_msg.edit_text(text)
            if isinstance(e, TokenExpiredError):
                self.log.warning(f"[{label}] Token expired during message handling")
                await _show_err(
                    "你的 GTO Wizard token 已過期，請重新點擊書籤工具並貼上 /settoken 指令。"
                )
                return
            await _show_err(f"❌ {str(e)}")

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
                f"Hero {hand['hero_position']} {cards_to_emoji(hand['hero_hand'])} "
                f"({hand['effective_bb']:.0f}bb, {hand.get('num_players', 8)}人)\n"
                f"Preflop: {hand['preflop_actions']}"
            )
            if hand.get("streets"):
                for s in hand["streets"]:
                    board = s.get("board", s.get("card", ""))
                    acts = " ".join(
                        f"{a['position']}:{a['action']}" for a in s["actions"]
                    )
                    hand_desc += f"\n{cards_to_emoji(board)} → {acts}"

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
                            title += f" | {cards_to_emoji(board)}"
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

    def _finalize_followups(self, chat_id: int, response: str,
                            include_gto_link: bool = False
                            ) -> tuple[str, InlineKeyboardMarkup | None]:
        """Strip any leaked FOLLOWUP lines from a reply and turn them into buttons.

        Safety net for the plain-chat follow-up path (_chat), which never ran
        _extract_followups — so the LLM's FOLLOWUP: lines surfaced as raw text
        instead of inline buttons. Re-extract here right before sending: strip
        the lines from the visible text and, when fresh questions are found,
        register them and rebuild the button set (clearing _followup_sent so the
        new questions render even after an earlier batch was already shown).

        On the initial-analysis paths the response is already clean (extraction
        happened upstream), so this is a no-op that just builds the markup.
        """
        clean, recovered = self.session_manager._extract_followups(response)
        if recovered:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if ctx is not None:
                ctx["followup_questions"] = recovered
                ctx["_followup_sent"] = False
        markup = self._build_followup_markup(
            chat_id, include_gto_link=include_gto_link)
        return clean, markup

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
                        f"{cards_to_emoji(hero_hand)} 在這裡應該用什麼 size？")
                elif is_icm:
                    questions.append("這個位置的 push 範圍有多寬？")
                else:
                    questions.append(
                        f"{cards_to_emoji(hero_hand)} 的 EV 跟其他手牌比如何？")

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
        """Deep-link to the GTOW /solutions node behind the current reply.

        When a follow-up answer was grounded on a specific street's hero
        decision (``_followup_node_street``), link to that exact node so the
        button's frequencies match the prose (turn 89% vs river 23%, H3515).
        Otherwise fall back to hero's last decision node (the played line).

        Never raises — a failed build just means no button.
        """
        try:
            from gtow_solution_url import (build_last_node_url,
                                           build_node_url_for_street)
            node_street = ctx.get("_followup_node_street")
            if node_street:
                url = build_node_url_for_street(ctx, node_street)
                if url:
                    return url
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
                # Follow-up answers can carry their own fresh FOLLOWUP: lines;
                # strip them from the text and re-render them as buttons so a
                # button press leads to more buttons, never raw FOLLOWUP text.
                response, markup = self._finalize_followups(chat_id, response)
                formatted = _format_for_telegram(response)
                chunks = [c for c in _split_message(formatted) if c.strip()]
                for i, chunk in enumerate(chunks):
                    chunk_markup = markup if i == len(chunks) - 1 else None
                    try:
                        await context.bot.send_message(
                            chat_id, chunk, parse_mode='Markdown',
                            reply_markup=chunk_markup,
                            read_timeout=30, write_timeout=30,
                            connect_timeout=30)
                    except Exception:
                        await context.bot.send_message(
                            chat_id, _strip_markdown(chunk),
                            reply_markup=chunk_markup,
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

    async def ingest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ingest — owner-only: queue an *incremental* GTOW Analyze pull.

        Enqueues a gtow_ingest_requests row (mode='incremental'); the 5s poller
        runs the same pipeline as the extension button (per-user DB token) and
        edits the reply in place with live progress. For a full-history import
        use /fullingest.
        """
        user_id = update.effective_user.id
        if not self.admin_chat_id or user_id != self.admin_chat_id:
            return
        self.log.info(f"[{self._user_label(update)}] /ingest")
        if not self.db or not self.db.pool:
            await update.message.reply_text("⚠️ 資料庫未連線，無法排入攝取佇列")
            return
        from src.ingest_runner import enqueue_request, register_status_message
        reused = await enqueue_request(self.db.pool, user_id)
        msg = await update.message.reply_text(
            "⏳ 已有一件同步在跑，完成後會通知你" if reused
            else "⏳ 已排入增量同步佇列，開始後會即時更新進度…")
        # Hand this reply to the poller so it edits progress in place on claim.
        # On a reused request the in-flight run already owns its own message, so
        # only register when we actually enqueued a fresh one.
        if not reused:
            register_status_message(user_id, msg.chat_id, msg.message_id)

    async def fullingest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/fullingest — owner-only: full-history import, behind a confirm menu.

        A full backfill re-sweeps every hand since the ledger epoch and can take
        many minutes, so it is gated by an explicit inline confirmation. Enqueue
        happens only on ✅; the same one-open-per-user rule blocks it while any
        other ingest (incremental or full) is already running.
        """
        if not self._is_owner(update):
            return
        self.log.info(f"[{self._user_label(update)}] /fullingest")
        if not self.db or not self.db.pool:
            await update.message.reply_text("⚠️ 資料庫未連線，無法排入攝取佇列")
            return
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ 確定全量匯入", callback_data="fullingest:confirm"),
            InlineKeyboardButton("❌ 取消", callback_data="fullingest:cancel"),
        ]])
        await update.message.reply_text(
            "⚠️ <b>全量匯入</b>會重抓 GTOW 上所有歷史手牌（自 ledger epoch 2026-03 起），"
            "可能需要數分鐘～十幾分鐘，期間無法同時跑其他同步。\n\n確定要全量匯入嗎？",
            parse_mode="HTML", reply_markup=kb)

    async def handle_fullingest_button(self, update: Update,
                                       context: ContextTypes.DEFAULT_TYPE):
        """Confirm/cancel for /fullingest. On ✅ enqueue mode='full' and hand the
        (now-edited) confirmation message to the poller for live progress."""
        query = update.callback_query
        if not self._is_owner(update):
            await query.answer("僅限 owner。", show_alert=True)
            return
        action = query.data.split(":", 1)[1]
        if action == "cancel":
            await query.answer("已取消")
            await query.edit_message_text("已取消全量匯入。")
            return
        await query.answer()
        if not self.db or not self.db.pool:
            await query.edit_message_text("⚠️ 資料庫未連線，無法排入攝取佇列")
            return
        from src.ingest_runner import enqueue_request, register_status_message
        user_id = query.from_user.id
        reused = await enqueue_request(self.db.pool, user_id, mode="full")
        if reused:
            await query.edit_message_text(
                "⏳ 已有一件同步在跑，完成後會通知你（全量匯入未排入，請稍後再試）")
            return
        await query.edit_message_text("⏳ 已排入全量匯入佇列，開始後會即時更新進度…")
        # query.message is the just-edited confirmation message; the poller edits
        # this same message in place with the live progress bar.
        register_status_message(user_id, query.message.chat_id,
                                query.message.message_id)

    # ── 線下流: /live /lives /queue /plan + inline buttons ───────────────────

    @staticmethod
    def _rows_to_markup(rows: list[list[dict]]) -> InlineKeyboardMarkup | None:
        """[[{"text", "url"|"callback_data"}]] -> InlineKeyboardMarkup."""
        kb = [[InlineKeyboardButton(b["text"], url=b.get("url"),
                                    callback_data=b.get("callback_data"))
               for b in row] for row in rows]
        return InlineKeyboardMarkup(kb) if kb else None

    @staticmethod
    def _mark_button_done(markup, tapped_data: str, done_text: str = "✅ 已排入"):
        """Copy `markup`, relabel the tapped button ✅. Its callback_data is kept
        so a re-tap is an idempotent enqueue no-op (enqueue_one dedupes)."""
        if not markup:
            return None
        kb = [[(InlineKeyboardButton(done_text, callback_data=b.callback_data)
                if b.callback_data == tapped_data else b)
               for b in row] for row in markup.inline_keyboard]
        return InlineKeyboardMarkup(kb)

    def _is_owner(self, update: Update) -> bool:
        return bool(self.admin_chat_id
                    and update.effective_user.id == self.admin_chat_id)

    async def live_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/live — owner-only: import a live-hand shorthand batch into the ledger."""
        if not self._is_owner(update):
            return
        self.log.info(f"[{self._user_label(update)}] /live")
        parts = (update.message.text or "").split(None, 1)
        payload = parts[1] if len(parts) > 1 else ""
        if payload.strip():
            async with self._user_lock(update.effective_chat.id):
                await self._process_live_batch(update, payload)
        else:
            self._live_pending.add(update.effective_chat.id)
            await update.message.reply_text(
                "貼上現場手牌（下一則訊息）。每手以「Eff <有效籌碼>」開頭，"
                "一則訊息可貼多手；街與街換行，例如：\n\n"
                "Eff 25bb co raise hero bb call As2s\n"
                "AhQhJh x b1.2 c\n2h x b1.5 f")

    async def live_sessions_command(self, update: Update,
                                    context: ContextTypes.DEFAULT_TYPE):
        """/lives — owner-only: list recent persisted live-session reports."""
        if not self._is_owner(update):
            return
        if not (self.db and self.db.pool):
            await update.message.reply_text("Database not connected.")
            return
        self.log.info(f"[{self._user_label(update)}] /lives")
        from live_flow import list_recent_sessions
        sessions = await list_recent_sessions(
            self.db.pool, update.effective_chat.id, LIVE_SESSION_LIST_LIMIT)
        html, buttons = _recent_live_sessions_payload(sessions)
        await update.message.reply_text(
            html, parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=self._rows_to_markup(buttons))

    async def online_sessions_command(self, update: Update,
                                      context: ContextTypes.DEFAULT_TYPE):
        """/sessions — owner-only: list recent online sessions for review resend."""
        if not self._is_owner(update):
            return
        if not (self.db and self.db.pool):
            await update.message.reply_text("Database not connected.")
            return
        self.log.info(f"[{self._user_label(update)}] /sessions")
        from session_review import list_recent_sessions
        sessions = await list_recent_sessions(
            self.db.pool, ONLINE_SESSION_LIST_LIMIT)
        html, buttons = _recent_online_sessions_payload(sessions)
        await update.message.reply_text(
            html, parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=self._rows_to_markup(buttons))

    async def _process_live_batch(self, update: Update, text: str):
        """Run scripts/live_flow.py on the batch, reply with the deviation
        report + [Hand N 詳細] callbacks + 🎯 drill URL buttons."""
        import json as _json
        from live_flow import split_batch
        label = self._user_label(update)
        n = len(split_batch(text))
        if n == 0:
            await update.message.reply_text(
                "沒有偵測到手牌 — 每手要以「Eff <有效籌碼>」開頭。")
            return
        eta_low, eta_high = _estimate_live_batch_minutes(n)
        msg = await update.message.reply_text(
            f"🃏 收到 {n} 手，解析評分中…（約 {eta_low}-{eta_high} 分鐘，依 solver 速度浮動）")
        refresh_token = await self._get_user_refresh_token(update.effective_user.id)
        if not refresh_token:
            await msg.edit_text("請先使用 /settoken 綁定你的 GTO Wizard 帳號。")
            return
        root = Path(__file__).resolve().parent.parent.parent
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text)
            tmp_in = f.name
        tmp_out = tmp_in + ".json"
        try:
            child_env = {
                **os.environ,
                "GTOW_USER_ID": str(update.effective_user.id),
            }
            child_env.pop("GTOW_REFRESH_TOKEN", None)
            child_env.pop("POKER_BOT_PROCESS", None)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "scripts/live_flow.py", "--file", tmp_in,
                "--json-out", tmp_out, cwd=str(root),
                env=child_env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
            if proc.returncode != 0 or not Path(tmp_out).exists():
                tail = out.decode(errors="replace")[-500:]
                await msg.edit_text(f"⚠️ 匯入失敗：\n{tail}")
                return
            result = _json.loads(Path(tmp_out).read_text())
            from live_flow import (hand_id_for, render_session_page,
                                   save_session, session_page_buttons,
                                   set_session_message)
            date_str = result.get("date")
            session_key = hand_id_for(text, date_str)
            async with self.db.pool.acquire() as conn:
                session_id = await save_session(
                    conn, session_key, update.effective_chat.id, result)
            html, _prev, _next = render_session_page(result, 0)
            markup = self._rows_to_markup(
                session_page_buttons(result, session_id, 0))
            try:
                await msg.delete()
            except Exception:
                pass
            sent = await update.message.reply_text(
                html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
            async with self.db.pool.acquire() as conn:
                await set_session_message(conn, session_id, sent.message_id)
            self.log.info(f"[{label}] /live done: {result['totals']}")
        except Exception as e:
            self.log.error(f"[{label}] /live failed: {e}", exc_info=True)
            try:
                await msg.edit_text(f"⚠️ 匯入失敗：{e}")
            except Exception:
                pass
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _apply_live_resend(self, update, context, session_id: int,
                                 hand_idx: int, block: str):
        from live_flow import (load_session, overwrite_hand, process_resend_block,
                               render_session_page, resend_entry_is_graded,
                               resend_failure_message, session_page_buttons,
                               set_session_message)

        msg = await update.message.reply_text("🔁 重新解析並覆蓋中…")
        async with self.db.pool.acquire() as conn:
            session_hint = await load_session(conn, session_id)
        if not session_hint:
            await msg.edit_text("這個線下 session 已過期，請重跑 /live。")
            return

        refresh_token = await self._get_user_refresh_token(update.effective_user.id)
        if not refresh_token:
            await msg.edit_text("請先使用 /settoken 綁定你的 GTO Wizard 帳號。")
            return

        # Gemini + solver grading is synchronous; do it off the event loop and
        # before acquiring the write transaction/connection.  The worker thread
        # must receive the requesting user's GTO token and clear it afterward.
        _user_id = update.effective_user.id
        _refresh_token = refresh_token
        _setup = self._setup_user_token
        _clear = self._clear_user_token
        _date = session_hint["result"].get("date")

        def _process_with_token():
            _setup(_user_id, _refresh_token)
            try:
                return process_resend_block(block, _date)
            finally:
                _clear()

        new_entry = await asyncio.to_thread(_process_with_token)
        if not resend_entry_is_graded(new_entry):
            await msg.edit_text(resend_failure_message(hand_idx, new_entry))
            return

        async with self.db.pool.acquire() as conn:
            applied = await overwrite_hand(conn, session_id, hand_idx, new_entry)
        if not applied.get("ok"):
            if applied.get("error") == "session_missing":
                await msg.edit_text("這個線下 session 已過期，請重跑 /live。")
            else:
                await msg.edit_text(resend_failure_message(
                    hand_idx, applied.get("entry") or new_entry))
            return

        session = applied["session"]
        result = applied["result"]
        page = applied["page"]
        html, _prev, _next = render_session_page(result, page)
        markup = self._rows_to_markup(
            session_page_buttons(result, session_id, page))
        try:
            await msg.delete()
        except Exception:
            pass
        if session.get("message_id"):
            try:
                await context.bot.edit_message_text(
                    html, chat_id=session["chat_id"],
                    message_id=session["message_id"], parse_mode="HTML",
                    disable_web_page_preview=True, reply_markup=markup)
                await update.message.reply_text(f"✅ Hand {hand_idx + 1} 已更新。")
                return
            except Exception:
                self.log.warning("live resend edit_message_text failed", exc_info=True)
        sent = await update.message.reply_text(
            html, parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=markup)
        async with self.db.pool.acquire() as conn:
            await set_session_message(conn, session_id, sent.message_id)

    async def _send_or_edit_session_page(self, query, session: dict, page: int):
        """Re-render one live session page and edit the report message in place."""
        from live_flow import (render_session_page, session_page_buttons,
                               update_session_result)

        result = session["result"]
        html, _prev, _next = render_session_page(result, page)
        markup = self._rows_to_markup(
            session_page_buttons(result, session["id"], page))
        async with self.db.pool.acquire() as conn:
            await update_session_result(conn, session["id"], result, page)
        try:
            await query.edit_message_text(
                html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
        except telegram.error.BadRequest as exc:
            if "Message is not modified" not in str(exc):
                raise

    async def _fetch_queue_page(self, page: int = 0):
        total = await self.db.pool.fetchval(
            "SELECT count(*) FROM drill_queue "
            "WHERE status IN ('pending','prescribed')")
        pages = max(1, (int(total) + QUEUE_PAGE_SIZE - 1) // QUEUE_PAGE_SIZE)
        page = max(0, min(int(page), pages - 1))
        rows = await self.db.pool.fetch(
            "SELECT id, spot_leaf, label, drill_url, review_anchor_url, "
            "review_anchor_street, status, n_sources, added_by, "
            "total_ev_loss_bb, kind, ref_hand_id, spot_category, "
            "bias_direction, bias_n, bias_ev_loss_bb, bias_share, depth_scope "
            "FROM drill_queue WHERE status IN ('pending','prescribed') "
            "ORDER BY (status='pending') DESC, total_ev_loss_bb DESC NULLS LAST "
            "LIMIT $1 OFFSET $2", QUEUE_PAGE_SIZE, page * QUEUE_PAGE_SIZE)
        return rows, int(total), page

    async def queue_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/queue — owner-only: pending/prescribed practice queue with buttons."""
        if not self._is_owner(update):
            return
        if not (self.db and self.db.pool):
            await update.message.reply_text("Database not connected.")
            return
        rows, total, page = await self._fetch_queue_page(0)
        html, buttons = _queue_payload(rows, page=page, total=total)
        await update.message.reply_text(
            html, parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=self._rows_to_markup(buttons))

    async def _queue_drill_detail(self, update: Update,
                                  context: ContextTypes.DEFAULT_TYPE,
                                  queue_id: int, page: int = 0,
                                  *, new_message: bool = False):
        """Ensure/reuse the matching GTOW Drill, then show its practice card."""
        query = update.callback_query
        chat_id = update.effective_chat.id
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        refresh_token = await self._get_user_refresh_token(update.effective_user.id)
        if not refresh_token:
            await query.answer("請先綁定 GTO Wizard token。", show_alert=True)
            return
        await query.answer("正在準備 GTOW Drill…")
        try:
            from gtow_drill_service import (GTOWDrillClient,
                                            settings_from_trainer_url,
                                            settings_hash, stats_json)
            client = GTOWDrillClient(update.effective_user.id, refresh_token)
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    item = await conn.fetchrow(
                        "SELECT id, spot_leaf, spot_category, label, drill_url, "
                        "source_hands, kind, n_sources, bias_direction, bias_n, "
                        "bias_ev_loss_bb, bias_share, "
                        "depth_scope, "
                        "gtow_drill_id, gtow_drill_name, gtow_settings_hash, "
                        "total_ev_loss_bb, gtow_target_hands, gtow_target_score, "
                        "gtow_training_started_at "
                        "FROM drill_queue WHERE id=$1 FOR UPDATE", queue_id)
                    if not item or item["kind"] != "drill":
                        await _present_queue_detail(
                            query, context, chat_id, "找不到這個 Drill queue item。",
                            None, new_message=new_message)
                        return
                    if not item["drill_url"]:
                        from queue_feed import _as_list, queue_drill_url_from_sources
                        rebuilt_url = await queue_drill_url_from_sources(
                            conn, _as_list(item["source_hands"]))
                        if rebuilt_url:
                            item = await conn.fetchrow(
                                "UPDATE drill_queue SET drill_url=$2, "
                                "gtow_drill_id=NULL, gtow_drill_name=NULL, "
                                "gtow_settings_hash=NULL, "
                                "gtow_drill_synced_at=NULL, last_added=NOW() "
                                "WHERE id=$1 RETURNING *",
                                queue_id, rebuilt_url)
                    if not item["drill_url"]:
                        await _present_queue_detail(
                            query, context, chat_id,
                            "這個項目目前沒有可精確重建的 GTOW Trainer 連結。",
                            None, new_message=new_message)
                        return
                    # Upgrade persisted pre-filter URLs before matching the
                    # GTOW Drill.  In particular, all MTT URLs now pin
                    # gmff_variant=with_limps.  Reset the old binding and
                    # attempt window so no-limp results cannot count toward
                    # the new prescription.
                    from gtow_trainer_url import apply_trainer_defaults
                    upgraded_url = apply_trainer_defaults(item["drill_url"])
                    if upgraded_url != item["drill_url"]:
                        item = await conn.fetchrow(
                            "UPDATE drill_queue SET drill_url=$2, "
                            "gtow_drill_id=NULL, gtow_drill_name=NULL, "
                            "gtow_settings_hash=NULL, gtow_drill_synced_at=NULL, "
                            "gtow_training_started_at=NULL, "
                            "gtow_baseline_totals=NULL, last_added=NOW() "
                            "WHERE id=$1 RETURNING *",
                            queue_id, upgraded_url)
                    fingerprint = settings_hash(
                        settings_from_trainer_url(item["drill_url"]))
                    # Serialize ensure/create across different queue rows that
                    # resolve to the same settings; rapid taps cannot create
                    # duplicate GTOW Drills.
                    await conn.fetchval(
                        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                        fingerprint)
                    from spot_naming import compact_spot_name
                    drill_name = compact_spot_name(item)
                    binding = await asyncio.to_thread(
                        client.ensure_drill, item["drill_url"], drill_name,
                        known_drill_id=(str(item["gtow_drill_id"])
                                        if item["gtow_drill_id"] else None),
                        known_drill_name=item["gtow_drill_name"],
                        known_settings_hash=item["gtow_settings_hash"])
                    item = await conn.fetchrow(
                        "UPDATE drill_queue SET gtow_drill_id=$2::uuid, "
                        "gtow_drill_name=$3, gtow_settings_hash=$4, "
                        "label=$6, "
                        "gtow_drill_synced_at=NOW(), "
                        "gtow_training_started_at="
                        "COALESCE(gtow_training_started_at, NOW()), "
                        "gtow_baseline_totals="
                        "COALESCE(gtow_baseline_totals, $5::jsonb) "
                        "WHERE id=$1 RETURNING *",
                        queue_id, binding.drill_id, binding.name,
                        binding.settings_hash, json.dumps(stats_json(binding.stats)),
                        drill_name)

            def load_stats():
                return (
                    client.drill_totals(binding.drill_id),
                    client.attempt_stats(
                        binding.drill_id, item["gtow_training_started_at"]),
                )

            lifetime, attempt = await asyncio.to_thread(load_stats)
            html, buttons = _queue_drill_detail_payload(
                dict(item), binding, lifetime, attempt, page=page)
            await _present_queue_detail(
                query, context, chat_id, html, self._rows_to_markup(buttons),
                new_message=new_message)
        except Exception as exc:
            self.log.error("GTOW Drill detail failed for queue %s: %s",
                           queue_id, exc, exc_info=True)
            await _present_queue_detail(
                query, context, chat_id,
                "⚠️ 無法準備 GTOW Drill。可能是 GTOW token 已失效或 API 暫時異常。",
                self._rows_to_markup([[
                    {"text": "🔄 重試", "callback_data": f"qdet:{queue_id}:{page}"},
                    {"text": "⬅ 返回 Queue", "callback_data": f"qpg:{page}"},
                ]]), new_message=new_message)

    async def _queue_show_sources(self, update: Update,
                                  context: ContextTypes.DEFAULT_TYPE,
                                  queue_id: int, page: int = 0,
                                  *, queue_page: int = 0,
                                  edit: bool = False):
        """Show the exact online/live hands that produced one queue item."""
        query = update.callback_query
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        item = await self.db.pool.fetchrow(
            "SELECT label, kind, spot_leaf, spot_category, drill_url, depth_scope, "
            "source_hands, ref_hand_id "
            "FROM drill_queue WHERE id=$1",
            queue_id)
        if not item:
            await query.answer("找不到這個 queue item。")
            return
        from queue_feed import (_as_list, queue_source_hand_ids,
                                resolve_queue_source_hands)
        entries = _as_list(item["source_hands"])
        hand_ids = queue_source_hand_ids(entries, item["ref_hand_id"])
        ledger_rows = []
        if hand_ids:
            ledger_rows = await self.db.pool.fetch(
                "SELECT gtow_hand_id, source, raw_text, played_at, position, hero_hand "
                "FROM ledger_hands WHERE gtow_hand_id = ANY($1::text[])",
                hand_ids)
        sources = resolve_queue_source_hands(
            entries, ledger_rows, ref_hand_id=item["ref_hand_id"])
        if item["kind"] == "drill":
            from spot_naming import compact_spot_name
            source_label = compact_spot_name(item)
        else:
            source_label = item["label"] or str(queue_id)
        html, buttons = _queue_source_payload(
            queue_id, source_label, sources, page=page,
            queue_page=queue_page, kind=item["kind"])
        markup = self._rows_to_markup(buttons)
        await query.answer()
        if edit:
            await query.edit_message_text(
                html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
        else:
            await context.bot.send_message(
                update.effective_chat.id, html, parse_mode="HTML",
                disable_web_page_preview=True, reply_markup=markup)

    async def _queue_send_live_raw(self, update: Update,
                                   context: ContextTypes.DEFAULT_TYPE,
                                   hand_id: str, *, queue_id: int | None = None,
                                   source_page: int = 0, queue_page: int = 0):
        """Replace the source menu with live shorthand + Study/back buttons."""
        query = update.callback_query
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        row = await self.db.pool.fetchrow(
            "SELECT raw_text, parsed_json FROM ledger_hands "
            "WHERE gtow_hand_id=$1 AND source='live'", hand_id)
        raw_text = row["raw_text"] if row else None
        if not raw_text:
            await query.answer("找不到這手的原始記錄。")
            return

        await query.answer("正在準備 Study 連結…")
        study_url = None
        try:
            parsed = row["parsed_json"]
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            decisions = await self.db.pool.fetch(
                "SELECT street, decision_idx FROM ledger_decisions "
                "WHERE gtow_hand_id=$1 AND source='live' "
                "AND grader='own_pipeline' AND excluded=FALSE",
                hand_id)
            refresh_token = await self._get_user_refresh_token(
                update.effective_user.id)
            if parsed and decisions and refresh_token:
                def build_study_url():
                    self._setup_user_token(
                        update.effective_user.id, refresh_token)
                    try:
                        from gtow_solution_url import build_last_hero_hand_url
                        return build_last_hero_hand_url(
                            parsed, [dict(d) for d in decisions])
                    finally:
                        self._clear_user_token()

                study_url = await asyncio.to_thread(build_study_url)
        except Exception:
            self.log.debug("Live raw Study URL build failed for %s",
                           hand_id, exc_info=True)

        payload = f"📝 線下原始紀錄\n\n{raw_text}"
        buttons = []
        if study_url:
            buttons.append([{
                "text": "🧙 查看 Study Spot", "url": study_url,
            }])
        if queue_id is not None:
            buttons.append([{
                "text": "⬅ 返回來源牌局",
                "callback_data":
                    f"qsrc:{queue_id}:{source_page}:{queue_page}",
            }])
        markup = self._rows_to_markup(buttons) if buttons else None
        chunks = _split_message(payload)
        await query.edit_message_text(
            chunks[0], disable_web_page_preview=True, reply_markup=markup)
        for chunk in chunks[1:]:
            await context.bot.send_message(update.effective_chat.id, chunk)

    async def plan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/plan — owner-only: resend the latest weekly training plan."""
        if not self._is_owner(update):
            return
        if not (self.db and self.db.pool):
            await update.message.reply_text("Database not connected.")
            return
        import json as _json
        row = await self.db.pool.fetchrow(
            "SELECT week, data_json FROM scorecards ORDER BY created_at DESC LIMIT 1")
        if not row:
            await update.message.reply_text("還沒有訓練計畫 — 週日 21:00 會自動產生。")
            return
        from scorecard import weekly_tg_payload
        data = (_json.loads(row["data_json"])
                if isinstance(row["data_json"], str) else row["data_json"])
        payload = weekly_tg_payload(row["week"], data)
        await update.message.reply_text(
            payload["html"], parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=self._rows_to_markup(payload["buttons"]))

    async def review_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/review [session_id] — owner-only: 這場（預設最近一個 online session）
        的復盤摘要（EV 加權、單場不作趨勢判斷）。只讀不動本週焦點 spot。"""
        if not self._is_owner(update):
            return
        if not (self.db and self.db.pool):
            await update.message.reply_text("Database not connected.")
            return
        parts = (update.message.text or "").split()
        sid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        from session_review import resolve_session
        session = await resolve_session(self.db.pool, sid)
        if not session:
            await update.message.reply_text(
                "還沒有可復盤的 session — 先用 ♠ 同步手牌或 /ingest。")
            return
        out = await self._online_session_review_payload(
            context, session, user_id=update.effective_user.id)
        await update.message.reply_text(
            out["html"], parse_mode="HTML", disable_web_page_preview=True,
            reply_markup=self._rows_to_markup(out["buttons"]))

    async def _online_session_review_payload(self, context, session: dict,
                                             user_id: int | None = None) -> dict:
        """Compute/cache/render one online session for /review and /sessions."""
        from session_review import compute, render_tg, session_callback_key

        data = await compute(self.db.pool, session, user_id=user_id)
        cache = context.application.bot_data.setdefault("srev", {})
        cache[data["session_id"]] = data
        cache[session_callback_key(data)] = data
        return render_tg(data)

    async def _session_review_enqueue(self, update: Update,
                                      context: ContextTypes.DEFAULT_TYPE, data: str):
        """srd|srv:<session_id>:<i> or srd2|srv2:<stable_key>:<i> — enqueue the i-th session-review spot(drill)/
        decision(review) into drill_queue (added_by='session', threshold-free). Idempotent
        via queue_feed.enqueue_one; the tapped button is relabelled ✅."""
        query = update.callback_query
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        kind, session_ref, i_s = data.split(":")
        i = int(i_s)
        stable = kind in {"srd2", "srv2"}
        cache_key = session_ref if stable else int(session_ref)
        cached = context.application.bot_data.get("srev", {}).get(cache_key)
        if cached is None:
            from session_review import (compute, resolve_session,
                                        resolve_session_key, session_callback_key)
            session = (await resolve_session_key(self.db.pool, session_ref) if stable
                       else await resolve_session(self.db.pool, int(session_ref)))
            if not session:
                await query.answer("找不到這個 session。")
                return
            cached = await compute(
                self.db.pool, session, user_id=update.effective_user.id)
            cache = context.application.bot_data.setdefault("srev", {})
            cache[cached["session_id"]] = cached
            cache[session_callback_key(cached)] = cached
        items = (cached["top_spots"] if kind.startswith("srd")
                 else (cached.get("top_decisions") or cached.get("top_hands") or []))
        if i >= len(items):
            await query.answer("這個項目已不在清單上。")
            return
        from queue_feed import enqueue_one
        result = await enqueue_one(self.db.pool, items[i]["enqueue_item"])
        await query.answer({"inserted": "✅ 已排入佇列", "merged": "✅ 已併入佇列",
                            "noop": "✔ 已在佇列中"}.get(result, "✅ 已排入"))
        try:
            await query.edit_message_reply_markup(
                reply_markup=self._mark_button_done(query.message.reply_markup, data))
        except telegram.error.BadRequest as exc:
            if "Message is not modified" not in str(exc):
                self.log.exception("session-review enqueue markup refresh failed")

    async def _queue_expand_review(self, update: Update,
                                   context: ContextTypes.DEFAULT_TYPE, queue_id: int):
        """qex:<queue_id> — show a review hand's graded decisions as a sub-menu;
        each row's ➕ button adds that decision as a manual drill (§6.2)."""
        query = update.callback_query
        chat_id = update.effective_chat.id
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        await query.answer()
        item = await self.db.pool.fetchrow(
            "SELECT ref_hand_id, label FROM drill_queue WHERE id=$1", queue_id)
        if not item or not item["ref_hand_id"]:
            await context.bot.send_message(chat_id, "找不到這個復盤項的來源手。")
            return
        rows = await self.db.pool.fetch(
            "SELECT id, gtow_hand_id, street, decision_idx, spot_category, spot_leaf, hero_cat, "
            "villain_cat, ip_oop, position, ev_loss_bb "
            "FROM ledger_decisions "
            "WHERE gtow_hand_id=$1 AND NOT excluded AND NOT discarded "
            "ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 "
            "WHEN 'turn' THEN 2 WHEN 'river' THEN 3 ELSE 9 END, decision_idx",
            item["ref_hand_id"])
        if not rows:
            await context.bot.send_message(chat_id, "這手沒有可加練的已評分決策。")
            return
        from queue_feed import qex_submenu
        from html import escape as _esc
        btn_rows = qex_submenu([dict(r) for r in rows], queue_id)
        await context.bot.send_message(
            chat_id,
            f"➕ <b>選一條 action line 加入練習</b>\n"
            f"{_esc(item['label'] or item['ref_hand_id'])}",
            parse_mode="HTML",
            reply_markup=self._rows_to_markup([[b] for b in btn_rows]))

    async def _live_add_menu(self, context, chat_id, session, hand_idx: int):
        """Expand a live hand's graded decisions as ➕ manual-add buttons."""
        from queue_feed import qex_submenu

        hands = session["result"]["hands"]
        if hand_idx < 0 or hand_idx >= len(hands) or not hands[hand_idx].get("ok"):
            await context.bot.send_message(chat_id, "這手沒有可加練的決策。")
            return

        hand_id = hands[hand_idx]["hand_id"]
        rows = await self.db.pool.fetch(
            "SELECT id, gtow_hand_id, street, decision_idx, spot_category, "
            "spot_leaf, hero_cat, villain_cat, ip_oop, position, ev_loss_bb "
            "FROM ledger_decisions "
            "WHERE gtow_hand_id=$1 AND source='live' AND NOT excluded "
            "AND NOT discarded "
            "ORDER BY CASE street WHEN 'preflop' THEN 0 WHEN 'flop' THEN 1 "
            "WHEN 'turn' THEN 2 WHEN 'river' THEN 3 ELSE 9 END, decision_idx",
            hand_id)
        if not rows:
            await context.bot.send_message(
                chat_id, "這手沒有可加練的已評分決策。")
            return

        btn_rows = qex_submenu([dict(r) for r in rows], queue_id=0)
        await context.bot.send_message(
            chat_id,
            f"➕ <b>選一條 action line 加入練習</b>\nHand {hand_idx + 1}",
            parse_mode="HTML",
            reply_markup=self._rows_to_markup([[b] for b in btn_rows]))

    async def _queue_add_manual(self, update: Update,
                                context: ContextTypes.DEFAULT_TYPE,
                                queue_id: int, decision_ref):
        """qad:<queue_id>:<decision_id> or
        qad2:<queue_id>:<gtow_hand_id>:<street>:<decision_idx> — add one graded decision as a manual
        drill (kind='drill', added_by='manual', source='manual'), §6.2."""
        query = update.callback_query
        chat_id = update.effective_chat.id
        if not (self.db and self.db.pool):
            await query.answer("Database not connected.")
            return
        select_cols = (
            "SELECT id, gtow_hand_id, street, decision_idx, spot_category, spot_leaf, "
            "hero_cat, villain_cat, ip_oop, position, pot_type, eff_stack, ev_loss_bb "
            "FROM ledger_decisions ")
        if isinstance(decision_ref, tuple):
            hid, street, decision_idx = decision_ref
            dec = await self.db.pool.fetchrow(
                select_cols +
                "WHERE gtow_hand_id=$1 AND street=$2 AND decision_idx=$3",
                hid, street, decision_idx)
        else:
            # Backward compatibility for old Telegram messages emitted before
            # stable qad2 callbacks existed.
            dec = await self.db.pool.fetchrow(
                select_cols + "WHERE id=$1", int(decision_ref))
        if not dec:
            await query.answer("找不到這個決策。")
            return
        from queue_feed import (manual_drill_item, enqueue,
                                queue_drill_url_from_sources)
        from html import escape as _esc
        async with self.db.pool.acquire() as conn:
            url = await queue_drill_url_from_sources(conn, [{
                "hand_id": dec["gtow_hand_id"], "street": dec["street"],
                "decision_idx": int(dec["decision_idx"]), "src": "manual",
            }])
            item = manual_drill_item(dict(dec), drill_url=url)
            await enqueue(conn, [item])
        await query.answer("➕ 已加入練習佇列")
        await context.bot.send_message(
            chat_id, f"➕ 已加入練習：{_esc(item['label'] or item['spot_leaf'] or '?')}\n"
                     "用 /queue 查看。")

    async def handle_live_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Practice-queue + live-hand callbacks (all owner-only, §6.3):
        lvd:<hand_id> — deep-dive a live hand via the normal coach path;
        src2:<stable_session_key>:<i> — coach the i-th online session decision;
        qsrc:<queue_id>[:source_page[:queue_page]] — exact online/live
        source-hand menu;
        qraw:<queue_id>:<source_page>:<queue_page>:<hand_id> — show stored
        live shorthand + Study link in place;
        qdet:<queue_id> — ensure/reuse GTOW Drill and show its detail menu;
        qdst:<queue_id> — refresh the same detail menu and practice results;
        qcl:<queue_id> — mark a queue item cleared (writes cleared_at);
        qex:<queue_id> — expand a review item into its decisions to hand-pick;
        qad:<queue_id>:<decision_id> — add one decision as a manual drill
        qad2:<queue_id>:<gtow_hand_id>:<street>:<decision_idx> — stable add."""
        query = update.callback_query
        data = query.data or ""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        if not self._is_owner(update):
            await query.answer()
            return

        if data.startswith("lvs:"):
            from live_flow import (load_session, render_session_page,
                                   session_page_buttons, set_session_message)

            _, sid = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session or session["chat_id"] != chat_id:
                await query.answer("找不到這個線下 session。")
                return
            await query.answer()
            html, _prev, _next = render_session_page(session["result"], 0)
            markup = self._rows_to_markup(
                session_page_buttons(session["result"], session["id"], 0))
            sent = await context.bot.send_message(
                chat_id, html, parse_mode="HTML", disable_web_page_preview=True,
                reply_markup=markup)
            async with self.db.pool.acquire() as conn:
                await set_session_message(conn, session["id"], sent.message_id)
            return

        if data.startswith("ors:"):
            from session_review import resolve_session_key

            _, stable_key = data.split(":", 1)
            session = await resolve_session_key(self.db.pool, stable_key)
            if not session:
                await query.answer("找不到這個線上 session，可能已被重建。")
                return
            await query.answer()
            out = await self._online_session_review_payload(
                context, session, user_id=user_id)
            await context.bot.send_message(
                chat_id, out["html"], parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=self._rows_to_markup(out["buttons"]))
            return

        if data.startswith("lvpg:"):
            from live_flow import load_session

            _, sid, page = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            await self._send_or_edit_session_page(query, session, int(page))
            return

        if data.startswith("lvadd:"):
            from live_flow import load_session

            _, sid, hand_idx = data.split(":")
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            await self._live_add_menu(context, chat_id, session, int(hand_idx))
            return

        if data.startswith("lvr:"):
            from html import escape as _esc
            from live_flow import load_session, _repair_explanation

            _, sid, hand_idx = data.split(":")
            hand_idx_i = int(hand_idx)
            async with self.db.pool.acquire() as conn:
                session = await load_session(conn, int(sid))
            if not session:
                await query.answer("這個線下 session 已過期，請重跑 /live。")
                return
            await query.answer()
            h = session["result"]["hands"][hand_idx_i]
            self._live_resend_pending[chat_id] = (user_id, int(sid), hand_idx_i)
            reps = "；".join(
                _repair_explanation(str(r)) for r in (h.get("repairs") or [])
            ) or "無"
            await context.bot.send_message(
                chat_id,
                f"請貼上 <b>Hand {hand_idx_i + 1}</b> 的單手更正版本"
                f"（Header / Flop / Turn / River 各一行）。\n"
                f"目前 echo：{_esc(h.get('echo') or '（無法評分）')}\n"
                f"目前校正：{_esc(reps)}",
                parse_mode="HTML")
            return

        if data.startswith("qsrc:"):
            parts = data.split(":")
            queue_id = int(parts[1])
            source_page = int(parts[2]) if len(parts) > 2 else 0
            queue_page = int(parts[3]) if len(parts) > 3 else 0
            await self._queue_show_sources(
                update, context, queue_id, source_page,
                queue_page=queue_page, edit=len(parts) > 2)
            return

        if data.startswith("qraw:"):
            parts = data.split(":", 4)
            if len(parts) == 5 and parts[1].isdigit():
                await self._queue_send_live_raw(
                    update, context, parts[4], queue_id=int(parts[1]),
                    source_page=int(parts[2]), queue_page=int(parts[3]))
            else:
                # Backward compatibility for buttons sent before this deploy.
                await self._queue_send_live_raw(update, context, data[5:])
            return

        if data.startswith("qdet:") or data.startswith("qdst:"):
            parts = data.split(":")
            origin = parts[3] if len(parts) > 3 else "queue"
            await self._queue_drill_detail(
                update, context, int(parts[1]),
                int(parts[2]) if len(parts) > 2 else 0,
                new_message=(data.startswith("qdet:") and origin == "plan"))
            return

        if data.startswith("qpg:"):
            if self.db and self.db.pool:
                page = int(data.split(":", 1)[1])
                rows, total, page = await self._fetch_queue_page(page)
                html, buttons = _queue_payload(rows, page=page, total=total)
                await query.answer()
                await query.edit_message_text(
                    html, parse_mode="HTML", disable_web_page_preview=True,
                    reply_markup=self._rows_to_markup(buttons))
            else:
                await query.answer("Database not connected.")
            return

        if data.startswith("qcl:"):
            if self.db and self.db.pool:
                parts = data.split(":")
                queue_id = int(parts[1])
                page = int(parts[2]) if len(parts) > 2 else 0
                reason = parts[3] if len(parts) > 3 else "completed"
                origin = parts[4] if len(parts) > 4 else "queue"
                if reason not in {"completed", "mistake", "skipped"}:
                    reason = "completed"
                await self.db.pool.execute(
                    "UPDATE drill_queue SET status='cleared', cleared_at=NOW(), "
                    "clear_reason=$2 WHERE id=$1", queue_id, reason)
                if origin == "plan":
                    await query.answer("✔ 已標記為完成")
                    try:
                        await query.edit_message_reply_markup(
                            reply_markup=self._mark_button_done(
                                query.message.reply_markup, data,
                                done_text="✅ 已完成"))
                    except telegram.error.BadRequest as exc:
                        if "Message is not modified" not in str(exc):
                            self.log.exception("Failed to refresh weekly plan after qcl")
                    return
                rows, total, page = await self._fetch_queue_page(page)
                html, buttons = _queue_payload(rows, page=page, total=total)
                answer = ("🗑 已移除誤植項目" if reason == "mistake"
                          else "✔ 已標記為完成")
                await query.answer(answer)
                try:
                    await query.edit_message_text(
                        html, parse_mode="HTML", disable_web_page_preview=True,
                        reply_markup=self._rows_to_markup(buttons))
                except telegram.error.BadRequest as exc:
                    # Completion is an in-place queue interaction.  Never add
                    # a separate chat message as a fallback; that makes review
                    # clears noisy and leaves the stale queue above it.
                    if "Message is not modified" not in str(exc):
                        self.log.exception("Failed to refresh queue after qcl")
            else:
                await query.answer("Database not connected.")
            return

        if data.startswith("qex:"):
            await self._queue_expand_review(update, context, int(data[4:]))
            return

        if data.startswith("qad:"):
            _, qid, did = data.split(":")
            await self._queue_add_manual(update, context, int(qid), int(did))
            return

        if data.startswith("qad2:"):
            _, qid, decision_key = data.split(":", 2)
            hid, street, didx = decision_key.rsplit(":", 2)
            await self._queue_add_manual(
                update, context, int(qid), (hid, street, int(didx)))
            return

        if (data.startswith("srd:") or data.startswith("srv:")
                or data.startswith("srd2:") or data.startswith("srv2:")):
            await self._session_review_enqueue(update, context, data)
            return

        if data.startswith("src2:"):
            _, stable_key, i_s = data.split(":")
            from session_review import (compute, resolve_session_key,
                                        session_callback_key, _load_detail)

            cache = context.application.bot_data.setdefault("srev", {})
            cached = cache.get(stable_key)
            if cached is None:
                session = await resolve_session_key(self.db.pool, stable_key)
                if not session:
                    await query.answer("找不到這個線上 session，可能已被重建。")
                    return
                cached = await compute(self.db.pool, session, user_id=user_id)
                cache[cached["session_id"]] = cached
                cache[session_callback_key(cached)] = cached
            decisions = cached.get("top_decisions") or []
            i = int(i_s)
            if i < 0 or i >= len(decisions):
                await query.answer("這個決策已不在清單上。")
                return
            hand_id = decisions[i].get("ref_hand_id")
            row = await self.db.pool.fetchrow(
                "SELECT gtow_hand_id, raw_path, position, hero_hand, total_players, "
                "preflop_depth_bb FROM ledger_hands "
                "WHERE gtow_hand_id=$1 AND source='online'", hand_id)
            detail = _load_detail(row["raw_path"] if row else None)
            if not row or not detail:
                await query.answer("找不到這手的 Analyzer 原始資料。", show_alert=True)
                return
            try:
                from analysis_fidelity_check import reconstruct_analyze_hand
                hand_json = reconstruct_analyze_hand(dict(row), detail)
            except Exception:
                self.log.exception("online session coach reconstruction failed: %s", hand_id)
                await query.answer("無法重建這手的教練分析資料。", show_alert=True)
                return

            await query.answer()
            label = f"online-session-coach-{chat_id}"
            self.log.info("[%s] deep dive %s", label, hand_id)
            await context.bot.send_message(chat_id, f"💬 教練分析：`{hand_id}`", parse_mode="Markdown")
            raw_status = await context.bot.send_message(chat_id, "🔍 分析中...")
            status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

            async def _on_status(m: str):
                await status_msg.edit_text(f"⏳ {m}")

            refresh_token = await self._get_user_refresh_token(user_id)
            try:
                response = await self._analyze_online_parsed_hand(
                    chat_id, user_id, hand_id, hand_json, _on_status, refresh_token)
                await status_msg.delete()
                if response and response.strip():
                    response, markup = self._finalize_followups(
                        chat_id, response, include_gto_link=True)
                    formatted = _format_for_telegram(response)
                    chunks = [c for c in _split_message(formatted) if c.strip()]
                    for j, chunk in enumerate(chunks):
                        chunk_markup = markup if j == len(chunks) - 1 else None
                        try:
                            await context.bot.send_message(
                                chat_id, chunk, parse_mode="Markdown",
                                reply_markup=chunk_markup)
                        except Exception:
                            await context.bot.send_message(
                                chat_id, _strip_markdown(chunk),
                                reply_markup=chunk_markup)
                await self._send_pending_range_images(update, chat_id, label)
            except Exception as exc:
                self.log.error("[%s] Error: %s", label, exc, exc_info=True)
                try:
                    await status_msg.delete()
                except Exception:
                    pass
                await context.bot.send_message(chat_id, "抱歉，教練分析時出錯了。")
            return

        await query.answer()
        hand_id = data[4:]                          # lvd:<hand_id>
        raw = None
        hand_json = None
        if self.db and self.db.pool:
            row = await self.db.pool.fetchrow(
                "SELECT raw_text, parsed_json FROM ledger_hands "
                "WHERE gtow_hand_id=$1 AND source='live'", hand_id)
            if row:
                raw = row["raw_text"]
                hand_json = self._decode_live_parsed_json(row["parsed_json"])
        if not raw or not hand_json:
            await context.bot.send_message(chat_id, "找不到這手的原始記錄。")
            return
        label = f"live-detail-{chat_id}"
        self.log.info(f"[{label}] deep dive {hand_id}")
        await context.bot.send_message(chat_id, f"💬 深入分析：\n{raw}")
        raw_status = await context.bot.send_message(chat_id, "🔍 分析中...")
        status_msg = _ResilientStatus(raw_status, log=self.log, label=label)

        async def _on_status(m: str):
            await status_msg.edit_text(f"⏳ {m}")

        refresh_token = await self._get_user_refresh_token(user_id)
        try:
            response = await self._analyze_live_parsed_hand(
                chat_id, user_id, hand_id, hand_json, _on_status, refresh_token)
            await status_msg.delete()
            if response and response.strip():
                response, markup = self._finalize_followups(
                    chat_id, response, include_gto_link=True)
                formatted = _format_for_telegram(response)
                chunks = [c for c in _split_message(formatted) if c.strip()]
                for i, chunk in enumerate(chunks):
                    chunk_markup = markup if i == len(chunks) - 1 else None
                    try:
                        await context.bot.send_message(
                            chat_id, chunk, parse_mode='Markdown',
                            reply_markup=chunk_markup)
                    except Exception:
                        await context.bot.send_message(
                            chat_id, _strip_markdown(chunk), reply_markup=chunk_markup)
            await self._send_pending_range_images(update, chat_id, label)
        except Exception as e:
            self.log.error(f"[{label}] Error: {e}", exc_info=True)
            try:
                await status_msg.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id, "抱歉，深入分析時出錯了。")

    @staticmethod
    def _decode_live_parsed_json(value) -> dict | None:
        """Return ledger_hands.parsed_json as a dict, accepting DB JSONB/string.

        Live detail buttons must reuse the already-ingested parsed hand.  The raw
        shorthand is only a user-facing echo; sending it back through the normal
        parser can produce a different hand than the one just graded.
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _live_hand_desc(hand_id: str, hand: dict) -> str:
        """Deterministic hand description for live deep-dive coaching."""
        desc = [
            f"Hand ID: {hand_id}",
            f"Hero {hand.get('hero_position')} {cards_to_emoji(hand.get('hero_hand'))} "
            f"({hand.get('effective_bb')}bb, {hand.get('players_at_table') or hand.get('num_players') or 8}人)",
            f"Preflop: {hand.get('preflop_actions')}",
        ]
        for street in hand.get("streets") or []:
            board = street.get("board") or street.get("card") or street.get("cards") or ""
            acts = " ".join(
                f"{a.get('position')}:{a.get('action')}"
                + (f"({a.get('size')}bb)" if a.get("size") is not None else "")
                for a in (street.get("actions") or [])
            )
            desc.append(f"{cards_to_emoji(board)} → {acts}".rstrip())
        repairs = hand.get("_repairs") or []
        if repairs:
            desc.append("Live parse repairs: " + "；".join(str(r) for r in repairs))
        return "\n".join(desc)

    async def _analyze_live_parsed_hand(self, chat_id: int, user_id: int,
                                        hand_id: str, hand_json: dict,
                                        on_status, refresh_token: str | None) -> str:
        """Analyze a stored live parsed_json without reparsing raw shorthand."""
        if not refresh_token:
            return "請先使用 /settoken 綁定你的 GTO Wizard 帳號。"

        if on_status:
            await on_status("查詢 GTO 策略中...")
        self._setup_user_token(user_id, refresh_token)
        try:
            from analyze_hand import analyze_hand_full
            context = analyze_hand_full(hand_json)
        finally:
            self._clear_user_token()

        gto_data = context["text"]
        self.session_manager.hand_contexts[chat_id] = context
        pending_images = getattr(self.session_manager, "pending_images", None)
        if isinstance(pending_images, dict):
            pending_images.pop(chat_id, None)

        if on_status:
            await on_status("分析回覆中...")
        prompt = (
            "這是 /live 入帳後的深入分析。手牌已經由 live_flow 解析、修補並入帳；"
            "請使用下面的穩定 parsed_json 摘要與 GTO Solver 數據，不要重新解析原始 shorthand。\n\n"
            f"手牌摘要：\n{self._live_hand_desc(hand_id, hand_json)}\n\n"
            f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
            "請根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
            "\n\n在回覆的最後，用以下格式輸出 3 個值得深入的 follow-up 問題（用戶可以點擊按鈕直接發送）：\n"
            "FOLLOWUP: 問題一\n"
            "FOLLOWUP: 問題二\n"
            "FOLLOWUP: 問題三\n"
        )
        response = await self.session_manager._chat_with_tools(
            chat_id, prompt, on_status=on_status,
            user_id=user_id, refresh_token=refresh_token,
            force_tool_eligible=False,
        )
        response, followups = self.session_manager._extract_followups(response)
        if followups:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if ctx is not None:
                ctx["followup_questions"] = followups
        warning = (context.get("validation") or {}).get("user_warning")
        if warning:
            response += f"\n\n{warning}"
        return f"📋 `{hand_id}`\n\n{response}"

    async def _analyze_online_parsed_hand(self, chat_id: int, user_id: int,
                                          hand_id: str, hand_json: dict,
                                          on_status, refresh_token: str | None) -> str:
        """Coach one archived online Analyzer hand through the grounded path."""
        if not refresh_token:
            return "請先使用 /settoken 綁定你的 GTO Wizard 帳號。"

        if on_status:
            await on_status("查詢 GTO 策略中...")
        def analyze_with_user_token():
            self._setup_user_token(user_id, refresh_token)
            try:
                from analyze_hand import analyze_hand_full
                return analyze_hand_full(hand_json)
            finally:
                self._clear_user_token()

        solver_context = await asyncio.to_thread(analyze_with_user_token)

        gto_data = solver_context["text"]
        self.session_manager.hand_contexts[chat_id] = solver_context
        pending_images = getattr(self.session_manager, "pending_images", None)
        if isinstance(pending_images, dict):
            pending_images.pop(chat_id, None)

        if on_status:
            await on_status("分析回覆中...")
        prompt = (
            "這是線上 session 總結中選出的 Analyzer 手牌。"
            "手牌已從封存的 GTOW Analyzer 原始資料精確重建；"
            "不要重新解析或改寫動作。\n\n"
            f"手牌摘要：\n{self._live_hand_desc(hand_id, hand_json)}\n\n"
            f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
            "請根據上面的 GTO 數據分析 hero 的行動，聚焦最大 EV loss 決策，"
            "並用可帶上桌的 heuristic 收尾。"
            "\n\n在回覆的最後，用以下格式輸出 3 個值得深入的 follow-up 問題：\n"
            "FOLLOWUP: 問題一\nFOLLOWUP: 問題二\nFOLLOWUP: 問題三\n"
        )
        response = await self.session_manager._chat_with_tools(
            chat_id, prompt, on_status=on_status, user_id=user_id,
            refresh_token=refresh_token, force_tool_eligible=False)
        response, followups = self.session_manager._extract_followups(response)
        if followups:
            ctx = self.session_manager.hand_contexts.get(chat_id)
            if ctx is not None:
                ctx["followup_questions"] = followups
        warning = (solver_context.get("validation") or {}).get("user_warning")
        if warning:
            response += f"\n\n{warning}"
        return f"📋 `{hand_id}`\n\n{response}"

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
            from gto_cache import entry_count as gto_cache_entry_count
            m["cache_total"] = gto_cache_entry_count()
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
        self.application.add_handler(CommandHandler("pair", self.pair_command))
        self.application.add_handler(CommandHandler("devices", self.devices_command))
        self.application.add_handler(CommandHandler("revoke", self.revoke_command))
        self.application.add_handler(CommandHandler("settoken", self.settoken_command))
        self.application.add_handler(CommandHandler("logout", self.logout_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler("ingest", self.ingest_command))
        self.application.add_handler(CommandHandler("fullingest", self.fullingest_command))
        self.application.add_handler(CommandHandler("live", self.live_command))
        self.application.add_handler(CommandHandler(
            ["lives", "live_sessions"], self.live_sessions_command))
        self.application.add_handler(CommandHandler(
            ["sessions", "online_sessions"], self.online_sessions_command))
        self.application.add_handler(CommandHandler("queue", self.queue_command))
        self.application.add_handler(CommandHandler("plan", self.plan_command))
        self.application.add_handler(CommandHandler("review", self.review_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.handle_photo))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        # pattern handlers must precede the generic follow-up handler
        self.application.add_handler(
            CallbackQueryHandler(
                self.handle_live_button,
                pattern=r"^(lvd|lvs|ors|lvpg|lvadd|lvr|qcl|qpg|qex|qad|qad2|qsrc|qraw|qdet|qdst|srd|srv|srd2|srv2|src2):"))
        self.application.add_handler(
            CallbackQueryHandler(self.handle_fullingest_button,
                                 pattern=r"^fullingest:"))
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
        self._text = None

    async def edit_text(self, text, **kwargs):
        if text == self._text:
            return
        try:
            await _tg_retry(
                lambda: self._msg.edit_text(
                    text, read_timeout=15, write_timeout=15, connect_timeout=15,
                    **kwargs,
                ),
                retries=2, label=self._label, log=self._log,
            )
            self._text = text
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
