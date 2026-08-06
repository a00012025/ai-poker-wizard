# src/gemini_session.py
"""Gemini-based session manager — direct API calls, no CLI subprocess.

Flow: user message → parse hand (Flash) → analyze_hand_full() → coaching (Pro)
Follow-ups: user message → parse (null) → Pro chat WITH query_gto tool → real data
"""
import asyncio
import contextvars
import copy
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

# Per-request correlation ID. Set at each entry point (send_message /
# send_image_message); ContextVar auto-propagates through asyncio tasks so
# every log line, tool call, and fire-and-forget task within the same request
# carries the same id — even under concurrent traffic.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    """Generate a short request id (8 hex chars)."""
    return uuid.uuid4().hex[:8]


class _RequestIdFilter(logging.Filter):
    """Inject request_id from ContextVar onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_LOG_DIR = _PROJECT_ROOT / "logs"
_FOLLOWUP_TIMEOUT_SECONDS = 180

# Allow importing from scripts/
sys.path.insert(0, str(_SCRIPTS_DIR))

from coach_prompts import (  # noqa: F401 — re-exported for existing importers
    PARSE_PROMPT, IMAGE_PARSE_PROMPT, HERO_HAND_ONLY_PROMPT, COACH_SYSTEM,
    _TERM_REPLACEMENTS, _normalize_terms, _GROUNDING_PATTERNS,
    _needs_solver_grounding,
)
from card_display import cards_to_emoji

QUERY_NEXT_ACTIONS_DECLARATION = types.FunctionDeclaration(
    name="query_next_actions",
    description=(
        "查詢某個決策點的所有可用動作及其 code。"
        "在建構假設情境（override actions）之前必須先呼叫此工具，以獲取正確的 action code（如 R3.6 而非猜測的 R1.2）。"
        "回傳每個可用動作的 code、betsize 和 betsize_by_pot。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "street": types.Schema(
                type=types.Type.STRING,
                enum=["preflop", "flop", "turn", "river"],
                description="要查詢哪條街的可用動作",
            ),
            "effective_bb": types.Schema(
                type=types.Type.NUMBER,
                description="有效籌碼深度（bb 數）。不同深度的 solver sizing 不同。不指定則使用目前手牌的深度。",
            ),
            "actions_so_far": types.Schema(
                type=types.Type.STRING,
                description="這條街到目前為止的動作序列（如果要查詢街中某個後續決策點）。例如查詢 flop 上 SB bet 後 BB 的選項，傳入 'R3.6'。留空表示查詢該街第一個行動者的選項。",
            ),
            "preflop_actions_override": types.Schema(
                type=types.Type.STRING,
                description=(
                    "覆蓋 preflop 動作序列（同 query_gto 的格式）。"
                    "用於查詢不同 preflop 路線下的可用動作。"
                ),
            ),
            "board_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的 board。",
            ),
            "flop_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的翻牌動作（查詢 turn/river 時使用）。",
            ),
            "turn_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的轉牌動作（查詢 river 時使用）。",
            ),
            "num_players": types.Schema(
                type=types.Type.INTEGER,
                description="桌上人數（6-9）。ICM 查詢時必須指定。",
            ),
            "icm_phase": types.Schema(
                type=types.Type.STRING,
                enum=["START", "PCT75", "PCT50", "PCT25", "PCT10", "PCT5",
                      "BUBBLEEARLY", "BUBBLEMID", "BUBBLELATE", "FT", "T2", "T3"],
                description=(
                    "ICM 錦標賽階段。指定後會使用 ICM solver 而非 Chip EV。"
                    "常見階段：START=初期, PCT25=剩25%人, BUBBLEMID=泡沫期, FT=決賽桌。"
                ),
            ),
            "player_stacks": types.Schema(
                type=types.Type.STRING,
                description=(
                    "ICM 各位置籌碼（bb），用逗號分隔，按座位順序（UTG 到 BB）。"
                    "例如 8 人桌全部 20bb: '20,20,20,20,20,20,20,20'。"
                    "不指定則預設所有人相同籌碼（= effective_bb）。"
                ),
            ),
        },
        required=["street"],
    ),
)

QUERY_GTO_DECLARATION = types.FunctionDeclaration(
    name="query_gto",
    description=(
        "查詢 GTO solver 策略數據。可以查詢目前手牌中任何位置在任何街的完整範圍或特定手牌策略。"
        "也可以修改 board 或 actions 來查詢假設情境。"
        "重要：使用 override actions 時，必須先用 query_next_actions 取得正確的 action code。"
        "查詢不同位置的 preflop 策略時，用 preflop_actions_override 指定到該位置行動前的動作序列。"
        "Raise size 不需要精確，系統會自動校正到最接近的 solver sizing（例如 R2 會自動校正為 R2.1）。"
        "\n\n用戶描述獨立情境（不基於已有手牌）時，必須同時提供："
        "effective_bb、preflop_actions_override、board_override，以及 flop/turn/river_actions_override。"
        "Board 必須帶花色（例如 QhTd3c），如果用戶沒指定花色就用 rainbow（不同花色）。"
        "Action 格式：X=check, C=call, F=fold, R{pot%}=bet/raise（如 R1.15 = ~33% pot bet）。"
        "查詢 turn 時，board_override 必須包含 turn 牌（4 張牌，例如 QhTd3c3s）。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "street": types.Schema(
                type=types.Type.STRING,
                enum=["preflop", "flop", "turn", "river"],
                description="要查詢哪條街的策略",
            ),
            "position": types.Schema(
                type=types.Type.STRING,
                description="要查詢哪個位置的範圍或策略（例如 BB, CO, BTN）。不指定則回傳當前行動者的整體策略。",
            ),
            "hand": types.Schema(
                type=types.Type.STRING,
                description=(
                    "查詢特定手牌的策略。不指定則回傳該位置的完整範圍概覽。\n"
                    "Postflop 查詢時，如果用戶指定了花色（如 Ah8h），必須傳入完整花色（如 Ah8h 而非 A8s），"
                    "因為不同花色在有同花/同花聽牌的牌面上策略差異極大。\n"
                    "例如 board Jc4d3s5d: Ad8d（方塊花聽）96% bet vs Ah8h（無聽牌）97% check。\n"
                    "Preflop 查詢用簡化格式即可：66, AKs, QTo。"
                ),
            ),
            "effective_bb": types.Schema(
                type=types.Type.NUMBER,
                description=(
                    "有效籌碼深度（bb 數）。當用戶問的情境深度與目前手牌不同時必須指定。"
                    "例如用戶問 '30bb effective' 就傳 30。系統會自動選擇最近的 solver 深度。"
                    "不指定則使用目前手牌的深度。"
                ),
            ),
            "preflop_actions_override": types.Schema(
                type=types.Type.STRING,
                description=(
                    "覆蓋 preflop 動作序列。格式：每個位置一個動作，按 UTG-UTG+1-LJ-HJ-CO-BTN-SB-BB 順序，用 - 分隔。"
                    "F=Fold, C=Call, RX=Raise to X, AI=All-in。Raise size 不用精確，系統會自動校正。"
                    "例如查詢 BB 面對 UTG+1 open 的策略：傳入 F-R2-F-F-F-F-F。"
                    "例如查詢 UTG+1 open 後 BB 3bet 後 UTG+1 的決策：傳入 F-R2-F-F-F-F-F-AI。"
                ),
            ),
            "board_override": types.Schema(
                type=types.Type.STRING,
                description=(
                    "指定 board 牌面（帶花色）。獨立情境查詢時必須提供。"
                    "Flop 查詢傳 3 張（如 QhTd3c），turn 查詢傳 4 張（如 QhTd3c3s），river 查詢傳 5 張。"
                    "也可用於覆蓋已有手牌的 board。"
                ),
            ),
            "flop_actions_override": types.Schema(
                type=types.Type.STRING,
                description=(
                    "翻牌動作序列。格式：X=check, C=call, F=fold, R{size}=bet/raise。\n"
                    "size 可以是絕對 bb 數（如 R3.7）或底池百分比（如 R50%）。系統會自動轉換百分比為正確的 bb 數。\n"
                    "推薦使用百分比格式，避免因 ante 導致底池計算錯誤。\n"
                    "例如 LJ bet 50% pot, BTN call = R50%-C。\n"
                    "查詢 flop 時：填到要查詢的決策點之前的動作。\n"
                    "查詢 turn 時：填完整的 flop 動作。"
                ),
            ),
            "turn_actions_override": types.Schema(
                type=types.Type.STRING,
                description=(
                    "轉牌動作序列。格式同上（支援 R50% 百分比格式）。"
                    "查詢 turn 某位置策略時，填到該位置行動前。"
                ),
            ),
            "river_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的河牌動作序列。格式同上（支援 R50% 百分比格式）。",
            ),
            "num_players": types.Schema(
                type=types.Type.INTEGER,
                description="桌上人數（6-9）。ICM 查詢時必須指定。",
            ),
            "icm_phase": types.Schema(
                type=types.Type.STRING,
                enum=["START", "PCT75", "PCT50", "PCT25", "PCT10", "PCT5",
                      "BUBBLEEARLY", "BUBBLEMID", "BUBBLELATE", "FT", "T2", "T3"],
                description=(
                    "ICM 錦標賽階段。指定後會使用 ICM solver 而非 Chip EV。"
                    "常見階段：START=初期, PCT25=剩25%人, BUBBLEMID=泡沫期, FT=決賽桌。"
                ),
            ),
            "player_stacks": types.Schema(
                type=types.Type.STRING,
                description=(
                    "ICM 各位置籌碼（bb），用逗號分隔，按座位順序（UTG 到 BB）。"
                    "例如 8 人桌全部 20bb: '20,20,20,20,20,20,20,20'。"
                    "不指定則預設所有人相同籌碼（= effective_bb）。"
                ),
            ),
        },
        required=["street"],
    ),
)

LOOKUP_HAND_DECLARATION = types.FunctionDeclaration(
    name="lookup_hand",
    description=(
        "根據 Hand ID 從用戶的手牌歷史中查詢手牌資料。"
        "用戶提到某個 Hand ID（如 H42 或 TM5600279272）時，使用此工具撈取手牌 JSON。"
        "可用於跨對話引用之前分析過的手牌。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "hand_id": types.Schema(
                type=types.Type.STRING,
                description="手牌 ID（如 H42 或 TM5600279272）",
            ),
        },
        required=["hand_id"],
    ),
)

EVALUATE_HAND_DECLARATION = types.FunctionDeclaration(
    name="evaluate_hand",
    description=(
        "判斷手牌在牌面上的確切牌型（成手牌 + 聽牌）。"
        "牌型判斷是 100% 確定性的，必須用此工具驗證，絕對不要自行推算。"
        "board 可省略，會自動使用當前最新牌面。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "hand": types.Schema(
                type=types.Type.STRING,
                description="手牌 (如 KQo, AhKh, T7s, 66)",
            ),
            "board": types.Schema(
                type=types.Type.STRING,
                description="牌面 (如 8hTc2sAc)，省略則用當前最新牌面",
            ),
        },
        required=["hand"],
    ),
)

# The strategy/range grounding gate must force a solver query, not the local
# postflop hand evaluator.  Keeping this separate from the full declaration
# list prevents Gemini from expanding a preflop range question into one
# evaluate_hand call per candidate combo (H3815).
_SOLVER_GROUNDING_TOOL_NAMES = ("query_gto", "query_next_actions")


# ── Training-loop Tool Declarations (ledger-backed; EV-weighted only) ──
# The frequency-era deviations tools (query_my_leaks / query_my_stats) were
# retired per North Star §7.3 — weakness/stats questions route to
# query_ledger_summary below.

GET_TRAINING_PLAN_DECLARATION = types.FunctionDeclaration(
    name="get_training_plan",
    description=(
        "取得本週訓練計畫（週日 21:00 自動生成的記分卡）：焦點 spot（EV loss 排序）"
        "+ GTOW Trainer drill 連結 + 上週焦點回讀 + 現場手牌練習佇列。"
        "當用戶問「我該練什麼」「給我訓練計畫」「本週計畫」時使用。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={},
        required=[],
    ),
)

GET_PROGRESS_DECLARATION = types.FunctionDeclaration(
    name="get_progress",
    description=(
        "查詢每週 EV loss 趨勢（bb/100 決策，帶樣本數 n）。可選按 spot 大類"
        "（RFI/vsOpen/vs3bet/…/flop/turn/river）或精確 spot_leaf 過濾。"
        "當用戶問「我有進步嗎」「XX 有改善嗎」時使用。"
        "注意：技能趨勢是月尺度，單週波動不是訊號。"
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "category": types.Schema(
                type=types.Type.STRING,
                description="spot 大類，如 vs3bet、flop（可選）",
            ),
            "spot_leaf": types.Schema(
                type=types.Type.STRING,
                description="精確 action-line spot leaf（可選）",
            ),
            "weeks": types.Schema(
                type=types.Type.INTEGER,
                description="查詢最近幾週（預設 8）",
            ),
        },
        required=[],
    ),
)

QUERY_LEDGER_SUMMARY_DECLARATION = types.FunctionDeclaration(
    name="query_ledger_summary",
    description=(
        "查詢全量帳本（GTOW Analyzer 評分的線上 MTT 決策，action-line 分類）的 "
        "EV loss 聚合。可按 spot 大類（RFI/vsOpen/vsRaiseCall/vs3bet/"
        "vs4bet/vsSqueeze/flop/turn/river）或 hero 位置類（EP/MP/LP/SB/BB）或天數過濾。"
        "回傳 EV loss/100 決策、總損失、樣本數 n、excluded 數與 top spot。"
        "使用者問『我最大的弱點 / 什麼地方打最差 / 我的統計 / 我哪裡漏 EV / "
        "某類 spot 表現如何 / 我 3bet pot 打得怎樣』時用這個。"
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "category": types.Schema(type=types.Type.STRING,
            description="spot 大類，如 vs3bet、flop、vsRaiseCall"),
        "hero_cat": types.Schema(type=types.Type.STRING,
            description="hero 位置類：EP/MP/LP/SB/BB"),
        "days": types.Schema(type=types.Type.INTEGER, description="回看天數，省略=全期"),
    }, required=[]),
)

QUERY_LEDGER_HANDS_DECLARATION = types.FunctionDeclaration(
    name="query_ledger_hands",
    description=(
        "列出帳本中符合條件的具體手牌（EV loss 排序），附 GTOW Analyze 復盤連結。"
        "使用者要看『哪幾手 / 最貴的手 / 某類 spot 的實例』時用這個。"
    ),
    parameters=types.Schema(type=types.Type.OBJECT, properties={
        "category": types.Schema(type=types.Type.STRING, description="spot 大類"),
        "min_ev_loss": types.Schema(type=types.Type.NUMBER, description="bb 門檻，預設 0.5"),
        "days": types.Schema(type=types.Type.INTEGER, description="預設 90"),
        "limit": types.Schema(type=types.Type.INTEGER, description="預設 5，最大 10"),
    }, required=[]),
)


class GeminiSessionManager:
    def __init__(self, db=None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境變數未設定")

        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.parse_model = os.getenv("GEMINI_PARSE_MODEL", "gemini-2.5-flash")
        self.image_parse_model = os.getenv("GEMINI_IMAGE_PARSE_MODEL", "gemini-pro-latest")
        self.max_turns = "N/A"  # for bot.py compat
        self.histories: Dict[int, List[types.Content]] = {}
        self.hand_contexts: Dict[int, dict] = {}
        self.pending_images: Dict[int, list] = {}  # chat_id → [(bytes, title)]
        # Last hand_id analyzed per chat — used to correlate follow-up
        # tool calls back to the hand they're asking about.
        self.last_hand_ids: Dict[int, str] = {}
        self.db = db

        # Logging. File + stdout, both carry request_id from ContextVar.
        _LOG_DIR.mkdir(exist_ok=True)
        self._logger = logging.getLogger("gemini_session")
        if not self._logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s [%(levelname)s] [req=%(request_id)s] %(message)s"
            )
            req_filter = _RequestIdFilter()

            file_handler = logging.FileHandler(
                _LOG_DIR / "gemini_session.log", encoding="utf-8"
            )
            file_handler.setFormatter(fmt)
            file_handler.addFilter(req_filter)
            self._logger.addHandler(file_handler)

            # Mirror to stdout so `docker logs` shows tool calls without
            # shelling into the container.
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(fmt)
            stream_handler.addFilter(req_filter)
            self._logger.addHandler(stream_handler)

            self._logger.setLevel(logging.DEBUG)
            # Don't propagate to root (avoid duplicate lines via bot.py's handler)
            self._logger.propagate = False

    @staticmethod
    def _extract_usage(response) -> dict:
        """Extract token usage from a Gemini API response."""
        um = getattr(response, "usage_metadata", None)
        if not um:
            return {}
        return {
            "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
            "cached_tokens": getattr(um, "cached_content_token_count", 0) or 0,
            "thinking_tokens": getattr(um, "thoughts_token_count", 0) or 0,
            "total_tokens": getattr(um, "total_token_count", 0) or 0,
        }

    @staticmethod
    def _accumulate_usage(acc: dict, usage: dict):
        """Add usage dict into accumulator."""
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens",
                     "thinking_tokens", "total_tokens"):
            acc[key] = acc.get(key, 0) + usage.get(key, 0)
        acc["api_calls"] = acc.get("api_calls", 0) + 1

    async def _save_usage(self, chat_id: int, request_type: str, model: str,
                           acc: dict, latency_ms: int | None = None):
        """Save accumulated token usage to DB."""
        if not self.db or not acc.get("api_calls"):
            return
        try:
            await self.db.log_token_usage(
                chat_id=chat_id, request_type=request_type, model=model,
                prompt_tokens=acc.get("prompt_tokens", 0),
                completion_tokens=acc.get("completion_tokens", 0),
                cached_tokens=acc.get("cached_tokens", 0),
                thinking_tokens=acc.get("thinking_tokens", 0),
                total_tokens=acc.get("total_tokens", 0),
                api_calls=acc.get("api_calls", 0),
                latency_ms=latency_ms,
            )
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Failed to log token usage: {e}")

    def _setup_user_token(self, user_id: int | None, refresh_token: str | None):
        """Set thread-local GTO token if user has one."""
        if user_id and refresh_token:
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

    async def _ensure_hand_context(self, chat_id: int,
                                   user_id: int | None = None,
                                   refresh_token: str | None = None) -> bool:
        """Rehydrate the in-memory follow-up context from the DB if missing.

        ``self.hand_contexts`` lives only in process memory, so a bot
        restart/deploy wipes every chat's "last analyzed hand".  A follow-up
        that arrives after such a restart then has no hand to reference and the
        coach replies "I need to know which hand" (H3515).

        When no context is cached for the chat, look up the most recent
        analysis snapshot, re-run ``analyze_hand_full`` to rebuild the full
        context (street states, hero spots, cached solutions) and cache it.
        Returns True if a context is available afterwards.

        Best-effort: needs the DB + the user's GTO token; any failure leaves
        the context empty (caller falls back to the existing no-context path).
        """
        if self.hand_contexts.get(chat_id) is not None:
            return True
        if not self.db or not refresh_token:
            return False
        try:
            last = await self.db.get_last_hand(chat_id)
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] get_last_hand failed: {e}")
            return False
        if not last:
            return False
        hand_json = last["hand"]
        hand_id = last.get("hand_id")
        try:
            def _analyze_with_token():
                self._setup_user_token(user_id, refresh_token)
                try:
                    from analyze_hand import analyze_hand_full
                    return analyze_hand_full(hand_json)
                finally:
                    self._clear_user_token()

            context = await asyncio.to_thread(_analyze_with_token)
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] rehydrate analyze_hand_full failed "
                f"({hand_id}): {e}")
            return False
        self.hand_contexts[chat_id] = context
        if hand_id:
            self.last_hand_ids[chat_id] = hand_id
        self._logger.info(
            f"[chat={chat_id}] Rehydrated hand context from DB "
            f"({hand_id}) after missing in-memory state")
        return True

    def _try_coach_facts(self, chat_id: int, user_text: str,
                         user_id: int | None = None,
                         refresh_token: str | None = None) -> str | None:
        """Deterministic grounded answer for P0/P1 follow-up intents.

        Routes the question through scripts/coach_facts (classify -> fetch the
        right spot-solution node(s) -> grounded narrate -> hard verify). Returns
        the answer string, or None to fall back to the tool-calling path
        ('other'/unknown intents, no cached hand, or any failure).

        Synchronous + blocking (Gemini classify/narrate + cached solver fetch);
        the async caller runs it via asyncio.to_thread.
        """
        ctx = self.hand_contexts.get(chat_id)
        if not ctx or not ctx.get("solutions"):
            return None
        try:
            import coach_facts
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] coach_facts import failed: {e}")
            return None
        try:
            self._setup_user_token(user_id, refresh_token)
            try:
                answer, facts = coach_facts.answer_followup_ex(coach_facts.Ctx(
                    question=user_text, hand_context=ctx,
                    user_id=user_id, refresh_token=refresh_token,
                ))
            finally:
                self._clear_user_token()
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] coach_facts failed: {e}")
            return None
        if answer:
            self._logger.info(
                f"[chat={chat_id}] coach_facts grounded answer ({len(answer)} chars)")
            # Range/strategy intents describe a node's distribution; draw the
            # 13x13 grid so range answers come with a chart like the tool path.
            self._queue_grounded_range_chart(chat_id, facts)
            # Point the GTO Wizard button at the exact node this answer is
            # grounded on (e.g. hero's turn decision) so the link's
            # frequencies match the prose — not the played-line river node.
            node_street = (getattr(facts, "meta", {}) or {}).get("node_street")
            if node_street:
                ctx["_followup_node_street"] = node_street
        return answer

    def _queue_grounded_range_chart(self, chat_id: int, facts) -> None:
        """Queue a range grid for a grounded coach_facts answer, if chartable.

        The deterministic coach_facts path bypasses _execute_query_gto (the
        only other place that queues range charts), so range/strategy
        follow-ups answered here had no image.  When the Facts carries a
        ``meta["chart"]`` (solution + acting position), render the grid and
        push it onto ``pending_images`` for the bot to flush after the reply.
        Non-critical: any failure just means no chart.
        """
        chart = getattr(facts, "meta", {}).get("chart") if facts else None
        if not chart:
            return
        sol = chart.get("solution")
        position = chart.get("position")
        if not sol or not position:
            return
        try:
            from range_image import generate_range_grid
            game = sol.get("game", {})
            st = game.get("current_street", {}).get("type", "").capitalize()
            board = game.get("board", "")
            title = f"{position} {st}".strip()
            if board:
                title += f" | {board}"
            img = generate_range_grid(sol, position, title=title)
            if chat_id not in self.pending_images:
                self.pending_images[chat_id] = []
            self.pending_images[chat_id].append((img, f"📊 {title}"))
            self._logger.info(
                f"[chat={chat_id}] queued grounded range chart ({title})")
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] grounded range chart failed: {e}")

    async def _save_snapshot(self, hand_id: str, chat_id: int,
                              source_type: str, user_input: str | None,
                              image_data: bytes | None,
                              parsed_json: dict, context: dict,
                              *, classifier_conf: float | None = None):
        """Fire-and-forget: save analysis snapshot to DB."""
        if not self.db or not hand_id:
            return
        try:
            await self.db.save_snapshot(
                hand_id=hand_id, chat_id=chat_id,
                source_type=source_type,
                user_input=user_input[:2000] if user_input else None,
                image_data=image_data,
                parsed_json=parsed_json,
                gto_text=context.get("text", ""),
                gto_compact=context.get("text_compact"),
                classifier_conf=classifier_conf,
            )
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Failed to save snapshot: {e}")

    async def _update_snapshot_coaching(self, hand_id: str, chat_id: int,
                                         coaching_text: str):
        """Fire-and-forget: update coaching text in snapshot."""
        if not self.db or not hand_id:
            return
        try:
            await self.db.update_snapshot_coaching(hand_id, coaching_text)
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Failed to update snapshot coaching: {e}")

    async def _cross_check_ocr_vs_gemini(self, chat_id: int, image_bytes: bytes,
                                           mime_type: str, user_text: str,
                                           ocr_hand: dict, ocr_conf: float):
        """Medium-tier safety net: call Gemini on the same image, compare
        hero/board cards, and log any disagreement as a labeled example for
        future classifier retraining. Fire-and-forget; never raises.

        The user-facing flow has already returned the OCR hand by the time
        this runs. We are not fixing this analysis — we are building the
        corpus that will fix the NEXT analysis after a retrain.
        """
        try:
            prompt_text = IMAGE_PARSE_PROMPT
            if user_text.strip():
                prompt_text += f"\n\n用戶留言：{user_text.strip()}"
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.image_parse_model,
                    contents=[
                        types.Content(role="user", parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                            types.Part(text=prompt_text),
                        ]),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        thinking_config=types.ThinkingConfig(thinking_budget=4096),
                    ),
                ),
                timeout=300,
            )
            text = response.text or ""
            m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            gemini_hand = (json.loads(m.group(1)) if m else json.loads(text.strip())).get("hand")
            if not gemini_hand:
                return
            self._normalize_cards(gemini_hand)
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] cross-check Gemini call failed: {e}")
            return

        disagreement = self._cards_disagreement(ocr_hand, gemini_hand)
        if not disagreement:
            self._logger.info(f"[chat={chat_id}] cross-check OK (OCR==Gemini)")
            return

        self._logger.info(
            f"[chat={chat_id}] cross-check DISAGREEMENT "
            f"(conf={ocr_conf:.2f}): {disagreement}"
        )
        if not self.db:
            return
        try:
            await self.db.log_classifier_disagreement(
                chat_id=chat_id,
                ocr_hand=ocr_hand,
                gemini_hand=gemini_hand,
                ocr_conf=ocr_conf,
                diff=disagreement,
            )
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] Failed to log classifier disagreement: {e}")

    async def _gemini_hero_hand_only(
        self, chat_id: int, image_bytes: bytes, mime_type: str,
        ocr_hand: dict, hints: dict | None = None,
        usage_acc: dict | None = None,
    ) -> str | None:
        """Cards-only Gemini call: returns hero_hand string or None.

        Used when OCR's structural fields (position, stacks, actions) look
        reliable but the card classifier's hero prediction was below
        threshold. Avoids the full IMAGE_PARSE_PROMPT, so Gemini cannot
        override OCR's hero_position / stacks / actions.
        """
        prompt_text = HERO_HAND_ONLY_PROMPT
        ctx = {
            "hero_position": ocr_hand.get("hero_position"),
            "players_at_table": ocr_hand.get("players_at_table"),
        }
        prompt_text += (
            f"\n\nOCR 已確定的上下文（請僅作為定位 hero 的參考，不要重新判斷）："
            f"{json.dumps(ctx, ensure_ascii=False)}"
        )
        if hints and hints.get("hero_card_suits"):
            suits = hints["hero_card_suits"]
            prompt_text += (
                f"\n\nCardCNN suit 分類器對 hero 兩張牌花色高信心，"
                f"由左至右為 {suits}（hero_hand 須以 rank 大者排前）。"
            )

        hero_image_bytes, hero_mime_type = self._hero_cards_image_for_micro_read(
            image_bytes, fallback_mime_type=mime_type
        )

        response = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.image_parse_model,
                        contents=[
                            types.Content(role="user", parts=[
                                types.Part.from_bytes(
                                    data=hero_image_bytes,
                                    mime_type=hero_mime_type),
                                types.Part(text=prompt_text),
                            ]),
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=2048),
                        ),
                    ),
                    timeout=180,
                )
                break
            except genai_errors.ServerError as e:
                if attempt == 2:
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"[chat={chat_id}] Cards-only Gemini transient error "
                    f"(attempt {attempt + 1}/3): {e}. Retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
        if response is None:
            return None
        if usage_acc is not None:
            self._accumulate_usage(usage_acc, self._extract_usage(response))

        text = response.text or ""
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        json_str = m.group(1) if m else text.strip()
        try:
            result = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            self._logger.warning(
                f"[chat={chat_id}] Cards-only Gemini JSON parse failed: {e}; "
                f"raw: {text[:200]}"
            )
            return None
        hero_hand = result.get("hero_hand") if isinstance(result, dict) else None
        if not hero_hand or not isinstance(hero_hand, str):
            return None
        # Sanity: must be 4 chars (RsRs format) and only valid rank/suit chars
        if len(hero_hand) != 4:
            self._logger.warning(
                f"[chat={chat_id}] Cards-only Gemini returned malformed "
                f"hero_hand: {hero_hand!r}"
            )
            return None
        ranks = set("23456789TJQKA")
        suits = set("cdhs")
        if (hero_hand[0] not in ranks or hero_hand[2] not in ranks
                or hero_hand[1] not in suits or hero_hand[3] not in suits):
            self._logger.warning(
                f"[chat={chat_id}] Cards-only Gemini returned non-card chars: "
                f"{hero_hand!r}"
            )
            return None
        return hero_hand

    @staticmethod
    def _hero_cards_image_for_micro_read(
        image_bytes: bytes,
        *,
        fallback_mime_type: str = "image/jpeg",
    ) -> tuple[bytes, str]:
        """Return a tightly scoped hero-card image for the cards-only VLM.

        The full-image fallback has measured regressions because it lets the
        VLM re-interpret position/action/board fields.  For the micro-route,
        send only the bottom hero-card crop when possible.  If the OCR cropper
        cannot localize the cards, fall back to the original bytes so the
        cards-only prompt still works rather than failing the whole parse.
        """
        try:
            import cv2
            import numpy as np
            from ocr.region_detector import detect_regions
            from ocr.table_parser import _locate_hero_cards

            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return image_bytes, fallback_mime_type
            regions = detect_regions(image)
            table = regions.get("table") if regions else image
            crops = _locate_hero_cards(table)
            if len(crops) >= 2:
                padded: list[np.ndarray] = []
                max_h = 0
                for crop in crops[:2]:
                    if crop is None or crop.size == 0:
                        continue
                    pad_y = max(2, int(crop.shape[0] * 0.08))
                    pad_x = max(2, int(crop.shape[1] * 0.08))
                    c = cv2.copyMakeBorder(
                        crop, pad_y, pad_y, pad_x, pad_x,
                        borderType=cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    padded.append(c)
                    max_h = max(max_h, c.shape[0])
                if len(padded) >= 2:
                    normalized = []
                    for c in padded:
                        if c.shape[0] < max_h:
                            extra = max_h - c.shape[0]
                            c = cv2.copyMakeBorder(
                                c, 0, extra, 0, 0,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(255, 255, 255),
                            )
                        normalized.append(c)
                    spacer = np.full((max_h, 10, 3), 255, dtype=np.uint8)
                    montage = np.hstack([normalized[0], spacer, normalized[1]])
                    ok, enc = cv2.imencode(".png", montage)
                    if ok:
                        return enc.tobytes(), "image/png"

            # Conservative fallback crop: bottom-center hero area of the table.
            h, w = table.shape[:2]
            hero = table[int(h * 0.55):int(h * 0.98),
                         int(w * 0.24):int(w * 0.72)]
            if hero.size:
                ok, enc = cv2.imencode(".png", hero)
                if ok:
                    return enc.tobytes(), "image/png"
        except Exception:
            pass
        return image_bytes, fallback_mime_type

    @staticmethod
    def _merge_ocr_with_gemini_hero_hand(
        ocr_hand: dict, gemini_hero_hand: str
    ) -> dict:
        """Take OCR's full hand parse, replace only hero_hand.

        Used when card_conf < threshold but structural_conf is high enough
        that we trust OCR's hero_position/stacks/actions/streets. Keeps the
        rest of OCR's parse intact so Gemini can't override blind-based
        position detection (regression: H2790, where Gemini's visual
        position read flipped SB → BB).
        """
        merged = dict(ocr_hand)
        merged["hero_hand"] = gemini_hero_hand
        return merged

    @staticmethod
    def _cards_only_merge_safe(
        ocr_result: dict,
        gemini_hero_hand: str | None,
    ) -> bool:
        """Whether a cards-only fallback may keep OCR's structural parse.

        The micro-route is high value only when Gemini is constrained to hero
        cards and OCR's position/action/board structure is already trustworthy.
        This predicate mirrors the 718-hand precision gate: accept changed
        hero cards only above a classifier-confidence floor, accept unchanged
        cards only for structurally stable OCR/VLM shapes, and reject known
        board/action-risk diagnostics before falling through to full Gemini.
        """
        hand = ocr_result.get("hand") or {}
        if not (hand.get("hero_position") and hand.get("preflop_actions")):
            return False

        diagnostics = ocr_result.get("diagnostics") or {}
        physics_issues = diagnostics.get("preflop_physics_issues") or []
        allowed_physics = (
            diagnostics.get("preflop_forced_collapse_repairs")
            and all(str(issue).startswith("too_few_initial_actions")
                    for issue in physics_issues)
        )
        if diagnostics.get("structural_risk_issues"):
            return False
        if physics_issues and not allowed_physics:
            return False

        card_conf = float(ocr_result.get("card_confidence") or 0.0)
        parts = ocr_result.get("confidence_parts") or {}
        ocr_conf = float(parts.get("ocr_confidence") or 0.0)
        pot_conf = float(parts.get("pot_consistency") or 0.0)
        player_conf = float(parts.get("player_tracking") or 0.0)
        original_hero = hand.get("hero_hand")
        changed = bool(
            gemini_hero_hand
            and original_hero
            and gemini_hero_hand != original_hero
        )

        if changed:
            return card_conf >= float(
                os.getenv("OCR_CARDS_ONLY_CHANGED_CARD_MIN", "0.38")
            )

        street_counts = diagnostics.get("street_entries_count") or {}
        street_pre_counts = (
            diagnostics.get("street_entries_pre_collapse_count") or {}
        )
        hidden_street_fragments = sum(
            max(
                0,
                int(street_pre_counts.get(name) or 0)
                - int(street_counts.get(name) or 0),
            )
            for name in ("flop", "turn", "river")
        )
        postflop_rows = sum(int(v or 0) for v in street_counts.values())
        pre_count = diagnostics.get("preflop_entries_count")
        pre_collapse = diagnostics.get("preflop_entries_pre_collapse_count")
        preloss = (
            pre_collapse - pre_count
            if isinstance(pre_collapse, int) and isinstance(pre_count, int)
            else 99
        )
        tokens = []
        for tok in str(hand.get("preflop_actions") or "").split("-"):
            if not tok:
                continue
            if tok.startswith("AI"):
                tokens.append("AI")
            elif tok.startswith("R"):
                tokens.append("R")
            else:
                tokens.append(tok)
        first = tokens[0] if tokens else ""
        last = tokens[-1] if tokens else ""
        ai_count = tokens.count("AI")
        raise_count = tokens.count("R")
        vlm_outcome = diagnostics.get("vlm_recheck_outcome")

        if (
            ocr_conf == 1.0
            and card_conf >= 0.30
            and pot_conf >= 0.5
            and player_conf >= 0.5
        ):
            return True

        if (
            vlm_outcome == "corrected"
            and card_conf >= 0.99
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and hidden_street_fragments == 0
        ):
            return True

        if (
            diagnostics.get("preflop_terminal_fold_repairs")
            and card_conf >= 0.60
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and postflop_rows == 0
        ):
            return True

        if (
            vlm_outcome == "corrected"
            and card_conf >= 0.99
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and postflop_rows == 0
            and hidden_street_fragments == 3
            and ai_count == 1
            and raise_count == 1
            and first == "F"
            and last == "C"
        ):
            return True

        if (
            vlm_outcome == "corrected"
            and card_conf >= 0.99
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and postflop_rows == 0
            and hidden_street_fragments == 4
            and first == "C"
        ):
            return True

        if (
            vlm_outcome == "corrected"
            and card_conf >= 0.99
            and pot_conf >= 1.0
            and player_conf >= 0.5
            and postflop_rows == 0
            and hidden_street_fragments == 3
            and first == "R"
            and ai_count == 1
            and raise_count == 1
            and preloss >= 10
        ):
            return True

        return False

    @staticmethod
    def _can_keep_ocr_abstain_after_cards_only(
        *,
        confidence_abstain_with_ocr: bool,
        hero_hand_present: bool,
        cards_need_fallback: bool,
        original_hero_hand: str | None,
        gemini_hero_hand: str | None,
    ) -> bool:
        """Whether to keep OCR after a cards-only re-read produced no repair.

        This is intentionally false when the card-confidence gate fired.  In
        that case OCR already told us its hero cards are untrustworthy; keeping
        the original hero_hand after Gemini returns nothing (or repeats the
        same bad value) reintroduces the exact low-confidence card hallucination.
        Regression: H3473, hero KhJc was emitted/coached as AhAs.
        """
        return (
            confidence_abstain_with_ocr
            and hero_hand_present
            and not cards_need_fallback
            and (
                not gemini_hero_hand
                or gemini_hero_hand == original_hero_hand
            )
        )

    @staticmethod
    def _gemini_ocr_context(
        ocr_result: dict | None,
        min_card_conf: float,
    ) -> tuple[dict | None, dict | None, bool]:
        """Return OCR hints/partial hand safe to pass into full Gemini parse.

        OCR structural context is valuable, but a below-threshold hero-card
        classifier prediction must not be used as a strong visual hint.  When
        card_confidence is below the hard floor, remove only the card-specific
        hero fields while preserving board/action/stacks context for Gemini.
        """
        if not ocr_result:
            return None, None, False

        hints = copy.deepcopy(ocr_result.get("hints") or {})
        partial = copy.deepcopy(ocr_result.get("hand") or {})
        try:
            card_conf = float(ocr_result.get("card_confidence") or 0.0)
        except (TypeError, ValueError):
            card_conf = 0.0
        low_card_conf = bool(partial) and card_conf < min_card_conf

        if low_card_conf:
            hints.pop("hero_cards", None)
            partial_hint = hints.get("partial_hand")
            if isinstance(partial_hint, dict):
                partial_hint.pop("hero_hand", None)
                partial_hint["hero_hand_low_confidence"] = True
            if partial:
                partial.pop("hero_hand", None)
                partial["hero_hand_low_confidence"] = True
            hints["hero_cards_low_confidence"] = {
                "card_confidence": card_conf,
                "min_required": min_card_conf,
                "instruction": (
                    "Ignore OCR hero_cards/hero_hand; re-read hero's two "
                    "bottom-center cards from the image."
                ),
            }

        return (hints or None), (partial or None), low_card_conf

    @staticmethod
    def _cards_disagreement(ocr_hand: dict, gemini_hand: dict) -> dict | None:
        """Compare hero_hand + per-street board cards. Return a small dict
        describing the disagreement, or None if the two parses agree on
        every card."""
        diffs: dict = {}
        o_hero = ocr_hand.get("hero_hand")
        g_hero = gemini_hand.get("hero_hand")
        if o_hero != g_hero:
            diffs["hero"] = {"ocr": o_hero, "gemini": g_hero}
        o_streets = ocr_hand.get("streets") or []
        g_streets = gemini_hand.get("streets") or []
        for i in range(max(len(o_streets), len(g_streets))):
            o = o_streets[i] if i < len(o_streets) else {}
            g = g_streets[i] if i < len(g_streets) else {}
            o_board = o.get("board", o.get("card", ""))
            g_board = g.get("board", g.get("card", ""))
            if o_board != g_board:
                diffs.setdefault("streets", {})[str(i)] = {
                    "ocr": o_board, "gemini": g_board,
                }
        return diffs or None

    async def _extract_deviations(self, chat_id: int, hand_id: str | None,
                                    hand_json: dict, context: dict):
        """Fire-and-forget deviation capture — delegates to
        scripts/deviation_extract.extract_deviations (god-file split)."""
        from deviation_extract import extract_deviations
        await extract_deviations(self.db, self._logger, chat_id, hand_id,
                                 hand_json, context)

    async def send_message(self, chat_id: int, user_text: str,
                           on_status: Callable[[str], Any] | None = None,
                           user_id: int | None = None,
                           refresh_token: str | None = None,
                           send_gto_callback: Callable[[str], Any] | None = None) -> str:
        """Main entry: parse hand → GTO analysis → coaching, or chat with tools.

        Args:
            on_status: optional async/sync callback(status_msg) for progress updates
            user_id: Telegram user ID for per-user token lookup
            refresh_token: user's GTO Wizard refresh token (if any)
            send_gto_callback: optional async/sync callback(text) to send the
                structured per-street GTO summary card immediately, before the
                slower coaching reply.  Mirrors the image pipeline's split
                response so text hands with a concrete hero hand feel faster.
        """
        request_id_var.set(new_request_id())
        t0 = time.time()
        self._logger.info(f"[chat={chat_id}] User: {user_text[:300]}")
        usage_acc = {}

        async def _status(msg: str):
            if on_status:
                r = on_status(msg)
                if asyncio.iscoroutine(r):
                    await r

        try:
            # Check for FT switch request on previous hand
            ft_switch_keywords = {"決賽桌分析", "FT分析", "用ICM", "用icm", "切換決賽桌", "final table分析"}
            stripped = user_text.strip().lower()
            if any(kw.lower() in stripped for kw in ft_switch_keywords):
                ctx = self.hand_contexts.get(chat_id)
                if ctx and not ctx.get("is_icm"):
                    prev_hand = ctx["hand"]
                    prev_hand["tournament_type"] = "icm"
                    prev_hand["phase"] = "FT"
                    # player_stacks should already be present from image parse
                    hand_json = prev_hand
                    t_parse = time.time()
                    self._logger.info(
                        f"[chat={chat_id}] FT switch: re-analyzing with ICM"
                    )

                    # Re-run GTO analysis with ICM
                    if not refresh_token:
                        return "請先使用 /settoken 綁定你的 GTO Wizard 帳號。"
                    await _status("切換到 ICM 決賽桌模式，重新查詢 GTO 策略...")
                    self._setup_user_token(user_id, refresh_token)
                    try:
                        from analyze_hand import analyze_hand_full
                        context = analyze_hand_full(hand_json)
                    finally:
                        self._clear_user_token()
                    gto_data = context["text"]
                    self.hand_contexts[chat_id] = context
                    self.pending_images.pop(chat_id, None)

                    coaching_prompt = (
                        f"用戶要求切換到 ICM 決賽桌模式重新分析。\n\n"
                        f"GTO Solver 數據（ICM 模式）：\n{gto_data}\n\n"
                        f"請分析 hero 在 ICM 決賽桌下的最佳策略，並與之前的 Chip EV 分析做比較。"
                    )
                    result = await self._verified_initial_coaching(
                        chat_id, coaching_prompt, context, user_text,
                        on_status=on_status, user_id=user_id,
                        refresh_token=refresh_token, usage_acc=usage_acc,
                    )
                    result, followups = self._extract_followups(result)
                    if followups:
                        ctx = self.hand_contexts.get(chat_id)
                        if ctx is not None:
                            ctx["followup_questions"] = followups
                    elapsed = time.time() - t0
                    await self._save_usage(chat_id, "hand_analysis", self.model,
                                           usage_acc, int(elapsed * 1000))
                    return result

            # Rehydrate the last analyzed hand from the DB when the in-memory
            # context was lost (bot restart/deploy) and this message looks like
            # a follow-up rather than a new hand.  Without this, follow-ups
            # after a deploy reply "I need to know which hand" (H3515).  Doing
            # it before parse also restores the _parse_hand guard so the
            # follow-up isn't misparsed into a fake hand.
            if (chat_id not in self.hand_contexts
                    and not self._text_looks_like_hand(user_text)):
                await self._ensure_hand_context(chat_id, user_id, refresh_token)

            # Step 1: Parse hand from user message (Flash — fast)
            await _status("解析手牌中...")
            hand_json = await asyncio.wait_for(
                self._parse_hand(chat_id, user_text, usage_acc=usage_acc), timeout=60,
            )
            if hand_json:
                hand_json = await self._reparse_if_rules_invalid(
                    chat_id, user_text, hand_json, usage_acc)
            t_parse = time.time()

            if hand_json:
                self._logger.info(
                    f"[chat={chat_id}] Parsed hand in {t_parse - t0:.1f}s "
                    f"(model={self.parse_model}): "
                    f"{json.dumps(hand_json, ensure_ascii=False)[:300]}"
                )

                # Save parsed hand to DB and get hand_id
                hand_id = None
                if self.db:
                    try:
                        hand_id = await self.db.save_hand_returning_id(
                            chat_id, hand_json, source_type="text",
                            user_input=user_text[:2000])
                    except Exception as e:
                        self._logger.warning(f"[chat={chat_id}] Failed to save hand: {e}")

                # Step 2: Require user token
                if not refresh_token:
                    return "請先使用 /settoken 綁定你的 GTO Wizard 帳號。"

                # Step 3: Run GTO analysis and cache context
                await _status("查詢 GTO 策略中...")
                self._setup_user_token(user_id, refresh_token)
                try:
                    from analyze_hand import analyze_hand_full
                    context = analyze_hand_full(hand_json)
                finally:
                    self._clear_user_token()
                gto_data = context["text"]
                self.hand_contexts[chat_id] = context
                if hand_id:
                    self.last_hand_ids[chat_id] = hand_id
                self.pending_images.pop(chat_id, None)
                # Save snapshot (fire-and-forget)
                import asyncio as _aio
                _aio.create_task(self._save_snapshot(
                    hand_id, chat_id, "text", user_text,
                    None, hand_json, context))
                # Extract deviations for leak detection (fire-and-forget)
                _aio.create_task(self._extract_deviations(
                    chat_id, hand_id, hand_json, context))

                t_analyze = time.time()
                self._logger.info(
                    f"[chat={chat_id}] GTO analysis in {t_analyze - t_parse:.1f}s "
                    f"({len(gto_data)} chars) — context cached"
                )
                self._logger.debug(f"[chat={chat_id}] GTO data:\n{gto_data}")

                # Exact ICM range requests must stay deterministic.  Letting
                # the coaching LLM rewrite these has caused two user-visible
                # failures: it skipped the approximation note and reordered
                # the user's FT stack list.  But returning the raw solver
                # artifact is too terse, so build a deterministic coach
                # response from the cached solver data: exact stack order,
                # explicit approximation details, action frequencies, and the
                # range breakdown without free-form hallucination risk.
                if context.get("no_hero_hand") and context.get("is_icm") and not hand_json.get("streets"):
                    result = self._format_icm_range_coach_response(context, gto_data)
                    context["followup_questions"] = self._build_icm_range_followups(context)
                    _aio.create_task(self._update_snapshot_coaching(
                        hand_id, chat_id, result))
                    if hand_id:
                        result = f"📋 `{hand_id}`\n\n{result}"
                    _vwarn = (context.get("validation") or {}).get("user_warning")
                    if _vwarn:
                        result += f"\n\n{_vwarn}"
                    t_total = time.time()
                    await self._save_usage(chat_id, "hand_analysis", self.model,
                                           usage_acc, int((t_total - t0) * 1000))
                    return result

                # Split response: send the structured per-street GTO summary
                # card immediately, before the slow coaching call.  Only for a
                # concrete hero hand (range-only questions have no per-hand
                # verdict to show) — mirrors the image pipeline so text hands
                # feel as responsive.  Non-fatal: a failed send just falls back
                # to the single combined reply.
                if send_gto_callback and not context.get("no_hero_hand"):
                    gto_summary = context.get("text_compact", gto_data)
                    if hand_id:
                        gto_summary = f"📋 `{hand_id}`\n\n{gto_summary}"
                    try:
                        r = send_gto_callback(gto_summary)
                        if asyncio.iscoroutine(r):
                            await r
                        self._logger.info(
                            f"[chat={chat_id}] GTO summary sent at "
                            f"{time.time() - t0:.1f}s"
                        )
                    except Exception:
                        self._logger.warning(
                            f"[chat={chat_id}] Failed to send GTO summary "
                            f"(non-fatal)"
                        )

                # Step 4: Coaching from LLM (with tools for follow-up queries)
                await _status("分析回覆中...")
                if context.get("no_hero_hand"):
                    coaching_instruction = (
                        "GTO 數據是該位置的整體範圍策略。"
                        "如果用戶問到特定手牌（如 Ks8s, AQs），你必須先用 query_gto 工具查詢該手牌的策略數據，"
                        "再用 evaluate_hand 工具確認該手牌的牌型和聽牌，然後根據工具回傳的數據回答。"
                        "絕對不要在沒有查詢的情況下自行編造特定手牌的頻率或聽牌描述！"
                    )
                else:
                    coaching_instruction = "請先根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
                followup_instruction = (
                    "\n\n在回覆的最後，用以下格式輸出 3 個值得深入的 follow-up 問題（用戶可以點擊按鈕直接發送）：\n"
                    "FOLLOWUP: 問題一\n"
                    "FOLLOWUP: 問題二\n"
                    "FOLLOWUP: 問題三\n"
                    "問題要具體、跟這手牌相關、能用 GTO solver 回答。例如「BB 在 turn 的 check-raise 範圍是什麼？」"
                    "「如果 flop 用 33% pot 下注會怎樣？」「對手 3-bet 的話 KQo 應該怎麼打？」"
                )
                coaching_prompt = (
                    f"用戶描述：\n{user_text}\n\n"
                    f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
                    f"{coaching_instruction}{followup_instruction}"
                )
                # Verified generation (retries transient 503/500 internally;
                # routes the verdict through the coach_facts claim verifier)
                result = await self._verified_initial_coaching(
                    chat_id, coaching_prompt, context, user_text,
                    on_status=on_status, user_id=user_id,
                    refresh_token=refresh_token, usage_acc=usage_acc,
                )
                result, followups = self._extract_followups(result)
                if followups:
                    ctx = self.hand_contexts.get(chat_id)
                    if ctx is not None:
                        ctx["followup_questions"] = followups
                # Update snapshot with coaching text
                _coaching_only = result.removeprefix(f"📋 `{hand_id}`\n\n") if hand_id else result
                _aio.create_task(self._update_snapshot_coaching(
                    hand_id, chat_id, _coaching_only))
                if hand_id:
                    result = f"📋 `{hand_id}`\n\n{result}"
                _vwarn = (context.get("validation") or {}).get("user_warning")
                if _vwarn:
                    result += f"\n\n{_vwarn}"
                t_total = time.time()
                self._logger.info(
                    f"[chat={chat_id}] Done: parse={t_parse - t0:.1f}s "
                    f"gto={t_analyze - t_parse:.1f}s "
                    f"coach={t_total - t_analyze:.1f}s "
                    f"total={t_total - t0:.1f}s"
                )
                await self._save_usage(chat_id, "hand_analysis", self.model,
                                       usage_acc, int((t_total - t0) * 1000))
                return result
            else:
                # Not a hand — chat (with tools if hand context exists)
                await _status("查詢中...")
                result = await self._run_followup_chat(
                    chat_id, user_text, on_status=on_status,
                    user_id=user_id, refresh_token=refresh_token,
                    usage_acc=usage_acc)
                elapsed = time.time() - t0
                self._logger.info(f"[chat={chat_id}] Chat response in {elapsed:.1f}s")
                await self._save_usage(chat_id, "follow_up", self.model,
                                       usage_acc, int(elapsed * 1000))
                return result

        except asyncio.TimeoutError:
            self._logger.error(f"[chat={chat_id}] Gemini API timeout")
            await self._save_usage(chat_id, "error", self.model, usage_acc,
                                   int((time.time() - t0) * 1000))
            raise RuntimeError("Gemini API 回應超時，請稍後再試。")
        except Exception as e:
            self._logger.error(f"[chat={chat_id}] Error: {e}", exc_info=True)
            await self._save_usage(chat_id, "error", self.model, usage_acc,
                                   int((time.time() - t0) * 1000))
            raise

    async def send_image_message(self, chat_id: int, image_bytes: bytes,
                                    mime_type: str = "image/jpeg",
                                    user_text: str = "",
                                    status_callback=None,
                                    send_gto_callback=None,
                                    user_id: int | None = None,
                                    refresh_token: str | None = None) -> str:
        """Main entry for image-based hand analysis: parse screenshot → GTO → coaching.

        status_callback: optional async callable(str) to update user-facing status.
        send_gto_callback: optional async callable(str) to send GTO summary immediately.
        user_id: Telegram user ID for per-user token lookup.
        refresh_token: user's GTO Wizard refresh token (if any).
        """
        request_id_var.set(new_request_id())
        t0 = time.time()
        self._logger.info(
            f"[chat={chat_id}] Image message ({len(image_bytes)} bytes), "
            f"caption: {user_text[:200]}"
        )
        usage_acc = {}

        async def _update_status(text: str):
            if status_callback:
                try:
                    await status_callback(text)
                except Exception:
                    pass

        try:
            # Step 1: Parse hand from screenshot
            await _update_status("🔍 正在辨識截圖中的手牌...")
            hand_json = await self._parse_hand_from_image(chat_id, image_bytes, mime_type,
                                                          user_text=user_text,
                                                          usage_acc=usage_acc)
            t_parse = time.time()

            if not hand_json:
                self._logger.info(f"[chat={chat_id}] No hand found in image")
                if user_text.strip():
                    result = await self._chat(chat_id, user_text,
                                              user_id=user_id, refresh_token=refresh_token,
                                              usage_acc=usage_acc)
                    await self._save_usage(chat_id, "image_analysis", self.image_parse_model,
                                           usage_acc, int((time.time() - t0) * 1000))
                    return result
                await self._save_usage(chat_id, "image_analysis", self.image_parse_model,
                                       usage_acc, int((time.time() - t0) * 1000))
                return "無法從截圖中辨識出撲克手牌。請確認截圖是手牌回放畫面（包含底部動作面板）。"

            self._logger.info(
                f"[chat={chat_id}] Parsed image hand in {t_parse - t0:.1f}s: "
                f"{json.dumps(hand_json, ensure_ascii=False)[:300]}"
            )

            # Handle possible_ft flag — extract before saving/analysis
            possible_ft = hand_json.pop("possible_ft", False)
            # Strip OCR conf off the hand dict so it doesn't leak into
            # downstream analysis or DB columns other than classifier_conf.
            ocr_conf_for_hand: float | None = hand_json.pop("__ocr_conf__", None)

            # Save parsed hand to DB and get hand_id
            hand_id = None
            if self.db:
                try:
                    hand_id = await self.db.save_hand_returning_id(
                        chat_id, hand_json, source_type="image",
                        user_input=(user_text[:2000] if user_text else "[screenshot]"))
                except Exception as e:
                    self._logger.warning(f"[chat={chat_id}] Failed to save image hand: {e}")

            # Step 2: Require user token
            eff_bb = hand_json.get('effective_bb')
            eff_str = f"({eff_bb:.0f}bb)" if eff_bb else ""
            await _update_status(
                f"📊 辨識完成：{hand_json['hero_position']} {cards_to_emoji(hand_json['hero_hand'])} "
                f"{eff_str}，正在查詢 GTO 策略..."
            )
            if not refresh_token:
                return "請先使用 /settoken 綁定你的 GTO Wizard 帳號。"

            # Step 3: GTO analysis
            self._setup_user_token(user_id, refresh_token)
            try:
                from analyze_hand import analyze_hand_full
                context = analyze_hand_full(hand_json)
            finally:
                self._clear_user_token()
            gto_data = context["text"]
            self.hand_contexts[chat_id] = context
            if hand_id:
                self.last_hand_ids[chat_id] = hand_id
            # Save snapshot with image bytes (fire-and-forget)
            import asyncio as _aio
            _aio.create_task(self._save_snapshot(
                hand_id, chat_id, "image", user_text or "[screenshot]",
                image_bytes, hand_json, context,
                classifier_conf=ocr_conf_for_hand))
            # Extract deviations for leak detection (fire-and-forget)
            _aio.create_task(self._extract_deviations(
                chat_id, hand_id, hand_json, context))

            t_analyze = time.time()
            self._logger.info(
                f"[chat={chat_id}] Image GTO analysis in {t_analyze - t_parse:.1f}s"
            )

            # Send GTO summary immediately (split response)
            if send_gto_callback:
                gto_summary = context.get("text_compact", gto_data)
                if hand_id:
                    gto_summary = f"📋 `{hand_id}`\n\n{gto_summary}"
                try:
                    r = send_gto_callback(gto_summary)
                    if asyncio.iscoroutine(r):
                        await r
                    self._logger.info(
                        f"[chat={chat_id}] GTO summary sent at {t_analyze - t0:.1f}s"
                    )
                except Exception:
                    self._logger.warning(
                        f"[chat={chat_id}] Failed to send GTO summary (non-fatal)"
                    )

            # Step 4: Coaching with user's caption/question
            eff_bb2 = hand_json.get('effective_bb')
            eff_str2 = f"({eff_bb2:.0f}bb)" if eff_bb2 else ""
            hand_desc = (
                f"Hero {hand_json['hero_position']}"
                f"{'' if hand_json.get('no_hero_hand') else ' ' + cards_to_emoji(hand_json['hero_hand'])} "
                f"{eff_str2}\n"
                f"Preflop: {hand_json['preflop_actions']}"
            )
            if hand_json.get("streets"):
                for s in hand_json["streets"]:
                    board = s.get("board", s.get("card", ""))
                    acts = " ".join(
                        f"{a['position']}:{a['action']}" for a in s["actions"]
                    )
                    hand_desc += f"\n{cards_to_emoji(board)} → {acts}"

            user_q = user_text.strip() if user_text.strip() else "請分析這手牌"
            if context.get("no_hero_hand"):
                img_coaching_instruction = (
                    "用戶沒有指定具體手牌，請根據 GTO 數據分析該位置的整體範圍策略。"
                    "不要提及或分析任何特定手牌（如 AA）的策略。"
                )
            else:
                img_coaching_instruction = "請先根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
            coaching_prompt = (
                f"用戶上傳了撲克截圖，已從截圖中解析出手牌：\n{hand_desc}\n\n"
                f"用戶留言：{user_q}\n\n"
                f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
                f"{img_coaching_instruction}"
                "\n\n在回覆的最後，用以下格式輸出 3 個值得深入的 follow-up 問題（用戶可以點擊按鈕直接發送）：\n"
                "FOLLOWUP: 問題一\n"
                "FOLLOWUP: 問題二\n"
                "FOLLOWUP: 問題三\n"
                "問題要具體、跟這手牌相關、能用 GTO solver 回答。例如「BB 在 turn 的 check-raise 範圍是什麼？」"
                "「如果 flop 用 33% pot 下注會怎樣？」「對手 3-bet 的話 KQo 應該怎麼打？」"
            )
            # Verified generation (retries transient 503/500 internally;
            # routes the verdict through the coach_facts claim verifier)
            result = await self._verified_initial_coaching(
                chat_id, coaching_prompt, context, user_q,
                user_id=user_id, refresh_token=refresh_token,
                usage_acc=usage_acc, disable_tools=True,
            )
            result, followups = self._extract_followups(result)
            if followups:
                ctx = self.hand_contexts.get(chat_id)
                if ctx is not None:
                    ctx["followup_questions"] = followups
            # Update snapshot with coaching text
            _coaching_only = result.removeprefix(f"📋 `{hand_id}`\n\n") if hand_id else result
            _aio.create_task(self._update_snapshot_coaching(
                hand_id, chat_id, _coaching_only))
            if hand_id:
                result = f"📋 `{hand_id}`\n\n{result}"

            if possible_ft and hand_json.get("tournament_type") != "icm":
                result += (
                    "\n\n💡 這看起來可能是決賽桌場景。"
                    "如果是的話，回覆「決賽桌分析」即可切換到 ICM 模式重新分析。"
                )

            # Rules-validator note (§5): the parse contained a poker-rules
            # contradiction (e.g. an orphan call) or a low-confidence signal.
            _vwarn = (context.get("validation") or {}).get("user_warning")
            if _vwarn:
                result += f"\n\n{_vwarn}"

            t_total = time.time()
            self._logger.info(
                f"[chat={chat_id}] Image done: parse={t_parse - t0:.1f}s "
                f"gto={t_analyze - t_parse:.1f}s total={t_total - t0:.1f}s"
            )
            await self._save_usage(chat_id, "image_analysis", self.model,
                                   usage_acc, int((t_total - t0) * 1000))
            return result

        except asyncio.TimeoutError:
            self._logger.error(f"[chat={chat_id}] Image Gemini API timeout")
            await self._save_usage(chat_id, "image_analysis", self.model, usage_acc,
                                   int((time.time() - t0) * 1000))
            raise RuntimeError("Gemini API 回應超時，請稍後再試。")
        except Exception as e:
            self._logger.error(f"[chat={chat_id}] Image error: {e}", exc_info=True)
            await self._save_usage(chat_id, "image_analysis", self.model, usage_acc,
                                   int((time.time() - t0) * 1000))
            raise

    async def _parse_hand_from_image(self, chat_id: int, image_bytes: bytes,
                                       mime_type: str = "image/jpeg",
                                       user_text: str = "",
                                       usage_acc: dict | None = None) -> dict | None:
        """Parse hand from a screenshot image with a tiered confidence gate.

        Three tiers (configurable via OCR_FAST_TIER_MIN / OCR_MEDIUM_TIER_MIN):
        - `>= OCR_FAST_TIER_MIN`  (default 0.95): trust OCR, skip Gemini.
        - `>= OCR_MEDIUM_TIER_MIN` (default 0.80): use OCR synchronously, AND
          fire a Gemini cross-check asynchronously to log disagreements to
          `classifier_disagreement_log`. User waits only for OCR; future
          retrain uses the disagreements as labeled examples.
        - below medium: fall back to Gemini synchronously, exactly as before.

        Confidence 0.1..medium: OCR's partial hand/hints are still passed
        into the Gemini prompt to anchor the parse.
        """
        self._logger.debug(f"[chat={chat_id}] Parsing hand from image ({len(image_bytes)} bytes)")

        FAST_TIER_MIN = float(os.getenv("OCR_FAST_TIER_MIN", "0.95"))
        MEDIUM_TIER_MIN = float(os.getenv("OCR_MEDIUM_TIER_MIN", "0.80"))
        # Hard floor on hero card-classifier confidence: even when the
        # blended overall score is high (good action tracking can mask a
        # bad rank prediction), force Gemini fallback when CardCNN itself
        # is shaky on hero. Regression: H2772 (overall=0.86, K rank
        # classified as 8 at 0.56 — overall passed MEDIUM, hand was wrong).
        MIN_CARD_CONF = float(os.getenv("OCR_MIN_CARD_CONF", "0.70"))

        # Step 1: Try OCR-based parsing (feature switch: OCR_ENABLED env var)
        ocr_result = None
        ocr_hints = None
        ocr_enabled = os.getenv("OCR_ENABLED", "false").lower() in ("true", "1", "yes")
        if ocr_enabled:
            try:
                from ocr.n8_parser import parse_n8_screenshot
                ocr_result = parse_n8_screenshot(image_bytes)
                ocr_conf = ocr_result.get("confidence", 0.0)
                card_conf = ocr_result.get("card_confidence", 0.0)
                self._logger.info(
                    f"[chat={chat_id}] OCR result (conf={ocr_conf:.2f}, "
                    f"card_conf={card_conf:.2f}): "
                    f"{json.dumps(ocr_result.get('hand'), ensure_ascii=False, default=str)[:500] if ocr_result.get('hand') else 'no hand'}"
                )

                # Structural fields don't depend on hero_hand — track
                # separately so we can still cards-only-fallback when OCR
                # cleared a duplicate-CNN hero (e.g. Ts9s misread as TcTc
                # → _resolve_hero_board_conflict drops both → hero_hand="").
                struct_ok = (
                    ocr_result.get("hand")
                    and ocr_result["hand"].get("hero_position")
                    and ocr_result["hand"].get("preflop_actions")
                )
                hero_hand_present = bool(
                    ocr_result.get("hand")
                    and ocr_result["hand"].get("hero_hand")
                )
                hand_ok = struct_ok and hero_hand_present

                # Hard card-confidence gate — overrides FAST and MEDIUM tiers.
                # Before fully demoting, try a cards-only Gemini call when
                # OCR's structural fields look reliable: keep OCR's
                # hero_position/stacks/actions, only ask Gemini to re-read
                # hero_hand. Regression: H2790 — full Gemini fallback was
                # flipping the correct OCR-detected SB to BB because
                # IMAGE_PARSE_PROMPT lets Gemini re-decide every field.
                # Also fires when hero_hand is empty: production has hit
                # cases where the CNN gave both hero crops the same label
                # (e.g. spades→clubs misclassification on Ts9s) so the
                # duplicate guard cleared hero_cards. Without this branch
                # we'd skip straight to full Gemini parse, which itself
                # has failed on those screenshots — the cards-only prompt
                # is more focused and reliable.
                cards_need_fallback = (
                    not hero_hand_present or card_conf < MIN_CARD_CONF
                )
                confidence_abstain_with_ocr = (
                    hand_ok and ocr_conf < MEDIUM_TIER_MIN
                )
                if struct_ok and (
                    cards_need_fallback or confidence_abstain_with_ocr
                ):
                    parts = ocr_result.get("confidence_parts") or {}
                    structural_conf = (
                        parts.get("pot_consistency", 0.0)
                        + parts.get("player_tracking", 0.0)
                        + parts.get("ocr_confidence", 0.0)
                    ) / 3.0
                    STRUCTURAL_MIN = float(
                        os.getenv("OCR_STRUCTURAL_MIN", "0.80")
                    )
                    ABSTAIN_STRUCTURAL_MIN = float(
                        os.getenv("OCR_ABSTAIN_STRUCTURAL_MIN", "0.50")
                    )
                    required_structural_min = (
                        STRUCTURAL_MIN
                        if cards_need_fallback
                        else ABSTAIN_STRUCTURAL_MIN
                    )
                    # Postflop entry-collapse loss is a hidden structural risk:
                    # pot/player/ocr consistency can all read 1.0 even when the
                    # collapse step quietly ate a re-action box (e.g. H3433 —
                    # river had 10 raw fragments collapsed to 3, eating BB's
                    # "Raise 13.6 BB All-In" sticker and turning a win into a
                    # parsed fold). When that happens, the cards-only fallback
                    # patches the hero hand but keeps the broken action chain
                    # — analysis then shows the user a wrong storyline. Treat
                    # large per-street collapse losses as structural-failure
                    # and demote to full Gemini reparse instead.
                    POSTFLOP_LOSS_MAX = int(
                        os.getenv("OCR_POSTFLOP_COLLAPSE_LOSS_MAX", "4")
                    )
                    diag = ocr_result.get("diagnostics") or {}
                    pre_map = diag.get("street_entries_pre_collapse_count") or {}
                    final_map = diag.get("street_entries_count") or {}
                    max_postflop_loss = 0
                    worst_street = None
                    for s, pre in pre_map.items():
                        if pre is None:
                            continue
                        loss = int(pre) - int(final_map.get(s) or 0)
                        if loss > max_postflop_loss:
                            max_postflop_loss = loss
                            worst_street = s
                    postflop_collapse_ok = max_postflop_loss <= POSTFLOP_LOSS_MAX
                    cards_only_attempt_ok = structural_conf >= required_structural_min
                    if (structural_conf >= required_structural_min
                            and (postflop_collapse_ok or confidence_abstain_with_ocr)):
                        if not hero_hand_present:
                            reason = "hero_hand missing"
                        elif cards_need_fallback:
                            reason = (
                                f"card_conf={card_conf:.2f} < "
                                f"{MIN_CARD_CONF:.2f}"
                            )
                        else:
                            reason = (
                                f"confidence_abstain conf={ocr_conf:.2f} < "
                                f"{MEDIUM_TIER_MIN:.2f}"
                            )
                        self._logger.info(
                            f"[chat={chat_id}] Cards-only Gemini fallback "
                            f"({reason}, structural_conf="
                            f"{structural_conf:.2f} >= "
                            f"{required_structural_min:.2f}) — keeping OCR's structural "
                            f"parse, asking Gemini for hero_hand only"
                        )
                        gemini_hero_hand = None
                        try:
                            gemini_hero_hand = await self._gemini_hero_hand_only(
                                chat_id, image_bytes, mime_type,
                                ocr_hand=ocr_result["hand"],
                                hints=ocr_result.get("hints"),
                                usage_acc=usage_acc,
                            )
                        except Exception as e:
                            self._logger.warning(
                                f"[chat={chat_id}] Cards-only Gemini failed: {e}; "
                                f"falling through to full Gemini parse"
                            )
                        if gemini_hero_hand:
                            hand = self._merge_ocr_with_gemini_hero_hand(
                                ocr_result["hand"], gemini_hero_hand
                            )
                            self._normalize_cards(hand)
                            self._fix_folded_players_guarded(hand, chat_id)
                            if self._cards_only_merge_safe(
                                ocr_result, gemini_hero_hand
                            ):
                                hand["__ocr_conf__"] = float(ocr_conf)
                                return hand
                            self._logger.info(
                                f"[chat={chat_id}] Cards-only Gemini hero_hand "
                                f"{gemini_hero_hand} did not satisfy OCR "
                                f"structure safety gate; falling through to "
                                f"full Gemini parse"
                            )
                        if self._can_keep_ocr_abstain_after_cards_only(
                            confidence_abstain_with_ocr=confidence_abstain_with_ocr,
                            hero_hand_present=hero_hand_present,
                            cards_need_fallback=cards_need_fallback,
                            original_hero_hand=ocr_result["hand"].get("hero_hand"),
                            gemini_hero_hand=gemini_hero_hand,
                        ):
                            hand = ocr_result["hand"]
                            if self._cards_only_merge_safe(ocr_result, None):
                                self._logger.info(
                                    f"[chat={chat_id}] Cards-only Gemini returned "
                                    f"no usable hero_hand; keeping OCR abstain "
                                    f"structure instead of destructive full parse"
                                )
                                self._normalize_cards(hand)
                                self._fix_folded_players_guarded(hand, chat_id)
                                hand["__ocr_conf__"] = float(ocr_conf)
                                return hand
                        # else fall through to full Gemini parse below
                    collapse_note = (
                        f", postflop_collapse_loss={max_postflop_loss}"
                        f"@{worst_street}>{POSTFLOP_LOSS_MAX}"
                        if not postflop_collapse_ok
                        else ""
                    )
                    self._logger.info(
                        f"[chat={chat_id}] Demoting to full Gemini fallback "
                        f"(card_conf={card_conf:.2f} < {MIN_CARD_CONF:.2f}, "
                        f"hero_hand_present={hero_hand_present}, "
                        f"overall={ocr_conf:.2f}, cards_only_attempt_ok="
                        f"{cards_only_attempt_ok}{collapse_note}) — OCR "
                        f"abstain is not safe for field-level merge"
                    )
                    hand_ok = False

                if hand_ok and ocr_conf >= FAST_TIER_MIN:
                    hand = ocr_result["hand"]
                    self._logger.info(
                        f"[chat={chat_id}] Using OCR result (FAST tier conf={ocr_conf:.2f})"
                    )
                    self._normalize_cards(hand)
                    self._fix_folded_players_guarded(hand, chat_id)
                    hand["__ocr_conf__"] = float(ocr_conf)
                    return hand

                if hand_ok and ocr_conf >= MEDIUM_TIER_MIN:
                    hand = ocr_result["hand"]
                    self._logger.info(
                        f"[chat={chat_id}] Using OCR result (MEDIUM tier conf={ocr_conf:.2f}, "
                        f"dispatching async Gemini cross-check)"
                    )
                    self._normalize_cards(hand)
                    self._fix_folded_players_guarded(hand, chat_id)
                    hand["__ocr_conf__"] = float(ocr_conf)
                    # Fire-and-forget cross-check — user sees OCR result
                    # immediately; disagreements become training data.
                    import asyncio as _aio
                    _aio.create_task(self._cross_check_ocr_vs_gemini(
                        chat_id=chat_id,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        user_text=user_text,
                        ocr_hand=hand,
                        ocr_conf=float(ocr_conf),
                    ))
                    return hand

                if 0.1 <= ocr_conf and ocr_result.get("hints"):
                    ocr_hints = ocr_result["hints"]
            except Exception as e:
                self._logger.warning(f"[chat={chat_id}] OCR failed: {e}")

        # Step 2: Fall back to Gemini vision
        prompt_text = IMAGE_PARSE_PROMPT
        if user_text.strip():
            prompt_text += f"\n\n用戶留言：{user_text.strip()}"

        safe_ocr_hints, safe_partial, low_card_conf = self._gemini_ocr_context(
            ocr_result, MIN_CARD_CONF
        )
        if ocr_hints and safe_ocr_hints is None:
            safe_ocr_hints = ocr_hints

        # Append OCR hints if available. Low-confidence hero cards are removed
        # by _gemini_ocr_context so Gemini gets structural anchors without
        # being anchored to the bad card-classifier guess.
        if safe_ocr_hints:
            hints_str = json.dumps(safe_ocr_hints, ensure_ascii=False, default=str)
            prompt_text += f"\n\nOCR 預處理提示（僅供參考，可能有誤）：{hints_str}"
            if low_card_conf:
                prompt_text += (
                    "\n\n重要：OCR hero card confidence 低於門檻；"
                    "上述 OCR 提示已刻意移除 hero_cards/hero_hand。"
                    "請直接從截圖底部中央兩張明牌重新讀取 hero_hand，"
                    "不要沿用 OCR 的低信心猜測。"
                )

        # Include partial hand from OCR if available
        if safe_partial:
            partial_str = json.dumps(safe_partial, ensure_ascii=False, default=str)
            prompt_text += f"\n\nOCR 解析結果（需要你驗證和補充，特別是 hero_hand）：{partial_str}"

        # Retry on transient Gemini server errors (503 UNAVAILABLE, 500 INTERNAL, 429)
        response = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.image_parse_model,
                        contents=[
                            types.Content(role="user", parts=[
                                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                types.Part(text=prompt_text),
                            ]),
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            thinking_config=types.ThinkingConfig(thinking_budget=4096),
                        ),
                    ),
                    timeout=300,
                )
                break
            except genai_errors.ServerError as e:
                last_err = e
                if attempt == 2:
                    raise
                backoff = 2 ** attempt
                self._logger.warning(
                    f"[chat={chat_id}] Gemini image parse transient error "
                    f"(attempt {attempt + 1}/3): {e}. Retrying in {backoff}s"
                )
                await asyncio.sleep(backoff)
        assert response is not None
        if usage_acc is not None:
            self._accumulate_usage(usage_acc, self._extract_usage(response))

        text = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Image parse response:\n{text}")

        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text.strip()

        try:
            result = json.loads(json_str)
            hand = result.get("hand")
            if hand and hand.get("hero_position") and hand.get("preflop_actions") and hand.get("hero_hand"):
                self._normalize_cards(hand)
                self._fix_folded_players_guarded(hand, chat_id)
                # Remove extra keys the vision model sometimes adds
                for street in hand.get("streets", []):
                    street.pop("street", None)

                # Log Gemini result + diff with OCR for debugging
                self._logger.info(
                    f"[chat={chat_id}] Gemini result: "
                    f"{json.dumps(hand, ensure_ascii=False, default=str)[:500]}"
                )
                if ocr_result and ocr_result.get("hand"):
                    ocr_hand = ocr_result["hand"]
                    diffs = []
                    for key in ["hero_hand", "hero_position", "players_at_table",
                                "preflop_actions", "effective_bb"]:
                        ov = ocr_hand.get(key)
                        gv = hand.get(key)
                        if ov and gv and str(ov) != str(gv):
                            diffs.append(f"{key}: OCR={ov} → Gemini={gv}")
                    if diffs:
                        self._logger.info(
                            f"[chat={chat_id}] OCR vs Gemini diffs: {'; '.join(diffs)}"
                        )

                # Record the OCR confidence even when Gemini produced the
                # final hand — the number tracks the classifier's population
                # conf distribution over time.
                if ocr_result is not None:
                    hand["__ocr_conf__"] = float(ocr_result.get("confidence", 0.0))
                return hand
        except (json.JSONDecodeError, AttributeError) as e:
            self._logger.warning(
                f"[chat={chat_id}] Image JSON parse failed: {e}\nRaw: {json_str[:500]}"
            )

        return None

    @staticmethod
    def _parse_structured_icm_range_query(user_text: str) -> dict | None:
        """Parse explicit text-only ICM range requests without an LLM round.

        This catches messages like:

            icm final table 剩餘 7 人, stack size 15/68/35/50/18/10/26
            這時 hero hj open range 如何

        The normal "existing context + not a hand" guard intentionally skips
        many follow-up questions so stack lists such as ``37/42/76`` are not
        misread as hole cards.  But explicit ICM + FT + stack-distribution
        range requests are complete solver scenarios, and sending them to the
        free-form chat path lets the LLM reorder stacks or answer from theory.
        Deterministically building the no-hero hand JSON preserves the user's
        exact stack order and lets analyze_hand.py choose the approximate FT
        solver config.
        """
        text = user_text.strip()
        low = text.lower()

        has_icm = "icm" in low or "決賽桌" in text or "final table" in low or re.search(r"\bft\b", low)
        has_stack = "stack" in low or "籌碼" in text
        asks_range = (
            "range" in low or "範圍" in text or "open" in low or "raise" in low
            or "all in" in low or "all-in" in low or "全下" in text
        )
        if not (has_icm and has_stack and asks_range):
            return None

        # Prefer the stack list immediately after "stack size"/"籌碼".
        stack_match = re.search(
            r"(?:stack(?:\s*size)?|籌碼(?:量|分佈|分布)?)[^\d]{0,20}"
            r"(\d+(?:\.\d+)?(?:\s*[/,，、]\s*\d+(?:\.\d+)?){1,8})",
            text,
            re.I,
        )
        if not stack_match:
            return None

        stacks = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", stack_match.group(1))]
        if not (2 <= len(stacks) <= 9):
            return None

        # "剩餘 7 人" on an FT means 7 players are seated.  If omitted, the
        # stack count is the best table-size signal.
        n_match = re.search(r"(?:剩(?:餘|下)?|剩餘|剩下)?\s*(\d)\s*(?:人|players?)", text, re.I)
        players_at_table = int(n_match.group(1)) if n_match else len(stacks)
        if players_at_table != len(stacks):
            # Keep exact user stacks.  If the count and list disagree, the
            # list is safer because it maps one-to-one onto solver positions.
            players_at_table = len(stacks)

        position_order = {
            9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
            5: ["HJ", "CO", "BTN", "SB", "BB"],
            4: ["CO", "BTN", "SB", "BB"],
            3: ["BTN", "SB", "BB"],
            2: ["SB", "BB"],
        }.get(players_at_table)
        if not position_order:
            return None

        pos_pattern = r"\b(utg\+?1|utg\+?2|utg|lj|hj|co|btn|sb|bb)\b"
        hero_match = re.search(r"hero\s*" + pos_pattern, low, re.I)
        if not hero_match:
            # Fall back to "HJ open range" without the word hero.
            hero_match = re.search(pos_pattern + r"\s*(?:open|raise|range|範圍)", low, re.I)
        if not hero_match:
            return None

        raw_pos = (hero_match.group(1) or hero_match.group(0)).upper().replace("UTG1", "UTG+1").replace("UTG2", "UTG+2")
        raw_pos = raw_pos.split()[0]
        if raw_pos not in position_order:
            return None

        hero_idx = position_order.index(raw_pos)
        effective_bb = stacks[hero_idx]

        # Build the decision point.  For "HJ open range", everyone before HJ
        # folded and HJ raises in the complete hand line; analyze_hand.py will
        # query the node before HJ acts.
        actions = ["F"] * players_at_table
        action_after_pos = low[hero_match.end(): hero_match.end() + 40]
        is_open_query = "open" in action_after_pos or "open" in low
        if is_open_query:
            actions[hero_idx] = "R2"
        else:
            raiser_match = re.search(pos_pattern + r"\s*(?:open|raise|加注)", low, re.I)
            if raiser_match:
                raiser_pos = raiser_match.group(1).upper().replace("UTG1", "UTG+1").replace("UTG2", "UTG+2")
                if raiser_pos in position_order and position_order.index(raiser_pos) < hero_idx:
                    actions[position_order.index(raiser_pos)] = "R2"

        return {
            "gametype": "MTTGeneral",
            "tournament_type": "icm",
            "phase": "FT" if ("final table" in low or "決賽桌" in text or re.search(r"\bft\b", low)) else "BUBBLE",
            "players_at_table": players_at_table,
            "player_stacks": stacks,
            "effective_bb": effective_bb,
            "hero_position": raw_pos,
            "hero_hand": "AA",
            "no_hero_hand": True,
            "preflop_actions": "-".join(actions),
        }

    def _fix_folded_players_guarded(self, hand: dict, chat_id: int):
        """``_fix_folded_players`` with a rules before/after double-check (§4a).

        If our own post-processing turns a rules-valid parse into an invalid one
        — exactly the H3517 failure mode where a real villain action was stripped
        onto a folded seat — log loudly and keep the pre-processing version.
        """
        try:
            from hand_validator import validate_hand
            import copy
            before_ok = validate_hand(hand).ok
            snapshot = copy.deepcopy(hand) if before_ok else None
        except Exception:
            self._fix_folded_players(hand)
            return
        self._fix_folded_players(hand)
        try:
            if before_ok and snapshot is not None and not validate_hand(hand).ok:
                self._logger.error(
                    f"[chat={chat_id}] _fix_folded_players corrupted a valid parse "
                    f"(rules now broken) — reverting to the pre-processing version")
                hand.clear()
                hand.update(snapshot)
        except Exception:
            pass

    @staticmethod
    def _fix_folded_players(hand: dict):
        """Remove actions from players who folded in earlier streets."""
        POSITION_ORDERS = {
            9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
            5: ["HJ", "CO", "BTN", "SB", "BB"],
            4: ["CO", "BTN", "SB", "BB"],
            3: ["BTN", "SB", "BB"],
            2: ["SB", "BB"],
        }
        n = hand.get("players_at_table", 8)
        pos_order = POSITION_ORDERS.get(n, POSITION_ORDERS[8])

        # Track who folded preflop
        folded = set()
        preflop_parts = hand.get("preflop_actions", "").split("-")
        for i, act in enumerate(preflop_parts[:len(pos_order)]):
            if act.upper() == "F":
                folded.add(pos_order[i])

        # Walk through streets, removing folded players and tracking new folds
        for street in hand.get("streets", []):
            actions = street.get("actions", [])
            if not isinstance(actions, list):
                continue
            cleaned = []
            for idx, a in enumerate(actions):
                pos = a.get("position", "")
                if pos in folded:
                    # A "folded" player with an aggressive postflop action that a
                    # later same-street Call/Raise depends on is a position
                    # MISLABEL, not a ghost action — dropping it orphans the call
                    # (Call with nothing to call) and kills the solver node
                    # (H3517: a BB 3-bettor's bet tagged LJ, then stripped here).
                    # Keep it; the label is the bug, the action is real.
                    code = (a.get("action", "") or "").upper()
                    is_aggressive = (
                        a.get("allin")
                        or code.startswith("R")
                        or code in ("B", "AI", "ALLIN")
                    )
                    later_depends = any(
                        (b.get("action", "") or "").upper().startswith(("C", "R"))
                        or (b.get("action", "") or "").upper() in ("AI", "ALLIN")
                        for b in actions[idx + 1:]
                        if b.get("position", "") not in folded
                    )
                    if not (is_aggressive and later_depends):
                        continue  # genuine folded-player ghost action — skip
                cleaned.append(a)
                if a.get("action", "").upper() == "F":
                    folded.add(pos)
            street["actions"] = cleaned

    @staticmethod
    def _normalize_cards(hand: dict):
        """Fix common Gemini vision mistakes in card notation (e.g. '10' → 'T')
        and convert string actions to structured format."""
        hand["hero_hand"] = re.sub(r"10", "T", hand["hero_hand"])
        for street in hand.get("streets", []):
            if "board" in street:
                street["board"] = re.sub(r"10", "T", street["board"])
            if "card" in street:
                street["card"] = re.sub(r"10", "T", street["card"])
            # Fix: vision model sometimes returns actions as a flat string
            # e.g. "X-X-R1.52-C" instead of [{position, action}, ...]
            acts = street.get("actions")
            if isinstance(acts, str):
                street["actions"] = GeminiSessionManager._parse_street_actions_string(
                    acts, hand
                )
            # …or as a list of bare action strings, e.g.
            # ["X", "R1.4", "R5.2", "F"]. Reuse the same positional
            # assignment by joining on "-" so _fix_folded_players (which
            # calls a.get("position")) doesn't choke on the raw strings.
            # Regression: 8h5h BB screenshot — un-normalized list raised
            # 'str' object has no attribute 'get' → bogus "無法辨識" reply.
            elif (isinstance(acts, list) and acts
                    and all(isinstance(a, str) for a in acts)):
                street["actions"] = GeminiSessionManager._parse_street_actions_string(
                    "-".join(acts), hand
                )

    @staticmethod
    def _parse_street_actions_string(actions_str: str, hand: dict) -> list[dict]:
        """Convert flat action string (e.g. 'X-X-R1.52-C') to structured actions.

        Uses postflop position order: SB first, then BB, then positions in order, BTN last.
        Only assigns positions to players still in the hand (didn't fold preflop).
        """
        POSITION_ORDERS = {
            9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
            5: ["HJ", "CO", "BTN", "SB", "BB"],
            4: ["CO", "BTN", "SB", "BB"],
            3: ["BTN", "SB", "BB"],
            2: ["SB", "BB"],
        }
        n = hand.get("players_at_table", 8)
        pos_order = POSITION_ORDERS.get(n, POSITION_ORDERS[8])

        # Find who's still in the hand after preflop
        preflop = hand.get("preflop_actions", "")
        preflop_parts = preflop.split("-")
        active_positions = []
        for i, act in enumerate(preflop_parts[:len(pos_order)]):
            if act.upper() != "F":
                active_positions.append(pos_order[i])

        # Postflop order: SB first, BB next, then others in order, BTN last
        postflop_order = []
        for pos in ["SB", "BB"] + [p for p in pos_order if p not in ("SB", "BB")]:
            if pos in active_positions:
                postflop_order.append(pos)

        parts = actions_str.split("-")
        result = []
        pos_idx = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            pos = postflop_order[pos_idx % len(postflop_order)] if postflop_order else "?"
            action_entry = {"position": pos}

            if part.upper() == "X":
                action_entry["action"] = "X"
            elif part.upper() == "C":
                action_entry["action"] = "C"
            elif part.upper() == "F":
                action_entry["action"] = "F"
            elif part.upper().startswith("R"):
                try:
                    size = float(part[1:])
                    action_entry["action"] = part
                    action_entry["size"] = size
                except ValueError:
                    action_entry["action"] = part
            elif part.upper().startswith("AI"):
                action_entry["action"] = part
            else:
                action_entry["action"] = part

            result.append(action_entry)
            pos_idx += 1

        return result

    def _text_looks_like_hand(self, user_text: str) -> bool:
        """Heuristic check: does the text contain enough info to be a new hand?

        Returns True if the text has core hand-defining elements (effective bb,
        board cards, or a hand+action combo). Follow-up questions like
        "hero turn bet 83% 的範圍有哪些" should return False.
        """
        t = user_text.lower()
        # Effective BB mentioned (e.g., "30bb", "有效 50bb", "effective 40")
        # Require word boundary + 1-3 digits so "H2672 BB" (hand id ref) doesn't
        # match as "2672 bb".
        has_bb = bool(re.search(r'\b\d{1,3}\s*bb\b', t, re.I)) or '有效' in t or 'effective' in t
        if has_bb:
            return True
        # Board cards (3+ cards with suits, e.g., "Js6h5s", "J♠6♥5♠")
        has_board = bool(re.search(r'[akqjt2-9][cdhs♠♥♦♣][akqjt2-9][cdhs♠♥♦♣][akqjt2-9][cdhs♠♥♦♣]', t))
        if has_board:
            return True
        # Specific hand + action (e.g., "TT raise", "AKs open", "66 call").
        # Bare digit pairs without an s/o suffix are only valid as same-rank
        # pairs (22-99); otherwise tournament stack lists like "37,15,42"
        # get mistaken for hands.
        has_hand = bool(re.search(
            r'\b(?:'
            r'[akqjt][akqjt2-9][so]?'      # at least one face rank
            r'|22|33|44|55|66|77|88|99'    # numeric pair
            r'|[2-9][2-9][so]'             # numeric non-pair requires s/o
            r')\b',
            t,
        ))
        has_action = bool(re.search(
            r'\b(raise|call|fold|open|3bet|4bet|limp|all.?in|shove|jam)\b', t, re.I
        )) or bool(re.search(r'(加注|跟注|棄牌|全下)', t))
        if has_hand and has_action:
            return True
        return False

    async def _parse_hand(self, chat_id: int, user_text: str,
                           usage_acc: dict | None = None,
                           feedback_hint: str = "") -> dict | None:
        """Parse user's natural language into hand JSON. Uses Flash for speed.

        ``feedback_hint`` (rules-validator Channel B) appends a precise
        correction note when re-parsing a hand whose first parse broke the rules.
        """
        structured_icm = self._parse_structured_icm_range_query(user_text)
        if structured_icm and not feedback_hint:
            self._logger.info(
                f"[chat={chat_id}] Parsed structured ICM range query: "
                f"{json.dumps(structured_icm, ensure_ascii=False)}"
            )
            return structured_icm

        # If there's already a hand context and the text doesn't look like a new
        # hand description, skip parsing to prevent follow-up questions from being
        # hallucinated into fake hands by the LLM parser.
        if chat_id in self.hand_contexts and not self._text_looks_like_hand(user_text):
            self._logger.debug(
                f"[chat={chat_id}] Skipping parse: existing context + "
                f"text doesn't look like a new hand"
            )
            return None

        prompt = f"{PARSE_PROMPT}\n\n用戶訊息：\n{user_text}"
        if feedback_hint:
            prompt += f"\n\n{feedback_hint}"
        self._logger.debug(f"[chat={chat_id}] Parse request: {user_text}")

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.parse_model,
                contents=prompt,
                # Hand parsing is deterministic structured extraction — Gemini
                # 2.5's default "thinking" adds ~7s with zero accuracy gain
                # (benchmarked 8 hands: 32/32 with thinking on AND off, 9.8s
                # vs 2.4s).  Disable it so text parse stays fast.
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ),
            timeout=60,
        )
        if usage_acc is not None:
            self._accumulate_usage(usage_acc, self._extract_usage(response))

        text = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Parse response:\n{text}")

        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text.strip()

        try:
            result = json.loads(json_str)
            hand = result.get("hand")
            if hand and hand.get("hero_position") and hand.get("preflop_actions") and hand.get("hero_hand"):
                return hand
        except (json.JSONDecodeError, AttributeError) as e:
            self._logger.warning(f"[chat={chat_id}] JSON parse failed: {e}\nRaw: {json_str[:300]}")

        return None

    async def _reparse_if_rules_invalid(
        self, chat_id: int, user_text: str, hand_json: dict,
        usage_acc: dict | None) -> dict:
        """Rules-validator Channel B: one re-parse if the text parse breaks the rules.

        Feeds the precise violation (street + orphan-call/act-after-fold/...) back
        to Gemini and re-reads the action order.  Keeps whichever parse the rules
        accept; falls back to the original if the retry doesn't improve.
        """
        try:
            from hand_validator import validate_hand, to_parser_feedback
        except Exception:
            return hand_json
        report = validate_hand(hand_json)
        if report.ok:
            return hand_json
        feedback = to_parser_feedback(report)
        self._logger.warning(
            f"[chat={chat_id}] Text parse broke poker rules, re-parsing once: "
            f"{[i.code for i in report.hard]}")
        try:
            retry = await asyncio.wait_for(
                self._parse_hand(chat_id, user_text, usage_acc=usage_acc,
                                 feedback_hint=feedback), timeout=60)
        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Channel-B re-parse failed: {e}")
            return hand_json
        if retry and validate_hand(retry).ok:
            self._logger.info(f"[chat={chat_id}] Channel-B re-parse fixed the hand")
            return retry
        return hand_json

    async def _run_followup_chat(self, chat_id: int, user_text: str,
                                 **kwargs) -> str:
        """Bound a complete follow-up, including every tool-call round.

        Individual Gemini calls already time out, but a multi-round tool loop
        could otherwise hold a bot handler for many minutes.  Cancellation
        propagates into the active async request so the Telegram error path can
        remove the stale status message and accept the next update.
        """
        return await asyncio.wait_for(
            self._chat(chat_id, user_text, **kwargs),
            timeout=_FOLLOWUP_TIMEOUT_SECONDS,
        )

    async def _chat(self, chat_id: int, user_text: str,
                     on_status: Callable[[str], Any] | None = None,
                     user_id: int | None = None,
                     refresh_token: str | None = None,
                     usage_acc: dict | None = None) -> str:
        """Chat with GTO tool access — always provides tools so model can query solver."""
        self._logger.debug(f"[chat={chat_id}] Chat with tools (model={self.model}): {user_text[:300]}")
        for attempt in range(3):
            try:
                return await self._chat_with_tools(
                    chat_id, user_text, on_status=on_status,
                    user_id=user_id, refresh_token=refresh_token,
                    usage_acc=usage_acc)
            except genai_errors.ServerError as e:
                if attempt == 2:
                    raise
                self._logger.warning(
                    f"[chat={chat_id}] Follow-up retry {attempt+1}/3: {e}")
                await asyncio.sleep(2 * (attempt + 1))

    async def _verified_initial_coaching(self, chat_id: int, coaching_prompt: str,
                                          context: dict, user_text: str, *,
                                          on_status=None, user_id: int | None = None,
                                          refresh_token: str | None = None,
                                          usage_acc: dict | None = None,
                                          disable_tools: bool = False) -> str:
        """Generate the initial coaching verdict.

        The initial combo-whitelist verifier is intentionally disabled: it was
        too coarse to distinguish solver-frequency claims from ordinary poker
        heuristics (H3689: mentioning TT as a possible stronger hand), causing
        useful analysis to be replaced by a warning. Follow-up answers still use
        the grounded solver/coach_facts verification path when users ask for
        ranges, specific combos, or hypotheticals.
        """
        for attempt in range(3):
            try:
                return await self._chat_with_tools(
                    chat_id, coaching_prompt, on_status=on_status,
                    user_id=user_id, refresh_token=refresh_token,
                    usage_acc=usage_acc, force_tool_eligible=False,
                    disable_tools=disable_tools)
            except genai_errors.ServerError as e:
                if attempt == 2:
                    raise
                self._logger.warning(
                    f"[chat={chat_id}] Coaching retry {attempt+1}/3: {e}")
                await asyncio.sleep(2 * (attempt + 1))

    async def _chat_with_tools(self, chat_id: int, user_text: str,
                                on_status: Callable[[str], Any] | None = None,
                                user_id: int | None = None,
                                refresh_token: str | None = None,
                                usage_acc: dict | None = None,
                                force_tool_eligible: bool = True,
                                disable_tools: bool = False) -> str:
        """Chat with GTO tools for data-driven follow-up answers.

        force_tool_eligible: when True (the follow-up path), strategy/range
        questions are detected by _needs_solver_grounding and the first
        generation round is hard-forced to a solver tool call (Gemini
        tool_config mode=ANY). Coaching/FT-switch callers pass False so the
        initial analysis isn't disturbed (its data is already computed).
        """
        # Drop any per-answer GTO-Wizard node override from a previous reply so
        # it can't leak onto this one; _try_coach_facts re-sets it when this
        # answer is grounded on a specific node.
        _ctx = self.hand_contexts.get(chat_id)
        if _ctx is not None:
            _ctx.pop("_followup_node_street", None)

        declarations = [
            QUERY_NEXT_ACTIONS_DECLARATION,
            QUERY_GTO_DECLARATION,
            EVALUATE_HAND_DECLARATION,
        ]
        if self.db:
            declarations.append(LOOKUP_HAND_DECLARATION)
            # Training-loop tools (require DB) — all ledger-backed, EV-weighted
            declarations.extend([
                GET_TRAINING_PLAN_DECLARATION,
                GET_PROGRESS_DECLARATION,
                QUERY_LEDGER_SUMMARY_DECLARATION,
                QUERY_LEDGER_HANDS_DECLARATION,
            ])
        tool = types.Tool(function_declarations=declarations)

        # Build system prompt with hand context
        hand_summary = self._build_hand_summary(chat_id)
        system = COACH_SYSTEM + "\n\n" + hand_summary

        history = self.histories.get(chat_id, [])
        messages = list(history) + [
            types.Content(role="user", parts=[types.Part(text=user_text)]),
        ]

        result_text = ""
        max_rounds = 8
        tools_called = 0

        # Intent gate: hard-force a solver tool call on round 0 for
        # strategy/range/hypothetical follow-ups so the model can't answer
        # from poker theory. Played-line range queries resolve against the
        # cached solution (no extra API latency). After the first tool runs
        # we revert to AUTO so the model synthesizes the grounded answer.
        force_tools = (
            force_tool_eligible
            and not disable_tools
            and _needs_solver_grounding(user_text)
        )
        if force_tools:
            self._logger.info(
                f"[chat={chat_id}] Solver-grounding gate matched — "
                f"forcing tool call on round 0 (mode=ANY)"
            )

        async def _status(msg: str):
            if on_status:
                r = on_status(msg)
                if asyncio.iscoroutine(r):
                    await r

        # Deterministic grounded path for P0/P1 follow-up intents (coach_facts).
        # Only when we have a cached analyzed hand and the grounding gate matched.
        # 'other'/unknown intents return None -> keep the existing tool loop below.
        if force_tools:
            grounded = await asyncio.to_thread(
                self._try_coach_facts, chat_id, user_text, user_id, refresh_token)
            if grounded:
                grounded = _normalize_terms(grounded)
                history = self.histories.get(chat_id, [])
                history.append(types.Content(role="user",
                                             parts=[types.Part(text=user_text)]))
                history.append(types.Content(role="model",
                                             parts=[types.Part(text=grounded)]))
                self.histories[chat_id] = history[-20:]
                return grounded

        for round_num in range(max_rounds):
            gen_kwargs = dict(
                system_instruction=system,
                tools=[] if disable_tools else [tool],
            )
            # Hard-force only until the first tool has actually executed;
            # subsequent rounds use AUTO so the model can return prose.
            if force_tools and tools_called == 0 and not disable_tools:
                gen_kwargs["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=_SOLVER_GROUNDING_TOOL_NAMES,
                    )
                )
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=types.GenerateContentConfig(**gen_kwargs),
                ),
                timeout=120,
            )
            if usage_acc is not None:
                self._accumulate_usage(usage_acc, self._extract_usage(response))

            # Check for function calls in response
            candidate = response.candidates[0]
            parts = (candidate.content and candidate.content.parts) or []
            function_calls = [
                p for p in parts
                if p.function_call
            ]

            # Extract any text parts from this response (model may return text + tool calls together)
            text_parts = [p.text for p in parts if p.text]
            if text_parts:
                result_text = "\n".join(text_parts)

            if not function_calls:
                # Model returned no function calls
                if round_num == 0 and not result_text.strip():
                    # Empty on first round — retry with explicit tool hint
                    finish = getattr(candidate, "finish_reason", "unknown")
                    self._logger.warning(
                        f"[chat={chat_id}] Empty response on round 0 "
                        f"(finish_reason={finish}), retrying with tool hint"
                    )
                    messages.append(types.Content(role="user", parts=[types.Part(text=(
                        "請使用 query_gto 工具查詢用戶問題所需的 GTO 策略數據。"
                        "例如查詢某位置在某條街的範圍，用 street 和 position 參數。"
                    ))]))
                    continue
                break

            # Execute tool calls and build response
            messages.append(candidate.content)

            for fc in function_calls:
                fn_name = fc.function_call.name
                args = dict(fc.function_call.args) if fc.function_call.args else {}
                self._logger.info(
                    f"[chat={chat_id}] Tool call #{round_num+1}: "
                    f"{fn_name}({json.dumps(args, ensure_ascii=False)})"
                )

                t_tool = time.time()

                if fn_name == "lookup_hand":
                    await _status("查詢手牌歷史...")
                    tool_result = await self._execute_lookup_hand(chat_id, args)
                elif fn_name == "evaluate_hand":
                    # Local deterministic eval — no API call needed
                    await _status("判斷牌型...")
                    tool_result = self._execute_evaluate_hand(chat_id, args)
                elif fn_name in ("get_training_plan", "get_progress"):
                    await _status("查詢訓練數據...")
                    tool_result = await self._execute_leak_tool(chat_id, fn_name, args, user_id)
                elif fn_name in ("query_ledger_summary", "query_ledger_hands"):
                    await _status("查詢帳本...")
                    tool_result = await self._execute_ledger_tool(fn_name, args)
                else:
                    # GTO API tools — need status + token
                    pos = args.get("position", "")
                    street = args.get("street", "")
                    icm = args.get("icm_phase", "")
                    tool_desc = f"查詢 {pos} {street}" if pos else f"查詢 {street} 策略"
                    if icm:
                        tool_desc += f" (ICM {icm})"
                    await _status(tool_desc + "...")

                    self._setup_user_token(user_id, refresh_token)
                    try:
                        if fn_name == "query_next_actions":
                            tool_result = self._execute_query_next_actions(chat_id, args)
                        else:
                            tool_result = self._execute_query_gto(chat_id, args)
                    finally:
                        self._clear_user_token()
                elapsed = time.time() - t_tool
                self._logger.debug(
                    f"[chat={chat_id}] Tool result ({elapsed:.1f}s, {len(tool_result)} chars):\n"
                    f"{tool_result[:500]}"
                )
                tools_called += 1

                # Persist the tool call for debugging (fire-and-forget).
                if self.db:
                    try:
                        asyncio.create_task(self.db.save_tool_call(
                            chat_id=chat_id,
                            request_id=request_id_var.get(),
                            hand_id=self.last_hand_ids.get(chat_id),
                            tool_name=fn_name,
                            tool_args=args,
                            tool_result=tool_result,
                            latency_ms=int(elapsed * 1000),
                        ))
                    except Exception as e:
                        self._logger.debug(f"[chat={chat_id}] save_tool_call dispatch failed: {e}")

                messages.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=fn_name,
                        response={"data": tool_result},
                    )],
                ))

        if not result_text.strip():
            self._logger.warning(
                f"[chat={chat_id}] Empty response after {round_num + 1} rounds "
                f"({tools_called} tool calls), requesting final answer"
            )
            if tools_called > 0:
                # Tools were called — ask model to summarize the results
                messages.append(types.Content(role="user", parts=[types.Part(text=(
                    "請根據以上工具查詢結果，給出完整的分析回覆。"
                    "不要包含任何 JSON 或原始數據，只用自然語言回覆。"
                ))]))
            else:
                # No tools were called — ask model to try answering directly
                messages.append(types.Content(role="user", parts=[types.Part(text=(
                    "請直接回答用戶的問題。如果需要 GTO 數據支持，"
                    "根據系統提示中的手牌資訊描述你所知道的策略。\n"
                    "重要：不要模擬工具呼叫、不要輸出 JSON、不要包含原始數據。"
                    "只用自然語言簡潔回覆。"
                ))]))
            await _status("生成回覆中...")
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=types.GenerateContentConfig(system_instruction=system),
                ),
                timeout=120,
            )
            if usage_acc is not None:
                self._accumulate_usage(usage_acc, self._extract_usage(response))
            result_text = _normalize_terms(
                response.text or "抱歉，分析過程中出現問題，請重新傳送手牌。"
            )

        self._logger.debug(f"[chat={chat_id}] Chat+tools response ({len(result_text)} chars):\n{result_text}")

        # ── Phase-2 reserved interface: post-answer grounding re-check ──
        # If a later phase enables it: when the gate matched but the answer
        # still enumerates hand-class → action with tools_called == 0 (model
        # ignored a forced call / gate missed), reject and re-ask with a
        # forced tool call. Intentionally a no-op for now.
        #
        #   if force_tools and tools_called == 0 \
        #           and self._answer_enumerates_hand_actions(result_text):
        #       result_text = await self._regrounded_retry(
        #           chat_id, user_text, messages, system, usage_acc)

        # Update history (user text only, not tool calls)
        history = self.histories.get(chat_id, [])
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=result_text)]))
        self.histories[chat_id] = history[-20:]

        return result_text

    def _build_standalone_context(self, args: dict) -> dict | None:
        """Build a minimal hand context from tool args when no cached context exists.

        Requires effective_bb at minimum. preflop_actions_override defaults to ""
        (UTG first to act) if not provided.
        """
        from gto_api import nearest_depth as _nearest_depth

        effective_bb = args.get("effective_bb")
        if not effective_bb:
            return None

        preflop_override = args.get("preflop_actions_override")
        if preflop_override is None:
            preflop_override = ""

        # ICM support
        icm_phase = args.get("icm_phase")
        if icm_phase:
            from icm_modes import find_icm_params
            num_players = args.get("num_players", 8)
            stacks_str = args.get("player_stacks", "")
            if stacks_str:
                player_stacks = [float(s.strip()) for s in stacks_str.split(",")]
            else:
                player_stacks = [float(effective_bb)] * num_players
            icm = find_icm_params(
                player_stacks=player_stacks,
                phase=icm_phase,
            )
            return {
                "gametype": icm["gametype"],
                "depth": icm["depth"],
                "stacks": icm["stacks"],
                "preflop_actions": preflop_override,
                "hero_position": "",
                "hero_hand": "",
                "hero_spots": [],
                "solutions": [],
                "street_states": {},
                "final_actions": {},
            }

        return {
            "gametype": "MTTGeneral",
            "depth": _nearest_depth(effective_bb),
            "stacks": "",
            "preflop_actions": preflop_override,
            "hero_position": "",
            "hero_hand": "",
            "hero_spots": [],
            "solutions": [],
            "street_states": {},
            "final_actions": {},
        }

    def _execute_evaluate_hand(self, chat_id: int, args: dict) -> str:
        """Execute evaluate_hand tool call. Returns deterministic hand type."""
        from hand_eval import evaluate as eval_hand

        hand = args.get("hand", "")
        board = args.get("board", "")

        # Auto-fill board from cached context if not provided
        if not board:
            ctx = self.hand_contexts.get(chat_id)
            if ctx:
                for street in ("river", "turn", "flop"):
                    if street in ctx.get("street_states", {}):
                        board = ctx["street_states"][street].get("board", "")
                        if board:
                            break

        if not board:
            return f"無法判斷牌型：沒有指定牌面，且當前沒有手牌 context。請提供 board 參數。"

        try:
            result = eval_hand(hand, board)
        except (ValueError, KeyError) as e:
            return f"無法判斷牌型：{e}。請確認 hand 格式（如 AKo, Th8c）。"
        return f"{cards_to_emoji(hand)} 在 {cards_to_emoji(board)}: {result['full_label']}"

    async def _execute_lookup_hand(self, chat_id: int, args: dict) -> str:
        """Look up a hand by ID from the user's history."""
        hand_id = args.get("hand_id", "")
        if not hand_id:
            return "錯誤：請提供 hand_id。"
        if not self.db:
            return "錯誤：資料庫未連接。"
        hand = await self.db.find_hand(chat_id, hand_id)
        if not hand:
            return f"找不到 Hand ID '{hand_id}' 的手牌記錄。"
        return json.dumps(hand, ensure_ascii=False)

    async def _execute_ledger_tool(self, fn_name: str, args: dict) -> str:
        """Execute the action-line ledger follow-up tools. Grounded, always with n."""
        if not self.db or not self.db.pool:
            return "暫時無法查詢帳本，請稍後再試"
        from ledger_service import query_ledger_summary, query_ledger_hands
        if fn_name == "query_ledger_summary":
            days = int(args["days"]) if args.get("days") else None
            s = await query_ledger_summary(self.db.pool, category=args.get("category"),
                                           hero_cat=args.get("hero_cat"), days=days)
            if not s["n"]:
                return "帳本裡沒有符合條件的決策資料。"
            scope = f"（最近 {days} 天）" if days else "（全期）"
            lines = [f"📒 帳本 EV loss{scope}：{s['per100']:.2f} bb/100 決策 · "
                     f"總損失 {s['total_bb']:.1f}bb · n={s['n']} · excluded {s['excluded_n']}"]
            if s["top_spots"]:
                lines.append("\ntop 漏 EV 的 spot（EV 排序，帶 n）：")
                for i, t in enumerate(s["top_spots"][:5], 1):
                    lines.append(f"{i}. `{t['spot']}` -{t['total_bb']:.1f}bb "
                                 f"（{t['per100']:.2f}bb/100, n={t['n']}）")
            return "\n".join(lines)
        # query_ledger_hands
        hands = await query_ledger_hands(
            self.db.pool, category=args.get("category"),
            min_ev_loss=float(args.get("min_ev_loss", 0.5)),
            days=int(args["days"]) if args.get("days") else 90,
            limit=int(args.get("limit", 5)))
        if not hands:
            return "帳本裡沒有符合條件的手牌。"
        lines = ["📒 符合的手牌（EV loss 排序）："]
        for h in hands:
            lines.append(f"· {h['played_at']} {cards_to_emoji(h['hero_hand'])} {h['position'] or ''} "
                         f"{cards_to_emoji(h['boards'] or '')} — `{h['spot']}` -{h['ev_loss_bb']:.2f}bb "
                         f"（{h['correctness']}）· [Analyze]({h['review_url']})")
        return "\n".join(lines)

    @staticmethod
    def _render_training_plan(week: str, data: dict) -> str:
        """Pure renderer for the get_training_plan tool (scorecard data_json)."""
        lines = [f"🎯 本週訓練計畫（{week}）："]
        if data.get("headline"):
            lines.append(data["headline"])
        for i, f in enumerate(data.get("focus", []), 1):
            lines.append(f"\n重點 {i}: {f['desc']}")
            lines.append(f"  {f['per100']:.2f} bb/100 · n={f['n']} · `{f['spot_leaf']}`")
            if f.get("drill_url"):
                lines.append(f"  → [GTOW Trainer 練這個]({f['drill_url']})")
        for r in (data.get("readback") or []):
            if r.get("current_per100") is not None and r.get("prescribed_per100") is not None:
                lines.append(
                    f"\n上週焦點 `{r['spot_leaf']}`：{r['prescribed_per100']:.1f} → "
                    f"{r['current_per100']:.1f} bb/100（n={r['n']}，{r['note']}）")
        dq = data.get("drill_queue") or []
        if dq:
            lines.append("\n📥 現場手牌練習佇列：")
            for q in dq:
                lines.append(f"· {q.get('label') or q.get('spot_leaf')}"
                             f"（{q.get('n_sources', 1)} 手，漏 {(q.get('total_ev_loss_bb') or 0):.1f}bb）")
        return "\n".join(lines)

    @staticmethod
    def _render_progress(scope: str, series: list[dict]) -> str:
        """Pure renderer for the get_progress tool: weekly EV-loss series with n.

        EV-weighted only (§7.3) and no single-week verdicts — skill trends are
        month-scale (§14.4), so the note replaces any ✅/⚠️ judgement."""
        lines = [f"📈 {scope} 每週 EV loss（bb/100 決策）："]
        for p in series:
            lines.append(f"  {p['week']}: {p['per100']:.2f} bb/100 (n={p['n']})")
        lines.append("")
        lines.append("技能趨勢是月尺度：單週樣本波動大，連續 4 週同向才算訊號；"
                     "不足 4 週先不下結論。")
        return "\n".join(lines)

    async def _execute_leak_tool(self, chat_id: int, fn_name: str,
                                  args: dict, user_id: int | None) -> str:
        """Training-loop tools over the ledger (EV-weighted, always with n).

        The frequency-era deviations tools (query_my_leaks / query_my_stats,
        deviation-rate trends) were retired per North Star §7.3 — weakness &
        stats questions route to query_ledger_summary instead."""
        if not self.db or not self.db.pool:
            return "暫時無法查詢你的資料，請稍後再試"

        try:
            if fn_name == "get_training_plan":
                from ledger_service import fetch_latest_scorecard
                row = await fetch_latest_scorecard(self.db.pool)
                if not row:
                    return ("本週訓練計畫還沒生成（每週日 21:00 自動產生）。"
                            "可以先用 query_ledger_summary 看目前最漏的 spot。")
                return self._render_training_plan(row["week"], row["data"])

            if fn_name == "get_progress":
                from ledger_service import query_progress_series
                weeks = int(args.get("weeks", 8))
                series = await query_progress_series(
                    self.db.pool, category=args.get("category"),
                    spot_leaf=args.get("spot_leaf"), weeks=weeks)
                if not series:
                    return "沒有符合條件的決策資料。"
                scope = args.get("spot_leaf") or args.get("category") or "整體"
                return self._render_progress(scope, series)

            return "未知的工具名稱"

        except Exception as e:
            self._logger.warning(f"[chat={chat_id}] Training tool error: {e}")
            return "暫時無法查詢你的資料，請稍後再試"

    def _execute_query_gto(self, chat_id: int, args: dict) -> str:
        """Execute a query_gto tool call. Returns formatted solver data."""
        from gto_api import get_spot_solution, get_next_actions, find_closest_action
        from gto_formatter import format_action_summary, format_hand_detail, format_range_overview

        from gto_api import nearest_depth as _nearest_depth
        from gto_api import nearest_cash_depth as _nearest_cash_depth

        # ICM args force standalone context (don't use cached chip EV context)
        if args.get("icm_phase"):
            ctx = self._build_standalone_context(args)
            if not ctx:
                return "錯誤：ICM 查詢需要提供 effective_bb。"
        else:
            ctx = self.hand_contexts.get(chat_id)
            if not ctx:
                ctx = self._build_standalone_context(args)
                if not ctx:
                    return "錯誤：沒有手牌 context 且未提供 effective_bb + preflop_actions_override。請先發送手牌描述，或同時指定 effective_bb 和 preflop_actions_override。"

        street = args.get("street", "flop")
        position = args.get("position")
        hand = args.get("hand")
        effective_bb = args.get("effective_bb")
        preflop_override = args.get("preflop_actions_override")
        board_override = args.get("board_override")
        flop_override = args.get("flop_actions_override")
        turn_override = args.get("turn_actions_override")
        river_override = args.get("river_actions_override")

        # Truncate preflop_override to target position's decision point
        # LLM often pads with trailing F's (e.g. F-R2-F-F-F-F-F-F for LJ's spot)
        # which means everyone folded = no solution. Strip to just before target position.
        if preflop_override and position and street == "preflop":
            POSITION_ORDER_8 = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
            try:
                target_idx = POSITION_ORDER_8.index(position)
                pf_parts = preflop_override.split("-")
                if len(pf_parts) > target_idx:
                    # Check if everything after target_idx is F (all folded past target)
                    tail = pf_parts[target_idx:]
                    if all(t == "F" for t in tail):
                        preflop_override = "-".join(pf_parts[:target_idx]) if target_idx > 0 else ""
                        self._logger.debug(
                            f"[chat={chat_id}] Truncated preflop to position {position}: "
                            f"{args.get('preflop_actions_override')} → {preflop_override or '(empty)'}"
                        )
            except ValueError:
                pass

        # Fix A: auto-pad preflop_override with leading F's to match the
        # context's internal preflop length. analyze_hand.py pads 7-max hands
        # to 8 positions for MTTGeneral (8-max solver); the LLM often echoes
        # back the unpadded 7-position string from the original parsed JSON,
        # which causes the solver to reject the spot. Pad leading F's to
        # recover — only when the override is shorter than the context.
        ctx_preflop = ctx.get("preflop_actions") or ""
        if (
            preflop_override
            and ctx_preflop
            and street != "preflop"  # preflop uses its own position-based handling below
        ):
            ctx_len = len([p for p in ctx_preflop.split("-") if p])
            override_parts = [p for p in preflop_override.split("-") if p]
            if 0 < len(override_parts) < ctx_len:
                padded = ["F"] * (ctx_len - len(override_parts)) + override_parts
                new_override = "-".join(padded)
                self._logger.debug(
                    f"[chat={chat_id}] Padded preflop_override leading F's: "
                    f"{preflop_override} → {new_override} (ctx_len={ctx_len})"
                )
                preflop_override = new_override

        # Override depth if effective_bb specified (only for non-ICM; ICM depth already set).
        # Cash games use integer depth (100.0), MTT uses .125 suffix (100.125).
        if effective_bb and not args.get("icm_phase"):
            is_cash = ctx.get("gametype", "").startswith("Cash")
            depth_override = _nearest_cash_depth(effective_bb) if is_cash else _nearest_depth(effective_bb)
        else:
            depth_override = None

        has_override = any([preflop_override, board_override, flop_override, turn_override, river_override, depth_override])

        # Fix B: cache hit when overrides match the played line.
        # The LLM often echoes back the full played line as "overrides" when
        # it just wants to query an existing spot. Detect this case and skip
        # the API call entirely. Compare against the cached hero_spot's
        # actual params (not street_states, which is a start-of-street
        # snapshot with incomplete action strings).
        if has_override and not args.get("icm_phase"):
            cached_spot = self._find_cached_spot(ctx, street)
            if cached_spot and self._overrides_match_played_line(
                cached_spot.get("params", {}),
                preflop_override, board_override,
                flop_override, turn_override, river_override,
                depth_override,
            ):
                solution = self._find_cached_solution(ctx, street)
                if solution:
                    self._logger.debug(
                        f"[chat={chat_id}] Overrides match played line; using cached {street} solution"
                    )
                    return self._format_solution(solution, position, hand)

        # Try cached solution first (no overrides)
        if not has_override:
            solution = self._find_cached_solution(ctx, street)
            if solution:
                return self._format_solution(solution, position, hand)

        # Build API params from context + overrides
        params = self._build_query_params(ctx, street, board_override,
                                          flop_override, turn_override, river_override,
                                          preflop_override=preflop_override)
        if not params:
            return f"無法建構 {street} 的查詢參數。"

        # Apply depth override
        if depth_override:
            params["depth"] = depth_override

        # Normalize any raise codes in override actions
        params = self._normalize_override_actions(params, street, flop_override, turn_override, river_override,
                                                  preflop_override=preflop_override)

        self._logger.debug(
            f"[chat={chat_id}] query_gto API params: {json.dumps(params, ensure_ascii=False)}"
        )

        try:
            solution = get_spot_solution(**params)
        except Exception as e:
            self._logger.warning(
                f"[chat={chat_id}] query_gto API exception: {e} "
                f"params={json.dumps(params, ensure_ascii=False)}"
            )
            return f"API 查詢失敗：{e}"

        if not solution:
            self._logger.warning(
                f"[chat={chat_id}] query_gto empty result. "
                f"params={json.dumps(params, ensure_ascii=False)}"
            )
            # Include the resolved params in the error so the LLM can
            # self-correct (e.g. realize preflop length was wrong).
            debug_params = {
                k: params.get(k)
                for k in ("preflop_actions", "board", "flop_actions",
                          "turn_actions", "river_actions", "depth")
                if params.get(k) not in (None, "")
            }
            return (
                f"{street} 沒有 solver 數據（可能是無效的 board 或 actions 組合）。"
                f"已發送的參數：{json.dumps(debug_params, ensure_ascii=False)}。"
                f"請檢查 preflop_actions 是否為 8 個位置（MTTGeneral 8-max），"
                f"board 花色是否合理，以及 action codes 是否符合 solver 格式。"
            )

        # Auto-pad preflop for position mismatch:
        # If position is specified but not found in the solution, try padding
        # the preflop to reach the correct decision point and retry.
        if position and street == "preflop":
            found = any(
                pi["player"]["position"] == position
                for pi in solution.get("players_info", [])
            )
            if not found:
                pf = params.get("preflop_actions", "")
                pf_parts = [p for p in pf.split("-") if p] if pf else []
                if len(pf_parts) < 8:
                    POSITION_ORDER = ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"]
                    try:
                        target_idx = POSITION_ORDER.index(position)
                    except ValueError:
                        target_idx = -1
                    if target_idx >= 0 and len(pf_parts) <= target_idx:
                        # Pad with F up to (but not including) the target position
                        while len(pf_parts) < target_idx:
                            pf_parts.append("F")
                        params["preflop_actions"] = "-".join(pf_parts)
                    elif len(pf_parts) < 8:
                        # Re-raise scenario: pad remaining positions with F
                        while len(pf_parts) < 8:
                            pf_parts.append("F")
                        params["preflop_actions"] = "-".join(pf_parts)
                    try:
                        solution = get_spot_solution(**params)
                    except Exception as e:
                        return f"API 查詢失敗：{e}"
                    if not solution:
                        return f"{street} 沒有 solver 數據（可能是無效的 board 或 actions 組合）。"

        result_text = self._format_solution(solution, position, hand)

        # Queue range grid image when querying a position's range (no specific hand)
        if position and not hand:
            try:
                from range_image import generate_range_grid
                game = solution.get("game", {})
                st = game.get("current_street", {}).get("type", "").capitalize()
                board = game.get("board", "")
                title = f"{position} {st}"
                if board:
                    title += f" | {cards_to_emoji(board)}"
                img = generate_range_grid(solution, position, title=title)
                if chat_id not in self.pending_images:
                    self.pending_images[chat_id] = []
                self.pending_images[chat_id].append((img, f"📊 {title}"))
            except Exception:
                pass  # non-critical

        return result_text

    def _find_cached_solution(self, ctx: dict, street: str) -> dict | None:
        """Find a cached spot-solution for the given street."""
        for spot, sol in zip(ctx["hero_spots"], ctx["solutions"]):
            if spot["street"] == street and sol is not None:
                return sol
        return None

    def _find_cached_spot(self, ctx: dict, street: str) -> dict | None:
        """Find the cached hero_spot for a given street (with solution)."""
        for spot, sol in zip(ctx.get("hero_spots", []), ctx.get("solutions", [])):
            if spot.get("street") == street and sol is not None:
                return spot
        return None

    @staticmethod
    def _norm_action_str(s: str | None) -> str:
        """Strip empty tokens from an action string for comparison."""
        if s is None:
            return ""
        return "-".join(p for p in s.split("-") if p)

    def _overrides_match_played_line(
        self,
        cached_params: dict,
        preflop_override: str | None,
        board_override: str | None,
        flop_override: str | None,
        turn_override: str | None,
        river_override: str | None,
        depth_override: float | None,
    ) -> bool:
        """Check whether all overrides redundantly describe the cached spot.

        The LLM often provides full override params when it just wants to
        query an already-analyzed spot. When the overrides exactly match
        the cached hero_spot's params, we can skip the API call entirely
        (faster + sidesteps raise-code normalization quirks).

        `cached_params` is the hero_spot["params"] dict that was used to
        fetch the cached solution.
        """
        if depth_override is not None:
            try:
                if float(depth_override) != float(cached_params.get("depth", 0)):
                    return False
            except (TypeError, ValueError):
                return False

        _n = self._norm_action_str

        if preflop_override is not None:
            if _n(preflop_override) != _n(cached_params.get("preflop_actions")):
                return False

        if board_override is not None:
            if (board_override or "").lower() != (cached_params.get("board", "") or "").lower():
                return False

        for key, ov in (
            ("flop_actions", flop_override),
            ("turn_actions", turn_override),
            ("river_actions", river_override),
        ):
            if ov is None:
                continue
            if _n(ov) != _n(cached_params.get(key)):
                return False

        return True

    def _build_query_params(self, ctx: dict, street: str,
                            board_override: str | None,
                            flop_override: str | None,
                            turn_override: str | None,
                            river_override: str | None,
                            preflop_override: str | None = None) -> dict | None:
        """Build API params for a query, using context + optional overrides."""
        states = ctx.get("street_states", {})
        base = states.get(street)
        preflop_actions = preflop_override or ctx["preflop_actions"]

        stacks = ctx.get("stacks", "")
        if street == "preflop":
            return dict(
                gametype=ctx["gametype"],
                depth=ctx["depth"],
                stacks=stacks,
                preflop_actions=preflop_actions,
            )

        if not base:
            # Street not in the analyzed hand — try to build from available data
            # For standalone queries (no street_states), build from overrides
            if board_override:
                return dict(
                    gametype=ctx["gametype"],
                    depth=ctx["depth"],
                    stacks=stacks,
                    preflop_actions=preflop_actions,
                    board=board_override,
                    flop_actions=flop_override or "",
                    turn_actions=turn_override or "",
                    river_actions=river_override or "",
                )
            # For hypotheticals on streets beyond what was played
            if street == "flop" and "flop" not in states:
                return None
            if street == "turn" and "flop" in states:
                flop_state = states["flop"]
                return dict(
                    gametype=ctx["gametype"],
                    depth=ctx["depth"],
                    stacks=stacks,
                    preflop_actions=preflop_actions,
                    board=board_override or flop_state["board"],
                    flop_actions=flop_override or flop_state["flop_actions"],
                    turn_actions=turn_override or "",
                    river_actions="",
                )
            return None

        return dict(
            gametype=ctx["gametype"],
            depth=ctx["depth"],
            stacks=stacks,
            preflop_actions=preflop_actions,
            board=board_override or base["board"],
            flop_actions=flop_override if flop_override is not None else base["flop_actions"],
            turn_actions=turn_override if turn_override is not None else base["turn_actions"],
            river_actions=river_override if river_override is not None else base["river_actions"],
        )

    def _normalize_override_actions(self, params: dict, street: str,
                                     flop_override: str | None,
                                     turn_override: str | None,
                                     river_override: str | None,
                                     preflop_override: str | None = None) -> dict:
        """Normalize raise codes in overridden action strings."""
        from gto_api import get_next_actions, find_closest_action, find_closest_action_by_pot_pct

        # Normalize preflop override (walk through each position's action)
        if preflop_override:
            parts = preflop_override.split("-")
            corrected = []
            for code in parts:
                if code in ("F", "C", ""):
                    corrected.append(code)
                elif code == "AI" or code.startswith("AI"):
                    # AI = all-in (no size), AI10 = all-in for 10bb (treat as raise to 10)
                    try:
                        check_params = dict(
                            gametype=params["gametype"],
                            depth=params["depth"],
                            stacks=params.get("stacks", ""),
                            preflop_actions="-".join(corrected) if corrected else "",
                        )
                        resp = get_next_actions(**check_params)
                        avail = resp["next_actions"]["available_actions"]
                        if code == "AI":
                            allin_code = next(
                                (a["action"]["code"] for a in avail if a["action"].get("allin")),
                                code,
                            )
                            corrected.append(allin_code)
                        else:
                            # AI{size} — find closest action by size
                            target = float(code[2:])
                            correct_code = find_closest_action(avail, target)
                            corrected.append(correct_code)
                    except Exception:
                        corrected.append(code)
                elif code.startswith("R"):
                    try:
                        check_params = dict(
                            gametype=params["gametype"],
                            depth=params["depth"],
                            stacks=params.get("stacks", ""),
                            preflop_actions="-".join(corrected) if corrected else "",
                        )
                        resp = get_next_actions(**check_params)
                        avail = resp["next_actions"]["available_actions"]
                        target = float(code[1:])
                        correct_code = find_closest_action(avail, target)
                        corrected.append(correct_code)
                    except Exception:
                        corrected.append(code)
                else:
                    corrected.append(code)
            params["preflop_actions"] = "-".join(corrected)

        # Normalize postflop overrides
        overrides = {
            "flop_actions": flop_override,
            "turn_actions": turn_override,
            "river_actions": river_override,
        }

        for key, override_val in overrides.items():
            if override_val is None:
                continue
            parts = override_val.split("-")
            corrected = []
            for code in parts:
                if code in ("X", "C", "F", ""):
                    corrected.append(code)
                elif code in ("AI", "RAI") or code.startswith("AI"):
                    # AI/RAI = all-in, AI{size} = all-in for specific size (treat as raise)
                    try:
                        check_params = dict(params)
                        check_params[key] = "-".join(corrected) if corrected else ""
                        resp = get_next_actions(**check_params)
                        avail = resp["next_actions"]["available_actions"]
                        if code in ("AI", "RAI"):
                            allin_code = next(
                                (a["action"]["code"] for a in avail if a["action"].get("allin")),
                                code,
                            )
                            corrected.append(allin_code)
                        else:
                            target = float(code[2:])
                            correct_code = find_closest_action(avail, target)
                            corrected.append(correct_code)
                    except Exception:
                        corrected.append(code)
                elif code.startswith("R"):
                    # Discover correct code from solver
                    try:
                        check_params = dict(params)
                        check_params[key] = "-".join(corrected) if corrected else ""
                        resp = get_next_actions(**check_params)
                        avail = resp["next_actions"]["available_actions"]
                        raw = code[1:]
                        if raw.endswith("%"):
                            # Percentage-based: R50% → convert to bb using solver pot
                            pct = float(raw[:-1]) / 100
                            solver_pot = float(resp["next_actions"]["game"]["pot"])
                            target = solver_pot * pct
                        else:
                            target = float(raw)
                        correct_code = find_closest_action_by_pot_pct(avail, target)
                        corrected.append(correct_code)
                    except Exception:
                        corrected.append(code)
                else:
                    corrected.append(code)
            params[key] = "-".join(corrected)

        return params

    def _format_solution(self, solution: dict, position: str | None, hand: str | None) -> str:
        """Format a spot-solution based on what was requested."""
        from gto_formatter import format_action_summary, format_hand_detail, format_range_by_action

        parts = [format_action_summary(solution)]

        if hand and position:
            parts.append("")
            parts.append(format_hand_detail(solution, hand, position))
        elif position:
            parts.append("")
            parts.append(format_range_by_action(solution, position))
        elif hand:
            # Hand specified but no position — use active position
            active_pos = solution["game"]["active_position"]
            parts.append("")
            parts.append(format_hand_detail(solution, hand, active_pos))

        return "\n".join(parts)

    def _execute_query_next_actions(self, chat_id: int, args: dict) -> str:
        """Execute a query_next_actions tool call. Returns available actions."""
        from gto_api import get_next_actions, nearest_depth as _nearest_depth
        from gto_api import nearest_cash_depth as _nearest_cash_depth

        # ICM args force standalone context
        if args.get("icm_phase"):
            ctx = self._build_standalone_context(args)
            if not ctx:
                return "錯誤：ICM 查詢需要提供 effective_bb。"
        else:
            ctx = self.hand_contexts.get(chat_id)
            if not ctx:
                ctx = self._build_standalone_context(args)
                if not ctx:
                    return "錯誤：沒有手牌 context 且未提供 effective_bb + preflop_actions_override。請先發送手牌描述，或同時指定 effective_bb 和 preflop_actions_override。"

        street = args.get("street", "flop")
        effective_bb = args.get("effective_bb")
        actions_so_far = args.get("actions_so_far", "")
        preflop_override = args.get("preflop_actions_override")
        board_override = args.get("board_override")
        flop_override = args.get("flop_actions_override")
        turn_override = args.get("turn_actions_override")

        # For ICM, depth is already set correctly in ctx.
        # Cash games use integer depth (100.0), MTT uses .125 suffix.
        if args.get("icm_phase"):
            depth = ctx["depth"]
        elif effective_bb:
            is_cash = ctx.get("gametype", "").startswith("Cash")
            depth = _nearest_cash_depth(effective_bb) if is_cash else _nearest_depth(effective_bb)
        else:
            depth = ctx["depth"]

        # Build params for the target street
        states = ctx.get("street_states", {})
        base = states.get(street, {})

        params = dict(
            gametype=ctx["gametype"],
            depth=depth,
            stacks=ctx.get("stacks", ""),
            preflop_actions=preflop_override or ctx["preflop_actions"],
        )

        if street != "preflop":
            params["board"] = board_override or base.get("board", "")
            params["flop_actions"] = (
                flop_override if flop_override is not None
                else base.get("flop_actions", "")
            )
            params["turn_actions"] = (
                turn_override if turn_override is not None
                else base.get("turn_actions", "")
            )
            params["river_actions"] = ""

        # Normalize raise codes (R2 → R2.1, AI → correct code)
        if preflop_override:
            params = self._normalize_override_actions(
                params, street, flop_override, turn_override, None,
                preflop_override=preflop_override,
            )

        # If actions_so_far provided, set it on the target street
        if actions_so_far:
            key = f"{street}_actions" if street != "preflop" else "preflop_actions"
            params[key] = actions_so_far

        try:
            resp = get_next_actions(**params)
        except Exception as e:
            return f"API 查詢失敗：{e}"

        avail = resp.get("next_actions", {}).get("available_actions", [])
        if not avail:
            return "此決策點沒有可用動作。"

        lines = [f"【{street} 可用動作】"]
        for entry in avail:
            action = entry["action"]
            code = action["code"]
            if code in ("X", "F", "C"):
                lines.append(f"  {code}")
            else:
                betsize = action.get("betsize", "?")
                pct = float(action.get("betsize_by_pot", 0)) * 100
                allin = " (all-in)" if action.get("allin") else ""
                lines.append(f"  {code} — betsize={betsize}bb（{pct:.0f}% pot）{allin}")

        return "\n".join(lines)

    @staticmethod
    def _extract_followups(text: str) -> tuple[str, list[str]]:
        """Strip FOLLOWUP: lines from response, return (clean_text, questions)."""
        followups: list[str] = []
        clean_lines: list[str] = []
        followup_re = re.compile(
            r"^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
            r"FOLLOW[\s_-]*UP(?:\*\*)?\s*[:：](?:\*\*)?\s*(.+?)\s*$",
            re.I,
        )
        for line in text.split("\n"):
            stripped = line.strip()
            m = followup_re.match(stripped)
            if m:
                q = m.group(1).strip()
                if q:
                    followups.append(q)
            else:
                clean_lines.append(line)
        if followups:
            return "\n".join(clean_lines).rstrip(), followups
        return text, []

    @staticmethod
    def _position_order(num_players: int) -> list[str]:
        """Return the preflop position order used by parser/analyzer."""
        return {
            9: ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            8: ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            7: ["UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
            6: ["LJ", "HJ", "CO", "BTN", "SB", "BB"],
            5: ["HJ", "CO", "BTN", "SB", "BB"],
            4: ["CO", "BTN", "SB", "BB"],
            3: ["BTN", "SB", "BB"],
            2: ["SB", "BB"],
        }.get(num_players, [])

    @staticmethod
    def _format_stack_list(stacks: list[Any]) -> str:
        """Format bb stacks without noisy .0 decimals."""
        formatted = []
        for stack in stacks:
            try:
                value = float(stack)
                formatted.append(f"{value:.0f}" if value.is_integer() else f"{value:g}")
            except (TypeError, ValueError):
                formatted.append(str(stack))
        return " / ".join(formatted)

    @staticmethod
    def _solver_stack_list(stacks: Any) -> list[float]:
        """Convert GTO Wizard ICM stack string (15.125-20.125) to bb values."""
        if not stacks:
            return []
        if isinstance(stacks, str):
            parts = [p for p in stacks.split("-") if p]
        elif isinstance(stacks, list):
            parts = stacks
        else:
            return []

        result: list[float] = []
        for part in parts:
            try:
                value = float(part)
            except (TypeError, ValueError):
                continue
            # GTO Wizard stack configs include the 0.125bb ante suffix.
            if value >= 0.125:
                value -= 0.125
            result.append(value)
        return result

    @staticmethod
    def _find_preflop_solution(context: dict) -> tuple[dict | None, dict | None]:
        """Return the cached preflop spot + solution for no-hero range coaching."""
        for spot, solution in zip(context.get("hero_spots", []), context.get("solutions", [])):
            if spot and spot.get("street") == "preflop" and solution:
                return spot, solution
        return None, None

    @staticmethod
    def _format_action_mix(solution: dict) -> str:
        """Format aggregate action frequencies from a spot solution."""
        from gto_formatter import _action_label

        lines = []
        for action_solution in solution.get("action_solutions", []):
            freq = float(action_solution.get("total_frequency", 0))
            if freq < 0.001:
                continue
            combos = float(action_solution.get("total_combos", 0))
            code = action_solution.get("action", {}).get("code", "")
            label = _action_label(code, solution)
            lines.append(f"- {label}: {freq * 100:.1f}%（{combos:.0f} combos）")
        return "\n".join(lines)

    @staticmethod
    def _format_nonfold_ranges(solution: dict, position: str) -> str:
        """Format the playable range by non-fold action for a position."""
        from gto_formatter import _action_label, _compress_range

        player_info = None
        for pi in solution.get("players_info", []):
            if pi.get("player", {}).get("position") == position:
                player_info = pi
                break
        if not player_info:
            return ""

        action_groups: dict[str, list[tuple[str, float, float]]] = {}
        for hand_name, data in (player_info.get("simple_hand_counters") or {}).items():
            freqs = data.get("actions_total_frequencies") or {}
            combos_by_action = data.get("actions_total_combos") or {}
            for code, freq in freqs.items():
                if code == "F" or float(freq) < 0.001:
                    continue
                combos = float(combos_by_action.get(code, 0))
                if combos < 0.01:
                    continue
                action_groups.setdefault(code, []).append((hand_name, float(freq), combos))

        if not action_groups:
            return ""

        lines = []
        for code, group in sorted(
            action_groups.items(), key=lambda item: -sum(h[2] for h in item[1])
        ):
            group.sort(key=lambda hand: -hand[2])
            total = sum(hand[2] for hand in group)
            lines.append(f"- {_action_label(code, solution)}（{total:.0f} combos）: {_compress_range(group)}")
        if any("~" in line for line in lines):
            lines.append("(~ = 該組已併入 >90% 高頻手牌，非 100% 純頻)")
        return "\n".join(lines)

    @staticmethod
    def _build_icm_range_followups(context: dict) -> list[str]:
        """Deterministic follow-up buttons for no-hero ICM preflop ranges."""
        hand = context.get("hand") or {}
        hero_pos = hand.get("hero_position") or context.get("hero_position") or "Hero"
        return [
            f"如果 {hero_pos} open 後被後位 3-bet all-in，要用哪些牌跟注？",
            f"這個 {hero_pos} open range 和 Chip EV 相比差在哪裡？",
            "如果短碼籌碼改變，ICM open range 會怎麼調整？",
        ]

    @staticmethod
    def _format_icm_range_coach_response(context: dict, fallback_text: str = "") -> str:
        """Build a deterministic coach response for no-hero ICM range queries.

        This deliberately does not call the LLM: stack order and approximation
        metadata are user-critical and must be reproduced exactly from the
        solver context.
        """
        hand = context.get("hand") or {}
        hero_pos = hand.get("hero_position") or context.get("hero_position", "Hero")
        num_players = int(hand.get("players_at_table") or len(hand.get("player_stacks") or []) or 0)
        pos_order = GeminiSessionManager._position_order(num_players)

        user_stacks = hand.get("player_stacks") or []
        solver_stacks = GeminiSessionManager._solver_stack_list(context.get("stacks"))
        user_stack_text = GeminiSessionManager._format_stack_list(user_stacks)
        solver_stack_text = GeminiSessionManager._format_stack_list(solver_stacks)
        gametype = context.get("gametype", "")

        hero_stack_line = ""
        if hero_pos in pos_order and len(user_stacks) > pos_order.index(hero_pos):
            hero_stack = user_stacks[pos_order.index(hero_pos)]
            hero_stack_line = f"按 {num_players} 人桌位置順序（{' / '.join(pos_order)}），{hero_pos} 對應 {hero_stack:g}bb。"

        max_diff = None
        if user_stacks and solver_stacks:
            diffs = [abs(float(a) - float(b)) for a, b in zip(user_stacks, solver_stacks)]
            if diffs:
                max_diff = max(diffs)

        spot, solution = GeminiSessionManager._find_preflop_solution(context)
        solver_pos = (spot or {}).get("solver_hero_pos", hero_pos)
        action_mix = GeminiSessionManager._format_action_mix(solution) if solution else ""
        nonfold_ranges = GeminiSessionManager._format_nonfold_ranges(solution, solver_pos) if solution else ""

        top_nonfold = ""
        if solution:
            nonfold_actions = [
                a for a in solution.get("action_solutions", [])
                if a.get("action", {}).get("code") != "F"
            ]
            if nonfold_actions:
                best = max(nonfold_actions, key=lambda a: float(a.get("total_frequency", 0)))
                code = best.get("action", {}).get("code", "")
                from gto_formatter import _action_label
                top_nonfold = (
                    f"主要可玩動作是 {_action_label(code, solution)}，"
                    f"總頻率 {float(best.get('total_frequency', 0)) * 100:.1f}%。"
                )

        lines = [
            "🎯 教練解讀",
            "",
            f"這是 {num_players} 人決賽桌 ICM 的 {hero_pos} preflop range 查詢。{hero_stack_line}",
            "",
            "⚠ 近似說明",
        ]
        if gametype:
            lines.append(f"- ICM 模式: {gametype}")
        if user_stack_text:
            lines.append(f"- 用戶籌碼: {user_stack_text}")
        if solver_stack_text:
            lines.append(f"- Solver 籌碼: {solver_stack_text}")
        if max_diff is not None:
            lines.append(f"- 最大差異: {max_diff:.0f}bb")
        lines.append(
            "- GTO Wizard ICM 只能查內建的 FT stack configuration；系統會用你的原始 stack order 去找最接近的 solver config。"
        )
        if max_diff is not None and max_diff > 10:
            lines.append(
                "- 這次最大差異偏大，所以請把它當作方向性近似：range 的鬆緊與核心手牌可參考，但邊界混頻手牌要更保守解讀。"
            )
        else:
            lines.append("- 這次 stack 差異較小，可把頻率當作較接近的參考。")

        if action_mix:
            lines.extend(["", "📊 Solver 策略", action_mix])
        if top_nonfold:
            lines.extend(["", f"重點：{top_nonfold}ICM 壓力下，這裡不是用一般 Chip EV 的寬 open，而是先保留能承受後位反擊的核心牌。"])
        if nonfold_ranges:
            lines.extend(["", f"✅ {hero_pos} 可玩範圍", nonfold_ranges])

        lines.extend([
            "",
            "實戰上可以這樣記：先照 solver 的主要 raise range 開局；像低對子、弱 Kxs/Qxs、弱 offsuit broadway 這類邊界牌，因為本次 solver stack 近似差異不小，不要把混頻數字當成絕對精準。"
        ])

        if not solution and fallback_text:
            lines.extend(["", "Solver 原始摘要：", fallback_text])

        return "\n".join(lines).strip()

    def _build_hand_summary(self, chat_id: int) -> str:
        """Build a concise hand summary for the system prompt."""
        ctx = self.hand_contexts.get(chat_id)
        if not ctx:
            return (
                "目前沒有分析中的手牌。\n"
                "你必須使用 query_gto 和 query_next_actions 工具查詢 GTO 策略數據。絕對不要在沒有工具數據的情況下回答策略問題！\n"
                "必須提供 effective_bb。\n"
                "\n"
                "Preflop 動作編碼：每個位置一個動作，按 UTG(0)-UTG+1(1)-LJ(2)-HJ(3)-CO(4)-BTN(5)-SB(6)-BB(7) 順序，用 - 分隔。\n"
                "F=Fold, C=Call, RX=Raise to X, AI=All-in。Raise size 不用精確，系統會自動校正。\n"
                "重要：MTTGeneral 每人有 0.125bb ante（8人桌 = 1bb），計算底池大小時必須加上！\n"
                "例：LJ open 2.1bb BTN call → pot = 0.5(SB) + 1(BB) + 1(antes) + 2.1 + 2.1 = 6.7bb\n"
                "查詢某位置的策略時，preflop_actions_override 只需包含到該位置行動前的動作。\n"
                "UTG 是第一個行動者，不需要 preflop_actions_override（留空即可）。\n"
                "\n"
                "例：查詢 60bb UTG open range → effective_bb=60, street='preflop', position='UTG'（不需要 preflop_actions_override）\n"
                "例：查詢 30bb 下 LJ open 後 SB 的策略 → effective_bb=30, preflop_actions_override='F-F-R2-F-F-F', street='preflop', position='SB'\n"
                "例：查詢 25bb 下 UTG+1 open 後 BB all-in 範圍 → effective_bb=25, preflop_actions_override='F-R2-F-F-F-F-F', street='preflop', position='BB'\n"
                "\n"
                "Postflop 查詢：\n"
                "先用 preflop_actions_override 建構完整 preflop 動作（包含所有 8 個位置），再加 board_override 和 street='flop'。\n"
                "例：40bb BTN open SB 3bet BTN call, flop Qs7h2d, SB 策略\n"
                "  → effective_bb=40, preflop_actions_override='F-F-F-F-F-R2-R8-F-C', board_override='Qs7h2d', street='flop', position='SB'\n"
                "\n"
                "重要：查詢面對 re-raise 的決策（如 UTG+1 open 後 BTN 3bet，UTG+1 要 call/fold）時，\n"
                "preflop_actions_override 必須包含完整 8 個位置（其他位置用 F），這樣才能查到該位置的第二次決策。\n"
                "例：UTG+1 面對 BTN 3bet SB 4bet → preflop_actions_override='F-R2-F-F-F-AI10-AI30-F', position='UTG+1'\n"
                "\n"
                "ICM 查詢：用戶提到 ICM / 錦標賽壓力 / 泡沫期 / 決賽桌 / 剩多少%人 時，必須使用 icm_phase 參數。\n"
                "同時指定 num_players（桌上人數）和 effective_bb。\n"
                "例：ICM 25% 8人桌 20bb → icm_phase='PCT25', num_players=8, effective_bb=20\n"
                "例：決賽桌 6人 30bb → icm_phase='FT', num_players=6, effective_bb=30"
            )

        lines = [
            "目前分析的手牌：",
            f"- Hero: {ctx['hero_position']}{'' if ctx.get('no_hero_hand') else ' ' + cards_to_emoji(ctx['hero_hand'])}, {float(ctx['depth']) - 0.125:.0f}bb depth",
            f"- Preflop: {ctx['preflop_actions']}",
        ]

        states = ctx.get("street_states", {})
        final = ctx.get("final_actions", {})
        for street_name in ["flop", "turn", "river"]:
            state = states.get(street_name)
            if not state:
                break
            board = state["board"]
            acts = final.get(f"{street_name}_actions", "")
            lines.append(f"- {street_name.capitalize()}: board={cards_to_emoji(board)} | actions={acts}")

        # Include range breakdown from cached solutions to prevent hallucination.
        # Gemini tends to fabricate range compositions instead of using tools.
        from gto_formatter import format_range_by_action
        hero_pos = ctx.get("hero_position", "")
        for spot, sol in zip(ctx.get("hero_spots", []), ctx.get("solutions", [])):
            if sol is None:
                continue
            street = spot.get("street", "")
            try:
                rb = format_range_by_action(sol, hero_pos)
                if rb:
                    lines.append(
                        f"\n{street.capitalize()} 策略分佈（回答「哪些手牌下注/過牌/加注」"
                        f"「某類牌怎麼打」類問題的唯一依據——必須照此分類回答，"
                        f"不可用撲克理論覆蓋）："
                    )
                    lines.append(rb)
            except Exception:
                pass

        lines.append("")
        lines.append(
            "工具使用指南：\n"
            "1. query_next_actions — 查詢某個決策點的所有可用動作和正確的 action code\n"
            "2. query_gto — 查詢完整策略數據（範圍、頻率、EV）\n"
            "\n"
            "重要規則：\n"
            "• 當用戶問假設情境（例如「如果 flop 打滿池」），先用 query_next_actions 查出正確的 action code，再用 query_gto。\n"
            "• Raise size 不需要精確（例如可以寫 R2），系統會自動校正到最近的 solver sizing（如 R2.1）。\n"
            "• 當用戶指定不同的籌碼深度（如 '30bb effective'），必須傳入 effective_bb 參數。不同深度的 solver sizing 不同！\n"
            "\n"
            "Preflop 動作編碼：每個位置一個動作，按 UTG(0)-UTG+1(1)-LJ(2)-HJ(3)-CO(4)-BTN(5)-SB(6)-BB(7) 順序，用 - 分隔。\n"
            "F=Fold, C=Call, RX=Raise to X, AI=All-in。\n"
            "查詢某位置的策略時，preflop_actions_override 只需包含到該位置行動前的動作。\n"
            "例：查詢 30bb 下 LJ open 後 BB 的策略 → effective_bb=30, preflop_actions_override='F-F-R2-F-F-F-F'\n"
            "例：查詢 UTG+1 open 後 BTN 3bet 範圍 → preflop_actions_override='F-R2-F-F-F'\n"
            "\n"
            "ICM 查詢：用戶提到 ICM / 錦標賽壓力 / 泡沫期 / 決賽桌 時，使用 icm_phase 參數。\n"
            "例：ICM 25% 8人桌 20bb → icm_phase='PCT25', num_players=8, effective_bb=20"
        )

        return "\n".join(lines)

    def clear_session(self, chat_id: int) -> None:
        """Clear conversation history and hand context for a chat."""
        self.histories.pop(chat_id, None)
        self.hand_contexts.pop(chat_id, None)
