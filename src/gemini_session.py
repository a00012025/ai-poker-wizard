# src/gemini_session.py
"""Gemini-based session manager — direct API calls, no CLI subprocess.

Flow: user message → parse hand (Flash) → analyze_hand_full() → coaching (Pro)
Follow-ups: user message → parse (null) → Pro chat WITH query_gto tool → real data
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from google import genai
from google.genai import types

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_LOG_DIR = _PROJECT_ROOT / "logs"

# Allow importing from scripts/
sys.path.insert(0, str(_SCRIPTS_DIR))

PARSE_PROMPT = """\
你是撲克手牌解析器。分析用戶訊息，如果包含手牌描述，提取為 JSON。
如果不是手牌（例如追問、閒聊），回覆 {"hand": null}。
重要：只要訊息包含足以構成手牌的資訊（有效籌碼、位置、手牌、preflop 動作），即使同時包含問題（如「該跟嗎？」「對手範圍？」），也要提取手牌 JSON！
例如「有效 30bb, hero +1 raise, btn all in, 我 TT 該跟嗎？」→ 這是手牌描述，要提取！

規則：
- 預設 MTT 8-max 位置順序：UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5), SB(6), BB(7)
- 不同人數位置順序（重要！按人數調整，preflop_actions 長度必須等於人數）：
  9人: UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB
  8人: UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB（預設）
  7人: UTG, LJ, HJ, CO, BTN, SB, BB
  6人: LJ, HJ, CO, BTN, SB, BB
  5人: HJ, CO, BTN, SB, BB
  4人: CO, BTN, SB, BB
  3人: BTN, SB, BB
- preflop_actions：必須列出所有位置的動作，用 - 分隔。F=Fold, C=Call, RX=Raise to X, AI=All-in, AI{size}=All-in for specific size
  如果用戶有提到 all-in 的大小（如 "all in 10bb"），必須用 AI{size} 格式（如 AI10）！只有不知道大小時才用 AI。
  重要：即使某些位置之後 fold 了，他們初始的 raise/call 動作仍要保留！
  例1：CO raise 2bb, BB call → F-F-F-F-R2-F-F-C
  例2（多人底池）：UTG+1 raise 2bb, LJ call, CO call, SB raise 10bb → F-R2-C-F-C-F-R10-F
  例3（3bet pot）：CO raise 2.5bb, BB raise 8bb, CO call → F-F-F-F-R2.5-F-F-R8-C
  注意例2：UTG+1 的 R2、LJ 的 C、CO 的 C 都要保留，不能省略成 F！
  注意例3：8 個位置後面的 -C 是 CO 面對 3bet 後 call 的動作（continuation action）
- 多人底池 + 3bet 後 continuation actions（重要！）：
  當有人 re-raise 後，之前 call 過的人會再次行動。這些動作接在 N 個位置後面，按原始行動順序排列。
  例：UTG+1 raise 2bb, LJ call, CO call, SB raise 10bb, UTG+1 fold, LJ fold, CO call
  → F-R2-C-F-C-F-R10-F-F-F-C（8個位置 + UTG+1 fold + LJ fold + CO call）
- Board 格式：Js6h5s（rank+suit: c/d/h/s）。如果用戶只說 "J65 two spade" 你要推斷出 Js6s5x 之類的（花色不確定的用最合理的猜測）
- 翻牌後行動順序（重要！）：SB 永遠先行動，然後 BB，然後其他位置按順序，BTN 最後。
  BvB 例子：SB bet, BB call → [{"position":"SB","action":"R2","size":2},{"position":"BB","action":"C"}]（SB 先行動，不要在前面加 BB check！）
- Postflop actions 只列出實際發生的動作，不要自己推測或補上未提及的 check
- streets：flop 用 "board"，turn/river 用 "card"
- hero_hand：如果用戶說 "66" 就用 "66"，如果說 "Ah Ks" 就用 "AhKs"
- effective_bb：取整數
- 翻牌後 size：必須是絕對 bb 值！如果用戶說 "bet 40%" 或 "bet 1/3"，請根據底池大小估算 bb 值。例如底池 5bb，bet 40% → size: 2.0（不是 40 或 0.4）

ICM 支援：
- 如果用戶提到 ICM、bubble、final table、錦標賽階段、不同位置有不同籌碼量，加入以下欄位：
  "tournament_type": "icm"（預設不寫 = chip EV）
  "pko": true/false（是否 PKO/bounty 錦標賽，預設 false）
  "tournament_size": 1000 或 200（錦標賽人數，預設 1000）
  "players_remaining": 數字（剩餘人數，例如 152）
  "phase": 階段名稱（可選，如 "BUBBLE", "FT", "PCT25" 等）
  "player_stacks": [每個位置的籌碼]（按位置順序排列，如 [50, 30, 45, 20, 35, 25, 15, 40]）
- 用戶說「ICM bubble 50bb」且沒提到個別籌碼 → 不需要 player_stacks，只需 tournament_type + phase
- 用戶說「6人 FT, BTN 60bb, SB 25bb...」→ player_stacks 按 6人順序（LJ, HJ, CO, BTN, SB, BB）
- phase 對應規則：
  early/開始 → "START"
  75% left → "PCT75"
  50% left → "PCT50"
  25% left → "PCT25"
  bubble → "BUBBLE"（泡沫）
  10% left → "PCT10"
  5% left → "PCT5"
  final table/FT → "FT"
  兩桌 → "T2"
  三桌 → "T3"

JSON 格式（Chip EV，預設）：
```json
{
  "hand": {
    "gametype": "MTTGeneral",
    "effective_bb": 32,
    "hero_position": "CO",
    "hero_hand": "66",
    "preflop_actions": "F-F-F-F-R2-F-F-C",
    "streets": [...]
  }
}
```

JSON 格式（ICM）：
```json
{
  "hand": {
    "gametype": "MTTGeneral",
    "tournament_type": "icm",
    "tournament_size": 1000,
    "players_remaining": 152,
    "phase": "BUBBLE",
    "player_stacks": [50, 50, 50, 50, 50, 50, 50, 50],
    "effective_bb": 50,
    "hero_position": "SB",
    "hero_hand": "A5s",
    "preflop_actions": "F-F-F-F-F-F"
  }
}
```

注意：
- 如果用戶沒給某些資訊（例如花色），用最合理的猜測並在 JSON 外加一句說明
- Raise size 如果用戶沒說具體金額，MTT preflop open 預設用 2bb（輸出 R2）
- 只回覆 JSON（可以用 ```json ``` 包住）
- 再次強調：翻牌後 SB 永遠第一個行動！BvB 時 SB bet → 不需要在前面加 BB check
- 再次強調：preflop_actions 必須保留所有位置的動作！多人底池不能省略成只有兩人！"""

IMAGE_PARSE_PROMPT = """\
你是撲克截圖解析器。從上傳的撲克手牌回放截圖中提取手牌資訊為 JSON。

截圖閱讀方式：
1. Hero = 畫面底部中央的玩家，手牌朝上展示（或有 WIN/LOSE 標記）
2. 底部面板分 Pre-Flop / Flop / Turn / River 欄位，每欄從上到下是行動順序
3. 每個玩家有位置標籤（UTG、CO、BTN、SB、BB 等）和籌碼量（XX BB）
4. 桌面中央是公共牌

提取規則：
- gametype: 固定 "MTTGeneral"
- 位置順序（按人數）：
  9人: UTG, UTG+1, UTG+2, LJ, HJ, CO, BTN, SB, BB
  8人: UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB（預設）
  6人: LJ, HJ, CO, BTN, SB, BB
- preflop_actions: 按位置順序列出所有動作，用 - 分隔
  F=Fold, C=Call, RX=Raise to Xbb, AIX=All-in Xbb
  3bet/4bet 後的 continuation actions 接在第一輪後面
  例：UTG+1 raise 2, CO call, SB raise 10, UTG+1 fold, CO call
  → F-R2-F-F-C-F-R10-F-F-C（8位置 + UTG+1 fold + CO call）
- effective_bb: min(hero 籌碼, 進入底池的對手中最小籌碼)
- 牌面記號：rank 用單字元 2-9, T, J, Q, K, A（十=T，不是10！）
  suit 用 c♣ d♦ h♥ s♠，如 "AsKc", "Ts4h"
- hero_hand: 兩張牌，如 "AsKc"
- streets: flop 用 "board"（如 "6cQs9d"），turn/river 用 "card"
- 翻牌後 action: X=Check, C=Call, F=Fold, R{size}=Bet/Raise（size 為 bb 絕對值）
- 翻牌後行動順序：靠近 SB 的位置先行動

JSON 格式：
```json
{
  "hand": {
    "gametype": "MTTGeneral",
    "effective_bb": 16,
    "hero_position": "LJ",
    "hero_hand": "AsKc",
    "preflop_actions": "F-F-R2-F-C-F-F-F",
    "streets": [
      {"board": "6cQs9d", "actions": [
        {"position": "LJ", "action": "X"},
        {"position": "CO", "action": "R1.9", "size": 1.9},
        {"position": "LJ", "action": "C"}
      ]},
      {"card": "4s", "actions": [
        {"position": "LJ", "action": "X"},
        {"position": "CO", "action": "X"}
      ]},
      {"card": "4h", "actions": [
        {"position": "LJ", "action": "R3", "size": 3},
        {"position": "CO", "action": "F"}
      ]}
    ]
  }
}
```

只回覆 JSON。如果截圖不是撲克手牌，回覆 {"hand": null}。
如果截圖不清楚某些資訊，用最合理的猜測。"""

COACH_SYSTEM = """\
你是專業 MTT 撲克教練 AI Poker Wizard。用繁體中文回覆。

格式規則（嚴格遵守！輸出直接發送到 Telegram）：
- 絕對不要用 # ## ### 等任何標題語法
- 絕對不要用 * 作為列表符號（Telegram 會誤判為粗體）
- 列表只用 1. 2. 3. 數字 或 • 符號
- 段落標題用 *粗體*（單星號），例如 *Preflop*
- 重點詞也用 *粗體*
- 不要用 **雙星號**、不要用表格

風格：
- 精簡直接，像教練用最少的話點出重點
- 不要廢話、不要重複已知資訊、不要客套開場
- 每條街 2-4 行就夠：GTO 怎麼打 → hero 怎麼打 → 差在哪 → 為什麼（一句話）
- 如果 hero 打得對，一句帶過就好，不用展開分析
- 數據引用要精準但不要列出所有選項，只提最重要的 1-2 個動作頻率
- 混合策略是重要資訊，必須標出頻率！不要說「所有口袋對都開」，要說「55+ 純開，22-44 混合（22 約 60%、33 約 75%、44 約 90%）」

重要原則：
- 分析必須完全基於 GTO Solver 數據，不要自行編造
- 如果訊息中已經包含「GTO Solver 數據」，這就是真實的 solver 分析結果！必須先根據這些數據分析 hero 的策略，不需要再用工具重複查詢
- 只有用戶的額外問題（如「對手範圍？」「不同位置的策略？」）才需要用 query_gto 工具查詢
- 「無 solver 數據」的街直接跳過，不要猜測或推斷該街的 GTO 策略
- 如果所有街都沒有 solver 數據，只簡短說明無法分析，不要輸出任何策略建議

回答流程（重要！）：
- 第一步：根據已提供的 GTO Solver 數據，分析 hero 的行動是否正確（頻率、EV）
- 第二步：如果用戶有額外問題（如對手範圍、假設場景），使用 query_gto 工具查詢後回答
- 兩個部分都要回答！不能只回答其中一個

多人底池簡化（重要！）：
- 當數據標記「⚠ 多人底池，簡化為 X open vs Y ... 單挑分析」時，表示原始多人底池已簡化為最接近的單挑場景
- Solver 的底池大小、籌碼深度會與用戶描述不同，這是正常的！不要因此拒絕分析
- 策略頻率（check/bet/raise 比例）和 EV 仍然是有效的參考
- 下注大小已按底池比例映射（例如實際 20% pot → solver 25% pot），直接用 solver 的百分比分析即可
- 分析時用 solver 的百分比（如「25% pot bet」），不要糾結於絕對 bb 數字的差異

近似場景分析（重要！）：
- 當數據包含「⚠ 近似說明」時，表示實際場景無法被 solver 完全模擬，使用了最接近的替代解
- 在分析開頭簡要說明近似方式（如「BTN all-in 10bb 被近似為 3bet 6.3bb」）以及可能的偏差
- 強調分析結果是參考性質，但仍有參考價值

分析結構：
1. 每條街的 GTO vs Hero 對比（只講有意義的差異）
2. 如果 hero 有明顯錯誤：指出最關鍵的 1 個錯誤 + 為什麼 + 一句改進建議
3. 如果 hero 全部打對：不需要「最關鍵的錯誤」或「改進建議」段落，直接結束即可"""

# ── Gemini tool schema for GTO queries ──

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
                description="查詢特定手牌的策略，例如 66, AhKs, QQ。不指定則回傳該位置的完整範圍概覽。",
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
                description="假設不同的 board（覆蓋實際 board）。例如查詢 turn 掉 Kd 而非 Kc：傳入 Js5s6hKd。",
            ),
            "flop_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的翻牌動作序列。格式：X=check, C=call, F=fold, R{size}=raise。例如 hero check through 用 X-X。",
            ),
            "turn_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的轉牌動作序列。",
            ),
            "river_actions_override": types.Schema(
                type=types.Type.STRING,
                description="假設不同的河牌動作序列。",
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


class GeminiSessionManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境變數未設定")

        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.parse_model = os.getenv("GEMINI_PARSE_MODEL", "gemini-2.5-flash")
        self.image_parse_model = os.getenv("GEMINI_IMAGE_PARSE_MODEL", "gemini-3-pro-preview")
        self.max_turns = "N/A"  # for bot.py compat
        self.histories: Dict[int, List[types.Content]] = {}
        self.hand_contexts: Dict[int, dict] = {}

        # Logging
        _LOG_DIR.mkdir(exist_ok=True)
        self._logger = logging.getLogger("gemini_session")
        if not self._logger.handlers:
            handler = logging.FileHandler(_LOG_DIR / "gemini_session.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    async def send_message(self, chat_id: int, user_text: str,
                           on_status: Callable[[str], Any] | None = None) -> str:
        """Main entry: parse hand → GTO analysis → coaching, or chat with tools.

        Args:
            on_status: optional async/sync callback(status_msg) for progress updates
        """
        t0 = time.time()
        self._logger.info(f"[chat={chat_id}] User: {user_text[:300]}")

        async def _status(msg: str):
            if on_status:
                r = on_status(msg)
                if asyncio.iscoroutine(r):
                    await r

        try:
            # Step 1: Parse hand from user message (Flash — fast)
            await _status("解析手牌中...")
            hand_json = await asyncio.wait_for(
                self._parse_hand(chat_id, user_text), timeout=60,
            )
            t_parse = time.time()

            if hand_json:
                self._logger.info(
                    f"[chat={chat_id}] Parsed hand in {t_parse - t0:.1f}s "
                    f"(model={self.parse_model}): "
                    f"{json.dumps(hand_json, ensure_ascii=False)[:300]}"
                )

                # Step 2: Ensure GTO Wizard session is valid
                from gto_token import ensure_session, capture_browser_token
                if not ensure_session():
                    self._logger.warning(f"[chat={chat_id}] Session expired, browser opened for login")
                    import asyncio as _aio
                    for _ in range(24):
                        await _aio.sleep(5)
                        if capture_browser_token():
                            self._logger.info(f"[chat={chat_id}] Browser login captured")
                            break
                    else:
                        return "GTO Wizard session 已過期，已開啟瀏覽器。請登入後重新傳送手牌。"

                # Step 3: Run GTO analysis and cache context
                await _status("查詢 GTO 策略中...")
                from analyze_hand import analyze_hand_full
                context = analyze_hand_full(hand_json)
                gto_data = context["text"]
                self.hand_contexts[chat_id] = context

                t_analyze = time.time()
                self._logger.info(
                    f"[chat={chat_id}] GTO analysis in {t_analyze - t_parse:.1f}s "
                    f"({len(gto_data)} chars) — context cached"
                )
                self._logger.debug(f"[chat={chat_id}] GTO data:\n{gto_data}")

                # Step 4: Coaching from LLM (with tools for follow-up queries)
                await _status("分析回覆中...")
                coaching_prompt = (
                    f"用戶描述：\n{user_text}\n\n"
                    f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
                    f"請先根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
                )
                result = await self._chat_with_tools(chat_id, coaching_prompt, on_status=on_status)
                t_total = time.time()
                self._logger.info(
                    f"[chat={chat_id}] Done: parse={t_parse - t0:.1f}s "
                    f"gto={t_analyze - t_parse:.1f}s "
                    f"coach={t_total - t_analyze:.1f}s "
                    f"total={t_total - t0:.1f}s"
                )
                return result
            else:
                # Not a hand — chat (with tools if hand context exists)
                await _status("查詢中...")
                result = await self._chat(chat_id, user_text, on_status=on_status)
                elapsed = time.time() - t0
                self._logger.info(f"[chat={chat_id}] Chat response in {elapsed:.1f}s")
                return result

        except asyncio.TimeoutError:
            self._logger.error(f"[chat={chat_id}] Gemini API timeout")
            raise RuntimeError("Gemini API 回應超時，請稍後再試。")
        except Exception as e:
            self._logger.error(f"[chat={chat_id}] Error: {e}", exc_info=True)
            raise

    async def send_image_message(self, chat_id: int, image_bytes: bytes,
                                    mime_type: str = "image/jpeg",
                                    user_text: str = "",
                                    status_callback=None) -> str:
        """Main entry for image-based hand analysis: parse screenshot → GTO → coaching.

        status_callback: optional async callable(str) to update user-facing status.
        """
        t0 = time.time()
        self._logger.info(
            f"[chat={chat_id}] Image message ({len(image_bytes)} bytes), "
            f"caption: {user_text[:200]}"
        )

        async def _update_status(text: str):
            if status_callback:
                try:
                    await status_callback(text)
                except Exception:
                    pass

        try:
            # Step 1: Parse hand from screenshot
            await _update_status("🔍 正在辨識截圖中的手牌...")
            hand_json = await self._parse_hand_from_image(chat_id, image_bytes, mime_type)
            t_parse = time.time()

            if not hand_json:
                self._logger.info(f"[chat={chat_id}] No hand found in image")
                if user_text.strip():
                    return await self._chat(chat_id, user_text)
                return "無法從截圖中辨識出撲克手牌。請確認截圖是手牌回放畫面（包含底部動作面板）。"

            self._logger.info(
                f"[chat={chat_id}] Parsed image hand in {t_parse - t0:.1f}s: "
                f"{json.dumps(hand_json, ensure_ascii=False)[:300]}"
            )

            # Step 2: Ensure GTO Wizard session
            await _update_status(
                f"📊 辨識完成：{hand_json['hero_position']} {hand_json['hero_hand']} "
                f"({hand_json['effective_bb']:.0f}bb)，正在查詢 GTO 策略..."
            )
            from gto_token import ensure_session
            if not ensure_session():
                return "GTO Wizard session 已過期，請管理員更新 token。"

            # Step 3: GTO analysis
            from analyze_hand import analyze_hand_full
            context = analyze_hand_full(hand_json)
            gto_data = context["text"]
            self.hand_contexts[chat_id] = context

            t_analyze = time.time()
            self._logger.info(
                f"[chat={chat_id}] Image GTO analysis in {t_analyze - t_parse:.1f}s"
            )

            # Step 4: Coaching with user's caption/question
            hand_desc = (
                f"Hero {hand_json['hero_position']} {hand_json['hero_hand']} "
                f"({hand_json['effective_bb']:.0f}bb)\n"
                f"Preflop: {hand_json['preflop_actions']}"
            )
            if hand_json.get("streets"):
                for s in hand_json["streets"]:
                    board = s.get("board", s.get("card", ""))
                    acts = " ".join(
                        f"{a['position']}:{a['action']}" for a in s["actions"]
                    )
                    hand_desc += f"\n{board} → {acts}"

            user_q = user_text.strip() if user_text.strip() else "請分析這手牌"
            coaching_prompt = (
                f"用戶上傳了撲克截圖，已從截圖中解析出手牌：\n{hand_desc}\n\n"
                f"用戶留言：{user_q}\n\n"
                f"GTO Solver 數據（已查詢完成，直接分析即可）：\n{gto_data}\n\n"
                f"請先根據上面的 GTO 數據分析 hero 的行動，再用工具回答用戶的其他問題。"
            )
            result = await self._chat_with_tools(chat_id, coaching_prompt)

            t_total = time.time()
            self._logger.info(
                f"[chat={chat_id}] Image done: parse={t_parse - t0:.1f}s "
                f"gto={t_analyze - t_parse:.1f}s total={t_total - t0:.1f}s"
            )
            return result

        except asyncio.TimeoutError:
            self._logger.error(f"[chat={chat_id}] Image Gemini API timeout")
            raise RuntimeError("Gemini API 回應超時，請稍後再試。")
        except Exception as e:
            self._logger.error(f"[chat={chat_id}] Image error: {e}", exc_info=True)
            raise

    async def _parse_hand_from_image(self, chat_id: int, image_bytes: bytes,
                                       mime_type: str = "image/jpeg") -> dict | None:
        """Parse hand from a screenshot image using Gemini vision."""
        self._logger.debug(f"[chat={chat_id}] Parsing hand from image ({len(image_bytes)} bytes)")

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.image_parse_model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        types.Part(text=IMAGE_PARSE_PROMPT),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=8192),
                ),
            ),
            timeout=120,
        )

        text = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Image parse response:\n{text}")

        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text.strip()

        try:
            result = json.loads(json_str)
            hand = result.get("hand")
            if hand and hand.get("hero_position") and hand.get("preflop_actions") and hand.get("hero_hand"):
                self._normalize_cards(hand)
                return hand
        except (json.JSONDecodeError, AttributeError) as e:
            self._logger.warning(
                f"[chat={chat_id}] Image JSON parse failed: {e}\nRaw: {json_str[:500]}"
            )

        return None

    @staticmethod
    def _normalize_cards(hand: dict):
        """Fix common Gemini vision mistakes in card notation (e.g. '10' → 'T')."""
        hand["hero_hand"] = re.sub(r"10", "T", hand["hero_hand"])
        for street in hand.get("streets", []):
            if "board" in street:
                street["board"] = re.sub(r"10", "T", street["board"])
            if "card" in street:
                street["card"] = re.sub(r"10", "T", street["card"])

    async def _parse_hand(self, chat_id: int, user_text: str) -> dict | None:
        """Parse user's natural language into hand JSON. Uses Flash for speed."""
        prompt = f"{PARSE_PROMPT}\n\n用戶訊息：\n{user_text}"
        self._logger.debug(f"[chat={chat_id}] Parse request: {user_text}")

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.parse_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            ),
            timeout=60,
        )

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

    async def _coach(self, chat_id: int, user_text: str, gto_data: str) -> str:
        """Generate coaching analysis from GTO solver data."""
        coaching_prompt = (
            f"用戶描述：\n{user_text}\n\n"
            f"GTO Solver 數據：\n{gto_data}"
        )
        self._logger.debug(
            f"[chat={chat_id}] Coach prompt (model={self.model}, "
            f"{len(coaching_prompt)} chars):\n{coaching_prompt}"
        )

        history = self.histories.get(chat_id, [])
        messages = list(history) + [
            types.Content(role="user", parts=[types.Part(text=coaching_prompt)]),
        ]

        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=COACH_SYSTEM,
                ),
            ),
            timeout=120,
        )

        result = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Coach response ({len(result)} chars):\n{result}")

        # Update history (keep user's original text, not the coaching prompt)
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=result)]))
        self.histories[chat_id] = history[-20:]

        return result

    async def _chat(self, chat_id: int, user_text: str,
                     on_status: Callable[[str], Any] | None = None) -> str:
        """Chat with GTO tool access — always provides tools so model can query solver."""
        self._logger.debug(f"[chat={chat_id}] Chat with tools (model={self.model}): {user_text[:300]}")
        return await self._chat_with_tools(chat_id, user_text, on_status=on_status)

    async def _chat_with_tools(self, chat_id: int, user_text: str,
                                on_status: Callable[[str], Any] | None = None) -> str:
        """Chat with GTO tools for data-driven follow-up answers."""
        tool = types.Tool(function_declarations=[
            QUERY_NEXT_ACTIONS_DECLARATION,
            QUERY_GTO_DECLARATION,
        ])

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

        async def _status(msg: str):
            if on_status:
                r = on_status(msg)
                if asyncio.iscoroutine(r):
                    await r

        for round_num in range(max_rounds):
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        tools=[tool],
                    ),
                ),
                timeout=120,
            )

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

                # Status update for tool calls
                pos = args.get("position", "")
                street = args.get("street", "")
                icm = args.get("icm_phase", "")
                tool_desc = f"查詢 {pos} {street}" if pos else f"查詢 {street} 策略"
                if icm:
                    tool_desc += f" (ICM {icm})"
                await _status(tool_desc + "...")

                t_tool = time.time()
                if fn_name == "query_next_actions":
                    tool_result = self._execute_query_next_actions(chat_id, args)
                else:
                    tool_result = self._execute_query_gto(chat_id, args)
                elapsed = time.time() - t_tool
                self._logger.debug(
                    f"[chat={chat_id}] Tool result ({elapsed:.1f}s, {len(tool_result)} chars):\n"
                    f"{tool_result[:500]}"
                )
                tools_called += 1

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
                ))]))
            else:
                # No tools were called — ask model to try answering directly
                messages.append(types.Content(role="user", parts=[types.Part(text=(
                    "請直接回答用戶的問題。如果需要 GTO 數據支持，"
                    "根據系統提示中的手牌資訊描述你所知道的策略。"
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
            result_text = response.text or "抱歉，分析過程中出現問題，請重新傳送手牌。"

        self._logger.debug(f"[chat={chat_id}] Chat+tools response ({len(result_text)} chars):\n{result_text}")

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

    def _execute_query_gto(self, chat_id: int, args: dict) -> str:
        """Execute a query_gto tool call. Returns formatted solver data."""
        from gto_api import get_spot_solution, get_next_actions, find_closest_action
        from gto_formatter import format_action_summary, format_hand_detail, format_range_overview

        from gto_api import nearest_depth as _nearest_depth

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

        # Override depth if effective_bb specified (only for non-ICM; ICM depth already set)
        depth_override = _nearest_depth(effective_bb) if effective_bb and not args.get("icm_phase") else None

        has_override = any([preflop_override, board_override, flop_override, turn_override, river_override, depth_override])

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

        try:
            solution = get_spot_solution(**params)
        except Exception as e:
            return f"API 查詢失敗：{e}"

        if not solution:
            return f"{street} 沒有 solver 數據（可能是無效的 board 或 actions 組合）。"

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

        return self._format_solution(solution, position, hand)

    def _find_cached_solution(self, ctx: dict, street: str) -> dict | None:
        """Find a cached spot-solution for the given street."""
        for spot, sol in zip(ctx["hero_spots"], ctx["solutions"]):
            if spot["street"] == street and sol is not None:
                return sol
        return None

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
        from gto_api import get_next_actions, find_closest_action

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
                        target = float(code[1:])
                        correct_code = find_closest_action(avail, target)
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

        # For ICM, depth is already set correctly in ctx
        depth = ctx["depth"] if args.get("icm_phase") else (_nearest_depth(effective_bb) if effective_bb else ctx["depth"])

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

    def _build_hand_summary(self, chat_id: int) -> str:
        """Build a concise hand summary for the system prompt."""
        ctx = self.hand_contexts.get(chat_id)
        if not ctx:
            return (
                "目前沒有分析中的手牌。\n"
                "你可以使用 query_gto 和 query_next_actions 工具查詢任何 GTO 策略，必須提供 effective_bb。\n"
                "\n"
                "Preflop 動作編碼：每個位置一個動作，按 UTG(0)-UTG+1(1)-LJ(2)-HJ(3)-CO(4)-BTN(5)-SB(6)-BB(7) 順序，用 - 分隔。\n"
                "F=Fold, C=Call, RX=Raise to X, AI=All-in。Raise size 不用精確，系統會自動校正。\n"
                "查詢某位置的策略時，preflop_actions_override 只需包含到該位置行動前的動作。\n"
                "UTG 是第一個行動者，不需要 preflop_actions_override（留空即可）。\n"
                "\n"
                "例：查詢 60bb UTG open range → effective_bb=60, street='preflop', position='UTG'（不需要 preflop_actions_override）\n"
                "例：查詢 30bb 下 LJ open 後 SB 的策略 → effective_bb=30, preflop_actions_override='F-F-R2-F-F-F', street='preflop', position='SB'\n"
                "例：查詢 25bb 下 UTG+1 open 後 BB all-in 範圍 → effective_bb=25, preflop_actions_override='F-R2-F-F-F-F-F', street='preflop', position='BB'\n"
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
            f"- Hero: {ctx['hero_position']} {ctx['hero_hand']}, {float(ctx['depth']) - 0.125:.0f}bb depth",
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
            lines.append(f"- {street_name.capitalize()}: board={board} | actions={acts}")

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
