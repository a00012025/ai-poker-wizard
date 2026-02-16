# src/gemini_session.py
"""Gemini-based session manager — direct API calls, no CLI subprocess.

Flow: user message → parse hand (Flash) → analyze_hand.py → coaching (Pro thinking)
"""
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

from google import genai
from google.genai import types

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_LOG_DIR = _PROJECT_ROOT / "logs"

# Allow importing analyze_hand from scripts/
sys.path.insert(0, str(_SCRIPTS_DIR))

PARSE_PROMPT = """\
你是撲克手牌解析器。分析用戶訊息，如果包含手牌描述，提取為 JSON。
如果不是手牌（例如追問、閒聊），回覆 {"hand": null}。

規則：
- MTT 8-max 位置順序：UTG(0), UTG+1(1), LJ(2), HJ(3), CO(4), BTN(5), SB(6), BB(7)
- preflop_actions：每個位置一個動作，用 - 分隔。F=Fold, C=Call, RX=Raise to X, AI=All-in
  例：CO raise 2bb, BB call → F-F-F-F-R2-F-F-C
- Board 格式：Js6h5s（rank+suit: c/d/h/s）。如果用戶只說 "J65 two spade" 你要推斷出 Js6s5x 之類的（花色不確定的用最合理的猜測）
- Postflop actions 必須包含所有動作（check, bet, call, fold 都要列出）
- streets：flop 用 "board"，turn/river 用 "card"
- hero_hand：如果用戶說 "66" 就用 "66"，如果說 "Ah Ks" 就用 "AhKs"
- effective_bb：取整數

JSON 格式：
```json
{
  "hand": {
    "gametype": "MTTGeneral",
    "effective_bb": 32,
    "hero_position": "CO",
    "hero_hand": "66",
    "preflop_actions": "F-F-F-F-R2-F-F-C",
    "streets": [
      {
        "board": "Js6h5s",
        "actions": [
          {"position": "BB", "action": "X"},
          {"position": "CO", "action": "R2", "size": 2.0},
          {"position": "BB", "action": "C"}
        ]
      },
      {
        "card": "Kc",
        "actions": [
          {"position": "BB", "action": "X"},
          {"position": "CO", "action": "R6.6", "size": 6.6},
          {"position": "BB", "action": "C"}
        ]
      }
    ]
  }
}
```

注意：
- 如果用戶沒給某些資訊（例如花色），用最合理的猜測並在 JSON 外加一句說明
- Raise size 如果用戶沒說具體金額，用常見的 size（preflop open 通常 2-2.5bb）
- 只回覆 JSON（可以用 ```json ``` 包住）"""

COACH_SYSTEM = """\
你是專業 MTT 撲克錦標賽教練，名叫 AI Poker Wizard。用繁體中文回覆。

格式規則（重要！Telegram 不支援 Markdown 標題）：
- 不要用 # ## 等標題語法
- 用 *粗體* 標記重點詞（單星號 *text*，不要用雙星號）
- 用數字或 • 列表
- 簡潔有力，像教練對學生說話
- 不要用表格

重要原則：
- 你的分析必須完全基於提供的 GTO Solver 數據
- 如果某條街顯示「無 solver 數據」，直接說明該街無法分析，不要猜測或自行編造 solver 的建議
- 只有在有具體數據（頻率、EV、combo 數）時才引用這些數字

分析框架：
1. 手牌概況 — 一句話摘要場景和結果
2. 每條街逐一分析（僅限有 solver 數據的街）：
   a) *Solver 建議*：GTO 在這個 spot 會怎麼打？各動作的頻率和 combo 數是多少？
   b) *Hero 實際打法*：hero 做了什麼？
   c) *差異比較*：hero 的選擇跟 solver 建議差多少？是純策略還是混合策略的偏差？
   d) *原因解釋*：為什麼 solver 推薦這樣打？背後的邏輯是什麼？（例如：range 保護、榨取價值、平衡頻率、阻擋牌效果等）
3. 關鍵錯誤 — 找出 EV 損失最大的決策點，用具體數據說明
4. 改進建議 — 1-2 個可以立即應用到牌桌上的調整"""


class GeminiSessionManager:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 環境變數未設定")

        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.parse_model = os.getenv("GEMINI_PARSE_MODEL", "gemini-2.5-flash")
        self.max_turns = "N/A"  # not applicable, for bot.py compat
        self.histories: Dict[int, List[types.Content]] = {}

        # Logging
        _LOG_DIR.mkdir(exist_ok=True)
        self._logger = logging.getLogger("gemini_session")
        if not self._logger.handlers:
            handler = logging.FileHandler(_LOG_DIR / "gemini_session.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    async def send_message(self, chat_id: int, user_text: str) -> str:
        """Main entry: parse hand → GTO analysis → coaching, or general chat."""
        t0 = time.time()
        self._logger.info(f"[chat={chat_id}] User: {user_text[:300]}")

        try:
            # Step 1: Parse hand from user message (Flash — fast)
            hand_json = await self._parse_hand(chat_id, user_text)
            t_parse = time.time()

            if hand_json:
                self._logger.info(
                    f"[chat={chat_id}] Parsed hand in {t_parse - t0:.1f}s "
                    f"(model={self.parse_model}): "
                    f"{json.dumps(hand_json, ensure_ascii=False)[:300]}"
                )

                # Step 2: Run GTO analysis (direct Python call, no subprocess)
                from analyze_hand import analyze_hand
                gto_data = analyze_hand(hand_json)
                t_analyze = time.time()
                self._logger.info(
                    f"[chat={chat_id}] GTO analysis in {t_analyze - t_parse:.1f}s "
                    f"({len(gto_data)} chars)"
                )
                self._logger.debug(f"[chat={chat_id}] GTO data:\n{gto_data}")

                # Step 3: Coaching from LLM (Pro thinking — thorough)
                result = await self._coach(chat_id, user_text, gto_data)
                t_total = time.time()
                self._logger.info(
                    f"[chat={chat_id}] Done: parse={t_parse - t0:.1f}s "
                    f"gto={t_analyze - t_parse:.1f}s "
                    f"coach={t_total - t_analyze:.1f}s "
                    f"total={t_total - t0:.1f}s"
                )
                return result
            else:
                # Not a hand — general chat
                result = await self._chat(chat_id, user_text)
                elapsed = time.time() - t0
                self._logger.info(f"[chat={chat_id}] Chat response in {elapsed:.1f}s")
                return result

        except Exception as e:
            self._logger.error(f"[chat={chat_id}] Error: {e}", exc_info=True)
            raise

    async def _parse_hand(self, chat_id: int, user_text: str) -> dict | None:
        """Parse user's natural language into hand JSON. Uses Flash for speed."""
        prompt = f"{PARSE_PROMPT}\n\n用戶訊息：\n{user_text}"
        self._logger.debug(f"[chat={chat_id}] Parse prompt ({len(prompt)} chars):\n{prompt}")

        response = await self.client.aio.models.generate_content(
            model=self.parse_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )

        text = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Parse response:\n{text}")

        # Extract JSON from response (may be wrapped in ```json ... ```)
        json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        json_str = json_match.group(1) if json_match else text.strip()

        try:
            result = json.loads(json_str)
            hand = result.get("hand")
            if hand and hand.get("hero_position") and hand.get("preflop_actions"):
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

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=COACH_SYSTEM,
            ),
        )

        result = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Coach response ({len(result)} chars):\n{result}")

        # Update history (keep user's original text, not the coaching prompt)
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=result)]))
        self.histories[chat_id] = history[-20:]  # keep last 10 turns

        return result

    async def _chat(self, chat_id: int, user_text: str) -> str:
        """General chat for non-hand messages (follow-ups, questions)."""
        self._logger.debug(f"[chat={chat_id}] Chat prompt (model={self.model}): {user_text[:300]}")

        history = self.histories.get(chat_id, [])
        messages = list(history) + [
            types.Content(role="user", parts=[types.Part(text=user_text)]),
        ]

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=COACH_SYSTEM,
            ),
        )

        result = response.text or ""
        self._logger.debug(f"[chat={chat_id}] Chat response ({len(result)} chars):\n{result}")

        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=result)]))
        self.histories[chat_id] = history[-20:]

        return result

    def clear_session(self, chat_id: int) -> None:
        """Clear conversation history for a chat."""
        self.histories.pop(chat_id, None)
