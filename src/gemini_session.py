# src/gemini_session.py
"""Gemini-based session manager — direct API calls, no CLI subprocess.

Flow: user message → parse hand (Flash) → analyze_hand_full() → coaching (Pro)
Follow-ups: user message → parse (null) → Pro chat WITH query_gto tool → real data
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

# Allow importing from scripts/
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
- 當你需要額外的 solver 數據（例如對手範圍、假設情境、特定手牌策略），使用 query_gto 工具查詢

分析框架：
1. 手牌概況 — 一句話摘要場景和結果
2. 每條街逐一分析（僅限有 solver 數據的街）：
   a) *Solver 建議*：GTO 在這個 spot 會怎麼打？各動作的頻率和 combo 數是多少？
   b) *Hero 實際打法*：hero 做了什麼？
   c) *差異比較*：hero 的選擇跟 solver 建議差多少？是純策略還是混合策略的偏差？
   d) *原因解釋*：為什麼 solver 推薦這樣打？背後的邏輯是什麼？（例如：range 保護、榨取價值、平衡頻率、阻擋牌效果等）
3. 關鍵錯誤 — 找出 EV 損失最大的決策點，用具體數據說明
4. 改進建議 — 1-2 個可以立即應用到牌桌上的調整"""

# ── Gemini tool schema for GTO queries ──

QUERY_GTO_DECLARATION = types.FunctionDeclaration(
    name="query_gto",
    description=(
        "查詢 GTO solver 策略數據。可以查詢目前手牌中任何位置在任何街的完整範圍或特定手牌策略。"
        "也可以修改 board 或 actions 來查詢假設情境（例如 hero check 後的策略、不同的 turn 牌）。"
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

    async def send_message(self, chat_id: int, user_text: str) -> str:
        """Main entry: parse hand → GTO analysis → coaching, or chat with tools."""
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

                # Step 2: Run GTO analysis and cache context
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

                # Step 3: Coaching from LLM
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
                # Not a hand — chat (with tools if hand context exists)
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
        self.histories[chat_id] = history[-20:]

        return result

    async def _chat(self, chat_id: int, user_text: str) -> str:
        """Chat with optional GTO tool access for follow-up questions."""
        has_context = chat_id in self.hand_contexts

        if has_context:
            self._logger.debug(f"[chat={chat_id}] Chat WITH tools (model={self.model}): {user_text[:300]}")
            return await self._chat_with_tools(chat_id, user_text)

        self._logger.debug(f"[chat={chat_id}] Plain chat (model={self.model}): {user_text[:300]}")

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

    async def _chat_with_tools(self, chat_id: int, user_text: str) -> str:
        """Chat with query_gto tool for data-driven follow-up answers."""
        tool = types.Tool(function_declarations=[QUERY_GTO_DECLARATION])

        # Build system prompt with hand context
        hand_summary = self._build_hand_summary(chat_id)
        system = COACH_SYSTEM + "\n\n" + hand_summary

        history = self.histories.get(chat_id, [])
        messages = list(history) + [
            types.Content(role="user", parts=[types.Part(text=user_text)]),
        ]

        result_text = ""
        max_rounds = 5

        for round_num in range(max_rounds):
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    tools=[tool],
                ),
            )

            # Check for function calls in response
            candidate = response.candidates[0]
            function_calls = [
                p for p in candidate.content.parts
                if p.function_call
            ]

            if not function_calls:
                # No more tool calls — extract final text
                result_text = response.text or ""
                break

            # Execute tool calls and build response
            messages.append(candidate.content)

            for fc in function_calls:
                args = dict(fc.function_call.args) if fc.function_call.args else {}
                self._logger.info(
                    f"[chat={chat_id}] Tool call #{round_num+1}: "
                    f"query_gto({json.dumps(args, ensure_ascii=False)})"
                )

                t_tool = time.time()
                tool_result = self._execute_query_gto(chat_id, args)
                elapsed = time.time() - t_tool
                self._logger.debug(
                    f"[chat={chat_id}] Tool result ({elapsed:.1f}s, {len(tool_result)} chars):\n"
                    f"{tool_result[:500]}"
                )

                messages.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name="query_gto",
                        response={"data": tool_result},
                    )],
                ))

        self._logger.debug(f"[chat={chat_id}] Chat+tools response ({len(result_text)} chars):\n{result_text}")

        # Update history (user text only, not tool calls)
        history = self.histories.get(chat_id, [])
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        history.append(types.Content(role="model", parts=[types.Part(text=result_text)]))
        self.histories[chat_id] = history[-20:]

        return result_text

    def _execute_query_gto(self, chat_id: int, args: dict) -> str:
        """Execute a query_gto tool call. Returns formatted solver data."""
        from gto_api import get_spot_solution, get_next_actions, find_closest_action
        from gto_formatter import format_action_summary, format_hand_detail, format_range_overview

        ctx = self.hand_contexts.get(chat_id)
        if not ctx:
            return "錯誤：沒有手牌 context，請先發送手牌描述。"

        street = args.get("street", "flop")
        position = args.get("position")
        hand = args.get("hand")
        board_override = args.get("board_override")
        flop_override = args.get("flop_actions_override")
        turn_override = args.get("turn_actions_override")
        river_override = args.get("river_actions_override")

        has_override = any([board_override, flop_override, turn_override, river_override])

        # Try cached solution first (no overrides)
        if not has_override:
            solution = self._find_cached_solution(ctx, street)
            if solution:
                return self._format_solution(solution, position, hand)

        # Build API params from context + overrides
        params = self._build_query_params(ctx, street, board_override,
                                          flop_override, turn_override, river_override)
        if not params:
            return f"無法建構 {street} 的查詢參數。"

        # Normalize any raise codes in override actions
        params = self._normalize_override_actions(params, street, flop_override, turn_override, river_override)

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
                            river_override: str | None) -> dict | None:
        """Build API params for a query, using context + optional overrides."""
        states = ctx.get("street_states", {})
        base = states.get(street)

        if street == "preflop":
            return dict(
                gametype=ctx["gametype"],
                depth=ctx["depth"],
                preflop_actions=ctx["preflop_actions"],
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
                    preflop_actions=ctx["preflop_actions"],
                    board=board_override or flop_state["board"],
                    flop_actions=flop_override or flop_state["flop_actions"],
                    turn_actions=turn_override or "",
                    river_actions="",
                )
            return None

        return dict(
            gametype=ctx["gametype"],
            depth=ctx["depth"],
            preflop_actions=ctx["preflop_actions"],
            board=board_override or base["board"],
            flop_actions=flop_override if flop_override is not None else base["flop_actions"],
            turn_actions=turn_override if turn_override is not None else base["turn_actions"],
            river_actions=river_override if river_override is not None else base["river_actions"],
        )

    def _normalize_override_actions(self, params: dict, street: str,
                                     flop_override: str | None,
                                     turn_override: str | None,
                                     river_override: str | None) -> dict:
        """Normalize raise codes in overridden action strings."""
        from gto_api import get_next_actions, find_closest_action

        # Only normalize the overridden street's actions
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
                if code in ("X", "C", "F", "AI", "RAI", ""):
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
        from gto_formatter import format_action_summary, format_hand_detail, format_range_overview

        parts = [format_action_summary(solution)]

        if hand and position:
            parts.append("")
            parts.append(format_hand_detail(solution, hand, position))
        elif position:
            parts.append("")
            parts.append(format_range_overview(solution, position))
        elif hand:
            # Hand specified but no position — use active position
            active_pos = solution["game"]["active_position"]
            parts.append("")
            parts.append(format_hand_detail(solution, hand, active_pos))

        return "\n".join(parts)

    def _build_hand_summary(self, chat_id: int) -> str:
        """Build a concise hand summary for the system prompt."""
        ctx = self.hand_contexts.get(chat_id)
        if not ctx:
            return ""

        lines = [
            "目前分析的手牌：",
            f"- Hero: {ctx['hero_position']} {ctx['hero_hand']}, {ctx['depth'] - 0.125:.0f}bb depth",
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
            "你可以使用 query_gto 工具查詢任何位置在任何街的 solver 數據。"
            "也可以傳入 board_override 或 actions_override 查詢假設情境（例如不同的動作或不同的牌面）。"
        )

        return "\n".join(lines)

    def clear_session(self, chat_id: int) -> None:
        """Clear conversation history and hand context for a chat."""
        self.histories.pop(chat_id, None)
        self.hand_contexts.pop(chat_id, None)
