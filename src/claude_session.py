# src/claude_session.py
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict

# Resolve project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_LOG_DIR = _PROJECT_ROOT / "logs"

SYSTEM_PROMPT = f"""\
你是 AI Poker Wizard — 一位專業的 MTT 撲克錦標賽教練。

當用戶描述手牌場景時，你**必須**用 Bash 執行 Python 腳本從 GTO Wizard API 抓取真實求解器數據，不要憑記憶猜測。

回覆規則：
- 一律使用繁體中文
- 提供具體數據和推理過程，不要泛泛而談
- 如果資訊不足，主動追問關鍵細節（例如：錦標賽階段、對手傾向、籌碼結構）
- **無論如何都必須回覆文字結果給用戶**，即使中間步驟失敗也要說明遇到了什麼問題

GTO 數據抓取：
使用 `python {_SCRIPTS_DIR}/analyze_hand.py --json '<json>'` 執行完整分析。

Hand JSON 格式：
{{
    "gametype": "MTTGeneral",
    "effective_bb": 32,
    "hero_position": "CO",
    "hero_hand": "66",
    "preflop_actions": "F-F-F-R2.1-F-F-F-C",
    "streets": [
        {{"board": "Js6h5s", "actions": [{{"position": "BB", "action": "X"}}, {{"position": "CO", "action": "R2", "size": 2.0}}]}},
        {{"card": "Kc", "actions": [{{"position": "BB", "action": "X"}}, {{"position": "CO", "action": "R6.6", "size": 6.6}}]}},
        {{"card": "2s", "actions": [{{"position": "BB", "action": "X"}}, {{"position": "CO", "action": "X"}}]}}
    ]
}}

Preflop action 編碼：
- MTT 8-max 位置: UTG, UTG+1, LJ, HJ, CO, BTN, SB, BB
- F=Fold, C=Call, RX=Raise to X（如 R2.1）, AI=All-in
- 範例：CO open, BB call → F-F-F-F-R2.1-F-F-C（前4個F=UTG到HJ fold, R2.1=CO raise, F=BTN fold, F=SB fold, C=BB call）

Board 記法：Js6h5s（rank+suit: c/d/h/s）

Action 記法：X=Check, C=Call, F=Fold, R+size=Bet/Raise（如 R2 表示 bet 2bb）

腳本輸出包含：每條街的 action summary（各動作頻率和 combos）以及 hero 手牌的具體策略。
基於這些數據提供教練分析：指出 hero 打法和 GTO 的差異，解釋為什麼 solver 推薦不同的行動。

也可以單獨使用個別腳本：
- `python {_SCRIPTS_DIR}/gto_api.py` — API 客戶端
- `python {_SCRIPTS_DIR}/gto_token.py` — Token 管理（輸出 access token）

工作目錄：{_PROJECT_ROOT}
"""


class ClaudeSessionManager:
    def __init__(self):
        self.model = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
        self.timeout = int(os.getenv("CLAUDE_TIMEOUT", "600"))
        self.max_turns = int(os.getenv("CLAUDE_MAX_TURNS", "100"))
        self.verbose = os.getenv("DEBUG", "").lower() == "true"
        # chat_id -> claude session_id
        self.sessions: Dict[int, str] = {}
        # Per-chat lock to serialize messages within the same session
        self._locks: Dict[int, asyncio.Lock] = {}
        # Setup logging
        _LOG_DIR.mkdir(exist_ok=True)
        self._logger = logging.getLogger("claude_session")
        if not self._logger.handlers:
            handler = logging.FileHandler(_LOG_DIR / "claude_session.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    def _get_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self._locks:
            self._locks[chat_id] = asyncio.Lock()
        return self._locks[chat_id]

    async def send_message(self, chat_id: int, user_text: str) -> str:
        """Send a message via claude CLI, maintaining per-chat sessions.

        Messages from the same chat are serialized via a lock to prevent
        concurrent --resume calls on the same session.
        """
        async with self._get_lock(chat_id):
            return await self._send(chat_id, user_text)

    async def _send(self, chat_id: int, user_text: str) -> str:
        session_id = self.sessions.get(chat_id)
        self._logger.info(f"[chat={chat_id}] User: {user_text[:200]}")

        # Always use stream-json + verbose to capture intermediate steps for logging
        cmd = [
            "claude", "-p",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--allowed-tools", "Bash",
            "--dangerously-skip-permissions",
        ]

        if session_id:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--system-prompt", SYSTEM_PROMPT])

        cmd.append(user_text)

        # Clear CLAUDECODE env to avoid nested session check
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        # In verbose mode, let stderr flow to terminal for live output
        stderr_target = None if self.verbose else asyncio.subprocess.PIPE

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr_target,
            cwd=str(_PROJECT_ROOT),
            env=env,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            self._logger.error(f"[chat={chat_id}] Timeout after {self.timeout}s")
            raise TimeoutError(f"Claude 回應超時（{self.timeout}s）")

        raw_out = (stdout or b"").decode()
        raw_err = (stderr or b"").decode().strip()

        # Parse stream-json output, log intermediate steps, extract result
        output = self._parse_output(raw_out, chat_id)

        if output.get("result"):
            if output.get("is_error"):
                self._logger.error(f"[chat={chat_id}] Claude error: {output['result'][:500]}")
                raise RuntimeError(f"Claude 錯誤：{output['result']}")
            # Store session for future messages
            if not session_id and output.get("session_id"):
                self.sessions[chat_id] = output["session_id"]
            self._logger.info(f"[chat={chat_id}] Result: {output['result'][:200]}...")
            return output["result"]

        # Fallback: extract last assistant text from stream
        if output.get("_last_text"):
            self._logger.warning(f"[chat={chat_id}] No result message, using last assistant text as fallback")
            if not session_id and output.get("session_id"):
                self.sessions[chat_id] = output["session_id"]
            return output["_last_text"]

        # No result found — handle as error
        if proc.returncode != 0:
            combined = raw_err or raw_out.strip() or f"exit code {proc.returncode}"
            self._logger.error(f"[chat={chat_id}] Exit {proc.returncode}: {combined[-500:]}")
            # Session might have expired — retry as new session
            if session_id and ("not found" in combined.lower() or "invalid" in combined.lower()):
                self.sessions.pop(chat_id, None)
                return await self._send(chat_id, user_text)
            raise RuntimeError(f"Claude 錯誤（exit {proc.returncode}）：{combined[-1000:]}")

        self._logger.error(f"[chat={chat_id}] Empty result. raw_out length={len(raw_out)}, returncode={proc.returncode}")
        raise RuntimeError("Claude 回傳空結果")

    def _parse_output(self, raw: str, chat_id: int = 0) -> dict:
        """Parse stream-json output, log intermediate steps, extract result."""
        lines = raw.strip().splitlines()
        if not lines:
            return {}

        result = {}
        last_text = ""
        session_id = None

        for line in lines:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            # Skip hook/system messages
            if msg_type == "system":
                if msg.get("subtype") == "init" and msg.get("session_id"):
                    session_id = msg["session_id"]
                continue

            # Result message — the final answer
            if msg_type == "result":
                result = msg
                if session_id and "session_id" not in result:
                    result["session_id"] = session_id
                continue

            # Non-stream json format fallback
            if "result" in msg and "session_id" in msg:
                result = msg
                continue

            # Assistant message — log tool calls and text
            if msg_type == "assistant":
                content = msg.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_name = block.get("name", "")
                        tool_input = block.get("input", {})
                        cmd_preview = tool_input.get("command", "")[:200]
                        self._logger.debug(f"[chat={chat_id}] Tool: {tool_name} → {cmd_preview}")
                        if self.verbose:
                            print(f"  🔧 Tool: {tool_name} → {cmd_preview[:120]}")
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            last_text = text
                            self._logger.debug(f"[chat={chat_id}] Text: {text[:300]}")
                            if self.verbose:
                                print(f"  💬 Text: {text[:100]}...")

            # Tool result — log output
            elif msg_type == "tool":
                content = msg.get("content", "")
                if isinstance(content, str):
                    preview = content[:300]
                elif isinstance(content, list):
                    preview = str(content)[:300]
                else:
                    preview = str(content)[:300]
                self._logger.debug(f"[chat={chat_id}] ToolResult: {preview}")

        if last_text and not result.get("result"):
            result["_last_text"] = last_text
        if session_id and "session_id" not in result:
            result["session_id"] = session_id
        return result

    def clear_session(self, chat_id: int) -> None:
        """Clear the session for a chat (next message starts fresh)."""
        self.sessions.pop(chat_id, None)
        self._locks.pop(chat_id, None)
