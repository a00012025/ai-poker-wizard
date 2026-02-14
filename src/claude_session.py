# src/claude_session.py
import asyncio
import json
import os
from pathlib import Path
from typing import Dict

# Resolve project root (parent of src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKILL_PATH = _PROJECT_ROOT / ".claude" / "skills" / "ai-poker-wizard" / "SKILL.md"
_SCRIPTS_PATH = _PROJECT_ROOT / "scripts" / "gto-wizard-extract.js"

SYSTEM_PROMPT = f"""\
你是 AI Poker Wizard — 一位專業的 MTT 撲克錦標賽教練。

當用戶描述手牌場景時，你**必須**使用 agent-browser 從 GTO Wizard 抓取真實求解器數據，不要憑記憶猜測。

回覆規則：
- 一律使用繁體中文
- 提供具體數據和推理過程，不要泛泛而談
- 如果資訊不足，主動追問關鍵細節（例如：錦標賽階段、對手傾向、籌碼結構）
- 寫 JS 給 agent-browser eval 時，先寫到 /tmp/ 檔案再用 cat 讀取執行，避免引號問題
- Postflop 只用 JS click 導航，不要用 URL 參數（flop_actions= 等），錯的參數會靜默回退到 preflop

收到第一個問題時，先用 Bash 執行 cat {_SKILL_PATH} 讀取完整的 GTO Wizard 自動化指南，然後按照指南操作。
JS 腳本路徑：{_SCRIPTS_PATH}
工作目錄：{_PROJECT_ROOT}
"""


class ClaudeSessionManager:
    def __init__(self):
        self.model = os.getenv("CLAUDE_MODEL", "sonnet")
        self.timeout = int(os.getenv("CLAUDE_TIMEOUT", "600"))
        self.max_turns = int(os.getenv("CLAUDE_MAX_TURNS", "30"))
        self.verbose = os.getenv("DEBUG", "").lower() == "true"
        # chat_id -> claude session_id
        self.sessions: Dict[int, str] = {}
        # Per-chat lock to serialize messages within the same session
        self._locks: Dict[int, asyncio.Lock] = {}

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

        cmd = [
            "claude", "-p",
            "--output-format", "stream-json" if self.verbose else "json",
            "--model", self.model,
            "--max-turns", str(self.max_turns),
            "--allowed-tools", "Bash",
            "--dangerously-skip-permissions",
        ]
        if self.verbose:
            cmd.append("--verbose")

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
            raise TimeoutError(f"Claude 回應超時（{self.timeout}s）")

        if proc.returncode != 0:
            err = (stderr or b"").decode().strip()
            out = (stdout or b"").decode().strip()
            combined = err or out or f"exit code {proc.returncode}"
            # Session might have expired — retry as new session
            if session_id and ("not found" in combined.lower() or "invalid" in combined.lower()):
                self.sessions.pop(chat_id, None)
                return await self._send(chat_id, user_text)
            raise RuntimeError(f"Claude 錯誤：{combined[:500]}")

        output = self._parse_output(stdout.decode())

        if output.get("is_error"):
            raise RuntimeError(f"Claude 錯誤：{output.get('result', 'unknown')}")

        # Store session for future messages
        if not session_id and output.get("session_id"):
            self.sessions[chat_id] = output["session_id"]

        return output["result"]

    def _parse_output(self, raw: str) -> dict:
        """Parse output — handles hook messages mixed into stdout."""
        lines = raw.strip().splitlines()

        # Try single JSON first (ideal case: no hooks, --output-format json)
        if len(lines) == 1:
            return json.loads(lines[0])

        # Multiple lines: hooks or stream-json mixed in. Find the result.
        result = {}
        for line in lines:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type", "")
            # Skip hook/system messages
            if msg_type == "system":
                continue
            if msg_type == "result":
                result = msg
            elif not self.verbose and "result" in msg and "session_id" in msg:
                # Non-verbose json format: the actual result object
                result = msg
            elif self.verbose and msg_type == "assistant":
                content = msg.get("message", {}).get("content", [])
                for block in content:
                    if block.get("type") == "tool_use":
                        print(f"  🔧 Tool: {block.get('name')} → {block.get('input', {}).get('command', '')[:120]}")
                    elif block.get("type") == "text":
                        preview = block.get("text", "")[:100]
                        print(f"  💬 Text: {preview}...")
        return result

    def clear_session(self, chat_id: int) -> None:
        """Clear the session for a chat (next message starts fresh)."""
        self.sessions.pop(chat_id, None)
        self._locks.pop(chat_id, None)
