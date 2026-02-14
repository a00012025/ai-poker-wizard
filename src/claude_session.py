# src/claude_session.py
import asyncio
import json
import os
from typing import Dict

SYSTEM_PROMPT = """你是 AI Poker Wizard — 一位專業的 MTT 撲克錦標賽教練。

你的職責：
1. 解析玩家描述的手牌場景
2. 基於 GTO 理論提供精確的策略分析
3. 考慮 ICM 和錦標賽特殊因素
4. 用中文提供專業、實用的教練建議

分析框架：
- 手牌概況：場景摘要、位置、籌碼深度
- GTO 策略：基於求解器原理的頻率分析
- 範圍分析：對手範圍推測和 equity 計算
- ICM 考量：錦標賽 chip EV vs $ EV
- 改進建議：具體可執行的策略建議

回覆規則：
- 一律使用繁體中文
- 使用 Markdown 格式方便閱讀
- 提供具體數據和推理過程，不要泛泛而談
- 如果資訊不足，主動追問關鍵細節（例如：錦標賽階段、對手傾向、籌碼結構）"""


class ClaudeSessionManager:
    def __init__(self):
        self.model = os.getenv("CLAUDE_MODEL", "sonnet")
        self.timeout = int(os.getenv("CLAUDE_TIMEOUT", "120"))
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
            "--output-format", "json",
            "--model", self.model,
            "--allowed-tools", "Bash(python*)",
            "--dangerously-skip-permissions",
        ]

        if session_id:
            cmd.extend(["--resume", session_id])
        else:
            cmd.extend(["--system-prompt", SYSTEM_PROMPT])

        cmd.append(user_text)

        # Clear CLAUDECODE env to avoid nested session check
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
            err = stderr.decode().strip()
            # Session might have expired — retry as new session
            if session_id and ("not found" in err.lower() or "invalid" in err.lower()):
                self.sessions.pop(chat_id, None)
                return await self._send(chat_id, user_text)
            raise RuntimeError(f"Claude 錯誤：{err}")

        output = json.loads(stdout.decode())

        if output.get("is_error"):
            raise RuntimeError(f"Claude 錯誤：{output.get('result', 'unknown')}")

        # Store session for future messages
        if not session_id and output.get("session_id"):
            self.sessions[chat_id] = output["session_id"]

        return output["result"]

    def clear_session(self, chat_id: int) -> None:
        """Clear the session for a chat (next message starts fresh)."""
        self.sessions.pop(chat_id, None)
        self._locks.pop(chat_id, None)
