# src/main_gemini.py
"""Entry point using Gemini API (fast, no Claude CLI subprocess)."""
import os

from dotenv import load_dotenv
load_dotenv()

from src.gemini_session import GeminiSessionManager
from src.telegram_bot.bot import PokerWizardBot


def main():
    print("🃏 AI Poker Wizard (Gemini) 正在啟動...")

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        print("❌ 錯誤：未設置 BOT_TOKEN 環境變數")
        return

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ 錯誤：未設置 GEMINI_API_KEY 環境變數")
        return

    session_manager = GeminiSessionManager()
    bot = PokerWizardBot(token=bot_token, session_manager=session_manager)

    print(f"✅ 系統就緒！模型：{session_manager.model}")
    print("🚀 正在啟動 Telegram Bot...")
    bot.run()


if __name__ == "__main__":
    main()
