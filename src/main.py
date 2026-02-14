# src/main.py
import os
from src.claude_session import ClaudeSessionManager
from src.telegram_bot.bot import PokerWizardBot


def main():
    """Main entry point for AI Poker Wizard"""
    print("🃏 AI Poker Wizard 正在啟動...")

    bot_token = os.getenv('BOT_TOKEN')
    if not bot_token:
        print("❌ 錯誤：未設置 BOT_TOKEN 環境變數")
        print("export BOT_TOKEN=你的bot_token")
        return

    session_manager = ClaudeSessionManager()
    bot = PokerWizardBot(token=bot_token, session_manager=session_manager)

    print(f"✅ 系統就緒！模型：{session_manager.model}")
    print("🚀 正在啟動 Telegram Bot...")
    bot.run()


if __name__ == "__main__":
    main()
