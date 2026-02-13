# src/main.py
import asyncio
import os
from src.telegram_bot.bot import PokerWizardBot
from src.wizard_core import PokerWizardCore

async def main():
    """Main entry point for AI Poker Wizard"""
    print("🃏 AI Poker Wizard 正在啟動...")

    # Check bot token
    bot_token = os.getenv('BOT_TOKEN')
    if not bot_token:
        print("❌ 錯誤：未設置 BOT_TOKEN 環境變數")
        print("請先設置你的 Telegram Bot Token：")
        print("export BOT_TOKEN=你的bot_token")
        return

    # Initialize core
    wizard_core = PokerWizardCore()

    # Initialize and run bot
    bot = PokerWizardBot(token=bot_token)

    # TODO: Integrate wizard_core with bot handlers

    print("✅ 系統就緒！正在啟動 Telegram Bot...")
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())