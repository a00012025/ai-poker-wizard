# src/telegram_bot/bot.py
import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

class PokerWizardBot:
    def __init__(self, token: str = None):
        self.token = token or os.getenv('BOT_TOKEN')
        self.application = None

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_msg = """🃏 **歡迎使用 AI Poker Wizard！**

我是你的專業撲克教練，可以幫你：
• 分析手牌歷史
• 提供 GTO 策略建議
• 復盤錦標賽手牌
• 改進決策技巧

📝 **使用方法：**
直接發送手牌描述，例如：
"Hero 42bb effective, UTG+1 raise 2bb, hero SB all-in A9s"

📁 **或上傳 Natural8 檔案** 並告訴我要分析哪一手牌

讓我們開始提升你的撲克技巧！💪"""

        await update.message.reply_text(welcome_msg, parse_mode='Markdown')

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_msg = """🆘 **使用說明**

**指令：**
/start - 開始使用
/help - 顯示說明

**手牌分析：**
直接描述你的手牌情況，我會：
1. 解析手牌資訊
2. 查詢 GTO Wizard 策略
3. 提供專業分析和建議

**範例格式：**
```
Hero 17bb effective
UTG raise 2bb, BTN call, Hero SB all-in A9s
UTG fold, BTN call
```

有問題隨時問我！"""

        await update.message.reply_text(help_msg, parse_mode='Markdown')

    async def handle_hand_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle hand analysis requests"""
        try:
            hand_text = update.message.text

            # Show typing indicator
            await update.message.chat.send_action(action="typing")

            # For now, simple response - will integrate with wizard_core
            response = f"🔍 **正在分析手牌...**\n\n收到描述：\n```\n{hand_text[:200]}...\n```\n\n⏳ 正在查詢 GTO Wizard 數據並生成分析報告..."

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            error_msg = f"❌ 分析時發生錯誤：{str(e)}\n\n請檢查手牌描述格式，或稍後再試。"
            await update.message.reply_text(error_msg)

    def setup_handlers(self):
        """Setup bot handlers"""
        if not self.application:
            self.application = Application.builder().token(self.token).build()

        # Add handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_hand_analysis))

        return self.application

    async def run(self):
        """Run the bot"""
        app = self.setup_handlers()
        print("🚀 AI Poker Wizard bot starting...")
        await app.run_polling(drop_pending_updates=True)