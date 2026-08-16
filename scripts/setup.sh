#!/bin/bash
set -e

echo "🃏 設置 AI Poker Wizard..."

# 創建虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements-dev.txt

# 複製環境模板
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ 已創建 .env 檔案，請編輯並填入你的 tokens 和憑證"
fi

# 創建數據目錄
mkdir -p data/cache data/logs

echo "🎯 設置完成！接下來的步驟："
echo "1. 編輯 .env 檔案，填入你的 Telegram bot token"
echo "2. 執行：source venv/bin/activate"
echo "3. 執行：python -m src.main_gemini"
