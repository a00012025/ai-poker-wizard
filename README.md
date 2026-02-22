# AI Poker Wizard

GTO 撲克教練 — 結合 GTO Wizard solver 數據與 Gemini LLM，提供即時手牌分析、策略教學與批次偏差比對。

**直接使用：** [t.me/ai_poker_wizard_bot](https://t.me/ai_poker_wizard_bot)

## 功能

### 手牌分析
- 用自然語言描述手牌，自動解析並查詢 GTO solver 策略
- 支援 Chip EV 與 ICM（MTT 錦標賽）模式
- 多街分析（preflop → flop → turn → river）
- 顯示各動作頻率、EV、以及 combo-level suit 差異

### 截圖辨識
- 上傳撲克回放截圖，自動辨識手牌資訊並分析 GTO 策略

### 手牌歷史批次分析
- 上傳 GGPoker 手牌歷史（.txt 或 .zip），批次比對 GTO 偏差
- 自動偵測 ICM 階段（bubble、final table 等）
- 按嚴重程度分類偏差報告
- 支援 follow-up：回覆 hand ID 可查看該手詳細分析

### AI 教練
- Gemini Pro 提供個人化教練回饋
- 多輪對話 — 可追問 range 細節、不同打法比較
- LLM 可即時查詢 solver 回答 follow-up 問題

### Chrome Extension
- 一鍵複製 GTO Wizard token，方便綁定帳號
- 原始碼在 `chrome-extension/` 目錄

## 使用教學

### 1. 綁定 GTO Wizard 帳號

使用前需要綁定你自己的 [GTO Wizard](https://www.gtowizard.com/) 帳號。

**方法一：Chrome Extension（推薦）**

1. 到 [Releases 頁面](https://github.com/a00012025/ai-poker-wizard/releases) 下載 `gto-wizard-ext-v1.0.zip`
2. 解壓縮後，Chrome 開啟 `chrome://extensions` → 開啟「開發人員模式」
3. 點「載入未封裝項目」→ 選擇解壓後的資料夾
4. 登入 [app.gtowizard.com](https://app.gtowizard.com) → 點擊 extension 圖示 → 自動複製指令
5. 回到 Telegram 貼上即可

**方法二：手動取得**

1. 登入 [app.gtowizard.com](https://app.gtowizard.com)
2. F12 開啟 Console
3. 貼上：`copy(localStorage.getItem('user_refresh'))`
4. 回到 Telegram 輸入：`/settoken <貼上>`

### 2. 分析手牌

直接傳送手牌描述，例如：

```
Hero 42bb effective, UTG+1 raise 2bb, hero SB all-in A9s
```

```
有效 50bb, CO open 2bb, hero SB AcTh raise 7.5bb, CO call
flop KsKhQd, SB bet 1/4 CO call
turn 3h, SB bet 60% CO fold
打得合理嗎？
```

也可以直接上傳撲克回放截圖。

### 3. 批次分析手牌歷史

上傳 GGPoker 匯出的 .txt 或 .zip 檔案，自動批次比對 GTO 偏差。

可在 caption 加上起始籌碼與錦標賽大小，例如 `10000 200`（起始 10000 chips，200 人錦標賽）。

分析完成後，回覆 hand ID（如 `TM5600279272`）可查看該手詳細 GTO 分析。

### Bot 指令

| 指令 | 說明 |
|------|------|
| `/start` | 顯示歡迎訊息與設定教學 |
| `/settoken` | 綁定 GTO Wizard token |
| `/logout` | 解除綁定 |
| `/clear` | 清除對話紀錄 |

## 自行部署

### 環境需求
- Python 3.11+
- Docker & Docker Compose
- Supabase 資料庫（用於儲存使用者 token 與手牌歷史）

### 環境變數（`.env`）

| 變數 | 說明 |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `BOT_TOKEN` | Telegram bot token |
| `SUPABASE_CONN` | Supabase 連線字串（transaction pooler） |
| `ADMIN_CHAT_ID` | 管理員 Telegram chat ID |

### 安裝與啟動

```bash
pip install -r requirements.txt

# 本地測試
python -m src.main_gemini

# Docker 部署
docker compose up -d
```

### 測試

```bash
# 回歸測試（76 tests）
python scripts/regression_test.py

# E2E 測試（不需 Telegram）
python scripts/e2e_test.py "Hero 20bb, UTG raise 2bb, hero BB all-in ATs"

# 互動模式（多輪對話）
python scripts/e2e_test.py -i "有效 50bb, CO open 2bb, hero SB AcTh raise 7.5bb ..."
```

## 架構

```
User (Telegram)
  ↓
PokerWizardBot
  ├── Token gate（檢查使用者是否已綁定 GTO Wizard）
  ↓
GeminiSessionManager
  ├── Parse hand (Gemini Flash)
  ├── GTO analysis (analyze_hand.py → GTO Wizard API)
  │     ├── gto_api.py — API client
  │     ├── gto_formatter.py — solver 數據 → 自然語言
  │     └── icm_modes.py — ICM 階段 & stack 配置
  └── Coaching (Gemini Pro + tool use for follow-ups)
```

## License

MIT
