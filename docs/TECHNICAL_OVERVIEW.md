# AI Poker Wizard 技術總覽

本文件承接 README 不適合展開的架構、部署與驗證細節。產品方向與不變量以 [`NORTH_STAR.md`](NORTH_STAR.md) 為準；實際操作驗收以各 runbook 為準。

## 系統路徑

### 即時分析

```text
Telegram message / image / HH file
        ↓
PokerWizardBot
        ↓
GeminiSessionManager
  ├── parse：Gemini Flash + deterministic validation
  ├── grade：analyze_hand.py → GTO Wizard API
  │     ├── gto_api.py
  │     ├── gto_formatter.py
  │     ├── hand_validator.py
  │     └── icm_modes.py
  └── coach
        ├── initial：deterministic 教學骨架 → GPT-5.6 Terra narrator → fact audit/fallback
        └── follow-up：Gemini Pro + grounded solver/ledger tools
```

### GTOW session 與 Ledger 攝取

```text
Chrome Extension page/content/background
        ↓ HTTPS + revocable device credential
Supabase Edge Function: gtow-sync
  ├── /token  → users.gto_* session bundle
  └── /ingest → gtow_ingest_requests
                    ↓ 5 秒 poller / single-flight
              ledger_ingest.py
                    ↓
       raw archive + ledger_hands/decisions
                    ↓
       backfill_spots.py + ledger_sessions.py + verify
```

Extension、Telegram `/ingest` 與每日排程都進同一個 request queue。Runner child process 只取得 `GTOW_USER_ID`，再由 `gto_credentials.py` 從 DB 解析該使用者的 session bundle。

### 訓練迴圈

```text
ledger_hands / ledger_decisions
        ↓
spot_taxonomy + session_review + spot_leaderboard
        ↓
queue_feed / scorecard / plan_scheduler
        ↓
GTOW Trainer URL + Drill API binding
        ↓
practice sessions / totals readback
```

## 主要模組

| 路徑 | 職責 |
|---|---|
| `src/gemini_session.py` | parse → solver tools → grounded coaching |
| `src/telegram_bot/bot.py` | Telegram handlers 與 live/review/queue/plan UI |
| `src/ingest_runner.py` | ingest request poller、進度、single-flight pipeline |
| `scripts/analyze_hand.py` | 多街 solver 分析 orchestration |
| `scripts/gto_api.py` | GTO Wizard next-actions／spot-solution client |
| `scripts/gto_formatter.py` | solver JSON → 自然語言與 combo breakdown |
| `scripts/gto_credentials.py` | browser-first per-user GTOW session provider |
| `scripts/ledger_ingest.py` | GTOW Analyze archive 與 Decision Ledger 攝取 |
| `scripts/ledger_distill.py` | raw detail → grader-agnostic decision rows |
| `scripts/spot_taxonomy.py` | action-line spot taxonomy |
| `scripts/ledger_sessions.py` | online session reconstruction |
| `scripts/session_review.py` | session 級 EV 加權復盤 |
| `scripts/queue_feed.py` | drill／review queue scan 與來源追蹤 |
| `scripts/spot_leaderboard.py` | EV-weighted family／leaf ranking 與 stack scope |
| `scripts/scorecard.py` | 每週焦點、課表與 GTOW drill 處方 |
| `scripts/plan_scheduler.py` | 處方 freshness 與完成判定 |
| `scripts/live_flow.py` | 線下 shorthand parse、repair、grade、ledger/queue |
| `scripts/gtow_trainer_url.py` | Trainer deep-link builder |
| `scripts/gtow_drill_service.py` | GTOW Drill provisioning 與成績 readback |
| `chrome-extension/` | GTOW session 捕捉、device pairing、一鍵 ingest |

## 資料與統計邊界

- `ledger_hands.source='online'`：GTOW Analyze 全量資料，可進聚合統計。
- `ledger_hands.source='live'`：使用者選擇性記錄，只能進線下復盤與 queue。
- 統計排除 `excluded`、`discarded`、低 confidence 與無法誠實評分的 decisions。
- Taxonomy 以 `spot_taxonomy.py` 的 action-line `spot_leaf`／`spot_category` 為準。
- 排序以 avg/total EV loss 為準；frequency difference 只可做解釋，不可當嚴重度。
- `played_depth_bb` 保存牌桌/list stack；`solver_depth_bb` 保存實際評分 game point。

## GTOW credential model

GTOW session bundle 只保存於 `users.gto_*` 欄位：

- `gto_access_token`
- `gto_access_token_iat` / `gto_access_token_exp`
- `gto_refresh_token`
- `gto_client_id`
- `gto_session_observed_at`
- `gto_access_token_source`
- `gto_backend_signing_keypair`

Browser access token 在實際 JWT expiry 前是唯一來源。只有 token 過期後，後端才可在 per-user PostgreSQL advisory lock 下，以持久化 keypair refresh。未過期 token 收到 401 時不得自行 refresh。

原始 access、refresh 與 keypair 不寫入 Chrome storage。Owner CLI／regression 透過 `scripts/gto_owner_token.py` 從 DB bootstrap，不使用 repo 內 token file。

## 排程

`src/main_gemini.py` 設定以下 Asia/Taipei jobs：

- 每 5 秒：poll `gtow_ingest_requests`。
- 每日 05:00：為 owner 排入 incremental ingest。
- 每週日 21:00：產生並推播 weekly scorecard。

Ingest pipeline：

```text
incremental/full ingest
  → taxonomy backfill
  → session rebuild
  → verify
  → mismatch 時自動 full sweep
```

## 自行部署

### 環境需求

- Python 3.12
- Docker 與 Docker Compose
- NVIDIA Container Runtime（目前 production compose 的 OCR／CV 配置使用 GPU）
- Supabase project 與 CLI
- Telegram Bot token
- Gemini API key
- 可使用 solver/API 的 GTO Wizard 帳號
- Node.js（Extension 開發與測試）

### 主要環境變數

複製 `.env.example` 為 `.env`：

| 變數 | 說明 |
|---|---|
| `BOT_TOKEN` | Telegram bot token |
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_MODEL` / `GEMINI_PARSE_MODEL` | coaching 與 hand parse 模型 |
| `GEMINI_LIVE_PARSE_MODEL` | `/live` lexical parser 模型 |
| `OPENAI_API_KEY` | 初始 grounded-coach narrator；未設定時安全退回 Gemini |
| `COACH_NARRATOR_PROVIDER` / `OPENAI_COACH_MODEL` | 初始教練 provider 與低成本 narrator 模型（預設有 key 時為 OpenAI / `gpt-5.6-terra`） |
| `SUPABASE_CONN` | PostgreSQL transaction-pooler DSN |
| `SUPABASE_ACCESS_TOKEN` | Supabase CLI deploy token |
| `SUPABASE_PROJECT_REF` | Supabase project ref |
| `ADMIN_CHAT_ID` | Owner/admin Telegram user ID |
| `OWNER_CHAT_ID` | 可選；明確指定 Ledger/scorecard owner |
| `GTOW_SYNC_PEPPER` | Bot 與 Edge Function 共用的 device HMAC secret，至少 32 字元 |
| `OCR_ENABLED` | 是否預載 OCR／CardCNN |

### 本地啟動

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m src.main_gemini
```

未設定 `SUPABASE_CONN` 時 Bot 可啟動，但 pairing、Ledger、ingest、queue 與排程不完整可用。

### 資料庫與 production

所有 schema 變更走 `supabase/migrations/`，使用 `supabase db push`，不要直接 raw `psql` 修改 production schema。

```bash
supabase db push
bash scripts/deploy.sh
```

`scripts/deploy.sh` 依序處理 pull、必要的 cache export、migration、taxonomy backfill、Edge Function secret/deploy、container rebuild。

Token sync 首次部署與 rollback 見 [`GTOW_TOKEN_SYNC_DEPLOY.md`](GTOW_TOKEN_SYNC_DEPLOY.md)。

## 測試與驗證

```bash
# 核心 solver / formatter / ICM / coaching / ledger regression suite
python scripts/regression_test.py

# pytest：OCR、parser、token sync 等
pytest -q

# Chrome Extension
node --check chrome-extension/background.js
node --check chrome-extension/content.js
node --check chrome-extension/page_session.js
node --check chrome-extension/popup.js
node --test chrome-extension/tests/*.test.js
bash scripts/package_extension.sh

# 需要有效 .env 與 GTOW session 的 E2E
python scripts/e2e_test.py "Hero 20bb, UTG raise 2bb, hero BB all-in ATs"
python scripts/e2e_test.py -i "有效 50bb, CO open 2bb, hero SB AcTh raise 7.5bb"
```

完整測試要求、snapshot workflow 與 worktree／PR 規範見 [`../AGENTS.md`](../AGENTS.md)。

部分 regression fixtures 與本地 solve cache 不進 Git；在新 worktree 執行完整 suite 前，需依專案開發規範連結必要的本地資料目錄。

## Runbooks

- [`phase1-loop-runbook.md`](phase1-loop-runbook.md) — Ledger 與週訓練迴圈
- [`analysis-fidelity-runbook.md`](analysis-fidelity-runbook.md) — GTOW Analyzer／自建 grader fidelity
- [`GTOW_TOKEN_SYNC_DEPLOY.md`](GTOW_TOKEN_SYNC_DEPLOY.md) — Extension／Edge Function 部署
- [`../chrome-extension/README.md`](../chrome-extension/README.md) — Extension 安裝、安全與測試
