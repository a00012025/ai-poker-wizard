# AI Poker Wizard

AI Poker Wizard 是疊加在 [GTO Wizard](https://www.gtowizard.com/) 之上的個人撲克教練系統。

GTO Wizard 是健身房與器材：提供 solver、Analyzer、Trainer 與題庫；AI Poker Wizard 是教練層：把真實牌局變成回饋、訓練、復盤與評估的循環，主動決定接下來最值得改善的地方。

**Telegram Bot：** [t.me/ai_poker_wizard_bot](https://t.me/ai_poker_wizard_bot)

## 理念

### 北極星不是 ROI，而是決策品質

MTT 短期結果的雜訊太大，因此系統用以下指標導航：

> **真實對局的 EV loss / 100 決策**
>
> EV 加權、信心過濾、按 spot family 分解，並持續往 0 推。

訓練分數、答題數與連勝都只是中間訊號。真正的「學會」是：練過的 spot family 回到真實牌局後，EV loss 可歸因地下降。

### 單手是入口，spot family 才是學習單位

一手牌可以指出問題，但不能建立穩定能力。系統會把相同行動線聚合起來，找出反覆付出最多 EV 的 family，再把具體錯誤升級成可反覆訓練的課題。

### 不是另一個 solver，也不是解說員

- GTOW 已有的 solver、Analyzer、Trainer 能力直接 reuse，不重造通用工具。
- AI 回答策略問題前會查詢 solver 或 Ledger，不用模糊的撲克常識代替事實。
- 訓練強調先作答、再看回饋，而不是被動閱讀更多分析。
- 系統只處理賽後分析與訓練，永不提供對局中的策略輸入。

完整產品憲法見 [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md)。

## 現在能做到什麼

### 1. 分析一手牌

- 直接用中文或英文描述手牌，取得 grounded GTO 分析。
- 上傳撲克回放截圖，自動辨識牌面、位置、籌碼與行動。
- 支援 preflop 到 river、多輪追問、range 與 combo-level 差異。
- 支援 MTT Chip EV、ICM preflop，以及現金桌常見配置。
- 解析結果會經過撲克規則驗證，避免不合法 action 靜默進入 solver。

### 2. 批次檢查 GGPoker 手牌

- 上傳 GGPoker `.txt` 或 `.zip`，逐 decision 比對 GTO 偏差。
- 回覆 hand ID 可回到單手詳細分析。
- 以 EV loss 呈現錯誤影響，不把單純 frequency 差異誇大成重大失誤。

### 3. 把 GTOW Analyze 變成 Decision Ledger

- Chrome Extension 一鍵同步新上傳的 Analyze hands。
- 保存每個 decision 的 EV loss、spot、stack depth、信心、近似標記與資料來源。
- 自動重建 sessions，完成後可立即復盤最近一場，而不是等到週報。
- 無效、低信心或無法誠實還原的 decision 會被隔離，不污染正式統計。

### 4. 從漏洞產生訓練處方

- `/review`：復盤最近 online session 最昂貴的決策。
- `/sessions`：查看近期 online sessions，點選後重新產生該場復盤。
- `/queue`：管理待練 drill 與待看的重大牌局，並可追溯到來源牌局。
- `/plan`：查看本週訓練計畫；每週日自動挑選新鮮焦點。
- 精確建立或重用 GTOW Drill，固定對應的 action line、位置與 stack scope。
- 讀回 GTOW practiced hands、GTO Score 與 EV loss，追蹤本次處方是否完成。

### 5. 記錄與訓練線下 MTT 手牌

- `/live` 接收一批 shorthand hands，解析、修補、評分並寫入 Ledger。
- 每手可打開 GTOW Study、詢問教練、加入練習或重傳修正。
- `/lives` 可找回最近保存的線下 sessions。
- 線下手是選擇性樣本，只用於復盤與 queue，不會混入 online leak 統計。

### 6. 安全同步個人 GTOW session

- Telegram 一次性配對 Chrome Extension，多台 Desktop Chrome 可分別撤銷。
- Extension 自動同步瀏覽器正在使用的 GTOW session，不需反覆手貼 token。
- 原始 token 不寫入 Chrome storage；裝置只保存可撤銷的 device credential。
- Popup 可一鍵攝取 Analyze hands，也保留 `/settoken` 手動備援。

Extension 安裝與安全說明見 [`chrome-extension/README.md`](chrome-extension/README.md)。

## 訓練迴圈

```text
打牌
  ↓
GTOW Analyze / 線下手牌入帳
  ↓
找出 EV loss 最高的 spot family
  ↓
復盤具體牌局，形成可測試的理解
  ↓
進 GTOW Trainer 練到門檻
  ↓
回到真實牌局觀察該 family 的 EV loss 是否下降
```

時間上分成四層：

| 迴圈 | 節奏 | 系統負責的事 |
|---|---|---|
| 內圈 | 每題 | 作答 → 即時 solver 回饋 |
| 中圈 | 每週 | 診斷 → 選焦點 → 復盤 → drill |
| 外圈 | 每月 | 比較練過與未練 family 的真實 EV loss |
| 元圈 | 每季 | 標準化測驗、級別與訓練方向檢討 |

目前已落地的重心是 Decision Ledger、session 復盤、訓練 queue、週課表與 GTOW Drill 串接；更長週期的 playbook、歸因與標準化測驗仍依 North Star 分階段建設。

## 快速開始

### 綁定 GTO Wizard

1. 到 [GitHub Releases](https://github.com/a00012025/ai-poker-wizard/releases) 下載最新版 Extension。
2. 私訊 Bot 輸入 `/pair`，把五分鐘配對碼貼到 Extension popup。
3. 登入 [app.gtowizard.com](https://app.gtowizard.com/)；之後 session 會自動同步。
4. 在 GTOW Analyze 上傳手牌後，點 Extension 的「♠ 同步手牌到 DB」。

### 分析單手

直接傳文字或截圖，例如：

```text
有效 50bb，CO open 2bb，hero SB AcTh raise 7.5bb，CO call
flop KsKhQd，SB bet 1/4，CO call
turn 3h，SB bet 60%，CO fold
打得合理嗎？
```

訊息提到 `cash`、`現金桌` 或 `ring game` 時會切換到現金桌分析；其餘預設為 MTT。

### 匯入線下手牌（owner）

輸入 `/live`，下一則訊息貼上一批手牌；每手以 `Eff <有效籌碼>` 開頭：

```text
Eff 25bb co raise hero bb call As2s
AhQhJh x b1.2 c
2h x b1.5 f
```

ICM 批次中，每手在 header 寫明階段與已知籌碼；`avg` 會約束 GTOW
config metadata 的 `avg_stack`，未知座位不會被偽造成對稱籌碼：

```text
Icm 30% avg 25bb hero has 28bb hj open ATo btn has 14bb all in hero call
Icm 10% avg 18bb hero has 12bb co open 77 sb has 8bb all in hero call
```

## 常用指令

一般功能：

| 指令 | 說明 |
|---|---|
| `/help` | 使用說明 |
| `/pair` | 配對 Chrome Extension（私訊） |
| `/devices` | 查看同步裝置（私訊） |
| `/revoke <ID>` | 撤銷同步裝置（私訊） |
| `/settoken <JWT>` | 手動 GTOW token 備援（私訊） |
| `/logout` | 移除 GTOW session 並撤銷裝置 |
| `/clear` | 清除對話上下文 |

個人訓練功能目前限 owner：

| 指令 | 說明 |
|---|---|
| `/ingest` | 增量攝取 GTOW Analyze hands |
| `/fullingest` | 確認後重掃 ledger epoch（2026-03）以來的完整歷史 |
| `/review [session_id]` | 復盤最近或指定 online session |
| `/sessions` | 查看近期 online sessions 並重傳復盤 |
| `/live` | 匯入線下 shorthand hands |
| `/lives` | 查看最近線下 sessions |
| `/queue` | 查看 drill／復盤工作清單 |
| `/plan` | 重送最新每週訓練計畫 |

## 誠實邊界

- Leak 與課表一律按 EV loss 排序，不按偏差次數製造焦慮。
- 每個統計結論都應帶樣本數；低信心與近似過重資料不進正式統計。
- Online 全量資料與 live 選擇性資料嚴格隔離。
- 無法精確重建 GTOW spot 時不提供誤導性 Trainer 連結。
- Drill 分數不是最終成效；真實牌局的 EV loss 下降才是裁決。
- 這是一個賽後教練系統，不是 RTA。

## 文件

- [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) — 願景、四層迴圈、指標與不變量
- [`docs/TECHNICAL_OVERVIEW.md`](docs/TECHNICAL_OVERVIEW.md) — 技術架構、主要模組、部署與驗證
- [`docs/phase1-loop-runbook.md`](docs/phase1-loop-runbook.md) — Ledger 與週訓練迴圈驗收
- [`docs/analysis-fidelity-runbook.md`](docs/analysis-fidelity-runbook.md) — Analyzer／grader 保真檢查
- [`docs/GTOW_TOKEN_SYNC_DEPLOY.md`](docs/GTOW_TOKEN_SYNC_DEPLOY.md) — Extension／Edge Function 部署
- [`chrome-extension/README.md`](chrome-extension/README.md) — Extension 安裝、安全與測試
- [`AGENTS.md`](AGENTS.md) — 專案結構、開發流程與測試規範
- [`TODOS.md`](TODOS.md) — 尚未完成的後續工作

## License

MIT
