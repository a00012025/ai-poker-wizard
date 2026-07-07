# Phase 1「帳本」設計：GTOW Analyze 全量攝取 + 最小教練迴圈

> **Status**: Approved design（brainstorming 定稿）
> **Date**: 2026-07-07
> **上游**: `docs/NORTH_STAR.md` §5.1/§5.2/§5.3/§10 Phase 1
> **下游**: 給 Opus 4.8 的執行計畫（本 spec 批准後產出）

## 0. 一句話

把選手 2026-03-01 至今上傳 GTO Wizard Analyzer 的**全部 33,608 手 MTT**（隨時間增長）攝取為誠實、保真、可重算的 Decision Ledger，並在其上讓最小教練迴圈轉起來：**打牌 → 自動入帳 → EV 加權診斷 → 週記分卡 + 焦點處方 → 隔週回讀有沒有變好**。

## 1. 已定案的 scope 決策（與選手逐項確認）

| 決策點 | 定案 |
|---|---|
| 迴圈邊界 | **B：帳本 + 診斷 + 處方** — 含每週焦點 family、Trainer 練習連結、隔週 delta 回讀；不含 Trainer practiced-hands 攝取（Phase 2） |
| 交付面 | **TG 推播摘要 + self-contained HTML 記分卡附件**；不建 web dashboard |
| 攝取節奏 | **每日 cron（05:00 台北）+ TG `/ingest` 手動觸發**；週日 21:00 台北記分卡 |
| Detail 抓取範圍 | **先全量再開工** — backfill 把 33,608 手的 detail 全部拉完落地，才開始建診斷（資料主權優先） |
| 診斷深度 | **leak 榜 + 知識型/邊界型判定 + session 重建與紀律型相關性 v0**；盲點矩陣不做（需線下流意圖標籤，出界） |
| 架構 | **方案 1：獨立攝取層 + 全新 `ledger_*` 表 + 本地 raw 檔案庫**；`deviations` 表不動 |

明確出界（本次不做）：盲點矩陣、線下捕獲流、GTOW Trainer 攝取（HAR 已看到 `/v1/poker/practice/drills/` 與 `/v4/trainer/sessions/last/`，Phase 2 的門已半開）、賽段（early/bubble/FT）維度（無可靠資料源，後續以 PokerCraft 名次/人數資料補）、web dashboard、任何新的訓練互動面。

## 2. User Story（驗收的敘事版）

> 我週間照常打牌、照常把 HH 上傳 GTOW Analyzer。系統每天凌晨自動把新分析入帳（或我上傳完打 `/ingest` 立刻入帳）。週日晚上 TG 推播週記分卡：一句話趨勢 + top leak（EV 排序、帶 n）+ 最貴 3 手（點開直達 GTOW 該手復盤）+ 本週焦點 family 與練習連結。下週記分卡自動回報：焦點 family 的 EV loss 有沒有降。平時任何時候我可以在 TG 追問帳本（「我這個月 BB 防守漏了多少」），得到與帳本一致、帶樣本數的回答。

## 3. 資料源事實（2026-07-07 live 探測，全部驗證過）

- Host `https://api.gtowizard.com`；auth = `Authorization: Bearer <access>` + `GWCLIENTID: <uuid>` header；access token 由 `.tokens.json` refresh token（5 年效期）經 `gto_token._refresh_access` 現 mint；data endpoints 不需 ECDSA 簽名。
- **List**: `POST /v4/hand-history/hands/`，body `{filters:{played_at__range:[iso,null], analyzer_game_format:"TOURNAMENT"}, pagination:{limit:100,offset,ordering:["played_at"]}, response_fields:[...]}` → `{items,total,limit,offset}`。
  - 可用欄位（實測）：`hand_id`(uuid)、`played_at`、`tournament_id/name/buyin`、`file_original_name`、`site`、`pot_type`、`player_position`、`hero_hand`、`boards`、`total_pot`、`blinds`、`board_flop_connectedness/pairedness`、`game_format`、`total_players`、`preflop_game_depth`、`solution_status`、`total_ev_loss`(bb)、`total_ev_loss_as_pot`、`avg_gto_score`、`avg_frequency_difference`、`player_winloss`(bb)、`hand_correctness`、`actions`（各街動作碼陣列）、`actions_with_correctness_{street}` = `[{action_code, correctness}]`（hero 動作才有 correctness，如 `BEST_MOVE/CORRECT_MOVE/BLUNDER`；非 hero 為 null）、`correctnesses` + `sequence_numbers`。
  - `ordering` 支援 `played_at`、`-total_ev_loss`（實測）；filter 不支援 `total_ev_loss__range`（422）。
- **Detail**: `GET /v4/hand-history/hands/{id}/` → `game_analysis.game_points[]`（每個動作一個節點，含全部 villain 動作）。hero 決策節點有 `analysis_solved.available_actions[]`：每個可選動作的 `frequency`、`frequency_difference`、`correctness`、`ev`、`ev_loss`(bb)、`ev_loss_as_pot`、`gto_score`、`selected`(hero 實選)；另有 `hand_eq`、`real_game`（pot、pot_odds、board、各家 stack）、`solved_action_sequence`（GTOW 標準化的 preflop/flop/turn/river 動作陣列）、頂層 `warning_status`、`approximation_reason`、`live_solved_from_street`、`live_solved_depth`、每節點 `gametype`（樣本為 `MTTGeneral` = chipEV）與 `depth`。
  - 大小：river 手 ~72KB / 15 節點；preflop fold 手遠小。全量估 <1GB 未壓縮。
- **規模**：3/1 至今 33,608 手；其中 **total_ev_loss > 0 者僅 2,448 手（7.3%）**，分佈極陡（rank 0 = 22.66bb，rank 244 = 0.63bb，rank 1224 = 0.08bb）。這符合 solver 語義（均衡支持內動作 EV loss = 0）。
- 疑似存在但未驗證：`POST /v4/hand-history/hand-statistics/`（422，body 格式待從 HAR 重放確認）、`.../hands/{id}/{study-link,practice-link}/`、`.../pokerwizard/tournaments/{id}/...`（404，路徑或 id 形式待確認）。**執行時先探測，全部有 fallback，不是 blocker。**
- Browser session 修復程序：repo skill `gtow-cdp-session`（dedicated CDP profile，FORCED_LOGOUT 5 分鐘修復）。

## 4. 架構

```
GTOW Analyze API ──► scripts/gtow_analyze_api.py ──► data/gtow_raw/*.jsonl.gz   ← 保真檔案庫（不變量 9）
     (list+detail)      (auth/GWCLIENTID/節流/分頁/退避)      │
                                                scripts/ledger_distill.py       ← 純函數：raw → 帳（可重跑）
                                                     │  spot_categorizer 分 family
                                                     │  誠實層映射（GTOW warning/approx → flags）
                                                     ▼
                                            Supabase ledger_* 表                 ← 單一事實來源
                                                     │
                    ┌────────────────────────────────┼──────────────────┐
        scripts/ledger_diagnostics.py       scripts/scorecard.py    LLM ledger 工具 ×2
        （EV 加權聚合/三型 v0/主指標）      （HTML+TG 推播）        （TG 追問）
                                                     │
                                              coach_focus 表 ←→ 處方與隔週回讀
```

原則：**raw 不動、帳可重算**。taxonomy/誠實層規則演進時，重跑 distill 即可，不需重新打 API。`deviations` 表與現有 bot 流程零接觸；Phase 2 再把自建管線評分以 `grader='own_pipeline'` 遷入 Ledger 收斂。

## 5. 資料模型（Supabase migrations，語義對齊北極星 §6）

### `ledger_hands` — 每手一筆
`id` PK、`gtow_hand_id` text UNIQUE、`played_at` timestamptz、`tournament_id` text、`tournament_name` text、`tournament_buyin` numeric、`file_name` text、`site` text、`position` text、`hero_hand` text、`boards` text、`pot_type` text（GTOW 的）、`total_players` int、`preflop_depth_bb` real、`total_ev_loss_bb` real、`total_ev_loss_pct_pot` real、`avg_gto_score` real、`winloss_bb` real、`hand_correctness` text、`solution_status` text、`session_id` FK nullable、`raw_path` text（指回檔案庫）、`detail_fetched` bool、`ingested_at`。

### `ledger_decisions` — 每個 hero 決策一筆（帳本主體，~5 萬筆起）
- 身分：`gtow_hand_id` + `street` + `decision_idx` UNIQUE；`source` text DEFAULT 'online'、`grader` text DEFAULT 'gtow_analyzer'
- 語境：`family` text（我們的 taxonomy）、`texture` text（我們的分類）、`gtow_texture` text（connectedness/pairedness 並存）、`depth_band` text ∈ {le15,15_25,25_40,40plus}、`position`、`pot_type`、`facing` text（面對的動作摘要）
- 判定：`taken_code`、`best_code`、`correctness`、`ev_loss_bb`、`ev_loss_pct_pot`、`taken_freq`、`freq_diff`、`gto_score`、`hand_eq`、`pot_bb`、`gametype`
- 誠實層（不變量 2）：`confidence` real（HH 源=1.0）、`approx_flags` jsonb（枚舉：`unsolved`、`warning:<status>`、`approx:<reason>`、`depth_snap_gap`、`chipev_grading`）、`excluded` bool（true 不進統計、留帳可查）
- `played_at`（denorm，查詢用）、`created_at`

### `ledger_sessions` — session 重建
`id`、`started_at`、`ended_at`、`duration_min`、`tournaments` jsonb（tournament_id 清單）、`max_concurrent_tables` int、`hands_count` int。重建規則：hands 按 `played_at` 排序，間隔 > 60 分鐘切段；同段內以重疊時間窗計併發桌數。

### `coach_focus` — 每週處方
`id`、`week` text（ISO週）、`families` jsonb、`rationale` jsonb（EV 金額、n、排名依據）、`prescriptions` jsonb（連結清單）、`readback` jsonb（隔週 delta 回填）、`created_at`。UNIQUE(week)。

### `scorecards` — 記分卡存檔（可稽核、可 regression）
`id`、`week` text UNIQUE、`html` text、`data_json` jsonb（全部呈現數字的來源）、`pushed_at`。

### Raw 檔案庫
`data/gtow_raw/list/YYYY-MM.jsonl.gz`（list 行）+ `data/gtow_raw/detail/YYYY-MM/{gtow_hand_id}.json.gz`。Append-only。加入 `.gitignore`。

## 6. 攝取管線

### `scripts/gtow_analyze_api.py`
- Session 帶 `origin: https://app.gtowizard.com`、UA 對齊 SPA、`GWCLIENTID`（首次生成 uuid4 持久化於 `.gtow_client_id`）、access token 走 `gto_token.get_access_token()`。
- `list_hands(since, until, offset, limit)`、`hand_detail(gtow_hand_id)`；**節流 2-3 rps + jitter**；429/5xx 指數退避（上限後熄火存 checkpoint）；401 時重 mint access 一次再試。

### `scripts/ledger_ingest.py`
- `--backfill --since 2026-03-01`：list sweep（337 請求）→ 寫 `ledger_hands`（`detail_fetched=false`）→ detail sweep（33,608 請求，~3-4 小時，逐手落 raw + 蒸餾 + `detail_fetched=true`）。**冪等**：`gtow_hand_id` 去重；中斷後重跑自動從未完成處續傳（查 `detail_fetched=false`）。
- `--incremental`：**重掃尾端 30 天視窗**（遲到上傳語義：使用者可能今天才上傳上週的 HH，`played_at` 是舊的，只拉昨天會漏）+ 去重；已知手不重拉 detail。~70 list 請求/日。
- `--verify`：1 個請求比對 API `total`（since 3/1）vs 帳本手數，不符 → TG 告警 + 輸出缺失區間（按月二分定位）。**大聲失敗，不安靜降級（§14.2）。**
- 觸發：PTB JobQueue 每日 05:00 台北跑 `--incremental` + `--verify`；TG `/ingest` 指令手動觸發（回報新入帳手數）。

### `scripts/ledger_distill.py`（純函數，可重跑）
- 輸入 raw detail JSON → 輸出 `ledger_decisions` 列。
- 動作序列用 `solved_action_sequence`；GTOW codes → categorizer tokens 映射（`B→R{size}`、`X→X`、`C→C`、`F→F`、`R{n}→R{n}`、`RAI→AI`；以固定映射表 + 全枚舉測試）。位置由 `total_players` + seat order 推（沿用 `POSITION_ORDERS`）。
- `categorize_spot()` 分 family + texture；`depth_band` 按 `preflop_game_depth` 切。
- 誠實層規則（枚舉）：
  - `correctness` null/`UNSOLVED` → `approx_flags += [unsolved]`，`excluded=true`
  - `solution_status != "OK"` 或 `warning_status != "OK"` → flag + `excluded=true`
  - `approximation_reason` 非空 → flag（是否 excluded 視 reason 白名單，預設不排除）
  - `|preflop_game_depth − game_point.depth| > 3bb` → `depth_snap_gap` flag（不排除）
  - `gametype` 為 chipEV 系（`MTTGeneral` 等）→ `chipev_grading` flag（不排除；記分卡附註呈現）
- 無 graded 決策的手照樣入 `ledger_hands`（0 決策列），數量可觀測。

## 7. 診斷（`scripts/ledger_diagnostics.py`，純查詢層）

- **主指標**：EV loss/100 決策（bb），信心過濾後（`excluded=false`），週序列自 3/1 全程回溯 — 第一份記分卡即有 18 週趨勢。同時輸出 `as_pot` 變體。
- **Leak 榜**：family（× depth_band）聚合：`sum(ev_loss_bb)`、決策數 n、EV loss/100、近 4 週趨勢。**n < 25 的 cell 不排名**，收進「樣本不足」桶（§14.3：統計主張旁必有 n）。
- **知識型 vs 邊界型 v0**：漏錢 family 內，若 ≥70% 損失集中於單一 depth_band 或 texture 子切片（子切片 n ≥ 10）→ `boundary`（規則太粗）；否則 `knowledge`（不會）。
- **紀律型 v0（只呈現相關性，不下因果結論）**：EV loss 率 vs session 第幾小時、vs 併發桌數、vs「大敗手後 15 分鐘窗口」（`winloss_bb < -20` 為 bad-beat proxy），全部帶 n。
- 所有輸出物件帶 `n` 與 `excluded_count`。

## 8. 記分卡 + 處方（`scripts/scorecard.py`，週日 21:00 台北）

- **HTML self-contained**（inline CSS + server-side SVG，零 CDN、零 JS 依賴）：
  1. 主指標卡 + 自 3/1 的週趨勢曲線
  2. Leak 榜 top 5（family × depth_band、bb、n、趨勢箭頭、知識/邊界標記）
  3. 最貴 3 手（牌面、行動線摘要、損失 bb、GTOW 復盤連結）
  4. 焦點處方區（本週 1-2 個焦點 family、選擇理由、練習連結、上週焦點回讀 delta）
  5. Session 觀察 v0（紀律型相關性）
  6. 誠實層附註（excluded 決策數、unsolved 占比、chipEV 評分占比、對數狀態）
- **處方**：按「family 總 EV loss（近 4 週）」排序（= 金額 × 頻率的自然乘積；可學性因子 v1 不做，spec 留註），取 top 1-2（過 n 門檻者）；練習連結**主路徑 = 現有 `scripts/gtow_trainer_url.build_trainer_url()`**（已建成、URL shape 驗證過，taxonomy 直接對映）；列該 family 最貴 3-5 手，每手附 Analyze table 日期過濾 URL（格式已驗證）作保底復盤連結；per-hand `study-link`/`practice-link` endpoint 為可選探測增強。寫入 `coach_focus`。
- **迴圈閉合**：讀上週 `coach_focus` → 本週該 family EV loss/100 delta 回填 `readback` 並呈現，掛變異數註記「單週讀數僅供參考，連續 4 週才算數」（§14.4）。
- **TG 推播**：摘要數行 + HTML 附件（sendDocument）+ 最貴手 inline 按鈕。HTML 與 `data_json` 存 `scorecards` 表。

## 9. TG 追問（LLM ledger 工具 ×2）

沿用現有 leak tools 註冊模式（`gemini_session`）：
- `query_ledger_summary(family?, depth_band?, since?, until?)` → 聚合數字 + n + excluded 數
- `query_ledger_hands(family?, min_ev_loss?, since?, order?, limit)` → 手牌清單 + GTOW 連結
輸出一律帶 n；coach 僅能引用工具結果作答（既有 grounding 紀律，不變量：任何新輸出面必路由事實驗證 §14.7）。

## 10. 測試策略（四層）

1. **單元/regression**（進 `scripts/regression_test.py`）：3-5 手真實 raw JSON 凍結為 fixtures（river blunder 手 / preflop fold 手 / UNSOLVED 手）→ 蒸餾輸出逐欄斷言；GTOW→categorizer 動作碼映射全枚舉；指標數學（EV loss/100、n 門檻、boundary/knowledge、session 聚類邊界）；冪等性（同 raw 蒸餾兩次 → 0 新列）。
2. **保真對數**（backfill 後一次）：帳本手數 == API total == Analyze UI 目測數；隨機抽 20 手 lossy 逐決策 `ev_loss_bb` 與 GTOW 網頁一致；`sum(total_ev_loss)` list 加總 == 帳本加總。
3. **首份診斷預覽 gate（硬性 STOP，選手驗收）**：backfill + 保真對數完成後、收尾（TG 佈線等）之前，用全量真實資料產出首份診斷預覽（記分卡 HTML + 摘要）交付選手驗收。預覽必須包含**具體、可反駁的發現**（例：「你在 facing_cbet_oop × 15-25bb 系統性漏 EV：n=…、共 −X bb、集中於 wet board → 邊界型」）+ 該 line 的樣本手連結，讓選手能以自身對局感 + GTOW UI 抽查驗證產出是真的。選手批准前不得進入後續任務。
4. **迴圈演練**（選手手動驗收）：上傳新 HH → `/ingest` → 查得到 → 週日收記分卡 → 連結全部可點開到正確位置 → 隔週見焦點回讀；TG 追問 3 問，答案與 HTML 一致。
5. **持續守門**：每日 `--verify` 對數告警常駐；記分卡 `data_json` 存表供未來 snapshot diff。

## 11. 驗收標準（Phase 1 完成的定義）

1. 3/1 至今全量入帳（backfill 當下 33,608 手，三方對數通過）。
2. 每筆決策有 grader、信心/近似標注；excluded 不進統計但可查（抽查驗證）。
3. 20 手抽查 per-decision EV loss 與 GTOW UI 一致率 100%。
4. 記分卡連續 2 週自動生成推播：EV 排序、分 family、全數字帶 n、含 18 週回溯趨勢。
5. 處方連結可用；`coach_focus` 有記錄；第 2 份記分卡出現焦點 delta 回讀。
6. 每日增量連續 7 天無漏（遲到上傳 48h 內入帳，30 天重掃視窗保證）。
7. TG 追問答案與帳本一致（3 問抽測）。
8. 首份診斷預覽 gate 已由選手驗收通過（測試策略第 3 層）。

（北極星 Phase 1 的「連續 4 週 100% 入帳」為持續驗收訊號：本次交付證明機制 7 天 + 對數 job 常駐，4 週訊號自然累積。§14.12：完成 = 驗收訊號被觀測。）

## 12. 風險與對沖

| 風險 | 對沖 |
|---|---|
| GTOW 非公開 API：改版/限流/封鎖 | raw 檔案庫先囤（已評分結果永久自有）；節流+退避+可續傳；headers 模仿 SPA；斷供時帳本與診斷照常運作 |
| 33.6k detail 一次拉引注意 | 夜間跑、2-3 rps + jitter、可分兩晚 |
| `practice-link`/`study-link` 未驗證 | 先探測；fallback = 現有 deep-link builder |
| chipEV 評分的 ICM 誤差 | `chipev_grading` flag + 記分卡附註；Phase 3 Endgame Lab 處理 |
| 遲到上傳漏帳 | 30 天重掃 + 每日對數告警 |
| token FORCED_LOGOUT | skill `gtow-cdp-session` 修復程序（5 分鐘） |
| Supabase 體積 | raw 在本地檔案庫；DB 只存蒸餾列（~5 萬列級） |

## 13. 北極星 §13 gate 自檢

1. **服務哪層迴圈**：中圈（診斷→處方→隔週回讀）+ 外圈資料地基。
2. **沉澱資產**：Ledger + coach_focus + scorecards。
3. **不變量**：2 信心標注✓、3 EV 加權✓、9 資料主權✓（raw 檔案庫）、11 回饋延遲✓（日入帳/週記分卡）；1 retrieval-first 不適用於診斷報告面（記分卡非教學互動；追問走既有 grounded 管線）。
4. **市場已有？** 評分完全 reuse Analyzer；只建它沒有的個人 taxonomy 聚合與迴圈管理。
5. **指標路徑**：本 phase 直接把北極星主指標（真實對局 EV loss/100，分 family）第一次真實算出。
6. **被動消費**：記分卡附處方行動出口 + 隔週回讀問責；作答機制屬 Phase 2 Dojo。
7. **維護成本**：非公開 API 為主要脆弱點，對沖見 §12；自建 pipeline 永為 fallback grader。

## 14. 給執行者的重點提醒

- 遵守 repo 開發流程：worktree + branch + PR；每個 bug fix 附 regression test；ad-hoc 探測寫 `scripts/_tmp.py`。
- Migrations 一律 `supabase db push`。
- `.tokens.json` 為 bind-mount，永不整檔替換（in-place write）。
- 探測未驗證 endpoints 時：HAR 重放法見 skill `gtow-cdp-session`（Method B）。
- 蒸餾邏輯改動後：重跑 distill 全量 + 記分卡 snapshot diff，確認無非預期漂移。
