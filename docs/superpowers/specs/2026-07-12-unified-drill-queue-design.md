# 統一練習 Queue：線上 leak 入列 + Review 復盤項 + 手動加練（Design Spec）

日期：2026-07-12
狀態：已實作（branch `feat/unified-drill-queue`）。唯一延後項：review 項的 GTOW Analyze
**單手**深連結格式需 gtow-cdp-session 驗證（§5.3）— v1 先用已存在的當日 Analyze table
fallback（`review_url` → `_single_hand_url` 待補）。DB 冪等/re-open/quota 已用 ephemeral
Postgres 端對端驗過（30 項 integration check 全綠），詳見 PR。
北極星對位：中圈（診斷 → 焦點 → drill）；Phase 1 → Phase 2 的橋（Dojo 處方 backlog 雛形）

## 1. 一句話

把 `drill_queue` 從「線下流專用」升級成**統一練習 backlog**：線上牌譜（rolling 60 天）
的高 EV loss 節點以兩種型態自動入列（系統性 leak → drill 項；單手災難 → review 復盤項），
並加上完整的互動生命週期 — drill 項可標記已練消除、review 項給 GTOW 復盤連結 +
「完成復盤」消除 + 「把這手的某條 action line 手動加入練習」的按鈕選單。

## 2. 背景（現況）

- `drill_queue`（migration `20260711000000`）目前只由線下流餵入
  （`live_flow.select_queue_items` → `enqueue`，單決策 ≥`QUEUE_EV_MIN`=0.1bb）。
  現有 17 筆 pending、全部 `source='live'`。
- 線上牌譜的 lossy 節點只在週記分卡（focus/leaderboard/top_hands）閃現，不進 queue。
- 資料量級（2026-07-12 實測，60 天窗、online、單決策 ≥0.1bb）：464 個 lossy 決策
  → 159 個 leaf；n≥3 且累計 ≥3bb 的 leaf 約 19 個（剔除 discarded 桶後）；
  單決策 ≥5bb 的災難手 14 手（合計 135bb，佔窗內總漏損 35%）。
- 已有交叉證據：`BB_vsOpen_SB`、`LJ_RFI` 同時出現在 live queue 與線上 top leak。

## 3. 名詞與型態

| kind | 語義 | 進列來源 | 動作 |
|---|---|---|---|
| `drill` | 系統性 leak，去 GTOW Trainer 練這個 spot | 線上掃描（auto）/ 線下流（auto）/ 手動（manual） | 🎯 練（Trainer URL）→ ✔ 已練（cleared） |
| `review` | 單手災難，去復盤那一手 | 線上掃描（auto） | 🔗 復盤（GTOW Analyze URL）→ ✔ 完成（cleared）；➕ 加練（拆解該手 action lines 供選擇，選中者以 manual drill 入列） |

`added_by`: `auto`（掃描/線下流）/ `manual`（owner 從 review 項按鈕加入）。

## 4. 資料模型變更（migration）

```sql
ALTER TABLE drill_queue
  ADD COLUMN kind TEXT NOT NULL DEFAULT 'drill',       -- 'drill' | 'review'
  ADD COLUMN ref_hand_id TEXT,                         -- review 項/手動項的參照手（gtow_hand_id 或 live hand id）
  ADD COLUMN added_by TEXT NOT NULL DEFAULT 'auto',    -- 'auto' | 'manual'
  ADD COLUMN cleared_at TIMESTAMPTZ;                   -- qcl 消除時間（re-open 判斷用）

DROP INDEX idx_drill_queue_pending_leaf;
CREATE UNIQUE INDEX idx_drill_queue_pending_leaf
  ON drill_queue(spot_leaf) WHERE status = 'pending' AND kind = 'drill';
CREATE UNIQUE INDEX idx_drill_queue_pending_review
  ON drill_queue(ref_hand_id) WHERE status = 'pending' AND kind = 'review';
```

- 既有 17 筆 live rows 落在 default `kind='drill'`, `added_by='auto'` — 語義正確，無需 backfill。
- `source` 欄位沿用：`'live'` / `'online'` / `'manual'`。
- review 項的 `spot_leaf` 填災難決策（該手最大 EV loss 決策）的 leaf，僅供顯示；
  唯一性以 `ref_hand_id` 判。

## 5. 線上掃描（新模組 `scripts/queue_feed.py`）

### 5.1 掃描規則（門檻已與 owner 定案）

常數（模組頂部，全部可調）：

```python
QUEUE_SCAN_WINDOW_DAYS = 60   # rolling window（owner 指定「過去兩個月」；刻意不對齊 scorecard 的 90d 焦點窗，兩者語義不同）
QUEUE_DRILL_MIN_N = 3         # 信心過濾：≥3 次才算 pattern（§2.1；n 只當門檻，排序仍純 EV，§7.3）
QUEUE_DRILL_MIN_TOTAL_BB = 3.0  # 值得一次 ≤20min Trainer session 的底線（北極星 §5.6 練習參數）
QUEUE_REVIEW_MIN_BB = 5.0     # 單決策災難門檻
```

**基礎 predicate（必須逐字重用 spot_leaderboard 的誠實條件）**：

```sql
NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL AND source='online'
AND played_at >= NOW() - INTERVAL '60 days'
```

（`NOT discarded` 自動剔除 `discarded:faced_limp` 等誠實排除桶 — 那些不是可練的 spot。）

**Drill 候選**：以 leaf 聚合，只計 `ev_loss_bb >= 0.1` 的決策；
`COUNT(*) >= QUEUE_DRILL_MIN_N AND SUM(ev_loss_bb) >= QUEUE_DRILL_MIN_TOTAL_BB`。
入列欄位：`kind='drill'`, `source='online'`, `added_by='auto'`,
`n_sources`=窗內 lossy 決策數, `total_ev_loss_bb`=窗內累計,
`source_hands`=窗內 lossy 決策清單（見 5.2 的 key 格式）。

**Review 候選**：單決策 `ev_loss_bb >= QUEUE_REVIEW_MIN_BB`，**以手聚合**（一手最多一筆
review 項；同手多個 ≥5bb 決策全部收進 `source_hands`，`spot_leaf`/label 取最大那個決策）。
入列欄位：`kind='review'`, `ref_hand_id=gtow_hand_id`, `total_ev_loss_bb`=該手合格決策 EV 加總。

### 5.2 冪等與 re-open（週週重掃 rolling window 的正確性）

掃描每週重跑、窗口重疊，**必須冪等**，否則 pending 行的 totals 會被同一批決策重複灌水：

- `source_hands` 每筆帶唯一 key：`{hand_id, street, decision_idx, ev_loss_bb, src}`
  （`src`: `'online'|'live'|'manual'`）。掃描 upsert 時先讀現有行，
  **只 append 不存在的 key，totals 只加新增條目的 EV**（Python 端 diff，不能只靠
  ENQUEUE_SQL 的無腦 `||` append）。
- **Cleared 的 drill leaf 不無腦復活**：同 leaf 存在 `status='cleared'` 行時，僅當窗內有
  **≥2 個 `played_at > cleared_at` 的 lossy 決策**（新證據）才重新入列（新開 pending 行）。
- **Review 項永不重複**：`ref_hand_id` 已有任何狀態的 review 行 → 跳過（復盤過就是過了）。
- Open（pending/prescribed）行沿用 `live_flow.MERGE_OPEN_SQL` 的合併精神（PR #91 的
  同 leaf 合併），但要把該 SQL 的合併路徑改為上述 dedupe-aware 版本，並將
  `enqueue()`/MERGE_OPEN_SQL/ENQUEUE_SQL 從 live_flow **抽到 queue_feed（或共用模組）**，
  live_flow 改 import — 單一 upsert 政策，禁止兩份 copy（同 PR #92 的 dedup 精神）。
  ENQUEUE_SQL 的 `ON CONFLICT` 目標索引已改為 partial index on (spot_leaf) WHERE
  pending AND kind='drill'，INSERT 需帶上新欄位。

### 5.3 Drill URL 與中文標籤（重用，不新造）

- Drill 項：per-leaf mode 參數（`mode() WITHIN GROUP` 取 hero_cat/villain_cat/ip_oop/
  hero_pos，同 `spot_leaderboard.LEADERBOARD_SQL`）+ `choose_depths`（stack band 集中
  判斷）→ `gtow_trainer_url.drill_url_for_spot`。與 leaderboard 同一條路徑（單一政策）。
- 中文 label：重用 `scorecard.spot_desc_zh`（live_flow.spot_label_zh 已在用）。
- Review 項 URL：**GTOW Analyze 單手頁面連結**。已知存在 hands table URL
  （`spot_leaderboard.analyze_table_url`），單手 URL 格式**需在實作時驗證**：
  用 repo skill `gtow-cdp-session` 開啟 Analyze 頁、點進一手、抄下 URL 格式
  （預期類似 `https://app.gtowizard.com/analyze/...?hand=<uuid>` 或 path param）。
  驗證後把格式寫進 agent memory。**Fallback**（驗證失敗或格式不含 hand id）：
  `analyze_table_url(該手 played_at 的台北日)` — 已存在的函式。
- Review 項 label 格式：`復盤 {M/D} {spot_desc_zh(最大損失決策)} −{ev:.1f}bb`；
  若該決策 `approx_flags` 含 `sizing_snap`/`depth_snap_gap`，label 尾附 `⚠近似`
  （§5.2 誠實層：災難值可能被近似灌水，讀保守）。

### 5.4 CLI 與排程

```bash
python scripts/queue_feed.py --scan [--window-days 60] [--dry-run]   # dry-run 印候選不落庫
```

- **初次 backfill = 第一次正式跑 `--scan`**（就是 owner 要的「過去兩個月」入列），
  不需要獨立 backfill 模式。先 `--dry-run` 給 owner 過目再落庫。
- **每週自動補進**：週日 21:00 記分卡 job 在建 payload / drain queue **之前**先呼叫
  `queue_feed.scan_online()`。順序：掃描入列 → 取 top 項處方（見 §7）→ 標 prescribed。

## 6. /queue 互動生命週期（bot 變更）

### 6.1 列表呈現

`/queue`（owner-only）維持單一列表，pending 先、各自 EV desc（現有排序），加 kind 圖示：

```
🎯 1. BB 對 SB open 防守 — 3 手 / 1.2bb   [pending]
🔍 2. 復盤 6/1 SB vs BB SRP river 面對下注 −22.7bb   [pending]
```

### 6.2 按鈕與 callback

| 項 | 按鈕 | callback / URL |
|---|---|---|
| drill | `🎯 練 i`（URL）、`✔ i 已練` | 既有 `qcl:<id>` |
| review | `🔗 復盤 i`（URL）、`✔ i 完成`、`➕ i 加練` | `qcl:<id>`、`qex:<id>` |

- `qcl:` 既有 handler 加一行：`SET cleared_at=NOW()`（re-open 判斷依賴它）。
- `qex:<queue_id>`（新）：讀該 review 項的 `ref_hand_id` → 查
  `ledger_decisions WHERE gtow_hand_id=$1 AND NOT excluded AND NOT discarded`
  （按 street 順序 preflop→flop→turn→river、decision_idx）→ 發一則子選單訊息：
  每個決策一行（街 + `spot_desc_zh` + EV loss 標注，含 0.0 的正確決策 —
  owner 可能想練打對但沒把握的線），每行配按鈕 `➕` → `qad:<queue_id>:<decision_row_id>`。
  **callback_data 用數字 id（ledger_decisions.id），絕不放 spot_leaf 字串**
  （Telegram callback_data 64 bytes 上限，leaf 如 `river:SRP:SBvBB:OOP:[b-c|x-b-c]:vs_bet` 會爆）。
- `qad:<queue_id>:<decision_id>`（新）：載入該決策 row → 以
  `kind='drill'`, `added_by='manual'`, `source='manual'`, `ref_hand_id=該手`,
  `source_hands=[{hand_id, street, decision_idx, ev_loss_bb, src:'manual'}]`,
  `total_ev_loss_bb=該決策 ev_loss`（可為 0）, drill_url + label 走 5.3 同一路徑，
  upsert 入列（同 leaf 已 open 則合併，`added_by` 保留原值）。
  回 callback answer + 確認訊息（含 leaf label 與 `/queue` 提示）。

### 6.3 GTOW Drill 詳細選單與成績（2026-07-16 延伸）

- drill 項的入口改為 `qdet:<queue_id>:<page>`，不直接離開 Telegram。打開時以完整
  Trainer settings（包含 GTOW 會自動注入的 169 手牌組）查找相同 Drill；有則復用，
  無則以 queue label 建立，名稱不加 `APW` 前綴。設定相同時 GTOW Trainer 會自動選中
  該 Drill，開始 session 的 request 會帶其 UUID。
- 詳細卡顯示處方來源、lifetime hands/decisions/score/EV loss，以及自首次開卡後、綁定
  Drill UUID 的本次 sessions 成績。`qdst` 原地更新；`🎯 開始練習` 才是 GTOW URL。
- 30 hands / 90% 是達標標籤，不是清除 gate。`qcf` 提供「已完成練習」與「誤植，直接
  清掉」，兩者都可隨時 `qcl`；以 `clear_reason` 區分，誤植不得記成 Drill 達標。
  來源手照常記在 `source_hands` + `ref_hand_id`，未來回看有跡可循。

### 6.3 權限

全部沿用 owner-only gate（`_is_owner`）。callback handler 同樣要驗 owner
（現有 `handle_live_button` 的 pattern）。

## 7. 週記分卡 / 訓練計畫變更（scorecard.py）

- `QUEUE_SQL`（現為單一 `LIMIT 5`）改為**分 kind 配額**：
  `QUEUE_DRILL_SLOTS = 3`、`QUEUE_REVIEW_SLOTS = 2`（常數可調）；
  一種不足時以另一種補滿 5 格。排序維持 pending 優先、`total_ev_loss_bb DESC`。
- 週日 payload 的佇列區塊按鈕：drill 照舊；review 用 `🔗 復盤` URL 按鈕 +
  `✔ 完成` / `➕ 加練` callback（同 §6.2）。
- Prescribe 行為不變：surfaced 的項標 `prescribed` + `prescribed_week`；
  aging（prescribed 未清重浮）既有機制自然涵蓋兩種 kind。
- **Readback 不受影響**：`prev_focus_readback` 讀 `ledger_decisions`（by focus leaf），
  不讀 drill_queue。加一條註解性不變量：未來任何從 drill_queue 取數的統計/readback
  一律 `kind='drill'` 且不得把 queue 當統計面（§5.2 來源隔離 — queue 是訓練面）。

## 8. 不變量對照（北極星 §13 gate）

- **服務中圈**：queue = EV 排序的練習 backlog；記分卡每週 drain top-5 處方。✅
- **§7.3 EV 加權排序**：一切排序按 `total_ev_loss_bb`；`n≥3` 是信心門檻（§2.1），
  永不作為排序鍵。✅
- **§5.2 來源隔離**：掃描只讀 `source='online'` + 誠實 predicate；queue 本身是
  訓練面、永不回餵任何統計。live/manual/online 在 queue 內共存是刻意的。✅
- **§5.2 誠實層**：excluded/discarded 永不入列；近似敏感的 review 項帶 ⚠ 標注。✅
- **單一政策**：upsert（enqueue）/drill URL/中文 label 各只有一份實作，
  live_flow 與 queue_feed 共用。✅

## 9. 驗收標準

1. `queue_feed.py --scan --dry-run`（60d 窗）產出 ≈19 drill + ≈14 review 候選
   （以 2026-07-12 資料為準；差異需能解釋，例如 excluded/discarded 過濾）。
2. 落庫後 `/queue` 正確顯示兩種 kind + 對應按鈕；`✔` 消除寫入 `cleared_at`；
   `➕ 加練` 展開該手全部已評分決策、選擇後以 manual drill 入列且帶 `ref_hand_id`。
3. **連跑兩次 `--scan` 結果一致**（totals 不灌水、不長重複行）— 冪等性驗證。
4. cleared 的 drill leaf 在沒有新證據時不復活；有 ≥2 筆新 lossy 決策時重開。
5. 週日記分卡：先掃描後 drain；佇列區塊 3 drill + 2 review；prescribed 標記正確。
6. 全套 regression 綠，且新增測試覆蓋：掃描聚合規則（drill/review 門檻與分組）、
   upsert 冪等/re-open 邏輯、qex 子選單構建（callback_data 長度）、qad 手動入列、
   QUEUE_SQL 配額混合。migration 檔存在且 `supabase db push` 可跑。

## 10. Out of scope（刻意不做，未來 phase）

- live 來源的 review 項（線下單手災難 ≥5bb 也轉復盤項）— v2。
- online review 手接自家教練深挖（需從 raw archive 重建 parse → coach 路徑）— v2；
  v1 只給 GTOW 連結。
- Review 手的 SRS 排程 / 同構變體 morph（北極星 §5.6 自建題型層）— Phase 2。
- leaderboard/LLM top/diagnostics 三面排序鍵統一 — 獨立 cleanup（審計遺留）。
- shrinkage 估計取代硬 n floor（§14.3 / 審計 D1）— 未來。

## 11. 實作備忘

- 開發流程照 repo 規範：worktree + branch（建議 `feat/unified-drill-queue`）→ PR。
- 本 spec 檔隨實作 PR 一起 commit。
- `.env` symlink 進 worktree；regression 的 `.gto_cache` 注意
  agent memory `worktree-gto-cache`（`cp -n` 不能 symlink）。
- DB 操作全走 migration + `supabase db push`（禁 raw psql DDL）。
- bot 改動屬於 `src/telegram_bot/bot.py`（非核心分析檔），但 scorecard.py 若改到
  仍建議跑全套 regression（本來就是 merge 前提）。
- 測試 fixture 可參考 regression_test.py 既有的 queue lifecycle 測試
  （prescribed aging / merge 行為，PR #91 加入）。
