# 週訓練計畫「新鮮度排課」設計（plan_scheduler）

日期：2026-07-29
狀態：已核可，實作中
分支：`feat/plan-scheduler-freshness`

## 1. 問題

`🃏 本週該練的地方` 每週把上週已經練過／已經復盤過的東西再寫一次。查 production DB 後確認是三個**互相獨立**的成因：

**1.1 焦點 spot 沒有任何記憶。** W29 與 W30 的兩個焦點完全相同（`river:SRP:OOP:vs_bet`、`river:SRP:IP:vs_bet`）。診斷視窗是滾動 90 天，每週只滑 7 天，排名幾乎不動。而這兩個 spot 的 EV 幾乎全來自 W22/W24 的舊尖峰（W22 OOP 377 bb/100 n=6、W24 IP 107 bb/100 n=14），**W29/W30 的實戰 per100 都是 0.0**（n=3、n=15）。已經不痛了，卻還會霸榜數週。

**1.2 處方一旦 `prescribed` 就永遠不會自己退場。** `scorecard.QUEUE_SQL` 取 `status IN ('pending','prescribed')`；id 23 從 W28、id 38/39/22/67/68 從 W29 一直掛到現在。原設計註解（§14.2「沒練的處方不能悄悄消失」）立意正確，但缺少老化、輪替與自動結案。

**1.3 復盤名額被兩張老手牌壟斷。** 全 DB 只剩 2 筆 open review（id 67 = 6/20 A♠K♠、id 68 = 5/17 6♠5♥，兩張都是 W29 開的），而 `mix_queue_quota` 固定保留 2 個復盤名額 → 這兩張每週必然重現。既有的「復盤過就是過了」規則只擋掃描重複建立，擋不住既存 row 重複被端上桌。

**1.4（延伸需求）線下手牌進不了計畫。** 佇列按 `total_ev_loss_bb` 排序，線下 pending 全落在 0.10–0.52bb，線上是 3.2–4.6bb，線下**結構性地永遠拿不到名額**。近 60 天 9 個線下 lossy spot 全部 `n=1`，連線上 drill 門檻（n≥3 且 ≥3bb）都摸不到。

| | 決策數 | 手數 | 總 EV loss | per100 |
|---|---|---|---|---|
| 線上 | 34,491 | 28,806 | 648.0 bb | 1.88 |
| 線下 | 59 | 23 | 2.7 bb | 4.60 |

另外系統其實**已經有客觀的練習證據**（`gtow_training_started_at` + GTOW attempt hands/score vs 目標 30 手 / 90%），但週計畫完全沒讀它，唯一結案途徑是手動 ✔。

## 2. 已核可的決策

| 決策 | 選擇 |
|---|---|
| 退場證據 | GTOW Drill 成績達標即自動結案 |
| 焦點選拔 | 新證據門檻 + 冷卻期 |
| 舊處方 | 名額分流：新鮮優先 + 舊帳輪替，標示「第 N 次」 |
| 復盤結案 | 端上桌一次就算送達 |
| 實作路線 | 抽出 `plan_scheduler.py` 純函式模組 |
| 線下納入 | 保留席次（不混算排名） |
| 席次分配 | 線上 3 / 線下 2 |

## 3. 不變量

- **§5.2 source isolation 不破**：北極星指標（bb/100 headline）、週趨勢、焦點排行榜維持 `source='online'`。線下是選擇性記錄的有偏樣本，其 4.60 bb/100 不可與線上 1.88 相比。線下的「加權」以**保留席次**實現，不使用任何無法辯護的係數。
- **§7.3 EV 加權排序不破**：桶內排序仍是 EV；新鮮度只決定分桶，不改變桶內順序（舊帳桶例外，依輪替時間排序，這是排程公平性而非重要性排序）。
- **§14.2 不悄悄丟棄未練處方**：舊帳永遠留在 `/queue`，只是不再自動霸佔週計畫的黃金名額。

## 4. 資料模型

Migration `supabase/migrations/20260729000000_plan_scheduler_freshness.sql`：

```sql
ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS surfaced_count     INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_surfaced_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_surfaced_week TEXT;

-- 時間戳回填自 prescribed_week（那週日曆上的週日，計畫送出的時間），不是 last_added：
-- 週掃描每次 merge 新來源手牌都會更新 last_added，會讓一筆久未理會的 row 看起來剛被看過，
-- 於是被排到輪替隊伍的最後面。
UPDATE drill_queue
   SET surfaced_count = 1,
       last_surfaced_week = prescribed_week,
       last_surfaced_at = to_date(prescribed_week, 'IYYY"-W"IW') + interval '6 days'
 WHERE prescribed_week IS NOT NULL AND surfaced_count = 0;

ALTER TABLE drill_queue DROP CONSTRAINT IF EXISTS drill_queue_clear_reason_check;
ALTER TABLE drill_queue ADD CONSTRAINT drill_queue_clear_reason_check
  CHECK (clear_reason IS NULL OR clear_reason IN
         ('completed', 'mistake', 'skipped', 'resend', 'drill_passed'));
```

`prescribed_week` 語意不變（第一次處方的那週），才算得出「第 N 次」。
`clear_reason='drill_passed'` 把自動結案與手動 ✔ 的 `'completed'` 分開，保留審計能力。

回填的直接效果：W31 一開跑，1.2/1.3 列出的所有陳年 row 立即歸入舊帳桶。

## 5. `scripts/plan_scheduler.py`

排課是**兩層配額**。外層是 source（§5.2 的隔離邊界），內層才是 kind 與新鮮度。

```
5 席
├─ 線上 track  3 席   drill 2 + 復盤 1     舊帳上限 1
└─ 線下 track  2 席   drill 2               舊帳上限 1
   （某 track 無候選 → 名額讓給另一邊；沒打線下的那週仍滿 5 席）
```

### 5.1 Track 歸屬

由 `source_hands` 指到的 `ledger_hands.source` 多數決決定，`drill_queue.source` 只當 fallback。理由：`queue_feed.resolve_queue_source_hands` 的既有註解已載明 `drill_queue.source` 記錄的是「怎麼被加進來的」，不是可靠的來源判別（手動從線下手牌加練的 row 目前掛 `source='manual'`）。

### 5.2 新鮮度分桶（純函式）

| 桶 | 條件 | 桶內排序 |
|---|---|---|
| 🆕 新鮮 | `surfaced_count = 0` | EV desc |
| 🔁 復發 | 端過，但 `last_surfaced_at` 之後該 `spot_leaf` 又累積 ≥ `RELAPSE_MIN_N`(=2) 個 lossy 決策 | EV desc |
| 📼 舊帳 | 端過、無新證據 | `last_surfaced_at` **asc**（最久沒端上桌的先輪到） |

`kind='review'` 一律不入復發桶：單一手牌不會「再犯」，`surfaced_count` 到 1 即永久落入舊帳（這就是「端上桌一次就算送達」）。
線下 drill 則**能**復發：下次線下又在同一 `spot_leaf` 犯錯時 `enqueue_one` 會 merge 新 source，其 `played_at` 晚於 `last_surfaced_at` → 自動升回黃金席次。重複犯的錯會自己浮上來。

### 5.3 填位規則

1. 每個 track 內先填新鮮＋復發（EV desc），至多 `slots - reserved`，其中
   `reserved = min(BACKLOG_SLOTS_PER_TRACK(=1), 實際舊帳數, slots)`。
   **輪替席是「保留」不是「剩餘」**：若寫成剩餘，只要新鮮項目填得滿就永遠輪不到舊帳，
   一筆 22bb 沒練的處方會無限期消失在小額新項目後面 —— 那正是 §14.2 要防的事。
   沒有舊帳可輪時該席位還給新鮮項目，不留空。
2. 舊帳依 `last_surfaced_at` asc 補，每 track 至多 1 席。
3. 某 track 候選不足 → 剩餘名額讓給另一 track 的**新鮮／復發**項目；舊帳上限絕不因讓位而放寬。
4. 仍填不滿 → **不補滿**。訊息改印一行「本週沒有更多新的漏洞，佇列還有 N 項舊帳（/queue）」。少即是誠實。

### 5.4 公開 API

```python
# 純函式（無 DB，可單測）
def resolve_track(row: dict, hand_sources: dict[str, str]) -> str        # 'online' | 'live'
def classify_freshness(row: dict, new_evidence_n: int) -> str            # 'fresh'|'relapse'|'backlog'
def select_weekly_slate(rows: list[dict]) -> dict                        # {'picked': [...], 'backlog_total': int}
def focus_cooldown_blocked(key, history, post_stats, global_per100) -> bool
def drill_attempt_passed(row: dict, attempt) -> bool

# async orchestration
async def annotate_rows(conn, rows) -> list[dict]      # track + new_evidence_n
async def autoclose_passed_drills(conn, client) -> dict
async def mark_surfaced(conn, ids, week) -> None
```

## 6. 焦點冷卻（`scorecard.py` + `spot_leaderboard.py`）

排除發生在 `compute_training_plan` 挑選焦點時，**不是**在 `hierarchical_leaderboard` 裡。
理由：排行榜同時餵「焦點」與「其他 EV 損失節點」，在排名層排除會讓被冷卻的 spot 從
排行榜整個消失，等於謊報 EV 流向。排名邏輯完全不動（§7.3）。

排除規則（讀完整 `coach_focus` 歷史，對每個曾被處方過的 `diagnosis_key`；
不可只取最近 12 週，因為 12 週只有 84 天，會在 90 天診斷窗留下 6 天空隙）：

1. 距上次處方 < `FOCUS_COOLDOWN_WEEKS`(=2) 週 → **硬性排除**。
2. 已過冷卻期 → 仍需通過新證據門檻才能重回焦點：處方後該 key 累積 `n ≥ FOCUS_RELAPSE_MIN_N`(=10) 個誠實決策，**且**處方後 per100 ≥ 全域平均 per100（`spot_leaderboard` 排序既有的收縮目標）。否則繼續排除。

也就是說：**曾被處方過的 spot 只能靠新證據回來，不會單純因為時間到就自動回鍋。** 這正是 1.1 的解 —— `river:SRP:OOP:vs_bet` 要等你真的又在那裡漏錢才會再上榜。被排除的 key 仍照常出現在「其他 EV 損失節點」清單，維持誠實。

## 7. GTOW 達標自動結案

- 對象：`status IN ('pending','prescribed') AND gtow_drill_id IS NOT NULL AND gtow_training_started_at IS NOT NULL`。
- 判定：`attempt.total_hands >= gtow_target_hands AND attempt.gto_score >= gtow_target_score`（與 `_queue_drill_detail_payload` 現行「✅ 本次 Drill 已達標」判準逐字一致）。
- 效率：`GTOWDrillClient` 新增 `attempts_by_drill(started_ats: dict[str, datetime]) -> dict[str, AttemptStats]`，把 `/sessions` 分頁**只讀一次**再依 drill 聚合；`attempt_stats` 改為呼叫它的單筆包裝，行為不變。否則每筆 row 各自分頁最多 10 次請求。
- 失敗軟著陸：GTOW token 失效或 API 異常時記 log 並略過自動結案，**絕不讓週計畫整個失敗**。
- 在 `scorecard._run` 的 `scan_online` **之前**執行，讓達標項目在排課前就退出候選。

## 8. 訊息呈現（`weekly_tg_html` / `render_html`）

- 佇列項目分 `🖥 線上` / `🎲 線下` 兩組列出。
- `surfaced_count >= 1` 的項目標「第 N 次」；舊帳桶項目加 `📼`。
- 新鮮候選不足時印「本週沒有新的漏洞，佇列還有 N 項舊帳（/queue）」。
- 焦點被冷卻擋下時不解釋細節（避免噪音），但 `--preview` 的 markdown 會列出被排除的 key 與原因，供調參。
- 既有按鈕結構（`qdet`/`qsrc`/`qcl`/`qex`）完全不動。

## 9. 測試

新檔 `scripts/regression_tests/test_plan_scheduler.py`：

1. `classify_freshness`：三桶分類，含 review 永不入復發桶。
2. `select_weekly_slate`：線上 3 / 線下 2；每 track 舊帳上限 1；track 空缺時讓位；候選不足時**不**補滿。
3. **1.3 回歸**：兩筆 `surfaced_count=1` 的 review row 不得佔用下一週的復盤席次。
4. **1.2 回歸**：W28/W29 的舊 prescribed row 不得排在新鮮 row 之前。
5. **1.4 回歸**：0.15bb 的線下 row 與 4.5bb 的線上 row 並存時，線下仍拿得到席次。
6. `focus_cooldown_blocked`：上週處方且無新證據 → 擋；有 ≥10 新證據且 per100 ≥ 全域 → 放行；過冷卻期但無新證據 → 仍擋。
7. `resolve_track`：ledger 來源優先於 `drill_queue.source`（含 `source='manual'` 指向線下手牌的案例）。
8. `drill_attempt_passed`：手數/分數兩個門檻的邊界。
9. Migration 欄位存在性檢查（比照 `test_gtow_drills.py` 既有作法）。

`scripts/regression_test.py` 全綠才可提交（CLAUDE.md 強制）。`scorecard.py` 未觸及 `analyze_hand`/`gto_api` 等核心分析檔，故不需 snapshot 套件，但仍會跑一次全套確認無旁生迴歸。

## 10. 不在本次範圍

- 線下復盤席（線下目前不產 `kind='review'` row）。若之後要「線下這場最貴的手」當復盤項，需另開 enqueue 規則。
- 完整訓練狀態機與 mastery 分數（原方案 C）。現階段 readback 樣本太小，撐不起自動狀態轉移。
- 焦點數量（維持 2）與 `FOCUS_WINDOW_DAYS`（維持 90）。
