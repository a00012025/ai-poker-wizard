# Session 復盤（Post-Session Review）— Design

> Status: approved (brainstorm 2026-07-15)
> Owner surface: Telegram bot, owner-only
> North Star fit: §7 不變量 11（回饋延遲預算：session 復盤能過夜不過週）、§4.2（理想每個 session 後入帳）、§5.9（session 級 EV loss）

## 問題

同步 500+ 手之後，使用者目前只看到一行 count summary（新增手牌 X…）。訓練計畫（`/plan`）只重送上週已存的記分卡、queue 只在週日 21:00 被掃描——**打完到學到中間最長空 7 天**。北極星把這個延遲視為 bug（§7-11 session 復盤「能過夜不過週」），且明確不是 RTA 紅線（RTA = 對局中；復盤是打完後）。

## 不變量（設計約束）

1. **只讀不改焦點**：session 復盤**不設定/不重排本週焦點 spot**。中圈（§3）需要一週穩定的焦點才能 drill→重測；每場重排 = 追噪音（§2.1 H3510 教訓）。復盤只更新「診斷畫面」與「backlog（drill_queue）」。
2. **EV 加權、單場不作趨勢判斷**：一場是一個樣本。輸出全部是「這場的事實」（共幾手、平均 EV loss、加總最多的 spot、最貴的手），**禁止**輸出「你進步/退步了」或任何頻率計數排序（§7.3）。保留一行「單場數字噪音大」的變異數警語，但**不做**近 N session 百分位（早期無基準、且非使用者所需）。
3. **口徑與週記分卡一致**：沿用 `_HONEST` 誠實過濾（`NOT excluded AND NOT discarded AND spot_leaf IS NOT NULL AND source='online' AND confidence >= 0.8`），per100 = `avg(ev_loss_bb)*100`，加總 = `sum(ev_loss_bb)`。source 隔離（§5.2）由 `source='online'` 保證。
4. **排入走既有 backlog**：兩個「加到 queue」動作對應既有 `drill_queue` 的兩種 kind——spot→`kind='drill'`、單手→`kind='review'`，`added_by='session'`。用 `queue_feed.enqueue_one`（idempotent、dedupe-aware），與週掃描產生同構的 rows。

## 使用者會看到什麼（TG 訊息）

- 標題：`🔍 這場復盤` + 時段
- `共 N 手，平均 X bb/100 決策，本場合計漏 Y bb`（+ 變異數警語）
- **漏最多的情境（EV 加總）**：top-2 spot，每個 → `[🎯 現在練]`(GTOW Trainer URL) `[📥 排入佇列]`(callback)
- **最貴 3 手**：每手 `combo board — 短描述 −Zbb` → `[① 手牌]`(Analyze 單手 URL) `[📖 復盤]`(Study 節點 URL) `[📥 排入]`(callback)
- 一行誠實 caveat（chipEV/limp 未計/低信心/覆蓋範圍）
- footer：`[🔕 這場略過]`（尊重依從，§7-11 推播可跳過）。**不放「全部排入」**——加入 backlog 一律逐項刻意選擇（時間預算紀律 §5.10）。

## 架構

新檔 `scripts/session_review.py`（純聚合 + render，reuse 既有 URL/label/enqueue helpers）：

- `resolve_session(conn, session_id=None) -> dict|None`：預設回傳最近一個 online session（`ledger_sessions` order by ended_at desc）。
- `compute(conn, session) -> dict`：session-scoped SQL（`gtow_hand_id IN (SELECT … FROM ledger_hands WHERE session_id=$1)` + `_HONEST`）算 overview / top_spots(2) / top_hands(3) / honesty。top-N 不套 queue 的 min-N 門檻（這是描述性 top-N，不是自動掃描）。
- `render_tg(data) -> {"html": str, "buttons": [[btn,…],…]}`：週記分卡口吻；URL 按鈕（練/手牌/復盤）+ callback 按鈕（排入/略過），callback_data = `srd:{sid}:{i}` / `srv:{sid}:{i}` / `srx:{sid}`（<64B）。
- helpers reuse：`queue_feed.{gtow_analyze_hands_urls, review_url, queue_drill_url_from_sources, drill_label, review_label, enqueue_one}`、`scorecard.spot_desc_zh`、`spot_leaderboard`。

CLI（本階段驗收用，dry-run 不送不寫）：
```
python scripts/session_review.py --latest        # 最近一個 session
python scripts/session_review.py --session-id N
python scripts/session_review.py --latest --json  # 結構化輸出
```

## 相依風險（誠實揭露）

- `📖 復盤` 的 Study 節點 URL 走 `queue_feed.review_url` → `_study_solution_url`（archived raw detail）→ 失敗 fallback 到當日 Analyze 表。此單手 Study URL 就是 unified-queue（PR #97）標記「待 gtow-cdp 驗證」的那條，是本設計唯一尚未端到端驗證的連結；其餘（Trainer drill URL、Analyze 單手 URL、enqueue）都是已上線機制。
- ~93% 手零損失 + limp/低信心剔除 → 有些場只會產出「這場沒什麼好復盤的，2 手小漏損」。這是特性（不製造戲劇），但要接受同步 500 手後常看到清淡結果。

## 分階段

- **Phase 1（本次，驗收 gate）**：`session_review.py` compute + render + CLI dry-run + regression test；對昨天 500+ 手 session 跑出真實 TG 訊息給 owner 驗收。
- **Phase 2（驗收後）**：`/review` 指令 + callback handlers（`srd/srv/srx` → `enqueue_one` + `edit_message` 即時回饋）接進 `bot.py`；（可選）同步完成後在 `ingest_runner` 收尾自動附一則復盤。部署。

## 測試

- Regression test `scripts/regression_tests/test_session_review.py`：以 fixture 資料驗 `render_tg`（訊息含手數/per100/top spot/top hand、按鈕結構、無趨勢字眼、callback_data <64B、無「全部排入」）。純函式、無 DB。
