# Handoff — Phase 1 Ledger + Version A 訓練閉環（2026-07-09）

## 做了什麼
把選手 2026-03-01 至今上傳 GTOW Analyzer 的**全部 33,608 手 MTT** 攝取成誠實、保真、可重算的 Decision Ledger，並在其上把 Version A 最小訓練閉環轉起來。

- **攝取層**：`gtow_analyze_api.py`（list+detail，節流/退避/分頁；404/403/204 軟跳過）、`ledger_ingest.py`（backfill/incremental/verify，冪等可續傳）→ 本地 raw 檔案庫 `data/gtow_raw/`（不變量 9）。
- **Decision Ledger**：`ledger_hands` / `ledger_decisions` / `ledger_sessions` / `coach_focus` / `scorecards`。每筆決策帶 grader、`approx_flags`、`excluded`、`confidence`（誠實層）。
- **Action-line taxonomy**（`spot_taxonomy.py`）：每個 hero 決策節點分類成階層行動線（見 plan v2 節）；`backfill_spots.py` 從 raw 重跑到 `ledger_decisions`（無 API）。limp spot 捨棄。
- **診斷 + 處方**：`spot_leaderboard.py`（avg-EV-loss 榜 + 精準多深度 GTOW Trainer drill + stack-band 分析）、`scorecard.py`（訓練計畫：焦點 spot + retrieval-first + 隔週回讀）。
- **GTOW Trainer 精準連結**：逆向出 `fh_hero/fh_opponent/fh_rel_positions/fh_actions/fh_start_spot/depth_list`（skill `gtow-trainer-drill`），`gtow_trainer_url.build_drill_url`。
- **TG**：`/ingest`（owner）、每日 05:00 攝取 job、週日 21:00 訓練計畫推播；2 個 grounded LLM 追問工具（`query_ledger_summary/hands`）。

## 驗收狀態（spec §11）
- [x] 全量入帳：33,608 手，`VERIFY OK api==db`，`hands_with_loss=2,448`（吻合 live 探測）。
- [x] 誠實層：每筆決策帶信心/近似標注；excluded + discarded 不進統計。
- [x] 20 手保真對數：0/20 mismatch。
- [x] Action-line 榜：avg-EV-loss，帶 n；top-5（n≥50）已由選手驗收（符合直覺，3bet pot 為已知弱點）。
- [x] 精準 drill 連結：live 驗證落點正確（hero/opponent/IP-OOP/多深度）。
- [x] 訓練計畫記分卡：焦點 spot + retrieval-first + drill + 回讀機制。
- [x] TG 佈線 + LLM 追問工具（直測 DB 通過）。
- [ ] 連續 2 週自動推播 / 每日 7 天無漏 / 隔週回讀 delta：**需真實時間累積**（機制已就緒，cron 常駐後自然累積）。

## 已知限制 / 下一步
- **chipEV 評分占比 100%**：postflop 全 chipEV，泡沫/FT 手含 ICM 近似誤差 → Phase 3 Endgame Lab。
- **快訊號未接**：Version A 用真實對局 EV-loss delta（慢）。Trainer practiced-hands 回收（`/v1/poker/practice/drills/`、`/v4/trainer/sessions/last/` 半開的門）→ Phase 2「drill 到 90%」快訊號。
- **postflop limp/iso pot**：保留但帶 `limp_origin`，選手若要也可一併捨棄（`spot_taxonomy` 一行）。
- **cold-3bet/4bet 無 GTOW drill 捷徑**：目前 fallback 到 vs3bet/vs4bet drill；精準化需 Trainer custom mode。

## 測試
- 全套 regression：**667 passed, 4 failed**；4 個失敗（3× L2-GTO 快照 H2492/H2494/H2499 + 1× validator corpus gate）**在 main 上也失敗**，與本 PR 無關（已比對確認）。
- 新增測試：taxonomy、drill URL、training plan、ledger service SQL、tool wiring、soft-404、session clustering 等。
- 註：worktree 需 symlink main 的 `data/*` 與 `.gto_cache` 才能跑到 effbb/OCR/split 相關測試（gitignored 本地資料）。

## 相關
- Plan：`docs/superpowers/plans/2026-07-07-phase1-ledger.md`（含 v2 節）
- Spec：`docs/superpowers/specs/2026-07-07-phase1-ledger-design.md`
- Runbook：`docs/phase1-loop-runbook.md`
- 憲法：`docs/NORTH_STAR.md`（Phase 1「帳本」→ Phase 2「迴圈」）
