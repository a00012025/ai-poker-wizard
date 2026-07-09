# Phase 1 Ledger — Version A 迴圈演練 Runbook

選手手動驗收清單。每步的預期結果都列出來，照著跑一遍就知道閉環是否轉動。

## 前置
- `.tokens.json` 有效（過期時用 skill `gtow-cdp-session` 修，或 `refresh-gto-token`）。
- Bot 已部署（`bash scripts/deploy.sh`），`OWNER_CHAT_ID`（可選）或 `users` 表恰一個 `is_active` 使用者。

## 1. 入帳（攝取）
```bash
python scripts/ledger_ingest.py --incremental   # 重掃尾端 30 天
python scripts/backfill_spots.py                 # 分類新手到 action-line spot
```
預期：`INGEST list=<新手> detail=<抓取> decisions=<n> skipped=<已知>`；`rows_with_spot_leaf` 上升。
TG：對 bot 打 `/ingest` → 回 `✅ INGEST ...`（僅 owner 可用）。

## 2. 對數（保真）
```bash
python scripts/ledger_ingest.py --verify         # 期望 VERIFY OK api==db
python scripts/ledger_fidelity_check.py          # 期望 mismatches: 0/20
```

## 3. 追問帳本（TG）
對 bot 打：「這三個月我在哪些 spot 漏最多 EV？特別是 3bet pot」
預期：引用帳本數字作答，帶 n（`query_ledger_summary` → top spot 榜；`query_ledger_hands` → 手牌 + Analyze 連結）。

## 4. 週訓練計畫（TG，週日 21:00 台北自動；手動驗收可直跑）
```bash
python scripts/scorecard.py --weekly             # 寫 scorecards + coach_focus，輸出 data/scorecards/<week>.html
python scripts/spot_leaderboard.py --min-n 50 --top 5   # 看焦點榜 + drill 連結
```
預期：TG 收到「📊 週訓練計畫」摘要 + 每個焦點 spot（描述 + 精準多深度 drill 連結）+ HTML 附件。drill 本身即 retrieval 練習（GTOW Trainer 先出手才顯示 GTO）。
驗收：點 drill 連結 → GTOW Trainer 落在描述的那個 spot（hero/opponent 位置、動作、IP/OOP、多深度 badge 正確）。

## 5. 隔週回讀
下週的訓練計畫應出現「上週焦點回讀」：該 spot 本週 avg EV loss/100 vs 處方當時，帶趨勢箭頭 + 「單週讀數僅供參考，連續 4 週才算數」註記。

## 每日自動（cron）
- 05:00 台北：`_daily_ledger_ingest_job` = incremental ingest + backfill_spots + sessions rebuild + verify（對數不符 → TG 告警）。
- 週日 21:00 台北：`_weekly_scorecard_job` = 產訓練計畫 + 推播。

## 誠實層自檢
- limp 相關 spot 已捨棄（`discarded=true`），不進統計。
- chipEV 評分占比（記分卡誠實層附註）——後期/泡沫手含 ICM 近似誤差，Phase 3 處理。
- 每筆決策帶 `approx_flags` 與 `excluded`；統計一律排除 `excluded` + `discarded`。
