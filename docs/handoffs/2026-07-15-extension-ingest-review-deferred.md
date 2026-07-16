# Extension ingest PR (#120) — review 後遺留的 refactor 建議

2026-07-15 code review（8 finder angles）結論。correctness 與 PR 範圍內的
清理已直接修在 `feat/extension-sync-trigger`；以下是刻意**不在本 PR 做**的
建議，留待之後獨立處理。

## 已修（記錄供對照）

- run_pipeline 重複段落抽成 `_pass()`；backfill_spots / ledger_sessions /
  verify 的 rc 全部檢查，任何 stage 掛掉都 loud-fail（不再綠勾配壞資料）。
- verify rc=1（script crash）不再被當成通過；rc=2 全量補齊後仍不符 → 結果
  帶 ⚠️ 警告而非 hard-fail（GTOW 端刪除 / epoch 前手牌是修不掉的）。
- stale 偵測改 heartbeat 制：subprocess 的 sweep 進度行即時寫回 row
  （`heartbeat_at`），45 分鐘無心跳才判中斷，且中斷會發 Telegram（原本靜默）。
  expiry 掃描 60s 一次而非每 5s tick。
- enqueue 去重改原子性：partial unique index（每 user 一件 open request）+
  `ON CONFLICT DO NOTHING`，bot `/ingest`、每日排程、edge RPC 三個入口共用
  `enqueue_request()` / 同一 index，check-then-insert race 消除。
- 每日 05:00 job 改為 enqueue（不再繞過 `_run_lock` 直接跑 pipeline），
  與 extension 觸發共用單飛路徑；排程失敗通知 ADMIN_CHAT_ID。
- `gtow_analyze_api` 改用現成的 `gto_token.get_user_access_token`（exp-aware
  + fingerprint cache），移除手刻 `_override_access`；401 re-mint 只消耗一次
  （原本 401 後每個 retry 都重 mint）。
- edge function 新 route 改用 `authenticateDevice()`（含 revoked_at 檢查）；
  ingest status 的 UUID regex 修正（原本接受任意 36 字 hex/hyphen、拒絕大寫）。
- content.js 輪詢 running 時退到 5s、tab hidden 時跳過。
- 全量 sweep 不再 hardcode `--since 2026-03-01`（epoch 預設在 ledger_ingest）。

## Deferred（之後再做）

1. **RESOLVED — legacy global auth 已移除**：access token 現由 DB refresh
   token mint；401 retry 透過 `invalidate_user_token` 強制重 mint。
2. **content.js / popup.js 輪詢迴圈重複**：trigger+poll 應下沉到
   background.js（單一 INGEST_RUN 訊息或 port streaming），兩個 UI 變純
   renderer。目前 popup 版沒有 hidden-skip/backoff，協定改動要改兩處。
3. **status 輪詢 → Supabase Realtime**：每次 ingest ~數百次 edge invocation
   純為讀狀態；改 extension 訂閱 `gtow_ingest_requests` row 變更可歸零。
4. **重複點擊時的冗餘 force token sync**：token 未變時可用 fingerprint
   短路（需比對 server 端狀態而非 local，否則會破壞 force 語意）。
5. **5-arg `sync_gtow_refresh_token` wrapper 清除**：edge function 部署穩定後
   出一個 follow-up migration DROP 掉（少一個 SECURITY DEFINER 入口）。
6. **owner gate 提前到 enqueue**：目前非 owner 的 request 會排隊數秒後才被
   runner 打回。等多用戶 ledger（`ledger_hands` 加 user 欄位）一起做。
7. **/settoken 降 iat 的接受風險**：手動 force-override 後，非 force 的
   auto-sync stale 基準跟著變舊（理論上放寬了舊 token replay 窗口，前提是
   已配對裝置被入侵）。接受此 trade-off；若要收緊，方向是「force 寫入不
   下修 iat 基準、只換 token body」。
8. **RPC 內 device 兩次 SELECT**：`sync_gtow_refresh_token` 先查 user_id 再
   FOR UPDATE，是上游鎖序（users 先於 device）的既有 pattern，勿貿然合併；
   若要簡化需保持鎖序。
9. **永久對數不符時的每日全量 sweep 成本**：mismatch 修不掉時每天一次全量
   sweep（~350 requests）。可加「上次全量後仍不符則 24h 內不再升級」的
   state guard（查最近一筆帶 ⚠️ 的 done request 即可）。
10. **`_run_script` 完全合併**：main_gemini 已改為 delegation；scorecard job
    遷走後可刪掉 main_gemini 的 wrapper。
