# Extension-Triggered Hand Ingest — Design

2026-07-15 · branch `feat/extension-sync-trigger`

## 問題

上傳手牌到 GTOW Analyze 後，把手牌拉進 ledger 需要手動跑 `/ingest` 或等每日
05:00 排程；而且 ingest 曾依賴全域 owner token，一旦使用者在
別處登入 GTOW 觸發 FORCED_LOGOUT，ingest 靜默失敗（bot 回「INGEST（無輸出）」
還打 ✅）。要一鍵：在 GTOW 頁面點一下 → 手牌進 DB，token 永遠用使用者當下有效
的那組，統一存入 `users.gto_refresh_token`。

## 既有基礎（v2 extension，直接沿用）

- `chrome-extension/`（MV3）已有裝置配對（`/pair` 12 碼、HMAC-pepper device
  credential）+ 自動 refresh-token 同步。
- Edge function `gtow-sync`（device auth）把 token 寫進
  `users.gto_refresh_token`（`sync_gtow_refresh_token` RPC，audit 只存
  fingerprint）。
- 亦即：**使用者當前有效 token 已經隨時在 DB 裡**，ingest 只需要改吃它。

## 架構

```
Extension（點「同步手牌」按鈕，popup 或 GTOW 頁面浮動按鈕）
  1. 先強制同步當前 tab 的 user_refresh（既有 SYNC_ACTIVE_TAB 流程）
  2. POST gtow-sync/ingest        (Device auth) → 建 gtow_ingest_requests row
  3. GET  gtow-sync/ingest/status (Device auth) 每 2s 輪詢 → 頁面 toast 顯示進度

Bot（JobQueue 每 5s）
  4. 撈最舊 pending row（FOR UPDATE SKIP LOCKED；v1 僅處理 owner 的 request）
  5. 讀 users.gto_refresh_token → 子程序 env GTOW_REFRESH_TOKEN
  6. ledger_ingest --incremental → backfill_spots → ledger_sessions --rebuild
  7. ledger_ingest --verify；不符 → 自動全量 sweep（--since 2026-03-01）→ 再 verify
  8. row 寫 progress/result；Telegram 通知 request 的 user
```

## 元件

### 1. Migration `gtow_ingest_requests`

`id uuid PK`, `user_id bigint → users`, `device_id uuid → gtow_sync_devices`,
`status text` (pending/running/done/error), `progress text`, `result text`,
`requested_at/started_at/finished_at timestamptz`。RLS 全鎖（僅 service_role
via edge function；bot 走直連 postgres）。同一 user 已有 pending/running 時
edge function 直接回傳既有 request id（防連點/防灌爆）。

### 2. Edge function `gtow-sync` 新 route

- `POST /ingest`（Device auth）→ insert or reuse pending row → `{request_id}`
- `GET /ingest/status?id=`（Device auth，只能查自己 user 的 row）→
  `{status, progress, result, requested_at, finished_at}`

### 3. `gtow_analyze_api` per-request token 模式

env `GTOW_REFRESH_TOKEN` 存在時：用 `gto_token._refresh_access` + 臨時
keypair mint access（in-memory cache，401 時 re-mint 一次）。未設 env 時，
owner-run CLI 由 `OWNER_CHAT_ID` 解析 DB token；bot request 漏接 token 則 fail closed。

### 4. Bot ingest runner（`src/ingest_runner.py` + main_gemini 註冊）

- run_repeating 5s；一次只跑一件（SKIP LOCKED claim）。
- v1 限 owner：非 owner request 標 error「ledger 目前僅支援 owner 帳號」。
- token 缺失/mint 失敗 → error「token 無效，請重新登入 GTOW 再點一次」。
- 階段更新 progress：攝取中 → 補 spot 分類 → 重建 sessions → 對數中 →
  （必要時）全量補齊中 → done；result = `INGEST list=X detail=Y decisions=Z`。
- 完成/失敗都發 Telegram 給 request 的 user。
- 啟動時把 stale `running`（>30min）標 error（bot 重啟中斷的殘留）。

### 5. Extension v2.1.0

- popup 加「🔄 同步手牌到 DB」按鈕；content script 在
  `app.gtowizard.com/analyze/*` 注入浮動按鈕 + toast 進度條。
- 點擊流程：SYNC_ACTIVE_TAB（確保 DB token 最新）→ INGEST_TRIGGER →
  每 2s INGEST_STATUS 直到 done/error；list=0 時 toast 提示「GTOW 可能還在
  處理剛上傳的檔案，稍後再點一次」。
- icon 換成 bot 的 Telegram avatar（16/32/48/128，`chrome-extension/icons/`）。

### 6. 手動觸發 force-override stale-token guard（用戶追加）

`sync_gtow_refresh_token` 的 `STALE_REFRESH_TOKEN`/`CONFLICTING_REFRESH_TOKEN`
guard 用 iat 比新舊，但 FORCED_LOGOUT 情境下「iat 較新」的 DB token 可能已
死、當下登入中的（iat 較舊）反而有效 — 手動觸發時把有效 token 擋下來沒有意
義。改法：RPC 加 `p_force boolean`（保留 5-arg wrapper 無痛升級），edge
function `/token` 接受 `force`，extension 所有**手動**路徑（sync 按鈕、
ingest 按鈕、剛配對後的首次同步）帶 `force:true`；被動 auto-sync
（TOKEN_DETECTED）維持 guard。`/settoken`（必為手動、且上游已向 GTOW 驗證
過）同步移除 `save_user_gto_token` 的 iat guard。安全性由 edge function 的
`validateWithGtow`（真的去 GTOW 換一次 access）把關。

### 7. 順手修

`/ingest`（bot.py）crash 時不再回「✅ INGEST（無輸出）」— 改回報 rc + 輸出
尾行；同時 `/ingest` 改走與 runner 相同的 per-user-token 路徑。

## 安全

- 觸發與查詢都走既有 Device credential（HMAC-pepper），無新 anon 面。
- refresh token 不進 request row（本來就在 `users` 表，由既有 RPC 管理）。
- 同 user 同時最多一件 pending/running；edge function 沿用 3s rate-limit 精神。

## 測試

- regression：`gtow_analyze_api` env-override 模式（mint/re-mint/不觸碰
  tokens.json）、runner 的 claim/stale/owner-gate 邏輯（mock conn）。
- `node --check` 三支 extension JS；`supabase functions deploy` 前 lint。
- E2E 驗收：真機點按鈕 → row 建立 → bot 撈走 → Telegram 收到結果。

## 不做（v1）

- 多用戶 ledger（`ledger_hands` 無 user 欄位）— 非 owner 只會收到明確 error。
- Realtime/LISTEN 即時觸發 — 5s 輪詢已夠。
