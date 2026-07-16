# Handoff: 永久移除 `.tokens.json` 依賴

**Date:** 2026-07-15

**Completed:** 2026-07-16

**Branch:** `fix/remove-tokens-json`

## 結果

執行期 credential source 已統一為 `users.gto_refresh_token`：

- Telegram 一般請求使用 requesting user 的 DB token，透過 thread-local
  access token 隔離。
- owner-run CLI、snapshot、E2E、regression 在沒有顯式
  `GTOW_REFRESH_TOKEN` 時，以 `OWNER_CHAT_ID` 解析 owner DB token。
- bot 內漏接 per-user token 的 solver request 會 fail closed，不會借用
  owner credential。
- 舊 file-backed auth API、Docker bind mount、操作 skill 與現行文件指示已移除。
- `.gitignore` / `.dockerignore` 仍保留舊檔名作為 **secret tombstone**，避免
  開發機殘留 credential 被 commit 或送入 Docker build context；這不是執行期依賴。

## 為什麼

舊的全域 credential 會建立第二個 GTOW session。GTOW session 上限可能觸發
`Too many sessions` / `FORCED_LOGOUT`，反過來踢掉 owner 瀏覽器 session；而
extension 正是從該瀏覽器 session 同步 token 到 DB。系統因此會破壞自己的
credential source。

## 已修的 auth gaps

1. **重啟後 follow-up rehydrate**：`_ensure_hand_context` 原本在 event-loop
   thread 設 token，再把 solver 丟到 worker thread，`threading.local` 不會跨
   thread。現在 setup / analyze / clear 全部在同一 worker thread。
2. **`/live` subprocess**：bot 先讀 owner 的 DB refresh token，再以
   `GTOW_REFRESH_TOKEN` 傳入 `live_flow.py` subprocess。
3. **ICM game-modes cache miss**：`_load_game_modes` 改走
   `gto_api._get_with_retry`，因此尊重 thread-local 或 owner DB bootstrap。
4. **CLI async safety**：`scripts/gto_owner_token.py` 使用同步 `psycopg2`，避免
   lazy fallback 在既有 event loop 中呼叫 `asyncio.run()`。
5. **E2E explicit parameter**：`e2e_test.py` 不只依賴 solver fallback；它在
   `asyncio.run()` 前 bootstrap owner DB token，並把 refresh token 傳進
   `GeminiSessionManager`。

## 主要檔案

- `scripts/gto_owner_token.py`：owner DB resolver + env bootstrap。
- `scripts/gto_api.py` / `scripts/gtow_analyze_api.py`：集中式 owner fallback、
  bot fail-closed。
- `scripts/gto_token.py`：只保留 per-user mint/cache/signing。
- `src/main_gemini.py`：標記 bot process；owner CLI child subprocess 會移除該標記。
- `src/gemini_session.py`、`src/telegram_bot/bot.py`、`scripts/icm_modes.py`：
  修復三個 auth gaps。

## 驗證證據

- Targeted auth regressions：全部通過。
- Full regression：**782 passed, 0 failed**。
- Owner DB token real mint：成功，且輸出不包含 token。
- `/live --dry-run` real solver smoke：1 hand、2 decisions、2 graded。
- ICM 無 disk cache real API smoke：成功取得 **1129 modes**。
- `python -m compileall -q scripts src`：通過。
- `git diff --check`：通過。

## 操作方式

- owner token 更新：重新登入 GTOW，使用 extension 的「立即同步目前 GTOW
  token」，或在 bot 私訊 `/settoken`。
- owner CLI 驗證：`python scripts/gto_token.py`。
- 不要建立、同步或掛載本地 token credential file。
