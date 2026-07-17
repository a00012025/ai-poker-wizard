# AI Poker Wizard — GTOW 自動同步 Extension

Chrome Extension 只需與 Telegram bot 配對一次；之後每次開啟
`app.gtowizard.com`，Extension 都會自動同步目前的 GTO Wizard refresh token。
v2.1 起，Extension popup 提供「♠ 同步手牌到 DB」按鈕：一鍵把新上傳的
Analyze 手牌攝取進 ledger，popup 顯示進度，完成後 Telegram 也會收到結果。

## 安裝

1. 到最新 GitHub Release 下載 `ai-poker-wizard-gtow-sync-v2.2.0.zip`。
2. 解壓縮下載檔案。
3. 開啟 `chrome://extensions`，啟用右上角「開發人員模式」。
4. 點擊「載入未封裝項目」，選擇剛才解壓縮的資料夾。
5. 將 **AI Poker Wizard - GTOW Sync** 固定在 Chrome 工具列。

## 第一次配對

1. 私訊 [@ai_poker_wizard_bot](https://t.me/ai_poker_wizard_bot)，輸入 `/pair`。
2. 五分鐘內將 12 碼配對碼貼到 Extension popup。
3. 正常開啟並登入 GTO Wizard。

Popup 會顯示最後成功同步時間。需要重試時可按「立即同步目前 GTOW token」。

## 手動 `/settoken` 備援

若自動同步服務暫時失敗，先切到已登入的 GTO Wizard 分頁，打開 Extension popup，
點「複製 `/settoken` 指令」，再直接貼到 Telegram bot。Extension 只會在點擊當下
讀取 token 並把完整指令寫入系統剪貼簿，不會顯示或保存 token；貼上後建議複製
其他文字覆蓋剪貼簿內容。

## 一鍵手牌同步（v2.1）

在 GTOW 上傳手牌後，開啟 Extension popup 並點「♠ 同步手牌到 DB」：Extension
會先把當前 token 同步上去（**手動觸發一律強制覆蓋伺服器版本** — 你當下登入中的 token
必定有效，即使它的 iat 比較舊），再排入攝取佇列；bot 幾秒內接手，用你的
token 跑 incremental ingest（對數不符時自動全量補齊）。進度顯示在 popup，
結果同時發到 Telegram。
若剛上傳完顯示「沒有新手牌」，是 GTOW 還在處理檔案，稍後再點一次即可。
Telegram `/devices` 可查看已配對瀏覽器；`/revoke <裝置ID>` 可撤銷單一裝置；
`/logout` 會移除 GTOW token 並撤銷所有已配對裝置。

同一個帳號可以配對多台 Desktop Chrome。請在每台 Chrome 依序取得新的 `/pair`
配對碼；每台裝置各自登入 GTOW 後都會自動同步到 Bot。Extension 不會把 A 電腦的
token 注入 B 電腦，因此不會替另一台瀏覽器自動登入。

## 隱私與安全

- Extension 只會在 `app.gtowizard.com` 讀取 `localStorage.user_refresh`。
- Extension 不會顯示、記錄，或將原始 GTOW token 寫入 Chrome storage；只有使用者
  主動點擊手動備援按鈕時，才會把 `/settoken` 指令寫入系統剪貼簿。
- Chrome storage 只保存可撤銷的 AI Poker Wizard device credential、token fingerprint
  與同步時間。
- 同步全程使用 HTTPS，初次連結必須使用 Telegram 產生的一次性配對碼。
- 同步服務會向 GTOW 驗證 token，且拒絕舊裝置以較舊 token 覆蓋目前版本。
- 若「解除這台裝置」遇到網路錯誤，Extension 會保留 credential 並提示重試，
  不會假裝已解除。
- `/settoken` 仍保留為手動備援方式。

## 開發檢查

```bash
node --check chrome-extension/background.js
node --check chrome-extension/content.js
node --check chrome-extension/popup.js
bash scripts/package_extension.sh
```

API URL 位於 `config.js`。如果更換 Supabase project，打包前必須同時修改
`config.js` 與 `manifest.json` 的 host permissions。
