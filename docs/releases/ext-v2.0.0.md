# AI Poker Wizard GTOW 自動同步 Extension v2.0.0

Extension v2 以一次性 Telegram 配對與自動同步，取代舊版複製、貼上 token 的流程。

## 新功能

- 使用 Telegram 私訊 `/pair` 指令與 Extension popup 配對
- 自動偵測並同步最新 GTOW token
- 提供「立即同步」與清楚的同步狀態
- 每台瀏覽器使用獨立、可撤銷的 device credential
- 同一帳號可配對多台 Desktop Chrome，並由每台已登入的瀏覽器自動同步
- 新增 Telegram `/devices`、`/revoke` 與更安全的 `/logout`
- 保留 `/settoken` 作為手動備援
- 同步前向 GTOW 驗證 token，並防止舊裝置回寫較舊 token
- Popup 開啟時會確認遠端撤銷狀態；離線解除失敗不會遺失本機 credential

## 安裝方式

1. 下載下方 `ai-poker-wizard-gtow-sync-v2.0.0.zip` 並解壓縮。
2. Chrome 開啟 `chrome://extensions`。
3. 啟用右上角「開發人員模式」。
4. 點擊「載入未封裝項目」，選擇解壓縮後的資料夾。
5. 開啟 [@ai_poker_wizard_bot](https://t.me/ai_poker_wizard_bot)，輸入 `/pair`，
   將五分鐘配對碼貼入 Extension popup。
6. 登入 [GTO Wizard](https://app.gtowizard.com)；後續同步會自動進行。

Extension 不會顯示或將原始 GTOW token 保存到 Chrome storage。
