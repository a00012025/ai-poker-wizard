# GTOW Token 自動同步部署手冊

本手冊用於部署 Chrome Extension v2 所需的 Supabase Edge Function。
不需要自訂 domain，也不需要公開 Docker host 的任何 port。

## 架構

```text
Chrome Extension
  └─ HTTPS → Supabase Edge Function（gtow-sync）
                ├─ 一次性 pairing RPC
                ├─ device authentication
                └─ 更新 users.gto_refresh_token

Telegram bot container
  ├─ /pair 在 Postgres 建立一次性配對碼
  ├─ /devices 與 /revoke 管理裝置
  └─ 透過既有 DB layer 讀取同步後的 token
```

## 前置需求

- Supabase CLI 已使用 `SUPABASE_ACCESS_TOKEN` 登入
- `.env` 已設定 `SUPABASE_CONN` 與 `SUPABASE_PASSWORD`
- 既有 Docker bot 可以正常部署
- Bot 與 Edge Function 共用同一個、至少 32 字元的 `GTOW_SYNC_PEPPER`

配對碼與裝置 credential 使用不同的 HMAC domain，避免跨用途重放。

只需生成一次共用 secret：

```bash
openssl rand -base64 48
```

將完整且相同的值加入 Docker host `.env`：

```dotenv
GTOW_SYNC_PEPPER=<剛才生成的值>
```

此值絕對不能 commit。

### Secret 輪替

目前裝置 credential 是以單一 pepper 的 HMAC 保存。輪替 pepper 會讓既有裝置失效，
因此請在維護時段依序執行：通知使用者、更新 Docker 與 Supabase secret、重新部署
Edge Function、要求使用者在 Telegram 私訊 `/pair` 重新配對。不要同時保留新舊
pepper；舊裝置應視為已撤銷。

## 第一次部署

目前正式 Extension 使用的 Supabase project ref 是 `ivtfzwsytdkdqolzxhkh`。

```bash
set -a && source .env && set +a

supabase link \
  --project-ref ivtfzwsytdkdqolzxhkh \
  --password "$SUPABASE_PASSWORD"

supabase db push

supabase secrets set \
  --project-ref ivtfzwsytdkdqolzxhkh \
  GTOW_SYNC_PEPPER="$GTOW_SYNC_PEPPER"

supabase functions deploy gtow-sync \
  --project-ref ivtfzwsytdkdqolzxhkh \
  --no-verify-jwt

docker compose build bot
docker compose up -d bot
```

也可以執行已包含檢查的部署腳本：

```bash
bash scripts/deploy_token_sync.sh
```

## 驗證

```bash
curl -fsS \
  https://ivtfzwsytdkdqolzxhkh.supabase.co/functions/v1/gtow-sync/health

docker compose logs --tail=100 bot
```

接著執行使用者流程 smoke test：

1. 在 Telegram bot 輸入 `/pair`。
2. 將配對碼貼到未封裝 Extension popup。
3. 開啟並登入 GTO Wizard。
4. 確認 popup 顯示同步成功。
5. 輸入 `/devices`，確認瀏覽器與最後同步時間存在。
6. 透過 bot 執行一次 solver 查詢。

配對與 token 指令只允許在 Telegram 私訊執行。Edge Function 會向 GTOW refresh
endpoint 驗證候選 token，並以 JWT `iat` 拒絕舊裝置覆蓋較新的 token。過期配對資料
保留 7 天、同步稽核資料保留 180 天，配對時會順便清理。

## 發布 Extension 新版本

```bash
ASSET=$(bash scripts/package_extension.sh)
gh release create ext-v2.0.0 "$ASSET" \
  --target main \
  --title "AI Poker Wizard GTOW 自動同步 Extension v2.0.0" \
  --notes-file docs/releases/ext-v2.0.0.md
```

Edge Function 尚未部署成功前，不得發布指向該 API 的 Extension asset。

## 回滾

1. 使用 `/revoke` 或 `/logout` 撤銷受影響裝置。
2. 重新部署前一個 commit 的 Edge Function。
3. 保留 `/settoken` 作為手動備援。
4. 新資料表均為 additive migration；事故期間不要直接刪表。
