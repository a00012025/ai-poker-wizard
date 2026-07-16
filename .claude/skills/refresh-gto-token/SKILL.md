---
name: refresh-gto-token
description: Refresh the owner's GTO Wizard DB token after expiry/FORCED_LOGOUT. Use for "更新 GTO token", "refresh gto token", "GTO Wizard 過期", or login tasks.
user_invocable: true
---

# Refresh GTO Wizard Owner Token

The database is the only refresh-token store. Never create a local token file.

## Process

1. Open the normal system Chrome at `https://app.gtowizard.com/login` and let
   the owner complete login. A fresh login clears the stale/FORCED_LOGOUT
   session state.
2. Open the AI Poker Wizard extension and click **立即同步目前 GTOW token**.
   Manual sync force-overrides the DB stale-iat guard because the currently
   logged-in browser token is authoritative.
3. Verify the owner DB token can mint access without printing either secret:

   ```bash
   python scripts/gto_token.py
   ```

   Expected: `OK: owner DB token minted access (length=...)`.
4. If extension sync is unavailable, use the bot's private `/settoken <refresh>`
   flow. Never paste a token in a group, logs, source files, or shell history.

The owner is resolved through `OWNER_CHAT_ID`; without it, resolution succeeds
only when exactly one active user exists.
