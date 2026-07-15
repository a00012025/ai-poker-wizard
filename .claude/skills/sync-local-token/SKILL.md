---
name: sync-local-token
description: Deprecated alias for syncing the owner's current GTO Wizard browser token into users.gto_refresh_token. No local token file is created.
user_invocable: true
---

# Sync Owner GTO Token to DB

This skill is retained as a compatibility alias. Authentication is DB-only.

1. In the logged-in GTO Wizard browser, open the AI Poker Wizard extension.
2. Click **立即同步目前 GTOW token**. Manual sync force-overrides stale-iat.
3. Verify:

   ```bash
   python scripts/gto_token.py
   ```

If browser sync is unavailable, use the private bot command
`/settoken <refresh>`. Do not create or synchronize a local credential file.
