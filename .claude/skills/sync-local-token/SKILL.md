---
name: sync-local-token
description: Sync local `.tokens.json` with the admin's GTO Wizard token from Supabase. Use when regression tests fail with token expiry or after the admin updates their token via `/settoken`.
user_invocable: true
---

# Sync Local Token from DB

Run this command to update `.tokens.json` with the admin's token from the database:

```bash
set -a && source .env && set +a && python3 -c "
import asyncio, os, json
from src.database import Database
from scripts.gto_token import _refresh_access, _save_tokens

async def update():
    db = Database()
    await db.connect()
    admin_id = int(os.getenv('ADMIN_CHAT_ID'))
    refresh = await db.get_user_gto_token(admin_id)
    await db.close()
    if not refresh:
        print('ERROR: No token in DB for admin user', admin_id)
        return
    access = _refresh_access(refresh)
    if not access:
        print('ERROR: DB token cannot be refreshed — admin needs to re-run /settoken')
        return
    _save_tokens({'refresh': refresh, 'access': access})
    print('OK: .tokens.json updated with admin token from DB')

asyncio.run(update())
"
```

After syncing, run regression tests to verify:

```bash
set -a && source .env && set +a && python scripts/regression_test.py
```
