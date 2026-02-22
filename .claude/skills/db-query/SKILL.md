---
name: db-query
description: Query the Supabase database (read-only). Use when checking users, hand histories, cache entries, or any DB data. Triggers on "query db", "check db", "查資料庫", "DB 有什麼", "有幾個用戶".
user_invocable: true
---

# Query Database (Read-Only)

Query the AI Poker Wizard Supabase database. **Read-only** — all write operations are blocked.

## Safety Rules

**BLOCKED operations** — NEVER execute SQL containing any of these keywords:
- `DELETE`, `DROP`, `TRUNCATE`, `UPDATE`, `ALTER`, `INSERT`, `CREATE`, `GRANT`, `REVOKE`

If the user asks to modify data, **refuse** and explain this skill is read-only. Suggest modifying data through the application code or Supabase dashboard instead.

## Database Schema

### `users`
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | BIGINT PK | Telegram user ID |
| `username` | TEXT | Telegram username |
| `name` | TEXT | Display name |
| `is_active` | BOOLEAN | Account active flag |
| `gto_refresh_token` | TEXT | GTO Wizard refresh JWT (sensitive — do NOT print full token) |
| `created_at` | TIMESTAMPTZ | Registration time |

### `hand_histories`
| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL PK | Auto-increment ID |
| `chat_id` | BIGINT | Telegram chat ID |
| `hand_id` | TEXT | GGPoker hand ID (e.g. TM5600279272) |
| `hand_data` | JSONB | Full parsed hand JSON |
| `uploaded_at` | TIMESTAMPTZ | Upload time |

Indexes: `idx_hh_chat_id`, `idx_hh_hand_id`

### `gto_api_cache`
| Column | Type | Description |
|--------|------|-------------|
| `cache_key` | TEXT PK | API request cache key |
| `response` | JSONB | Cached API response |
| `is_null` | BOOLEAN | True if API returned no data (204/403) |

## How to Query

Run a Python snippet that connects via `asyncpg` using `SUPABASE_CONN` from `.env`:

```bash
set -a && source .env && set +a && python3 -c "
import asyncio, asyncpg, os, json
async def main():
    conn = await asyncpg.connect(os.environ['SUPABASE_CONN'], statement_cache_size=0)
    rows = await conn.fetch('''
        YOUR_SELECT_QUERY_HERE
    ''')
    for r in rows:
        print(dict(r))
    await conn.close()
asyncio.run(main())
"
```

## Privacy

- When displaying user data, mask `gto_refresh_token` — show only `eyJ...xxxx` (first 3 + last 4 chars)
- Do not log or display full tokens

## Common Queries

**List all users:**
```sql
SELECT user_id, username, is_active, gto_refresh_token IS NOT NULL as has_token, created_at FROM users ORDER BY created_at
```

**Count hands per user:**
```sql
SELECT chat_id, COUNT(*) as hands, MIN(uploaded_at) as first, MAX(uploaded_at) as last FROM hand_histories GROUP BY chat_id ORDER BY last DESC
```

**Cache stats:**
```sql
SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE is_null) as null_entries FROM gto_api_cache
```
