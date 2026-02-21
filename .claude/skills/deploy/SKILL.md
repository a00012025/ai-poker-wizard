---
name: deploy
description: Use when deploying AI Poker Wizard bot to production, updating the running container, or troubleshooting deployment issues. Triggers on "deploy", "部署", "上線", "restart bot", "重啟".
user_invocable: true
---

# Deploy AI Poker Wizard

## Prerequisites

Before deploying, ensure:

1. **`.env`** exists at project root with all required vars:
   - `BOT_TOKEN` — Telegram bot token
   - `GEMINI_API_KEY` — Gemini API key
   - `SUPABASE_CONN` — Supabase pooler connection string (transaction mode, port 6543)
   - `SUPABASE_ACCESS_TOKEN` — Supabase personal access token (for CLI migrations)
   - `ALLOWED_USERS` — comma-separated Telegram user IDs
   - `ADMIN_CHAT_ID` — admin Telegram user ID for token expiry alerts

2. **`.tokens.json`** exists at project root with valid GTO Wizard tokens:
   ```json
   {"refresh": "eyJ...", "access": "eyJ..."}
   ```

3. **`.game_modes_cache.json`** exists (can be empty `{}` — will populate on first use)

4. **Docker** and **docker compose** are installed

5. **Supabase CLI** installed (`supabase --version`). Project already linked (`supabase/` dir).

## Deploy Steps

### Step 1: Run regression tests

```bash
set -a && source .env && set +a && python scripts/regression_test.py
```

All non-API tests must pass. 429 rate-limit failures from GTO Wizard API are expected and OK to ignore.

### Step 2: Commit and push changes

Only if there are uncommitted changes. Do NOT commit `.env` or `.tokens.json`.

### Step 3: Build and deploy

```bash
bash scripts/deploy.sh
```

This runs:
1. `git pull`
2. `supabase db push` — apply pending migrations to remote DB
3. `docker compose build && docker compose up -d` — rebuild and restart container

The old container receives SIGTERM, finishes any in-flight analysis (up to 5 min grace period), then the new container starts.

### Step 4: Verify

```bash
# Check container is running
docker compose ps

# Check startup logs
docker compose logs --tail=30

# Verify DB connection (look for "Database tables verified" in logs)
docker compose logs | grep -i database
```

## Database Migrations (Supabase CLI)

Schema is managed by Supabase CLI. Migration files live in `supabase/migrations/`.
The project is already linked to `ivtfzwsytdkdqolzxhkh` (ap-northeast-2).

### Creating a new migration

```bash
supabase migration new <name>
```

This creates `supabase/migrations/<timestamp>_<name>.sql`. Write your DDL in that file.

### Applying migrations to remote DB

```bash
set -a && source .env && set +a
supabase db push
```

This is also run automatically by `scripts/deploy.sh`. Uses the linked project — requires `SUPABASE_ACCESS_TOKEN` in env.

### Checking migration status

```bash
set -a && source .env && set +a
supabase migration list
```

### NEVER use psql directly

Do NOT run `psql "$SUPABASE_CONN" -f migration.sql` to apply migrations. This bypasses Supabase's migration tracking and `supabase db push` will fail or re-apply. Always use `supabase db push` — it tracks which migrations have been applied.

### Important: adding tables

If you add a new table, also add its name to `_REQUIRED_TABLES` in `src/database.py` so the startup check catches missing migrations.

## What Happens on Startup

1. Bot connects to Supabase via `asyncpg` pool (transaction pooler, `statement_cache_size=0`)
2. Verifies required tables exist (fails fast with clear error if migrations haven't been applied)
3. `ALLOWED_USERS` are seeded into `users` table with `is_active=TRUE`
4. Bot starts polling Telegram for updates

## Troubleshooting

### Bot won't start — "Table not found"
Migrations haven't been applied. Run:
```bash
set -a && source .env && set +a
supabase db push
```

### Bot won't start — other errors
```bash
docker compose logs --tail=50
```

### GTO Wizard token expired
Admin gets a Telegram notification. Update `.tokens.json` on the host — the file is volume-mounted, so the bot picks it up on next API call without restart.

### DB connection failed
Check `SUPABASE_CONN` in `.env`. Uses Supabase transaction pooler (port 6543, IPv4).
Format: `postgresql://postgres.<ref>:<password>@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres`

### Rollback
```bash
git log --oneline -5          # find previous commit
git checkout <commit>
docker compose build && docker compose up -d
```

## Architecture

```
Host                          Container (/app)
─────────────────────────────────────────────
.env ──── env_file ────────→  environment vars
.tokens.json ── volume ───→  /app/.tokens.json
.game_modes_cache.json ───→  /app/.game_modes_cache.json
logs/ ──── volume ────────→  /app/logs/
```

- `stop_grace_period: 5m` — in-flight HH analysis (up to 10 min timeout) finishes before container dies
- `restart: unless-stopped` — auto-restart on crash
- DB schema managed by Supabase CLI migrations (`supabase/migrations/`)
- DB connection uses Supabase transaction pooler (IPv4, port 6543, `statement_cache_size=0`)
