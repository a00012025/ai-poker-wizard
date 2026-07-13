#!/bin/bash
set -e
cd /home/harry/ai-poker-wizard

# Source env for Supabase access token
set -a && source .env && set +a

# The Supabase v2 shim and its Go CLI must travel together. Resolve the
# companion before git/database/container side effects so a broken install
# fails closed instead of interrupting deployment halfway through.
source scripts/supabase_cli.sh
configure_supabase_cli

git pull

# Run Supabase migrations (project already linked)
supabase db push

# Resumable post-migration data upgrade.  The default selector only touches
# hands missing the current taxonomy/depth contract, and exits non-zero if any
# honest online rows remain partial.  Run before the bot/weekly job can publish
# a mixed-schema training focus.
python scripts/backfill_spots.py

# Token 同步是本版 Bot 公開功能；缺少 secret 時禁止部署半套版本。
: "${GTOW_SYNC_PEPPER:?GTOW_SYNC_PEPPER 未設定，停止部署}"
if (( ${#GTOW_SYNC_PEPPER} < 32 )); then
  echo "GTOW_SYNC_PEPPER 至少需要 32 個字元" >&2
  exit 1
fi
PROJECT_REF="${SUPABASE_PROJECT_REF:-ivtfzwsytdkdqolzxhkh}"
supabase secrets set --project-ref "$PROJECT_REF" \
  GTOW_SYNC_PEPPER="$GTOW_SYNC_PEPPER"
supabase functions deploy gtow-sync \
  --project-ref "$PROJECT_REF" --no-verify-jwt
curl -fsS "https://${PROJECT_REF}.supabase.co/functions/v1/gtow-sync/health"
echo

# Build and deploy container
docker compose build
docker compose up -d
