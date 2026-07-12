#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
set -a
source .env
set +a

PROJECT_REF=${SUPABASE_PROJECT_REF:-ivtfzwsytdkdqolzxhkh}
: "${SUPABASE_PASSWORD:?SUPABASE_PASSWORD is required}"
: "${GTOW_SYNC_PEPPER:?GTOW_SYNC_PEPPER is required}"
if (( ${#GTOW_SYNC_PEPPER} < 32 )); then
  echo "GTOW_SYNC_PEPPER must be at least 32 characters" >&2
  exit 1
fi

supabase link --project-ref "$PROJECT_REF" --password "$SUPABASE_PASSWORD"
supabase db push
supabase secrets set --project-ref "$PROJECT_REF" \
  GTOW_SYNC_PEPPER="$GTOW_SYNC_PEPPER"
supabase functions deploy gtow-sync --project-ref "$PROJECT_REF" --no-verify-jwt
curl -fsS "https://${PROJECT_REF}.supabase.co/functions/v1/gtow-sync/health"
echo
docker compose build bot
docker compose up -d bot
echo "GTOW token sync deployment verified."
