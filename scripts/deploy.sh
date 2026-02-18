#!/bin/bash
set -e
cd /home/harry/ai-poker-wizard

# Source env for Supabase access token
set -a && source .env && set +a

git pull

# Run Supabase migrations (project already linked)
supabase db push

# Build and deploy container
docker compose build
docker compose up -d
