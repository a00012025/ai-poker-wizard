-- Per-user GTO Wizard token storage
ALTER TABLE users ADD COLUMN IF NOT EXISTS gto_refresh_token TEXT;
