-- Solver responses live in the bind-mounted .gto_cache personal solve library.
-- scripts/deploy.sh exports and verifies every row while the bot is stopped
-- before this migration is applied.
DROP TABLE IF EXISTS public.gto_api_cache;
