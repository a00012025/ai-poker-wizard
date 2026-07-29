-- Close the only public table reported by Supabase's Security Advisor as
-- reachable without Row-Level Security.
--
-- live_sessions contains Telegram identifiers and the full parsed live-session
-- payload. The bot reads and writes it through SUPABASE_CONN (direct Postgres);
-- no browser or authenticated Data API client needs access. Therefore the
-- correct public API policy is deny-by-default: enable RLS and create no
-- anon/authenticated policies.

BEGIN;

ALTER TABLE public.live_sessions ENABLE ROW LEVEL SECURITY;

-- Defense in depth: even if RLS is disabled accidentally later, the public API
-- roles still have no table or sequence privileges.
REVOKE ALL ON TABLE public.live_sessions FROM anon, authenticated;
REVOKE ALL ON SEQUENCE public.live_sessions_id_seq FROM anon, authenticated;

-- Migrations run as postgres. New public tables must opt into Data API access
-- explicitly instead of inheriting full CRUD grants and relying on a developer
-- to remember RLS afterward.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE ALL ON TABLES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE ALL ON SEQUENCES FROM anon, authenticated;

COMMIT;
