-- Enable Row Level Security on all public tables.
-- All app access uses direct PostgreSQL connections (asyncpg with postgres role),
-- which bypasses RLS. This protects against unintended PostgREST (anon/authenticated) access.

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hand_histories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.gto_api_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.message_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.token_usage ENABLE ROW LEVEL SECURITY;
