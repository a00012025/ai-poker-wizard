-- Ingest requests gain a mode: 'incremental' (default, 30-day re-sweep) or
-- 'full' (backfill all history since the ledger epoch). The extension ♠-sync
-- button and the daily job keep inserting without a mode → default
-- 'incremental'; only the owner's /fullingest confirmation enqueues 'full'.
--
-- The existing one-open-request-per-user unique index is mode-agnostic, so it
-- already enforces "at most one ingest running per user regardless of mode".

ALTER TABLE public.gtow_ingest_requests
  ADD COLUMN mode text NOT NULL DEFAULT 'incremental'
    CHECK (mode IN ('incremental', 'full'));
