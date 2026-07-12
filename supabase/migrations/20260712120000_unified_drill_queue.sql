-- Unified practice queue (design spec docs/superpowers/specs/2026-07-12-unified-drill-queue-design.md):
-- promote drill_queue from a live-flow-only backlog to a unified practice
-- backlog that also ingests online-hand leaks in two shapes —
--   kind='drill'  : systematic leak (n>=3, total>=3bb)  -> GTOW Trainer drill
--   kind='review' : single-hand disaster (>=5bb)         -> GTOW Analyze review
-- added_by: 'auto' (scan / live flow) | 'manual' (owner adds a line from a review).
-- cleared_at powers the re-open gate (a cleared drill leaf only revives on >=2
-- new post-clear lossy decisions).

ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'drill',        -- 'drill' | 'review'
  ADD COLUMN IF NOT EXISTS ref_hand_id TEXT,                         -- review/manual item's referenced hand
  ADD COLUMN IF NOT EXISTS added_by TEXT NOT NULL DEFAULT 'auto',    -- 'auto' | 'manual'
  ADD COLUMN IF NOT EXISTS cleared_at TIMESTAMPTZ;                   -- qcl clear time (re-open gate)

-- Existing 17 live rows fall on the defaults (kind='drill', added_by='auto') —
-- semantically correct, no backfill needed.

-- Uniqueness is now per-kind: one pending drill row per leaf; one pending
-- review row per referenced hand.
DROP INDEX IF EXISTS idx_drill_queue_pending_leaf;
CREATE UNIQUE INDEX idx_drill_queue_pending_leaf
  ON drill_queue(spot_leaf) WHERE status = 'pending' AND kind = 'drill';
CREATE UNIQUE INDEX idx_drill_queue_pending_review
  ON drill_queue(ref_hand_id) WHERE status = 'pending' AND kind = 'review';

-- The scan looks up any-status review rows by ref_hand_id (復盤過就是過了).
CREATE INDEX IF NOT EXISTS idx_drill_queue_ref_hand
  ON drill_queue(ref_hand_id) WHERE ref_hand_id IS NOT NULL;
