-- Taxonomy convergence (North Star §4.2): the action-line taxonomy
-- (spot_leaf / spot_category, scripts/spot_taxonomy.py) is the single
-- official classifier. The legacy ~15-bucket `family` (+ its `texture`)
-- stops being written by ledger_distill / live_flow; columns stay for
-- historical rows, so the constraint and its index go.
ALTER TABLE ledger_decisions ALTER COLUMN family DROP NOT NULL;
DROP INDEX IF EXISTS idx_ledger_decisions_family;
