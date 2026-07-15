-- Distinguish intentionally skipped zero-loss details from retryable pending
-- details. The boolean stays for compatibility with existing fidelity tools.

ALTER TABLE ledger_hands
  ADD COLUMN IF NOT EXISTS detail_status TEXT NOT NULL DEFAULT 'pending';

UPDATE ledger_hands
SET detail_status = CASE
  WHEN detail_fetched THEN 'fetched'
  ELSE 'pending'
END;

ALTER TABLE ledger_hands
  DROP CONSTRAINT IF EXISTS ledger_hands_detail_status_check;
ALTER TABLE ledger_hands
  ADD CONSTRAINT ledger_hands_detail_status_check
  CHECK (detail_status IN ('pending', 'fetched', 'skipped_zeroloss'));

DROP INDEX IF EXISTS idx_ledger_hands_detail;
CREATE INDEX idx_ledger_hands_detail_pending
  ON ledger_hands(played_at)
  WHERE source='online' AND detail_status='pending';

CREATE INDEX idx_ledger_hands_detail_skipped_zeroloss
  ON ledger_hands(played_at)
  WHERE source='online' AND detail_status='skipped_zeroloss';
