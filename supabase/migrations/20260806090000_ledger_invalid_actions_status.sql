-- A hand can remain visible in the GTOW Analyze list while its detail endpoint
-- permanently returns VALIDATION_ERROR / Incorrect actions. Keep the list row
-- for count reconciliation, exclude it from decisions, and stop retrying it.

ALTER TABLE ledger_hands
  DROP CONSTRAINT IF EXISTS ledger_hands_detail_status_check;

ALTER TABLE ledger_hands
  ADD CONSTRAINT ledger_hands_detail_status_check
  CHECK (detail_status IN (
    'pending',
    'fetched',
    'skipped_zeroloss',
    'skipped_invalid_actions'
  ));
