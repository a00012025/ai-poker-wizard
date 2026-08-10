-- Queue URL refresh can discover that a legacy all-depth pending drill and an
-- existing band-specific pending drill are the same prescription.  The source
-- rows are merged into the band-specific row and the duplicate is retained as
-- cleared history with this explicit audit reason.
ALTER TABLE drill_queue DROP CONSTRAINT IF EXISTS drill_queue_clear_reason_check;
ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_clear_reason_check
    CHECK (clear_reason IS NULL OR clear_reason IN
           ('completed', 'mistake', 'skipped', 'resend', 'drill_passed',
            'scope_dedupe'));
