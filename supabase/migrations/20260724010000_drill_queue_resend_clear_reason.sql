-- Single-hand live resend can clear auto/live queue rows whose only source hand
-- was replaced. Track that lifecycle separately from owner-completed/skipped.

ALTER TABLE drill_queue
  DROP CONSTRAINT IF EXISTS drill_queue_clear_reason_check;

ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_clear_reason_check
    CHECK (clear_reason IS NULL OR clear_reason IN ('completed', 'mistake', 'skipped', 'resend'));
