-- Weekly-plan freshness scheduling (design spec
-- docs/superpowers/specs/2026-07-29-weekly-plan-freshness-design.md).
--
-- The weekly plan used to re-prescribe whatever it prescribed last week: a row
-- stayed 'prescribed' forever until a manual ✔, so W28/W29 items were still
-- taking W30's slots. Track how often and how recently each row was actually
-- put in front of the owner, so the scheduler can prefer fresh work and rotate
-- the backlog instead of replaying it (§14.2 still holds: nothing is dropped,
-- the backlog just stops monopolising the plan).

ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS surfaced_count     INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_surfaced_at   TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_surfaced_week TEXT;

-- Rows already prescribed have been surfaced at least once. prescribed_week
-- keeps its existing meaning (the week of the FIRST prescription) so that
-- "第 N 次" can be computed from surfaced_count without losing that anchor.
--
-- Backfill the timestamp from prescribed_week (the Sunday the plan went out),
-- NOT from last_added: the weekly scan bumps last_added whenever it merges a
-- new source hand into an open row, which would make a long-ignored row look
-- freshly seen and send it to the back of the rotation.
UPDATE drill_queue
   SET surfaced_count = 1,
       last_surfaced_week = prescribed_week,
       last_surfaced_at = to_date(prescribed_week, 'IYYY"-W"IW') + interval '6 days'
 WHERE prescribed_week IS NOT NULL AND surfaced_count = 0;

-- 'drill_passed' = auto-closed because the bound GTOW Drill attempt met both
-- targets. Kept distinct from the manual ✔ ('completed') for auditability.
ALTER TABLE drill_queue DROP CONSTRAINT IF EXISTS drill_queue_clear_reason_check;
ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_clear_reason_check
    CHECK (clear_reason IS NULL OR clear_reason IN
           ('completed', 'mistake', 'skipped', 'resend', 'drill_passed'));

CREATE INDEX IF NOT EXISTS idx_drill_queue_open_surfaced
  ON drill_queue(last_surfaced_at NULLS FIRST)
  WHERE status IN ('pending', 'prescribed');
