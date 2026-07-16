-- Bind a queue prescription to the GTOW Drill selected by identical Trainer
-- settings.  Opening a drill detail menu establishes the baseline; practice
-- sessions after that timestamp are the current attempt, while lifetime GTOW
-- totals remain available for context.

ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS gtow_drill_id UUID,
  ADD COLUMN IF NOT EXISTS gtow_drill_name TEXT,
  ADD COLUMN IF NOT EXISTS gtow_settings_hash TEXT,
  ADD COLUMN IF NOT EXISTS gtow_drill_synced_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS gtow_training_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS gtow_baseline_totals JSONB,
  ADD COLUMN IF NOT EXISTS gtow_target_hands INT NOT NULL DEFAULT 30,
  ADD COLUMN IF NOT EXISTS gtow_target_score REAL NOT NULL DEFAULT 0.90,
  ADD COLUMN IF NOT EXISTS clear_reason TEXT;

ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_gtow_target_hands_positive
    CHECK (gtow_target_hands > 0),
  ADD CONSTRAINT drill_queue_gtow_target_score_range
    CHECK (gtow_target_score >= 0 AND gtow_target_score <= 1),
  ADD CONSTRAINT drill_queue_clear_reason_check
    CHECK (clear_reason IS NULL OR clear_reason IN ('completed', 'mistake', 'skipped'));

CREATE INDEX IF NOT EXISTS idx_drill_queue_gtow_settings_hash
  ON drill_queue(gtow_settings_hash)
  WHERE gtow_settings_hash IS NOT NULL;
