-- Sparse EV-backed tendency attached to a queue prescription.
-- NULL means no robust dominant direction; the UI must render nothing.
ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS bias_key TEXT,
  ADD COLUMN IF NOT EXISTS bias_direction TEXT,
  ADD COLUMN IF NOT EXISTS bias_n INT,
  ADD COLUMN IF NOT EXISTS bias_ev_loss_bb REAL,
  ADD COLUMN IF NOT EXISTS bias_share REAL;

ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_bias_direction_check
  CHECK (bias_direction IS NULL OR bias_direction IN
         ('overfold', 'overcall', 'overraise', 'too_passive'));
