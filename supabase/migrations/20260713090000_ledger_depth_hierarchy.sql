-- Decision-level stack truth + stable learning-family rollup.
--
-- played_depth_bb: physical/list-row stack retained for audit.
-- solver_depth_bb: GTOW game-point depth that actually graded the decision.
-- depth_band/eff_stack are backfilled from solver_depth_bb (falling back to
-- played depth only when the game point omitted depth).

ALTER TABLE ledger_decisions
  ADD COLUMN IF NOT EXISTS played_depth_bb REAL,
  ADD COLUMN IF NOT EXISTS solver_depth_bb REAL,
  ADD COLUMN IF NOT EXISTS spot_parent TEXT;

CREATE INDEX IF NOT EXISTS idx_ledger_decisions_spot_parent
  ON ledger_decisions(spot_parent);

CREATE INDEX IF NOT EXISTS idx_ledger_decisions_training_confidence
  ON ledger_decisions(played_at, spot_parent)
  WHERE source='online' AND NOT excluded AND NOT discarded AND confidence >= 0.8;
