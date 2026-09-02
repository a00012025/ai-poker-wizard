ALTER TABLE ledger_decisions
  ADD COLUMN IF NOT EXISTS strategy_context TEXT NOT NULL DEFAULT 'chipev';

ALTER TABLE ledger_decisions
  DROP CONSTRAINT IF EXISTS ledger_decisions_strategy_context_check;

ALTER TABLE ledger_decisions
  ADD CONSTRAINT ledger_decisions_strategy_context_check
  CHECK (strategy_context IN ('chipev', 'icm', 'icm_postflop_chipev'));

CREATE INDEX IF NOT EXISTS idx_ledger_decisions_strategy_context
  ON ledger_decisions(strategy_context);
