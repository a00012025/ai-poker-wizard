-- Action-line spot taxonomy columns on ledger_decisions (v2).
-- Populated by scripts/backfill_spots.py from the archived raw (no API).
-- North Star §4.2: the spot taxonomy is ours; this is the real classifier
-- replacing the coarse ~15-bucket `family` starting point.

ALTER TABLE ledger_decisions
  ADD COLUMN IF NOT EXISTS spot_category TEXT,       -- RFI/vsOpen/.../flop/turn/river/discarded
  ADD COLUMN IF NOT EXISTS spot_leaf TEXT,           -- the leaf action-line key (primary grouping)
  ADD COLUMN IF NOT EXISTS spot_keys JSONB,          -- hierarchical keys for tree rollup
  ADD COLUMN IF NOT EXISTS hero_cat TEXT,            -- EP/MP/LP/SB/BB
  ADD COLUMN IF NOT EXISTS villain_cat TEXT,
  ADD COLUMN IF NOT EXISTS ip_oop TEXT,              -- IP/OOP (postflop)
  ADD COLUMN IF NOT EXISTS flop_seq TEXT,            -- abbreviated flop action line (turn/river)
  ADD COLUMN IF NOT EXISTS turn_seq TEXT,
  ADD COLUMN IF NOT EXISTS eff_stack TEXT,           -- large/medium/short
  ADD COLUMN IF NOT EXISTS board_suit TEXT,          -- monotone/two_tone/rainbow
  ADD COLUMN IF NOT EXISTS board_conn TEXT,          -- GTOW connectedness
  ADD COLUMN IF NOT EXISTS board_paired TEXT,        -- GTOW pairedness
  ADD COLUMN IF NOT EXISTS discarded BOOLEAN NOT NULL DEFAULT FALSE,   -- limp-involved, unreliable grade
  ADD COLUMN IF NOT EXISTS limp_origin BOOLEAN NOT NULL DEFAULT FALSE; -- postflop limp/iso pot caveat

CREATE INDEX IF NOT EXISTS idx_ledger_decisions_spot_leaf ON ledger_decisions(spot_leaf);
CREATE INDEX IF NOT EXISTS idx_ledger_decisions_spot_cat ON ledger_decisions(spot_category);
