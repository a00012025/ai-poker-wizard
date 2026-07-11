-- Live flow v1 (North Star §5.1 stream 3): offline hands enter the ledger
-- (source='live', grader='own_pipeline') and deviated action lines feed a
-- practice queue that the weekly training plan drains.

-- ledger_hands: source discrimination + raw capture for re-grading (§5.2
-- grader-agnostic: keep the original text + parse so judgments can be redone).
ALTER TABLE ledger_hands
  ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'online',
  ADD COLUMN IF NOT EXISTS raw_text TEXT,
  ADD COLUMN IF NOT EXISTS parsed_json JSONB,
  ADD COLUMN IF NOT EXISTS intent_tag TEXT;          -- uncertain / curious (線下流意圖標籤)

CREATE INDEX IF NOT EXISTS idx_ledger_hands_source ON ledger_hands(source);

-- Practice queue: deviated action lines (spot leaves) awaiting drill.
-- One pending row per leaf; repeat offenders accumulate source_hands/n_sources.
CREATE TABLE drill_queue (
  id BIGSERIAL PRIMARY KEY,
  spot_leaf TEXT NOT NULL,
  spot_category TEXT,
  label TEXT,                                        -- zh human description of the line
  drill_url TEXT,
  status TEXT NOT NULL DEFAULT 'pending',            -- pending / prescribed / cleared
  source TEXT NOT NULL DEFAULT 'live',
  source_hands JSONB NOT NULL DEFAULT '[]',          -- [{hand_id, street, ev_loss_bb}]
  n_sources INT NOT NULL DEFAULT 1,
  total_ev_loss_bb REAL,
  prescribed_week TEXT,                              -- ISO week when surfaced in the plan
  first_added TIMESTAMPTZ DEFAULT NOW(),
  last_added TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_drill_queue_pending_leaf ON drill_queue(spot_leaf) WHERE status = 'pending';
CREATE INDEX idx_drill_queue_status ON drill_queue(status);

ALTER TABLE drill_queue ENABLE ROW LEVEL SECURITY;
