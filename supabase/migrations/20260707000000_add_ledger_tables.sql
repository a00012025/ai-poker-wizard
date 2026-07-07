-- Ledger tables: grader-agnostic decision ledger (North Star §5.2/§6).
-- N=1 system: no chat_id on ledger tables (single-player, NORTH_STAR §0).

CREATE TABLE ledger_hands (
  id BIGSERIAL PRIMARY KEY,
  gtow_hand_id TEXT NOT NULL UNIQUE,
  played_at TIMESTAMPTZ NOT NULL,
  tournament_id TEXT,
  tournament_name TEXT,
  tournament_buyin NUMERIC,
  file_name TEXT,
  site TEXT,
  position TEXT,
  hero_hand TEXT,
  boards TEXT,
  pot_type TEXT,                    -- GTOW pot type (Preflop/SRP/3bet/...)
  total_players INT,
  preflop_depth_bb REAL,
  total_ev_loss_bb REAL,
  total_ev_loss_pct_pot REAL,
  avg_gto_score REAL,
  winloss_bb REAL,
  hand_correctness TEXT,
  solution_status TEXT,
  session_id BIGINT,                -- FK filled by session rebuild
  raw_path TEXT,                    -- local raw archive path (detail JSON)
  detail_fetched BOOLEAN NOT NULL DEFAULT FALSE,
  ingested_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_ledger_hands_played ON ledger_hands(played_at);
CREATE INDEX idx_ledger_hands_detail ON ledger_hands(detail_fetched) WHERE NOT detail_fetched;
CREATE INDEX idx_ledger_hands_tourney ON ledger_hands(tournament_id);

CREATE TABLE ledger_decisions (
  id BIGSERIAL PRIMARY KEY,
  gtow_hand_id TEXT NOT NULL REFERENCES ledger_hands(gtow_hand_id) ON DELETE CASCADE,
  street TEXT NOT NULL,             -- preflop/flop/turn/river
  decision_idx INT NOT NULL,        -- 0-based hero decision counter within street
  source TEXT NOT NULL DEFAULT 'online',
  grader TEXT NOT NULL DEFAULT 'gtow_analyzer',
  family TEXT NOT NULL,             -- our spot_categorizer taxonomy
  texture TEXT,                     -- our classify_board_texture
  gtow_texture TEXT,                -- GTOW connectedness/pairedness "oesd_possible/not_paired"
  depth_band TEXT NOT NULL,         -- le15 / 15_25 / 25_40 / 40plus
  position TEXT,
  pot_type TEXT,
  facing TEXT,                      -- e.g. "vs_R16.65", "unopened", "checked_to"
  taken_code TEXT,
  best_code TEXT,
  correctness TEXT,                 -- GTOW grade on taken action
  ev_loss_bb REAL,
  ev_loss_pct_pot REAL,
  taken_freq REAL,
  freq_diff REAL,
  gto_score REAL,
  hand_eq REAL,
  pot_bb REAL,
  gametype TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  approx_flags JSONB NOT NULL DEFAULT '[]',
  excluded BOOLEAN NOT NULL DEFAULT FALSE,
  played_at TIMESTAMPTZ,            -- denormalized for time queries
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(gtow_hand_id, street, decision_idx)
);
CREATE INDEX idx_ledger_decisions_family ON ledger_decisions(family, depth_band);
CREATE INDEX idx_ledger_decisions_played ON ledger_decisions(played_at);
CREATE INDEX idx_ledger_decisions_loss ON ledger_decisions(ev_loss_bb) WHERE ev_loss_bb > 0;

CREATE TABLE ledger_sessions (
  id BIGSERIAL PRIMARY KEY,
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ NOT NULL,
  duration_min REAL,
  tournaments JSONB NOT NULL DEFAULT '[]',
  max_concurrent_tables INT,
  hands_count INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE coach_focus (
  id BIGSERIAL PRIMARY KEY,
  week TEXT NOT NULL UNIQUE,        -- ISO week label e.g. "2026-W28"
  families JSONB NOT NULL,          -- [{family, depth_band, rationale...}]
  rationale JSONB,
  prescriptions JSONB,              -- [{label, url}]
  readback JSONB,                   -- next-week delta, filled by following scorecard
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scorecards (
  id BIGSERIAL PRIMARY KEY,
  week TEXT NOT NULL UNIQUE,
  html TEXT,
  data_json JSONB,
  pushed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE ledger_hands ENABLE ROW LEVEL SECURITY;
ALTER TABLE ledger_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ledger_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE coach_focus ENABLE ROW LEVEL SECURITY;
ALTER TABLE scorecards ENABLE ROW LEVEL SECURITY;
