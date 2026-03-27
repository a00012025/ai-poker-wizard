-- Deviations table: one row per hero decision point where GTO was queried.
-- Used for leak detection and coaching memory.

CREATE TABLE deviations (
  id BIGSERIAL PRIMARY KEY,
  chat_id BIGINT NOT NULL REFERENCES users(user_id),
  hand_history_id BIGINT REFERENCES hand_histories(id),
  street TEXT NOT NULL,                   -- preflop/flop/turn/river
  action_index INT NOT NULL DEFAULT 0,   -- multiple decisions per street (e.g., hero bets, gets raised, decides again)
  spot_category TEXT NOT NULL,            -- e.g. "open_raise", "facing_cbet_oop"
  position TEXT NOT NULL,                 -- hero's position
  hero_action TEXT NOT NULL,              -- what hero did (solver code)
  gto_action TEXT NOT NULL,               -- what GTO recommends (solver code)
  hero_freq REAL,                         -- GTO frequency of hero's action (0-100)
  gto_freq REAL,                          -- GTO frequency of recommended action (0-100)
  ev_loss_estimate REAL,                  -- estimated EV loss in bb (if computable)
  board_texture TEXT,                     -- dry/wet/paired/monotone (postflop only)
  effective_bb REAL,                      -- stack depth
  is_deviation BOOLEAN NOT NULL,          -- true if hero_freq < 10%
  meta JSONB,                             -- extra context (IP/OOP, multiway, etc.)
  played_at TIMESTAMPTZ,                  -- when the hand was played (from HH)
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(hand_history_id, street, action_index)
);

CREATE INDEX idx_deviations_chat_spot ON deviations(chat_id, spot_category);
CREATE INDEX idx_deviations_chat_street ON deviations(chat_id, street);
CREATE INDEX idx_deviations_chat_created ON deviations(chat_id, created_at);

-- Enable RLS (app uses direct postgres connection, bypasses RLS)
ALTER TABLE deviations ENABLE ROW LEVEL SECURITY;
