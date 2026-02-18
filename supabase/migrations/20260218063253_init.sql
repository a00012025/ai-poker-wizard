-- Initial schema for AI Poker Wizard

CREATE TABLE IF NOT EXISTS users (
    user_id    BIGINT PRIMARY KEY,
    username   TEXT,
    name       TEXT,
    is_active  BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hand_histories (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     BIGINT NOT NULL,
    hand_id     TEXT NOT NULL,
    hand_data   JSONB NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hh_chat_id ON hand_histories (chat_id);
CREATE INDEX IF NOT EXISTS idx_hh_hand_id ON hand_histories (hand_id);
