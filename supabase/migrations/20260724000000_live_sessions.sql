CREATE TABLE IF NOT EXISTS live_sessions (
    id           bigserial PRIMARY KEY,
    session_key  text UNIQUE NOT NULL,
    chat_id      bigint NOT NULL,
    message_id   bigint,
    page         int NOT NULL DEFAULT 0,
    result_json  jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS live_sessions_created_at_idx
    ON live_sessions (created_at DESC);
