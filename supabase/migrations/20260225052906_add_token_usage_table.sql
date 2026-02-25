CREATE TABLE IF NOT EXISTS token_usage (
  id BIGSERIAL PRIMARY KEY,
  chat_id BIGINT NOT NULL,
  request_type TEXT NOT NULL,       -- 'hand_analysis', 'image_analysis', 'follow_up'
  model TEXT NOT NULL,              -- e.g. 'gemini-2.0-flash', 'gemini-2.5-pro'
  prompt_tokens INT NOT NULL DEFAULT 0,
  completion_tokens INT NOT NULL DEFAULT 0,
  cached_tokens INT NOT NULL DEFAULT 0,
  thinking_tokens INT NOT NULL DEFAULT 0,
  total_tokens INT NOT NULL DEFAULT 0,
  api_calls INT NOT NULL DEFAULT 1, -- number of Gemini API calls in this request
  latency_ms INT,                   -- total request latency in milliseconds
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_token_usage_chat_id ON token_usage (chat_id);
CREATE INDEX idx_token_usage_created_at ON token_usage (created_at);
