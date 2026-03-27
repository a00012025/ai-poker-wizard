-- Analysis snapshots for E2E regression testing.
-- Captures full pipeline state: input → parse → GTO output → coaching.
CREATE TABLE analysis_snapshots (
    id BIGSERIAL PRIMARY KEY,
    hand_id TEXT NOT NULL UNIQUE,
    chat_id BIGINT,
    source_type TEXT NOT NULL,          -- 'text' | 'image'
    user_input TEXT,                     -- original user message / caption
    image_data BYTEA,                   -- raw screenshot bytes (NULL for text)
    parsed_json JSONB NOT NULL,         -- what Gemini/OCR parsed
    expected_json JSONB,                -- corrected parse (set during bug fix, NULL until then)
    gto_text TEXT NOT NULL,             -- analyze_hand_full()["text"]
    gto_compact TEXT,                   -- analyze_hand_full()["text_compact"]
    coaching_text TEXT,                  -- final LLM coaching response (reference only)
    is_regression BOOLEAN DEFAULT FALSE,-- flagged for regression testing
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_snapshots_hand_id ON analysis_snapshots (hand_id);
CREATE INDEX idx_snapshots_regression ON analysis_snapshots (is_regression) WHERE is_regression = TRUE;
