-- Persist LLM tool calls for debugging follow-up analyses.
-- Captures every query_gto / query_next_actions / leak-tool invocation
-- with the resolved args + result + request_id for correlation.

CREATE TABLE tool_calls (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    request_id TEXT NOT NULL,
    hand_id TEXT,
    tool_name TEXT NOT NULL,
    tool_args JSONB NOT NULL,
    tool_result TEXT,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tool_calls_chat_id_created_at ON tool_calls (chat_id, created_at DESC);
CREATE INDEX idx_tool_calls_request_id ON tool_calls (request_id);
CREATE INDEX idx_tool_calls_hand_id ON tool_calls (hand_id) WHERE hand_id IS NOT NULL;

ALTER TABLE tool_calls ENABLE ROW LEVEL SECURITY;
