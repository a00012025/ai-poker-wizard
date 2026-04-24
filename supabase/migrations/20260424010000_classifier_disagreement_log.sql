-- Medium-tier OCR/Gemini cross-check log. Every row is a snapshot where
-- the OCR fast-path returned a hand at medium confidence (default band
-- OCR_MEDIUM_TIER_MIN <= conf < OCR_FAST_TIER_MIN) AND the async Gemini
-- re-parse disagreed with it on hero_hand or any street board. These rows
-- feed the periodic relabel audit: pick a winner (usually Gemini), write
-- expected_json, retrain.
CREATE TABLE IF NOT EXISTS classifier_disagreement_log (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    ocr_hand JSONB NOT NULL,
    gemini_hand JSONB NOT NULL,
    ocr_conf REAL NOT NULL,
    diff JSONB NOT NULL,
    resolved_winner TEXT CHECK (resolved_winner IN ('ocr', 'gemini', 'both_wrong', NULL)),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_disagreement_log_unresolved
  ON classifier_disagreement_log(created_at)
  WHERE resolved_winner IS NULL;

CREATE INDEX IF NOT EXISTS idx_disagreement_log_chat
  ON classifier_disagreement_log(chat_id, created_at DESC);
