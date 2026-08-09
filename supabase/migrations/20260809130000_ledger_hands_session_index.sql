-- Session summaries filter ledger_hands by session_id on every resend.
-- Keep that membership lookup indexed as the online ledger grows.
CREATE INDEX IF NOT EXISTS idx_ledger_hands_session
ON ledger_hands(session_id);
