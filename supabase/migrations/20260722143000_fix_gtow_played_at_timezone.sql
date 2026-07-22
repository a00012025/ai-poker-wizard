-- GTOW Analyze returns hand-history wall-clock played_at values with a trailing
-- Z. They were previously interpreted as true UTC, shifting online ledger hands
-- eight hours into the future for the Taipei/GTOW session clock. Normalize the
-- already-ingested online rows to real UTC. Live/offline rows are excluded.

UPDATE ledger_hands
SET played_at = played_at - interval '8 hours'
WHERE source = 'online';

UPDATE ledger_decisions
SET played_at = played_at - interval '8 hours'
WHERE source = 'online'
  AND played_at IS NOT NULL;
