-- Normalize preflop taxonomy wording: remove the new-surface "Cold3bet" name.
-- Plain non-opener facing a 3bet becomes vs3bet. If the same hand has an
-- earlier hero flat-call versus an open before the 3bet decision, classify that
-- later decision as flat_vsSqueeze so caller-facing-squeeze is distinct from
-- opener-facing-squeeze.

WITH cold AS (
  SELECT d.id, d.gtow_hand_id, d.decision_idx, d.hero_cat, d.villain_cat, d.ip_oop,
         EXISTS (
           SELECT 1 FROM ledger_decisions prior
           WHERE prior.gtow_hand_id = d.gtow_hand_id
             AND prior.source = d.source
             AND prior.street = 'preflop'
             AND prior.decision_idx < d.decision_idx
             AND prior.spot_category = 'vsOpen'
             AND prior.taken_code = 'C'
         ) AS hero_flat_called_open
  FROM ledger_decisions d
  WHERE d.street = 'preflop'
    AND d.spot_category = 'vsCold3bet'
)
UPDATE ledger_decisions d
SET spot_category = CASE WHEN cold.hero_flat_called_open THEN 'vsSqueeze' ELSE 'vs3bet' END,
    spot_parent = CASE
      WHEN cold.hero_flat_called_open THEN cold.hero_cat || 'flat_vsSqueeze'
      ELSE cold.hero_cat || '_vs3bet'
    END,
    spot_leaf = CASE
      WHEN cold.hero_flat_called_open THEN cold.hero_cat || 'flat_vsSqueeze_v' || cold.villain_cat || '_' || COALESCE(cold.ip_oop, '?')
      ELSE cold.hero_cat || '_vs3bet_v' || cold.villain_cat || '_' || COALESCE(cold.ip_oop, '?')
    END
FROM cold
WHERE d.id = cold.id;

-- Queue rows denormalize the leaf/category for open prescriptions. Re-key rows
-- whose source decisions are now caller-facing-squeeze first, then rename the
-- remaining legacy Cold3bet rows to plain vs3bet.
UPDATE drill_queue q
SET spot_category = 'vsSqueeze',
    spot_leaf = replace(q.spot_leaf, '_vsCold3bet', 'flat_vsSqueeze'),
    label = replace(replace(q.label, '｜Cold vs ', ' flat vs '), 'Cold vs ', 'flat vs '),
    last_added = NOW()
WHERE q.spot_category = 'vsCold3bet'
  AND EXISTS (
    SELECT 1
    FROM jsonb_array_elements(q.source_hands) AS src(entry)
    JOIN ledger_decisions d ON d.gtow_hand_id = src.entry->>'hand_id'
      AND d.street = src.entry->>'street'
      AND d.decision_idx = CASE
        WHEN (src.entry->>'decision_idx') ~ '^[0-9]+$' THEN (src.entry->>'decision_idx')::int
        ELSE NULL
      END
    WHERE d.spot_leaf LIKE '%flat_vsSqueeze%'
  );

UPDATE drill_queue
SET spot_category = 'vs3bet',
    spot_leaf = replace(spot_leaf, '_vsCold3bet', '_vs3bet'),
    label = replace(replace(label, '｜Cold vs ', ' vs '), 'Cold vs ', 'vs '),
    last_added = NOW()
WHERE spot_category = 'vsCold3bet';
