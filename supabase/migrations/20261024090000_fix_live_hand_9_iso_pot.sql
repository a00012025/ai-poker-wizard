-- Session 36, Hand 9: SB completed, BB raised to 4bb, SB called.
-- The old classifier recorded this ISO pot as a limp pot. Keep the
-- preflop faced-limp classification and limp_origin flag unchanged.
UPDATE ledger_hands
SET pot_type = 'iso'
WHERE gtow_hand_id = 'live:2026-10-23:0f4cb64d66'
  AND source = 'live'
  AND pot_type = 'limp';

UPDATE ledger_decisions
SET pot_type = 'iso',
    spot_leaf = CASE WHEN street = 'flop'
                     THEN replace(spot_leaf, 'flop:limp', 'flop:iso')
                     ELSE spot_leaf END,
    spot_parent = CASE WHEN street = 'flop'
                       THEN replace(spot_parent, 'flop:limp', 'flop:iso')
                       ELSE spot_parent END,
    spot_keys = CASE WHEN street = 'flop'
                     THEN replace(spot_keys::text, 'flop:limp', 'flop:iso')::jsonb
                     ELSE spot_keys END
WHERE gtow_hand_id = 'live:2026-10-23:0f4cb64d66'
  AND source = 'live'
  AND pot_type = 'limp';

UPDATE live_sessions
SET result_json = jsonb_set(
    result_json, '{hands,8}',
    replace(
      replace((result_json #> '{hands,8}')::text,
              '"pot_type": "limp"', '"pot_type": "iso"'),
      'flop:limp', 'flop:iso')::jsonb)
WHERE id = 36
  AND result_json #>> '{hands,8,hand_row,gtow_hand_id}'
      = 'live:2026-10-23:0f4cb64d66';
