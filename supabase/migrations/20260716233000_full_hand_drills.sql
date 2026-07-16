-- Trainer exercises are full-hand practice, never single-spot practice.
-- Preserve each URL's exact action line while upgrading only the GTOW UI mode.
-- A changed URL changes Drill identity, so clear the cached settings hash; the
-- bound UUID remains authoritative and is PATCHed in place on the next sync.

UPDATE drill_queue
SET drill_url = CASE
      WHEN drill_url ~ '([?&])fh_trainer_mode='
        THEN regexp_replace(
          drill_url,
          '([?&])fh_trainer_mode=[^&]*',
          '\1fh_trainer_mode=stop_end_of_hand'
        )
      WHEN drill_url LIKE '%?%'
        THEN drill_url || '&fh_trainer_mode=stop_end_of_hand'
      ELSE drill_url || '?fh_trainer_mode=stop_end_of_hand'
    END,
    gtow_settings_hash = NULL,
    gtow_drill_synced_at = NULL
WHERE kind = 'drill'
  AND drill_url LIKE 'https://app.gtowizard.com/practice/trainer%'
  AND position('fh_trainer_mode=stop_end_of_hand' IN drill_url) = 0;

ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_full_hand_trainer_url
  CHECK (
    kind <> 'drill'
    OR drill_url IS NULL
    OR drill_url NOT LIKE 'https://app.gtowizard.com/practice/trainer%'
    OR position('fh_trainer_mode=stop_end_of_hand' IN drill_url) > 0
  );
