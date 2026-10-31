-- Replace the former Full hand invariant before writing Street-mode URLs.
ALTER TABLE drill_queue DROP CONSTRAINT IF EXISTS drill_queue_full_hand_trainer_url;

UPDATE drill_queue
SET drill_url = CASE
      WHEN drill_url ~ '([?&])fh_trainer_mode='
        THEN regexp_replace(drill_url, '([?&])fh_trainer_mode=[^&]*',
                            '\1fh_trainer_mode=stop_end_of_street', 'g')
      WHEN drill_url LIKE '%?%'
        THEN drill_url || '&fh_trainer_mode=stop_end_of_street'
      ELSE drill_url || '?fh_trainer_mode=stop_end_of_street'
    END,
    gtow_settings_hash = NULL,
    gtow_drill_synced_at = NULL,
    gtow_training_started_at = NULL,
    gtow_baseline_totals = NULL
WHERE kind = 'drill'
  AND drill_url LIKE 'https://app.gtowizard.com/practice/trainer%'
  AND drill_url !~ '([?&])fh_trainer_mode=stop_end_of_street(&|$)';

ALTER TABLE drill_queue ADD CONSTRAINT drill_queue_street_trainer_url
CHECK (
  kind <> 'drill'
  OR drill_url IS NULL
  OR drill_url NOT LIKE 'https://app.gtowizard.com/practice/trainer%'
  OR drill_url ~ '([?&])fh_trainer_mode=stop_end_of_street(&|$)'
);
