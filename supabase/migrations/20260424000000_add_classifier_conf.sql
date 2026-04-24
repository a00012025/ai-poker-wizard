-- Adds the classifier_conf column used for OOD monitoring of the CardCNN
-- rank/suit classifier. Populated by gemini_session._save_snapshot on every
-- image-based hand analysis; NULL for text-based hands and for legacy rows
-- captured before the classifier landed. weekly_report.py reads it to
-- compute a rolling 7-day mean confidence + low-conf share.
ALTER TABLE analysis_snapshots
  ADD COLUMN IF NOT EXISTS classifier_conf REAL;

CREATE INDEX IF NOT EXISTS idx_snapshots_classifier_conf
  ON analysis_snapshots(classifier_conf)
  WHERE classifier_conf IS NOT NULL;
