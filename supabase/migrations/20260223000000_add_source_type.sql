-- Add source_type to hand_histories to distinguish text/image/file origins
ALTER TABLE hand_histories
  ADD COLUMN IF NOT EXISTS source_type TEXT NOT NULL DEFAULT 'file';
