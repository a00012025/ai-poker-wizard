-- Different stack bands are different training prescriptions even when the
-- action-line taxonomy leaf is identical.  Persist that scope so short/mid/
-- deep evidence and GTOW Drill bindings cannot overwrite one another.
ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS depth_scope TEXT NOT NULL DEFAULT 'all';

UPDATE drill_queue
SET depth_scope = CASE
  WHEN drill_url ~* 'depth_list=10\.125(%2c|,)12\.125(%2c|,)14\.125(%2c|,)17\.125(%2c|,)20\.125(&|$)'
    THEN 'short'
  WHEN drill_url ~* 'depth_list=25\.125(%2c|,)30\.125(%2c|,)35\.125(%2c|,)40\.125(&|$)'
    THEN 'medium'
  WHEN drill_url ~* 'fh_start_spot=preflop'
    AND drill_url ~* 'depth_list=40\.125(&|$)'
    THEN 'large'
  ELSE 'all'
END
WHERE kind = 'drill';

ALTER TABLE drill_queue
  ADD CONSTRAINT drill_queue_depth_scope_check
  CHECK (depth_scope IN ('all', 'short', 'medium', 'large'));

DROP INDEX IF EXISTS idx_drill_queue_pending_leaf;
CREATE UNIQUE INDEX idx_drill_queue_pending_leaf
  ON drill_queue(spot_leaf, depth_scope)
  WHERE status = 'pending' AND kind = 'drill';
