-- Make depth-restricted queue/GTOW Drill names self-describing.  The normal
-- all-depth MTT prescription intentionally keeps its compact name unchanged.
WITH named AS (
  SELECT id,
         CASE
           WHEN drill_url ~* 'depth_list=10\.125(%2c|,)12\.125(%2c|,)14\.125(%2c|,)17\.125(%2c|,)20\.125(&|$)'
             THEN ' (≤20bb)'
           WHEN drill_url ~* 'depth_list=25\.125(%2c|,)30\.125(%2c|,)35\.125(%2c|,)40\.125(&|$)'
             THEN ' (20-50bb)'
           WHEN drill_url ~* 'fh_start_spot=preflop'
             AND drill_url ~* 'depth_list=40\.125(&|$)'
             THEN ' (>50bb)'
           WHEN drill_url ~* 'depth_list=10\.125(%2c|,)12\.125(%2c|,)14\.125(%2c|,)17\.125(%2c|,)20\.125(%2c|,)25\.125(%2c|,)30\.125(%2c|,)35\.125(%2c|,)40\.125(&|$)'
             THEN ''
           ELSE NULL
         END suffix
  FROM drill_queue
  WHERE kind = 'drill' AND drill_url IS NOT NULL
)
UPDATE drill_queue q
SET label = CASE WHEN q.label IS NULL THEN NULL ELSE
      regexp_replace(q.label, ' \((≤20bb|20-50bb|>50bb)\)$', '') || named.suffix END,
    gtow_drill_name = CASE WHEN q.gtow_drill_name IS NULL THEN NULL ELSE
      regexp_replace(q.gtow_drill_name, ' \((≤20bb|20-50bb|>50bb)\)$', '') || named.suffix END
FROM named
WHERE q.id = named.id AND named.suffix IS NOT NULL;
