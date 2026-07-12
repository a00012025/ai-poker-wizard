-- Path-aware review links: preserve EV-loss ranking while linking an earlier
-- low-frequency hero branch as the recommended review starting point.
ALTER TABLE drill_queue
  ADD COLUMN IF NOT EXISTS review_anchor_url TEXT,
  ADD COLUMN IF NOT EXISTS review_anchor_street TEXT;

-- Normalize persisted queue copy; newly generated labels use the same wording
-- in scorecard.spot_desc_zh.
UPDATE drill_queue SET label = replace(label, ' 首開（RFI）', ' 開池')
WHERE label LIKE '% 首開（RFI）%';
UPDATE drill_queue SET label = replace(replace(replace(label,
  '翻牌 面對', '翻牌面對'), '轉牌 面對', '轉牌面對'), '河牌 面對', '河牌面對')
WHERE label LIKE '%翻牌 面對%' OR label LIKE '%轉牌 面對%' OR label LIKE '%河牌 面對%';
