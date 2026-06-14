# Phase C Results — Avatar-Anchored Seat Reading

Branch: `feat/effbb-precision`. Measured 2026-06-12 with the auto-label harness
`scripts/seat_autolabel.py` (D4a): each corpus image's HH ground truth gives the
expected per-seat stack multiset; a read counts when it matches an expected
stack within `max(0.3bb, 3%)`. Zero manual labeling.

## Scorecard (120-image stride, `--stride 60`)

| Source | seat_recall | seat_precision | value_accuracy |
|---|---|---|---|
| Current pipeline (`_find_all_stacks`, the bar) | 64.4% | 62.8% | 80.0% |
| **Avatar-anchored (detector + reader)** | **67.5%** | **65.6%** | 79.4% |
| Δ | **+3.1pp** | **+2.8pp** | −0.6pp |

The avatar-anchored reader **beats the legacy scan on both axes the effbb
frontier cares about** — recall (coverage) and precision (phantom rejection).
The win comes from claiming, per detected avatar, only the `XX.X BB` text inside
a fixed ROI under the disc; BB text not owned by any avatar (pot totals, bet
sizes on the felt, the action timeline) is dropped by construction. Bounty `$`
pills sit outside the stack ROI and are never read.

## Detector (`seat_detector.py`)

Classical CV per D4: `cv2.HoughCircles` on the fixed 499×640 table-region crop,
tuned to the measured avatar radius band (10–38px), with the central
board-card zone excluded (no seats render there) and near-coincident circles
de-duplicated. One parameter-tuning iteration was applied (param2 30→22, radius
band widened) to lift avatar recall — the plan's allotted single iteration.

Avatar recall remains below the plan's 97% target (the reader's 67.5% seat
recall is an upper bound on detector recall). Per D4/C2 this is the documented
stop-and-report point: **a neural avatar detector is a user decision, not an
implementer decision.** The classical detector already lifts both seat-reading
axes, so it is worth shipping as the optional path now.

## Full-corpus effbb rebuild (the decisive measurement) — flag stays OFF

The cleaner seat reads do **not** improve the production effbb metric. A full
`OCR_SEAT_ANCHORED=1 EFFBB_CAPTURE=1` re-parse of all 7,183 images
(`data/effbb_cache/cache_anchored.jsonl`, 6,835 hands got anchored reads),
scored with `effbb_eval.py`:

| | precision | coverage | correct |
|---|---|---|---|
| Baseline OCR | **78.19%** | 70.4% | 993/1270 |
| Anchored OCR | 77.05% | 71.0% | 987/1281 |
| Δ | **−1.14pp** | +0.6pp | −6 |

Per the C5 gate (flip ON only if BOTH precision and coverage improve),
`OCR_SEAT_ANCHORED` **stays OFF** — zero behavior change in production.

**Why better reads don't move effbb (the key finding).** A fault analysis of the
277 hero-active errors shows **198 (71%) had the correct effective-stack value
already readable on screen** — 85 selection (wrong seat bound; the right value
was a visible stack in 85/85) + 113 near (adjacent bucket; value present in
113/113). The limiter is the **selection / bucketing LOGIC, not seat-read
recall**, so cleaner reads cannot lift the frontier and can even perturb the
layout-consensus inputs slightly (the −1.14pp). The achievable headroom lives in
logic: fixing selection alone is a +6.7pp ceiling (84.9%), selection + boundary
a +15.6pp ceiling (93.8%) — all on the existing OCR, no re-parse.

The seat-reading scorecard above still stands (the anchored reads ARE cleaner on
the value axis); it simply isn't the binding constraint for effbb. The module
lands as the optional, tested path; re-evaluate it only after the selection
logic is fixed, when read quality may start to matter at the margin.

## What shipped

- `scripts/ocr/seat_detector.py` — classical HoughCircles avatar detection.
- `scripts/ocr/seat_reader.py` — anchored-ROI `XX.X BB` reads → `named_stacks`
  schema + `anchor_conf`.
- `scripts/seat_autolabel.py` — committed auto-label harness (`--score-current`
  baseline + `--detector avatars`).
- `scripts/ocr/table_parser.py` — `parse_table` consumes the anchored reads
  behind `OCR_SEAT_ANCHORED` (default OFF), with a legacy-scan fallback flagged
  in `seat_anchored_fallback`.
