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

## Landing (D4b / C5)

`OCR_SEAT_ANCHORED` **defaults OFF**. Flipping it on changes
`_compute_effective_bb` inputs and therefore requires a full `EFFBB_CAPTURE=1`
corpus re-parse (~3h) to rebuild `data/effbb_cache/cache.jsonl` before any effbb
number can be quoted (D4b). That rebuild is the user-gated next step:

```bash
OCR_SEAT_ANCHORED=1 EFFBB_CAPTURE=1 python scripts/effbb_cache.py   # ~3h
python scripts/effbb_eval.py --cache data/effbb_cache/cache_anchored.jsonl
```

The flag flips to ON by default only if that rebuild shows the effbb frontier
improving on BOTH precision and coverage (C5). Until then it lands OFF — zero
behavior change, zero regression — with the seat-reading scorecard above as the
evidence that the cleaner reads are worth validating.

## What shipped

- `scripts/ocr/seat_detector.py` — classical HoughCircles avatar detection.
- `scripts/ocr/seat_reader.py` — anchored-ROI `XX.X BB` reads → `named_stacks`
  schema + `anchor_conf`.
- `scripts/seat_autolabel.py` — committed auto-label harness (`--score-current`
  baseline + `--detector avatars`).
- `scripts/ocr/table_parser.py` — `parse_table` consumes the anchored reads
  behind `OCR_SEAT_ANCHORED` (default OFF), with a legacy-scan fallback flagged
  in `seat_anchored_fallback`.
