# OCR 99% Baseline History

Append a new row after every phase re-baseline. Source data lives next to the row's `out_dir`.

| Date | Phase | Bucket | Scored | hand_exact | hero_hand | hero_position | board | preflop_types | table_size | parse_none | ece_10bin | tau_target | precision@tau | coverage@tau | out_dir | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-20 | pre-phase-0 | test | 657 | 23.288% | 83.866% | 49.772% | 94.368% | 47.184% | 47.793% | 61 | N/A | N/A | N/A | N/A | data/ocr_precision_card_v2_test | baseline from commit c2c0c53 |
| 2026-05-21 | post-phase-0 | test | 657 | 23.288% | 83.866% | 49.772% | 94.368% | 47.184% | 47.793% | 61 | 0.638609 | N/A | N/A | N/A | data/ocr_precision_phase0_test | summary.json + diagnostics_summary.json + calibration_summary.json |
| 2026-05-21 | post-phase-0 | train-dev1500 | 1390 | 26.475% | 83.885% | 56.547% | 93.741% | 50.432% | 51.583% | 110 | 0.605964 | N/A | N/A | N/A | data/ocr_precision_phase0_dev1500 | summary.json + diagnostics_summary.json + calibration_summary.json |
