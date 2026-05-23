# Three-Day 99% Push — Wrong-Emitted Audit (2026-05-23)

**Source:** `data/ocr_precision_current/diffs.jsonl` (regenerated 2026-05-23 against `data/splits/card_classifier_v2.json` test bucket).

**Summary metrics (from `summary.json`):**

| Metric | Value |
| --- | --- |
| paired | 718 |
| emitted | 612 |
| coverage | 85.237% |
| hand_exact | 95.588% (585/612) |
| wrong emitted | 27 |
| abstained correctly (wrong if emitted) | 32 |
| abstained loss (exact if emitted) | 58 |
| ECE 10-bin | 0.041 |
| τ_target (99%@70%) | None |

**Three-day target:** ≥503 emitted, ≤5 wrong, `hero_hand ≥99.8%`, `critical_error ≤0.25%`, `ece_10bin ≤0.04`.

## Wrong-emitted breakdown

- **position_wrong**: 22
- **preflop_action_types_wrong**: 3
- **board_wrong**: 1
- **hero_cards_wrong**: 1

## board_wrong (1)

| hand_id | conf | parts (pot/player/ocr/card) | diag-levers | safe_emit | GT vs parsed |
| --- | --- | --- | --- | --- | --- |
| TM5896602712 | 0.900 | 0.50/1.00/1.00/1.00 | pre_collapse_loss=4; button_lowconf=0.0; pot_consistency=0.5; safe_emit=simple_preflop_high_card; conf<0.90=0.900 | simple_preflop_high_card | table_size: `6` → `None` <br> num_players: `5` → `None` <br> effective_bb: `9.2` → `None` |

## hero_cards_wrong (1)

| hand_id | conf | parts (pot/player/ocr/card) | diag-levers | safe_emit | GT vs parsed |
| --- | --- | --- | --- | --- | --- |
| TM5900728345 | 0.906 | 1.00/1.00/1.00/0.76 | pre_collapse_loss=7; button_lowconf=0.0; card_conf=0.765 | - | hero_hand: `8cTd` → `Td3c` <br> preflop_actions: `R2.0-F-F-F-F-F-F-C` → `R2-F-F-F-F-F-F-C` <br> table_size: `8` → `None` <br> num_players: `8` → `None` |

## position_wrong (22)

| hand_id | conf | parts (pot/player/ocr/card) | diag-levers | safe_emit | GT vs parsed |
| --- | --- | --- | --- | --- | --- |
| TM5913031183 | 1.000 | 1.00/1.00/1.00/1.00 | pre_collapse_loss=7; players_raw_vs_final=9->8; button_lowconf=0.0 | - | hero_hand: `3dJc` → `Jc3d` <br> hero_position: `HJ` → `BB` <br> preflop_actions: `F-R2.0-F-F-C-F-C-C` → `F-R2-F-F-C-F-C-AI3-C` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `1.5` → `None` |
| TM5913201917 | 0.998 | 1.00/1.00/1.00/1.00 | pre_collapse_loss=6; button_lowconf=0.0 | - | hero_hand: `JdKh` → `KhJd` <br> hero_position: `BTN` → `SB` <br> preflop_actions: `R2.0-C-C-F-C-F-C` → `R2-C-C-F-C-F-C` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `0.1` → `33.5` |
| TM5900097060 | 0.997 | 1.00/1.00/1.00/0.99 | pre_collapse_loss=7; reaction_signal; button_lowconf=0.0 | - | hero_position: `HJ` → `CO` <br> preflop_actions: `F-F-F-F-R2.0-F-R3.7-C` → `F-F-F-F-R2-F-R3.68-C` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `0.1` → `23.8` |
| TM5875440783 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=3; reaction_signal; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `7h8s` → `8s7h` <br> hero_position: `LJ` → `CO` <br> preflop_actions: `F-F-F-AI23.1-F-F` → `F-F-F-AI23.05-F-F` <br> table_size: `6` → `None` <br> num_players: `6` → `None` <br> effective_bb: `20.2` → `24.9` |
| TM5875585050 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=5; reaction_signal; button_lowconf=0.0; player_tracking=0.5; conf<0.90=0.900 | - | hero_hand: `KcAc` → `AcKc` <br> hero_position: `BB` → `SB` <br> preflop_actions: `F-F-F-F-F-R2.0-AI50.6-C` → `F-F-F-F-F-R2-AI50.56-C-AI2` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `18.2` → `None` |
| TM5879883906 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=4; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `3h5s` → `5s3h` <br> hero_position: `LJ` → `CO` <br> preflop_actions: `F-F-F-AI16.5-F-F-F` → `F-F-F-AI16.54-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `14.7` → `None` |
| TM5963073078 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=1; players_raw_vs_final=5->6; button_lowconf=0.0; player_tracking=0.5; safe_emit=simple_preflop_high_card; conf<0.90=0.900 | simple_preflop_high_card | hero_hand: `4h6s` → `6s4h` <br> hero_position: `BTN` → `LJ` <br> table_size: `6` → `None` <br> num_players: `6` → `None` |
| TM5947075456 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=4; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `JcQd` → `QdJc` <br> hero_position: `SB` → `BB` <br> preflop_actions: `F-F-F-R2.0-AI40.1-F-F-F-F` → `F-F-F-R2-AI40.13-F` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `3.9` → `41.5` |
| TM5873873878 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=7; reaction_signal; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `AcAh` → `AhAc` <br> hero_position: `BB` → `SB` <br> preflop_actions: `F-F-F-F-F-C-R3.0-AI18.5-C` → `F-F-F-F-F-C-R3-AI52-C` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `17.7` → `52.0` |
| TM5932682790 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=9; players_raw_vs_final=9->8; button_lowconf=0.0; player_tracking=0.5; conf<0.90=0.900 | - | hero_position: `HJ` → `CO` <br> preflop_actions: `F-AI13.6-F-F-F-AI16.4-F-F` → `F-F-AI13.64-F-F-F-AI16.42-F-F-F` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `13.6` → `None` |
| TM5901482662 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=21; reaction_signal; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `9sQs` → `Qs9s` <br> hero_position: `HJ` → `CO` <br> preflop_actions: `C-AI4.2-F-F-AI95.4-C` → `C-AI-AI-F-F-AI95.43-C` <br> table_size: `6` → `None` <br> num_players: `6` → `None` <br> effective_bb: `0.7` → `None` |
| TM5963739343 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=8; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `CO` → `BTN` <br> preflop_actions: `F-F-F-AI8.2-AI59.1-F-F` → `F-F-F-AI8.16-AI59.07-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `8.2` → `None` |
| TM5880191974 | 0.900 | 0.50/1.00/1.00/1.00 | pre_collapse_loss=10; reaction_signal; pot_consistency=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `HJ` → `CO` <br> preflop_actions: `F-R2.0-C-C-AI6.8-F-F-C-C` → `F-R2-C-C-AI6.84-F-F-C-C` <br> table_size: `6` → `None` <br> num_players: `6` → `None` <br> effective_bb: `6.8` → `16.3` |
| TM5896105025 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=1; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `TdQs` → `QsTd` <br> hero_position: `CO` → `HJ` <br> preflop_actions: `F-R2.0-AI10.0-F-F-F` → `F-R2-AI9.97-F-F-F` <br> table_size: `6` → `None` <br> num_players: `5` → `None` <br> effective_bb: `10.0` → `16.2` |
| TM5963739858 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=4; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `SB` → `BB` <br> preflop_actions: `F-F-F-AI13.7-F-F-F` → `F-F-F-AI13.73-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `13.7` → `17.0` |
| TM5932645601 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=3; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `BB` → `SB` <br> preflop_actions: `F-R2.0-R5.1-F-F-F-F-F` → `F-R2-R5.1-F-F-F-F-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `16.6` → `86.0` |
| TM5875113375 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=2; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `2dAs` → `As2d` <br> hero_position: `BTN` → `SB` <br> preflop_actions: `F-F-F-F-AI11.6-F-F` → `F-F-F-F-AI11.58-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `11.0` → `22.2` |
| TM5887365005 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=7; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `6sQh` → `Qh6s` <br> hero_position: `UTG` → `CO` <br> preflop_actions: `F-F-F-C-F-AI6.5-F` → `F-F-F-C-AI-F-AI6.47-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `0.3` → `8.4` |
| TM5896664443 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=1; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `BB` → `SB` <br> preflop_actions: `F-F-F-R2.0-AI18.8-F-F-F` → `F-F-F-R2-AI18.82-F-F-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `18.8` → `25.4` |
| TM5947799144 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=10; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_position: `CO` → `SB` <br> preflop_actions: `F-F-F-C-F` → `F-F-F-C-AI-F` <br> table_size: `8` → `None` <br> num_players: `6` → `None` <br> effective_bb: `0.7` → `19.3` |
| TM5963740052 | 0.899 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=1; button_lowconf=0.0; player_tracking=0.5; conf<0.90=0.899 | - | hero_position: `SB` → `BTN` <br> preflop_actions: `F-F-F-F-F-C-AI13.6-F` → `F-F-F-F-F-C-AI13.61-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `13.6` → `26.2` |
| TM5896148295 | 0.899 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=3; reaction_signal; button_lowconf=0.0; player_tracking=0.5; conf<0.90=0.899 | - | hero_hand: `3h8c` → `8c3h` <br> hero_position: `SB` → `BTN` <br> preflop_actions: `F-C-R3.5-F-F-F-AI16.5-F-F` → `F-C-R3.5-F-F-F-AI16.45-F-F` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `16.5` → `31.1` |

## preflop_action_types_wrong (3)

| hand_id | conf | parts (pot/player/ocr/card) | diag-levers | safe_emit | GT vs parsed |
| --- | --- | --- | --- | --- | --- |
| TM5866478558 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=6; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `6sQs` → `Qs6s` <br> preflop_actions: `C-F-F-F-F-F-AI25.1-F-F` → `C-F-F-F-F-F-AI25.07-F` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `25.1` → `73.6` |
| TM5895757896 | 0.900 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=7; button_lowconf=0.0; player_tracking=0.5; safe_emit=high_card_complex_non_danger; conf<0.90=0.900 | high_card_complex_non_danger | hero_hand: `5d7d` → `7d5d` <br> preflop_actions: `F-R3.0-F-F-F-AI12.3-F-C` → `F-R3-F-F-F-AI12.25-C` <br> table_size: `8` → `None` <br> num_players: `7` → `None` <br> effective_bb: `12.3` → `52.1` |
| TM5879884236 | 0.898 | 1.00/0.50/1.00/1.00 | pre_collapse_loss=5; reaction_signal; player_tracking=0.5; conf<0.90=0.898 | - | hero_hand: `6c6s` → `6s6c` <br> preflop_actions: `R2.0-F-F-F-F-F-C-AI50.7-C-F` → `R2-F-F-F-F-F-C-AI50.68-C-AI2-F` <br> table_size: `8` → `None` <br> num_players: `8` → `None` <br> effective_bb: `17.0` → `None` |

## Calibrator viability analysis

- Calibrator-abstainable (≥1 diagnostic lever): **27** / 27
- Calibrator-hard (no obvious lever): **0** / 27

## Recoverable abstain pool (coverage upside)

- Currently abstained but exact: **58** — recoverable coverage.
- Currently abstained and wrong: **32** — calibrator must keep these abstained.

Top 20 abstained-exact (sorted by confidence) — recoverable candidates:

| hand_id | conf | card_conf | safe_emit | players_raw/final | preflop_pre/post |
| --- | --- | --- | --- | --- | --- |
| TM5888029705 | 0.875 | 0.855 | - | 6/6 | 9/7 |
| TM5920962255 | 0.873 | 0.683 | - | 6/6 | 12/6 |
| TM5866468863 | 0.867 | 1.000 | - | 8/8 | 12/9 |
| TM5932645246 | 0.863 | 0.858 | - | 8/8 | 9/8 |
| TM5919995242 | 0.862 | 0.905 | - | 6/6 | 11/6 |
| TM5900096435 | 0.861 | 0.902 | - | 8/8 | 16/8 |
| TM5874599522 | 0.848 | 0.869 | - | 7/8 | 7/7 |
| TM5900854136 | 0.841 | 0.852 | - | 8/8 | 8/8 |
| TM5874599558 | 0.840 | 0.993 | - | 8/8 | 10/8 |
| TM5900729022 | 0.837 | 0.843 | - | 8/8 | 18/9 |
| TM5879884033 | 0.833 | 0.998 | - | 8/8 | 12/9 |
| TM5920327196 | 0.832 | 0.829 | - | 8/8 | 15/8 |
| TM5864260800 | 0.831 | 0.578 | - | 6/6 | 8/6 |
| TM5920962008 | 0.820 | 0.551 | - | 5/5 | 9/5 |
| TM5887942304 | 0.820 | 0.800 | - | 6/6 | 7/6 |
| TM5846885345 | 0.800 | 1.000 | - | 9/8 | 15/10 |
| TM5901453300 | 0.800 | 1.000 | - | 7/7 | 16/8 |
| TM5920962533 | 0.800 | 0.999 | - | 6/6 | 14/7 |
| TM5875127138 | 0.800 | 0.999 | - | 8/8 | 8/8 |
| TM5880480341 | 0.799 | 0.999 | - | 7/7 | 15/9 |

## Proposed fixtures (≥12)

Each fixture asserts a **reusable feature shape**, not a hand-ID lookup.

### position_wrong → must abstain

- `TM5873873878` — all-in re-action ambiguity
- `TM5887365005` — raw vs final player count mismatch
- `TM5896105025` — preflop pre/post entry collapse loss
- `TM5913201917` — reaction-signal with low confidence

### preflop_action_types_wrong → must abstain

- `TM5866478558` — missing call after all-in
- `TM5879884236` — phantom raise fragment
- `TM5895757896` — wrong extra re-action row

### hero/board critical → must abstain

- `TM5900728345` — raw vs WIN-masked card disagreement
- `TM5896602712` — board street count mismatch

### positive high-conf exact (must keep emitting)

- `H2894` — Th9h hero, WIN overlay, must not flip to 9d (snapshot fixture).
- Top-3 calibration-recoverable exact hands from the table above (Day 2 confirms via gate harness).

## Day 2 calibrator hypothesis

Lever frequency across the 27 wrong:

- `pre_collapse_loss`: 27 / 27
- `button_lowconf`: 25 / 27
- `conf<0.90`: 23 / 27
- `player_tracking`: 21 / 27
- `safe_emit`: 18 / 27
- `reaction_signal`: 8 / 27
- `players_raw_vs_final`: 3 / 27
- `pot_consistency`: 2 / 27
- `card_conf`: 1 / 27

**Implication:** a calibrator over `preflop_entries_pre_collapse - preflop_entries_count`, `players_raw_vs_final` mismatch, `pot_consistency`, `player_tracking`, `card_confidence`, `confidence`, plus a hard-abstain rule for all-in re-action grammar and missing-call-after-all-in, should rank most wrong hands below the bulk of exact hands. The calibrator-hard residual (table above) is the Day 3 parser-fix list.
