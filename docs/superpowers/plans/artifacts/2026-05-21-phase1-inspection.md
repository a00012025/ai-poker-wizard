# Phase 1 Inspection - 30 position_wrong hands

Diagnostic overlays live under `data/ocr_diagnostics/phase1/<hand_id>/`.

Status: hand set and overlays generated; visual taxonomy pass remains.

Taxonomy used:

- **PR-DROP-ROW: panel parser dropped or filtered a preflop row, causing table-size undercount**
- **PR-MERGE-ROW: adjacent entries merged before split/collapse**
- **PR-SPLIT-FAIL: `_split_multi_action_groups` failed to split a merged group**
- **PR-NAMELESS-REACTION: re-action with no player_name slipped past `_estimate_table_size`**
- **BTN-LOWCONF: dealer button visible but confidence below override threshold**
- **BTN-MISALIGNED: dealer button detected but mapped to the wrong seat**
- **BLIND-COL-OVERRIDE: blinds column inferred SB/BB incorrectly**
- **OTHER: requires visual notes**

| hand_id | GT pos | parsed pos | dealer_button_conf | pre_collapse | final | players_at_table (parsed / GT) | category | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TM5846884632 | HJ | CO | 0.000 | 8 | 6 | 6 / 7 | TBD | metrics: undercount |
| TM5932941555 | HJ | CO | 0.000 | 11 | 6 | 5 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5896980767 | SB | BB | 0.000 | 7 | 6 | 6 / 7 | TBD | metrics: undercount |
| TM5887364726 | CO | BTN | 0.000 | 7 | 7 | 7 / 8 | TBD | metrics: undercount |
| TM5896655012 | LJ | HJ | 0.000 | 7 | 7 | 7 / 8 | TBD | metrics: undercount |
| TM5920467483 | SB | BB | 0.000 | 13 | 7 | 7 / 8 | TBD | metrics: undercount |
| TM5896801988 | SB | BB | 0.000 | 7 | 6 | 6 / 7 | TBD | metrics: undercount |
| TM5947473013 | CO | BTN | 0.000 | 7 | 6 | 6 / 7 | TBD | metrics: undercount |
| TM5947939232 | BTN | SB | 0.000 | 7 | 6 | 6 / 7 | TBD | metrics: undercount |
| TM5962716413 | HJ | BB | 0.000 | 7 | 6 | 3 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5846884791 | BTN | SB | 0.000 | 10 | 8 | 8 / 8 | TBD |  |
| TM5963421207 | LJ | SB | 0.000 | 8 | 6 | 3 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5963609478 | LJ | BB | 1.000 | 18 | 10 | 8 / 8 | TBD | reaction_signal=True; button high-conf |
| TM5962678301 | SB | BB | 0.000 | 8 | 8 | 7 / 8 | TBD | metrics: undercount |
| TM5962447954 | BTN | BB | 0.000 | 6 | 6 | 3 / 6 | TBD | metrics: undercount; reaction_signal=True |
| TM5875127138 | LJ | HJ | 0.000 | 8 | 7 | 7 / 8 | TBD | metrics: undercount |
| TM5947939293 | UTG | BB | 1.000 | 19 | 10 | 8 / 8 | TBD | reaction_signal=True; button high-conf |
| TM5896105081 | BB | BTN | 0.000 | 14 | 6 | 3 / 5 | TBD | metrics: undercount; reaction_signal=True |
| TM5963739754 | LJ | BB | 0.000 | 7 | 7 | 2 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5875749434 | BTN | HJ | 0.000 | 8 | 8 | 5 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5896285939 | UTG+1 | LJ | 0.000 | 9 | 7 | 7 / 8 | TBD | metrics: undercount |
| TM5880529980 | CO | BB | 0.000 | 8 | 6 | 4 / 7 | TBD | metrics: undercount; reaction_signal=True |
| TM5875362766 | BTN | SB | 1.000 | 19 | 5 | 5 / 7 | TBD | metrics: undercount; button high-conf |
| TM5846885345 | LJ | BB | 0.000 | 15 | 9 | 9 / 8 | TBD | metrics: overcount |
| TM5879884125 | BTN | BB | 1.000 | 20 | 9 | 7 / 8 | TBD | metrics: undercount; reaction_signal=True; button high-conf |
| TM5875210374 | HJ | BB | 0.000 | 9 | 8 | 4 / 8 | TBD | metrics: undercount; reaction_signal=True |
| TM5963600626 | SB | LJ | 0.000 | 14 | 6 | 6 / 6 | TBD |  |
| TM5896731755 | HJ | CO | 0.000 | 7 | 5 | 4 / 5 | TBD | metrics: undercount; reaction_signal=True |
| TM5875037999 | BTN | SB | 0.000 | 8 | 8 | 7 / 8 | TBD | metrics: undercount |
| TM5947075050 | HJ | SB | 0.000 | 8 | 8 | 3 / 8 | TBD | metrics: undercount; reaction_signal=True |
