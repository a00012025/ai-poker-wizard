# Card Classifier v2 Follow-up

Date: 2026-05-20
Branch: `feat/card-classifier-v2`

## Current Status

This branch implements the CardCNN v2 plan surface:

- Persistent train/val/test split loading for `data/splits/card_classifier_v2.json`.
- PokerCraft crop extractor for `data/cards_v2`, including N8 visual-order hero labels.
- Training-time augmentation and calibration plumbing.
- `CardCNNv2` plus MobileNetV3-small fallback architecture.
- `CardClassifier` metadata dispatch and v2 checkpoint defaulting.
- Dealer-button detector integration for high-confidence 8-max position override.
- Top-2 hero/board duplicate fallback.
- `ocr_precision.py --split/--bucket` test-bucket filtering.

The deployed bot should use the new model after this PR is merged and deployed, provided:

- `scripts/ocr/models/card_cnn_v2.pt` and `.json` are included in the deployed artifact.
- `requirements.txt` is installed, including `torchvision`.
- The Telegram bot process is restarted/redeployed, because `CardClassifier` loads lazily and caches the singleton.

## Verification Evidence

Classifier split evidence:

- Best checkpoint architecture: `mobilenet_v3_small`
- Val: `card=0.9672`, `rank=0.9719`, `suit=0.9757`
- Test split via `scripts/ocr/classifier/eval.py`: `card=0.9772`, `rank=0.9791`, `suit=0.9832`
- Gate still fails: val per-class F1 below 0.95 for rank `9` and `K`.

End-to-end OCR evidence:

- Command: `python scripts/ocr_precision.py --split data/splits/card_classifier_v2.json --bucket test --workers 4 --max-failures 120 --out data/ocr_precision_card_v2_test`
- Summary: `data/ocr_precision_card_v2_test/summary.json`
- `hand_exact`: `23.288%` (`153/657`)
- `hero_hand`: `83.866%`
- `hero_position`: `49.772%`
- `board`: `94.368%`
- `preflop_types`: `47.184%`
- `preflop_sized`: `45.053%`
- `table_size`: `47.793%`

Failure modes:

- `position_wrong`: 262
- `preflop_action_types_wrong`: 108
- `parse_none`: 61
- `hero_cards_missing`: 60
- `hero_cards_wrong`: 46
- `board_wrong`: 28

## What Improved

The original crop training data was badly mislabeled because HH `hero_hand` order does not match N8 visual card order. Fixes applied:

- Different-rank hero hands are sorted by visual rank order, high to low.
- Pocket pairs are sorted by observed N8 suit display order: `s > d > h > c`.
- Raw and WIN-masked hero crops are both included.

This moved classifier quality from the earlier broken range around 50-60% card accuracy to roughly 97-98% per-card on held-out split data.

## Remaining Bottleneck

The next session should not start by training another card model. The strongest evidence says `hand_exact` is low because structural parsing is weak:

- `hero_position` is only 49.772%.
- `preflop_types` is only 47.184%.
- `table_size` is only 47.793%.
- `position_wrong` is the largest failure bucket.

Card model work still matters, but it is no longer the top bottleneck. The next session should attack position/action parsing first.

## Recommended Next Session Plan

1. Start from `data/ocr_precision_card_v2_test/diffs.jsonl`.
2. Bucket `position_wrong` by table size, `players_at_table`, and whether `dealer_button` was detected.
3. Inspect the first 30 `position_wrong` failures manually against source PNGs.
4. Fix `n8_parser._estimate_table_size`, action-entry first-round/reaction detection, and button-based position mapping.
5. Add regression tests for each parser failure pattern before patching.
6. Re-run `ocr_precision.py --split ... --bucket test` after each parser patch.
7. Only return to card model training if `hero_cards_wrong` remains a top failure after position/action fixes.

Target remains: materially raise `hand_exact`, regardless of method. The fastest path is likely parser correctness, not more CNN capacity.

## Known Risks

- The current PR does not meet the original 99.9% headline target.
- `torchvision` is now required for the selected MobileNetV3 checkpoint.
- The bot must be redeployed/restarted for the cached classifier singleton to load the new model.
- `data/cards_v2` and split files are rebuildable/ignored; the committed checkpoint is the artifact used in runtime.
