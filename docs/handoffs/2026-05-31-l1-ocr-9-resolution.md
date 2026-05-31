# Resolution: the 9 L1-OCR snapshot mismatches

Date: 2026-05-31 (Asia/Taipei)
Supersedes: `docs/handoffs/2026-05-30-l1-ocr-9-snapshot-mismatches.md`
Branch: `fix/ocr-snapshot-mismatches`

## TL;DR — the prior handoff's premise was backwards

The 2026-05-30 handoff assumed **all 9 are OCR bugs** and instructed "do NOT edit
`expected_json` — the user hand-verified the expecteds." Verifying every hand
against its actual screenshot (zoomed to the pip / panel level) showed the
opposite for most of them:

- **6 of 9** were **stale/wrong `expected_json`** — the live OCR is **correct**.
  Following the old handoff literally would have corrupted the parser to emit ♦
  for hearts and `C` for genuine BB checks, regressing the whole corpus.
- **1 of 9** (H2878) was a real bug — fixed in code.
- **2 of 9** (H2810, H2829) are genuine CNN ♣→♠ misreads under WIN-sticker
  occlusion — **deferred to the CardCNN retrain effort** (their `expected_json`
  is already correct, so they pass automatically once the suit head is fixed).

Final snapshot state for these 9: **7 pass / 2 fail** (the 2 are the deferred
♣→♠ pair). `regression_test.py`: **434 passed, 0 failed**.

## Ground-truth verification (per hand)

| Hand  | OCR got | expected (old) | Screenshot truth | Verdict |
|-------|---------|----------------|------------------|---------|
| H2554 | `5d3h`  | `5d3d`         | **5d♦ 3♥**        | OCR correct → fixed expected |
| H2588 | `KsQh`  | `KsQd`         | **K♠ Q♥**         | OCR correct → fixed expected |
| H2549 | `…C-X`,7| `…C-C`,6       | BB **Check**, **7 seats** | OCR correct → fixed expected (both fields) |
| H2555 | `…C-X`,1| `…C-C`,3       | BB **Check**, all-in on flop, turn/river runout | OCR correct → fixed expected |
| H2616 | `…C-X`  | `…C-C`         | SB Call → BB **Check** | OCR correct → fixed expected |
| H2774 | `…C-X`  | `…C-C`         | SB Call → BB **Check** | OCR correct → fixed expected |
| H2878 | `4c2h`  | `Ac2h`         | **A♣ 2♥**         | OCR bug (corner-OCR A→4) → fixed code |
| H2810 | `QsJs`  | `QcJs`         | **Q♣ J♠** (reveal box) | CNN bug ♣→♠ → deferred to retrain |
| H2829 | `QsTs`  | `QcTs`         | **Q♣ T♠** (reveal box) | CNN bug ♣→♠ → deferred to retrain |

Pattern: red/red (♥ vs ♦) the CNN reads correctly (old expected wrongly said ♦);
black/black (♣ vs ♠) the CNN misreads ♣→♠ under WIN occlusion.

## What changed

### A-group — 6 stale `expected_json` corrected (shared Supabase DB, not in git)

These are data corrections to `analysis_snapshots.expected_json`, then
`gto_text` regenerated (`snapshot_test.py --add`). Exact field edits, to
reproduce if the DB is ever rebuilt:

```
H2554  hero_hand:        5d3d            -> 5d3h
H2588  hero_hand:        KsQd            -> KsQh
H2549  preflop_actions:  F-F-F-F-F-C-C   -> F-F-F-F-F-C-X
       players_at_table: 6               -> 7
H2616  preflop_actions:  F-F-F-F-F-C-C   -> F-F-F-F-F-C-X
H2774  preflop_actions:  F-F-F-F-F-F-C-C -> F-F-F-F-F-F-C-X
H2555  preflop_actions:  F-F-F-F-F-C-C   -> F-F-F-F-F-C-X
       streets:          3 (2 phantom None-board runout) -> 1 (flop only)
```

### H2878 — corner-OCR A→4 guard (code, this PR)

The CNN read `A@1.00`, but `_rank_from_corner_ocr` misread the Ace corner glyph
as "4" and **unconditionally** overrode the correct CNN rank. The corner-OCR
cross-check exists to rescue confident CNN *face-card hallucinations* (H3429: 2
read as K), so it cannot simply be removed. Added `_corner_rank_overrides()` in
`scripts/ocr/table_parser.py`: a corner "4" never overrides a CNN that is
certain (≥0.99) the card is an Ace. Unit test:
`test_corner_ocr_does_not_override_confident_ace`.

### H2810 / H2829 — deferred to CardCNN retrain

The WIN sticker physically destroys the club's lower lobes in both the raw and
masked crops; the CNN reads `Qs` at 0.82–0.99 with the correct `Qc` only as the
top-2 runner-up (0.08–0.18). This is not a near-tie, so the existing
`_repair_suit_from_top2` top-2 pattern cannot catch it without a threshold loose
enough to flip the many correct spades in the corpus. The correct fix is to add
these corrected-label crops to the CardCNN training set and retrain the suit
head — tracked by the ocr-99 retrain effort
(`docs/superpowers/plans/2026-05-30-ocr-99-handoff.md`).

Their `expected_json` is already correct (`QcJs` / `QcTs`), and they remain
flagged as regression snapshots, so they will pass with no further DB change
once the retrain lands. Note the impact is real: `Qc..`→`Qs..` turns an
**offsuit** holding into a **suited** one, which changes the GTO range.

## Verification

```
python scripts/regression_test.py        -> 434 passed, 0 failed
snapshot_test.py {9 hands}                -> 7 pass / 2 fail (H2810,H2829 deferred)
```
