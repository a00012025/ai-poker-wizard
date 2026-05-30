# Handoff: 9 pre-existing L1-OCR snapshot mismatches

Date: 2026-05-30 (Asia/Taipei)
Repo: `/home/harry/ai-poker-wizard`
Surfaced in: PR #36 (`fix/allin-call-deviation`) full snapshot run — these are
**NOT** caused by that PR. They reproduce identically on `main` and every one
of them **passes Layer-2 (GTO)**; only **Layer-1 (OCR re-parse vs expected_json)**
fails. They were pre-existing and simply became visible when the full
`snapshot_test.py` suite was run.

## TL;DR

`python scripts/snapshot_test.py` → **77 passed, 9 failed**. All 9 failures are
`FAIL L1-OCR: Parse mismatch`. The deterministic GTO analysis (L2) is fine for
all of them. The job for the next session is to fix the OCR so the re-parse
matches the human-verified `expected_json`.

The 9 split into **two clusters**:

- **Cluster A — same-color suit (and one rank) misread on the hero's hole cards** (5 hands)
- **Cluster B — preflop trailing `C` parsed as `X`, plus structural drift** (4 hands)

Every misread is **deterministic** (verified by running each hand twice — byte-
identical output), so these are systematic parser bugs, not flaky noise. They
are debuggable by dumping the specific image and inspecting the classifier.

## The 9 failures (exact mismatches)

| Hand  | conf | Mismatch (got → expected) | Cluster |
|-------|------|---------------------------|---------|
| H2554 | 1.00 | `hero_hand: 5d3h → 5d3d` (♥→♦, red↔red) | A |
| H2588 | 0.99 | `hero_hand: KsQh → KsQd` (♥→♦, red↔red) | A |
| H2810 | 0.97 | `hero_hand: QsJs → QcJs` (♠→♣, black↔black) | A |
| H2829 | 0.93 | `hero_hand: QsTs → QcTs` (♠→♣, black↔black) | A |
| H2878 | 0.89 | `hero_hand: 4c2h → Ac2h` (rank 4→A) | A |
| H2549 | 1.00 | `preflop_actions: …-C-X → …-C-C`; `players_at_table: 7 → 6` | B |
| H2555 | 1.00 | `preflop_actions: …-C-X → …-C-C`; `streets count: 1 → 3` | B |
| H2616 | 1.00 | `preflop_actions: …-C-X → …-C-C` | B |
| H2774 | 1.00 | `preflop_actions: …-F-C-X → …-F-C-C` | B |

(Confidence is the parser's own score. Note Cluster B is all `conf=1.00` —
confidently wrong, so the abstain gate will not catch these; they need a
parser-logic fix, not a confidence threshold.)

## Reproduce

```bash
cd ~/ai-poker-wizard            # clean main reproduces all 9

# One hand (dumps nothing; just pass/fail + mismatch):
python scripts/snapshot_test.py H2554

# All snapshots (buffered when piped — use -u if redirecting):
PYTHONUNBUFFERED=1 python -u scripts/snapshot_test.py | tee /tmp/snap.txt
grep -A3 "FAIL L1-OCR" /tmp/snap.txt
```

To get the image + the stored expected for a hand, follow the `/fix-hand`
skill Step 1 (dump `image_data` from `analysis_snapshots` to `/tmp/HXXXX.jpeg`
and **Read it visually** to confirm the ground truth). Set corrected fields
with `snapshot_test.py --set-expected` only if the stored expected is itself
wrong — but the user already hand-verified these, so treat `expected_json` as
ground truth and fix the OCR.

## Cluster A — hero hole-card misreads (5 hands)

Pattern: **same-color suit confusion** dominates — ♦↔♥ (both red) and ♠↔♣
(both black) — plus one rank slip (4↔A, H2878). The rank/suit of the WRONG
card varies (1st card in H2810/H2829/H2878, 2nd card in H2554/H2588), so this
is not a fixed-position crop bug; it is the **suit classifier failing the
red-vs-red / black-vs-black sub-decision**, and CardCNN missing A→4 on a
low-confidence (0.89) card.

Where to look:
- `scripts/ocr/table_parser.py` — `_find_hero_cards`, `_locate_hero_cards`,
  suit detection (`_detect_suit_bgr` / corner OCR / `_repair_suit_from_top2`),
  rank path (`_rank_from_corner_ocr`, `_repair_rank_from_top2`).
- `scripts/ocr/card_matcher.py` — CardCNN rank/suit classifier.
- The `retrain-card-classifier` skill — if the misreads trace to the CNN's
  same-color suit head being under-trained, a targeted relabel + retrain on
  these 5 (plus siblings) is the likely fix. Confirm first whether it's the
  CNN or the BGR/contour suit heuristic that is wrong for each hand.

Suggested approach: for each of the 5, dump the hero-card crops and print the
classifier's top-2 rank and top-2 suit with confidences. If the correct answer
is consistently top-2-but-not-top-1 on the same-color suit, that localizes the
fix to the suit head (threshold/feature or retrain). H2878 (A→4) is a separate
rank-head miss — check whether the ace pip/corner is being cropped.

## Cluster B — preflop trailing `C`→`X` + structure (4 hands)

All four mis-parse the **last preflop action as a check (`X`) instead of a call
(`C`)** — i.e. a limped pot where the BB's "Call"/complete is being read as a
"Check". This is the classic limp-pot Call-vs-Check ambiguity in the panel
action classifier. Two also have a structural error on top:
- **H2549**: `players_at_table 7 → 6` (table-size over-count).
- **H2555**: `streets count 1 → 3` (flop/turn/river dropped — only preflop
  survived; a much bigger parse failure, investigate separately/first).

Where to look:
- `scripts/ocr/panel_parser.py` — action word classification (`Call` vs
  `Check`), `_classify_group`. The yellow/white sticker + OCR text decides
  Call vs Check; check whether a sized "Call 1 BB" / blind-complete is losing
  its size and defaulting to Check.
- `scripts/ocr/n8_parser.py` — `_assemble_hand` preflop walk that emits
  `preflop_actions`, the `players_at_table` inference, and (for H2555) the
  street assembly that dropped flop/turn/river.

Note: a BB **checking** its option vs **calling** can look semantically close,
but the user-verified ground truth is `C`. Decide on the real rule (in a
limped, unraised pot the BB closing action is a check, but these expecteds say
C — likely SB completed so BB is calling the completion). Verify against each
image before changing the classifier, so you don't regress genuine BB-check
hands elsewhere in the corpus.

## Acceptance criteria

- `python scripts/snapshot_test.py` for each of the 9 hands → **L1-OCR PASS**
  (and keep L2-GTO passing).
- Full suite returns to the pre-existing-clean baseline (the 9 gone, nothing
  new broken). The hand corpus has known same-color suit fragility, so re-run
  the **whole** suite, not just the 9, to catch collateral regressions.
- Per the repo rule, **every fix gets a regression test** in
  `scripts/regression_test.py` (unit test on the classifier/parser helper) in
  addition to the snapshot passing. Prefer a deterministic unit test over
  relying on the snapshot alone.

## Do NOT

- Do not "fix" these by editing `expected_json` — the user hand-verified the
  expecteds; the OCR is what's wrong.
- Do not touch the all-in/call deviation logic from PR #36 — it is orthogonal
  and already green.
- Do not raise/lower the confidence gate to mask Cluster B — they are
  `conf=1.00` confident errors; the gate is the wrong lever.

## Environment notes

- Fresh worktrees lack `.gto_cache/` and the gitignored `data/` dir; symlink
  both from the main checkout before running (see `[[worktree-gto-cache]]`).
  L2 needs the cache; the CardCNN crop tests need `data/pokercraft_corpus/`.
- OCR is deterministic per image — if you can't reproduce a misread, you have
  an environment problem (wrong model weights / missing `data/`), not a fixed bug.
