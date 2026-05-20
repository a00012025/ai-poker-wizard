---
name: pokercraft-scraping
description: End-to-end workflow for harvesting a benchmark-grade hand corpus from my.pokercraft.com — bulk Hand-History download, per-hand replay-image scrape, ground-truth derivation, and the title-OCR integrity loop that catches silent label corruption. Use when building, extending, or repairing the OCR ground-truth dataset, or when reviving the scraper after a session expiry. Triggers on phrases like "爬 PokerCraft", "更新 hand history corpus", "重跑 ground truth", "OCR dataset 校驗", "scrape replay images".
---

# PokerCraft Data Extraction

The pipeline that produced `data/hand_images/img/*.png` (7,183 replay images) +
`data/pokercraft_corpus/hh/*.txt` (HH source) +
`data/pokercraft_corpus/ground_truth/ground_truth.jsonl` (the OCR benchmark
oracle), keyed end-to-end by `hand_id`.

Two non-obvious truths drove the final design — internalize them before
writing any code:

1. **The replay image's title bar is the only authoritative "which hand is
   this image"**. No DOM element ever exposes it. Anything else (the in-modal
   right arrow's order, the list table's position, the row you clicked) is a
   *hint* that can drift; only the title-bar pixels can be trusted.
2. **Build the integrity check before the scraper, not after.** A scraper
   that's silently wrong looks identical to one that's right. The title-OCR
   verifier is what made the Daily-Hyper-1 off-by-one visible.

## When to use

- First-time corpus build (past N days of tournaments).
- Backfilling new tournaments after the corpus is stale.
- Repairing the dataset after any scraper change or suspected corruption.
- Recovering after a session/token expiry mid-run.

## Prerequisites

- **Single-use PokerCraft session URL** from the user. Format:
  `https://my.pokercraft.com/?token=<...>&lang=en`. Expires in ~1–1.5 h;
  ask for a fresh one right before each run, not at the start of the chat.
- **agent-browser** installed; use an isolated session
  (`AGENT_BROWSER_SESSION=pokercraft`) so the scraper never collides with
  the user's main browsing.
- **tesseract** (`brew install tesseract`) for title OCR. The macOS sandbox
  blocks tesseract from reading temp files — always feed PNG bytes via
  stdin (see `scripts/title_ocr.py`).
- **caffeinate** for long unattended runs:
  `nohup caffeinate -dimsu >/dev/null 2>&1 &`. Kill it when done.
- Local Python deps: `pip install -r requirements.txt`. The scraper imports
  from `src/` for the production parse; the benchmark does too.

## The pipeline

```
[1] tournament list  →  [2] bulk HH (ZIP)  →  [3] ground_truth.jsonl
                                                     ↓
                                              [5] OCR benchmark
                                                     ↑
[4] per-hand replay images  ──────────────────────────
              ↑↓
   [6] title-OCR verifier + repair  (run this BEFORE trusting any of the above)
```

### 1. Open the session and pin a tab

```bash
AGENT_BROWSER_SESSION=pokercraft agent-browser open \
  "https://my.pokercraft.com/?token=<...>&lang=en"
AGENT_BROWSER_SESSION=pokercraft agent-browser open \
  "https://my.pokercraft.com/tournament"
# pin this tab id — pass it to every script via --tab
AGENT_BROWSER_SESSION=pokercraft agent-browser tab list
```

Set the date filter you want (default UI filter is fine; the scraper reads
whichever rows the list shows). 30 days yields ~165 tournaments / ~7 K
hands for a moderately active player.

### 2. Bulk Hand-History download (ZIP, fast)

```bash
AGENT_BROWSER_SESSION=pokercraft python scripts/pokercraft_download.py \
  --tab t1 --out data/pokercraft_corpus/hh
```

PokerCraft's "Download Game Histories" button returns a blob ZIP per
selection. The script clicks "Select All" + "Download" and unzips to the
output dir. One `.txt` per tournament, e.g.
`GG20260419-1200 - 35 Mega to APT Taipei 2.2M GTD ME Live Day 1 [7 Seats].txt`.

**Naming gotcha**: `GG{YYYYMMDD}-{seq}` — the `{seq}` is a per-day *batch* id,
NOT the tournament time. Multiple distinct tournaments share a single batch
id (e.g. `GG20260503-1041` covers Bounty Hunters Sunday Big One, Sunday
Special, Mini Superstack Turbo 3, Sunday Turbo 2 — four different
tournaments). Don't pivot on it; pivot on the GT body's `tournament_id`.

### 3. Derive ground truth from HH

```bash
python scripts/build_ground_truth.py \
  --hh data/pokercraft_corpus/hh \
  --out data/pokercraft_corpus/ground_truth/ground_truth.jsonl
```

Per row:
```json
{"hand_id":"TM5846885824","tournament_id":"277038423",
 "source_file":"GG20260419-1200 - 35 Mega ... .txt",
 "ground_truth":{"hero_hand":"TdAc","hero_position":"UTG",
                 "effective_bb":0.1,"preflop_actions":"F-F-...",
                 "streets":[...]}}
```

`hh_parser.parse_hand` is the oracle. Regression-tested for ante-level
formats and the sub-ante-all-in walk gate. If a parse returns None for a
non-walk hand, **fix the parser, don't drop the hand** — the corpus must be
exhaustive or the benchmark will silently miss the failure modes.

### 4. Per-hand replay images (the careful part)

```bash
AGENT_BROWSER_SESSION=pokercraft python scripts/scrape_hand_images.py \
  --tab t1 --out data/hand_images \
  --gt data/pokercraft_corpus/ground_truth/ground_truth.jsonl
```

The scraper is **self-correcting**: it OCRs each captured frame's own
title bar and saves the file by THAT id. Arrow position is just a fast
walk; the name is never trusted to it.

For every tournament:
1. `open_game_history(row_i)` — clicks the row's date cell (tagged with
   `aria-label='SCRAPE_ROW'`, then Playwright-ref clicked; **NEVER match by
   `innerText`** — Angular renders garbled multi-cell text that the a11y
   label never equals, which silently skipped half the older events).
2. `set_page_size_max()` — picks "100 Rows" (or largest available) so the
   per-page anchor covers the whole tournament in one walk.
3. For each list page: `open_hand_by_id(page_ids[0])` (anchor), `set_bb_view()`
   once, then `nav_right()` through the rest. After every grab:
   `read_title_id(png, valid=gt_ids)` decides the filename.

#### Critical idioms (each one is a scar, not a preference)

- **Modal close: JS-click `.cdk-overlay-backdrop`.** Escape doesn't work. The
  "OK" button is inert/off-viewport. A stranded backdrop is fatal — every
  subsequent agent-browser snapshot only sees the overlay and all table
  ref-clicks fail. The single reliable close is the backdrop click; do it
  in a loop until `_overlay_gone()`.
- **BB toggle: raw-mouse only.** `agent-browser click @ref` is silently
  ignored on the BB header button. Use `mouse move / mouse down / mouse up`
  and poll `button.getAttribute('active')==='true' && #hand-scene.src
  changed` to confirm. The third button in `app-hand-scene-modal` is the
  BB button (no stable selector exposed).
- **`#hand-scene` exists ≠ scene is the right hand.** The modal reuses one
  `<img>`, so the previous frame lingers. `open_hand_by_id` must wait for
  the src to *change from the pre-click value AND remain stable across two
  reads*. This is the single defence against the Daily-Hyper-1
  off-by-one — clicking row X always loads X's distinct image, so
  "src ≠ pre AND stopped changing" pins the right hand without OCR.
- **The hand id is NOT in the DOM.** Verified with a live probe: modal
  text is just `"OK"`, no row gains an `aria-selected`/`active` class as
  you arrow, the URL never changes, the `<img>` has no useful attribute.
  The title-bar pixels are the only source.
- **Tournament rows can duplicate.** Same name + same time may render as
  several rows; only the one whose Game History page-ids intersect the
  target set is the right tournament. Don't dedupe by tkey alone —
  hand-id intersection (`want = page_ids ∩ gt_ids − done`) is the
  arbiter.
- **`done_tournaments.json` skip-opt.** A tournament fully covered
  (`want_seen == 0 and not errored`) is recorded by `tkey = date|name`
  and skipped on resume without even opening Game History. Saves entire
  token windows on big corpora. Has an escape hatch (`--ignore-done-tours`)
  for when a tkey was wrongly added (e.g. an earlier buggy run errored
  silently).
- **The right arrow walks the loaded *page*, not the tournament.** It
  stalls at ~page_size steps. For >100-hand tournaments, set page size
  to 100 and re-anchor per list page; do NOT try to arrow across the
  full tournament from a single anchor.

#### Filenames are not labels — the title is

```python
from title_ocr import read_title_id
oid, votes, total = read_title_id(png_bytes, valid=gt_ids)
# save as f"{oid}.png" — not as page_ids[k]
```

`read_title_id` is a calibrated multi-variant majority-voted reader (top
~4% crop, LANCZOS 6–8×, tesseract psm 7/6/11 with digit whitelist). On the
production dataset it returns a confident id for 100% of non-truncated
titles. Treat anything < majority as None and flag for one retry, then
record `ocr_failed: true` in the manifest.

#### Long-name live events: titles get truncated, use `--by-row-match`

```
HH $35 Mega to APT Taipei $2.2M GTD ME Live Day 1 [7 Seats]
```

That title fills the bar; the `-#TM<id>` suffix is dropped entirely.
`read_title_id` returns None for the whole tournament — there is no id in
the image to recover. For these events, the file's correctness must come
from clicking the right row in the first place:

```bash
... --by-row-match "Mega to APT"
```

By-row mode opens *each* wanted hand by its own row click (no arrow,
no title OCR). Because `open_hand_by_id` now waits for a fresh settled
scene, the file we save IS the hand we clicked — correct by construction.
Slow (~5–8 s/hand vs ~1 s for arrow), so reserve it for truncated-title
tournaments only.

#### Targeted backfill: `--only-date` + `--ignore-done-tours`

When you know a backfill belongs to a narrow date window (e.g. "the 38
missing hands are all on May 03"), filter the row list cheaply by the
date cell — no Game History open needed:

```bash
... --gt data/hand_images/may03_targets.jsonl \
    --only-date "May 03" --ignore-done-tours
```

Combined with the page-1 fast-skip the scraper already does (break out
of a tournament if its page-1 ids share nothing with the target set),
this cuts a 165-row walk to a focused ~13-row pass that finishes in
minutes.

#### Resumability

- `<hand_id>.png` already present → skipped automatically.
- `done_tournaments.json` skips already-covered tournaments without
  opening them.
- `manifest.jsonl` records every grab attempt with `page`, `k`, `by_row`,
  and `ocr_failed` flags — your audit trail. Don't delete it.

### 5. OCR benchmark (the deliverable the corpus enables)

```bash
python scripts/ocr_benchmark.py \
  --images data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --out data/benchmark --limit 200 --concurrency 4
```

Pairs `<hand_id>.png` with its GT row, runs the *production* image parse
(`GeminiSessionManager._parse_hand_from_image`), canonicalises both sides
(card-set order, `R2` vs `R2.0`, `effective_bb` tolerance), and emits
hand-exact precision + per-field accuracy + critical-error rate.

Critical fields = `{hero_hand, hero_position, board, preflop_action types}`.
Headline metric is "hand-level exact match on critical fields". Target:
**≥ 99.9 %**. The full corpus is what makes that target meaningful.

### 6. Integrity loop — the thing that catches silent corruption

If you only do one thing in this pipeline besides scraping, do this:

```bash
# pre-flight scan (parallel, ~12 min for 7 K images): no mutation
python scripts/verify_image_labels.py data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --workers 10 --report data/hand_images/label_report.json

# if mislabels found, apply the saved plan deterministically (no re-OCR)
python scripts/verify_image_labels.py data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --from-report data/hand_images/label_report.json --apply
```

The verifier reads each image's own title and classifies every file as
CORRECT / MISLABEL / UNREADABLE. `--apply` two-phase-renames mislabeled
files to their true id, quarantines collisions and "untrackable" (true id
∉ GT) into `data/hand_images/quarantine/`. The plan is JSON so you can
review it before mutating.

**Expected pass on a healthy dataset**: 0 mislabel, 0 unreadable —
*except* for tournaments scraped via `--by-row-match` (truncated titles),
which legitimately have no `#TM` in the image. Filter those out before
asserting:

```python
mega = {h for h,sf in gt.items() if sf.startswith("GG20260419-1200")}
# expect title-unreadable ⊆ mega, mislabel == 0
```

## Recovery playbooks

### "Session expired mid-run"

The script will start failing on snapshot/click. Just stop it (it's
resumable):
```bash
pkill -f scrape_hand_images.py
```
Get a fresh tokenized URL from the user, re-open it in the pinned tab,
relaunch the same scraper command. Resume picks up from existing
`<hand_id>.png` files + `done_tournaments.json`.

### "Image saved, but the title says the wrong id"

Run the integrity loop (step 6) on the affected slice. If it confirms
mislabels:
1. `--apply` the rename plan (renames to true ids, quarantines dupes).
2. Recompute the missing set: GT ids with no resulting image.
3. Re-scrape just those missing via the normal `--gt full` run (with
   `--ignore-done-tours` if the affected tkey was wrongly recorded done).

### "Modal won't close, all subsequent ref-clicks fail"

A `.cdk-overlay-backdrop` is stranded. The scraper's `clear_overlay()`
should handle this; if not, inject:
```js
document.querySelector('.cdk-overlay-backdrop')?.click()
```
Loop until `!document.querySelector('.cdk-overlay-backdrop')`.

### "A whole tournament has 0/N images and the log says `could not open Game History`"

Almost always row-tagging fell over because the row was virtualized or
out of viewport. The scraper now scrolls + tags via `aria-label='SCRAPE_ROW'`
+ ref-click; if it still fails, the tournament name probably has special
characters in the date cell that broke the snapshot. Verify the row is
present via `tournament_rows()` and the date cell is non-empty.

### "Arrow stalls at k≈100 in a 200-hand tournament"

Page size wasn't maxed. Confirm `set_page_size_max()` returned 100 (or
larger). If the "N Rows" menu trigger isn't present (some smaller
tournaments lack it), the default is 10 and the walk will indeed stall —
in that case break and re-anchor per page (this is what the current
per-page anchor loop already does).

## Lessons (architectural, not just operational)

- **Authoritative-by-construction beats clever heuristics.** The arrow
  walk was clever and ~99 % right; "the title is the label" is dumb and
  100 % right. Pick the dumb one.
- **An integrity oracle that's stronger than the scraper is non-optional.**
  The old verify printed `unreadable=29 %` and `mislabelled=0`, looking
  fine. The bulletproof reader showed `unreadable=0 %` and
  `mislabelled=78` — same data, opposite conclusion. If your check is
  weaker than your producer, your producer is unchecked.
- **Resume optimisations must have escape hatches.** `done_tournaments`
  was great until a buggy earlier run silently added wrong tkeys.
  `--ignore-done-tours` is the safety valve.
- **Filter cheap, then deep.** `--only-date` filters on a row cell text
  (~free) before paying the ~15 s of Game-History open. A 165-row walk
  becomes 13 rows × full work. Apply the cheapest discriminator first.
- **Two-phase renames are mandatory.** When the relabel plan is a permutation
  (A → B, B → C, C → A), single-step renames clobber. Stage every move
  to `__stage_<target>.png`, then promote in a second pass.
- **The benchmark is the point.** Everything in this skill exists so that
  `ocr_benchmark.py` is measuring real, correctly-paired data. A 25 %
  accuracy on correctly-paired data is a finding. A 99 % accuracy on
  mispaired data is fiction.

## File map (for the next person to read)

| Script | Role |
|---|---|
| `scripts/pokercraft_download.py` | Bulk HH ZIP downloader |
| `scripts/hh_parser.py` | HH text → structured hand (the GT oracle) |
| `scripts/build_ground_truth.py` | HH dir → `ground_truth.jsonl` |
| `scripts/scrape_hand_images.py` | Per-hand replay PNG scraper (self-correcting, by-row, only-date) |
| `scripts/title_ocr.py` | Bulletproof title-bar reader |
| `scripts/verify_image_labels.py` | Integrity verifier + relabel/quarantine |
| `scripts/ocr_benchmark.py` | Pair images with GT, run production parse, score |

| Data | What |
|---|---|
| `data/pokercraft_corpus/hh/*.txt` | Raw HH source (~165 files) |
| `data/pokercraft_corpus/ground_truth/ground_truth.jsonl` | Parsed GT, one row per hand |
| `data/hand_images/img/<hand_id>.png` | Replay PNG, BB view, named by true id |
| `data/hand_images/manifest.jsonl` | Audit log of every grab |
| `data/hand_images/done_tournaments.json` | Skip-opt cache |
| `data/hand_images/quarantine/` | Dupes & untrackable files from repair runs |
