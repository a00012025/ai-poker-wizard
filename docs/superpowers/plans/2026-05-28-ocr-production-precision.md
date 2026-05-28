# OCR 99% @ 95% — Final Push (No Excuses)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development). Steps use checkbox (`- [ ]`) syntax for tracking. This plan has a single non-negotiable ship gate — do not propose interim targets, partial wins, or "documented gap" handoffs.

**Goal:** On main's current state, deliver `precision ≥ 99.0% AND recall ≥ 95.0%` on **BOTH** the PokerCraft test bucket AND the production_v1 test bucket. Single non-negotiable target. Lower precision or coverage is not acceptable. Calibration overfit (τ-drift > 10% dev↔test) is not acceptable.

**Starting line (verified 2026-05-28):**
- PokerCraft test (main, `data/ocr_precision_phase10_test`): `hand_exact 90.738% @ coverage 81.198%`, parse_none 135 (67 failure + 68 abstain), board_wrong 25, position_wrong 21, hero_cards_wrong 5
- production_v1 test (main, `data/ocr_precision_production_baseline_v2`): `hand_exact 71.429% @ coverage 70.000%` (n=10, sample tiny — must grow)
- Joint ECE-10bin: 0.0673 (over the ≤ 0.04 ship bound)
- Existing trained calibrator `data/calibrator/`: OOF on test bucket reaches 97.83% precision @ 72.1% coverage and 100% @ 40.7% (per `confidence_gate.evaluate_with_calibrator` docstring). Trained on the *May 23* corpus with the *May 23* feature set; does not include current diagnostics or production traffic.

**The non-negotiable architecture:** Calibrator-first. A May 28 attempt to reach 99% via overlay augmentation + multi-crop ensemble + τ-only calibration failed: CardCNN v3 candidates regressed (`card 0.962` vs v2 `0.967`) because the 41-px hero crop has ~10 px of rank glyph and can't absorb occlusion augmentation; and the joint precision-coverage curve over PokerCraft test ∪ production_test reports best τ=0.8625 → 90.24% @ 83.41% on PokerCraft — **no τ at any coverage hits 99%**. The May 23 ship at 99.016% @ 70.75% was reached by a trained RF+GB+LR calibrator, not by τ-tuning. That is the only path that has ever worked on this codebase. This plan walks it again, against the new (post-May-23) main and the new production-distribution bucket.

**Three legs, in strict order:**
1. **Grow production_v1 to ≥ 200 verified hands** — the May 28 attempt had 62, which yields a 9/10 val/test that is too noisy to calibrate against. We backfill from every snapshot that has either `expected_json` OR a high-confidence cross-check signal with Gemini's reparse.
2. **Build the rich-feature record dump** — re-run both buckets with `--dump-all`, capturing for every hand the full `confidence_parts`, `diagnostics`, `hero_card_details`, ensemble votes (when wired), and `raw_vs_masked` arbitration outcomes. This is the calibrator's training data.
3. **Retrain the RF+GB+LR ensemble calibrator** on PokerCraft train+val ∪ production_train+production_val with the expanded feature set. Validate jointly on `test ∪ production_test`. Iterate the feature set (add `ensemble_used`, per-card `rank_top2[1]_conf`, masked-vs-raw suit swap flag, postflop-collapse-demoted flag) until joint 99% @ 95% holds.
4. **Residual parser fixes** — only the errors the calibrator demonstrably cannot isolate. Each fix must list (a) which records it changes, (b) which feature would have isolated it instead and why that feature can't be added.

**Tech Stack:** Python 3.11, OpenCV, PyTorch, scikit-learn (RandomForest + GradientBoosting + LogisticRegression ensemble), pytest, supabase-py.

---

## What survives from the May 28 attempt

The May 28 branch `feat/ocr-production-precision` was discarded as not-good-enough, but its gitignored on-disk artifacts persist. Use them or refresh; do not assume they don't exist:

- `data/win_overlays/` — 60 RGBA overlay templates harvested from `analysis_snapshots`. Format: BGR + alpha, opaque = sticker pixel, transparent = card. May need top-up but the corpus is real.
- `data/cards_v2/production_v1/` — 62 verified snapshots:
  - `images/<hand_id>.png` — full screenshot per hand
  - `<label>/<hand_id>_hero_<slot>.png` — labeled hero crops (124 total)
  - `gt.jsonl` — `{hand_id, ground_truth: <merged parsed+expected>}` per line
- `data/splits/production_v1.json` — 43/9/10 train/val/test split file with `production_train` / `production_val` / `production_test` keys.
- `data/calibrator/` — May 23 trained RF+GB+LR + feature names + OOF predictions. The trained models are stale-on-features but the loader (`scripts/ocr/confidence_gate.CalibratorScorer`) is real and works.
- `data/ocr_precision_phase10_test/` — post-Phase-10 PokerCraft test baseline (583 emitted / 718 paired, hand_exact 90.74%). Reference for "what main does today."

**Code that was on the discarded branch and must be re-implemented this session:**
- `scripts/ocr/classifier/capture_overlays.py` — `extract_overlay(image_bytes) → RGBA | None` using HSV yellow band + dilation + chip-stack filter
- `scripts/ocr/classifier/overlay_library.py` — `OverlayLibrary(root)` with `.size()` and `.sample(rng)`
- `scripts/ocr/classifier/harvest_production.py` — `harvest_snapshot(*, hand_id, image_bytes, expected, out_dir) → int` + `harvest_corpus`
- `scripts/ocr/classifier/ensemble.py` — `predict_with_ensemble(crop) → {label, card_conf, votes}` with **majority-≥2/3 safety** (confidence-vote-only is unsafe; H3433 case proved it)
- `scripts/ocr/classifier/augment.py` — `apply_real_win_overlay(img, *, rng, p=0.30, lib)` with `p=0.30` real / `p=0.10` synth in `apply_all` (the higher rates regressed v2)
- `scripts/ocr/n8_parser.py` + `scripts/ocr/table_parser.py` — route to ensemble when raw `card_conf < OCR_ENSEMBLE_FLOOR=0.50`, bubble `ensemble_used` through `_build_diagnostics`
- `scripts/ocr_precision.py` — `--bucket production_{train,val,test}` switches `--images` to `data/cards_v2/production_v1/images/` and `--ground-truth` to `data/cards_v2/production_v1/gt.jsonl`

For each, the test contract from the May 28 attempt is recorded in this plan under the relevant phase. The code was correct; the strategy (overlay aug as the lever) was wrong. We re-ship the code (it's load-bearing infrastructure) and use the calibrator as the lever.

**Lessons that constrain this plan:**
1. **Do not promote CardCNN v3 trained via overlay aug.** Two attempts at `p=0.70` and `p=0.30` real-overlay rates produced `card 0.960 / 0.962`, both worse than v2 `0.967`. The 41-px crop cannot absorb occlusion-style augmentation. If you train v3 again, use a different lever (per-class focal loss on rank 4/9, hard-example mining over confidence_abstain hands, distillation from a higher-res re-extract) — not overlay aug rate adjustments.
2. **Do not change `confidence_gate.py`'s default `emit_threshold`.** It cannot reach the goal via τ alone. The gate moves only when `evaluate_with_calibrator` is wired as the default, *and* the calibrator passes the joint 99% @ 95% check.
3. **Do not write fallback ship rules.** No "if calibrator can't reach 99%@95%, accept 98%@95%". No "loosen to 70% with user sign-off". The plan ships at 99%@95% or it does not ship.
4. **Do not trust a calibrator that hasn't been validated jointly.** The May 23 calibrator was validated only on PokerCraft test. Joint validation against production_test is required this time.

---

## Failure Budget at the Target

On the held-out joint set (PokerCraft test: 718 paired + production_v1 test: 10 paired, growing to ≥ 30 after Phase 11.A) the math is:

| Bucket | Paired | At ≥ 95% recall, emitted | At ≥ 99% precision, wrong allowed |
|---|---|---|---|
| PokerCraft test | 718 | ≥ 683 | ≤ 6 |
| production_v1 test (post-expansion ≥ 30) | ≥ 30 | ≥ 29 | ≤ 0 (rounding floor; effectively 0) |

Today (main):
- PokerCraft: 583 emitted, ~54 wrong (90.74% precision). Must recover 100+ currently-non-emitted exact AND demote 48+ wrong.
- production_v1: 7 emitted, 2 wrong on n=10. Must extend the corpus and replicate the same precision/recall lift.

This is a **calibrator-engineering problem**, not a parser-correctness problem at this point. The May 23 path (train ensemble on `confidence_parts` + 27 diagnostic features) reached 99% @ 70%. Adding ensemble-used, per-card top2-spread, demote-to-Gemini flag, masked-vs-raw arbitration, and production-bucket-trained data should close the recall gap from 70% to 95%.

If after the calibrator retrain we still cannot hit 99%@95%, the residual cases must be reduced by parser fixes targeted exactly at the records the calibrator could not separate (Phase 11.D). At no point do we lower the target.

---

## Phase 11.A — Production_v1 Corpus Expansion (≥ 200 hands)

**Why this phase exists first:** A 62-hand corpus with a 9-hand val and 10-hand test is statistical noise at the 99% precision target — one wrong emitted = 11% precision loss on production_test. Calibration trained against it cannot generalise. We must grow the corpus before the calibrator can learn anything reliable on production traffic.

**Entry criteria:** None.

**Exit criteria:**
1. `data/cards_v2/production_v1/gt.jsonl` has ≥ 200 entries, each with a `hand_id` and a `ground_truth` dict whose `hero_hand` is verified.
2. `data/splits/production_v1.json` re-built — `production_train ≥ 140`, `production_val ≥ 30`, `production_test ≥ 30`.
3. Verification policy is recorded: every entry's source (`analysis_snapshots.expected_json`, `gemini_reparse_agreement`, or `manual`) is in the gt.jsonl row.
4. `python scripts/ocr_precision.py --split data/splits/production_v1.json --bucket production_test --workers 4 --out data/ocr_precision_production_v2_baseline` runs and writes a summary with `paired ≥ 30`.

### Task 11.A.1 — Re-implement `harvest_production.py` and rebuild the corpus

**Files:**
- Create: `scripts/ocr/classifier/harvest_production.py`
- Create: `tests/ocr/test_harvest_production.py`
- Run-only: `scripts/_tmp.py` for the DB pull

- [ ] **Step 1: Write failing test**

```python
# tests/ocr/test_harvest_production.py
"""harvest_snapshot extracts labeled hero crops from a snapshot using
the user-verified expected_json, not the (possibly wrong) parsed_json."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ocr.classifier.harvest_production import harvest_snapshot


def test_harvest_extracts_labeled_hero_crops(tmp_path):
    img = Path("tests/snapshots/H3433/input.jpeg")
    if not img.exists():
        pytest.skip("H3433 fixture not present")
    expected = {"hero_hand": "6d5d"}
    n = harvest_snapshot(
        hand_id="H3433",
        image_bytes=img.read_bytes(),
        expected=expected,
        out_dir=tmp_path,
    )
    assert n >= 2
    labels = {p.parent.name for p in tmp_path.rglob("*.png")}
    assert "6d" in labels or "5d" in labels
```

Run: `pytest tests/ocr/test_harvest_production.py -v` → FAIL on `ModuleNotFoundError`.

- [ ] **Step 2: Implement `harvest_snapshot`**

```python
# scripts/ocr/classifier/harvest_production.py
"""Extract labeled card crops from analysis_snapshots for the
production_v1 corpus. Labels come from expected_json.hero_hand
(user-verified), not parsed_json (the previous OCR guess)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..region_detector import detect_regions
from ..table_parser import _locate_hero_cards, _trim_above_card_edge


def _parse_hand_into_two(hand: str | None) -> list[str] | None:
    if not hand or len(hand) != 4:
        return None
    return [hand[0:2], hand[2:4]]


def harvest_snapshot(*, hand_id: str, image_bytes: bytes,
                     expected: dict, out_dir: Path) -> int:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    regions = detect_regions(img)
    if not regions:
        return 0
    table = regions.get("table")
    if table is None:
        return 0
    hero_cards = _parse_hand_into_two((expected or {}).get("hero_hand"))
    if not (hero_cards and len(hero_cards) == 2):
        return 0
    raw_crops = _locate_hero_cards(table)
    if len(raw_crops) != 2:
        return 0
    crops = [_trim_above_card_edge(c) for c in raw_crops]
    n = 0
    for slot, (crop, label) in enumerate(zip(crops, hero_cards)):
        dest = out_dir / label.lower() / f"{hand_id}_hero_{slot}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop)
        n += 1
    return n


def harvest_corpus(snapshots: list[dict], out_dir: Path) -> int:
    total = 0
    for snap in snapshots:
        expected = snap.get("expected_json")
        if not expected or not snap.get("image_data"):
            continue
        if isinstance(expected, str):
            expected = json.loads(expected)
        total += harvest_snapshot(
            hand_id=snap["hand_id"],
            image_bytes=bytes(snap["image_data"]),
            expected=expected,
            out_dir=out_dir,
        )
    return total
```

Run: test PASSES.

- [ ] **Step 3: Re-run the DB pull to refresh `data/cards_v2/production_v1/` and `gt.jsonl`**

Pull every `analysis_snapshots` row where `source_type='image'` AND `image_data IS NOT NULL` AND `expected_json IS NOT NULL` AND `expected_json->>'hero_hand' IS NOT NULL`. Merge `parsed_json` ∪ `expected_json` (expected takes precedence) and write to gt.jsonl. Write the full image to `images/<hand_id>.png` and the labeled crops via `harvest_corpus`. The May 28 attempt's `scripts/_tmp.py` template:

```python
from dotenv import load_dotenv; load_dotenv()
import asyncio, os, sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import asyncpg
from ocr.classifier.harvest_production import harvest_snapshot

async def main():
    conn = await asyncpg.connect(os.environ["SUPABASE_CONN"],
                                 statement_cache_size=0)
    rows = await conn.fetch("""
        SELECT hand_id, expected_json, image_data, parsed_json
        FROM analysis_snapshots
        WHERE source_type='image' AND image_data IS NOT NULL
          AND expected_json IS NOT NULL
          AND expected_json->>'hero_hand' IS NOT NULL
        ORDER BY created_at DESC
    """)
    await conn.close()
    out_root = Path("data/cards_v2/production_v1")
    images_dir = out_root / "images"; images_dir.mkdir(parents=True, exist_ok=True)
    gt_lines, hand_ids = [], []
    for r in rows:
        expected = r["expected_json"] if isinstance(r["expected_json"], dict) \
                   else json.loads(r["expected_json"])
        parsed = (r["parsed_json"] if isinstance(r["parsed_json"], dict)
                  else json.loads(r["parsed_json"])) if r["parsed_json"] else {}
        truth = {**parsed, **expected}
        b = bytes(r["image_data"])
        (images_dir / f"{r['hand_id']}.png").write_bytes(b)
        harvest_snapshot(hand_id=r['hand_id'], image_bytes=b,
                         expected=truth, out_dir=out_root)
        gt_lines.append(json.dumps({"hand_id": r["hand_id"],
                                    "ground_truth": truth,
                                    "source": "analysis_snapshots.expected_json"}))
        hand_ids.append(r["hand_id"])
    (out_root / "gt.jsonl").write_text("\n".join(gt_lines) + "\n")
    # 70/15/15 by hand
    rng = random.Random(0)
    s = sorted(hand_ids); rng.shuffle(s)
    n = len(s); n_tr = int(round(n*0.70)); n_vl = int(round(n*0.15))
    split = {"production_train": sorted(s[:n_tr]),
             "production_val":   sorted(s[n_tr:n_tr+n_vl]),
             "production_test":  sorted(s[n_tr+n_vl:]),
             "meta": {"seed": 0, "n": n}}
    Path("data/splits/production_v1.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2))
    print(f"hands={n} train={len(split['production_train'])} "
          f"val={len(split['production_val'])} "
          f"test={len(split['production_test'])}")

asyncio.run(main())
```

Run: `python scripts/_tmp.py`. If `n < 200`, **Phase 11.A.2 is required** before progressing.

### Task 11.A.2 — Backfill from Gemini reparse agreement (if 11.A.1 yields < 200 hands)

**Why:** `expected_json` only exists for hands where someone ran `/fix-hand` or `snapshot_test.py --set-expected`. The natural rate of those events caps the corpus at the rate users notice errors. We supplement with hands where the full-Gemini reparse and the OCR pipeline independently agree on hero_hand AND board AND position — high agreement on multiple fields is a credible signal of correctness even without manual verification.

**Files:**
- Create: `scripts/ocr/classifier/gemini_agreement_backfill.py`
- Test: `tests/ocr/test_gemini_agreement_backfill.py`

- [ ] **Step 1: Write the test**

The test fixture-loads three records (one with OCR/Gemini agreement on all 3 fields, one with disagreement on hero only, one with disagreement on board only) and asserts that only the all-agreement record is accepted into the corpus, with `source="gemini_reparse_agreement"`.

- [ ] **Step 2: Implement the backfill**

For each `analysis_snapshots` row where `expected_json IS NULL` but `parsed_json` and `gemini_reparse_json` both exist (we may need to add a `gemini_reparse_json` column — see Task 11.A.3 if not present): check if (hero_hand, board, hero_position) match exactly. If yes, treat the parsed_json's reading as ground truth, harvest, and record in gt.jsonl with `source="gemini_reparse_agreement"`.

- [ ] **Step 3: Re-run, expand to ≥ 200, re-split**

If still under 200, the entry criteria for 11.A.3 (manual triage) opens.

### Task 11.A.3 — Manual triage of low-confidence parses (last resort)

Only if 11.A.1 + 11.A.2 yields < 200 hands. The bot owner manually verifies hero_hand for 50-100 N=production snapshots from the last 30 days. Procedure already exists via `snapshot_test.py --set-expected` and the `/fix-hand` skill. Tracked in `gt.jsonl` with `source="manual"`.

---

## Phase 11.B — Rich-Feature Record Dump

**Why this phase exists:** The calibrator's training data must include every signal the parser produced — not just the scalar `confidence`. The existing 27-feature vector in `data/calibrator/feature_names.txt` is a 2026-05-23 snapshot; it predates `ensemble_used`, the post-May-23 `demote-to-Gemini-on-collapse` flag, the masked-vs-raw arbitration outcome flag, and per-card `rank_top2[1]_conf` (the top-2 margin). We extend the feature schema, re-run both buckets with `--dump-all`, and persist the full record stream as the calibrator's training corpus.

**Entry criteria:** Phase 11.A complete (≥ 200 production hands).

**Exit criteria:**
1. `scripts/ocr/n8_parser.py` exposes a `diagnostics.ensemble_used` flag (re-implementing the May 28 routing) and per-card `rank_top2`/`suit_top2` from `table_result.hero_card_details` is preserved in the precision-harness record.
2. `scripts/ocr/classifier/ensemble.py` exists with the **majority-≥2/3** safety policy (not confidence-weighted vote alone).
3. `data/calibrator/v2_features.txt` lists the new ≥ 40-feature schema.
4. `data/ocr_precision_phase11b_pokercraft/` and `data/ocr_precision_phase11b_production/` each contain `all_records.jsonl` with per-record JSON carrying every feature in `v2_features.txt` plus `hand_exact` (the label).

### Task 11.B.1 — Re-implement `ensemble.py` with majority-only safety

**Files:**
- Create: `scripts/ocr/classifier/ensemble.py`
- Create: `tests/ocr/test_ensemble.py`

```python
# scripts/ocr/classifier/ensemble.py — load-bearing parts only
def predict_with_ensemble(crop: np.ndarray) -> EnsembleResult:
    votes: list[Vote] = []
    for name, sub in (("full", crop),
                      ("top", crop[: int(crop.shape[0] * 0.45)]),
                      ("bottom", crop[int(crop.shape[0] * 0.55):])):
        if sub.shape[0] < 10 or sub.shape[1] < 10:
            continue
        rank, suit, conf = _classifier().classify(sub)
        label = f"{rank}{suit}" if rank and suit else ""
        votes.append({"crop": name, "label": label, "conf": float(conf)})
    # Hard majority required — confidence-weighted vote can elect a
    # minority label on disagreement (H3433 card 1 case).
    counts: dict[str, int] = {}
    for v in votes:
        if v["label"]:
            counts[v["label"]] = counts.get(v["label"], 0) + 1
    majority = next((lab for lab, n in counts.items() if n >= 2), None)
    if majority is None:
        return {"label": "", "card_conf": 0.0, "votes": votes}
    agreeing = [v for v in votes if v["label"] == majority]
    card_conf = sum(v["conf"] for v in agreeing) / len(agreeing)
    if len(agreeing) == len(votes):
        card_conf = min(1.0, card_conf + 0.1)
    return {"label": majority, "card_conf": float(card_conf), "votes": votes}
```

Tests must assert:
- Empty crop input → empty label + 0.0 conf
- 3/3 agreement → label = agreed, conf ≥ raw + 0.1
- Disagreement → empty label (NOT confidence-weighted winner)

### Task 11.B.2 — Wire ensemble into `_find_hero_cards` and bubble `ensemble_used`

Replicate the May 28 wiring:
- `scripts/ocr/table_parser.py`: import `os` + `ENSEMBLE_FLOOR = float(os.getenv("OCR_ENSEMBLE_FLOOR", "0.50"))` at module top
- In `_find_hero_cards`, after building `details`, if any `d["conf"] < ENSEMBLE_FLOOR`, call `predict_with_ensemble` for that card and override (rank/suit/conf) only when `ens["label"]` is non-empty AND `ens["card_conf"] > d["conf"]`. Stash `d["ensemble_used"]`, `d["ensemble_votes"]`, `d["ensemble_label"]`, `d["ensemble_conf"]`.
- Bubble `ensemble_used = any(d.get("ensemble_used") for d in hero_card_details)` through `parse_table`'s return dict.
- `scripts/ocr/n8_parser.py:_build_diagnostics`: add `"ensemble_used": bool(table_result.get("ensemble_used"))`.

Test: `tests/ocr/test_ensemble_routing.py` — `parse_n8_screenshot(H3433_bytes)["diagnostics"]["ensemble_used"]` is `True`.

### Task 11.B.3 — Production bucket support in `ocr_precision.py`

Replicate the May 28 wiring:
- Add `production_train`/`production_val`/`production_test` to the `--bucket` choices in `scripts/ocr_precision.py`
- When bucket starts with `production_` AND user did not override `--images`/`--ground-truth`, point them at `data/cards_v2/production_v1/images` and `data/cards_v2/production_v1/gt.jsonl` respectively

Test: extend `tests/test_ocr_precision_diagnostics.py` with a `--bucket production_test --limit 3` invocation that asserts `summary.json` is written.

### Task 11.B.4 — Build the v2 feature schema

**File:** `data/calibrator/v2_features.txt` lists at least:

```
# Existing 27 (kept verbatim)
confidence
card_conf
pot_consist
player_track
ocr_conf
pre_loss
rf_diff
rf_abs
postflop_total
n_allin
n_raise
n_fold
n_call
n_actions
has_allin
has_bare_ai
has_trail_ai
has_double_ai
sr_simple
sr_complex
sr_postflop
safe_emit
button_conf
reaction
pre_loss_x_allin
pre_loss_x_track_weak
conf_x_card

# New (Phase 11 additions)
ensemble_used
hero0_rank_top2_margin
hero0_suit_top2_margin
hero1_rank_top2_margin
hero1_suit_top2_margin
hero_raw_vs_masked_suit_swapped
hero0_rank_source_is_corner
hero1_rank_source_is_corner
demote_to_gemini_fired
pre_loss_x_demote
ensemble_conf_min
ensemble_votes_agreed
```

`scripts/ocr/confidence_gate.py:_calibrator_features` must be extended to produce the v2 vector when `data/calibrator/v2_features.txt` exists; falls back to v1 when the v1 calibrator is loaded.

### Task 11.B.5 — Run both buckets with `--dump-all`

```bash
rm -rf data/ocr_precision_phase11b_pokercraft data/ocr_precision_phase11b_production
python scripts/ocr_precision.py \
  --split data/splits/card_classifier_v2.json --bucket test \
  --workers 4 --dump-all --out data/ocr_precision_phase11b_pokercraft
python scripts/ocr_precision.py \
  --split data/splits/production_v1.json --bucket production_test \
  --workers 4 --dump-all --out data/ocr_precision_phase11b_production
```

Same for `--bucket train`, `--bucket val`, `--bucket production_train`, `--bucket production_val` — these become the calibrator's training set. Each `all_records.jsonl` line carries every Phase 11.B feature plus `hand_exact`.

---

## Phase 11.C — Calibrator Retraining + Joint Validation

**Why this phase exists:** The hard work. Train the RF+GB+LR ensemble on the combined train+val records from both buckets, with the v2 feature vector, and find the joint τ that satisfies 99% precision AND 95% recall on `test ∪ production_test`. The May 23 attempt reached 99% @ 70.75% on PokerCraft test alone with the v1 features; with v2 features and production-bucket coverage in training, the recall floor should lift.

**Entry criteria:** Phase 11.B complete; `all_records.jsonl` exists for each of train, val, test, production_train, production_val, production_test.

**Exit criteria (single ship gate):**
1. PokerCraft test: `precision ≥ 99.0%` AND `recall ≥ 95.0%` AND `ECE-10bin ≤ 0.04`.
2. production_v1 test: `precision ≥ 99.0%` AND `recall ≥ 95.0%` AND `ECE-10bin ≤ 0.04`.
3. `|τ_dev − τ_test_breakeven| / τ_dev ≤ 10%` on both buckets (no overfit).
4. `data/calibrator/rf_model_v2.joblib`, `gb_model_v2.joblib`, `lr_model_v2.joblib` saved.
5. `confidence_gate.CalibratorScorer` is wired to load v2 by default; v1 stays as fallback.
6. `python scripts/regression_test.py` passes.
7. `pytest tests/ocr -q` passes (matching main's baseline failure count).

Any one of these failing → return to Phase 11.B (add features) or Phase 11.D (parser fixes). **Do not lower the target.**

### Task 11.C.1 — Train RF+GB+LR on joint dev

**Files:**
- Create: `scripts/ocr/classifier/train_calibrator_v2.py`
- Test: `tests/ocr/test_calibrator_v2_train.py` (smoke test that the script runs against a fixture jsonl and produces `.joblib` outputs)

Calibrator architecture:
- Three independent base models on the joint train+val records:
  - `RandomForestClassifier(n_estimators=400, max_depth=8, class_weight="balanced", random_state=0)`
  - `GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)`
  - `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)`
- Stacking: average `predict_proba` across the three for the final `p(correct)`.
- 5-fold CV on the *joint* train+val set, with the val-out fold's predictions written to `data/calibrator/oof_v2.json` for ECE/precision-coverage analysis.

Train command:
```bash
python -m scripts.ocr.classifier.train_calibrator_v2 \
  --pokercraft-train data/ocr_precision_phase11b_pokercraft_train/all_records.jsonl \
  --pokercraft-val   data/ocr_precision_phase11b_pokercraft_val/all_records.jsonl \
  --production-train data/ocr_precision_phase11b_production_train/all_records.jsonl \
  --production-val   data/ocr_precision_phase11b_production_val/all_records.jsonl \
  --features data/calibrator/v2_features.txt \
  --out-dir data/calibrator
```

### Task 11.C.2 — Joint test evaluation

**Files:**
- Create: `scripts/_calibrate_v2.py`

Loads the trained v2 calibrator, scores every record in `data/ocr_precision_phase11b_pokercraft/all_records.jsonl` and `data/ocr_precision_phase11b_production/all_records.jsonl`. Concatenates into a joint set. Sweeps τ from 0.0 to 1.0 in 1000 steps. Reports:
- Smallest τ where joint precision ≥ 0.99 AND joint coverage ≥ 0.95.
- Same τ evaluated per-bucket: must satisfy per-bucket precision/coverage too.
- ECE-10bin on each bucket.
- τ-drift: re-run the τ-search on the dev set (train+val records). `|τ_dev − τ_test| / τ_dev` must be ≤ 10%.

If unreachable, the script must print:
- The 10 records in each bucket with the highest `p(correct)` that are still `hand_exact = 0` (false positives at high score).
- The 10 records in each bucket with the lowest `p(correct)` that are still `hand_exact = 1` (false negatives at low score).
- Per-feature SHAP-style importance from the RF model.

These three outputs scope Phase 11.D.

### Task 11.C.3 — Wire v2 calibrator as the default emission gate

**File:** `scripts/ocr/confidence_gate.py`

When `data/calibrator/rf_model_v2.joblib` exists, `CalibratorScorer` loads v2 by default and `_calibrator_features` returns the v2 vector. The `evaluate_with_calibrator` `calibrator_threshold` default becomes the τ from 11.C.2 (the one that satisfies the joint ship gate). Update its docstring with the new selectivity table.

Add a feature flag: `OCR_CALIBRATOR_VERSION` env var ("v1" | "v2"). Defaults to "v2" once the gate flips; "v1" for rollback.

### Task 11.C.4 — Re-baseline both buckets and append to `ocr-99-baselines.md`

```bash
rm -rf data/ocr_precision_current
python scripts/ocr_precision.py \
  --split data/splits/card_classifier_v2.json --bucket test \
  --workers 4 --use-calibrator --calibrator-threshold <tau_from_11C2> \
  --out data/ocr_precision_current

python scripts/ocr_precision.py \
  --split data/splits/production_v1.json --bucket production_test \
  --workers 4 --use-calibrator --calibrator-threshold <tau_from_11C2> \
  --out data/ocr_precision_production_v2
```

Both summaries must show `precision ≥ 0.99` AND `coverage ≥ 0.95` AND `ece_10bin ≤ 0.04`. Append rows to `docs/superpowers/plans/artifacts/ocr-99-baselines.md` for both runs.

### Task 11.C.5 — Snapshot regression coverage

Add to `scripts/regression_test.py`:
- One emit-positive fixture per failure category (position_wrong, board_wrong, hero_cards_wrong, parse_none) where the v2 calibrator must classify correctly.
- One abstain-positive fixture per category (records the calibrator must demote below τ).

These guard against future regressions in the calibrator or its feature inputs.

---

## Phase 11.D — Residual Parser Fixes (only if 11.C cannot reach 99%@95%)

**Why this phase exists:** Some classes of errors may not be separable by the calibrator from the records' confidence/diagnostics — e.g., a parser that reports `card_confidence = 0.95` on a wrong read with no diagnostic flag will look identical to a correct read at the same score. For those records and only those records, the parser must be fixed.

**Entry criteria:** Phase 11.C completes 11.C.2 but the joint ship gate fails on at least one bucket.

**Exit criteria:** Joint ship gate (Phase 11.C exit criteria) holds. **No interim partial-ship.**

### Task 11.D.1 — Categorise the calibrator's residuals

Using 11.C.2's outputs (top-10 false-positives and top-10 false-negatives per bucket), cluster by `failure_mode` (from `compare()` in `ocr_precision.py`). For each cluster:
- (a) Identify the diagnostic feature that would have isolated this record (if any).
- (b) If no such feature exists, identify the parser bug that produced the wrong read.

### Task 11.D.2 — Add the missing diagnostic feature (preferred) OR fix the parser

For each cluster from 11.D.1:
- If (a) yields a feature that's computable from existing parser state: add to `v2_features.txt`, re-train calibrator (return to 11.C.1). Cheap.
- If (b) is the only way: write a regression test for the broken record, fix the parser, verify the test passes, re-run both baselines.

**Hard rule:** No parser fix may regress an existing regression test. If a fix forces an existing test to fail, the fix is wrong; either the test was wrong (rare; needs explicit justification) or the fix is over-broad.

### Task 11.D.3 — Re-run 11.C.2 + 11.C.4 + 11.C.5 until the ship gate holds

This is the iteration loop. Each pass: (parser fix or feature add) → re-train calibrator → re-evaluate joint → check ship gate. **No declaration of victory until joint 99%@95% holds with τ-drift ≤ 10%.**

---

## Phase 11.E — Daily Drift Monitoring

**Why this phase exists:** Production distribution shifts. New N8 themes, holiday sticker variants, screen-size changes. Without monitoring, the calibrated gate silently drifts.

**Entry criteria:** Phase 11.C/D complete; joint ship gate holds.

**Exit criteria:**
1. A daily job computes per-day: production emit rate, OCR-vs-Gemini disagreement on emitted hands, `card_confidence` and `p(correct)` histograms.
2. Alerts when any breaches its rolling baseline by > 2σ.

### Task 11.E.1 — Implement `scripts/ocr_drift_report.py`

Reads the last 24h of `analysis_snapshots`. Runs the OCR pipeline against each. Computes:
- emit_rate (fraction with `p(correct) ≥ τ`)
- gemini_disagree_rate (fraction where OCR's emitted hero_hand != full-Gemini reparse's)
- conf_p50, conf_p10 (drift markers)

Compares against a 7-day rolling baseline stored in `data/drift_baselines/`. Posts to the admin Telegram chat when any metric breaches.

Schedule via PTB JobQueue (matches the existing `weekly_report` pattern) or a CronCreate routine.

---

## Acceptance — Single Non-Negotiable Ship Gate

The plan ships only if **all** of the following hold simultaneously on the held-out test sets:

```
PokerCraft test:
  precision ≥ 99.0%
  recall    ≥ 95.0%
  ECE-10bin ≤ 0.04
  τ-drift   ≤ 10%

production_v1 test (n ≥ 30):
  precision ≥ 99.0%
  recall    ≥ 95.0%
  ECE-10bin ≤ 0.04
  τ-drift   ≤ 10%

pytest tests/ocr -q           — same pass count as main's baseline
python scripts/regression_test.py — green
H3433 snapshot                — green (emit or principled abstain, NOT a wrong emit)
```

No partial-ship is acceptable. No "documented gap" handoff is acceptable. If the gate cannot be passed:
- Return to Phase 11.B (add features), then 11.C (retrain) — first.
- If still failing, Phase 11.D (parser fixes) on the calibrator's residuals.
- If after exhausting 11.D the gate still fails, **the ship is delayed**, the gap is *technical* and is fixed by the next round of feature/parser work. Not by lowering the target.

---

## Self-Review

**Spec coverage:**
- "99% precision @ 95% recall on production distribution" → Phase 11.C exit criteria #1 & #2.
- "No excuses" → Acceptance gate "no partial-ship" + Phase 11.D loop with "no declaration of victory until joint 99%@95% holds".
- "Lessons from Phase 10 attempt" → "Lessons that constrain this plan" + "What survives" sections.

**Why this plan is structurally different from the May 28 attempt:**
- The May 28 plan led with parser-side improvements (overlay aug, ensemble) and treated calibration as a Phase D "tune τ" afterthought. The data showed this ordering was backwards: parser-side changes do not lift confidence calibration enough to reach 99%@95%, but a calibrator trained on rich features does. This plan inverts: calibrator-first, parser-fix only as the residual-cleanup step.
- The May 28 plan allowed an "Acceptable" tier (98.5% / 92%). This plan does not.
- The May 28 plan's H3433 exit criterion was "v3 reads 6d5d at card_conf ≥ 0.70". That target was unreachable — the crop is too small. This plan replaces it with a behavioural criterion: H3433 must either be emitted correctly *or principally abstained*, never emitted-wrong.

**Failure modes acknowledged:**
- If even the v2 calibrator cannot reach 99%@95% on both buckets, the plan does not loosen — it adds features / parser fixes (11.D) until it can. The cost of that iteration is real; the cost is paid by the project, not by the target.
