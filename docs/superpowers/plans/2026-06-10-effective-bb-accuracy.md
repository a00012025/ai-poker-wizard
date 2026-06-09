# Effective_bb Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise hero-active `effective_bb` solver-depth-bucket precision to ≥ 99.5% on emitted values by rewriting `_compute_effective_bb` (min-with-shorter-villain, pot-bounded reconstruction, confidence-gated abstain), backed by an input-level feature cache for seconds-fast iteration.

**Architecture:** Three new modules — `scripts/effbb_metrics.py` (pure metric helpers, TDD'd), `scripts/effbb_cache.py` (one-time 3h parse that captures the *inputs* to `_compute_effective_bb` per corpus hand), `scripts/effbb_eval.py` (runs any candidate function over the cache in seconds, reports coverage/precision/fault-breakdown). The production change is a rewrite of `_compute_effective_bb` in `scripts/ocr/n8_parser.py`, TDD'd against the real cached hands that currently fail.

**Tech Stack:** Python 3.13, OpenCV/CNN OCR pipeline (`scripts/ocr/`), `gto_api.nearest_depth` for buckets, HH-derived ground truth at `data/pokercraft_corpus/ground_truth/ground_truth.jsonl`, pytest-free repo test runner (`scripts/regression_test.py` `@test` decorator + `assert_eq`/`assert_true`).

**Spec:** `docs/superpowers/specs/2026-06-10-effective-bb-accuracy-design.md`

---

## File Structure

- **Create** `scripts/effbb_metrics.py` — pure functions: `depth_bucket`, `bucket_match`, `hero_folded_preflop`, `classify_fault`. No I/O, no OCR. Imported by the eval harness and by tests.
- **Create** `scripts/effbb_cache.py` — CLI that parses the corpus once and writes `data/effbb_cache/cache.jsonl` (inputs to `_compute_effective_bb` + GT + OCR-module hash).
- **Create** `scripts/effbb_eval.py` — CLI that loads the cache, runs `_compute_effective_bb` over it, prints coverage/precision/fault-breakdown + precision-coverage curve.
- **Modify** `scripts/ocr/n8_parser.py` — (a) env-gated capture of `_compute_effective_bb` inputs inside `_assemble_hand`; (b) rewrite `_compute_effective_bb` to return `(effective_bb, hero_starting_stack, confidence)` with min-over-villains + pot-bounded estimators + abstain; (c) update the call site at `:1713`; (d) remove the `displayed×5` gate at `:1760-1779`.
- **Modify** `scripts/regression_test.py` — update the existing `_compute_effective_bb` test (`:382-415`, now a 3-tuple) and add one test per fault class.

---

## Task 1: Metric helpers (`effbb_metrics.py`)

**Files:**
- Create: `scripts/effbb_metrics.py`
- Test: `scripts/regression_test.py` (new `@test` functions)

- [ ] **Step 1: Write the failing tests** in `scripts/regression_test.py` (append near the other OCR tests):

```python
@test("effbb_metrics: depth_bucket snaps to AVAILABLE_DEPTHS")
def test_effbb_depth_bucket():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from effbb_metrics import depth_bucket
    assert_eq(depth_bucket(21.6), 20)
    assert_eq(depth_bucket(24.0), 25)   # |25-24|=1 < |20-24|=4
    assert_eq(depth_bucket(29.3), 30)
    assert_eq(depth_bucket(None), None)
    assert_eq(depth_bucket("x"), None)


@test("effbb_metrics: bucket_match compares snapped depths")
def test_effbb_bucket_match():
    from effbb_metrics import bucket_match
    assert_true(bucket_match(21.6, 19.0))    # both -> 20
    assert_true(not bucket_match(29.3, 36.2)) # 30 vs 35
    assert_true(not bucket_match(None, 20.0))


@test("effbb_metrics: hero_folded_preflop reads position-ordered action string")
def test_effbb_hero_folded():
    from effbb_metrics import hero_folded_preflop
    # 8-max, hero UTG (index 0), preflop UTG folds
    gt = {"num_players": 8, "table_size": 8, "hero_position": "UTG",
          "preflop_actions": "F-F-F-F-F-R2.0-F-C"}
    assert_eq(hero_folded_preflop(gt), True)
    # hero UTG+1 raises
    gt2 = {"num_players": 8, "table_size": 8, "hero_position": "UTG+1",
           "preflop_actions": "F-R2.2-C-F-F-C-F-F"}
    assert_eq(hero_folded_preflop(gt2), False)


@test("effbb_metrics: classify_fault buckets the 4 error classes")
def test_effbb_classify_fault():
    from effbb_metrics import classify_fault
    # overshoot beyond any table stack -> impossible_over
    assert_eq(classify_fault(p_eff=162.9, gt_eff=20.4, hero_start=20.4,
                             gt_max=63.0), "impossible_over")
    # returned hero's own start, a shorter villain existed -> selection
    assert_eq(classify_fault(p_eff=36.2, gt_eff=29.3, hero_start=36.2,
                             gt_max=69.4), "selection")
    # under-compute
    assert_eq(classify_fault(p_eff=7.4, gt_eff=24.1, hero_start=24.1,
                             gt_max=80.0), "undershoot")
    # adjacent-bucket near miss
    assert_eq(classify_fault(p_eff=40.0, gt_eff=37.1, hero_start=45.0,
                             gt_max=78.0), "near")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python scripts/regression_test.py 2>&1 | grep -i effbb`
Expected: FAIL — `No module named 'effbb_metrics'`.

- [ ] **Step 3: Write `scripts/effbb_metrics.py`**

```python
"""Pure metric helpers for effective_bb evaluation. No OCR, no I/O."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gto_api import nearest_depth

try:
    from analyze_hand import POSITION_ORDERS
except Exception:  # pragma: no cover - analyze_hand import is heavy
    POSITION_ORDERS = {}


def depth_bucket(bb):
    """Snap a bb value to its solver depth bucket (int). None on bad input."""
    try:
        return int(round(nearest_depth(float(bb))))
    except (TypeError, ValueError):
        return None


def bucket_match(a, b) -> bool:
    """True iff a and b snap to the same solver depth bucket."""
    ba, bb = depth_bucket(a), depth_bucket(b)
    return ba is not None and ba == bb


def hero_folded_preflop(gt: dict):
    """True/False if hero's preflop code is F, else None (unknown order)."""
    pa = (gt.get("preflop_actions") or "").split("-")
    hp = gt.get("hero_position")
    order = POSITION_ORDERS.get(gt.get("num_players")) or \
        POSITION_ORDERS.get(gt.get("table_size"))
    if not order or hp not in order:
        return None
    idx = order.index(hp)
    if idx < len(pa):
        return pa[idx] == "F"
    return None


def classify_fault(*, p_eff, gt_eff, hero_start, gt_max) -> str:
    """Bucket an emitted-but-wrong hand into one of 4 fault classes."""
    ratio = (p_eff / gt_eff) if gt_eff else 0.0
    if ratio >= 1.4 and gt_max and p_eff > gt_max * 1.1:
        return "impossible_over"
    if ratio >= 1.4:
        return "selection"      # over-large but within table stacks
    if ratio <= 0.71:
        return "undershoot"
    return "near"               # adjacent bucket, < 1.4x off
```

Note `nearest_depth` returns `depth.125` floats (e.g. `20.125`); `int(round(...))` yields the bucket integer `20`.

- [ ] **Step 4: Run to verify they pass**

Run: `python scripts/regression_test.py 2>&1 | grep -i effbb`
Expected: 4 effbb_metrics tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/ai-poker-wizard-effbb-accuracy
git add scripts/effbb_metrics.py scripts/regression_test.py
git commit -m "feat(effbb): pure metric helpers (bucket match, hero-folded, fault class)"
```

---

## Task 2: Env-gated input capture in `_assemble_hand`

Capture the exact inputs `_compute_effective_bb` receives, so the cache can replay them. Guarded by an env var → zero production impact when unset.

**Files:**
- Modify: `scripts/ocr/n8_parser.py:1713` (call site region)

- [ ] **Step 1: Add the capture block** immediately AFTER the `_compute_effective_bb` call at `scripts/ocr/n8_parser.py:1713-1715`. Insert after the closing `)` of that call:

```python
    effective_bb, hero_starting_stack = _compute_effective_bb(
        columns, hero_stack, hero_position, stacks, named_stacks,
    )

    # Phase-0 effbb cache: stash the raw inputs so effbb_eval can replay
    # _compute_effective_bb without re-OCR. Gated by env var; no prod cost.
    if os.getenv("EFFBB_CAPTURE"):
        try:
            hand_capture = {
                "columns": columns,
                "hero_stack": hero_stack,
                "hero_position": hero_position,
                "stacks": stacks,
                "named_stacks": named_stacks,
            }
        except Exception:
            hand_capture = None
    else:
        hand_capture = None
```

(`os` is already imported at the top of `n8_parser.py`.)

- [ ] **Step 2: Attach the capture to the returned hand.** Find where `_assemble_hand` sets `hand["player_stacks"] = stacks` (`scripts/ocr/n8_parser.py:1839`) and add right after it:

```python
        hand["player_stacks"] = stacks
        if hand_capture is not None:
            hand["__effbb_inputs__"] = hand_capture
```

- [ ] **Step 3: Smoke-test the capture** on one image:

```bash
cd ~/ai-poker-wizard-effbb-accuracy
EFFBB_CAPTURE=1 python -c "
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/ocr')
from ocr.n8_parser import parse_n8_screenshot
from pathlib import Path
img=sorted(Path('data/hand_images/img').glob('*.png'))[0]
h=parse_n8_screenshot(img.read_bytes())['hand']
print('has inputs:', bool(h and '__effbb_inputs__' in h))
print('keys:', list(h['__effbb_inputs__'].keys()) if h and '__effbb_inputs__' in h else None)
"
```

Expected: `has inputs: True` and the 5 keys. (If `h` is None on this image, try index `1`/`2` — pick one that parses.)

- [ ] **Step 4: Verify no capture when env unset:**

```bash
python -c "
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'scripts/ocr')
from ocr.n8_parser import parse_n8_screenshot
from pathlib import Path
img=sorted(Path('data/hand_images/img').glob('*.png'))[0]
h=parse_n8_screenshot(img.read_bytes())['hand']
print('no capture:', not (h and '__effbb_inputs__' in h))
"
```

Expected: `no capture: True`.

- [ ] **Step 5: Commit**

```bash
git add scripts/ocr/n8_parser.py
git commit -m "feat(effbb): env-gated capture of _compute_effective_bb inputs for cache"
```

---

## Task 3: Cache builder (`effbb_cache.py`)

**Files:**
- Create: `scripts/effbb_cache.py`
- Output: `data/effbb_cache/cache.jsonl`

- [ ] **Step 1: Write `scripts/effbb_cache.py`**

```python
#!/usr/bin/env python3
"""Build the effective_bb input cache: one full parse over the corpus,
capturing the inputs to _compute_effective_bb + HH ground truth + a hash of
the OCR modules that produce those inputs (for staleness detection).

Usage:
  EFFBB_CAPTURE=1 python scripts/effbb_cache.py \
      --images data/hand_images/img \
      --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
      --out data/effbb_cache/cache.jsonl [--limit N]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ocr"))

from ocr.n8_parser import parse_n8_screenshot

# Modules whose changes invalidate the cache (see spec staleness contract).
_OCR_MODULE_FILES = [
    "ocr/panel_parser.py",
    "ocr/table_parser.py",
    "ocr/n8_parser.py",
]


def ocr_modules_hash() -> str:
    h = hashlib.sha256()
    base = Path(__file__).resolve().parent
    for rel in _OCR_MODULE_FILES:
        h.update((base / rel).read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not os.getenv("EFFBB_CAPTURE"):
        sys.exit("Set EFFBB_CAPTURE=1 so _assemble_hand stashes __effbb_inputs__.")

    gt = {}
    with open(args.ground_truth, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            gt[o["hand_id"]] = o["ground_truth"]

    imgs = sorted(Path(args.images).glob("*.png"))
    pairs = [(p, gt[p.stem]) for p in imgs if p.stem in gt]
    if args.limit:
        pairs = pairs[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mod_hash = ocr_modules_hash()
    n_ok = n_inputs = 0
    with out.open("w", encoding="utf-8") as wf:
        for i, (path, g) in enumerate(pairs):
            rec = {"hand_id": path.stem, "ocr_hash": mod_hash,
                   "gt": {k: g.get(k) for k in
                          ("effective_bb", "stacks_bb", "preflop_actions",
                           "num_players", "table_size", "hero_position")}}
            try:
                res = parse_n8_screenshot(path.read_bytes())
                hand = res.get("hand")
                rec["confidence"] = round(res.get("confidence", 0.0), 3)
                rec["hand_none"] = hand is None
                if hand and "__effbb_inputs__" in hand:
                    rec["inputs"] = hand["__effbb_inputs__"]
                    n_inputs += 1
                n_ok += 1
            except Exception as e:
                rec["err"] = str(e)[:120]
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(pairs)} (inputs={n_inputs})", flush=True)
    print(f"[effbb_cache] wrote {len(pairs)} rows -> {out} "
          f"(parsed={n_ok}, with_inputs={n_inputs}, ocr_hash={mod_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Validate on a 20-hand slice** (fast, proves the format):

```bash
cd ~/ai-poker-wizard-effbb-accuracy
EFFBB_CAPTURE=1 python scripts/effbb_cache.py \
  --images data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --out data/effbb_cache/cache_smoke.jsonl --limit 20
```

Expected: prints `wrote 20 rows ... with_inputs=` a number > 0; file exists.

- [ ] **Step 3: Eyeball one cached row has the inputs + gt:**

```bash
python -c "
import json
r=[json.loads(l) for l in open('data/effbb_cache/cache_smoke.jsonl')]
withi=[x for x in r if 'inputs' in x][0]
print('keys:', list(withi.keys()))
print('input keys:', list(withi['inputs'].keys()))
print('gt:', withi['gt'])
"
```

Expected: `inputs` has `columns/hero_stack/hero_position/stacks/named_stacks`; `gt` has `effective_bb/stacks_bb/...`.

- [ ] **Step 4: Commit the script** (not the cache data — it is large/regenerable; add to `.gitignore`):

```bash
cd ~/ai-poker-wizard-effbb-accuracy
echo "data/effbb_cache/" >> .gitignore
git add scripts/effbb_cache.py .gitignore
git commit -m "feat(effbb): cache builder — capture compute inputs + GT + ocr hash"
```

---

## Task 4: Eval harness (`effbb_eval.py`)

**Files:**
- Create: `scripts/effbb_eval.py`

- [ ] **Step 1: Write `scripts/effbb_eval.py`**

```python
#!/usr/bin/env python3
"""Evaluate _compute_effective_bb over the input cache in seconds.

Replays cached inputs through the CURRENT _compute_effective_bb, scores at the
solver depth bucket level vs HH ground truth, splits hero-active vs hero-folded,
and prints the fault breakdown + a precision/coverage curve over confidence.

Usage: python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ocr"))

from effbb_metrics import bucket_match, hero_folded_preflop, classify_fault, depth_bucket
from ocr.n8_parser import _compute_effective_bb


def recompute(inp):
    """Replay one cached input tuple through _compute_effective_bb.
    Tolerates both the 2-tuple (legacy) and 3-tuple (rewritten) returns."""
    res = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"],
    )
    if isinstance(res, tuple) and len(res) == 3:
        return res                      # (eff, hero_start, confidence)
    eff, hero_start = res
    return eff, hero_start, 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/effbb_cache/cache.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="emit only when confidence >= this (precision/coverage knob)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]
    active, folded = [], []
    for r in rows:
        gt = r.get("gt") or {}
        ge = gt.get("effective_bb")
        if ge is None or ge < 1.0 or "inputs" not in r:
            continue
        p_eff, hero_start, conf = recompute(r["inputs"])
        if conf < args.min_conf:
            p_eff = None
        rec = {"hid": r["hand_id"], "gt_eff": ge, "p_eff": p_eff,
               "hero_start": hero_start, "conf": conf,
               "gt_max": max(gt.get("stacks_bb") or [0]) or None}
        hf = hero_folded_preflop(gt)
        (folded if hf else active).append(rec) if hf is not None else None

    def score(name, subset):
        emitted = [x for x in subset if x["p_eff"] is not None]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(subset) if subset else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"\n## {name}: n={len(subset)} emitted={len(emitted)} "
              f"coverage={cov:.1f}% bucket-precision={prec:.2f}% "
              f"({len(ok)}/{len(emitted)})")
        wrong = [x for x in emitted if x not in ok]
        faults = {}
        for x in wrong:
            f = classify_fault(p_eff=x["p_eff"], gt_eff=x["gt_eff"],
                               hero_start=x["hero_start"] or x["gt_eff"],
                               gt_max=x["gt_max"])
            faults[f] = faults.get(f, 0) + 1
        if wrong:
            print("   faults:", faults)
        return prec, cov

    score("HERO ACTIVE (target population)", active)
    score("HERO FOLDED (context only)", folded)

    print("\n--- precision/coverage curve (hero-active, by confidence floor) ---")
    for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        emitted = [x for x in active if x["p_eff"] is not None and x["conf"] >= thr]
        ok = [x for x in emitted if bucket_match(x["p_eff"], x["gt_eff"])]
        cov = 100 * len(emitted) / len(active) if active else 0
        prec = 100 * len(ok) / len(emitted) if emitted else 0
        print(f"  conf>={thr:.1f}: coverage={cov:5.1f}%  precision={prec:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run against the 20-hand smoke cache** (proves it runs end-to-end before the 3h build):

```bash
cd ~/ai-poker-wizard-effbb-accuracy
python scripts/effbb_eval.py --cache data/effbb_cache/cache_smoke.jsonl
```

Expected: prints HERO ACTIVE / HERO FOLDED blocks and a curve (numbers noisy on 20 hands — we only check it runs without error).

- [ ] **Step 3: Commit**

```bash
git add scripts/effbb_eval.py
git commit -m "feat(effbb): eval harness — bucket precision/coverage + fault breakdown + PR curve"
```

---

## Task 5: Build the full cache + record the baseline (operational, ~3h)

**Files:** none (produces `data/effbb_cache/cache.jsonl` + a baseline note)

- [ ] **Step 1: Kick off the full parse in the background:**

```bash
cd ~/ai-poker-wizard-effbb-accuracy
EFFBB_CAPTURE=1 nohup python scripts/effbb_cache.py \
  --images data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --out data/effbb_cache/cache.jsonl > /tmp/effbb_cache_build.log 2>&1 &
echo "pid $!"
```

- [ ] **Step 2: Wait for completion** (monitor `tail -f /tmp/effbb_cache_build.log`). Done when the log prints `[effbb_cache] wrote 7183 rows`.

- [ ] **Step 3: Record the CURRENT-logic baseline:**

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl | tee docs/superpowers/plans/effbb-baseline-current.txt
```

Expected: HERO ACTIVE block with the real coverage/precision (the rough sample said coverage ≈75%, precision ≈52%) and the fault breakdown. **This is the number Task 8 must beat.**

- [ ] **Step 4: Commit the baseline note** (text only; cache stays gitignored):

```bash
git add docs/superpowers/plans/effbb-baseline-current.txt
git commit -m "docs(effbb): record current-logic hero-active baseline over full cache"
```

---

## Task 6: Rewrite `_compute_effective_bb` — signature + min-over-villains (the selection fix)

The single biggest fault class (21/39). Change the return to a 3-tuple and take `min` over **all** active villains. Golden cases come straight from the cache.

**Files:**
- Modify: `scripts/ocr/n8_parser.py` (`_compute_effective_bb` def + call site `:1713` + gate `:1760-1779`)
- Modify: `scripts/regression_test.py:382-415` (existing test now 3-tuple) + new tests

- [ ] **Step 1: Write the failing golden test.** Append to `scripts/regression_test.py`. It loads two real cached hands and asserts the right bucket. (Requires Task 5's cache.)

```python
@test("effbb: multiway returns min(hero, shortest active villain) bucket")
def test_effbb_multiway_selection():
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..",
                         "data/effbb_cache/cache.jsonl")
    by_id = {}
    for line in open(cache, encoding="utf-8"):
        o = json.loads(line)
        if "inputs" in o:
            by_id[o["hand_id"]] = o
    # TM5873208532: hero 36.2 vs callers 29.3/30.4 -> effective 29.3 -> bucket 30
    o = by_id["TM5873208532"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_eq(depth_bucket(eff), 30)
```

- [ ] **Step 2: Run to verify it fails** (current logic returns 36.2 → bucket 35):

Run: `python scripts/regression_test.py 2>&1 | grep -i "multiway returns min"`
Expected: FAIL — got bucket 35, want 30 (or a tuple-unpack error if the 3-tuple isn't in yet).

- [ ] **Step 3: Change the return signature to a 3-tuple.** In `scripts/ocr/n8_parser.py`, update `_compute_effective_bb`:
  - The early return `if hero_stack_displayed is None: return None, None` → `return None, None, 0.0`.
  - The final `return effective_bb, round(hero_starting, 1)` (`:1299`) → `return effective_bb, round(hero_starting, 1), 1.0` (confidence filled in Task 8; 1.0 placeholder now).

- [ ] **Step 4: Implement min-over-villains.** Locate the postflop opponent-selection logic (the block deriving `opp_preflop_total` / continuing opponent around `:1030-1130` and the final `effective_bb = round(min(all_starting), 1)` at `:1292`). Ensure `all_starting` includes **every active (non-folded) villain's** reconstructed start, not just the heads-up continuing opponent. Concretely, build the villain start set from all opponents who are non-folded at showdown/last street and take `min(hero_starting, *villain_starts)`. Keep `hero_starting` as computed.

(The exact edit depends on the current internal structure; the test in Step 1 plus the eval harness in Step 6 are the acceptance signal. Implement the minimal change that makes `min` span all active villains.)

- [ ] **Step 5: Update the production call site** `scripts/ocr/n8_parser.py:1713`:

```python
    effective_bb, hero_starting_stack, _effbb_conf = _compute_effective_bb(
        columns, hero_stack, hero_position, stacks, named_stacks,
    )
```

- [ ] **Step 6: Update the existing regression test** at `scripts/regression_test.py:415`. Change:

```python
    eff, hero_start = _compute_effective_bb(
```
to:
```python
    eff, hero_start, _conf = _compute_effective_bb(
```

- [ ] **Step 7: Run both effbb tests:**

Run: `python scripts/regression_test.py 2>&1 | grep -iE "effbb|effective"`
Expected: the existing `_compute_effective_bb` test still PASSes (3-tuple) and `multiway returns min` PASSes.

- [ ] **Step 8: Re-score on the cache** (no re-parse needed):

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
```

Expected: HERO ACTIVE precision **up** vs Task 5 baseline; `selection` fault count **down**.

- [ ] **Step 9: Commit**

```bash
git add scripts/ocr/n8_parser.py scripts/regression_test.py
git commit -m "feat(effbb): 3-tuple return + min over all active villains (selection fix)"
```

---

## Task 7: Pot-bounded over-compute guard

Kill the "start > displayed + pot" inflations (49.0 when hero had 29.0; 162.9 from disp 3.0).

**Files:**
- Modify: `scripts/ocr/n8_parser.py` (`_compute_effective_bb` reconstruction)
- Modify: `scripts/regression_test.py` (new test)

- [ ] **Step 1: Write the failing test.** Pick a cached `impossible_over` hand from Task 5's baseline fault list (replace `HID_OVER` with an actual id printed by `effbb_eval`, e.g. one where `p_eff > gt_max*1.1`). Assert the rewritten function no longer emits a value above the table max (it should either correct or abstain → `None`).

```python
@test("effbb: over-compute past table max is rejected (bounded or abstain)")
def test_effbb_overcompute_bounded():
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(os.path.dirname(__file__), "..",
                         "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8")}
    o = rows["HID_OVER"]; inp = o["inputs"]
    gt_max = max(o["gt"]["stacks_bb"])
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is None or eff <= gt_max * 1.1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python scripts/regression_test.py 2>&1 | grep -i "over-compute"`
Expected: FAIL (current logic emits the inflated value).

- [ ] **Step 3: Add the pot bound.** In `_compute_effective_bb`, after each player's `start = displayed + investment` is formed, clamp/reject: compute the table pot total available (`max` street `pot` header, or sum of contributions) and treat any `start > displayed + pot_total` as invalid → drop that estimate (do not let it enter `all_starting`). If the *effective* player's estimate is dropped and no consistent alternative remains, mark for abstain (Task 8 turns this into low confidence).

- [ ] **Step 4: Run to verify it passes**

Run: `python scripts/regression_test.py 2>&1 | grep -i "over-compute"`
Expected: PASS.

- [ ] **Step 5: Re-score:**

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
```

Expected: `impossible_over` fault count → ~0; precision up.

- [ ] **Step 6: Commit**

```bash
git add scripts/ocr/n8_parser.py scripts/regression_test.py
git commit -m "feat(effbb): reject reconstructions exceeding displayed+pot (over-compute guard)"
```

---

## Task 8: Confidence-gated abstain — second estimator + agreement; remove the old gate

Add the pot-header estimator, set `confidence` from agreement, and delete the `displayed×5` gate.

**Files:**
- Modify: `scripts/ocr/n8_parser.py` (`_compute_effective_bb` + remove gate `:1760-1779`)
- Modify: `scripts/regression_test.py` (abstain test + H3522-style deep-invested test)

- [ ] **Step 1: Write two failing tests.**

```python
@test("effbb: deep-invested hero keeps a real value (no displayed*5 false-null)")
def test_effbb_deep_invested_not_nulled():
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    from effbb_metrics import depth_bucket
    cache = os.path.join(os.path.dirname(__file__), "..",
                         "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8")}
    # H3522 analogue: hero starts ~29.4bb, displayed ~5bb after heavy action.
    # Use the real cached hand id once identified in the corpus (TM-id).
    o = rows["HID_DEEP"]; inp = o["inputs"]
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(eff is not None)
    assert_eq(depth_bucket(eff), depth_bucket(o["gt"]["effective_bb"]))


@test("effbb: diverging estimators abstain (None) rather than guess")
def test_effbb_abstain_on_divergence():
    import json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ocr"))
    from ocr.n8_parser import _compute_effective_bb
    cache = os.path.join(os.path.dirname(__file__), "..",
                         "data/effbb_cache/cache.jsonl")
    rows = {json.loads(l)["hand_id"]: json.loads(l)
            for l in open(cache, encoding="utf-8")}
    o = rows["HID_DIVERGE"]; inp = o["inputs"]   # an undershoot/ambiguous hand
    eff, hero_start, conf = _compute_effective_bb(
        inp["columns"], inp["hero_stack"], inp["hero_position"],
        inp["stacks"], inp["named_stacks"])
    assert_true(conf < 0.7)
```

(Identify `HID_DEEP` and `HID_DIVERGE` from the baseline fault list: a deep-invested hero whose old output was `None`, and an `undershoot` hand. Replace the placeholders with the real TM ids.)

- [ ] **Step 2: Run to verify they fail.**

Run: `python scripts/regression_test.py 2>&1 | grep -iE "deep-invested|abstain"`
Expected: FAIL.

- [ ] **Step 3: Add the pot-header estimator.** In `_compute_effective_bb`, compute a second independent `effective` estimate for the deciding player using pot-header deltas (the function already collects `pot_by_street`/`pot_sequence`). Name it `eff_pothdr` alongside the action-walk `eff_actionwalk`.

- [ ] **Step 4: Set confidence from agreement.** Replace the trailing `return effective_bb, round(hero_starting,1), 1.0` with:

```python
    from effbb_metrics import depth_bucket as _bucket   # local import, no cycle
    if eff_pothdr is not None and eff_actionwalk is not None:
        confidence = 1.0 if _bucket(eff_pothdr) == _bucket(eff_actionwalk) else 0.3
    else:
        confidence = 0.6   # single estimator only
    if confidence < 0.7:
        return None, round(hero_starting, 1), confidence
    return effective_bb, round(hero_starting, 1), confidence
```

(Adjust the `0.7` floor in Task 9 against the precision/coverage curve.)

- [ ] **Step 5: Remove the old gate.** Delete the sanity-gate block at `scripts/ocr/n8_parser.py:1760-1779` (the `if effective_bb < max_preflop_raise` / `elif ... > hero_stack*5` nulling). Abstain now lives inside `_compute_effective_bb`. Verify `effective_bb is None` is still handled downstream (`analyze_hand.py:1144-1156` fallback — unchanged).

- [ ] **Step 6: Run the new tests + full effbb suite:**

Run: `python scripts/regression_test.py 2>&1 | grep -iE "effbb|effective"`
Expected: all effbb tests PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/ocr/n8_parser.py scripts/regression_test.py
git commit -m "feat(effbb): dual-estimator confidence + abstain; remove displayed*5 gate"
```

---

## Task 9: Tune the confidence floor to ≥ 99.5% precision (operational)

**Files:** `scripts/ocr/n8_parser.py` (the floor constant)

- [ ] **Step 1: Read the precision/coverage curve:**

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
```

Look at the `precision/coverage curve (hero-active)` block.

- [ ] **Step 2: Pick the lowest confidence floor whose hero-active precision ≥ 99.5%.** Set that as the abstain threshold (the `0.7` in Task 8 Step 4, and make it a module constant `_EFFBB_CONF_FLOOR = float(os.getenv("OCR_EFFBB_CONF_FLOOR", "<chosen>"))`).

- [ ] **Step 3: Re-score and confirm:**

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl
```

Expected: HERO ACTIVE `bucket-precision ≥ 99.50%`; note the coverage. **If 99.5% is unreachable at any usable coverage**, stop and report the curve to the user (per spec §6 risk) before forcing the threshold.

- [ ] **Step 4: Commit**

```bash
git add scripts/ocr/n8_parser.py
git commit -m "feat(effbb): set confidence floor to the 99.5%-precision operating point"
```

---

## Task 10: Full validation re-parse + snapshot relocks

The cache froze OCR inputs at Task 5; one real parse confirms the end-to-end numbers and the gate removal didn't shift snapshots.

**Files:** snapshot DB (via `snapshot_test.py`), no code

- [ ] **Step 1: Run the standard regression suite** (must be green before re-parse):

```bash
cd ~/ai-poker-wizard-effbb-accuracy
python scripts/regression_test.py
```

Expected: all pass (the effbb tests + existing suite).

- [ ] **Step 2: Run the snapshot suite:**

```bash
python scripts/snapshot_test.py
```

Expected: pass, OR a small set of hands whose `effective_bb`/depth legitimately shifted (e.g. a deep-invested hand now emitting a value).

- [ ] **Step 3: For each legitimately-shifted snapshot, verify against the real screenshot, then re-lock:**

```bash
python scripts/snapshot_test.py --update H####
python scripts/snapshot_test.py --add H####
python scripts/snapshot_test.py H####
```

(Only re-lock hands you confirmed correct against the image — per `validation-backlog-mostly-stale`: verify real data first.)

- [ ] **Step 4: Re-run the full cache build once** (the production code changed → cache is stale by the staleness contract; this is the Phase-2 confirmation parse):

```bash
EFFBB_CAPTURE=1 nohup python scripts/effbb_cache.py \
  --images data/hand_images/img \
  --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \
  --out data/effbb_cache/cache.jsonl > /tmp/effbb_cache_final.log 2>&1 &
```

Wait for `wrote 7183 rows`, then:

```bash
python scripts/effbb_eval.py --cache data/effbb_cache/cache.jsonl | tee docs/superpowers/plans/effbb-final-result.txt
```

Expected: hero-active precision ≥ 99.5% confirmed on a fresh end-to-end parse (not just the replay).

- [ ] **Step 5: Commit the final result note**

```bash
git add docs/superpowers/plans/effbb-final-result.txt
git commit -m "docs(effbb): final end-to-end hero-active precision/coverage result"
```

---

## Task 11: Validator soft-signal + PR

Ensure `effective_bb=None` is a soft signal (no hard warning) and open the PR.

**Files:**
- Inspect: `scripts/hand_validator.py` (EFFECTIVE_BB check), `scripts/ocr/n8_parser.py:2700`

- [ ] **Step 1: Confirm abstain does not raise a hard validator flag.** Check the `EFFECTIVE_BB` path in `scripts/hand_validator.py` and the `effective_missing` handling at `scripts/ocr/n8_parser.py:2700`. If `effective_bb is None` currently produces a blocking/warning flag, downgrade it to an informational note (abstain is expected and handled downstream). Add/adjust a unit test asserting `None` does not produce a blocking validation issue.

- [ ] **Step 2: Run the suite once more:**

```bash
python scripts/regression_test.py
```

Expected: all pass.

- [ ] **Step 3: Push + open PR:**

```bash
cd ~/ai-poker-wizard-effbb-accuracy
git push -u origin feat/effective-bb-accuracy
gh pr create --title "feat(effbb): hero-active depth-bucket precision >=99.5% (logic rewrite + abstain)" \
  --body "Rewrites _compute_effective_bb: min over all active villains, pot-bounded reconstruction, dual-estimator confidence + abstain (replaces displayed*5 gate). Adds effbb_metrics/effbb_cache/effbb_eval. Baseline vs final in docs/superpowers/plans/. Closes the H3522 false-null. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: After merge, clean up the worktree:**

```bash
cd ~/ai-poker-wizard
git worktree remove ~/ai-poker-wizard-effbb-accuracy
```

---

## Self-Review notes

- **Spec coverage:** Component 1 → Tasks 6–9; Component 2 → Tasks 2,3,5; Component 3 → Task 4; success metric (hero-active bucket precision) → Tasks 4,5,9,10; abstain-as-soft-signal → Tasks 8,11; testing → Tasks 1,6,7,8,10,11; staleness contract → Task 3 (`ocr_modules_hash`). All spec sections mapped.
- **Known-late-bound ids:** `HID_OVER` / `HID_DEEP` / `HID_DIVERGE` are intentionally resolved from the Task-5 baseline fault dump (the real corpus ids aren't known until the cache exists). Each placeholder is called out in-step with how to obtain it. The golden multiway ids (`TM5873208532`) are real and verified.
- **Type consistency:** `_compute_effective_bb` returns the 3-tuple `(effective_bb, hero_starting_stack, confidence)` from Task 6 onward; every caller (prod `:1713`, `effbb_eval.recompute`, all tests) unpacks 3. `depth_bucket` returns int|None throughout.
