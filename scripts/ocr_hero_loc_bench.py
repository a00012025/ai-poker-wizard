#!/usr/bin/env python3
"""Fast hero-card LOCALIZATION benchmark — single-process, deterministic, local.

Isolates `_locate_hero_cards` + CardCNN (skips the slow panel EasyOCR and the
Gemini fallback), so it iterates in minutes where the full ocr_benchmark.py
(production parse incl. Gemini) takes far longer. Use it to tune hero detection
and prove a change is net-positive against authoritative pokercraft labels
*before* running the full no-regression gate (see the verify-ocr-no-regression
skill, which remains the hard commit gate).

It replicates production's confidence-gated two-stage localization (bright blob,
then a whiteness retry when hero card_conf < OCR_HERO_RELOCATE_CONF), using a
raw CardCNN read (no corner-OCR) — a fair proxy for localization quality, since
corner-OCR helps old and new equally.

Modes
-----
  # Eval the live localizer on N pokercraft hands, save per-hand results:
  python scripts/ocr_hero_loc_bench.py run <label> [N]

  # Diff two saved runs (run the baseline on `main`, your change on the branch):
  python scripts/ocr_hero_loc_bench.py cmp <labelA> <labelB>

  # Label-free speed metric on real Telegram traffic (DB blocking-Gemini rate):
  python scripts/ocr_hero_loc_bench.py live [N]

Decision rule: a change ships only with REGRESS == 0 and FIXED > 0 in `cmp`.

CUDA note: this is single-process on purpose. ocr_precision.py with
`--workers N` on GPU dies with CUBLAS_NOT_INITIALIZED (N CUDA contexts on one
device); run that one CPU-only (CUDA_VISIBLE_DEVICES="").
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL = Path("data/_ocr_loc_eval")
GT_PATH = "data/pokercraft_corpus/ground_truth/ground_truth.jsonl"
IMG_DIR = Path("data/hand_images/img")
RELOCATE_CONF = float(os.getenv("OCR_HERO_RELOCATE_CONF", "0.70"))
_RANK = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT = {s: i for i, s in enumerate("cdhs")}


def _canon(s: str) -> tuple:
    s = (s or "").replace(" ", "")
    cards = [s[i:i + 2] for i in range(0, len(s) - 1, 2)]
    return tuple(sorted(cards, key=lambda c: (_RANK.get(c[0], -1), _SUIT.get(c[1], -1))))


def _load_gt() -> dict:
    gt = {}
    for line in open(GT_PATH):
        d = json.loads(line)
        g = d.get("ground_truth", d)
        if g.get("hero_hand"):
            gt[d["hand_id"]] = g["hero_hand"]
    return gt


def _gated_hero(img, regions, TP, clf):
    """Production conf-gated 3-stage hero read (raw CNN). -> (cards, conf).

    Stage 1 bright (table), stage 2 whiteness (table), stage 3 whiteness on a
    divider-spanning band of the full image. Mirrors _find_hero_cards minus the
    corner-OCR rescue (which helps all stages equally), so it's a fair, fast
    proxy for localization quality.
    """
    table = regions.get("table")
    divider_y = regions.get("divider_y")

    def classify(crops):
        if not crops:
            return "", 0.0
        det = clf.classify_batch_detailed_tta([TP._trim_above_card_edge(c) for c in crops])
        return ("".join(f"{d['rank']}{d['suit']}" for d in det),
                min(min(d["rank_conf"], d["suit_conf"]) for d in det))

    cards, conf = classify(TP._locate_hero_bright(table))
    for crops in (
        TP._locate_hero_white(table) if conf < RELOCATE_CONF else None,
        TP._locate_hero_white(img, divider_y=divider_y)
        if (conf < RELOCATE_CONF and divider_y) else None,
    ):
        if conf >= RELOCATE_CONF or not crops:
            continue
        c2, cf2 = classify(crops)
        if cf2 > conf:
            cards, conf = c2, cf2
    return cards, conf


def run(label: str, n: int) -> None:
    from ocr.region_detector import detect_regions
    from ocr import table_parser as TP
    from ocr.classifier.infer import CardClassifier
    EVAL.mkdir(parents=True, exist_ok=True)
    gt = _load_gt()
    clf = CardClassifier()
    paired = sorted(p for p in IMG_DIR.glob("*.png") if p.stem in gt)
    if n and len(paired) > n:
        paired = paired[::max(1, len(paired) // n)][:n]
    out = open(EVAL / f"{label}.jsonl", "w")
    correct = correct_emit = low = nocrop = 0
    for i, p in enumerate(paired):
        img = cv2.imread(str(p))
        regions = detect_regions(img) if img is not None else None
        cards, conf = _gated_hero(img, regions, TP, clf) if regions else ("", 0.0)
        ok = bool(cards) and _canon(cards) == _canon(gt[p.stem])
        out.write(json.dumps({"hand_id": p.stem, "gt": gt[p.stem],
                              "cards": cards, "conf": round(conf, 3), "ok": ok}) + "\n")
        if not cards:
            nocrop += 1
        if ok:
            correct += 1
            if conf >= RELOCATE_CONF:
                correct_emit += 1
        if conf < RELOCATE_CONF:
            low += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paired)} ...", flush=True)
    out.close()
    tot = len(paired)
    print(f"\n[{label}] n={tot}")
    print(f"  hero_hand correct:                 {correct} ({100*correct/tot:.1f}%)")
    print(f"  correct & conf>={RELOCATE_CONF} (local emit): {correct_emit} ({100*correct_emit/tot:.1f}%)")
    print(f"  conf<{RELOCATE_CONF} (blocking Gemini):       {low} ({100*low/tot:.1f}%)")
    print(f"  no crop located:                   {nocrop} ({100*nocrop/tot:.1f}%)")


def runf(label: str, n: int) -> None:
    """Like run(), but uses the REAL production hero path _find_hero_cards
    (corner-OCR rank rescue + top-2 repairs + ensemble + all 3 localizer
    stages). Slower (EasyOCR corner read per card) but the true production-local
    hero accuracy — still no Gemini fallback."""
    from ocr.region_detector import detect_regions
    from ocr import table_parser as TP
    EVAL.mkdir(parents=True, exist_ok=True)
    gt = _load_gt()
    paired = sorted(p for p in IMG_DIR.glob("*.png") if p.stem in gt)
    if n and len(paired) > n:
        paired = paired[::max(1, len(paired) // n)][:n]
    out = open(EVAL / f"{label}.jsonl", "w")
    correct = correct_emit = low = 0
    for i, p in enumerate(paired):
        img = cv2.imread(str(p))
        regions = detect_regions(img) if img is not None else None
        if not regions:
            cards, conf = "", 0.0
        else:
            cards_list, conf, _details, _ens = TP._find_hero_cards(
                regions["table"], full_image=img, divider_y=regions.get("divider_y"))
            cards = "".join(cards_list)
        ok = bool(cards) and _canon(cards) == _canon(gt[p.stem])
        out.write(json.dumps({"hand_id": p.stem, "gt": gt[p.stem],
                              "cards": cards, "conf": round(conf, 3), "ok": ok}) + "\n")
        if ok:
            correct += 1
            if conf >= RELOCATE_CONF:
                correct_emit += 1
        if conf < RELOCATE_CONF:
            low += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paired)} ...", flush=True)
    out.close()
    tot = len(paired)
    print(f"\n[{label}] n={tot}  (full production hero path, corner-OCR on)")
    print(f"  hero_hand correct:                 {correct} ({100*correct/tot:.2f}%)")
    print(f"  correct & conf>={RELOCATE_CONF} (local emit): {correct_emit} ({100*correct_emit/tot:.2f}%)")
    print(f"  conf<{RELOCATE_CONF} (blocking Gemini):       {low} ({100*low/tot:.2f}%)")


def cmp(a: str, b: str) -> None:
    ra = {r["hand_id"]: r for r in (json.loads(x) for x in open(EVAL / f"{a}.jsonl"))}
    rb = {r["hand_id"]: r for r in (json.loads(x) for x in open(EVAL / f"{b}.jsonl"))}
    keys = sorted(set(ra) & set(rb))
    fixed = [k for k in keys if not ra[k]["ok"] and rb[k]["ok"]]
    regr = [k for k in keys if ra[k]["ok"] and not rb[k]["ok"]]
    print(f"compare {a} -> {b}  ({len(keys)} hands)")
    print(f"  {a} correct: {sum(r['ok'] for r in ra.values())}")
    print(f"  {b} correct: {sum(r['ok'] for r in rb.values())}")
    print(f"  FIXED   (wrong->right): {len(fixed)}")
    print(f"  REGRESS (right->wrong): {len(regr)}   <-- must be 0 to ship")
    for k in regr[:20]:
        print(f"    REGRESS {k}: gt={ra[k]['gt']}  {a}={ra[k]['cards']}({ra[k]['conf']})  "
              f"{b}={rb[k]['cards']}({rb[k]['conf']})")


def live(n: int) -> None:
    import asyncio
    import asyncpg
    from ocr.region_detector import detect_regions
    from ocr import table_parser as TP
    from ocr.classifier.infer import CardClassifier

    async def _fetch():
        c = await asyncpg.connect(os.environ["SUPABASE_CONN"], statement_cache_size=0)
        try:
            rows = await c.fetch("SELECT hand_id,image_data FROM analysis_snapshots "
                                 "WHERE image_data IS NOT NULL ORDER BY hand_id DESC "
                                 "LIMIT $1", n)
            return [(r["hand_id"], bytes(r["image_data"])) for r in rows]
        finally:
            await c.close()

    clf = CardClassifier()
    data = asyncio.run(_fetch())
    old_block = new_block = recovered = tot = 0
    for _hid, b in data:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        regions = detect_regions(img) if img is not None else None
        if not regions:
            continue
        tot += 1

        def classify(crops):
            if not crops:
                return 0.0
            det = clf.classify_batch_detailed_tta([TP._trim_above_card_edge(c) for c in crops])
            return min(min(d["rank_conf"], d["suit_conf"]) for d in det)

        oc = classify(TP._locate_hero_bright(regions.get("table")))
        _, nc = _gated_hero(img, regions, TP, clf)
        if oc < RELOCATE_CONF:
            old_block += 1
            if nc >= RELOCATE_CONF:
                recovered += 1
        if nc < RELOCATE_CONF:
            new_block += 1
    print(f"\nLIVE Telegram traffic (n={tot})")
    print(f"  OLD blocking-Gemini (conf<{RELOCATE_CONF}): {old_block} ({100*old_block/tot:.1f}%)")
    print(f"  NEW blocking-Gemini (conf<{RELOCATE_CONF}): {new_block} ({100*new_block/tot:.1f}%)")
    print(f"  recovered to local emit: {recovered} ({100*recovered/max(1,old_block):.1f}% of old blocks)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1]
    if mode == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif mode == "runf":
        runf(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    elif mode == "cmp":
        cmp(sys.argv[2], sys.argv[3])
    elif mode == "live":
        live(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
