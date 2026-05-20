#!/usr/bin/env python3
"""Pre-Gemini OCR precision benchmark.

Measures the deterministic CV+CNN pipeline (`parse_n8_screenshot`) alone — no
Gemini fallback. Reports per-field accuracy, confidence calibration, and
failure-mode breakdown so we can drive the standalone pipeline to 99.9% and
let Gemini stay only as a safety net.

Compares against PokerCraft HH ground truth produced by build_ground_truth.py.

Usage:
  python scripts/ocr_precision.py                                # all paired
  python scripts/ocr_precision.py --limit 500                    # quick sample
  python scripts/ocr_precision.py --limit 500 --workers 8        # parallel
  python scripts/ocr_precision.py --out data/ocr_precision       # custom out
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT = {s: i for i, s in enumerate("cdhs")}


# ---------- canonicalization (must match ocr_benchmark.py semantics) ----------

def _cards(s: str) -> list[str]:
    s = (s or "").replace(" ", "")
    return [s[i : i + 2] for i in range(0, len(s) - 1, 2)]


def _canon_cardset(s: str) -> tuple:
    return tuple(sorted(
        _cards(s),
        key=lambda c: (_RANK.get(c[0], -1), _SUIT.get(c[1], -1)),
    ))


def _canon_action(tok: str) -> str:
    tok = (tok or "").strip().upper()
    if tok.startswith("AI"):
        return "AI"
    if tok.startswith("R"):
        try:
            return f"R{float(tok[1:]):g}"
        except ValueError:
            return tok
    return tok


def _action_types(s: str) -> list[str]:
    out = []
    for t in (s or "").split("-"):
        c = _canon_action(t)
        out.append("R" if c.startswith("R") else c)
    return out


def _actions_sized(s: str) -> list[str]:
    return [_canon_action(t) for t in (s or "").split("-") if t]


def _streets(hand: dict) -> list[tuple]:
    res = []
    for st in (hand.get("streets") or []):
        key = st.get("board") if "board" in st else st.get("card")
        if key:
            res.append(_canon_cardset(key))
    return res


def compare(parsed: dict, gt: dict) -> dict:
    f: dict = {}
    f["hero_hand"] = _canon_cardset(parsed.get("hero_hand", "")) == \
        _canon_cardset(gt.get("hero_hand", ""))
    f["hero_position"] = (str(parsed.get("hero_position", "")).upper()
                          == str(gt.get("hero_position", "")).upper())
    f["board"] = _streets(parsed) == _streets(gt)
    f["preflop_types"] = _action_types(parsed.get("preflop_actions", "")) == \
        _action_types(gt.get("preflop_actions", ""))
    f["preflop_sized"] = _actions_sized(parsed.get("preflop_actions", "")) == \
        _actions_sized(gt.get("preflop_actions", ""))
    pe, ge = parsed.get("effective_bb"), gt.get("effective_bb")
    if isinstance(pe, (int, float)) and isinstance(ge, (int, float)):
        f["effective_bb_tol"] = abs(pe - ge) <= max(1.0, 0.12 * ge)
    else:
        f["effective_bb_tol"] = pe == ge
    # OCR's `players_at_table` counts entries it actually saw acting, which
    # matches GT's `num_players` (active in hand). `table_size` (max seats)
    # is a different concept and not what the OCR exposes.
    pt = parsed.get("players_at_table")
    gt_size = gt.get("num_players") or gt.get("table_size")
    f["table_size"] = (pt == gt_size) if (pt and gt_size) else False
    f["hand_exact"] = (f["hero_hand"] and f["hero_position"]
                       and f["board"] and f["preflop_types"])
    f["hand_exact_sized"] = f["hand_exact"] and f["preflop_sized"]
    f["critical_error"] = (not f["hero_hand"]) or (not f["board"])
    return f


# ---------- per-image worker ----------

_GT_MAP: dict[str, dict] = {}


def _init_worker(gt_path: str) -> None:
    """Load GT into module globals once per worker (avoid pickling 7k dict)."""
    global _GT_MAP
    with open(gt_path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            _GT_MAP[o["hand_id"]] = o["ground_truth"]


def _classify_failure(parsed: dict | None, gt: dict, fields: dict) -> str:
    """Bucket the failure into a coarse mode so we can group them."""
    if parsed is None:
        return "parse_none"
    if not parsed.get("hero_hand"):
        return "hero_cards_missing"
    if not fields["hero_hand"]:
        return "hero_cards_wrong"
    if not fields["board"]:
        return "board_wrong"
    if not fields["hero_position"]:
        return "position_wrong"
    if not fields["preflop_types"]:
        return "preflop_action_types_wrong"
    if not fields["preflop_sized"]:
        return "preflop_sizes_wrong"
    if not fields["table_size"]:
        return "table_size_wrong"
    return "other"


def _run_one(img_path_str: str) -> dict:
    p = Path(img_path_str)
    hand_id = p.stem
    gt = _GT_MAP.get(hand_id)
    if gt is None:
        return {"hand_id": hand_id, "skipped": True}
    # Import inside worker so each process owns its own CNN / EasyOCR.
    from ocr.n8_parser import parse_n8_screenshot
    t0 = time.time()
    try:
        result = parse_n8_screenshot(p.read_bytes())
    except Exception as e:
        return {
            "hand_id": hand_id,
            "image": str(p),
            "error": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc().splitlines()[-3:],
            "elapsed_s": time.time() - t0,
        }
    parsed = result.get("hand")
    conf = float(result.get("confidence") or 0.0)
    card_conf = float(result.get("card_confidence") or 0.0)
    parts = result.get("confidence_parts") or {}

    if parsed is None:
        return {
            "hand_id": hand_id,
            "image": str(p),
            "parsed_none": True,
            "confidence": conf,
            "card_confidence": card_conf,
            "confidence_parts": parts,
            "elapsed_s": time.time() - t0,
        }
    fields = compare(parsed, gt)
    return {
        "hand_id": hand_id,
        "image": str(p),
        "confidence": conf,
        "card_confidence": card_conf,
        "confidence_parts": parts,
        "fields": fields,
        "failure_mode": None if fields["hand_exact"] else _classify_failure(parsed, gt, fields),
        "parsed": {k: parsed.get(k) for k in (
            "hero_hand", "hero_position", "players_at_table",
            "preflop_actions", "effective_bb")},
        "parsed_streets": _streets(parsed),
        "gt": {k: gt.get(k) for k in (
            "hero_hand", "hero_position", "table_size", "num_players",
            "preflop_actions", "effective_bb")},
        "gt_streets": _streets(gt),
        "elapsed_s": time.time() - t0,
    }


# ---------- driver ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/hand_images/img")
    ap.add_argument("--ground-truth",
                    default="data/pokercraft_corpus/ground_truth/ground_truth.jsonl")
    ap.add_argument("--out", default="data/ocr_precision")
    ap.add_argument("--limit", type=int, default=0,
                    help="Evenly sample N pairs (for fast iteration)")
    ap.add_argument("--workers", type=int,
                    default=min(4, os.cpu_count() or 2))
    ap.add_argument("--max-failures", type=int, default=40)
    args = ap.parse_args()

    gt_path = str(Path(args.ground_truth).resolve())
    img_dir = Path(args.images)

    # Quick scan for pairs (no GT loaded in main proc; workers do that).
    gt_ids: set[str] = set()
    with open(gt_path, encoding="utf-8") as fh:
        for line in fh:
            gt_ids.add(json.loads(line)["hand_id"])
    imgs = sorted(img_dir.glob("*.png"))
    pairs = [p for p in imgs if p.stem in gt_ids]
    if args.limit and len(pairs) > args.limit:
        step = max(1, len(pairs) // args.limit)
        pairs = pairs[::step][: args.limit]
    if not pairs:
        sys.exit("no paired images")

    print(f"[ocr_precision] paired={len(pairs)}  images={len(imgs)}  "
          f"gt={len(gt_ids)}  workers={args.workers}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs_path = out_dir / "diffs.jsonl"
    diffs = diffs_path.open("w", encoding="utf-8")

    agg = Counter()
    failure_modes = Counter()
    conf_buckets = {
        ">=0.95": Counter(), "0.80-0.95": Counter(),
        "0.50-0.80": Counter(), "0.30-0.50": Counter(),
        "<0.30": Counter(),
    }

    def bucket(c: float) -> str:
        if c >= 0.95: return ">=0.95"
        if c >= 0.80: return "0.80-0.95"
        if c >= 0.50: return "0.50-0.80"
        if c >= 0.30: return "0.30-0.50"
        return "<0.30"

    n = parse_none = errors = 0
    written_failures = 0
    elapsed_sum = 0.0
    t_start = time.time()

    paths = [str(p) for p in pairs]
    if args.workers > 1:
        ctx = mp.get_context("spawn")  # CNN + EasyOCR are not fork-safe
        pool = ctx.Pool(args.workers, initializer=_init_worker,
                        initargs=(gt_path,))
        iterator = pool.imap_unordered(_run_one, paths, chunksize=2)
    else:
        _init_worker(gt_path)
        iterator = (_run_one(p) for p in paths)
        pool = None

    try:
        for r in iterator:
            n += 1
            elapsed_sum += float(r.get("elapsed_s") or 0.0)
            if r.get("skipped"):
                continue
            if r.get("error"):
                errors += 1
                if written_failures < args.max_failures:
                    diffs.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                    written_failures += 1
                continue
            if r.get("parsed_none"):
                parse_none += 1
                failure_modes["parse_none"] += 1
                conf_buckets[bucket(r["confidence"])]["total"] += 1
                if written_failures < args.max_failures:
                    diffs.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                    written_failures += 1
                continue
            f = r["fields"]
            b = bucket(r["confidence"])
            conf_buckets[b]["total"] += 1
            if f["hand_exact"]:
                conf_buckets[b]["exact"] += 1
            for k, v in f.items():
                if v:
                    agg[k] += 1
            if not f["hand_exact"]:
                failure_modes[r["failure_mode"]] += 1
                if written_failures < args.max_failures:
                    diffs.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                    written_failures += 1
            if n % 25 == 0:
                exact = agg["hand_exact"]
                rate = (exact / max(1, n - parse_none - errors)) * 100
                print(f"  {n}/{len(pairs)}  exact={exact} "
                      f"({rate:.2f}%)  parse_none={parse_none}  err={errors}  "
                      f"avg_ms={elapsed_sum/max(1,n)*1000:.0f}",
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        diffs.close()

    scored = max(1, n - parse_none - errors)

    def pct(x: int) -> str:
        return f"{x/scored*100:.3f}%"

    field_keys = ("hero_hand", "hero_position", "board", "preflop_types",
                  "preflop_sized", "effective_bb_tol", "table_size",
                  "hand_exact", "hand_exact_sized", "critical_error")
    summary = {
        "paired": len(pairs),
        "scored": scored,
        "parse_none": parse_none,
        "errors": errors,
        "elapsed_total_s": round(time.time() - t_start, 1),
        "avg_ms_per_image": round(elapsed_sum / max(1, n) * 1000, 1),
        "field_accuracy": {k: pct(agg[k]) for k in field_keys},
        "failure_modes": dict(failure_modes.most_common()),
        "confidence_buckets": {
            b: {
                "n": c["total"],
                "exact": c["exact"],
                "exact_rate": (f"{c['exact']/c['total']*100:.2f}%"
                               if c["total"] else "n/a"),
            } for b, c in conf_buckets.items()
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("=" * 72)
    print(f"PRE-GEMINI OCR PRECISION  ({scored} scored, "
          f"{parse_none} parse_none, {errors} errors)")
    print(f"  hand_exact (headline)   : {pct(agg['hand_exact'])} "
          f"[{agg['hand_exact']}/{scored}]")
    print(f"  hand_exact_sized        : {pct(agg['hand_exact_sized'])}")
    print(f"  critical_error          : {pct(agg['critical_error'])}")
    print("  field accuracy:")
    for k in ("hero_hand", "hero_position", "board",
              "preflop_types", "preflop_sized",
              "effective_bb_tol", "table_size"):
        print(f"    {k:18}: {pct(agg[k])}")
    print("  failure modes:")
    for mode, cnt in failure_modes.most_common():
        print(f"    {mode:30}: {cnt}")
    print("  confidence calibration (exact-match rate per bucket):")
    for b in (">=0.95", "0.80-0.95", "0.50-0.80", "0.30-0.50", "<0.30"):
        c = conf_buckets[b]
        if not c["total"]:
            continue
        print(f"    {b:10}  n={c['total']:5}  "
              f"exact={c['exact']:5}  "
              f"rate={c['exact']/c['total']*100:6.2f}%")
    print(f"  avg latency             : {summary['avg_ms_per_image']} ms/image")
    print(f"  diffs (first {written_failures}) -> {diffs_path}")
    print(f"  summary                 -> {out_dir/'summary.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
