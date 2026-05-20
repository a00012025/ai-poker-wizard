#!/usr/bin/env python3
"""OCR accuracy benchmark: scraped replay images vs HH ground truth.

Pairs each <hand_id>.png with its ground_truth.jsonl entry (HH-derived, the
authoritative label), runs the *production* image parse, canonicalizes both
sides, and reports field-level + hand-level accuracy plus a critical-error
breakdown.

Why canonicalize: the OCR and HH express the same fact differently —
"9cTh" vs "Th9c", "R2" vs "R2.0", effective_bb 10 vs 10.1. Those are not
OCR errors, so the metric must compare normalized forms or it will report
false failures (verified on TM5963540471, which OCR'd correctly yet diffed
on raw strings).

Precision definition (headline number):
  hand-level exact match on the critical fields
  {hero_hand, hero_position, board-per-street, preflop action *types*}.
  Target: >= 99.9%. Also reported: per-field accuracy, a lenient variant
  (raise sizes within tolerance), and a critical-error rate
  (hero_hand or any board card wrong — the errors that most distort GTO).

Usage:
  python scripts/ocr_benchmark.py --images data/hand_images/img \\
      --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \\
      --out data/benchmark --limit 200 --concurrency 4
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_RANK = {r: i for i, r in enumerate("23456789TJQKA")}
_SUIT = {s: i for i, s in enumerate("cdhs")}


def _cards(s: str) -> list[str]:
    """Split a card string like '9cTh' / 'Js6h5s' into ['9c','Th']."""
    s = (s or "").replace(" ", "")
    return [s[i : i + 2] for i in range(0, len(s) - 1, 2)]


def _canon_cardset(s: str) -> tuple:
    """Order-independent canonical form of a set of cards."""
    return tuple(sorted(_cards(s),
                        key=lambda c: (_RANK.get(c[0], -1), _SUIT.get(c[1], -1))))


def _canon_action(tok: str) -> str:
    """F/X/C unchanged; R2.0->R2, R2.50->R2.5; AI/RAI->AI."""
    tok = (tok or "").strip().upper()
    if tok in ("AI", "RAI") or tok.startswith("AI"):
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
        out.append("R" if c.startswith("R") else c)  # type only, size dropped
    return out


def _actions_sized(s: str) -> list[str]:
    return [_canon_action(t) for t in (s or "").split("-") if t]


def _streets(hand: dict) -> list[tuple]:
    """[(canon flop set,), (turn,), (river,)] from a parsed/gt hand."""
    res = []
    for st in hand.get("streets") or []:
        if "board" in st:
            res.append(_canon_cardset(st["board"]))
        elif "card" in st:
            res.append(_canon_cardset(st["card"]))
    return res


def compare(parsed: dict, gt: dict) -> dict:
    """Return per-field booleans + the canonical values for diffing."""
    f = {}
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
    # Headline: critical fields, action *types* (sizes are a softer signal).
    f["hand_exact"] = (f["hero_hand"] and f["hero_position"]
                       and f["board"] and f["preflop_types"])
    f["hand_exact_sized"] = f["hand_exact"] and f["preflop_sized"]
    f["critical_error"] = (not f["hero_hand"]) or (not f["board"])
    return f


async def _parse(image_bytes: bytes):
    from gemini_session import GeminiSessionManager
    sess = GeminiSessionManager(db=None)
    acc = {}
    hand = await sess._parse_hand_from_image(
        0, image_bytes, "image/png", user_text="", usage_acc=acc)
    if hand:
        hand.pop("possible_ft", None)
        hand.pop("__ocr_conf__", None)
    return hand


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    gt = {}
    with open(args.ground_truth, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            gt[o["hand_id"]] = o["ground_truth"]

    imgs = sorted(Path(args.images).glob("*.png"))
    pairs = [(p, gt[p.stem]) for p in imgs if p.stem in gt]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        sys.exit("No image/ground-truth pairs found.")
    print(f"[benchmark] {len(pairs)} paired hands "
          f"({len(imgs)} images, {len(gt)} GT rows)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    diffs = (out / "diffs.jsonl").open("w", encoding="utf-8")

    sem = asyncio.Semaphore(args.concurrency)
    agg = {k: 0 for k in ("hero_hand", "hero_position", "board",
                           "preflop_types", "preflop_sized", "effective_bb_tol",
                           "hand_exact", "hand_exact_sized", "critical_error")}
    n = parse_fail = 0

    async def work(path: Path, g: dict):
        nonlocal n, parse_fail
        async with sem:
            try:
                parsed = await _parse(path.read_bytes())
            except Exception as e:  # noqa: BLE001
                parsed = None
                err = f"{type(e).__name__}: {e}"
            else:
                err = None
        n += 1
        if not parsed:
            parse_fail += 1
            diffs.write(json.dumps(
                {"hand_id": path.stem, "parse_failed": True,
                 "error": err}, ensure_ascii=False) + "\n")
            return
        f = compare(parsed, g)
        for k in agg:
            if f[k]:
                agg[k] += 1
        if not f["hand_exact"]:
            diffs.write(json.dumps({
                "hand_id": path.stem,
                "fields": {k: f[k] for k in
                           ("hero_hand", "hero_position", "board",
                            "preflop_types", "preflop_sized")},
                "ocr": {k: parsed.get(k) for k in
                        ("hero_hand", "hero_position", "preflop_actions")},
                "gt": {k: g.get(k) for k in
                       ("hero_hand", "hero_position", "preflop_actions")},
            }, ensure_ascii=False) + "\n")
        if n % 25 == 0:
            print(f"  {n}/{len(pairs)}  exact={agg['hand_exact']}  "
                  f"crit_err={agg['critical_error']}  pfail={parse_fail}")

    await asyncio.gather(*(work(p, g) for p, g in pairs))
    diffs.close()

    scored = n - parse_fail
    def pct(x):
        return f"{(x / scored * 100):.3f}%" if scored else "n/a"

    summary = {
        "paired": len(pairs), "scored": scored, "parse_failed": parse_fail,
        "hand_exact": agg["hand_exact"], "hand_exact_rate": pct(agg["hand_exact"]),
        "hand_exact_sized_rate": pct(agg["hand_exact_sized"]),
        "critical_error": agg["critical_error"],
        "critical_error_rate": pct(agg["critical_error"]),
        "field_accuracy": {k: pct(agg[k]) for k in
                           ("hero_hand", "hero_position", "board",
                            "preflop_types", "preflop_sized",
                            "effective_bb_tol")},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"PRECISION (hand-level exact, critical fields): "
          f"{summary['hand_exact_rate']}  "
          f"[{agg['hand_exact']}/{scored}]")
    print(f"  + raise sizes exact            : "
          f"{summary['hand_exact_sized_rate']}")
    print(f"  critical-error rate (hand/board): "
          f"{summary['critical_error_rate']}")
    print(f"  parse failures                 : {parse_fail}")
    print("  field accuracy:")
    for k, v in summary["field_accuracy"].items():
        print(f"    {k:18}: {v}")
    print(f"  diffs -> {out/'diffs.jsonl'}   summary -> {out/'summary.json'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
