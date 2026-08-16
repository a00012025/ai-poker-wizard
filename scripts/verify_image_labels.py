#!/usr/bin/env python3
"""Verify (and optionally repair) the scraped replay dataset by each image's
OWN rendered title bar — the only authoritative "which hand is this".

The scraper used to name each file from the arrow-walk position, assuming
the in-modal right arrow steps in hand-list order. That assumption broke
for at least one tournament (Daily Hyper 1: a stale anchor frame shifted
every file by one), so a file's name cannot be trusted on its own. The
replay PNG, however, bakes "HH <tournament> -#TM<id>" into its top strip;
title_ocr.read_title_id recovers it reliably (calibrated + majority-voted).

For every <name>.png this OCRs the title and classifies:

  CORRECT    title id == filename stem
  MISLABEL   title id is a *different* valid ground-truth id
             -> the file actually holds that hand; rename it
  UNREADABLE title strip could not be read confidently -> re-scrape

With --apply it rewrites the dataset to be self-consistent: every file
ends up named by the hand it truly contains. Collisions (two files whose
titles read the same id) are quarantined, not silently overwritten.

Usage:
  python scripts/verify_image_labels.py data/hand_images/img \\
      --ground-truth data/pokercraft_corpus/ground_truth/ground_truth.jsonl \\
      [--limit N] [--apply] [--report data/hand_images/label_report.json]
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from title_ocr import read_title_id  # noqa: E402

_GT: set[str] | None = None


def _init(gt: set[str] | None) -> None:
    global _GT
    _GT = gt


def _scan(path_str: str) -> tuple[str, str | None]:
    """Worker: (stem, true_id|None). Module-level for picklability."""
    p = Path(path_str)
    tid, _, _ = read_title_id(p.read_bytes(), valid=_GT)
    return p.stem, tid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--ground-truth", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="even sample across the set (validation runs)")
    ap.add_argument("--apply", action="store_true",
                    help="rename mislabeled files to their true id")
    ap.add_argument("--from-report", default="",
                    help="apply a previously computed report's plan instead "
                         "of re-OCRing (deterministic, no scan)")
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 4))
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    gt_ids: set[str] | None = None
    if args.ground_truth:
        gt_ids = set()
        with open(args.ground_truth, encoding="utf-8") as fh:
            for line in fh:
                gt_ids.add(json.loads(line)["hand_id"])

    img_dir = Path(args.images)
    correct = unreadable = 0
    mislabel: list[tuple[str, str]] = []   # (stem, true_id)
    bad_read: list[str] = []

    if args.from_report:
        rep = json.loads(Path(args.from_report).read_text())
        mislabel = [tuple(x) for x in rep.get("mislabel", [])]
        bad_read = list(rep.get("unreadable", []))
        unreadable = len(bad_read)
        n = rep.get("checked", len(mislabel))
        correct = rep.get("correct", n - len(mislabel) - unreadable)
        print(f"[from-report] {args.from_report}: "
              f"{len(mislabel)} mislabel, {unreadable} unreadable")
    else:
        pngs = sorted(img_dir.glob("*.png"))
        if args.limit:
            step = max(1, len(pngs) // args.limit)
            pngs = pngs[::step][: args.limit]
        if not pngs:
            sys.exit("no images")
        done_n = 0
        with Pool(args.workers, initializer=_init,
                  initargs=(gt_ids,)) as pool:
            for stem, tid in pool.imap_unordered(
                    _scan, [str(p) for p in pngs], chunksize=8):
                if not tid:
                    unreadable += 1
                    bad_read.append(stem)
                elif tid == stem:
                    correct += 1
                else:
                    mislabel.append((stem, tid))
                done_n += 1
                if done_n % 500 == 0:
                    print(f"  {done_n}/{len(pngs)}  ok={correct} "
                          f"mislabel={len(mislabel)} unreadable={unreadable}",
                          flush=True)
        n = len(pngs)

    # A title id may be valid yet outside GT only if GT is partial; flag
    # mislabels whose true id we cannot benchmark against.
    relabel, untrack = [], []
    for stem, tid in mislabel:
        (relabel if gt_ids is None or tid in gt_ids else untrack).append(
            (stem, tid))

    print("=" * 64)
    print(f"checked        : {n} images")
    print(f"label CORRECT  : {correct}  ({correct/n*100:.3f}%)")
    print(f"MISLABELLED    : {len(mislabel)}  "
          f"(relabelable={len(relabel)} untrackable={len(untrack)})")
    print(f"title UNREADABLE: {unreadable}  ({unreadable/n*100:.3f}%)")
    for stem, tid in mislabel[:25]:
        print(f"  file={stem}  true={tid}")

    applied = quarantined = 0
    if args.apply and (relabel or untrack):
        qdir = img_dir.parent / "quarantine"
        qdir.mkdir(exist_ok=True)
        # Untrackable: file's true id is not in GT — its name is wrong AND
        # its content can't be benchmarked. Move it out so img/ is clean.
        for stem, tid in untrack:
            src = img_dir / f"{stem}.png"
            if src.exists():
                src.replace(qdir / f"untrackable_{stem}_is_{tid}.png")
                quarantined += 1
        # Two-phase: stage to temp names so an A->B, B->C cycle can't clobber.
        staged = []
        for stem, tid in relabel:
            src = img_dir / f"{stem}.png"
            if not src.exists():
                continue
            tmp = img_dir / f"__stage_{tid}.png"
            if tmp.exists():
                src.replace(qdir / f"dupe_{stem}_as_{tid}.png")
                quarantined += 1
                continue
            src.rename(tmp)
            staged.append((tmp, tid))
        for tmp, tid in staged:
            dst = img_dir / f"{tid}.png"
            if dst.exists():
                # a CORRECT file already owns this id -> our copy is a dupe
                tmp.replace(qdir / f"dupe_as_{tid}.png")
                quarantined += 1
            else:
                tmp.rename(dst)
                applied += 1
        print("-" * 64)
        print(f"renamed {applied} files to their true id; "
              f"quarantined {quarantined} files -> {qdir}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "checked": n, "correct": correct,
            "mislabel": mislabel, "unreadable": bad_read,
            "untrackable": untrack,
            "applied": applied, "quarantined": quarantined,
        }, ensure_ascii=False, indent=1))
        print(f"report -> {args.report}")
    print("=" * 64)
    # Non-zero only on conditions that need a re-scrape, not on repaired ones.
    return 1 if (unreadable or untrack) else 0


if __name__ == "__main__":
    raise SystemExit(main())
