#!/usr/bin/env python3
"""Verify every scraped replay image is labelled with the correct hand id.

The PokerCraft replay renders the hand id in the title bar
("HH Daily Classic $3 -#TM5963880955"). The scraper names each file
<hand_id>.png from the recorded list order; if the in-modal right arrow ever
stepped out of sync with that order, the file would carry the wrong id.

This OCRs the title strip of every image and asserts the embedded
"#TM<digits>" equals the filename stem — a free, definitive, dataset-wide
integrity check (no Gemini, no ground truth needed). Any mismatch means the
arrow/list order assumption broke and those images must be re-scraped.

Usage:
  python scripts/verify_image_labels.py data/hand_images/img [--limit N]
"""

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

TM_RE = re.compile(r"TM\d{6,}")


def ocr_title(png: Path) -> str:
    # Title strip (top ~40px), grayscale, aspect-preserving 3x upscale.
    # Feed via stdin — the sandbox blocks tesseract from reading temp files.
    im = Image.open(png)
    w, _ = im.size
    crop = im.crop((0, 0, w, 40)).convert("L").resize((w * 3, 120))
    buf = io.BytesIO()
    crop.save(buf, "PNG")
    r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "7"],
                        input=buf.getvalue(), capture_output=True)
    out = r.stdout.decode("utf-8", "ignore").replace(" ", "")
    m = TM_RE.search(out)
    return m.group(0) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pngs = sorted(Path(args.images).glob("*.png"))
    if args.limit:
        # Even sample across the set, not just the first N.
        step = max(1, len(pngs) // args.limit)
        pngs = pngs[::step][: args.limit]
    if not pngs:
        sys.exit("no images")

    ok = mismatch = unreadable = 0
    bad = []
    for i, p in enumerate(pngs):
        title_id = ocr_title(p)
        if not title_id:
            unreadable += 1
            bad.append((p.stem, "UNREADABLE"))
        elif title_id == p.stem:
            ok += 1
        else:
            mismatch += 1
            bad.append((p.stem, title_id))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pngs)}  ok={ok} mismatch={mismatch} "
                  f"unreadable={unreadable}")

    n = len(pngs)
    print("=" * 60)
    print(f"checked        : {n} images")
    print(f"label CORRECT  : {ok}  ({ok/n*100:.2f}%)")
    print(f"MISLABELLED    : {mismatch}")
    print(f"title unreadable: {unreadable}")
    if bad[:20]:
        print("-" * 60)
        for stem, got in bad[:20]:
            print(f"  file={stem}  title={got}")
    print("=" * 60)
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
