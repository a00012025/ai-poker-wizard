#!/usr/bin/env python3
"""Bulletproof reader for the hand id rendered in a PokerCraft replay's
title bar ("HH <tournament> -#TM<digits>").

The replay PNG bakes the hand id into the top strip; no DOM element ever
exposes it (verified live: modal text is just "OK", no row is marked, the
URL never changes). So this strip is the *only* authoritative answer to
"which hand is this image" — it must be read reliably, because the scraper
relies on it to name files and to detect a stale/duplicate frame.

Recipe was calibrated against scenes whose id was known from a direct row
click (anchor) and confirmed across hands of differing heights: crop the
top band (a small fraction of image height), upscale heavily with LANCZOS,
optional binarization, tesseract psm 7/6/11 with a digit whitelist. Votes
across several variants are tallied; the winner needs an outright majority
(and, when a validator is supplied, must be an id we actually expect).

tesseract is fed via stdin — the sandbox blocks it from reading temp files.
"""

import io
import re
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image

TM_RE = re.compile(r"TM\d{6,}")

# (height-fraction, upscale, threshold|None, psm) — calibrated variants,
# cheapest/most-reliable first so confident reads exit early.
_VARIANTS = [
    (0.040, 6, None, 7), (0.040, 6, 140, 7), (0.030, 6, None, 7),
    (0.055, 6, None, 7), (0.040, 8, 140, 6), (0.040, 6, 170, 11),
    (0.030, 8, None, 6), (0.055, 4, 140, 7),
]


def _tess(img: Image.Image, psm: int) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", str(psm),
         "-c", "tessedit_char_whitelist=#TM0123456789-"],
        input=buf.getvalue(), capture_output=True)
    return r.stdout.decode("utf-8", "ignore")


def _read(strip_src: Image.Image, w: int, h: int) -> Counter:
    votes: Counter = Counter()
    cache: dict[float, Image.Image] = {}
    for frac, scale, thr, psm in _VARIANTS:
        th = max(14, int(h * frac))
        if frac not in cache:
            cache[frac] = strip_src.crop((0, 0, w, th))
        big = cache[frac].resize((w * scale, th * scale), Image.LANCZOS)
        if thr is not None:
            big = big.point(lambda p, q=thr: 0 if p < q else 255)
        m = TM_RE.search(_tess(big, psm).replace(" ", "").replace("#", ""))
        if m:
            votes[m.group(0)] += 1
    return votes


def read_title_id(png: Path | bytes, *, valid: set[str] | None = None
                   ) -> tuple[str | None, int, int]:
    """Return (hand_id|None, winner_votes, total_votes).

    A result is returned only on a strict majority (> half the variants
    that produced any id). If ``valid`` is given, a non-majority winner is
    still accepted when it is the *only* voted id that is a known id.
    """
    im = Image.open(io.BytesIO(png) if isinstance(png, bytes) else png)
    w, h = im.size
    gray = im.convert("L")
    votes = _read(gray, w, h)
    if not votes:
        return None, 0, 0
    total = sum(votes.values())
    best, nbest = votes.most_common(1)[0]
    if nbest * 2 > total:
        return best, nbest, total
    if valid is not None:
        in_gt = [k for k in votes if k in valid]
        if len(in_gt) == 1:
            return in_gt[0], votes[in_gt[0]], total
    return None, nbest, total
