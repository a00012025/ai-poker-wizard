"""Avatar-anchored stack reading for N8 replayer screenshots (Phase C, D4).

For each avatar anchor, read the stack value that belongs to THAT seat: N8
renders the green "XX.X BB" stack directly below the avatar disc. We OCR the
table region once, then claim, per avatar, the nearest "XX.X BB" text inside a
fixed offset box under the disc. BB texts not claimed by any avatar (pot
displays, bet sizes on the felt, the action timeline) are dropped by
construction — that is the phantom-rejection win. Bounty "$" pills sit left/
right of the avatar and are never inside the stack ROI, so no $-filtering is
needed.

Output is the SAME named_stacks schema the rest of the pipeline consumes, plus
``anchor_conf`` per row.
"""
import re

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

_BB = re.compile(r"(\d+\.?\d*)\s*BB", re.IGNORECASE)

# Stack-ROI offset relative to an avatar of radius r (measured on 499x640 table
# crops): the green stack plate sits just below the disc, roughly disc-width.
_ROI_DOWN = 1.4      # box top starts ~1.4r below the centre ...
_ROI_H = 2.6         # ... and extends 2.6r tall
_ROI_HALF_W = 2.2    # half-width 2.2r


def _parse_bb(text):
    m = _BB.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 0.5 < v < 500 else None


def read_seats(table_region: np.ndarray, avatars: list[dict]) -> list[dict]:
    """Return ``[{"name","stack","x","y","anchor_conf"}, ...]`` — one row per
    avatar that owns a nearby stack read. ``table_region`` is the cropped table
    image; ``avatars`` come from seat_detector.detect_avatars (same frame)."""
    if cv2 is None or table_region is None or not avatars:
        return []
    from .ocr_utils import ocr_full_image
    toks = ocr_full_image(table_region)
    bb_toks = []
    for t in toks:
        v = _parse_bb(t["text"])
        if v is not None:
            bb_toks.append({"v": v, "x": t["center_x"], "y": t["center_y"]})
    name_toks = [t for t in toks
                 if len(t["text"].strip()) >= 2 and _parse_bb(t["text"]) is None
                 and not re.match(r"^[\d.]+$", t["text"].strip())]

    used = set()
    rows = []
    for av in avatars:
        cx, cy, r = av["cx"], av["cy"], av["r"]
        top = cy + _ROI_DOWN * r - 0.3 * r
        bot = cy + (_ROI_DOWN + _ROI_H) * r
        half_w = _ROI_HALF_W * r
        best, best_d = None, None
        for i, b in enumerate(bb_toks):
            if i in used:
                continue
            if not (top - 0.6 * r <= b["y"] <= bot and abs(b["x"] - cx) <= half_w):
                continue
            d = abs(b["y"] - (cy + _ROI_DOWN * r)) + 0.5 * abs(b["x"] - cx)
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is None:
            continue
        used.add(best)
        b = bb_toks[best]
        # nearest name token above the stack, within the seat column
        nm, nm_d = None, None
        for t in name_toks:
            if abs(t["center_x"] - cx) <= half_w and (cy - 1.6 * r) <= t["center_y"] <= b["y"]:
                d = abs(t["center_y"] - cy)
                if nm_d is None or d < nm_d:
                    nm, nm_d = t["text"].strip(), d
        rows.append({"name": nm, "stack": float(b["v"]),
                     "x": float(b["x"]), "y": float(b["y"]),
                     "anchor_conf": float(av["conf"])})
    return rows
