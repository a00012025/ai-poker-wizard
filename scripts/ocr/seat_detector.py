"""Classical avatar detection for N8 replayer screenshots (Phase C, D4).

Find the people first, then read their numbers. N8 avatars are fixed-size
circular discs arranged on a ring around the oval table. We isolate the table
region (region_detector), run a Hough circle transform tuned to the measured
avatar radius band, then drop candidates in the central board-card zone (no
seats there). NO neural detector in this phase — a CNN is a fallback decision
only if classical recall measures < 97% on the auto-label harness.

Pure CV module: numpy + OpenCV only.
"""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present in the OCR stack
    cv2 = None

# Measured on 499x640 table-region crops (the fixed output of
# region_detector.detect_regions()["table"]). Avatar discs render ~18-30px
# radius; the board cards sit in the central horizontal band.
_R_MIN = 10
_R_MAX = 38
_MIN_DIST = 28


def detect_avatars(table_region: np.ndarray, table_size_hint: int | None = None) -> list[dict]:
    """Return avatar anchors as ``[{"cx","cy","r","conf"}, ...]`` in the
    table-region frame. ``table_region`` is the cropped table image
    (region_detector.detect_regions()["table"]).

    Candidates inside the central board zone (where community cards render) are
    dropped — no seat sits there. ``table_size_hint`` is accepted for API
    parity (used only to bound the returned count when the ring is crowded).
    """
    if cv2 is None or table_region is None:
        return []
    table = table_region
    h, w = table.shape[:2]
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY) if table.ndim == 3 else table
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=_MIN_DIST,
        param1=100, param2=22, minRadius=_R_MIN, maxRadius=_R_MAX,
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)

    # Central board-card exclusion zone: the community cards render across the
    # horizontal mid-band of the oval. Seats ring the outside.
    cx0, cy0 = w / 2.0, h / 2.0
    board_half_w = 0.34 * w
    board_half_h = 0.16 * h
    out = []
    for x, y, r in circles:
        if (abs(x - cx0) < board_half_w) and (abs(y - cy0) < board_half_h):
            continue  # board zone — not a seat
        # confidence: prefer mid-band radii (true avatars cluster there)
        conf = 1.0 - abs(r - (_R_MIN + _R_MAX) / 2.0) / (_R_MAX - _R_MIN)
        out.append({"cx": float(x), "cy": float(y), "r": float(r),
                    "conf": round(max(0.1, conf), 2)})
    # de-dup near-coincident circles, keep the higher-confidence one
    out.sort(key=lambda c: -c["conf"])
    kept: list[dict] = []
    for c in out:
        if all((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2 > _MIN_DIST ** 2
               for k in kept):
            kept.append(c)
    return kept
