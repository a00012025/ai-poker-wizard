"""Card crop augmentation for CardCNN v2/v3 training."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .overlay_library import OverlayLibrary

REPO_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _default_overlay_library() -> OverlayLibrary:
    return OverlayLibrary(REPO_ROOT / "data" / "win_overlays")


def apply_real_win_overlay(
    img: np.ndarray,
    *,
    rng: np.random.Generator,
    p: float = 0.50,
    lib: OverlayLibrary | None = None,
) -> np.ndarray:
    if rng.random() > p:
        return img
    if lib is None:
        lib = _default_overlay_library()
    overlay = lib.sample(rng)
    if overlay is None:
        return img
    h, w = img.shape[:2]
    target_w = int(rng.uniform(0.6, 1.0) * w)
    target_w = max(1, min(target_w, w))
    scale = target_w / max(overlay.shape[1], 1)
    target_h = max(1, int(overlay.shape[0] * scale))
    target_h = min(target_h, max(1, h - 1))
    resized = cv2.resize(
        overlay, (target_w, target_h), interpolation=cv2.INTER_AREA
    )
    x_hi = max(1, w - target_w)
    x0 = int(rng.integers(0, x_hi))
    y_lo = int(h * 0.30)
    y_hi = max(y_lo + 1, h - target_h)
    y0 = int(rng.integers(y_lo, y_hi))
    out = img.copy()
    bgr = resized[:, :, :3]
    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
    alpha = alpha * float(rng.uniform(0.7, 1.0))
    roi = out[y0:y0 + target_h, x0:x0 + target_w]
    blended = (
        bgr.astype(np.float32) * alpha
        + roi.astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)
    out[y0:y0 + target_h, x0:x0 + target_w] = blended
    return out


def apply_win_sticker(
    img: np.ndarray,
    *,
    rng: np.random.Generator,
    p: float = 0.25,
) -> np.ndarray:
    if rng.random() > p:
        return img
    out = img.copy()
    h, w = out.shape[:2]
    sticker_w = int(rng.integers(max(1, int(w * 0.5)), max(2, int(w * 0.95))))
    sticker_h = int(rng.integers(max(1, int(h * 0.18)), max(2, int(h * 0.32))))
    x0 = int(rng.integers(0, max(1, w - sticker_w)))
    y0 = int(rng.integers(int(h * 0.25), max(int(h * 0.25) + 1, h - sticker_h)))
    overlay = out.copy()
    color = (
        int(rng.integers(0, 80)),
        int(rng.integers(180, 230)),
        int(rng.integers(220, 255)),
    )
    alpha = float(rng.uniform(0.6, 0.95))
    cv2.rectangle(overlay, (x0, y0), (x0 + sticker_w, y0 + sticker_h), color, thickness=-1)
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def color_jitter(
    img: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: float = 0.2,
) -> np.ndarray:
    factors = 1.0 + (rng.random(3) - 0.5) * 2 * strength
    out = img.astype(np.float32) * factors.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_geometric(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        float(rng.uniform(-2.0, 2.0)),
        float(rng.uniform(0.97, 1.03)),
    )
    matrix[0, 2] += float(rng.integers(-2, 3))
    matrix[1, 2] += float(rng.integers(-2, 3))
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_all(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    img = light_geometric(img, rng=rng)
    img = color_jitter(img, rng=rng, strength=0.2)
    # 70% real overlay (when corpus available), 20% synthetic block, 10% clean.
    # The real-overlay path no-ops when the library is empty, so this stays
    # safe in CI/dev before Task A.1's harvest has run.
    img = apply_real_win_overlay(img, rng=rng, p=0.70)
    img = apply_win_sticker(img, rng=rng, p=0.20)
    return img
