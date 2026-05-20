"""Card crop augmentation for CardCNN v2 training."""
from __future__ import annotations

import cv2
import numpy as np


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
    return apply_win_sticker(img, rng=rng, p=0.25)
