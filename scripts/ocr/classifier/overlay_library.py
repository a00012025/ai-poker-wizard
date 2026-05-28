"""Loader + sampler for real captured WIN overlays."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OverlayLibrary:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: list[np.ndarray] = []
        if self.root.exists():
            for png in sorted(self.root.glob("*.png")):
                rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
                if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                    continue
                self._cache.append(rgba)

    def size(self) -> int:
        return len(self._cache)

    def sample(self, rng: np.random.Generator) -> np.ndarray | None:
        if not self._cache:
            return None
        idx = int(rng.integers(0, len(self._cache)))
        return self._cache[idx]
