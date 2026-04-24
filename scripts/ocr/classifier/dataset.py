"""CardDataset — loads labeled crops from data/cards/{rank}/{suit}/*.png."""
from __future__ import annotations

import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import RANK_CLASSES, SUIT_CLASSES

INPUT_H, INPUT_W = 48, 64
_RANK_TO_IDX = {r: i for i, r in enumerate(RANK_CLASSES)}
_SUIT_TO_IDX = {s: i for i, s in enumerate(SUIT_CLASSES)}
_FILENAME_RE = re.compile(r"^(?P<hand>[A-Za-z0-9]+)_(?P<src>hero|board)_(?P<slot>\d+)\.png$")


def _letterbox(img: np.ndarray, h: int = INPUT_H, w: int = INPUT_W) -> np.ndarray:
    """Resize + pad to (h, w) preserving aspect ratio, BGR."""
    ih, iw = img.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(ih * scale)), max(1, int(iw * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    top = (h - nh) // 2
    left = (w - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def _apply_aug(img: np.ndarray) -> np.ndarray:
    dx, dy = random.randint(-2, 2), random.randint(-2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         borderMode=cv2.BORDER_REPLICATE)
    scale = random.uniform(0.9, 1.1)
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.1, 0.5))
    return img


class CardDataset(Dataset):
    """Every crop under root/{rank}/{suit}/{hand}_{src}_{slot}.png."""

    def __init__(self, root: Path, augment: bool = True):
        self.root = Path(root)
        self.augment = augment
        self.samples: list[tuple[str, str, int, int]] = []
        self.paths: list[Path] = []
        for rank_dir in sorted(self.root.iterdir() if self.root.exists() else []):
            if not rank_dir.is_dir() or rank_dir.name not in _RANK_TO_IDX:
                continue
            for suit_dir in sorted(rank_dir.iterdir()):
                if not suit_dir.is_dir() or suit_dir.name not in _SUIT_TO_IDX:
                    continue
                for png in sorted(suit_dir.glob("*.png")):
                    m = _FILENAME_RE.match(png.name)
                    if not m:
                        continue
                    self.paths.append(png)
                    self.samples.append((
                        m.group("hand"), f"{m.group('src')}_{m.group('slot')}",
                        _RANK_TO_IDX[rank_dir.name],
                        _SUIT_TO_IDX[suit_dir.name],
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        img = cv2.imread(str(self.paths[idx]))
        assert img is not None, f"unreadable: {self.paths[idx]}"
        img = _letterbox(img)
        if self.augment:
            img = _apply_aug(img)
        _, _, r_idx, s_idx = self.samples[idx]
        return _to_tensor(img), r_idx, s_idx


def split_by_hand_id(
    samples: list[tuple], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[tuple], list[tuple]]:
    """Split samples into train/val by hand_id (first element of each tuple)."""
    by_hand: dict[str, list[tuple]] = {}
    for s in samples:
        by_hand.setdefault(s[0], []).append(s)
    hands = sorted(by_hand.keys())
    rng = random.Random(seed)
    rng.shuffle(hands)
    n_val = max(1, int(len(hands) * val_frac))
    val_hands = set(hands[:n_val])
    train, val = [], []
    for h, rows in by_hand.items():
        (val if h in val_hands else train).extend(rows)
    return train, val
