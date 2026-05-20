"""CardDataset — loads labeled crops from data/cards*/{rank}/{suit}/*.png."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .model import RANK_CLASSES, SUIT_CLASSES

INPUT_H, INPUT_W = 192, 128
_RANK_TO_IDX = {r: i for i, r in enumerate(RANK_CLASSES)}
_SUIT_TO_IDX = {s: i for i, s in enumerate(SUIT_CLASSES)}
_FILENAME_RE = re.compile(r"^(?P<hand>[A-Za-z0-9]+)_(?P<src>hero|hero_masked|board)_(?P<slot>\d+)\.png$")


def _letterbox(img: np.ndarray, h: int = INPUT_H, w: int = INPUT_W) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def _apply_aug(img: np.ndarray) -> np.ndarray:
    # No translation — the rank glyph is tiny (~10px) and ±2px shifts
    # destroyed rank signal in v1 (rank acc 0.35). Brightness + mild blur
    # are safe because they don't move the glyph.
    scale = random.uniform(0.9, 1.1)
    img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.1, 0.5))
    return img


class CardDataset(Dataset):
    """Every crop under root/{rank}/{suit}/{hand}_{src}_{slot}.png."""

    def __init__(
        self,
        root: Path,
        augment: bool = True,
        hand_ids: set[str] | None = None,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.root = Path(root)
        self.augment = augment
        self.augment_fn = augment_fn
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
                    if hand_ids is not None and m.group("hand") not in hand_ids:
                        continue
                    self.paths.append(png)
                    self.samples.append((
                        m.group("hand"), f"{m.group('src')}_{m.group('slot')}",
                        _RANK_TO_IDX[rank_dir.name],
                        _SUIT_TO_IDX[suit_dir.name],
                    ))

    @classmethod
    def from_split_json(
        cls,
        root: Path,
        split_path: Path,
        bucket: str,
        augment: bool = False,
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "CardDataset":
        with Path(split_path).open() as fh:
            split = json.load(fh)
        if bucket not in split:
            raise ValueError(f"unknown split bucket: {bucket}")
        return cls(
            root,
            augment=augment,
            hand_ids=set(split[bucket]),
            augment_fn=augment_fn,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        img = cv2.imread(str(self.paths[idx]))
        assert img is not None, f"unreadable: {self.paths[idx]}"
        img = _letterbox(img)
        if self.augment:
            img = self.augment_fn(img) if self.augment_fn else _apply_aug(img)
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
