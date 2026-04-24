"""CardCNN — shared backbone + RankHead (13) + SuitHead (4)."""
from __future__ import annotations

import torch
import torch.nn as nn

RANK_CLASSES = ["2", "3", "4", "5", "6", "7", "8", "9",
                "T", "J", "Q", "K", "A"]
SUIT_CLASSES = ["c", "d", "h", "s"]

# Input tensor shape: (B, 3, H=192, W=128).
# Design notes:
#   v1 (48x64, pool→1): rank_acc=0.35 — rank glyph ~5px, spatial info pooled away.
#   v2 (96x128, pool→4): rank_acc=0.89 — better but ceiling from letterbox padding
#     + limited capacity.
#   v3 (+cosine LR, +150 epochs): 0.90 — confirmed convergence, not overfitting.
#   v4 (192x128, pool→4, 4 conv blocks, stretch-resize): target 0.99.
_POOL_SIZE = 4
_LAST_CH = 128
_FEAT_DIM = _LAST_CH * _POOL_SIZE * _POOL_SIZE  # 2048


class CardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 96 × 64
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 48 × 32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 24 × 16
            nn.Conv2d(64, _LAST_CH, 3, padding=1), nn.BatchNorm2d(_LAST_CH),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                              # 12 × 8
            nn.AdaptiveAvgPool2d(_POOL_SIZE),                             # 4 × 4
            nn.Flatten(),
        )
        self.rank_head = nn.Linear(_FEAT_DIM, len(RANK_CLASSES))
        self.suit_head = nn.Linear(_FEAT_DIM, len(SUIT_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)
