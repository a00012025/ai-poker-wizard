"""CardCNN — shared backbone + RankHead (13) + SuitHead (4)."""
from __future__ import annotations

import torch
import torch.nn as nn

RANK_CLASSES = ["2", "3", "4", "5", "6", "7", "8", "9",
                "T", "J", "Q", "K", "A"]
SUIT_CLASSES = ["c", "d", "h", "s"]

# Input tensor shape: (B, 3, H=48, W=64)


class CardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.rank_head = nn.Linear(64, len(RANK_CLASSES))
        self.suit_head = nn.Linear(64, len(SUIT_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)
