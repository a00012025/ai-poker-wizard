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


class CardCNNv2(nn.Module):
    def __init__(self):
        super().__init__()

        def block(ci: int, co: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 192),
            block(192, 256),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.3),
        )
        feat = 256 * 4 * 4
        self.rank_head = nn.Linear(feat, len(RANK_CLASSES))
        self.suit_head = nn.Linear(feat, len(SUIT_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.rank_head(f), self.suit_head(f)


class CardMobileNetV3Small(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = mobilenet_v3_small(weights=weights)
        self.register_buffer(
            "_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[0].in_features
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
        )
        self.rank_head = nn.Linear(512, len(RANK_CLASSES))
        self.suit_head = nn.Linear(512, len(SUIT_CLASSES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self._mean) / self._std
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        f = self.shared(f)
        return self.rank_head(f), self.suit_head(f)
