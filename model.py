from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MultiImageLivenessNet(nn.Module):
    """Classify a set of images as human / fake.

    Input shape: [B, N, C, H, W]
    Output shape: [B, 2]
    """

    def __init__(self, pretrained: bool = False, freeze_backbone: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # -> [B, 512, 1, 1]

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input with 5 dims [B,N,C,H,W], got {tuple(x.shape)}")

        b, n, c, h, w = x.shape
        x = x.view(b * n, c, h, w)
        feats = self.feature_extractor(x)           # [B*N, 512, 1, 1]
        feats = feats.flatten(1)                    # [B*N, 512]
        feats = feats.view(b, n, 512)               # [B, N, 512]
        pooled = feats.mean(dim=1)                  # [B, 512]
        logits = self.classifier(pooled)
        return logits
