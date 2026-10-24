"""Model registry: every compared network is created by name with a 2-class output (NORMAL, PNEUMONIA)."""

from __future__ import annotations

import timm
import torch
from torch import nn

NUM_CLASSES = 2

# Exact timm weight tags (ImageNet-1k) so a future timm release cannot silently change the pretrained weights.
TIMM_MODELS = {
    "vgg16": "vgg16.tv_in1k",
    "resnet50": "resnet50.a1_in1k",
    "densenet121": "densenet121.ra_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "mobilenetv3_large": "mobilenetv3_large_100.ra_in1k",
}


class BaselineCNN(nn.Module):
    """Small CNN trained from scratch: 4 × [Conv3×3 → BatchNorm → ReLU → MaxPool], global average pooling, dropout, linear."""

    def __init__(self, num_classes: int = NUM_CLASSES, channels: tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_channels, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


MODEL_NAMES = ("baseline_cnn", *TIMM_MODELS)


def create_model(name: str, pretrained: bool = True, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Build a model by its registry name.

    timm models get ImageNet weights (when `pretrained`) and a new `num_classes` head; the baseline is always
    trained from scratch.
    """
    if name == "baseline_cnn":
        return BaselineCNN(num_classes)
    if name in TIMM_MODELS:
        return timm.create_model(TIMM_MODELS[name], pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
