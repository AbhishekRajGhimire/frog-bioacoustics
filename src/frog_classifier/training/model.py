"""ResNet-18 for one-channel spectrogram images."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


ARCHITECTURE = "resnet18"
PRETRAINED_WEIGHTS = "IMAGENET1K_V1"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SpectrogramResNet(nn.Module):
    """Takes images shaped (batch, 1, mels, frames) in [0, 1]; returns one frog logit per image."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        # The pretrained filters expect three ImageNet-normalized colour channels.
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError(f"expected images shaped (batch, 1, mels, frames), got {tuple(images.shape)}")
        colour = images.expand(-1, 3, -1, -1)
        return self.backbone((colour - self.mean) / self.std).squeeze(1)


def to_model_input(pixels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """8-bit pixels shaped (batch, mels, frames) as float (batch, 1, mels, frames) in [0, 1]."""
    tensor = torch.as_tensor(pixels)
    if tensor.dtype != torch.uint8 or tensor.ndim != 3:
        raise ValueError("expected uint8 pixels shaped (batch, mels, frames)")
    return tensor.unsqueeze(1).float().div_(255.0)
