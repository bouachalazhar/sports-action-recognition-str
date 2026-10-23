"""Two-dimensional classifier factory.

The paper compares a light MobileNet V4 classifier against heavier 3D
architectures. Every model here consumes one STR image and returns one logit
vector, which keeps the per-image cost constant across the family.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torchvision
from torch import nn

try:  # timm is an optional heavy dependency for the MobileNet V4 variants only.
    import timm
except ImportError:  # pragma: no cover - exercised only in minimal installs
    timm = None  # type: ignore[assignment]

RESNET18 = "resnet18"
MOBILENET_V4 = "mobilenet_v4"
EFFICIENTNET_V2 = "efficientnet_v2_s"
CONVNEXT = "convnext_tiny"
SWIN_V2 = "swin_v2_t"
VIT_B16 = "vit_b_16"

SUPPORTED_MODELS = (
    RESNET18,
    MOBILENET_V4,
    EFFICIENTNET_V2,
    CONVNEXT,
    SWIN_V2,
    VIT_B16,
)


def _replace_head(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Replace the classification head of a torchvision backbone in place.

    The head attribute differs per family (``fc``, ``classifier``,
    ``head``, ``heads.head``) and torchvision stubs vary across releases,
    so the backbone is intentionally accessed at a single documented
    dynamic boundary.
    """
    backbone = cast(Any, model)
    if name == RESNET18:
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    elif name in (EFFICIENTNET_V2, CONVNEXT):
        classifier = backbone.classifier
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
    elif name == SWIN_V2:
        backbone.head = nn.Linear(backbone.head.in_features, num_classes)
    elif name == VIT_B16:
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, num_classes)
    return model


def build_model(
    name: str = RESNET18,
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a 2D classifier by name.

    Args:
        name: One of :data:`SUPPORTED_MODELS`.
        num_classes: Number of output logits.
        pretrained: Load ImageNet weights when available.
        dropout: Optional dropout before the classification head.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {SUPPORTED_MODELS}")

    if name == RESNET18:
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model = _replace_head(model, name, num_classes)
    elif name == MOBILENET_V4:
        if timm is None:
            raise ImportError("timm is required for the MobileNet V4 variants")
        model = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    else:
        factory = {
            EFFICIENTNET_V2: torchvision.models.efficientnet_v2_s,
            CONVNEXT: torchvision.models.convnext_tiny,
            SWIN_V2: torchvision.models.swin_v2_t,
            VIT_B16: torchvision.models.vit_b_16,
        }[name]
        model = factory(weights="DEFAULT" if pretrained else None)
        model = _replace_head(model, name, num_classes)

    if dropout > 0:
        model = nn.Sequential(model, nn.Dropout(dropout))
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


@torch.no_grad()
def predict_probabilities(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Return softmax probabilities for a batch of STR images."""
    model.eval()
    return torch.softmax(model(batch), dim=1)


__all__ = [
    "CONVNEXT",
    "EFFICIENTNET_V2",
    "MOBILENET_V4",
    "RESNET18",
    "SUPPORTED_MODELS",
    "SWIN_V2",
    "VIT_B16",
    "build_model",
    "count_parameters",
    "predict_probabilities",
]
