"""Tests for the classifier factory."""

from __future__ import annotations

import pytest
import torch

from str_action.models import build_model, count_parameters


def test_resnet18_forward() -> None:
    model = build_model("resnet18", num_classes=2)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)
    assert count_parameters(model) > 0


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValueError):
        build_model("alexnet", num_classes=2)


def test_mobilenet_v4_forward() -> None:
    pytest.importorskip("timm")
    model = build_model("mobilenet_v4", num_classes=2)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 2)
