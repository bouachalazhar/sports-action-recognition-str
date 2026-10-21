"""Tests for batched STR inference."""

from __future__ import annotations

import numpy as np
import torch

from str_action.infer import predict_probabilities


class _MeanModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        means = batch.mean(dim=(1, 2, 3))
        return torch.stack((means, -means), dim=1)


def test_predict_probabilities_uses_bounded_batches() -> None:
    generator = np.random.default_rng(3)
    images = generator.integers(0, 256, size=(5, 8, 8, 3), dtype=np.uint8)
    model = _MeanModel()

    probabilities = predict_probabilities(
        images,
        model,
        image_size=8,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert probabilities.shape == (5, 2)
    assert model.calls == 3
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    reference = predict_probabilities(
        images,
        _MeanModel(),
        image_size=8,
        batch_size=len(images),
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(probabilities, reference, rtol=1e-6, atol=1e-7)


def test_predict_probabilities_rejects_non_positive_batch_size() -> None:
    model = _MeanModel()
    try:
        predict_probabilities(
            np.zeros((1, 4, 4, 3), dtype=np.uint8),
            model,
            image_size=4,
            batch_size=0,
            device=torch.device("cpu"),
        )
    except ValueError:
        return
    raise AssertionError("a non-positive batch size should raise")
