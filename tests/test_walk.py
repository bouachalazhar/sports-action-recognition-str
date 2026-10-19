"""Tests for the random walk sampler."""

from __future__ import annotations

import numpy as np
import pytest

from str_action.walk import (
    aggregate_pixels,
    random_walk_frames,
    shape_offsets,
    temporal_pad,
)


def test_shape_offsets_point() -> None:
    assert shape_offsets("point", {}, 1) == [(0, 0)]


def test_shape_offsets_cross() -> None:
    offsets = shape_offsets("cross", {"arm_length": 2}, 1)
    assert (0, 0) in offsets
    assert (2, 0) in offsets
    assert (0, -2) in offsets
    assert len(offsets) == 9


def test_shape_offsets_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError):
        shape_offsets("hexagon", {}, 1)


def test_temporal_pad_duplicate_and_loop() -> None:
    video = np.arange(3 * 2 * 2 * 1, dtype=np.uint8).reshape(3, 2, 2, 1)
    duplicated = temporal_pad(video, 6, "duplicate")
    assert duplicated.shape[0] == 6
    looped = temporal_pad(video, 5, "loop")
    assert looped.shape[0] == 5
    assert np.array_equal(looped[0], video[0])
    assert np.array_equal(looped[3], video[0])


def test_temporal_pad_truncates() -> None:
    video = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    assert temporal_pad(video, 4, "bounce").shape[0] == 4


def test_temporal_pad_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        temporal_pad(np.zeros((2, 4, 4, 3), dtype=np.uint8), 4, "mirror")


def test_aggregate_pixels_modes() -> None:
    pixels = np.array([[[10, 20, 30], [30, 20, 10]]], dtype=np.uint8)
    assert aggregate_pixels(pixels, "mean").shape == (1, 3)
    assert aggregate_pixels(pixels, "max").shape == (1, 3)
    assert aggregate_pixels(pixels, "stack").shape == (1, 6)
    assert aggregate_pixels(pixels, "sobel").shape == (1, 3)
    assert aggregate_pixels(pixels, "laplacian").shape == (1, 3)


def test_random_walk_is_seeded() -> None:
    rng = np.random.default_rng(0)
    video = rng.integers(0, 255, size=(8, 32, 32, 3), dtype=np.uint8)
    first, points_a = random_walk_frames(video, 4, 16, seed=7)
    second, points_b = random_walk_frames(video, 4, 16, seed=7)
    assert first.shape == (4, 8, 16, 3)
    assert np.array_equal(first, second)
    assert points_a == points_b


def test_random_walk_rejects_oversized_shape() -> None:
    video = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        random_walk_frames(
            video, 1, 4, shape_type="cross", shape_params={"arm_length": 10}
        )
