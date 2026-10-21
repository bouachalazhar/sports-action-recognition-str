"""Tests for the random walk sampler."""

from __future__ import annotations

import numpy as np
import pytest

from str_action.walk import (
    aggregate_pixels,
    random_walk_frames,
    resolve_temporal_target_length,
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


def test_random_walk_starts_inside_mask() -> None:
    video = np.zeros((6, 16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    images, points = random_walk_frames(video, 8, 8, start_mask=mask, seed=5)
    assert images.shape == (8, 6, 8, 3)
    starts = points[::8]
    assert all(8 <= row < 12 and 8 <= column < 12 for row, column in starts)


def test_random_walk_stays_inside_mask_at_every_step() -> None:
    video = np.zeros((6, 16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 4:10] = 255
    _images, points = random_walk_frames(video, 16, 24, walk_mask=mask, seed=11)
    assert all(4 <= row < 10 and 4 <= column < 10 for row, column in points)


def test_random_walk_self_loops_on_a_single_pixel_mask() -> None:
    video = np.zeros((6, 16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[5, 5] = 255
    _images, points = random_walk_frames(video, 4, 12, walk_mask=mask, seed=11)
    assert points == [(5, 5)] * 48


def test_temporal_pad_bounce_survives_one_frame() -> None:
    video = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    padded = temporal_pad(video, 7, "bounce")
    assert padded.shape[0] == 7
    assert np.array_equal(padded, np.repeat(video, 7, axis=0))


def test_random_walk_rejects_empty_mask() -> None:
    video = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        random_walk_frames(video, 2, 4, start_mask=np.zeros((16, 16)), seed=5)


def test_random_walk_rejects_oversized_shape() -> None:
    video = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        random_walk_frames(
            video, 1, 4, shape_type="cross", shape_params={"arm_length": 10}
        )


def test_preserve_policy_keeps_the_decoded_frame_count() -> None:
    assert resolve_temporal_target_length(90, 224, "preserve") == 90


def test_min_half_walk_is_an_explicit_legacy_policy() -> None:
    assert resolve_temporal_target_length(90, 224, "min-half-walk") == 113
    assert resolve_temporal_target_length(400, 224, "min-half-walk") == 400


def test_temporal_policy_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unknown temporal policy"):
        resolve_temporal_target_length(90, 224, "stretch")
