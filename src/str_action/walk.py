"""Random walk sampling over a video volume.

A video is a tensor of shape ``(T, H, W, C)``. A random walk starts at a pixel
and moves through the 8-connected neighbourhood for a fixed number of steps.
The pixels visited by one walk, across all frames, form a planar RGB image
called an STR (spatio-temporal random walk representation).

The implementation is vectorised over walks and uses Numba for the coordinate
generation, which is the only loop that cannot be expressed as a tensor
operation.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

# 8-connected neighbourhood, in (row, column) order.
_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def shape_offsets(
    shape_type: str, shape_params: dict[str, Any], dilation: int
) -> list[tuple[int, int]]:
    """Return the (dy, dx) offsets of a sampling shape.

    Args:
        shape_type: One of ``point``, ``cross``, ``diagonal_cross``, ``square``,
            ``rectangle``, ``line_h``, ``line_v``.
        shape_params: Shape parameters (``arm_length``, ``side``, ``height``,
            ``width``, ``length``).
        dilation: Distance between two consecutive samples of the shape.
    """
    if shape_type == "point":
        return [(0, 0)]

    if shape_type == "cross":
        arm = int(shape_params.get("arm_length", 1))
        if arm < 1:
            raise ValueError("cross arm_length must be >= 1")
        offsets = [(0, 0)]
        for i in range(1, arm + 1):
            offsets.extend(
                [
                    (-i * dilation, 0),
                    (i * dilation, 0),
                    (0, -i * dilation),
                    (0, i * dilation),
                ]
            )
        return offsets

    if shape_type == "diagonal_cross":
        arm = int(shape_params.get("arm_length", 1))
        if arm < 1:
            raise ValueError("diagonal_cross arm_length must be >= 1")
        offsets = [(0, 0)]
        for i in range(1, arm + 1):
            offsets.extend(
                [
                    (-i * dilation, -i * dilation),
                    (-i * dilation, i * dilation),
                    (i * dilation, -i * dilation),
                    (i * dilation, i * dilation),
                ]
            )
        return offsets

    if shape_type == "square":
        side = int(shape_params.get("side", 2))
        if side < 1:
            raise ValueError("square side must be >= 1")
        return [
            (dy * dilation, dx * dilation) for dy in range(side) for dx in range(side)
        ]

    if shape_type == "rectangle":
        height = int(shape_params.get("height", 2))
        width = int(shape_params.get("width", 1))
        if height < 1 or width < 1:
            raise ValueError("rectangle dimensions must be >= 1")
        return [
            (dy * dilation, dx * dilation)
            for dy in range(height)
            for dx in range(width)
        ]

    if shape_type == "line_h":
        length = int(shape_params.get("length", 3))
        if length < 1:
            raise ValueError("line_h length must be >= 1")
        half = length // 2
        return [(0, dx * dilation) for dx in range(-half, length - half)]

    if shape_type == "line_v":
        length = int(shape_params.get("length", 3))
        if length < 1:
            raise ValueError("line_v length must be >= 1")
        half = length // 2
        return [(dy * dilation, 0) for dy in range(-half, length - half)]

    raise ValueError(f"unknown shape_type: {shape_type}")


def temporal_pad(video: np.ndarray, target_length: int, mode: str) -> np.ndarray:
    """Pad or truncate a video to ``target_length`` frames.

    Args:
        video: Array of shape ``(T, H, W, C)``.
        target_length: Requested number of frames.
        mode: ``duplicate``, ``bounce`` or ``loop``.
    """
    frames = video.shape[0]
    if frames >= target_length:
        return video[:target_length]
    if frames == 0:
        raise ValueError("cannot pad an empty video")

    if mode == "bounce" and frames == 1:
        return np.repeat(video, target_length, axis=0)

    if mode == "duplicate":
        repeats = np.ones(frames, dtype=int) * (target_length // frames)
        repeats[: target_length % frames] += 1
        indices = np.repeat(np.arange(frames), repeats)
        return video[indices]

    if mode == "bounce":
        indices: list[int] = []
        current = 0
        direction = 1
        while len(indices) < target_length:
            indices.append(current)
            if current + direction >= frames or current + direction < 0:
                direction *= -1
                if frames > 1:
                    current += direction
            else:
                current += direction
        return video[indices]

    if mode == "loop":
        return video[[i % frames for i in range(target_length)]]

    raise ValueError(f"unknown temporal_pad mode: {mode}")


TEMPORAL_POLICIES = ("preserve", "min-half-walk")


def resolve_temporal_target_length(
    total_frames: int, walk_length: int, policy: str
) -> int:
    """Return the frame count used to build one STR image.

    ``preserve`` keeps the decoded temporal axis unchanged. ``min-half-walk``
    is the legacy convenience policy that pads very short clips to at least half
    the walk length; it is not the manuscript protocol when ``L=224`` and a
    three-second clip has 90 frames.
    """
    if total_frames <= 0:
        raise ValueError("a clip must contain at least one frame")
    if walk_length <= 0:
        raise ValueError("walk_length must be positive")
    if policy == "preserve":
        return total_frames
    if policy == "min-half-walk":
        return max(total_frames, walk_length // 2 + 1)
    raise ValueError(f"unknown temporal policy: {policy}")


def aggregate_pixels(pixels: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate the pixels sampled by one walk into one value per frame.

    Args:
        pixels: Array of shape ``(T, N_pixels, C)``.
        mode: ``mean``, ``max``, ``stack``, ``sobel`` or ``laplacian``.
    """
    if mode == "mean":
        return np.mean(pixels, axis=1).astype(np.uint8)
    if mode == "max":
        return np.max(pixels, axis=1).astype(np.uint8)
    if mode == "stack":
        time, npix, channels = pixels.shape
        return pixels.reshape(time, npix * channels)
    if mode == "sobel":
        gradient = np.std(pixels.astype(np.float32), axis=1) * 2
        return np.clip(gradient, 0, 255).astype(np.uint8)
    if mode == "laplacian":
        center = pixels[:, 0, :].astype(np.float32)
        mean = np.mean(pixels.astype(np.float32), axis=1)
        laplacian = np.abs(center - mean) * 4
        return np.clip(laplacian, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown aggregate mode: {mode}")


@numba.njit
def _walk_coordinates(
    walk_count: int,
    walk_length: int,
    height: int,
    width: int,
    stride: int,
    seed: int,
    start_mask: np.ndarray | None,
    walk_mask: np.ndarray | None,
) -> np.ndarray:
    if seed != -1:
        np.random.seed(seed)

    valid_rows = np.empty(0, dtype=np.int64)
    valid_columns = np.empty(0, dtype=np.int64)
    if start_mask is not None:
        valid_rows, valid_columns = np.nonzero(start_mask)
        if valid_rows.size == 0:
            raise ValueError("start_mask contains no valid starting pixel")

    coordinates = np.zeros((walk_count, walk_length, 2), dtype=np.int32)
    deltas = np.array(
        [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=np.int32,
    )

    for index in range(walk_count):
        if start_mask is not None:
            chosen_start = np.random.randint(0, valid_rows.size)
            row = valid_rows[chosen_start]
            column = valid_columns[chosen_start]
        else:
            row = np.random.randint(0, height)
            column = np.random.randint(0, width)

        for step in range(walk_length):
            coordinates[index, step, 0] = row
            coordinates[index, step, 1] = column

            if step < walk_length - 1:
                valid_count = 0
                neighbours = np.zeros((8, 2), dtype=np.int32)

                for delta in range(8):
                    next_row = row + deltas[delta, 0] * stride
                    next_column = column + deltas[delta, 1] * stride
                    if not (0 <= next_row < height and 0 <= next_column < width):
                        continue
                    if walk_mask is not None and walk_mask[next_row, next_column] == 0:
                        continue
                    neighbours[valid_count, 0] = next_row
                    neighbours[valid_count, 1] = next_column
                    valid_count += 1

                if valid_count > 0:
                    chosen = np.random.randint(0, valid_count)
                    row = neighbours[chosen, 0]
                    column = neighbours[chosen, 1]
                # Otherwise the walk self-loops on the current pixel, as in
                # the manuscript's absorbing-boundary rule.

    return coordinates


def random_walk_frames(
    video: np.ndarray,
    walk_count: int,
    walk_length: int,
    *,
    temporal_pad_mode: str = "bounce",
    temporal_target_length: int | None = None,
    shape_type: str = "point",
    shape_params: dict[str, Any] | None = None,
    aggregate_mode: str = "mean",
    walk_stride: int = 1,
    shape_dilation: int = 1,
    start_mask: np.ndarray | None = None,
    walk_mask: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build STR images from a video volume.

    Args:
        video: Array of shape ``(T, H, W, C)``, dtype ``uint8``.
        walk_count: Number of random walks (number of STR images).
        walk_length: Number of steps per walk, which sets the image width.
        temporal_pad_mode: Padding mode when the video is shorter than
            ``temporal_target_length``.
        temporal_target_length: Optional frame count to pad or truncate to.
        shape_type: Sampling shape, see :func:`shape_offsets`.
        shape_params: Sampling shape parameters.
        aggregate_mode: Pixel aggregation mode, see :func:`aggregate_pixels`.
        walk_stride: Step size of the walk, in pixels.
        shape_dilation: Sample spacing inside the sampling shape.
        start_mask: Optional binary mask restricting the first pixel of each
            walk. Defaults to ``walk_mask`` when only that mask is supplied.
        walk_mask: Optional binary mask restricting every pixel of the walk.
            This is the manuscript's active support: a candidate neighbour is
            accepted only when it falls inside the mask. When no neighbour is
            available the walk self-loops on its current pixel.
        seed: Optional seed for reproducible walks.

    Returns:
        A tuple ``(images, points)`` where ``images`` has shape
        ``(walk_count, T, walk_length, C)`` and ``points`` lists every visited
        pixel as ``(row, column)``.
    """
    if shape_params is None:
        shape_params = {}
    if temporal_target_length is not None:
        video = temporal_pad(video, temporal_target_length, temporal_pad_mode)

    _frames, height, width, _channels = video.shape

    if walk_mask is not None:
        walk_mask = np.asarray(walk_mask)
        if walk_mask.shape != (height, width):
            raise ValueError(
                "walk_mask must have the same height and width as the video"
            )
        if not np.any(walk_mask):
            raise ValueError("walk_mask contains no valid pixel")

    if start_mask is not None:
        start_mask = np.asarray(start_mask)
        if start_mask.shape != (height, width):
            raise ValueError(
                "start_mask must have the same height and width as the video"
            )
        if walk_mask is not None:
            start_mask = np.logical_and(start_mask, walk_mask).astype(np.uint8)
        if not np.any(start_mask):
            raise ValueError("start_mask contains no valid starting pixel")
    elif walk_mask is not None:
        start_mask = walk_mask

    offsets = shape_offsets(shape_type, shape_params, shape_dilation)
    pixel_count = len(offsets)
    max_dy = max(abs(dy) for dy, _dx in offsets)
    max_dx = max(abs(dx) for _dy, dx in offsets)
    if max_dy * 2 >= height or max_dx * 2 >= width:
        raise ValueError(f"shape parameters are too large for {height}x{width} images")

    coordinates = _walk_coordinates(
        walk_count,
        walk_length,
        height,
        width,
        walk_stride,
        seed if seed is not None else -1,
        start_mask,
        walk_mask,
    )

    offsets_array = np.array(offsets, dtype=np.int32).reshape(1, 1, pixel_count, 2)
    centers = coordinates.reshape(walk_count, walk_length, 1, 2)
    pixels = centers + offsets_array

    rows = pixels[..., 0]
    columns = pixels[..., 1]
    valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    rows = np.where(valid, rows, centers[..., 0, 0][..., np.newaxis])
    columns = np.where(valid, columns, centers[..., 0, 1][..., np.newaxis])

    values = video[:, rows, columns, :]
    values = values.transpose(1, 0, 2, 3, 4)
    walks, frames, steps, npix, channels = values.shape
    flat = values.reshape(walks * frames * steps, npix, channels)
    aggregated = aggregate_pixels(flat, aggregate_mode)
    images = aggregated.reshape(walks, frames, steps, aggregated.shape[-1])

    points = [(int(row), int(column)) for walk in coordinates for row, column in walk]
    return images, points


__all__ = [
    "TEMPORAL_POLICIES",
    "aggregate_pixels",
    "random_walk_frames",
    "resolve_temporal_target_length",
    "shape_offsets",
    "temporal_pad",
]
