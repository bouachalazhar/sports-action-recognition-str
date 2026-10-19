"""Tests for ROI extraction on a synthetic video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from str_action.roi import green_chromaticity, optical_flow, roi_str


def write_synthetic_video(path: Path, frames: int = 8, size: int = 64) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (size, size)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, 1] = 180  # green field
        offset = 8 + index * 2
        frame[20:32, offset : offset + 8] = (0, 0, 220)  # moving object
        writer.write(frame)
    writer.release()


def test_roi_shapes(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    write_synthetic_video(video)
    field, players, roi = roi_str(str(video))
    assert field.shape == (64, 64)
    assert players.shape == (64, 64)
    assert roi.shape == (64, 64)
    assert roi.dtype == np.uint8


def test_optical_flow_rejects_empty_video(tmp_path: Path) -> None:
    empty = tmp_path / "empty.avi"
    empty.write_bytes(b"")
    with pytest.raises((ValueError, cv2.error)):
        optical_flow(str(empty))


def test_green_chromaticity_returns_masks(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    write_synthetic_video(video)
    field, players = green_chromaticity(str(video))
    assert field.max() > 0
    assert players.max() > 0
