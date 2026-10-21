"""Tests for the STR builder and its on-disk layout."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np

from str_action.build import build_split
from str_action.dataset import discover_clips, grouped_split


def _write_video(path: Path, *, frames: int = 8, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            color = np.array(
                [(index * 31) % 255, (index * 67) % 255, (index * 97) % 255],
                dtype=np.uint8,
            )
            image = np.broadcast_to(color, (size, size, 3)).copy()
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_build_preserves_temporal_axis_and_group(tmp_path: Path) -> None:
    source = tmp_path / "videos"
    _write_video(source / "action" / "match_a" / "clip.mp4")
    destination = tmp_path / "str"

    written = build_split(
        source,
        destination,
        walks=2,
        walk_length=8,
        roi_mode="none",
        image_size=16,
        seed=3,
    )

    assert written == 1
    clip_dir = destination / "action" / "match_a" / "clip"
    image = cv2.imread(str(clip_dir / "str" / "0.png"))
    assert image is not None
    assert image.shape == (16, 16, 3)
    # A middle-frame implementation collapses all temporal rows to one colour.
    assert len(np.unique(image.reshape(16, -1), axis=0)) > 1

    clips = discover_clips(destination)
    assert len(clips) == 1
    assert clips[0].label_name == "action"
    assert clips[0].group == "match_a"


def test_grouped_split_keeps_match_groups_together(tmp_path: Path) -> None:
    root = tmp_path / "str"
    for label in ("action", "no_action"):
        for match in ("m1", "m2", "m3"):
            clip = root / "train" / label / match / "clip"
            (clip / "str").mkdir(parents=True)
            cv2.imwrite(str(clip / "str" / "0.png"), np.zeros((8, 8, 3), np.uint8))

    clips = discover_clips(root)
    train, validation = grouped_split(clips, validation_fraction=0.34, seed=7)

    assert {clip.group for clip in train}.isdisjoint(
        {clip.group for clip in validation}
    )
    assert {clip.label_name for clip in train} == {"action", "no_action"}
    assert {clip.label_name for clip in validation} == {"action", "no_action"}


def test_points_are_opt_in(tmp_path: Path) -> None:
    source = tmp_path / "videos"
    _write_video(source / "action" / "match_a" / "clip.mp4")
    destination = tmp_path / "str"

    build_split(
        source,
        destination,
        walks=2,
        walk_length=8,
        roi_mode="none",
        image_size=16,
        seed=3,
    )
    clip_dir = destination / "action" / "match_a" / "clip"
    assert not (clip_dir / "points.json").exists()

    build_split(
        source,
        destination,
        walks=2,
        walk_length=8,
        roi_mode="none",
        image_size=16,
        seed=3,
        save_points=True,
    )
    points = (clip_dir / "points.json").read_text(encoding="utf-8")
    assert points.startswith("[[")
