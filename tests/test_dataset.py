"""Tests for the STR dataset and grouped splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from str_action.dataset import (
    STRImageDataset,
    discover_clips,
    grouped_split,
    label_names,
)


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def make_dataset(root: Path, clips_per_label: int = 3) -> Path:
    for split in ("train", "val"):
        for label, value in (("non_action", 10), ("action", 200)):
            for index in range(clips_per_label):
                clip = root / split / label / f"clip_{index}"
                write_image(clip / "str" / "0.png", value)
                write_image(clip / "str" / "1.png", value)
    return root


def test_discover_clips(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root)
    assert len(clips) == 12
    assert label_names(clips) == ["action", "non_action"]
    assert {clip.label for clip in clips} == {0, 1}
    assert all(len(clip.images) == 2 for clip in clips)


def test_discover_clips_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_clips(tmp_path / "missing")


def test_dataset_item_shapes(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root)
    dataset = STRImageDataset(clips, image_size=32)
    assert len(dataset) == 24
    image, label, clip_index = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == np.float32
    assert label == int(dataset.clip_labels[clip_index])


def test_grouped_split_keeps_clips_disjoint(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root)
    train, validation = grouped_split(clips, validation_fraction=0.34, seed=17)
    assert train
    assert validation
    assert {clip.clip_id for clip in train}.isdisjoint(
        {clip.clip_id for clip in validation}
    )


def test_grouped_split_is_deterministic(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root)
    first = grouped_split(clips, 0.34, seed=29)
    second = grouped_split(clips, 0.34, seed=29)
    assert [clip.clip_id for clip in first[0]] == [clip.clip_id for clip in second[0]]


def test_discover_clips_reuses_checkpoint_label_order(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root, label_order=["non_action", "action"])
    by_name = {clip.label_name: clip.label for clip in clips}
    assert by_name == {"non_action": 0, "action": 1}
    with pytest.raises(ValueError):
        discover_clips(root, label_order=["action"])


def test_grouped_split_rejects_single_source_group(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for label in ("action", "non_action"):
        clip = root / "train" / label / "match_1" / "clip"
        write_image(clip / "str" / "0.png", 128)
    clips = discover_clips(root)
    with pytest.raises(ValueError, match="only one source group"):
        grouped_split(clips)


def test_grouped_split_covers_every_label_with_distinct_groups(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    for label, prefix in (("action", "a"), ("non_action", "b")):
        for index in range(3):
            clip = root / "train" / label / f"{prefix}{index}" / "clip"
            write_image(clip / "str" / "0.png", 128)

    clips = discover_clips(root)
    train, validation = grouped_split(clips, validation_fraction=0.34, seed=7)

    assert {clip.group for clip in train}.isdisjoint(
        {clip.group for clip in validation}
    )
    assert {clip.label_name for clip in train} == {"action", "non_action"}
    assert {clip.label_name for clip in validation} == {"action", "non_action"}


def test_grouped_split_rejects_invalid_fraction(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    clips = discover_clips(root)
    with pytest.raises(ValueError):
        grouped_split(clips, validation_fraction=1.0)
