"""STR image datasets, label mapping, and grouped splits."""

from __future__ import annotations

import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLIT_NAMES = {"train", "training", "val", "validation", "test"}
_GROUP_SUFFIX = re.compile(r"[_-]?\d+$")


@dataclass(frozen=True)
class ClipSample:
    """One clip, its source group and its pre-computed STR images."""

    clip_id: str
    label: int
    label_name: str
    group: str
    images: tuple[Path, ...]


def _layout_parts(root: Path, directory: Path) -> tuple[str, str]:
    """Return ``(label_name, group)`` for one clip directory.

    Accepted layouts are ``<root>/<label>/<group>/<clip>/str`` and
    ``<root>/<split>/<label>/<group>/<clip>/str``. A flat clip name is used as
    its own group only when no group directory is present.
    """
    parts = directory.relative_to(root).parts
    offset = 1 if parts and parts[0].lower() in SPLIT_NAMES else 0
    if len(parts) <= offset:
        raise ValueError(f"cannot infer a label from {directory}")
    label_name = parts[offset]
    group_parts = parts[offset + 1 : -1]
    group = "/".join(group_parts) if group_parts else directory.name
    if group == label_name:
        group = _GROUP_SUFFIX.sub("", directory.name) or directory.name
    return label_name, group


def discover_clips(
    root: Path, label_order: list[str] | None = None
) -> list[ClipSample]:
    """Scan ``root/<label>/<group>/<clip>/str``.

    ``label_order`` lets evaluation reuse the class order stored in a
    checkpoint. Unknown labels are rejected instead of being silently remapped.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    directories = sorted({path.parent for path in root.rglob("str") if path.is_dir()})
    if not directories:
        raise FileNotFoundError(f"no STR directory found under {root}")

    entries: list[tuple[Path, str, str, tuple[Path, ...]]] = []
    for directory in directories:
        images = tuple(
            sorted(
                path
                for path in (directory / "str").iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
        )
        if not images:
            continue
        label_name, group = _layout_parts(root, directory)
        entries.append((directory, label_name, group, images))

    labels = sorted({label_name for _directory, label_name, _group, _images in entries})
    if label_order is not None:
        unknown = set(labels) - set(label_order)
        if unknown:
            raise ValueError(
                f"STR labels absent from the checkpoint: {sorted(unknown)}"
            )
        missing = set(label_order) - set(labels)
        if missing:
            warnings.warn(
                "checkpoint labels absent from the STR data: "
                f"{sorted(missing)}; their recall and F1 are reported as zero",
                RuntimeWarning,
                stacklevel=2,
            )
        ordered_labels = list(label_order)
    else:
        ordered_labels = labels
    label_index = {name: index for index, name in enumerate(ordered_labels)}

    clips: list[ClipSample] = []
    for directory, label_name, group, images in entries:
        clips.append(
            ClipSample(
                clip_id=str(directory.relative_to(root)),
                label=label_index[label_name],
                label_name=label_name,
                group=group,
                images=images,
            )
        )

    if not clips:
        raise FileNotFoundError(f"no STR image found under {root}")
    return clips


def label_names(clips: list[ClipSample]) -> list[str]:
    """Return label names ordered by label index."""
    mapping = {clip.label: clip.label_name for clip in clips}
    return [mapping[index] for index in sorted(mapping)]


def grouped_split(
    clips: list[ClipSample],
    validation_fraction: float = 0.1,
    seed: int = 17,
) -> tuple[list[ClipSample], list[ClipSample]]:
    """Split globally by source group, without group leakage.

    A group is assigned to exactly one partition, even when it contains clips
    from several labels. The split also keeps every label in both partitions;
    if the requested fraction is too small to satisfy that coverage, the
    validation set grows to the minimum feasible size. A group that is the only
    source for a label cannot be split without leakage, so it raises instead of
    silently producing a fake validation set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if not clips:
        raise ValueError("clip list is empty")

    groups = sorted({clip.group for clip in clips})
    labels = sorted({clip.label for clip in clips})
    group_labels = {
        group: {clip.label for clip in clips if clip.group == group} for group in groups
    }
    for label in labels:
        label_groups = [group for group in groups if label in group_labels[group]]
        if len(label_groups) < 2:
            raise ValueError(
                f"label {label} has only one source group; provide at least two "
                "groups to build a leak-free train/validation split"
            )

    generator = random.Random(seed)
    shuffled_groups = list(groups)
    generator.shuffle(shuffled_groups)
    shuffled_labels = list(labels)
    generator.shuffle(shuffled_labels)

    target = max(1, int(round(len(groups) * validation_fraction)))
    target = min(target, len(groups) - 1)
    selected: set[str] = set()

    def train_covers_all_labels() -> bool:
        train_labels = {clip.label for clip in clips if clip.group not in selected}
        return set(labels) <= train_labels

    # First make sure that validation covers every label. A shared group can
    # cover several labels at once, so this may select fewer than len(labels).
    for label in shuffled_labels:
        if any(label in group_labels[group] for group in selected):
            continue
        uncovered = set(labels) - {
            covered for group in selected for covered in group_labels[group]
        }
        candidates = [
            group
            for group in shuffled_groups
            if label in group_labels[group] and group not in selected
        ]
        generator.shuffle(candidates)
        candidates.sort(key=lambda group: -len(group_labels[group] & uncovered))
        for group in candidates:
            selected.add(group)
            if train_covers_all_labels():
                break
            selected.remove(group)
        else:
            raise ValueError(
                f"cannot cover label {label} in both partitions; "
                "provide more source groups"
            )

    # Fill the validation set towards the requested fraction, but never remove
    # a label from the training partition.
    for group in shuffled_groups:
        if len(selected) >= target:
            break
        if group in selected:
            continue
        selected.add(group)
        if not train_covers_all_labels():
            selected.remove(group)

    validation = [clip for clip in clips if clip.group in selected]
    train = [clip for clip in clips if clip.group not in selected]
    if not train or not validation:
        raise ValueError("the requested split produced an empty partition")
    if {clip.label for clip in train} != set(labels):
        raise ValueError("grouped split removed a label from training")
    if {clip.label for clip in validation} != set(labels):
        raise ValueError("grouped split could not cover every label in validation")
    return sorted(train, key=lambda clip: clip.clip_id), sorted(
        validation, key=lambda clip: clip.clip_id
    )


class STRImageDataset(Dataset[tuple[np.ndarray, int, int]]):
    """One item per STR image.

    Each sample returns the image, its clip label and the clip index, so that
    evaluation can aggregate predictions at the clip level.
    """

    def __init__(
        self,
        clips: list[ClipSample],
        image_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

        self.items: list[tuple[Path, int, int]] = []
        for clip_index, clip in enumerate(clips):
            for image in clip.images:
                self.items.append((image, clip.label, clip_index))
        if not self.items:
            raise ValueError("dataset is empty")

        self.clip_labels = np.asarray([clip.label for clip in clips], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int, int]:
        path, label, clip_index = self.items[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        array = (array - self.mean) / self.std
        return array, label, clip_index


__all__ = [
    "ClipSample",
    "STRImageDataset",
    "discover_clips",
    "grouped_split",
    "label_names",
]
