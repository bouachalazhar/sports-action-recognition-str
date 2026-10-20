"""STR image datasets, label mapping, and grouped splits."""

from __future__ import annotations

import random
import re
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
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[list[ClipSample], list[ClipSample]]:
    """Split by source group, stratified by label, without group leakage.

    A flat tree where every clip name is its own group is supported for small
    demonstrations. A tree with one group per label cannot be split without
    leaking that group, so it raises instead of silently producing a fake
    validation set.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

    by_label: dict[int, dict[str, list[ClipSample]]] = {}
    for clip in clips:
        by_label.setdefault(clip.label, {}).setdefault(clip.group, []).append(clip)

    generator = random.Random(seed)
    train: list[ClipSample] = []
    validation: list[ClipSample] = []
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        if len(groups) < 2:
            raise ValueError(
                f"label {label} has only one source group; provide at least two "
                "groups to build a leak-free train/validation split"
            )
        generator.shuffle(groups)
        validation_count = max(1, int(round(len(groups) * validation_fraction)))
        validation_count = min(validation_count, len(groups) - 1)
        for group in groups[:validation_count]:
            validation.extend(by_label[label][group])
        for group in groups[validation_count:]:
            train.extend(by_label[label][group])

    if not train or not validation:
        raise ValueError("the requested split produced an empty partition")
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
