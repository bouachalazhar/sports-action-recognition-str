"""STR image datasets, label mapping, and grouped splits."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ClipSample:
    """One clip and its pre-computed STR images."""

    clip_id: str
    label: int
    label_name: str
    images: tuple[Path, ...]


def discover_clips(root: Path) -> list[ClipSample]:
    """Scan ``root/<label>/<clip>/str`` or ``root/<split>/<label>/<clip>/str``.

    The clip identifier is the path of the clip directory relative to ``root``,
    so two identical clip names in different splits never collide.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    directories = sorted({path.parent for path in root.rglob("str") if path.is_dir()})
    if not directories:
        raise FileNotFoundError(f"no STR directory found under {root}")

    labels = sorted({directory.parent.name for directory in directories})
    label_index = {name: index for index, name in enumerate(labels)}

    clips: list[ClipSample] = []
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
        label_name = directory.parent.name
        clips.append(
            ClipSample(
                clip_id=str(directory.relative_to(root)),
                label=label_index[label_name],
                label_name=label_name,
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
    """Split clips into train and validation sets without clip leakage."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")

    by_label: dict[int, list[ClipSample]] = {}
    for clip in clips:
        by_label.setdefault(clip.label, []).append(clip)

    generator = random.Random(seed)
    train: list[ClipSample] = []
    validation: list[ClipSample] = []
    for label in sorted(by_label):
        samples = list(by_label[label])
        generator.shuffle(samples)
        cut = max(1, int(round(len(samples) * (1.0 - validation_fraction))))
        cut = min(cut, len(samples) - 1) if len(samples) > 1 else cut
        train.extend(samples[:cut])
        validation.extend(samples[cut:])

    if not validation:
        validation = [train.pop()]
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
