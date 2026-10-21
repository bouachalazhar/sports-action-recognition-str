"""Clip-level evaluation shared by training and the evaluation entry point."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from str_action.checkpoints import load_checkpoint
from str_action.config import apply_defaults, config_path, load_config, section
from str_action.metrics import (
    confusion_matrix,
    macro_f1,
    per_class_recall,
    top1_accuracy,
)

CLIP_AGGREGATIONS = ("vote", "softmax")


@torch.no_grad()
def collect_image_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return image probabilities, labels and clip indices for a loader."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    clip_indices: list[np.ndarray] = []

    for images, targets, clip_index in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(targets.numpy())
        clip_indices.append(clip_index.numpy())

    return (
        np.concatenate(probabilities, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(clip_indices, axis=0),
    )


def clip_vote_fractions(
    probabilities: np.ndarray, clip_indices: np.ndarray, num_clips: int
) -> np.ndarray:
    """Return the fraction of STR images voting for each class per clip."""
    votes = np.zeros((num_clips, probabilities.shape[1]), dtype=np.float64)
    winners = np.argmax(probabilities, axis=1)
    np.add.at(votes, (clip_indices, winners), 1.0)
    counts = np.bincount(clip_indices, minlength=num_clips).astype(np.float64)
    return votes / np.maximum(counts, 1)[:, None]


def aggregate_clips(
    probabilities: np.ndarray,
    clip_indices: np.ndarray,
    num_clips: int,
    strategy: str = "vote",
) -> np.ndarray:
    """Return one clip decision, using the manuscript vote rule by default.

    ``vote`` reproduces the article: every STR image votes for one class and
    the class with the largest proportion wins. ``softmax`` is kept as a
    diagnostic alternative that averages the image probabilities instead.
    Ties are broken by the mean probability, which does not change a strict
    majority but makes the result deterministic.
    """
    if strategy not in CLIP_AGGREGATIONS:
        raise ValueError(f"unknown clip aggregation: {strategy}")
    sums = np.zeros((num_clips, probabilities.shape[1]), dtype=np.float64)
    counts = np.zeros(num_clips, dtype=np.int64)
    np.add.at(sums, clip_indices, probabilities)
    np.add.at(counts, clip_indices, 1)
    counts = np.maximum(counts, 1)
    mean_probabilities = sums / counts[:, None]
    if strategy == "softmax":
        return np.argmax(mean_probabilities, axis=1)
    votes = clip_vote_fractions(probabilities, clip_indices, num_clips)
    return np.argmax(votes + 1e-6 * mean_probabilities, axis=1)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    num_clips: int,
    clip_labels: np.ndarray,
    device: torch.device,
    class_names: list[str],
    *,
    aggregation: str = "vote",
) -> dict[str, Any]:
    """Evaluate clip-level predictions and return a report dictionary."""
    probabilities, _image_labels, clip_indices = collect_image_probabilities(
        model, loader, device
    )
    predictions = aggregate_clips(
        probabilities, clip_indices, num_clips, strategy=aggregation
    )
    num_classes = len(class_names)

    matrix = confusion_matrix(clip_labels, predictions, num_classes)
    return {
        "clips": int(clip_labels.size),
        "images": int(probabilities.shape[0]),
        "aggregation": aggregation,
        "top1": top1_accuracy(clip_labels, predictions),
        "macro_f1": macro_f1(clip_labels, predictions, num_classes),
        "per_class_recall": per_class_recall(clip_labels, predictions, num_classes),
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
    }


def build_parser() -> Any:
    """Build the evaluation command-line parser."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--clip-aggregation",
        choices=CLIP_AGGREGATIONS,
        default=None,
        help="clip decision rule; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="allow trusted v1.0.0 checkpoints (unsafe unpickler)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Any:
    """Parse arguments with optional YAML defaults."""
    parser = build_parser()
    config = load_config(config_path(argv))
    apply_defaults(parser, section(config, "evaluate"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate a checkpoint on a directory of pre-computed STR images."""
    from torch.utils.data import DataLoader

    from str_action.dataset import (
        STRImageDataset,
        discover_clips,
    )
    from str_action.models import build_model
    from str_action.utils import resolve_device, write_json

    args = parse_args(argv)

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(
        args.checkpoint, allow_legacy=args.allow_legacy_checkpoint
    )
    model_name = checkpoint["model_name"]
    classes: list[str] = checkpoint["class_names"]
    image_size = args.image_size or int(checkpoint.get("image_size", 224))
    training_metadata = checkpoint.get("training", {})
    aggregation = args.clip_aggregation or (
        training_metadata.get("clip_aggregation", "vote")
        if isinstance(training_metadata, dict)
        else "vote"
    )

    model = build_model(
        model_name,
        num_classes=len(classes),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    clips = discover_clips(args.data_root, label_order=classes)
    dataset = STRImageDataset(clips, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    report = evaluate_model(
        model,
        loader,
        num_clips=len(clips),
        clip_labels=dataset.clip_labels,
        device=device,
        class_names=classes,
        aggregation=aggregation,
    )
    report["model"] = model_name
    report["data_root"] = str(args.data_root)
    write_json(args.output, report)
    print(
        f"clips={report['clips']} top1={report['top1']:.4f} "
        f"macro_f1={report['macro_f1']:.4f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLIP_AGGREGATIONS",
    "aggregate_clips",
    "clip_vote_fractions",
    "collect_image_probabilities",
    "evaluate_model",
    "main",
    "parse_args",
]
