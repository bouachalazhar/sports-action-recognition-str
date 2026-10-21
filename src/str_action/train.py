"""Supervised training on pre-computed STR images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from str_action.config import (
    REPRESENTATION_DEFAULTS,
    apply_defaults,
    config_path,
    load_config,
    representation,
    section,
)
from str_action.dataset import (
    ClipSample,
    STRImageDataset,
    discover_clips,
    grouped_split,
    label_names,
)
from str_action.evaluate import CLIP_AGGREGATIONS, evaluate_model
from str_action.models import SUPPORTED_MODELS, build_model, count_parameters
from str_action.utils import resolve_device, seed_everything, write_json
from str_action.walk import TEMPORAL_POLICIES


def class_weights(
    clips: list[ClipSample], num_classes: int, mode: str = "balanced"
) -> torch.Tensor | None:
    """Return inverse-frequency class weights, or ``None`` when disabled.

    Counts are taken at clip level because every clip contributes the same
    target to the clip-level decision, even when it contains a different number
    of STR images.
    """
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"unknown class-weight mode: {mode}")
    counts = torch.bincount(
        torch.tensor([clip.label for clip in clips], dtype=torch.long),
        minlength=num_classes,
    ).to(torch.float64)
    if torch.any(counts == 0):
        raise ValueError("every class must have at least one training clip")
    weights = counts.sum() / (num_classes * counts)
    return weights.to(torch.float32)


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/str"))
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet18")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--image-size", type=int, default=REPRESENTATION_DEFAULTS["image_size"]
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "fraction of source groups held out for validation; "
            "the test tree is external"
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--class-weights",
        choices=("none", "balanced"),
        default="balanced",
        help="weight the cross-entropy by inverse class frequency",
    )
    parser.add_argument(
        "--clip-aggregation",
        choices=CLIP_AGGREGATIONS,
        default="vote",
        help="clip decision rule used for validation and model selection",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="stop after this many epochs without a validation improvement; 0 disables",
    )
    parser.add_argument(
        "--walks",
        type=int,
        default=REPRESENTATION_DEFAULTS["walks"],
        help="STR metadata stored in the checkpoint",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=REPRESENTATION_DEFAULTS["walk_length"],
        help="STR metadata stored in the checkpoint",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("all", "none"),
        default=REPRESENTATION_DEFAULTS["roi_mode"],
        help="STR metadata stored in the checkpoint",
    )
    parser.add_argument(
        "--temporal-policy",
        choices=TEMPORAL_POLICIES,
        default=REPRESENTATION_DEFAULTS["temporal_policy"],
        help="STR metadata stored in the checkpoint",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments, applying YAML defaults before explicit flags."""
    parser = build_parser()
    config = load_config(config_path(argv))
    apply_defaults(parser, representation(config), known_only=True)
    apply_defaults(parser, section(config, "train"))
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> Path:
    """Train one model and return the path of the best checkpoint."""
    seed_everything(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clips = discover_clips(args.data_root)
    train_clips, validation_clips = grouped_split(
        clips, validation_fraction=args.validation_fraction, seed=args.seed
    )
    classes = label_names(clips)

    train_dataset = STRImageDataset(train_clips, image_size=args.image_size)
    validation_dataset = STRImageDataset(validation_clips, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=len(train_dataset) > args.batch_size,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model(
        args.model,
        num_classes=len(classes),
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    weights = class_weights(train_clips, len(classes), args.class_weights)
    criterion = nn.CrossEntropyLoss(
        weight=None if weights is None else weights.to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best.pt"
    representation_metadata = {
        "walks": args.walks,
        "walk_length": args.walk_length,
        "image_size": args.image_size,
        "roi_mode": args.roi_mode,
        "temporal_policy": args.temporal_policy,
    }
    training_metadata = {
        "class_weights": args.class_weights,
        "clip_aggregation": args.clip_aggregation,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        started = time.perf_counter()

        for images, targets, _clip_index in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * targets.numel()
            seen += targets.numel()

        report = evaluate_model(
            model,
            validation_loader,
            num_clips=len(validation_clips),
            clip_labels=validation_dataset.clip_labels,
            device=device,
            class_names=classes,
            aggregation=args.clip_aggregation,
        )
        epoch_report = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_top1": report["top1"],
            "val_macro_f1": report["macro_f1"],
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report))

        if report["macro_f1"] > best_macro_f1:
            best_macro_f1 = report["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_names": classes,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "representation": representation_metadata,
                    "training": training_metadata,
                },
                best_path,
            )

        if (
            args.early_stopping_patience > 0
            and epoch - best_epoch >= args.early_stopping_patience
        ):
            break

    write_json(
        args.output_dir / "history.json",
        {
            "model": args.model,
            "parameters": count_parameters(model),
            "train_clips": len(train_clips),
            "validation_clips": len(validation_clips),
            "class_names": classes,
            "representation": representation_metadata,
            "class_weight_mode": args.class_weights,
            "clip_aggregation": args.clip_aggregation,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "checkpoint": str(best_path),
        },
    )
    return best_path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    checkpoint = train(args)
    print(f"best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
