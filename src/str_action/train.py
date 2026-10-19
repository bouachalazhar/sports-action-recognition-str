"""Supervised training on pre-computed STR images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from str_action.dataset import (
    STRImageDataset,
    discover_clips,
    grouped_split,
    label_names,
)
from str_action.evaluate import evaluate_model
from str_action.models import SUPPORTED_MODELS, build_model, count_parameters
from str_action.utils import resolve_device, seed_everything, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/str"))
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
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
        drop_last=len(train_dataset) >= args.batch_size,
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, float]] = []
    best_macro_f1 = -1.0
    best_path = args.output_dir / "best.pt"

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
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_names": classes,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_macro_f1": best_macro_f1,
                    "args": vars(args),
                },
                best_path,
            )

    write_json(
        args.output_dir / "history.json",
        {
            "model": args.model,
            "parameters": count_parameters(model),
            "train_clips": len(train_clips),
            "validation_clips": len(validation_clips),
            "class_names": classes,
            "history": history,
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
