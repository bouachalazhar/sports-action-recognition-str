"""End-to-end inference on a raw video."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from str_action.checkpoints import load_checkpoint
from str_action.config import apply_defaults, config_path, load_config, section
from str_action.evaluate import CLIP_AGGREGATIONS, clip_vote_fractions
from str_action.models import build_model
from str_action.roi import roi_str
from str_action.utils import resolve_device, seed_everything, write_json
from str_action.walk import (
    TEMPORAL_POLICIES,
    random_walk_frames,
    resolve_temporal_target_length,
)


def read_video(path: Path) -> np.ndarray:
    """Decode a video into an RGB array of shape ``(T, H, W, C)``."""
    container = av.open(str(path))
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    if not frames:
        raise ValueError(f"no frame could be decoded from {path}")
    return np.stack(frames, axis=0)


def _checkpoint_representation(checkpoint: dict[str, object]) -> dict[str, object]:
    """Return the representation metadata of a checkpoint, with safe defaults."""
    payload = checkpoint.get("representation", {})
    return dict(payload) if isinstance(payload, dict) else {}


def classify_video(
    video_path: Path,
    checkpoint_path: Path,
    *,
    walks: int | None = None,
    walk_length: int | None = None,
    image_size: int | None = None,
    roi_mode: str | None = None,
    temporal_policy: str | None = None,
    clip_aggregation: str | None = None,
    seed: int = 17,
    device: str = "auto",
    output_dir: Path | None = None,
    allow_legacy_checkpoint: bool = False,
) -> dict[str, object]:
    """Classify one video and return the per-STR probabilities and the decision.

    When an argument is omitted, the value stored by training is reused. This
    prevents an inference run from silently using a different walk length or
    ROI rule than the checkpoint it evaluates.
    """
    seed_everything(seed)
    torch_device = resolve_device(device)

    checkpoint = load_checkpoint(checkpoint_path, allow_legacy=allow_legacy_checkpoint)
    classes: list[str] = checkpoint["class_names"]
    metadata = _checkpoint_representation(checkpoint)
    resolved_image_size = int(
        image_size
        if image_size is not None
        else checkpoint.get("image_size", metadata.get("image_size", 224))
    )
    resolved_walks = int(walks if walks is not None else metadata.get("walks", 1000))
    resolved_walk_length = int(
        walk_length if walk_length is not None else metadata.get("walk_length", 224)
    )
    resolved_roi_mode = str(
        roi_mode if roi_mode is not None else metadata.get("roi_mode", "all")
    )
    resolved_temporal_policy = str(
        temporal_policy
        if temporal_policy is not None
        else metadata.get("temporal_policy", "preserve")
    )
    training_metadata = checkpoint.get("training", {})
    resolved_aggregation = str(
        clip_aggregation
        if clip_aggregation is not None
        else (
            training_metadata.get("clip_aggregation", "vote")
            if isinstance(training_metadata, dict)
            else "vote"
        )
    )
    if resolved_roi_mode not in ("all", "none"):
        raise ValueError(f"unknown roi_mode: {resolved_roi_mode}")
    if resolved_temporal_policy not in TEMPORAL_POLICIES:
        raise ValueError(f"unknown temporal policy: {resolved_temporal_policy}")
    if resolved_aggregation not in CLIP_AGGREGATIONS:
        raise ValueError(f"unknown clip aggregation: {resolved_aggregation}")

    model = build_model(
        str(checkpoint["model_name"]),
        num_classes=len(classes),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch_device)
    model.eval()

    _field, _players, roi = roi_str(str(video_path))
    video = read_video(video_path)
    use_roi = resolved_roi_mode == "all" and bool(np.any(roi))
    if resolved_roi_mode == "all" and not np.any(roi):
        warnings.warn(
            f"empty ROI for {video_path}; falling back to the full frame",
            RuntimeWarning,
            stacklevel=2,
        )

    target_length = resolve_temporal_target_length(
        video.shape[0], resolved_walk_length, resolved_temporal_policy
    )
    images, points = random_walk_frames(
        video,
        walk_count=resolved_walks,
        walk_length=resolved_walk_length,
        temporal_target_length=target_length,
        temporal_pad_mode="duplicate",
        start_mask=roi if use_roi else None,
        walk_mask=roi if use_roi else None,
        seed=seed,
    )

    resized = np.stack(
        [
            cv2.resize(
                image,
                (resolved_image_size, resolved_image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            for image in images
        ]
    )
    batch = torch.from_numpy(
        resized.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    ).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    with torch.no_grad():
        logits = model(batch.to(torch_device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    mean_probabilities = probabilities.mean(axis=0)
    clip_indices = np.zeros(probabilities.shape[0], dtype=np.int64)
    vote_fractions = clip_vote_fractions(probabilities, clip_indices, 1)[0]
    if resolved_aggregation == "vote":
        predicted = int(np.argmax(vote_fractions + 1e-6 * mean_probabilities))
    else:
        predicted = int(np.argmax(mean_probabilities))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            cv2.imwrite(
                str(output_dir / f"str_{index:04d}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )
        cv2.imwrite(str(output_dir / "roi.png"), roi)

    report: dict[str, object] = {
        "video": str(video_path),
        "prediction": classes[predicted],
        "prediction_index": predicted,
        "clip_aggregation": resolved_aggregation,
        "probabilities": {
            name: float(value)
            for name, value in zip(classes, mean_probabilities, strict=True)
        },
        "vote_fractions": {
            name: float(value)
            for name, value in zip(classes, vote_fractions, strict=True)
        },
        "walks": resolved_walks,
        "walk_length": resolved_walk_length,
        "image_size": resolved_image_size,
        "roi_mode": resolved_roi_mode,
        "temporal_policy": resolved_temporal_policy,
        "points": len(points),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the inference command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--walks",
        type=int,
        default=None,
        help="number of STR images; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=None,
        help="walk length; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="resize before batching; defaults to the checkpoint image size",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("all", "none"),
        default=None,
        help="ROI rule; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--temporal-policy",
        choices=TEMPORAL_POLICIES,
        default=None,
        help="temporal axis policy; defaults to the checkpoint metadata",
    )
    parser.add_argument(
        "--clip-aggregation",
        choices=CLIP_AGGREGATIONS,
        default=None,
        help="clip decision rule; defaults to the checkpoint metadata",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--allow-legacy-checkpoint",
        action="store_true",
        help="allow trusted v1.0.0 checkpoints (unsafe unpickler)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=Path("inference.json"))
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments with optional YAML defaults."""
    parser = build_parser()
    config = load_config(config_path(argv))
    apply_defaults(parser, section(config, "inference"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)

    report = classify_video(
        args.video,
        args.checkpoint,
        walks=args.walks,
        walk_length=args.walk_length,
        image_size=args.image_size,
        roi_mode=args.roi_mode,
        temporal_policy=args.temporal_policy,
        clip_aggregation=args.clip_aggregation,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        allow_legacy_checkpoint=args.allow_legacy_checkpoint,
    )
    write_json(args.report, report)
    print(f"{args.video}: {report['prediction']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
