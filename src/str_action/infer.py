"""End-to-end inference on a raw video."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from str_action.checkpoints import load_checkpoint
from str_action.models import build_model
from str_action.roi import roi_str
from str_action.utils import resolve_device, seed_everything, write_json
from str_action.walk import random_walk_frames


def read_video(path: Path) -> np.ndarray:
    """Decode a video into an RGB array of shape ``(T, H, W, C)``."""
    container = av.open(str(path))
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    container.close()
    if not frames:
        raise ValueError(f"no frame could be decoded from {path}")
    return np.stack(frames, axis=0)


def classify_video(
    video_path: Path,
    checkpoint_path: Path,
    *,
    walks: int = 64,
    walk_length: int = 112,
    image_size: int | None = None,
    roi_mode: str = "all",
    seed: int = 17,
    device: str = "auto",
    output_dir: Path | None = None,
    allow_legacy_checkpoint: bool = False,
) -> dict[str, object]:
    """Classify one video and return the per-STR probabilities and the decision."""
    seed_everything(seed)
    torch_device = resolve_device(device)

    checkpoint = load_checkpoint(checkpoint_path, allow_legacy=allow_legacy_checkpoint)
    classes: list[str] = checkpoint["class_names"]
    resolved_image_size = image_size or int(checkpoint.get("image_size", 224))
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(classes),
        dropout=float(checkpoint.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch_device)
    model.eval()

    _field, _players, roi = roi_str(str(video_path))
    video = read_video(video_path)
    use_roi = roi_mode == "all" and bool(np.any(roi))

    images, points = random_walk_frames(
        video,
        walk_count=walks,
        walk_length=walk_length,
        temporal_target_length=max(video.shape[0], walk_length // 2 + 1),
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
        "probabilities": {
            name: float(value)
            for name, value in zip(classes, mean_probabilities, strict=True)
        },
        "walks": walks,
        "walk_length": walk_length,
        "points": len(points),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--walks", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=112)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="resize before batching; defaults to the checkpoint image size",
    )
    parser.add_argument(
        "--roi-mode",
        choices=("all", "none"),
        default="all",
        help="apply the active ROI mask to every walk, or use the full frame",
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
    args = parser.parse_args(argv)

    report = classify_video(
        args.video,
        args.checkpoint,
        walks=args.walks,
        walk_length=args.walk_length,
        image_size=args.image_size,
        roi_mode=args.roi_mode,
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
