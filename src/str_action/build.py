"""Build an STR image dataset from a labelled video tree.

Expected input layout: ``<source>/<label>/<group>/<clip>.mp4``. The output
preserves the label and group directories, with one folder per clip containing
its full spatio-temporal STR images, the ROI masks and the sampled points.
``--roi-mode all`` reproduces the ROI variant of the article and applies the
active support mask to every step of the walk. ``--roi-mode none`` reproduces
the no-ROI variant.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from str_action.config import (
    REPRESENTATION_DEFAULTS,
    apply_defaults,
    config_path,
    load_config,
    representation,
    section,
)
from str_action.infer import read_video
from str_action.roi import roi_str
from str_action.utils import seed_everything
from str_action.walk import (
    TEMPORAL_POLICIES,
    random_walk_frames,
    resolve_temporal_target_length,
)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov"}


def build_split(
    source: Path,
    destination: Path,
    *,
    walks: int,
    walk_length: int,
    roi_mode: str,
    image_size: int,
    seed: int,
    temporal_policy: str = "preserve",
    save_points: bool = False,
) -> int:
    """Convert every video of one source tree into full STR images."""
    if temporal_policy not in TEMPORAL_POLICIES:
        raise ValueError(f"unknown temporal policy: {temporal_policy}")
    videos = sorted(
        path for path in source.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES
    )
    written = 0

    for counter, video_path in enumerate(tqdm(videos, desc=source.name)):
        relative = video_path.relative_to(source)
        folder = destination / relative.parent / video_path.stem
        folder.mkdir(parents=True, exist_ok=True)

        field, players, roi = roi_str(str(video_path))
        cv2.imwrite(str(folder / "field.png"), field)
        cv2.imwrite(str(folder / "players.png"), players)
        cv2.imwrite(str(folder / "roi.png"), roi)

        video = read_video(video_path)
        use_roi = roi_mode == "all" and bool(np.any(roi))
        if roi_mode == "all" and not np.any(roi):
            warnings.warn(
                f"empty ROI for {video_path}; falling back to the full frame",
                RuntimeWarning,
                stacklevel=2,
            )
        target_length = resolve_temporal_target_length(
            video.shape[0], walk_length, temporal_policy
        )
        images, points = random_walk_frames(
            video,
            walk_count=walks,
            walk_length=walk_length,
            temporal_target_length=target_length,
            temporal_pad_mode="duplicate",
            start_mask=roi if use_roi else None,
            walk_mask=roi if use_roi else None,
            seed=seed + counter,
        )

        str_dir = folder / "str"
        str_dir.mkdir(exist_ok=True)
        for index, image in enumerate(images):
            resized = cv2.resize(
                image, (image_size, image_size), interpolation=cv2.INTER_LINEAR
            )
            cv2.imwrite(
                str(str_dir / f"{index}.png"),
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            )
        if save_points:
            (folder / "points.json").write_text(json.dumps(points), encoding="utf-8")
        written += 1

    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--walks",
        type=int,
        default=REPRESENTATION_DEFAULTS["walks"],
        help="number of STR images per video",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=REPRESENTATION_DEFAULTS["walk_length"],
        help="number of pixels visited by one walk",
    )
    parser.add_argument(
        "--image-size", type=int, default=REPRESENTATION_DEFAULTS["image_size"]
    )
    parser.add_argument(
        "--roi-mode",
        choices=("all", "none"),
        default=REPRESENTATION_DEFAULTS["roi_mode"],
        help="apply the active ROI mask to every walk, or use the full frame",
    )
    parser.add_argument(
        "--temporal-policy",
        choices=TEMPORAL_POLICIES,
        default=REPRESENTATION_DEFAULTS["temporal_policy"],
        help="preserve the decoded temporal axis, or pad clips shorter than L/2",
    )
    parser.add_argument(
        "--save-points",
        action=argparse.BooleanOptionalAction,
        default=REPRESENTATION_DEFAULTS["save_points"],
        help="write the sampled walk coordinates to points.json",
    )
    parser.add_argument("--seed", type=int, default=17)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments, applying YAML defaults before explicit flags."""
    parser = build_parser()
    config = load_config(config_path(argv))
    apply_defaults(parser, representation(config), known_only=True)
    apply_defaults(parser, section(config, "build"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)

    seed_everything(args.seed)
    count = build_split(
        args.source,
        args.destination,
        walks=args.walks,
        walk_length=args.walk_length,
        roi_mode=args.roi_mode,
        image_size=args.image_size,
        seed=args.seed,
        temporal_policy=args.temporal_policy,
        save_points=args.save_points,
    )
    print(f"converted {count} videos into {args.destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
