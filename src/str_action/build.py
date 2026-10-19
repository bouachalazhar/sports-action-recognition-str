"""Build an STR image dataset from a labelled video tree.

Expected input layout: ``<source>/<split>/<label>/<clip>.mp4``. The output
layout mirrors it with one folder per clip containing its STR images, the ROI
masks and the sampled points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from str_action.infer import read_video
from str_action.roi import roi_str
from str_action.utils import seed_everything
from str_action.walk import random_walk_frames

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov"}


def build_split(
    source: Path,
    destination: Path,
    *,
    walks: int,
    walk_length: int,
    seed: int,
) -> int:
    """Convert every video of one source tree into STR images."""
    videos = sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )
    written = 0

    for counter, video_path in enumerate(tqdm(videos, desc=source.name)):
        relative = video_path.relative_to(source)
        label = relative.parts[0]
        clip = video_path.stem
        folder = destination / label / clip
        folder.mkdir(parents=True, exist_ok=True)

        field, players, roi = roi_str(str(video_path))
        cv2.imwrite(str(folder / "field.png"), field)
        cv2.imwrite(str(folder / "players.png"), players)
        cv2.imwrite(str(folder / "roi.png"), roi)

        video = read_video(video_path)
        images, points = random_walk_frames(
            video,
            walk_count=walks,
            walk_length=walk_length,
            temporal_target_length=max(video.shape[0], walk_length // 2 + 1),
            temporal_pad_mode="duplicate",
            seed=seed + counter,
        )

        middle = images.shape[1] // 2
        str_dir = folder / "str"
        str_dir.mkdir(exist_ok=True)
        for index, image in enumerate(images):
            resized = cv2.resize(
                image[middle], (224, 224), interpolation=cv2.INTER_LINEAR
            )
            cv2.imwrite(
                str(str_dir / f"{index}.png"),
                cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
            )
        (folder / "points.json").write_text(
            json.dumps(points), encoding="utf-8"
        )
        written += 1

    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--walks", type=int, default=1000)
    parser.add_argument("--walk-length", type=int, default=112)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    count = build_split(
        args.source,
        args.destination,
        walks=args.walks,
        walk_length=args.walk_length,
        seed=args.seed,
    )
    print(f"converted {count} videos into {args.destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
