"""Deterministic synthetic fixture for the demo pipeline.

The fixture generates small, closed-form synthetic clips (no random numbers
at all) so that every downstream artefact - STR images, the trained checkpoint
and the evaluation metrics - is reproducible bit-for-bit inside the environment
pinned by ``uv.lock``. The clips are encoded with the same codec used by the
test suite so no platform FFmpeg build can alter the bytes.

Usage:
    uv run demo/generate_fixture.py
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np

FRAMES = 16
SIZE = 64
RATE = 8
TRAIN_TREE = Path("demo/data/train")
TEST_TREE = Path("demo/data/test")

TRAIN_PLAN = (("match_001", 3), ("match_002", 3))
TEST_PLAN = (("match_101", 2), ("match_102", 2))


def clip_pixels(t: int, moving: bool) -> np.ndarray:
    """Return one synthetic ``RGB`` frame for the given time index."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    red = (xx * 7 + t) % 251
    green = (yy * 5 + t) % 251
    blue = ((xx + yy) * 3 + t) % 251
    frame = np.stack((red, green, blue), axis=2).astype(np.uint8)

    offset = int(round(SIZE / 4 * np.sin(2 * np.pi * t / FRAMES))) if moving else 0
    top = (SIZE // 3) if moving else SIZE // 3
    left = (SIZE // 3 + offset) % (SIZE - 16)
    frame[top : top + 16, left : left + 16] = 220
    return frame


def write_clip(path: Path, moving: bool) -> None:
    """Encode one deterministic synthetic clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=RATE)
        stream.width = SIZE
        stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        for t in range(FRAMES):
            frame = av.VideoFrame.from_ndarray(clip_pixels(t, moving), format="rgb24")
            frame.pts = t
            frame.time_base = Fraction(1, RATE)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_tree(root: Path, plan: tuple[tuple[str, int], ...]) -> None:
    """Write the ``<label>/<group>/<clip>.mp4`` demo tree."""
    counter = 0
    for label, moving in (("action", True), ("non_action", False)):
        for group, count in plan:
            for _index in range(count):
                counter += 1
                path = root / label / group / f"clip_{counter:03d}.mp4"
                write_clip(path, moving)


def main() -> None:
    for tree in (TRAIN_TREE, TEST_TREE):
        if tree.exists():
            import shutil

            shutil.rmtree(tree)
    build_tree(TRAIN_TREE, TRAIN_PLAN)
    build_tree(TEST_TREE, TEST_PLAN)
    print(f"wrote fixture trees under {TRAIN_TREE} and {TEST_TREE}")


if __name__ == "__main__":
    main()
