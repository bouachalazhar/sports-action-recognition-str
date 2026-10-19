"""Smoke test of the training loop on a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from str_action.train import main


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 24, 3), value, dtype=np.uint8)).save(path)


def test_training_runs_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for label, value in (("non_action", 20), ("action", 220)):
        for clip in range(3):
            directory = root / "train" / label / f"clip_{clip}" / "str"
            write_image(directory / "0.png", value)
            write_image(directory / "1.png", value)

    output = tmp_path / "run"
    exit_code = main(
        [
            "--data-root",
            str(root),
            "--output-dir",
            str(output),
            "--model",
            "resnet18",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--image-size",
            "32",
            "--workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert (output / "best.pt").is_file()
    assert (output / "history.json").is_file()
