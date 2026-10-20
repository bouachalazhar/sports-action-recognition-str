"""Tests for checkpoint compatibility helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from str_action.checkpoints import load_checkpoint


def test_load_checkpoint_reads_modern_payload(tmp_path: Path) -> None:
    path = tmp_path / "modern.pt"
    torch.save({"state_dict": {}, "dropout": 0.25}, path)
    checkpoint = load_checkpoint(path)
    assert checkpoint["dropout"] == 0.25


def test_legacy_checkpoint_requires_explicit_trust(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {"state_dict": {}, "args": {"dropout": 0.5, "output": Path("run")}}, path
    )
    with pytest.raises(RuntimeError, match="weights_only"):
        load_checkpoint(path)

    checkpoint = load_checkpoint(path, allow_legacy=True)
    assert checkpoint["dropout"] == 0.5
    assert "args" not in checkpoint
