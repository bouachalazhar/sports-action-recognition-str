"""Shared helpers: device selection, seeding, and JSON output."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Return the best available device for the requested policy."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")


__all__ = ["resolve_device", "seed_everything", "write_json"]
