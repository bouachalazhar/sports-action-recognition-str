"""Checkpoint loading with an explicit legacy escape hatch."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    """Load a checkpoint, preferring PyTorch's restricted unpickler.

    Checkpoints produced by v1.0.0 stored the raw argument namespace, including
    :class:`pathlib.Path` values that ``weights_only=True`` rejects. Such files
    can be converted in memory only when the caller explicitly marks the file
    as trusted with ``allow_legacy=True``. New checkpoints never take this path.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError) as error:
        if not allow_legacy:
            raise RuntimeError(
                "checkpoint is not compatible with weights_only=True; "
                "retrain with v1.0.1 or pass --allow-legacy-checkpoint only "
                "for a trusted local file"
            ) from error
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    args = checkpoint.get("args")
    if isinstance(args, dict) and "dropout" not in checkpoint:
        checkpoint["dropout"] = float(args.get("dropout", 0.0))
    checkpoint.pop("args", None)
    return checkpoint
