"""Spatio-temporal random walk (STR) representations for action recognition.

The package implements the method described in the CVIU paper associated with
this repository:

    L. Bouacha, T. Magnien, L. Wendling, C. Kurtz. Recognition of action
    phases from spatio-temporal intermediate representations in amateur sports
    competition videos. Computer Vision and Image Understanding, 268:104748,
    2026.

Modules
-------
walk     Random walk sampling over a video volume.
roi      Region-of-interest extraction from green chromaticity and optical flow.
dataset  STR image datasets and grouping utilities.
models   Two-dimensional classifier factory (ResNet-18, MobileNet V4, ...).
train    Supervised training on pre-computed STR images.
evaluate Clip-level evaluation (Top-1, macro-F1, confusion matrix).
infer    End-to-end inference on a raw video.
build    STR dataset construction from a labelled video tree.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.2"
