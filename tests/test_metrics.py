"""Tests for the NumPy metrics."""

from __future__ import annotations

import numpy as np
import pytest

from str_action.metrics import (
    confusion_matrix,
    macro_f1,
    per_class_recall,
    top1_accuracy,
)


def test_top1_accuracy() -> None:
    targets = np.array([0, 1, 1])
    predictions = np.array([0, 0, 1])
    assert top1_accuracy(targets, predictions) == pytest.approx(2 / 3)


def test_confusion_matrix_layout() -> None:
    matrix = confusion_matrix(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2)
    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_macro_f1_balanced() -> None:
    targets = np.array([0, 0, 1, 1])
    assert macro_f1(targets, targets, 2) == pytest.approx(1.0)


def test_macro_f1_chance() -> None:
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 0, 0])
    # Class 0 F1 = 2/3, class 1 F1 = 0, macro = 1/3.
    assert macro_f1(targets, predictions, 2) == pytest.approx(1 / 3)


def test_per_class_recall() -> None:
    targets = np.array([0, 1, 1, 1])
    predictions = np.array([0, 0, 1, 1])
    assert per_class_recall(targets, predictions, 2) == [1.0, pytest.approx(2 / 3)]


def test_empty_targets_rejected() -> None:
    with pytest.raises(ValueError):
        top1_accuracy(np.array([]), np.array([]))
