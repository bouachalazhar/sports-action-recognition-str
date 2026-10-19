"""Metrics computed with NumPy only, so evaluation stays dependency-light."""

from __future__ import annotations

import numpy as np


def top1_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Fraction of exact matches."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.size == 0:
        raise ValueError("targets must not be empty")
    return float((targets == predictions).mean())


def confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Rows are targets, columns are predictions."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    pairs = zip(np.asarray(targets), np.asarray(predictions), strict=True)
    for target, prediction in pairs:
        matrix[int(target), int(prediction)] += 1
    return matrix


def macro_f1(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> float:
    """Unweighted mean of the per-class F1 scores."""
    matrix = confusion_matrix(targets, predictions, num_classes)
    scores: list[float] = []
    for index in range(num_classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            # A class absent from both target and prediction contributes a
            # neutral score instead of an arbitrary zero.
            scores.append(0.0)
        else:
            scores.append(2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def per_class_recall(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> list[float]:
    """Recall of each class, zero when the class is absent from the targets."""
    matrix = confusion_matrix(targets, predictions, num_classes)
    recalls: list[float] = []
    for index in range(num_classes):
        support = matrix[index, :].sum()
        recalls.append(float(matrix[index, index] / support) if support else 0.0)
    return recalls


__all__ = ["confusion_matrix", "macro_f1", "per_class_recall", "top1_accuracy"]
