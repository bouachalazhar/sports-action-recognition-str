"""Tests for clip-level aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from str_action.evaluate import aggregate_clips, clip_vote_fractions


def test_hard_vote_uses_the_majority_of_str_images() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
            [0.55, 0.45],
        ],
        dtype=np.float64,
    )
    indices = np.asarray([0, 0, 0, 0, 0], dtype=np.int64)

    fractions = clip_vote_fractions(probabilities, indices, num_clips=1)
    assert np.allclose(fractions[0], [0.6, 0.4])
    assert aggregate_clips(probabilities, indices, num_clips=1, strategy="vote")[0] == 0


def test_softmax_aggregation_averages_probabilities() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.55, 0.45],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )
    indices = np.asarray([0, 0, 0, 0, 0], dtype=np.int64)

    # The mean probability is [0.49, 0.51], so softmax flips to class 1.
    mean = probabilities.mean(axis=0)
    assert mean[1] > mean[0]
    softmax_prediction = aggregate_clips(
        probabilities, indices, num_clips=1, strategy="softmax"
    )[0]
    assert softmax_prediction == 1
    # Votes are 2/3 for class 0, so the manuscript rule still favours class 0.
    assert aggregate_clips(probabilities, indices, num_clips=1, strategy="vote")[0] == 0


def test_vote_fractions_support_multiple_clips() -> None:
    probabilities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]],
        dtype=np.float64,
    )
    indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    fractions = clip_vote_fractions(probabilities, indices, num_clips=2)
    assert np.allclose(fractions[0], [0.5, 0.5])
    assert np.allclose(fractions[1], [0.5, 0.5])


def test_unknown_aggregation_is_rejected() -> None:
    probabilities = np.asarray([[0.5, 0.5]], dtype=np.float64)
    indices = np.asarray([0], dtype=np.int64)
    with pytest.raises(ValueError, match="unknown clip aggregation"):
        aggregate_clips(probabilities, indices, num_clips=1, strategy="median")
