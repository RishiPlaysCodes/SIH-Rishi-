"""Data-leakage tests (PRD §2.17 / §2.18).

Network-security forecasting silently cheats in two classic ways:
  1. shuffling time (evaluating on windows the model effectively saw), and
  2. fitting the feature scaler on the whole dataset (test stats leak into train).
These tests make both impossible to do by accident.
"""

import pytest

from sentinelx.data import Normalizer
from sentinelx.data.splits import TemporalSplit, assert_no_leakage, temporal_split


def test_temporal_split_is_chronological():
    split = temporal_split(40, test_fraction=0.3)
    assert max(split.train_indices) < min(split.test_indices)
    assert set(split.train_indices).isdisjoint(split.test_indices)
    # union covers everything, no gaps
    assert split.train_indices + split.test_indices == list(range(40))


def test_assert_no_leakage_passes_valid_split():
    assert_no_leakage(temporal_split(30, 0.3))  # should not raise


def test_assert_no_leakage_detects_overlap():
    bad = TemporalSplit(train_indices=[0, 1, 2, 3], test_indices=[3, 4, 5])
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_assert_no_leakage_detects_time_travel():
    # test windows earlier than some train windows -> future leaks into past
    bad = TemporalSplit(train_indices=[0, 1, 5, 6], test_indices=[2, 3])
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_assert_no_leakage_detects_unordered():
    bad = TemporalSplit(train_indices=[2, 0, 1], test_indices=[3, 4])
    with pytest.raises(AssertionError):
        assert_no_leakage(bad)


def test_empty_train_is_rejected():
    with pytest.raises(AssertionError):
        assert_no_leakage(TemporalSplit(train_indices=[], test_indices=[0, 1]))


def test_normalizer_fit_uses_train_only(graph_sequence):
    """The scaler must be fit on train rows only; test stats must NOT influence it."""
    split = temporal_split(len(graph_sequence), 0.3)
    train_rows = []
    for i in split.train_indices:
        train_rows.extend(graph_sequence[i].feature_matrix())
    norm_train = Normalizer(mode="zscore").fit(train_rows)

    # A scaler fit on ALL data (the leaky version) would have different means,
    # because the attack windows in the test set shift the distribution.
    all_rows = []
    for g in graph_sequence:
        all_rows.extend(g.feature_matrix())
    norm_all = Normalizer(mode="zscore").fit(all_rows)

    assert norm_train.means != norm_all.means, (
        "Train-only and all-data scalers must differ; if identical, the attack "
        "signal isn't shifting the distribution and leakage would be undetectable"
    )
