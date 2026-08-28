"""Leakage-safe temporal splitting.

For network-security forecasting, the ONLY valid split is chronological: the
model may train on the past and is evaluated on the strictly-later future.
Random shuffling would leak future information. These helpers enforce that
invariant and :func:`assert_no_leakage` is exercised directly by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class TemporalSplit:
    train_indices: List[int]
    test_indices: List[int]

    @property
    def boundary(self) -> int:
        return self.test_indices[0] if self.test_indices else len(self.train_indices)


def temporal_split(num_windows: int, test_fraction: float = 0.3) -> TemporalSplit:
    """Split window indices chronologically into (train, test).

    The last ``test_fraction`` of windows become the test set; everything before
    is training. At least one window is guaranteed in each side when possible.
    """
    if num_windows <= 0:
        return TemporalSplit([], [])
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be in (0, 1)")
    n_test = max(1, int(round(num_windows * test_fraction)))
    n_test = min(n_test, num_windows - 1) if num_windows > 1 else 0
    boundary = num_windows - n_test
    return TemporalSplit(
        train_indices=list(range(boundary)),
        test_indices=list(range(boundary, num_windows)),
    )


def assert_no_leakage(split: TemporalSplit) -> None:
    """Raise ``AssertionError`` if the split could leak future into past."""
    train, test = split.train_indices, split.test_indices
    if not train:
        raise AssertionError("Temporal split has an empty training set")
    overlap = set(train) & set(test)
    if overlap:
        raise AssertionError(f"Train/test overlap detected: {sorted(overlap)}")
    if test and max(train) >= min(test):
        raise AssertionError(
            f"Temporal ordering violated: max(train)={max(train)} >= min(test)={min(test)}"
        )
    # Indices must be contiguous and non-decreasing (no interleaving).
    if train != sorted(train) or test != sorted(test):
        raise AssertionError("Split indices are not chronologically ordered")
