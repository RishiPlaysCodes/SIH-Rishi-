"""Numeric primitives: matrix ops, solver, ridge regression."""

import math

import pytest

from sentinelx import linalg


def test_matmul_and_transpose():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    assert linalg.matmul(a, b) == [[19.0, 22.0], [43.0, 50.0]]
    assert linalg.transpose(a) == [[1.0, 3.0], [2.0, 4.0]]


def test_solve_identity():
    a = [[2.0, 0.0], [0.0, 4.0]]
    b = [[2.0], [8.0]]
    x = linalg.solve(a, b)
    assert abs(x[0][0] - 1.0) < 1e-9
    assert abs(x[1][0] - 2.0) < 1e-9


def test_solve_recovers_known_system():
    a = [[3.0, 2.0], [1.0, 2.0]]
    # true x = [2, 3]; b = a @ x
    b = [[3 * 2 + 2 * 3], [1 * 2 + 2 * 3]]
    x = linalg.solve(a, b)
    assert abs(x[0][0] - 2.0) < 1e-6
    assert abs(x[1][0] - 3.0) < 1e-6


def test_ridge_fit_recovers_linear_map():
    # y = 2*x0 + 3 (with bias column). Fit should recover ~[2, 3].
    xs = [[float(i), 1.0] for i in range(20)]
    ys = [[2.0 * i + 3.0] for i in range(20)]
    w = linalg.ridge_fit(xs, ys, lam=1e-6)
    assert abs(w[0][0] - 2.0) < 1e-2
    assert abs(w[1][0] - 3.0) < 1e-2


def test_vector_ops():
    assert linalg.euclidean([0, 0], [3, 4]) == 5.0
    assert abs(linalg.mse([1, 2, 3], [1, 2, 3])) < 1e-12
    assert linalg.cosine([1, 0], [0, 1]) == 0.0
    assert abs(linalg.cosine([1, 1], [1, 1]) - 1.0) < 1e-9


def test_solve_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        linalg.solve([[1.0, 2.0]], [[1.0]])  # non-square A


def test_sigmoid_bounds():
    assert 0.0 < linalg.sigmoid(-50) < 1e-10 or linalg.sigmoid(-50) >= 0
    assert 0.5 == linalg.sigmoid(0.0)
    assert math.isclose(linalg.sigmoid(50), 1.0, abs_tol=1e-6)
