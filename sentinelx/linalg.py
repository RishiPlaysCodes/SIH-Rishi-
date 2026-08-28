"""Pure-python numeric primitives.

A tiny, dependency-free stand-in for the NumPy/PyTorch tensor operations used by
Sentinel-X. Everything is expressed with nested ``list[list[float]]`` matrices
and ``list[float]`` vectors. It is deliberately small: just enough linear algebra
to fit ridge-regression world models and compute the distance/error metrics the
forecasting and analytics layers need.

When the full stack is installed, model code swaps these helpers for NumPy /
Torch without changing call sites (the signatures mirror common ndarray ops).
"""

from __future__ import annotations

import math
from typing import List, Sequence

Vector = List[float]
Matrix = List[List[float]]


# --------------------------------------------------------------------------- #
# Vector ops
# --------------------------------------------------------------------------- #
def vzeros(n: int) -> Vector:
    return [0.0] * n


def vadd(a: Sequence[float], b: Sequence[float]) -> Vector:
    _check_len(a, b)
    return [x + y for x, y in zip(a, b)]


def vsub(a: Sequence[float], b: Sequence[float]) -> Vector:
    _check_len(a, b)
    return [x - y for x, y in zip(a, b)]


def vscale(a: Sequence[float], s: float) -> Vector:
    return [x * s for x in a]


def vdot(a: Sequence[float], b: Sequence[float]) -> float:
    _check_len(a, b)
    return sum(x * y for x, y in zip(a, b))


def l2(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    _check_len(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mse(a: Sequence[float], b: Sequence[float]) -> float:
    _check_len(a, b)
    if not a:
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = l2(a), l2(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return vdot(a, b) / (na * nb)


def mean_vector(rows: Sequence[Sequence[float]]) -> Vector:
    if not rows:
        return []
    dim = len(rows[0])
    acc = vzeros(dim)
    for row in rows:
        for j in range(dim):
            acc[j] += row[j]
    return [v / len(rows) for v in acc]


def std_vector(rows: Sequence[Sequence[float]]) -> Vector:
    if not rows:
        return []
    mu = mean_vector(rows)
    dim = len(mu)
    acc = vzeros(dim)
    for row in rows:
        for j in range(dim):
            acc[j] += (row[j] - mu[j]) ** 2
    return [math.sqrt(v / len(rows)) for v in acc]


# --------------------------------------------------------------------------- #
# Matrix ops
# --------------------------------------------------------------------------- #
def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(m: Matrix) -> Matrix:
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    if not a or not b:
        return []
    if len(a[0]) != len(b):
        raise ValueError(f"Shape mismatch for matmul: {len(a[0])} vs {len(b)}")
    bt = transpose(b)
    return [[vdot(row, col) for col in bt] for row in a]


def matvec(m: Matrix, v: Sequence[float]) -> Vector:
    return [vdot(row, v) for row in m]


def madd(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mscale(a: Matrix, s: float) -> Matrix:
    return [[v * s for v in row] for row in a]


def solve(a: Matrix, b: Matrix) -> Matrix:
    """Solve ``A X = B`` for X via Gauss-Jordan elimination with partial pivoting.

    ``A`` must be square (n x n); ``B`` is n x m. Returns X (n x m).
    """
    n = len(a)
    if any(len(row) != n for row in a):
        raise ValueError("solve() requires a square matrix A")
    if len(b) != n:
        raise ValueError("solve() dimension mismatch between A and B")
    m = len(b[0]) if b else 0
    # Augmented matrix [A | B].
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        # Partial pivot: find the largest magnitude entry in this column.
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            # Singular; nudge with a tiny value to keep the solve well-posed.
            aug[pivot][col] += 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        aug[col] = [v / pivot_val for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != 0.0:
                aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [row[n : n + m] for row in aug]


def ridge_fit(x: Matrix, y: Matrix, lam: float) -> Matrix:
    """Closed-form multi-output ridge regression.

    Solves ``W = (XᵀX + λI)⁻¹ Xᵀ Y`` where X is (samples x in_dim),
    Y is (samples x out_dim). Returns W as (in_dim x out_dim).
    Callers are expected to append a bias column to X if a bias term is wanted.
    """
    if not x or not y:
        return []
    xt = transpose(x)
    xtx = matmul(xt, x)
    in_dim = len(xtx)
    reg = [[xtx[i][j] + (lam if i == j else 0.0) for j in range(in_dim)] for i in range(in_dim)]
    xty = matmul(xt, y)
    return solve(reg, xty)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _check_len(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
