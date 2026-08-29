"""Linear transition world model (PRD Baseline 2), closed-form ridge regression.

Learns a shared per-node mapping X_{t+1}[v] ~= [X_t[v], 1] . W from every
(consecutive-window, co-present node) pair in the training sequence. Sharing W
across nodes lets the model generalise to nodes it has never seen individually
(the vertex set V_t changes each window), which is essential for a dynamic
graph. This is the numeric stand-in for the GraphSAGE+GRU encoder that replaces
it in the full stack.
"""

from __future__ import annotations

from collections.abc import Sequence

from sentinelx.graph.types import GraphState
from sentinelx.linalg import Matrix, matvec, ridge_fit, transpose
from sentinelx.models.base import WorldModel, apply_dropout
from sentinelx.seeding import Rng


class LinearTransitionModel(WorldModel):
    name = "linear_transition"

    def __init__(self, ridge_lambda: float = 0.05):
        self.ridge_lambda = ridge_lambda
        self.W: Matrix = []          # (in_dim+1) x out_dim
        self._wt: Matrix = []        # cached transpose for fast matvec
        self.in_dim = 0
        self.out_dim = 0
        self._fitted = False

    def fit(self, train_graphs: Sequence[GraphState]) -> LinearTransitionModel:
        xs: list[list[float]] = []
        ys: list[list[float]] = []
        for g_t, g_next in zip(train_graphs, list(train_graphs)[1:]):
            for key, node in g_t.nodes.items():
                nxt = g_next.nodes.get(key)
                if nxt is None:
                    continue
                xs.append(list(node.features) + [1.0])  # bias term
                ys.append(list(nxt.features))
        if len(xs) < 2:
            self._fitted = False
            return self
        self.in_dim = len(xs[0])
        self.out_dim = len(ys[0])
        self.W = ridge_fit(xs, ys, self.ridge_lambda)
        self._wt = transpose(self.W)  # out_dim x in_dim  -> matvec(_wt, x) = x.W
        self._fitted = True
        return self

    def predict_next(
        self, history: Sequence[GraphState], dropout: float = 0.0, rng: Rng | None = None
    ) -> GraphState:
        if not history:
            raise ValueError("LinearTransitionModel.predict_next requires non-empty history")
        last = history[-1]
        pred = self._skeleton_from_last(last)
        for key, node in last.nodes.items():
            feats = apply_dropout(node.features, dropout, rng)
            if self._fitted and len(feats) + 1 == self.in_dim:
                x = list(feats) + [1.0]
                y = matvec(self._wt, x)  # out_dim vector
            else:
                y = list(feats)  # graceful fallback to persistence
            self._set_features(pred, key, y)
        return pred
