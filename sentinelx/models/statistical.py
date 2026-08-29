"""Statistical baselines (PRD Baseline 1).

These are intentionally simple and *honest* references the learned models must
beat. If a GNN cannot outperform persistence, the GNN is not earning its keep —
exactly the kind of ablation the PRD's success metrics call for.
"""

from __future__ import annotations

from collections.abc import Sequence

from sentinelx.graph.types import GraphState
from sentinelx.models.base import WorldModel, apply_dropout
from sentinelx.seeding import Rng


class PersistenceModel(WorldModel):
    """Predict G_{t+1} = G_t (the "no change" baseline)."""

    name = "baseline_statistical"

    def fit(self, train_graphs: Sequence[GraphState]) -> PersistenceModel:
        return self

    def predict_next(
        self, history: Sequence[GraphState], dropout: float = 0.0, rng: Rng | None = None
    ) -> GraphState:
        if not history:
            raise ValueError("PersistenceModel.predict_next requires non-empty history")
        last = history[-1]
        pred = self._skeleton_from_last(last)
        for key, node in last.nodes.items():
            self._set_features(pred, key, apply_dropout(node.features, dropout, rng))
        return pred


class EWMAModel(WorldModel):
    """Exponentially-weighted moving average over each node's feature history."""

    name = "ewma"

    def __init__(self, alpha: float = 0.4):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("EWMA alpha must be in (0, 1]")
        self.alpha = alpha

    def fit(self, train_graphs: Sequence[GraphState]) -> EWMAModel:
        return self

    def predict_next(
        self, history: Sequence[GraphState], dropout: float = 0.0, rng: Rng | None = None
    ) -> GraphState:
        if not history:
            raise ValueError("EWMAModel.predict_next requires non-empty history")
        last = history[-1]
        pred = self._skeleton_from_last(last)
        dim = len(last.node_feature_names)
        for key in last.nodes:
            # Walk history oldest->newest, updating the EWMA for this node.
            ewma: list[float] | None = None
            for g in history:
                node = g.nodes.get(key)
                if node is None:
                    continue
                if ewma is None:
                    ewma = list(node.features)
                else:
                    ewma = [
                        self.alpha * f + (1.0 - self.alpha) * e
                        for f, e in zip(node.features, ewma)
                    ]
            if ewma is None:
                ewma = [0.0] * dim
            self._set_features(pred, key, apply_dropout(ewma, dropout, rng))
        return pred
