"""Forecasting engine: K-step rollouts and rolling one-step deviation scoring.

Wraps a fitted :class:`~sentinelx.models.base.WorldModel` and provides the two
operations the rest of the platform needs:

* :meth:`forecast` — produce Ĝ_{t+1..t+k} from an observed history (Forecast /
  Counterfactual views);
* :meth:`rolling_deviation` — for each window t, predict G_t from the strictly
  earlier history and score the observed vs predicted graph (Network / anomaly
  views). This is strictly causal (no future leakage into any prediction).
"""

from __future__ import annotations

from collections.abc import Sequence

from sentinelx.forecast.deviation import DeviationResult, compute_deviation
from sentinelx.graph.types import GraphState
from sentinelx.models.base import WorldModel
from sentinelx.seeding import Rng


class ForecastEngine:
    def __init__(
        self,
        model: WorldModel,
        weights: dict[str, float] | None = None,
        anomaly_threshold: float = 0.55,
        deviating_threshold: float = 0.35,
    ):
        self.model = model
        self.weights = weights
        self.anomaly_threshold = anomaly_threshold
        self.deviating_threshold = deviating_threshold

    def forecast(
        self,
        history: Sequence[GraphState],
        k: int,
        dropout: float = 0.0,
        rng: Rng | None = None,
    ) -> list[GraphState]:
        if k < 1:
            raise ValueError("Forecast horizon k must be >= 1")
        return self.model.predict_sequence(history, k, dropout=dropout, rng=rng)

    def score_window(self, graphs: Sequence[GraphState], t: int) -> DeviationResult:
        """Predict graph ``t`` from ``graphs[:t]`` and score against the truth."""
        if t < 1 or t >= len(graphs):
            raise IndexError("score_window requires 1 <= t < len(graphs)")
        predicted = self.model.predict_next(graphs[:t])
        return compute_deviation(
            predicted=predicted,
            actual=graphs[t],
            previous=graphs[t - 1],
            weights=self.weights,
            anomaly_threshold=self.anomaly_threshold,
            deviating_threshold=self.deviating_threshold,
        )

    def rolling_deviation(self, graphs: Sequence[GraphState]) -> list[DeviationResult]:
        results: list[DeviationResult] = []
        for t in range(1, len(graphs)):
            results.append(self.score_window(graphs, t))
        return results

    @staticmethod
    def apply_statuses(graph: GraphState, result: DeviationResult) -> None:
        """Write per-node deviation statuses back onto a graph snapshot."""
        for key, node in graph.nodes.items():
            dev = result.per_node.get(key)
            node.status = dev.status if dev else "normal"
