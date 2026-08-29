"""Uncertainty engine via MC-Dropout (PRD §2.10).

Runs N stochastic forward passes of the world model (input feature dimensions
randomly masked with inverted-dropout scaling) and summarises the spread of the
predictions into a mean, a standard deviation, and a LOW / MEDIUM / HIGH
confidence label. This is the honesty layer the PRD insists on: every forecast
is reported *with* its uncertainty, never as a bare point estimate. Deep
ensembles / conformal prediction are the documented future comparison points.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sentinelx.graph.types import GraphState
from sentinelx.linalg import mean_vector, std_vector
from sentinelx.models.base import WorldModel
from sentinelx.seeding import Rng


@dataclass
class UncertaintyResult:
    mean_prediction: float
    std_dev: float
    label: str  # LOW | MEDIUM | HIGH
    per_node_sigma: dict[str, float] = field(default_factory=dict)


def _label(sigma: float, low: float, high: float) -> str:
    if sigma <= low:
        return "LOW"
    if sigma >= high:
        return "HIGH"
    return "MEDIUM"


def estimate_uncertainty(
    model: WorldModel,
    history: Sequence[GraphState],
    num_passes: int = 30,
    dropout: float = 0.2,
    rng: Rng | None = None,
    low_sigma: float = 0.30,
    high_sigma: float = 0.50,
) -> UncertaintyResult:
    if not history:
        raise ValueError("estimate_uncertainty requires non-empty history")
    if num_passes < 2:
        raise ValueError("num_passes must be >= 2 to estimate variance")
    rng = rng or Rng(0)

    # Collect, per node, the stack of predicted feature vectors across passes.
    stacks: dict[str, list[list[float]]] = {}
    for p in range(num_passes):
        pass_rng = rng.spawn(f"mc-{p}")
        pred = model.predict_next(history, dropout=dropout, rng=pass_rng)
        for key, node in pred.nodes.items():
            stacks.setdefault(key, []).append(list(node.features))

    per_node_sigma: dict[str, float] = {}
    node_means: list[float] = []
    for key, rows in stacks.items():
        sigma_vec = std_vector(rows)
        mean_vec = mean_vector(rows)
        per_node_sigma[key] = sum(sigma_vec) / len(sigma_vec) if sigma_vec else 0.0
        if mean_vec:
            node_means.append(sum(mean_vec) / len(mean_vec))

    graph_sigma = (
        sum(per_node_sigma.values()) / len(per_node_sigma) if per_node_sigma else 0.0
    )
    mean_prediction = sum(node_means) / len(node_means) if node_means else 0.0
    return UncertaintyResult(
        mean_prediction=mean_prediction,
        std_dev=graph_sigma,
        label=_label(graph_sigma, low_sigma, high_sigma),
        per_node_sigma=per_node_sigma,
    )
