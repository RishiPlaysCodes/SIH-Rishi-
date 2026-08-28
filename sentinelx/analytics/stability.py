"""Forecast stability engine (PRD §2.12).

Slightly perturbs the most recent observed evidence, re-runs the forecast, and
measures how much the prediction moves. A forecast that swings wildly under tiny
input changes should not be trusted; a forecast that barely moves is robust.
The output is a stability score in [0, 1] (higher = more stable) plus a
STABLE / UNSTABLE label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from sentinelx.graph.types import GraphState
from sentinelx.linalg import euclidean
from sentinelx.models.base import WorldModel
from sentinelx.seeding import Rng


@dataclass
class StabilityResult:
    stability_score: float  # 1 = perfectly stable, 0 = highly sensitive
    label: str              # STABLE | UNSTABLE
    mean_delta: float       # raw mean output sensitivity


def _perturb(graph: GraphState, sigma: float, rng: Rng) -> GraphState:
    noisy = graph.clone()
    for node in noisy.nodes.values():
        node.features = [f + rng.gauss(0.0, sigma) for f in node.features]
    return noisy


def _forecast_delta(base: GraphState, other: GraphState) -> float:
    keys = set(base.nodes) & set(other.nodes)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        a = base.nodes[k].features
        b = other.nodes[k].features
        dim = len(a) or 1
        total += euclidean(a, b) / math.sqrt(dim)
    return total / len(keys)


def assess_stability(
    model: WorldModel,
    history: Sequence[GraphState],
    perturbation: float = 0.03,
    num_trials: int = 12,
    unstable_threshold: float = 0.12,
    rng: Rng | None = None,
) -> StabilityResult:
    if not history:
        raise ValueError("assess_stability requires non-empty history")
    rng = rng or Rng(0)
    baseline = model.predict_next(history)

    deltas: List[float] = []
    for t in range(num_trials):
        trial_rng = rng.spawn(f"stab-{t}")
        perturbed_last = _perturb(history[-1], perturbation, trial_rng)
        perturbed_history = list(history[:-1]) + [perturbed_last]
        pred = model.predict_next(perturbed_history)
        deltas.append(_forecast_delta(baseline, pred))

    mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
    stability_score = 1.0 / (1.0 + mean_delta)  # bounded (0, 1], 1 = stable
    label = "UNSTABLE" if mean_delta > unstable_threshold else "STABLE"
    return StabilityResult(stability_score=stability_score, label=label, mean_delta=mean_delta)
