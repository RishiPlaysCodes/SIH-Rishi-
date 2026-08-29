"""Cyber-epidemiology / propagation layer (PRD §2.9).

Treats anomalous behaviour like an infection spreading over the graph: a node
that is anomalous at window t-1 and has an edge to a node that *becomes*
anomalous at window t is treated as having propagated to it. From these
transitions we derive:

    propagation_velocity              newly-affected nodes per second
    propagation_intensity             mean deviation of the newly-affected nodes
    effective_reproduction_number Rₑ  secondary infections per prior infection

These are descriptive signals surfaced to the analyst; per the PRD's honesty
principle they are reported, not assumed to improve forecasting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sentinelx.forecast.deviation import DeviationResult
from sentinelx.graph.types import GraphState


@dataclass
class PropagationEvent:
    window_index: int
    source: str
    target: str
    propagation_velocity: float
    propagation_intensity: float
    effective_reproduction_number: float


def _infected(dev: DeviationResult) -> dict[str, float]:
    """Map of anomalous-node -> deviation score for a window."""
    return {
        k: d.deviation_score
        for k, d in dev.per_node.items()
        if d.status in ("anomalous", "deviating")
    }


def compute_propagation(
    graphs: Sequence[GraphState],
    dev_by_index: dict[int, DeviationResult],
    window_seconds: float,
) -> list[PropagationEvent]:
    """Detect propagation events across a scored graph sequence."""
    events: list[PropagationEvent] = []
    dt = window_seconds if window_seconds > 0 else 1.0
    by_index = {g.index: g for g in graphs}

    ordered = sorted(dev_by_index.keys())
    for t in ordered:
        if (t - 1) not in dev_by_index:
            continue
        prev_dev = dev_by_index[t - 1]
        curr_dev = dev_by_index[t]
        prev_inf = _infected(prev_dev)
        curr_inf = _infected(curr_dev)
        newly = {k: v for k, v in curr_inf.items() if k not in prev_inf}
        if not newly or not prev_inf:
            continue

        velocity = len(newly) / dt
        re = len(newly) / max(len(prev_inf), 1)
        intensity = sum(newly.values()) / len(newly)

        # Attribute each newly-infected node to an infected upstream neighbour.
        graph = by_index.get(t) or by_index.get(t - 1)
        if graph is None:
            continue
        adj = graph.adjacency()
        for src in prev_inf:
            for tgt in adj.get(src, []):
                if tgt in newly:
                    events.append(
                        PropagationEvent(
                            window_index=t,
                            source=src,
                            target=tgt,
                            propagation_velocity=velocity,
                            propagation_intensity=intensity,
                            effective_reproduction_number=re,
                        )
                    )
    return events
