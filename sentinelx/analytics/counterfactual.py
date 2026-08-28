"""Counterfactual "what-if" intervention engine (PRD §2.13).

Sits *over* the world model rather than being a separate model: apply a
hypothetical defensive action to the current graph, re-run the forecast, and
compare a risk metric before vs after.

    Gₜ                      -> world model -> Forecast A -> Risk_A
    Gₜ + intervention -> Gₜ' -> world model -> Forecast B -> Risk_B
    ΔRisk = Risk_A - Risk_B   (positive => the action reduced risk)

Risk is computed structurally on the forecasted graph as the exposure of the
"crown-jewel" servers (and the wider estate) to the currently-compromised nodes,
via directed reachability weighted by predicted traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sentinelx.graph.types import GraphState
from sentinelx.models.base import WorldModel

VALID_ACTIONS = {
    "ISOLATE_NODE",
    "BLOCK_EDGE",
    "BLOCK_PORT",
    "DISABLE_COMMUNICATION",
    "RATE_LIMIT",
}


@dataclass
class Intervention:
    action_type: str
    target_node: Optional[str] = None
    target_edge: Optional[Tuple[str, str]] = None
    port: Optional[int] = None
    rate_factor: float = 0.2  # RATE_LIMIT: fraction of traffic retained

    def __post_init__(self):
        if self.action_type not in VALID_ACTIONS:
            raise ValueError(
                f"Unknown action_type={self.action_type!r}. Valid: {sorted(VALID_ACTIONS)}"
            )


@dataclass
class CounterfactualResult:
    action_type: str
    target: str
    risk_before: float
    risk_after: float
    delta_risk: float
    components_before: Dict[str, float] = field(default_factory=dict)
    components_after: Dict[str, float] = field(default_factory=dict)


def apply_intervention(graph: GraphState, iv: Intervention) -> GraphState:
    """Return a copy of ``graph`` with the intervention applied to its structure."""
    g = graph.clone()
    if iv.action_type == "ISOLATE_NODE" and iv.target_node:
        g.nodes.pop(iv.target_node, None)
        g.edges = [e for e in g.edges if e.src != iv.target_node and e.dst != iv.target_node]
    elif iv.action_type == "DISABLE_COMMUNICATION" and iv.target_node:
        g.edges = [e for e in g.edges if e.src != iv.target_node and e.dst != iv.target_node]
    elif iv.action_type == "BLOCK_EDGE" and iv.target_edge:
        s, d = iv.target_edge
        g.edges = [e for e in g.edges if not (e.src == s and e.dst == d)]
    elif iv.action_type == "BLOCK_PORT" and iv.port is not None:
        g.edges = [
            e
            for e in g.edges
            if not (e.dst_port == iv.port and (iv.target_node is None or e.dst == iv.target_node))
        ]
    elif iv.action_type == "RATE_LIMIT" and iv.target_node:
        factor = max(0.0, min(1.0, iv.rate_factor))
        for e in g.edges:
            if e.src == iv.target_node:
                e.weight *= factor
                e.features = [f * factor for f in e.features]
    return g


def _reachable(graph: GraphState, sources: Set[str]) -> Set[str]:
    """Directed BFS reachable set from ``sources`` (excluding the sources)."""
    adj = graph.adjacency()
    seen: Set[str] = set()
    frontier: List[str] = [s for s in sources if s in graph.nodes]
    while frontier:
        cur = frontier.pop()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen - set(sources)


def compute_risk(graph: GraphState, compromised: Set[str]) -> Dict[str, float]:
    """Structural risk components in [0, 1] for a (forecasted) graph."""
    servers = set(graph.server_keys())
    total_nodes = max(graph.node_count() - len(compromised), 1)
    reachable = _reachable(graph, compromised)

    reachable_servers = reachable & servers
    server_risk = (len(reachable_servers) / len(servers)) if servers else 0.0
    lateral_risk = len(reachable) / total_nodes

    # Alternate-path risk: fraction of servers reachable via >1 distinct
    # compromised entry, i.e. resilience of the attack to a single block.
    alt = 0.0
    if servers:
        multi = 0
        for srv in servers:
            entries = sum(1 for c in compromised if srv in _reachable(graph, {c}))
            if entries > 1:
                multi += 1
        alt = multi / len(servers)

    overall = min(1.0, 0.6 * server_risk + 0.4 * lateral_risk)
    return {
        "db_risk": server_risk,
        "lateral_movement_risk": lateral_risk,
        "alternate_path_risk": alt,
        "overall": overall,
    }


def run_counterfactual(
    model: WorldModel,
    history: Sequence[GraphState],
    intervention: Intervention,
    compromised: Sequence[str],
    horizon: int = 3,
) -> CounterfactualResult:
    if not history:
        raise ValueError("run_counterfactual requires non-empty history")
    compromised_set = set(compromised)

    forecast_a = model.predict_sequence(history, horizon)
    modified_last = apply_intervention(history[-1], intervention)
    history_b = list(history[:-1]) + [modified_last]
    forecast_b = model.predict_sequence(history_b, horizon)

    # Evaluate risk on the final forecasted horizon step.
    risk_a = compute_risk(forecast_a[-1], compromised_set)
    # After an ISOLATE/DISABLE the compromised node may be gone; that is the point.
    risk_b = compute_risk(forecast_b[-1], compromised_set)

    target = intervention.target_node or (
        f"{intervention.target_edge}" if intervention.target_edge else f"port:{intervention.port}"
    )
    return CounterfactualResult(
        action_type=intervention.action_type,
        target=str(target),
        risk_before=risk_a["overall"],
        risk_after=risk_b["overall"],
        delta_risk=risk_a["overall"] - risk_b["overall"],
        components_before=risk_a,
        components_after=risk_b,
    )
