"""Behavioural Deviation Score: D_t = d(G_t, Ĝ_t).

Compares the *observed* graph against the graph the world model *predicted* for
the same window, decomposed into interpretable components (PRD §2.8):

    feature_pred_error   per-node MSE between predicted and observed features
    node_state_error     per-node normalised euclidean feature error
    structural_error     Jaccard distance between predicted / observed edge sets
    edge_state_error     normalised edge-weight disagreement
    temporal_error       error in the *magnitude of change* vs the previous graph

These are combined with configurable weights into a per-node score in [0, 1],
then thresholded into normal / deviating / anomalous. All components are bounded
in [0, 1] so the weighted sum is interpretable and comparable across windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sentinelx.graph.types import GraphState
from sentinelx.linalg import euclidean

# Feature-error aggregation. An attack typically spikes ONE or TWO features hard
# (e.g. unique_destinations, failed_connections) while the rest stay normal, so
# a pure mean over dimensions dilutes the signal. We blend the RMS residual with
# the single worst dimension's absolute residual (in z-score space).
_RMS_WEIGHT = 0.4
_MAX_WEIGHT = 0.6

# Saturating scale constants (in units of standardised residual). ``saturate``
# maps an unbounded non-negative residual to [0, 1) so the final score stays
# interpretable regardless of how far out-of-distribution the input is. The
# constants set "how many standard deviations of forecast error counts as fully
# anomalous" for each component.
_SAT_FEATURE = 2.5
_SAT_NODE = 3.5
_SAT_TEMPORAL = 3.0


def saturate(x: float, scale: float) -> float:
    """Monotonic map [0, inf) -> [0, 1):  x / (x + scale)."""
    x = abs(x)
    return x / (x + scale) if (x + scale) > 0 else 0.0

DEFAULT_WEIGHTS = {
    "feature": 0.5,
    "node_state": 0.18,
    "temporal": 0.15,
    "structural": 0.10,
    "edge_state": 0.07,
}


@dataclass
class NodeDeviation:
    key: str
    deviation_score: float
    feature_pred_error: float
    node_state_error: float
    structural_error: float
    edge_state_error: float
    temporal_error: float
    status: str = "normal"


@dataclass
class DeviationResult:
    graph_index: int
    graph_score: float
    structural_error: float
    edge_state_error: float
    per_node: Dict[str, NodeDeviation] = field(default_factory=dict)

    def anomalous_keys(self) -> List[str]:
        return [k for k, d in self.per_node.items() if d.status == "anomalous"]

    def deviating_keys(self) -> List[str]:
        return [k for k, d in self.per_node.items() if d.status == "deviating"]


def _rms(residuals) -> float:
    if not residuals:
        return 0.0
    return math.sqrt(sum(r * r for r in residuals) / len(residuals))


def _feature_error(pred, actual) -> float:
    """Bounded blend of RMS and worst-dimension residual (z-score space)."""
    if not pred:
        return 0.0
    resid = [p - a for p, a in zip(pred, actual)]
    rms = _rms(resid)
    max_abs = max(abs(r) for r in resid)
    return _RMS_WEIGHT * saturate(rms, _SAT_FEATURE) + _MAX_WEIGHT * saturate(max_abs, _SAT_FEATURE)


def _node_state_error(pred, actual) -> float:
    if not pred:
        return 0.0
    return saturate(euclidean(pred, actual), _SAT_NODE)


def _structural_error(predicted: GraphState, actual: GraphState) -> float:
    pe, ae = predicted.edge_set(), actual.edge_set()
    union = pe | ae
    if not union:
        return 0.0
    return 1.0 - (len(pe & ae) / len(union))


def _edge_state_error(predicted: GraphState, actual: GraphState) -> float:
    pw = {(e.src, e.dst): e.weight for e in predicted.edges}
    aw = {(e.src, e.dst): e.weight for e in actual.edges}
    keys = set(pw) | set(aw)
    if not keys:
        return 0.0
    max_w = max([abs(v) for v in list(pw.values()) + list(aw.values())] + [1.0])
    total = sum(abs(pw.get(k, 0.0) - aw.get(k, 0.0)) / max_w for k in keys)
    return min(1.0, total / len(keys))


def compute_deviation(
    predicted: GraphState,
    actual: GraphState,
    previous: Optional[GraphState] = None,
    weights: Optional[Dict[str, float]] = None,
    anomaly_threshold: float = 0.55,
    deviating_threshold: float = 0.35,
) -> DeviationResult:
    w = weights or DEFAULT_WEIGHTS
    structural_error = _structural_error(predicted, actual)
    edge_state_error = _edge_state_error(predicted, actual)

    per_node: Dict[str, NodeDeviation] = {}
    scored_keys = set(actual.nodes) & set(predicted.nodes)
    for key in scored_keys:
        p_feat = predicted.nodes[key].features
        a_feat = actual.nodes[key].features
        feature_err = _feature_error(p_feat, a_feat)
        node_state_err = _node_state_error(p_feat, a_feat)

        temporal_err = 0.0
        if previous is not None and key in previous.nodes:
            prev_feat = previous.nodes[key].features
            actual_delta = euclidean(a_feat, prev_feat)
            pred_delta = euclidean(p_feat, prev_feat)
            temporal_err = saturate(abs(actual_delta - pred_delta), _SAT_TEMPORAL)

        score = (
            w.get("feature", 0.0) * feature_err
            + w.get("node_state", 0.0) * node_state_err
            + w.get("structural", 0.0) * structural_error
            + w.get("edge_state", 0.0) * edge_state_error
            + w.get("temporal", 0.0) * temporal_err
        )
        score = max(0.0, min(1.0, score))
        status = (
            "anomalous"
            if score >= anomaly_threshold
            else "deviating"
            if score >= deviating_threshold
            else "normal"
        )
        per_node[key] = NodeDeviation(
            key=key,
            deviation_score=score,
            feature_pred_error=feature_err,
            node_state_error=node_state_err,
            structural_error=structural_error,
            edge_state_error=edge_state_error,
            temporal_error=temporal_err,
            status=status,
        )

    graph_score = (
        sum(d.deviation_score for d in per_node.values()) / len(per_node) if per_node else 0.0
    )
    return DeviationResult(
        graph_index=actual.index,
        graph_score=graph_score,
        structural_error=structural_error,
        edge_state_error=edge_state_error,
        per_node=per_node,
    )
