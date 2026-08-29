"""Typed representations of a dynamic network graph state.

A :class:`GraphState` is the atomic unit the world model consumes and predicts.
Node identity is a stable ``node_key`` (a hashed asset identifier, never a raw
IP) so nodes can be *aligned across time windows* even as the vertex set
V_t changes from one window to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinelx.linalg import Vector, mean_vector


@dataclass
class NodeState:
    """A host/server within a snapshot."""

    key: str
    label: str
    features: Vector
    status: str = "normal"  # normal | deviating | anomalous
    is_server: bool = False


@dataclass
class EdgeState:
    """A directed connection between two nodes within a snapshot."""

    src: str
    dst: str
    protocol: str
    features: Vector
    weight: float = 1.0
    dst_port: int = 0  # dominant destination port on this aggregated edge


@dataclass
class GraphState:
    """A single time-window snapshot G_t = (V_t, E_t, X_t)."""

    index: int
    window_start: float
    window_end: float
    nodes: dict[str, NodeState] = field(default_factory=dict)
    edges: list[EdgeState] = field(default_factory=list)
    node_feature_names: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)

    # ---- convenience accessors -------------------------------------------- #
    def node_keys(self) -> list[str]:
        return sorted(self.nodes.keys())

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def feature_matrix(self, keys: list[str] | None = None) -> list[Vector]:
        """Node feature rows aligned to ``keys`` (defaults to sorted node keys)."""
        keys = keys if keys is not None else self.node_keys()
        dim = len(self.node_feature_names)
        rows: list[Vector] = []
        for k in keys:
            node = self.nodes.get(k)
            rows.append(list(node.features) if node else [0.0] * dim)
        return rows

    def edge_set(self) -> set[tuple[str, str]]:
        return {(e.src, e.dst) for e in self.edges}

    def adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {k: [] for k in self.nodes}
        for e in self.edges:
            adj.setdefault(e.src, []).append(e.dst)
        return adj

    def embedding(self) -> Vector:
        """A fixed-length graph-level summary used for novelty/OOD comparison.

        Concatenates the mean node-feature vector with a few scalar structural
        descriptors so two graphs of different sizes are still comparable.
        """
        rows = self.feature_matrix()
        mean_feat = mean_vector(rows) if rows else []
        n = float(self.node_count())
        e = float(self.edge_count())
        density = (e / (n * (n - 1))) if n > 1 else 0.0
        anomalous = sum(1 for nd in self.nodes.values() if nd.status == "anomalous")
        return list(mean_feat) + [n, e, density, float(anomalous)]

    def anomalous_keys(self) -> list[str]:
        return [k for k, nd in self.nodes.items() if nd.status == "anomalous"]

    def server_keys(self) -> list[str]:
        return [k for k, nd in self.nodes.items() if nd.is_server]

    def clone(self) -> GraphState:
        return GraphState(
            index=self.index,
            window_start=self.window_start,
            window_end=self.window_end,
            nodes={
                k: NodeState(v.key, v.label, list(v.features), v.status, v.is_server)
                for k, v in self.nodes.items()
            },
            edges=[
                EdgeState(e.src, e.dst, e.protocol, list(e.features), e.weight, e.dst_port)
                for e in self.edges
            ],
            node_feature_names=list(self.node_feature_names),
            edge_feature_names=list(self.edge_feature_names),
        )

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "node_feature_names": self.node_feature_names,
            "edge_feature_names": self.edge_feature_names,
            "nodes": [
                {
                    "key": nd.key,
                    "label": nd.label,
                    "features": nd.features,
                    "status": nd.status,
                    "is_server": nd.is_server,
                }
                for nd in (self.nodes[k] for k in self.node_keys())
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "protocol": e.protocol,
                    "features": e.features,
                    "weight": e.weight,
                    "dst_port": e.dst_port,
                }
                for e in self.edges
            ],
        }
