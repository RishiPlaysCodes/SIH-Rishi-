"""Construct dynamic graph snapshots from windowed flows.

Each :class:`WindowSlice` becomes one :class:`GraphState`:

* nodes = every host/server observed in the window, with the behavioural
  feature vector from :func:`window_node_features`;
* edges = one aggregated directed edge per (src, dst) pair, carrying summed
  traffic features and a weight (total packets);
* servers are flagged heuristically (explicit ``SERVER-*`` key, or high inbound
  fan-in) so the risk/propagation layers know which assets are "crown jewels".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sentinelx.data.features import WindowSlice, build_windows, window_node_features
from sentinelx.data.schema import EDGE_FEATURES, NODE_FEATURES, FlowRecord
from sentinelx.graph.types import EdgeState, GraphState, NodeState

_SERVER_FANIN_THRESHOLD = 5


def _detect_servers(flows: Sequence[FlowRecord], nodes: Sequence[str]) -> set[str]:
    fan_in: dict[str, set] = {n: set() for n in nodes}
    for f in flows:
        fan_in.setdefault(f.dst, set()).add(f.src)
    servers = set()
    for n in nodes:
        if n.upper().startswith("SERVER") or len(fan_in.get(n, set())) >= _SERVER_FANIN_THRESHOLD:
            servers.add(n)
    return servers


def build_graph_state(window: WindowSlice, min_edge_weight: float = 1.0) -> GraphState:
    node_features = window_node_features(window.flows)
    nodes = sorted(node_features.keys())
    servers = _detect_servers(window.flows, nodes)

    graph = GraphState(
        index=window.index,
        window_start=window.start,
        window_end=window.end,
        node_feature_names=list(NODE_FEATURES),
        edge_feature_names=list(EDGE_FEATURES),
    )
    for n in nodes:
        graph.nodes[n] = NodeState(
            key=n,
            label=n,
            features=node_features[n],
            status="normal",
            is_server=n in servers,
        )

    # Aggregate flows into directed edges.
    agg: dict[tuple[str, str], dict] = {}
    for f in window.flows:
        key = (f.src, f.dst)
        bucket = agg.setdefault(
            key,
            {"packets": 0, "bytes": 0, "duration": 0.0, "protocols": Counter(), "ports": Counter()},
        )
        bucket["packets"] += f.packets
        bucket["bytes"] += f.bytes
        bucket["duration"] += f.duration
        bucket["protocols"][f.protocol] += 1
        bucket["ports"][f.dst_port] += 1

    for (src, dst), b in agg.items():
        weight = float(b["packets"])
        if weight < min_edge_weight:
            continue
        duration = b["duration"] if b["duration"] > 0 else 1e-6
        protocol = b["protocols"].most_common(1)[0][0]
        dst_port = b["ports"].most_common(1)[0][0]
        edge_feat = [
            float(b["packets"]),
            float(b["bytes"]),
            float(b["duration"]),
            b["packets"] / duration,
            b["bytes"] / duration,
        ]
        graph.edges.append(
            EdgeState(
                src=src, dst=dst, protocol=protocol, features=edge_feat,
                weight=weight, dst_port=dst_port,
            )
        )
    return graph


def build_graph_sequence(
    flows: Sequence[FlowRecord], window_seconds: int, min_edge_weight: float = 1.0
) -> list[GraphState]:
    windows = build_windows(flows, window_seconds)
    return [build_graph_state(w, min_edge_weight=min_edge_weight) for w in windows]
