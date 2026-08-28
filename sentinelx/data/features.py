"""Time-windowing and per-node feature engineering.

Flows are bucketed into fixed-width time windows; each window yields one graph
snapshot. For every node we compute the behavioural + temporal feature vector
declared in :data:`sentinelx.data.schema.NODE_FEATURES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from sentinelx.data.schema import NODE_FEATURES, FlowRecord
from sentinelx.linalg import Vector


@dataclass
class WindowSlice:
    index: int
    start: float
    end: float
    flows: List[FlowRecord]


def build_windows(flows: Sequence[FlowRecord], window_seconds: int) -> List[WindowSlice]:
    """Group time-sorted flows into contiguous fixed-width windows.

    Empty interior windows are preserved (as empty slices) so the temporal
    sequence has no silent gaps — important for a forecasting model.
    """
    if not flows:
        return []
    ordered = sorted(flows, key=lambda f: f.ts)
    # Anchor windows to the time grid (a multiple of window_seconds) rather than
    # to the exact first-flow timestamp. This keeps window boundaries aligned to
    # meaningful clock intervals and ensures a data source that emits flows on a
    # ``w * window_seconds`` schedule maps 1:1 onto window indices.
    t_min = (ordered[0].ts // window_seconds) * window_seconds
    t_max = ordered[-1].ts
    num_windows = int((t_max - t_min) // window_seconds) + 1
    buckets: List[List[FlowRecord]] = [[] for _ in range(num_windows)]
    for f in ordered:
        idx = int((f.ts - t_min) // window_seconds)
        idx = min(idx, num_windows - 1)
        buckets[idx].append(f)
    slices: List[WindowSlice] = []
    for i, bucket in enumerate(buckets):
        start = t_min + i * window_seconds
        slices.append(WindowSlice(index=i, start=start, end=start + window_seconds, flows=bucket))
    return slices


def window_node_features(flows: Sequence[FlowRecord]) -> Dict[str, Vector]:
    """Compute the fixed-order node feature vector for every node in a window."""
    outbound: Dict[str, List[FlowRecord]] = {}
    inbound_bytes: Dict[str, int] = {}
    all_nodes: set[str] = set()

    for f in flows:
        all_nodes.add(f.src)
        all_nodes.add(f.dst)
        outbound.setdefault(f.src, []).append(f)
        inbound_bytes[f.dst] = inbound_bytes.get(f.dst, 0) + f.bytes

    features: Dict[str, Vector] = {}
    for node in all_nodes:
        out_flows = outbound.get(node, [])
        out_flows_sorted = sorted(out_flows, key=lambda f: f.ts)

        connection_frequency = float(len(out_flows))
        unique_destinations = float(len({f.dst for f in out_flows}))
        unique_ports = float(len({f.dst_port for f in out_flows}))
        failed_connections = float(sum(1 for f in out_flows if f.failed))

        out_bytes = sum(f.bytes for f in out_flows)
        in_bytes = inbound_bytes.get(node, 0)
        total_bytes = out_bytes + in_bytes
        outbound_ratio = (out_bytes / total_bytes) if total_bytes > 0 else 0.0

        if out_flows:
            mean_packet_rate = sum(f.packet_rate for f in out_flows) / len(out_flows)
            mean_byte_rate = sum(f.byte_rate for f in out_flows) / len(out_flows)
        else:
            mean_packet_rate = 0.0
            mean_byte_rate = 0.0

        if len(out_flows_sorted) >= 2:
            gaps = [
                out_flows_sorted[i + 1].ts - out_flows_sorted[i].ts
                for i in range(len(out_flows_sorted) - 1)
            ]
            mean_iat = sum(gaps) / len(gaps)
        else:
            mean_iat = 0.0

        vec = [
            connection_frequency,
            unique_destinations,
            unique_ports,
            failed_connections,
            outbound_ratio,
            mean_packet_rate,
            mean_byte_rate,
            mean_iat,
        ]
        assert len(vec) == len(NODE_FEATURES)
        features[node] = vec
    return features
