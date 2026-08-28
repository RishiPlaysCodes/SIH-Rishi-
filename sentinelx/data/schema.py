"""Canonical flow record + feature name registries.

``FlowRecord`` is the single normalised representation every ingestion path
(synthetic generator, CIC-IDS2018 CSV, PCAP-via-Scapy later) is mapped onto, so
downstream code never depends on a specific dataset's column layout.
"""

from __future__ import annotations

from dataclasses import dataclass


# Node-level (behavioural + temporal) feature vector, fixed order.
NODE_FEATURES = [
    "connection_frequency",   # number of outbound flows in the window
    "unique_destinations",    # distinct destination hosts contacted
    "unique_ports",           # distinct destination ports contacted
    "failed_connections",     # flows with no response (bwd_packets == 0)
    "outbound_ratio",         # outbound_bytes / (inbound + outbound)
    "mean_packet_rate",       # mean packets per second across the node's flows
    "mean_byte_rate",         # mean bytes per second across the node's flows
    "mean_iat",               # mean inter-arrival time between the node's flows
]

# Edge-level feature vector, fixed order.
EDGE_FEATURES = [
    "packets",
    "bytes",
    "duration",
    "packet_rate",
    "byte_rate",
]


@dataclass
class FlowRecord:
    """A single bidirectional network flow, normalised across sources."""

    ts: float                 # flow start time (epoch seconds)
    src: str                  # source asset identifier (pre-hash)
    dst: str                  # destination asset identifier (pre-hash)
    src_port: int
    dst_port: int
    protocol: str             # 'TCP' | 'UDP' | 'ICMP' | ...
    duration: float           # flow duration in seconds
    fwd_packets: int
    bwd_packets: int
    fwd_bytes: int
    bwd_bytes: int
    label: str = "Benign"     # dataset label, kept for evaluation only

    # ---- derived views ---------------------------------------------------- #
    @property
    def packets(self) -> int:
        return self.fwd_packets + self.bwd_packets

    @property
    def bytes(self) -> int:
        return self.fwd_bytes + self.bwd_bytes

    @property
    def packet_rate(self) -> float:
        return self.packets / self.duration if self.duration > 0 else float(self.packets)

    @property
    def byte_rate(self) -> float:
        return self.bytes / self.duration if self.duration > 0 else float(self.bytes)

    @property
    def failed(self) -> bool:
        """A connection attempt that received no response."""
        return self.bwd_packets == 0

    @property
    def is_attack(self) -> bool:
        return self.label.strip().lower() not in ("benign", "normal", "")
