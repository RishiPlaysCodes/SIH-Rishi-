"""Deterministic synthetic network traffic generator.

Produces a stream of :class:`FlowRecord` spanning many time windows so the full
pipeline runs end-to-end without the (multi-GB, download-required) CIC-IDS2018
dataset.

Design rationale: a forecasting anomaly detector only works if *benign* traffic
is **predictable**. Real enterprise hosts are far more regular than uniform
noise — each workstation repeatedly talks to the same handful of servers on the
same ports with similar volumes. So every host is given a stable *profile* and
benign windows only jitter mildly around it. This lets the world model learn a
tight baseline; the injected attack then breaks it sharply, producing a clean,
honest separation between benign and malicious forecast error.

Attack scenarios (begin at ``attack_start_window``):
    lateral_movement  a compromised host fans out to many hosts/ports with many
                      failed (no-response) connections, then pivots to a server.
    exfiltration      a host opens a ramping high-byte-rate channel to a server.
"""

from __future__ import annotations

from typing import Dict, List

from sentinelx.data.schema import FlowRecord
from sentinelx.seeding import Rng


def _host_id(i: int) -> str:
    return f"HOST-{i:02d}"


def _server_id(i: int) -> str:
    return f"SERVER-{i:02d}"


class _HostProfile:
    """Stable per-host behavioural fingerprint for benign traffic."""

    def __init__(self, rng: Rng, servers: List[str]):
        self.flows_per_window = rng.randint(4, 7)
        self.home_servers = rng.sample(servers, k=min(2, len(servers)))
        self.port = rng.choice([80, 443, 53])
        self.fwd_pkts = rng.randint(12, 24)
        self.bwd_pkts = rng.randint(12, 24)
        self.fwd_size = rng.randint(60, 100)
        self.bwd_size = rng.randint(80, 160)
        self.duration = rng.uniform(1.0, 2.0)


def generate_synthetic_flows(
    num_windows: int = 40,
    window_seconds: int = 60,
    num_hosts: int = 18,
    num_servers: int = 4,
    attack_start_window: int = 30,
    attack_type: str = "lateral_movement",
    seed: int = 1337,
) -> List[FlowRecord]:
    """Generate a reproducible list of flows sorted by timestamp."""
    rng = Rng(seed).spawn("synthetic")
    hosts = [_host_id(i) for i in range(num_hosts)]
    servers = [_server_id(i) for i in range(num_servers)]
    scan_ports = [21, 22, 23, 135, 139, 445, 3389, 8080]
    profiles: Dict[str, _HostProfile] = {h: _HostProfile(rng, servers) for h in hosts}
    flows: List[FlowRecord] = []

    # A compromise CHAIN: the primary attacker infects a secondary host, which
    # later infects a tertiary host. Each compromised host begins its own
    # fan-out with a staggered start, so the propagation engine can observe the
    # infection actually spreading across the graph (attacker -> B -> C).
    chain = rng.sample(hosts, k=min(3, num_hosts))
    chain_starts = [0, 3, 6]  # windows after attack_start_window each activates

    for w in range(num_windows):
        t0 = float(w * window_seconds)
        for h in hosts:
            _benign_host_window(flows, rng, h, profiles[h], t0, window_seconds)

        if w < attack_start_window:
            continue
        if attack_type == "lateral_movement":
            for idx, actor in enumerate(chain):
                start = attack_start_window + chain_starts[idx]
                if w < start:
                    continue
                progress = w - start
                # Force the actor to also touch the next chain member, so the
                # infection edge exists in the graph for attribution.
                forced = chain[idx + 1] if idx + 1 < len(chain) else None
                _inject_lateral_movement(
                    flows, rng, actor, hosts, servers, scan_ports, t0,
                    window_seconds, progress, forced_target=forced,
                )
        elif attack_type == "exfiltration":
            _inject_exfiltration(
                flows, rng, chain[0], servers, t0, window_seconds, w - attack_start_window
            )

    flows.sort(key=lambda f: f.ts)
    return flows


def _jitter(rng: Rng, base: float, frac: float = 0.08) -> float:
    """Small multiplicative jitter around a base value (kept positive)."""
    return max(0.0, base * (1.0 + rng.gauss(0.0, frac)))


def _benign_host_window(flows, rng, host, prof: "_HostProfile", t0, wsec) -> None:
    n_flows = max(1, round(_jitter(rng, prof.flows_per_window, 0.12)))
    spacing = wsec / (n_flows + 1)
    for j in range(n_flows):
        dst = rng.choice(prof.home_servers)
        ts = t0 + spacing * (j + 1) + rng.gauss(0.0, spacing * 0.1)
        ts = min(max(ts, t0), t0 + wsec - 1)
        fwd_p = max(1, round(_jitter(rng, prof.fwd_pkts)))
        bwd_p = max(1, round(_jitter(rng, prof.bwd_pkts)))
        flows.append(
            FlowRecord(
                ts=ts,
                src=host,
                dst=dst,
                src_port=rng.randint(1024, 65535),
                dst_port=prof.port,
                protocol="TCP",
                duration=_jitter(rng, prof.duration) + 0.05,
                fwd_packets=fwd_p,
                bwd_packets=bwd_p,
                fwd_bytes=fwd_p * prof.fwd_size,
                bwd_bytes=bwd_p * prof.bwd_size,
                label="Benign",
            )
        )


def _inject_lateral_movement(
    flows, rng, attacker, hosts, servers, ports, t0, wsec, progress, forced_target=None
):
    # Fan-out grows over time: the attacker probes more hosts/ports each window.
    scan_targets = min(len(hosts) - 1, 5 + progress * 3)
    victims = [h for h in rng.sample(hosts, k=min(scan_targets + 1, len(hosts))) if h != attacker]
    victims = victims[:scan_targets]
    # Guarantee the infection edge to the next chain member exists early.
    if forced_target and forced_target != attacker and forced_target not in victims:
        victims.append(forced_target)
    for v in victims:
        responded = rng.random() < 0.2  # most scans get no response -> failed
        bwd_p = rng.randint(1, 3) if responded else 0
        flows.append(
            FlowRecord(
                ts=t0 + rng.uniform(0, wsec - 1),
                src=attacker,
                dst=v,
                src_port=rng.randint(1024, 65535),
                dst_port=rng.choice(ports),
                protocol="TCP",
                duration=abs(rng.gauss(0.15, 0.05)) + 0.01,
                fwd_packets=rng.randint(1, 3),
                bwd_packets=bwd_p,
                fwd_bytes=rng.randint(40, 100),
                bwd_bytes=bwd_p * rng.randint(40, 80),
                label="LateralMovement",
            )
        )
    # After a couple of windows, the attacker pivots toward a server.
    if progress >= 2:
        flows.append(
            FlowRecord(
                ts=t0 + rng.uniform(0, wsec - 1),
                src=attacker,
                dst=servers[0],
                src_port=rng.randint(1024, 65535),
                dst_port=445,
                protocol="TCP",
                duration=abs(rng.gauss(3.0, 1.0)) + 0.1,
                fwd_packets=rng.randint(80, 160),
                bwd_packets=rng.randint(80, 160),
                fwd_bytes=rng.randint(8000, 20000),
                bwd_bytes=rng.randint(8000, 20000),
                label="LateralMovement",
            )
        )


def _inject_exfiltration(flows, rng, attacker, servers, t0, wsec, progress):
    # Bulk exfiltration is a *sustained* high-byte-rate channel, not one packet:
    # emit several large outbound flows per window so the host's aggregate
    # byte-rate / outbound ratio clearly break from its benign profile.
    target = servers[-1]
    n_flows = 3 + progress
    for _ in range(n_flows):
        volume = 80000 * (1 + progress) + rng.randint(0, 20000)
        flows.append(
            FlowRecord(
                ts=t0 + rng.uniform(0, wsec - 1),
                src=attacker,
                dst=target,
                src_port=rng.randint(1024, 65535),
                dst_port=443,
                protocol="TCP",
                duration=abs(rng.gauss(12.0, 2.0)) + 1.0,
                fwd_packets=rng.randint(400, 800),
                bwd_packets=rng.randint(5, 15),
                fwd_bytes=volume,
                bwd_bytes=rng.randint(200, 800),
                label="Exfiltration",
            )
        )
