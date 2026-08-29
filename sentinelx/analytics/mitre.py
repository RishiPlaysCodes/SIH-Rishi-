"""MITRE ATT&CK stage mapping (PRD §2.14).

Purely a *post-hoc interpretability* layer: the world model produces the
forecast and deviation; this maps the dominant contributing features to an
ATT&CK stage so an analyst gets a familiar label. It is explicitly NOT a
detector and makes no security determination on its own.
"""

from __future__ import annotations

# Ordered by kill-chain progression; earlier = earlier stage.
ATTACK_STAGES = [
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Exfiltration",
    "Command and Control",
]

# Which feature spiking most strongly implies which stage.
_FEATURE_TO_STAGE = {
    "failed_connections": "Discovery",
    "unique_ports": "Discovery",
    "unique_destinations": "Lateral Movement",
    "connection_frequency": "Lateral Movement",
    "mean_byte_rate": "Exfiltration",
    "outbound_ratio": "Exfiltration",
    "mean_packet_rate": "Command and Control",
    "mean_iat": "Command and Control",
}


def map_mitre_stage(top_features: list[str]) -> str:
    """Map ranked contributing feature names to the most advanced likely stage.

    Given features ordered by contribution (most first), we take the stage of
    the top contributor but, if a later kill-chain stage also appears strongly
    (in the top two), prefer the more advanced stage — an analyst cares most
    about how far along an intrusion may be.
    """
    if not top_features:
        return "Unknown"
    candidates = [
        _FEATURE_TO_STAGE[f] for f in top_features[:2] if f in _FEATURE_TO_STAGE
    ]
    if not candidates:
        return "Unknown"
    return max(candidates, key=lambda s: ATTACK_STAGES.index(s))
