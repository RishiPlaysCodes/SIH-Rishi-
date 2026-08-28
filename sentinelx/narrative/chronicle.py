"""Templated incident narration (PRD §1.5 CyberChronicle, §3.2 F).

Deliberately *templated*, not LLM-generated: per the PRD's guiding principle, no
AI-generated prose is treated as ground-truth security judgment. These produce
deterministic, auditable sentences from structured results. An LLM summarisation
layer is a documented later phase that would sit on top of this, never replace
the structured signals.
"""

from __future__ import annotations

from typing import List

from sentinelx.analytics.counterfactual import CounterfactualResult
from sentinelx.analytics.propagation import PropagationEvent


def _clock(window_start: float) -> str:
    total = int(window_start)
    return f"{(total // 3600) % 24:02d}:{(total // 60) % 60:02d}"


def narrate_deviation(
    window_start: float, node_key: str, score: float, top_features: List[str], mitre_stage: str
) -> str:
    feats = ", ".join(top_features) if top_features else "multiple signals"
    return (
        f"{_clock(window_start)} — Behavioral deviation detected on {node_key}. "
        f"Observed network state diverged from the learned baseline "
        f"(deviation {score:.0%}); dominant signals: {feats}. "
        f"Mapped ATT&CK stage: {mitre_stage}."
    )


def narrate_propagation(window_start: float, ev: PropagationEvent) -> str:
    return (
        f"{_clock(window_start)} — Propagation detected. Anomalous behavior moved "
        f"from {ev.source} toward {ev.target} "
        f"(effective reproduction number {ev.effective_reproduction_number:.2f})."
    )


def narrate_forecast(window_start: float, horizon: int, risk: float, target: str) -> str:
    return (
        f"{_clock(window_start)} — Forecast (T+{horizon}): trajectory toward {target} "
        f"at {risk:.0%} predicted risk."
    )


def narrate_intervention(window_start: float, cf: CounterfactualResult) -> str:
    direction = "reduced" if cf.delta_risk >= 0 else "increased"
    return (
        f"{_clock(window_start)} — Intervention simulated: {cf.action_type} on {cf.target}. "
        f"Predicted risk {direction} from {cf.risk_before:.0%} to {cf.risk_after:.0%} "
        f"(Δ {cf.delta_risk:+.0%})."
    )
