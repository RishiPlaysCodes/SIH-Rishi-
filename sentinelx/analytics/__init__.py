"""Higher-order analytics over the world model's forecasts.

    propagation     cyber-epidemiology: velocity, intensity, effective R
    uncertainty     MC-dropout mean/variance -> confidence label
    novelty         trajectory novelty / OOD: KNOWN..UNKNOWN
    stability       perturbation sensitivity of a forecast: STABLE / UNSTABLE
    counterfactual  hypothetical interventions over the world model
    mitre           ATT&CK stage mapping (interpretability, post-hoc)
    explain         per-feature contribution to a node's deviation
"""

from sentinelx.analytics.counterfactual import (
    CounterfactualResult,
    Intervention,
    run_counterfactual,
)
from sentinelx.analytics.explain import feature_contributions
from sentinelx.analytics.mitre import map_mitre_stage
from sentinelx.analytics.novelty import NoveltyResult, NoveltyScorer
from sentinelx.analytics.propagation import PropagationEvent, compute_propagation
from sentinelx.analytics.stability import StabilityResult, assess_stability
from sentinelx.analytics.uncertainty import UncertaintyResult, estimate_uncertainty

__all__ = [
    "CounterfactualResult",
    "Intervention",
    "NoveltyResult",
    "NoveltyScorer",
    "PropagationEvent",
    "StabilityResult",
    "UncertaintyResult",
    "assess_stability",
    "compute_propagation",
    "estimate_uncertainty",
    "feature_contributions",
    "map_mitre_stage",
    "run_counterfactual",
]
