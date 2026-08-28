"""Sentinel-X: Predictive Network Behavior & Threat Forecasting Platform.

A dynamic-graph world model that *forecasts* where network behaviour is heading,
quantifies uncertainty and novelty, and lets analysts test hypothetical
interventions before acting.

The package is organised as:

    sentinelx.config        Reproducible YAML/dict configuration + seeding
    sentinelx.linalg        Pure-python numeric primitives (matrix ops, RNG)
    sentinelx.data          Ingestion, cleaning, feature engineering, splits
    sentinelx.graph         Dynamic graph construction (G_t = (V_t, E_t, X_t))
    sentinelx.models        World models behind a common interface
    sentinelx.forecast      K-step forecasting + behavioural deviation scoring
    sentinelx.analytics     Propagation, uncertainty, novelty, stability,
                            counterfactuals, MITRE mapping, explainability
    sentinelx.persistence   SQLite schema + repository
    sentinelx.narrative     Templated CyberChronicle incident narration
    sentinelx.pipeline      End-to-end orchestration
    sentinelx.api           Stdlib HTTP API serving inference + the dashboard
    sentinelx.cli           Command-line entry point
"""

__version__ = "0.1.0"
