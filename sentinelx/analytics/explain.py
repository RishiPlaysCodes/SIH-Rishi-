"""Explainability layer (PRD §2.15).

Attributes a node's behavioural deviation to individual input features by their
share of the squared forecast residual. This is a first-order, indicative
explanation — the UI must always surface the caveat below. SHAP / attention
attribution over the GNN are the documented future upgrades.
"""

from __future__ import annotations

from collections.abc import Sequence

EXPLANATION_CAVEAT = (
    "Feature contributions are indicative attributions of forecast error, "
    "not a guaranteed-faithful causal explanation."
)


def feature_contributions(
    predicted: Sequence[float],
    actual: Sequence[float],
    feature_names: Sequence[str],
    top_k: int | None = None,
) -> list[dict[str, float]]:
    """Rank features by their share of the squared prediction residual.

    Returns a list of ``{"feature": name, "contribution": share}`` sorted by
    descending contribution, where shares sum to 1 (unless the residual is zero).
    """
    if not (len(predicted) == len(actual) == len(feature_names)):
        raise ValueError("predicted, actual and feature_names must be the same length")
    sq = [(p - a) ** 2 for p, a in zip(predicted, actual)]
    total = sum(sq)
    if total <= 0:
        contributions = [
            {"feature": name, "contribution": 0.0} for name in feature_names
        ]
    else:
        contributions = [
            {"feature": name, "contribution": s / total}
            for name, s in zip(feature_names, sq)
        ]
    contributions.sort(key=lambda c: c["contribution"], reverse=True)
    return contributions[:top_k] if top_k else contributions


def top_feature_names(
    predicted: Sequence[float],
    actual: Sequence[float],
    feature_names: Sequence[str],
    top_k: int = 3,
) -> list[str]:
    return [c["feature"] for c in feature_contributions(predicted, actual, feature_names, top_k)]
