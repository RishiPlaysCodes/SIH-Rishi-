"""Model registry: build a world model by name from config."""

from __future__ import annotations

from typing import Any, Dict, List

from sentinelx.models.base import WorldModel
from sentinelx.models.linear import LinearTransitionModel
from sentinelx.models.statistical import EWMAModel, PersistenceModel

_ALIASES = {
    "baseline_statistical": "persistence",
    "statistical": "persistence",
}


def available_models() -> List[str]:
    return ["persistence", "ewma", "linear_transition"]


def build_model(model_type: str, model_cfg: Dict[str, Any] | None = None) -> WorldModel:
    cfg = model_cfg or {}
    key = _ALIASES.get(model_type, model_type)
    if key == "persistence":
        return PersistenceModel()
    if key == "ewma":
        return EWMAModel(alpha=float(cfg.get("ewma_alpha", 0.4)))
    if key == "linear_transition":
        return LinearTransitionModel(ridge_lambda=float(cfg.get("ridge_lambda", 0.05)))
    raise ValueError(
        f"Unknown model_type={model_type!r}. Available: {available_models()} "
        f"(GNN models require the full stack; see requirements-full.txt)"
    )
