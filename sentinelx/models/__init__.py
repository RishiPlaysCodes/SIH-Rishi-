"""World models: learn network transition dynamics behind a common interface.

Progressive complexity (PRD build order):

    statistical  -> PersistenceModel / EWMAModel      (Baseline 1)
    linear       -> LinearTransitionModel (ridge)     (Baseline 2)
    graphsage_gru / sentinel_x                        (framework swap-in; see
                                                       requirements-full.txt)

All models implement :class:`~sentinelx.models.base.WorldModel`, so the
forecasting, deviation, uncertainty and counterfactual layers are model-agnostic
and a PyTorch-Geometric temporal GNN drops in without touching call sites.
"""

from sentinelx.models.base import WorldModel
from sentinelx.models.statistical import EWMAModel, PersistenceModel
from sentinelx.models.linear import LinearTransitionModel
from sentinelx.models.registry import available_models, build_model

__all__ = [
    "WorldModel",
    "PersistenceModel",
    "EWMAModel",
    "LinearTransitionModel",
    "build_model",
    "available_models",
]
