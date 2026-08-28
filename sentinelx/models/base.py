"""The ``WorldModel`` interface every forecasting model implements.

A world model learns p(G_{t+1} | G_{<=t}). In this reference implementation a
"prediction" is a :class:`GraphState` whose:

* node set equals the last observed graph's node set,
* node features are the model's forecast,
* edges are carried forward from the last observed graph (structural
  persistence) — a fair, shared structural baseline across all models so the
  *feature* forecast is what distinguishes them.

Stochastic forward passes (for MC-dropout uncertainty) are supported uniformly
via the ``dropout``/``rng`` arguments: input feature dimensions are randomly
masked with inverted-dropout scaling, which induces output variance for *any*
model, including the deterministic statistical baselines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from sentinelx.graph.types import GraphState, NodeState
from sentinelx.seeding import Rng


def apply_dropout(vec: Sequence[float], dropout: float, rng: Optional[Rng]) -> List[float]:
    """Inverted dropout: zero each dim with prob ``dropout``, scale survivors."""
    if dropout <= 0.0 or rng is None:
        return list(vec)
    keep = 1.0 - dropout
    scale = 1.0 / keep if keep > 0 else 0.0
    return [(0.0 if rng.random() < dropout else v * scale) for v in vec]


class WorldModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_graphs: Sequence[GraphState]) -> "WorldModel":
        """Learn transition dynamics from a chronological training sequence."""

    @abstractmethod
    def predict_next(
        self, history: Sequence[GraphState], dropout: float = 0.0, rng: Optional[Rng] = None
    ) -> GraphState:
        """Forecast the next graph state given observed history."""

    # ---- shared helpers --------------------------------------------------- #
    def _skeleton_from_last(self, last: GraphState) -> GraphState:
        """A predicted graph that inherits the last graph's structure/edges."""
        pred = last.clone()
        pred.index = last.index + 1
        pred.window_start = last.window_end
        pred.window_end = last.window_end + (last.window_end - last.window_start)
        return pred

    def predict_sequence(
        self, history: Sequence[GraphState], k: int, dropout: float = 0.0, rng: Optional[Rng] = None
    ) -> List[GraphState]:
        """Autoregressive K-step rollout (feeds predictions back as input)."""
        rolling: List[GraphState] = list(history)
        out: List[GraphState] = []
        for _ in range(k):
            nxt = self.predict_next(rolling, dropout=dropout, rng=rng)
            out.append(nxt)
            rolling = list(rolling) + [nxt]
        return out

    @staticmethod
    def _set_features(pred: GraphState, key: str, features: List[float]) -> None:
        node = pred.nodes.get(key)
        if node is None:
            pred.nodes[key] = NodeState(key=key, label=key, features=features)
        else:
            node.features = features
