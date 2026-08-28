"""Unknown-trajectory / out-of-distribution engine (PRD §2.11).

Combines four independent signals into a single Trajectory Novelty Score and a
qualitative label on the KNOWN -> UNKNOWN scale:

    embedding distance   how far the current graph embedding is from anything
                         seen during (benign) training;
    prediction error     the behavioural deviation score for the window;
    uncertainty          the MC-dropout sigma for the forecast.

The distance is calibrated against the natural spread of the *training*
embeddings so "far" means far relative to normal variation, not an arbitrary
constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from sentinelx.forecast.deviation import saturate
from sentinelx.graph.types import GraphState
from sentinelx.linalg import euclidean

_W_DIST = 0.5
_W_ERROR = 0.3
_W_UNCERTAINTY = 0.2
_SIGMA_SCALE = 0.1


@dataclass
class NoveltyResult:
    score: float
    label: str  # KNOWN | FAMILIAR | UNUSUAL | EMERGING | UNKNOWN
    embedding_distance: float


class NoveltyScorer:
    def __init__(self, unusual: float = 0.4, emerging: float = 0.6, unknown: float = 0.8):
        self.unusual = unusual
        self.emerging = emerging
        self.unknown = unknown
        self._train_embeddings: List[List[float]] = []
        self._dist_scale = 1.0
        self._fitted = False

    def fit(self, train_graphs: Sequence[GraphState]) -> "NoveltyScorer":
        self._train_embeddings = [g.embedding() for g in train_graphs if g.node_count() > 0]
        # Calibrate the distance scale to the mean nearest-neighbour distance
        # among training embeddings (the natural spread of "normal").
        nn: List[float] = []
        for i, e in enumerate(self._train_embeddings):
            best = None
            for j, o in enumerate(self._train_embeddings):
                if i == j:
                    continue
                d = euclidean(e, o)
                best = d if best is None or d < best else best
            if best is not None:
                nn.append(best)
        self._dist_scale = (sum(nn) / len(nn)) if nn else 1.0
        self._dist_scale = max(self._dist_scale, 1e-6)
        self._fitted = True
        return self

    def _nearest_distance(self, emb: List[float]) -> float:
        if not self._train_embeddings:
            return 0.0
        return min(euclidean(emb, e) for e in self._train_embeddings)

    def _label(self, score: float) -> str:
        if score < self.unusual * 0.6:
            return "KNOWN"
        if score < self.unusual:
            return "FAMILIAR"
        if score < self.emerging:
            return "UNUSUAL"
        if score < self.unknown:
            return "EMERGING"
        return "UNKNOWN"

    def score(
        self, graph: GraphState, prediction_error: float, uncertainty_sigma: float = 0.0
    ) -> NoveltyResult:
        if not self._fitted:
            raise RuntimeError("NoveltyScorer.score called before fit")
        emb = graph.embedding()
        dist = self._nearest_distance(emb)
        novelty_dist = saturate(dist, self._dist_scale)
        err_term = max(0.0, min(1.0, prediction_error))
        unc_term = saturate(uncertainty_sigma, _SIGMA_SCALE)
        score = _W_DIST * novelty_dist + _W_ERROR * err_term + _W_UNCERTAINTY * unc_term
        score = max(0.0, min(1.0, score))
        return NoveltyResult(score=score, label=self._label(score), embedding_distance=dist)
