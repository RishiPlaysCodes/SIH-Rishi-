"""Deterministic seeding for reproducibility.

Every stochastic component in Sentinel-X (synthetic data, MC-dropout passes,
stability perturbations) draws from an explicit :class:`Rng` derived from a
single experiment seed, so a run is fully reproducible from ``config + seed``.
"""

from __future__ import annotations

import hashlib
import random


class Rng:
    """Thin, explicit wrapper over :class:`random.Random`.

    Using a dedicated object (rather than the global ``random`` state) means
    concurrent components never perturb each other's streams.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._r = random.Random(self.seed)

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return self._r.uniform(lo, hi)

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def randint(self, lo: int, hi: int) -> int:
        return self._r.randint(lo, hi)

    def random(self) -> float:
        return self._r.random()

    def choice(self, seq):
        return self._r.choice(seq)

    def shuffle(self, seq: list) -> None:
        self._r.shuffle(seq)

    def sample(self, population, k):
        return self._r.sample(population, k)

    def spawn(self, label: str) -> Rng:
        """Deterministically derive a child stream keyed by ``label``."""
        digest = hashlib.sha256(f"{self.seed}:{label}".encode()).hexdigest()
        return Rng(int(digest[:16], 16))


def make_rng(seed: int) -> Rng:
    return Rng(seed)
