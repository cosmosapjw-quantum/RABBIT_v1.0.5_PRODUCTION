from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _simpson_weights(n: int, h: float) -> np.ndarray:
    if n < 3 or n % 2 == 0:
        raise ValueError("Composite Simpson rule requires an odd n >= 3")
    w = np.ones(n, dtype=np.float64)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (h / 3.0)


@dataclass(frozen=True)
class DecouplingGrid:
    """Uniform comoving-momentum grid for isotropic neutrino decoupling.

    We follow the standard early-universe decoupling convention of solving on a
    fixed comoving momentum variable y = p a / m_e, so free-streaming does not
    induce advection in momentum space.
    """

    n_y: int = 101
    y_min: float = 0.02
    y_max: float = 20.0

    def __post_init__(self) -> None:
        if self.n_y < 3 or self.n_y % 2 == 0:
            raise ValueError("DecouplingGrid requires an odd n_y >= 3")
        if not (self.y_max > self.y_min > 0.0):
            raise ValueError("Require y_max > y_min > 0")

        nodes = np.linspace(self.y_min, self.y_max, self.n_y, dtype=np.float64)
        h = float(nodes[1] - nodes[0])
        weights = _simpson_weights(self.n_y, h)

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)

    def integrate(self, values: np.ndarray, power: int = 0) -> float:
        vals = np.asarray(values, dtype=np.float64)
        return float(np.sum(self.weights * (self.nodes ** power) * vals, dtype=np.float64))
