"""
rabbit.weak.quadrature — Quadrature nodes for weak rate integrals.

Two quadrature rules are used:

1. Gauss–Laguerre (semi-infinite channels a, b, d, e):
   ∫₀^∞ f(x) e^{-x} dx ≈ Σ w_i f(x_i)
   Default N_E = 32 nodes.

2. Gauss–Legendre (bounded channels c, f):
   ∫₋₁^1 f(x) dx ≈ Σ w_i f(x_i)
   Mapped to [m_e, Δ] = [1, q] in m_e units.
   Default N_leg = 32 nodes.

Nodes are pre-computed and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class WeakQuadrature:
    """Pre-computed quadrature nodes for weak rate integration.

    Parameters
    ----------
    N_laguerre : int
        Gauss–Laguerre nodes for semi-infinite channels.
    N_legendre : int
        Gauss–Legendre nodes for bounded channels.
    """
    N_laguerre: int = 32
    N_legendre: int = 32
    lag_nodes: np.ndarray = field(init=False, repr=False, compare=False)
    lag_weights: np.ndarray = field(init=False, repr=False, compare=False)
    leg_nodes: np.ndarray = field(init=False, repr=False, compare=False)
    leg_weights: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        x_lag, w_lag = laggauss(self.N_laguerre)
        x_leg, w_leg = leggauss(self.N_legendre)
        object.__setattr__(self, 'lag_nodes', x_lag)
        object.__setattr__(self, 'lag_weights', w_lag)
        object.__setattr__(self, 'leg_nodes', x_leg)
        object.__setattr__(self, 'leg_weights', w_leg)

    def __eq__(self, other):
        if not isinstance(other, WeakQuadrature):
            return NotImplemented
        return (self.N_laguerre == other.N_laguerre and
                self.N_legendre == other.N_legendre)

    def __hash__(self):
        return hash((self.N_laguerre, self.N_legendre))


# ═══════════════════════════════════════════════════════════════════════
# Default quadrature
# ═══════════════════════════════════════════════════════════════════════

#: Production default: 32 nodes each.
DEFAULT_WEAK_QUADRATURE = WeakQuadrature(N_laguerre=32, N_legendre=32)

#: High-precision: 64 nodes each (for validation).
HIGHRES_WEAK_QUADRATURE = WeakQuadrature(N_laguerre=64, N_legendre=64)

#: Fast-inference candidate. Not a production default; promote only after
#: explicit 24-vs-32-vs-64 convergence gates pass for the target observable.
FAST_INFERENCE_WEAK_QUADRATURE = WeakQuadrature(N_laguerre=24, N_legendre=24)


def weak_quadrature_for_mode(mode: str = "production") -> WeakQuadrature:
    """Return the locked weak-rate quadrature for a named fidelity mode."""
    key = str(mode).strip().lower().replace("-", "_")
    if key in {"production", "default", "prod"}:
        return DEFAULT_WEAK_QUADRATURE
    if key in {"validation", "highres", "high_res", "reference"}:
        return HIGHRES_WEAK_QUADRATURE
    if key in {"fast", "inference_fast", "candidate_fast"}:
        return FAST_INFERENCE_WEAK_QUADRATURE
    raise ValueError(
        "unknown weak quadrature mode "
        f"{mode!r}; expected production, validation, or inference_fast"
    )
