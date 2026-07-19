"""
rabbit.config.grids — Momentum and multipole grid specifications.

MomentumGrid wraps Gauss–Laguerre quadrature nodes for the comoving
neutrino momentum q.  The Laguerre weight function e^{-q} naturally
absorbs the exponential tail of Fermi–Dirac distributions, giving
root-exponential convergence in N_q.

MultipoleSpec encodes which ℓ values are active for a given Bianchi
type and provides the DOF count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple

import numpy as np
from numpy.polynomial.laguerre import laggauss

from rabbit.config.conventions import (
    ACTIVE_ELL_TYPE_I,
    ACTIVE_ELL_TYPE_I_GENERIC,
    ELLMAX_TYPE_I,
    N_SPECIES_TRANSPORT,
    BianchiType,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Momentum grid
# ═══════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=32)
def _laggauss_cached(N_q: int) -> tuple[np.ndarray, np.ndarray]:
    q, w = laggauss(N_q)
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    q.setflags(write=False)
    w.setflags(write=False)
    return q, w

@dataclass(frozen=True)
class MomentumGrid:
    """Gauss–Laguerre momentum grid for neutrino transport.

    Parameters
    ----------
    N_q : int
        Number of quadrature nodes.  Default 80 (R01 recommendation).
        Convergence is root-exponential; N_q = 80 vs 100 differs by
        < 10^{-4} relative.

    Attributes
    ----------
    nodes : ndarray, shape (N_q,)
        Quadrature abscissae (comoving momentum q_i > 0).
    weights : ndarray, shape (N_q,)
        Standard Gauss-Laguerre weights for ∫₀^∞ e^{-q} g(q) dq.
        To approximate ∫₀^∞ h(q) dq with these raw weights, include the
        e^{+q} factor explicitly in the integrand via g(q)=e^{q}h(q).
    """
    N_q: int = 80
    nodes: np.ndarray = field(init=False, repr=False, compare=False)
    weights: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.N_q < 1:
            raise ValueError(f"N_q must be >= 1, got {self.N_q}")
        q, w = _laggauss_cached(self.N_q)
        # frozen dataclass requires object.__setattr__
        object.__setattr__(self, 'nodes', q)
        object.__setattr__(self, 'weights', w)

    def __eq__(self, other):
        if not isinstance(other, MomentumGrid):
            return NotImplemented
        return self.N_q == other.N_q

    def __hash__(self):
        return hash(self.N_q)


# ═══════════════════════════════════════════════════════════════════════
# §2. Multipole specification
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MultipoleSpec:
    """Specification of active multipoles for a Bianchi type.

    For Type I, ℓ_max = 2 is EXACT (not a truncation choice).
    For curved types (II, VII₀, …), ℓ_max is a convergence parameter
    that must be validated.

    Parameters
    ----------
    bianchi_type : BianchiType
        Determines whether ℓ_max = 2 is exact or approximate.
    ell_max : int
        Maximum multipole order.  For Type I this MUST be 2.
    active_ells : tuple of int
        Which ℓ values are evolved.  For Type I orthogonal: (0, 2).
    """
    bianchi_type: BianchiType = BianchiType.TYPE_I
    ell_max: int = ELLMAX_TYPE_I
    active_ells: Tuple[int, ...] = ACTIVE_ELL_TYPE_I

    def __post_init__(self):
        if self.bianchi_type == BianchiType.TYPE_I and self.ell_max != 2:
            raise ValueError(
                f"Type I: ℓ_max must be exactly 2 (exact, not truncation). "
                f"Got ℓ_max = {self.ell_max}."
            )
        if self.ell_max < 2:
            raise ValueError(f"ℓ_max must be >= 2, got {self.ell_max}")

    @property
    def is_exact(self) -> bool:
        """Whether the multipole truncation is exact (no closure needed)."""
        return self.bianchi_type == BianchiType.TYPE_I

    @property
    def n_ell(self) -> int:
        """Number of active multipole components."""
        return len(self.active_ells)

    @property
    def n_transport_dof_per_species(self) -> int:
        """Transport DOF per species: n_ell × N_q (requires a grid)."""
        raise TypeError(
            "Call transport_dof(grid) for the full count; "
            "this property cannot know N_q."
        )

    def transport_dof(self, grid: MomentumGrid) -> int:
        """Total transport DOF for all species.

        For Type I with N_q = 80:  3 species × 2 ℓ × 80 bins = 960.
        """
        return N_SPECIES_TRANSPORT * self.n_ell * grid.N_q

    def total_dof(self, grid: MomentumGrid, n_nuclear: int = 8) -> int:
        """Total state vector DOF.

        Decomposition: thermo(1) + geometry(2) + transport + nuclear.
        """
        n_thermo = 1     # T_gamma
        n_geom = 2       # Sigma_plus, Sigma_minus
        n_transport = self.transport_dof(grid)
        return n_thermo + n_geom + n_transport + n_nuclear


# ═══════════════════════════════════════════════════════════════════════
# §3. Default configurations
# ═══════════════════════════════════════════════════════════════════════

#: Production defaults for Type I.
DEFAULT_GRID = MomentumGrid(N_q=80)
DEFAULT_MULTIPOLE = MultipoleSpec(
    bianchi_type=BianchiType.TYPE_I,
    ell_max=2,
    active_ells=(0, 2),
)

#: Generic (non-LRS) Type I: two quadrupole polarizations.
DEFAULT_MULTIPOLE_GENERIC = MultipoleSpec(
    bianchi_type=BianchiType.TYPE_I,
    ell_max=2,
    active_ells=(0, 2, -2),
)
