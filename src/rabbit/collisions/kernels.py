"""
rabbit.collisions.kernels — Collision operator protocol and constants.

Defines the interface that all collision operators must satisfy, plus
the weak coupling constants for each neutrino species.

Standard Model couplings (V−A theory):
    ν_e + e: CC + NC → G_L = 1/2 + sin²θ_W,  G_R = sin²θ_W
    ν_x + e: NC only → G_L = -1/2 + sin²θ_W,  G_R = sin²θ_W

where sin²θ_W = 0.23122 (PDG 2024).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# §1. Weinberg angle and coupling constants
# ═══════════════════════════════════════════════════════════════════════

#: sin²θ_W (PDG 2024, MS-bar at M_Z).
SIN2_THETA_W: float = 0.23122

#: Fermi constant [MeV⁻²].
G_F_MEV: float = 1.1663788e-11

#: Denominator for the two-dimensional Hannestad-Madsen/DHS reduced kernels
#: used by the deterministic references.  The accompanying reduced matrix
#: polynomials carry the code's spin/root convention, so the remaining
#: normalization is ``1 / (4*pi**3)``.
HM_REDUCED_COLLISION_DENOMINATOR: float = 4.0 * np.pi**3

#: ν_e coupling (CC + NC):
G_L_NUE: float = 0.5 + SIN2_THETA_W     # ≈ 0.7312
G_R_NUE: float = SIN2_THETA_W            # ≈ 0.2312

#: ν_x coupling (NC only, x = μ, τ):
G_L_NUX: float = -0.5 + SIN2_THETA_W    # ≈ -0.2688
G_R_NUX: float = SIN2_THETA_W            # ≈ 0.2312

#: Coupling combinations for scattering |M|²:
#:   |M|² ∝ (G_L² + G_R²) × (s² + u²) + (G_L² - G_R²)² × t²/su ...
#: For the total rate (summed over all channels), the relevant combination is:
GL2_GR2_NUE: float = G_L_NUE**2 + G_R_NUE**2   # ≈ 0.5882
GL2_GR2_NUX: float = G_L_NUX**2 + G_R_NUX**2   # ≈ 0.1257

#: For annihilation ν ν̄ → e⁺ e⁻:
#:   ν_e: (G_L − G_R)² for t-channel, but full channel has both CC + NC
#: The combined scattering + annihilation coefficient for the total rate:
#:   ν_e: a_e = 1 + 4 sin²θ_W + 8 sin⁴θ_W
#:   ν_x: a_x = 1 - 4 sin²θ_W + 8 sin⁴θ_W
#: (Notzold & Raffelt 1988, Enqvist, Kainulainen & Thomson 1992)
_s2 = SIN2_THETA_W
_s4 = SIN2_THETA_W**2
A_TOTAL_NUE: float = 1.0 + 4.0 * _s2 + 8.0 * _s4   # ≈ 2.353
A_TOTAL_NUX: float = 1.0 - 4.0 * _s2 + 8.0 * _s4   # ≈ 0.503

#: Total rate ratio ν_e / ν_x (scattering + annihilation combined).
COUPLING_RATIO_NUE_NUX: float = A_TOTAL_NUE / A_TOTAL_NUX  # ≈ 4.68


def hm_reduced_collision_prefactor(T_MeV: float) -> float:
    """Return the MeV-rate prefactor for dimensionless HM/DHS kernels.

    The reduced integrals in ``rabbit.collisions`` are dimensionless functions
    of ``y = p/T``.  With ``G_F_MEV`` in MeV^-2, a collision field consumed as
    ``df/dt`` must therefore scale as ``G_F^2 T^5``.
    """

    T = float(T_MeV)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T_MeV must be positive and finite.")
    return G_F_MEV**2 * T**5 / HM_REDUCED_COLLISION_DENOMINATOR


# ═══════════════════════════════════════════════════════════════════════
# §2. Collision operator protocol
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class CollisionOperator(Protocol):
    """Protocol for collision operators acting on the neutrino monopole.

    Implementations must satisfy:
        1. Detailed balance: C[f_eq] = 0 (to numerical precision).
        2. Energy conservation: ∫ q³ ε(q) C[f](q) dq = 0.
        3. Number non-conservation (for annihilation): ∫ q² C[f](q) dq ≠ 0 in general.

    The collision integral acts on the PERTURBATION δf = f − f_eq,
    not on f directly.  This ensures property (1) by construction.
    """

    def evaluate(
        self,
        delta_f: np.ndarray,
        q_nodes: np.ndarray,
        T_MeV: float,
    ) -> np.ndarray:
        """Evaluate C[δf](q_i) at each momentum node.

        Parameters
        ----------
        delta_f : ndarray, shape (N_q,)
            Perturbation δf = f − f_eq at each node.
        q_nodes : ndarray, shape (N_q,)
            Comoving momentum nodes.
        T_MeV : float
            Plasma temperature [MeV].

        Returns
        -------
        ndarray, shape (N_q,)
            Collision integral C[δf](q_i) in units of [s⁻¹].
        """
        ...

    def energy_transfer_rate(
        self,
        delta_f: np.ndarray,
        q_nodes: np.ndarray,
        q_weights: np.ndarray,
        T_MeV: float,
    ) -> float:
        """Total energy transfer rate dQ/dt = ∫ ε C[f] dq.

        Should be zero to numerical precision for detailed-balance-
        respecting perturbations, but nonzero during transients.

        Returns
        -------
        float
            Energy transfer rate [MeV⁵ s⁻¹] (in natural units).
        """
        ...
