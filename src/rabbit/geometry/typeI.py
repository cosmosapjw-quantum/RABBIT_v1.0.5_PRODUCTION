"""
rabbit.geometry.typeI — Canonical Type I Bianchi geometry.

Hubble-normalized shear evolution for orthogonal Bianchi Type I:

    dΣ₊/dN = −(2 − q) Σ₊ + Π₊
    dΣ₋/dN = −(2 − q) Σ₋ + Π₋

where Π₊, Π₋ are the anisotropic stress components from the neutrino
transport sector (NOT hardcoded — received as function arguments).

The Friedmann constraint Ω = 1 − Σ₊² − Σ₋² is algebraic and NOT
evolved.  The deceleration parameter for radiation (γ = 4/3) is
q = 1 + Σ².

Key physics:
    - Σ ∝ a^{−1/2} (Hubble-normalized decay), σ ∝ a^{−5/2} (physical)
    - Log-periodic oscillations with ω = √(64 f_ν/5 − 1)/2 ≈ 1.023
    - f_ν ≈ 0.4052 for N_eff = 3.044
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# §1. Physical constants
# ═══════════════════════════════════════════════════════════════════════

#: Standard neutrino energy fraction for N_eff = 3.044.
#: f_ν = (7/8)(4/11)^{4/3} N_eff / [1 + (7/8)(4/11)^{4/3} N_eff]
F_NU_STANDARD: float = 0.40520

#: σ–π feedback coefficient A = 6 (geometry ← transport).
A_FEEDBACK: float = 6.0

#: Quadrupole source coefficient B = 8/15 (transport ← geometry).
B_SOURCE: float = 8.0 / 15.0

#: Oscillation frequency ω = √(64 f_ν/5 − 1)/2.
OMEGA_OSCILLATION: float = np.sqrt(64.0 * F_NU_STANDARD / 5.0 - 1.0) / 2.0

#: Hubble-normalized decay rate: Re(λ) = −1/2.
SIGMA_DECAY_RATE: float = -0.5


# ═══════════════════════════════════════════════════════════════════════
# §2. State container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TypeIShearState:
    """Accessor for Type I shear variables.

    Attributes
    ----------
    Sigma_plus : float
        Σ₊ shear variable.
    Sigma_minus : float
        Σ₋ shear variable.
    """
    Sigma_plus: float
    Sigma_minus: float

    @property
    def Sigma_sq(self) -> float:
        """Scalar shear invariant Σ² = Σ₊² + Σ₋²."""
        return self.Sigma_plus**2 + self.Sigma_minus**2

    @property
    def Omega(self) -> float:
        """Energy density parameter from Friedmann constraint."""
        return max(0.0, 1.0 - self.Sigma_sq)


# ═══════════════════════════════════════════════════════════════════════
# §3. Deceleration parameter
# ═══════════════════════════════════════════════════════════════════════

def compute_q_radiation(Sigma_sq: float) -> float:
    """Deceleration parameter for radiation domination.

    q = 2Σ² + Ω = 2Σ² + (1 − Σ²) = 1 + Σ²

    The (3/2)Π correction is subdominant and omitted in the q
    evaluation following standard practice (Wainwright–Ellis).

    Parameters
    ----------
    Sigma_sq : float
        Σ² = Σ₊² + Σ₋².

    Returns
    -------
    float
        Deceleration parameter q.
    """
    return 1.0 + Sigma_sq


# ═══════════════════════════════════════════════════════════════════════
# §4. Geometry RHS
# ═══════════════════════════════════════════════════════════════════════

def compute_typeI_geometry_rhs(
    Sigma_plus: float,
    Sigma_minus: float,
    pi_shear_plus: float,
    pi_shear_minus: float,
    Omega: float,
) -> Tuple[float, float]:
    """Compute dΣ₊/dN and dΣ₋/dN for Type I orthogonal.

    dΣ₊/dN = −(2 − q) Σ₊ + Π₊
    dΣ₋/dN = −(2 − q) Σ₋ + Π₋

    where q = 1 + Σ² for radiation domination.

    Parameters
    ----------
    Sigma_plus : float
        Hubble-normalized shear Σ₊.
    Sigma_minus : float
        Hubble-normalized shear Σ₋.
    pi_shear_plus : float
        Π₊ anisotropic stress from neutrino sector.
    pi_shear_minus : float
        Π₋ anisotropic stress from neutrino sector.
    Omega : float
        Energy density parameter (from Friedmann constraint).
        Passed explicitly for generality; for Type I in radiation
        domination Ω = 1 − Σ₊² − Σ₋².

    Returns
    -------
    dSigma_plus_dN : float
    dSigma_minus_dN : float
    """
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    q = compute_q_radiation(Sigma_sq)

    damping = -(2.0 - q)  # = -(1 - Σ²)

    dSigma_plus_dN = damping * Sigma_plus + pi_shear_plus
    dSigma_minus_dN = damping * Sigma_minus + pi_shear_minus

    return dSigma_plus_dN, dSigma_minus_dN


def compute_typeI_geometry_rhs_lrs(
    Sigma_H: float,
    pi_shear: float,
) -> float:
    """Simplified LRS variant (Σ₋ = 0).

    Parameters
    ----------
    Sigma_H : float
        Single shear variable Σ_H = Σ₊.
    pi_shear : float
        Anisotropic stress Π₊.

    Returns
    -------
    float
        dΣ_H/dN.
    """
    dS, _ = compute_typeI_geometry_rhs(Sigma_H, 0.0, pi_shear, 0.0,
                                        max(0.0, 1.0 - Sigma_H**2))
    return dS


# ═══════════════════════════════════════════════════════════════════════
# §5. Coupled σ–π system (for eigenvalue analysis and testing)
# ═══════════════════════════════════════════════════════════════════════

def sigma_pi_system_matrix(f_nu: float = F_NU_STANDARD) -> np.ndarray:
    """System matrix for the linearized σ–π coupled ODE.

    The 2×2 system for (Σ, Π) in the linear regime is:

        dΣ/dN = −Σ + A f_ν Π
        dΠ/dN = −B Σ

    with A = 6, B = 8/15.  Eigenvalues: λ = −1/2 ± iω,
    ω = √(A·B·f_ν − 1/4).

    Parameters
    ----------
    f_nu : float
        Neutrino energy fraction.

    Returns
    -------
    ndarray, shape (2, 2)
        System matrix [[−1, A·f_ν], [−B, 0]].
    """
    return np.array([
        [-1.0, A_FEEDBACK * f_nu],
        [-B_SOURCE, 0.0],
    ])


def oscillation_frequency(f_nu: float = F_NU_STANDARD) -> float:
    """Oscillation frequency ω = √(64 f_ν/5 − 1)/2.

    Parameters
    ----------
    f_nu : float
        Neutrino energy fraction.

    Returns
    -------
    float
        ω (imaginary part of eigenvalue).
    """
    discriminant = A_FEEDBACK * B_SOURCE * f_nu - 0.25
    if discriminant < 0:
        raise ValueError(
            f"No oscillation: f_ν = {f_nu} < 5/64 = {5/64:.4f}. "
            f"Discriminant = {discriminant:.6f} < 0."
        )
    return np.sqrt(discriminant)
