"""
rabbit.geometry.general_classA — Unified Class A Bianchi geometry.

Implements the Wainwright–Hsu evolution equations for all Class A types
(I, II, VI₀, VII₀, VIII, IX) with a single kernel parameterized by
the structure-constant variables (N₁, N₂, N₃).

State: (Σ₊, Σ₋, N₁, N₂, N₃) — 5 DOF for diagonal shear + curvature.

Evolution equations (Wainwright & Ellis 1997, §6.2):
    dΣ₊/dN = -(2-q)Σ₊ - S₊ + Π₊
    dΣ₋/dN = -(2-q)Σ₋ - S₋ + Π₋
    dN₁/dN = (q - 4Σ₊)N₁
    dN₂/dN = (q + 2Σ₊ + 2√3 Σ₋)N₂
    dN₃/dN = (q + 2Σ₊ - 2√3 Σ₋)N₃

Curvature sources (Wainwright–Hsu):
    S₊ = (1/6)[(N₂-N₃)² - N₁(2N₁-N₂-N₃)]
    S₋ = (√3/6)(N₃-N₂)(N₁-N₂-N₃)

Gauss curvature invariant:
    K = (1/12)(N₁²+N₂²+N₃²-2N₁N₂-2N₂N₃-2N₃N₁)

Friedmann constraint: Ω = 1 - Σ² - K

Type recovery:
    I:    N₁=N₂=N₃=0 → S₊=S₋=K=0 → identical to typeI.py
    II:   N₁≠0, N₂=N₃=0 → S₊=-(1/3)N₁², S₋=0
    VII₀: N₂≠0, N₃≠0, N₁=0
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from rabbit.config.conventions import BianchiType

_SQRT3 = np.sqrt(3.0)


# ═══════════════════════════════════════════════════════════════════════
# §1. Curvature sources
# ═══════════════════════════════════════════════════════════════════════

def curvature_source_S(N1: float, N2: float, N3: float) -> Tuple[float, float]:
    """Wainwright–Hsu curvature sources S₊, S₋.

    S₊ = (1/6)[(N₂-N₃)² - N₁(2N₁-N₂-N₃)]
    S₋ = (√3/6)(N₃-N₂)(N₁-N₂-N₃)
    """
    S_plus = (1.0/6.0) * ((N2 - N3)**2 - N1 * (2.0*N1 - N2 - N3))
    S_minus = (_SQRT3/6.0) * (N3 - N2) * (N1 - N2 - N3)
    return S_plus, S_minus


def gauss_curvature_K(N1: float, N2: float, N3: float) -> float:
    """Gauss curvature invariant K.

    K = (1/12)(N₁²+N₂²+N₃²-2N₁N₂-2N₂N₃-2N₃N₁)
    """
    return (1.0/12.0) * (N1**2 + N2**2 + N3**2
                          - 2.0*N1*N2 - 2.0*N2*N3 - 2.0*N3*N1)


# ═══════════════════════════════════════════════════════════════════════
# §2. Deceleration parameter and Friedmann constraint
# ═══════════════════════════════════════════════════════════════════════

def compute_Omega(Sigma_plus, Sigma_minus, N1, N2, N3):
    """Ω = 1 - Σ² - K (algebraic Friedmann constraint)."""
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = gauss_curvature_K(N1, N2, N3)
    return 1.0 - Sigma_sq - K


def compute_q(Sigma_plus, Sigma_minus, N1, N2, N3):
    """Deceleration parameter for radiation domination.

    For a radiation fluid with vanishing 4-acceleration,
        q = 2Σ² + Ω,
    and the Class A Friedmann constraint gives
        Ω = 1 - Σ² - K,
    hence
        q = 1 + Σ² - K.

    This recovers the Collins–Stewart Type-II equilibrium:
        Σ₊ = 1/4, K = 1/16, q = 1.
    """
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = gauss_curvature_K(N1, N2, N3)
    return 1.0 + Sigma_sq - K


def friedmann_residual(Sigma_plus, Sigma_minus, N1, N2, N3, Omega=None):
    """Friedmann constraint residual |1 - Σ² - K - Ω|."""
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = gauss_curvature_K(N1, N2, N3)
    if Omega is None:
        Omega = 1.0 - Sigma_sq - K
    return abs(1.0 - Sigma_sq - K - Omega)


# ═══════════════════════════════════════════════════════════════════════
# §3. Unified Class A RHS
# ═══════════════════════════════════════════════════════════════════════

def classA_geometry_rhs(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    pi_shear_plus: float = 0.0,
    pi_shear_minus: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """Compute the 5-component RHS for Class A Bianchi geometry.

    Returns (dΣ₊/dN, dΣ₋/dN, dN₁/dN, dN₂/dN, dN₃/dN).

    Parameters
    ----------
    Sigma_plus, Sigma_minus : float
        Hubble-normalized diagonal shear.
    N1, N2, N3 : float
        Structure-constant variables.
    pi_shear_plus, pi_shear_minus : float
        Anisotropic stress from neutrino sector.
    """
    q = compute_q(Sigma_plus, Sigma_minus, N1, N2, N3)
    S_plus, S_minus = curvature_source_S(N1, N2, N3)

    damping = -(2.0 - q)

    dSigma_plus = damping * Sigma_plus - S_plus + pi_shear_plus
    dSigma_minus = damping * Sigma_minus - S_minus + pi_shear_minus

    dN1 = (q - 4.0 * Sigma_plus) * N1
    dN2 = (q + 2.0 * Sigma_plus + 2.0 * _SQRT3 * Sigma_minus) * N2
    dN3 = (q + 2.0 * Sigma_plus - 2.0 * _SQRT3 * Sigma_minus) * N3

    return dSigma_plus, dSigma_minus, dN1, dN2, dN3


# ═══════════════════════════════════════════════════════════════════════
# §4. Type-specific IC helpers
# ═══════════════════════════════════════════════════════════════════════

def typeI_initial(Sigma_plus, Sigma_minus=0.0):
    """Type I: N₁=N₂=N₃=0."""
    return (Sigma_plus, Sigma_minus, 0.0, 0.0, 0.0)

def typeII_initial(Sigma_plus, N1, Sigma_minus=0.0):
    """Type II: N₁≠0, N₂=N₃=0."""
    return (Sigma_plus, Sigma_minus, N1, 0.0, 0.0)

def typeVII0_initial(Sigma_plus, N2, N3, Sigma_minus=0.0):
    """Type VII₀: N₁=0, N₂≠0, N₃≠0."""
    return (Sigma_plus, Sigma_minus, 0.0, N2, N3)

def typeIX_initial(Sigma_plus, N1, N2, N3, Sigma_minus=0.0):
    """Type IX: all N_i ≠ 0, same sign."""
    return (Sigma_plus, Sigma_minus, N1, N2, N3)


# ═══════════════════════════════════════════════════════════════════════
# §5. Collins–Stewart equilibrium (Type II radiation)
# ═══════════════════════════════════════════════════════════════════════

# For Type II with γ=4/3 radiation, the Collins–Stewart attractor is:
#   Σ₊ = 1/4, N₁² = 3, K = 1/4, Ω = 1/2, Σ₋ = 0
# Actually from R06: Σ₊ = 0.25, N₁² = 1/12 → K = N₁²/12
# Let me verify: with N₂=N₃=0, K = N₁²/12.
# Collins-Stewart: Σ₊ = 1/4, Ω = 7/8, K = 1 - Σ₊² - Ω = 1 - 1/16 - 7/8 = 1/16
# So N₁² = 12K = 12/16 = 3/4, N₁ = √(3/4)
COLLINS_STEWART_SIGMA_PLUS = 0.25
COLLINS_STEWART_OMEGA = 7.0 / 8.0
COLLINS_STEWART_K = 1.0 / 16.0
COLLINS_STEWART_N1_SQ = 12.0 * COLLINS_STEWART_K  # = 0.75

# ═══════════════════════════════════════════════════════════════════════
# Collins–Stewart equilibrium (pure radiation, no Π)
# ═══════════════════════════════════════════════════════════════════════
# Non-tilted Bianchi II equilibrium family (Hewitt–Bridson–Wainwright):
#   Σ₊ = (3γ−2)/8,
#   N₁ = (3/4) sqrt[(3γ−2)(2−γ)],
#   Ω = (3/16)(6−γ),
#   q = (3γ−2)/2.
# For radiation γ = 4/3 this gives:
#   Σ₊ = 1/4,  N₁² = 3/4,  K = N₁²/12 = 1/16,  Ω = 7/8,  q = 1.
CS_SIGMA_PLUS = 1.0 / 4.0
CS_Q = 1.0
CS_N1_SQ = 3.0 / 4.0
CS_K = 1.0 / 16.0
CS_OMEGA = 7.0 / 8.0
