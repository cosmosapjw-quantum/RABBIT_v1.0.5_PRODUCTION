"""
rabbit.transport.characteristic_rays — Exact collisionless Boltzmann transport
via characteristic curves for LRS Bianchi Type I.

Exact solution: f(q, μ, N) = f₀(q exp(2I(N, μ₀)))

State per ray: (I_j, J_j) where
    dI_j/dN = Σ P₂(μ_j(S))       — energy shift integral
    dJ_j/dN = 3Σ(1-3μ_j²)J_j     — angular Jacobian
    dS/dN   = Σ                    — accumulated shear integral

μ_j(S) recovered analytically from the LRS separable ODE:
    μ²/(1-μ²) = [μ₀²/(1-μ₀²)] exp(6S)

Observables (proven to match production code convention at linear order):
    Π₊ = f_ν Σ_j w_j J_j P₂(μ_j) exp(-8I_j)
    f_mono(q) = (1/2) Σ_j w_j J_j f₀(q exp(2I_j))

Properties (COLLISIONLESS / free-streaming transport only — see Scope):
    - No ℓ-truncation (angular structure exact to quadrature order)
    - No Ψ-linearization (nonlinear monopole generation captured)
    - No coordinate singularity (exp(-8I), exp(2I) always finite and positive)
    - Automatic positivity: f ∈ [0,1] by construction
    - DOF: 2 N_μ + 1 per transport block (vs 1 for linearized, vs n_ℓ × N_q for PSTF)

Scope (BD598 B-R1):
    The "no ℓ-truncation / angular-exact" claim holds for the COLLISIONLESS
    characteristic transport above. It does NOT extend to the collisional
    augmented (no-QKE) yield path, which uses a fixed ℓ_max=2 S₂ projection of
    the collision source with no demonstrated ℓ-convergence. Do not cite this
    exactness for collisional runs.

Generalization path:
    - All Bianchi types: replace μ-ODE with e^a-ODE on S², I-ODE with full σ_{ab}e^ae^b
    - Tilted models: add ȧ_a e^a/H term to I-ODE
    - Non-LRS: use S² quadrature (Lebedev) instead of GL on [-1,1]
    - Collisions: add source term along characteristics (integral equation)

References:
    - Session 37 derivation (2026-04-10): q-power decoupling proof,
      Teff coordinate singularity analysis, stress normalization calibration.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════
# §1. Grid setup
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=8)
def setup_ray_grid(N_mu: int = 12):
    """Gauss-Legendre nodes for ray tracking.

    Returns
    -------
    mu0 : (N_mu,) initial ray directions
    w0 : (N_mu,) quadrature weights
    X0 : (N_mu,) precomputed μ₀²/(1-μ₀²) for forward map
    signs : (N_mu,) sign(μ₀)
    """
    mu0, w0 = leggauss(N_mu)
    mu0 = mu0.astype(np.float64)
    w0 = w0.astype(np.float64)
    X0 = mu0**2 / np.maximum(1 - mu0**2, 1e-30)
    signs = np.sign(mu0)
    return mu0, w0, X0, signs


# ═══════════════════════════════════════════════════════════════════════
# §2. LRS characteristic map (analytical)
# ═══════════════════════════════════════════════════════════════════════

def mu_current(X0: np.ndarray, signs: np.ndarray, S: float) -> np.ndarray:
    """Forward map: μ_j(S) from precomputed X0_j.

    Uses the analytical solution of dμ/dN = 3Σ μ(1-μ²):
        μ²/(1-μ²) = X₀ exp(6S)
    """
    X = X0 * np.exp(6 * S)
    return signs * np.sqrt(np.minimum(X / (1 + X), 1 - 1e-15))


# ═══════════════════════════════════════════════════════════════════════
# §3. Transport RHS
# ═══════════════════════════════════════════════════════════════════════

def characteristic_transport_rhs(
    Sigma_plus: float,
    I: np.ndarray,
    J: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (dI/dN, dJ/dN, dS/dN) for characteristic rays.

    Parameters
    ----------
    Sigma_plus : current Hubble-normalized shear
    I : (N_mu,) energy shift integrals
    J : (N_mu,) angular Jacobians
    mu : (N_mu,) current ray directions

    Returns
    -------
    dI : (N_mu,)
    dJ : (N_mu,)
    dS : float
    """
    P2 = 0.5 * (3.0 * mu**2 - 1.0)
    dI = Sigma_plus * P2
    dJ = 3 * Sigma_plus * (1 - 3 * mu**2) * J
    dS = Sigma_plus
    return dI, dJ, dS


# ═══════════════════════════════════════════════════════════════════════
# §4. Observable extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_stress(
    I: np.ndarray,
    J: np.ndarray,
    mu: np.ndarray,
    w0: np.ndarray,
    f_nu: float,
) -> float:
    """Anisotropic stress Π₊ from characteristic ray data.

    Formula: Π₊ = f_ν Σ_j w_j J_j P₂(μ_j) exp(-8 I_j)

    Calibrated against production code at linear order:
        Π₊ → 6 f_ν Ψ₂ when |I| → 0, matching dΨ₂/dN = -(8/15)Σ.
    """
    P2 = 0.5 * (3.0 * mu**2 - 1.0)
    return f_nu * np.sum(w0 * J * P2 * np.exp(-8 * I))


def extract_monopole(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
) -> np.ndarray:
    """Angle-averaged neutrino distribution f_mono(q).

    Formula: f_mono(q) = (1/2) Σ_j w_j J_j f₀(q exp(2I_j))

    Returns f_mono ∈ [0, 1] by construction (FD structure preserved).
    """
    alpha = np.exp(2 * I)                              # (N_mu,)
    qa = q_nodes[:, None] * alpha[None, :]             # (N_q, N_mu)
    f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)  # (N_q, N_mu)
    return 0.5 * f_vals @ (w0 * J)                     # (N_q,)


def extract_monopole_from_background(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    f_background: np.ndarray,
) -> np.ndarray:
    """Angle-averaged monopole from an arbitrary isotropic background spectrum.

    If the isotropic background is ``f_bg(q)``, characteristic transport gives
    ``f(q, μ_j) = f_bg(q exp(2 I_j))``. The monopole is then

        f_mono(q) = (1/2) Σ_j w_j J_j f_bg(q exp(2 I_j)).

    The background is assumed to be sampled on the same ``q_nodes`` grid.
    Values beyond the tabulated support are clipped to ``f_background[0]`` on
    the low-q side and to zero on the high-q side.
    """
    q_arr = np.asarray(q_nodes, dtype=np.float64)
    f_bg = np.clip(np.asarray(f_background, dtype=np.float64), 0.0, 1.0)
    alpha = np.exp(2.0 * np.asarray(I, dtype=np.float64))
    qa = q_arr[:, None] * alpha[None, :]
    f_vals = np.empty_like(qa, dtype=np.float64)
    left = float(f_bg[0]) if f_bg.size else 0.0
    for j in range(qa.shape[1]):
        f_vals[:, j] = np.interp(qa[:, j], q_arr, f_bg, left=left, right=0.0)
    return 0.5 * f_vals @ (np.asarray(w0, dtype=np.float64) * np.asarray(J, dtype=np.float64))


# ═══════════════════════════════════════════════════════════════════════
# §5. Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def ray_diagnostics(
    I: np.ndarray,
    J: np.ndarray,
    mu: np.ndarray,
    w0: np.ndarray,
    S: float,
    q_nodes: np.ndarray,
) -> dict:
    """Comprehensive diagnostic snapshot of ray state."""
    f0 = 1.0 / (np.exp(np.minimum(q_nodes, 500)) + 1)
    f_mono = extract_monopole(I, J, w0, q_nodes)

    return {
        'S': float(S),
        'max_abs_I': float(np.max(np.abs(I))),
        'J_min': float(np.min(J)),
        'J_max': float(np.max(J)),
        'mu_min': float(np.min(mu)),
        'mu_max': float(np.max(mu)),
        'f_mono_min': float(np.min(f_mono)),
        'f_mono_max': float(np.max(f_mono)),
        'delta_f_max': float(np.max(np.abs(f_mono - f0))),
        'delta_f_rel_max': float(np.max(
            np.abs(f_mono - f0) / np.maximum(f0, 1e-30))),
        'energy_density_ratio': float(
            0.5 * np.sum(w0 * J * np.exp(-8 * I))),
            # = ∫dΩ/(4π) J(μ) exp(-8I(μ)) ≈ 1 + O(Σ²)
            # Measures conservation of the energy-weighted angular integral.
            # Deviates from 1 due to nonlinear anisotropic redistribution.
    }
