"""
rabbit.jax.nonlinear_transport — Fully nonlinear discrete ordinate Boltzmann transport.

Solves the exact collisionless Boltzmann equation for massless neutrinos
in Bianchi Type I LRS spacetimes on a (q, μ) grid, with NO linearization
around FD equilibrium.

Exact equation (Hubble-normalized, LRS Type I):

    ∂f/∂N = -Σ₊(3μ²-1) q ∂f/∂q - 3Σ₊ μ(1-μ²) ∂f/∂μ

      Term 1: gravitational blueshifting/redshifting (energy shift)
      Term 2: angular aberration from differential expansion (angular drift)

Linearized PSTF approximation (current code):

    dΨ₂/dN = -(8/15)Σ₊ - 4Ψ₂

    - Treats Ψ₂ as q-independent
    - Drops angular drift (O(ΣΨ) = second order)
    - D=4 damping from neutrino stress back-reaction

This module provides both kernels and a comparison function to quantify
the nonlinear correction as a function of Σ₊.

References:
    Ellis & van Elst 1999, NATO Sci. C 541, 1 (1+3 covariant formalism)
    Pontzen & Challinor 2007, PRD 76, 063529 (PSTF hierarchy)
    RABBIT Paper I §III (linearized PSTF hierarchy)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss


# ═══════════════════════════════════════════════════════════════
# §1. Grid setup
# ═══════════════════════════════════════════════════════════════

def build_grids(N_q: int = 20, N_mu: int = 24):
    """Build momentum × angular grids for discrete ordinate transport."""
    q_nodes, q_weights = laggauss(N_q)
    mu_nodes, mu_weights = leggauss(N_mu)
    return q_nodes, q_weights, mu_nodes, mu_weights


def fd_equilibrium(q: np.ndarray) -> np.ndarray:
    """Fermi-Dirac equilibrium: f₀(q) = 1/(e^q + 1)."""
    return 1.0 / (np.exp(np.clip(q, -500, 500)) + 1.0)


def fd_equilibrium_2d(q_nodes: np.ndarray, N_mu: int) -> np.ndarray:
    """FD equilibrium on (q, μ) grid. Shape (N_q, N_μ)."""
    f0 = fd_equilibrium(q_nodes)
    return np.tile(f0[:, None], (1, N_mu))


# ═══════════════════════════════════════════════════════════════
# §2. Differentiation matrices
# ═══════════════════════════════════════════════════════════════

def _validate_q_diff_nodes(q_nodes: np.ndarray) -> np.ndarray:
    q = np.asarray(q_nodes, dtype=float)
    if q.ndim != 1 or q.size < 3:
        raise ValueError("q_nodes must be a one-dimensional grid with at least 3 nodes")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_nodes must be finite")
    if np.any(np.diff(q) <= 0.0):
        raise ValueError("q_nodes must be strictly increasing")
    return q


def _build_q_diff_matrix_3point(q_nodes: np.ndarray) -> np.ndarray:
    """Legacy 3-point non-uniform finite-difference q derivative."""

    q = _validate_q_diff_nodes(q_nodes)
    N = len(q)
    D = np.zeros((N, N))

    # Interior points: 3-point non-uniform FD
    for i in range(1, N - 1):
        h_m = q[i] - q[i - 1]
        h_p = q[i + 1] - q[i]
        D[i, i - 1] = -h_p / (h_m * (h_m + h_p))
        D[i, i] = (h_p - h_m) / (h_m * h_p)
        D[i, i + 1] = h_m / (h_p * (h_m + h_p))

    # Left boundary: forward difference (3-point)
    h0 = q[1] - q[0]
    h1 = q[2] - q[1]
    D[0, 0] = -(2 * h0 + h1) / (h0 * (h0 + h1))
    D[0, 1] = (h0 + h1) / (h0 * h1)
    D[0, 2] = -h0 / (h1 * (h0 + h1))

    # Right boundary: backward difference (3-point)
    hm = q[-2] - q[-3]
    hp = q[-1] - q[-2]
    D[-1, -3] = hp / (hm * (hm + hp))
    D[-1, -2] = -(hm + hp) / (hm * hp)
    D[-1, -1] = (2 * hp + hm) / (hp * (hm + hp))

    return D


def _local_polynomial_first_derivative_weights(
    x0: float,
    stencil_nodes: np.ndarray,
) -> np.ndarray:
    offsets = np.asarray(stencil_nodes, dtype=float) - float(x0)
    width = int(offsets.size)
    vandermonde = np.vstack([offsets**power for power in range(width)])
    rhs = np.zeros(width, dtype=float)
    rhs[1] = float(math.factorial(1))
    return np.linalg.solve(vandermonde, rhs)


def _effective_q_diff_stencil_width(q_node_count: int, requested_width: int) -> int:
    # Manufactured Laguerre profiles regress on q4-q8 with the wider stencil.
    if int(q_node_count) < 9:
        return 3
    return min(int(requested_width), int(q_node_count))


def build_q_diff_matrix(
    q_nodes: np.ndarray,
    *,
    stencil_width: int = 5,
) -> np.ndarray:
    """Non-uniform finite difference differentiation matrix for q-grid.

    The default uses a local five-point polynomial stencil on sufficiently
    resolved grids.  This keeps the operator local while reducing the large
    high-order Laguerre interior/tail errors exposed by the BD148 manufactured
    profiles.  ``stencil_width=3`` preserves the legacy 3-point operator for
    comparison probes.

    Returns D such that f' ≈ D @ f.
    """
    q = _validate_q_diff_nodes(q_nodes)
    N = len(q)
    if isinstance(stencil_width, (bool, np.bool_)):
        raise ValueError("stencil_width must be an integer >= 3")
    width = int(stencil_width)
    if width < 3:
        raise ValueError("stencil_width must be >= 3")
    width = _effective_q_diff_stencil_width(N, width)
    if width <= 3:
        return _build_q_diff_matrix_3point(q)
    D = np.zeros((N, N))
    left_width = width // 2
    for i in range(N):
        if i < left_width:
            indices = np.arange(width)
        elif i > N - 1 - left_width:
            indices = np.arange(N - width, N)
        else:
            start = i - left_width
            indices = np.arange(start, start + width)
        D[i, indices] = _local_polynomial_first_derivative_weights(q[i], q[indices])
    if not np.all(np.isfinite(D)):
        raise ValueError("q differentiation matrix contains non-finite entries")
    return D


def _q_diff_profile_error_summary(
    q_nodes: np.ndarray,
    D_q: np.ndarray,
    values: np.ndarray,
    exact_derivative: np.ndarray,
) -> dict[str, float | int]:
    approx_derivative = D_q @ values
    error = approx_derivative - exact_derivative
    abs_error = np.abs(error)
    rel_error = abs_error / np.maximum(np.abs(exact_derivative), 1.0e-300)
    argmax_index = int(np.argmax(abs_error))
    return {
        "max_abs_error": float(abs_error[argmax_index]),
        "max_relative_error": float(np.max(rel_error)),
        "right_boundary_abs_error": float(abs_error[-1]),
        "right_boundary_relative_error": float(rel_error[-1]),
        "argmax_index": argmax_index,
        "q_at_argmax": float(q_nodes[argmax_index]),
    }


def q_diff_manufactured_error_summary(q_nodes: np.ndarray) -> dict[str, object]:
    """Summarize manufactured q-derivative errors for the current q stencil.

    This is a diagnostic attribution helper for high-order Laguerre grids.  It
    does not alter transport evolution; it exposes whether the existing
    one-sided boundary row produces profile-dependent derivative errors that can
    be compared with q-grid observable drift.
    """
    q = _validate_q_diff_nodes(q_nodes)

    D_q = build_q_diff_matrix(q)
    legacy_D_q = build_q_diff_matrix(q, stencil_width=3)

    fd = fd_equilibrium(q)
    fd_exact = -fd * (1.0 - fd)

    logit_shift = 0.05
    shifted = fd_equilibrium(q + logit_shift)
    shifted_exact = -shifted * (1.0 - shifted)

    tail = np.exp(-0.5 * q) / (1.0 + q)
    tail_exact = tail * (-0.5 - 1.0 / (1.0 + q))

    return {
        "contract": "q_diff_manufactured_error_summary_v1",
        "q_node_count": int(q.size),
        "q_diff_stencil_width": int(_effective_q_diff_stencil_width(q.size, 5)),
        "legacy_q_diff_stencil_width": 3,
        "right_boundary_row_abs_max": float(np.max(np.abs(D_q[-1]))),
        "right_boundary_row_abs_sum": float(np.sum(np.abs(D_q[-1]))),
        "profiles": {
            "fd_equilibrium": _q_diff_profile_error_summary(q, D_q, fd, fd_exact),
            "logit_shifted": _q_diff_profile_error_summary(
                q, D_q, shifted, shifted_exact
            ),
            "tail_sensitive": _q_diff_profile_error_summary(
                q, D_q, tail, tail_exact
            ),
        },
        "legacy_profiles": {
            "fd_equilibrium": _q_diff_profile_error_summary(q, legacy_D_q, fd, fd_exact),
            "logit_shifted": _q_diff_profile_error_summary(
                q, legacy_D_q, shifted, shifted_exact
            ),
            "tail_sensitive": _q_diff_profile_error_summary(
                q, legacy_D_q, tail, tail_exact
            ),
        },
    }


def build_mu_diff_matrix(mu_nodes: np.ndarray) -> np.ndarray:
    """Barycentric spectral differentiation matrix for Legendre nodes.

    Uses the standard Lagrange interpolation derivative formula.
    """
    N = len(mu_nodes)

    # Barycentric weights
    w = np.ones(N)
    for j in range(N):
        for k in range(N):
            if k != j:
                w[j] /= (mu_nodes[j] - mu_nodes[k])

    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = w[j] / w[i] / (mu_nodes[i] - mu_nodes[j])
        D[i, i] = -np.sum(D[i, :])

    return D


# ═══════════════════════════════════════════════════════════════
# §3. Exact nonlinear Boltzmann RHS
# ═══════════════════════════════════════════════════════════════

def nonlinear_boltzmann_rhs(
    f: np.ndarray,
    Sigma_plus: float,
    q_nodes: np.ndarray,
    mu_nodes: np.ndarray,
    D_q: np.ndarray,
    D_mu: np.ndarray,
) -> np.ndarray:
    """Exact collisionless Boltzmann RHS for LRS Type I.

    ∂f/∂N = -Σ₊(3μ²-1) q ∂f/∂q - 3Σ₊ μ(1-μ²) ∂f/∂μ

    Parameters
    ----------
    f : ndarray, shape (N_q, N_μ)
        Distribution function on the (q, μ) grid.
    Sigma_plus : float
        Hubble-normalized shear (fixed during this call).
    q_nodes, mu_nodes : 1D arrays
    D_q, D_mu : differentiation matrices

    Returns
    -------
    ndarray, shape (N_q, N_μ)
        Time derivative df/dN.
    """
    N_q, N_mu = f.shape

    # Energy shift coefficient: S(μ) = Σ₊(3μ²-1)
    S_mu = Sigma_plus * (3.0 * mu_nodes**2 - 1.0)  # shape (N_μ,)

    # Angular drift coefficient: A(μ) = 3Σ₊ μ(1-μ²)
    A_mu = 3.0 * Sigma_plus * mu_nodes * (1.0 - mu_nodes**2)  # shape (N_μ,)

    # q-derivative: ∂f/∂q at each μ (apply D_q row-wise)
    df_dq = D_q @ f  # shape (N_q, N_μ)

    # μ-derivative: ∂f/∂μ at each q (apply D_mu column-wise)
    df_dmu = f @ D_mu.T  # shape (N_q, N_μ)

    # Assemble RHS
    # Term 1: energy shift (q × S(μ) × ∂f/∂q)
    term_energy = -S_mu[None, :] * q_nodes[:, None] * df_dq

    # Term 2: angular drift (A(μ) × ∂f/∂μ)
    term_angular = -A_mu[None, :] * df_dmu

    return term_energy + term_angular


# ═══════════════════════════════════════════════════════════════
# §4. Linearized PSTF RHS (for comparison)
# ═══════════════════════════════════════════════════════════════

def pstf_rhs(
    psi2: float,
    Sigma_plus: float,
    B: float = 8.0 / 15.0,
    D_damp: float = 4.0,
) -> float:
    """Linearized PSTF hierarchy for the quadrupole.

    dΨ₂/dN = -BΣ₊ - DΨ₂

    Returns dΨ₂/dN.
    """
    return -B * Sigma_plus - D_damp * psi2


# ═══════════════════════════════════════════════════════════════
# §5. Observable extraction
# ═══════════════════════════════════════════════════════════════

def extract_multipoles(
    f: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    mu_nodes: np.ndarray,
    mu_weights: np.ndarray,
    ell_max: int = 6,
) -> dict:
    """Extract Legendre multipole moments from the discrete ordinate distribution.

    F_ℓ(q) = (2ℓ+1)/2 ∫ f(q,μ) P_ℓ(μ) dμ

    π̃_ℓ = ∫ w(q) δF_ℓ(q) dq / ∫ w(q) f₀(q) dq

    where δF_ℓ = F_ℓ - f₀ δ_{ℓ0} and w(q) = q³ f₀(q) [Laguerre weight correction].
    """
    from numpy.polynomial.legendre import legval

    f0_eq = fd_equilibrium(q_nodes)
    N_q = len(q_nodes)

    # Energy weight for anisotropic stress projection
    energy_weight = q_weights * np.exp(q_nodes) * q_nodes**3 * f0_eq
    norm = np.sum(energy_weight)

    result = {}
    for ell in range(0, ell_max + 1, 2):  # only even ℓ for LRS
        # P_ℓ(μ) at quadrature nodes
        coeffs = np.zeros(ell + 1)
        coeffs[ell] = 1.0
        P_ell = legval(mu_nodes, coeffs)

        # Angular integration: F_ℓ(q_i) = (2ℓ+1)/2 × Σ_j f(q_i, μ_j) P_ℓ(μ_j) w_j
        F_ell_q = (2 * ell + 1) / 2.0 * (f * (P_ell * mu_weights)[None, :]).sum(axis=1)

        # Perturbation relative to equilibrium
        if ell == 0:
            delta_F = F_ell_q - f0_eq
        else:
            delta_F = F_ell_q

        # Fractional perturbation Ψ_ℓ(q)
        psi_ell_q = delta_F / np.maximum(f0_eq, 1e-30)

        # Energy-averaged reduced stress
        pi_tilde_ell = np.sum(energy_weight * psi_ell_q) / max(norm, 1e-30)

        result[ell] = {
            'psi_q': psi_ell_q,
            'pi_tilde': pi_tilde_ell,
            'F_q': F_ell_q,
        }

    return result


# ═══════════════════════════════════════════════════════════════
# §6. Time integration (RK4)
# ═══════════════════════════════════════════════════════════════

def rk4_step(f, dt, rhs_fn):
    k1 = rhs_fn(f)
    k2 = rhs_fn(f + 0.5 * dt * k1)
    k3 = rhs_fn(f + 0.5 * dt * k2)
    k4 = rhs_fn(f + dt * k3)
    return f + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_nonlinear(
    f0: np.ndarray,
    Sigma_plus: float,
    N_final: float,
    q_nodes: np.ndarray,
    mu_nodes: np.ndarray,
    D_q: np.ndarray,
    D_mu: np.ndarray,
    n_steps: int = 2000,
) -> np.ndarray:
    """Evolve the exact nonlinear Boltzmann equation for N_final e-folds."""
    dt = N_final / n_steps
    f = f0.copy()

    def rhs(f_):
        return nonlinear_boltzmann_rhs(f_, Sigma_plus, q_nodes, mu_nodes, D_q, D_mu)

    for _ in range(n_steps):
        f = rk4_step(f, dt, rhs)
        f = np.clip(f, 0.0, 1.0)  # physical bound

    return f


def evolve_pstf(
    Sigma_plus: float,
    N_final: float,
    n_steps: int = 2000,
    B: float = 8.0 / 15.0,
    D_damp: float = 4.0,
) -> np.ndarray:
    """Evolve the linearized PSTF quadrupole for N_final e-folds."""
    dt = N_final / n_steps
    psi2_history = np.zeros(n_steps + 1)
    psi2 = 0.0

    for i in range(n_steps):
        psi2_history[i] = psi2
        k1 = pstf_rhs(psi2, Sigma_plus, B, D_damp)
        k2 = pstf_rhs(psi2 + 0.5 * dt * k1, Sigma_plus, B, D_damp)
        k3 = pstf_rhs(psi2 + 0.5 * dt * k2, Sigma_plus, B, D_damp)
        k4 = pstf_rhs(psi2 + dt * k3, Sigma_plus, B, D_damp)
        psi2 += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    psi2_history[-1] = psi2
    return psi2_history


# ═══════════════════════════════════════════════════════════════
# §7. Comparison driver
# ═══════════════════════════════════════════════════════════════

@dataclass
class NonlinearComparisonResult:
    """Result of linearized vs nonlinear transport comparison."""
    Sigma_plus: float
    N_final: float
    N_q: int
    N_mu: int

    # Linearized PSTF
    pstf_pi_tilde_2: float
    pstf_psi2_final: float

    # Exact nonlinear
    exact_pi_tilde_0: float   # monopole perturbation (spectral distortion)
    exact_pi_tilde_2: float   # quadrupole
    exact_pi_tilde_4: float   # hexadecapole (generated by nonlinearity)
    exact_pi_tilde_6: float   # ℓ=6

    # Psi_2(q) profile from exact (to show q-dependence)
    exact_psi2_q: np.ndarray
    q_nodes: np.ndarray

    # Diagnostics
    relative_error_pi2: float    # |exact - pstf| / |exact| for π̃₂
    higher_multipole_ratio: float  # |π̃₄| / |π̃₂| (measure of nonlinear mode generation)
    spectral_distortion: float     # |π̃₀| (monopole shift from nonlinear energy redistribution)


def compare_linearized_vs_nonlinear(
    Sigma_plus: float,
    N_final: float = 3.0,
    N_q: int = 20,
    N_mu: int = 24,
    n_steps: int = 3000,
) -> NonlinearComparisonResult:
    """Run a fixed-Σ comparison between linearized PSTF and exact nonlinear transport.

    Parameters
    ----------
    Sigma_plus : float
        Hubble-normalized shear (held constant during evolution).
    N_final : float
        Number of e-folds to evolve.
    N_q, N_mu : int
        Grid resolution.
    n_steps : int
        Number of RK4 time steps.
    """
    q_nodes, q_weights, mu_nodes, mu_weights = build_grids(N_q, N_mu)
    D_q = build_q_diff_matrix(q_nodes)
    D_mu = build_mu_diff_matrix(mu_nodes)

    # Initial condition: FD equilibrium at all angles
    f0 = fd_equilibrium_2d(q_nodes, N_mu)

    # ── Evolve exact nonlinear ──
    f_exact = evolve_nonlinear(
        f0, Sigma_plus, N_final, q_nodes, mu_nodes, D_q, D_mu, n_steps)

    # ── Evolve linearized PSTF ──
    pstf_history = evolve_pstf(Sigma_plus, N_final, n_steps)
    pstf_psi2_final = pstf_history[-1]

    # ── Extract multipoles from exact ──
    multipoles = extract_multipoles(f_exact, q_nodes, q_weights, mu_nodes, mu_weights, ell_max=6)

    exact_pi2 = multipoles[2]['pi_tilde']
    exact_pi0 = multipoles[0]['pi_tilde']
    exact_pi4 = multipoles[4]['pi_tilde']
    exact_pi6 = multipoles[6]['pi_tilde']
    exact_psi2_q = multipoles[2]['psi_q']

    # PSTF π̃₂: since PSTF Ψ₂ is q-independent, π̃₂ = Ψ₂ × (normalization)
    # In PSTF convention: π̃ = Ψ₂ (energy-averaged = value itself since q-independent)
    pstf_pi2 = pstf_psi2_final

    rel_err = abs(exact_pi2 - pstf_pi2) / max(abs(exact_pi2), 1e-30)
    higher_ratio = abs(exact_pi4) / max(abs(exact_pi2), 1e-30)

    return NonlinearComparisonResult(
        Sigma_plus=Sigma_plus,
        N_final=N_final,
        N_q=N_q,
        N_mu=N_mu,
        pstf_pi_tilde_2=pstf_pi2,
        pstf_psi2_final=pstf_psi2_final,
        exact_pi_tilde_0=exact_pi0,
        exact_pi_tilde_2=exact_pi2,
        exact_pi_tilde_4=exact_pi4,
        exact_pi_tilde_6=exact_pi6,
        exact_psi2_q=exact_psi2_q,
        q_nodes=q_nodes,
        relative_error_pi2=rel_err,
        higher_multipole_ratio=higher_ratio,
        spectral_distortion=abs(exact_pi0),
    )
