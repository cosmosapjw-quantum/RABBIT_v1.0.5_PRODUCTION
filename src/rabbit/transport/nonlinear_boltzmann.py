"""
rabbit.transport.nonlinear_boltzmann — Exact nonlinear collisionless Boltzmann
transport for LRS Bianchi Type I.

Solves the FULL nonlinear equation (no Ψ-linearization):

    ∂Ψ/∂N = 2Σ₊ P₂(μ) q(1-f₀)(1+Ψ) - 2Σ₊ P₂(μ) q ∂Ψ/∂q
             - 3Σ₊ μ(1-μ²) ∂Ψ/∂μ

compared to the linearized version in typeI_hierarchy.py:

    dΨ₀/dN = 0
    dΨ₂/dN = -(8/15) Σ₊

Three independent approximations are disentangled:
    1. ℓ-truncation: controlled by ell_max parameter
    2. Ψ-linearization: controlled by linearized=True/False
    3. q-independent source: controlled by q_independent_source=True/False

Method: μ-grid (Gauss-Legendre) for angular integrals, Legendre
projection for mode extraction. This avoids explicit Gaunt coefficient
computation and handles all nonlinear terms exactly up to quadrature error.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════
# §1. Precomputed angular grid and Legendre matrices
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=32)
def _angular_grid(N_mu: int):
    """Gauss-Legendre nodes and weights on [-1, 1]."""
    mu, w = leggauss(N_mu)
    return mu.astype(np.float64), w.astype(np.float64)


@lru_cache(maxsize=32)
def _legendre_matrices(N_mu: int, ell_max: int):
    """Precompute P_ℓ(μ_j) and dP_ℓ/dμ(μ_j) for even ℓ ≤ ell_max.

    Returns
    -------
    ell_list : tuple of int
        Even ℓ values: (0, 2, 4, ..., ell_max).
    P : ndarray (n_ell, N_mu)
        P[k, j] = P_{ℓ_k}(μ_j).
    dP : ndarray (n_ell, N_mu)
        dP[k, j] = dP_{ℓ_k}/dμ(μ_j).
    """
    mu, _ = _angular_grid(N_mu)
    ell_list = tuple(range(0, ell_max + 1, 2))
    n_ell = len(ell_list)

    P = np.zeros((n_ell, N_mu))
    dP = np.zeros((n_ell, N_mu))

    for k, ell in enumerate(ell_list):
        P[k, :] = eval_legendre(ell, mu)
        # dP_ℓ/dμ via recurrence: (1-μ²)P_ℓ' = ℓ(P_{ℓ-1} - μ P_ℓ)
        if ell == 0:
            dP[k, :] = 0.0
        else:
            Pm1 = eval_legendre(ell - 1, mu)
            dP[k, :] = ell * (Pm1 - mu * P[k, :]) / np.maximum(1 - mu**2, 1e-30)
            # Fix endpoints where 1-μ² → 0: use L'Hôpital or explicit formula
            mask = np.abs(1 - mu**2) < 1e-14
            if np.any(mask):
                dP[k, mask] = ell * (ell + 1) / 2 * mu[mask]**(ell - 1)  # approx

    return ell_list, P, dP


# ═══════════════════════════════════════════════════════════════════════
# §2. q-derivative on non-uniform grid
# ═══════════════════════════════════════════════════════════════════════

def _q_derivative(q_nodes: np.ndarray, f_q: np.ndarray) -> np.ndarray:
    """Compute df/dq on non-uniform Gauss-Laguerre nodes.

    Uses second-order centered differences (first-order at boundaries).
    """
    N = len(q_nodes)
    df = np.zeros(N)
    if N < 2:
        return df

    # Forward at left boundary
    df[0] = (f_q[1] - f_q[0]) / (q_nodes[1] - q_nodes[0])

    # Centered in interior
    for i in range(1, N - 1):
        h_m = q_nodes[i] - q_nodes[i - 1]
        h_p = q_nodes[i + 1] - q_nodes[i]
        df[i] = (f_q[i + 1] * h_m**2 - f_q[i - 1] * h_p**2 +
                 f_q[i] * (h_p**2 - h_m**2)) / (h_m * h_p * (h_m + h_p))

    # Backward at right boundary
    df[-1] = (f_q[-1] - f_q[-2]) / (q_nodes[-1] - q_nodes[-2])

    return df


# ═══════════════════════════════════════════════════════════════════════
# §3. Nonlinear transport RHS
# ═══════════════════════════════════════════════════════════════════════

def nonlinear_transport_rhs(
    Sigma_plus: float,
    psi_modes: np.ndarray,
    q_nodes: np.ndarray,
    f0_eq: np.ndarray,
    ell_max: int = 2,
    N_mu: int = 0,
    linearized: bool = False,
    q_independent_source: bool = False,
) -> np.ndarray:
    """Compute dΨ_ℓ/dN for all species and momentum bins.

    Parameters
    ----------
    Sigma_plus : float
        Hubble-normalized shear.
    psi_modes : ndarray (n_species, n_ell, N_q)
        Current Ψ_ℓ(q) for each species.
    q_nodes : ndarray (N_q,)
        Gauss-Laguerre momentum nodes.
    f0_eq : ndarray (N_q,)
        Equilibrium Fermi-Dirac: f₀(q) = 1/(exp(q)+1).
    ell_max : int
        Maximum even multipole.
    N_mu : int
        Angular quadrature order. If 0, auto-select = 2*ell_max + 4.
    linearized : bool
        If True, drop O(Σ×Ψ) terms (reproduces standard linear hierarchy).
    q_independent_source : bool
        If True, use constant source -(8/15)Σ₊ instead of 2Σ₊ q(1-f₀).

    Returns
    -------
    dpsi : ndarray (n_species, n_ell, N_q)
    """
    n_species, n_ell_in, N_q = psi_modes.shape
    if N_mu == 0:
        N_mu = max(2 * ell_max + 4, 16)

    ell_list, P_mat, dP_mat = _legendre_matrices(N_mu, ell_max)
    mu_nodes, mu_weights = _angular_grid(N_mu)
    n_ell = len(ell_list)

    # Pad/truncate modes to match ell_max
    psi = np.zeros((n_species, n_ell, N_q))
    n_copy = min(n_ell_in, n_ell)
    psi[:, :n_copy, :] = psi_modes[:, :n_copy, :]

    # P₂(μ) values
    idx_ell2 = ell_list.index(2)
    P2_mu = P_mat[idx_ell2, :]  # (N_mu,)

    dpsi = np.zeros((n_species, n_ell, N_q))

    for s in range(n_species):
        # Reconstruct Ψ(q, μ) = Σ_ℓ Ψ_ℓ(q) P_ℓ(μ)
        # Shape: (N_q, N_mu) = (n_ell, N_q).T @ (n_ell, N_mu)
        Psi_qmu = psi[s, :, :].T @ P_mat  # (N_q, N_mu)

        # q-derivative of Ψ at each μ: ∂Ψ/∂q(q_i, μ_j)
        dPsi_dq = np.zeros((N_q, N_mu))
        for j in range(N_mu):
            dPsi_dq[:, j] = _q_derivative(q_nodes, Psi_qmu[:, j])

        # μ-derivative: ∂Ψ/∂μ = Σ_ℓ Ψ_ℓ(q) P_ℓ'(μ)
        dPsi_dmu = psi[s, :, :].T @ dP_mat  # (N_q, N_mu)

        # Full RHS in (q, μ) space
        RHS = np.zeros((N_q, N_mu))

        for i in range(N_q):
            q = q_nodes[i]
            f0 = f0_eq[i]

            if q_independent_source:
                # Standard linear source: -(8/15) Σ₊ at ℓ=2, nothing at ℓ=0
                # Reconstruct in μ-space: -(8/15)Σ₊ P₂(μ)
                source_mu = -(8.0 / 15.0) * Sigma_plus * P2_mu
            else:
                # Exact q-dependent source: 2Σ₊ q(1-f₀) P₂(μ)
                source_mu = 2.0 * Sigma_plus * q * (1.0 - f0) * P2_mu

            for j in range(N_mu):
                mu = mu_nodes[j]

                # Term 1: Source (background)
                term = source_mu[j]

                if not linearized:
                    # Term 2: Nonlinear source correction: 2Σ₊ P₂ q(1-f₀) Ψ
                    if not q_independent_source:
                        term += 2.0 * Sigma_plus * P2_mu[j] * q * (1.0 - f0) * Psi_qmu[i, j]
                    else:
                        # Nonlinear with q-independent source:
                        # Add -(8/15)Σ₊ P₂(μ) × Ψ
                        term += -(8.0 / 15.0) * Sigma_plus * P2_mu[j] * Psi_qmu[i, j]

                    # Term 3: -2Σ₊ P₂(μ) q ∂Ψ/∂q
                    term += -2.0 * Sigma_plus * P2_mu[j] * q * dPsi_dq[i, j]

                    # Term 4: -3Σ₊ μ(1-μ²) ∂Ψ/∂μ
                    term += -3.0 * Sigma_plus * mu * (1.0 - mu**2) * dPsi_dmu[i, j]
                else:
                    # Linear: only keep source, skip O(Σ×Ψ) terms
                    # q-derivative and μ-derivative of Ψ are O(Ψ), so dropped
                    pass

                RHS[i, j] = term

        # Project back to ℓ-modes: dΨ_ℓ/dN = (2ℓ+1)/2 ∫ RHS P_ℓ dμ
        for k, ell in enumerate(ell_list):
            weight = (2 * ell + 1) / 2.0
            dpsi[s, k, :] = weight * (RHS @ (mu_weights * P_mat[k, :]))

    return dpsi


# ═══════════════════════════════════════════════════════════════════════
# §4. Diagnostic: extract Ψ₀ amplitude
# ═══════════════════════════════════════════════════════════════════════

def psi0_max_amplitude(psi_modes: np.ndarray) -> float:
    """Max |Ψ₀(q)| across species and momentum bins."""
    return float(np.max(np.abs(psi_modes[:, 0, :])))


def psi2_max_amplitude(psi_modes: np.ndarray) -> float:
    """Max |Ψ₂(q)| across species and momentum bins."""
    if psi_modes.shape[1] < 2:
        return 0.0
    return float(np.max(np.abs(psi_modes[:, 1, :])))


def psi_ell_amplitudes(psi_modes: np.ndarray, ell_list: tuple) -> dict:
    """Max |Ψ_ℓ| for each ℓ."""
    out = {}
    for k, ell in enumerate(ell_list):
        if k < psi_modes.shape[1]:
            out[f"max_Psi_{ell}"] = float(np.max(np.abs(psi_modes[:, k, :])))
    return out
