"""
rabbit.jax.teff_correction_jax — JAX-native Teff closure reconstruction.

This mirrors the Teff closure semantics in the NumPy reference:
- π̃ → T₂ via energy-density matching
- perturbative spectral hardening in the small-variance regime
- exact *in-manifold* closure reconstruction by angular quadrature
- tangency diagnostic using the actual Laguerre quadrature weights

The core exact correction function is JIT-safe so it can be inlined
inside the tier-2 JAX driver's live-weak evaluation path.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss as _leggauss

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════
# §1. Module-level quadrature constants (frozen at import time)
# ═══════════════════════════════════════════════════════════════

_N_MU_DEFAULT = 16
_mu_nodes_np, _mu_weights_np = _leggauss(_N_MU_DEFAULT)
_MU_NODES = jnp.asarray(_mu_nodes_np, dtype=jnp.float64)
_MU_WEIGHTS = jnp.asarray(_mu_weights_np, dtype=jnp.float64)
# P₂(μ) = ½(3μ²−1) at each quadrature node
_P2_MU = 0.5 * (3.0 * _MU_NODES**2 - 1.0)


# ═══════════════════════════════════════════════════════════════
# §2. Helper functions
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TeffClosureResult:
    pi_tilde: float
    T2: float
    Sigma2: float
    mode: str


def pi_tilde_to_teff_quadrupole_jax(pi_tilde):
    return jnp.asarray(pi_tilde, dtype=jnp.float64) / 4.0


def teff_angular_variance_jax(T2):
    return jnp.asarray(T2, dtype=jnp.float64) ** 2 / 5.0


@jax.jit
def spectral_hardening_fd_jax(q, Sigma2):
    """Perturbative spectral hardening δf₀(q) at O(Σ²).  JIT-safe."""
    q = jnp.asarray(q, dtype=jnp.float64)
    Sigma2 = jnp.asarray(Sigma2, dtype=jnp.float64)
    f0 = 1.0 / (jnp.exp(jnp.clip(q, -500.0, 500.0)) + 1.0)
    bracket = q**2 * (1.0 - 2.0 * f0) - 2.0 * q
    return 0.5 * Sigma2 * f0 * (1.0 - f0) * bracket


# ═══════════════════════════════════════════════════════════════
# §3. JIT-safe exact Teff correction
# ═══════════════════════════════════════════════════════════════

@jax.jit
def _prepare_logit_residual(q_nodes, f0):
    """Return δ(q)=logit(f(q))+q so exact FD stays an exact fixed point."""
    eps = jnp.asarray(1e-14, dtype=f0.dtype)
    f_safe = jnp.clip(f0, eps, 1.0 - eps)
    return jnp.log(f_safe) - jnp.log1p(-f_safe) + q_nodes


@jax.jit
def _interp_scaled(q_nodes, delta_q, Theta_i):
    """Interpolate the logit residual for f₀(q/Θ) at one μ quadrature point."""
    q_scaled = q_nodes / jnp.maximum(Theta_i, 1e-30)
    idx = jnp.searchsorted(q_nodes, q_scaled, side='right') - 1
    idx = jnp.clip(idx, 0, q_nodes.shape[0] - 2)

    x_i = q_nodes[idx]
    x_ip1 = q_nodes[idx + 1]
    h_i = jnp.maximum(x_ip1 - x_i, 1e-30)
    t = (q_scaled - x_i) / h_i
    delta_eval = (1.0 - t) * delta_q[idx] + t * delta_q[idx + 1]
    delta_eval = jnp.where(q_scaled <= q_nodes[0], delta_q[0], delta_eval)
    delta_eval = jnp.where(q_scaled >= q_nodes[-1], 0.0, delta_eval)
    vals = 1.0 / (1.0 + jnp.exp(q_scaled - delta_eval))
    return vals


@jax.jit
def teff_corrected_monopole_exact_jit(
    f0: jnp.ndarray,
    q_nodes: jnp.ndarray,
    pi_tilde: jnp.ndarray,
    spline_matrix: jnp.ndarray = None,
) -> jnp.ndarray:
    """Exact in-manifold Teff correction of a neutrino monopole.  JIT-safe.

    Computes ⟨f₀(q/Θ(μ))⟩_μ via N_mu-point Gauss–Legendre quadrature
    over the P₂(cos θ) anisotropy. The isotropic baseline is reconstructed
    through the stabilized logit residual δ(q)=logit(f)+q, matching the
    improved live-weak semantics and preserving exact FD spectra for any N_q.

    Parameters
    ----------
    f0 : jnp.ndarray, shape (N_q,)
        Equilibrium or transported monopole at the momentum grid nodes.
    q_nodes : jnp.ndarray, shape (N_q,)
        Gauss–Laguerre quadrature nodes (sorted).
    pi_tilde : jnp.ndarray, scalar
        Dimensionless anisotropic stress π̃ from the transport hierarchy.
    spline_matrix : jnp.ndarray or None
        Precomputed not-a-knot spline matrix. If None, built on the fly.

    Returns
    -------
    jnp.ndarray, shape (N_q,)
        Teff-corrected monopole. Physical-bound violations are not silently clipped.
    """
    del spline_matrix

    T2 = pi_tilde / 4.0
    Sigma2 = T2**2 / 5.0

    # Θ(μ_i) = 1 + T₂ P₂(μ_i) for each quadrature point
    Theta = 1.0 + T2 * _P2_MU  # shape (N_mu,)
    delta_q = _prepare_logit_residual(q_nodes, f0)

    # Evaluate f₀(q/Θ_i) for each μ via logit-residual interpolation, then average
    f_at_mu = jax.vmap(lambda th: _interp_scaled(q_nodes, delta_q, th))(Theta)

    f_exact = 0.5 * jnp.sum(_MU_WEIGHTS[:, None] * f_at_mu, axis=0)

    # Perturbative fallback: used when any Θ ≤ 0
    f_pert = f0 + spectral_hardening_fd_jax(q_nodes, Sigma2)

    # Check if all Θ > 0 (physical regime for exact method)
    all_positive = jnp.all(Theta > 0.0)
    f_corrected = jnp.where(all_positive, f_exact, f_pert)

    # If π̃ ≈ 0, return f0 unchanged
    is_zero = jnp.abs(pi_tilde) < 1e-15
    f_corrected = jnp.where(is_zero, f0, f_corrected)

    return f_corrected


# ═══════════════════════════════════════════════════════════════
# §4. Driver-facing apply function
# ═══════════════════════════════════════════════════════════════

@jax.jit
def apply_teff_correction_to_monopoles(
    f_nue: jnp.ndarray,
    f_nuebar: jnp.ndarray,
    q_nodes: jnp.ndarray,
    pi_tilde_nue: jnp.ndarray,
    pi_tilde_nuebar: jnp.ndarray,
    spline_matrix: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Teff spectral hardening to ν_e and ν̄_e monopoles.  JIT-safe.

    This is the single entry point for the JAX driver's Teff wiring.

    Parameters
    ----------
    f_nue, f_nuebar : jnp.ndarray, shape (N_q,)
        Transported monopole distributions at momentum grid nodes.
    q_nodes : jnp.ndarray, shape (N_q,)
        Gauss–Laguerre nodes.
    pi_tilde_nue, pi_tilde_nuebar : jnp.ndarray, scalar
        Per-species π̃ from the transport hierarchy.
    spline_matrix : jnp.ndarray or None
        Precomputed not-a-knot spline matrix (shared between species).

    Returns
    -------
    (f_nue_corrected, f_nuebar_corrected) : tuple of jnp.ndarray
    """
    if spline_matrix is None:
        from rabbit.jax.weak_live_jax import build_not_a_knot_matrix
        spline_matrix = build_not_a_knot_matrix(q_nodes)
    f_nue_corr = teff_corrected_monopole_exact_jit(f_nue, q_nodes, pi_tilde_nue, spline_matrix)
    f_nuebar_corr = teff_corrected_monopole_exact_jit(f_nuebar, q_nodes, pi_tilde_nuebar, spline_matrix)
    return f_nue_corr, f_nuebar_corr


# ═══════════════════════════════════════════════════════════════
# §5. Legacy non-JIT interface (backward compatible)
# ═══════════════════════════════════════════════════════════════

def teff_corrected_monopole_exact_jax(f0, q_nodes, pi_tilde: float, N_mu: int = 16):
    """Non-JIT exact Teff correction (legacy interface).

    Kept for backward compatibility with existing tests.  For the default
    quadrature it reuses the JIT kernel to keep semantics identical to the
    driver-facing path.  For non-default N_mu it falls back to the vectorized
    NumPy reference implementation.
    """
    f0 = jnp.asarray(f0, dtype=jnp.float64)
    q_nodes = jnp.asarray(q_nodes, dtype=jnp.float64)
    if abs(float(pi_tilde)) < 1e-15:
        return f0

    if int(N_mu) == _N_MU_DEFAULT:
        return teff_corrected_monopole_exact_jit(f0, q_nodes, jnp.asarray(pi_tilde, dtype=jnp.float64))

    from rabbit.weak.teff_correction import teff_corrected_monopole_exact
    out = teff_corrected_monopole_exact(np.asarray(f0), np.asarray(q_nodes), float(pi_tilde), N_mu=int(N_mu))
    return jnp.asarray(out, dtype=jnp.float64)


def teff_tangency_diagnostic_jax(f_monopole, q_nodes, pi_tilde: float, q_weights=None) -> float:
    """Tangency diagnostic (non-JIT, diagnostic only)."""
    if abs(float(pi_tilde)) < 1e-15:
        return 0.0
    f_monopole = np.asarray(f_monopole, dtype=float)
    q_nodes = np.asarray(q_nodes, dtype=float)
    f_teff = np.asarray(teff_corrected_monopole_exact_jax(f_monopole, q_nodes, pi_tilde))
    residual = f_monopole - f_teff
    f0 = np.clip(f_monopole, 1e-30, 1.0 - 1e-30)
    entropy_weight = 1.0 / (f0 * (1.0 - f0))
    q = np.clip(q_nodes, 1e-30, None)
    if q_weights is None:
        from numpy.polynomial.laguerre import laggauss
        _, q_weights = laggauss(len(q_nodes))
    q_weights = np.asarray(q_weights, dtype=float)
    w_diag = q_weights * np.exp(q) / q
    G_norm = residual * entropy_weight
    W = w_diag
    S0 = np.sum(W)
    S1 = np.sum(W * q)
    Sq = np.sum(W * q**2)
    Sg = np.sum(W * G_norm)
    Sgq = np.sum(W * q * G_norm)
    denom = S0 * Sq - S1**2
    if abs(denom) < 1e-50:
        return 0.0
    a_fit = (Sq * Sg - S1 * Sgq) / denom
    b_fit = (S0 * Sgq - S1 * Sg) / denom
    resid_n2plus = G_norm - a_fit - b_fit * q
    D_sq = np.sum(W * resid_n2plus**2)
    return float(np.sqrt(max(D_sq, 0.0)))
