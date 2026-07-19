"""
rabbit.jax.weak_corrections_jax — JAX-friendly weak correction substrate.

This module promotes the Coulomb/Sirlin correction hierarchy into a reusable
kernel for future all-type weak-rate backends.  The implementation mirrors the
reference formulas in :mod:`rabbit.weak.corrections`, but keeps the public API
small and explicit: correction factors, I0 normalization, and per-level budget
summaries.

Maturity: experimental_substrate.
It is intended for parity-locked CL0/CL2 weak kernels, not yet for canonical
JIT-heavy production use.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax.scipy.special as jsp_special
from numpy.polynomial.legendre import leggauss

from rabbit.weak.corrections import (
    ALPHA, M_ELECTRON_MEV, M_PROTON_MEV, Q_NP_MEV, M_P_OVER_M_E,
    W0_NEUTRON_DECAY,
)


@dataclass(frozen=True)
class WeakCorrectionBudget:
    channel: str
    coulomb_mean: float
    radiative_mean: float
    total_mean: float
    correction_level: int


@dataclass(frozen=True)
class WeakRateBudget:
    lambda_np: float
    lambda_pn: float
    I0: float
    correction_level: int
    weak_mode: str


def fermi_sommerfeld_jax(E_e, Z: int = 1):
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    result = jnp.ones_like(E_e)
    mask = E_e > 1.0 + 1e-15
    W = jnp.where(mask, E_e, 1.0)
    p = jnp.sqrt(jnp.maximum(W**2 - 1.0, 1e-300))
    eta = ALPHA * Z * W / p
    arg = jnp.clip(-2.0 * jnp.pi * eta, -500.0, 500.0)
    denom = 1.0 - jnp.exp(arg)
    small = jnp.abs(eta) < 1e-12
    F = jnp.where(small, 1.0, 2.0 * jnp.pi * eta / denom)
    return jnp.where(mask, F, result)


def coulomb_correction_per_channel_jax(E_e, channel: str):
    if channel in ('a', 'c', 'd', 'f'):
        return fermi_sommerfeld_jax(E_e, Z=+1)
    if channel in ('b', 'e'):
        return fermi_sommerfeld_jax(E_e, Z=-1)
    raise ValueError(f"Unknown channel: {channel}")


def sirlin_g_function_jax(W, W0: float = W0_NEUTRON_DECAY):
    """JIT-safe Sirlin g(W) factor.

    The algebra matches the NumPy reference but is expressed entirely in JAX
    primitives so the experimental CL2 branch can be traced inside the JAX
    driver.
    """
    W = jnp.asarray(W, dtype=jnp.float64)
    g = jnp.zeros_like(W)
    mask = (W > 1.0 + 1e-10) & (W < W0 - 1e-10)
    Wm = jnp.where(mask, W, 1.1)
    beta = jnp.sqrt(jnp.maximum(1.0 - 1.0 / Wm**2, 0.0))
    beta_clip = jnp.clip(beta, -0.99999, 0.99999)
    L = jnp.arctanh(beta_clip)
    delta_W = W0 - Wm
    t1 = 3.0 * jnp.log(M_P_OVER_M_E) - 0.75
    beta_safe = jnp.clip(beta, 1e-15, None)
    L_over_beta = L / beta_safe
    bracket2 = delta_W / (3.0 * Wm) - 1.5 + jnp.log(jnp.clip(2.0 * delta_W, 1e-30, None))
    t2 = 4.0 * (L_over_beta - 1.0) * bracket2
    t3 = L_over_beta * (2.0 * (1.0 + beta**2) + delta_W**2 / (6.0 * Wm**2) - 4.0 * L)
    x_arg = 2.0 * beta / (1.0 + beta_safe)
    S_val = -jsp_special.spence(1.0 - x_arg)
    t4 = (4.0 / beta_safe) * S_val
    g_val = t1 + t2 + t3 + t4
    return jnp.where(mask, g_val, g)


def radiative_correction_factor_jax(E_e, W0: float = W0_NEUTRON_DECAY):
    g = sirlin_g_function_jax(E_e, W0)
    return 1.0 + (ALPHA / (2.0 * jnp.pi)) * g


def weak_correction_factor_jax(E_e, channel: str, enable_coulomb: bool = True, enable_radiative: bool = True):
    factor = jnp.ones_like(jnp.asarray(E_e, dtype=jnp.float64))
    if enable_coulomb:
        factor = factor * coulomb_correction_per_channel_jax(E_e, channel)
    if enable_radiative and channel in ('c', 'f'):
        factor = factor * radiative_correction_factor_jax(E_e, W0_NEUTRON_DECAY)
    return factor


def correction_budget_for_channel_jax(E_e, channel: str, correction_level: int) -> WeakCorrectionBudget:
    E_e = jnp.asarray(E_e, dtype=jnp.float64)
    coul = coulomb_correction_per_channel_jax(E_e, channel)
    rad = radiative_correction_factor_jax(E_e, W0_NEUTRON_DECAY) if channel in ('c', 'f') else jnp.ones_like(E_e)
    total = weak_correction_factor_jax(
        E_e, channel,
        enable_coulomb=correction_level >= 1,
        enable_radiative=correction_level >= 2,
    )
    return WeakCorrectionBudget(
        channel=channel,
        coulomb_mean=float(jnp.mean(coul)),
        radiative_mean=float(jnp.mean(rad)),
        total_mean=float(jnp.mean(total)),
        correction_level=int(correction_level),
    )


def compute_I0_corrected_jax(enable_coulomb: bool = True, enable_radiative: bool = True, n_leg: int = 64) -> float:
    x, w = leggauss(n_leg)
    W0 = W0_NEUTRON_DECAY
    W = 0.5 * (W0 - 1.0) * x + 0.5 * (W0 + 1.0)
    jac = 0.5 * (W0 - 1.0)
    p = np.sqrt(np.maximum(W**2 - 1.0, 0.0))
    E_nu = W0 - W
    integrand = W * E_nu**2 * p
    corr = np.asarray(weak_correction_factor_jax(
        jnp.asarray(W), 'c',
        enable_coulomb=enable_coulomb,
        enable_radiative=enable_radiative,
    ))
    integrand *= corr
    return float(jac * np.sum(w * integrand))


# ═══════════════════════════════════════════════════════════════
# §5. Corrected Born rates (drop-in for all JAX backends)
# ═══════════════════════════════════════════════════════════════

# Default quadrature for equilibrium monopole (cached)
_DEFAULT_Q_NODES = None
_DEFAULT_SPLINE_MATRIX = None

def _ensure_default_quadrature(N_q: int = 20):
    global _DEFAULT_Q_NODES, _DEFAULT_SPLINE_MATRIX
    if _DEFAULT_Q_NODES is None or len(_DEFAULT_Q_NODES) != N_q:
        from numpy.polynomial.laguerre import laggauss
        q_np, _ = laggauss(N_q)
        _DEFAULT_Q_NODES = jnp.asarray(q_np, dtype=jnp.float64)
        from rabbit.jax.weak_live_jax import build_not_a_knot_matrix
        _DEFAULT_SPLINE_MATRIX = build_not_a_knot_matrix(_DEFAULT_Q_NODES)
    return _DEFAULT_Q_NODES, _DEFAULT_SPLINE_MATRIX


def compute_corrected_born_rates(
    T_gamma: jnp.ndarray,
    T_nu: jnp.ndarray,
    tau_n: jnp.ndarray,
    correction_level: int = 0,
    q_nodes: jnp.ndarray = None,
    spline_matrix: jnp.ndarray = None,
) -> tuple:
    """Compute λ_{np}, λ_{pn} with CL0/CL1/CL2/CL3 corrections.

    Drop-in replacement for compute_born_rates with proper CL support.
    Uses the full live weak rate machinery with equilibrium FD monopole,
    applying Coulomb, Sirlin, and finite-mass corrections at the integrand
    level (channel-by-channel, energy-dependent).

    Parameters
    ----------
    T_gamma, T_nu : temperature in MeV
    tau_n : neutron lifetime in seconds
    correction_level : 0 (Born), 1 (+Coulomb), 2 (+Sirlin), 3 (+finite mass)
    q_nodes : optional Gauss-Laguerre nodes (default: 20-point)
    spline_matrix : optional spline matrix for interpolation

    Returns
    -------
    (lambda_np, lambda_pn) in s⁻¹
    """
    from rabbit.jax.weak_jax import compute_born_rates

    if correction_level == 0:
        return compute_born_rates(T_gamma, T_nu, tau_n)

    # Use live weak machinery with equilibrium FD monopole
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_monopoles_level_specialized_jax,
    )

    if q_nodes is None:
        q_nodes, spline_matrix = _ensure_default_quadrature(20)

    # Equilibrium Fermi-Dirac monopole (isotropic limit)
    f_eq = 1.0 / (jnp.exp(jnp.minimum(q_nodes, 500.0)) + 1.0)

    lnp, lpn, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma, T_nu, tau_n, q_nodes, f_eq, f_eq,
        correction_level=correction_level,
        spline_matrix=spline_matrix,
    )
    return lnp, lpn
