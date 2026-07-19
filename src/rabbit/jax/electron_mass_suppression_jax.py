"""rabbit.jax.electron_mass_suppression_jax — F(m_e/T) suppression in JAX.

v3.2 Phase χ-2 (Plan §χ-2). Ports the SciPy ``F(m_e/T)`` electron-mass
suppression factor from :mod:`rabbit.thermo.nudec_tables` to JAX.

Physics meaning
---------------
The ν-e collision rate has an electron-mass threshold around T ~ m_e
because pair production becomes Boltzmann-suppressed. The suppression
factor

    F(T) = ρ_{e±}^{exact}(T) / ρ_{e±}^{rel}(T)

with ``ρ_{e±}^{exact} = (2/π²) ∫₀^∞ p² E / (e^{E/T}+1) dp`` (E = √(p²+m_e²))
and ``ρ_{e±}^{rel} = (7/4)(π²/15) T⁴`` is the ratio of exact to
ultrarelativistic e± energy densities. F(T → ∞) = 1; F(T → 0) = 0.

This factor multiplies the leading-order ν-e collision rate when the
plasma temperature is at or below m_e ≈ 0.51 MeV — i.e. exactly the
neutrino-decoupling regime — and is one of the dominant
beyond-Mangano-coefficient corrections to N_eff.

Implementation strategy (Plan §χ-2 Path B)
------------------------------------------
The Padé closed-form path was ruled out because the SciPy reference
already exists as a precomputed table built via ``scipy.integrate.quad``.
We import that table at module load (200-point geomspace from 0.01
to 100 MeV) and expose it as JAX arrays for ``jnp.interp``-based
evaluation. The result is bit-identical to the SciPy reference to
table-quadrature precision and is jax.jit + jax.grad compatible.

Anisotropy independence (Plan §0.3, v3.1 anisotropy audit)
----------------------------------------------------------
F(m_e/T) is a **particle-physics layer** quantity (pure thermal QED
property of the electron plasma). It depends only on the photon
temperature and the electron mass; no anisotropic Bianchi state
enters its derivation. Safe to drop in directly.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Table import + JAX conversion
# ═══════════════════════════════════════════════════════════════════════

def _import_scipy_table() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load the SciPy-built F(T) table as JAX arrays.

    Imported at module load time so the SciPy quadrature pays its cost
    once. The returned arrays are read-only JAX float64 vectors.
    """
    from rabbit.thermo.nudec_tables import _F_TABLE_LOG_T, _F_TABLE_VALUES
    log_t = jnp.asarray(_F_TABLE_LOG_T, dtype=jnp.float64)
    f_vals = jnp.asarray(_F_TABLE_VALUES, dtype=jnp.float64)
    return log_t, f_vals


_F_LOG_T, _F_VALUES = _import_scipy_table()
"""Module-level JAX arrays: log(T_MeV) → F(m_e/T)."""


# ═══════════════════════════════════════════════════════════════════════
# §2. JAX-traceable evaluation
# ═══════════════════════════════════════════════════════════════════════

def electron_mass_suppression_jax(T_gamma_MeV: jnp.ndarray) -> jnp.ndarray:
    """JAX mirror of :func:`rabbit.thermo.nudec_tables._electron_mass_suppression`.

    Returns F(m_e/T_γ) ∈ [0, 1] via linear interpolation on the
    SciPy-built log-spaced table. Asymptotes:

      - T_γ ≪ m_e ≈ 0.51 MeV: F → 0 (electron decoupled)
      - T_γ ≫ m_e: F → 1 (relativistic limit)

    Boundary handling matches the SciPy implementation:
      - ``T_γ > 50 MeV``: return 1.0 (above the table; relativistic)
      - ``T_γ < 0.01 MeV``: return 0.0 (below table; fully decoupled)

    Parameters
    ----------
    T_gamma_MeV : scalar or array-like
        Photon temperature in MeV.

    Returns
    -------
    F : jnp.ndarray
        Suppression factor in [0, 1], same shape as input.

    Notes
    -----
    Uses ``jnp.interp`` (linear in log T). Bit-identical to SciPy
    reference to table-quadrature precision (~1e-6 floor).

    jax.grad-compatible. The derivative w.r.t. T_γ at the boundary
    extrapolation regions is zero by construction (clamp).
    """
    T = jnp.asarray(T_gamma_MeV, dtype=jnp.float64)
    log_T = jnp.log(jnp.maximum(T, 1e-300))
    # jnp.interp clamps at boundaries (returns f_vals[0] for log_T below
    # f_log_t[0] and f_vals[-1] for log_T above f_log_t[-1]). The SciPy
    # version uses explicit clamps at T < 0.01 (F=0) and T > 50 (F=1),
    # which match our table edges.
    F_interp = jnp.interp(log_T, _F_LOG_T, _F_VALUES)
    # Apply the explicit boundary masks matching SciPy semantics
    F_high = jnp.where(T > 50.0, 1.0, F_interp)
    F = jnp.where(T < 0.01, 0.0, F_high)
    return F


# ═══════════════════════════════════════════════════════════════════════
# §3. Convenience: rate-multiplier closure
# ═══════════════════════════════════════════════════════════════════════

def make_F_me_multiplier_jax():
    """Return a closure ``f(T_γ) → F(m_e/T_γ)`` for production drivers.

    Used to thread the suppression factor through the production rate
    pipeline without exposing the table internals to callers. The
    returned closure is JAX-traceable.
    """
    def F(T_gamma_MeV):
        return electron_mass_suppression_jax(T_gamma_MeV)
    return F


__all__ = [
    "electron_mass_suppression_jax",
    "make_F_me_multiplier_jax",
]
