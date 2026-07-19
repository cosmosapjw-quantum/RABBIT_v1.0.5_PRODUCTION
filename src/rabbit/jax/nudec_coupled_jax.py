"""JAX substrate for minimal 3-temperature neutrino decoupling.

This is a parity-oriented port of the current NumPy Tier-2 thermodynamics
helpers.  It is intentionally limited to the momentum-averaged 3T subset:

- H(T_gamma, T_nue, T_nux, Sigma^2)
- (dT_gamma/dN, dT_nue/dN, dT_nux/dN)
- N_eff inference from the 3T state

Maturity: experimental_substrate.
It is *not* yet wired into the end-to-end JAX Type-I driver, but it defines the
thermo-side contract needed for future tier-2 integration without changing the
current tier-1 parity-locked path.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from rabbit.jax.thermo_jax import (
    G_N,
    PI,
    RHO_GAMMA_PREFACTOR,
    drho_dT,
    entropy_density_plasma,
    rho_photon,
    rho_plasma,
)
from rabbit.thermo.nudec_tables import (
    _C_RATE,
    _F_TABLE_LOG_T,
    _F_TABLE_VALUES,
    _G_F,
    _PI,
    A_NUE,
    A_NUX,
)
from rabbit.jax.collision_rates_jax import gamma_nu_nu_over_H_jax


@jax.jit
def entropy_ratio_S_jax(T_gamma: jnp.ndarray) -> jnp.ndarray:
    """Plasma entropy ratio S(T)=s_plasma/s_gamma with the canonical JAX EOS."""
    s_gamma = (4.0 / 3.0) * rho_photon(T_gamma) / jnp.maximum(T_gamma, 1e-30)
    s_plasma = entropy_density_plasma(T_gamma)
    return s_plasma / jnp.maximum(s_gamma, 1e-100)


@jax.jit
def _electron_mass_suppression_jax(T_gamma: jnp.ndarray) -> jnp.ndarray:
    """Table-driven suppression F(m_e/T) with JAX interpolation.

    Uses the exact same tabulated values as :mod:`rabbit.thermo.nudec_tables`.
    This keeps the tier-2 substrate parity-safe with respect to the current
    SciPy/NumPy reference formulas.
    """
    T_gamma = jnp.asarray(T_gamma, dtype=jnp.float64)
    logT = jnp.log(jnp.maximum(T_gamma, 1e-300))
    interp = jnp.interp(logT, jnp.asarray(_F_TABLE_LOG_T), jnp.asarray(_F_TABLE_VALUES))
    return jnp.where(
        T_gamma > 50.0,
        1.0,
        jnp.where(T_gamma < 0.01, 0.0, interp),
    )


@jax.jit
def energy_transfer_rate_nue_jax(T_gamma: jnp.ndarray, T_nu_e: jnp.ndarray) -> jnp.ndarray:
    """Energy gain rate of one ν_e or ν̄_e species [MeV^5]."""
    F = _electron_mass_suppression_jax(T_gamma)
    return (_C_RATE / _PI**5) * _G_F**2 * A_NUE * F * T_gamma**4 * T_nu_e**4 * (T_gamma - T_nu_e)


@jax.jit
def energy_transfer_rate_nux_jax(T_gamma: jnp.ndarray, T_nu_x: jnp.ndarray) -> jnp.ndarray:
    """Energy gain rate of one ν_x or ν̄_x species [MeV^5]."""
    F = _electron_mass_suppression_jax(T_gamma)
    return (_C_RATE / _PI**5) * _G_F**2 * A_NUX * F * T_gamma**4 * T_nu_x**4 * (T_gamma - T_nu_x)


@jax.jit
def total_energy_transfer_jax(T_gamma: jnp.ndarray, T_nu_e: jnp.ndarray, T_nu_x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Total energy gain of (ν_e pair, 2 ν_x pairs) [MeV^5]."""
    dQ_nue = 2.0 * energy_transfer_rate_nue_jax(T_gamma, T_nu_e)
    dQ_nux = 4.0 * energy_transfer_rate_nux_jax(T_gamma, T_nu_x)
    return dQ_nue, dQ_nux


@jax.jit
def nu_nu_temperature_equilibration_sources_jax(
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    H_MeV: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Energy-conserving diagonal ν-ν temperature equilibration per e-fold.

    The 3T model carries one electron-neutrino pair and two identical
    heavy-flavour pairs.  Diagonal ν-ν scattering cannot exchange energy
    with the electromagnetic plasma, but it does relax the two neutrino
    temperature sectors toward the common energy-conserving temperature

        T_c^4 = (T_νe^4 + 2 T_νx^4) / 3.

    Returns ``(dQ_νe_pair/dN, dQ_νx_bank/dN)`` where the second term is the
    total source for the two heavy-flavour pairs.  The two terms sum to zero
    algebraically up to floating-point roundoff.
    """
    T_nu_e = jnp.asarray(T_nu_e, dtype=jnp.float64)
    T_nu_x = jnp.asarray(T_nu_x, dtype=jnp.float64)
    H_MeV = jnp.asarray(H_MeV, dtype=jnp.float64)

    T_common_4 = jnp.maximum((T_nu_e**4 + 2.0 * T_nu_x**4) / 3.0, 0.0)
    T_common = T_common_4 ** 0.25
    rho_e = _rho_nu_pair_jax(T_nu_e)
    rho_common = _rho_nu_pair_jax(T_common)

    # ν_e scatters against the two distinguishable heavy-flavour pairs.
    gamma_over_H = 2.0 * gamma_nu_nu_over_H_jax(
        T_common,
        H_MeV,
        alpha_eq_beta=False,
    )
    rho_delta = rho_common - rho_e
    equal_temperature = jnp.abs(T_nu_e - T_nu_x) <= (
        1.0e-14 * jnp.maximum(jnp.maximum(jnp.abs(T_nu_e), jnp.abs(T_nu_x)), 1.0)
    )
    rho_delta = jnp.where(equal_temperature, 0.0, rho_delta)
    dQ_nue_pair_N = gamma_over_H * rho_delta
    dQ_nux_bank_N = -dQ_nue_pair_N
    return dQ_nue_pair_N, dQ_nux_bank_N


@jax.jit
def _rho_nu_pair_jax(T_nu: jnp.ndarray) -> jnp.ndarray:
    return (7.0 / 8.0) * RHO_GAMMA_PREFACTOR * T_nu**4


@jax.jit
def _drho_nu_pair_dT_jax(T_nu: jnp.ndarray) -> jnp.ndarray:
    return 4.0 * (7.0 / 8.0) * RHO_GAMMA_PREFACTOR * T_nu**3


@jax.jit
def hubble_3T_jax(T_gamma: jnp.ndarray, T_nu_e: jnp.ndarray, T_nu_x: jnp.ndarray, Sigma_sq: jnp.ndarray = 0.0) -> jnp.ndarray:
    """Hubble rate [MeV] for one ν_e pair and two ν_x pairs."""
    rho_em = rho_plasma(T_gamma)
    rho_nu = _rho_nu_pair_jax(T_nu_e) + 2.0 * _rho_nu_pair_jax(T_nu_x)
    Omega = 1.0 - Sigma_sq
    H_sq = (8.0 * PI * G_N / 3.0) * (rho_em + rho_nu) / Omega
    return jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)


@jax.jit
def coupled_3T_rhs_jax(T_gamma: jnp.ndarray, T_nu_e: jnp.ndarray, T_nu_x: jnp.ndarray, H_MeV: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Minimal 3T thermodynamic RHS per e-fold.

    This mirrors :func:`rabbit.thermo.nudec_coupled.coupled_3T_rhs`, but uses
    the canonical JAX EOS derivatives where available.
    """
    T_gamma = jnp.asarray(T_gamma, dtype=jnp.float64)
    T_nu_e = jnp.asarray(T_nu_e, dtype=jnp.float64)
    T_nu_x = jnp.asarray(T_nu_x, dtype=jnp.float64)

    if H_MeV is None:
        H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x)

    small_T = T_gamma < 1e-6
    small_H = H_MeV < 1e-100

    S = entropy_ratio_S_jax(T_gamma)
    dSdT = jax.grad(entropy_ratio_S_jax)(T_gamma)
    dT_base = -T_gamma / (1.0 + T_gamma * dSdT / jnp.maximum(3.0 * S, 1e-100))

    dQ_nue, dQ_nux = total_energy_transfer_jax(T_gamma, T_nu_e, T_nu_x)
    dQ_nue_N = dQ_nue / jnp.maximum(H_MeV, 1e-100)
    dQ_nux_N = dQ_nux / jnp.maximum(H_MeV, 1e-100)
    dQ_total_N = dQ_nue_N + dQ_nux_N

    drdT_plasma = drho_dT(T_gamma)
    dT_source = -dQ_total_N / jnp.maximum(drdT_plasma, 1e-100)
    dT_gamma_dN = dT_base + dT_source

    d1 = _drho_nu_pair_dT_jax(T_nu_e)
    d2 = 2.0 * _drho_nu_pair_dT_jax(T_nu_x)
    dT_nue_dN = -T_nu_e + dQ_nue_N / jnp.maximum(d1, 1e-100)
    dT_nux_dN = -T_nu_x + dQ_nux_N / jnp.maximum(d2, 1e-100)

    dT_gamma_dN = jnp.where(small_T, 0.0, jnp.where(small_H, -T_gamma, dT_gamma_dN))
    dT_nue_dN = jnp.where(small_T, 0.0, jnp.where(small_H, -T_nu_e, dT_nue_dN))
    dT_nux_dN = jnp.where(small_T, 0.0, jnp.where(small_H, -T_nu_x, dT_nux_dN))
    return dT_gamma_dN, dT_nue_dN, dT_nux_dN


@jax.jit
def coupled_3T_rhs_from_collision_moments_jax(
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    *,
    dQ_nue_pair_N: jnp.ndarray,
    dQ_nux_bank_N: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """3T thermo RHS sourced directly by collision moments per e-fold."""
    T_gamma = jnp.asarray(T_gamma, dtype=jnp.float64)
    T_nu_e = jnp.asarray(T_nu_e, dtype=jnp.float64)
    T_nu_x = jnp.asarray(T_nu_x, dtype=jnp.float64)
    dQ_nue_pair_N = jnp.asarray(dQ_nue_pair_N, dtype=jnp.float64)
    dQ_nux_bank_N = jnp.asarray(dQ_nux_bank_N, dtype=jnp.float64)

    small_T = T_gamma < 1e-6

    S = entropy_ratio_S_jax(T_gamma)
    dSdT = jax.grad(entropy_ratio_S_jax)(T_gamma)
    dT_base = -T_gamma / (1.0 + T_gamma * dSdT / jnp.maximum(3.0 * S, 1e-100))

    drdT_plasma = drho_dT(T_gamma)
    dQ_total_N = dQ_nue_pair_N + dQ_nux_bank_N
    dT_source = -dQ_total_N / jnp.maximum(drdT_plasma, 1e-100)
    dT_gamma_dN = dT_base + dT_source

    d1 = _drho_nu_pair_dT_jax(T_nu_e)
    d2 = 2.0 * _drho_nu_pair_dT_jax(T_nu_x)
    dT_nue_dN = -T_nu_e + dQ_nue_pair_N / jnp.maximum(d1, 1e-100)
    dT_nux_dN = -T_nu_x + dQ_nux_bank_N / jnp.maximum(d2, 1e-100)

    dT_gamma_dN = jnp.where(small_T, 0.0, dT_gamma_dN)
    dT_nue_dN = jnp.where(small_T, 0.0, dT_nue_dN)
    dT_nux_dN = jnp.where(small_T, 0.0, dT_nux_dN)
    return dT_gamma_dN, dT_nue_dN, dT_nux_dN


@jax.jit
def collision_moment_3T_source_terms_jax(
    T_gamma: jnp.ndarray,
    T_nu_e: jnp.ndarray,
    T_nu_x: jnp.ndarray,
    *,
    dQ_nue_pair_N: jnp.ndarray,
    dQ_nux_bank_N: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transport-induced 3T source terms from collision moments only."""
    T_gamma = jnp.asarray(T_gamma, dtype=jnp.float64)
    T_nu_e = jnp.asarray(T_nu_e, dtype=jnp.float64)
    T_nu_x = jnp.asarray(T_nu_x, dtype=jnp.float64)
    dQ_nue_pair_N = jnp.asarray(dQ_nue_pair_N, dtype=jnp.float64)
    dQ_nux_bank_N = jnp.asarray(dQ_nux_bank_N, dtype=jnp.float64)

    drdT_plasma = drho_dT(T_gamma)
    dQ_total_N = dQ_nue_pair_N + dQ_nux_bank_N
    dT_gamma_source = -dQ_total_N / jnp.maximum(drdT_plasma, 1e-100)

    d1 = _drho_nu_pair_dT_jax(T_nu_e)
    d2 = 2.0 * _drho_nu_pair_dT_jax(T_nu_x)
    dT_nue_source = dQ_nue_pair_N / jnp.maximum(d1, 1e-100)
    dT_nux_source = dQ_nux_bank_N / jnp.maximum(d2, 1e-100)
    return dT_gamma_source, dT_nue_source, dT_nux_source


@jax.jit
def N_eff_from_3T_jax(T_gamma: jnp.ndarray, T_nu_e: jnp.ndarray, T_nu_x: jnp.ndarray) -> jnp.ndarray:
    """Infer N_eff from the 3T state using the standard asymptotic ratio."""
    T_std = T_gamma * (4.0 / 11.0) ** (1.0 / 3.0)
    return jnp.where(
        T_std < 1e-30,
        jnp.nan,
        (T_nu_e / T_std) ** 4 + 2.0 * (T_nu_x / T_std) ** 4,
    )
