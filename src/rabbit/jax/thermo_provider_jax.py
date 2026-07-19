"""Thermo provider split for JAX Type-I development.

The current end-to-end JAX Type-I driver remains tier-1 only.  This module
extracts the thermo-side contract so future tier-2 wiring can reuse the same
interface without perturbing the parity-locked tier-1 path.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from rabbit.physics.contracts import ThermoResult
from rabbit.jax.thermo_jax import entropy_density_plasma, rho_photon
from rabbit.jax.nudec_coupled_jax import (
    N_eff_from_3T_jax,
    coupled_3T_rhs_jax,
    hubble_3T_jax,
)

_T_DEC = 2.0
_T_NU_OVER_T_GAMMA = (4.0 / 11.0) ** (1.0 / 3.0)


@jax.jit
def tier1_T_nu_from_T_gamma_jax(T_gamma: jnp.ndarray) -> jnp.ndarray:
    """Tier 1 entropy-based decoupling map for a common neutrino temperature."""
    s_dec = entropy_density_plasma(jnp.array(_T_DEC))
    s_now = entropy_density_plasma(T_gamma)
    T_nu_decoupled = _T_DEC * jnp.power(
        jnp.maximum(s_now / jnp.maximum(s_dec, 1e-100), 1e-100), 1.0 / 3.0)
    return jnp.where(T_gamma >= _T_DEC, T_gamma, T_nu_decoupled)


@jax.jit
def tier1_hubble_aniso_jax(
    T_gamma: jnp.ndarray,
    T_nu: jnp.ndarray,
    N_eff: jnp.ndarray,
    Sigma_sq: jnp.ndarray,
) -> jnp.ndarray:
    """Tier 1 anisotropic Hubble rate [MeV]."""
    from rabbit.jax.thermo_jax import PI, G_N, rho_plasma

    rho_pl = rho_plasma(T_gamma)
    rho_nu = N_eff * (7.0 / 8.0) * rho_photon(T_nu)
    rho_total = rho_pl + rho_nu
    Omega = 1.0 - Sigma_sq
    H_sq = 8.0 * PI / 3.0 * G_N * rho_total / Omega
    return jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)


@jax.jit
def tier1_dT_gamma_dN_jax(T_gamma: jnp.ndarray) -> jnp.ndarray:
    """Tier 1 entropy-conserving plasma cooling law."""
    s_val = entropy_density_plasma(T_gamma)
    ds_dT = jax.grad(entropy_density_plasma)(T_gamma)
    return -3.0 * s_val / jnp.maximum(ds_dT, 1e-100)


@jax.jit
def tier1_N_eff_from_T_ratio_jax(T_gamma: jnp.ndarray, T_nu: jnp.ndarray) -> jnp.ndarray:
    T_nu_std = T_gamma * _T_NU_OVER_T_GAMMA
    return jnp.where(T_nu_std < 1e-30, 3.044, 3.0 * (T_nu / T_nu_std) ** 4)


@dataclass(frozen=True)
class Tier1ThermoProvider:
    """Current parity-locked JAX thermo provider.

    This is the exact tier-1 contract already used by the end-to-end JAX
    Type-I driver, now surfaced as an explicit provider object.
    """

    N_eff: float = 3.044
    mode: str = "tier1_entropy_parametric"

    def __call__(self, *, T_gamma: float, Sigma_sq: float = 0.0, **_: float) -> ThermoResult:
        T_gamma_j = jnp.asarray(T_gamma, dtype=jnp.float64)
        Sigma_sq_j = jnp.asarray(Sigma_sq, dtype=jnp.float64)
        T_nu = tier1_T_nu_from_T_gamma_jax(T_gamma_j)
        return ThermoResult(
            hubble=float(tier1_hubble_aniso_jax(T_gamma_j, T_nu, jnp.asarray(self.N_eff), Sigma_sq_j)),
            dT_gamma_dN=float(tier1_dT_gamma_dN_jax(T_gamma_j)),
            tier=1,
            T_nu_for_weak=float(T_nu),
            T_nu_e=float(T_nu),
            T_nu_x=float(T_nu),
            dT_nu_e_dN=None,
            dT_nu_x_dN=None,
            N_eff_effective=float(tier1_N_eff_from_T_ratio_jax(T_gamma_j, T_nu)),
            mode=self.mode,
        )


@dataclass(frozen=True)
class Tier2ThermoProvider:
    """JAX 3T coupled thermo provider.

    Evolves (T_γ, T_νe, T_νx) as 3 independent temperatures via
    the coupled QED entropy-transfer equations. Driver-wired via
    thermo_tier=2 in JAXTypeIConfig.
    """

    mode: str = "tier2_3T_coupled"

    def __call__(
        self,
        *,
        T_gamma: float,
        T_nu_e: float,
        T_nu_x: float,
        Sigma_sq: float = 0.0,
        **_: float,
    ) -> ThermoResult:
        T_gamma_j = jnp.asarray(T_gamma, dtype=jnp.float64)
        T_nu_e_j = jnp.asarray(T_nu_e, dtype=jnp.float64)
        T_nu_x_j = jnp.asarray(T_nu_x, dtype=jnp.float64)
        Sigma_sq_j = jnp.asarray(Sigma_sq, dtype=jnp.float64)

        H = hubble_3T_jax(T_gamma_j, T_nu_e_j, T_nu_x_j, Sigma_sq_j)
        dTg, dTne, dTnx = coupled_3T_rhs_jax(T_gamma_j, T_nu_e_j, T_nu_x_j, H_MeV=H)
        N_eff_eff = N_eff_from_3T_jax(T_gamma_j, T_nu_e_j, T_nu_x_j)
        return ThermoResult(
            hubble=float(H),
            dT_gamma_dN=float(dTg),
            tier=2,
            T_nu_for_weak=float(T_nu_e_j),
            T_nu_e=float(T_nu_e_j),
            T_nu_x=float(T_nu_x_j),
            dT_nu_e_dN=float(dTne),
            dT_nu_x_dN=float(dTnx),
            N_eff_effective=float(N_eff_eff),
            mode=self.mode,
        )
