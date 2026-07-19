"""Tier-3 collision energy-moment contracts for the no-QKE surface."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.collisions_jax import laguerre_grid
from rabbit.jax.driver_typeI_full_boltzmann import (
    _COLLISION_PREFLIGHT_C_RATE,
    _MEV_TO_S,
    _collision_ap_unified_bank_core_jax,
    _collision_bank_energy_exchange_jax,
    _rate_enforcement_scale_jax,
)
from rabbit.jax.nudec_coupled_jax import (
    collision_moment_3T_source_terms_jax,
    coupled_3T_rhs_from_collision_moments_jax,
    hubble_3T_jax,
    total_energy_transfer_jax,
)
from rabbit.thermo.nudec_tables import _C_RATE


jax.config.update("jax_enable_x64", True)


def _fd(q: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


@pytest.mark.production
def test_full_boltzmann_collision_prefactor_uses_canonical_3t_rate() -> None:
    """The phase-space driver must not carry a second hidden Mangano rate."""
    assert _COLLISION_PREFLIGHT_C_RATE == pytest.approx(_C_RATE, rel=0.0, abs=0.0)


@pytest.mark.production
@pytest.mark.parametrize(
    ("T_gamma", "T_nu_e", "T_nu_x"),
    [
        (2.0, 1.7, 1.6),
        (1.6, 1.8, 1.75),
        (1.7, 1.6, 1.8),
    ],
)
def test_ap_unified_bank_energy_moment_matches_3t_heating_and_cooling(
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
) -> None:
    q_nodes, q_weights = laguerre_grid(6)
    bank = np.concatenate([_fd(q_nodes), _fd(q_nodes), _fd(q_nodes)])
    H_MeV = hubble_3T_jax(
        jnp.asarray(T_gamma),
        jnp.asarray(T_nu_e),
        jnp.asarray(T_nu_x),
        jnp.asarray(0.0),
    )

    collision = _collision_ap_unified_bank_core_jax(
        jnp.asarray(bank),
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_gamma=jnp.asarray(T_gamma),
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
        H_inv_sec=H_MeV * _MEV_TO_S,
        enforce_rate=True,
    )
    dQ_nue_pair_N, dQ_nux_bank_N = _collision_bank_energy_exchange_jax(
        collision,
        q_nodes=jnp.asarray(q_nodes),
        q_weights=jnp.asarray(q_weights),
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
    )
    target_nue_pair, target_nux_bank = total_energy_transfer_jax(
        jnp.asarray(T_gamma),
        jnp.asarray(T_nu_e),
        jnp.asarray(T_nu_x),
    )

    assert float(dQ_nue_pair_N) == pytest.approx(
        float(target_nue_pair / H_MeV),
        rel=1e-12,
        abs=1e-12,
    )
    assert float(dQ_nux_bank_N) == pytest.approx(
        float(target_nux_bank / H_MeV),
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.production
def test_standard_3t_collision_moment_sign_heats_cooler_heavy_bank() -> None:
    T_gamma = jnp.asarray(0.8)
    T_nu_e = jnp.asarray(0.77)
    T_nu_x = jnp.asarray(0.70)
    dQ_nue_pair_N = jnp.asarray(0.0)
    dQ_nux_bank_N = jnp.asarray(1.0e-6)

    _dT_gamma_source, _dT_nue_source, dT_nux_source = collision_moment_3T_source_terms_jax(
        T_gamma,
        T_nu_e,
        T_nu_x,
        dQ_nue_pair_N=dQ_nue_pair_N,
        dQ_nux_bank_N=dQ_nux_bank_N,
    )
    _dT_gamma, _dT_nue, dT_nux = coupled_3T_rhs_from_collision_moments_jax(
        T_gamma,
        T_nu_e,
        T_nu_x,
        dQ_nue_pair_N=dQ_nue_pair_N,
        dQ_nux_bank_N=dQ_nux_bank_N,
    )

    assert float(dT_nux_source) > 0.0
    assert float(dT_nux) > -float(T_nu_x)


@pytest.mark.production
def test_rate_enforcement_scale_never_flips_wrong_sign_energy_moment() -> None:
    scale_ok = _rate_enforcement_scale_jax(
        jnp.asarray(2.0),
        jnp.asarray(4.0),
        jnp.asarray(1.0e-6),
        jnp.asarray(0.1),
        jnp.asarray(5.0),
    )
    scale_wrong = _rate_enforcement_scale_jax(
        jnp.asarray(-2.0),
        jnp.asarray(4.0),
        jnp.asarray(1.0e-6),
        jnp.asarray(0.1),
        jnp.asarray(5.0),
    )
    scale_tiny_same_sign = _rate_enforcement_scale_jax(
        jnp.asarray(1.0e-4),
        jnp.asarray(1.0),
        jnp.asarray(1.0e-2),
        jnp.asarray(0.1),
        jnp.asarray(5.0),
    )
    scale_zero_target = _rate_enforcement_scale_jax(
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-6),
        jnp.asarray(0.1),
        jnp.asarray(5.0),
    )

    assert float(scale_ok) == pytest.approx(2.0)
    assert float(scale_wrong) == pytest.approx(0.0)
    assert float(scale_tiny_same_sign) == pytest.approx(5.0)
    assert float(scale_zero_target) == pytest.approx(0.0)
