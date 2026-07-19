"""Temperature and degeneracy weighting of full-Boltzmann anisotropic stress."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.characteristic_rays_jax import P2_jax
from rabbit.jax.driver_typeI_full_boltzmann import (
    _cached_equilibrium_distribution,
    _cached_ray_grid,
    _get_full_boltzmann_rhs,
    _initial_transport_state,
    _species_energy_weights_jax,
)
from rabbit.jax.solver_jax_rodas5p import materialize_low_rank_jacobian


jax.config.update("jax_enable_x64", True)


def _phase1_collisionless_system():
    return _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="collisionless",
        collision_preflight_relaxation=1.0,
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        n_network_species=2,
        n_reactions=12,
        tau_n=878.4,
        eta=6.104e-10,
        N_eff=3.044,
        f_nu=0.40520,
    )


def _state(layout: dict, *, T_nu_e: float = 2.9, T_nu_x: float = 2.9) -> np.ndarray:
    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.0
    y0[layout["i_Sm"]] = 0.0
    y0[layout["i_S"]] = 0.0
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 3.0
    y0[layout["i_tne"]] = T_nu_e
    y0[layout["i_tnx"]] = T_nu_x
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87])
    return y0


def _species_slice(layout: dict, species_index: int) -> slice:
    stride = int(layout["N_mu"]) * int(layout["N_q"])
    start = int(layout["i_transport"]) + int(species_index) * stride
    return slice(start, start + stride)


def _p2_spectral_perturbation() -> np.ndarray:
    mu0, _w0, _x0, _signs = _cached_ray_grid(4)
    p2 = np.asarray(P2_jax(mu0), dtype=np.float64)
    f0 = np.asarray(_cached_equilibrium_distribution(6), dtype=np.float64)
    return p2[:, None] * f0[None, :] * (1.0 - f0[None, :])


def _stress_rhs_delta(
    rhs_fn,
    layout: dict,
    y0: np.ndarray,
    *,
    species_index: int,
) -> float:
    base = np.asarray(rhs_fn(jnp.asarray(0.0), jnp.asarray(y0)), dtype=np.float64)
    y = y0.copy()
    transport = y[layout["i_transport"] : layout["i_transport"] + layout["n_transport"]]
    shaped = transport.reshape((4, 4, 6))
    shaped[species_index] += 0.02 * _p2_spectral_perturbation()
    y[layout["i_transport"] : layout["i_transport"] + layout["n_transport"]] = shaped.reshape(-1)
    shifted = np.asarray(rhs_fn(jnp.asarray(0.0), jnp.asarray(y)), dtype=np.float64)
    return float(shifted[layout["i_Sp"]] - base[layout["i_Sp"]])


@pytest.mark.production
def test_species_energy_weights_use_temperature_fourth_power_and_degeneracy() -> None:
    f_nu = 0.40520
    equal = np.asarray(
        _species_energy_weights_jax(
            jnp.asarray(f_nu),
            jnp.asarray(2.9),
            jnp.asarray(2.9),
        ),
        dtype=np.float64,
    )
    assert np.sum(equal) == pytest.approx(f_nu, rel=0.0, abs=1.0e-14)
    assert equal == pytest.approx(
        np.array([f_nu / 6.0, f_nu / 6.0, f_nu / 3.0, f_nu / 3.0]),
        rel=0.0,
        abs=1.0e-14,
    )

    split = np.asarray(
        _species_energy_weights_jax(
            jnp.asarray(f_nu),
            jnp.asarray(2.9),
            jnp.asarray(2.4),
        ),
        dtype=np.float64,
    )
    expected_raw = np.array([2.9**4, 2.9**4, 2.0 * 2.4**4, 2.0 * 2.4**4])
    assert split == pytest.approx(f_nu * expected_raw / expected_raw.sum(), rel=1.0e-14)
    assert split[2] / split[0] == pytest.approx(2.0 * (2.4 / 2.9) ** 4, rel=1.0e-14)


@pytest.mark.production
def test_stress_rhs_uses_species_degeneracy_weighted_energy_share() -> None:
    rhs_fn, _jac_fn, layout, *_ = _phase1_collisionless_system()
    y_equal = _state(layout, T_nu_e=2.9, T_nu_x=2.9)

    d_nue = _stress_rhs_delta(rhs_fn, layout, y_equal, species_index=0)
    d_nuebar = _stress_rhs_delta(rhs_fn, layout, y_equal, species_index=1)
    d_nux = _stress_rhs_delta(rhs_fn, layout, y_equal, species_index=2)
    d_nuxbar = _stress_rhs_delta(rhs_fn, layout, y_equal, species_index=3)

    assert d_nuebar / d_nue == pytest.approx(1.0, rel=1.0e-12)
    assert d_nux / d_nue == pytest.approx(2.0, rel=1.0e-12)
    assert d_nuxbar / d_nue == pytest.approx(2.0, rel=1.0e-12)

    y_split = _state(layout, T_nu_e=2.9, T_nu_x=2.4)
    split_ratio = (
        _stress_rhs_delta(rhs_fn, layout, y_split, species_index=2)
        / _stress_rhs_delta(rhs_fn, layout, y_split, species_index=0)
    )
    assert split_ratio == pytest.approx(2.0 * (2.4 / 2.9) ** 4, rel=1.0e-12)


@pytest.mark.production
def test_low_rank_jacobian_stress_columns_use_same_species_weights() -> None:
    _rhs_fn, jac_fn, layout, *_ = _phase1_collisionless_system()
    y = jnp.asarray(_state(layout, T_nu_e=2.9, T_nu_x=2.4))
    dense_jac = np.asarray(
        materialize_low_rank_jacobian(jac_fn(jnp.asarray(0.0), y)),
        dtype=np.float64,
    )
    perturb = _p2_spectral_perturbation().reshape(-1)
    i_sp = int(layout["i_Sp"])

    def directional(species_index: int) -> float:
        cols = _species_slice(layout, species_index)
        return float(dense_jac[i_sp, cols] @ perturb)

    d_nue = directional(0)
    d_nuebar = directional(1)
    d_nux = directional(2)
    d_nuxbar = directional(3)

    assert d_nuebar / d_nue == pytest.approx(1.0, rel=1.0e-12)
    assert d_nux / d_nue == pytest.approx(2.0 * (2.4 / 2.9) ** 4, rel=1.0e-12)
    assert d_nuxbar / d_nue == pytest.approx(2.0 * (2.4 / 2.9) ** 4, rel=1.0e-12)
