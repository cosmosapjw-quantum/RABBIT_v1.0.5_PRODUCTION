"""Separate ν_e / ν̄_e weak-rate coupling in the tier-3 full-Boltzmann path."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.driver_typeI_full_boltzmann import (
    _get_full_boltzmann_rhs,
    _initial_transport_state,
)
from rabbit.jax.solver_jax_rodas5p import materialize_low_rank_jacobian


jax.config.update("jax_enable_x64", True)


def _phase1_system():
    return _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode="jax_kernel_projected_preflight",
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


def _state(layout: dict) -> np.ndarray:
    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.0
    y0[layout["i_S"]] = 0.0
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 3.0
    y0[layout["i_tne"]] = 2.9
    y0[layout["i_tnx"]] = 2.8
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87])
    return y0


def _species_slice(layout: dict, species_index: int) -> slice:
    stride = int(layout["N_mu"]) * int(layout["N_q"])
    start = int(layout["i_transport"]) + int(species_index) * stride
    return slice(start, start + stride)


@pytest.mark.production
def test_full_boltzmann_weak_rhs_uses_separate_nuebar_monopole() -> None:
    rhs_fn, _jac_fn, layout, *_ = _phase1_system()
    y0 = _state(layout)

    base = np.asarray(rhs_fn(jnp.asarray(0.0), jnp.asarray(y0)), dtype=np.float64)

    y_nuebar = y0.copy()
    y_nuebar[_species_slice(layout, 1)] *= 0.5
    shifted_nuebar = np.asarray(
        rhs_fn(jnp.asarray(0.0), jnp.asarray(y_nuebar)), dtype=np.float64
    )

    y_nux = y0.copy()
    y_nux[_species_slice(layout, 2)] *= 0.5
    shifted_nux = np.asarray(
        rhs_fn(jnp.asarray(0.0), jnp.asarray(y_nux)), dtype=np.float64
    )

    i_xn = int(layout["i_net"])
    assert abs(shifted_nuebar[i_xn] - base[i_xn]) > 1.0
    assert shifted_nux[i_xn] == pytest.approx(base[i_xn], rel=0.0, abs=1.0e-12)


@pytest.mark.production
def test_full_boltzmann_low_rank_jacobian_has_nuebar_weak_columns() -> None:
    _rhs_fn, jac_fn, layout, *_ = _phase1_system()
    y = jnp.asarray(_state(layout))
    dense_jac = np.asarray(
        materialize_low_rank_jacobian(jac_fn(jnp.asarray(0.0), y)),
        dtype=np.float64,
    )

    i_xn = int(layout["i_net"])
    nuebar_cols = np.arange(_species_slice(layout, 1).start, _species_slice(layout, 1).stop)
    nux_cols = np.arange(_species_slice(layout, 2).start, _species_slice(layout, 2).stop)

    assert np.max(np.abs(dense_jac[i_xn, nuebar_cols])) > 1.0
    assert np.max(np.abs(dense_jac[i_xn, nux_cols])) < 1.0e-12
