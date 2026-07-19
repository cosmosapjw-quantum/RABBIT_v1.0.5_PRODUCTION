"""Jacobian-preconditioned remapped JAX kernel preflight checks."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pytest

os.environ.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")

pytest.importorskip("jax")

import jax
import jax.numpy as jnp

from rabbit.jax.driver_typeI_full_boltzmann import (
    JAXFullBoltzmannConfig,
    _get_full_boltzmann_rhs,
    _initial_transport_state,
    run_full_boltzmann_jax,
)


jax.config.update("jax_enable_x64", True)


def _phase1_rhs_and_jac(mode: str):
    return _get_full_boltzmann_rhs(
        phase=1,
        correction_level=0,
        collision_mode=mode,
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


def _remapped_kernel_test_state(layout: dict) -> jnp.ndarray:
    y0 = np.zeros(layout["n_total"], dtype=np.float64)
    y0[layout["i_Sp"]] = 0.0
    y0[layout["i_S"]] = 0.0
    i0 = layout["i_transport"]
    i1 = i0 + layout["n_transport"]
    y0[i0:i1] = _initial_transport_state(4, 6).reshape(-1)
    y0[layout["i_tg"]] = 3.0
    y0[layout["i_tne"]] = 2.85
    y0[layout["i_tnx"]] = 2.80
    y0[layout["i_net"] : layout["i_net"] + 2] = np.array([0.13, 0.87])
    return jnp.asarray(y0)


@lru_cache(maxsize=None)
def _run_preconditioned_remap(
    *,
    N_q: int = 6,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    max_steps: int = 2000,
):
    cfg = JAXFullBoltzmannConfig(
        Sigma_H_plus=0.0,
        N_mu=4,
        N_q=N_q,
        correction_level=0,
        n_reactions=12,
        collision_mode="jax_kernel_remap_preconditioned_preflight",
        thermo_tier=2,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        event_refine_steps=12,
    )
    return run_full_boltzmann_jax(cfg)


@pytest.mark.production
def test_jax_kernel_remap_preconditioner_changes_jacobian_not_rhs() -> None:
    raw_rhs, raw_jac, layout, *_ = _phase1_rhs_and_jac("jax_kernel_remap_preflight")
    pre_rhs, pre_jac, pre_layout, *_ = _phase1_rhs_and_jac(
        "jax_kernel_remap_preconditioned_preflight"
    )
    assert pre_layout["n_total"] == layout["n_total"]

    y = _remapped_kernel_test_state(layout)
    raw_dy = np.asarray(raw_rhs(jnp.asarray(0.0), y), dtype=np.float64)
    pre_dy = np.asarray(pre_rhs(jnp.asarray(0.0), y), dtype=np.float64)
    np.testing.assert_allclose(pre_dy, raw_dy, rtol=0.0, atol=0.0)

    raw_core = np.asarray(raw_jac(jnp.asarray(0.0), y).core_matrix, dtype=np.float64)
    pre_core = np.asarray(pre_jac(jnp.asarray(0.0), y).core_matrix, dtype=np.float64)
    diff = pre_core - raw_core
    n_bank = 3 * 6
    bank_delta = diff[-n_bank:, -n_bank:]

    assert raw_core.shape == (26, 37)
    assert pre_core.shape == raw_core.shape
    assert np.max(np.abs(diff[:-n_bank, :])) == pytest.approx(0.0, abs=0.0)
    assert np.max(np.abs(diff[:, :-n_bank])) == pytest.approx(0.0, abs=0.0)
    assert np.max(np.abs(bank_delta - np.diag(np.diag(bank_delta)))) == pytest.approx(
        0.0, abs=0.0
    )
    assert np.max(np.abs(np.diag(bank_delta))) > 1.0e3
    assert np.all(np.diag(bank_delta) <= 0.0)


@pytest.mark.production
@pytest.mark.slow
@pytest.mark.gold
def test_jax_kernel_remap_preconditioned_preflight_full_ode_smoke() -> None:
    result = _run_preconditioned_remap()

    assert result.success, result.metadata
    assert result.metadata["collision_scope_contract"] == (
        "jax_kernel_temperature_remap_jacobian_preconditioned_v1"
    )
    assert result.metadata["kernel_jacobian_preconditioner_enabled"] is True
    assert result.metadata["weak_monopole_scope_contract"] == "separate_nue_nuebar_live_f0_v1"
    assert result.metadata["stress_species_weight_contract"] == (
        "temperature4_degeneracy_weighted_species_stress_v1"
    )
    assert result.metadata["phase1_solver_diagnostics"]["target_reached"] is True
    assert result.n_steps_p1 > 0
    assert 3.04 < result.metadata["N_eff_measured"] < 3.10
    assert 0.23 < result.Yp < 0.25


@pytest.mark.production
@pytest.mark.slow
@pytest.mark.gold
def test_jax_kernel_remap_preconditioned_tolerance_envelope() -> None:
    loose = _run_preconditioned_remap(rtol=1.0e-4, atol=1.0e-6, max_steps=2000)
    production = _run_preconditioned_remap()
    tight = _run_preconditioned_remap(rtol=3.0e-6, atol=3.0e-8, max_steps=6000)

    assert loose.success, loose.metadata
    assert production.success, production.metadata
    assert tight.success, tight.metadata
    assert tight.n_steps_p1 > production.n_steps_p1

    assert (
        abs(production.metadata["N_eff_measured"] - tight.metadata["N_eff_measured"])
        < 2.0e-3
    )
    assert (
        abs(loose.metadata["N_eff_measured"] - tight.metadata["N_eff_measured"])
        < 5.0e-3
    )
    assert abs(production.Yp - tight.Yp) < 2.0e-4
    assert abs(loose.Yp - tight.Yp) < 4.0e-4


@pytest.mark.production
@pytest.mark.slow
def test_jax_kernel_remap_preconditioned_grid_sensitivity_is_documented() -> None:
    n6 = _run_preconditioned_remap()
    n8 = _run_preconditioned_remap(N_q=8, rtol=1.0e-5, atol=1.0e-7, max_steps=3000)
    n10 = _run_preconditioned_remap(N_q=10, rtol=1.0e-5, atol=1.0e-7, max_steps=3000)

    assert n6.success, n6.metadata
    assert n8.success, n8.metadata
    assert n10.success, n10.metadata

    n_eff_values = [
        n6.metadata["N_eff_measured"],
        n8.metadata["N_eff_measured"],
        n10.metadata["N_eff_measured"],
    ]
    assert n_eff_values[0] > n_eff_values[1] > n_eff_values[2]
    assert 1.5e-2 < (n_eff_values[0] - n_eff_values[2]) < 2.5e-2
    assert 3.04 < n_eff_values[2] < n_eff_values[0] < 3.10

    yps = [n6.Yp, n8.Yp, n10.Yp]
    assert max(yps) - min(yps) < 7.0e-4
