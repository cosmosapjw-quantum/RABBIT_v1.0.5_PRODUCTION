from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")


def _fd(q):
    return 1.0 / (np.exp(np.minimum(q, 500.0)) + 1.0)


def _kernel_energy_exchange(
    *,
    T_gamma: float,
    T_nu_e: float,
    T_nu_x: float,
    project_mangano_rate: bool = False,
):
    import jax.numpy as jnp

    from rabbit.jax.collisions_jax import laguerre_grid
    from rabbit.jax.driver_typeI_full_boltzmann import (
        _collision_bank_energy_exchange_jax,
        _collision_jax_kernel_bank_core_jax,
    )

    q_nodes_np, q_weights_np = laguerre_grid(8)
    f = _fd(q_nodes_np)
    bank = jnp.asarray(np.concatenate([f, f, f]))
    q_nodes = jnp.asarray(q_nodes_np)
    q_weights = jnp.asarray(q_weights_np)
    collision_rhs = _collision_jax_kernel_bank_core_jax(
        bank,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=jnp.asarray(T_gamma),
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
        H_inv_sec=jnp.asarray(1.0e20),
        temperature_frame_remap=True,
        project_mangano_rate=project_mangano_rate,
    )
    dQ_e, dQ_x = _collision_bank_energy_exchange_jax(
        collision_rhs,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_nu_e=jnp.asarray(T_nu_e),
        T_nu_x=jnp.asarray(T_nu_x),
    )
    return float(dQ_e), float(dQ_x), float(np.max(np.abs(np.asarray(collision_rhs))))


@pytest.mark.production
@pytest.mark.gold
def test_jax_kernel_temperature_remap_preserves_detailed_balance():
    dQ_e, dQ_x, max_abs_rhs = _kernel_energy_exchange(
        T_gamma=2.0,
        T_nu_e=2.0,
        T_nu_x=2.0,
    )
    assert abs(dQ_e) < 1.0e-30
    assert abs(dQ_x) < 1.0e-30
    assert max_abs_rhs < 1.0e-28


@pytest.mark.production
@pytest.mark.gold
def test_jax_kernel_temperature_remap_heats_neutrinos_when_plasma_hotter():
    dQ_e, dQ_x, max_abs_rhs = _kernel_energy_exchange(
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
    )
    assert dQ_e > 0.0
    assert dQ_x > 0.0
    assert max_abs_rhs > 0.0


@pytest.mark.production
@pytest.mark.gold
def test_jax_kernel_temperature_remap_mangano_projection_preserves_heating_sign():
    raw_e, raw_x, _ = _kernel_energy_exchange(
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        project_mangano_rate=False,
    )
    projected_e, projected_x, _ = _kernel_energy_exchange(
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        project_mangano_rate=True,
    )
    assert raw_e > 0.0 and raw_x > 0.0
    assert projected_e > 0.0 and projected_x > 0.0
    assert abs(projected_e - raw_e) > 0.0
