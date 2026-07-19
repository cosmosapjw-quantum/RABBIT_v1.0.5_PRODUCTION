from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX required")

import jax.numpy as jnp
from numpy.polynomial.laguerre import laggauss

from rabbit.jax.weak_corrections_jax import (
    weak_correction_factor_jax, compute_I0_corrected_jax, correction_budget_for_channel_jax,
)
from rabbit.jax.weak_live_jax import (
    compute_live_rates_with_budget_from_monopoles,
    compute_live_rates_from_monopoles_cl012_jax,
    compute_live_rates_from_monopoles_level_specialized_jax,
)
from rabbit.weak.corrections import weak_correction_factor, compute_I0_corrected
from rabbit.weak.live_rates import compute_live_weak_rates


@pytest.mark.parametrize("channel", list("abcdef"))
def test_channel_correction_factor_matches_numpy_reference(channel):
    E = np.linspace(1.02, 2.45, 9)
    ref = weak_correction_factor(E, channel, enable_coulomb=True, enable_radiative=True)
    got = np.asarray(weak_correction_factor_jax(jnp.asarray(E), channel, True, True))
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_I0_corrected_matches_numpy_reference(level):
    enable_coulomb = level >= 1
    enable_radiative = level >= 2
    ref = compute_I0_corrected(enable_coulomb=enable_coulomb, enable_radiative=enable_radiative)
    got = compute_I0_corrected_jax(enable_coulomb=enable_coulomb, enable_radiative=enable_radiative)
    assert abs(got - ref) / ref < 1e-10


def test_budget_object_records_level_and_mode():
    E = jnp.linspace(1.02, 2.45, 8)
    bud = correction_budget_for_channel_jax(E, 'c', 2)
    assert bud.channel == 'c'
    assert bud.correction_level == 2
    assert bud.total_mean >= bud.coulomb_mean


@pytest.mark.parametrize("level", [0, 1, 2])
def test_live_weak_budget_parity_fd_window(level):
    q_nodes_np, _ = laggauss(6)
    q_nodes = jnp.asarray(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    got = compute_live_rates_with_budget_from_monopoles(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0, correction_level=level
    )
    ref = compute_live_weak_rates(
        np.asarray(f0), np.asarray(f0), np.asarray(q_nodes), 1.0, 1.0, 878.4,
        compute_iso_reference=False, correction_level=level,
    )
    assert abs(got.lambda_np - ref.lambda_np) / ref.lambda_np < 5e-9
    if ref.lambda_pn > 0:
        assert abs(got.lambda_pn - ref.lambda_pn) / ref.lambda_pn < 5e-9
    assert got.correction_level == level


@pytest.mark.parametrize("level", [0, 1, 2])
def test_jittable_live_weak_cl012_kernel_matches_numpy_reference(level):
    q_nodes_np, _ = laggauss(6)
    q_nodes = jnp.asarray(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    lnp, lpn, I0 = compute_live_rates_from_monopoles_cl012_jax(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0, correction_level=level
    )
    ref = compute_live_weak_rates(
        np.asarray(f0), np.asarray(f0), np.asarray(q_nodes), 1.0, 1.0, 878.4,
        compute_iso_reference=False, correction_level=level,
    )
    assert abs(float(lnp) - ref.lambda_np) / ref.lambda_np < 5e-9
    if ref.lambda_pn > 0:
        assert abs(float(lpn) - ref.lambda_pn) / ref.lambda_pn < 5e-9
    assert float(I0) > 0.0


@pytest.mark.parametrize("level", [0, 1, 2])
def test_level_specialized_live_weak_kernel_matches_generic_dispatch(level):
    q_nodes_np, _ = laggauss(6)
    q_nodes = jnp.asarray(q_nodes_np)
    f0 = 1.0 / (jnp.exp(q_nodes) + 1.0)
    generic = compute_live_rates_from_monopoles_cl012_jax(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0, correction_level=level
    )
    specialized = compute_live_rates_from_monopoles_level_specialized_jax(
        jnp.array(1.0), jnp.array(1.0), jnp.array(878.4), q_nodes, f0, f0, correction_level=level
    )
    assert np.allclose(np.asarray(generic[0]), np.asarray(specialized[0]), rtol=0.0, atol=0.0)
    assert np.allclose(np.asarray(generic[1]), np.asarray(specialized[1]), rtol=0.0, atol=0.0)
    assert np.allclose(np.asarray(generic[2]), np.asarray(specialized[2]), rtol=0.0, atol=0.0)
