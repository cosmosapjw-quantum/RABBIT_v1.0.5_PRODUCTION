"""JAX/XLA collision-rate backend dispatch tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


jax.config.update("jax_enable_x64", True)


def test_default_backend_is_jax():
    from rabbit.jax.collision_rates_jax import (
        total_rate_jax,
        total_rate_nu_e_jax,
        total_rate_nu_x_jax,
    )

    T = jnp.array([0.5, 1.0, 2.0])
    assert jnp.allclose(total_rate_nu_e_jax(T), total_rate_nu_e_jax(T, backend="jax"))
    assert jnp.allclose(total_rate_nu_x_jax(T), total_rate_nu_x_jax(T, backend="jax"))
    assert jnp.allclose(total_rate_jax(T, "nue"), total_rate_jax(T, "nue", backend="jax"))


def test_kernel_backend_auto_is_jax():
    from rabbit.jax.collision_rates_jax import (
        gamma_over_H_jax,
        total_rate_jax,
        total_rate_nu_e_jax,
        total_rate_nu_x_jax,
    )

    T = jnp.array([0.5, 1.0, 2.0])
    assert jnp.allclose(total_rate_nu_e_jax(T), total_rate_nu_e_jax(T, kernel_backend="auto"))
    assert jnp.allclose(total_rate_nu_x_jax(T), total_rate_nu_x_jax(T, kernel_backend="auto"))
    assert jnp.allclose(total_rate_jax(T, "nue"), total_rate_jax(T, "nue", kernel_backend="auto"))
    assert jnp.allclose(
        gamma_over_H_jax(T, jnp.ones_like(T), "nue"),
        gamma_over_H_jax(T, jnp.ones_like(T), "nue", kernel_backend="auto"),
    )


def test_kernel_backend_metadata_surface_is_jax_only():
    from rabbit.jax.collision_rates_jax import collision_rate_kernel_backend_report

    meta = collision_rate_kernel_backend_report("jax")
    assert meta["kernel_backend_scope"] == "collision_rate"
    assert meta["kernel_backend_jit_safe"] is True
    assert meta["kernel_backend_host_mediated"] is False
    assert meta["kernel_backend_external_call"] is False


@pytest.mark.parametrize("backend", ["rust", "rust_ffi", "gpu_rust_ffi", "wgpu"])
def test_removed_backends_raise(backend):
    from rabbit.jax.collision_rates_jax import total_rate_nu_e_jax

    with pytest.raises(ValueError, match="removed"):
        total_rate_nu_e_jax(jnp.array([1.0]), backend=backend)


def test_unknown_backend_raises():
    from rabbit.jax.collision_rates_jax import total_rate_nu_e_jax

    with pytest.raises(ValueError, match=r"unknown backend"):
        total_rate_nu_e_jax(jnp.array([1.0]), backend="bogus")
