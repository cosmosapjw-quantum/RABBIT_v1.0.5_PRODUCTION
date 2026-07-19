"""JAX/XLA-only kernel-backend honesty contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


jax.config.update("jax_enable_x64", True)


def test_jax_backend_report_is_authoritative_default():
    from rabbit.jax.kernel_backend import inspect_kernel_backend

    report = inspect_kernel_backend("jax", scope="collision_rate")
    assert report.effective == "jax"
    assert report.available is True
    assert report.jit_safe is True
    assert report.grad_safe is True
    assert report.host_mediated is False
    assert report.as_dict()["kernel_backend_external_call"] is False


def test_auto_backend_reports_jax():
    from rabbit.jax.kernel_backend import inspect_kernel_backend

    report = inspect_kernel_backend("auto", scope="collision_rate")
    assert report.effective == "jax"
    assert report.available is True
    assert report.jit_safe is True


@pytest.mark.parametrize("backend", ["rust", "rust_ffi", "gpu_rust_ffi", "wgpu"])
def test_removed_external_backends_raise(backend):
    from rabbit.jax.kernel_backend import inspect_kernel_backend

    with pytest.raises(ValueError, match="removed"):
        inspect_kernel_backend(backend, scope="collision_rate")


def test_collision_rate_auto_backend_computes_jax_baseline():
    from rabbit.jax.collision_rates_jax import (
        collision_rate_kernel_backend_report,
        total_rate_nu_e_jax,
    )

    t = jnp.array([0.5, 1.0, 2.0])
    assert jnp.allclose(total_rate_nu_e_jax(t), total_rate_nu_e_jax(t, kernel_backend="auto"))
    metadata = collision_rate_kernel_backend_report("auto")
    assert metadata["kernel_backend_effective"] == "jax"
    assert metadata["kernel_backend_external_call"] is False


def test_jax_backend_jit_probe_matches_report():
    from rabbit.jax.collision_rates_jax import total_rate_jax

    t = jnp.array([0.5, 1.0, 2.0])
    compiled = jax.jit(lambda x: total_rate_jax(x, "nue", kernel_backend="jax"))
    assert jnp.allclose(compiled(t), total_rate_jax(t, "nue"))
