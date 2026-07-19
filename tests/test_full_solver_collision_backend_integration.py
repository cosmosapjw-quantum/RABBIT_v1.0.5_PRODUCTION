"""Full-solver collision-backend integration gates."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")


def test_ap_preconditioner_backend_auto_matches_jax():
    import jax.numpy as jnp
    from rabbit.jax.collision_ap_preconditioner_jax import compute_ap_preconditioner_diag

    q = jnp.array([0.5, 1.0, 2.0, 4.0])
    auto = compute_ap_preconditioner_diag(1.0, 1.0, q, N_mu=2, kernel_backend="auto")
    jax = compute_ap_preconditioner_diag(1.0, 1.0, q, N_mu=2, kernel_backend="jax")
    assert auto.shape == (3, 8)
    assert jnp.allclose(auto, jax)


def test_full_boltzmann_config_accepts_auto_as_jax_safe_backend():
    from rabbit.jax.driver_typeI_full_boltzmann import JAXFullBoltzmannConfig

    cfg = JAXFullBoltzmannConfig(
        collision_mode="ap_unified_preflight",
        thermo_tier=2,
        N_mu=4,
        N_q=6,
        collision_kernel_backend="auto",
    )
    assert cfg.collision_kernel_backend == "jax"
    assert cfg._collision_kernel_metadata["kernel_backend_effective"] == "jax"
    assert cfg._collision_kernel_metadata["kernel_backend_jit_safe"] is True


def test_full_boltzmann_config_rejects_removed_external_backend():
    from rabbit.jax.driver_typeI_full_boltzmann import JAXFullBoltzmannConfig

    with pytest.raises(ValueError, match="removed"):
        JAXFullBoltzmannConfig(
            collision_mode="ap_unified_preflight",
            thermo_tier=2,
            N_mu=4,
            N_q=6,
            collision_kernel_backend="rust",
        )


def test_public_ap_unified_endpoint_is_retired_before_solve():
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend="jax_ap_unified_tier3", N_q=6)
