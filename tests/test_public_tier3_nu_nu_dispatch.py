"""Retirement contracts for the former public AP-unified tier-3 endpoint."""
from __future__ import annotations

import inspect

import pytest


@pytest.mark.production
def test_ap_unified_absent_from_public_registry_but_component_metadata_preserved() -> None:
    from rabbit.config.backend_capabilities import (
        CAPABILITY_BY_BACKEND,
        CAPABILITY_BY_KEY,
    )

    assert "jax_ap_unified_tier3" not in CAPABILITY_BY_BACKEND
    assert "jax_typeI_full_boltzmann_tier3_preflight" in CAPABILITY_BY_KEY


@pytest.mark.production
def test_ap_unified_public_dispatch_fails_closed() -> None:
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired"):
        canonical_forward_solver(backend="jax_ap_unified_tier3")


@pytest.mark.production
def test_public_tier3_option_removed_from_canonical_signature() -> None:
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    assert "jax_tier3_nu_nu" not in inspect.signature(
        canonical_forward_solver
    ).parameters


@pytest.mark.production
@pytest.mark.parametrize(
    "mode",
    [
        "ap_unified_nu_nu_preflight",
        "ap_unified_nu_nu_spectral_preflight",
        "ap_unified_nu_nu_spectral_accuracy_preflight",
    ],
)
def test_private_full_boltzmann_component_retains_nu_nu_modes(mode: str) -> None:
    pytest.importorskip("jax", reason="JAX component oracle required")
    from rabbit.jax.driver_typeI_full_boltzmann import JAXFullBoltzmannConfig

    cfg = JAXFullBoltzmannConfig(
        N_mu=4,
        N_q=6,
        thermo_tier=2,
        collision_mode=mode,
    )
    assert cfg.collision_mode == mode


@pytest.mark.production
def test_private_full_boltzmann_component_rejects_qke_mode() -> None:
    pytest.importorskip("jax", reason="JAX component oracle required")
    from rabbit.jax.driver_typeI_full_boltzmann import JAXFullBoltzmannConfig

    with pytest.raises(ValueError, match="collision_mode"):
        JAXFullBoltzmannConfig(collision_mode="flavour_qke")
