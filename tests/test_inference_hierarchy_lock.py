"""F06 public-forward authority and retained component-catalog lock."""
from __future__ import annotations

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY


HIERARCHY = {
    "auto": {"key": "scipy_typeI_reference", "tier": "canonical", "scope": "TypeI"},
    "scipy": {"key": "scipy_typeI_reference", "tier": "canonical", "scope": "TypeI"},
}

EXPECTED_CATALOG_KEYS = [
    "scipy_typeI_reference",
    "scipy_typeI_tier2_per_species",
    "scipy_typeI_tier3_weak_budget",
    "jax_typeI_full_boltzmann_tier3_preflight",
    "jax_typeI_liveweak_cl3_tier2_teff_candidate",
    "jax_typeI_augmented_pstf_noqke_staging",
    "jax_classA_geometry",
    "jax_classA_driver",
    "jax_classA_characteristic",
    "jax_classB_driver",
    "jax_tilted_bbn",
    "jax_tilted_full_coupled",
    "jax_weak_cl3_kernel",
    "jax_curved_hierarchy",
    "jax_rodas5p_solver",
]


@pytest.mark.parametrize("backend", HIERARCHY)
def test_public_backend_contract(backend):
    cap = CAPABILITY_BY_BACKEND[backend]
    expected = HIERARCHY[backend]
    assert cap.key == expected["key"]
    assert cap.tier == expected["tier"]
    assert cap.physics_scope == expected["scope"]


def test_no_extra_public_backends():
    assert set(CAPABILITY_BY_BACKEND) == set(HIERARCHY)
    assert CAPABILITY_BY_BACKEND["auto"] is CAPABILITY_BY_BACKEND["scipy"]


def test_retained_component_and_geometry_catalog_is_exact():
    assert set(CAPABILITY_BY_KEY) == set(EXPECTED_CATALOG_KEYS)
    for cap in CAPABILITY_BY_BACKEND.values():
        assert cap.key in CAPABILITY_BY_KEY


@pytest.mark.parametrize(
    "backend",
    [
        "jax",
        "jax_advanced",
        "jax_characteristic",
        "jax_characteristic_tier2",
        "jax_characteristic_nonlrs",
        "jax_ap_unified_tier3",
        "jax_classA",
        "jax_classB",
        "jax_tilted",
        "jax_tilted_full_coupled",
    ],
)
def test_retired_backend_is_rejected(backend):
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend=backend)


def test_unknown_backend_raises():
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="Unknown backend"):
        canonical_forward_solver(Sigma_H=0.0, backend="nonexistent")
