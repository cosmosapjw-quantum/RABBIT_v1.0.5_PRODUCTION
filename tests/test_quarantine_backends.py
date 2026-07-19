"""F06 hard-retirement and retained component-oracle registry lock."""
from __future__ import annotations

from rabbit.config.backend_capabilities import (
    ACTIVE_CANONICAL_BACKENDS,
    CAPABILITY_BY_BACKEND,
    CAPABILITY_BY_KEY,
    QUARANTINED_BACKENDS,
    is_quarantined,
)


def test_only_scipy_and_auto_have_public_forward_authority():
    assert set(CAPABILITY_BY_BACKEND) == {"scipy", "auto"}
    assert ACTIVE_CANONICAL_BACKENDS == frozenset({"scipy", "auto"})
    assert CAPABILITY_BY_BACKEND["auto"] is CAPABILITY_BY_BACKEND["scipy"]


def test_callable_quarantine_is_empty_after_hard_retirement():
    assert QUARANTINED_BACKENDS == frozenset()
    for backend in (
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
        "scipy",
        "auto",
        "nonexistent",
    ):
        assert is_quarantined(backend) is False


def test_component_oracles_and_geometry_metadata_remain_non_dispatchable():
    retained = {
        "jax_typeI_full_boltzmann_tier3_preflight",
        "jax_typeI_liveweak_cl3_tier2_teff_candidate",
        "jax_typeI_augmented_pstf_noqke_staging",
        "jax_weak_cl3_kernel",
        "jax_rodas5p_solver",
        "jax_classA_geometry",
        "jax_classA_driver",
        "jax_classA_characteristic",
        "jax_classB_driver",
        "jax_tilted_bbn",
        "jax_tilted_full_coupled",
    }
    assert retained <= set(CAPABILITY_BY_KEY)
    assert all(key not in CAPABILITY_BY_BACKEND for key in retained)
