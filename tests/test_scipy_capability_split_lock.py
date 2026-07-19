from rabbit.config.backend_capabilities import (
    CAPABILITY_BY_BACKEND,
    CAPABILITY_BY_KEY,
)


def test_scipy_dispatch_remains_canonical_reference():
    assert CAPABILITY_BY_BACKEND["scipy"].key == "scipy_typeI_reference"
    # BD619: auto now defaults to the scipy Type-I reference.
    assert CAPABILITY_BY_BACKEND["auto"].key == "scipy_typeI_reference"


def test_scipy_subsurface_capabilities_present():
    assert "scipy_typeI_tier2_per_species" in CAPABILITY_BY_KEY
    assert "scipy_typeI_tier3_weak_budget" in CAPABILITY_BY_KEY


def test_scipy_subsurface_capabilities_are_candidate_not_canonical():
    tier2 = CAPABILITY_BY_KEY["scipy_typeI_tier2_per_species"]
    tier3 = CAPABILITY_BY_KEY["scipy_typeI_tier3_weak_budget"]

    assert tier2.tier == "candidate"
    assert tier3.tier == "candidate"
    assert tier2.backend == "scipy"
    assert tier3.backend == "scipy"
