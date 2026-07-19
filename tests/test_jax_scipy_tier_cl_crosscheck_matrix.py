"""F06 authority replacement for the retired endpoint crosscheck matrix."""
from __future__ import annotations

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize("backend", ["jax", "jax_advanced"])
def test_retired_jax_matrix_endpoint_is_rejected(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend=backend, Sigma_H=0.0, N_q=6)


def test_scipy_tier_capabilities_remain_catalogued():
    assert set(CAPABILITY_BY_BACKEND) == {"scipy", "auto"}
    assert {
        "scipy_typeI_reference",
        "scipy_typeI_tier2_per_species",
        "scipy_typeI_tier3_weak_budget",
    } <= set(CAPABILITY_BY_KEY)


def test_retired_typei_forward_capability_keys_are_absent():
    retired = {
        "jax_typeI_liveweak_cl3_tier1",
        "jax_typeI_tier3_weak_budget",
    }
    assert retired.isdisjoint(CAPABILITY_BY_KEY)
