"""F06 retirement lock for the former JAX tier-2/3 endpoint."""
from __future__ import annotations

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize("correction_level", [2, 3])
def test_jax_advanced_tier23_endpoint_is_retired(correction_level):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(
            backend="jax_advanced",
            Sigma_H=0.0,
            correction_level=correction_level,
            N_q=6,
        )


def test_tier23_forward_capability_is_not_public_or_catalogued():
    assert "jax_advanced" not in CAPABILITY_BY_BACKEND
    assert "jax_typeI_tier3_weak_budget" not in CAPABILITY_BY_KEY
