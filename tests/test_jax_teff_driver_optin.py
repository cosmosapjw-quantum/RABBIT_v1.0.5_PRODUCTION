"""Teff and retired-JAX public-runtime fences."""
from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize("backend", ["jax", "jax_advanced"])
def test_retired_jax_endpoint_precedes_teff_policy(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend=backend, enable_teff=True, N_q=6)


def test_auto_rejects_deprecated_teff():
    with pytest.raises(ValueError, match="deprecated legacy"):
        canonical_forward_solver(
            Sigma_H=0.0, backend="auto", enable_teff=True, N_q=6
        )


def test_jax_advanced_remains_absent_from_public_registry():
    from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND

    assert "jax_advanced" not in CAPABILITY_BY_BACKEND
