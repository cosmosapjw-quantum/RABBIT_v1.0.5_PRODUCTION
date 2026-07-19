"""F06 retirement lock for tilted public-forward dispatch names."""
from __future__ import annotations

import inspect

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_tilted_metadata_remains_catalog_only():
    assert "jax_tilted" not in CAPABILITY_BY_BACKEND
    assert "jax_tilted_full_coupled" not in CAPABILITY_BY_BACKEND
    assert CAPABILITY_BY_KEY["jax_tilted_bbn"].physics_scope == "tilted_all_types"
    assert (
        CAPABILITY_BY_KEY["jax_tilted_full_coupled"].physics_scope
        == "tilted_all_types_full_coupled"
    )


@pytest.mark.parametrize("backend", ["jax_tilted", "jax_tilted_full_coupled"])
def test_tilted_backend_names_are_hard_retired(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend=backend)


def test_tilted_kwargs_are_absent_from_public_signature():
    params = inspect.signature(canonical_forward_solver).parameters
    for name in (
        "v0",
        "tilt_stress_feedback",
        "tilt_weak_rate_boost",
        "tilt_cl3_angular_kernel",
    ):
        assert name not in params
