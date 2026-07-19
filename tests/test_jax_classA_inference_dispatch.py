"""F06 retirement lock for the former Class-A inference dispatch."""
from __future__ import annotations

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_classa_metadata_is_retained_but_not_dispatchable():
    assert "jax_classA" not in CAPABILITY_BY_BACKEND
    cap = CAPABILITY_BY_KEY["jax_classA_driver"]
    assert cap.tier == "substrate"
    assert cap.physics_scope == "ClassA_6types"
    assert cap.max_correction_level == 3


def test_auto_does_not_resolve_to_classa():
    assert CAPABILITY_BY_BACKEND["auto"].key == "scipy_typeI_reference"


def test_retired_classa_backend_is_rejected_before_legacy_kwargs():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend="jax_classA")


def test_removed_classa_kwargs_are_not_in_public_signature():
    import inspect

    params = inspect.signature(canonical_forward_solver).parameters
    for name in ("bianchi_type", "N1_init", "N2_init", "N3_init"):
        assert name not in params
