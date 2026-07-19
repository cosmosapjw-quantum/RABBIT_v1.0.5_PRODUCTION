"""F06 hard-retirement lock for the former AP-unified public endpoint."""
from __future__ import annotations

import inspect

import pytest

from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND, CAPABILITY_BY_KEY
from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_ap_unified_name_is_not_registered_for_dispatch():
    assert "jax_ap_unified_tier3" not in CAPABILITY_BY_BACKEND
    assert "jax_typeI_ap_unified_tier3_candidate" not in CAPABILITY_BY_KEY


def test_ap_unified_name_is_rejected_as_retired():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend="jax_ap_unified_tier3")


def test_full_boltzmann_preflight_remains_a_nonpublic_component_oracle():
    cap = CAPABILITY_BY_KEY["jax_typeI_full_boltzmann_tier3_preflight"]
    assert cap.key not in {item.key for item in CAPABILITY_BY_BACKEND.values()}
    assert cap.validated_default is False


def test_retired_ap_specific_kwargs_are_absent_from_public_signature():
    params = inspect.signature(canonical_forward_solver).parameters
    for name in (
        "jax_tier3_collision_mode",
        "jax_tier3_nu_nu",
        "Sigma_H_minus",
    ):
        assert name not in params
