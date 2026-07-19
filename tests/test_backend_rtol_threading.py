"""F06 tolerance-surface authority lock."""
from __future__ import annotations

import inspect

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_scipy_public_signature_retains_solver_tolerances():
    params = inspect.signature(canonical_forward_solver).parameters
    assert "rtol" in params
    assert "atol" in params


def test_retired_classa_backend_cannot_claim_tolerance_threading():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(
            backend="jax_classA", Sigma_H=0.05, rtol=1.234e-9, atol=5.6e-11
        )
