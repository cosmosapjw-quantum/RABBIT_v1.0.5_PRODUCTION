"""F06 closes synthetic-recovery claims on the retired JAX endpoint."""
from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_jax_advanced_synthetic_forward_is_unavailable():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend="jax_advanced", Sigma_H=0.0, N_q=6)


def test_no_retired_jax_backend_is_publicly_registered():
    from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND

    assert set(CAPABILITY_BY_BACKEND) == {"scipy", "auto"}
