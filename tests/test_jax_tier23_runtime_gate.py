"""F06 prevents runtime-performance gates for the retired JAX endpoint."""
from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


def test_jax_tier23_runtime_path_is_unavailable():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(
            backend="jax_advanced",
            Sigma_H=0.01,
            correction_level=3,
            N_q=6,
        )
