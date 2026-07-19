"""F06 prevents retired JAX endpoint comparisons from regaining authority."""
from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.parametrize("backend", ["jax", "jax_characteristic"])
def test_retired_shear_endpoint_is_unavailable(backend):
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(backend=backend, Sigma_H=0.1, N_q=20)
