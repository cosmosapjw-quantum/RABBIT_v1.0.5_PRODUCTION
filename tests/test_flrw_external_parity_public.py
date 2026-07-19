"""Fail-closed lock for the retired public AP-unified FLRW endpoint.

The former tests compared an unmatched public JAX endpoint with published
numbers.  F06 keeps the underlying research module but removes that endpoint
from public inference, so no abundance or N_eff parity claim is made here.
"""
from __future__ import annotations

import pytest


@pytest.mark.production
@pytest.mark.gold
def test_external_parity_endpoint_absent_from_public_registry() -> None:
    from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND

    assert "jax_ap_unified_tier3" not in CAPABILITY_BY_BACKEND
    assert set(CAPABILITY_BY_BACKEND) == {"scipy", "auto"}


@pytest.mark.production
@pytest.mark.gold
def test_external_parity_endpoint_public_dispatch_fails_closed() -> None:
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired"):
        canonical_forward_solver(backend="jax_ap_unified_tier3")
