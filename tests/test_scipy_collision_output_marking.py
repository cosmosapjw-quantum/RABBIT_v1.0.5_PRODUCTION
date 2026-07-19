"""tests/test_scipy_collision_output_marking.py — BD612-F7 regression.

The public ``canonical_forward_solver`` hard-blocks ``enable_collisions=True`` on
every JAX/batch backend, but the default SciPy path ACCEPTS it (a documented
opt-in boundary) and runs the calibrated-RTA gather-scatter collision bridge —
whose collisional characteristic reference is a documented anomaly (B4-PR4 /
``test_b4_collisional_char_reference_unsound``). BD612-F7 marks that opt-in output
so a caller cannot mistake an unvalidated candidate result for a canonical one.

This locks the marker; it does NOT validate collision physics.
"""

from __future__ import annotations

import pytest

from rabbit.inference.forward_likelihood import canonical_forward_solver


@pytest.mark.slow
def test_scipy_collision_opt_in_output_is_marked_candidate():
    """enable_collisions=True on the SciPy path tags metadata as a calibrated-RTA
    candidate; the default (collisions off) leaves it None."""
    on = canonical_forward_solver(
        0.0, N_q=8, n_reactions=12, enable_collisions=True,
    )
    off = canonical_forward_solver(
        0.0, N_q=8, n_reactions=12, enable_collisions=False,
    )
    assert on.metadata.get("collision_model") == "calibrated_rta_candidate"
    assert off.metadata.get("collision_model") is None


def test_retired_jax_collision_endpoint_stays_fail_closed():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(0.0, enable_collisions=True, backend="jax_characteristic")
