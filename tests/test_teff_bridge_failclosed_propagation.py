"""tests/test_teff_bridge_failclosed_propagation.py — BD612-F6 regression.

The calibrated-RTA gather-scatter bridge used to SILENTLY sanitize a non-finite
runtime collision field (``np.nan_to_num`` on ``C_monopole`` and ``delta_I``),
zeroing a bad state instead of surfacing it. That violated the repo's
preserve-raw-failed-states rule. BD612-F6 removed those runtime sanitizers in
BOTH twins so a non-finite collision field now PROPAGATES (fails loud downstream)
rather than being masked as a benign zero shift.

These tests lock the propagation contract and its numpy<->JAX symmetry. They do
NOT validate collision physics — the bridge is a default-OFF candidate surface.

Note: the deterministic ``rho_ref`` construction guard (zeros on a pathological
q-grid) is intentionally retained — it guards a build-time constant, not a
runtime state.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MomentumGrid
from rabbit.transport.teff_collision_bridge import (
    apply_gather_scatter_collision,
    scatter_collision_to_rays,
)

_N_Q, _N_MU = 20, 8


def _rays():
    mu, w0 = np.polynomial.legendre.leggauss(_N_MU)
    I = 0.05 * mu
    J = 1.0 + 0.1 * mu
    grid = MomentumGrid(N_q=_N_Q)
    return I, J, w0, grid.nodes, grid.weights


def test_numpy_scatter_propagates_non_finite_collision_field():
    """A non-finite C_monopole must PROPAGATE to a non-finite delta_I, not be
    silently zeroed (the pre-BD612-F6 behavior)."""
    I, J, w0, q_nodes, q_weights = _rays()
    C_monopole = np.zeros(_N_Q)
    C_monopole[3] = np.nan  # inject a bad runtime collision value
    delta_I = scatter_collision_to_rays(C_monopole, I, J, w0, q_nodes, q_weights, H_inv_sec=1.0)
    assert not np.all(np.isfinite(delta_I)), (
        "non-finite collision field was silently sanitized — F-6 regression"
    )


def test_numpy_scatter_finite_input_stays_finite():
    """Sanity: a finite collision field still yields a finite, non-trivial delta_I."""
    I, J, w0, q_nodes, q_weights = _rays()
    C_monopole = np.linspace(-1e-3, 1e-3, _N_Q)
    delta_I = scatter_collision_to_rays(C_monopole, I, J, w0, q_nodes, q_weights, H_inv_sec=1.0)
    assert np.all(np.isfinite(delta_I))


def test_numpy_bridge_propagates_non_finite_ray_state():
    """End-to-end numpy bridge: a NaN in the ray state must propagate to delta_I."""
    I, J, w0, q_nodes, q_weights = _rays()
    I = I.copy()
    I[0] = np.nan
    res = apply_gather_scatter_collision(I, J, w0, q_nodes, q_weights, 2.0, 1.6, 1.0)
    assert not np.all(np.isfinite(res.delta_I)), (
        "NaN ray state was silently sanitized to a finite delta_I — F-6 regression"
    )


def test_jax_twin_propagates_non_finite_ray_state():
    """The JAX twin must mirror the numpy propagation contract (parity on the error
    path, not just on finite inputs)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from rabbit.jax.teff_collision_bridge_jax import apply_gather_scatter_collision_jax

    I, J, w0, q_nodes, q_weights = _rays()
    I = I.copy()
    I[0] = np.nan
    res = apply_gather_scatter_collision_jax(
        jax.numpy.asarray(I), jax.numpy.asarray(J), jax.numpy.asarray(w0),
        jax.numpy.asarray(q_nodes), jax.numpy.asarray(q_weights), 2.0, 1.6, 1.0,
    )
    assert not bool(np.all(np.isfinite(np.asarray(res.delta_I)))), (
        "JAX twin silently sanitized a NaN ray state — F-6 twin-parity regression"
    )
