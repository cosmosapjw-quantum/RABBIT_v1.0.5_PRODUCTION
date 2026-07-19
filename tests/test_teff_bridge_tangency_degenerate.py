"""tests/test_teff_bridge_tangency_degenerate.py — PR#24 (AUDIT2-1): lock the tangency diagnostic
as DEGENERATE, so its docstring can never be silently re-promoted to a validity guard.

External audit BD601 (round 2) flagged `transport.teff_collision_bridge.tangency_diagnostic` as
documented like a validated Teff-manifold guard ("guarantees entropy cost < 1e-4") while being a
degenerate proxy: its P2 term is the input-independent constant -0.75, and it only sees per-ray Θ
variance (never the spectral shape along a ray). These tests pin that degeneracy:
  * D2 stays ~0 across the whole physical shear range (it never fires);
  * D2 stays ~0 even for a strongly-collided state (it is blind to the distortion it claims to flag);
  * D2 == sqrt(max(Θ_var/Θ_mean² - 0.5625, 0)) exactly -- the 0.5625 = (-0.75)² is a CONSTANT, proving
    the P2 term is input-independent and the diagnostic depends only on cross-ray Θ variance, not on
    any actual spectral distortion.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MomentumGrid
from rabbit.transport.teff_collision_bridge import (
    apply_gather_scatter_collision,
    tangency_diagnostic,
)

_N_Q, _N_MU = 20, 8


def _rays(i_scale: float):
    mu, w0 = np.polynomial.legendre.leggauss(_N_MU)
    I = i_scale * mu
    J = 1.0 + 0.1 * mu
    return I, J, w0


@pytest.mark.parametrize("i_scale", [0.05, 0.1, 0.2, 0.3])
def test_tangency_stays_zero_across_physical_shear(i_scale):
    """Across the entire physical shear range the diagnostic is ~0 -- it never fires, so it is not a
    real distortion guard."""
    grid = MomentumGrid(N_q=_N_Q)
    I, J, w0 = _rays(i_scale)
    D2 = tangency_diagnostic(I, J, w0, grid.nodes, grid.weights)
    assert D2 < 1e-9, f"tangency unexpectedly nonzero at i_scale={i_scale}: {D2}"


def test_tangency_is_blind_to_strong_collision_distortion():
    """A strongly-heated collision step (T_g=5, T_nu=1) produces a real, large monopole collision
    integral, yet the tangency diagnostic reports ~0 -- it does NOT detect the distortion it was
    meant to measure."""
    grid = MomentumGrid(N_q=_N_Q)
    I, J, w0 = _rays(0.3)
    res = apply_gather_scatter_collision(
        I, J, w0, grid.nodes, grid.weights, 5.0, 1.0, 1.0, compute_tangency=True)
    assert np.max(np.abs(res.C_monopole)) > 0.0   # the collision genuinely distorts the distribution
    assert res.tangency_D2 < 1e-9                  # ... yet the diagnostic is blind to it


@pytest.mark.parametrize("i_scale", [0.1, 0.5, 1.0])
def test_tangency_p2_term_is_the_constant_minus_0p75(i_scale):
    """D2 == sqrt(max(T2 - 0.5625, 0)) exactly, where T2 = Θ_var/Θ_mean² and 0.5625 = (-0.75)² is a
    CONSTANT independent of the ray state. This proves the P2 term collapses to -0.75, so the
    diagnostic is governed solely by cross-ray Θ variance against a fixed threshold -- not by any
    spectral distortion. (A genuine, input-dependent P2 would break this fixed-0.5625 prediction.)"""
    grid = MomentumGrid(N_q=_N_Q)
    I, J, w0 = _rays(i_scale)
    theta = np.exp(-2.0 * I)
    theta_mean = 0.5 * np.sum(w0 * J * theta)
    theta_var = 0.5 * np.sum(w0 * J * (theta - theta_mean) ** 2)
    T2 = theta_var / max(theta_mean ** 2, 1e-30)
    expected = float(np.sqrt(max(T2 - 0.5625, 0.0)))   # 0.5625 = (-0.75)^2, the constant P2 term
    D2 = tangency_diagnostic(I, J, w0, grid.nodes, grid.weights)
    np.testing.assert_allclose(D2, expected, rtol=1e-12, atol=1e-15)
