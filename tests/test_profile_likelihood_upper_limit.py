"""tests/test_profile_likelihood_upper_limit.py — BD599 W2-PR2 (INF-2).

The Σ_H profile-likelihood upper limit must not report a grid-edge or
solver-failure artifact as a data constraint. When the allowed region reaches the
last finite grid point, Σ_H is unconstrained (the "limit" would just track grid
extent); and +inf χ² points (solver failures, e.g. logL=-inf at Σ_H~1) must be
masked so they cannot fabricate a finite exclusion.
"""

from __future__ import annotations

import numpy as np

from rabbit.inference.forward_likelihood import GridResult
from rabbit.inference.model_comparison import profile_likelihood


def _grid_result(grid, chi2):
    grid = np.asarray(grid, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    return GridResult(
        param_names=["Sigma_H"],
        param_grids={"Sigma_H": grid},
        log_likelihood=-0.5 * chi2,
        chi2=chi2,
        best_fit={"Sigma_H": float(grid[int(np.argmin(chi2))])},
        best_chi2=float(np.min(chi2[np.isfinite(chi2)])),
    )


def test_unconstrained_when_allowed_region_reaches_grid_edge():
    """Flat χ² (Δχ²=0 everywhere) → Σ_H unconstrained, not limit==grid[-1]."""
    r = profile_likelihood(_grid_result([0.0, 0.25, 0.5, 0.75], [0.0, 0.0, 0.0, 0.0]))
    assert r.upper_limit_unconstrained[0.95] is True
    assert not np.isfinite(r.upper_limits[0.95])  # inf, not the grid edge 0.75


def test_solver_failure_points_masked_not_fabricating_limit():
    """A +inf χ² (solver failure) at high Σ_H must not fabricate a finite limit."""
    r = profile_likelihood(_grid_result([0.0, 0.25, 0.5, 0.75],
                                        [0.0, 0.0, 0.0, np.inf]))
    # allowed region reaches the last FINITE point (0.5) → unconstrained
    assert r.upper_limit_unconstrained[0.95] is True
    assert not np.isfinite(r.upper_limits[0.95])


def test_genuine_constraint_still_reported():
    """A χ² that rises above threshold mid-grid gives a real, finite limit."""
    r = profile_likelihood(_grid_result([0.0, 0.25, 0.5, 0.75],
                                        [0.0, 0.0, 10.0, 20.0]))
    assert r.upper_limit_unconstrained[0.95] is False
    assert r.upper_limits[0.95] == 0.25  # last allowed grid point, not the edge
