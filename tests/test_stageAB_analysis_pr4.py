from __future__ import annotations

import numpy as np

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_analysis import compute_stageB_closure_residual, fit_power_law
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS


def test_stageB_closure_residual_zero_on_flrw_equilibrium():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_B_ELLS, n_species=1)
    resid = compute_stageB_closure_residual(state, sigma_h=0.0)
    assert resid.keep_l2 == 0.0
    assert resid.omit_self_l2 == 0.0
    assert resid.omit_f4_l2 == 0.0
    assert resid.omit_total_l2 == 0.0


def test_stageB_closure_residual_splits_self_and_f4_terms():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.zeros(grid, STAGE_B_ELLS, n_species=1)
    q = grid.nodes
    state.data[0, 0, :] = fermi_dirac(q)
    state.data[0, 1, :] = 1.0e-3 * q * np.exp(-q / 2.0)
    state.data[0, 2, :] = 5.0e-4 * q**2 * np.exp(-q / 2.5)
    resid = compute_stageB_closure_residual(state, sigma_h=0.1)
    assert resid.keep_l2 > 0.0
    assert resid.omit_self_l2 > 0.0
    assert resid.omit_f4_l2 > 0.0
    assert resid.omit_total_l2 > 0.0
    assert resid.ratio_total_to_keep > 0.0


def test_fit_power_law_recovers_known_slope():
    xs = [1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2]
    ys = [2.0 * x**2 for x in xs]
    fit = fit_power_law(xs, ys)
    assert abs(fit['slope'] - 2.0) < 1.0e-10
