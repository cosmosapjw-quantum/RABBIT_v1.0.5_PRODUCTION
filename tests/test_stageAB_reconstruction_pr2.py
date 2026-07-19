from __future__ import annotations

import numpy as np

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_stageA_hierarchy import STAGE_A_ELLS
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS
from rabbit.transport.typeI_stageAB_reconstruct import (
    compute_reconstruction_diagnostics,
    gauss_legendre_mu_quadrature,
    reconstruct_nodal_distribution,
    reproject_active_moments,
)


def _build_stageB_test_state(n_q: int = 20) -> AxisymmetricHierarchyState:
    grid = MomentumGrid(N_q=n_q)
    state = AxisymmetricHierarchyState.zeros(grid, STAGE_B_ELLS, n_species=1)
    q = grid.nodes
    state.data[0, 0, :] = fermi_dirac(q)
    state.data[0, 1, :] = 5.0e-3 * q * np.exp(-q / 2.0)
    state.data[0, 2, :] = 8.0e-4 * q**2 * np.exp(-q / 2.5)
    return state


def test_r1_flrw_contract_stageA():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_A_ELLS, n_species=1)
    quad, f_nodal = reconstruct_nodal_distribution(state, mode="R1")
    expected = fermi_dirac(grid.nodes)[:, None]
    assert np.allclose(f_nodal[0], expected, atol=1e-14, rtol=1e-14)


def test_r2_flrw_contract_stageB():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_B_ELLS, n_species=1)
    quad, f_nodal = reconstruct_nodal_distribution(state, mode="R2")
    expected = fermi_dirac(grid.nodes)[:, None]
    assert np.allclose(f_nodal[0], expected, atol=1e-12, rtol=1e-12)


def test_r1_reprojection_consistent_for_stageB_state():
    state = _build_stageB_test_state(n_q=20)
    quadrature, f_nodal = reconstruct_nodal_distribution(state, mode="R1")
    reproj = reproject_active_moments(f_nodal, state.active_ells, quadrature)
    assert np.allclose(reproj, state.data, atol=1e-12, rtol=1e-12)


def test_r2_discrete_constraints_hold_for_stageB_state():
    state = _build_stageB_test_state(n_q=20)
    quadrature, f_nodal = reconstruct_nodal_distribution(state, mode="R2")
    reproj = reproject_active_moments(f_nodal, state.active_ells, quadrature)
    assert np.allclose(reproj, state.data, atol=1e-11, rtol=1e-11)


def test_reconstruction_diagnostics_report_zero_violation_in_flrw():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_A_ELLS, n_species=1)
    diag_r1 = compute_reconstruction_diagnostics(state, mode="R1", n_mu=17)
    diag_r2 = compute_reconstruction_diagnostics(state, mode="R2", n_mu=17)
    assert diag_r1.violation_fraction == 0.0
    assert diag_r2.violation_fraction == 0.0
    assert diag_r1.reprojection_l2_abs < 1e-12
    assert diag_r2.reprojection_l2_abs < 1e-10
