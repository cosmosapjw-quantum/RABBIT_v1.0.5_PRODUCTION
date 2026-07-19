from __future__ import annotations

import numpy as np

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_stageA_hierarchy import (
    STAGE_A_ELLS,
    compute_hierarchy_rhs_typeI_stageA,
)
from rabbit.transport.typeI_stageB_hierarchy import (
    STAGE_B_ELLS,
    compute_hierarchy_rhs_typeI_stageB,
)


def test_stageA_flrw_fd_invariant_rhs_zero():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_A_ELLS, n_species=2)
    rhs = compute_hierarchy_rhs_typeI_stageA(state, Sigma_H=0.0)
    assert np.allclose(rhs, 0.0, atol=0.0, rtol=0.0)


def test_stageB_flrw_fd_invariant_rhs_zero():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_B_ELLS, n_species=2)
    rhs = compute_hierarchy_rhs_typeI_stageB(state, Sigma_H=0.0)
    assert np.allclose(rhs, 0.0, atol=0.0, rtol=0.0)


def test_stageA_small_shear_sources_quadrupole_from_fd_only():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_A_ELLS, n_species=1)
    rhs = AxisymmetricHierarchyState.from_flat(
        compute_hierarchy_rhs_typeI_stageA(state, Sigma_H=1.0e-3),
        grid,
        STAGE_A_ELLS,
        n_species=1,
    )
    assert np.allclose(rhs.moment(0, 0), 0.0, atol=1e-14)
    f2_rhs = rhs.moment(0, 2)
    assert np.linalg.norm(f2_rhs) > 0.0
    assert np.all(f2_rhs[:-1] < 0.0)
    assert abs(f2_rhs[-1]) < 1.0e-20


def test_stageB_small_shear_sources_quadrupole_but_not_hexadecapole_from_fd_only():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_B_ELLS, n_species=1)
    rhs = AxisymmetricHierarchyState.from_flat(
        compute_hierarchy_rhs_typeI_stageB(state, Sigma_H=1.0e-3),
        grid,
        STAGE_B_ELLS,
        n_species=1,
    )
    assert np.allclose(rhs.moment(0, 0), 0.0, atol=1e-14)
    f2_rhs = rhs.moment(0, 2)
    assert np.linalg.norm(f2_rhs) > 0.0
    assert np.all(f2_rhs[:-1] < 0.0)
    assert abs(f2_rhs[-1]) < 1.0e-20
    assert np.allclose(rhs.moment(0, 4), 0.0, atol=1e-14)


def test_stageB_existing_quadrupole_sources_hexadecapole():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.zeros(grid, STAGE_B_ELLS, n_species=1)
    state.data[0, 0, :] = fermi_dirac(grid.nodes)
    state.data[0, 1, :] = 1.0e-3 * np.exp(-grid.nodes / 3.0)
    rhs = AxisymmetricHierarchyState.from_flat(
        compute_hierarchy_rhs_typeI_stageB(state, Sigma_H=0.1),
        grid,
        STAGE_B_ELLS,
        n_species=1,
    )
    assert np.linalg.norm(rhs.moment(0, 4)) > 0.0
