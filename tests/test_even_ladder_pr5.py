from __future__ import annotations

import numpy as np

from rabbit.config.grids import MomentumGrid
from rabbit.transport.even_ladder_analysis import compare_even_ladder_to_reference
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState, fermi_dirac
from rabbit.transport.typeI_even_ladder_hierarchy import (
    active_even_ells,
    compute_hierarchy_rhs_typeI_even_ladder,
)
from rabbit.transport.typeI_stageB_hierarchy import STAGE_B_ELLS, compute_hierarchy_rhs_typeI_stageB



def test_even_ladder_flrw_fd_invariant_rhs_zero_for_2_4_6_8():
    grid = MomentumGrid(N_q=20)
    for lmax in (2, 4, 6, 8):
        active = active_even_ells(lmax)
        state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active, n_species=1)
        rhs = compute_hierarchy_rhs_typeI_even_ladder(state, Sigma_H=0.0)
        assert np.allclose(rhs, 0.0, atol=0.0, rtol=0.0)



def test_even_ladder_lmax4_matches_stageB_exactly():
    grid = MomentumGrid(N_q=20)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, STAGE_B_ELLS, n_species=1)
    rhs_generic = compute_hierarchy_rhs_typeI_even_ladder(state, Sigma_H=0.1)
    rhs_stageb = compute_hierarchy_rhs_typeI_stageB(state, Sigma_H=0.1)
    assert np.allclose(rhs_generic, rhs_stageb, atol=1e-14, rtol=1e-14)



def test_even_ladder_existing_f4_sources_f6_when_lmax6():
    grid = MomentumGrid(N_q=20)
    active = active_even_ells(6)
    state = AxisymmetricHierarchyState.zeros(grid, active, n_species=1)
    q = grid.nodes
    state.data[0, 0, :] = fermi_dirac(q)
    state.data[0, state.ell_index(2), :] = 1.0e-3 * np.exp(-q / 3.0)
    state.data[0, state.ell_index(4), :] = 5.0e-4 * q * np.exp(-q / 2.5)
    rhs = AxisymmetricHierarchyState.from_flat(
        compute_hierarchy_rhs_typeI_even_ladder(state, Sigma_H=0.1),
        grid,
        active,
        n_species=1,
    )
    assert np.linalg.norm(rhs.moment(0, 6)) > 0.0



def test_even_ladder_existing_f6_sources_f8_when_lmax8():
    grid = MomentumGrid(N_q=20)
    active = active_even_ells(8)
    state = AxisymmetricHierarchyState.zeros(grid, active, n_species=1)
    q = grid.nodes
    state.data[0, 0, :] = fermi_dirac(q)
    state.data[0, state.ell_index(2), :] = 1.0e-3 * np.exp(-q / 3.0)
    state.data[0, state.ell_index(4), :] = 5.0e-4 * q * np.exp(-q / 2.5)
    state.data[0, state.ell_index(6), :] = 2.0e-4 * q**2 * np.exp(-q / 2.0)
    rhs = AxisymmetricHierarchyState.from_flat(
        compute_hierarchy_rhs_typeI_even_ladder(state, Sigma_H=0.1),
        grid,
        active,
        n_species=1,
    )
    assert np.linalg.norm(rhs.moment(0, 8)) > 0.0



def test_compare_even_ladder_to_reference_zero_for_identical_state():
    grid = MomentumGrid(N_q=20)
    active = active_even_ells(8)
    state = AxisymmetricHierarchyState.from_fd_equilibrium(grid, active, n_species=1)
    cmp = compare_even_ladder_to_reference(state, state)
    assert cmp['deltaF0_diff_to_ref_L2'] == 0.0
    assert cmp['F2_diff_to_ref_L2'] == 0.0
