import numpy as np
import jax.numpy as jnp

from rabbit.config.grids import MomentumGrid, MultipoleSpec
from rabbit.transport.state import HierarchyState
from rabbit.transport.projectors import extract_aniso_stress as extract_aniso_stress_numpy
from rabbit.transport.typeI_hierarchy import compute_hierarchy_rhs_typeI
from rabbit.jax.transport_ops_jax import (
    apply_flat_streaming_rhs,
    extract_aniso_stress_operator,
    extract_monopole_distributions_operator,
)


def test_flat_streaming_operator_matches_numpy_typeI_hierarchy():
    grid = MomentumGrid(N_q=8)
    multipole = MultipoleSpec()
    state = HierarchyState.from_isotropic(grid, multipole)
    flat = state.to_flat()

    dflat_jax = np.asarray(apply_flat_streaming_rhs(0.07, -0.01, jnp.asarray(flat), n_ell=2, n_species=6))
    dflat_np = compute_hierarchy_rhs_typeI(state, 0.07, -0.01)
    assert np.allclose(dflat_jax, dflat_np, rtol=0.0, atol=1e-15)


def test_transport_projector_matches_numpy_projector_for_typeI_state():
    grid = MomentumGrid(N_q=8)
    multipole = MultipoleSpec()
    state = HierarchyState.from_isotropic(grid, multipole)
    state.data[:, 1, :] = 0.03
    stress_np = extract_aniso_stress_numpy(state, grid)
    pi_plus_jax, pi_minus_jax = extract_aniso_stress_operator(
        jnp.asarray(state.to_flat()), grid.N_q, multipole.n_ell,
        jnp.asarray(grid.nodes), jnp.asarray(grid.weights), jnp.asarray(0.40520)
    )
    assert abs(float(pi_plus_jax) - stress_np.Pi_plus) < 1e-14
    assert abs(float(pi_minus_jax)) < 1e-14


def test_monopole_distribution_operator_returns_physical_distributions():
    grid = MomentumGrid(N_q=6)
    multipole = MultipoleSpec()
    state = HierarchyState.from_isotropic(grid, multipole)
    state.data[0, 0, :] = 0.1
    state.data[1, 0, :] = -0.05
    f_nue, f_nuebar = extract_monopole_distributions_operator(
        jnp.asarray(state.to_flat()), grid.N_q, multipole.n_ell, jnp.asarray(grid.nodes), species_pair=(0, 1)
    )
    f_eq = 1.0 / (np.exp(grid.nodes) + 1.0)
    assert np.allclose(np.asarray(f_nue), np.clip(f_eq * 1.1, 0.0, 1.0), atol=1e-14)
    assert np.allclose(np.asarray(f_nuebar), np.clip(f_eq * 0.95, 0.0, 1.0), atol=1e-14)
