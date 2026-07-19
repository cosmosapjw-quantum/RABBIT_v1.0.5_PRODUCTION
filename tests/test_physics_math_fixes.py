from __future__ import annotations

import numpy as np
import pytest

from rabbit.geometry.general_classA import (
    compute_q,
    gauss_curvature_K,
    CS_SIGMA_PLUS,
    CS_Q,
    CS_N1_SQ,
    CS_K,
    CS_OMEGA,
)
from rabbit.network.abundances_standard import mass_conservation_residual
from rabbit.inference.forward_likelihood import canonical_forward_solver
from rabbit.config.grids import MomentumGrid
from rabbit.transport.state import HierarchyState
from rabbit.config.grids import MultipoleSpec
from rabbit.transport.projectors import extract_aniso_stress as extract_aniso_stress_numpy

pytest.importorskip("jax", reason="JAX required for JAX projector checks")
import jax.numpy as jnp
from rabbit.jax.rhs_typeI import extract_aniso_stress as extract_aniso_stress_jax
from rabbit.weak.teff_correction import (
    teff_corrected_monopole_exact,
    teff_tangency_diagnostic,
)


def test_classA_q_uses_minus_curvature():
    Sp, Sm, N1, N2, N3 = 0.2, 0.0, np.sqrt(0.75), 0.0, 0.0
    Sigma_sq = Sp**2 + Sm**2
    K = gauss_curvature_K(N1, N2, N3)
    expected = 1.0 + Sigma_sq - K
    got = compute_q(Sp, Sm, N1, N2, N3)
    assert np.isclose(got, expected, rtol=0, atol=1e-14)


def test_collins_stewart_radiation_constants():
    assert np.isclose(CS_SIGMA_PLUS, 0.25)
    assert np.isclose(CS_Q, 1.0)
    assert np.isclose(CS_N1_SQ, 0.75)
    assert np.isclose(CS_K, 1.0 / 16.0)
    assert np.isclose(CS_OMEGA, 7.0 / 8.0)


def test_mass_fraction_residual_uses_sum_X():
    X = np.array([0.2, 0.3, 0.5] + [0.0] * 6)
    assert np.isclose(mass_conservation_residual(X), 0.0)


def test_jax_projector_matches_numpy_standard_laguerre():
    grid = MomentumGrid(N_q=12)
    state = HierarchyState.from_isotropic(grid, MultipoleSpec())
    # Identical species, q-dependent quadrupole profile.
    profile = 0.03 + 0.01 * np.tanh(grid.nodes - 2.0)
    for s in range(state.n_species):
        state.data[s, 1, :] = profile

    stress_np = extract_aniso_stress_numpy(state, grid)
    psi_flat = jnp.array(state.data.reshape(-1))
    pi_plus_jax, _ = extract_aniso_stress_jax(
        psi_flat,
        grid.N_q,
        state.n_ell,
        jnp.array(grid.nodes),
        jnp.array(grid.weights),
        jnp.array(0.40520),
    )
    assert np.isclose(float(pi_plus_jax), stress_np.Pi_plus, rtol=0, atol=1e-12)


def test_teff_closure_reconstruction_preserves_generic_baseline_at_zero_quadrupole():
    q = np.linspace(0.05, 12.0, 50)
    f0 = 0.85 / (np.exp(q) + 1.0)
    out = teff_corrected_monopole_exact(f0, q, 0.0)
    assert np.allclose(out, f0)


def test_teff_tangency_uses_consistent_quadrature_weights():
    grid = MomentumGrid(N_q=16)
    f0 = 1.0 / (np.exp(grid.nodes) + 1.0)
    D = teff_tangency_diagnostic(f0, grid.nodes, 0.02, q_weights=grid.weights)
    assert np.isfinite(D)
    assert D >= 0.0


def test_canonical_forward_auto_falls_back_to_scipy_reference_outside_jax_default_scope():
    pred = canonical_forward_solver(Sigma_H=0.0, N_q=6, backend="auto")
    assert pred.success
    assert pred.metadata["backend"] == "scipy_typeI_reference"


def test_canonical_forward_jax_endpoint_is_hard_retired():
    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, N_q=20, backend="jax", correction_level=3)
