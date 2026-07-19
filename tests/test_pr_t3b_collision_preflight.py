from __future__ import annotations

import numpy as np

from rabbit.jax.full_boltzmann_collision_preflight import (
    build_lifted_transport_collision_preflight,
    evaluate_collision_bank_and_3t_sources_from_state,
    evaluate_collision_moment_core,
    materialize_collision_bank_and_3t_source_augmented_jacobian,
    extract_bank_monopoles_from_transport_rays,
    materialize_collision_bank_and_3t_source_jacobian,
    materialize_collision_moment_core_augmented_jacobian,
    materialize_collision_moment_core_jacobian,
    stack_bank_monopoles_with_active_scalars,
    stack_bank_monopoles,
)


def _fd_equilibrium(q_nodes: np.ndarray) -> np.ndarray:
    return 1.0 / (np.exp(np.clip(q_nodes, -500.0, 500.0)) + 1.0)


def _materialize_lifted_factors(factors) -> np.ndarray:
    return (
        np.asarray(factors.base_jacobian, dtype=np.float64)
        + np.asarray(factors.left_factor, dtype=np.float64)
        @ np.asarray(factors.core_matrix, dtype=np.float64)
        @ np.asarray(factors.right_factor, dtype=np.float64)
    )


def test_extract_bank_monopoles_from_transport_rays_matches_weighted_average():
    f = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.2, 0.4], [0.6, 0.8]],
            [[0.4, 0.6], [0.8, 1.0]],
        ],
        dtype=np.float64,
    )
    jac = np.ones(2, dtype=np.float64)
    w0 = np.ones(2, dtype=np.float64)

    nue, nuebar, nux = extract_bank_monopoles_from_transport_rays(f, jac, w0)

    np.testing.assert_allclose(nue, np.array([0.2, 0.3]), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(nuebar, np.array([0.6, 0.7]), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(nux, np.array([0.5, 0.7]), rtol=0.0, atol=1e-14)


def test_collision_preflight_vanishes_at_equal_temperatures_and_equilibrium():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(6)
    f_eq = _fd_equilibrium(q_nodes)
    result = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.0,
        T_nu_e=2.0,
        T_nu_x=2.0,
        H_inv_sec=1.0e21,
    )

    assert np.max(np.abs(result.C_nue)) < 1e-14
    assert np.max(np.abs(result.C_nuebar)) < 1e-14
    assert np.max(np.abs(result.C_nux)) < 1e-14
    assert np.max(np.abs(result.target_plasma_energy_exchange)) < 1e-14
    assert np.max(np.abs(result.achieved_plasma_energy_exchange)) < 1e-14
    assert result.closure_mode == "source_only"
    assert result.closure_strength == 0.0


def test_collision_preflight_nue_coupling_exceeds_nux_for_hotter_plasma():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(6)
    f_eq = _fd_equilibrium(q_nodes)
    result = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.2,
        T_nu_e=1.8,
        T_nu_x=1.8,
        H_inv_sec=1.0e21,
    )

    assert result.gamma_over_H[0] > result.gamma_over_H[2]
    assert result.gamma_over_H[1] > result.gamma_over_H[2]
    assert np.linalg.norm(result.C_nue) > np.linalg.norm(result.C_nux)
    assert np.linalg.norm(result.C_nuebar) > np.linalg.norm(result.C_nux)
    assert np.all(result.achieved_plasma_energy_exchange < 0.0)


def test_collision_preflight_jacobian_is_finite_and_block_diagonal_by_species():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(6)
    f_eq = _fd_equilibrium(q_nodes)
    state = stack_bank_monopoles(
        f_eq * (1.0 + 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 - 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 + 0.02 * q_nodes / np.max(q_nodes)),
    )
    jac_source_only = materialize_collision_moment_core_jacobian(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
    )
    assert np.linalg.norm(jac_source_only) < 1e-10

    jac = materialize_collision_moment_core_jacobian(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="spectral_relaxation",
    )

    n_q = len(q_nodes)
    assert jac.shape == (3 * n_q + 3, 3 * n_q)
    assert np.isfinite(jac).all()

    rows_nue = np.concatenate([np.arange(0, n_q), np.array([3 * n_q])])
    rows_nuebar = np.concatenate([np.arange(n_q, 2 * n_q), np.array([3 * n_q + 1])])
    rows_nux = np.concatenate([np.arange(2 * n_q, 3 * n_q), np.array([3 * n_q + 2])])

    block_nue = jac[np.ix_(rows_nue, np.arange(0, n_q))]
    block_nuebar = jac[np.ix_(rows_nuebar, np.arange(n_q, 2 * n_q))]
    block_nux = jac[np.ix_(rows_nux, np.arange(2 * n_q, 3 * n_q))]
    assert np.linalg.norm(block_nue) > 0.0
    assert np.linalg.norm(block_nuebar) > 0.0
    assert np.linalg.norm(block_nux) > 0.0

    offdiag = jac.copy()
    offdiag[np.ix_(rows_nue, np.arange(0, n_q))] = 0.0
    offdiag[np.ix_(rows_nuebar, np.arange(n_q, 2 * n_q))] = 0.0
    offdiag[np.ix_(rows_nux, np.arange(2 * n_q, 3 * n_q))] = 0.0
    assert np.linalg.norm(offdiag) < 1e-8


def test_projected_physical_preflight_is_state_dependent_and_equilibrates():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(6)
    f_eq = _fd_equilibrium(q_nodes)
    eq_result = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.0,
        T_nu_e=2.0,
        T_nu_x=2.0,
        H_inv_sec=1.0e21,
        closure_mode="projected_physical",
    )
    assert np.max(np.abs(eq_result.C_nue)) < 1e-14
    assert np.max(np.abs(eq_result.C_nuebar)) < 1e-14
    assert np.max(np.abs(eq_result.C_nux)) < 1e-14

    state = stack_bank_monopoles(
        f_eq * (1.0 + 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 - 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 + 0.02 * q_nodes / np.max(q_nodes)),
    )
    jac = materialize_collision_moment_core_jacobian(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="projected_physical",
    )

    n_q = len(q_nodes)
    assert jac.shape == (3 * n_q + 3, 3 * n_q)
    assert np.isfinite(jac).all()
    assert np.linalg.norm(jac) > 0.0

    rows_nue = np.concatenate([np.arange(0, n_q), np.array([3 * n_q])])
    rows_nuebar = np.concatenate([np.arange(n_q, 2 * n_q), np.array([3 * n_q + 1])])
    rows_nux = np.concatenate([np.arange(2 * n_q, 3 * n_q), np.array([3 * n_q + 2])])

    offdiag = jac.copy()
    offdiag[np.ix_(rows_nue, np.arange(0, n_q))] = 0.0
    offdiag[np.ix_(rows_nuebar, np.arange(n_q, 2 * n_q))] = 0.0
    offdiag[np.ix_(rows_nux, np.arange(2 * n_q, 3 * n_q))] = 0.0
    assert np.linalg.norm(offdiag) < 1e-8


def test_direct_kernel_preflight_respects_detailed_balance_and_heats_hotter_plasma():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(20)
    f_eq = _fd_equilibrium(q_nodes)
    eq_result = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.0,
        T_nu_e=2.0,
        T_nu_x=2.0,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    assert np.max(np.abs(eq_result.C_nue)) < 1e-18
    assert np.max(np.abs(eq_result.C_nuebar)) < 1e-18
    assert np.max(np.abs(eq_result.C_nux)) < 1e-18

    hot_result = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.2,
        T_nu_e=1.8,
        T_nu_x=1.7,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    assert np.linalg.norm(hot_result.C_nue) > np.linalg.norm(hot_result.C_nux)
    assert np.linalg.norm(hot_result.C_nuebar) > np.linalg.norm(hot_result.C_nux)
    assert np.all(hot_result.achieved_plasma_energy_exchange < 0.0)


def test_direct_kernel_preflight_scales_like_inverse_hubble():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(16)
    f_eq = _fd_equilibrium(q_nodes)
    slow = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.2,
        T_nu_e=1.8,
        T_nu_x=1.7,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    fast = evaluate_collision_moment_core(
        f_nue=f_eq,
        f_nuebar=f_eq,
        f_nux=f_eq,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.2,
        T_nu_e=1.8,
        T_nu_x=1.7,
        H_inv_sec=2.0e21,
        closure_mode="direct_kernel",
    )
    np.testing.assert_allclose(fast.C_nue, 0.5 * slow.C_nue, rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(fast.C_nuebar, 0.5 * slow.C_nuebar, rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(fast.C_nux, 0.5 * slow.C_nux, rtol=2.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(
        fast.achieved_plasma_energy_exchange,
        0.5 * slow.achieved_plasma_energy_exchange,
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(fast.gamma_over_H, 0.5 * slow.gamma_over_H, rtol=2.0e-6, atol=1.0e-12)


def test_direct_kernel_combined_bank_and_3t_source_jacobian_is_finite():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(8)
    f_eq = np.clip(_fd_equilibrium(q_nodes), 1.0e-4, 1.0 - 1.0e-4)
    state = stack_bank_monopoles(
        f_eq * (1.0 + 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 - 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 + 0.02 * q_nodes / np.max(q_nodes)),
    )
    out = evaluate_collision_bank_and_3t_sources_from_state(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    jac = materialize_collision_bank_and_3t_source_jacobian(
        state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    assert out.shape == (3 * len(q_nodes) + 3,)
    assert jac.shape == (3 * len(q_nodes) + 3, 3 * len(q_nodes))
    assert np.isfinite(out).all()
    assert np.isfinite(jac).all()
    assert np.linalg.norm(jac) > 0.0


def test_direct_kernel_augmented_active_jacobian_is_finite_and_extends_bank_state_block():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(8)
    f_eq = np.clip(_fd_equilibrium(q_nodes), 1.0e-4, 1.0 - 1.0e-4)
    bank_state = stack_bank_monopoles(
        f_eq * (1.0 + 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 - 0.01 * q_nodes / np.max(q_nodes)),
        f_eq * (1.0 + 0.02 * q_nodes / np.max(q_nodes)),
    )
    augmented_state = stack_bank_monopoles_with_active_scalars(
        bank_state[: len(q_nodes)],
        bank_state[len(q_nodes) : 2 * len(q_nodes)],
        bank_state[2 * len(q_nodes) :],
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
    )

    jac_core = materialize_collision_moment_core_jacobian(
        bank_state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="direct_kernel",
    )
    jac_core_aug = materialize_collision_moment_core_augmented_jacobian(
        augmented_state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        closure_mode="direct_kernel",
    )
    jac_aug = materialize_collision_bank_and_3t_source_augmented_jacobian(
        augmented_state,
        q_nodes=q_nodes,
        q_weights=q_weights,
        closure_mode="direct_kernel",
    )

    n_q = len(q_nodes)
    assert jac_core_aug.shape == (3 * n_q + 3, 3 * n_q + 4)
    assert jac_aug.shape == (3 * n_q + 3, 3 * n_q + 4)
    assert np.isfinite(jac_core_aug).all()
    assert np.isfinite(jac_aug).all()
    np.testing.assert_allclose(jac_core_aug[:, : 3 * n_q], jac_core, rtol=3.0e-5, atol=3.0e-7)
    active_block_core = jac_core_aug[:, 3 * n_q :]
    active_block_aug = jac_aug[:, 3 * n_q :]
    assert np.linalg.norm(active_block_core[:, 0]) > 0.0  # T_gamma
    assert np.linalg.norm(active_block_core[:, 1]) > 0.0  # T_nu_e
    assert np.linalg.norm(active_block_core[:, 2]) > 0.0  # T_nu_x
    assert np.linalg.norm(active_block_core[:, 3]) > 0.0  # H_inv_sec
    assert np.linalg.norm(active_block_aug) > 0.0


def test_lifted_transport_collision_rhs_is_uniform_per_bank_and_heavy_flavor_shared():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(4)
    f_eq = _fd_equilibrium(q_nodes)
    f = np.tile(f_eq[None, None, :], (4, 3, 1))
    f[0] *= 1.01
    f[1] *= 0.99
    f[2] *= 1.02
    f[3] *= 0.98

    lifted = build_lifted_transport_collision_preflight(
        f_species_rays=f,
        jacobian_weights=np.ones(3, dtype=np.float64),
        w0=np.ones(3, dtype=np.float64),
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="spectral_relaxation",
    )

    assert lifted.transport_rhs.shape == f.shape
    np.testing.assert_allclose(lifted.transport_rhs[0, 0], lifted.transport_rhs[0, 1], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(lifted.transport_rhs[1, 0], lifted.transport_rhs[1, 2], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(lifted.transport_rhs[2], lifted.transport_rhs[3], rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        lifted.transport_rhs[0, 0],
        lifted.bank_result.C_nue,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lifted.transport_rhs[1, 1],
        lifted.bank_result.C_nuebar,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lifted.transport_rhs[2, 2],
        lifted.bank_result.C_nux,
        rtol=0.0,
        atol=1e-14,
    )


def test_lifted_transport_collision_factorization_matches_dense_fd():
    q_nodes, q_weights = np.polynomial.laguerre.laggauss(4)
    f_eq = np.clip(_fd_equilibrium(q_nodes), 1.0e-4, 1.0 - 1.0e-4)
    f = np.tile(f_eq[None, None, :], (4, 2, 1))
    f[0, 0] *= 1.01
    f[0, 1] *= 0.99
    f[1, 0] *= 1.02
    f[1, 1] *= 0.98
    f[2, 0] *= 1.03
    f[3, 1] *= 0.97
    x = f.reshape(-1)

    def lifted_rhs(flat: np.ndarray) -> np.ndarray:
        built = build_lifted_transport_collision_preflight(
            f_species_rays=flat.reshape(4, 2, 4),
            jacobian_weights=np.array([0.9, 1.1], dtype=np.float64),
            w0=np.array([1.2, 0.8], dtype=np.float64),
            q_nodes=q_nodes,
            q_weights=q_weights,
            T_gamma=2.1,
            T_nu_e=1.7,
            T_nu_x=1.6,
            H_inv_sec=1.0e21,
            closure_mode="spectral_relaxation",
        )
        return built.state_rhs

    lifted = build_lifted_transport_collision_preflight(
        f_species_rays=f,
        jacobian_weights=np.array([0.9, 1.1], dtype=np.float64),
        w0=np.array([1.2, 0.8], dtype=np.float64),
        q_nodes=q_nodes,
        q_weights=q_weights,
        T_gamma=2.1,
        T_nu_e=1.7,
        T_nu_x=1.6,
        H_inv_sec=1.0e21,
        closure_mode="spectral_relaxation",
    )
    fact = _materialize_lifted_factors(lifted.factors)
    assert np.linalg.norm(fact) > 0.0

    fd = np.zeros_like(fact)
    for i in range(x.size):
        step = 1.0e-6 * max(1.0, abs(x[i]))
        x_hi = x.copy()
        x_lo = x.copy()
        x_hi[i] += step
        x_lo[i] -= step
        fd[:, i] = (lifted_rhs(x_hi) - lifted_rhs(x_lo)) / (2.0 * step)

    np.testing.assert_allclose(fact, fd, rtol=2.0e-6, atol=2.0e-6)
