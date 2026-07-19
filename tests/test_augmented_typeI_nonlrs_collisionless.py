from __future__ import annotations

import numpy as np
import pytest

from rabbit.transport.augmented_typeI_nonlrs_collisionless import (
    AugmentedNonLRSSourceCollisionlessConfig,
    NonLRSS2Grid,
    augmented_nonlrs_source_collisionless_rhs,
    build_non_lrs_s2_grid,
    non_lrs_quadrupole_source_rhs,
    run_augmented_nonlrs_source_collisionless_solve,
)


def test_non_lrs_s2_grid_has_expected_shapes_and_area() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)

    assert grid.mu.shape == (48,)
    assert grid.phi.shape == (48,)
    assert grid.angular_weights.shape == (48,)
    assert grid.basis_matrix.shape == (3, 48)
    assert float(np.sum(grid.angular_weights)) == pytest.approx(4.0 * np.pi)


# NOTE (Phase 3 deflation): the non-LRS↔LRS linearized-quadrupole parity test against
# augmented_fd_boltzmann.augmented_fd_transport_rhs was removed with the AP65 audit track.
# The non-LRS source's plus/minus-mode behavior is still covered by the tests below.


def test_non_lrs_source_populates_minus_mode_from_sigma_minus() -> None:
    grid = build_non_lrs_s2_grid(N_mu=14, N_phi=20)
    q = np.array([0.5, 1.5])
    A = np.zeros((1, 3, q.size))

    rhs = non_lrs_quadrupole_source_rhs(0.0, -0.03, A, q, grid=grid)

    assert np.allclose(rhs.dA_modes[0, 0, :], 0.0, atol=1.0e-14)
    assert np.allclose(rhs.dA_modes[0, 1, :], 0.0, atol=1.0e-14)
    assert np.allclose(rhs.dA_modes[0, 2, :], 0.06 * q, atol=1.0e-13)
    assert rhs.closure == "quadrupole_source_projection_only_v1"


def test_non_lrs_source_collisionless_rhs_matches_source_projection_and_geometry() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)
    q = np.array([0.5, 1.4, 2.3])
    q_weights = np.array([0.7, 1.1, 1.5])
    A = np.zeros((2, 3, q.size))
    sigma_plus = 0.025
    sigma_minus = -0.012

    rhs = augmented_nonlrs_source_collisionless_rhs(
        sigma_plus,
        sigma_minus,
        A,
        q,
        q_weights,
        grid=grid,
        config=AugmentedNonLRSSourceCollisionlessConfig(N_mu=6, N_phi=8),
    )
    source = non_lrs_quadrupole_source_rhs(
        sigma_plus,
        sigma_minus,
        A,
        q,
        grid=grid,
    )

    sigma_sq = sigma_plus**2 + sigma_minus**2
    assert np.allclose(rhs.dA_modes, source.dA_modes)
    assert rhs.Pi_plus == pytest.approx(0.0, abs=1.0e-14)
    assert rhs.Pi_minus == pytest.approx(0.0, abs=1.0e-14)
    assert rhs.dSigma_plus == pytest.approx(-((1.0 - sigma_sq) * sigma_plus))
    assert rhs.dSigma_minus == pytest.approx(-((1.0 - sigma_sq) * sigma_minus))
    assert rhs.closure == "non_lrs_quadrupole_source_coevolution_v1"


def test_non_lrs_source_collisionless_rhs_uses_live_minus_stress_feedback() -> None:
    grid = build_non_lrs_s2_grid(N_mu=8, N_phi=10)
    q = np.array([0.4, 1.1, 2.0])
    q_weights = np.array([0.5, 1.0, 1.4])
    A = np.zeros((1, 3, q.size))
    A[:, 2, :] = 0.05
    config = AugmentedNonLRSSourceCollisionlessConfig(
        N_mu=8,
        N_phi=10,
        f_nu=0.4,
        feedback_factor=6.0,
    )

    rhs = augmented_nonlrs_source_collisionless_rhs(
        0.0,
        0.0,
        A,
        q,
        q_weights,
        grid=grid,
        config=config,
    )

    assert rhs.moments.pi_minus_tilde is not None
    assert abs(rhs.Pi_minus) > 1.0e-5
    assert abs(rhs.Pi_minus) > abs(rhs.Pi_plus)
    assert rhs.dSigma_plus == pytest.approx(-rhs.Pi_plus)
    assert rhs.dSigma_minus == pytest.approx(-rhs.Pi_minus)


def test_run_non_lrs_source_collisionless_solve_evolves_plus_and_minus_modes() -> None:
    config = AugmentedNonLRSSourceCollisionlessConfig(N_mu=6, N_phi=8)

    result = run_augmented_nonlrs_source_collisionless_solve(
        0.018,
        -0.011,
        N_span=(0.0, 0.02),
        N_q=3,
        n_species=2,
        config=config,
        rtol=1.0e-7,
        atol=1.0e-9,
    )

    assert result.success
    assert result.Sigma_plus.shape == result.Sigma_minus.shape == result.N.shape
    assert result.A_modes_final.shape == (2, 3, 3)
    assert np.max(np.abs(result.A_modes_final[:, 1, :])) > 0.0
    assert np.max(np.abs(result.A_modes_final[:, 2, :])) > 0.0
    assert np.isfinite(result.Pi_plus_final)
    assert np.isfinite(result.Pi_minus_final)
    assert result.closure == "non_lrs_quadrupole_source_coevolution_v1"


def test_non_lrs_source_rejects_bad_mode_shape() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)
    q = np.array([0.5, 1.5])

    with pytest.raises(ValueError, match="three non-LRS modes"):
        non_lrs_quadrupole_source_rhs(0.0, 0.0, np.zeros((1, 2, q.size)), q, grid=grid)


def test_non_lrs_source_rejects_empty_species_or_q_grid() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)

    with pytest.raises(ValueError, match="at least one species"):
        non_lrs_quadrupole_source_rhs(
            0.0,
            0.0,
            np.zeros((0, 3, 2)),
            np.array([0.5, 1.5]),
            grid=grid,
        )
    with pytest.raises(ValueError, match="non-empty 1D"):
        non_lrs_quadrupole_source_rhs(
            0.0,
            0.0,
            np.zeros((1, 3, 0)),
            np.array([]),
            grid=grid,
        )


def test_non_lrs_s2_grid_rejects_fractional_resolution() -> None:
    with pytest.raises(TypeError, match="exact integer"):
        build_non_lrs_s2_grid(N_mu=6.5, N_phi=8)


def test_non_lrs_s2_grid_rejects_degenerate_resolution() -> None:
    with pytest.raises(ValueError, match="N_mu must be at least 3"):
        build_non_lrs_s2_grid(N_mu=2, N_phi=8)
    with pytest.raises(ValueError, match="N_phi must be at least 5"):
        build_non_lrs_s2_grid(N_mu=6, N_phi=4)


def test_non_lrs_source_rejects_non_increasing_q_nodes() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)
    q = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="strictly increasing"):
        non_lrs_quadrupole_source_rhs(0.0, 0.0, np.zeros((1, 3, q.size)), q, grid=grid)


def test_non_lrs_s2_grid_rejects_bad_manual_basis_shape() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)

    with pytest.raises(ValueError, match="three non-LRS modes"):
        NonLRSS2Grid(
            mu=grid.mu,
            phi=grid.phi,
            angular_weights=grid.angular_weights,
            W_plus=grid.W_plus,
            W_minus=grid.W_minus,
            basis_matrix=grid.basis_matrix[:2, :],
        )


def test_non_lrs_s2_grid_rejects_mismatched_manual_basis() -> None:
    grid = build_non_lrs_s2_grid(N_mu=6, N_phi=8)
    basis = grid.basis_matrix.copy()
    basis[[1, 2], :] = basis[[2, 1], :]

    with pytest.raises(ValueError, match="must equal"):
        NonLRSS2Grid(
            mu=grid.mu,
            phi=grid.phi,
            angular_weights=grid.angular_weights,
            W_plus=grid.W_plus,
            W_minus=grid.W_minus,
            basis_matrix=basis,
        )
