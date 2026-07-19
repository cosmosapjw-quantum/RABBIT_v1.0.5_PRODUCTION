from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.laguerre import laggauss

from rabbit.jax.nonlinear_transport import (
    build_mu_diff_matrix,
    build_q_diff_matrix,
    fd_equilibrium,
    nonlinear_boltzmann_rhs,
)
from rabbit.transport.augmented_nonlrs_transport import (
    NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT,
    augmented_nonlrs_nonlinear_mode_rhs,
    augmented_nonlrs_nonlinear_nodal_rhs,
    augmented_nonlrs_nonlinear_transport_coevolution_rhs,
    build_nonlrs_nonlinear_transport_grid,
    run_augmented_nonlrs_nonlinear_transport_solve,
)
from rabbit.transport.augmented_pstf_distribution import (
    modes_to_nodal,
    project_nodal_to_modes,
    reconstruct_distribution,
)
from rabbit.transport import augmented_nonlrs_transport as nonlrs_transport


def _q_nodes(n: int = 5) -> np.ndarray:
    q, _w = laggauss(n)
    return np.asarray(q, dtype=float)


def _fd_lrs_distribution(q: np.ndarray, n_mu: int, n_phi: int) -> np.ndarray:
    f_mu = np.tile(fd_equilibrium(q)[:, None], (1, n_mu))
    return np.repeat(f_mu[:, :, None], n_phi, axis=2).reshape(q.size, n_mu * n_phi)


def _direct_logit_transport_expected(
    A_modes: np.ndarray,
    q: np.ndarray,
    *,
    Sigma_plus: float,
    Sigma_minus: float,
    grid,
) -> tuple[np.ndarray, np.ndarray]:
    nodal_A = modes_to_nodal(A_modes, grid.s2_grid.basis_matrix)
    A_grid = nodal_A.reshape((A_modes.shape[0], q.size, grid.N_mu, grid.N_phi))
    dA_dq = np.einsum("ij,sja->sia", build_q_diff_matrix(q), nodal_A)
    dA_dmu = np.einsum("ij,sqjp->sqip", grid.D_mu, A_grid).reshape(nodal_A.shape)
    dA_dphi = np.einsum("pj,sqij->sqip", grid.D_phi, A_grid).reshape(nodal_A.shape)
    energy_shift = (
        float(Sigma_plus) * grid.energy_shift_plus
        + float(Sigma_minus) * grid.energy_shift_minus
    )
    mu_drift = (
        float(Sigma_plus) * grid.mu_drift_plus
        + float(Sigma_minus) * grid.mu_drift_minus
    )
    phi_drift = float(Sigma_minus) * grid.phi_drift_minus
    nodal_dA = (
        -energy_shift[None, None, :] * q[None, :, None] * (1.0 + dA_dq)
        - mu_drift[None, None, :] * dA_dmu
        - phi_drift[None, None, :] * dA_dphi
    )
    f = reconstruct_distribution(A_modes, q, grid.s2_grid.basis_matrix)
    df_dN = -f * (1.0 - f) * nodal_dA
    projected = project_nodal_to_modes(
        nodal_dA,
        grid.s2_grid.basis_matrix,
        grid.s2_grid.angular_weights,
        projection_matrix=grid.s2_grid.projection_matrix,
    )
    return df_dN, projected


def test_nonlrs_nonlinear_transport_grid_precomputes_drift_factors() -> None:
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    mu = grid.mu
    phi = grid.phi
    one_minus_mu2 = 1.0 - mu * mu

    np.testing.assert_allclose(grid.energy_shift_plus, 2.0 * grid.W_plus)
    np.testing.assert_allclose(grid.energy_shift_minus, 2.0 * grid.W_minus)
    np.testing.assert_allclose(grid.mu_drift_plus, 3.0 * mu * one_minus_mu2)
    np.testing.assert_allclose(
        grid.mu_drift_minus,
        -np.sqrt(3.0) * mu * one_minus_mu2 * np.cos(2.0 * phi),
    )
    np.testing.assert_allclose(
        grid.phi_drift_minus,
        -np.sqrt(3.0) * np.sin(2.0 * phi),
    )


def test_nonlrs_nonlinear_transport_grid_reuses_cached_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []
    real_builder = nonlrs_transport.build_non_lrs_s2_grid
    if hasattr(nonlrs_transport, "_NONLINEAR_TRANSPORT_GRID_CACHE"):
        nonlrs_transport._NONLINEAR_TRANSPORT_GRID_CACHE.clear()

    def _counted_builder(*, N_mu: int, N_phi: int):
        calls.append((int(N_mu), int(N_phi)))
        return real_builder(N_mu=N_mu, N_phi=N_phi)

    monkeypatch.setattr(nonlrs_transport, "build_non_lrs_s2_grid", _counted_builder)

    first = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    second = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    third = build_nonlrs_nonlinear_transport_grid(N_mu=7, N_phi=9)

    assert first is second
    assert third is not first
    assert calls == [(5, 7), (7, 9)]


def test_nonlrs_nonlinear_transport_reuses_q_diff_matrix_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)
    calls: list[np.ndarray] = []
    real_build = nonlrs_transport.build_q_diff_matrix
    if hasattr(nonlrs_transport, "_Q_DIFF_MATRIX_CACHE"):
        nonlrs_transport._Q_DIFF_MATRIX_CACHE.clear()

    def _counted_build(q_nodes: np.ndarray) -> np.ndarray:
        calls.append(np.asarray(q_nodes, dtype=float).copy())
        return real_build(q_nodes)

    monkeypatch.setattr(nonlrs_transport, "build_q_diff_matrix", _counted_build)

    for q_input in (q, q.copy()):
        augmented_nonlrs_nonlinear_nodal_rhs(
            f[None, :, :],
            q_input,
            Sigma_plus=0.02,
            Sigma_minus=-0.01,
            grid=grid,
        )

    assert len(calls) == 1


def test_nonlrs_nonlinear_transport_reduces_to_lrs_rhs_when_sigma_minus_zero() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)

    report = augmented_nonlrs_nonlinear_nodal_rhs(
        f[None, :, :],
        q,
        Sigma_plus=0.02,
        Sigma_minus=0.0,
        grid=grid,
    )

    mu = grid.mu.reshape(grid.N_mu, grid.N_phi)[:, 0]
    expected = nonlinear_boltzmann_rhs(
        f.reshape(q.size, grid.N_mu, grid.N_phi)[:, :, 0],
        0.02,
        q,
        mu,
        build_q_diff_matrix(q),
        build_mu_diff_matrix(mu),
    )
    actual = report.df_dN_nodal[0].reshape(q.size, grid.N_mu, grid.N_phi)
    for phi_idx in range(grid.N_phi):
        np.testing.assert_allclose(actual[:, :, phi_idx], expected, rtol=1.0e-12, atol=1.0e-12)
    assert report.dA_modes[0, 2] == pytest.approx(np.zeros(q.size), abs=1.0e-12)
    assert report.transport_scope_contract == NONLRS_S2_NONLINEAR_TRANSPORT_CONTRACT


def test_nonlrs_nonlinear_transport_projects_logit_rhs() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    A_modes = np.zeros((1, 3, q.size), dtype=float)
    A_modes[:, 0, :] = 1.0e-3 * np.linspace(-1.0, 1.0, q.size)
    A_modes[:, 1, :] = 8.0e-4 * q
    A_modes[:, 2, :] = -6.0e-4 * q[::-1]

    report = augmented_nonlrs_nonlinear_mode_rhs(
        Sigma_plus=0.02,
        Sigma_minus=-0.012,
        A_modes=A_modes,
        q_nodes=q,
        grid=grid,
    )
    expected_df, expected_dA = _direct_logit_transport_expected(
        A_modes,
        q,
        Sigma_plus=0.02,
        Sigma_minus=-0.012,
        grid=grid,
    )

    np.testing.assert_allclose(
        report.df_dN_nodal,
        expected_df,
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        report.dA_modes,
        expected_dA,
        rtol=2.0e-11,
        atol=2.0e-12,
    )


def test_nonlrs_nonlinear_transport_is_flrw_quiet() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)

    report = augmented_nonlrs_nonlinear_nodal_rhs(
        f[None, :, :],
        q,
        Sigma_plus=0.0,
        Sigma_minus=0.0,
        grid=grid,
    )

    assert np.max(np.abs(report.df_dN_nodal)) == pytest.approx(0.0, abs=1.0e-14)
    assert np.max(np.abs(report.dA_modes)) == pytest.approx(0.0, abs=1.0e-14)


def test_nonlrs_nonlinear_transport_minus_mode_changes_sign_with_sigma_minus() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)

    plus = augmented_nonlrs_nonlinear_nodal_rhs(
        f[None, :, :],
        q,
        Sigma_plus=0.0,
        Sigma_minus=0.03,
        grid=grid,
    )
    minus = augmented_nonlrs_nonlinear_nodal_rhs(
        f[None, :, :],
        q,
        Sigma_plus=0.0,
        Sigma_minus=-0.03,
        grid=grid,
    )

    assert np.max(np.abs(plus.dA_modes[0, 2])) > 0.0
    np.testing.assert_allclose(plus.dA_modes[0, 2], -minus.dA_modes[0, 2], rtol=1.0e-12, atol=1.0e-12)
    assert plus.dA_modes[0, 1] == pytest.approx(np.zeros(q.size), abs=1.0e-12)


def test_nonlrs_nonlinear_mode_rhs_reconstructs_and_projects_modes() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    A_modes = np.zeros((2, 3, q.size), dtype=float)
    A_modes[:, 0, :] = fd_equilibrium(q)

    report = augmented_nonlrs_nonlinear_mode_rhs(
        Sigma_plus=0.01,
        Sigma_minus=-0.02,
        A_modes=A_modes,
        q_nodes=q,
        grid=grid,
    )

    expected_df, expected_dA = _direct_logit_transport_expected(
        A_modes,
        q,
        Sigma_plus=0.01,
        Sigma_minus=-0.02,
        grid=grid,
    )
    np.testing.assert_allclose(report.df_dN_nodal, expected_df)
    np.testing.assert_allclose(report.dA_modes, expected_dA)


def test_nonlrs_nonlinear_transport_coevolution_reuses_single_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rabbit.transport import augmented_nonlrs_transport as transport_module

    q = _q_nodes()
    q_weights = q**3
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    A_modes = np.zeros((2, 3, q.size), dtype=float)
    A_modes[:, 0, :] = fd_equilibrium(q)

    original_reconstruct = transport_module.reconstruct_distribution
    call_count = 0

    def counting_reconstruct(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_reconstruct(*args, **kwargs)

    monkeypatch.setattr(transport_module, "reconstruct_distribution", counting_reconstruct)

    report = augmented_nonlrs_nonlinear_transport_coevolution_rhs(
        0.01,
        -0.02,
        A_modes,
        q,
        q_weights,
        grid=grid,
    )

    assert call_count == 1
    assert report.dA_modes.shape == A_modes.shape


def test_nonlrs_coevolution_shear_stress_feedback_damps_with_minus_pi() -> None:
    """BD622-R2 / audit F-020 (SL-R seal, audit/reference/shear_pi_adjudication.md).

    The augmented ``<f W_+>/<f>`` Pi object carries the polarity Sigma_+ > 0 => Pi_+ > 0,
    so the Site-3 coevolution shear RHS must couple the stress with ``-Pi`` (H-theorem
    damping); the pre-fix ``+Pi`` anti-damps (seal integration: shear grows ~1600x,
    diverging to the Sigma ~ 1 attractor). Mirrors the seal's sign-isolating
    discriminator: build the quadrupole dynamically at frozen Sigma_+ > 0, then check
    the shear increment opposes the shear, and integrate to confirm damping.
    """
    q = _q_nodes()
    q_weights = np.asarray(laggauss(q.size)[1], dtype=float) * np.exp(q) * q**3
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=8, N_phi=8)
    sigma_plus = 0.05

    # Probe state: quadrupole generated by the code's own transport source at frozen
    # Sigma_+ > 0 (explicit Euler from the FLRW state), as in the seal's derivation.
    A = np.zeros((1, 3, q.size))
    A[:, 0, :] = fd_equilibrium(q)
    for _ in range(50):
        rhs = augmented_nonlrs_nonlinear_mode_rhs(
            Sigma_plus=sigma_plus, Sigma_minus=0.0, A_modes=A, q_nodes=q, grid=grid
        )
        A = A + 1.0e-2 * rhs.dA_modes

    report = augmented_nonlrs_nonlinear_transport_coevolution_rhs(
        sigma_plus, 0.0, A, q, q_weights, grid=grid
    )
    free_decay = -(1.0 - sigma_plus**2) * sigma_plus
    # Polarity precondition (the seal's K = +3.89 > 0): Sigma_+ > 0 => Pi_+ > 0.
    assert report.Pi_plus > 0.0
    # Corrected assembly: the stress feedback enters with -Pi ...
    assert report.dSigma_plus == pytest.approx(free_decay - report.Pi_plus, rel=1.0e-12)
    # ... and therefore strengthens the damping (the pre-fix +Pi gives
    # dSigma_plus = free_decay + Pi_plus > 0 here: anti-damping).
    assert report.dSigma_plus < free_decay < 0.0

    # Seal discriminator, integrated: over N in [0, 6] the shear DAMPS (seal ratio
    # ~0.034 for -Pi vs ~1600x growth for +Pi; pre-fix the +Pi runaway drives Sigma
    # non-finite and this solve raises).
    result = run_augmented_nonlrs_nonlinear_transport_solve(
        Sigma_plus0=sigma_plus,
        Sigma_minus0=0.0,
        N_span=(0.0, 6.0),
        N_q=4,
        n_species=1,
        N_mu=8,
        N_phi=8,
        method="LSODA",
        rtol=1.0e-7,
        atol=1.0e-10,
    )
    assert result.success
    assert abs(result.Sigma_plus[-1]) < 0.2 * sigma_plus
    assert np.max(np.abs(result.Sigma_plus)) <= 3.0 * sigma_plus


def test_nonlrs_nonlinear_nodal_rhs_uses_grid_projection_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rabbit.transport import augmented_pstf_distribution as pstf_distribution

    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)

    def fail_dynamic_gram(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("nonlinear transport should reuse the grid projection matrix")

    monkeypatch.setattr(pstf_distribution, "gram_matrix_from_basis", fail_dynamic_gram)

    report = augmented_nonlrs_nonlinear_nodal_rhs(
        f[None, :, :],
        q,
        Sigma_plus=0.01,
        Sigma_minus=-0.02,
        grid=grid,
    )

    assert report.dA_modes.shape == (1, 3, q.size)


def test_nonlrs_nonlinear_transport_rejects_invalid_inputs() -> None:
    q = _q_nodes()
    grid = build_nonlrs_nonlinear_transport_grid(N_mu=5, N_phi=7)
    f = _fd_lrs_distribution(q, grid.N_mu, grid.N_phi)

    with pytest.raises(ValueError, match="N_mu must be an exact integer"):
        build_nonlrs_nonlinear_transport_grid(N_mu=3.2, N_phi=7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="q_nodes must be strictly increasing"):
        augmented_nonlrs_nonlinear_nodal_rhs(f[None], q[::-1], Sigma_plus=0.0, Sigma_minus=0.0, grid=grid)
    with pytest.raises(ValueError, match="f_species must have shape"):
        augmented_nonlrs_nonlinear_nodal_rhs(f, q, Sigma_plus=0.0, Sigma_minus=0.0, grid=grid)
    with pytest.raises(ValueError, match="A_modes must have shape"):
        augmented_nonlrs_nonlinear_mode_rhs(
            Sigma_plus=0.0,
            Sigma_minus=0.0,
            A_modes=np.zeros((3, q.size)),
            q_nodes=q,
            grid=grid,
        )


def test_nonlrs_nonlinear_transport_solve_runs_finite_short_span() -> None:
    result = run_augmented_nonlrs_nonlinear_transport_solve(
        Sigma_plus0=0.02,
        Sigma_minus0=-0.01,
        N_span=(0.0, 1.0e-3),
        N_q=4,
        n_species=2,
        N_mu=5,
        N_phi=7,
        method="LSODA",
    )

    assert result.success
    assert result.nfev > 0
    assert result.closure == "nonlrs_s2_nonlinear_transport_coevolution_v1"
    assert np.all(np.isfinite(result.A_modes_final))
    assert np.isfinite(result.Pi_plus_final)
    assert np.isfinite(result.Pi_minus_final)


def test_nonlrs_nonlinear_transport_solve_is_flrw_quiet() -> None:
    result = run_augmented_nonlrs_nonlinear_transport_solve(
        Sigma_plus0=0.0,
        Sigma_minus0=0.0,
        N_span=(0.0, 1.0e-3),
        N_q=4,
        n_species=1,
        N_mu=5,
        N_phi=7,
        method="LSODA",
    )

    assert result.success
    assert np.max(np.abs(result.Sigma_plus)) == pytest.approx(0.0, abs=1.0e-14)
    assert np.max(np.abs(result.Sigma_minus)) == pytest.approx(0.0, abs=1.0e-14)
    assert np.max(np.abs(result.A_modes_final[:, 1:])) == pytest.approx(0.0, abs=1.0e-14)


def test_nonlrs_nonlinear_transport_solve_preserves_sigma_minus_zero_reduction() -> None:
    result = run_augmented_nonlrs_nonlinear_transport_solve(
        Sigma_plus0=0.02,
        Sigma_minus0=0.0,
        N_span=(0.0, 1.0e-3),
        N_q=4,
        n_species=1,
        N_mu=5,
        N_phi=7,
        method="LSODA",
    )

    assert result.success
    assert np.max(np.abs(result.Sigma_minus)) == pytest.approx(0.0, abs=1.0e-12)
    assert np.max(np.abs(result.A_modes_final[:, 2])) == pytest.approx(0.0, abs=1.0e-12)


def test_nonlrs_nonlinear_transport_solve_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="N_span"):
        run_augmented_nonlrs_nonlinear_transport_solve(
            Sigma_plus0=0.0,
            Sigma_minus0=0.0,
            N_span=(1.0, 0.0),
        )
    with pytest.raises(ValueError, match="initial_A_modes"):
        run_augmented_nonlrs_nonlinear_transport_solve(
            Sigma_plus0=0.0,
            Sigma_minus0=0.0,
            N_q=4,
            n_species=2,
            initial_A_modes=np.zeros((1, 3, 4)),
        )
