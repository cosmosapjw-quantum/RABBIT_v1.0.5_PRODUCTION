from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from rabbit.transport.augmented_pstf_distribution import (
    distribution_rhs_to_augmented_rhs,
    fermi_dirac_from_logit,
    gram_matrix_from_basis,
    mode_norms_from_basis,
    modes_to_nodal,
    project_nodal_to_modes,
    projection_matrix_from_basis,
    reconstruct_distribution,
)


def _lrs_basis(ell_max: int = 4, n_mu: int = 16):
    mu, weights = leggauss(n_mu)
    modes = []
    for ell in range(0, ell_max + 1, 2):
        if ell == 0:
            modes.append(np.ones_like(mu))
        elif ell == 2:
            modes.append(0.5 * (3.0 * mu**2 - 1.0))
        elif ell == 4:
            modes.append((35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0)
        else:
            raise ValueError("test helper only supports ell <= 4")
    return np.asarray(modes), mu, weights


def test_fermi_dirac_from_logit_is_overflow_safe() -> None:
    values = fermi_dirac_from_logit(np.array([-1000.0, 0.0, 1000.0]))

    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(0.0)


def test_zero_augmented_modes_reconstruct_fd_monopole_on_all_angles() -> None:
    basis, _, _ = _lrs_basis(ell_max=4)
    q = np.array([0.5, 1.0, 2.0])
    A = np.zeros((basis.shape[0], q.size))

    f = reconstruct_distribution(A, q, basis)
    expected = fermi_dirac_from_logit(q)

    assert f.shape == (q.size, basis.shape[1])
    assert np.allclose(f, expected[:, None])


def test_modes_to_nodal_and_reconstruct_support_species_axis() -> None:
    basis, _, _ = _lrs_basis(ell_max=2)
    q = np.array([0.25, 0.75])
    A = np.zeros((4, basis.shape[0], q.size))
    A[:, 1, :] = 0.1

    nodal = modes_to_nodal(A, basis)
    f = reconstruct_distribution(A, q, basis)

    assert nodal.shape == (4, q.size, basis.shape[1])
    assert f.shape == (4, q.size, basis.shape[1])
    assert np.all((0.0 <= f) & (f <= 1.0))


def test_project_nodal_to_modes_roundtrips_lrs_polynomial_modes() -> None:
    basis, _, weights = _lrs_basis(ell_max=4, n_mu=24)
    q = np.array([0.5, 1.5, 3.0])
    A = np.zeros((basis.shape[0], q.size))
    A[0, :] = [0.1, -0.2, 0.3]
    A[1, :] = [0.01, 0.02, 0.03]
    A[2, :] = [-0.04, 0.0, 0.04]

    nodal = modes_to_nodal(A, basis)
    restored = project_nodal_to_modes(nodal, basis, weights)

    assert np.allclose(restored, A, atol=1.0e-13)


def test_project_nodal_to_modes_solves_non_orthogonal_gram_system() -> None:
    basis = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    weights = np.array([1.0, 1.0, 1.0])
    A = np.array([[0.5, -0.2], [0.1, 0.2]])

    nodal = modes_to_nodal(A, basis)
    restored = project_nodal_to_modes(nodal, basis, weights)

    assert not np.allclose(
        project_nodal_to_modes(
            nodal,
            basis,
            weights,
            mode_norms=mode_norms_from_basis(basis, weights),
        ),
        A,
    )
    assert np.allclose(restored, A, atol=1.0e-13)


def test_project_nodal_to_modes_accepts_precomputed_projection_matrix() -> None:
    basis = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    weights = np.array([1.0, 1.0, 1.0])
    A = np.array([[0.5, -0.2], [0.1, 0.2]])
    nodal = modes_to_nodal(A, basis)

    projection_matrix = projection_matrix_from_basis(basis, weights)
    restored = project_nodal_to_modes(
        nodal,
        basis,
        weights,
        projection_matrix=projection_matrix,
    )

    np.testing.assert_allclose(restored, A, rtol=0.0, atol=1.0e-13)


def test_distribution_rhs_to_augmented_rhs_recovers_known_mode_rhs() -> None:
    basis, _, weights = _lrs_basis(ell_max=2, n_mu=20)
    q = np.array([0.5, 1.0])
    A = np.zeros((basis.shape[0], q.size))
    dA_expected = np.array([[0.1, -0.2], [0.03, 0.04]])

    f = reconstruct_distribution(A, q, basis)
    nodal_dA = modes_to_nodal(dA_expected, basis)
    df = -f * (1.0 - f) * nodal_dA

    dA = distribution_rhs_to_augmented_rhs(df, f, basis, weights)

    assert np.allclose(dA, dA_expected, atol=1.0e-13)


def test_distribution_rhs_to_augmented_rhs_handles_pauli_floor() -> None:
    basis = np.ones((1, 2))
    weights = np.array([1.0, 1.0])
    f = np.array([[0.0, 1.0]])
    df = np.array([[1.0e-20, -1.0e-20]])

    dA = distribution_rhs_to_augmented_rhs(df, f, basis, weights, eps_f=1.0e-12)

    assert dA.shape == (1, 1)
    assert np.all(np.isfinite(dA))


def test_projection_rejects_near_zero_basis_norm() -> None:
    basis = np.zeros((1, 2))
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="near-zero quadrature norm"):
        mode_norms_from_basis(basis, weights)


def test_gram_matrix_rejects_singular_basis() -> None:
    basis = np.array([[1.0, 1.0], [2.0, 2.0]])
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="singular"):
        gram_matrix_from_basis(basis, weights)
