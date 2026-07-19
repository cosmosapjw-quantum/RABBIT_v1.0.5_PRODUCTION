from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from rabbit.transport.augmented_typeI_observables import (
    angular_monopole,
    lrs_quadrupole_kernel,
    non_lrs_quadrupole_kernels,
    stress_moments_from_distribution,
)


def test_angular_monopole_preserves_species_and_q_axes() -> None:
    f = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    weights = np.ones(4)

    monopole = angular_monopole(f, weights)

    assert monopole.shape == (2, 3)
    assert np.allclose(monopole, np.mean(f, axis=-1))


def test_lrs_isotropic_distribution_has_zero_quadrupole_stress() -> None:
    mu, w_mu = leggauss(24)
    W_plus = lrs_quadrupole_kernel(mu)
    f = 0.5 * np.ones((3, mu.size))
    q_energy_weights = np.array([0.2, 0.3, 0.5])

    moments = stress_moments_from_distribution(f, q_energy_weights, w_mu, W_plus)

    assert moments.rho_weighted > 0.0
    assert abs(float(moments.pi_plus_tilde)) < 1.0e-14
    assert np.allclose(moments.monopole_q, 0.5)


def test_lrs_quadrupole_distortion_sources_plus_stress_with_expected_sign() -> None:
    mu, w_mu = leggauss(32)
    W_plus = lrs_quadrupole_kernel(mu)
    q_energy_weights = np.array([1.0])
    f = 0.5 + 0.05 * W_plus[None, :]

    moments = stress_moments_from_distribution(f, q_energy_weights, w_mu, W_plus)

    assert float(moments.pi_plus_tilde) > 0.0


def test_non_lrs_phi_independent_distribution_reduces_to_lrs_plus_stress() -> None:
    mu, w_mu = leggauss(16)
    phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    mu_grid = np.broadcast_to(mu[:, None], (mu.size, phi.size)).reshape(-1)
    phi_grid = np.broadcast_to(phi[None, :], (mu.size, phi.size)).reshape(-1)
    weights = np.broadcast_to((w_mu[:, None] * (2.0 * np.pi / phi.size)), (mu.size, phi.size)).reshape(-1)
    W_plus, W_minus = non_lrs_quadrupole_kernels(mu_grid, phi_grid)

    f_s2 = 0.5 + 0.02 * W_plus[None, :]
    q_energy_weights = np.array([1.0])
    s2 = stress_moments_from_distribution(
        f_s2,
        q_energy_weights,
        weights,
        W_plus,
        W_minus=W_minus,
    )

    f_lrs = 0.5 + 0.02 * lrs_quadrupole_kernel(mu)[None, :]
    lrs = stress_moments_from_distribution(
        f_lrs,
        q_energy_weights,
        w_mu,
        lrs_quadrupole_kernel(mu),
    )

    assert s2.pi_minus_tilde is not None
    assert abs(float(s2.pi_minus_tilde)) < 1.0e-14
    assert np.allclose(s2.pi_plus_tilde, lrs.pi_plus_tilde, atol=1.0e-14)


def test_non_lrs_cos2phi_distortion_sources_minus_stress() -> None:
    mu, w_mu = leggauss(16)
    phi = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    mu_grid = np.broadcast_to(mu[:, None], (mu.size, phi.size)).reshape(-1)
    phi_grid = np.broadcast_to(phi[None, :], (mu.size, phi.size)).reshape(-1)
    weights = np.broadcast_to((w_mu[:, None] * (2.0 * np.pi / phi.size)), (mu.size, phi.size)).reshape(-1)
    W_plus, W_minus = non_lrs_quadrupole_kernels(mu_grid, phi_grid)
    f = 0.5 + 0.03 * W_minus[None, :]

    moments = stress_moments_from_distribution(
        f,
        np.array([1.0]),
        weights,
        W_plus,
        W_minus=W_minus,
    )

    assert moments.pi_minus_tilde is not None
    assert float(moments.pi_minus_tilde) > 0.0
    assert abs(float(moments.pi_plus_tilde)) < 1.0e-14


def test_stress_moments_compute_from_single_validated_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rabbit.transport import augmented_typeI_observables as observables

    mu, w_mu = leggauss(12)
    W_plus = lrs_quadrupole_kernel(mu)
    f = 0.5 + 0.02 * W_plus[None, :]

    def fail_repeated_energy_integral(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("stress_moments_from_distribution should not revalidate via _energy_integral")

    monkeypatch.setattr(observables, "_energy_integral", fail_repeated_energy_integral)

    moments = stress_moments_from_distribution(f, np.array([1.0]), w_mu, W_plus)

    assert moments.rho_weighted > 0.0
    assert float(moments.pi_plus_tilde) > 0.0


def test_stress_moments_reject_unphysical_distribution_values() -> None:
    mu, w_mu = leggauss(8)
    W_plus = lrs_quadrupole_kernel(mu)
    f = np.full((1, mu.size), 1.1)

    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        stress_moments_from_distribution(f, np.array([1.0]), w_mu, W_plus)
