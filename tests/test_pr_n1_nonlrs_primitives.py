from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("jax", reason="JAX required for PR-N1 non-LRS primitive tests")
pytest.importorskip("scipy", reason="SciPy required for direction-map cross-check")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from rabbit.jax.characteristic_rays_jax import (
    extract_monopole_jax,
    extract_stress_jax,
    intensity_shift_jax,
    jacobian_jax,
    mu_current_jax,
)
from rabbit.jax.characteristic_rays_nonlrs_jax import (
    extract_monopole_S2,
    extract_stress_minus_S2,
    extract_stress_plus_S2,
    mu_phi_current_jax,
    setup_ray_grid_S2,
)

_SQRT3 = math.sqrt(3.0)


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _direction_components(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sin_theta = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), mu],
        axis=-1,
    )


def _direction_ode(_N: float, n: np.ndarray, sigma_plus: float, sigma_minus: float) -> np.ndarray:
    lam_x = sigma_plus + _SQRT3 * sigma_minus
    lam_y = sigma_plus - _SQRT3 * sigma_minus
    lam_z = -2.0 * sigma_plus
    nx, ny, nz = n
    kappa = lam_x * nx * nx + lam_y * ny * ny + lam_z * nz * nz
    return np.array(
        [
            (kappa - lam_x) * nx,
            (kappa - lam_y) * ny,
            (kappa - lam_z) * nz,
        ],
        dtype=np.float64,
    )


def test_setup_ray_grid_s2_shapes_and_weight_sum():
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(11, 8)

    assert mu0.shape == (88,)
    assert phi0.shape == (88,)
    assert w_s2.shape == (88,)
    assert X0.shape == (88,)
    assert signs.shape == (88,)
    assert float(jnp.sum(w_s2)) == pytest.approx(2.0, abs=1e-14)
    assert float(jnp.min(phi0)) > 0.0
    assert float(jnp.max(phi0)) < 2.0 * math.pi

    x0_expected = np.asarray(mu0) ** 2 / np.maximum(1.0 - np.asarray(mu0) ** 2, 1e-30)
    np.testing.assert_allclose(np.asarray(X0), x0_expected, rtol=0.0, atol=1e-15)

    center = int(np.argmin(np.abs(np.asarray(mu0))))
    assert float(np.asarray(mu0)[center]) == pytest.approx(0.0, abs=1e-15)
    assert float(np.asarray(signs)[center]) == pytest.approx(1.0)


def test_mu_phi_current_reduces_to_lrs_map():
    n_theta = 11
    mu0, phi0, _w_s2, X0, signs = setup_ray_grid_S2(n_theta, 1)
    S_plus = jnp.asarray(0.37)
    mu_nonlrs, phi_nonlrs = mu_phi_current_jax(mu0, phi0, S_plus, jnp.asarray(0.0))
    mu_lrs = mu_current_jax(X0, signs, S_plus)

    np.testing.assert_allclose(np.asarray(mu_nonlrs), np.asarray(mu_lrs), rtol=0.0, atol=1e-14)
    phi_delta = _wrap_angle(np.asarray(phi_nonlrs) - np.asarray(phi0))
    np.testing.assert_allclose(phi_delta, 0.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize(
    ("sigma_plus", "sigma_minus", "mu0", "phi0", "delta_N"),
    [
        (0.12, 0.04, 0.35, 0.40, 1.75),
        (0.09, -0.03, -0.55, 1.20, 2.10),
        (0.05, 0.02, 0.10, 2.30, 3.00),
    ],
)
def test_mu_phi_current_matches_constant_shear_direction_ode(
    sigma_plus: float,
    sigma_minus: float,
    mu0: float,
    phi0: float,
    delta_N: float,
):
    mu_analytic, phi_analytic = mu_phi_current_jax(
        jnp.asarray([mu0]),
        jnp.asarray([phi0]),
        jnp.asarray(sigma_plus * delta_N),
        jnp.asarray(sigma_minus * delta_N),
    )
    vec0 = _direction_components(np.asarray([mu0]), np.asarray([phi0]))[0]
    sol = solve_ivp(
        _direction_ode,
        (0.0, delta_N),
        vec0,
        args=(sigma_plus, sigma_minus),
        rtol=1e-11,
        atol=1e-13,
    )
    assert sol.success
    vec_num = sol.y[:, -1]
    vec_num /= np.linalg.norm(vec_num)
    vec_analytic = _direction_components(np.asarray(mu_analytic), np.asarray(phi_analytic))[0]
    np.testing.assert_allclose(vec_analytic, vec_num, rtol=0.0, atol=3e-11)


def test_lrs_reduction_stress_plus_matches_existing_lrs_primitive():
    n_theta = 12
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(n_theta, 1)
    _mu_nodes, w_mu = np.polynomial.legendre.leggauss(n_theta)
    S_plus = jnp.asarray(0.42)
    mu_lrs = mu_current_jax(X0, signs, S_plus)
    mu_nonlrs, _phi_nonlrs = mu_phi_current_jax(mu0, phi0, S_plus, jnp.asarray(0.0))
    I_vals = intensity_shift_jax(X0, S_plus)
    J_vals = jacobian_jax(X0, S_plus, mu_lrs)

    pi_plus_lrs = extract_stress_jax(I_vals, J_vals, mu_lrs, jnp.asarray(w_mu), 0.40520)
    pi_plus_nonlrs = extract_stress_plus_S2(I_vals, J_vals, mu_nonlrs, w_s2, 0.40520)

    assert float(pi_plus_nonlrs) == pytest.approx(float(pi_plus_lrs), abs=1e-12)


def test_lrs_reduction_monopole_matches_existing_lrs_primitive():
    n_theta = 12
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(n_theta, 1)
    _mu_nodes, w_mu = np.polynomial.legendre.leggauss(n_theta)
    q_nodes, _ = np.polynomial.laguerre.laggauss(6)
    S_plus = jnp.asarray(0.31)
    mu_lrs = mu_current_jax(X0, signs, S_plus)
    I_vals = intensity_shift_jax(X0, S_plus)
    J_vals = jacobian_jax(X0, S_plus, mu_lrs)

    mono_lrs = extract_monopole_jax(I_vals, J_vals, jnp.asarray(w_mu), jnp.asarray(q_nodes))
    mono_nonlrs = extract_monopole_S2(I_vals, J_vals, w_s2, jnp.asarray(q_nodes))

    np.testing.assert_allclose(np.asarray(mono_nonlrs), np.asarray(mono_lrs), rtol=0.0, atol=1e-12)


def test_pi_minus_vanishes_at_sigma_minus_zero():
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(12, 16)
    S_plus = jnp.asarray(0.28)
    mu, phi = mu_phi_current_jax(mu0, phi0, S_plus, jnp.asarray(0.0))
    I_vals = intensity_shift_jax(X0, S_plus)
    J_vals = jacobian_jax(X0, S_plus, mu_current_jax(X0, signs, S_plus))

    pi_minus = extract_stress_minus_S2(I_vals, J_vals, mu, phi, w_s2, 0.40520)
    assert abs(float(pi_minus)) < 1e-12


def test_s2_quadrupole_kernel_normalizations_are_locked():
    from rabbit.jax.characteristic_rays_nonlrs_jax import (
        stress_minus_kernel_S2_jax,
        stress_plus_kernel_S2_jax,
    )

    mu0, phi0, w_s2, _X0, _signs = setup_ray_grid_S2(8, 8)
    k_plus = stress_plus_kernel_S2_jax(mu0)
    k_minus = stress_minus_kernel_S2_jax(mu0, phi0)

    plus_norm = jnp.sum(w_s2 * k_plus * k_plus)
    minus_norm = jnp.sum(w_s2 * k_minus * k_minus)

    assert float(plus_norm) == pytest.approx(2.0 / 5.0, abs=1e-14)
    assert float(minus_norm) == pytest.approx(2.0 / 5.0, abs=1e-14)


def test_xy_swap_symmetry_of_nonlrs_forward_map():
    n_theta, n_phi = 10, 8
    mu0, phi0, w_s2, _X0, _signs = setup_ray_grid_S2(n_theta, n_phi)
    S_plus = jnp.asarray(0.19)
    S_minus = jnp.asarray(0.11)

    mu_a, phi_a = mu_phi_current_jax(mu0, phi0, S_plus, S_minus)
    mu_b, phi_b = mu_phi_current_jax(mu0, (0.5 * math.pi) - phi0, S_plus, -S_minus)

    mu_a_grid = np.asarray(mu_a).reshape(n_theta, n_phi)
    phi_a_grid = np.asarray(phi_a).reshape(n_theta, n_phi)
    mu_b_grid = np.asarray(mu_b).reshape(n_theta, n_phi)
    phi_b_grid = np.asarray(phi_b).reshape(n_theta, n_phi)

    np.testing.assert_allclose(mu_b_grid, mu_a_grid, rtol=0.0, atol=1e-14)
    phi_residual = _wrap_angle(phi_b_grid - ((0.5 * math.pi) - phi_a_grid))
    np.testing.assert_allclose(phi_residual, 0.0, rtol=0.0, atol=1e-13)

    I_vals = jnp.zeros_like(mu_a)
    J_vals = jnp.ones_like(mu_a)
    pi_plus_a = extract_stress_plus_S2(I_vals, J_vals, mu_a, w_s2, 0.40520)
    pi_plus_b = extract_stress_plus_S2(I_vals, J_vals, mu_b, w_s2, 0.40520)
    pi_minus_a = extract_stress_minus_S2(I_vals, J_vals, mu_a, phi_a, w_s2, 0.40520)
    pi_minus_b = extract_stress_minus_S2(I_vals, J_vals, mu_b, phi_b, w_s2, 0.40520)

    assert float(pi_plus_a) == pytest.approx(float(pi_plus_b), abs=1e-12)
    assert float(pi_minus_a) == pytest.approx(-float(pi_minus_b), abs=1e-12)
