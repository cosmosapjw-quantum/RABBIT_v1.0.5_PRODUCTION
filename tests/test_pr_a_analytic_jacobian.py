"""Characteristic analytic transport regressions."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.integrate import solve_ivp

from rabbit.jax.characteristic_rays_jax import (
    intensity_shift_jax,
    jacobian_jax,
    mu_current_jax,
)


@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.3, 0.5])
def test_analytic_jacobian_matches_numerical_solution(sigma: float):
    """Paper eq. (51) must match the direct numerical ODE solution."""
    n_mu = 12
    mu0, _ = np.polynomial.legendre.leggauss(n_mu)
    x0 = mu0**2 / np.maximum(1.0 - mu0**2, 1e-30)
    signs = np.where(mu0 >= 0.0, 1.0, -1.0)

    def numerical_j(n_final: float) -> np.ndarray:
        def rhs(_n, y):
            j_vals = y[:n_mu]
            mu_vals = y[n_mu:]
            d_j = 3.0 * sigma * (1.0 - 3.0 * mu_vals**2) * j_vals
            d_mu = 3.0 * sigma * mu_vals * (1.0 - mu_vals**2)
            return np.concatenate([d_j, d_mu])

        y0 = np.concatenate([np.ones(n_mu), mu0])
        sol = solve_ivp(
            rhs,
            (0.0, n_final),
            y0,
            method="Radau",
            rtol=1e-12,
            atol=1e-14,
        )
        assert sol.success
        return sol.y[:n_mu, -1]

    for n_final in (0.5, 2.0, 5.0, 8.0):
        s_val = sigma * n_final
        j_num = numerical_j(n_final)
        mu_now = np.asarray(
            mu_current_jax(jnp.asarray(x0), jnp.asarray(signs), jnp.asarray(s_val))
        )
        j_ana = np.asarray(
            jacobian_jax(jnp.asarray(x0), jnp.asarray(s_val), jnp.asarray(mu_now))
        )
        max_err = float(np.max(np.abs(j_num - j_ana)))
        assert max_err < 1e-9, (
            f"Σ={sigma}, N={n_final}: analytic J mismatch {max_err:.2e}"
        )


@pytest.mark.parametrize("sigma", [0.05, 0.1, 0.3, 0.5])
def test_analytic_intensity_shift_matches_numerical_solution(sigma: float):
    """Closed-form I_j(S) must match the directly integrated transport ODE."""
    n_mu = 11
    mu0, _ = np.polynomial.legendre.leggauss(n_mu)
    x0 = mu0**2 / np.maximum(1.0 - mu0**2, 1e-30)
    signs = np.where(mu0 >= 0.0, 1.0, -1.0)

    def numerical_i(n_final: float) -> np.ndarray:
        def rhs(_n, y):
            i_vals = y[:n_mu]
            mu_vals = y[n_mu:]
            d_i = sigma * 0.5 * (3.0 * mu_vals**2 - 1.0)
            d_mu = 3.0 * sigma * mu_vals * (1.0 - mu_vals**2)
            return np.concatenate([d_i, d_mu])

        y0 = np.concatenate([np.zeros(n_mu), mu0])
        sol = solve_ivp(
            rhs,
            (0.0, n_final),
            y0,
            method="Radau",
            rtol=1e-12,
            atol=1e-14,
        )
        assert sol.success
        return sol.y[:n_mu, -1]

    for n_final in (0.5, 2.0, 5.0, 8.0):
        s_val = sigma * n_final
        i_num = numerical_i(n_final)
        i_ana = np.asarray(intensity_shift_jax(jnp.asarray(x0), jnp.asarray(s_val)))
        mu_now = np.asarray(
            mu_current_jax(jnp.asarray(x0), jnp.asarray(signs), jnp.asarray(s_val))
        )
        assert np.all(np.isfinite(i_ana))
        assert float(i_ana[n_mu // 2]) == pytest.approx(-0.5 * s_val)
        max_err = float(np.max(np.abs(i_num - i_ana)))
        assert max_err < 1e-9, (
            f"Σ={sigma}, N={n_final}: analytic I mismatch {max_err:.2e}"
        )
        assert np.all(np.abs(mu_now) <= 1.0)
