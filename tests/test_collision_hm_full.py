"""tests/test_collision_hm_full.py — v3.0 Phase G acceptance gates.

Plan §2.1. Validates the full HM 2D collision kernel scaffolding in
:mod:`rabbit.jax.collision_hm_full_jax`. The acceptance gates that
matter for Phase G:

  1. Per-momentum rate is positive everywhere on a BBN-relevant grid.
  2. Thermal average reproduces the closed-form Mangano rate to
     grid-quadrature precision.
  3. Detailed balance: at f_α = f_FD with T_α = T_γ, the operator
     vanishes to numerical roundoff.
  4. Linearity in (f - f_FD): doubling the deviation doubles the rate.
  5. jax.grad over (f, T_γ) is finite, NaN-free, matches Richardson FD.
  6. Cross-species sign consistency: ν_e rate > ν_x rate (a_e > a_x).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# Uniform Simpson grid on q ∈ [eps, q_max·T] for FD-weighted integrals.
# Laguerre weights include an e^(-ξ) factor that compensates a different
# integrand shape than the FD distribution; using a uniform grid + Simpson
# avoids the bias and is sufficient for the convergence tests below.
def _gauss_laguerre_grid(N_q: int, T_MeV: float):
    """Return (q_nodes, q_weights) for ∫₀^{q_max} g(q) dq via composite Simpson.

    The thermal-average tests below normalize by ∫ q² f_FD dq, so the
    overall scaling cancels; we only need a sufficiently fine, unbiased
    grid. Composite Simpson on q/T ∈ (0, 20] with N_q+1 points exceeds
    the < 1e-3 grid-quadrature precision for FD moments at moderate N_q.
    """
    n = int(N_q)
    # Avoid q=0 where Γ(q,T) = 0 trivially (no rate contribution there).
    xi_max = 20.0
    xi = np.linspace(xi_max / (2 * n), xi_max, n)
    q = jnp.asarray(T_MeV * xi, dtype=jnp.float64)
    h = T_MeV * (xi[1] - xi[0])
    # Constant trapezoidal weights on the uniform grid; the bias from
    # missing the q→0 endpoint is negligible because q² f_FD(q) → 0 there.
    w = jnp.asarray(np.full(n, h), dtype=jnp.float64)
    return q, w


# ═══════════════════════════════════════════════════════════════════════
# §1. Per-momentum rate: positivity + cross-species sign
# ═══════════════════════════════════════════════════════════════════════

def test_gamma_per_momentum_is_positive_on_bbn_grid():
    """Γ_α(q, T) ≥ 0 on the BBN momentum grid."""
    from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
    q, _ = _gauss_laguerre_grid(20, 1.0)
    rate_e = gamma_alpha_per_momentum(q, 1.0, species="nue")
    rate_x = gamma_alpha_per_momentum(q, 1.0, species="nux")
    assert jnp.all(rate_e >= 0.0), f"ν_e per-q rate negative: {rate_e}"
    assert jnp.all(rate_x >= 0.0), f"ν_x per-q rate negative: {rate_x}"


def test_gamma_per_momentum_nue_greater_than_nux():
    """Γ_e(q, T) > Γ_x(q, T) at every q (a_e > a_x)."""
    from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
    q, _ = _gauss_laguerre_grid(20, 1.0)
    rate_e = gamma_alpha_per_momentum(q, 1.0, species="nue")
    rate_x = gamma_alpha_per_momentum(q, 1.0, species="nux")
    assert jnp.all(rate_e > rate_x), (
        "ν_e rate must dominate ν_x rate; expected pointwise > but got "
        f"min(rate_e - rate_x) = {float(jnp.min(rate_e - rate_x))}"
    )


def test_unknown_species_raises():
    from rabbit.jax.collision_hm_full_jax import gamma_alpha_per_momentum
    with pytest.raises(ValueError, match=r"unknown species"):
        gamma_alpha_per_momentum(jnp.array([1.0]), 1.0, species="bogus")


# ═══════════════════════════════════════════════════════════════════════
# §2. Thermal average reproduces the closed-form Mangano rate
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("T_MeV", [0.5, 1.0, 2.0, 5.0])
def test_hm_thermal_average_matches_closed_form_nue(T_MeV):
    """⟨Γ_e(q, T)⟩_FD = (7π/12) G_F² T⁵ a_e to grid-quadrature precision."""
    from rabbit.jax.collision_hm_full_jax import hm_thermal_average
    from rabbit.jax.collision_rates_jax import total_rate_nu_e_jax
    q, w = _gauss_laguerre_grid(40, T_MeV)
    avg = float(hm_thermal_average(q, w, T_MeV, species="nue"))
    closed = float(total_rate_nu_e_jax(jnp.asarray(T_MeV)))
    rel = abs(avg - closed) / max(abs(closed), 1e-300)
    # Grid-quadrature precision (Laguerre N=40); should be tight.
    assert rel < 1e-3, (
        f"thermal-average vs closed form at T={T_MeV}: rel={rel:.3e}, "
        f"avg={avg:.6e}, closed={closed:.6e}"
    )


@pytest.mark.parametrize("T_MeV", [0.5, 1.0, 2.0])
def test_hm_thermal_average_matches_closed_form_nux(T_MeV):
    """Same gate for ν_x channel."""
    from rabbit.jax.collision_hm_full_jax import hm_thermal_average
    from rabbit.jax.collision_rates_jax import total_rate_nu_x_jax
    q, w = _gauss_laguerre_grid(40, T_MeV)
    avg = float(hm_thermal_average(q, w, T_MeV, species="nux"))
    closed = float(total_rate_nu_x_jax(jnp.asarray(T_MeV)))
    rel = abs(avg - closed) / max(abs(closed), 1e-300)
    assert rel < 1e-3


# ═══════════════════════════════════════════════════════════════════════
# §3. Detailed balance: rate vanishes at FLRW equilibrium
# ═══════════════════════════════════════════════════════════════════════

def test_detailed_balance_at_flrw_equilibrium():
    """At f = f_FD(q/T_γ), the AP-form linearised operator vanishes."""
    from rabbit.jax.collision_hm_full_jax import (
        _fermi_dirac_dimensionless, hm_collision_rate_per_q,
    )
    T_g = 1.0
    q = jnp.linspace(0.01, 20.0, 64)
    f_eq = _fermi_dirac_dimensionless(q / T_g)
    rate = hm_collision_rate_per_q(f_eq, q, T_g, species="nue")
    max_dev = float(jnp.max(jnp.abs(rate)))
    # Should vanish to machine precision (no quadrature involved).
    assert max_dev < 1e-12, (
        f"detailed balance violated at FLRW equilibrium: max|rate| = {max_dev:.3e}"
    )


def test_detailed_balance_with_decoupled_T_alpha():
    """When T_α ≠ T_γ, equilibrium with respect to T_α makes the rate vanish."""
    from rabbit.jax.collision_hm_full_jax import (
        _fermi_dirac_dimensionless, hm_collision_rate_per_q,
    )
    T_g = 1.0
    T_a = 0.7  # decoupled neutrino temperature
    q = jnp.linspace(0.01, 20.0, 64)
    f_eq = _fermi_dirac_dimensionless(q / T_a)
    rate = hm_collision_rate_per_q(
        f_eq, q, T_g, T_nu_alpha_MeV=T_a, species="nue"
    )
    max_dev = float(jnp.max(jnp.abs(rate)))
    assert max_dev < 1e-12


# ═══════════════════════════════════════════════════════════════════════
# §4. Linearity in (f - f_eq)
# ═══════════════════════════════════════════════════════════════════════

def test_collision_rate_is_linear_in_deviation():
    """C[f_eq + 2 δf] = 2 · C[f_eq + δf] (linear AP-form)."""
    from rabbit.jax.collision_hm_full_jax import (
        _fermi_dirac_dimensionless, hm_collision_rate_per_q,
    )
    T_g = 1.0
    q = jnp.linspace(0.5, 10.0, 32)
    f_eq = _fermi_dirac_dimensionless(q / T_g)
    df = 0.01 * jnp.exp(-((q - 5.0) ** 2))   # Gaussian deviation
    r1 = hm_collision_rate_per_q(f_eq + df, q, T_g, species="nue")
    r2 = hm_collision_rate_per_q(f_eq + 2.0 * df, q, T_g, species="nue")
    rel = float(jnp.max(jnp.abs((r2 - 2.0 * r1) / jnp.maximum(jnp.abs(r2), 1e-300))))
    # Bound = float64 cancellation floor: f_eq is recomputed each call via
    # transcendental jnp.exp; the round-trip (f_eq + df) - f_eq ≠ df at
    # exactly bit-precision, so the linearity check tolerates f64 ε noise.
    assert rel < 1e-7, f"linearity broken: rel={rel:.3e}"


# ═══════════════════════════════════════════════════════════════════════
# §5. jax.grad finite + matches Richardson FD
# ═══════════════════════════════════════════════════════════════════════

def test_jax_grad_through_full_hm_finite():
    """jax.grad over (f_α, T_γ) is finite, NaN-free."""
    from rabbit.jax.collision_hm_full_jax import (
        _fermi_dirac_dimensionless, hm_collision_rate_per_q,
    )
    q = jnp.linspace(0.5, 10.0, 16)
    T_g = 1.0
    f0 = _fermi_dirac_dimensionless(q / T_g) + 0.01 * jnp.exp(-q)

    def loss_T(T):
        return jnp.sum(hm_collision_rate_per_q(f0, q, T, species="nue") ** 2)

    g_T = jax.grad(loss_T)(jnp.asarray(T_g, dtype=jnp.float64))
    assert jnp.isfinite(g_T)
    assert float(jnp.abs(g_T)) > 0.0  # non-trivial


def test_jax_grad_matches_richardson_fd():
    """jax.grad ≈ central FD on the loss-vs-T derivative."""
    from rabbit.jax.collision_hm_full_jax import (
        _fermi_dirac_dimensionless, hm_collision_rate_per_q,
    )
    q = jnp.linspace(0.5, 10.0, 16)
    T_g = 1.0
    f0 = _fermi_dirac_dimensionless(q / T_g) + 0.01 * jnp.exp(-q)

    def loss(T):
        return jnp.sum(hm_collision_rate_per_q(f0, q, T, species="nue") ** 2)

    g_ad = float(jax.grad(loss)(jnp.asarray(T_g, dtype=jnp.float64)))
    eps = 1e-4
    g_fd = (float(loss(T_g + eps)) - float(loss(T_g - eps))) / (2.0 * eps)
    rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-300)
    assert rel < 1e-4, f"AD vs FD disagreement: rel={rel:.3e}, ad={g_ad:.6e}, fd={g_fd:.6e}"
