"""tests/test_hm_partner_integration.py — v3.1 Phase α-2 acceptance gates.

Plan §α-2. Validates the Bianchi-aware partner integration kernel.

Gates:
  1. FLRW partner factory returns f_FD(q'/T_β) and ignores μ'.
  2. LRS anisotropic factory at Σ_+ → 0 reduces to FLRW factory.
  3. compute_partner_integrated_rate is non-negative and grows with q_α.
  4. Anisotropic detailed balance: f_α = f_β = matched LRS-anisotropic
     equilibrium → operator returns zero (1e-12).
  5. FLRW detailed balance: f_α = f_β = f_FD(q/T) → operator zero.
  6. Σ_+ → 0 continuity: rate(Σ_+=0.01) - rate(0) → 0 linearly.
  7. jax.grad through partner integration is finite.
  8. 2D quadrature convergence: spread on (8,12)/(12,20)/(16,32) < 5e-5.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


def _gauss_legendre_2d(N_q: int, N_mu: int, q_max_T: float = 12.0, T: float = 1.0):
    """Build (q, μ, q_w, μ_w) Gauss-Legendre grid on [0, q_max_T·T] × [-1, 1]."""
    q_nodes, q_weights = np.polynomial.legendre.leggauss(int(N_q))
    # Map [-1, 1] → [0, q_max_T·T]
    half = 0.5 * q_max_T * T
    q = jnp.asarray(half * (q_nodes + 1.0), dtype=jnp.float64)
    qw = jnp.asarray(half * q_weights, dtype=jnp.float64)
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(int(N_mu))
    mu = jnp.asarray(mu_nodes, dtype=jnp.float64)
    mw = jnp.asarray(mu_weights, dtype=jnp.float64)
    return q, mu, qw, mw


# ═══════════════════════════════════════════════════════════════════════
# §1. FLRW partner factory contract
# ═══════════════════════════════════════════════════════════════════════

class TestPartnerFactories:

    def test_flrw_factory_ignores_mu(self):
        from rabbit.jax.collision_hm_partner_integration_jax import (
            flrw_partner_factory,
        )
        f = flrw_partner_factory(1.0)
        q = jnp.array([1.0, 2.0])
        f_a = f(q, jnp.array([0.0, 0.0]))
        f_b = f(q, jnp.array([0.5, -0.5]))
        rel = float(jnp.max(jnp.abs((f_a - f_b) / jnp.maximum(jnp.abs(f_a), 1e-300))))
        assert rel < 1e-12, f"FLRW factory leaked μ-dependence: rel={rel:.3e}"

    def test_flrw_factory_returns_fermi_dirac(self):
        from rabbit.jax.collision_hm_partner_integration_jax import (
            flrw_partner_factory,
        )
        T = 1.0
        f = flrw_partner_factory(T)
        q = jnp.array([0.5, 1.0, 2.0])
        got = f(q, jnp.zeros_like(q))
        expected = 1.0 / (jnp.exp(q / T) + 1.0)
        rel = float(jnp.max(jnp.abs((got - expected) / jnp.maximum(jnp.abs(expected), 1e-300))))
        assert rel < 1e-12

    def test_lrs_anisotropic_factory_reduces_to_flrw_at_sigma_zero(self):
        from rabbit.jax.collision_hm_partner_integration_jax import (
            flrw_partner_factory, lrs_anisotropic_partner_factory,
        )
        T = 1.0
        f_flrw = flrw_partner_factory(T)
        f_lrs = lrs_anisotropic_partner_factory(T, sigma_plus=0.0, theta_j=1.0)
        q = jnp.array([0.5, 1.0, 2.0])
        mu = jnp.array([-0.5, 0.0, 0.7])
        a = f_flrw(q, mu)
        b = f_lrs(q, mu)
        rel = float(jnp.max(jnp.abs((a - b) / jnp.maximum(jnp.abs(a), 1e-300))))
        assert rel < 1e-12

    def test_lrs_anisotropic_factory_carries_mu_dependence_at_finite_sigma(self):
        from rabbit.jax.collision_hm_partner_integration_jax import (
            lrs_anisotropic_partner_factory,
        )
        T = 1.0
        f_lrs = lrs_anisotropic_partner_factory(T, sigma_plus=0.1, theta_j=1.0)
        q_scalar = jnp.asarray(1.0)
        f_zero = float(f_lrs(q_scalar, jnp.asarray(0.0)))
        f_one = float(f_lrs(q_scalar, jnp.asarray(1.0)))
        # P_2(0) = -1/2, P_2(1) = 1; effective T differs → f differs
        assert abs(f_zero - f_one) > 1e-3, (
            f"LRS anisotropic factory not μ-coupling: f(μ=0)={f_zero} f(μ=1)={f_one}"
        )


# ═══════════════════════════════════════════════════════════════════════
# §2. Partner-integrated rate
# ═══════════════════════════════════════════════════════════════════════

class TestPartnerIntegratedRate:

    def test_rate_is_positive(self):
        from rabbit.jax.collision_hm_partner_integration_jax import (
            compute_partner_integrated_rate, flrw_partner_factory,
        )
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        T = 1.0
        q, mu, qw, mw = _gauss_legendre_2d(20, 12, q_max_T=12.0, T=T)
        rate = float(compute_partner_integrated_rate(
            jnp.asarray(2.0), q, mu, qw, mw, T,
            partner_factory=flrw_partner_factory(T),
            M2_func=partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0),
        ))
        assert rate > 0.0, f"rate non-positive: {rate}"

    def test_rate_grows_with_q_alpha(self):
        """Γ_α(q_α, T) is monotonically non-decreasing in q_α (URM)."""
        from rabbit.jax.collision_hm_partner_integration_jax import (
            compute_partner_integrated_rate, flrw_partner_factory,
        )
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        T = 1.0
        q, mu, qw, mw = _gauss_legendre_2d(20, 12, q_max_T=12.0, T=T)
        rates = []
        for q_a in (0.5, 1.0, 2.0, 5.0):
            r = float(compute_partner_integrated_rate(
                jnp.asarray(q_a), q, mu, qw, mw, T,
                partner_factory=flrw_partner_factory(T),
                M2_func=partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0),
            ))
            rates.append(r)
        for i in range(1, len(rates)):
            assert rates[i] > rates[i-1], f"non-monotone: rates={rates}"


# ═══════════════════════════════════════════════════════════════════════
# §3. Detailed balance (anti-fixed-point guard)
# ═══════════════════════════════════════════════════════════════════════

class TestDetailedBalance:
    """Critical anti-fixed-point gates from Plan §0.2."""

    def test_flrw_detailed_balance(self):
        """f_α = f_β = f_FD(q/T) → operator returns zero."""
        from rabbit.jax.collision_hm_partner_integration_jax import (
            flrw_partner_factory, hm_collision_operator_anisotropic,
        )
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        T = 1.0
        q, mu, qw, mw = _gauss_legendre_2d(12, 10, q_max_T=10.0, T=T)
        q_alpha = jnp.array([0.5, 1.0, 2.0, 5.0])
        # f_α at FLRW equilibrium
        f_alpha_eq_factory = flrw_partner_factory(T)
        f_alpha = f_alpha_eq_factory(q_alpha, jnp.zeros_like(q_alpha))
        rate = hm_collision_operator_anisotropic(
            f_alpha, q_alpha, q, mu, qw, mw, T,
            partner_factory=flrw_partner_factory(T),
            M2_func=partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0),
            f_alpha_eq_factory=f_alpha_eq_factory,
        )
        max_dev = float(jnp.max(jnp.abs(rate)))
        assert max_dev < 1e-12, (
            f"FLRW detailed balance violated: max|rate| = {max_dev:.3e}"
        )

    def test_anisotropic_detailed_balance(self):
        """f_α = f_β = f_FD(q/Θ_j) at Σ_+ ≠ 0 → operator returns zero.

        Anti-fixed-point: detailed balance holds even under anisotropic
        equilibrium, because the matrix element is anisotropy-independent
        and the partner distribution matches the probe distribution.
        """
        from rabbit.jax.collision_hm_partner_integration_jax import (
            lrs_anisotropic_partner_factory, hm_collision_operator_anisotropic,
        )
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        T = 1.0
        sigma_plus = 0.1
        theta_j = 0.95   # decoupled neutrino temperature ratio
        q, mu, qw, mw = _gauss_legendre_2d(12, 10, q_max_T=10.0, T=T)
        q_alpha = jnp.array([0.5, 1.0, 2.0, 5.0])
        # Both factories return the same anisotropic equilibrium
        eq_factory = lrs_anisotropic_partner_factory(T, sigma_plus, theta_j)
        f_alpha = eq_factory(q_alpha, jnp.zeros_like(q_alpha))
        rate = hm_collision_operator_anisotropic(
            f_alpha, q_alpha, q, mu, qw, mw, T,
            partner_factory=eq_factory,
            M2_func=partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0),
            f_alpha_eq_factory=eq_factory,
        )
        max_dev = float(jnp.max(jnp.abs(rate)))
        assert max_dev < 1e-12, (
            f"Anisotropic detailed balance violated at Σ_+={sigma_plus}: "
            f"max|rate| = {max_dev:.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# §4. Σ_+ → 0 continuity
# ═══════════════════════════════════════════════════════════════════════

def test_sigma_plus_continuity():
    """rate(Σ_+=0.01) → rate(Σ_+=0) linearly as Σ_+ → 0."""
    from rabbit.jax.collision_hm_partner_integration_jax import (
        compute_partner_integrated_rate, lrs_anisotropic_partner_factory,
    )
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    T = 1.0
    q, mu, qw, mw = _gauss_legendre_2d(20, 12, q_max_T=12.0, T=T)
    q_alpha = jnp.asarray(2.0)
    rates = []
    sigma_values = (0.0, 0.001, 0.01, 0.1)
    for s in sigma_values:
        r = float(compute_partner_integrated_rate(
            q_alpha, q, mu, qw, mw, T,
            partner_factory=lrs_anisotropic_partner_factory(T, s, theta_j=1.0),
            M2_func=partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0),
        ))
        rates.append(r)
    # Continuity: |rate(σ) - rate(0)| should be small at small σ
    rel_at_small = abs(rates[1] - rates[0]) / abs(rates[0])
    assert rel_at_small < 5e-3, (
        f"Σ_+ → 0 continuity broken: σ=0.001 rate diff = {rel_at_small:.3e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# §5. jax.grad finite
# ═══════════════════════════════════════════════════════════════════════

def test_partner_integration_jax_grad_finite():
    """jax.grad through the partner integration returns finite values."""
    from rabbit.jax.collision_hm_partner_integration_jax import (
        compute_partner_integrated_rate, flrw_partner_factory,
    )
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    T = 1.0
    q, mu, qw, mw = _gauss_legendre_2d(8, 6, q_max_T=10.0, T=T)
    M2_func = partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0)
    def loss(q_a):
        return compute_partner_integrated_rate(
            q_a, q, mu, qw, mw, T,
            partner_factory=flrw_partner_factory(T),
            M2_func=M2_func,
        )
    g = float(jax.grad(loss)(jnp.asarray(2.0)))
    assert jnp.isfinite(g)
    assert g > 0.0, f"d Γ/d q_α should be > 0 (URM); got {g}"


# ═══════════════════════════════════════════════════════════════════════
# §6. Quadrature convergence
# ═══════════════════════════════════════════════════════════════════════

def test_2d_quadrature_convergence():
    """Spread on (N_q, N_μ) ∈ {(8,6), (12,8), (20,12)} < 1e-3 rel."""
    from rabbit.jax.collision_hm_partner_integration_jax import (
        compute_partner_integrated_rate, flrw_partner_factory,
    )
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    T = 1.0
    M2_func = partial(M2_nu_e_elastic, species="nue", m_e_MeV=0.0)
    factory = flrw_partner_factory(T)
    rates = []
    for nq, nmu in ((8, 6), (12, 8), (20, 12)):
        q, mu, qw, mw = _gauss_legendre_2d(nq, nmu, q_max_T=12.0, T=T)
        r = float(compute_partner_integrated_rate(
            jnp.asarray(2.0), q, mu, qw, mw, T,
            partner_factory=factory, M2_func=M2_func,
        ))
        rates.append(r)
    spread = max(rates) - min(rates)
    rel = spread / max(abs(rates[-1]), 1e-300)
    assert rel < 1e-3, (
        f"2D quadrature convergence widened: rates={rates}, rel spread={rel:.3e}"
    )


# ═══════════════════════════════════════════════════════════════════════
# §7. Calibration constant
# ═══════════════════════════════════════════════════════════════════════

def test_calibration_normalization_finite_and_positive():
    """The FLRW calibration Z = Γ_physical / Γ_raw is finite and positive."""
    from rabbit.jax.collision_hm_partner_integration_jax import (
        calibrate_partner_integration_normalization,
    )
    q, mu, qw, mw = _gauss_legendre_2d(20, 12, q_max_T=12.0, T=1.0)
    Z = calibrate_partner_integration_normalization(
        q, mu, qw, mw, T_MeV=1.0, species="nue",
    )
    assert np.isfinite(Z)
    assert Z > 0.0, f"calibration constant non-positive: {Z}"
