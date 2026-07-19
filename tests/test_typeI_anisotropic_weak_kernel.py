"""tests/test_typeI_anisotropic_weak_kernel.py — Phase γ-1 gate.

Validates the leading-order Σ_+-coupled CL3 angular weak-rate kernel
implemented in :mod:`rabbit.weak.sigma_plus_kernel`. Three gate criteria
mirror the Plan §2.1 acceptance tests:

1. Null recovery: ``Σ_+ = 0`` returns the exact baseline (factor = 1.0).
2. CL gating: ``CL < 3`` returns the exact baseline regardless of Σ_+.
3. Finite-Σ signal: ``Σ_+ != 0`` and CL3 returns a non-trivial factor
   with the correct sign (Pitrou+ 2018 eq. 4.31; arXiv:2502.20893).
4. NumPy / JAX parity: numpy and JAX implementations agree to 1e-12.
5. Range guard: ``|Σ_+| > 0.75`` raises (legacy runtime guard; not a
   validated physical domain).

A driver-integrated test that propagates this correction to a final
Y_p / D/H is deferred to Phase γ-2 (where the canonical SciPy and JAX
forward solvers will accept ``sigma_plus`` as a kwarg to
``compute_live_weak_rates`` and apply the multiplier).
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.weak.sigma_plus_kernel import (
    compute_kappa2,
    compute_kappa2_pair_from_quadrupole_profiles,
    compute_kappa2_from_quadrupole_profile,
    sigma_plus_K2_correction_factor,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Test fixtures: canonical Gauss-Laguerre grid + Fermi-Dirac monopole
# ═══════════════════════════════════════════════════════════════════════

def _gl_nodes_and_fd(N_q: int = 20, T_nu_MeV: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Standard Gauss-Laguerre nodes plus equilibrium FD monopole."""
    from numpy.polynomial.laguerre import laggauss
    q_nodes, _ = laggauss(N_q)
    f_fd = 1.0 / (1.0 + np.exp(q_nodes))  # Equilibrium FD on q = E_nu / T_nu
    return q_nodes, f_fd


# ═══════════════════════════════════════════════════════════════════════
# §2. Acceptance gates
# ═══════════════════════════════════════════════════════════════════════

class TestSigmaPlusKernelNullRecovery:
    """Σ_+ = 0 limit must recover baseline exactly."""

    def test_factor_at_sigma_zero(self):
        q, f = _gl_nodes_and_fd()
        factor = sigma_plus_K2_correction_factor(
            sigma_plus=0.0, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
            correction_level=3,
        )
        assert factor == 1.0

    def test_factor_at_cl_below_3(self):
        q, f = _gl_nodes_and_fd()
        for cl in (0, 1, 2):
            factor = sigma_plus_K2_correction_factor(
                sigma_plus=0.1, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
                correction_level=cl,
            )
            assert factor == 1.0, f"CL={cl} must short-circuit; got factor={factor}"


class TestSigmaPlusKernelFiniteSignal:
    """Finite Σ_+, CL3 must produce a non-trivial physical correction."""

    def test_kappa2_negative(self):
        """K_2 = -(2/3) p_e E_nu / m_N has negative sign (Pitrou eq. 4.31).

        Order of magnitude (T_nu = 1 MeV thermal): with q ~ 3 typical,
        E_nu ~ 6 m_e, E_e ~ Q + E_nu ~ 8.5 m_e, p_e ~ 8.4 m_e,
        kappa_2 ~ -(2/3) * 8.4 * 6 / 1836 ≈ -1.8e-2. So |kappa_2| ~ 0.02.
        """
        q, f = _gl_nodes_and_fd()
        kappa = compute_kappa2(f, q, T_nu_MeV=1.0)
        assert kappa < 0.0, f"kappa_2 expected < 0 (Pitrou sign); got {kappa}"
        # Bracket the expected magnitude conservatively
        assert 1.0e-3 < abs(kappa) < 1.0e-1, (
            f"|kappa_2|={abs(kappa)} outside expected [1e-3, 1e-1]"
        )

    def test_quadrupole_pair_helper_matches_single_profile_helper(self):
        """Paired ν_e / anti-ν_e profile evaluation must preserve CL3 factors."""
        q, f = _gl_nodes_and_fd(N_q=5, T_nu_MeV=0.8)
        nue_profile = 0.012 * np.exp(-q)
        nuebar_profile = -0.007 * np.exp(-q)

        delta_np, delta_pn = compute_kappa2_pair_from_quadrupole_profiles(
            nue_profile,
            f,
            nuebar_profile,
            f,
            q,
            T_nu_MeV=0.8,
        )

        assert delta_np == pytest.approx(
            compute_kappa2_from_quadrupole_profile(nue_profile, f, q, 0.8),
            abs=0.0,
        )
        assert delta_pn == pytest.approx(
            compute_kappa2_from_quadrupole_profile(nuebar_profile, f, q, 0.8),
            abs=0.0,
        )

    def test_quadrupole_pair_helper_rejects_shape_mismatch(self):
        q, f = _gl_nodes_and_fd(N_q=5, T_nu_MeV=0.8)

        with pytest.raises(ValueError, match="share shape"):
            compute_kappa2_pair_from_quadrupole_profiles(
                np.ones(q.size - 1),
                f,
                np.ones_like(q),
                f,
                q,
                T_nu_MeV=0.8,
            )

    def test_factor_linear_in_sigma_plus(self):
        """At leading order, factor = 1 + Σ_+ · κ_2 must be linear in Σ_+."""
        q, f = _gl_nodes_and_fd()
        kappa = compute_kappa2(f, q, T_nu_MeV=1.0)
        for sigma in (0.01, 0.05, 0.1, 0.5):
            factor = sigma_plus_K2_correction_factor(
                sigma_plus=sigma, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
                correction_level=3,
            )
            expected = 1.0 + sigma * kappa
            assert abs(factor - expected) < 1.0e-12

    def test_factor_outside_legacy_guarded_range_raises(self):
        q, f = _gl_nodes_and_fd()
        with pytest.raises(ValueError, match=r"sigma_plus=.*outside legacy guarded"):
            sigma_plus_K2_correction_factor(
                sigma_plus=0.8, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
                correction_level=3,
            )


class TestSigmaPlusKernelJaxParity:
    """NumPy and JAX implementations must agree to numerical precision."""

    def test_jax_matches_numpy_at_sigma_zero(self):
        pytest.importorskip("jax")
        from rabbit.weak.sigma_plus_kernel import sigma_plus_K2_correction_factor_jax

        q, f = _gl_nodes_and_fd()
        factor_np = sigma_plus_K2_correction_factor(
            sigma_plus=0.0, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
            correction_level=3,
        )
        factor_jax = float(sigma_plus_K2_correction_factor_jax(
            0.0, f, q, 1.0, correction_level=3,
        ))
        assert abs(factor_np - factor_jax) < 1.0e-12

    def test_jax_matches_numpy_at_finite_sigma(self):
        pytest.importorskip("jax")
        from rabbit.weak.sigma_plus_kernel import sigma_plus_K2_correction_factor_jax

        q, f = _gl_nodes_and_fd()
        for sigma in (0.05, 0.10, 0.20):
            factor_np = sigma_plus_K2_correction_factor(
                sigma_plus=sigma, f_nue_monopole=f, q_nodes=q, T_nu_MeV=1.0,
                correction_level=3,
            )
            factor_jax = float(sigma_plus_K2_correction_factor_jax(
                sigma, f, q, 1.0, correction_level=3,
            ))
            assert abs(factor_np - factor_jax) < 1.0e-10, (
                f"sigma={sigma}: np={factor_np}, jax={factor_jax}"
            )

    def test_jax_supports_jax_grad_through_sigma(self):
        """Differentiability gate: jax.grad of the factor wrt sigma_plus is finite."""
        import jax
        import jax.numpy as jnp
        from rabbit.weak.sigma_plus_kernel import sigma_plus_K2_correction_factor_jax

        q, f = _gl_nodes_and_fd()

        def factor(sigma):
            return sigma_plus_K2_correction_factor_jax(
                sigma, f, q, 1.0, correction_level=3,
            )

        g = jax.grad(factor)(0.05)
        assert jnp.isfinite(g)
        # The gradient should equal kappa_2 (constant)
        kappa = compute_kappa2(f, q, T_nu_MeV=1.0)
        assert abs(float(g) - kappa) < 1.0e-10


# ═══════════════════════════════════════════════════════════════════════
# §3. Driver-integration smoke (deferred wire-up — Phase γ-2)
# ═══════════════════════════════════════════════════════════════════════

class TestSigmaPlusFactorThroughComputeLiveWeakRates:
    """Phase γ-2: compute_live_weak_rates now accepts sigma_plus_correction_factor.

    Default 1.0 is no-op (backward compatible). A non-1.0 factor multiplies
    lambda_np and lambda_pn, demonstrating that the API can carry the
    Σ+-coupled CL3 angular correction into the canonical rate computation
    without altering the integrand.
    """

    def _build_inputs(self, N_q: int = 20, T_nu_MeV: float = 1.0):
        from numpy.polynomial.laguerre import laggauss
        q_nodes, _ = laggauss(N_q)
        f_fd = 1.0 / (1.0 + np.exp(q_nodes))
        return q_nodes, f_fd

    def test_default_factor_is_noop(self):
        """factor=1.0 must reproduce the bit-exact baseline."""
        from rabbit.weak.live_rates import compute_live_weak_rates
        q, f = self._build_inputs()
        T_g_MeV = 1.0
        T_nu_MeV = T_g_MeV * (4.0 / 11.0) ** (1.0 / 3.0)
        r1 = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=2,
        )
        r2 = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=2,
            sigma_plus_correction_factor=1.0,
        )
        assert r1.lambda_np == r2.lambda_np
        assert r1.lambda_pn == r2.lambda_pn

    def test_finite_factor_multiplies_lambdas(self):
        """factor=1+ε must scale lambda_np and lambda_pn by 1+ε exactly."""
        from rabbit.weak.live_rates import compute_live_weak_rates
        q, f = self._build_inputs()
        T_g_MeV = 1.0
        T_nu_MeV = T_g_MeV * (4.0 / 11.0) ** (1.0 / 3.0)
        epsilon = 0.001
        r0 = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=2,
        )
        r1 = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=2,
            sigma_plus_correction_factor=1.0 + epsilon,
        )
        # Note: lambda_np has a floor at 1/tau_n (~1.14e-3 s^-1); lambda_pn has 0.
        # We test that the scaled value matches the analytical product, accepting
        # that the floor is the only place where exact multiplicativity may fail.
        if r0.lambda_np > 1.0 / 878.4 * 1.001:
            assert abs(r1.lambda_np / r0.lambda_np - (1.0 + epsilon)) < 1.0e-12
        if r0.lambda_pn > 0.0:
            assert abs(r1.lambda_pn / r0.lambda_pn - (1.0 + epsilon)) < 1.0e-12

    def test_factor_signs_propagate(self):
        """Negative ε (decreasing rate) must shrink lambda_np proportionally."""
        from rabbit.weak.live_rates import compute_live_weak_rates
        q, f = self._build_inputs()
        T_g_MeV = 0.7  # weak rates are non-trivial here
        T_nu_MeV = T_g_MeV * (4.0 / 11.0) ** (1.0 / 3.0)
        r_plus = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=3,
            sigma_plus_correction_factor=1.01,
        )
        r_minus = compute_live_weak_rates(
            f, f, q, T_g_MeV, T_nu_MeV, tau_n=878.4, correction_level=3,
            sigma_plus_correction_factor=0.99,
        )
        assert r_plus.lambda_np > r_minus.lambda_np
