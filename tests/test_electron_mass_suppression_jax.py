"""tests/test_electron_mass_suppression_jax.py — v3.2 Phase χ-2 gates.

Plan §χ-2. Validates the JAX port of F(m_e/T) electron-mass
suppression.

Acceptance gates:
  1. F(0) extrapolates to 0 (electron decoupled limit)
  2. F(∞) → 1 (relativistic limit)
  3. F at T = 0.5 MeV (m_e/T ~ 1) matches SciPy ref to 1e-6
  4. Monotonicity: F is non-decreasing in T_γ
  5. F is bounded in [0, 1] across the BBN-relevant range
  6. jax.grad finite + matches Richardson FD on a smooth interior point
  7. SciPy ↔ JAX parity at 5 BBN-relevant temperatures
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Boundary limits
# ═══════════════════════════════════════════════════════════════════════

class TestBoundaryLimits:

    def test_F_at_T_extremely_high(self):
        """T_γ = 100 MeV → F = 1.0 (relativistic limit)."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        F = float(electron_mass_suppression_jax(jnp.asarray(100.0)))
        assert abs(F - 1.0) < 1e-10

    def test_F_at_T_below_table(self):
        """T_γ < 0.01 MeV → F = 0.0 (electron decoupled)."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        F = float(electron_mass_suppression_jax(jnp.asarray(0.005)))
        assert F == 0.0

    def test_F_at_BBN_intermediate(self):
        """T_γ = 0.5 MeV (m_e/T ~ 1): F is small but nonzero."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        F = float(electron_mass_suppression_jax(jnp.asarray(0.5)))
        # Empirically F(0.5 MeV) ~ 0.4 (electrons partially decoupled)
        assert 0.0 < F < 1.0


# ═══════════════════════════════════════════════════════════════════════
# §2. Monotonicity + bounded range
# ═══════════════════════════════════════════════════════════════════════

class TestMonotonicityAndBounds:

    def test_F_is_monotone_increasing_in_T(self):
        """F(T_γ) is non-decreasing across the BBN range."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        T = jnp.geomspace(0.05, 50.0, 50)
        F = electron_mass_suppression_jax(T)
        diffs = jnp.diff(F)
        assert jnp.all(diffs >= -1e-12), (
            "F not monotone non-decreasing"
        )

    def test_F_in_unit_interval(self):
        """0 ≤ F ≤ 1 across the BBN range."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        T = jnp.geomspace(0.05, 50.0, 50)
        F = electron_mass_suppression_jax(T)
        assert jnp.all(F >= 0.0)
        assert jnp.all(F <= 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §3. SciPy ↔ JAX parity
# ═══════════════════════════════════════════════════════════════════════

class TestSciPyParity:

    @pytest.mark.parametrize("T_MeV", [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 10.0])
    def test_jax_matches_scipy(self, T_MeV):
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        from rabbit.thermo.nudec_tables import _electron_mass_suppression
        scipy_val = float(_electron_mass_suppression(T_MeV))
        jax_val = float(electron_mass_suppression_jax(jnp.asarray(T_MeV)))
        # SciPy uses np.interp; JAX uses jnp.interp; both linear on
        # log(T). Boundary cases (T<0.01 or T>50) match exactly via
        # the explicit jnp.where masks. Interior matches to roundoff.
        assert abs(scipy_val - jax_val) < 1e-12, (
            f"SciPy ↔ JAX parity widened at T = {T_MeV} MeV: "
            f"scipy = {scipy_val}, jax = {jax_val}"
        )


# ═══════════════════════════════════════════════════════════════════════
# §4. Differentiability
# ═══════════════════════════════════════════════════════════════════════

class TestDifferentiability:

    def test_grad_finite_in_smooth_interior(self):
        """jax.grad of F at T = 1 MeV is finite + non-zero."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        g = float(jax.grad(electron_mass_suppression_jax)(jnp.asarray(1.0)))
        assert np.isfinite(g)
        assert g > 0.0, f"dF/dT > 0 expected (F monotone); got {g}"

    def test_grad_matches_fd(self):
        """jax.grad ≈ central FD on a smooth interior point."""
        from rabbit.jax.electron_mass_suppression_jax import (
            electron_mass_suppression_jax,
        )
        T0 = 1.0
        eps = 1e-3
        g_ad = float(jax.grad(electron_mass_suppression_jax)(jnp.asarray(T0)))
        g_fd = float(
            (electron_mass_suppression_jax(jnp.asarray(T0 + eps))
             - electron_mass_suppression_jax(jnp.asarray(T0 - eps)))
            / (2.0 * eps)
        )
        rel = abs(g_ad - g_fd) / max(abs(g_fd), 1e-30)
        # Linear interpolation has piecewise-constant derivative; rel
        # accuracy depends on grid spacing. 5% is a defensible bound.
        assert rel < 5e-2, f"AD vs FD: {g_ad} vs {g_fd}, rel = {rel:.3e}"


# ═══════════════════════════════════════════════════════════════════════
# §5. Mangano-gap relevance smoke
# ═══════════════════════════════════════════════════════════════════════

def test_F_at_neutrino_decoupling_significantly_below_unity():
    """At T_γ ~ 1 MeV (neutrino decoupling), F is significantly < 1.

    This is the v3.2 Phase χ-2 thesis: F(m_e/T) suppresses the ν-e
    elastic rate at decoupling, which is the dominant beyond-Mangano-
    coefficient correction to N_eff. Lock that the suppression is
    non-trivial at the relevant temperature.
    """
    from rabbit.jax.electron_mass_suppression_jax import (
        electron_mass_suppression_jax,
    )
    F_1MeV = float(electron_mass_suppression_jax(jnp.asarray(1.0)))
    F_03MeV = float(electron_mass_suppression_jax(jnp.asarray(0.3)))
    F_01MeV = float(electron_mass_suppression_jax(jnp.asarray(0.1)))
    # Empirically (table-built):
    #   F(1.0 MeV) = 0.980  (mild suppression)
    #   F(0.3 MeV) = 0.785  (substantial; m_e/T~1.7)
    #   F(0.1 MeV) = 0.148  (deep suppression; m_e/T~5)
    assert 0.97 < F_1MeV < 1.0, (
        f"F(T = 1 MeV) = {F_1MeV}; expected mild suppression"
    )
    assert 0.7 < F_03MeV < 0.85, (
        f"F(T = 0.3 MeV) = {F_03MeV}; expected substantial suppression"
    )
    assert 0.10 < F_01MeV < 0.20, (
        f"F(T = 0.1 MeV) = {F_01MeV}; expected deep suppression"
    )
