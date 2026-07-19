"""tests/test_hm_matrix_elements.py — v3.1 Phase α-1 acceptance gates.

Plan §α-1. Validates the closed-form |M|² transcriptions in
:mod:`rabbit.jax.hm_matrix_elements_jax`.

Gates:
  1. Non-negativity on a BBN (q, q', μ) grid.
  2. Crossing symmetry: |M|²_νe→νe(s, t, u) = |M|²_νν̄→ee(u, t, s).
  3. Mandelstam closure: URM s + t + u = 0; with m_e: s + t + u = 2 m_e².
  4. ν-ν symmetry: M²_diagonal = 2 · M²_off-diagonal at fixed (s, t, u).
  5. URM reduction: ν-e |M|² at m_e=0 equals the pure-(s²,u²) form.
  6. a_α reproduction: the coupling structure satisfies
     ``4(G_L² + G_R²) = a_α`` for both species.
  7. jax.grad finite + matches Richardson FD over (s, u).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Non-negativity on BBN grid
# ═══════════════════════════════════════════════════════════════════════

class TestNonNegativity:
    """|M|² must be ≥ 0 everywhere on physical kinematics."""

    def _bbn_grid(self, n: int = 20):
        from rabbit.jax.hm_matrix_elements_jax import (
            mandelstam_from_qq_mu_with_me,
        )
        q = jnp.linspace(0.5, 10.0, n)        # MeV
        qp = jnp.linspace(0.5, 10.0, n)
        mu = jnp.linspace(-1.0, 1.0, n)
        Q, QP, MU = jnp.meshgrid(q, qp, mu, indexing="ij")
        return mandelstam_from_qq_mu_with_me(Q.ravel(), QP.ravel(), MU.ravel())

    def test_M2_nu_e_elastic_nue_non_negative(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        s, t, u = self._bbn_grid()
        m2 = M2_nu_e_elastic(s, t, u, species="nue")
        assert jnp.all(m2 >= 0.0), (
            f"M²_νe→νe (nue) negative on BBN grid: min = {float(jnp.min(m2)):.3e}"
        )

    def test_M2_nu_e_elastic_nux_non_negative(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        s, t, u = self._bbn_grid()
        m2 = M2_nu_e_elastic(s, t, u, species="nux")
        assert jnp.all(m2 >= 0.0)

    def test_M2_nu_nubar_to_ee_nue_non_negative(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_nubar_to_ee
        s, t, u = self._bbn_grid()
        m2 = M2_nu_nubar_to_ee(s, t, u, species="nue")
        assert jnp.all(m2 >= 0.0)

    def test_M2_nu_nu_diagonal_non_negative(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_nu_diagonal
        # ν-ν is massless; use the URM Mandelstam helper (s, t, u)
        # but with m_e=0 the same grid works for non-negativity.
        from rabbit.jax.hm_matrix_elements_jax import (
            mandelstam_from_qq_mu_urm,
        )
        q = jnp.linspace(0.5, 10.0, 12)
        qp = jnp.linspace(0.5, 10.0, 12)
        mu = jnp.linspace(-1.0, 1.0, 12)
        Q, QP, MU = jnp.meshgrid(q, qp, mu, indexing="ij")
        s, t, u = mandelstam_from_qq_mu_urm(Q.ravel(), QP.ravel(), MU.ravel())
        m2 = M2_nu_nu_diagonal(s, t, u)
        assert jnp.all(m2 >= 0.0)

    def test_M2_nu_nu_off_diagonal_non_negative(self):
        from rabbit.jax.hm_matrix_elements_jax import (
            M2_nu_nu_off_diagonal, mandelstam_from_qq_mu_urm,
        )
        q = jnp.linspace(0.5, 10.0, 12)
        qp = jnp.linspace(0.5, 10.0, 12)
        mu = jnp.linspace(-1.0, 1.0, 12)
        Q, QP, MU = jnp.meshgrid(q, qp, mu, indexing="ij")
        s, t, u = mandelstam_from_qq_mu_urm(Q.ravel(), QP.ravel(), MU.ravel())
        m2 = M2_nu_nu_off_diagonal(s, t, u)
        assert jnp.all(m2 >= 0.0)


# ═══════════════════════════════════════════════════════════════════════
# §2. Crossing symmetry
# ═══════════════════════════════════════════════════════════════════════

class TestCrossingSymmetry:
    """ν-e elastic ↔ ν-ν̄ annihilation by s ↔ u crossing."""

    def test_nu_e_nu_nubar_crossing(self):
        """|M|²_νe→νe(s, t, u) = |M|²_νν̄→ee(u, t, s)."""
        from rabbit.jax.hm_matrix_elements_jax import (
            M2_nu_e_elastic, M2_nu_nubar_to_ee,
        )
        s = jnp.array([2.0, 5.0, 10.0])
        t = jnp.array([0.0, 0.0, 0.0])
        u = jnp.array([1.5, 4.0, 8.0])
        for species in ("nue", "nux"):
            a = M2_nu_e_elastic(s, t, u, species=species)
            b = M2_nu_nubar_to_ee(u, t, s, species=species)
            rel = float(jnp.max(jnp.abs((a - b) / jnp.maximum(jnp.abs(a), 1e-300))))
            assert rel < 1.0e-12, (
                f"crossing symmetry broken for {species}: rel={rel:.3e}"
            )


# ═══════════════════════════════════════════════════════════════════════
# §3. Mandelstam closure
# ═══════════════════════════════════════════════════════════════════════

class TestMandelstamClosure:
    """s + t + u closure identity."""

    def test_urm_closure_zero(self):
        """In URM (massless): s + t + u = 0."""
        from rabbit.jax.hm_matrix_elements_jax import mandelstam_from_qq_mu_urm
        q = jnp.array([1.0, 2.0, 5.0])
        qp = jnp.array([0.5, 1.5, 4.0])
        mu = jnp.array([-0.5, 0.0, 0.7])
        s, t, u = mandelstam_from_qq_mu_urm(q, qp, mu)
        closure = float(jnp.max(jnp.abs(s + t + u)))
        assert closure < 1e-12, f"URM Mandelstam closure broken: {closure:.3e}"

    def test_with_me_closure_2_me_squared(self):
        """With m_e ≠ 0: s + t + u = 2 m_e² (one massive incoming particle)."""
        from rabbit.jax.hm_matrix_elements_jax import (
            mandelstam_from_qq_mu_with_me, M_E_MEV,
        )
        q = jnp.array([1.0, 2.0, 5.0])
        qp = jnp.array([0.5, 1.5, 4.0])
        mu = jnp.array([-0.5, 0.0, 0.7])
        s, t, u = mandelstam_from_qq_mu_with_me(q, qp, mu)
        expected = 2.0 * M_E_MEV ** 2
        rel = float(jnp.max(jnp.abs((s + t + u - expected) / expected)))
        assert rel < 1e-12, f"closure with m_e widened: rel={rel:.3e}"


# ═══════════════════════════════════════════════════════════════════════
# §4. ν-ν symmetrization factor
# ═══════════════════════════════════════════════════════════════════════

def test_nu_nu_diagonal_is_twice_off_diagonal():
    """|M|²_diagonal = 2 · |M|²_off-diagonal at the same (s, u)."""
    from rabbit.jax.hm_matrix_elements_jax import (
        M2_nu_nu_diagonal, M2_nu_nu_off_diagonal,
    )
    s = jnp.array([2.0, 5.0, 10.0])
    t = jnp.zeros_like(s)
    u = jnp.array([1.5, 4.0, 8.0])
    diag = M2_nu_nu_diagonal(s, t, u)
    offd = M2_nu_nu_off_diagonal(s, t, u)
    rel = float(jnp.max(jnp.abs((diag - 2.0 * offd) / jnp.maximum(jnp.abs(diag), 1e-300))))
    assert rel < 1e-12, f"identical-particle factor broken: rel={rel:.3e}"


# ═══════════════════════════════════════════════════════════════════════
# §5. URM reduction of ν-e elastic
# ═══════════════════════════════════════════════════════════════════════

def test_nu_e_elastic_urm_reduction():
    """At m_e → 0, |M|²_νe = 8 G_F² (G_L² s² + G_R² u²)."""
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    from rabbit.collisions.kernels import G_F_MEV, G_L_NUE, G_R_NUE
    s_val = 5.0
    u_val = -5.0  # URM closure s + u = 0
    s = jnp.asarray(s_val)
    t = jnp.asarray(0.0)
    u = jnp.asarray(u_val)
    got = float(M2_nu_e_elastic(s, t, u, species="nue", m_e_MeV=0.0))
    expected = 8.0 * G_F_MEV ** 2 * (
        G_L_NUE ** 2 * s_val ** 2 + G_R_NUE ** 2 * u_val ** 2
    )
    rel = abs(got - expected) / max(abs(expected), 1e-300)
    assert rel < 1e-12, f"URM reduction widened: rel={rel:.3e}"


# ═══════════════════════════════════════════════════════════════════════
# §6. a_α coefficient
# ═══════════════════════════════════════════════════════════════════════

class TestAlphaCoefficient:
    """Mangano 2005 a_α = 4(G_L² + G_R²) reproduced by the closed form."""

    def test_a_e_matches_published_value(self):
        """a_e ≈ 1 + 4 sin²θ_W + 8 sin⁴θ_W ≈ 2.353."""
        from rabbit.jax.hm_matrix_elements_jax import closed_form_a_alpha
        from rabbit.collisions.kernels import A_TOTAL_NUE
        a = closed_form_a_alpha("nue")
        rel = abs(a - A_TOTAL_NUE) / abs(A_TOTAL_NUE)
        assert rel < 1e-12, f"a_e closed form drift: {a} vs {A_TOTAL_NUE}"

    def test_a_x_matches_published_value(self):
        """a_x ≈ 1 - 4 sin²θ_W + 8 sin⁴θ_W ≈ 0.503."""
        from rabbit.jax.hm_matrix_elements_jax import closed_form_a_alpha
        from rabbit.collisions.kernels import A_TOTAL_NUX
        a = closed_form_a_alpha("nux")
        rel = abs(a - A_TOTAL_NUX) / abs(A_TOTAL_NUX)
        assert rel < 1e-12

    def test_unknown_species_raises(self):
        from rabbit.jax.hm_matrix_elements_jax import closed_form_a_alpha
        with pytest.raises(ValueError, match=r"unknown species"):
            closed_form_a_alpha("bogus")


# ═══════════════════════════════════════════════════════════════════════
# §7. jax.grad finite + matches Richardson FD
# ═══════════════════════════════════════════════════════════════════════

class TestDifferentiability:
    """|M|² is jax.grad-able over Mandelstam invariants."""

    def test_grad_over_s_finite(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        def f(s_scalar):
            s = jnp.asarray(s_scalar)
            t = jnp.asarray(0.0)
            u = jnp.asarray(-s_scalar)
            return M2_nu_e_elastic(s, t, u, species="nue", m_e_MeV=0.0)
        g = float(jax.grad(f)(jnp.asarray(5.0)))
        assert jnp.isfinite(g)
        assert g > 0.0, f"d|M²|/ds should be > 0 in URM; got {g}"

    def test_grad_matches_richardson_fd(self):
        from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
        def f(s_scalar):
            s = jnp.asarray(s_scalar)
            t = jnp.asarray(0.0)
            u = jnp.asarray(-s_scalar)
            return M2_nu_e_elastic(s, t, u, species="nue", m_e_MeV=0.0)
        s0 = 5.0
        g_ad = float(jax.grad(f)(jnp.asarray(s0)))
        eps = 1e-3
        g_fd = (float(f(s0 + eps)) - float(f(s0 - eps))) / (2.0 * eps)
        rel = abs(g_ad - g_fd) / abs(g_fd)
        assert rel < 1e-4, (
            f"AD vs FD disagreement: rel={rel:.3e}, ad={g_ad:.6e}, fd={g_fd:.6e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# §8. Unknown species
# ═══════════════════════════════════════════════════════════════════════

def test_unknown_species_in_M2_nu_e_elastic_raises():
    from rabbit.jax.hm_matrix_elements_jax import M2_nu_e_elastic
    with pytest.raises(ValueError, match=r"unknown species"):
        M2_nu_e_elastic(jnp.array([5.0]), jnp.array([0.0]), jnp.array([-5.0]),
                        species="bogus")
