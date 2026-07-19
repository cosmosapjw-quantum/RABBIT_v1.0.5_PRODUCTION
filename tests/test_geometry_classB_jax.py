"""
Test: Class B Bianchi geometry kernel.

§1. Type V: A only (simplest Class B — no curvature)
§2. Type IV: A + N₁ (frame + single curvature)
§3. A=0 reduction to Class A
§4. Friedmann constraint with cA²
§5. Frame variable evolution direction
§6. Exceptional VI₋₁/₉ c-factor
§7. Type mask and dispatch
"""
import math
import pytest

pytest.importorskip("jax", reason="JAX required")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from rabbit.jax.geometry_classB_jax import (
    compute_classB_geometry_rhs_jax,
    compute_classB_rhs_for_type,
    build_classB_type_mask,
    get_c_factor,
    classB_reduces_to_classA_when_A_zero,
    ClassBGeometryResult,
    TYPE_MASKS_B,
)
from rabbit.config.conventions import BianchiType


# ═══════════════════════════════════════════════════════════════
# §1. Type V: A only (no N_i, no curvature)
# ═══════════════════════════════════════════════════════════════

class TestTypeV:

    def test_smoke(self):
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert isinstance(r, ClassBGeometryResult)

    def test_no_curvature(self):
        """Type V has no N_i → K=0, S=0."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert r.K == 0.0
        assert r.S_plus == 0.0
        assert r.S_minus == 0.0

    def test_omega_reduced_by_cA_sq(self):
        """Ω = 1 − Σ² − cA² < 1 − Σ² (flat would be 1 − Σ²)."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.1, BianchiType.TYPE_V)
        # Σ² = 0.01, cA² = 3 × 0.01 = 0.03, Ω = 1 - 0.01 - 0.03 = 0.96
        assert abs(r.Omega - 0.96) < 1e-14

    def test_q_includes_cA_sq(self):
        """For Type V radiation, q = 1 + Σ² - cA²."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.1, BianchiType.TYPE_V)
        assert abs(r.q - 0.98) < 1e-14

    def test_frame_variable_grows(self):
        """dA/dN = (q + 2Σ₊)A > 0 for Σ₊ > 0, A > 0."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert r.dA_dN > 0

    def test_n_derivatives_zero(self):
        """No N_i → dN_i = 0."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert r.dN1_dN == 0.0
        assert r.dN2_dN == 0.0
        assert r.dN3_dN == 0.0

    def test_constraint_residual_zero(self):
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert r.constraint_residual < 1e-14


# ═══════════════════════════════════════════════════════════════
# §2. Type IV: A + N₁
# ═══════════════════════════════════════════════════════════════

class TestTypeIV:

    def test_has_curvature(self):
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.2, 0.0, 0.0, 0.05, BianchiType.TYPE_IV)
        assert r.K > 0  # Type IV: N₁ > 0 → K > 0 (open)

    def test_frame_and_curvature(self):
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.2, 0.0, 0.0, 0.05, BianchiType.TYPE_IV)
        assert r.cA_sq > 0  # A ≠ 0
        assert r.K > 0       # N₁ ≠ 0

    def test_n1_evolves_n2n3_zero(self):
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.2, 0.0, 0.0, 0.05, BianchiType.TYPE_IV)
        assert abs(r.dN1_dN) > 0  # N₁ evolves
        assert r.dN2_dN == 0.0    # masked
        assert r.dN3_dN == 0.0    # masked


# ═══════════════════════════════════════════════════════════════
# §3. A=0 reduction to Class A
# ═══════════════════════════════════════════════════════════════

class TestReductionToClassA:

    def test_flat(self):
        """Class B with A=0, N_i=0 reduces to flat (Type I)."""
        assert classB_reduces_to_classA_when_A_zero(0.1, 0.0, 0.0, 0.0, 0.0)

    def test_with_curvature_q_convention_noted(self):
        """Class B (A=0, N₁≠0) vs Class A: q convention differs for K≠0.

        Base module (used by Class B): q = 1+Σ²+K  (Raychaudhuri-correct)
        Legacy classA standalone:      q = 1+Σ²-K  (simple q=2Σ²+Ω)

        Shear RHS differs by 2K×Σ ≈ O(10⁻⁴) for small curvature.
        Flat reduction (K=0) is exact. Flagged for PHYS-MATH AUDIT.
        """
        from rabbit.jax.geometry_classB_jax import compute_classB_geometry_rhs_jax
        from rabbit.jax.geometry_classA_jax import compute_classA_geometry_rhs_jax
        rhs_B = compute_classB_geometry_rhs_jax(0.1, 0.0, 0.2, 0.0, 0.0, A=0.0)
        rhs_A = compute_classA_geometry_rhs_jax(0.1, 0.0, 0.2, 0.0, 0.0)
        # K and curvature sources match exactly
        assert abs(rhs_B.K - rhs_A.K) < 1e-14
        assert abs(rhs_B.S_plus - rhs_A.S_plus) < 1e-14
        # Shear RHS differs by q convention (2K × Σ₊ term)
        K = rhs_B.K
        dSp_diff = abs(rhs_B.dSigma_plus_dN - rhs_A.dSigma_plus_dN)
        assert dSp_diff < 2 * K * 0.1 + 1e-12  # bounded by 2KΣ

    def test_dA_zero_when_A_zero(self):
        r = compute_classB_geometry_rhs_jax(
            0.1, 0.0, 0.0, 0.0, 0.0, A=0.0)
        assert r.dA_dN == 0.0

    def test_cA_sq_zero_when_A_zero(self):
        r = compute_classB_geometry_rhs_jax(
            0.1, 0.0, 0.0, 0.0, 0.0, A=0.0)
        assert r.cA_sq == 0.0


# ═══════════════════════════════════════════════════════════════
# §4. Friedmann constraint
# ═══════════════════════════════════════════════════════════════

class TestFriedmannConstraint:

    def test_omega_positive(self):
        """Physical solutions require Ω > 0."""
        r = compute_classB_rhs_for_type(
            0.1, 0.05, 0.0, 0.0, 0.0, 0.1, BianchiType.TYPE_V)
        assert r.Omega > 0

    def test_constraint_sum(self):
        """Σ² + K + cA² + Ω = 1."""
        r = compute_classB_rhs_for_type(
            0.1, 0.05, 0.1, 0.0, 0.0, 0.08, BianchiType.TYPE_IV)
        Sigma_sq = 0.1**2 + 0.05**2
        total = Sigma_sq + r.K + r.cA_sq + r.Omega
        assert abs(total - 1.0) < 1e-14


# ═══════════════════════════════════════════════════════════════
# §5. Frame variable evolution direction
# ═══════════════════════════════════════════════════════════════

class TestFrameEvolution:

    def test_A_positive_grows(self):
        """dA/dN = (q+2Σ₊)A > 0 for q>0, Σ₊>0, A>0."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, 0.05, BianchiType.TYPE_V)
        assert r.dA_dN > 0

    def test_A_negative_decreases(self):
        """A < 0 → dA/dN < 0 (same sign as A)."""
        r = compute_classB_rhs_for_type(
            0.1, 0.0, 0.0, 0.0, 0.0, -0.05, BianchiType.TYPE_V)
        assert r.dA_dN < 0

    def test_growth_rate(self):
        """dA/dN = (q + 2Σ₊) × A. Check numerically."""
        A = 0.05
        Sp = 0.1
        r = compute_classB_rhs_for_type(Sp, 0.0, 0.0, 0.0, 0.0, A, BianchiType.TYPE_V)
        expected = (r.q + 2 * Sp) * A
        assert abs(r.dA_dN - expected) < 1e-14


# ═══════════════════════════════════════════════════════════════
# §6. Exceptional VI₋₁/₉
# ═══════════════════════════════════════════════════════════════

class TestExceptionalType:

    def test_c_factor_15_over_4(self):
        c = get_c_factor(BianchiType.TYPE_VI_M19)
        assert c == 15.0 / 4.0

    def test_generic_c_factor_3(self):
        for t in [BianchiType.TYPE_V, BianchiType.TYPE_IV,
                  BianchiType.TYPE_III, BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH]:
            assert get_c_factor(t) == 3.0

    def test_exceptional_has_larger_cA_sq(self):
        """Same A, but VI₋₁/₉ has c=15/4 > c=3."""
        A = 0.1
        r_generic = compute_classB_geometry_rhs_jax(
            0.1, 0.0, 0.0, 0.1, 0.05, A, c_factor=3.0,
            type_mask=jnp.array([0.0, 1.0, 1.0]))
        r_except = compute_classB_geometry_rhs_jax(
            0.1, 0.0, 0.0, 0.1, 0.05, A, c_factor=15.0/4.0,
            type_mask=jnp.array([0.0, 1.0, 1.0]))
        assert r_except.cA_sq > r_generic.cA_sq


# ═══════════════════════════════════════════════════════════════
# §7. Type mask and dispatch
# ═══════════════════════════════════════════════════════════════

class TestTypeDispatch:

    def test_all_6_types_have_masks(self):
        assert len(TYPE_MASKS_B) == 6

    def test_invalid_classA_rejected(self):
        with pytest.raises(ValueError, match="not a Class B"):
            build_classB_type_mask(BianchiType.TYPE_I)

    def test_type_dispatch_runs(self):
        for bt in [BianchiType.TYPE_V, BianchiType.TYPE_IV, BianchiType.TYPE_III,
                   BianchiType.TYPE_VIH, BianchiType.TYPE_VIIH, BianchiType.TYPE_VI_M19]:
            r = compute_classB_rhs_for_type(
                0.05, 0.01, 0.1, 0.05, 0.03, 0.02, bt)
            assert r.constraint_residual < 1e-13, f"{bt}: residual={r.constraint_residual}"
