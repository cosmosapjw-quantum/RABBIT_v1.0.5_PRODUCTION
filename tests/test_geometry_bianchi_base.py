"""
Test: geometry_bianchi_base — common Bianchi geometry interface.

Validates the shared algebra layer that Class A and future Class B use.
§1. Curvature invariants (K, S₊, S₋) match Class A reference
§2. q, Ω, Friedmann constraint
§3. Shear RHS matches Class A reference
§4. Structure constant eigenvalues and evolution
§5. Class B extension points (frame variable, c-factor)
§6. Type mask registry completeness
§7. Backward compatibility: Class A refactored code unchanged
"""
import math
import pytest

pytest.importorskip("jax", reason="JAX required")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════
# §1. Curvature invariants
# ═══════════════════════════════════════════════════════════════

class TestCurvatureInvariants:

    def test_flat_gives_zero(self):
        from rabbit.jax.geometry_bianchi_base import curvature_invariants
        K, Sp, Sm = curvature_invariants(0.0, 0.0, 0.0)
        assert float(K) == 0.0
        assert float(Sp) == 0.0
        assert float(Sm) == 0.0

    def test_typeII_K_positive(self):
        from rabbit.jax.geometry_bianchi_base import gauss_curvature_K
        K = gauss_curvature_K(jnp.array(0.5), jnp.array(0.0), jnp.array(0.0))
        assert float(K) > 0  # Type II: open

    def test_typeIX_K_negative(self):
        from rabbit.jax.geometry_bianchi_base import gauss_curvature_K
        K = gauss_curvature_K(jnp.array(0.1), jnp.array(0.1), jnp.array(0.1))
        assert float(K) < 0  # Type IX: closed

    def test_matches_classA_reference(self):
        """Base module K,S must match geometry_classA_jax exactly."""
        from rabbit.jax.geometry_bianchi_base import curvature_invariants
        from rabbit.jax.geometry_classA_jax import compute_curvature_invariants_jax
        for N1, N2, N3 in [(0.5, 0.0, 0.0), (0.0, 0.2, 0.1), (0.1, 0.1, 0.1)]:
            K_base, Sp_base, Sm_base = curvature_invariants(
                jnp.array(N1), jnp.array(N2), jnp.array(N3))
            K_ref, Sp_ref, Sm_ref = compute_curvature_invariants_jax(N1, N2, N3)
            assert abs(float(K_base - K_ref)) < 1e-15
            assert abs(float(Sp_base - Sp_ref)) < 1e-15
            assert abs(float(Sm_base - Sm_ref)) < 1e-15


# ═══════════════════════════════════════════════════════════════
# §2. q and Ω
# ═══════════════════════════════════════════════════════════════

class TestDeceleration:

    def test_flat_q_equals_one_plus_sigma_sq(self):
        from rabbit.jax.geometry_bianchi_base import compute_q
        q = compute_q(jnp.array(0.04), jnp.array(0.0))
        assert abs(float(q) - 1.04) < 1e-15

    def test_omega_flat(self):
        from rabbit.jax.geometry_bianchi_base import compute_Omega
        Om = compute_Omega(jnp.array(0.04), jnp.array(0.0))
        assert abs(float(Om) - 0.96) < 1e-15

    def test_classB_q_with_frame_variable(self):
        """For radiation, q = 1 + Σ² - K - cA² for Class B."""
        from rabbit.jax.geometry_bianchi_base import compute_q
        q = compute_q(jnp.array(0.04), jnp.array(0.01), cA_sq=jnp.array(0.03))
        assert abs(float(q) - 1.0) < 1e-15

    def test_classB_omega_with_frame_variable(self):
        from rabbit.jax.geometry_bianchi_base import compute_Omega
        Om = compute_Omega(jnp.array(0.04), jnp.array(0.01), cA_sq=jnp.array(0.03))
        assert abs(float(Om) - 0.92) < 1e-15

    def test_friedmann_residual_zero_when_consistent(self):
        from rabbit.jax.geometry_bianchi_base import compute_Omega, friedmann_residual
        Sigma_sq = jnp.array(0.04)
        K = jnp.array(0.01)
        Om = compute_Omega(Sigma_sq, K)
        res = friedmann_residual(Sigma_sq, K, Om)
        assert float(res) < 1e-15


# ═══════════════════════════════════════════════════════════════
# §3. Shear RHS
# ═══════════════════════════════════════════════════════════════

class TestShearRHS:

    def test_flat_damping(self):
        """In flat limit (q≈1): dΣ/dN ≈ -Σ (exponential decay)."""
        from rabbit.jax.geometry_bianchi_base import shear_rhs
        dSp, dSm = shear_rhs(jnp.array(0.1), jnp.array(0.0),
                              jnp.array(1.0), jnp.array(0.0), jnp.array(0.0))
        assert abs(float(dSp) + 0.1) < 1e-15  # -(2-1)*0.1 = -0.1

    def test_curvature_source_enters(self):
        from rabbit.jax.geometry_bianchi_base import shear_rhs
        dSp1, _ = shear_rhs(jnp.array(0.1), jnp.array(0.0),
                             jnp.array(1.0), jnp.array(0.0), jnp.array(0.0))
        dSp2, _ = shear_rhs(jnp.array(0.1), jnp.array(0.0),
                             jnp.array(1.0), jnp.array(0.05), jnp.array(0.0))
        assert float(dSp2) < float(dSp1)  # S₊ > 0 makes dΣ more negative

    def test_stress_backreaction(self):
        from rabbit.jax.geometry_bianchi_base import shear_rhs
        dSp1, _ = shear_rhs(jnp.array(0.1), jnp.array(0.0),
                             jnp.array(1.0), jnp.array(0.0), jnp.array(0.0))
        dSp2, _ = shear_rhs(jnp.array(0.1), jnp.array(0.0),
                             jnp.array(1.0), jnp.array(0.0), jnp.array(0.0),
                             Pi_plus=jnp.array(0.02))
        assert float(dSp2) > float(dSp1)  # Π₊ > 0 sustains shear


# ═══════════════════════════════════════════════════════════════
# §4. Structure constant evolution
# ═══════════════════════════════════════════════════════════════

class TestStructureEvolution:

    def test_eigenvalue_sum(self):
        """λ₁ + λ₂ + λ₃ = 3q (trace of q·I + 2Σ = 3q since tr(Σ)=0)."""
        from rabbit.jax.geometry_bianchi_base import structure_eigenvalues
        l1, l2, l3 = structure_eigenvalues(jnp.array(1.1), jnp.array(0.2), jnp.array(0.05))
        assert abs(float(l1 + l2 + l3) - 3 * 1.1) < 1e-14

    def test_masked_type_gives_zero(self):
        from rabbit.jax.geometry_bianchi_base import structure_rhs
        dN1, dN2, dN3 = structure_rhs(
            jnp.array(0.0), jnp.array(0.2), jnp.array(0.1),  # N₁=0 (masked)
            jnp.array(1.0), jnp.array(0.1), jnp.array(0.0))
        assert float(dN1) == 0.0  # N₁=0 → dN₁=0


# ═══════════════════════════════════════════════════════════════
# §5. Class B extensions
# ═══════════════════════════════════════════════════════════════

class TestClassBExtensions:

    def test_frame_variable_rhs(self):
        from rabbit.jax.geometry_bianchi_base import frame_variable_rhs
        dA = frame_variable_rhs(jnp.array(0.1), jnp.array(1.0), jnp.array(0.2))
        assert abs(float(dA) - 0.1 * (1.0 + 0.4)) < 1e-15

    def test_c_factor_generic(self):
        from rabbit.jax.geometry_bianchi_base import classB_c_factor
        assert classB_c_factor("TYPE_V") == 3.0
        assert classB_c_factor("TYPE_VIH") == 3.0

    def test_c_factor_exceptional(self):
        from rabbit.jax.geometry_bianchi_base import classB_c_factor
        assert classB_c_factor("TYPE_VI_M19") == 15.0 / 4.0


# ═══════════════════════════════════════════════════════════════
# §6. Type mask registry
# ═══════════════════════════════════════════════════════════════

class TestTypeMaskRegistry:

    def test_all_classA_present(self):
        from rabbit.jax.geometry_bianchi_base import get_type_mask, CLASS_A_TYPES
        for t in CLASS_A_TYPES:
            mask = get_type_mask(t)
            assert len(mask) == 3

    def test_all_classB_present(self):
        from rabbit.jax.geometry_bianchi_base import get_type_mask, CLASS_B_TYPES
        for t in CLASS_B_TYPES:
            mask = get_type_mask(t)
            assert len(mask) == 3

    def test_typeI_all_zero(self):
        from rabbit.jax.geometry_bianchi_base import get_type_mask
        assert get_type_mask("TYPE_I") == (0.0, 0.0, 0.0)

    def test_typeV_all_zero_ni(self):
        """Type V has A only, no N_i."""
        from rabbit.jax.geometry_bianchi_base import get_type_mask
        assert get_type_mask("TYPE_V") == (0.0, 0.0, 0.0)

    def test_class_identification(self):
        from rabbit.jax.geometry_bianchi_base import is_class_A, is_class_B, has_frame_variable
        assert is_class_A("TYPE_I") and not is_class_B("TYPE_I")
        assert is_class_B("TYPE_V") and not is_class_A("TYPE_V")
        assert not has_frame_variable("TYPE_II")
        assert has_frame_variable("TYPE_V")
        assert has_frame_variable("TYPE_VIH")

    def test_total_12_types(self):
        from rabbit.jax.geometry_bianchi_base import TYPE_MASK_REGISTRY
        assert len(TYPE_MASK_REGISTRY) == 12


# ═══════════════════════════════════════════════════════════════
# §7. Backward compatibility: Class A geometry tests
# ═══════════════════════════════════════════════════════════════

class TestClassABackwardCompat:
    """Ensure refactored Class A module still passes basic checks."""

    def test_type_masks_unchanged(self):
        from rabbit.jax.geometry_classA_jax import TYPE_MASKS, build_type_mask
        from rabbit.config.conventions import BianchiType
        assert float(TYPE_MASKS[BianchiType.TYPE_I].sum()) == 0.0
        assert float(TYPE_MASKS[BianchiType.TYPE_II].sum()) == 1.0
        m = build_type_mask(BianchiType.TYPE_IX)
        assert float(m.sum()) == 3.0

    def test_curvature_function_exists(self):
        from rabbit.jax.geometry_classA_jax import compute_curvature_invariants_jax
        K, Sp, Sm = compute_curvature_invariants_jax(0.5, 0.0, 0.0)
        assert float(K) > 0

    def test_geometry_rhs_exists(self):
        from rabbit.jax.geometry_classA_jax import compute_classA_geometry_rhs_jax
        r = compute_classA_geometry_rhs_jax(0.1, 0.0, 0.0, 0.0, 0.0)
        assert hasattr(r, 'dSigma_plus_dN')

    def test_bianchi_type_enum_has_classB(self):
        from rabbit.config.conventions import BianchiType
        assert hasattr(BianchiType, 'TYPE_V')
        assert hasattr(BianchiType, 'TYPE_VIH')
        assert hasattr(BianchiType, 'TYPE_VI_M19')
