"""
Test: Class A geometry — official candidate gate tests.

Four gates for jax_classA_geometry candidate promotion:
  G1. Type I reduction: ClassA(I) = TypeI to machine precision
  G2. Collins-Stewart equilibrium: dΣ/dN = 0 at attractor point
  G3. Type II representative: finite curvature, physical Ω, correct K sign
  G4. Constraint residual: Friedmann |1-Σ²-K-Ω| < 1e-12 for all 6 types
  G5. NumPy parity: JAX matches NumPy reference for all 6 types (1e-12)
"""
import math
import pytest

from rabbit.config.conventions import BianchiType
from rabbit.geometry.general_classA import (
    CS_K, CS_N1_SQ, CS_OMEGA, CS_Q, CS_SIGMA_PLUS,
    classA_geometry_rhs,
)
from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.jax.geometry_classA_jax import (
    TYPE_MASKS, build_type_mask,
    compute_classA_geometry_rhs_jax,
    compute_curvature_invariants_jax,
    compute_Omega_classA_jax,
    compute_q_classA_jax,
)


# Representative test points for all 6 types
ALL_TYPE_POINTS = {
    BianchiType.TYPE_I:    (0.1,  -0.05, 0.0,  0.0,   0.0),
    BianchiType.TYPE_II:   (0.08,  0.0,  0.2,  0.0,   0.0),
    BianchiType.TYPE_VI0:  (0.05,  0.02, 0.0,  0.15,  0.1),
    BianchiType.TYPE_VII0: (0.04,  0.01, 0.0,  0.2,   0.2),
    BianchiType.TYPE_VIII: (0.03,  0.01, 0.15, 0.1,  -0.05),
    BianchiType.TYPE_IX:   (0.02,  0.01, 0.08, 0.08,  0.08),
}


# ═══════════════════════════════════════════════════════════════
# G1. Type I reduction
# ═══════════════════════════════════════════════════════════════

class TestGate1TypeIReduction:
    """ClassA with Type I mask = dedicated TypeI geometry to machine precision."""

    @pytest.mark.parametrize("Sp,Sm,pip,pim", [
        (0.09, -0.02, 1.5e-3, -7e-4),
        (0.2, 0.0, 0.0, 0.0),
        (0.0, 0.05, 1e-3, 1e-3),
    ])
    def test_rhs_parity(self, Sp, Sm, pip, pim):
        Omega = 1.0 - Sp**2 - Sm**2
        dSp_ref, dSm_ref = compute_typeI_geometry_rhs(Sp, Sm, pip, pim, Omega)
        out = compute_classA_geometry_rhs_jax(
            Sp, Sm, 0.0, 0.0, 0.0,
            pi_shear_plus=pip, pi_shear_minus=pim,
            type_mask=TYPE_MASKS[BianchiType.TYPE_I])
        assert abs(out.dSigma_plus_dN - dSp_ref) < 1e-12
        assert abs(out.dSigma_minus_dN - dSm_ref) < 1e-12
        assert out.dN1_dN == 0.0
        assert out.dN2_dN == 0.0
        assert out.dN3_dN == 0.0

    def test_type_mask_zeros(self):
        mask = build_type_mask(BianchiType.TYPE_I)
        assert tuple(float(x) for x in mask) == (0.0, 0.0, 0.0)

    def test_flat_curvature(self):
        K, _, _ = compute_curvature_invariants_jax(0.0, 0.0, 0.0)
        assert abs(float(K)) < 1e-15


# ═══════════════════════════════════════════════════════════════
# G2. Collins-Stewart equilibrium (Type II attractor)
# ═══════════════════════════════════════════════════════════════

class TestGate2CollinsStewart:
    """At the CS radiation attractor: dΣ±/dN = 0 exactly."""

    def test_invariants_match_reference(self):
        N1 = math.sqrt(CS_N1_SQ)
        K, S_plus, S_minus = compute_curvature_invariants_jax(N1, 0.0, 0.0)
        Omega = compute_Omega_classA_jax(CS_SIGMA_PLUS, 0.0, N1, 0.0, 0.0)
        q = compute_q_classA_jax(CS_SIGMA_PLUS, 0.0, N1, 0.0, 0.0)
        assert abs(float(K) - CS_K) < 1e-12
        assert abs(float(Omega) - CS_OMEGA) < 1e-12
        assert abs(float(q) - CS_Q) < 1e-12

    def test_equilibrium_rhs_vanishes(self):
        N1 = math.sqrt(CS_N1_SQ)
        rhs = compute_classA_geometry_rhs_jax(
            CS_SIGMA_PLUS, 0.0, N1, 0.0, 0.0,
            type_mask=TYPE_MASKS[BianchiType.TYPE_II])
        assert abs(rhs.dSigma_plus_dN) < 1e-12
        assert abs(rhs.dSigma_minus_dN) < 1e-12

    def test_lrs_sigma_minus_source_zero(self):
        N1 = math.sqrt(CS_N1_SQ)
        _, _, S_minus = compute_curvature_invariants_jax(N1, 0.0, 0.0)
        assert abs(float(S_minus)) < 1e-12


# ═══════════════════════════════════════════════════════════════
# G3. Type II representative (finite curvature physics)
# ═══════════════════════════════════════════════════════════════

class TestGate3TypeIIRepresentative:
    """Type II at a non-equilibrium point: curvature sources are active."""

    def test_nonzero_curvature(self):
        K, S_plus, S_minus = compute_curvature_invariants_jax(0.3, 0.0, 0.0)
        assert abs(float(K)) > 0.001

    def test_omega_positive_and_less_than_one(self):
        Omega = compute_Omega_classA_jax(0.08, 0.0, 0.3, 0.0, 0.0)
        assert 0.0 < float(Omega) < 1.0

    def test_rhs_finite(self):
        rhs = compute_classA_geometry_rhs_jax(
            0.08, 0.0, 0.3, 0.0, 0.0,
            type_mask=TYPE_MASKS[BianchiType.TYPE_II])
        assert math.isfinite(rhs.dSigma_plus_dN)
        assert math.isfinite(rhs.dN1_dN)
        assert rhs.dN2_dN == 0.0  # N2 masked out for Type II

    def test_n1_evolves(self):
        rhs = compute_classA_geometry_rhs_jax(
            0.08, 0.0, 0.3, 0.0, 0.0,
            type_mask=TYPE_MASKS[BianchiType.TYPE_II])
        assert abs(rhs.dN1_dN) > 1e-6  # N1 must evolve


# ═══════════════════════════════════════════════════════════════
# G4. Constraint residual for all 6 types
# ═══════════════════════════════════════════════════════════════

class TestGate4ConstraintResidual:
    """Friedmann |1 - Σ² - K - Ω| < 1e-12 for all 6 types."""

    @pytest.mark.parametrize("btype", list(ALL_TYPE_POINTS.keys()),
                             ids=[b.value for b in ALL_TYPE_POINTS])
    def test_residual(self, btype):
        sp, sm, n1, n2, n3 = ALL_TYPE_POINTS[btype]
        rhs = compute_classA_geometry_rhs_jax(
            sp, sm, n1, n2, n3, type_mask=TYPE_MASKS[btype])
        assert abs(rhs.constraint_residual) < 1e-12

    @pytest.mark.parametrize("btype", list(ALL_TYPE_POINTS.keys()),
                             ids=[b.value for b in ALL_TYPE_POINTS])
    def test_omega_positive(self, btype):
        sp, sm, n1, n2, n3 = ALL_TYPE_POINTS[btype]
        rhs = compute_classA_geometry_rhs_jax(
            sp, sm, n1, n2, n3, type_mask=TYPE_MASKS[btype])
        assert rhs.Omega > 0


# ═══════════════════════════════════════════════════════════════
# G5. JAX ↔ NumPy parity for all 6 types
# ═══════════════════════════════════════════════════════════════

class TestGate5NumPyParity:
    """JAX and NumPy give identical RHS for all 6 types."""

    @pytest.mark.parametrize("btype", list(ALL_TYPE_POINTS.keys()),
                             ids=[b.value for b in ALL_TYPE_POINTS])
    def test_rhs_parity(self, btype):
        sp, sm, n1, n2, n3 = ALL_TYPE_POINTS[btype]
        dSp_np, dSm_np, dN1_np, dN2_np, dN3_np = classA_geometry_rhs(
            sp, sm, n1, n2, n3)
        rhs = compute_classA_geometry_rhs_jax(
            sp, sm, n1, n2, n3, type_mask=TYPE_MASKS[btype])
        assert abs(rhs.dSigma_plus_dN - dSp_np) < 1e-12
        assert abs(rhs.dSigma_minus_dN - dSm_np) < 1e-12
