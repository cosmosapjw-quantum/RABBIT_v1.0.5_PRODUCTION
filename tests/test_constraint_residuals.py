"""Foundation F0 — Hamiltonian + momentum G^0_i constraint residuals.

Asserts:
  * Friedmann residuals reduce correctly across Type I -> Class A -> Class B.
  * General-dispatcher matches the per-class implementations.
  * Momentum residual is identically zero for orthogonal models.
  * Momentum residual responds correctly to a tilt v_i in the matter source.
  * Class B Friedmann picks up the c A^2 term and the c=15/4 exception.
"""
from __future__ import annotations

import math

import pytest

from rabbit.geometry.constraints import (
    GAMMA_RAD_DEFAULT,
    curvature_K_classA,
    friedmann_residual_classA,
    friedmann_residual_classB,
    friedmann_residual_general,
    friedmann_residual_typeI,
    momentum_geometry_source_classA,
    momentum_geometry_source_classB,
    momentum_matter_source,
    required_momentum_dipole,
    momentum_residual,
)


@pytest.mark.production
class TestFriedmannResiduals:
    def test_typeI_at_flrw_is_zero(self):
        assert friedmann_residual_typeI(0.0, 0.0) == 0.0

    def test_typeI_with_shear_self_consistent(self):
        for s in [0.05, 0.1, 0.2, 0.5, 0.7]:
            assert friedmann_residual_typeI(s, 0.0) < 1e-15
            assert friedmann_residual_typeI(0.0, s) < 1e-15

    def test_classA_zero_curvature_reduces_to_typeI(self):
        # N_i = 0 -> K = 0, so Class A residual must equal Type I residual at the same Omega.
        Omega = 1.0 - 0.05 ** 2
        r_typeI = friedmann_residual_typeI(0.05, 0.0)
        r_classA = friedmann_residual_classA(0.05, 0.0, 0.0, 0.0, 0.0, Omega)
        assert math.isclose(r_typeI, r_classA, abs_tol=1e-15)

    def test_classA_typeIX_curvature_picks_up_K(self):
        # All N_i nonzero (Type IX seed): K positive.
        K = curvature_K_classA(0.1, 0.1, 0.1)
        # K = (1/12)(3*0.01 - 6*0.01) = (1/12)(-0.03) = -0.0025 (Type IX seed gives negative K
        # because all-equal triplet is the closed mode).  We just check the formula.
        expected = (1.0 / 12.0) * (3 * 0.01 - 6 * 0.01)
        assert math.isclose(K, expected, abs_tol=1e-15)

    def test_classB_zero_A_reduces_to_classA(self):
        Omega = 0.7
        r_A = friedmann_residual_classA(0.1, 0.05, 0.0, 0.2, 0.2, Omega)
        r_B = friedmann_residual_classB(0.1, 0.05, 0.0, 0.2, 0.2, A=0.0, Omega=Omega, c_factor=3.0)
        assert math.isclose(r_A, r_B, abs_tol=1e-15)

    def test_classB_picks_up_c_A_squared(self):
        # With A nonzero, residual must include c*A^2.
        Sp, Sm = 0.0, 0.0
        N1, N2, N3 = 0.0, 0.0, 0.0
        Omega = 1.0  # baseline
        A = 0.1
        # Friedmann is now |1 - 1 - 0 - 0 - c*0.01| = c*0.01
        r3 = friedmann_residual_classB(Sp, Sm, N1, N2, N3, A, Omega, c_factor=3.0)
        r15_4 = friedmann_residual_classB(Sp, Sm, N1, N2, N3, A, Omega, c_factor=15.0 / 4.0)
        assert math.isclose(r3, 0.03, abs_tol=1e-15)
        assert math.isclose(r15_4, 0.0375, abs_tol=1e-15)

    def test_general_dispatcher_matches_classA(self):
        Omega = 0.85
        r = friedmann_residual_general(0.1, 0.0, 0.0, 0.0, 0.0, A=0.0,
                                        Omega=Omega, bianchi_type_str="TYPE_I")
        ref = friedmann_residual_classA(0.1, 0.0, 0.0, 0.0, 0.0, Omega)
        assert math.isclose(r, ref, abs_tol=1e-15)

    def test_general_dispatcher_matches_classB_VI_m19(self):
        Omega = 0.7
        r = friedmann_residual_general(0.05, 0.05, 0.0, 0.1, 0.1, A=0.05,
                                        Omega=Omega, bianchi_type_str="TYPE_VI_M19")
        # VI_{-1/9} uses c = 15/4.
        ref = friedmann_residual_classB(0.05, 0.05, 0.0, 0.1, 0.1, A=0.05,
                                         Omega=Omega, c_factor=15.0 / 4.0)
        assert math.isclose(r, ref, abs_tol=1e-15)

    def test_general_dispatcher_rejects_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown Bianchi type"):
            friedmann_residual_general(0.0, 0.0, 0.0, 0.0, 0.0, A=0.0,
                                        Omega=1.0, bianchi_type_str="TYPE_NONSENSE")


@pytest.mark.production
class TestMomentumResidual:
    """G^0_i monitor required by every tilted non-Type-I cell PR."""

    def test_classA_orthogonal_residual_is_zero(self):
        # No tilt, no dipole, hypersurface-orthogonal IC -> exactly zero.
        for type_str in ["TYPE_I", "TYPE_II", "TYPE_VI0", "TYPE_VII0", "TYPE_VIII", "TYPE_IX"]:
            r = momentum_residual(
                Sigma_plus=0.1, Sigma_minus=0.05,
                N1=0.1, N2=0.1, N3=0.1,
                A=0.0,
                bianchi_type_str=type_str,
                v_vec=(0.0, 0.0, 0.0),
                rho_plus_p_over_3H2=GAMMA_RAD_DEFAULT * (1.0 / 3.0),
                psi_dipole=(0.0, 0.0, 0.0),
            )
            assert r == 0.0, f"{type_str} orthogonal residual != 0: {r}"

    def test_classB_LRS_orthogonal_picks_up_A_Sigma_coupling(self):
        # For V/IV (LRS), s_1 = -3 A * Sigma_plus; s_2 = s_3 = 0 since
        # Sigma_minus = 0 in LRS.  The matter side is 0 (no tilt).
        r = momentum_residual(
            Sigma_plus=0.05, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.1,
            bianchi_type_str="TYPE_V",
        )
        # Expected magnitude = 3 * 0.1 * 0.05 = 0.015
        assert math.isclose(r, 0.015, rel_tol=1e-12)

    def test_classB_LRS_zero_A_zero_residual(self):
        r = momentum_residual(
            Sigma_plus=0.05, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_V",
        )
        assert r == 0.0

    def test_tilt_v_creates_matter_side_residual(self):
        # No geometry source (Class A orthogonal), tilt v in x-direction.
        r = momentum_residual(
            Sigma_plus=0.0, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_I",
            v_vec=(0.05, 0.0, 0.0),
        )
        # m_1 = (4/9) * gamma^2 * 0.05 with rho_plus_p_over_3H2 = 4/9.
        v_sq = 0.05 ** 2
        gamma_sq = 1.0 / (1.0 - v_sq)
        expected = (4.0 / 9.0) * gamma_sq * 0.05
        assert math.isclose(r, expected, rel_tol=1e-12)

    def test_psi_dipole_alone_creates_residual(self):
        r = momentum_residual(
            Sigma_plus=0.0, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_I",
            v_vec=(0.0, 0.0, 0.0),
            psi_dipole=(0.0, 0.001, 0.0),
        )
        assert math.isclose(r, 0.001, rel_tol=1e-12)

    def test_required_dipole_closes_typeI_tilt_residual(self):
        psi = required_momentum_dipole(
            Sigma_plus=0.0, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_I",
            v_vec=(0.05, 0.0, 0.0),
        )
        assert psi[0] < 0.0
        assert psi[1] == 0.0
        assert psi[2] == 0.0
        r = momentum_residual(
            Sigma_plus=0.0, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_I",
            v_vec=(0.05, 0.0, 0.0),
            psi_dipole=psi,
        )
        assert r < 1.0e-15

    def test_required_dipole_closes_classB_geometry_and_tilt(self):
        psi = required_momentum_dipole(
            Sigma_plus=0.05, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.1,
            bianchi_type_str="TYPE_V",
            v_vec=(0.0, 0.0, 0.02),
        )
        assert psi[0] > 0.0
        assert psi[1] == 0.0
        assert psi[2] < 0.0
        r = momentum_residual(
            Sigma_plus=0.05, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.1,
            bianchi_type_str="TYPE_V",
            v_vec=(0.0, 0.0, 0.02),
            psi_dipole=psi,
        )
        assert r < 1.0e-15

    def test_momentum_residual_reduces_orthogonal_classA_at_flrw(self):
        r = momentum_residual(
            Sigma_plus=0.0, Sigma_minus=0.0,
            N1=0.0, N2=0.0, N3=0.0,
            A=0.0,
            bianchi_type_str="TYPE_I",
        )
        assert r == 0.0


@pytest.mark.production
class TestGeometrySourceShapes:
    def test_classA_geometry_source_returns_3vec(self):
        s = momentum_geometry_source_classA(0.1, 0.05, 0.1, 0.1, 0.1)
        assert isinstance(s, tuple)
        assert len(s) == 3

    def test_classB_geometry_source_returns_3vec(self):
        s = momentum_geometry_source_classB(0.1, 0.05, 0.0, 0.1, 0.1, 0.05)
        assert isinstance(s, tuple)
        assert len(s) == 3

    def test_classA_orthogonal_returns_zeros(self):
        # F0 contract: orthogonal Class A returns zeros; per-cell PRs override.
        for inputs in [(0.0, 0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, 0.0, 0.0),
                       (0.0, 0.1, 0.0, 0.0, 0.0)]:
            s = momentum_geometry_source_classA(*inputs)
            assert s == (0.0, 0.0, 0.0)

    def test_matter_source_velocity_zero_returns_dipole_only(self):
        m = momentum_matter_source((0.0, 0.0, 0.0), 0.5,
                                    psi_dipole=(0.001, 0.002, 0.003))
        assert m == (0.001, 0.002, 0.003)
