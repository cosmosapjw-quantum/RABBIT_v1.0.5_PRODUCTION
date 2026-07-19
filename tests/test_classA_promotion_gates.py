"""
tests/test_classA_promotion_gates.py — Class A promotion gate tests.
"""
import pytest
import numpy as np


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassAGateG1_TypeILimit:
    """N1=N2=N3=0 must recover Type I exactly."""

    def test_curvature_source_zero(self):
        from rabbit.geometry.general_classA import curvature_source_S
        Sp, Sm = curvature_source_S(0, 0, 0)
        assert abs(Sp) < 1e-15 and abs(Sm) < 1e-15

    def test_gauss_curvature_zero(self):
        from rabbit.geometry.general_classA import gauss_curvature_K
        assert abs(gauss_curvature_K(0, 0, 0)) < 1e-15

    def test_friedmann_at_typeI(self):
        from rabbit.geometry.general_classA import compute_Omega
        Om = compute_Omega(0.1, 0.0, 0.0, 0.0, 0.0)
        expected = 1.0 - 0.1**2  # Ω = 1 - Σ²
        assert abs(Om - expected) < 1e-10

    def test_geometry_rhs_typeI_limit(self):
        from rabbit.geometry.general_classA import classA_geometry_rhs
        rhs = classA_geometry_rhs(0.1, 0.0, 0.0, 0.0, 0.0, 0.01, 1.0)
        # dN1=dN2=dN3=0 at Type I limit
        assert abs(rhs[2]) < 1e-15  # dN1
        assert abs(rhs[3]) < 1e-15  # dN2
        assert abs(rhs[4]) < 1e-15  # dN3


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassAGateG2_CurvatureSources:
    """Each type must have distinct curvature signature."""

    def test_typeII_nonzero_S_plus(self):
        from rabbit.geometry.general_classA import curvature_source_S
        Sp, _ = curvature_source_S(0.1, 0.0, 0.0)
        assert abs(Sp) > 1e-5, "Type II must have nonzero S+"

    def test_typeIX_nonzero_K(self):
        from rabbit.geometry.general_classA import gauss_curvature_K
        K = gauss_curvature_K(0.1, 0.1, 0.1)
        assert abs(K) > 1e-5, "Type IX must have nonzero K"

    def test_typeVII0_isotropic_cancellation(self):
        from rabbit.geometry.general_classA import curvature_source_S
        Sp, Sm = curvature_source_S(0.0, 0.1, 0.1)
        # VII₀ has symmetric N2=N3, S should be small or structured
        assert np.isfinite(Sp) and np.isfinite(Sm)


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassAGateG3_SmoothDeformation:
    """Small curvature must give smooth deformation from Type I."""

    def test_smooth_N1_variation(self):
        from rabbit.geometry.general_classA import classA_geometry_rhs
        rhs_values = []
        for N1 in [0.0, 0.001, 0.01, 0.05, 0.1]:
            rhs = classA_geometry_rhs(0.1, 0.0, N1, 0.0, 0.0, 0.01, 1.0)
            rhs_values.append(rhs[0])  # dΣ+
        # Check monotonicity or at least smoothness
        diffs = [abs(rhs_values[i+1]-rhs_values[i]) for i in range(len(rhs_values)-1)]
        assert max(diffs) < 0.1, f"Non-smooth: max jump = {max(diffs)}"

    def test_deceleration_parameter_physical(self):
        from rabbit.geometry.general_classA import compute_q
        for S in [0.0, 0.1, 0.3]:
            q = compute_q(S, 0, 0, 0, 0)
            assert q > 0, f"q={q} non-positive at Σ={S}"
            assert q < 3, f"q={q} unphysically large"


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassAGateG4_RetiredPublicDispatch:
    """Class A source remains inspectable but is not a public backend."""

    def test_classA_absent_from_public_map_but_metadata_preserved(self):
        from rabbit.config.backend_capabilities import (
            CAPABILITY_BY_BACKEND,
            CAPABILITY_BY_KEY,
        )
        assert 'jax_classA' not in CAPABILITY_BY_BACKEND
        assert 'jax_classA_driver' in CAPABILITY_BY_KEY

    def test_classA_public_dispatch_fails_closed(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        with pytest.raises(ValueError, match="retired"):
            canonical_forward_solver(backend='jax_classA')

    def test_classA_config_constructible(self):
        from rabbit.jax.driver_classA import JAXClassAConfig
        cfg = JAXClassAConfig(bianchi_type="TYPE_II", N1_init=0.01,
                               Sigma_H_plus=0.1, N_q=20)
        assert cfg.bianchi_type == "TYPE_II"


@pytest.mark.production
@pytest.mark.release_smoke
class TestClassAGateG5_Docs:
    def test_promotion_packet_exists(self):
        import os
        assert os.path.exists("docs/CLASSA_PROMOTION_PACKET.md")

    def test_approximation_documented(self):
        with open("SUPPORTED_CAPABILITIES.md") as f:
            cap = f.read()
        assert "reduced transport" in cap.lower()
