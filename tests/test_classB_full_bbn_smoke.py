"""
Test: Class B full BBN (Phase-1 + Phase-2) smoke.

Type V:  A only (FLRW + frame correction), A_init=0.0001
Type IV: A + N₁ (frame + curvature), A_init=1e-5

Small A_init required: frame variable grows exponentially during
radiation domination. Stability threshold:
  Type V:  A_init ≲ 0.0001
  Type IV: A_init ≲ 1e-5 (with N₁=0.001)
"""
import pytest

pytest.importorskip("jax", reason="JAX required")


@pytest.fixture(scope="module")
def jax_setup():
    import jax; jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def typeV_bbn(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax
    return run_classB_jax(JAXClassBConfig(
        bianchi_type='V', Sigma_H_plus=0.02, A_init=0.0001,
        N_q=6, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def typeIV_bbn(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax
    return run_classB_jax(JAXClassBConfig(
        bianchi_type='IV', Sigma_H_plus=0.01, N1_init=0.001, A_init=1e-5,
        N_q=6, n_ell=2, correction_level=0))


@pytest.fixture(scope="module")
def flrw_ref(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_jax
    return run_classB_jax(JAXClassBConfig(
        bianchi_type='V', Sigma_H_plus=0.0, A_init=1e-10,
        N_q=6, n_ell=2, correction_level=0))


class TestTypeVFullBBN:

    def test_success(self, typeV_bbn):
        assert typeV_bbn.success

    def test_yp_physical(self, typeV_bbn):
        assert 0.20 < typeV_bbn.Yp < 0.30

    def test_dh_physical(self, typeV_bbn):
        assert 1e-7 < typeV_bbn.DH < 1e-3

    def test_yp_near_flrw(self, typeV_bbn, flrw_ref):
        """Type V with small A should be close to FLRW."""
        assert abs(typeV_bbn.Yp - flrw_ref.Yp) < 0.005

    def test_phase_full_bbn(self, typeV_bbn):
        assert typeV_bbn.metadata['phase'] == 'full_bbn'

    def test_A_stayed_bounded(self, typeV_bbn):
        assert typeV_bbn.metadata['A_final'] < 0.577  # Ω > 0 requires A < 1/√3


class TestTypeIVFullBBN:

    def test_success(self, typeIV_bbn):
        assert typeIV_bbn.success

    def test_yp_physical(self, typeIV_bbn):
        assert 0.20 < typeIV_bbn.Yp < 0.30

    def test_dh_physical(self, typeIV_bbn):
        assert 1e-7 < typeIV_bbn.DH < 1e-3

    def test_has_curvature_effect(self, typeIV_bbn, flrw_ref):
        """N₁ > 0 should produce measurable Y_p shift from FLRW."""
        # Type IV has both A and N₁ — at least small deviation expected
        assert typeIV_bbn.Yp != flrw_ref.Yp  # not identical

    def test_metadata_complete(self, typeIV_bbn):
        for key in ['backend', 'phase', 'bianchi_type', 'c_factor',
                     'A_init', 'A_final', 'N1_final', 'T_final']:
            assert key in typeIV_bbn.metadata, f"Missing: {key}"


class TestFLRWReduction:

    def test_flrw_success(self, flrw_ref):
        assert flrw_ref.success

    def test_flrw_yp_range(self, flrw_ref):
        assert 0.20 < flrw_ref.Yp < 0.30

    def test_flrw_neff_finite(self, flrw_ref):
        assert flrw_ref.N_eff == flrw_ref.N_eff
