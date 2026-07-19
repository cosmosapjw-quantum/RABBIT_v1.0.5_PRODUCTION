"""
Test: Class B Phase-1 freeze-out smoke.

Validates the Class B driver produces physical Phase-1 results for:
  Type V  (A only, no N_i)
  Type IV (A + N₁)

Small initial conditions required: A ≲ 0.001 for stability
(frame variable grows exponentially during radiation domination).
"""
import pytest

pytest.importorskip("jax", reason="JAX required")


@pytest.fixture(scope="module")
def jax_setup():
    import jax; jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def typeV_phase1(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_phase1
    result, _, _ = run_classB_phase1(JAXClassBConfig(
        bianchi_type='V', Sigma_H_plus=0.02, A_init=0.001))
    return result


@pytest.fixture(scope="module")
def typeIV_phase1(jax_setup):
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_phase1
    result, _, _ = run_classB_phase1(JAXClassBConfig(
        bianchi_type='IV', Sigma_H_plus=0.01, N1_init=0.01, A_init=0.0005))
    return result


class TestTypeVPhase1:

    def test_success(self, typeV_phase1):
        assert typeV_phase1.success

    def test_xn_physical(self, typeV_phase1):
        assert 0.05 < typeV_phase1.Xn_freeze < 0.35

    def test_A_grew(self, typeV_phase1):
        """Frame variable grows exponentially during radiation era."""
        assert typeV_phase1.metadata['A_final'] > 0.001  # grew from init

    def test_shear_decayed(self, typeV_phase1):
        assert abs(typeV_phase1.metadata['Sigma_plus_final']) < 0.02

    def test_backend_metadata(self, typeV_phase1):
        assert typeV_phase1.metadata['backend'] == 'jax_classB_driver'
        assert typeV_phase1.metadata['bianchi_type'] == 'V'
        assert typeV_phase1.metadata['c_factor'] == 3.0

    def test_no_curvature(self, typeV_phase1):
        """Type V has no N_i → N_i should remain zero."""
        assert abs(typeV_phase1.metadata['N1_final']) < 1e-15
        assert abs(typeV_phase1.metadata['N2_final']) < 1e-15
        assert abs(typeV_phase1.metadata['N3_final']) < 1e-15


class TestTypeIVPhase1:

    def test_success(self, typeIV_phase1):
        assert typeIV_phase1.success

    def test_xn_physical(self, typeIV_phase1):
        assert 0.05 < typeIV_phase1.Xn_freeze < 0.35

    def test_A_grew(self, typeIV_phase1):
        assert typeIV_phase1.metadata['A_final'] > 0.0005

    def test_N1_evolved(self, typeIV_phase1):
        """N₁ should have grown from initial value."""
        assert abs(typeIV_phase1.metadata['N1_final']) > 0.01

    def test_backend_metadata(self, typeIV_phase1):
        assert typeIV_phase1.metadata['backend'] == 'jax_classB_driver'
        assert typeIV_phase1.metadata['bianchi_type'] == 'IV'
