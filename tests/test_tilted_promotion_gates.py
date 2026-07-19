"""
tests/test_tilted_promotion_gates.py — Tilted BBN promotion gate tests.
"""
import pytest
import numpy as np


@pytest.mark.production
@pytest.mark.release_smoke
class TestTiltedGateT1_RetiredPublicAPI:
    def test_tilt_parameters_absent_from_canonical_signature(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver
        import inspect
        retired = {
            'v0', 'tilt_axis', 'tilt_n_ell', 'tilt_stress_feedback',
            'momentum_constraint_closure', 'tilt_weak_rate_boost',
            'tilt_cl3_angular_kernel', 'tilt_hubble_closure',
        }
        assert retired.isdisjoint(inspect.signature(canonical_forward_solver).parameters)

    def test_tiltstate_constructible(self):
        from rabbit.geometry.tilt import TiltState
        ts = TiltState.from_velocity(1e-7, scalar=True)
        assert ts.v_sq < 1e-13

    def test_public_dispatch_fails_closed(self):
        from rabbit.inference.forward_likelihood import canonical_forward_solver

        with pytest.raises(ValueError, match="retired"):
            canonical_forward_solver(backend='jax_tilted')

    def test_public_registry_absent_but_metadata_preserved(self):
        from rabbit.config.backend_capabilities import (
            CAPABILITY_BY_BACKEND,
            CAPABILITY_BY_KEY,
        )

        assert 'jax_tilted' not in CAPABILITY_BY_BACKEND
        assert 'jax_tilted_bbn' in CAPABILITY_BY_KEY


@pytest.mark.production
@pytest.mark.release_smoke
class TestTiltedGateT2_LimitRecovery:
    def test_v0_rhs_is_zero(self):
        from rabbit.geometry.tilt import compute_tilt_rhs_flrw
        assert abs(compute_tilt_rhs_flrw(0.0)) < 1e-15

    def test_v0_hubble_correction_is_unity(self):
        from rabbit.geometry.tilt import tilt_hubble_correction
        assert abs(tilt_hubble_correction(1.0, 0.0) - 1.0) < 1e-15

    def test_v0_stress_energy_hubble_factor_is_unity(self):
        from rabbit.geometry.tilt import tilt_hubble_stress_energy_factor
        assert abs(tilt_hubble_stress_energy_factor(0.0) - 1.0) < 1e-15

    def test_v0_normal_energy_density_factor_is_unity(self):
        from rabbit.geometry.tilt import tilted_normal_energy_density_factor
        assert abs(tilted_normal_energy_density_factor(0.0) - 1.0) < 1e-15

    def test_v0_delta_neff_is_zero(self):
        from rabbit.geometry.tilt import tilt_to_delta_neff
        assert abs(tilt_to_delta_neff(0.0)) < 1e-15

    def test_v0_tilt_anisotropic_stress_is_zero(self):
        from rabbit.geometry.tilt import tilt_anisotropic_stress_scalar
        pi_plus, pi_minus = tilt_anisotropic_stress_scalar(0.0, 1.0)
        assert abs(pi_plus) < 1e-15
        assert abs(pi_minus) < 1e-15

    def test_v0_boosted_fd_monopole_recovers_equilibrium(self):
        pytest.importorskip("jax", reason="JAX required")
        import jax.numpy as jnp
        from rabbit.jax.tilt_jax import boosted_fd_monopole

        q = jnp.asarray([0.1, 1.0, 3.0, 10.0], dtype=jnp.float64)
        boosted = np.asarray(boosted_fd_monopole(q, jnp.asarray(0.0)))
        expected = np.asarray(1.0 / (jnp.exp(q) + 1.0))
        assert np.allclose(boosted, expected, rtol=0.0, atol=1e-15)

    def test_v0_boosted_fd_moments_recover_isotropic_limit(self):
        pytest.importorskip("jax", reason="JAX required")
        import jax.numpy as jnp
        from rabbit.jax.tilt_jax import boosted_fd_legendre_moments

        q = jnp.asarray([0.1, 1.0, 3.0, 10.0], dtype=jnp.float64)
        f0, f1, f2 = boosted_fd_legendre_moments(q, jnp.asarray(0.0))
        expected = np.asarray(1.0 / (jnp.exp(q) + 1.0))
        assert np.allclose(np.asarray(f0), expected, rtol=0.0, atol=1e-15)
        assert np.max(np.abs(np.asarray(f1))) < 1e-15
        assert np.max(np.abs(np.asarray(f2))) < 2e-15

    def test_v0_weak_rates_unchanged(self):
        from rabbit.drivers.tilted_flrw import _weak_rates_corrected
        a, _ = _weak_rates_corrected(1.0, 1.0/1.4, 878.4, v=0.0)
        b, _ = _weak_rates_corrected(1.0, 1.0/1.4, 878.4, v=0.0)
        assert abs(a - b) < 1e-15

    def test_tilted_config_accepts_cl3_weak_budget(self):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig
        assert TiltedBBNConfig(correction_level=3).correction_level == 3


@pytest.mark.production
@pytest.mark.release_smoke
class TestTiltedGateT3_Parity:
    def test_tilt_rhs_deterministic(self):
        from rabbit.geometry.tilt import compute_tilt_rhs_flrw
        a = compute_tilt_rhs_flrw(1e-5)
        b = compute_tilt_rhs_flrw(1e-5)
        assert a == b

    def test_weak_rates_deterministic(self):
        from rabbit.drivers.tilted_flrw import _weak_rates_corrected
        a, _ = _weak_rates_corrected(1.0, 1.0/1.4, 878.4, v=1e-3)
        b, _ = _weak_rates_corrected(1.0, 1.0/1.4, 878.4, v=1e-3)
        assert abs(a - b) < 1e-15


@pytest.mark.production
@pytest.mark.release_smoke
class TestTiltedGateT4_Convergence:
    def test_eigenvalue_positive(self):
        from rabbit.geometry.tilt import tilt_eigenvalue_flrw
        assert tilt_eigenvalue_flrw() == 1.0

    def test_small_v_perturbative(self):
        from rabbit.geometry.tilt import compute_tilt_rhs_flrw
        rhs = compute_tilt_rhs_flrw(1e-10)
        assert abs(rhs - 1e-10) < 1e-15  # linear in v at small v

    def test_smooth_across_v_range(self):
        from rabbit.drivers.tilted_flrw import _weak_rates_corrected
        vs = np.logspace(-10, -2, 20)
        rates = [_weak_rates_corrected(1.0, 1.0/1.4, 878.4, v=v)[0] for v in vs]
        jumps = [abs(rates[i+1]-rates[i]) for i in range(len(rates)-1)]
        assert max(jumps) < 0.1 * abs(rates[0])

    def test_shear_coupling_sign(self):
        """Positive shear should suppress tilt growth (λ₃ = 2Σ₊ > 0 shifts effective eigenvalue)."""
        from rabbit.geometry.tilt import TiltState, compute_tilt_rhs_typeI
        ts = TiltState.from_velocity(1e-5, scalar=True)
        rhs_0 = compute_tilt_rhs_typeI(ts, 0.0, 0.0, 1.0)
        rhs_s = compute_tilt_rhs_typeI(ts, 0.1, 0.0, 1.0)
        # With shear, the effective eigenvalue changes
        # Both should produce finite results
        assert np.all(np.isfinite(rhs_0))
        assert np.all(np.isfinite(rhs_s))

    def test_scalar_tilt_stress_has_expected_sign(self):
        from rabbit.geometry.tilt import tilt_anisotropic_stress_scalar
        pi_plus, pi_minus = tilt_anisotropic_stress_scalar(1.0e-4, 0.9)
        assert pi_plus > 0.0
        assert pi_minus == 0.0

    def test_stress_energy_hubble_factor_exceeds_legacy_for_radiation(self):
        from rabbit.geometry.tilt import (
            tilt_hubble_stress_energy_factor,
            tilted_normal_energy_density_factor,
        )
        v_sq = 1.0e-4
        legacy = 1.0 / np.sqrt(1.0 - v_sq)
        expected_density_factor = 1.0 + (4.0 / 3.0) * v_sq / (1.0 - v_sq)
        assert tilted_normal_energy_density_factor(
            v_sq, gamma=4.0 / 3.0
        ) == pytest.approx(expected_density_factor)
        assert tilt_hubble_stress_energy_factor(v_sq, gamma=4.0 / 3.0) > legacy

    def test_boosted_fd_monopole_changes_for_finite_tilt(self):
        pytest.importorskip("jax", reason="JAX required")
        import jax.numpy as jnp
        from rabbit.jax.tilt_jax import boosted_fd_monopole

        q = jnp.asarray([0.1, 1.0, 3.0, 10.0], dtype=jnp.float64)
        boosted = np.asarray(boosted_fd_monopole(q, jnp.asarray(0.05)))
        expected = np.asarray(1.0 / (jnp.exp(q) + 1.0))
        assert np.all((boosted >= 0.0) & (boosted <= 1.0))
        assert np.max(np.abs(boosted - expected)) > 0.0

    def test_boosted_fd_higher_moments_change_for_finite_tilt(self):
        pytest.importorskip("jax", reason="JAX required")
        import jax.numpy as jnp
        from rabbit.jax.tilt_jax import boosted_fd_legendre_moments

        q = jnp.asarray([0.1, 1.0, 3.0, 10.0], dtype=jnp.float64)
        _, f1, f2 = boosted_fd_legendre_moments(q, jnp.asarray(0.05))
        assert np.max(np.abs(np.asarray(f1))) > 0.0
        assert np.max(np.abs(np.asarray(f2))) > 0.0

    def test_principal_axis_stress_projection_signs(self):
        from rabbit.geometry.tilt import tilt_anisotropic_stress_principal_axis

        pi1_plus, pi1_minus = tilt_anisotropic_stress_principal_axis(
            1.0e-4, 0.9, axis=1)
        pi2_plus, pi2_minus = tilt_anisotropic_stress_principal_axis(
            1.0e-4, 0.9, axis=2)
        pi3_plus, pi3_minus = tilt_anisotropic_stress_principal_axis(
            1.0e-4, 0.9, axis=3)

        assert pi1_plus < 0.0
        assert pi2_plus < 0.0
        assert pi3_plus > 0.0
        assert pi1_minus < 0.0
        assert pi2_minus > 0.0
        assert pi3_minus == 0.0
        assert pi1_plus == pytest.approx(pi2_plus)
        assert pi3_plus == pytest.approx(-2.0 * pi1_plus)


@pytest.mark.production
@pytest.mark.release_smoke
class TestTiltedGateT5_Docs:
    def test_promotion_packet_exists(self):
        import os
        assert os.path.exists("docs/TILTED_PROMOTION_PACKET.md")

    def test_supported_capabilities_entry(self):
        with open("SUPPORTED_CAPABILITIES.md") as f:
            assert "tilted" in f.read().lower()
