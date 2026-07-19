"""
Test: Tilted BBN for all Bianchi types.

§1. Type I: tilt grows, ΔY_p > 0 (expansion channel)
§2. Class A curved types: tilt + curvature
§3. Class B: tilt + frame variable
§4. v₀=0 reduces to untilted
§5. Tilt growth direction (v increases during radiation era)
"""
import pytest

pytest.importorskip("jax", reason="JAX required")


@pytest.fixture(scope="module")
def jax_setup():
    import jax; jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def flrw(jax_setup):
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
    return run_tilted_bbn(TiltedBBNConfig(bianchi_type='TYPE_I', v0=0, Sigma_H_plus=0.0))


@pytest.fixture(scope="module")
def typeI_tilted(jax_setup):
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
    return run_tilted_bbn(TiltedBBNConfig(bianchi_type='TYPE_I', v0=1e-7, Sigma_H_plus=0.05))


@pytest.fixture(scope="module")
def typeII_tilted(jax_setup):
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
    return run_tilted_bbn(TiltedBBNConfig(
        bianchi_type='TYPE_II', v0=1e-7, Sigma_H_plus=0.05, N1_init=0.1))


@pytest.fixture(scope="module")
def typeV_tilted(jax_setup):
    from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
    return run_tilted_bbn(TiltedBBNConfig(
        bianchi_type='V', v0=1e-7, Sigma_H_plus=0.02, A_init=0.0001))


class TestTypeITilted:

    def test_success(self, typeI_tilted):
        assert typeI_tilted.success
        assert typeI_tilted.metadata["final_state_ok"] is True
        assert typeI_tilted.metadata["final_state_reason"] == "ok"

    def test_yp_physical(self, typeI_tilted):
        assert 0.20 < typeI_tilted.Yp < 0.30

    def test_tilt_grew(self, typeI_tilted):
        assert abs(typeI_tilted.v_final) > 1e-7
        assert typeI_tilted.metadata["tilt_hubble_factor_final"] > 1.0
        assert typeI_tilted.metadata["Omega_tilted_final"] < typeI_tilted.metadata["Omega_final"]
        assert typeI_tilted.metadata["momentum_constraint_residual_final"] > 0.0
        assert typeI_tilted.metadata["momentum_constraint_ok_final"] is False
        assert typeI_tilted.metadata["momentum_constraint_enforced"] is False

    def test_delta_yp_positive(self, typeI_tilted, flrw):
        """Tilt produces ΔY_p > 0 (expansion channel: H/√(1-v²))."""
        assert typeI_tilted.Yp >= flrw.Yp - 1e-6


class TestTypeIITilted:

    def test_success(self, typeII_tilted):
        assert typeII_tilted.success
        assert typeII_tilted.metadata["final_state_ok"] is True

    def test_yp_physical(self, typeII_tilted):
        assert 0.20 < typeII_tilted.Yp < 0.30


class TestClassBTilted:

    def test_success(self, typeV_tilted):
        assert typeV_tilted.success
        assert typeV_tilted.metadata["final_state_ok"] is True

    def test_yp_physical(self, typeV_tilted):
        assert 0.20 < typeV_tilted.Yp < 0.30

    def test_tilt_grew(self, typeV_tilted):
        assert abs(typeV_tilted.v_final) > 1e-7

    def test_classB_metadata(self, typeV_tilted):
        assert typeV_tilted.metadata['bianchi_class'] == 'B'
        assert typeV_tilted.metadata["Omega_tilted_final"] < typeV_tilted.metadata["Omega_final"]
        assert typeV_tilted.metadata["momentum_constraint_residual_final"] > 0.0
        assert typeV_tilted.metadata["momentum_constraint_source"] == "scalar_v3_no_dipole_diagnostic"


class TestUntiltedReduction:

    def test_v0_zero_gives_zero_tilt(self, flrw):
        assert flrw.v_final == 0.0
        assert flrw.metadata["Omega_tilted_final"] == pytest.approx(flrw.metadata["Omega_final"])
        assert flrw.metadata["tilt_hubble_factor_final"] == pytest.approx(1.0)
        assert flrw.metadata["momentum_constraint_residual_final"] == pytest.approx(0.0)
        assert flrw.metadata["momentum_constraint_ok_final"] is True

    def test_v0_zero_yp_physical(self, flrw):
        assert 0.240 < flrw.Yp < 0.245


class TestFLRWTilt:
    """FLRW tilt = Type I with Σ=0, v≠0. Tests the 'FLRW' alias."""

    @pytest.fixture(scope="class")
    def flrw_tilted(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn
        return run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='FLRW', v0=1e-5, Sigma_H_plus=0.0))

    def test_alias_works(self, flrw_tilted):
        assert flrw_tilted.success

    def test_yp_physical(self, flrw_tilted):
        assert 0.20 < flrw_tilted.Yp < 0.30

    def test_tilt_grew(self, flrw_tilted):
        """v₀=10⁻⁵ should grow significantly during ~14 e-folds."""
        assert abs(flrw_tilted.v_final) > 1e-3

    def test_yp_near_flrw(self, flrw_tilted, flrw):
        """Small tilt should not drastically change Y_p."""
        assert abs(flrw_tilted.Yp - flrw.Yp) < 0.01

    def test_metadata_bianchi_type(self, flrw_tilted):
        assert flrw_tilted.metadata['bianchi_type'] == 'FLRW'


class TestTiltStressFeedback:
    def test_opt_in_tilt_stress_feedback_sources_shear(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        baseline = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_stress_feedback=False,
        ))
        feedback = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_stress_feedback=True,
        ))

        assert baseline.success is True
        assert feedback.success is True
        assert baseline.metadata["tilt_stress_feedback_enabled"] is False
        assert feedback.metadata["tilt_stress_feedback_enabled"] is True
        assert feedback.metadata["tilt_perfect_fluid_pi_plus_final"] > 0.0
        assert feedback.metadata["Sigma_plus_final"] > baseline.metadata["Sigma_plus_final"]
        assert feedback.Yp != pytest.approx(baseline.Yp, abs=1.0e-12)

    def test_axis1_tilt_stress_sources_both_shear_components(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        feedback = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            tilt_axis=1,
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.0,
            N_q=6,
            correction_level=0,
            tilt_stress_feedback=True,
        ))

        assert feedback.success is True
        assert feedback.metadata["tilt_axis"] == 1
        assert feedback.metadata["tilt_vector_final"][0] == pytest.approx(feedback.v_final)
        assert feedback.metadata["tilt_vector_final"][1] == pytest.approx(0.0)
        assert feedback.metadata["tilt_vector_final"][2] == pytest.approx(0.0)
        assert feedback.metadata["tilt_perfect_fluid_pi_plus_final"] < 0.0
        assert feedback.metadata["tilt_perfect_fluid_pi_minus_final"] < 0.0
        assert feedback.metadata["Sigma_plus_final"] < 0.0
        assert feedback.metadata["Sigma_minus_final"] < 0.0
        assert feedback.metadata["momentum_constraint_source"] == "principal_axis_v1_no_dipole_diagnostic"


class TestTiltMomentumClosure:
    def test_algebraic_heat_flux_closes_momentum_constraint(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        closed = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            tilt_axis=1,
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.0,
            N_q=6,
            correction_level=0,
            tilt_stress_feedback=True,
            momentum_constraint_closure="algebraic_heat_flux",
        ))

        assert closed.success is True
        assert closed.metadata["momentum_constraint_closure_mode"] == "algebraic_heat_flux"
        assert closed.metadata["momentum_constraint_enforced"] is True
        assert closed.metadata["momentum_constraint_dynamical_heat_flux"] is False
        assert closed.metadata["momentum_constraint_residual_no_dipole_final"] > 0.0
        assert closed.metadata["momentum_required_dipole_norm_final"] > 0.0
        assert closed.metadata["momentum_applied_dipole_final"] == pytest.approx(
            closed.metadata["momentum_required_dipole_final"]
        )
        assert closed.metadata["momentum_constraint_residual_final"] < 1.0e-12
        assert closed.metadata["momentum_constraint_ok_final"] is True
        assert closed.metadata["momentum_constraint_source"] == "principal_axis_v1_algebraic_heat_flux"


class TestTiltedNonLRSTransport:
    def test_n_ell3_activates_sigma_minus_transport_feedback(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        lrs = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=0.0,
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.05,
            N_q=6,
            n_ell=2,
            correction_level=0,
        ))
        nonlrs_cfg = TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=0.0,
            Sigma_H_plus=0.0,
            Sigma_H_minus=0.05,
            N_q=6,
            n_ell=3,
            correction_level=0,
        )
        nonlrs = run_tilted_bbn(nonlrs_cfg)

        assert lrs.success is True
        assert nonlrs.success is True
        assert lrs.metadata["transport_pi_minus_enabled"] is False
        assert nonlrs.metadata["transport_pi_minus_enabled"] is True
        assert lrs.metadata["transport_pi_minus_final"] == pytest.approx(0.0)
        assert abs(nonlrs.metadata["transport_pi_minus_final"]) > 0.0
        assert nonlrs.metadata["transport_mode"] == "tilted_kappa_cascade_lmax2"
        assert nonlrs.metadata["transport_quadrupole_mode"] == "diagonal_nonlrs_plus_minus"
        assert nonlrs.metadata["Sigma_minus_final"] != pytest.approx(
            lrs.metadata["Sigma_minus_final"], abs=1.0e-12
        )


class TestTiltWeakRateBoost:
    def test_opt_in_weak_rate_boost_changes_forward_solution(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        baseline = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_weak_rate_boost=False,
        ))
        boosted = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_weak_rate_boost=True,
        ))

        assert baseline.success is True
        assert boosted.success is True
        assert baseline.metadata["tilt_weak_rate_boost_enabled"] is False
        assert boosted.metadata["tilt_weak_rate_boost_enabled"] is True
        assert boosted.metadata["tilt_weak_rate_boost_mode"] == "boosted_fd_monopole_ratio"
        assert boosted.metadata["tilt_weak_rate_boost_monopole_delta_final"] > 0.0
        assert boosted.metadata["tilt_weak_rate_boost_lnp_factor_final"] != pytest.approx(1.0, abs=1.0e-12)
        assert boosted.Yp != pytest.approx(baseline.Yp, abs=1.0e-12)


class TestTiltedCL3WeakBudget:
    def test_tilted_runner_supports_cl3_finite_mass_budget(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        cl2 = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-7,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=2,
        ))
        cl3 = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-7,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=3,
        ))

        assert cl2.success is True
        assert cl3.success is True
        assert cl2.metadata["weak_budget_mode"] == "born+coulomb+sirlin"
        assert cl3.metadata["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
        assert cl3.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is True
        assert cl3.metadata["correction_budget_channels"]["finite_mass_angular_kernel_coupled"] is False
        assert cl3.Yp > cl2.Yp

    def test_boosted_tilted_weak_rates_support_cl3_budget(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        boosted = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=3,
            tilt_weak_rate_boost=True,
        ))

        assert boosted.success is True
        assert boosted.metadata["weak_budget_mode"] == "born+coulomb+sirlin+finite_mass"
        assert boosted.metadata["tilt_weak_rate_boost_enabled"] is True
        assert boosted.metadata["tilt_weak_rate_boost_monopole_delta_final"] > 0.0
        assert boosted.metadata["correction_budget_channels"]["finite_mass_recoil_wm"] is True

    def test_cl3_angular_kernel_couples_boosted_moments_to_forward_solution(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        scalar = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=3,
            tilt_weak_rate_boost=True,
        ))
        angular = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-5,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=3,
            tilt_weak_rate_boost=True,
            tilt_cl3_angular_kernel=True,
        ))

        assert scalar.success is True
        assert angular.success is True
        assert angular.metadata["tilt_cl3_angular_kernel_enabled"] is True
        assert angular.metadata["tilt_cl3_angular_kernel_mode"] == "boosted_fd_f0_f1_f2_principal_axis"
        assert angular.metadata["tilt_weak_rate_boost_mode"] == "boosted_fd_l012_angular_kernel_ratio"
        assert angular.metadata["tilt_cl3_angular_f1_absmax_final"] > 0.0
        assert angular.metadata["tilt_cl3_angular_f2_absmax_final"] > 0.0
        assert angular.metadata["correction_budget_channels"]["finite_mass_angular_kernel_coupled"] is True
        assert (
            angular.metadata["tilt_weak_rate_boost_lnp_factor_final"]
            != scalar.metadata["tilt_weak_rate_boost_lnp_factor_final"]
        )


class TestTiltHubbleClosure:
    def test_opt_in_stress_energy_hubble_closure_changes_forward_solution(self, jax_setup):
        from rabbit.jax.run_tilted_bbn import TiltedBBNConfig, run_tilted_bbn

        baseline = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-3,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_hubble_closure="legacy_gamma",
        ))
        stress_energy = run_tilted_bbn(TiltedBBNConfig(
            bianchi_type='TYPE_I',
            v0=1.0e-3,
            Sigma_H_plus=0.0,
            N_q=6,
            correction_level=0,
            tilt_stress_feedback=True,
            tilt_hubble_closure="stress_energy_t00",
        ))

        assert baseline.success is True
        assert stress_energy.success is True
        assert baseline.metadata["tilt_hubble_closure_mode"] == "legacy_gamma"
        assert stress_energy.metadata["tilt_hubble_closure_mode"] == "stress_energy_t00"
        assert stress_energy.metadata["tilt_hubble_factor_final"] == pytest.approx(
            stress_energy.metadata["tilt_hubble_stress_energy_factor_final"]
        )
        assert (
            stress_energy.metadata["tilt_hubble_stress_energy_factor_final"]
            > stress_energy.metadata["tilt_hubble_legacy_gamma_factor_final"]
        )
        assert stress_energy.metadata["Omega_tilted_final"] == pytest.approx(
            stress_energy.metadata["Omega_tilted_stress_energy_final"]
        )
        assert (
            stress_energy.metadata["Omega_tilted_stress_energy_final"]
            < stress_energy.metadata["Omega_tilted_legacy_gamma_final"]
        )
        gamma = 4.0 / 3.0
        v_sq = stress_energy.metadata["v_sq_final"]
        expected_pi_plus = (
            gamma
            * stress_energy.metadata["Omega_tilted_stress_energy_final"]
            * v_sq
            / (1.0 - v_sq)
            / 3.0
        )
        assert stress_energy.metadata["tilt_perfect_fluid_pi_plus_final"] == pytest.approx(
            expected_pi_plus
        )
        assert stress_energy.Yp != pytest.approx(baseline.Yp, abs=1.0e-12)


class TestTiltGrowth:

    def test_v_final_gt_v_init(self, typeI_tilted):
        """Tilt grows during radiation era (eigenvalue +1)."""
        assert abs(typeI_tilted.v_final) > 1e-7  # grew from 1e-7
