//! Standalone neutron-fraction freeze-out on the ideal two-leg FLRW substrate.
//!
//! This is the first consumer coupling F-01, F-02, and the six-channel F-03a
//! Born rates.  It evolves only `(T_gamma, X_n)` versus `N=ln(a/a_start)`.
//! The FLRW leg can select the scalar finite-temperature QED EOS and the weak
//! leg can select the zero-temperature Coulomb/radiative correction and the
//! finite-nucleon-mass correction with weak magnetism either disabled or set
//! to PRIMAT's physical magnetic-moment coefficient.  The complete four-term
//! finite-temperature radiative block can be selected only as the next layer
//! on that physical-WM base.  Nuclear reactions remain absent here.

#![cfg_attr(not(test), allow(dead_code))]

use crate::born_weak::{WeakRateModel, evaluate_weak_rates};
use crate::flrw::IdealFlrwSystem;
use crate::ode::OdeSystem;
use crate::qed_eos::FiniteTemperatureQed;
use crate::thermal_weak::THERMAL_RADIATIVE_MAXIMUM_MEV;

#[derive(Clone, Copy, Debug)]
struct FreezeoutDerivative {
    rhs: [f64; 2],
    dxn_dxn: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BornFreezeoutSystem {
    flrw_leg: IdealFlrwSystem,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
    weak_rate_model: WeakRateModel,
}

impl BornFreezeoutSystem {
    pub(crate) fn common_bath_leg(
        decoupling_temperature_mev: f64,
        neutron_lifetime_seconds: f64,
        quadrature_order: usize,
    ) -> Self {
        Self {
            flrw_leg: IdealFlrwSystem::common_bath_leg(decoupling_temperature_mev),
            neutron_lifetime_seconds,
            quadrature_order,
            weak_rate_model: WeakRateModel::Born,
        }
    }

    pub(crate) fn electromagnetic_bath_leg(
        decoupling_temperature_mev: f64,
        neutron_lifetime_seconds: f64,
        quadrature_order: usize,
    ) -> Self {
        Self {
            flrw_leg: IdealFlrwSystem::electromagnetic_bath_leg(decoupling_temperature_mev),
            neutron_lifetime_seconds,
            quadrature_order,
            weak_rate_model: WeakRateModel::Born,
        }
    }

    pub(crate) fn ideal_high_temperature_instantaneous_decoupling(
        neutron_lifetime_seconds: f64,
        quadrature_order: usize,
    ) -> Self {
        Self {
            flrw_leg: IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling(),
            neutron_lifetime_seconds,
            quadrature_order,
            weak_rate_model: WeakRateModel::Born,
        }
    }

    pub(crate) fn high_temperature_instantaneous_decoupling_with_physics(
        neutron_lifetime_seconds: f64,
        quadrature_order: usize,
        qed_model: FiniteTemperatureQed,
        weak_rate_model: WeakRateModel,
    ) -> Self {
        Self {
            flrw_leg: IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(
                qed_model,
            ),
            neutron_lifetime_seconds,
            quadrature_order,
            weak_rate_model,
        }
    }

    /// Matched leading-QED/CCR freeze-out including the F08B finite-mass term.
    ///
    /// Weak magnetism is explicitly absent from the selected weak-rate model.
    pub(crate) fn primat_leading_qed_ccr_finite_mass_no_weak_magnetism(
        neutron_lifetime_seconds: f64,
        quadrature_order: usize,
    ) -> Self {
        Self::high_temperature_instantaneous_decoupling_with_physics(
            neutron_lifetime_seconds,
            quadrature_order,
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
        )
    }

    fn derivative(&self, t_gamma_mev: f64, neutron_fraction: f64) -> Option<FreezeoutDerivative> {
        if !neutron_fraction.is_finite() || !(0.0..=1.0).contains(&neutron_fraction) {
            return None;
        }
        let background = self.flrw_leg.thermo_state(t_gamma_mev).ok()?;
        let rates = evaluate_weak_rates(
            t_gamma_mev,
            background.t_nu_mev,
            self.neutron_lifetime_seconds,
            self.quadrature_order,
            self.weak_rate_model,
        )
        .ok()?;
        let inverse_hubble = background.h_inverse_seconds.recip();
        let dxn_d_lna = (-rates.neutron_to_proton_per_second * neutron_fraction
            + rates.proton_to_neutron_per_second * (1.0 - neutron_fraction))
            * inverse_hubble;
        let dxn_dxn = -(rates.neutron_to_proton_per_second + rates.proton_to_neutron_per_second)
            * inverse_hubble;
        let derivative = FreezeoutDerivative {
            rhs: [background.d_tgamma_d_lna, dxn_d_lna],
            dxn_dxn,
        };
        if derivative
            .rhs
            .iter()
            .chain([derivative.dxn_dxn].iter())
            .any(|value| !value.is_finite())
        {
            return None;
        }
        Some(derivative)
    }
}

impl OdeSystem for BornFreezeoutSystem {
    fn dimension(&self) -> usize {
        2
    }

    fn rhs(&self, _ln_a: f64, y: &[f64], out: &mut [f64]) {
        match self.derivative(y[0], y[1]) {
            Some(derivative) => out.copy_from_slice(&derivative.rhs),
            None => out.fill(f64::NAN),
        }
    }

    fn jacobian(&self, _ln_a: f64, y: &[f64], out: &mut [f64]) {
        let step = (1.0e-5 * y[0].abs()).max(1.0e-10);
        let center = self.derivative(y[0], y[1]);
        let plus = self.derivative(y[0] + step, y[1]);
        let minus = self.derivative(y[0] - step, y[1]);
        let Some(center) = center else {
            out.fill(f64::NAN);
            return;
        };
        let temperature_derivative = match (plus, minus) {
            (Some(plus), Some(minus)) => [
                (plus.rhs[0] - minus.rhs[0]) / (2.0 * step),
                (plus.rhs[1] - minus.rhs[1]) / (2.0 * step),
            ],
            (None, Some(minus))
                if self.weak_rate_model
                    == WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism
                    && y[0] <= THERMAL_RADIATIVE_MAXIMUM_MEV
                    && y[0] + step > THERMAL_RADIATIVE_MAXIMUM_MEV =>
            {
                // The F08D table has an explicit 10 MeV upper authority
                // boundary.  Retain second-order accuracy with a three-point
                // in-domain stencil; no other failed probe is masked.
                let Some(minus_two) = self.derivative(y[0] - 2.0 * step, y[1]) else {
                    out.fill(f64::NAN);
                    return;
                };
                [
                    (3.0 * center.rhs[0] - 4.0 * minus.rhs[0] + minus_two.rhs[0]) / (2.0 * step),
                    (3.0 * center.rhs[1] - 4.0 * minus.rhs[1] + minus_two.rhs[1]) / (2.0 * step),
                ]
            }
            _ => {
                out.fill(f64::NAN);
                return;
            }
        };
        out.copy_from_slice(&[
            temperature_derivative[0],
            0.0,
            temperature_derivative[1],
            center.dxn_dxn,
        ]);
    }

    fn dfdt(&self, _ln_a: f64, _y: &[f64], out: &mut [f64]) {
        out.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::born_weak::{
        DEFAULT_BORN_WEAK_QUADRATURE_ORDER, DEFAULT_NEUTRON_LIFETIME_SECONDS, evaluate_weak_rates,
    };
    use crate::born_weak::{
        NEUTRON_PROTON_MASS_DIFFERENCE_MEV, PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
    };
    use crate::minimal_bbn::PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV;
    use crate::ode::{OdeConfig, OdeResult, SolverKind, TerminalEvent, solve};

    const INITIAL_TEMPERATURE_MEV: f64 = 10.0;
    const DECOUPLING_TEMPERATURE_MEV: f64 = 2.0;
    const FINAL_TEMPERATURE_MEV: f64 = 0.05;
    const EXTERNAL_ORACLE_N_TOLERANCE: f64 = 3.0e-7;
    const EXTERNAL_ORACLE_XN_TOLERANCE: f64 = 2.0e-8;
    const FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE: f64 = 6.0e-8;
    const F08D_EXTERNAL_ORACLE_XN_TOLERANCE: f64 = 8.0e-8;

    fn config() -> OdeConfig {
        OdeConfig {
            rtol: 2.0e-8,
            atol: vec![1.0e-10, 1.0e-11],
            h_init: 1.0e-5,
            h_min: 1.0e-13,
            h_max: 0.03,
            max_attempts: 100_000,
        }
    }

    fn equilibrium_neutron_fraction(temperature_mev: f64) -> f64 {
        1.0 / (1.0 + (NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp())
    }

    fn integrate_freezeout(kind: SolverKind, order: usize) -> (OdeResult, OdeResult) {
        let common = BornFreezeoutSystem::common_bath_leg(
            DECOUPLING_TEMPERATURE_MEV,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
        );
        let electromagnetic = BornFreezeoutSystem::electromagnetic_bath_leg(
            DECOUPLING_TEMPERATURE_MEV,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
        );
        let decoupling_event_fn = |_ln_a: f64, y: &[f64]| y[0] - DECOUPLING_TEMPERATURE_MEV;
        let decoupling_event = TerminalEvent {
            value: &decoupling_event_fn,
            direction: -1,
        };
        let final_event_fn = |_ln_a: f64, y: &[f64]| y[0] - FINAL_TEMPERATURE_MEV;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let initial = [
            INITIAL_TEMPERATURE_MEV,
            equilibrium_neutron_fraction(INITIAL_TEMPERATURE_MEV),
        ];
        let at_decoupling = solve(
            kind,
            &common,
            (0.0, 5.0),
            &initial,
            &config(),
            Some(&decoupling_event),
        );
        assert_eq!(at_decoupling.failure, None, "{kind:?}: {at_decoupling:?}");
        assert!(at_decoupling.event_reached, "{kind:?}: {at_decoupling:?}");
        let terminal = solve(
            kind,
            &electromagnetic,
            (at_decoupling.t, 12.0),
            &at_decoupling.y,
            &config(),
            Some(&final_event),
        );
        (at_decoupling, terminal)
    }

    fn integrate_leading_qed_weak_freezeout(
        kind: SolverKind,
        system: BornFreezeoutSystem,
        weak_rate_model: WeakRateModel,
    ) -> (OdeResult, OdeResult) {
        assert_eq!(system.weak_rate_model, weak_rate_model);
        let initial_background = system
            .flrw_leg
            .thermo_state(INITIAL_TEMPERATURE_MEV)
            .unwrap();
        let initial_rates = evaluate_weak_rates(
            INITIAL_TEMPERATURE_MEV,
            initial_background.t_nu_mev,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            weak_rate_model,
        )
        .unwrap();
        let initial_rate_sum =
            initial_rates.neutron_to_proton_per_second + initial_rates.proton_to_neutron_per_second;
        let initial = [
            INITIAL_TEMPERATURE_MEV,
            initial_rates.proton_to_neutron_per_second / initial_rate_sum,
        ];
        let activation_event_fn =
            |_ln_a: f64, y: &[f64]| y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV;
        let activation_event = TerminalEvent {
            value: &activation_event_fn,
            direction: -1,
        };
        let final_event_fn = |_ln_a: f64, y: &[f64]| y[0] - FINAL_TEMPERATURE_MEV;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let at_activation = solve(
            kind,
            &system,
            (0.0, 5.0),
            &initial,
            &config(),
            Some(&activation_event),
        );
        assert_eq!(at_activation.failure, None, "{kind:?}: {at_activation:?}");
        assert!(at_activation.event_reached, "{kind:?}: {at_activation:?}");
        let terminal = solve(
            kind,
            &system,
            (at_activation.t, at_activation.t + 8.0),
            &at_activation.y,
            &config(),
            Some(&final_event),
        );
        (at_activation, terminal)
    }

    fn integrate_leading_qed_ccr_freezeout(kind: SolverKind) -> (OdeResult, OdeResult) {
        integrate_leading_qed_weak_freezeout(
            kind,
            BornFreezeoutSystem::high_temperature_instantaneous_decoupling_with_physics(
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                FiniteTemperatureQed::PrimatLeadingE2E3,
                WeakRateModel::PrimatZeroTemperatureCcr,
            ),
            WeakRateModel::PrimatZeroTemperatureCcr,
        )
    }

    fn integrate_leading_qed_ccr_finite_mass_no_weak_magnetism_freezeout(
        kind: SolverKind,
    ) -> (OdeResult, OdeResult) {
        integrate_leading_qed_weak_freezeout(
            kind,
            BornFreezeoutSystem::primat_leading_qed_ccr_finite_mass_no_weak_magnetism(
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            ),
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
        )
    }

    fn integrate_leading_qed_ccr_finite_mass_physical_weak_magnetism_freezeout(
        kind: SolverKind,
    ) -> (OdeResult, OdeResult) {
        let model = WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism;
        integrate_leading_qed_weak_freezeout(
            kind,
            BornFreezeoutSystem::high_temperature_instantaneous_decoupling_with_physics(
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                FiniteTemperatureQed::PrimatLeadingE2E3,
                model,
            ),
            model,
        )
    }

    fn integrate_leading_qed_complete_thermal_radiative_freezeout(
        kind: SolverKind,
    ) -> (OdeResult, OdeResult) {
        let model = WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism;
        integrate_leading_qed_weak_freezeout(
            kind,
            BornFreezeoutSystem::high_temperature_instantaneous_decoupling_with_physics(
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                FiniteTemperatureQed::PrimatLeadingE2E3,
                model,
            ),
            model,
        )
    }

    #[test]
    fn equilibrium_and_boundary_vector_fields_have_physical_signs() {
        let system = BornFreezeoutSystem::common_bath_leg(
            DECOUPLING_TEMPERATURE_MEV,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
        );
        let temperature = 1.0;
        let equilibrium = equilibrium_neutron_fraction(temperature);
        let at_equilibrium = system.derivative(temperature, equilibrium).unwrap();
        let at_zero = system.derivative(temperature, 0.0).unwrap();
        let at_one = system.derivative(temperature, 1.0).unwrap();
        assert!(at_equilibrium.rhs[1].abs() < 2.0e-13);
        assert!(at_zero.rhs[1] > 0.0);
        assert!(at_one.rhs[1] < 0.0);
    }

    #[test]
    fn analytic_neutron_fraction_jacobian_matches_centered_difference() {
        let system = BornFreezeoutSystem::electromagnetic_bath_leg(
            DECOUPLING_TEMPERATURE_MEV,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
        );
        let state = [0.7, 0.2];
        let mut jacobian = [0.0; 4];
        system.jacobian(0.0, &state, &mut jacobian);
        let step = 1.0e-6;
        let plus = system.derivative(state[0], state[1] + step).unwrap();
        let minus = system.derivative(state[0], state[1] - step).unwrap();
        let finite_difference = (plus.rhs[1] - minus.rhs[1]) / (2.0 * step);
        assert!((jacobian[3] - finite_difference).abs() < 2.0e-10);
    }

    #[test]
    fn negative_neutron_fraction_fails_without_clipping() {
        let system = BornFreezeoutSystem::common_bath_leg(
            DECOUPLING_TEMPERATURE_MEV,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
        );
        let initial = [1.0, -0.01];
        let result = solve(
            SolverKind::Rodas5P,
            &system,
            (0.0, 0.1),
            &initial,
            &config(),
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("nonfinite_rhs"));
        assert_eq!(result.y, initial);
    }

    #[test]
    fn both_solvers_reach_the_standalone_freezeout_endpoint() {
        // Independent SciPy 1.17.1 DOP853 integration used T_gamma as the
        // independent variable, adaptive y=p/T EOS quadrature, and direct
        // adaptive Born momentum integrals.  It therefore shares neither this
        // ODE variable nor this fixed Gauss-Legendre implementation.
        let independent_decoupling = (1.610_173_333_678_325_5, 0.349_445_111_642_624_8);
        let independent_terminal = (5.632_995_443_079_69, 0.087_722_120_022_667_17);
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let (at_decoupling, terminal) =
                integrate_freezeout(kind, DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
            assert_eq!(terminal.failure, None, "{kind:?}: {terminal:?}");
            assert!(terminal.event_reached, "{kind:?}: {terminal:?}");
            assert!((at_decoupling.y[0] - DECOUPLING_TEMPERATURE_MEV).abs() < 1.0e-9);
            assert!((terminal.y[0] - FINAL_TEMPERATURE_MEV).abs() < 1.0e-9);
            assert!(terminal.y[1] > 0.08 && terminal.y[1] < 0.2);
            assert!((at_decoupling.t - independent_decoupling.0).abs() < 3.0e-7);
            assert!((at_decoupling.y[1] - independent_decoupling.1).abs() < 3.0e-7);
            assert!((terminal.t - independent_terminal.0).abs() < 3.0e-7);
            assert!((terminal.y[1] - independent_terminal.1).abs() < 3.0e-7);
            endpoints.push((terminal.t, terminal.y[1]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 3.0e-7);
        assert!((endpoints[0].1 - endpoints[1].1).abs() < 2.0e-7);
    }

    #[test]
    fn freezeout_endpoint_converges_from_64_to_96_nodes() {
        let (_, default) = integrate_freezeout(SolverKind::Bdf, DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
        let (_, refined) = integrate_freezeout(SolverKind::Bdf, 96);
        assert_eq!(default.failure, None, "{default:?}");
        assert_eq!(refined.failure, None, "{refined:?}");
        assert!((default.y[1] - refined.y[1]).abs() < 2.0e-9);
        assert!((default.t - refined.t).abs() < 2.0e-9);
    }

    #[test]
    fn leading_qed_ccr_standalone_freezeout_reaches_both_solver_endpoints() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08a = &fixture["f08a_zero_temperature_ccr"];
        let oracle = &f08a["independent_temperature_variable_freezeout"];
        assert_eq!(oracle["execution_status"].as_str(), Some("VALIDATED"));
        let external = &oracle["ccr_baseline"];
        assert_eq!(
            oracle["configuration"]["rust_standalone_rtol"].as_f64(),
            Some(config().rtol)
        );
        assert_eq!(
            oracle["configuration"]["rust_standalone_atol"][0].as_f64(),
            Some(config().atol[0])
        );
        assert_eq!(
            oracle["configuration"]["rust_standalone_atol"][1].as_f64(),
            Some(config().atol[1])
        );
        assert_eq!(
            oracle["rust_acceptance"]["max_absolute_N_difference"].as_f64(),
            Some(EXTERNAL_ORACLE_N_TOLERANCE)
        );
        assert_eq!(
            oracle["rust_acceptance"]["max_absolute_Xn_difference"].as_f64(),
            Some(EXTERNAL_ORACLE_XN_TOLERANCE)
        );
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let (activation, terminal) = integrate_leading_qed_ccr_freezeout(kind);
            assert_eq!(terminal.failure, None, "{kind:?}: {terminal:?}");
            assert!(terminal.event_reached, "{kind:?}: {terminal:?}");
            assert!(
                (activation.y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV).abs()
                    < 2.0e-9
            );
            assert!((terminal.y[0] - FINAL_TEMPERATURE_MEV).abs() < 2.0e-9);
            assert!(terminal.y[1] > 0.08 && terminal.y[1] < 0.2);
            let stored = match kind {
                SolverKind::Bdf => &f08a["rust"]["bdf"],
                SolverKind::Rodas5P => &f08a["rust"]["rodas5p"],
            };
            assert!((activation.y[1] - stored["Xn_handoff"].as_f64().unwrap()).abs() < 2.0e-8);
            assert!(
                (activation.t - external["activation"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (activation.y[1] - external["activation"]["Xn"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_XN_TOLERANCE
            );
            assert!(
                (terminal.t - external["final"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (terminal.y[1] - external["final"]["Xn"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_XN_TOLERANCE
            );
            println!(
                "{kind:?} leading-QED+CCR standalone: activation N={:.16e}, Xn={:.16e}; final N={:.16e}, Xn={:.16e}",
                activation.t, activation.y[1], terminal.t, terminal.y[1]
            );
            endpoints.push((activation.t, activation.y[1], terminal.t, terminal.y[1]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 3.0e-7);
        assert!((endpoints[0].1 - endpoints[1].1).abs() < 2.0e-7);
        assert!((endpoints[0].2 - endpoints[1].2).abs() < 3.0e-7);
        assert!((endpoints[0].3 - endpoints[1].3).abs() < 2.0e-7);
    }

    #[test]
    fn finite_mass_no_weak_magnetism_constructor_dispatches_both_solvers() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08b = &fixture["f08b_finite_nucleon_mass_no_weak_magnetism"];
        let oracle = &f08b["standalone_temperature_variable_freezeout"];
        assert_eq!(
            f08b["schema_version"].as_str(),
            Some("f08b_finite_nucleon_mass_no_weak_magnetism_v1")
        );
        assert_eq!(f08b["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(oracle["execution_status"].as_str(), Some("VALIDATED"));
        assert_eq!(
            oracle["rust_acceptance"]["max_absolute_N_difference"].as_f64(),
            Some(EXTERNAL_ORACLE_N_TOLERANCE)
        );
        assert_eq!(
            oracle["rust_acceptance"]["max_absolute_Xn_difference"].as_f64(),
            Some(FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE)
        );
        let external = &oracle["f08b_gl160"];
        let system = BornFreezeoutSystem::primat_leading_qed_ccr_finite_mass_no_weak_magnetism(
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
        );
        assert_eq!(
            system.weak_rate_model,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism
        );

        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let (activation, terminal) =
                integrate_leading_qed_ccr_finite_mass_no_weak_magnetism_freezeout(kind);
            assert_eq!(terminal.failure, None, "{kind:?}: {terminal:?}");
            assert!(terminal.event_reached, "{kind:?}: {terminal:?}");
            assert!(
                (activation.y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV).abs()
                    < 2.0e-9
            );
            assert!((terminal.y[0] - FINAL_TEMPERATURE_MEV).abs() < 2.0e-9);
            assert!(terminal.y[1] > 0.08 && terminal.y[1] < 0.2);
            let stored = match kind {
                SolverKind::Bdf => &f08b["rust"]["bdf"],
                SolverKind::Rodas5P => &f08b["rust"]["rodas5p"],
            };
            assert!((activation.y[1] - stored["Xn_handoff"].as_f64().unwrap()).abs() < 2.0e-15);
            assert!(
                (activation.t - external["activation"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (activation.y[1] - external["activation"]["Xn"].as_f64().unwrap()).abs()
                    < FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE
            );
            assert!(
                (terminal.t - external["final"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (terminal.y[1] - external["final"]["Xn"].as_f64().unwrap()).abs()
                    < FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE
            );
            println!(
                "{kind:?} leading-QED+CCR+finite-mass(no-WM) standalone: activation N={:.16e}, Xn={:.16e}; final N={:.16e}, Xn={:.16e}",
                activation.t, activation.y[1], terminal.t, terminal.y[1]
            );
            endpoints.push((activation.t, activation.y[1], terminal.t, terminal.y[1]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 3.0e-7);
        assert!((endpoints[0].1 - endpoints[1].1).abs() < 2.0e-7);
        assert!((endpoints[0].2 - endpoints[1].2).abs() < 3.0e-7);
        assert!((endpoints[0].3 - endpoints[1].3).abs() < 2.0e-7);
    }

    #[test]
    fn physical_weak_magnetism_selector_dispatches_both_solvers() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08c = &fixture["f08c_physical_weak_magnetism"];
        let oracle = &f08c["standalone_temperature_variable_freezeout"];
        assert_eq!(
            f08c["schema_version"].as_str(),
            Some("f08c_physical_weak_magnetism_v1")
        );
        assert_eq!(f08c["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08c["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(oracle["execution_status"].as_str(), Some("VALIDATED"));
        assert_eq!(
            f08c["convention"]["weak_magnetism_delta_kappa"].as_f64(),
            Some(PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT)
        );
        assert_eq!(
            oracle["configuration"]["reference_order"].as_u64(),
            Some(320)
        );
        assert!(
            oracle["numerical_stability"]["f08c_max_absolute_Xn_gl320_minus_gl160"]
                .as_f64()
                .unwrap()
                < oracle["numerical_stability"]["required_max_absolute_Xn_gl320_minus_gl160"]
                    .as_f64()
                    .unwrap()
        );
        let acceptance = &oracle["rust_acceptance"];
        assert_eq!(
            acceptance["max_absolute_N_difference"].as_f64(),
            Some(EXTERNAL_ORACLE_N_TOLERANCE)
        );
        assert_eq!(
            acceptance["max_absolute_Xn_difference"].as_f64(),
            Some(FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE)
        );
        assert_eq!(
            acceptance["frozen_before_any_rust_value_was_read"].as_bool(),
            Some(true)
        );
        let external = &oracle["f08c_physical_weak_magnetism_gl320"];
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let (activation, terminal) =
                integrate_leading_qed_ccr_finite_mass_physical_weak_magnetism_freezeout(kind);
            assert_eq!(activation.failure, None, "{kind:?}: {activation:?}");
            assert_eq!(terminal.failure, None, "{kind:?}: {terminal:?}");
            assert!(activation.event_reached, "{kind:?}: {activation:?}");
            assert!(terminal.event_reached, "{kind:?}: {terminal:?}");
            assert!(
                (activation.y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV).abs()
                    < 2.0e-9
            );
            assert!((terminal.y[0] - FINAL_TEMPERATURE_MEV).abs() < 2.0e-9);
            assert!(activation.y[1] > 0.2 && activation.y[1] < 0.3);
            assert!(terminal.y[1] > 0.08 && terminal.y[1] < 0.2);
            let stored = match kind {
                SolverKind::Bdf => &f08c["rust"]["bdf"],
                SolverKind::Rodas5P => &f08c["rust"]["rodas5p"],
            };
            assert!((activation.y[1] - stored["Xn_handoff"].as_f64().unwrap()).abs() < 2.0e-15);
            assert!(
                (activation.t - external["activation"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (activation.y[1] - external["activation"]["Xn"].as_f64().unwrap()).abs()
                    < FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE
            );
            assert!(
                (terminal.t - external["final"]["N"].as_f64().unwrap()).abs()
                    < EXTERNAL_ORACLE_N_TOLERANCE
            );
            assert!(
                (terminal.y[1] - external["final"]["Xn"].as_f64().unwrap()).abs()
                    < FINITE_MASS_EXTERNAL_ORACLE_XN_TOLERANCE
            );
            println!(
                "{kind:?} leading-QED+CCR+finite-mass(physical-WM) standalone: activation N={:.16e}, Xn={:.16e}; final N={:.16e}, Xn={:.16e}",
                activation.t, activation.y[1], terminal.t, terminal.y[1]
            );
            endpoints.push((activation.t, activation.y[1], terminal.t, terminal.y[1]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 3.0e-7);
        assert!((endpoints[0].1 - endpoints[1].1).abs() < 2.0e-7);
        assert!((endpoints[0].2 - endpoints[1].2).abs() < 3.0e-7);
        assert!((endpoints[0].3 - endpoints[1].3).abs() < 2.0e-7);
    }

    #[test]
    fn complete_thermal_radiative_selector_has_finite_boundary_jacobian_and_dual_endpoints() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08d = &fixture["f08d_complete_thermal_radiative"];
        assert_eq!(f08d["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08d["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(f08d["promotion_status"].as_str(), Some("BLOCKED"));
        let standalone = &f08d["standalone_temperature_variable_freezeout"];
        assert_eq!(
            standalone["strict_external_parity_status"].as_str(),
            Some("BLOCKED")
        );
        let model = WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism;
        let system = BornFreezeoutSystem::high_temperature_instantaneous_decoupling_with_physics(
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            FiniteTemperatureQed::PrimatLeadingE2E3,
            model,
        );
        let initial = [
            INITIAL_TEMPERATURE_MEV,
            equilibrium_neutron_fraction(INITIAL_TEMPERATURE_MEV),
        ];
        let mut jacobian = [0.0; 4];
        system.jacobian(0.0, &initial, &mut jacobian);
        assert!(
            jacobian.iter().all(|value| value.is_finite()),
            "F08D 10 MeV in-domain Jacobian={jacobian:?}"
        );

        // Frozen before any Rust F08D endpoint was read.  This comparator is
        // Python global-quadratic interpolation over deterministic C knots;
        // its interpolation spread remains an explicit promotion blocker.
        let external_activation = (
            standalone["external"]["activation"]["N"].as_f64().unwrap(),
            standalone["external"]["activation"]["Xn"].as_f64().unwrap(),
        );
        let external_final = (
            standalone["external"]["final"]["N"].as_f64().unwrap(),
            standalone["external"]["final"]["Xn"].as_f64().unwrap(),
        );
        assert_eq!(
            standalone["frozen_strict_acceptance"]["max_absolute_Xn_difference"].as_f64(),
            Some(F08D_EXTERNAL_ORACLE_XN_TOLERANCE)
        );
        let mut endpoints = Vec::new();
        let mut strict_external_xn_differences = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let (activation, terminal) =
                integrate_leading_qed_complete_thermal_radiative_freezeout(kind);
            assert_eq!(activation.failure, None, "{kind:?}: {activation:?}");
            assert_eq!(terminal.failure, None, "{kind:?}: {terminal:?}");
            assert!(activation.event_reached, "{kind:?}: {activation:?}");
            assert!(terminal.event_reached, "{kind:?}: {terminal:?}");
            println!(
                "{kind:?} leading-QED+F08D standalone: activation N={:.16e}, Xn={:.16e}; final N={:.16e}, Xn={:.16e}",
                activation.t, activation.y[1], terminal.t, terminal.y[1]
            );
            assert!((activation.t - external_activation.0).abs() < EXTERNAL_ORACLE_N_TOLERANCE);
            assert!((terminal.t - external_final.0).abs() < EXTERNAL_ORACLE_N_TOLERANCE);
            assert!(activation.y[1] > 0.2 && activation.y[1] < 0.3);
            assert!(terminal.y[1] > 0.08 && terminal.y[1] < 0.2);
            strict_external_xn_differences.extend([
                (activation.y[1] - external_activation.1).abs(),
                (terminal.y[1] - external_final.1).abs(),
            ]);
            let stored = &standalone["rust"][match kind {
                SolverKind::Bdf => "bdf",
                SolverKind::Rodas5P => "rodas5p",
            }];
            for (actual, key) in [
                (activation.t, "activation_N"),
                (activation.y[1], "activation_Xn"),
                (terminal.t, "final_N"),
                (terminal.y[1], "final_Xn"),
            ] {
                assert!((actual - stored[key].as_f64().unwrap()).abs() < 2.0e-15);
            }
            endpoints.push((activation.t, activation.y[1], terminal.t, terminal.y[1]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 3.0e-7);
        assert!((endpoints[0].1 - endpoints[1].1).abs() < 2.0e-7);
        assert!((endpoints[0].2 - endpoints[1].2).abs() < 3.0e-7);
        assert!((endpoints[0].3 - endpoints[1].3).abs() < 2.0e-7);
        assert!(
            strict_external_xn_differences
                .iter()
                .any(|difference| *difference >= F08D_EXTERNAL_ORACLE_XN_TOLERANCE),
            "strict C-knot/global-quadratic standalone parity unexpectedly passed; review and update the documented F08D interpolation-authority blocker"
        );
    }
}
