//! Endpoint-consumed three-temperature thermal-moment BBN slice.
//!
//! This module couples the F09B finite-vacuum-electron-mass FD/Pauli electron
//! energy moment to a finite-electron-mass, leading-QED FLRW background, the F08C
//! lifetime-normalised weak model, and the selected 31-reaction network.  The
//! weak leg evolves `(T_gamma, T_nue, T_nux, X_n)` from 10 MeV to the PRIMAT
//! network handoff.  A background-only continuation fixes the actual elapsed
//! `ln(a)` to the cold endpoint, so the late baryon-to-photon ratio can anchor
//! `n_b propto a^-3` without borrowing the instantaneous-decoupling entropy
//! map.  The network then consumes the same three-temperature RHS.
//!
//! This is deliberately a bounded thermal-closure foundation.  The collision
//! moment has finite vacuum electron mass and FD/Pauli statistics, but no
//! collision thermal masses, radiative collision corrections, spectral
//! Boltzmann evolution, or QKE.  It excludes F08D's profile-specific
//! thermal-radiative weak table and carries no precision-abundance authority.

#![cfg_attr(not(test), allow(dead_code))]

use core::fmt;

use crate::born_weak::{
    DEFAULT_BORN_WEAK_QUADRATURE_ORDER, DEFAULT_NEUTRON_LIFETIME_SECONDS, WeakRateModel,
    evaluate_weak_rates,
};
use crate::flrw::ThreeTemperatureFlrwSystem;
use crate::minimal_bbn::{
    FINAL_TEMPERATURE_MEV, INITIAL_TEMPERATURE_MEV, LATE_BARYON_TO_PHOTON_RATIO,
    MinimalBbnObservables, ObservableError, PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV,
    observables_from_endpoint, primat_saha_mass_fractions,
};
#[cfg(test)]
use crate::minimal_network::{CHARGE_NUMBERS, MASS_NUMBERS};
use crate::minimal_network::{
    MinimalNetwork, N_SPECIES, NetworkExtent, Species, photon_number_density_per_cm3,
};
use crate::ode::{OdeConfig, OdeResult, OdeSystem, SolverKind, TerminalEvent, solve};
use crate::qed_eos::FiniteTemperatureQed;

const TGAMMA_INDEX: usize = 0;
const TNUE_INDEX: usize = 1;
const TNUX_INDEX: usize = 2;
const NEUTRON_FRACTION_INDEX: usize = 3;
const FREEZEOUT_DIMENSION: usize = 4;

const LOG_BARYON_DENSITY_INDEX: usize = 3;
const ABUNDANCE_START: usize = 4;
const NETWORK_DIMENSION: usize = ABUNDANCE_START + N_SPECIES;

const QED_MODEL: FiniteTemperatureQed = FiniteTemperatureQed::PrimatLeadingE2E3;
const WEAK_MODEL: WeakRateModel =
    WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism;

#[derive(Clone, Debug)]
pub(crate) struct ThermalBbnRun {
    pub(crate) weak_at_activation: OdeResult,
    pub(crate) background_endpoint: OdeResult,
    /// Physical coordinates `(T_gamma, T_nue, T_nux, ln(n_b), X_i...)`.
    pub(crate) network_initial_state: Vec<f64>,
    /// Physical coordinates `(T_gamma, T_nue, T_nux, ln(n_b), X_i...)`.
    pub(crate) network_endpoint: OdeResult,
    pub(crate) expected_terminal_baryon_density_per_cm3: f64,
    /// Unprojected closed-form Saha baryon sum at the handoff.
    pub(crate) raw_saha_baryon_sum: f64,
    /// Single conservation projection applied to every Saha mass fraction.
    pub(crate) saha_projection_factor: f64,
}

#[derive(Clone, Debug)]
pub(crate) enum ThermalBbnRunError {
    Construction(&'static str),
    WeakActivation(Box<OdeResult>),
    BackgroundEndpoint(Box<OdeResult>),
}

impl fmt::Display for ThermalBbnRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Construction(message) => write!(formatter, "construction failure: {message}"),
            Self::WeakActivation(result) => write!(
                formatter,
                "thermal weak activation failure at N={} ({:?})",
                result.t, result.failure
            ),
            Self::BackgroundEndpoint(result) => write!(
                formatter,
                "thermal background endpoint failure at N={} ({:?})",
                result.t, result.failure
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ThermalFreezeoutSystem {
    background: ThreeTemperatureFlrwSystem,
}

impl ThermalFreezeoutSystem {
    const fn new() -> Self {
        Self {
            background: ThreeTemperatureFlrwSystem::new(QED_MODEL),
        }
    }

    fn derivative(&self, state: &[f64]) -> Option<[f64; FREEZEOUT_DIMENSION]> {
        let background = self
            .background
            .thermo_state([state[TGAMMA_INDEX], state[TNUE_INDEX], state[TNUX_INDEX]])
            .ok()?;
        let weak = evaluate_weak_rates(
            state[TGAMMA_INDEX],
            state[TNUE_INDEX],
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            WEAK_MODEL,
        )
        .ok()?;
        let neutron = state[NEUTRON_FRACTION_INDEX];
        let proton = 1.0 - neutron;
        let inverse_hubble = background.h_inverse_seconds.recip();
        let output = [
            background.d_tgamma_d_lna,
            background.d_tnue_d_lna,
            background.d_tnux_d_lna,
            (-weak.neutron_to_proton_per_second * neutron
                + weak.proton_to_neutron_per_second * proton)
                * inverse_hubble,
        ];
        output
            .iter()
            .all(|value| value.is_finite())
            .then_some(output)
    }
}

impl OdeSystem for ThermalFreezeoutSystem {
    fn dimension(&self) -> usize {
        FREEZEOUT_DIMENSION
    }

    fn state_is_valid(&self, state: &[f64]) -> bool {
        state.len() == FREEZEOUT_DIMENSION
            && state[..NEUTRON_FRACTION_INDEX]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
            && state[NEUTRON_FRACTION_INDEX].is_finite()
            && (0.0..=1.0).contains(&state[NEUTRON_FRACTION_INDEX])
    }

    fn rhs(&self, _ln_a: f64, state: &[f64], output: &mut [f64]) {
        match self.derivative(state) {
            Some(derivative) => output.copy_from_slice(&derivative),
            None => output.fill(f64::NAN),
        }
    }

    fn jacobian(&self, ln_a: f64, state: &[f64], output: &mut [f64]) {
        finite_difference_jacobian(self, ln_a, state, output);
    }

    fn dfdt(&self, _ln_a: f64, _state: &[f64], output: &mut [f64]) {
        output.fill(0.0);
    }
}

#[derive(Clone, Copy, Debug)]
struct NetworkDerivative {
    encoded: [f64; NETWORK_DIMENSION],
    abundance_jacobian_per_lna: [f64; N_SPECIES * N_SPECIES],
}

#[derive(Clone, Debug)]
struct ThermalNetworkSystem {
    background: ThreeTemperatureFlrwSystem,
    network: MinimalNetwork,
}

impl ThermalNetworkSystem {
    fn new() -> Result<Self, &'static str> {
        Ok(Self {
            background: ThreeTemperatureFlrwSystem::new(QED_MODEL),
            network: MinimalNetwork::from_embedded_selected_31_table()
                .map_err(|_| "invalid embedded selected-31 network")?,
        })
    }

    fn encode(&self, physical: &[f64]) -> Option<Vec<f64>> {
        if physical.len() != NETWORK_DIMENSION {
            return None;
        }
        let mut encoded = physical.to_vec();
        for value in &mut encoded[ABUNDANCE_START..] {
            if !value.is_finite() || *value <= 0.0 {
                return None;
            }
            *value = value.ln();
        }
        Some(encoded)
    }

    fn decode(&self, encoded: &[f64]) -> Option<Vec<f64>> {
        if encoded.len() != NETWORK_DIMENSION {
            return None;
        }
        let mut physical = encoded.to_vec();
        for value in &mut physical[ABUNDANCE_START..] {
            *value = value.exp();
            if !value.is_finite() || *value <= 0.0 {
                return None;
            }
        }
        Some(physical)
    }

    fn derivative(&self, encoded: &[f64]) -> Option<NetworkDerivative> {
        let physical = self.decode(encoded)?;
        let abundances: [f64; N_SPECIES] = physical[ABUNDANCE_START..].try_into().ok()?;
        let baryon_density = physical[LOG_BARYON_DENSITY_INDEX].exp();
        if !baryon_density.is_finite() || baryon_density <= 0.0 {
            return None;
        }
        let background = self
            .background
            .thermo_state([
                physical[TGAMMA_INDEX],
                physical[TNUE_INDEX],
                physical[TNUX_INDEX],
            ])
            .ok()?;
        let weak = evaluate_weak_rates(
            physical[TGAMMA_INDEX],
            physical[TNUE_INDEX],
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            WEAK_MODEL,
        )
        .ok()?;
        let nuclear = self
            .network
            .stage_rhs_with_baryon_number_density(
                &abundances,
                physical[TGAMMA_INDEX],
                baryon_density,
            )
            .ok()?;
        let nuclear_jacobian = self
            .network
            .stage_jacobian_with_baryon_number_density(
                &abundances,
                physical[TGAMMA_INDEX],
                baryon_density,
            )
            .ok()?;
        let inverse_hubble = background.h_inverse_seconds.recip();
        let neutron = Species::Neutron as usize;
        let proton = Species::Proton as usize;
        let weak_neutron_jacobian = -weak.neutron_to_proton_per_second * inverse_hubble;
        let weak_proton_jacobian = weak.proton_to_neutron_per_second * inverse_hubble;
        let weak_flow =
            weak_neutron_jacobian * abundances[neutron] + weak_proton_jacobian * abundances[proton];

        let mut physical_abundance_derivative = nuclear.map(|value| value * inverse_hubble);
        physical_abundance_derivative[neutron] += weak_flow;
        physical_abundance_derivative[proton] -= weak_flow;

        let mut physical_jacobian = nuclear_jacobian.map(|value| value * inverse_hubble);
        physical_jacobian[neutron * N_SPECIES + neutron] += weak_neutron_jacobian;
        physical_jacobian[neutron * N_SPECIES + proton] += weak_proton_jacobian;
        physical_jacobian[proton * N_SPECIES + neutron] -= weak_neutron_jacobian;
        physical_jacobian[proton * N_SPECIES + proton] -= weak_proton_jacobian;

        let mut encoded_derivative = [0.0; NETWORK_DIMENSION];
        encoded_derivative[TGAMMA_INDEX] = background.d_tgamma_d_lna;
        encoded_derivative[TNUE_INDEX] = background.d_tnue_d_lna;
        encoded_derivative[TNUX_INDEX] = background.d_tnux_d_lna;
        encoded_derivative[LOG_BARYON_DENSITY_INDEX] = -3.0;
        for species in 0..N_SPECIES {
            encoded_derivative[ABUNDANCE_START + species] =
                physical_abundance_derivative[species] / abundances[species];
        }

        let mut encoded_jacobian = [0.0; N_SPECIES * N_SPECIES];
        for row in 0..N_SPECIES {
            for column in 0..N_SPECIES {
                let mut value = physical_jacobian[row * N_SPECIES + column] * abundances[column]
                    / abundances[row];
                if row == column {
                    value -= encoded_derivative[ABUNDANCE_START + row];
                }
                encoded_jacobian[row * N_SPECIES + column] = value;
            }
        }
        (encoded_derivative.iter().all(|value| value.is_finite())
            && encoded_jacobian.iter().all(|value| value.is_finite()))
        .then_some(NetworkDerivative {
            encoded: encoded_derivative,
            abundance_jacobian_per_lna: encoded_jacobian,
        })
    }
}

impl OdeSystem for ThermalNetworkSystem {
    fn dimension(&self) -> usize {
        NETWORK_DIMENSION
    }

    fn state_is_valid(&self, encoded: &[f64]) -> bool {
        encoded.len() == NETWORK_DIMENSION
            && encoded.iter().all(|value| value.is_finite())
            && encoded[..LOG_BARYON_DENSITY_INDEX]
                .iter()
                .all(|value| *value > 0.0)
            && encoded[LOG_BARYON_DENSITY_INDEX].exp().is_finite()
            && encoded[LOG_BARYON_DENSITY_INDEX].exp() > 0.0
            && self.decode(encoded).is_some()
    }

    fn rhs(&self, _ln_a: f64, encoded: &[f64], output: &mut [f64]) {
        match self.derivative(encoded) {
            Some(derivative) => output.copy_from_slice(&derivative.encoded),
            None => output.fill(f64::NAN),
        }
    }

    fn jacobian(&self, ln_a: f64, encoded: &[f64], output: &mut [f64]) {
        output.fill(0.0);
        let Some(center) = self.derivative(encoded) else {
            output.fill(f64::NAN);
            return;
        };

        // Only temperature and density columns need differencing.  The
        // abundance block is the selected network's analytic Jacobian in log
        // coordinates, including the weak n<->p block.
        for column in 0..=LOG_BARYON_DENSITY_INDEX {
            let step = if column == LOG_BARYON_DENSITY_INDEX {
                1.0e-6
            } else {
                (1.0e-5 * encoded[column].abs()).max(1.0e-10)
            };
            let mut plus = encoded.to_vec();
            let mut minus = encoded.to_vec();
            plus[column] += step;
            minus[column] -= step;
            let mut plus_rhs = vec![0.0; NETWORK_DIMENSION];
            let mut minus_rhs = vec![0.0; NETWORK_DIMENSION];
            self.rhs(ln_a, &plus, &mut plus_rhs);
            self.rhs(ln_a, &minus, &mut minus_rhs);
            for row in 0..NETWORK_DIMENSION {
                output[row * NETWORK_DIMENSION + column] =
                    (plus_rhs[row] - minus_rhs[row]) / (2.0 * step);
            }
        }
        for row in 0..N_SPECIES {
            for column in 0..N_SPECIES {
                output[(ABUNDANCE_START + row) * NETWORK_DIMENSION + ABUNDANCE_START + column] =
                    center.abundance_jacobian_per_lna[row * N_SPECIES + column];
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            output.fill(f64::NAN);
        }
    }

    fn dfdt(&self, _ln_a: f64, _state: &[f64], output: &mut [f64]) {
        output.fill(0.0);
    }
}

fn finite_difference_jacobian<S: OdeSystem>(system: &S, t: f64, state: &[f64], output: &mut [f64]) {
    let dimension = state.len();
    for column in 0..dimension {
        let step = (1.0e-5 * state[column].abs()).max(1.0e-10);
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[column] += step;
        minus[column] -= step;
        let mut plus_rhs = vec![0.0; dimension];
        let mut minus_rhs = vec![0.0; dimension];
        system.rhs(t, &plus, &mut plus_rhs);
        system.rhs(t, &minus, &mut minus_rhs);
        for row in 0..dimension {
            output[row * dimension + column] = (plus_rhs[row] - minus_rhs[row]) / (2.0 * step);
        }
    }
}

fn thermal_solver_config(dimension: usize, relative_tolerance: f64) -> OdeConfig {
    OdeConfig {
        rtol: relative_tolerance,
        // All evolved network coordinates other than the temperatures are
        // logarithmic.  Keep the scalar absolute floor below the cold-endpoint
        // temperature tolerance so the declared relative-tolerance ladder is
        // not silently masked at T_gamma = 0.005 MeV.
        atol: vec![1.0e-13; dimension],
        h_init: 1.0e-10,
        h_min: 1.0e-18,
        h_max: 0.02,
        max_attempts: 500_000,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ThermalSolverTolerances {
    /// Accuracy of the collision-coupled weak/freeze-out leg.
    pub(crate) freezeout_rtol: f64,
    /// Accuracy of the background continuation and selected-31 endpoint.
    pub(crate) endpoint_rtol: f64,
}

fn initial_neutron_fraction() -> Result<f64, ThermalBbnRunError> {
    let rates = evaluate_weak_rates(
        INITIAL_TEMPERATURE_MEV,
        INITIAL_TEMPERATURE_MEV,
        DEFAULT_NEUTRON_LIFETIME_SECONDS,
        DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
        WEAK_MODEL,
    )
    .map_err(|_| ThermalBbnRunError::Construction("initial F08C weak rates"))?;
    let sum = rates.neutron_to_proton_per_second + rates.proton_to_neutron_per_second;
    let fraction = rates.proton_to_neutron_per_second / sum;
    if !sum.is_finite() || sum <= 0.0 || !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
        return Err(ThermalBbnRunError::Construction(
            "initial weak-rate equilibrium",
        ));
    }
    Ok(fraction)
}

/// Run the fixed F09B collision-consuming endpoint.  Both tolerances are
/// numerical accuracy only; there is no collision-off or profile selector.
/// Solver failures and event misses remain in the returned raw `OdeResult`s;
/// the observable builder rejects either condition before reading abundances.
pub(crate) fn integrate_thermal_bbn(
    kind: SolverKind,
    tolerances: ThermalSolverTolerances,
) -> Result<ThermalBbnRun, ThermalBbnRunError> {
    let freezeout = ThermalFreezeoutSystem::new();
    let activation_fn = |_ln_a: f64, state: &[f64]| {
        state[TGAMMA_INDEX] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV
    };
    let activation_event = TerminalEvent {
        value: &activation_fn,
        direction: -1,
    };
    let initial = [
        INITIAL_TEMPERATURE_MEV,
        INITIAL_TEMPERATURE_MEV,
        INITIAL_TEMPERATURE_MEV,
        initial_neutron_fraction()?,
    ];
    let weak_at_activation = solve(
        kind,
        &freezeout,
        (0.0, 5.0),
        &initial,
        &thermal_solver_config(FREEZEOUT_DIMENSION, tolerances.freezeout_rtol),
        Some(&activation_event),
    );
    if weak_at_activation.failure.is_some() || !weak_at_activation.event_reached {
        return Err(ThermalBbnRunError::WeakActivation(Box::new(
            weak_at_activation,
        )));
    }

    let background = ThreeTemperatureFlrwSystem::new(QED_MODEL);
    let final_fn = |_ln_a: f64, state: &[f64]| state[TGAMMA_INDEX] - FINAL_TEMPERATURE_MEV;
    let final_event = TerminalEvent {
        value: &final_fn,
        direction: -1,
    };
    let background_endpoint = solve(
        kind,
        &background,
        (weak_at_activation.t, weak_at_activation.t + 8.0),
        &weak_at_activation.y[..3],
        &thermal_solver_config(3, tolerances.endpoint_rtol),
        Some(&final_event),
    );
    if background_endpoint.failure.is_some() || !background_endpoint.event_reached {
        return Err(ThermalBbnRunError::BackgroundEndpoint(Box::new(
            background_endpoint,
        )));
    }

    let final_density = LATE_BARYON_TO_PHOTON_RATIO
        * photon_number_density_per_cm3(background_endpoint.y[TGAMMA_INDEX])
            .map_err(|_| ThermalBbnRunError::Construction("terminal photon density"))?;
    let delta_lna = background_endpoint.t - weak_at_activation.t;
    let activation_density = final_density * (3.0 * delta_lna).exp();
    if !delta_lna.is_finite()
        || delta_lna <= 0.0
        || !activation_density.is_finite()
        || activation_density <= 0.0
    {
        return Err(ThermalBbnRunError::Construction(
            "three-temperature baryon-density anchor",
        ));
    }
    let activation_eta = activation_density
        / photon_number_density_per_cm3(weak_at_activation.y[TGAMMA_INDEX])
            .map_err(|_| ThermalBbnRunError::Construction("activation photon density"))?;
    let mut abundances = primat_saha_mass_fractions(
        weak_at_activation.y[TGAMMA_INDEX],
        activation_eta,
        weak_at_activation.y[NEUTRON_FRACTION_INDEX],
        1.0 - weak_at_activation.y[NEUTRON_FRACTION_INDEX],
        NetworkExtent::Selected31,
    )
    .map_err(ThermalBbnRunError::Construction)?;
    // The existing closed-form Saha helper treats the supplied free-nucleon
    // fractions as leading-order values and then adds bound nuclei.  Project
    // that initial NSE approximation onto the exact baryon simplex once,
    // before integration.  This is deterministic normalisation of an input
    // state, not a runtime abundance floor, clamp, or endpoint repair.
    let saha_baryon_sum = abundances.iter().sum::<f64>();
    if !saha_baryon_sum.is_finite() || saha_baryon_sum <= 0.0 {
        return Err(ThermalBbnRunError::Construction(
            "thermal Saha baryon normalisation",
        ));
    }
    let saha_projection_factor = saha_baryon_sum.recip();
    for abundance in &mut abundances {
        *abundance *= saha_projection_factor;
    }

    let system = ThermalNetworkSystem::new().map_err(ThermalBbnRunError::Construction)?;
    let mut network_initial_state = vec![0.0; NETWORK_DIMENSION];
    network_initial_state[..3].copy_from_slice(&weak_at_activation.y[..3]);
    network_initial_state[LOG_BARYON_DENSITY_INDEX] = activation_density.ln();
    network_initial_state[ABUNDANCE_START..].copy_from_slice(&abundances);
    let encoded_initial =
        system
            .encode(&network_initial_state)
            .ok_or(ThermalBbnRunError::Construction(
                "thermal network log-coordinate state",
            ))?;
    let mut network_endpoint = solve(
        kind,
        &system,
        (weak_at_activation.t, weak_at_activation.t + 8.0),
        &encoded_initial,
        &thermal_solver_config(NETWORK_DIMENSION, tolerances.endpoint_rtol),
        Some(&final_event),
    );
    match system.decode(&network_endpoint.y) {
        Some(physical) => network_endpoint.y = physical,
        None => {
            network_endpoint.failure = Some("invalid_log_coordinate_state".to_string());
            network_endpoint.event_reached = false;
        }
    }

    Ok(ThermalBbnRun {
        weak_at_activation,
        background_endpoint,
        network_initial_state,
        network_endpoint,
        expected_terminal_baryon_density_per_cm3: final_density,
        raw_saha_baryon_sum: saha_baryon_sum,
        saha_projection_factor,
    })
}

/// Reuse the locked minimal-BBN observable definitions after an explicit
/// coordinate projection; no abundance is clipped or renormalised.
pub(crate) fn thermal_observables_from_endpoint(
    endpoint: &OdeResult,
) -> Result<MinimalBbnObservables, ObservableError> {
    if endpoint.y.len() != NETWORK_DIMENSION {
        return Err(ObservableError::InvalidStateLength);
    }
    let mut projected = endpoint.clone();
    projected.y = Vec::with_capacity(2 + N_SPECIES);
    projected.y.push(endpoint.y[TGAMMA_INDEX]);
    projected.y.push(endpoint.y[LOG_BARYON_DENSITY_INDEX]);
    projected
        .y
        .extend_from_slice(&endpoint.y[ABUNDANCE_START..]);
    observables_from_endpoint(&projected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    const DEFAULT_TOLERANCES: ThermalSolverTolerances = ThermalSolverTolerances {
        freezeout_rtol: 2.0e-8,
        endpoint_rtol: 1.0e-10,
    };
    const TIGHT_TOLERANCES: ThermalSolverTolerances = ThermalSolverTolerances {
        freezeout_rtol: 6.0e-9,
        endpoint_rtol: 3.0e-11,
    };
    const DUAL_SOLVER_TEMPERATURE_RELATIVE_BUDGET: f64 = 1.0e-7;
    const DUAL_SOLVER_OBSERVABLE_RELATIVE_BUDGETS: [f64; 5] = [2.0e-6; 5];
    const TIGHT_BDF_STATE_RELATIVE_BUDGET: f64 = 1.0e-6;
    const TIGHT_BDF_OBSERVABLE_RELATIVE_BUDGETS: [f64; 5] = [1.0e-6; 5];
    // Measured Rust regression anchors frozen after the first successful F09B
    // endpoint. They are not independent physics acceptance values.
    const RAW_SAHA_BARYON_SUM_ANCHOR: f64 = 1.000_000_000_001_778;
    const SAHA_PROJECTION_FACTOR_ANCHOR: f64 = 0.999_999_999_998_222_1;
    const SAHA_ANCHOR_ABSOLUTE_BUDGET: f64 = 4.0e-15;

    fn assert_success_and_invariants(run: &ThermalBbnRun) {
        assert_eq!(run.network_initial_state.len(), NETWORK_DIMENSION);
        assert!(run.raw_saha_baryon_sum > 1.0);
        assert!(
            (run.raw_saha_baryon_sum - RAW_SAHA_BARYON_SUM_ANCHOR).abs()
                <= SAHA_ANCHOR_ABSOLUTE_BUDGET
        );
        assert!(
            (run.saha_projection_factor - SAHA_PROJECTION_FACTOR_ANCHOR).abs()
                <= SAHA_ANCHOR_ABSOLUTE_BUDGET
        );
        assert_eq!(
            run.saha_projection_factor.to_bits(),
            run.raw_saha_baryon_sum.recip().to_bits()
        );
        assert!(
            (run.network_initial_state[ABUNDANCE_START..]
                .iter()
                .sum::<f64>()
                - 1.0)
                .abs()
                <= 4.0 * f64::EPSILON
        );
        for endpoint in [
            &run.weak_at_activation,
            &run.background_endpoint,
            &run.network_endpoint,
        ] {
            assert_eq!(endpoint.failure, None, "raw endpoint failure: {endpoint:?}");
            assert!(
                endpoint.event_reached,
                "terminal event missing: {endpoint:?}"
            );
        }
        for state in [
            &run.weak_at_activation.y[..3],
            &run.background_endpoint.y[..3],
            &run.network_endpoint.y[..3],
        ] {
            assert!(state.iter().all(|value| value.is_finite() && *value > 0.0));
            assert!(state[TNUE_INDEX] > state[TNUX_INDEX]);
        }
        let abundances = &run.network_endpoint.y[ABUNDANCE_START..];
        assert!(
            abundances
                .iter()
                .all(|value| value.is_finite() && *value > 0.0)
        );
        assert!(
            (abundances.iter().sum::<f64>() - 1.0).abs() <= 5.0e-9,
            "raw baryon residual={}",
            abundances.iter().sum::<f64>() - 1.0
        );
        let charge_per_baryon = abundances
            .iter()
            .enumerate()
            .map(|(species, abundance)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * abundance)
            .sum::<f64>();
        assert!(charge_per_baryon.is_finite() && (0.0..=1.0).contains(&charge_per_baryon));
        let terminal_density = run.network_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
        assert!(
            (terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0).abs() <= 2.0e-8,
            "late-density residual={}",
            terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0
        );
        thermal_observables_from_endpoint(&run.network_endpoint).unwrap();
    }

    fn observable_array(run: &ThermalBbnRun) -> [f64; 5] {
        let value = thermal_observables_from_endpoint(&run.network_endpoint).unwrap();
        [
            value.helium4_mass_fraction,
            value.deuterium_to_hydrogen,
            value.helium3_to_hydrogen,
            value.lithium7_plus_beryllium7_to_hydrogen,
            value.lithium6_to_hydrogen,
        ]
    }

    fn assert_relative_budget(left: &[f64], right: &[f64], budgets: &[f64]) {
        for ((left, right), budget) in left.iter().zip(right).zip(budgets) {
            let residual = (left / right - 1.0).abs();
            assert!(residual <= *budget, "residual={residual} budget={budget}");
        }
    }

    fn assert_bitwise_same_result(first: &OdeResult, repeat: &OdeResult) {
        assert_eq!(first.t.to_bits(), repeat.t.to_bits());
        assert_eq!(
            first
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeat
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            (
                first.accepted,
                first.rejected,
                first.jacobian_evaluations,
                first.linear_setups,
                first.event_reached,
                &first.failure,
            ),
            (
                repeat.accepted,
                repeat.rejected,
                repeat.jacobian_evaluations,
                repeat.linear_setups,
                repeat.event_reached,
                &repeat.failure,
            )
        );
    }

    fn report_endpoint(label: &str, run: &ThermalBbnRun, wall: Duration) {
        let observables = observable_array(run);
        println!(
            "F09B {label}: wall={:.6}s; activation N={:.17e} T=[{:.17e},{:.17e},{:.17e}] Xn={:.17e} steps={}/{}; background N={:.17e} T=[{:.17e},{:.17e},{:.17e}] steps={}/{}; network N={:.17e} T=[{:.17e},{:.17e},{:.17e}] steps={}/{}; observables=[{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}]",
            wall.as_secs_f64(),
            run.weak_at_activation.t,
            run.weak_at_activation.y[TGAMMA_INDEX],
            run.weak_at_activation.y[TNUE_INDEX],
            run.weak_at_activation.y[TNUX_INDEX],
            run.weak_at_activation.y[NEUTRON_FRACTION_INDEX],
            run.weak_at_activation.accepted,
            run.weak_at_activation.rejected,
            run.background_endpoint.t,
            run.background_endpoint.y[TGAMMA_INDEX],
            run.background_endpoint.y[TNUE_INDEX],
            run.background_endpoint.y[TNUX_INDEX],
            run.background_endpoint.accepted,
            run.background_endpoint.rejected,
            run.network_endpoint.t,
            run.network_endpoint.y[TGAMMA_INDEX],
            run.network_endpoint.y[TNUE_INDEX],
            run.network_endpoint.y[TNUX_INDEX],
            run.network_endpoint.accepted,
            run.network_endpoint.rejected,
            observables[0],
            observables[1],
            observables[2],
            observables[3],
            observables[4],
        );
    }

    #[test]
    fn f09b_coupled_log_jacobian_matches_five_point_difference() {
        let run = integrate_thermal_bbn(SolverKind::Bdf, DEFAULT_TOLERANCES).unwrap();
        let system = ThermalNetworkSystem::new().unwrap();
        let encoded = system.encode(&run.network_initial_state).unwrap();
        assert!(
            system.derivative(&encoded).is_some(),
            "representative coupled state has no finite derivative"
        );

        let mut analytic = vec![0.0; NETWORK_DIMENSION * NETWORK_DIMENSION];
        system.jacobian(0.0, &encoded, &mut analytic);
        assert!(
            analytic.iter().all(|value| value.is_finite()),
            "first non-finite analytic cell: {:?}",
            analytic.iter().position(|value| !value.is_finite())
        );

        for column in 0..NETWORK_DIMENSION {
            let step = 2.0e-6 * encoded[column].abs().max(1.0);
            let mut plus_two = encoded.clone();
            let mut plus_one = encoded.clone();
            let mut minus_one = encoded.clone();
            let mut minus_two = encoded.clone();
            plus_two[column] += 2.0 * step;
            plus_one[column] += step;
            minus_one[column] -= step;
            minus_two[column] -= 2.0 * step;
            let mut f_plus_two = vec![0.0; NETWORK_DIMENSION];
            let mut f_plus_one = vec![0.0; NETWORK_DIMENSION];
            let mut f_minus_one = vec![0.0; NETWORK_DIMENSION];
            let mut f_minus_two = vec![0.0; NETWORK_DIMENSION];
            system.rhs(0.0, &plus_two, &mut f_plus_two);
            system.rhs(0.0, &plus_one, &mut f_plus_one);
            system.rhs(0.0, &minus_one, &mut f_minus_one);
            system.rhs(0.0, &minus_two, &mut f_minus_two);
            let reference = (0..NETWORK_DIMENSION)
                .map(|row| {
                    (-f_plus_two[row] + 8.0 * f_plus_one[row] - 8.0 * f_minus_one[row]
                        + f_minus_two[row])
                        / (12.0 * step)
                })
                .collect::<Vec<_>>();
            let column_scale = reference
                .iter()
                .chain(analytic.iter().skip(column).step_by(NETWORK_DIMENSION))
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
            for row in 0..NETWORK_DIMENSION {
                let actual = analytic[row * NETWORK_DIMENSION + column];
                let expected = reference[row];
                let scale = actual.abs().max(expected.abs()).max(1.0e-10 * column_scale);
                let residual = (actual - expected).abs() / scale;
                assert!(
                    residual <= 5.0e-4,
                    "row={row} column={column} actual={actual:.17e} expected={expected:.17e} residual={residual:.17e}"
                );
            }
        }
    }

    #[test]
    fn f09b_bdf_and_rodas_reach_collision_consuming_selected31_endpoint() {
        let bdf_start = Instant::now();
        let bdf = integrate_thermal_bbn(SolverKind::Bdf, DEFAULT_TOLERANCES).unwrap();
        let bdf_wall = bdf_start.elapsed();
        let rodas_start = Instant::now();
        let rodas = integrate_thermal_bbn(SolverKind::Rodas5P, DEFAULT_TOLERANCES).unwrap();
        let rodas_wall = rodas_start.elapsed();
        println!(
            "F09B raw Saha sums/projections: BDF=[{:.17e},{:.17e}] Rodas5P=[{:.17e},{:.17e}]",
            bdf.raw_saha_baryon_sum,
            bdf.saha_projection_factor,
            rodas.raw_saha_baryon_sum,
            rodas.saha_projection_factor
        );
        assert_success_and_invariants(&bdf);
        assert_success_and_invariants(&rodas);
        assert_relative_budget(
            &bdf.network_endpoint.y[..3],
            &rodas.network_endpoint.y[..3],
            &[DUAL_SOLVER_TEMPERATURE_RELATIVE_BUDGET; 3],
        );
        assert_relative_budget(
            &observable_array(&bdf),
            &observable_array(&rodas),
            &DUAL_SOLVER_OBSERVABLE_RELATIVE_BUDGETS,
        );
        report_endpoint("BDF_freezeout2e-8_endpoint1e-10", &bdf, bdf_wall);
        report_endpoint("Rodas5P_freezeout2e-8_endpoint1e-10", &rodas, rodas_wall);
    }

    #[test]
    fn f09b_tighter_bdf_converges_and_immediate_repeat_is_bitwise_deterministic() {
        let first_start = Instant::now();
        let first = integrate_thermal_bbn(SolverKind::Bdf, DEFAULT_TOLERANCES).unwrap();
        let first_wall = first_start.elapsed();
        let repeat_start = Instant::now();
        let repeat = integrate_thermal_bbn(SolverKind::Bdf, DEFAULT_TOLERANCES).unwrap();
        let repeat_wall = repeat_start.elapsed();
        assert_bitwise_same_result(&first.weak_at_activation, &repeat.weak_at_activation);
        assert_bitwise_same_result(&first.background_endpoint, &repeat.background_endpoint);
        assert_bitwise_same_result(&first.network_endpoint, &repeat.network_endpoint);

        let tight_start = Instant::now();
        let tight = integrate_thermal_bbn(SolverKind::Bdf, TIGHT_TOLERANCES).unwrap();
        let tight_wall = tight_start.elapsed();
        assert_success_and_invariants(&tight);
        assert_relative_budget(
            &[
                first.weak_at_activation.t,
                first.background_endpoint.t,
                first.network_endpoint.t,
            ],
            &[
                tight.weak_at_activation.t,
                tight.background_endpoint.t,
                tight.network_endpoint.t,
            ],
            &[TIGHT_BDF_STATE_RELATIVE_BUDGET; 3],
        );
        assert_relative_budget(
            &first.weak_at_activation.y,
            &tight.weak_at_activation.y,
            &[TIGHT_BDF_STATE_RELATIVE_BUDGET; FREEZEOUT_DIMENSION],
        );
        assert_relative_budget(
            &first.background_endpoint.y,
            &tight.background_endpoint.y,
            &[TIGHT_BDF_STATE_RELATIVE_BUDGET; 3],
        );
        assert_relative_budget(
            &first.network_endpoint.y[..3],
            &tight.network_endpoint.y[..3],
            &[TIGHT_BDF_STATE_RELATIVE_BUDGET; 3],
        );
        assert_relative_budget(
            &observable_array(&first),
            &observable_array(&tight),
            &TIGHT_BDF_OBSERVABLE_RELATIVE_BUDGETS,
        );
        report_endpoint("BDF_freezeout2e-8_endpoint1e-10_first", &first, first_wall);
        report_endpoint(
            "BDF_freezeout2e-8_endpoint1e-10_repeat",
            &repeat,
            repeat_wall,
        );
        report_endpoint("BDF_freezeout6e-9_endpoint3e-11", &tight, tight_wall);
    }
}
