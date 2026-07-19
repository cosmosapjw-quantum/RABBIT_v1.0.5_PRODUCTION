//! Private coupled Rust BBN foundation built bottom-up from F-02 through F-08N.
//!
//! The phase-2 state is `(T_gamma, ln(n_b/cm^-3), X_n, X_p, X_D, X_T,
//! X_He3, X_He4, X_Li6, X_Li7, X_Be7)` versus `N=ln(a/a_initial)`.
//! The configured baryon-to-photon ratio is a late-time value: `n_b` is
//! anchored at the cold terminal temperature and evolves exactly as `a^-3`.
//! An explicit switch optionally propagates the scalar finite-temperature QED
//! EOS through the background, weak handoff, entropy-normalized baryon density,
//! and nuclear endpoint.  A second switch can apply the zero-temperature
//! Coulomb/radiative weak correction and the finite-nucleon-mass correction
//! with weak magnetism either disabled or set to PRIMAT's physical magnetic-
//! moment coefficient through the same lifetime-normalized six-channel weak
//! leg.  The complete four-term finite-temperature radiative correction can
//! then be added as a separate signed directional contribution without
//! assigning its multi-body processes to a fictitious Born channel.  The
//! named network is selectable between the original 12-reaction backbone and
//! the accepted 31-row AC2024 subset. Baryon/nuclear Hubble backreaction,
//! neutrino collisions, shear, and tuned abundance seeds remain absent.

#![cfg_attr(not(test), allow(dead_code))]

use crate::born_freezeout::BornFreezeoutSystem;
#[cfg(test)]
use crate::born_weak::DEFAULT_BORN_WEAK_QUADRATURE_ORDER;
use crate::born_weak::{
    DEFAULT_NEUTRON_LIFETIME_SECONDS, NEUTRON_PROTON_MASS_DIFFERENCE_MEV, WeakRateModel,
    evaluate_weak_rates,
};
#[cfg(test)]
use crate::flrw::electromagnetic_eos;
use crate::flrw::{IdealFlrwSystem, electromagnetic_eos_for_qed};
#[cfg(test)]
use crate::minimal_network::N_BACKBONE_REACTIONS;
use crate::minimal_network::{
    CHARGE_NUMBERS, MASS_NUMBERS, MinimalNetwork, N_SPECIES, NetworkExtent, Species,
    photon_number_density_per_cm3,
};
use crate::ode::{OdeConfig, OdeResult, OdeSystem, SolverKind, TerminalEvent, solve};
use crate::qed_eos::FiniteTemperatureQed;
use core::fmt;
use std::f64::consts::PI;

const TEMPERATURE_INDEX: usize = 0;
const LOG_BARYON_DENSITY_INDEX: usize = 1;
const ABUNDANCE_START: usize = 2;
const STATE_DIMENSION: usize = ABUNDANCE_START + N_SPECIES;

pub(crate) const INITIAL_TEMPERATURE_MEV: f64 = 10.0;
pub(crate) const DECOUPLING_TEMPERATURE_MEV: f64 = 2.0;
pub(crate) const NETWORK_ACTIVATION_TEMPERATURE_MEV: f64 = 0.08;
// 0.9999e10 K: just inside the 10 GK upper edge of the AC2024 table.  PRIMAT
// v0.3.2 starts its MT solve at 1 MeV and linearly extrapolates above that
// edge; the Rust authority path keeps the raw table-domain failure contract
// and instead extends the weak-only leg to this physically equivalent NSE
// handoff.
pub(crate) const PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV: f64 = 0.861_647_152_867_38;
pub(crate) const FINAL_TEMPERATURE_MEV: f64 = 0.005;
pub(crate) const LATE_BARYON_TO_PHOTON_RATIO: f64 = 6.1e-10;
const NORMALIZATION_TOLERANCE: f64 = 2.0e-7;

/// One consolidated selector for the matched standard-physics endpoint.
///
/// Keeping the EOS and weak model together prevents a new public wrapper for
/// every staged correction while preserving the historical Born/off entry
/// points below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatchedStandardPhysics {
    qed_model: FiniteTemperatureQed,
    weak_rate_model: WeakRateModel,
}

impl MatchedStandardPhysics {
    pub(crate) const fn new(
        qed_model: FiniteTemperatureQed,
        weak_rate_model: WeakRateModel,
    ) -> Self {
        Self {
            qed_model,
            weak_rate_model,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MinimalBbnSystem {
    background: IdealFlrwSystem,
    network: MinimalNetwork,
    neutron_lifetime_seconds: f64,
    born_quadrature_order: usize,
    weak_rate_model: WeakRateModel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MinimalBbnObservables {
    pub(crate) helium4_mass_fraction: f64,
    pub(crate) deuterium_to_hydrogen: f64,
    pub(crate) helium3_to_hydrogen: f64,
    pub(crate) lithium7_plus_beryllium7_to_hydrogen: f64,
    pub(crate) lithium6_to_hydrogen: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ObservableError {
    SolverFailed,
    TerminalEventMissing,
    InvalidStateLength,
    NonFiniteState,
    NegativeState { index: usize },
    UnnormalizedState { residual: f64 },
    NonPositiveHydrogen,
    NonFiniteObservable,
}

#[derive(Clone, Debug)]
pub(crate) struct MinimalBbnRun {
    pub(crate) activation_temperature_mev: f64,
    pub(crate) weak_at_decoupling: OdeResult,
    pub(crate) weak_at_activation: OdeResult,
    pub(crate) coupled_initial_state: Vec<f64>,
    pub(crate) coupled_endpoint: OdeResult,
    pub(crate) expected_terminal_baryon_density_per_cm3: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct PrimatMatchedMinimalBbnRun {
    /// Exact staged-physics selector used by both the weak and coupled legs.
    pub(crate) physics: MatchedStandardPhysics,
    pub(crate) network_extent: NetworkExtent,
    pub(crate) weak_at_activation: OdeResult,
    pub(crate) coupled_initial_state: Vec<f64>,
    pub(crate) coupled_endpoint: OdeResult,
    pub(crate) expected_terminal_baryon_density_per_cm3: f64,
}

#[derive(Clone, Debug)]
pub(crate) enum MinimalBbnRunError {
    Construction(&'static str),
    WeakDecoupling(Box<OdeResult>),
    MatchedWeak(Box<OdeResult>),
    WeakActivation {
        at_decoupling: Box<OdeResult>,
        at_activation: Box<OdeResult>,
    },
}

impl fmt::Display for MinimalBbnRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Construction(message) => write!(formatter, "construction failure: {message}"),
            Self::WeakDecoupling(result) => write!(
                formatter,
                "weak decoupling failure at N={} ({:?})",
                result.t, result.failure
            ),
            Self::MatchedWeak(result) => write!(
                formatter,
                "matched weak failure at N={} ({:?})",
                result.t, result.failure
            ),
            Self::WeakActivation {
                at_decoupling,
                at_activation,
            } => write!(
                formatter,
                "weak activation failure after N={} at N={} ({:?})",
                at_decoupling.t, at_activation.t, at_activation.failure
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CoupledDerivative {
    values: [f64; STATE_DIMENSION],
    abundance_jacobian_per_lna: [f64; N_SPECIES * N_SPECIES],
    weak_neutron_derivative: f64,
    weak_proton_derivative: f64,
}

impl MinimalBbnSystem {
    pub(crate) fn new(
        neutron_lifetime_seconds: f64,
        born_quadrature_order: usize,
    ) -> Result<Self, &'static str> {
        if !neutron_lifetime_seconds.is_finite() || neutron_lifetime_seconds <= 0.0 {
            return Err("invalid neutron lifetime");
        }
        Self::with_background(
            IdealFlrwSystem::electromagnetic_bath_leg(DECOUPLING_TEMPERATURE_MEV),
            neutron_lifetime_seconds,
            born_quadrature_order,
        )
    }

    fn primat_matched_with_physics_and_network(
        neutron_lifetime_seconds: f64,
        born_quadrature_order: usize,
        physics: MatchedStandardPhysics,
        network_extent: NetworkExtent,
    ) -> Result<Self, &'static str> {
        let background = if physics.qed_model == FiniteTemperatureQed::Off {
            IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling()
        } else {
            IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(physics.qed_model)
        };
        Self::with_background_weak_and_network(
            background,
            neutron_lifetime_seconds,
            born_quadrature_order,
            physics.weak_rate_model,
            network_extent,
        )
    }

    fn with_background(
        background: IdealFlrwSystem,
        neutron_lifetime_seconds: f64,
        born_quadrature_order: usize,
    ) -> Result<Self, &'static str> {
        Self::with_background_and_weak(
            background,
            neutron_lifetime_seconds,
            born_quadrature_order,
            WeakRateModel::Born,
        )
    }

    fn with_background_and_weak(
        background: IdealFlrwSystem,
        neutron_lifetime_seconds: f64,
        born_quadrature_order: usize,
        weak_rate_model: WeakRateModel,
    ) -> Result<Self, &'static str> {
        Self::with_background_weak_and_network(
            background,
            neutron_lifetime_seconds,
            born_quadrature_order,
            weak_rate_model,
            NetworkExtent::Backbone12,
        )
    }

    fn with_background_weak_and_network(
        background: IdealFlrwSystem,
        neutron_lifetime_seconds: f64,
        born_quadrature_order: usize,
        weak_rate_model: WeakRateModel,
        network_extent: NetworkExtent,
    ) -> Result<Self, &'static str> {
        if !neutron_lifetime_seconds.is_finite() || neutron_lifetime_seconds <= 0.0 {
            return Err("invalid neutron lifetime");
        }
        let network = match network_extent {
            NetworkExtent::Backbone12 => MinimalNetwork::from_embedded_canonical_table(),
            NetworkExtent::Selected31 => MinimalNetwork::from_embedded_selected_31_table(),
        }
        .map_err(|_| "invalid embedded named network")?;
        Ok(Self {
            background,
            network,
            neutron_lifetime_seconds,
            born_quadrature_order,
            weak_rate_model,
        })
    }

    fn unpack_abundances(state: &[f64]) -> Option<[f64; N_SPECIES]> {
        if state.len() != STATE_DIMENSION {
            return None;
        }
        state[ABUNDANCE_START..].try_into().ok()
    }

    fn derivative(&self, state: &[f64]) -> Option<CoupledDerivative> {
        let abundances = Self::unpack_abundances(state)?;
        if !state[TEMPERATURE_INDEX].is_finite()
            || !state[LOG_BARYON_DENSITY_INDEX].is_finite()
            || abundances.iter().any(|value| !value.is_finite())
        {
            return None;
        }
        let temperature = state[TEMPERATURE_INDEX];
        let baryon_density = state[LOG_BARYON_DENSITY_INDEX].exp();
        if !baryon_density.is_finite() || baryon_density <= 0.0 {
            return None;
        }
        let background = self.background.thermo_state(temperature).ok()?;
        let weak = evaluate_weak_rates(
            temperature,
            background.t_nu_mev,
            self.neutron_lifetime_seconds,
            self.born_quadrature_order,
            self.weak_rate_model,
        )
        .ok()?;
        let nuclear_per_second = self
            .network
            .stage_rhs_with_baryon_number_density(&abundances, temperature, baryon_density)
            .ok()?;
        let nuclear_jacobian_per_second = self
            .network
            .stage_jacobian_with_baryon_number_density(&abundances, temperature, baryon_density)
            .ok()?;
        let inverse_hubble = background.h_inverse_seconds.recip();
        let mut values = [0.0; STATE_DIMENSION];
        values[TEMPERATURE_INDEX] = background.d_tgamma_d_lna;
        values[LOG_BARYON_DENSITY_INDEX] = -3.0;
        for species in 0..N_SPECIES {
            values[ABUNDANCE_START + species] = nuclear_per_second[species] * inverse_hubble;
        }
        let neutron = abundances[Species::Neutron as usize];
        let proton = abundances[Species::Proton as usize];
        let weak_neutron_derivative = -weak.neutron_to_proton_per_second * inverse_hubble;
        let weak_proton_derivative = weak.proton_to_neutron_per_second * inverse_hubble;
        let weak_flow = weak_neutron_derivative * neutron + weak_proton_derivative * proton;
        values[ABUNDANCE_START + Species::Neutron as usize] += weak_flow;
        values[ABUNDANCE_START + Species::Proton as usize] -= weak_flow;
        let abundance_jacobian_per_lna =
            nuclear_jacobian_per_second.map(|value| value * inverse_hubble);
        if values.iter().any(|value| !value.is_finite())
            || abundance_jacobian_per_lna
                .iter()
                .any(|value| !value.is_finite())
        {
            return None;
        }
        Some(CoupledDerivative {
            values,
            abundance_jacobian_per_lna,
            weak_neutron_derivative,
            weak_proton_derivative,
        })
    }
}

impl OdeSystem for MinimalBbnSystem {
    fn dimension(&self) -> usize {
        STATE_DIMENSION
    }

    fn state_is_valid(&self, state: &[f64]) -> bool {
        Self::unpack_abundances(state).is_some_and(|abundances| {
            state[TEMPERATURE_INDEX].is_finite()
                && state[TEMPERATURE_INDEX] > 0.0
                && state[LOG_BARYON_DENSITY_INDEX].is_finite()
                && state[LOG_BARYON_DENSITY_INDEX].exp().is_finite()
                && state[LOG_BARYON_DENSITY_INDEX].exp() > 0.0
                && abundances
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
        })
    }

    fn rhs(&self, _ln_a: f64, state: &[f64], output: &mut [f64]) {
        match self.derivative(state) {
            Some(derivative) => output.copy_from_slice(&derivative.values),
            None => output.fill(f64::NAN),
        }
    }

    fn jacobian(&self, _ln_a: f64, state: &[f64], output: &mut [f64]) {
        let Some(center) = self.derivative(state) else {
            output.fill(f64::NAN);
            return;
        };
        output.fill(0.0);

        for column in [TEMPERATURE_INDEX, LOG_BARYON_DENSITY_INDEX] {
            let step = if column == TEMPERATURE_INDEX {
                (1.0e-5 * state[column].abs()).max(1.0e-10)
            } else {
                1.0e-6
            };
            let mut plus = state.to_vec();
            let mut minus = state.to_vec();
            plus[column] += step;
            minus[column] -= step;
            let (Some(plus), Some(minus)) = (self.derivative(&plus), self.derivative(&minus))
            else {
                output.fill(f64::NAN);
                return;
            };
            for row in 0..STATE_DIMENSION {
                output[row * STATE_DIMENSION + column] =
                    (plus.values[row] - minus.values[row]) / (2.0 * step);
            }
        }

        for row in 0..N_SPECIES {
            for column in 0..N_SPECIES {
                output[(ABUNDANCE_START + row) * STATE_DIMENSION + ABUNDANCE_START + column] =
                    center.abundance_jacobian_per_lna[row * N_SPECIES + column];
            }
        }
        let neutron = Species::Neutron as usize;
        let proton = Species::Proton as usize;
        output[(ABUNDANCE_START + neutron) * STATE_DIMENSION + ABUNDANCE_START + neutron] +=
            center.weak_neutron_derivative;
        output[(ABUNDANCE_START + neutron) * STATE_DIMENSION + ABUNDANCE_START + proton] +=
            center.weak_proton_derivative;
        output[(ABUNDANCE_START + proton) * STATE_DIMENSION + ABUNDANCE_START + neutron] -=
            center.weak_neutron_derivative;
        output[(ABUNDANCE_START + proton) * STATE_DIMENSION + ABUNDANCE_START + proton] -=
            center.weak_proton_derivative;
    }

    fn dfdt(&self, _ln_a: f64, _state: &[f64], output: &mut [f64]) {
        output.fill(0.0);
    }
}

/// Positive-coordinate form used only by the high-temperature NSE-started
/// comparison lane. Every species active in the selected network is evolved
/// as `ln X`; Li6 remains an exact zero spectator only for the 12-reaction
/// backbone. This prevents an implicit stage from turning a physically tiny
/// NSE seed negative without clipping or flooring any abundance.
#[derive(Clone, Debug)]
struct PrimatMatchedLogSystem {
    physical: MinimalBbnSystem,
    evolves_lithium6: bool,
}

impl PrimatMatchedLogSystem {
    fn new(physical: MinimalBbnSystem) -> Self {
        let evolves_lithium6 = physical.network.extent().evolves_lithium6();
        Self {
            physical,
            evolves_lithium6,
        }
    }

    fn encode(&self, physical: &[f64]) -> Option<Vec<f64>> {
        if physical.len() != STATE_DIMENSION {
            return None;
        }
        let mut encoded = physical.to_vec();
        for species in Species::ALL {
            let index = ABUNDANCE_START + species as usize;
            if species == Species::Lithium6 && !self.evolves_lithium6 {
                if physical[index] != 0.0 {
                    return None;
                }
                encoded[index] = 0.0;
            } else {
                if !physical[index].is_finite() || physical[index] <= 0.0 {
                    return None;
                }
                encoded[index] = physical[index].ln();
            }
        }
        Some(encoded)
    }

    fn decode(&self, encoded: &[f64]) -> Option<Vec<f64>> {
        if encoded.len() != STATE_DIMENSION {
            return None;
        }
        let mut physical = encoded.to_vec();
        for species in Species::ALL {
            let index = ABUNDANCE_START + species as usize;
            if species == Species::Lithium6 && !self.evolves_lithium6 {
                physical[index] = 0.0;
            } else {
                physical[index] = encoded[index].exp();
                if !physical[index].is_finite() || physical[index] <= 0.0 {
                    return None;
                }
            }
        }
        Some(physical)
    }
}

impl OdeSystem for PrimatMatchedLogSystem {
    fn dimension(&self) -> usize {
        STATE_DIMENSION
    }

    fn state_is_valid(&self, encoded: &[f64]) -> bool {
        encoded.len() == STATE_DIMENSION
            && encoded.iter().all(|value| value.is_finite())
            && encoded[TEMPERATURE_INDEX] > 0.0
            && encoded[LOG_BARYON_DENSITY_INDEX].exp().is_finite()
            && encoded[LOG_BARYON_DENSITY_INDEX].exp() > 0.0
            && (self.evolves_lithium6
                || encoded[ABUNDANCE_START + Species::Lithium6 as usize] == 0.0)
    }

    fn rhs(&self, _ln_a: f64, encoded: &[f64], output: &mut [f64]) {
        let Some(physical) = self.decode(encoded) else {
            output.fill(f64::NAN);
            return;
        };
        let Some(derivative) = self.physical.derivative(&physical) else {
            output.fill(f64::NAN);
            return;
        };
        output[TEMPERATURE_INDEX] = derivative.values[TEMPERATURE_INDEX];
        output[LOG_BARYON_DENSITY_INDEX] = derivative.values[LOG_BARYON_DENSITY_INDEX];
        for species in Species::ALL {
            let index = ABUNDANCE_START + species as usize;
            output[index] = if species == Species::Lithium6 && !self.evolves_lithium6 {
                0.0
            } else {
                derivative.values[index] / physical[index]
            };
        }
        if output.iter().any(|value| !value.is_finite()) {
            output.fill(f64::NAN);
        }
    }

    fn jacobian(&self, ln_a: f64, encoded: &[f64], output: &mut [f64]) {
        let Some(physical) = self.decode(encoded) else {
            output.fill(f64::NAN);
            return;
        };
        let Some(derivative) = self.physical.derivative(&physical) else {
            output.fill(f64::NAN);
            return;
        };
        let mut physical_jacobian = vec![0.0; STATE_DIMENSION * STATE_DIMENSION];
        self.physical
            .jacobian(ln_a, &physical, &mut physical_jacobian);
        if physical_jacobian.iter().any(|value| !value.is_finite()) {
            output.fill(f64::NAN);
            return;
        }
        output.fill(0.0);

        for row in [TEMPERATURE_INDEX, LOG_BARYON_DENSITY_INDEX] {
            for column in [TEMPERATURE_INDEX, LOG_BARYON_DENSITY_INDEX] {
                output[row * STATE_DIMENSION + column] =
                    physical_jacobian[row * STATE_DIMENSION + column];
            }
            for species in Species::ALL {
                if species == Species::Lithium6 && !self.evolves_lithium6 {
                    continue;
                }
                let column = ABUNDANCE_START + species as usize;
                output[row * STATE_DIMENSION + column] =
                    physical_jacobian[row * STATE_DIMENSION + column] * physical[column];
            }
        }

        for row_species in Species::ALL {
            if row_species == Species::Lithium6 && !self.evolves_lithium6 {
                continue;
            }
            let row = ABUNDANCE_START + row_species as usize;
            let row_abundance = physical[row];
            for column in [TEMPERATURE_INDEX, LOG_BARYON_DENSITY_INDEX] {
                output[row * STATE_DIMENSION + column] =
                    physical_jacobian[row * STATE_DIMENSION + column] / row_abundance;
            }
            for column_species in Species::ALL {
                if column_species == Species::Lithium6 && !self.evolves_lithium6 {
                    continue;
                }
                let column = ABUNDANCE_START + column_species as usize;
                let mut value = physical_jacobian[row * STATE_DIMENSION + column]
                    * physical[column]
                    / row_abundance;
                if row == column {
                    value -= derivative.values[row] / row_abundance;
                }
                output[row * STATE_DIMENSION + column] = value;
            }
        }
        if output.iter().any(|value| !value.is_finite()) {
            output.fill(f64::NAN);
        }
    }

    fn dfdt(&self, _ln_a: f64, _encoded: &[f64], output: &mut [f64]) {
        output.fill(0.0);
    }
}

fn weak_solver_config() -> OdeConfig {
    OdeConfig {
        rtol: 2.0e-8,
        atol: vec![1.0e-10, 1.0e-11],
        h_init: 1.0e-5,
        h_min: 1.0e-13,
        h_max: 0.03,
        max_attempts: 100_000,
    }
}

fn coupled_solver_config(relative_tolerance: f64) -> OdeConfig {
    OdeConfig {
        rtol: relative_tolerance,
        atol: vec![
            1.0e-10, 1.0e-10, 1.0e-11, 1.0e-11, 1.0e-18, 1.0e-18, 1.0e-18, 1.0e-15, 1.0e-25,
            1.0e-22, 1.0e-22,
        ],
        h_init: 1.0e-12,
        h_min: 1.0e-18,
        h_max: 0.02,
        max_attempts: 500_000,
    }
}

fn equilibrium_neutron_fraction(temperature_mev: f64) -> f64 {
    1.0 / (1.0 + (NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp())
}

const PRIMAT_ATOMIC_MASS_UNIT_MEV: f64 = 931.494_061;
const PRIMAT_NEUTRON_MASS_MEV: f64 = 939.565_420_52;
const PRIMAT_PROTON_MASS_MEV: f64 = 938.272_088_16;
const PRIMAT_MASS_EXCESS_KEV: [f64; N_SPECIES] = [
    8_071.318_1,
    7_288.971_064,
    13_135.722_895,
    14_949.810_9,
    14_931.218_88,
    2_424.915_87,
    14_086.880_400,
    14_907.105,
    15_769.0,
];
const PRIMAT_SPINS: [f64; N_SPECIES] = [0.5, 0.5, 1.0, 0.5, 0.5, 0.0, 1.0, 1.5, 1.5];

/// PRIMAT-form NSE seed, converted from number-per-baryon abundances `Y_A` to
/// this module's mass fractions `X_A=A Y_A`. The selected-31 extent adds Li6
/// with the NUBASE2020 mass excess `14086.880400 keV` and ground-state spin 1;
/// the 12-reaction comparison keeps its historical exact-zero spectator.
pub(crate) fn primat_saha_mass_fractions(
    temperature_mev: f64,
    baryon_to_photon_ratio: f64,
    neutron_fraction: f64,
    proton_fraction: f64,
    network_extent: NetworkExtent,
) -> Result<[f64; N_SPECIES], &'static str> {
    if !temperature_mev.is_finite()
        || temperature_mev <= 0.0
        || !baryon_to_photon_ratio.is_finite()
        || baryon_to_photon_ratio <= 0.0
        || !neutron_fraction.is_finite()
        || neutron_fraction < 0.0
        || !proton_fraction.is_finite()
        || proton_fraction < 0.0
    {
        return Err("invalid Saha input");
    }
    let mut mass_fractions = [0.0; N_SPECIES];
    mass_fractions[Species::Neutron as usize] = neutron_fraction;
    mass_fractions[Species::Proton as usize] = proton_fraction;
    for species in Species::ALL {
        if matches!(species, Species::Neutron | Species::Proton)
            || (species == Species::Lithium6 && !network_extent.evolves_lithium6())
        {
            continue;
        }
        let index = species as usize;
        let a = MASS_NUMBERS[index] as i32;
        let z = CHARGE_NUMBERS[index] as i32;
        let n = a - z;
        let excess_kev = PRIMAT_MASS_EXCESS_KEV[index];
        let nuclear_mass_mev = a as f64 * PRIMAT_ATOMIC_MASS_UNIT_MEV + excess_kev * 1.0e-3
            - z as f64 * crate::flrw::ELECTRON_MASS_MEV;
        let binding_mev = (n as f64 * PRIMAT_MASS_EXCESS_KEV[Species::Neutron as usize]
            + z as f64 * PRIMAT_MASS_EXCESS_KEV[Species::Proton as usize]
            - excess_kev)
            * 1.0e-3;
        let mass_temperature_factor = (nuclear_mass_mev
            / (PRIMAT_NEUTRON_MASS_MEV.powi(n) * PRIMAT_PROTON_MASS_MEV.powi(z)))
        .powf(1.5)
            * temperature_mev.powf(1.5 * (a - 1) as f64);
        let number_abundance = (2.0 * PRIMAT_SPINS[index] + 1.0)
            * crate::minimal_network::APERY_ZETA_THREE.powi(a - 1)
            * PI.powf((1 - a) as f64 / 2.0)
            * 2.0_f64.powf((3 * a - 5) as f64 / 2.0)
            * mass_temperature_factor
            * baryon_to_photon_ratio.powi(a - 1)
            * proton_fraction.powi(z)
            * neutron_fraction.powi(n)
            * (binding_mev / temperature_mev).exp();
        let mass_fraction = a as f64 * number_abundance;
        if !mass_fraction.is_finite() || mass_fraction < 0.0 {
            return Err("invalid Saha abundance");
        }
        mass_fractions[index] = mass_fraction;
    }
    Ok(mass_fractions)
}

fn baryon_density_at_activation(
    activation_temperature_mev: f64,
) -> Result<(f64, f64), &'static str> {
    baryon_density_at_activation_with_qed(activation_temperature_mev, FiniteTemperatureQed::Off)
}

fn baryon_density_at_activation_with_qed(
    activation_temperature_mev: f64,
    qed_model: FiniteTemperatureQed,
) -> Result<(f64, f64), &'static str> {
    if !activation_temperature_mev.is_finite()
        || activation_temperature_mev <= FINAL_TEMPERATURE_MEV
        || activation_temperature_mev >= INITIAL_TEMPERATURE_MEV
    {
        return Err("invalid activation temperature");
    }
    let entropy_activation = electromagnetic_eos_for_qed(activation_temperature_mev, qed_model)
        .map_err(|_| "activation entropy failure")?
        .entropy;
    let entropy_final = electromagnetic_eos_for_qed(FINAL_TEMPERATURE_MEV, qed_model)
        .map_err(|_| "terminal entropy failure")?
        .entropy;
    let delta_lna = (entropy_activation / entropy_final).ln() / 3.0;
    let final_density = LATE_BARYON_TO_PHOTON_RATIO
        * photon_number_density_per_cm3(FINAL_TEMPERATURE_MEV)
            .map_err(|_| "terminal photon density failure")?;
    let activation_density = final_density * (3.0 * delta_lna).exp();
    if !delta_lna.is_finite()
        || delta_lna <= 0.0
        || !activation_density.is_finite()
        || activation_density <= 0.0
    {
        return Err("invalid baryon-density anchor");
    }
    Ok((activation_density, final_density))
}

fn integrate_weak_to_activation(
    kind: SolverKind,
    born_quadrature_order: usize,
    activation_temperature_mev: f64,
) -> Result<(OdeResult, OdeResult), MinimalBbnRunError> {
    let common = BornFreezeoutSystem::common_bath_leg(
        DECOUPLING_TEMPERATURE_MEV,
        DEFAULT_NEUTRON_LIFETIME_SECONDS,
        born_quadrature_order,
    );
    let electromagnetic = BornFreezeoutSystem::electromagnetic_bath_leg(
        DECOUPLING_TEMPERATURE_MEV,
        DEFAULT_NEUTRON_LIFETIME_SECONDS,
        born_quadrature_order,
    );
    let decoupling_fn = |_ln_a: f64, state: &[f64]| state[0] - DECOUPLING_TEMPERATURE_MEV;
    let decoupling = TerminalEvent {
        value: &decoupling_fn,
        direction: -1,
    };
    let activation_fn = |_ln_a: f64, state: &[f64]| state[0] - activation_temperature_mev;
    let activation = TerminalEvent {
        value: &activation_fn,
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
        &weak_solver_config(),
        Some(&decoupling),
    );
    if at_decoupling.failure.is_some() || !at_decoupling.event_reached {
        return Err(MinimalBbnRunError::WeakDecoupling(Box::new(at_decoupling)));
    }
    let at_activation = solve(
        kind,
        &electromagnetic,
        (at_decoupling.t, at_decoupling.t + 5.0),
        &at_decoupling.y,
        &weak_solver_config(),
        Some(&activation),
    );
    if at_activation.failure.is_some() || !at_activation.event_reached {
        return Err(MinimalBbnRunError::WeakActivation {
            at_decoupling: Box::new(at_decoupling),
            at_activation: Box::new(at_activation),
        });
    }
    Ok((at_decoupling, at_activation))
}

fn integrate_primat_matched_weak_to_activation_with_physics(
    kind: SolverKind,
    born_quadrature_order: usize,
    physics: MatchedStandardPhysics,
) -> Result<OdeResult, MinimalBbnRunError> {
    let background = if physics.qed_model == FiniteTemperatureQed::Off {
        IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling()
    } else {
        IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(physics.qed_model)
    };
    let initial_background = background
        .thermo_state(INITIAL_TEMPERATURE_MEV)
        .map_err(|_| MinimalBbnRunError::Construction("matched initial FLRW state"))?;
    let initial_rates = evaluate_weak_rates(
        INITIAL_TEMPERATURE_MEV,
        initial_background.t_nu_mev,
        DEFAULT_NEUTRON_LIFETIME_SECONDS,
        born_quadrature_order,
        physics.weak_rate_model,
    )
    .map_err(|_| MinimalBbnRunError::Construction("matched initial weak rates"))?;
    let rate_sum =
        initial_rates.neutron_to_proton_per_second + initial_rates.proton_to_neutron_per_second;
    if !rate_sum.is_finite() || rate_sum <= 0.0 {
        return Err(MinimalBbnRunError::Construction(
            "matched initial weak equilibrium",
        ));
    }
    let initial_neutron_fraction = initial_rates.proton_to_neutron_per_second / rate_sum;
    let system = if physics.qed_model == FiniteTemperatureQed::Off
        && physics.weak_rate_model == WeakRateModel::Born
    {
        BornFreezeoutSystem::ideal_high_temperature_instantaneous_decoupling(
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            born_quadrature_order,
        )
    } else {
        BornFreezeoutSystem::high_temperature_instantaneous_decoupling_with_physics(
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            born_quadrature_order,
            physics.qed_model,
            physics.weak_rate_model,
        )
    };
    let activation_fn =
        |_ln_a: f64, state: &[f64]| state[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV;
    let activation = TerminalEvent {
        value: &activation_fn,
        direction: -1,
    };
    let endpoint = solve(
        kind,
        &system,
        (0.0, 5.0),
        &[INITIAL_TEMPERATURE_MEV, initial_neutron_fraction],
        &weak_solver_config(),
        Some(&activation),
    );
    if endpoint.failure.is_some() || !endpoint.event_reached {
        return Err(MinimalBbnRunError::MatchedWeak(Box::new(endpoint)));
    }
    Ok(endpoint)
}

pub(crate) fn integrate_primat_matched_minimal_bbn(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    integrate_primat_matched_minimal_bbn_with_physics(
        kind,
        born_quadrature_order,
        relative_tolerance,
        MatchedStandardPhysics::new(FiniteTemperatureQed::Off, WeakRateModel::Born),
    )
}

pub(crate) fn integrate_primat_matched_minimal_bbn_with_qed(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
    qed_model: FiniteTemperatureQed,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    integrate_primat_matched_minimal_bbn_with_physics(
        kind,
        born_quadrature_order,
        relative_tolerance,
        MatchedStandardPhysics::new(qed_model, WeakRateModel::Born),
    )
}

/// Integrate the matched leading-QED/CCR endpoint with the F08B finite-mass
/// correction and weak magnetism explicitly disabled.
pub(crate) fn integrate_primat_matched_minimal_bbn_with_ccr_finite_mass_no_weak_magnetism(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    integrate_primat_matched_minimal_bbn_with_physics(
        kind,
        born_quadrature_order,
        relative_tolerance,
        MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
        ),
    )
}

pub(crate) fn integrate_primat_matched_minimal_bbn_with_physics(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
    physics: MatchedStandardPhysics,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    integrate_primat_matched_bbn_with_physics_and_network(
        kind,
        born_quadrature_order,
        relative_tolerance,
        physics,
        NetworkExtent::Backbone12,
    )
}

pub(crate) fn integrate_primat_matched_selected_31_bbn_with_physics(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
    physics: MatchedStandardPhysics,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    integrate_primat_matched_bbn_with_physics_and_network(
        kind,
        born_quadrature_order,
        relative_tolerance,
        physics,
        NetworkExtent::Selected31,
    )
}

fn integrate_primat_matched_bbn_with_physics_and_network(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
    physics: MatchedStandardPhysics,
    network_extent: NetworkExtent,
) -> Result<PrimatMatchedMinimalBbnRun, MinimalBbnRunError> {
    let weak_at_activation = integrate_primat_matched_weak_to_activation_with_physics(
        kind,
        born_quadrature_order,
        physics,
    )?;
    let physical_system = MinimalBbnSystem::primat_matched_with_physics_and_network(
        DEFAULT_NEUTRON_LIFETIME_SECONDS,
        born_quadrature_order,
        physics,
        network_extent,
    )
    .map_err(MinimalBbnRunError::Construction)?;
    let system = PrimatMatchedLogSystem::new(physical_system);
    let (activation_density, final_density) = baryon_density_at_activation_with_qed(
        PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV,
        physics.qed_model,
    )
    .map_err(MinimalBbnRunError::Construction)?;
    let activation_eta = activation_density
        / photon_number_density_per_cm3(PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV)
            .map_err(|_| MinimalBbnRunError::Construction("matched activation photon density"))?;
    let abundances = primat_saha_mass_fractions(
        PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV,
        activation_eta,
        weak_at_activation.y[1],
        1.0 - weak_at_activation.y[1],
        network_extent,
    )
    .map_err(MinimalBbnRunError::Construction)?;
    let mut initial = vec![0.0; STATE_DIMENSION];
    initial[TEMPERATURE_INDEX] = PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV;
    initial[LOG_BARYON_DENSITY_INDEX] = activation_density.ln();
    initial[ABUNDANCE_START..].copy_from_slice(&abundances);
    let encoded_initial = system
        .encode(&initial)
        .ok_or(MinimalBbnRunError::Construction(
            "matched log-coordinate state",
        ))?;
    let final_event_fn = |_ln_a: f64, state: &[f64]| state[0] - FINAL_TEMPERATURE_MEV;
    let final_event = TerminalEvent {
        value: &final_event_fn,
        direction: -1,
    };
    let mut coupled_endpoint = solve(
        kind,
        &system,
        (weak_at_activation.t, weak_at_activation.t + 8.0),
        &encoded_initial,
        &coupled_solver_config(relative_tolerance),
        Some(&final_event),
    );
    match system.decode(&coupled_endpoint.y) {
        Some(physical_endpoint) => coupled_endpoint.y = physical_endpoint,
        None => {
            coupled_endpoint.failure = Some("invalid_log_coordinate_state".to_string());
            coupled_endpoint.event_reached = false;
        }
    }
    Ok(PrimatMatchedMinimalBbnRun {
        physics,
        network_extent,
        weak_at_activation,
        coupled_initial_state: initial,
        coupled_endpoint,
        expected_terminal_baryon_density_per_cm3: final_density,
    })
}

pub(crate) fn integrate_minimal_bbn(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
) -> Result<MinimalBbnRun, MinimalBbnRunError> {
    integrate_minimal_bbn_at_activation(
        kind,
        born_quadrature_order,
        relative_tolerance,
        NETWORK_ACTIVATION_TEMPERATURE_MEV,
    )
}

fn integrate_minimal_bbn_at_activation(
    kind: SolverKind,
    born_quadrature_order: usize,
    relative_tolerance: f64,
    activation_temperature_mev: f64,
) -> Result<MinimalBbnRun, MinimalBbnRunError> {
    let (at_decoupling, at_activation) =
        integrate_weak_to_activation(kind, born_quadrature_order, activation_temperature_mev)?;
    let system = MinimalBbnSystem::new(DEFAULT_NEUTRON_LIFETIME_SECONDS, born_quadrature_order)
        .map_err(MinimalBbnRunError::Construction)?;
    let (activation_density, final_density) =
        baryon_density_at_activation(activation_temperature_mev)
            .map_err(MinimalBbnRunError::Construction)?;
    let mut initial = vec![0.0; STATE_DIMENSION];
    initial[TEMPERATURE_INDEX] = activation_temperature_mev;
    initial[LOG_BARYON_DENSITY_INDEX] = activation_density.ln();
    initial[ABUNDANCE_START + Species::Neutron as usize] = at_activation.y[1];
    initial[ABUNDANCE_START + Species::Proton as usize] = 1.0 - at_activation.y[1];
    let final_event_fn = |_ln_a: f64, state: &[f64]| state[0] - FINAL_TEMPERATURE_MEV;
    let final_event = TerminalEvent {
        value: &final_event_fn,
        direction: -1,
    };
    let coupled_endpoint = solve(
        kind,
        &system,
        (at_activation.t, at_activation.t + 8.0),
        &initial,
        &coupled_solver_config(relative_tolerance),
        Some(&final_event),
    );
    Ok(MinimalBbnRun {
        activation_temperature_mev,
        weak_at_decoupling: at_decoupling,
        weak_at_activation: at_activation,
        coupled_initial_state: initial,
        coupled_endpoint,
        expected_terminal_baryon_density_per_cm3: final_density,
    })
}

pub(crate) fn observables_from_endpoint(
    endpoint: &OdeResult,
) -> Result<MinimalBbnObservables, ObservableError> {
    if endpoint.failure.is_some() {
        return Err(ObservableError::SolverFailed);
    }
    if !endpoint.event_reached {
        return Err(ObservableError::TerminalEventMissing);
    }
    if endpoint.y.len() != STATE_DIMENSION {
        return Err(ObservableError::InvalidStateLength);
    }
    if endpoint.y.iter().any(|value| !value.is_finite()) {
        return Err(ObservableError::NonFiniteState);
    }
    let abundances = &endpoint.y[ABUNDANCE_START..];
    if let Some((index, _)) = abundances
        .iter()
        .enumerate()
        .find(|(_, value)| **value < 0.0)
    {
        return Err(ObservableError::NegativeState { index });
    }
    let residual = (abundances.iter().sum::<f64>() - 1.0).abs();
    if residual > NORMALIZATION_TOLERANCE {
        return Err(ObservableError::UnnormalizedState { residual });
    }
    let hydrogen = abundances[Species::Proton as usize];
    if hydrogen <= 0.0 {
        return Err(ObservableError::NonPositiveHydrogen);
    }
    let observables = MinimalBbnObservables {
        helium4_mass_fraction: abundances[Species::Helium4 as usize],
        deuterium_to_hydrogen: abundances[Species::Deuterium as usize] / (2.0 * hydrogen),
        helium3_to_hydrogen: (abundances[Species::Helium3 as usize]
            + abundances[Species::Tritium as usize])
            / (3.0 * hydrogen),
        lithium7_plus_beryllium7_to_hydrogen: (abundances[Species::Lithium7 as usize]
            + abundances[Species::Beryllium7 as usize])
            / (7.0 * hydrogen),
        lithium6_to_hydrogen: abundances[Species::Lithium6 as usize] / (6.0 * hydrogen),
    };
    if [
        observables.helium4_mass_fraction,
        observables.deuterium_to_hydrogen,
        observables.helium3_to_hydrogen,
        observables.lithium7_plus_beryllium7_to_hydrogen,
        observables.lithium6_to_hydrogen,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(ObservableError::NonFiniteObservable);
    }
    Ok(observables)
}

pub(crate) fn baryon_residual(state: &[f64]) -> Option<f64> {
    let abundances = MinimalBbnSystem::unpack_abundances(state)?;
    Some((abundances.iter().sum::<f64>() - 1.0).abs())
}

pub(crate) fn nuclear_charge_per_baryon(state: &[f64]) -> Option<f64> {
    let abundances = MinimalBbnSystem::unpack_abundances(state)?;
    Some(
        abundances
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn test_system(order: usize) -> MinimalBbnSystem {
        MinimalBbnSystem::new(DEFAULT_NEUTRON_LIFETIME_SECONDS, order).unwrap()
    }

    fn positive_fixed_state() -> [f64; STATE_DIMENSION] {
        let (activation_density, _) =
            baryon_density_at_activation(NETWORK_ACTIVATION_TEMPERATURE_MEV).unwrap();
        let mut state = [0.0; STATE_DIMENSION];
        state[TEMPERATURE_INDEX] = 0.1;
        state[LOG_BARYON_DENSITY_INDEX] = (activation_density * 0.1).ln();
        state[ABUNDANCE_START..]
            .copy_from_slice(&[0.12, 0.50, 0.08, 0.05, 0.04, 0.16, 0.005, 0.02, 0.025]);
        state
    }

    #[test]
    fn late_eta_anchor_and_a_cubed_density_history_are_exact() {
        let (activation_density, final_density) =
            baryon_density_at_activation(NETWORK_ACTIVATION_TEMPERATURE_MEV).unwrap();
        let entropy_activation = electromagnetic_eos(NETWORK_ACTIVATION_TEMPERATURE_MEV)
            .unwrap()
            .entropy;
        let entropy_final = electromagnetic_eos(FINAL_TEMPERATURE_MEV).unwrap().entropy;
        let delta_lna = (entropy_activation / entropy_final).ln() / 3.0;
        assert!(
            (activation_density * (-3.0 * delta_lna).exp() / final_density - 1.0).abs() < 2.0e-15
        );
        assert!(
            (final_density
                / photon_number_density_per_cm3(FINAL_TEMPERATURE_MEV).unwrap()
                / LATE_BARYON_TO_PHOTON_RATIO
                - 1.0)
                .abs()
                < 2.0e-15
        );
    }

    #[test]
    fn selected_31_saha_adds_only_authorized_positive_lithium6_seed() {
        assert_eq!(
            PRIMAT_MASS_EXCESS_KEV[Species::Lithium6 as usize],
            14_086.880_400
        );
        assert_eq!(PRIMAT_SPINS[Species::Lithium6 as usize], 1.0);
        let temperature = PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV;
        let (density, _) = baryon_density_at_activation_with_qed(
            temperature,
            FiniteTemperatureQed::PrimatLeadingE2E3,
        )
        .unwrap();
        let eta = density / photon_number_density_per_cm3(temperature).unwrap();
        let backbone =
            primat_saha_mass_fractions(temperature, eta, 0.24, 0.76, NetworkExtent::Backbone12)
                .unwrap();
        let selected =
            primat_saha_mass_fractions(temperature, eta, 0.24, 0.76, NetworkExtent::Selected31)
                .unwrap();
        for species in Species::ALL {
            let index = species as usize;
            if species == Species::Lithium6 {
                assert_eq!(backbone[index], 0.0);
                assert!(selected[index].is_finite() && selected[index] > 0.0);
            } else {
                assert_eq!(selected[index], backbone[index]);
            }
        }
    }

    #[test]
    fn selected_31_log_coordinate_abundance_jacobian_matches_five_point_difference() {
        let physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism,
        );
        let physical = MinimalBbnSystem::primat_matched_with_physics_and_network(
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            physics,
            NetworkExtent::Selected31,
        )
        .unwrap();
        let system = PrimatMatchedLogSystem::new(physical);
        let encoded = system.encode(&positive_fixed_state()).unwrap();
        let mut analytic = vec![0.0; STATE_DIMENSION * STATE_DIMENSION];
        system.jacobian(0.0, &encoded, &mut analytic);
        assert!(analytic.iter().all(|value| value.is_finite()));

        let mut resolved_entries = 0_usize;
        let mut resolved_lithium6_row_entries = 0_usize;
        for column_species in Species::ALL {
            let column = ABUNDANCE_START + column_species as usize;
            let five_point = |step: f64| {
                let rhs_at = |offset: f64| {
                    let mut state = encoded.clone();
                    state[column] += offset * step;
                    let mut rhs = vec![0.0; STATE_DIMENSION];
                    system.rhs(0.0, &state, &mut rhs);
                    assert!(rhs.iter().all(|value| value.is_finite()));
                    rhs
                };
                let minus_two = rhs_at(-2.0);
                let minus_one = rhs_at(-1.0);
                let plus_one = rhs_at(1.0);
                let plus_two = rhs_at(2.0);
                let derivative = (0..STATE_DIMENSION)
                    .map(|row| {
                        (-plus_two[row] + 8.0 * plus_one[row] - 8.0 * minus_one[row]
                            + minus_two[row])
                            / (12.0 * step)
                    })
                    .collect::<Vec<_>>();
                let cancellation_bound = (0..STATE_DIMENSION)
                    .map(|row| {
                        64.0 * f64::EPSILON
                            * (plus_two[row].abs()
                                + 8.0 * plus_one[row].abs()
                                + 8.0 * minus_one[row].abs()
                                + minus_two[row].abs())
                            / (12.0 * step)
                    })
                    .collect::<Vec<_>>();
                (derivative, cancellation_bound)
            };
            // A smaller 1e-5/1e-3 stencil is cancellation-dominated for the
            // largest log-RHS entries.  These 1% and 0.5% log perturbations
            // keep the O(h^4) five-point truncation small while providing an
            // explicit stencil-refinement check.
            let (coarse, coarse_roundoff) = five_point(1.0e-2);
            let (refined, refined_roundoff) = five_point(5.0e-3);
            for row_species in Species::ALL {
                let row = ABUNDANCE_START + row_species as usize;
                let exact = analytic[row * STATE_DIMENSION + column];
                let scale = exact.abs().max(refined[row].abs()).max(coarse[row].abs());
                assert!(
                    (coarse[row] - refined[row]).abs()
                        <= 3.0e-6 * scale + coarse_roundoff[row] + refined_roundoff[row],
                    "row={row_species:?} column={column_species:?}: coarse={:.17e}, refined={:.17e}, cancellation_bounds=[{:.17e},{:.17e}]",
                    coarse[row],
                    refined[row],
                    coarse_roundoff[row],
                    refined_roundoff[row]
                );
                if scale > 16.0 * refined_roundoff[row].max(f64::MIN_POSITIVE) {
                    resolved_entries += 1;
                    if row_species == Species::Lithium6 {
                        resolved_lithium6_row_entries += 1;
                    }
                    assert!(
                        (exact - refined[row]).abs() <= 2.0e-6 * scale + refined_roundoff[row],
                        "row={row_species:?} column={column_species:?}: analytic={exact:.17e}, refined_five_point={:.17e}, cancellation_bound={:.17e}",
                        refined[row],
                        refined_roundoff[row]
                    );
                } else {
                    assert!(
                        exact.abs() <= 16.0 * refined_roundoff[row] + 1.0e-300,
                        "unresolved row={row_species:?} column={column_species:?}: analytic={exact:.17e}, cancellation_bound={:.17e}",
                        refined_roundoff[row]
                    );
                }
            }
        }
        assert!(resolved_entries >= N_SPECIES * N_SPECIES / 2);
        assert!(resolved_lithium6_row_entries >= N_SPECIES / 2);
    }

    #[test]
    fn fixed_state_rhs_has_cooling_density_and_baryon_invariants() {
        let system = test_system(DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
        let state = positive_fixed_state();
        let derivative = system.derivative(&state).unwrap();
        assert!(derivative.values[TEMPERATURE_INDEX] < 0.0);
        assert_eq!(derivative.values[LOG_BARYON_DENSITY_INDEX], -3.0);
        // Frozen integration regression after the independently anchored EOS,
        // Born-rate, and direct-Python nuclear-rate component tests.  This
        // combined vector is not a second independent physics validation.
        let independent_rhs_per_lna: [f64; STATE_DIMENSION] = [
            -8.333_900_568_019_657e-2,
            -3.0,
            3.092_919_700_170_706_5e5,
            3.103_897_451_476_886e5,
            -6.198_696_901_081_904e5,
            7.184_582_071_610_607e2,
            8.680_202_691_901_74e4,
            1.179_015_603_078_361_6e5,
            0.0,
            1.655_779_292_927_473_5e3,
            -2.068_898_497_835_109_6e5,
        ];
        for (actual, expected) in derivative.values.iter().zip(independent_rhs_per_lna) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() < 2.0e-9 * scale);
        }
        let abundance_sum: f64 = derivative.values[ABUNDANCE_START..].iter().sum();
        let scale: f64 = derivative.values[ABUNDANCE_START..]
            .iter()
            .map(|value| value.abs())
            .sum();
        assert!(abundance_sum.abs() < 3.0e-15 * scale);

        let abundances = MinimalBbnSystem::unpack_abundances(&state).unwrap();
        let baryon_density = state[LOG_BARYON_DENSITY_INDEX].exp();
        let nuclear_per_second = system
            .network
            .rhs_with_baryon_number_density(&abundances, state[TEMPERATURE_INDEX], baryon_density)
            .unwrap();
        let nuclear_charge_derivative: f64 = nuclear_per_second
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum();
        let nuclear_scale: f64 = nuclear_per_second.iter().map(|value| value.abs()).sum();
        assert!(nuclear_charge_derivative.abs() < 3.0e-15 * nuclear_scale);

        let weak_flow = derivative.weak_neutron_derivative * abundances[Species::Neutron as usize]
            + derivative.weak_proton_derivative * abundances[Species::Proton as usize];
        let total_charge_derivative: f64 = derivative.values[ABUNDANCE_START..]
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum();
        let charge_scale: f64 = derivative.values[ABUNDANCE_START..]
            .iter()
            .enumerate()
            .map(|(species, value)| (CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value).abs())
            .sum::<f64>()
            + weak_flow.abs();
        assert!((total_charge_derivative + weak_flow).abs() < 5.0e-15 * charge_scale);
        assert!(nuclear_charge_per_baryon(&state).unwrap().is_finite());
    }

    #[test]
    fn coupled_analytic_abundance_jacobian_matches_centered_difference() {
        let system = test_system(DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
        let state = positive_fixed_state();
        let mut analytic = [0.0; STATE_DIMENSION * STATE_DIMENSION];
        system.jacobian(0.0, &state, &mut analytic);
        for column in ABUNDANCE_START..STATE_DIMENSION {
            let step = 1.0e-3 * state[column];
            let mut plus = state;
            let mut minus = state;
            plus[column] += step;
            minus[column] -= step;
            let plus = system.derivative(&plus).unwrap();
            let minus = system.derivative(&minus).unwrap();
            for row in ABUNDANCE_START..STATE_DIMENSION {
                let finite = (plus.values[row] - minus.values[row]) / (2.0 * step);
                let exact = analytic[row * STATE_DIMENSION + column];
                let scale = exact.abs().max(finite.abs()).max(1.0e-20);
                let cancellation =
                    128.0 * f64::EPSILON * plus.values[row].abs().max(minus.values[row].abs())
                        / step;
                assert!((exact - finite).abs() < 4.0e-9 * scale + cancellation);
            }
        }
    }

    #[test]
    fn negative_abundance_fails_raw_without_observable_repair() {
        let system = test_system(DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
        let mut state = positive_fixed_state();
        state[ABUNDANCE_START + Species::Lithium6 as usize] = -1.0e-30;
        let result = solve(
            SolverKind::Rodas5P,
            &system,
            (0.0, 0.01),
            &state,
            &coupled_solver_config(1.0e-7),
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("invalid_initial_state"));
        assert_eq!(result.y, state);
        assert_eq!(
            observables_from_endpoint(&result),
            Err(ObservableError::SolverFailed)
        );
    }

    #[test]
    fn observable_builder_uses_raw_state_and_rejects_bad_contracts() {
        let mut endpoint = OdeResult {
            t: 1.0,
            y: positive_fixed_state().to_vec(),
            accepted: 1,
            rejected: 0,
            jacobian_evaluations: 1,
            linear_setups: 1,
            event_reached: true,
            failure: None,
            rhs_evaluations: 0,
            dfdt_evaluations: 0,
        };
        let observables = observables_from_endpoint(&endpoint).unwrap();
        assert_eq!(observables.helium4_mass_fraction, 0.16);
        assert_eq!(observables.deuterium_to_hydrogen, 0.08);
        assert_eq!(observables.helium3_to_hydrogen, (0.04 + 0.05) / 1.5);
        assert_eq!(
            observables.lithium7_plus_beryllium7_to_hydrogen,
            (0.02 + 0.025) / 3.5
        );
        assert_eq!(observables.lithium6_to_hydrogen, 0.005 / 3.0);
        endpoint.y[ABUNDANCE_START + Species::Lithium6 as usize] = -1.0e-6;
        endpoint.y[ABUNDANCE_START + Species::Proton as usize] += 0.005_001;
        assert_eq!(
            observables_from_endpoint(&endpoint),
            Err(ObservableError::NegativeState {
                index: Species::Lithium6 as usize
            })
        );
        endpoint.y = positive_fixed_state().to_vec();
        endpoint.y[ABUNDANCE_START] += 0.01;
        assert!(matches!(
            observables_from_endpoint(&endpoint),
            Err(ObservableError::UnnormalizedState { .. })
        ));
        endpoint.failure = Some("raw".to_string());
        assert_eq!(
            observables_from_endpoint(&endpoint),
            Err(ObservableError::SolverFailed)
        );
    }

    #[test]
    fn both_solvers_reach_the_first_coupled_bbn_endpoint() {
        let mut outputs = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let run =
                integrate_minimal_bbn(kind, DEFAULT_BORN_WEAK_QUADRATURE_ORDER, 2.0e-7).unwrap();
            assert_eq!(run.coupled_endpoint.failure, None, "{kind:?}: {run:?}");
            assert!(run.coupled_endpoint.event_reached, "{kind:?}: {run:?}");
            assert_eq!(
                run.activation_temperature_mev,
                NETWORK_ACTIVATION_TEMPERATURE_MEV
            );
            assert!(
                (run.weak_at_decoupling.y[TEMPERATURE_INDEX] - DECOUPLING_TEMPERATURE_MEV).abs()
                    < 2.0e-15
            );
            assert!(
                (run.weak_at_activation.y[TEMPERATURE_INDEX] - NETWORK_ACTIVATION_TEMPERATURE_MEV)
                    .abs()
                    < 2.0e-15
            );
            assert_eq!(
                run.coupled_initial_state[TEMPERATURE_INDEX],
                NETWORK_ACTIVATION_TEMPERATURE_MEV
            );
            assert_eq!(
                run.coupled_initial_state[ABUNDANCE_START + Species::Neutron as usize]
                    + run.coupled_initial_state[ABUNDANCE_START + Species::Proton as usize],
                1.0
            );
            assert!(
                run.coupled_initial_state[ABUNDANCE_START + Species::Deuterium as usize..]
                    .iter()
                    .all(|value| *value == 0.0)
            );
            assert!((run.coupled_endpoint.y[0] - FINAL_TEMPERATURE_MEV).abs() < 1.0e-9);
            let terminal_density = run.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            eprintln!(
                "{kind:?}: density_residual={:.16e}, temperature_residual={:.16e}, accepted={}, rejected={}",
                terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0,
                run.coupled_endpoint.y[0] / FINAL_TEMPERATURE_MEV - 1.0,
                run.coupled_endpoint.accepted,
                run.coupled_endpoint.rejected,
            );
            assert!(
                (terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 2.0e-6
            );
            assert!(baryon_residual(&run.coupled_endpoint.y).unwrap() < 2.0e-7);
            let observables = observables_from_endpoint(&run.coupled_endpoint).unwrap();
            outputs.push((run.coupled_endpoint.y, observables));
        }
        for species in 0..N_SPECIES {
            let left = outputs[0].0[ABUNDANCE_START + species];
            let right = outputs[1].0[ABUNDANCE_START + species];
            eprintln!(
                "species={species}: bdf={left:.16e}, rodas={right:.16e}, abs={:.16e}, rel={:.16e}",
                (left - right).abs(),
                (left - right).abs() / left.abs().max(right.abs()).max(1.0e-300),
            );
            assert!(
                (left - right).abs() < 3.0e-6 * left.abs().max(right.abs()).max(1.0e-20) + 2.0e-15
            );
        }
    }

    #[test]
    fn born_quadrature_and_solver_tolerance_converge_at_endpoint() {
        let baseline =
            integrate_minimal_bbn(SolverKind::Bdf, DEFAULT_BORN_WEAK_QUADRATURE_ORDER, 2.0e-7)
                .unwrap();
        let refined_quadrature = integrate_minimal_bbn(SolverKind::Bdf, 96, 2.0e-7).unwrap();
        let refined_tolerance =
            integrate_minimal_bbn(SolverKind::Bdf, DEFAULT_BORN_WEAK_QUADRATURE_ORDER, 5.0e-8)
                .unwrap();
        for run in [&baseline, &refined_quadrature, &refined_tolerance] {
            assert_eq!(run.coupled_endpoint.failure, None, "{run:?}");
            assert!(run.coupled_endpoint.event_reached, "{run:?}");
        }
        let baseline_density_residual = (baseline.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX]
            .exp()
            / baseline.expected_terminal_baryon_density_per_cm3
            - 1.0)
            .abs();
        let refined_density_residual =
            (refined_tolerance.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp()
                / refined_tolerance.expected_terminal_baryon_density_per_cm3
                - 1.0)
                .abs();
        eprintln!(
            "late-eta density residual: rtol=2e-7 {baseline_density_residual:.16e}, rtol=5e-8 {refined_density_residual:.16e}"
        );
        assert!(baseline_density_residual < 2.0e-6);
        assert!(refined_density_residual < 4.0e-7);
        assert!(refined_density_residual < baseline_density_residual);
        for species in 0..N_SPECIES {
            let reference = refined_tolerance.coupled_endpoint.y[ABUNDANCE_START + species];
            let quadrature = refined_quadrature.coupled_endpoint.y[ABUNDANCE_START + species];
            let base = baseline.coupled_endpoint.y[ABUNDANCE_START + species];
            let scale = reference.abs().max(1.0e-20);
            assert!((base - reference).abs() < 2.0e-5 * scale + 2.0e-20);
            let quadrature_budget = if species == Species::Neutron as usize {
                // The residual free-neutron tail is O(1e-11); its relative
                // shift is more sensitive than every reported abundance.
                1.0e-5
            } else {
                2.0e-6
            };
            assert!(
                (quadrature - base).abs() < quadrature_budget * scale + 2.0e-20,
                "species={species}: quadrature relative shift={:.16e}",
                (quadrature - base).abs() / scale
            );
        }
    }

    #[test]
    fn zero_seed_activation_is_stable_from_point_zero_eight_to_point_one_two_mev() {
        let mut results = Vec::new();
        for activation_temperature_mev in [0.08, 0.10, 0.12] {
            let run = integrate_minimal_bbn_at_activation(
                SolverKind::Bdf,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                5.0e-8,
                activation_temperature_mev,
            )
            .unwrap();
            assert_eq!(run.coupled_endpoint.failure, None, "{run:?}");
            assert!(run.coupled_endpoint.event_reached, "{run:?}");
            assert!(
                run.coupled_initial_state[ABUNDANCE_START + Species::Deuterium as usize..]
                    .iter()
                    .all(|value| *value == 0.0)
            );
            if activation_temperature_mev == NETWORK_ACTIVATION_TEMPERATURE_MEV {
                // Independent T_gamma-variable DOP853 weak-only handoff.
                assert!(
                    (run.coupled_initial_state[ABUNDANCE_START + Species::Neutron as usize]
                        - 0.128_115_520_070_838_43)
                        .abs()
                        < 3.0e-8
                );
            }
            let observables = observables_from_endpoint(&run.coupled_endpoint).unwrap();
            eprintln!(
                "activation={activation_temperature_mev:.2}: Yp={:.16e}, D/H={:.16e}, He3/H={:.16e}, Li7/H={:.16e}",
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            );
            results.push(observables);
        }

        let baseline = results[0];
        // Independent T_gamma-variable SciPy Radau endpoint with continuum
        // EOS, adaptive Born integrals and a refined 180/480 interpolation
        // grid. This is a shared-table formulation cross-check, not external
        // nuclear-rate authority.
        let independent_observables = [
            0.242_304_816_771_178_65,
            2.434_002_545_762_628_6e-5,
            1.040_062_426_462_625_5e-5,
            5.417_107_906_847_936e-10,
        ];
        for (actual, expected) in [
            baseline.helium4_mass_fraction,
            baseline.deuterium_to_hydrogen,
            baseline.helium3_to_hydrogen,
            baseline.lithium7_plus_beryllium7_to_hydrogen,
        ]
        .iter()
        .zip(independent_observables)
        {
            // Legacy zero-seed diagnostic only.  F06 promotion authority is
            // the matched Saha-started PRIMAT/LINX lane below.
            assert!((actual / expected - 1.0).abs() < 5.0e-6);
        }
        for comparison in &results[1..] {
            assert!(
                (comparison.helium4_mass_fraction - baseline.helium4_mass_fraction).abs() < 5.0e-4
            );
            assert!(
                (comparison.deuterium_to_hydrogen / baseline.deuterium_to_hydrogen - 1.0).abs()
                    < 2.0e-2
            );
            assert!(
                (comparison.helium3_to_hydrogen / baseline.helium3_to_hydrogen - 1.0).abs()
                    < 2.0e-2
            );
            assert!(
                (comparison.lithium7_plus_beryllium7_to_hydrogen
                    / baseline.lithium7_plus_beryllium7_to_hydrogen
                    - 1.0)
                    .abs()
                    < 5.0e-2
            );
        }
    }

    #[test]
    fn primat_matched_saha_handoff_and_dual_solver_endpoint_are_live_anchored() {
        // Official PRIMAT v0.3.2, commit
        // 21ff8f39fa18e3937e9fdf386cfa982361bfdfce, run independently through
        // both its C and Python backends with eta0=6.1e-10, tau_n=878.4 s,
        // small network, instantaneous decoupling, and all QED/radiative/
        // finite-mass/thermal corrections disabled.  The twelve RABBIT
        // AC2024 rows were supplied through PRIMAT's official custom-network
        // API on a dense representation of the same piecewise-loglinear rate
        // law.  This is a matched-table port/integrator anchor, not an
        // independent nuclear-data validation.  The Python anchor is used
        // below; the observed C-Python spread is smaller than every budget.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f06 = &fixture["f06_matched_standard_anchors"];
        let config = &f06["matched_configuration"];
        assert_eq!(config["geometry"].as_str(), Some("FLRW"));
        assert_eq!(config["reaction_count"].as_u64(), Some(12));
        assert_eq!(
            config["eta_late"].as_f64(),
            Some(LATE_BARYON_TO_PHOTON_RATIO)
        );
        assert_eq!(
            config["neutron_lifetime_seconds"].as_f64(),
            Some(DEFAULT_NEUTRON_LIFETIME_SECONDS)
        );
        assert_eq!(
            config["temperature_start_mev"].as_f64(),
            Some(INITIAL_TEMPERATURE_MEV)
        );
        assert_eq!(
            config["temperature_end_mev"].as_f64(),
            Some(FINAL_TEMPERATURE_MEV)
        );
        assert_eq!(
            config["mev_to_t9"].as_f64(),
            Some(crate::minimal_network::MEV_TO_T9)
        );
        let serialized_reactions = config["reaction_identities"].as_array().unwrap();
        assert_eq!(serialized_reactions.len(), N_BACKBONE_REACTIONS);
        for (serialized, implemented) in serialized_reactions
            .iter()
            .zip(MinimalNetwork::canonical_reaction_names())
        {
            assert_eq!(serialized.as_str(), Some(implemented));
        }
        let primat_anchor =
            &f06["primat"]["matched_rabbit_60_node_piecewise_loglinear_custom_table"]["python"];
        let primat = MinimalBbnObservables {
            helium4_mass_fraction: primat_anchor["Yp"].as_f64().unwrap(),
            deuterium_to_hydrogen: primat_anchor["DH"].as_f64().unwrap(),
            helium3_to_hydrogen: primat_anchor["He3H"].as_f64().unwrap(),
            lithium7_plus_beryllium7_to_hydrogen: primat_anchor["Li7H"].as_f64().unwrap(),
            lithium6_to_hydrogen: 0.0,
        };
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let wall_start = Instant::now();
            let run = integrate_primat_matched_minimal_bbn(
                kind,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                1.0e-9,
            )
            .unwrap();
            assert_eq!(run.network_extent, NetworkExtent::Backbone12);
            let endpoint_wall = wall_start.elapsed();
            assert_eq!(run.coupled_endpoint.failure, None, "{kind:?}: {run:?}");
            assert!(run.coupled_endpoint.event_reached, "{kind:?}: {run:?}");
            assert!(
                (run.weak_at_activation.y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV)
                    .abs()
                    < 2.0e-9
            );
            let initial_baryon_residual =
                baryon_residual(&run.coupled_initial_state).expect("matched initial state");
            assert!(initial_baryon_residual < 5.0e-11);
            let terminal_density = run.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
            let terminal_baryon_residual =
                baryon_residual(&run.coupled_endpoint.y).expect("matched terminal state");
            assert!(
                terminal_baryon_residual < 5.0e-8,
                "{kind:?}: terminal baryon residual={terminal_baryon_residual:.16e}"
            );
            let observables = observables_from_endpoint(&run.coupled_endpoint).unwrap();
            let serialized = match kind {
                SolverKind::Bdf => &f06["rust"]["bdf_rtol_1e_9"],
                SolverKind::Rodas5P => &f06["rust"]["rodas5p_rtol_1e_9"],
            };
            assert_eq!(serialized["rtol"].as_f64(), Some(1.0e-9));
            assert!(
                (run.weak_at_activation.y[1] - serialized["Xn_handoff"].as_f64().unwrap()).abs()
                    < 2.0e-8
            );
            for (actual, key) in [
                (observables.helium4_mass_fraction, "Yp"),
                (observables.deuterium_to_hydrogen, "DH"),
                (observables.helium3_to_hydrogen, "He3H"),
                (observables.lithium7_plus_beryllium7_to_hydrogen, "Li7H"),
            ] {
                let stored = serialized[key].as_f64().unwrap();
                assert!((actual / stored - 1.0).abs() < 2.0e-7, "{kind:?} {key}");
            }
            println!(
                "{kind:?}: wall={:.6}s, Xn(handoff)={:.16e}, Yp={:.16e}, D/H={:.16e}, He3/H={:.16e}, Li7/H={:.16e}",
                endpoint_wall.as_secs_f64(),
                run.weak_at_activation.y[1],
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            );
            assert!(
                (observables.helium4_mass_fraction - primat.helium4_mass_fraction).abs() <= 1.0e-5
            );
            assert!(
                (observables.deuterium_to_hydrogen / primat.deuterium_to_hydrogen - 1.0).abs()
                    <= 5.0e-4
            );
            assert!(
                (observables.helium3_to_hydrogen / primat.helium3_to_hydrogen - 1.0).abs()
                    <= 1.0e-3
            );
            assert!(
                (observables.lithium7_plus_beryllium7_to_hydrogen
                    / primat.lithium7_plus_beryllium7_to_hydrogen
                    - 1.0)
                    .abs()
                    <= 3.0e-3
            );
            endpoints.push(observables);
        }
        assert!(
            (endpoints[0].helium4_mass_fraction - endpoints[1].helium4_mass_fraction).abs()
                < 2.0e-6
        );
        assert!(
            (endpoints[0].deuterium_to_hydrogen / endpoints[1].deuterium_to_hydrogen - 1.0).abs()
                < 2.0e-5
        );
    }

    #[test]
    fn complete_qed_matched_endpoint_has_isolated_shift_and_dual_solver_convergence() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f07 = &fixture["f07_finite_temperature_qed"];
        assert_eq!(
            f07["schema_version"].as_str(),
            Some("f07_finite_temperature_qed_v1")
        );
        assert_eq!(f07["claim_status"].as_str(), Some("VALIDATED"));
        let off = integrate_primat_matched_minimal_bbn(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
        )
        .unwrap();
        let leading = integrate_primat_matched_minimal_bbn_with_qed(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            FiniteTemperatureQed::PrimatLeadingE2E3,
        )
        .unwrap();
        let complete_bdf = integrate_primat_matched_minimal_bbn_with_qed(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        )
        .unwrap();
        let complete_rodas = integrate_primat_matched_minimal_bbn_with_qed(
            SolverKind::Rodas5P,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        )
        .unwrap();
        for run in [&off, &leading, &complete_bdf, &complete_rodas] {
            assert_eq!(run.coupled_endpoint.failure, None, "{run:?}");
            assert!(run.coupled_endpoint.event_reached, "{run:?}");
            assert!(baryon_residual(&run.coupled_endpoint.y).unwrap() < 5.0e-8);
            let terminal_density = run.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
        }
        let off_observables = observables_from_endpoint(&off.coupled_endpoint).unwrap();
        let leading_observables = observables_from_endpoint(&leading.coupled_endpoint).unwrap();
        let complete_bdf_observables =
            observables_from_endpoint(&complete_bdf.coupled_endpoint).unwrap();
        let complete_rodas_observables =
            observables_from_endpoint(&complete_rodas.coupled_endpoint).unwrap();
        let to_array = |observables: MinimalBbnObservables| {
            [
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            ]
        };
        let off_values = to_array(off_observables);
        let leading_values = to_array(leading_observables);
        let complete_bdf_values = to_array(complete_bdf_observables);
        let complete_rodas_values = to_array(complete_rodas_observables);
        let stored_array = |record: &serde_json::Value| {
            ["Yp", "DH", "He3H", "Li7H"].map(|key| record[key].as_f64().unwrap())
        };
        for (actual, stored) in [
            (off_values, stored_array(&f07["rust"]["off_bdf"])),
            (leading_values, stored_array(&f07["rust"]["leading_bdf"])),
            (
                complete_bdf_values,
                stored_array(&f07["rust"]["complete_bdf"]),
            ),
            (
                complete_rodas_values,
                stored_array(&f07["rust"]["complete_rodas5p"]),
            ),
        ] {
            for observable in 0..4 {
                assert!((actual[observable] / stored[observable] - 1.0).abs() < 2.0e-7);
            }
        }
        println!(
            "QED matched endpoints: off={off_values:?}, leading={leading_values:?}, complete_bdf={complete_bdf_values:?}, complete_rodas={complete_rodas_values:?}"
        );
        let primat_off = stored_array(&f07["primat_leading_endpoint"]["qed_off_python"]);
        let primat_leading = stored_array(&f07["primat_leading_endpoint"]["qed_leading_python"]);
        for observable in 0..4 {
            let leading_shift = leading_values[observable] - off_values[observable];
            let exchange_shift = complete_bdf_values[observable] - leading_values[observable];
            let external_leading_shift = primat_leading[observable] - primat_off[observable];
            println!(
                "observable={observable}: leading_shift={leading_shift:.16e}, exchange_shift={exchange_shift:.16e}, ratio={:.16e}",
                exchange_shift.abs() / leading_shift.abs()
            );
            assert!(leading_shift != 0.0);
            assert!(exchange_shift != 0.0);
            assert_eq!(
                leading_shift.is_sign_positive(),
                external_leading_shift.is_sign_positive()
            );
            assert!((leading_shift / external_leading_shift - 1.0).abs() < 2.5e-2);
            assert!((complete_bdf_values[observable] / off_values[observable] - 1.0).abs() < 0.01);
            assert!(
                (complete_bdf_values[observable] - complete_rodas_values[observable]).abs()
                    < 3.0e-6
                        * complete_bdf_values[observable]
                            .abs()
                            .max(complete_rodas_values[observable].abs())
                        + 2.0e-15
            );
        }
        println!(
            "QED handoff Xn: off={:.16e}, leading={:.16e}, complete_bdf={:.16e}, complete_rodas={:.16e}",
            off.weak_at_activation.y[1],
            leading.weak_at_activation.y[1],
            complete_bdf.weak_at_activation.y[1],
            complete_rodas.weak_at_activation.y[1],
        );
    }

    #[test]
    fn zero_temperature_ccr_matched_endpoint_tracks_live_primat_delta() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08a = &fixture["f08a_zero_temperature_ccr"];
        assert_eq!(
            f08a["schema_version"].as_str(),
            Some("f08a_zero_temperature_ccr_v1")
        );
        assert_eq!(f08a["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08a["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(
            f08a["component_anchors"]["ccr_fn"].as_f64(),
            Some(1.758_384_386_757_194_2)
        );

        let born = integrate_primat_matched_minimal_bbn_with_qed(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            FiniteTemperatureQed::PrimatLeadingE2E3,
        )
        .unwrap();
        let ccr_physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatZeroTemperatureCcr,
        );
        let bdf_start = Instant::now();
        let ccr_bdf = integrate_primat_matched_minimal_bbn_with_physics(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            ccr_physics,
        )
        .unwrap();
        let bdf_wall = bdf_start.elapsed();
        let repeat_start = Instant::now();
        let ccr_bdf_repeat = integrate_primat_matched_minimal_bbn_with_physics(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            ccr_physics,
        )
        .unwrap();
        let repeat_wall = repeat_start.elapsed();
        let rodas_start = Instant::now();
        let ccr_rodas = integrate_primat_matched_minimal_bbn_with_physics(
            SolverKind::Rodas5P,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
            ccr_physics,
        )
        .unwrap();
        let rodas_wall = rodas_start.elapsed();
        for run in [&born, &ccr_bdf, &ccr_bdf_repeat, &ccr_rodas] {
            assert_eq!(run.coupled_endpoint.failure, None, "{run:?}");
            assert!(run.coupled_endpoint.event_reached, "{run:?}");
            assert!(baryon_residual(&run.coupled_endpoint.y).unwrap() < 5.0e-8);
            let terminal_density = run.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / run.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
        }
        assert_eq!(
            ccr_bdf.coupled_endpoint.t.to_bits(),
            ccr_bdf_repeat.coupled_endpoint.t.to_bits()
        );
        assert_eq!(
            ccr_bdf
                .coupled_endpoint
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            ccr_bdf_repeat
                .coupled_endpoint
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            (
                ccr_bdf.coupled_endpoint.accepted,
                ccr_bdf.coupled_endpoint.rejected,
                ccr_bdf.coupled_endpoint.jacobian_evaluations,
                ccr_bdf.coupled_endpoint.linear_setups,
            ),
            (
                ccr_bdf_repeat.coupled_endpoint.accepted,
                ccr_bdf_repeat.coupled_endpoint.rejected,
                ccr_bdf_repeat.coupled_endpoint.jacobian_evaluations,
                ccr_bdf_repeat.coupled_endpoint.linear_setups,
            )
        );

        let to_array = |run: &PrimatMatchedMinimalBbnRun| {
            let observables = observables_from_endpoint(&run.coupled_endpoint).unwrap();
            [
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            ]
        };
        let fixture_array = |record: &serde_json::Value| {
            ["Yp", "DH", "He3H", "Li7H"].map(|key| record[key].as_f64().unwrap())
        };
        let born_values = to_array(&born);
        let bdf_values = to_array(&ccr_bdf);
        let rodas_values = to_array(&ccr_rodas);
        let external_born = fixture_array(&f08a["primat_matched_endpoint"]["born_python"]);
        let external_ccr = fixture_array(&f08a["primat_matched_endpoint"]["ccr_python"]);
        let external_delta =
            fixture_array(&f08a["primat_matched_endpoint"]["ccr_minus_born_python"]);
        for (run, values, stored) in [
            (&ccr_bdf, bdf_values, &f08a["rust"]["bdf"]),
            (&ccr_rodas, rodas_values, &f08a["rust"]["rodas5p"]),
        ] {
            assert!(
                (run.weak_at_activation.y[1] - stored["Xn_handoff"].as_f64().unwrap()).abs()
                    < 2.0e-8
            );
            let stored_values = fixture_array(stored);
            for observable in 0..4 {
                assert!((values[observable] / stored_values[observable] - 1.0).abs() < 2.0e-7);
            }
        }
        let external_value_budgets = [2.0e-5, 1.0e-3, 2.0e-3, 4.0e-3];
        for observable in 0..4 {
            let rust_delta = bdf_values[observable] - born_values[observable];
            assert_eq!(
                rust_delta.is_sign_positive(),
                external_delta[observable].is_sign_positive()
            );
            assert!(
                (rust_delta / external_delta[observable] - 1.0).abs() < 1.0e-3,
                "observable={observable}: rust_delta={rust_delta:.16e}, external_delta={:.16e}",
                external_delta[observable]
            );
            let endpoint_difference = if observable == 0 {
                (bdf_values[observable] - external_ccr[observable]).abs()
            } else {
                (bdf_values[observable] / external_ccr[observable] - 1.0).abs()
            };
            assert!(
                endpoint_difference < external_value_budgets[observable],
                "observable={observable}: Rust={:.16e}, PRIMAT={:.16e}, difference={endpoint_difference:.16e}",
                bdf_values[observable],
                external_ccr[observable]
            );
            assert!(
                (bdf_values[observable] - rodas_values[observable]).abs()
                    < 3.0e-6
                        * bdf_values[observable]
                            .abs()
                            .max(rodas_values[observable].abs())
                        + 2.0e-15
            );
        }
        println!(
            "CCR matched endpoint: external_born={external_born:?}, external_ccr={external_ccr:?}, Rust_born={born_values:?}, Rust_bdf={bdf_values:?}, Rust_rodas={rodas_values:?}, Xn_bdf={:.16e}, Xn_rodas={:.16e}, BDF_wall={:.6}s, BDF_repeat_wall={:.6}s, Rodas_wall={:.6}s",
            ccr_bdf.weak_at_activation.y[1],
            ccr_rodas.weak_at_activation.y[1],
            bdf_wall.as_secs_f64(),
            repeat_wall.as_secs_f64(),
            rodas_wall.as_secs_f64(),
        );
    }

    #[test]
    fn finite_mass_no_weak_magnetism_entrypoint_labels_and_supports_both_solvers() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08a = &fixture["f08a_zero_temperature_ccr"];
        let f08b = &fixture["f08b_finite_nucleon_mass_no_weak_magnetism"];
        assert_eq!(
            f08b["schema_version"].as_str(),
            Some("f08b_finite_nucleon_mass_no_weak_magnetism_v1")
        );
        assert_eq!(f08b["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08b["claim_status"].as_str(), Some("VALIDATED"));
        let expected_physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
        );
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let run = integrate_primat_matched_minimal_bbn_with_ccr_finite_mass_no_weak_magnetism(
                kind,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                1.0e-9,
            )
            .unwrap();
            assert_eq!(run.physics, expected_physics);
            assert_eq!(run.coupled_endpoint.failure, None, "{kind:?}: {run:?}");
            assert!(run.coupled_endpoint.event_reached, "{kind:?}: {run:?}");
            assert!(
                (run.weak_at_activation.y[0] - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV)
                    .abs()
                    < 2.0e-9
            );
            assert!((run.coupled_endpoint.y[0] - FINAL_TEMPERATURE_MEV).abs() < 2.0e-9);
            assert!(baryon_residual(&run.coupled_endpoint.y).unwrap() < 5.0e-8);
            let observables = observables_from_endpoint(&run.coupled_endpoint).unwrap();
            endpoints.push([
                run.weak_at_activation.y[1],
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            ]);
        }
        let fixture_observables = |record: &serde_json::Value| {
            ["Yp", "DH", "He3H", "Li7H"].map(|key| record[key].as_f64().unwrap())
        };
        let external_f08b = fixture_observables(&f08b["primat_matched_endpoint"]["f08b_python"]);
        let external_delta =
            fixture_observables(&f08b["primat_matched_endpoint"]["f08b_minus_ccr_python"]);
        for (index, (values, name)) in endpoints.iter().zip(["bdf", "rodas5p"]).enumerate() {
            let stored = &f08b["rust"][name];
            assert!((values[0] - stored["Xn_handoff"].as_f64().unwrap()).abs() < 2.0e-15);
            let stored_values = fixture_observables(stored);
            for observable in 0..4 {
                let relative_residual =
                    (values[observable + 1] / stored_values[observable] - 1.0).abs();
                assert!(
                    relative_residual < 2.0e-13,
                    "solver_index={index} observable={observable}: live={:.17e}, stored={:.17e}, relative_residual={relative_residual:.17e}",
                    values[observable + 1],
                    stored_values[observable]
                );
            }
        }

        let f08a_ccr_bdf = fixture_observables(&f08a["rust"]["bdf"]);
        let endpoint_budgets = [2.0e-5, 1.0e-3, 2.0e-3, 4.0e-3];
        for observable in 0..4 {
            let rust_delta = endpoints[0][observable + 1] - f08a_ccr_bdf[observable];
            assert_eq!(
                rust_delta.is_sign_positive(),
                external_delta[observable].is_sign_positive()
            );
            assert!(
                (rust_delta / external_delta[observable] - 1.0).abs() < 2.0e-3,
                "observable={observable}: Rust F08B-CCR={rust_delta:.16e}, PRIMAT F08B-CCR={:.16e}",
                external_delta[observable]
            );
            let endpoint_difference = if observable == 0 {
                (endpoints[0][observable + 1] - external_f08b[observable]).abs()
            } else {
                (endpoints[0][observable + 1] / external_f08b[observable] - 1.0).abs()
            };
            assert!(
                endpoint_difference < endpoint_budgets[observable],
                "observable={observable}: Rust={:.16e}, PRIMAT={:.16e}, difference={endpoint_difference:.16e}",
                endpoints[0][observable + 1],
                external_f08b[observable]
            );
        }
        println!(
            "F08B matched endpoints [Xn,Yp,DH,He3H,Li7H]: BDF={:?}, Rodas5P={:?}",
            endpoints[0], endpoints[1]
        );
        for (observable, (&bdf, &rodas5p)) in endpoints[0].iter().zip(&endpoints[1]).enumerate() {
            let scale = bdf.abs().max(rodas5p.abs());
            assert!(
                (bdf - rodas5p).abs() < 3.0e-6 * scale + 2.0e-15,
                "observable={observable}: BDF={:.16e}, Rodas5P={:.16e}",
                bdf,
                rodas5p
            );
        }
    }

    #[test]
    fn physical_weak_magnetism_generic_selector_reaches_deterministic_dual_solver_endpoint() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08b = &fixture["f08b_finite_nucleon_mass_no_weak_magnetism"];
        let f08c = &fixture["f08c_physical_weak_magnetism"];
        assert_eq!(
            f08c["schema_version"].as_str(),
            Some("f08c_physical_weak_magnetism_v1")
        );
        assert_eq!(f08c["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08c["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(f08c["rust"]["execution_status"].as_str(), Some("VALIDATED"));
        assert_eq!(
            f08c["rust"]["profile"]["weak_magnetism_delta_kappa"].as_f64(),
            Some(crate::born_weak::PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT)
        );
        assert_eq!(
            f08c["primat_matched_endpoint"]["repeat_determinism"]["python"]["exact_float_equality"]
                .as_bool(),
            Some(true)
        );
        assert_eq!(
            f08c["primat_matched_endpoint"]["repeat_determinism"]["c"]["exact_float_equality"]
                .as_bool(),
            Some(true)
        );
        assert_eq!(
            f08c["primat_matched_endpoint"]["rust_acceptance"]
                ["frozen_before_any_rust_value_was_read"]
                .as_bool(),
            Some(true)
        );
        let physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism,
        );
        let run = |kind| {
            integrate_primat_matched_minimal_bbn_with_physics(
                kind,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                1.0e-9,
                physics,
            )
            .unwrap()
        };

        let bdf_start = Instant::now();
        let bdf = run(SolverKind::Bdf);
        let bdf_wall = bdf_start.elapsed();
        let repeat_start = Instant::now();
        let bdf_repeat = run(SolverKind::Bdf);
        let repeat_wall = repeat_start.elapsed();
        let rodas_start = Instant::now();
        let rodas = run(SolverKind::Rodas5P);
        let rodas_wall = rodas_start.elapsed();

        for candidate in [&bdf, &bdf_repeat, &rodas] {
            assert_eq!(candidate.physics, physics);
            assert_eq!(candidate.weak_at_activation.failure, None, "{candidate:?}");
            assert_eq!(candidate.coupled_endpoint.failure, None, "{candidate:?}");
            assert!(candidate.weak_at_activation.event_reached);
            assert!(candidate.coupled_endpoint.event_reached);
            assert!(
                (candidate.weak_at_activation.y[0]
                    - PRIMAT_MATCHED_NETWORK_ACTIVATION_TEMPERATURE_MEV)
                    .abs()
                    < 2.0e-9
            );
            assert!((candidate.coupled_endpoint.y[0] - FINAL_TEMPERATURE_MEV).abs() < 2.0e-9);
            assert!(baryon_residual(&candidate.coupled_endpoint.y).unwrap() < 5.0e-8);
            let terminal_density = candidate.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / candidate.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
        }

        for (first, repeat) in [
            (&bdf.weak_at_activation, &bdf_repeat.weak_at_activation),
            (&bdf.coupled_endpoint, &bdf_repeat.coupled_endpoint),
        ] {
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
                ),
                (
                    repeat.accepted,
                    repeat.rejected,
                    repeat.jacobian_evaluations,
                    repeat.linear_setups,
                )
            );
        }

        let endpoint = |candidate: &PrimatMatchedMinimalBbnRun| {
            let observables = observables_from_endpoint(&candidate.coupled_endpoint).unwrap();
            [
                candidate.weak_at_activation.y[1],
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            ]
        };
        let bdf_values = endpoint(&bdf);
        let rodas_values = endpoint(&rodas);
        let fixture_observables = |record: &serde_json::Value| {
            ["Yp", "DH", "He3H", "Li7H"].map(|key| record[key].as_f64().unwrap())
        };
        for (solver_index, (values, name)) in [bdf_values, rodas_values]
            .iter()
            .zip(["bdf", "rodas5p"])
            .enumerate()
        {
            let stored = &f08c["rust"][name];
            assert!((values[0] - stored["Xn_handoff"].as_f64().unwrap()).abs() < 2.0e-15);
            let stored_observables = fixture_observables(stored);
            for observable in 0..4 {
                assert!(
                    (values[observable + 1] / stored_observables[observable] - 1.0).abs() < 2.0e-13,
                    "solver_index={solver_index} observable={observable}"
                );
            }
        }

        let external_endpoint = &f08c["primat_matched_endpoint"];
        let external_f08c = fixture_observables(&external_endpoint["f08c_python"]);
        let external_delta =
            fixture_observables(&external_endpoint["f08c_python_minus_f08b_python"]);
        let f08b_bdf = fixture_observables(&f08b["rust"]["bdf"]);
        let acceptance = &external_endpoint["rust_acceptance"];
        let endpoint_budgets = [
            acceptance["direct_endpoint_budgets"]["Yp_absolute"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["DH_relative"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["He3H_relative"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["Li7H_relative"]
                .as_f64()
                .unwrap(),
        ];
        let delta_ceilings = [
            acceptance["isolated_f08c_minus_f08b_relative_ceilings"]["Yp"]
                .as_f64()
                .unwrap(),
            acceptance["isolated_f08c_minus_f08b_relative_ceilings"]["DH"]
                .as_f64()
                .unwrap(),
            acceptance["isolated_f08c_minus_f08b_relative_ceilings"]["He3H"]
                .as_f64()
                .unwrap(),
            acceptance["isolated_f08c_minus_f08b_relative_ceilings"]["Li7H"]
                .as_f64()
                .unwrap(),
        ];
        assert_eq!(endpoint_budgets, [2.0e-5, 1.0e-3, 2.0e-3, 4.0e-3]);
        assert_eq!(delta_ceilings, [2.5e-3, 1.5e-2, 1.5e-2, 1.5e-2]);
        for observable in 0..4 {
            let rust_delta = bdf_values[observable + 1] - f08b_bdf[observable];
            assert_eq!(
                rust_delta.is_sign_positive(),
                external_delta[observable].is_sign_positive()
            );
            assert!(
                (rust_delta / external_delta[observable] - 1.0).abs() < delta_ceilings[observable],
                "observable={observable}: Rust F08C-F08B={rust_delta:.16e}, PRIMAT F08C-F08B={:.16e}",
                external_delta[observable]
            );
            let endpoint_difference = if observable == 0 {
                (bdf_values[observable + 1] - external_f08c[observable]).abs()
            } else {
                (bdf_values[observable + 1] / external_f08c[observable] - 1.0).abs()
            };
            assert!(
                endpoint_difference < endpoint_budgets[observable],
                "observable={observable}: Rust={:.16e}, PRIMAT={:.16e}, difference={endpoint_difference:.16e}",
                bdf_values[observable + 1],
                external_f08c[observable]
            );
        }
        for (observable, (&bdf_value, &rodas_value)) in
            bdf_values.iter().zip(&rodas_values).enumerate()
        {
            let scale = bdf_value.abs().max(rodas_value.abs());
            assert!(
                (bdf_value - rodas_value).abs() < 3.0e-6 * scale + 2.0e-15,
                "observable={observable}: BDF={bdf_value:.16e}, Rodas5P={rodas_value:.16e}"
            );
        }
        println!(
            "F08C matched endpoints [Xn,Yp,DH,He3H,Li7H]: BDF={bdf_values:?}, Rodas5P={rodas_values:?}; BDF_wall={:.6}s, repeat_wall={:.6}s, Rodas_wall={:.6}s",
            bdf_wall.as_secs_f64(),
            repeat_wall.as_secs_f64(),
            rodas_wall.as_secs_f64(),
        );
    }

    #[test]
    fn complete_thermal_radiative_generic_selector_reaches_conditional_dual_solver_endpoint() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08c = &fixture["f08c_physical_weak_magnetism"];
        let f08d = &fixture["f08d_complete_thermal_radiative"];
        assert_eq!(f08d["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08d["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(f08d["promotion_status"].as_str(), Some("BLOCKED"));
        assert_eq!(
            f08d["matched_endpoint"]["execution_status"].as_str(),
            Some("VALIDATED")
        );
        let physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism,
        );
        let run = |kind| {
            integrate_primat_matched_minimal_bbn_with_physics(
                kind,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                1.0e-9,
                physics,
            )
            .unwrap()
        };

        let bdf_start = Instant::now();
        let bdf = run(SolverKind::Bdf);
        let bdf_wall = bdf_start.elapsed();
        let repeat_start = Instant::now();
        let bdf_repeat = run(SolverKind::Bdf);
        let repeat_wall = repeat_start.elapsed();
        let rodas_start = Instant::now();
        let rodas = run(SolverKind::Rodas5P);
        let rodas_wall = rodas_start.elapsed();

        for candidate in [&bdf, &bdf_repeat, &rodas] {
            assert_eq!(candidate.physics, physics);
            assert_eq!(candidate.weak_at_activation.failure, None, "{candidate:?}");
            assert_eq!(candidate.coupled_endpoint.failure, None, "{candidate:?}");
            assert!(candidate.weak_at_activation.event_reached);
            assert!(candidate.coupled_endpoint.event_reached);
            assert!(baryon_residual(&candidate.coupled_endpoint.y).unwrap() < 5.0e-8);
            let terminal_density = candidate.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / candidate.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
        }
        for (first, repeat) in [
            (&bdf.weak_at_activation, &bdf_repeat.weak_at_activation),
            (&bdf.coupled_endpoint, &bdf_repeat.coupled_endpoint),
        ] {
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
        }

        let endpoint = |candidate: &PrimatMatchedMinimalBbnRun| {
            let observables = observables_from_endpoint(&candidate.coupled_endpoint).unwrap();
            [
                candidate.weak_at_activation.y[1],
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
            ]
        };
        let bdf_values = endpoint(&bdf);
        let rodas_values = endpoint(&rodas);
        let matched = &f08d["matched_endpoint"];
        let external_record = &matched["external_deterministic_c_table"]["python"];
        let external_python =
            ["Yp", "DH", "He3H", "Li7H"].map(|key| external_record[key].as_f64().unwrap());
        let acceptance = &matched["frozen_acceptance"];
        let direct_endpoint_budgets = [
            acceptance["direct_endpoint_budgets"]["Yp_absolute"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["DH_relative"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["He3H_relative"]
                .as_f64()
                .unwrap(),
            acceptance["direct_endpoint_budgets"]["Li7H_relative"]
                .as_f64()
                .unwrap(),
        ];
        let isolated_envelopes = &acceptance["isolated_absolute_magnitude_envelopes"];
        let isolated_magnitude_envelopes = ["Yp", "DH", "He3H", "Li7H"].map(|key| {
            (
                isolated_envelopes[key][0].as_f64().unwrap(),
                isolated_envelopes[key][1].as_f64().unwrap(),
            )
        });
        let f08c_bdf = &f08c["rust"]["bdf"];
        let f08c_values = ["Yp", "DH", "He3H", "Li7H"].map(|key| f08c_bdf[key].as_f64().unwrap());
        for observable in 0..4 {
            let direct_difference = if observable == 0 {
                (bdf_values[observable + 1] - external_python[observable]).abs()
            } else {
                (bdf_values[observable + 1] / external_python[observable] - 1.0).abs()
            };
            assert!(
                direct_difference < direct_endpoint_budgets[observable],
                "observable={observable}: Rust={:.17e}, external={:.17e}, difference={direct_difference:.17e}",
                bdf_values[observable + 1],
                external_python[observable]
            );
            let isolated = bdf_values[observable + 1] - f08c_values[observable];
            assert!(
                isolated.is_sign_negative(),
                "observable={observable}: F08D-F08C={isolated:.17e}"
            );
            assert!(
                (isolated_magnitude_envelopes[observable].0
                    ..=isolated_magnitude_envelopes[observable].1)
                    .contains(&isolated.abs()),
                "observable={observable}: |F08D-F08C|={:.17e}, envelope={:?}",
                isolated.abs(),
                isolated_magnitude_envelopes[observable]
            );
        }
        for (observable, (&bdf_value, &rodas_value)) in
            bdf_values.iter().zip(&rodas_values).enumerate()
        {
            let scale = bdf_value.abs().max(rodas_value.abs());
            assert!(
                (bdf_value - rodas_value).abs() < 3.0e-6 * scale + 2.0e-15,
                "observable={observable}: BDF={bdf_value:.17e}, Rodas5P={rodas_value:.17e}"
            );
        }
        for (values, solver) in [(bdf_values, "bdf"), (rodas_values, "rodas5p")] {
            let stored = &matched["rust"][solver];
            for (observable, key) in ["Xn_handoff", "Yp", "DH", "He3H", "Li7H"]
                .into_iter()
                .enumerate()
            {
                let stored_value = stored[key].as_f64().unwrap();
                let absolute_residual = (values[observable] - stored_value).abs();
                assert!(
                    absolute_residual < 2.0e-15,
                    "solver={solver} key={key}: live={:.17e}, stored={stored_value:.17e}, absolute_residual={absolute_residual:.17e}",
                    values[observable]
                );
            }
        }
        println!(
            "F08D matched endpoints [Xn,Yp,DH,He3H,Li7H]: BDF={bdf_values:?}, Rodas5P={rodas_values:?}; cold_BDF_wall={:.6}s, repeat_BDF_wall={:.6}s, Rodas_wall={:.6}s",
            bdf_wall.as_secs_f64(),
            repeat_wall.as_secs_f64(),
            rodas_wall.as_secs_f64(),
        );
    }

    #[test]
    fn selected_31_complete_endpoint_satisfies_pre_rust_external_and_nested_budgets() {
        // F08N budgets were frozen by the independent external campaign before
        // this Rust endpoint was first executed.  The exact selected-31 PRIMAT
        // record used the AC2024 table's stored reverse coefficients rather
        // than reconstructing them from rounded Q values.
        const EXTERNAL_SELECTED_31: [f64; 5] = [
            0.246_833_899_514_248_03,
            2.456_571_050_928_129e-5,
            1.043_378_368_726_396_1e-5,
            5.437_134_223_492_424e-10,
            7.819_032_381_929_294e-15,
        ];
        const DIRECT_BUDGETS: [f64; 4] = [2.0e-5, 1.0e-3, 2.0e-3, 4.0e-3];
        const LI6H_RANGE: (f64, f64) = (2.0e-15, 2.0e-14);
        const DELTA_YP_ABSOLUTE_CEILING: f64 = 2.0e-6;
        const DELTA_DH_RANGE: (f64, f64) = (2.0e-10, 2.0e-9);
        const DELTA_HE3H_RANGE: (f64, f64) = (-1.6e-9, -3.0e-10);
        const RELATIVE_DELTA_LI7H_RANGE: (f64, f64) = (-2.0e-2, -4.0e-3);
        const BARYON_RESIDUAL_CEILING: f64 = 5.0e-9;
        const PRIMARY_RELATIVE_TOLERANCE: f64 = 1.0e-10;
        const REFINED_RELATIVE_TOLERANCE: f64 = 3.0e-11;
        const SOLVER_CONVERGENCE_RELATIVE_CEILING: f64 = 3.0e-6;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/flrw_gold_v861.json"
        )))
        .unwrap();
        let f08n = &fixture["f08n_selected_31_network"];
        assert_eq!(f08n["implementation_status"].as_str(), Some("IMPLEMENTED"));
        assert_eq!(f08n["claim_status"].as_str(), Some("VALIDATED"));
        assert_eq!(f08n["promotion_status"].as_str(), Some("BLOCKED"));
        assert_eq!(f08n["authority_state"].as_str(), Some("CONDITIONAL"));
        assert_eq!(
            f08n["external"]["frozen_acceptance"]["frozen_before_any_rust_f08n_value_was_read"]
                .as_bool(),
            Some(true)
        );
        assert_eq!(
            f08n["network_contract"]["selected_table_sha256"].as_str(),
            Some("8dacb12bece202a798bae67f2ca89d1fad62f9f02bad38da0f59c19147960b85")
        );
        assert_eq!(
            f08n["network_contract"]["reaction_count"].as_u64(),
            Some(31)
        );
        let stored_reaction_order = f08n["network_contract"]["reaction_order"]
            .as_array()
            .unwrap();
        assert_eq!(
            stored_reaction_order.len(),
            crate::minimal_network::N_REACTIONS
        );
        for (stored, implemented) in stored_reaction_order
            .iter()
            .zip(MinimalNetwork::selected_31_reaction_names())
        {
            assert_eq!(stored.as_str(), Some(implemented));
        }
        let stored_external = &f08n["external"]["exact_selected_31"];
        let stored_external_values =
            ["Yp", "DH", "He3H", "Li7H", "Li6H"].map(|key| stored_external[key].as_f64().unwrap());
        for observable in 0..5 {
            let scale = EXTERNAL_SELECTED_31[observable]
                .abs()
                .max(stored_external_values[observable].abs());
            assert!(
                (EXTERNAL_SELECTED_31[observable] - stored_external_values[observable]).abs()
                    <= 2.0e-15 * scale + 1.0e-30,
                "external fixture observable={observable}: literal={:.17e}, stored={:.17e}",
                EXTERNAL_SELECTED_31[observable],
                stored_external_values[observable]
            );
        }
        let stored_direct_budgets =
            &f08n["external"]["frozen_acceptance"]["direct_selected31_endpoint_budgets"];
        assert_eq!(
            DIRECT_BUDGETS,
            [
                stored_direct_budgets["Yp_absolute"].as_f64().unwrap(),
                stored_direct_budgets["DH_relative"].as_f64().unwrap(),
                stored_direct_budgets["He3H_relative"].as_f64().unwrap(),
                stored_direct_budgets["Li7H_relative"].as_f64().unwrap(),
            ]
        );
        assert_eq!(
            [LI6H_RANGE.0, LI6H_RANGE.1],
            stored_direct_budgets["Li6H_absolute_range"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap())
                .collect::<Vec<_>>()
                .as_slice()
        );
        assert_eq!(
            stored_direct_budgets["baryon_sum_abs_residual"].as_f64(),
            Some(BARYON_RESIDUAL_CEILING)
        );
        let f08d = &fixture["f08d_complete_thermal_radiative"]["matched_endpoint"]["rust"];
        let physics = MatchedStandardPhysics::new(
            FiniteTemperatureQed::PrimatLeadingE2E3,
            WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism,
        );
        let run = |kind, rtol| {
            integrate_primat_matched_selected_31_bbn_with_physics(
                kind,
                DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
                rtol,
                physics,
            )
            .unwrap()
        };

        let bdf_start = Instant::now();
        let bdf = run(SolverKind::Bdf, PRIMARY_RELATIVE_TOLERANCE);
        let bdf_wall = bdf_start.elapsed();
        let repeat_start = Instant::now();
        let bdf_repeat = run(SolverKind::Bdf, PRIMARY_RELATIVE_TOLERANCE);
        let repeat_wall = repeat_start.elapsed();
        let rodas_start = Instant::now();
        let rodas = run(SolverKind::Rodas5P, PRIMARY_RELATIVE_TOLERANCE);
        let rodas_wall = rodas_start.elapsed();
        let ladder_start = Instant::now();
        let bdf_rtol_3e11 = run(SolverKind::Bdf, REFINED_RELATIVE_TOLERANCE);
        let ladder_wall = ladder_start.elapsed();

        for (label, candidate) in [
            ("bdf", &bdf),
            ("bdf_repeat", &bdf_repeat),
            ("rodas5p", &rodas),
            ("bdf_rtol_3e-11", &bdf_rtol_3e11),
        ] {
            assert_eq!(candidate.physics, physics);
            assert_eq!(candidate.network_extent, NetworkExtent::Selected31);
            assert_eq!(candidate.weak_at_activation.failure, None, "{candidate:?}");
            assert_eq!(candidate.coupled_endpoint.failure, None, "{candidate:?}");
            assert!(candidate.weak_at_activation.event_reached);
            assert!(candidate.coupled_endpoint.event_reached);
            let baryon_residual = baryon_residual(&candidate.coupled_endpoint.y).unwrap();
            assert!(
                baryon_residual <= BARYON_RESIDUAL_CEILING,
                "{label} selected-31 baryon residual={baryon_residual:.17e}"
            );
            let charge = nuclear_charge_per_baryon(&candidate.coupled_endpoint.y).unwrap();
            assert!(charge.is_finite() && (0.0..=1.0).contains(&charge));
            let terminal_density = candidate.coupled_endpoint.y[LOG_BARYON_DENSITY_INDEX].exp();
            assert!(
                (terminal_density / candidate.expected_terminal_baryon_density_per_cm3 - 1.0).abs()
                    < 5.0e-6
            );
        }
        for (first, repeat) in [
            (&bdf.weak_at_activation, &bdf_repeat.weak_at_activation),
            (&bdf.coupled_endpoint, &bdf_repeat.coupled_endpoint),
        ] {
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
                ),
                (
                    repeat.accepted,
                    repeat.rejected,
                    repeat.jacobian_evaluations,
                    repeat.linear_setups,
                )
            );
        }

        let endpoint = |candidate: &PrimatMatchedMinimalBbnRun| {
            let observables = observables_from_endpoint(&candidate.coupled_endpoint).unwrap();
            [
                observables.helium4_mass_fraction,
                observables.deuterium_to_hydrogen,
                observables.helium3_to_hydrogen,
                observables.lithium7_plus_beryllium7_to_hydrogen,
                observables.lithium6_to_hydrogen,
            ]
        };
        let bdf_values = endpoint(&bdf);
        let rodas_values = endpoint(&rodas);
        let ladder_values = endpoint(&bdf_rtol_3e11);
        for (name, values) in [("bdf", bdf_values), ("rodas5p", rodas_values)] {
            let stored = &f08n["rust"][name];
            let stored_values =
                ["Yp", "DH", "He3H", "Li7H", "Li6H"].map(|key| stored[key].as_f64().unwrap());
            for observable in 0..5 {
                let scale = values[observable]
                    .abs()
                    .max(stored_values[observable].abs());
                assert!(
                    (values[observable] - stored_values[observable]).abs()
                        <= 5.0e-12 * scale + 1.0e-24,
                    "stored {name} observable={observable}: live={:.17e}, stored={:.17e}",
                    values[observable],
                    stored_values[observable]
                );
            }
        }
        let stored_ladder = ["Yp", "DH", "He3H", "Li7H", "Li6H"].map(|key| {
            f08n["rust"]["refined_bdf_rtol_3e_11"][key]
                .as_f64()
                .unwrap()
        });
        for observable in 0..5 {
            let scale = ladder_values[observable]
                .abs()
                .max(stored_ladder[observable].abs());
            assert!(
                (ladder_values[observable] - stored_ladder[observable]).abs()
                    <= 5.0e-12 * scale + 1.0e-24,
                "stored refined BDF observable={observable}: live={:.17e}, stored={:.17e}",
                ladder_values[observable],
                stored_ladder[observable]
            );
        }
        for (solver, values) in [("bdf", bdf_values), ("rodas5p", rodas_values)] {
            assert!((values[0] - EXTERNAL_SELECTED_31[0]).abs() <= DIRECT_BUDGETS[0]);
            for observable in 1..4 {
                let relative_difference =
                    (values[observable] / EXTERNAL_SELECTED_31[observable] - 1.0).abs();
                assert!(
                    relative_difference <= DIRECT_BUDGETS[observable],
                    "{solver} observable={observable}: Rust={:.17e}, external={:.17e}, relative_difference={relative_difference:.17e}",
                    values[observable],
                    EXTERNAL_SELECTED_31[observable]
                );
            }
            assert!(
                (LI6H_RANGE.0..=LI6H_RANGE.1).contains(&values[4]),
                "{solver} Li6/H={:.17e}",
                values[4]
            );

            let backbone = &f08d[solver];
            let backbone_values =
                ["Yp", "DH", "He3H", "Li7H"].map(|key| backbone[key].as_f64().unwrap());
            let delta = [
                values[0] - backbone_values[0],
                values[1] - backbone_values[1],
                values[2] - backbone_values[2],
                values[3] / backbone_values[3] - 1.0,
            ];
            assert!(delta[0].abs() <= DELTA_YP_ABSOLUTE_CEILING);
            assert!((DELTA_DH_RANGE.0..=DELTA_DH_RANGE.1).contains(&delta[1]));
            assert!((DELTA_HE3H_RANGE.0..=DELTA_HE3H_RANGE.1).contains(&delta[2]));
            assert!(
                (RELATIVE_DELTA_LI7H_RANGE.0..=RELATIVE_DELTA_LI7H_RANGE.1).contains(&delta[3])
            );
        }
        for observable in 0..5 {
            let dual_scale = bdf_values[observable]
                .abs()
                .max(rodas_values[observable].abs());
            assert!(
                (bdf_values[observable] - rodas_values[observable]).abs()
                    <= SOLVER_CONVERGENCE_RELATIVE_CEILING * dual_scale,
                "dual solver observable={observable}: BDF={:.17e}, Rodas5P={:.17e}",
                bdf_values[observable],
                rodas_values[observable]
            );
            let ladder_scale = bdf_values[observable]
                .abs()
                .max(ladder_values[observable].abs());
            assert!(
                (bdf_values[observable] - ladder_values[observable]).abs()
                    <= SOLVER_CONVERGENCE_RELATIVE_CEILING * ladder_scale,
                "tolerance ladder observable={observable}: rtol1e-10={:.17e}, rtol3e-11={:.17e}",
                bdf_values[observable],
                ladder_values[observable]
            );
        }
        println!(
            "F08N selected-31 endpoints [Yp,DH,He3H,Li7H,Li6H]: BDF={bdf_values:?}, Rodas5P={rodas_values:?}, BDF_rtol3e-11={ladder_values:?}; baryon_residuals=[{:.17e},{:.17e},{:.17e}]; charges=[{:.17e},{:.17e},{:.17e}]; cold_BDF_wall={:.6}s, repeat_BDF_wall={:.6}s, Rodas_wall={:.6}s, ladder_wall={:.6}s",
            baryon_residual(&bdf.coupled_endpoint.y).unwrap(),
            baryon_residual(&rodas.coupled_endpoint.y).unwrap(),
            baryon_residual(&bdf_rtol_3e11.coupled_endpoint.y).unwrap(),
            nuclear_charge_per_baryon(&bdf.coupled_endpoint.y).unwrap(),
            nuclear_charge_per_baryon(&rodas.coupled_endpoint.y).unwrap(),
            nuclear_charge_per_baryon(&bdf_rtol_3e11.coupled_endpoint.y).unwrap(),
            bdf_wall.as_secs_f64(),
            repeat_wall.as_secs_f64(),
            rodas_wall.as_secs_f64(),
            ladder_wall.as_secs_f64(),
        );
    }

    #[test]
    fn repeated_full_matched_bdf_endpoint_is_bitwise_deterministic_and_reports_wall_time() {
        let first_start = Instant::now();
        let first = integrate_primat_matched_minimal_bbn(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
        )
        .unwrap();
        let first_wall = first_start.elapsed();
        let repeat_start = Instant::now();
        let repeat = integrate_primat_matched_minimal_bbn(
            SolverKind::Bdf,
            DEFAULT_BORN_WEAK_QUADRATURE_ORDER,
            1.0e-9,
        )
        .unwrap();
        let repeat_wall = repeat_start.elapsed();
        assert_eq!(first.coupled_endpoint.failure, None, "{first:?}");
        assert_eq!(repeat.coupled_endpoint.failure, None, "{repeat:?}");
        assert!(first.coupled_endpoint.event_reached);
        assert!(repeat.coupled_endpoint.event_reached);
        assert_eq!(
            first.coupled_endpoint.t.to_bits(),
            repeat.coupled_endpoint.t.to_bits()
        );
        assert_eq!(
            first
                .coupled_endpoint
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeat
                .coupled_endpoint
                .y
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            (
                first.coupled_endpoint.accepted,
                first.coupled_endpoint.rejected,
                first.coupled_endpoint.jacobian_evaluations,
                first.coupled_endpoint.linear_setups,
            ),
            (
                repeat.coupled_endpoint.accepted,
                repeat.coupled_endpoint.rejected,
                repeat.coupled_endpoint.jacobian_evaluations,
                repeat.coupled_endpoint.linear_setups,
            )
        );
        eprintln!(
            "full matched BDF endpoint wall: first={:.6}s, repeat={:.6}s",
            first_wall.as_secs_f64(),
            repeat_wall.as_secs_f64()
        );
    }
}
