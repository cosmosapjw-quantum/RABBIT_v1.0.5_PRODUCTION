//! Neutron-proton weak rates for the bottom-up Rust BBN path.
//!
//! The stable baseline is the six Born charged-current processes with finite
//! electron mass and zero chemical potentials.  The opt-in PRIMAT zero-
//! temperature CCR model multiplies that same phase space by the relativistic
//! Fermi-Coulomb factor and resummed zero-temperature radiative factor used by
//! PRIMAT v0.3.2.  Its measured-neutron-lifetime normalization contains the
//! same corrections as its thermal rates.  The next opt-in layer adds PRIMAT's
//! first-order finite-nucleon-mass Fokker-Planck correction, first with
//! anomalous weak magnetism disabled and then with PRIMAT's physical magnetic-
//! moment coefficient.  The next complete opt-in layer adds Brown--Sawyer's
//! four-part finite-temperature radiative correction as a signed directional
//! term; its direct Rust evaluator feeds a private in-memory table tied to the
//! leading-QED instantaneous-decoupling profile.  Energies inside the
//! integrals are measured in electron-mass units; temperatures at the
//! interface are MeV and rates are returned in inverse seconds.  Clipping and
//! heuristic rate floors remain absent.

#![cfg_attr(not(test), allow(dead_code))]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{Display, Formatter};

use crate::thermal_weak::{ThermalRadiativeError, complete_thermal_radiative_raw};

pub(crate) const ELECTRON_MASS_MEV: f64 = 0.510_998_950_0;
pub(crate) const NEUTRON_PROTON_MASS_DIFFERENCE_MEV: f64 = 1.293_332_36;
pub(crate) const DEFAULT_NEUTRON_LIFETIME_SECONDS: f64 = 878.4;
pub(crate) const DEFAULT_BORN_WEAK_QUADRATURE_ORDER: usize = 64;

const MIN_QUADRATURE_ORDER: usize = 2;
const MAX_QUADRATURE_ORDER: usize = 512;
const ROOT_ITERATION_LIMIT: usize = 64;

// PRIMAT v0.3.2 constants entering FermiCoulomb and RadCorrResum.  These are
// deliberately local to this first correction layer so a later constants
// authority change cannot silently alter the validated Born path.
const FINE_STRUCTURE_CONSTANT: f64 = 1.0 / 137.035_999_084;
const PROTON_MASS_MEV: f64 = 938.272_088_16;
const NEUTRON_MASS_MEV: f64 = 939.565_420_52;
const NUCLEON_AXIAL_COUPLING: f64 = 1.2756;
const F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT: f64 = 0.0;
/// PRIMAT's `delta_kappa = kappa_p - kappa_n`, not the paper convention
/// `f_wm = delta_kappa / 2`.
pub(crate) const PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT: f64 = 3.705_890_074_63;
const PROTON_CHARGE_RADIUS_CM: f64 = 0.8409e-13;
const HBAR_ERG_SECONDS: f64 = 6.626_070_15 / (2.0 * PI) * 1.0e-27;
const SPEED_OF_LIGHT_CM_PER_SECOND: f64 = 2.997_924_58e10;
const MEV_IN_ERG: f64 = 1.602_176_634e-6;
const FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeakRateModel {
    Born,
    PrimatZeroTemperatureCcr,
    PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
    PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism,
    PrimatCompleteThermalRadiativePhysicalWeakMagnetism,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BornWeakProcess {
    ElectronNeutrinoCaptureOnNeutron,
    ElectronCaptureOnProton,
    PositronCaptureOnNeutron,
    ElectronAntineutrinoCaptureOnProton,
    FreeNeutronDecay,
    InverseNeutronDecay,
}

impl Display for BornWeakProcess {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::ElectronNeutrinoCaptureOnNeutron => "nu_e + n -> p + e-",
            Self::ElectronCaptureOnProton => "p + e- -> n + nu_e",
            Self::PositronCaptureOnNeutron => "e+ + n -> p + anti-nu_e",
            Self::ElectronAntineutrinoCaptureOnProton => "p + anti-nu_e -> n + e+",
            Self::FreeNeutronDecay => "n -> p + e- + anti-nu_e",
            Self::InverseNeutronDecay => "p + e- + anti-nu_e -> n",
        };
        formatter.write_str(name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BornWeakError {
    InvalidTemperature {
        field: &'static str,
        raw_value_mev: f64,
    },
    InvalidNeutronLifetime {
        raw_value_seconds: f64,
    },
    InvalidQuadratureOrder {
        raw_order: usize,
        minimum: usize,
        maximum: usize,
    },
    InvalidNormalization {
        raw_lifetime_seconds: f64,
        raw_normalization: f64,
    },
    QuadratureRootDidNotConverge {
        order: usize,
        root_index: usize,
        raw_root: f64,
    },
    NonFiniteIntegral {
        process: BornWeakProcess,
        raw_value: f64,
    },
    NegativeIntegral {
        process: BornWeakProcess,
        raw_value: f64,
    },
    NonFiniteRate {
        direction: &'static str,
        raw_value_per_second: f64,
    },
    NegativeRate {
        direction: &'static str,
        raw_value_per_second: f64,
    },
    ThermalRadiative(ThermalRadiativeError),
}

impl Display for BornWeakError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTemperature {
                field,
                raw_value_mev,
            } => write!(
                formatter,
                "{field} must be finite and positive, got {raw_value_mev} MeV"
            ),
            Self::InvalidNeutronLifetime { raw_value_seconds } => write!(
                formatter,
                "neutron lifetime must be finite and positive, got {raw_value_seconds} s"
            ),
            Self::InvalidQuadratureOrder {
                raw_order,
                minimum,
                maximum,
            } => write!(
                formatter,
                "quadrature order must lie in [{minimum}, {maximum}], got {raw_order}"
            ),
            Self::InvalidNormalization {
                raw_lifetime_seconds,
                raw_normalization,
            } => write!(
                formatter,
                "Born normalization from tau_n={raw_lifetime_seconds} s is invalid: {raw_normalization}"
            ),
            Self::QuadratureRootDidNotConverge {
                order,
                root_index,
                raw_root,
            } => write!(
                formatter,
                "Gauss-Legendre root {root_index} of order {order} did not converge; last root {raw_root}"
            ),
            Self::NonFiniteIntegral { process, raw_value } => {
                write!(formatter, "{process} integral is nonfinite: {raw_value}")
            }
            Self::NegativeIntegral { process, raw_value } => {
                write!(formatter, "{process} integral is negative: {raw_value}")
            }
            Self::NonFiniteRate {
                direction,
                raw_value_per_second,
            } => write!(
                formatter,
                "{direction} total rate is nonfinite: {raw_value_per_second} s^-1"
            ),
            Self::NegativeRate {
                direction,
                raw_value_per_second,
            } => write!(
                formatter,
                "{direction} total rate is negative: {raw_value_per_second} s^-1"
            ),
            Self::ThermalRadiative(error) => {
                write!(
                    formatter,
                    "finite-temperature radiative correction failed: {error:?}"
                )
            }
        }
    }
}

impl Error for BornWeakError {}

impl From<ThermalRadiativeError> for BornWeakError {
    fn from(error: ThermalRadiativeError) -> Self {
        Self::ThermalRadiative(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BornWeakChannelRates {
    pub(crate) electron_neutrino_capture_on_neutron_per_second: f64,
    pub(crate) electron_capture_on_proton_per_second: f64,
    pub(crate) positron_capture_on_neutron_per_second: f64,
    pub(crate) electron_antineutrino_capture_on_proton_per_second: f64,
    pub(crate) free_neutron_decay_per_second: f64,
    pub(crate) inverse_neutron_decay_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BornWeakRates {
    pub(crate) channels: BornWeakChannelRates,
    /// Complete multi-body thermal-radiative correction; it is not assigned
    /// artificially to any one of the six Born channels.
    pub(crate) thermal_radiative_neutron_to_proton_per_second: f64,
    pub(crate) thermal_radiative_proton_to_neutron_per_second: f64,
    pub(crate) neutron_to_proton_per_second: f64,
    pub(crate) proton_to_neutron_per_second: f64,
}

#[derive(Debug, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }
}

/// Analytic zero-temperature neutron-decay phase-space normalization.
///
/// With `q = (m_n - m_p) / m_e`,
///
/// `I0 = sqrt(q^2 - 1) (2 q^4 - 9 q^2 - 8) / 60 + q acosh(q) / 4`.
pub(crate) fn born_neutron_decay_normalization() -> f64 {
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let momentum = (q * q - 1.0).sqrt();
    momentum * (2.0 * q.powi(4) - 9.0 * q * q - 8.0) / 60.0 + q * q.acosh() / 4.0
}

/// Evaluate the requested weak-rate model.
pub(crate) fn evaluate_weak_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
    model: WeakRateModel,
) -> Result<BornWeakRates, BornWeakError> {
    match model {
        WeakRateModel::Born => evaluate_born_weak_rates_impl(
            photon_temperature_mev,
            neutrino_temperature_mev,
            neutron_lifetime_seconds,
            quadrature_order,
        ),
        WeakRateModel::PrimatZeroTemperatureCcr => evaluate_primat_zero_temperature_ccr_rates(
            photon_temperature_mev,
            neutrino_temperature_mev,
            neutron_lifetime_seconds,
            quadrature_order,
        ),
        WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism => {
            evaluate_primat_zero_temperature_ccr_finite_mass_no_weak_magnetism_rates(
                photon_temperature_mev,
                neutrino_temperature_mev,
                neutron_lifetime_seconds,
                quadrature_order,
            )
        }
        WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism => {
            evaluate_primat_zero_temperature_ccr_finite_mass_physical_weak_magnetism_rates(
                photon_temperature_mev,
                neutrino_temperature_mev,
                neutron_lifetime_seconds,
                quadrature_order,
            )
        }
        WeakRateModel::PrimatCompleteThermalRadiativePhysicalWeakMagnetism => {
            evaluate_primat_complete_thermal_radiative_physical_weak_magnetism_rates(
                photon_temperature_mev,
                neutrino_temperature_mev,
                neutron_lifetime_seconds,
                quadrature_order,
            )
        }
    }
}

/// Evaluate all six tree-level charged-current rates.
///
/// This compatibility wrapper intentionally dispatches to the unchanged Born
/// implementation, preserving the pre-CCR numerical path bit for bit.
pub(crate) fn evaluate_born_weak_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    evaluate_weak_rates(
        photon_temperature_mev,
        neutrino_temperature_mev,
        neutron_lifetime_seconds,
        quadrature_order,
        WeakRateModel::Born,
    )
}

fn evaluate_born_weak_rates_impl(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    validate_positive_temperature("neutrino_temperature_mev", neutrino_temperature_mev)?;
    if !neutron_lifetime_seconds.is_finite() || neutron_lifetime_seconds <= 0.0 {
        return Err(BornWeakError::InvalidNeutronLifetime {
            raw_value_seconds: neutron_lifetime_seconds,
        });
    }

    let nodes = gauss_legendre_unit_nodes(quadrature_order)?;
    let t_gamma = photon_temperature_mev / ELECTRON_MASS_MEV;
    let t_nu = neutrino_temperature_mev / ELECTRON_MASS_MEV;
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    // A common positive scale keeps each forward/reverse pair on identical
    // nodes (so equilibrium detailed balance is tested without interpolation)
    // while following the support of whichever thermal bath is hotter.
    let tail_scale = t_gamma + t_nu;

    let (nu_capture, electron_capture) = integrate_semi_infinite_pair(
        &nodes,
        tail_scale,
        BornWeakProcess::ElectronNeutrinoCaptureOnNeutron,
        BornWeakProcess::ElectronCaptureOnProton,
        |neutrino_energy| {
            let electron_energy = neutrino_energy + q;
            let electron_momentum = (electron_energy * electron_energy - 1.0).sqrt();
            let phase_space = neutrino_energy.powi(2) * electron_energy * electron_momentum;
            let f_nu = fermi_dirac(neutrino_energy, t_nu);
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            (
                phase_space * f_nu * (1.0 - f_electron),
                phase_space * f_electron * (1.0 - f_nu),
            )
        },
    )?;

    let (positron_capture, antineutrino_capture) = integrate_semi_infinite_pair(
        &nodes,
        tail_scale,
        BornWeakProcess::PositronCaptureOnNeutron,
        BornWeakProcess::ElectronAntineutrinoCaptureOnProton,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q + electron_energy;
            let phase_space = electron_momentum.powi(2) * antineutrino_energy.powi(2);
            let f_positron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            (
                phase_space * f_positron * (1.0 - f_antineutrino),
                phase_space * f_antineutrino * (1.0 - f_positron),
            )
        },
    )?;

    let endpoint_momentum = (q * q - 1.0).sqrt();
    let (free_decay, inverse_decay) = integrate_bounded_pair(
        &nodes,
        endpoint_momentum,
        BornWeakProcess::FreeNeutronDecay,
        BornWeakProcess::InverseNeutronDecay,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q - electron_energy;
            let phase_space = electron_momentum.powi(2) * antineutrino_energy.powi(2);
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            (
                phase_space * (1.0 - f_electron) * (1.0 - f_antineutrino),
                phase_space * f_electron * f_antineutrino,
            )
        },
    )?;

    let normalization = neutron_lifetime_seconds * born_neutron_decay_normalization();
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(BornWeakError::InvalidNormalization {
            raw_lifetime_seconds: neutron_lifetime_seconds,
            raw_normalization: normalization,
        });
    }
    let channels = BornWeakChannelRates {
        electron_neutrino_capture_on_neutron_per_second: nu_capture / normalization,
        electron_capture_on_proton_per_second: electron_capture / normalization,
        positron_capture_on_neutron_per_second: positron_capture / normalization,
        electron_antineutrino_capture_on_proton_per_second: antineutrino_capture / normalization,
        free_neutron_decay_per_second: free_decay / normalization,
        inverse_neutron_decay_per_second: inverse_decay / normalization,
    };
    let neutron_to_proton_per_second = channels.electron_neutrino_capture_on_neutron_per_second
        + channels.positron_capture_on_neutron_per_second
        + channels.free_neutron_decay_per_second;
    let proton_to_neutron_per_second = channels.electron_capture_on_proton_per_second
        + channels.electron_antineutrino_capture_on_proton_per_second
        + channels.inverse_neutron_decay_per_second;
    validate_total_rate("neutron_to_proton", neutron_to_proton_per_second)?;
    validate_total_rate("proton_to_neutron", proton_to_neutron_per_second)?;

    Ok(BornWeakRates {
        channels,
        thermal_radiative_neutron_to_proton_per_second: 0.0,
        thermal_radiative_proton_to_neutron_per_second: 0.0,
        neutron_to_proton_per_second,
        proton_to_neutron_per_second,
    })
}

fn evaluate_primat_zero_temperature_ccr_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    validate_positive_temperature("neutrino_temperature_mev", neutrino_temperature_mev)?;
    if !neutron_lifetime_seconds.is_finite() || neutron_lifetime_seconds <= 0.0 {
        return Err(BornWeakError::InvalidNeutronLifetime {
            raw_value_seconds: neutron_lifetime_seconds,
        });
    }

    let nodes = gauss_legendre_unit_nodes(quadrature_order)?;
    let t_gamma = photon_temperature_mev / ELECTRON_MASS_MEV;
    let t_nu = neutrino_temperature_mev / ELECTRON_MASS_MEV;
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let tail_scale = t_gamma + t_nu;

    let (nu_capture, electron_capture) = integrate_semi_infinite_pair(
        &nodes,
        tail_scale,
        BornWeakProcess::ElectronNeutrinoCaptureOnNeutron,
        BornWeakProcess::ElectronCaptureOnProton,
        |neutrino_energy| {
            let electron_energy = neutrino_energy + q;
            let phase_space =
                corrected_neutrino_electron_phase_space(neutrino_energy, electron_energy);
            let f_nu = fermi_dirac(neutrino_energy, t_nu);
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            (
                phase_space * f_nu * (1.0 - f_electron),
                phase_space * f_electron * (1.0 - f_nu),
            )
        },
    )?;

    let (positron_capture, antineutrino_capture) = integrate_semi_infinite_pair(
        &nodes,
        tail_scale,
        BornWeakProcess::PositronCaptureOnNeutron,
        BornWeakProcess::ElectronAntineutrinoCaptureOnProton,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q + electron_energy;
            // PRIMAT's _fermi_stat is unity on this positron branch: there is
            // no heuristic repulsive F(-1) factor.
            let phase_space = corrected_positron_momentum_phase_space(
                electron_momentum,
                antineutrino_energy,
                electron_energy,
            );
            let f_positron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            (
                phase_space * f_positron * (1.0 - f_antineutrino),
                phase_space * f_antineutrino * (1.0 - f_positron),
            )
        },
    )?;

    let endpoint_momentum = (q * q - 1.0).sqrt();
    let (free_decay, inverse_decay) = integrate_bounded_pair(
        &nodes,
        endpoint_momentum,
        BornWeakProcess::FreeNeutronDecay,
        BornWeakProcess::InverseNeutronDecay,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q - electron_energy;
            let phase_space = corrected_electron_momentum_phase_space(
                electron_momentum,
                antineutrino_energy,
                electron_energy,
            );
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            (
                phase_space * (1.0 - f_electron) * (1.0 - f_antineutrino),
                phase_space * f_electron * f_antineutrino,
            )
        },
    )?;

    let decay_normalization = corrected_neutron_decay_normalization(&nodes, q)?;
    let normalization = neutron_lifetime_seconds * decay_normalization;
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(BornWeakError::InvalidNormalization {
            raw_lifetime_seconds: neutron_lifetime_seconds,
            raw_normalization: normalization,
        });
    }
    let channels = BornWeakChannelRates {
        electron_neutrino_capture_on_neutron_per_second: nu_capture / normalization,
        electron_capture_on_proton_per_second: electron_capture / normalization,
        positron_capture_on_neutron_per_second: positron_capture / normalization,
        electron_antineutrino_capture_on_proton_per_second: antineutrino_capture / normalization,
        free_neutron_decay_per_second: free_decay / normalization,
        inverse_neutron_decay_per_second: inverse_decay / normalization,
    };
    let neutron_to_proton_per_second = channels.electron_neutrino_capture_on_neutron_per_second
        + channels.positron_capture_on_neutron_per_second
        + channels.free_neutron_decay_per_second;
    let proton_to_neutron_per_second = channels.electron_capture_on_proton_per_second
        + channels.electron_antineutrino_capture_on_proton_per_second
        + channels.inverse_neutron_decay_per_second;
    validate_total_rate("neutron_to_proton", neutron_to_proton_per_second)?;
    validate_total_rate("proton_to_neutron", proton_to_neutron_per_second)?;

    Ok(BornWeakRates {
        channels,
        thermal_radiative_neutron_to_proton_per_second: 0.0,
        thermal_radiative_proton_to_neutron_per_second: 0.0,
        neutron_to_proton_per_second,
        proton_to_neutron_per_second,
    })
}

/// PRIMAT zero-temperature Coulomb/radiative rates plus the signed,
/// first-order finite-nucleon-mass Fokker-Planck correction.
///
/// The correction is evaluated in physical-channel coordinates so every
/// radiative `u = 0` singularity is an integration endpoint.  It is additive,
/// not a scalar rescaling of the CCR integrand.  This F08B model deliberately
/// fixes the anomalous weak-magnetism coefficient to zero.
fn evaluate_primat_zero_temperature_ccr_finite_mass_no_weak_magnetism_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    evaluate_primat_zero_temperature_ccr_finite_mass_rates(
        photon_temperature_mev,
        neutrino_temperature_mev,
        neutron_lifetime_seconds,
        quadrature_order,
        F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
    )
}

/// F08C adds physical anomalous weak magnetism to the already validated F08B
/// finite-mass source.  The rest of the CCR and finite-mass path is shared.
fn evaluate_primat_zero_temperature_ccr_finite_mass_physical_weak_magnetism_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    evaluate_primat_zero_temperature_ccr_finite_mass_rates(
        photon_temperature_mev,
        neutrino_temperature_mev,
        neutron_lifetime_seconds,
        quadrature_order,
        PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
    )
}

/// F08D complete finite-temperature radiative block on the F08C base.
///
/// The four Brown--Sawyer terms are retained as one signed directional
/// correction.  They share the unchanged physical-WM vacuum normalization;
/// no thermal term is inserted into `F_n` and no correction is clipped.
fn evaluate_primat_complete_thermal_radiative_physical_weak_magnetism_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
) -> Result<BornWeakRates, BornWeakError> {
    let mut rates = evaluate_primat_zero_temperature_ccr_finite_mass_physical_weak_magnetism_rates(
        photon_temperature_mev,
        neutrino_temperature_mev,
        neutron_lifetime_seconds,
        quadrature_order,
    )?;
    let raw = complete_thermal_radiative_raw(photon_temperature_mev, neutrino_temperature_mev)?;
    let nodes = gauss_legendre_unit_nodes(quadrature_order)?;
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let proton_mass = PROTON_MASS_MEV / ELECTRON_MASS_MEV;
    let decay_normalization = finite_mass_decay_normalization(
        &nodes,
        q,
        proton_mass,
        PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
    )?;
    let normalization = neutron_lifetime_seconds * decay_normalization;
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(BornWeakError::InvalidNormalization {
            raw_lifetime_seconds: neutron_lifetime_seconds,
            raw_normalization: normalization,
        });
    }
    let neutron_correction = raw.neutron_to_proton / normalization;
    let proton_correction = raw.proton_to_neutron / normalization;
    rates.thermal_radiative_neutron_to_proton_per_second = neutron_correction;
    rates.thermal_radiative_proton_to_neutron_per_second = proton_correction;
    rates.neutron_to_proton_per_second += neutron_correction;
    rates.proton_to_neutron_per_second += proton_correction;
    validate_total_rate("neutron_to_proton", rates.neutron_to_proton_per_second)?;
    validate_total_rate("proton_to_neutron", rates.proton_to_neutron_per_second)?;
    Ok(rates)
}

fn evaluate_primat_zero_temperature_ccr_finite_mass_rates(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    neutron_lifetime_seconds: f64,
    quadrature_order: usize,
    anomalous_weak_magnetism_coefficient: f64,
) -> Result<BornWeakRates, BornWeakError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    validate_positive_temperature("neutrino_temperature_mev", neutrino_temperature_mev)?;
    if !neutron_lifetime_seconds.is_finite() || neutron_lifetime_seconds <= 0.0 {
        return Err(BornWeakError::InvalidNeutronLifetime {
            raw_value_seconds: neutron_lifetime_seconds,
        });
    }

    let nodes = gauss_legendre_unit_nodes(quadrature_order)?;
    let t_gamma = photon_temperature_mev / ELECTRON_MASS_MEV;
    let t_nu = neutrino_temperature_mev / ELECTRON_MASS_MEV;
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let proton_mass = PROTON_MASS_MEV / ELECTRON_MASS_MEV;
    let neutron_mass = NEUTRON_MASS_MEV / ELECTRON_MASS_MEV;
    let finite_mass_parameters = FiniteMassParameters {
        proton_mass,
        neutron_mass,
        anomalous_weak_magnetism_coefficient,
    };
    // y = E - q >= 0.  The finite-mass source is originally integrated in
    // electron momentum, hence p^2 dp = p E dy supplies the explicit p E
    // Jacobian below.
    let (nu_capture, electron_capture) = integrate_semi_infinite_pair_with_scales(
        &nodes,
        t_nu,
        t_gamma,
        BornWeakProcess::ElectronNeutrinoCaptureOnNeutron,
        BornWeakProcess::ElectronCaptureOnProton,
        |neutrino_energy| {
            let electron_energy = neutrino_energy + q;
            let electron_momentum = (electron_energy * electron_energy - 1.0).sqrt();
            let beta = electron_momentum / electron_energy;
            let radiative =
                resummed_zero_temperature_radiative_factor(beta, neutrino_energy, electron_energy);
            let ccr_factor = fermi_coulomb_factor(beta) * radiative;
            let phase_space =
                corrected_neutrino_electron_phase_space(neutrino_energy, electron_energy);
            let f_nu = fermi_dirac(neutrino_energy, t_nu);
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            let momentum_jacobian = electron_momentum * electron_energy;
            let forward_correction = momentum_jacobian
                * finite_mass_chi(
                    electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    1,
                    finite_mass_parameters,
                )
                * ccr_factor;
            let reverse_correction = momentum_jacobian
                * finite_mass_chi(
                    -electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    -1,
                    finite_mass_parameters,
                )
                * ccr_factor;
            (
                phase_space * f_nu * (1.0 - f_electron) + forward_correction,
                phase_space * f_electron * (1.0 - f_nu) + reverse_correction,
            )
        },
    )?;

    let (positron_capture, antineutrino_capture) = integrate_semi_infinite_pair_with_scales(
        &nodes,
        t_gamma,
        t_nu,
        BornWeakProcess::PositronCaptureOnNeutron,
        BornWeakProcess::ElectronAntineutrinoCaptureOnProton,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q + electron_energy;
            let beta = electron_momentum / electron_energy;
            let radiative = resummed_zero_temperature_radiative_factor(
                beta,
                antineutrino_energy,
                electron_energy,
            );
            let phase_space = corrected_positron_momentum_phase_space(
                electron_momentum,
                antineutrino_energy,
                electron_energy,
            );
            let f_positron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            let momentum_squared = electron_momentum.powi(2);
            let forward_correction = momentum_squared
                * finite_mass_chi(
                    -electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    1,
                    finite_mass_parameters,
                )
                * radiative;
            let reverse_correction = momentum_squared
                * finite_mass_chi(
                    electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    -1,
                    finite_mass_parameters,
                )
                * radiative;
            (
                phase_space * f_positron * (1.0 - f_antineutrino) + forward_correction,
                phase_space * f_antineutrino * (1.0 - f_positron) + reverse_correction,
            )
        },
    )?;

    let endpoint_momentum = (q * q - 1.0).sqrt();
    let (free_decay, inverse_decay) = integrate_bounded_pair(
        &nodes,
        endpoint_momentum,
        BornWeakProcess::FreeNeutronDecay,
        BornWeakProcess::InverseNeutronDecay,
        |electron_momentum| {
            let electron_energy = electron_momentum.hypot(1.0);
            let antineutrino_energy = q - electron_energy;
            let beta = electron_momentum / electron_energy;
            let radiative = resummed_zero_temperature_radiative_factor(
                beta,
                antineutrino_energy,
                electron_energy,
            );
            let ccr_factor = fermi_coulomb_factor(beta) * radiative;
            let phase_space = corrected_electron_momentum_phase_space(
                electron_momentum,
                antineutrino_energy,
                electron_energy,
            );
            let f_electron = fermi_dirac(electron_energy, t_gamma);
            let f_antineutrino = fermi_dirac(antineutrino_energy, t_nu);
            let momentum_squared = electron_momentum.powi(2);
            let forward_correction = momentum_squared
                * finite_mass_chi(
                    electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    1,
                    finite_mass_parameters,
                )
                * ccr_factor;
            let reverse_correction = momentum_squared
                * finite_mass_chi(
                    -electron_energy,
                    electron_momentum,
                    t_gamma,
                    t_nu,
                    -1,
                    finite_mass_parameters,
                )
                * ccr_factor;
            (
                phase_space * (1.0 - f_electron) * (1.0 - f_antineutrino) + forward_correction,
                phase_space * f_electron * f_antineutrino + reverse_correction,
            )
        },
    )?;

    let decay_normalization = finite_mass_decay_normalization(
        &nodes,
        q,
        proton_mass,
        anomalous_weak_magnetism_coefficient,
    )?;
    let normalization = neutron_lifetime_seconds * decay_normalization;
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(BornWeakError::InvalidNormalization {
            raw_lifetime_seconds: neutron_lifetime_seconds,
            raw_normalization: normalization,
        });
    }
    let channels = BornWeakChannelRates {
        electron_neutrino_capture_on_neutron_per_second: nu_capture / normalization,
        electron_capture_on_proton_per_second: electron_capture / normalization,
        positron_capture_on_neutron_per_second: positron_capture / normalization,
        electron_antineutrino_capture_on_proton_per_second: antineutrino_capture / normalization,
        free_neutron_decay_per_second: free_decay / normalization,
        inverse_neutron_decay_per_second: inverse_decay / normalization,
    };
    let neutron_to_proton_per_second = channels.electron_neutrino_capture_on_neutron_per_second
        + channels.positron_capture_on_neutron_per_second
        + channels.free_neutron_decay_per_second;
    let proton_to_neutron_per_second = channels.electron_capture_on_proton_per_second
        + channels.electron_antineutrino_capture_on_proton_per_second
        + channels.inverse_neutron_decay_per_second;
    validate_total_rate("neutron_to_proton", neutron_to_proton_per_second)?;
    validate_total_rate("proton_to_neutron", proton_to_neutron_per_second)?;

    Ok(BornWeakRates {
        channels,
        thermal_radiative_neutron_to_proton_per_second: 0.0,
        thermal_radiative_proton_to_neutron_per_second: 0.0,
        neutron_to_proton_per_second,
        proton_to_neutron_per_second,
    })
}

fn finite_mass_decay_normalization(
    nodes: &[(f64, f64)],
    q: f64,
    proton_mass: f64,
    anomalous_weak_magnetism_coefficient: f64,
) -> Result<f64, BornWeakError> {
    Ok(corrected_neutron_decay_normalization(nodes, q)?
        + finite_mass_neutron_decay_normalization_correction(
            nodes,
            q,
            proton_mass,
            anomalous_weak_magnetism_coefficient,
        )?)
}

fn corrected_neutron_decay_normalization(
    nodes: &[(f64, f64)],
    q: f64,
) -> Result<f64, BornWeakError> {
    let endpoint_momentum = (q * q - 1.0).sqrt();
    let mut integral = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let electron_momentum = endpoint_momentum * unit_node;
        let electron_energy = electron_momentum.hypot(1.0);
        let antineutrino_energy = q - electron_energy;
        let phase_space = corrected_electron_momentum_phase_space(
            electron_momentum,
            antineutrino_energy,
            electron_energy,
        );
        integral.add(unit_weight * endpoint_momentum * phase_space);
    }
    checked_integral(BornWeakProcess::FreeNeutronDecay, integral.sum)
}

#[derive(Debug, Clone, Copy)]
struct FiniteMassDerivativeKernels {
    g20: f64,
    g30: f64,
    g21: f64,
    g31: f64,
    g41: f64,
    g22: f64,
    g32: f64,
    g42: f64,
}

#[derive(Debug, Clone, Copy)]
struct FiniteMassParameters {
    proton_mass: f64,
    neutron_mass: f64,
    anomalous_weak_magnetism_coefficient: f64,
}

/// Compact forms of d^j/du^j [u^n f_FD(z_nu u)] used by PRIMAT's
/// Fokker-Planck finite-mass correction at zero neutrino chemical potential.
fn finite_mass_derivative_kernels(u: f64, z_nu: f64) -> FiniteMassDerivativeKernels {
    if !u.is_finite() || !z_nu.is_finite() || z_nu <= 0.0 {
        return FiniteMassDerivativeKernels {
            g20: f64::NAN,
            g30: f64::NAN,
            g21: f64::NAN,
            g31: f64::NAN,
            g41: f64::NAN,
            g22: f64::NAN,
            g32: f64::NAN,
            g42: f64::NAN,
        };
    }

    let z = z_nu * u;
    let f = fermi_dirac_from_exponent(z);
    let h = 1.0 - f;
    let (
        first_bracket_2,
        first_bracket_3,
        first_bracket_4,
        second_bracket_2,
        second_bracket_3,
        second_bracket_4,
    ) = if f == 0.0 {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    } else if h == 0.0 {
        (2.0, 3.0, 4.0, 2.0, 6.0, 12.0)
    } else {
        let curvature = z.powi(2) * h * (1.0 - 2.0 * f);
        (
            2.0 - z * h,
            3.0 - z * h,
            4.0 - z * h,
            2.0 - 4.0 * z * h + curvature,
            6.0 - 6.0 * z * h + curvature,
            12.0 - 8.0 * z * h + curvature,
        )
    };

    FiniteMassDerivativeKernels {
        g20: u.powi(2) * f,
        g30: u.powi(3) * f,
        g21: u * f * first_bracket_2,
        g31: u.powi(2) * f * first_bracket_3,
        g41: u.powi(3) * f * first_bracket_4,
        g22: f * second_bracket_2,
        g32: u * f * second_bracket_3,
        g42: u.powi(2) * f * second_bracket_4,
    }
}

fn finite_mass_couplings(
    direction: i8,
    anomalous_weak_magnetism_coefficient: f64,
) -> (f64, f64, f64) {
    if (direction != 1 && direction != -1) || !anomalous_weak_magnetism_coefficient.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let signed_direction = f64::from(direction);
    let denominator = 1.0 + 3.0 * NUCLEON_AXIAL_COUPLING.powi(2);
    let f1 = ((1.0 + signed_direction * NUCLEON_AXIAL_COUPLING).powi(2)
        + 2.0 * anomalous_weak_magnetism_coefficient * signed_direction * NUCLEON_AXIAL_COUPLING)
        / denominator;
    let f2 = ((1.0 - signed_direction * NUCLEON_AXIAL_COUPLING).powi(2)
        - 2.0 * anomalous_weak_magnetism_coefficient * signed_direction * NUCLEON_AXIAL_COUPLING)
        / denominator;
    let f3 = (NUCLEON_AXIAL_COUPLING.powi(2) - 1.0) / denominator;
    (f1, f2, f3)
}

/// Signed PRIMAT finite-mass chi for one signed-electron-energy branch.
///
/// `direction=+1` is n->p and uses the daughter proton mass; `direction=-1`
/// is p->n and uses the daughter neutron mass.  The returned value already
/// contains the electron and neutrino occupation/blocking factors.
fn finite_mass_chi(
    signed_electron_energy: f64,
    electron_momentum: f64,
    t_gamma: f64,
    t_nu: f64,
    direction: i8,
    parameters: FiniteMassParameters,
) -> f64 {
    if !signed_electron_energy.is_finite()
        || signed_electron_energy.abs() < 1.0
        || !electron_momentum.is_finite()
        || electron_momentum < 0.0
        || !t_gamma.is_finite()
        || t_gamma <= 0.0
        || !t_nu.is_finite()
        || t_nu <= 0.0
        || !parameters.proton_mass.is_finite()
        || parameters.proton_mass <= 0.0
        || !parameters.neutron_mass.is_finite()
        || parameters.neutron_mass <= 0.0
        || (direction != 1 && direction != -1)
        || !parameters.anomalous_weak_magnetism_coefficient.is_finite()
    {
        return f64::NAN;
    }

    let signed_direction = f64::from(direction);
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let nucleon_mass = if direction == 1 {
        parameters.proton_mass
    } else {
        parameters.neutron_mass
    };
    let x = 1.0 / t_gamma;
    let z_nu = 1.0 / t_nu;
    let neutrino_energy = signed_electron_energy - signed_direction * q;
    let kernels = finite_mass_derivative_kernels(neutrino_energy, z_nu);
    let electron_factor = fermi_dirac_from_exponent(-signed_electron_energy * x);
    let (f1, f2, f3) =
        finite_mass_couplings(direction, parameters.anomalous_weak_magnetism_coefficient);
    let coupling_sum = f1 + f2 + f3;
    let momentum_squared = electron_momentum.powi(2);

    electron_factor
        * (f1 * kernels.g20 * momentum_squared / (nucleon_mass * signed_electron_energy)
            - f2 * kernels.g30 / nucleon_mass
            + coupling_sum / (2.0 * x * nucleon_mass)
                * (kernels.g42 + kernels.g22 * momentum_squared)
            + coupling_sum / (2.0 * nucleon_mass) * (kernels.g41 + kernels.g21 * momentum_squared)
            - (f1 + f2) / (x * nucleon_mass)
                * (kernels.g31 + kernels.g21 * momentum_squared / (-signed_electron_energy))
            - 3.0 * f3 / (x * nucleon_mass) * kernels.g20
            + f3 / (3.0 * nucleon_mass) * kernels.g31 * momentum_squared / signed_electron_energy
            + f3 / (3.0 * x * nucleon_mass) * kernels.g32 * momentum_squared
                / signed_electron_energy
            - coupling_sum * 3.0 / (2.0 * x)
                * (1.0
                    - (parameters.neutron_mass / parameters.proton_mass)
                        .powi(i32::from(direction)))
                * kernels.g21)
}

/// Vacuum n->p limit of `finite_mass_chi`, using mp/me.
fn finite_mass_vacuum_chi(
    electron_energy: f64,
    electron_momentum: f64,
    q: f64,
    proton_mass: f64,
    anomalous_weak_magnetism_coefficient: f64,
) -> f64 {
    if !electron_energy.is_finite()
        || electron_energy < 1.0
        || !electron_momentum.is_finite()
        || electron_momentum < 0.0
        || !q.is_finite()
        || !proton_mass.is_finite()
        || proton_mass <= 0.0
        || !anomalous_weak_magnetism_coefficient.is_finite()
    {
        return f64::NAN;
    }
    let (f1, f2, f3) = finite_mass_couplings(1, anomalous_weak_magnetism_coefficient);
    let coupling_sum = f1 + f2 + f3;
    let neutrino_energy = electron_energy - q;
    let momentum_squared = electron_momentum.powi(2);
    f1 * neutrino_energy.powi(2) * momentum_squared / (proton_mass * electron_energy)
        - f2 * neutrino_energy.powi(3) / proton_mass
        + coupling_sum / (2.0 * proton_mass)
            * (4.0 * neutrino_energy.powi(3) + 2.0 * neutrino_energy * momentum_squared)
        + f3 / proton_mass * neutrino_energy.powi(2) * momentum_squared / electron_energy
}

fn finite_mass_neutron_decay_normalization_correction(
    nodes: &[(f64, f64)],
    q: f64,
    proton_mass: f64,
    anomalous_weak_magnetism_coefficient: f64,
) -> Result<f64, BornWeakError> {
    let endpoint_momentum = (q * q - 1.0).sqrt();
    let mut integral = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let electron_momentum = endpoint_momentum * unit_node;
        let electron_energy = electron_momentum.hypot(1.0);
        let antineutrino_energy = q - electron_energy;
        let beta = electron_momentum / electron_energy;
        let correction = electron_momentum.powi(2)
            * finite_mass_vacuum_chi(
                electron_energy,
                electron_momentum,
                q,
                proton_mass,
                anomalous_weak_magnetism_coefficient,
            )
            * fermi_coulomb_factor(beta)
            * resummed_zero_temperature_radiative_factor(
                beta,
                antineutrino_energy,
                electron_energy,
            );
        integral.add(unit_weight * endpoint_momentum * correction);
    }
    if !integral.sum.is_finite() {
        return Err(BornWeakError::NonFiniteIntegral {
            process: BornWeakProcess::FreeNeutronDecay,
            raw_value: integral.sum,
        });
    }
    // This is a signed additive correction; a negative value is physical.
    Ok(integral.sum)
}

fn corrected_neutrino_electron_phase_space(neutrino_energy: f64, electron_energy: f64) -> f64 {
    if !electron_energy.is_finite() || electron_energy < 1.0 {
        return f64::NAN;
    }
    let electron_momentum = (electron_energy * electron_energy - 1.0).sqrt();
    let beta = electron_momentum / electron_energy;
    electron_energy.powi(2)
        * beta_times_fermi_coulomb_factor(beta)
        * neutrino_energy_squared_times_radiative_factor(beta, neutrino_energy, electron_energy)
}

fn corrected_electron_momentum_phase_space(
    electron_momentum: f64,
    neutrino_energy: f64,
    electron_energy: f64,
) -> f64 {
    if !electron_momentum.is_finite()
        || electron_momentum < 0.0
        || !electron_energy.is_finite()
        || electron_energy < 1.0
    {
        return f64::NAN;
    }
    let beta = electron_momentum / electron_energy;
    electron_momentum
        * electron_energy
        * beta_times_fermi_coulomb_factor(beta)
        * neutrino_energy_squared_times_radiative_factor(beta, neutrino_energy, electron_energy)
}

fn corrected_positron_momentum_phase_space(
    electron_momentum: f64,
    neutrino_energy: f64,
    electron_energy: f64,
) -> f64 {
    if !electron_momentum.is_finite()
        || electron_momentum < 0.0
        || !electron_energy.is_finite()
        || electron_energy < 1.0
    {
        return f64::NAN;
    }
    let beta = electron_momentum / electron_energy;
    electron_momentum.powi(2)
        * neutrino_energy_squared_times_radiative_factor(beta, neutrino_energy, electron_energy)
}

/// The finite physical product beta F(beta), including its beta=0 limit.
fn beta_times_fermi_coulomb_factor(beta: f64) -> f64 {
    if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
        return f64::NAN;
    }
    if beta < FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD {
        return threshold_beta_times_fermi_coulomb();
    }
    beta * fermi_coulomb_factor(beta)
}

/// The finite physical product y^2 R(beta,y,E), including its y=0 limit.
fn neutrino_energy_squared_times_radiative_factor(
    beta: f64,
    neutrino_energy: f64,
    electron_energy: f64,
) -> f64 {
    if neutrino_energy == 0.0
        && beta.is_finite()
        && (0.0..1.0).contains(&beta)
        && electron_energy.is_finite()
        && electron_energy >= 1.0
    {
        return 0.0;
    }
    neutrino_energy.powi(2)
        * resummed_zero_temperature_radiative_factor(beta, neutrino_energy, electron_energy)
}

fn threshold_beta_times_fermi_coulomb() -> f64 {
    let gamma = (1.0 - FINE_STRUCTURE_CONSTANT.powi(2)).sqrt() - 1.0;
    let compton_wavelength_cm =
        HBAR_ERG_SECONDS * SPEED_OF_LIGHT_CM_PER_SECOND / (ELECTRON_MASS_MEV * MEV_IN_ERG);
    let log_product = (gamma / 2.0).ln_1p()
        + 4.0_f64.ln()
        + 2.0 * gamma * (2.0 * PROTON_CHARGE_RADIUS_CM / compton_wavelength_cm).ln()
        - 2.0 * log_abs_gamma_lanczos(3.0 + 2.0 * gamma, 0.0)
        + (2.0 * PI).ln()
        + (1.0 + 2.0 * gamma) * FINE_STRUCTURE_CONSTANT.ln();
    log_product.exp()
}

/// Exact relativistic Fermi function used by PRIMAT v0.3.2.
///
/// Below beta=1e-6 it uses the leading threshold asymptotic F=C/beta; above
/// that threshold it evaluates the exact expression in log space.  The
/// finite product beta F is consumed directly by the phase-space helpers.
pub(crate) fn fermi_coulomb_factor(beta: f64) -> f64 {
    if !beta.is_finite() || beta <= 0.0 || beta >= 1.0 {
        return f64::NAN;
    }
    if beta < FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD {
        return threshold_beta_times_fermi_coulomb() / beta;
    }
    let gamma = (1.0 - FINE_STRUCTURE_CONSTANT.powi(2)).sqrt() - 1.0;
    let compton_wavelength_cm =
        HBAR_ERG_SECONDS * SPEED_OF_LIGHT_CM_PER_SECOND / (ELECTRON_MASS_MEV * MEV_IN_ERG);
    let log_factor = (gamma / 2.0).ln_1p()
        + 4.0_f64.ln()
        + 2.0 * gamma * (2.0 * PROTON_CHARGE_RADIUS_CM * beta / compton_wavelength_cm).ln()
        - 2.0 * log_abs_gamma_lanczos(3.0 + 2.0 * gamma, 0.0)
        + PI * FINE_STRUCTURE_CONSTANT / beta
        - gamma * (-beta * beta).ln_1p()
        + 2.0 * log_abs_gamma_lanczos(1.0 + gamma, FINE_STRUCTURE_CONSTANT / beta);
    log_factor.exp()
}

/// PRIMAT v0.3.2's resummed zero-temperature radiative factor.
fn resummed_zero_temperature_radiative_factor(
    beta: f64,
    neutrino_energy: f64,
    electron_energy: f64,
) -> f64 {
    if !beta.is_finite()
        || !(0.0..1.0).contains(&beta)
        || !neutrino_energy.is_finite()
        || neutrino_energy <= 0.0
        || !electron_energy.is_finite()
        || electron_energy < 1.0
    {
        return f64::NAN;
    }

    let sirlin = if beta == 0.0 {
        // atanh(beta)/beta -> 1 and Li_2(2 beta/(1+beta))/beta -> 2.
        3.0 * (PROTON_MASS_MEV / ELECTRON_MASS_MEV).ln() - 27.0 / 4.0
            + neutrino_energy.powi(2) / (6.0 * electron_energy.powi(2))
    } else {
        let rd = beta.atanh() / beta;
        let dilogarithm_argument = 2.0 * beta / (1.0 + beta);
        3.0 * (PROTON_MASS_MEV / ELECTRON_MASS_MEV).ln() - 0.75
            + 4.0
                * (rd - 1.0)
                * (neutrino_energy / (3.0 * electron_energy) - 1.5 + (2.0 * neutrino_energy).ln())
            + rd * (2.0 * (1.0 + beta * beta)
                + neutrino_energy.powi(2) / (6.0 * electron_energy.powi(2))
                - 4.0 * beta * rd)
            - 4.0 / beta * dilogarithm_unit_interval(dilogarithm_argument)
    };
    let mass_difference_mev = NEUTRON_PROTON_MASS_DIFFERENCE_MEV;
    let outer = 1.0
        + FINE_STRUCTURE_CONSTANT / (2.0 * PI)
            * (sirlin - 3.0 * (PROTON_MASS_MEV / (2.0 * mass_difference_mev)).ln());
    let long_distance = 1.020_94 + FINE_STRUCTURE_CONSTANT / PI * 0.891 - 0.000_43;
    let short_distance =
        1.022_48 + 1.0 / (134.0 * 2.0 * PI) * ((PROTON_MASS_MEV / 1_200.0).ln() - 0.34) - 0.000_1;
    outer * long_distance * short_distance
}

/// Li_2(x) on 0 <= x <= 1, with the reflection identity near one.
fn dilogarithm_unit_interval(argument: f64) -> f64 {
    if !argument.is_finite() || !(0.0..=1.0).contains(&argument) {
        return f64::NAN;
    }
    if argument == 0.0 {
        return 0.0;
    }
    if argument == 1.0 {
        return PI.powi(2) / 6.0;
    }
    if argument > 0.5 {
        return PI.powi(2) / 6.0
            - argument.ln() * (-argument).ln_1p()
            - dilogarithm_unit_interval(1.0 - argument);
    }

    let mut sum = 0.0;
    let mut power = argument;
    for index in 1_u32..=256 {
        let denominator = f64::from(index).powi(2);
        let term = power / denominator;
        sum += term;
        if term.abs() <= f64::EPSILON * sum.abs() {
            break;
        }
        power *= argument;
    }
    sum
}

/// Real part of log Gamma(real + i imaginary), using the g=7 Lanczos series.
fn log_abs_gamma_lanczos(real: f64, imaginary: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let shifted_real = real - 1.0;
    let mut series_real = COEFFICIENTS[0];
    let mut series_imaginary = 0.0;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        let denominator_real = shifted_real + index as f64;
        let denominator_norm = denominator_real.powi(2) + imaginary.powi(2);
        series_real += coefficient * denominator_real / denominator_norm;
        series_imaginary -= coefficient * imaginary / denominator_norm;
    }

    let t_real = shifted_real + 7.5;
    let log_t_norm = t_real.hypot(imaginary).ln();
    let t_argument = imaginary.atan2(t_real);
    0.5 * (2.0 * PI).ln() + (shifted_real + 0.5) * log_t_norm - imaginary * t_argument - t_real
        + series_real.hypot(series_imaginary).ln()
}

fn validate_positive_temperature(
    field: &'static str,
    raw_value_mev: f64,
) -> Result<(), BornWeakError> {
    if !raw_value_mev.is_finite() || raw_value_mev <= 0.0 {
        return Err(BornWeakError::InvalidTemperature {
            field,
            raw_value_mev,
        });
    }
    Ok(())
}

fn validate_total_rate(
    direction: &'static str,
    raw_value_per_second: f64,
) -> Result<(), BornWeakError> {
    if !raw_value_per_second.is_finite() {
        return Err(BornWeakError::NonFiniteRate {
            direction,
            raw_value_per_second,
        });
    }
    if raw_value_per_second < 0.0 {
        return Err(BornWeakError::NegativeRate {
            direction,
            raw_value_per_second,
        });
    }
    Ok(())
}

fn fermi_dirac(energy: f64, temperature: f64) -> f64 {
    let boltzmann = (-energy / temperature).exp();
    boltzmann / (1.0 + boltzmann)
}

/// Stable `1 / (exp(exponent) + 1)` for the signed-energy derivative kernels.
///
/// Unlike `fermi_dirac`, this helper must accept negative signed energies.
/// The branch form avoids `inf / inf` without clipping a physical argument.
fn fermi_dirac_from_exponent(exponent: f64) -> f64 {
    if exponent.is_nan() {
        return f64::NAN;
    }
    if exponent == f64::INFINITY {
        return 0.0;
    }
    if exponent == f64::NEG_INFINITY {
        return 1.0;
    }
    if exponent >= 0.0 {
        let inverse_boltzmann = (-exponent).exp();
        inverse_boltzmann / (1.0 + inverse_boltzmann)
    } else {
        1.0 / (1.0 + exponent.exp())
    }
}

fn checked_integral(process: BornWeakProcess, raw_value: f64) -> Result<f64, BornWeakError> {
    if !raw_value.is_finite() {
        return Err(BornWeakError::NonFiniteIntegral { process, raw_value });
    }
    if raw_value < 0.0 {
        return Err(BornWeakError::NegativeIntegral { process, raw_value });
    }
    Ok(raw_value)
}

fn integrate_semi_infinite_pair<F>(
    nodes: &[(f64, f64)],
    scale: f64,
    first_process: BornWeakProcess,
    second_process: BornWeakProcess,
    mut integrand: F,
) -> Result<(f64, f64), BornWeakError>
where
    F: FnMut(f64) -> (f64, f64),
{
    let mut first = CompensatedSum::new();
    let mut second = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let denominator = 1.0 - unit_node;
        let coordinate = scale * unit_node / denominator;
        let jacobian = scale / denominator.powi(2);
        let (first_value, second_value) = integrand(coordinate);
        first.add(unit_weight * jacobian * first_value);
        second.add(unit_weight * jacobian * second_value);
    }
    Ok((
        checked_integral(first_process, first.sum)?,
        checked_integral(second_process, second.sum)?,
    ))
}

/// Integrate a forward/reverse pair on direction-specific thermal scales.
///
/// This is required once signed finite-mass corrections are added: at extreme
/// bath-temperature ratios the physical rate can be the small difference of
/// a base and correction integral.  Sampling both directions on the hotter
/// bath's scale can then miss that cancellation without either integrand
/// being individually ill-defined.  Equal scales retain the shared-node path
/// and its exact discrete detailed-balance cancellation.
fn integrate_semi_infinite_pair_with_scales<F>(
    nodes: &[(f64, f64)],
    first_scale: f64,
    second_scale: f64,
    first_process: BornWeakProcess,
    second_process: BornWeakProcess,
    mut integrand: F,
) -> Result<(f64, f64), BornWeakError>
where
    F: FnMut(f64) -> (f64, f64),
{
    if first_scale.to_bits() == second_scale.to_bits() {
        return integrate_semi_infinite_pair(
            nodes,
            first_scale,
            first_process,
            second_process,
            integrand,
        );
    }

    let mut first = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let denominator = 1.0 - unit_node;
        let coordinate = first_scale * unit_node / denominator;
        let jacobian = first_scale / denominator.powi(2);
        let (first_value, _) = integrand(coordinate);
        first.add(unit_weight * jacobian * first_value);
    }

    let mut second = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let denominator = 1.0 - unit_node;
        let coordinate = second_scale * unit_node / denominator;
        let jacobian = second_scale / denominator.powi(2);
        let (_, second_value) = integrand(coordinate);
        second.add(unit_weight * jacobian * second_value);
    }

    Ok((
        checked_integral(first_process, first.sum)?,
        checked_integral(second_process, second.sum)?,
    ))
}

fn integrate_bounded_pair<F>(
    nodes: &[(f64, f64)],
    upper_bound: f64,
    first_process: BornWeakProcess,
    second_process: BornWeakProcess,
    mut integrand: F,
) -> Result<(f64, f64), BornWeakError>
where
    F: FnMut(f64) -> (f64, f64),
{
    let mut first = CompensatedSum::new();
    let mut second = CompensatedSum::new();
    for &(unit_node, unit_weight) in nodes {
        let coordinate = upper_bound * unit_node;
        let (first_value, second_value) = integrand(coordinate);
        first.add(unit_weight * upper_bound * first_value);
        second.add(unit_weight * upper_bound * second_value);
    }
    Ok((
        checked_integral(first_process, first.sum)?,
        checked_integral(second_process, second.sum)?,
    ))
}

fn gauss_legendre_unit_nodes(order: usize) -> Result<Vec<(f64, f64)>, BornWeakError> {
    if !(MIN_QUADRATURE_ORDER..=MAX_QUADRATURE_ORDER).contains(&order) {
        return Err(BornWeakError::InvalidQuadratureOrder {
            raw_order: order,
            minimum: MIN_QUADRATURE_ORDER,
            maximum: MAX_QUADRATURE_ORDER,
        });
    }

    let mut nodes = vec![(0.0, 0.0); order];
    let root_count = order.div_ceil(2);
    let order_f64 = order as f64;
    for root_index in 0..root_count {
        let phase = PI * (root_index as f64 + 0.75) / (order_f64 + 0.5);
        let mut root = phase.cos();
        let mut converged = false;
        for _ in 0..ROOT_ITERATION_LIMIT {
            let (polynomial, derivative) = legendre_polynomial_and_derivative(order, root);
            let update = polynomial / derivative;
            root -= update;
            if update.abs() <= 8.0 * f64::EPSILON * (1.0 + root.abs()) {
                converged = true;
                break;
            }
        }
        if !converged || !root.is_finite() {
            return Err(BornWeakError::QuadratureRootDidNotConverge {
                order,
                root_index,
                raw_root: root,
            });
        }

        let (_, derivative) = legendre_polynomial_and_derivative(order, root);
        let full_weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        let unit_weight = 0.5 * full_weight;
        nodes[root_index] = (0.5 * (1.0 - root), unit_weight);
        nodes[order - 1 - root_index] = (0.5 * (1.0 + root), unit_weight);
    }
    Ok(nodes)
}

fn legendre_polynomial_and_derivative(order: usize, coordinate: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = coordinate;
    if order == 1 {
        return (current, 1.0);
    }
    for degree in 2..=order {
        let degree_f64 = degree as f64;
        let next = ((2.0 * degree_f64 - 1.0) * coordinate * current
            - (degree_f64 - 1.0) * previous)
            / degree_f64;
        previous = current;
        current = next;
    }
    let derivative =
        order as f64 * (coordinate * current - previous) / (coordinate * coordinate - 1.0);
    (current, derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE_RELATIVE_TOLERANCE: f64 = 2.0e-11;
    // SciPy's absolute-error-controlled semi-infinite reference loses relative
    // resolution for the 1e-22 and 1e-60 reverse rates.  The separate
    // pointwise detailed-balance test is the stronger check in that regime.
    const TINY_REFERENCE_RELATIVE_TOLERANCE: f64 = 1.0e-7;

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs()
    }

    fn matches_scipy_anchor(actual: f64, expected: f64) -> bool {
        let tolerance = if expected.abs() < 1.0e-18 {
            TINY_REFERENCE_RELATIVE_TOLERANCE
        } else {
            REFERENCE_RELATIVE_TOLERANCE
        };
        relative_error(actual, expected) < tolerance
    }

    fn rates(t_gamma_mev: f64, t_nu_mev: f64, order: usize) -> BornWeakRates {
        evaluate_born_weak_rates(
            t_gamma_mev,
            t_nu_mev,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
        )
        .expect("physical Born weak input should evaluate")
    }

    fn ccr_rates(t_gamma_mev: f64, t_nu_mev: f64, order: usize) -> BornWeakRates {
        evaluate_weak_rates(
            t_gamma_mev,
            t_nu_mev,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
            WeakRateModel::PrimatZeroTemperatureCcr,
        )
        .expect("physical PRIMAT zero-temperature CCR input should evaluate")
    }

    fn f08b_rates(t_gamma_mev: f64, t_nu_mev: f64, order: usize) -> BornWeakRates {
        evaluate_weak_rates(
            t_gamma_mev,
            t_nu_mev,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
        )
        .expect("physical F08B finite-mass no-weak-magnetism input should evaluate")
    }

    fn f08c_rates(t_gamma_mev: f64, t_nu_mev: f64, order: usize) -> BornWeakRates {
        evaluate_weak_rates(
            t_gamma_mev,
            t_nu_mev,
            DEFAULT_NEUTRON_LIFETIME_SECONDS,
            order,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism,
        )
        .expect("physical F08C finite-mass weak-magnetism input should evaluate")
    }

    fn finite_mass_test_parameters(
        anomalous_weak_magnetism_coefficient: f64,
    ) -> FiniteMassParameters {
        FiniteMassParameters {
            proton_mass: PROTON_MASS_MEV / ELECTRON_MASS_MEV,
            neutron_mass: NEUTRON_MASS_MEV / ELECTRON_MASS_MEV,
            anomalous_weak_magnetism_coefficient,
        }
    }

    #[test]
    fn analytic_i0_matches_independent_bounded_momentum_integral() {
        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let endpoint = (q * q - 1.0).sqrt();
        let nodes = gauss_legendre_unit_nodes(16).expect("quadrature nodes");
        let mut numerical = CompensatedSum::new();
        for (node, weight) in nodes {
            let momentum = endpoint * node;
            let electron_energy = momentum.hypot(1.0);
            numerical.add(weight * endpoint * momentum.powi(2) * (q - electron_energy).powi(2));
        }
        let analytic = born_neutron_decay_normalization();
        assert_eq!(analytic.to_bits(), 1.636_097_956_008_729_4_f64.to_bits());
        assert!(relative_error(numerical.sum, analytic) < 2.0e-14);
    }

    #[test]
    fn generated_unit_nodes_integrate_legendre_moments() {
        let nodes = gauss_legendre_unit_nodes(32).expect("quadrature nodes");
        for degree in 0_i32..=20 {
            let actual: f64 = nodes
                .iter()
                .map(|(node, weight)| weight * node.powi(degree))
                .sum();
            let expected = 1.0 / (degree as f64 + 1.0);
            assert!((actual - expected).abs() < 2.0e-15, "degree={degree}");
        }
    }

    #[test]
    fn born_compatibility_wrapper_is_bitwise_identical_to_model_dispatch() {
        let wrapped = evaluate_born_weak_rates(1.0, 0.9, 878.4, 96).expect("wrapped Born");
        let dispatched =
            evaluate_weak_rates(1.0, 0.9, 878.4, 96, WeakRateModel::Born).expect("dispatched Born");
        let wrapped_values = [
            wrapped
                .channels
                .electron_neutrino_capture_on_neutron_per_second,
            wrapped.channels.electron_capture_on_proton_per_second,
            wrapped.channels.positron_capture_on_neutron_per_second,
            wrapped
                .channels
                .electron_antineutrino_capture_on_proton_per_second,
            wrapped.channels.free_neutron_decay_per_second,
            wrapped.channels.inverse_neutron_decay_per_second,
            wrapped.thermal_radiative_neutron_to_proton_per_second,
            wrapped.thermal_radiative_proton_to_neutron_per_second,
            wrapped.neutron_to_proton_per_second,
            wrapped.proton_to_neutron_per_second,
        ];
        let dispatched_values = [
            dispatched
                .channels
                .electron_neutrino_capture_on_neutron_per_second,
            dispatched.channels.electron_capture_on_proton_per_second,
            dispatched.channels.positron_capture_on_neutron_per_second,
            dispatched
                .channels
                .electron_antineutrino_capture_on_proton_per_second,
            dispatched.channels.free_neutron_decay_per_second,
            dispatched.channels.inverse_neutron_decay_per_second,
            dispatched.thermal_radiative_neutron_to_proton_per_second,
            dispatched.thermal_radiative_proton_to_neutron_per_second,
            dispatched.neutron_to_proton_per_second,
            dispatched.proton_to_neutron_per_second,
        ];
        for (wrapped_value, dispatched_value) in wrapped_values.into_iter().zip(dispatched_values) {
            assert_eq!(wrapped_value.to_bits(), dispatched_value.to_bits());
        }
    }

    #[test]
    fn ccr_control_path_bit_pattern_is_locked_at_the_f08b_boundary() {
        let rates = ccr_rates(1.0, 0.9, 96);
        let bits = [
            rates
                .channels
                .electron_neutrino_capture_on_neutron_per_second
                .to_bits(),
            rates
                .channels
                .electron_capture_on_proton_per_second
                .to_bits(),
            rates
                .channels
                .positron_capture_on_neutron_per_second
                .to_bits(),
            rates
                .channels
                .electron_antineutrino_capture_on_proton_per_second
                .to_bits(),
            rates.channels.free_neutron_decay_per_second.to_bits(),
            rates.channels.inverse_neutron_decay_per_second.to_bits(),
            rates.neutron_to_proton_per_second.to_bits(),
            rates.proton_to_neutron_per_second.to_bits(),
        ];
        assert_eq!(
            bits,
            [
                4_602_493_099_605_770_829,
                4_597_020_189_075_190_908,
                4_605_122_449_415_857_839,
                4_592_775_193_201_587_542,
                4_557_695_234_299_008_559,
                4_548_870_677_098_775_935,
                4_608_360_028_644_663_342,
                4_599_627_711_240_289_485,
            ]
        );
    }

    #[test]
    fn ccr_component_factors_match_live_primat_v032() {
        let fermi_references = [
            (0.001, 45.879_806_811_006_52),
            (0.01, 4.635_230_004_846_99),
            (0.1, 1.247_368_465_585_479),
            (0.5, 1.047_006_044_588_675),
            (0.9, 1.026_064_918_276_222_3),
            (0.999, 1.023_369_136_097_811_4),
        ];
        for (beta, expected) in fermi_references {
            let actual = fermi_coulomb_factor(beta);
            assert!(
                relative_error(actual, expected) < 3.0e-12,
                "F(beta={beta}) actual={actual:.17e} expected={expected:.17e}"
            );
        }

        let radiative_references = [
            (0.01, 0.1, 1.000_05, 1.042_467_709_996_628_4),
            (0.1, 0.2, 1.005_037_815_259_212, 1.042_425_124_654_481_1),
            (0.5, 1.0, 1.154_700_538_379_251_7, 1.041_980_197_999_113_5),
            (0.9, 3.0, 2.294_157_338_705_618, 1.040_329_673_731_661_4),
            (0.999, 0.01, 22.366_272_042_129_37, 0.916_405_249_537_196_7),
        ];
        for (beta, neutrino_energy, electron_energy, expected) in radiative_references {
            let actual =
                resummed_zero_temperature_radiative_factor(beta, neutrino_energy, electron_energy);
            assert!(
                relative_error(actual, expected) < 3.0e-13,
                "R(beta={beta}, y={neutrino_energy}, E={electron_energy}) actual={actual:.17e} expected={expected:.17e}"
            );
        }
    }

    #[test]
    fn ccr_threshold_products_use_the_analytic_physical_limits() {
        // High-precision evaluation of the PRIMAT v0.3.2 relativistic Fermi
        // function's beta -> 0 asymptotic coefficient.
        let threshold_product = 0.045_879_810_642_536_6;
        assert!(relative_error(threshold_beta_times_fermi_coulomb(), threshold_product) < 3.0e-13);
        for beta in [0.0, 1.0e-8, 0.5e-6] {
            assert!(
                relative_error(beta_times_fermi_coulomb_factor(beta), threshold_product) < 3.0e-13,
                "beta={beta} beta*F={:.17e}",
                beta_times_fermi_coulomb_factor(beta)
            );
        }
        assert!(
            relative_error(1.0e-8 * fermi_coulomb_factor(1.0e-8), threshold_product,) < 3.0e-13
        );
        let just_below = FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD * (1.0 - 1.0e-6);
        let just_above = FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD * (1.0 + 1.0e-6);
        let below_product = beta_times_fermi_coulomb_factor(just_below);
        let above_product = beta_times_fermi_coulomb_factor(just_above);
        assert!(relative_error(above_product, below_product) < 5.0e-12);
        assert!(
            relative_error(
                FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD
                    * fermi_coulomb_factor(FERMI_COULOMB_ASYMPTOTIC_BETA_THRESHOLD),
                0.045_879_810_642_608_21,
            ) < 5.0e-12
        );

        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let endpoint_momentum = (q * q - 1.0).sqrt();
        assert_eq!(corrected_neutrino_electron_phase_space(0.0, q), 0.0);
        assert_eq!(
            corrected_electron_momentum_phase_space(0.0, q - 1.0, 1.0),
            0.0
        );
        assert_eq!(
            corrected_electron_momentum_phase_space(endpoint_momentum, 0.0, q),
            0.0
        );
        assert_eq!(
            corrected_positron_momentum_phase_space(0.0, q + 1.0, 1.0),
            0.0
        );
        assert!(resummed_zero_temperature_radiative_factor(0.0, q - 1.0, 1.0).is_finite());
        assert_eq!(
            neutrino_energy_squared_times_radiative_factor(0.5, 0.0, 1.2),
            0.0
        );
    }

    #[test]
    fn corrected_lifetime_normalization_matches_live_primat_v032() {
        let nodes = gauss_legendre_unit_nodes(96).expect("quadrature nodes");
        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let actual = corrected_neutron_decay_normalization(&nodes, q).expect("CCR normalization");
        let primat_compute_fn = 1.758_384_386_757_194_2;
        assert!(relative_error(actual, primat_compute_fn) < 3.0e-10);
        assert!(actual > born_neutron_decay_normalization());
    }

    #[test]
    fn all_six_ccr_channels_match_independent_primat_scipy_anchors() {
        // Direct scipy.quad integration of PRIMAT v0.3.2 FermiCoulomb and
        // RadCorrResum, normalized by its ComputeFn with finite-mass and
        // finite-temperature corrections disabled.
        let channels = ccr_rates(1.0, 1.0, 96).channels;
        let actual = [
            channels.electron_neutrino_capture_on_neutron_per_second,
            channels.electron_capture_on_proton_per_second,
            channels.positron_capture_on_neutron_per_second,
            channels.electron_antineutrino_capture_on_proton_per_second,
            channels.free_neutron_decay_per_second,
            channels.inverse_neutron_decay_per_second,
        ];
        let references = [
            0.787_744_754_886_909_6,
            0.216_121_718_872_307_55,
            0.768_579_143_161_680_4,
            0.210_863_537_305_753_72,
            0.000_484_432_719_840_826_3,
            0.000_132_906_542_938_539_58,
        ];
        for (index, (value, reference)) in actual.into_iter().zip(references).enumerate() {
            assert!(
                relative_error(value, reference) < 3.0e-9,
                "CCR channel={index} actual={value:.17e} reference={reference:.17e}"
            );
        }
    }

    #[test]
    fn ccr_totals_match_independent_temperature_anchors() {
        let references = [
            (
                0.1,
                0.1,
                0.001_204_473_292_123_202_3,
                2.910_221_884_867_795_4e-9,
            ),
            (
                0.3,
                0.3,
                0.009_537_177_344_752_982,
                0.000_127_976_285_622_832_13,
            ),
            (1.0, 0.9, 1.261_481_912_723_214, 0.330_629_486_147_411),
            (1.0, 1.0, 1.556_808_330_768_430_8, 0.427_118_162_720_999_8),
            (10.0, 10.0, 92_817.162_924_876_06, 81_556.687_613_630_54),
        ];
        for (t_gamma, t_nu, neutron_to_proton, proton_to_neutron) in references {
            let actual = ccr_rates(t_gamma, t_nu, 96);
            assert!(
                relative_error(actual.neutron_to_proton_per_second, neutron_to_proton) < 5.0e-9,
                "T_gamma={t_gamma} T_nu={t_nu} n->p actual={:.17e}",
                actual.neutron_to_proton_per_second
            );
            assert!(
                relative_error(actual.proton_to_neutron_per_second, proton_to_neutron) < 5.0e-9,
                "T_gamma={t_gamma} T_nu={t_nu} p->n actual={:.17e}",
                actual.proton_to_neutron_per_second
            );
        }
    }

    #[test]
    fn ccr_preserves_equal_temperature_detailed_balance() {
        let temperature_mev = 0.7;
        let channels = ccr_rates(temperature_mev, temperature_mev, 96).channels;
        let equilibrium_factor = (-NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp();
        let pairs = [
            (
                channels.electron_neutrino_capture_on_neutron_per_second,
                channels.electron_capture_on_proton_per_second,
            ),
            (
                channels.positron_capture_on_neutron_per_second,
                channels.electron_antineutrino_capture_on_proton_per_second,
            ),
            (
                channels.free_neutron_decay_per_second,
                channels.inverse_neutron_decay_per_second,
            ),
        ];
        for (forward, reverse) in pairs {
            assert!(relative_error(reverse, equilibrium_factor * forward) < 2.0e-14);
        }
    }

    #[test]
    fn ccr_quadrature_converges_and_low_temperature_has_no_rate_floor() {
        let reference = 1.261_481_912_723_214;
        let order_32 = ccr_rates(1.0, 0.9, 32).neutron_to_proton_per_second;
        let order_64 = ccr_rates(1.0, 0.9, 64).neutron_to_proton_per_second;
        let order_96 = ccr_rates(1.0, 0.9, 96).neutron_to_proton_per_second;
        let errors = [
            (order_32 - reference).abs(),
            (order_64 - reference).abs(),
            (order_96 - reference).abs(),
        ];
        assert!(errors[1] < errors[0], "errors={errors:?}");
        assert!(errors[2] < errors[1], "errors={errors:?}");
        assert!(relative_error(order_96, reference) < 5.0e-9);

        let cold = ccr_rates(0.000_1, 0.000_1, 96);
        assert!(
            relative_error(
                cold.neutron_to_proton_per_second,
                1.0 / DEFAULT_NEUTRON_LIFETIME_SECONDS,
            ) < 1.0e-9
        );
        assert_eq!(cold.proton_to_neutron_per_second, 0.0);
    }

    #[test]
    fn ccr_invalid_inputs_and_factor_domains_remain_raw() {
        assert!(matches!(
            evaluate_weak_rates(0.0, 1.0, 878.4, 64, WeakRateModel::PrimatZeroTemperatureCcr),
            Err(BornWeakError::InvalidTemperature {
                field: "photon_temperature_mev",
                raw_value_mev: 0.0,
            })
        ));
        assert!(matches!(
            evaluate_weak_rates(1.0, 1.0, -2.0, 64, WeakRateModel::PrimatZeroTemperatureCcr),
            Err(BornWeakError::InvalidNeutronLifetime {
                raw_value_seconds: -2.0,
            })
        ));
        assert!(matches!(
            evaluate_weak_rates(1.0, 1.0, 878.4, 1, WeakRateModel::PrimatZeroTemperatureCcr),
            Err(BornWeakError::InvalidQuadratureOrder { raw_order: 1, .. })
        ));
        assert!(matches!(
            evaluate_weak_rates(
                1.0,
                1.0,
                f64::MAX,
                64,
                WeakRateModel::PrimatZeroTemperatureCcr,
            ),
            Err(BornWeakError::InvalidNormalization {
                raw_lifetime_seconds,
                raw_normalization,
            }) if raw_lifetime_seconds == f64::MAX && raw_normalization.is_infinite()
        ));

        assert!(fermi_coulomb_factor(0.0).is_nan());
        assert!(fermi_coulomb_factor(1.0).is_nan());
        assert!(resummed_zero_temperature_radiative_factor(0.5, 0.0, 1.2).is_nan());
        assert!(dilogarithm_unit_interval(-0.1).is_nan());
    }

    #[test]
    fn f08b_compact_chi_matches_independent_expanded_primat_points() {
        // The references expand PRIMAT v0.3.2's derivative operators before
        // numerical evaluation.  They are therefore independent of the
        // compact derivative polynomials used by the Rust implementation.
        let momentum: f64 = 0.5;
        let electron_energy = momentum.hypot(1.0);
        let temperature = 1.0 / ELECTRON_MASS_MEV;
        let parameters = finite_mass_test_parameters(F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        let references = [
            (1, 1.0, -0.003_650_209_289_080_508_4),
            (1, -1.0, -0.011_807_914_363_094_158),
            (-1, 1.0, -0.000_877_859_435_380_232),
            (-1, -1.0, -0.000_513_625_341_110_382_2),
        ];
        for (direction, energy_sign, expected) in references {
            let actual = finite_mass_chi(
                energy_sign * electron_energy,
                momentum,
                temperature,
                temperature,
                direction,
                parameters,
            );
            assert!(
                relative_error(actual, expected) < 3.0e-12,
                "direction={direction} energy_sign={energy_sign} actual={actual:.17e} expected={expected:.17e}"
            );
        }
    }

    #[test]
    fn f08b_coefficients_and_lifetime_normalization_lock_no_weak_magnetism() {
        assert_eq!(
            F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT.to_bits(),
            0.0_f64.to_bits()
        );
        let (f1, f2, f3) = finite_mass_couplings(1, F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        assert!(relative_error(f1, 0.880_453_153_952_389_8) < 3.0e-15);
        assert!(relative_error(f2, 0.012_914_358_251_301_864) < 3.0e-15);
        assert!(relative_error(f3, 0.106_632_487_796_308_1) < 3.0e-15);
        assert!(((f1 + f2 + f3) - 1.0).abs() < 5.0e-16);

        let nodes = gauss_legendre_unit_nodes(160).expect("quadrature nodes");
        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let proton_mass = PROTON_MASS_MEV / ELECTRON_MASS_MEV;
        let ccr = corrected_neutron_decay_normalization(&nodes, q).expect("CCR normalization");
        let finite_mass = finite_mass_neutron_decay_normalization_correction(
            &nodes,
            q,
            proton_mass,
            F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
        )
        .expect("signed F08B normalization correction");
        let total = ccr + finite_mass;
        assert!(finite_mass < 0.0);
        assert!(relative_error(ccr, 1.758_384_386_757_194_2) < 3.0e-10);
        assert!(relative_error(finite_mass, -0.003_620_163_815_600_761) < 3.0e-10);
        assert!(relative_error(total, 1.754_764_222_941_593_5) < 3.0e-10);
        // Stock physical weak magnetism is a parked F08C slice.  Its PRIMAT
        // normalization is a negative control, not an allowed F08B target.
        let physical_weak_magnetism_total = 1.754_751_046_224_002_4;
        assert!((total - physical_weak_magnetism_total).abs() > 1.0e-5);
    }

    #[test]
    fn f08b_all_six_channels_match_independent_unequal_temperature_anchor() {
        let channels = f08b_rates(1.0, 0.8, 160).channels;
        let actual = [
            channels.electron_neutrino_capture_on_neutron_per_second,
            channels.electron_capture_on_proton_per_second,
            channels.positron_capture_on_neutron_per_second,
            channels.electron_antineutrino_capture_on_proton_per_second,
            channels.free_neutron_decay_per_second,
            channels.inverse_neutron_decay_per_second,
        ];
        let references = [
            0.288_267_703_858_452_85,
            0.219_012_104_584_137_45,
            0.762_153_048_440_150_6,
            0.054_424_445_204_397_97,
            0.000_505_075_432_243_711_5,
            0.000_123_073_807_520_430_45,
        ];
        for (index, (value, reference)) in actual.into_iter().zip(references).enumerate() {
            assert!(
                relative_error(value, reference) < 5.0e-9,
                "F08B channel={index} actual={value:.17e} reference={reference:.17e}"
            );
        }
    }

    #[test]
    fn f08b_direction_scaled_quadrature_resolves_extreme_bath_ratios() {
        // The tiny-channel anchors use explicitly threshold-split adaptive
        // PRIMAT-formula integrals.  An unsplit interval spanning p~2.3..782
        // can silently skip this narrow cold-bath boundary layer.
        let anchors = [
            (
                10.0,
                0.01,
                40_003.587_322_158_9,
                40_094.367_956_691_03,
                2.758_718_400_076_742_2e-8,
                true,
            ),
            (
                0.01,
                10.0,
                42_416.963_263_316_32,
                32_632.813_184_859_537,
                2.142_901_130_326_829e-64,
                false,
            ),
        ];
        for (t_gamma, t_nu, expected_n_to_p, expected_p_to_n, tiny_rate, tiny_is_nu) in anchors {
            // Order 32 used to return a negative integral for the cold-bath
            // channel when both directions shared the hotter bath's scale.
            for order in [32, 64, 96] {
                let rates = f08b_rates(t_gamma, t_nu, order);
                let channels = rates.channels;
                for value in [
                    channels.electron_neutrino_capture_on_neutron_per_second,
                    channels.electron_capture_on_proton_per_second,
                    channels.positron_capture_on_neutron_per_second,
                    channels.electron_antineutrino_capture_on_proton_per_second,
                    channels.free_neutron_decay_per_second,
                    channels.inverse_neutron_decay_per_second,
                ] {
                    assert!(value >= 0.0 && value.is_finite());
                }
            }

            let refined = f08b_rates(t_gamma, t_nu, 320);
            assert!(relative_error(refined.neutron_to_proton_per_second, expected_n_to_p) < 5.0e-9);
            assert!(relative_error(refined.proton_to_neutron_per_second, expected_p_to_n) < 5.0e-9);
            let resolved_tiny_rate = if tiny_is_nu {
                refined
                    .channels
                    .electron_neutrino_capture_on_neutron_per_second
            } else {
                refined.channels.electron_capture_on_proton_per_second
            };
            assert!(
                relative_error(resolved_tiny_rate, tiny_rate) < 2.0e-6,
                "Tgamma={t_gamma} Tnu={t_nu} tiny={resolved_tiny_rate:.17e} expected={tiny_rate:.17e}"
            );
        }
    }

    #[test]
    fn f08b_totals_match_independent_adaptive_temperature_anchors() {
        let references = [
            (0.3, 0.009_514_094_338_920_922, 0.000_127_932_120_445_981_3),
            (1.0, 1.541_611_535_946_057_2, 0.423_839_523_709_042_3),
            (3.0, 253.821_350_899_704_24, 165.289_555_611_405_2),
        ];
        for (temperature, neutron_to_proton, proton_to_neutron) in references {
            let actual = f08b_rates(temperature, temperature, 160);
            assert!(
                relative_error(actual.neutron_to_proton_per_second, neutron_to_proton) < 5.0e-9,
                "T={temperature} n->p actual={:.17e}",
                actual.neutron_to_proton_per_second
            );
            assert!(
                relative_error(actual.proton_to_neutron_per_second, proton_to_neutron) < 5.0e-9,
                "T={temperature} p->n actual={:.17e}",
                actual.proton_to_neutron_per_second
            );
        }
    }

    #[test]
    fn f08b_exposes_first_order_modified_detailed_balance_residual() {
        let temperature_mev: f64 = 1.0;
        let actual = f08b_rates(temperature_mev, temperature_mev, 160);
        let target = (NEUTRON_MASS_MEV / PROTON_MASS_MEV).powf(1.5)
            * (-NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp();
        let total_ratio = actual.proton_to_neutron_per_second / actual.neutron_to_proton_per_second;
        let residual = relative_error(total_ratio, target);
        assert!(
            relative_error(target, 0.274_922_468_233_241_73) < 3.0e-14,
            "target={target:.17e}"
        );
        assert!(
            residual > 1.0e-6 && residual < 2.0e-4,
            "O(1/M_N) residual must be exposed rather than tuned away: target={target:.17e} ratio={total_ratio:.17e} residual={residual:.17e}"
        );
        let born_target = (-NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp();
        assert!(relative_error(total_ratio, target) < relative_error(total_ratio, born_target));
    }

    #[test]
    fn f08b_quadrature_converges_and_low_temperature_has_no_rate_floor() {
        let reference = 1.541_611_535_946_057_2;
        let order_32 = f08b_rates(1.0, 1.0, 32).neutron_to_proton_per_second;
        let order_64 = f08b_rates(1.0, 1.0, 64).neutron_to_proton_per_second;
        let order_96 = f08b_rates(1.0, 1.0, 96).neutron_to_proton_per_second;
        let errors = [
            (order_32 - reference).abs(),
            (order_64 - reference).abs(),
            (order_96 - reference).abs(),
        ];
        assert!(errors[1] < errors[0], "errors={errors:?}");
        assert!(errors[2] < errors[1], "errors={errors:?}");
        assert!(relative_error(order_96, reference) < 5.0e-9);

        let warm = f08b_rates(0.001, 0.001, 96);
        let cold = f08b_rates(0.000_1, 0.000_1, 96);
        let inverse_lifetime = 1.0 / DEFAULT_NEUTRON_LIFETIME_SECONDS;
        let warm_error = relative_error(warm.neutron_to_proton_per_second, inverse_lifetime);
        let cold_error = relative_error(cold.neutron_to_proton_per_second, inverse_lifetime);
        assert!(
            cold_error < warm_error / 5.0,
            "warm={warm_error} cold={cold_error}"
        );
        assert!(cold_error < 2.0e-7, "cold relative error={cold_error}");
        assert_eq!(cold.proton_to_neutron_per_second, 0.0);

        // At a deliberately over-extreme temperature the signed O(1/M_N)
        // correction can dominate a thermally underflowed capture integral.
        // The implementation must expose that breakdown, not clip or floor it.
        assert!(matches!(
            evaluate_weak_rates(
                1.0e-8,
                1.0e-8,
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                96,
                WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
            ),
            Err(BornWeakError::NegativeIntegral {
                process: BornWeakProcess::ElectronNeutrinoCaptureOnNeutron,
                raw_value,
            }) if raw_value < 0.0
        ));
    }

    #[test]
    fn f08b_invalid_inputs_and_internal_domains_remain_raw() {
        let model = WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism;
        assert!(matches!(
            evaluate_weak_rates(0.0, 1.0, 878.4, 64, model),
            Err(BornWeakError::InvalidTemperature {
                field: "photon_temperature_mev",
                raw_value_mev: 0.0,
            })
        ));
        assert!(matches!(
            evaluate_weak_rates(1.0, 1.0, -2.0, 64, model),
            Err(BornWeakError::InvalidNeutronLifetime {
                raw_value_seconds: -2.0,
            })
        ));
        assert!(matches!(
            evaluate_weak_rates(1.0, 1.0, 878.4, 1, model),
            Err(BornWeakError::InvalidQuadratureOrder { raw_order: 1, .. })
        ));
        let invalid_kernels = finite_mass_derivative_kernels(1.0, 0.0);
        assert!(invalid_kernels.g20.is_nan());
        assert!(finite_mass_couplings(0, 0.0).0.is_nan());
        assert!(finite_mass_couplings(1, f64::NAN).0.is_nan());
        assert!(
            finite_mass_chi(
                0.5,
                0.0,
                1.0,
                1.0,
                1,
                FiniteMassParameters {
                    proton_mass: 1.0,
                    neutron_mass: 1.0,
                    anomalous_weak_magnetism_coefficient: 0.0,
                },
            )
            .is_nan()
        );
    }

    #[test]
    fn f08c_physical_weak_magnetism_coefficients_and_normalization_match_primat() {
        let kappa_proton: f64 = 2.792_847_344_63 - 1.0;
        let kappa_neutron: f64 = -1.913_042_73;
        assert_eq!(
            PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT.to_bits(),
            (kappa_proton - kappa_neutron).to_bits()
        );

        let (forward_f1, forward_f2, forward_f3) =
            finite_mass_couplings(1, PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        let (reverse_f1, reverse_f2, reverse_f3) =
            finite_mass_couplings(-1, PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        assert!(relative_error(forward_f1, 2.487_954_860_124_953) < 3.0e-15);
        assert!(relative_error(forward_f2, -1.594_587_347_921_261_1) < 3.0e-15);
        assert!(relative_error(forward_f3, 0.106_632_487_796_308_1) < 3.0e-15);
        assert_eq!(forward_f1.to_bits(), reverse_f2.to_bits());
        assert_eq!(forward_f2.to_bits(), reverse_f1.to_bits());
        assert_eq!(forward_f3.to_bits(), reverse_f3.to_bits());

        let nodes = gauss_legendre_unit_nodes(160).expect("quadrature nodes");
        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let proton_mass = PROTON_MASS_MEV / ELECTRON_MASS_MEV;
        let ccr = corrected_neutron_decay_normalization(&nodes, q).expect("CCR normalization");
        let no_weak_magnetism = finite_mass_neutron_decay_normalization_correction(
            &nodes,
            q,
            proton_mass,
            F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
        )
        .expect("F08B normalization correction");
        let physical_weak_magnetism = finite_mass_neutron_decay_normalization_correction(
            &nodes,
            q,
            proton_mass,
            PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT,
        )
        .expect("F08C normalization correction");
        let f08c_total = ccr + physical_weak_magnetism;
        let isolated_weak_magnetism = physical_weak_magnetism - no_weak_magnetism;
        assert!(isolated_weak_magnetism < 0.0);
        assert!(relative_error(physical_weak_magnetism, -0.003_633_340_905_019_523) < 3.0e-10);
        assert!(relative_error(isolated_weak_magnetism, -1.317_708_941_876_2e-5) < 3.0e-10);
        assert!(relative_error(f08c_total, 1.754_751_046_224_002_4) < 3.0e-10);

        // Without Coulomb/radiative weighting, the anomalous-moment terms in
        // the integrated vacuum correction cancel.  The nonzero F08C shift
        // above is therefore a coupled CCR effect, not a free rescaling.
        let unweighted_finite_mass = |coefficient| {
            let endpoint_momentum = (q * q - 1.0).sqrt();
            let mut integral = CompensatedSum::new();
            for &(unit_node, unit_weight) in &nodes {
                let momentum = endpoint_momentum * unit_node;
                let energy = momentum.hypot(1.0);
                integral.add(
                    unit_weight
                        * endpoint_momentum
                        * momentum.powi(2)
                        * finite_mass_vacuum_chi(energy, momentum, q, proton_mass, coefficient),
                );
            }
            integral.sum
        };
        let unweighted_shift =
            unweighted_finite_mass(PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT)
                - unweighted_finite_mass(F08B_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        assert!(
            unweighted_shift.abs() < 2.0e-15,
            "unweighted weak-magnetism shift={unweighted_shift:.17e}"
        );
    }

    #[test]
    fn pre_f08d_models_have_no_hidden_thermal_directional_term() {
        for model in [
            WeakRateModel::Born,
            WeakRateModel::PrimatZeroTemperatureCcr,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism,
            WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism,
        ] {
            let rates = evaluate_weak_rates(1.0, 0.9, 878.4, 64, model)
                .expect("pre-F08D model remains evaluable");
            assert_eq!(
                rates
                    .thermal_radiative_neutron_to_proton_per_second
                    .to_bits(),
                0.0_f64.to_bits()
            );
            assert_eq!(
                rates
                    .thermal_radiative_proton_to_neutron_per_second
                    .to_bits(),
                0.0_f64.to_bits()
            );
        }
        assert!(matches!(
            validate_total_rate("probe", -f64::MIN_POSITIVE),
            Err(BornWeakError::NegativeRate {
                direction: "probe",
                ..
            })
        ));
    }

    #[test]
    fn f08c_compact_chi_matches_independent_expanded_primat_points() {
        let momentum: f64 = 0.5;
        let electron_energy = momentum.hypot(1.0);
        let temperature = 1.0 / ELECTRON_MASS_MEV;
        let parameters = finite_mass_test_parameters(PHYSICAL_ANOMALOUS_WEAK_MAGNETISM_COEFFICIENT);
        let references = [
            (1, 1.0, -0.004_544_341_548_612_97),
            (1, -1.0, -0.025_915_912_592_864_136),
            (-1, 1.0, -0.004_743_131_443_165_837),
            (-1, -1.0, -0.000_758_597_330_597_541_3),
        ];
        for (direction, energy_sign, expected) in references {
            let actual = finite_mass_chi(
                energy_sign * electron_energy,
                momentum,
                temperature,
                temperature,
                direction,
                parameters,
            );
            assert!(
                relative_error(actual, expected) < 3.0e-12,
                "direction={direction} energy_sign={energy_sign} actual={actual:.17e} expected={expected:.17e}"
            );
        }
    }

    #[test]
    fn f08c_all_six_channels_and_equal_temperature_totals_match_primat_anchors() {
        let channels = f08c_rates(1.0, 0.8, 160).channels;
        let actual = [
            channels.electron_neutrino_capture_on_neutron_per_second,
            channels.electron_capture_on_proton_per_second,
            channels.positron_capture_on_neutron_per_second,
            channels.electron_antineutrino_capture_on_proton_per_second,
            channels.free_neutron_decay_per_second,
            channels.inverse_neutron_decay_per_second,
        ];
        let references = [
            0.292_489_539_069_280_66,
            0.222_935_031_328_633_07,
            0.748_250_761_627_631,
            0.053_618_882_579_454_14,
            0.000_505_067_314_247_246_2,
            0.000_123_075_198_235_504_23,
        ];
        for (index, (value, reference)) in actual.into_iter().zip(references).enumerate() {
            assert!(
                relative_error(value, reference) < 5.0e-9,
                "F08C channel={index} actual={value:.17e} reference={reference:.17e}"
            );
        }

        let total_references = [
            (0.3, 0.009_519_719_396_921_074, 0.000_128_007_500_685_520_63),
            (1.0, 1.541_934_485_109_423_4, 0.423_928_015_526_180_04),
            (3.0, 253.957_985_186_276_66, 165.378_220_977_871_6),
        ];
        for (temperature, neutron_to_proton, proton_to_neutron) in total_references {
            let rates = f08c_rates(temperature, temperature, 160);
            assert!(
                relative_error(rates.neutron_to_proton_per_second, neutron_to_proton) < 5.0e-9,
                "T={temperature} n->p actual={:.17e}",
                rates.neutron_to_proton_per_second
            );
            assert!(
                relative_error(rates.proton_to_neutron_per_second, proton_to_neutron) < 5.0e-9,
                "T={temperature} p->n actual={:.17e}",
                rates.proton_to_neutron_per_second
            );
        }
    }

    #[test]
    fn f08c_exposes_the_physical_first_order_detailed_balance_residual() {
        let temperature_mev: f64 = 1.0;
        let rates = f08c_rates(temperature_mev, temperature_mev, 160);
        let target = (NEUTRON_MASS_MEV / PROTON_MASS_MEV).powf(1.5)
            * (-NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp();
        let ratio = rates.proton_to_neutron_per_second / rates.neutron_to_proton_per_second;
        let residual = relative_error(ratio, target);
        assert!(relative_error(ratio, 0.274_932_573_089_248_95) < 5.0e-9);
        assert!(
            residual > 1.0e-6 && residual < 2.0e-4,
            "F08C detailed-balance target={target:.17e} ratio={ratio:.17e} residual={residual:.17e}"
        );
    }

    #[test]
    fn f08c_quadrature_converges_and_raw_low_temperature_failure_is_exposed() {
        let reference = 1.541_934_485_109_423_4;
        let order_32 = f08c_rates(1.0, 1.0, 32).neutron_to_proton_per_second;
        let order_64 = f08c_rates(1.0, 1.0, 64).neutron_to_proton_per_second;
        let order_96 = f08c_rates(1.0, 1.0, 96).neutron_to_proton_per_second;
        let errors = [
            (order_32 - reference).abs(),
            (order_64 - reference).abs(),
            (order_96 - reference).abs(),
        ];
        assert!(errors[1] < errors[0], "errors={errors:?}");
        assert!(errors[2] < errors[1], "errors={errors:?}");
        assert!(relative_error(order_96, reference) < 5.0e-9);

        let warm = f08c_rates(0.001, 0.001, 96);
        let cold = f08c_rates(0.000_1, 0.000_1, 96);
        let inverse_lifetime = 1.0 / DEFAULT_NEUTRON_LIFETIME_SECONDS;
        let warm_error = relative_error(warm.neutron_to_proton_per_second, inverse_lifetime);
        let cold_error = relative_error(cold.neutron_to_proton_per_second, inverse_lifetime);
        assert!(cold_error < warm_error / 5.0);
        assert!(cold_error < 2.0e-7);
        assert_eq!(cold.proton_to_neutron_per_second, 0.0);

        let model = WeakRateModel::PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism;
        assert!(matches!(
            evaluate_weak_rates(0.0, 1.0, 878.4, 64, model),
            Err(BornWeakError::InvalidTemperature {
                field: "photon_temperature_mev",
                raw_value_mev: 0.0,
            })
        ));
        assert!(matches!(
            evaluate_weak_rates(
                1.0e-8,
                1.0e-8,
                DEFAULT_NEUTRON_LIFETIME_SECONDS,
                96,
                model,
            ),
            Err(BornWeakError::NegativeIntegral {
                process: BornWeakProcess::ElectronNeutrinoCaptureOnNeutron,
                raw_value,
            }) if raw_value < 0.0
        ));
    }

    #[test]
    fn all_six_physical_channels_match_independent_scipy_anchors() {
        let channels = rates(1.0, 1.0, 96).channels;
        let actual = [
            channels.electron_neutrino_capture_on_neutron_per_second,
            channels.electron_capture_on_proton_per_second,
            channels.positron_capture_on_neutron_per_second,
            channels.electron_antineutrino_capture_on_proton_per_second,
            channels.free_neutron_decay_per_second,
            channels.inverse_neutron_decay_per_second,
        ];
        let scipy_quad = [
            0.805_577_740_767_347_8,
            0.221_014_287_863_976_7,
            0.797_465_233_303_688_5,
            0.218_788_580_810_348_33,
            0.000_484_377_436_379_039_27,
            0.000_132_891_375_643_917_55,
        ];
        for (index, (value, reference)) in actual.into_iter().zip(scipy_quad).enumerate() {
            assert!(
                value > 0.0 && matches_scipy_anchor(value, reference),
                "channel={index} actual={value:.17e} reference={reference:.17e}"
            );
        }
    }

    #[test]
    fn total_rates_match_independent_scipy_temperature_grid() {
        let anchors = [
            (0.01, 1.138_439_407_334_316_2e-3, 7.719_678_939_288_619e-60),
            (0.03, 1.138_910_808_318_371e-3, 2.155_689_178_417_225e-22),
            (0.1, 1.206_164_662_897_369_4e-3, 2.914_308_537_759_024e-9),
            (0.3, 9.751_348_354_688_931e-3, 1.308_501_768_538_54e-4),
            (1.0, 1.603_527_351_507_415, 0.439_935_760_049_968_9),
            (3.0, 271.276_877_179_332, 176.272_166_234_618_5),
            (10.0, 96_771.013_183_167_38, 85_030.861_141_726_32),
        ];
        for (temperature, neutron_to_proton, proton_to_neutron) in anchors {
            let actual = rates(temperature, temperature, 96);
            assert!(
                matches_scipy_anchor(actual.neutron_to_proton_per_second, neutron_to_proton),
                "T={temperature} n->p actual={:.17e}",
                actual.neutron_to_proton_per_second
            );
            assert!(
                matches_scipy_anchor(actual.proton_to_neutron_per_second, proton_to_neutron),
                "T={temperature} p->n actual={:.17e}",
                actual.proton_to_neutron_per_second
            );
        }
    }

    #[test]
    fn unequal_temperatures_match_independent_scipy_anchor() {
        let actual = rates(1.0, 0.9, 96);
        assert!(
            relative_error(actual.neutron_to_proton_per_second, 1.301_409_078_884_550_5)
                < REFERENCE_RELATIVE_TOLERANCE
        );
        assert!(
            relative_error(actual.proton_to_neutron_per_second, 0.339_709_231_766_509_3)
                < REFERENCE_RELATIVE_TOLERANCE
        );
    }

    #[test]
    fn totals_match_live_primat_born_instantaneous_decoupling_grid() {
        // PRIMAT v0.3.2 public runtime weak-rate API, sampled at 640 points
        // per temperature decade with every higher-order weak correction,
        // QED, spectral distortion, and incomplete decoupling disabled.
        let references = [
            (10.0, 9.675_665_305_739_222e4, 8.501_791_273_070_23e4),
            (1.0, 1.582_285_335_901_359_3, 4.324_978_115_832_039e-1),
            (
                0.861_647_152_867_38,
                8.086_295_194_961_61e-1,
                1.792_226_124_815_928_5e-1,
            ),
            (0.3, 8.723_352_040_685_488e-3, 1.085_739_855_147_212_4e-4),
            (0.1, 1.167_152_587_931_267e-3, 9.105_828_441_005_356e-10),
        ];
        let background =
            crate::flrw::IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling();
        for (temperature, neutron_to_proton, proton_to_neutron) in references {
            let t_nu = background.thermo_state(temperature).unwrap().t_nu_mev;
            let actual = rates(temperature, t_nu, DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
            println!(
                "T={temperature}: n2p={:.16e} ({:.3e}), p2n={:.16e} ({:.3e})",
                actual.neutron_to_proton_per_second,
                actual.neutron_to_proton_per_second / neutron_to_proton - 1.0,
                actual.proton_to_neutron_per_second,
                actual.proton_to_neutron_per_second / proton_to_neutron - 1.0,
            );
            assert!(
                relative_error(actual.neutron_to_proton_per_second, neutron_to_proton) < 2.0e-6
            );
            assert!(
                relative_error(actual.proton_to_neutron_per_second, proton_to_neutron) < 2.0e-6
            );
        }
    }

    #[test]
    fn pairwise_detailed_balance_holds_only_at_equal_temperature() {
        let temperature_mev = 0.7;
        let equal = rates(temperature_mev, temperature_mev, 96).channels;
        let equilibrium_factor = (-NEUTRON_PROTON_MASS_DIFFERENCE_MEV / temperature_mev).exp();
        let pairs = [
            (
                equal.electron_neutrino_capture_on_neutron_per_second,
                equal.electron_capture_on_proton_per_second,
            ),
            (
                equal.positron_capture_on_neutron_per_second,
                equal.electron_antineutrino_capture_on_proton_per_second,
            ),
            (
                equal.free_neutron_decay_per_second,
                equal.inverse_neutron_decay_per_second,
            ),
        ];
        for (forward, reverse) in pairs {
            assert!(relative_error(reverse, equilibrium_factor * forward) < 2.0e-14);
        }

        let unequal = rates(0.7, 0.5, 96).channels;
        let unequal_ratio = unequal.electron_capture_on_proton_per_second
            / unequal.electron_neutrino_capture_on_neutron_per_second;
        assert!((unequal_ratio / equilibrium_factor - 1.0).abs() > 0.05);
    }

    #[test]
    fn quadrature_converges_across_32_64_96_nodes() {
        let reference = 1.301_409_078_884_550_5;
        let order_32 = rates(1.0, 0.9, 32).neutron_to_proton_per_second;
        let order_64 = rates(1.0, 0.9, 64).neutron_to_proton_per_second;
        let order_96 = rates(1.0, 0.9, 96).neutron_to_proton_per_second;
        let errors = [
            (order_32 - reference).abs(),
            (order_64 - reference).abs(),
            (order_96 - reference).abs(),
        ];
        assert!(errors[1] < errors[0], "errors={errors:?}");
        assert!(errors[2] < errors[1], "errors={errors:?}");
        assert!(relative_error(order_96, reference) < REFERENCE_RELATIVE_TOLERANCE);
    }

    #[test]
    fn default_order_tracks_96_for_both_totals_over_full_and_extreme_grid() {
        let temperature_pairs = [
            (0.01, 0.01),
            (0.03, 0.03),
            (0.1, 0.1),
            (0.3, 0.3),
            (1.0, 1.0),
            (3.0, 3.0),
            (10.0, 10.0),
            (1.0, 0.9),
            (0.01, 10.0),
            (10.0, 0.01),
        ];
        for (t_gamma, t_nu) in temperature_pairs {
            let default = rates(t_gamma, t_nu, DEFAULT_BORN_WEAK_QUADRATURE_ORDER);
            let refined = rates(t_gamma, t_nu, 96);
            assert!(
                relative_error(
                    default.neutron_to_proton_per_second,
                    refined.neutron_to_proton_per_second,
                ) < 3.0e-11,
                "T_gamma={t_gamma} T_nu={t_nu} n->p"
            );
            assert!(
                relative_error(
                    default.proton_to_neutron_per_second,
                    refined.proton_to_neutron_per_second,
                ) < 3.0e-11,
                "T_gamma={t_gamma} T_nu={t_nu} p->n"
            );
        }
    }

    #[test]
    fn bounded_decay_phase_space_vanishes_continuously_at_both_thresholds() {
        let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
        let endpoint = (q * q - 1.0).sqrt();
        let phase_space = |momentum: f64| {
            let neutrino_energy = q - momentum.hypot(1.0);
            momentum.powi(2) * neutrino_energy.powi(2)
        };
        let interior = phase_space(0.5 * endpoint);
        assert!(interior > 0.0);
        assert!(phase_space(0.0) < 1.0e-28);
        assert!(phase_space(endpoint) < 1.0e-28);
        assert!(phase_space(1.0e-8 * endpoint) < 1.0e-12 * interior);
        assert!(phase_space((1.0 - 1.0e-8) * endpoint) < 1.0e-12 * interior);
    }

    #[test]
    fn low_temperature_limit_is_free_neutron_decay() {
        let actual = rates(0.0001, 0.0001, 96);
        let free_decay = 1.0 / DEFAULT_NEUTRON_LIFETIME_SECONDS;
        assert!(relative_error(actual.neutron_to_proton_per_second, free_decay) < 2.0e-10);
        assert!(actual.proton_to_neutron_per_second < 1.0e-100);
    }

    #[test]
    fn high_temperature_rate_approaches_fifth_power_scaling() {
        let low = rates(10.0, 10.0, 96);
        let high = rates(20.0, 20.0, 96);
        let observed_power = (high.neutron_to_proton_per_second / low.neutron_to_proton_per_second)
            .ln()
            / 2.0_f64.ln();
        assert!(
            (observed_power - 5.0).abs() < 0.08,
            "power={observed_power}"
        );
        let equilibrium_neutron_fraction = high.proton_to_neutron_per_second
            / (high.neutron_to_proton_per_second + high.proton_to_neutron_per_second);
        assert!((equilibrium_neutron_fraction - 0.5).abs() < 0.02);
    }

    #[test]
    fn every_channel_scales_exactly_with_inverse_neutron_lifetime() {
        let base = evaluate_born_weak_rates(1.0, 0.9, 800.0, 96).expect("base rates");
        let doubled = evaluate_born_weak_rates(1.0, 0.9, 1600.0, 96).expect("scaled rates");
        let base_channels = [
            base.channels
                .electron_neutrino_capture_on_neutron_per_second,
            base.channels.electron_capture_on_proton_per_second,
            base.channels.positron_capture_on_neutron_per_second,
            base.channels
                .electron_antineutrino_capture_on_proton_per_second,
            base.channels.free_neutron_decay_per_second,
            base.channels.inverse_neutron_decay_per_second,
            base.neutron_to_proton_per_second,
            base.proton_to_neutron_per_second,
        ];
        let doubled_channels = [
            doubled
                .channels
                .electron_neutrino_capture_on_neutron_per_second,
            doubled.channels.electron_capture_on_proton_per_second,
            doubled.channels.positron_capture_on_neutron_per_second,
            doubled
                .channels
                .electron_antineutrino_capture_on_proton_per_second,
            doubled.channels.free_neutron_decay_per_second,
            doubled.channels.inverse_neutron_decay_per_second,
            doubled.neutron_to_proton_per_second,
            doubled.proton_to_neutron_per_second,
        ];
        for (first, second) in base_channels.into_iter().zip(doubled_channels) {
            assert_eq!(first, 2.0 * second);
        }
    }

    #[test]
    fn invalid_inputs_return_typed_raw_errors() {
        for raw_temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                evaluate_born_weak_rates(raw_temperature, 1.0, 878.4, 64),
                Err(BornWeakError::InvalidTemperature {
                    field: "photon_temperature_mev",
                    raw_value_mev,
                }) if raw_value_mev.to_bits() == raw_temperature.to_bits()
            ));
            assert!(matches!(
                evaluate_born_weak_rates(1.0, raw_temperature, 878.4, 64),
                Err(BornWeakError::InvalidTemperature {
                    field: "neutrino_temperature_mev",
                    raw_value_mev,
                }) if raw_value_mev.to_bits() == raw_temperature.to_bits()
            ));
        }
        for raw_lifetime in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                evaluate_born_weak_rates(1.0, 1.0, raw_lifetime, 64),
                Err(BornWeakError::InvalidNeutronLifetime { raw_value_seconds })
                    if raw_value_seconds.to_bits() == raw_lifetime.to_bits()
            ));
        }
        for raw_order in [0, 1, MAX_QUADRATURE_ORDER + 1] {
            assert!(matches!(
                evaluate_born_weak_rates(1.0, 1.0, 878.4, raw_order),
                Err(BornWeakError::InvalidQuadratureOrder { raw_order: value, .. })
                    if value == raw_order
            ));
        }
        assert!(matches!(
            evaluate_born_weak_rates(1.0, 1.0, f64::MAX, 64),
            Err(BornWeakError::InvalidNormalization {
                raw_lifetime_seconds,
                raw_normalization,
            }) if raw_lifetime_seconds == f64::MAX && raw_normalization.is_infinite()
        ));
    }

    #[test]
    fn f08c_physical_rates_remain_raw_and_positive_at_extreme_bath_ratios() {
        for (t_gamma, t_nu) in [(10.0, 0.01), (0.01, 10.0)] {
            for order in [32, 64, 96] {
                let rates = f08c_rates(t_gamma, t_nu, order);
                for value in [
                    rates
                        .channels
                        .electron_neutrino_capture_on_neutron_per_second,
                    rates.channels.electron_capture_on_proton_per_second,
                    rates.channels.positron_capture_on_neutron_per_second,
                    rates
                        .channels
                        .electron_antineutrino_capture_on_proton_per_second,
                    rates.channels.free_neutron_decay_per_second,
                    rates.channels.inverse_neutron_decay_per_second,
                ] {
                    assert!(
                        value >= 0.0 && value.is_finite(),
                        "Tgamma={t_gamma} Tnu={t_nu} order={order} raw channel={value:.17e}"
                    );
                }
            }

            let order_96 = f08c_rates(t_gamma, t_nu, 96);
            let order_320 = f08c_rates(t_gamma, t_nu, 320);
            assert!(
                relative_error(
                    order_96.neutron_to_proton_per_second,
                    order_320.neutron_to_proton_per_second,
                ) < 2.0e-8
            );
            assert!(
                relative_error(
                    order_96.proton_to_neutron_per_second,
                    order_320.proton_to_neutron_per_second,
                ) < 2.0e-8
            );
        }
    }
}
