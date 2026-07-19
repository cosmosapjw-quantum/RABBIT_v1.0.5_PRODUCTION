//! Tree-level, spatially flat FLRW thermodynamics for the minimal BBN core.
//!
//! Conventions: `c = hbar = k_B = 1`; masses and temperatures are in MeV,
//! energy densities in MeV^4, Newton's constant in MeV^-2, and `H` in MeV.
//! The Friedmann convention is `H^2 = 8 pi G_N rho / 3`, equivalently the
//! unreduced Planck mass `M_Pl = G_N^(-1/2)`.
//! The ideal one-temperature system contains photons, equilibrium e-/e+ at
//! zero chemical potential, and three massless zero-chemical-potential
//! neutrino flavours.  Its explicit switch optionally adds the scalar PRIMAT
//! finite-temperature QED pressure; shear, collisions, baryon rest energy,
//! and nuclear backreaction remain absent from that ideal system.  The
//! separate three-temperature substrate below adds only the bounded thermal
//! electron-collision energy-transfer closure documented on its type.

#![cfg_attr(not(test), allow(dead_code))]

use std::f64::consts::PI;

use crate::electron_thermal_fd::finite_mass_fd_electron_energy_transfer;
use crate::ode::OdeSystem;
use crate::qed_eos::{
    FiniteTemperatureQed, QedEosError, high_temperature_em_entropy_coefficient, qed_correction,
};

pub(crate) const ELECTRON_MASS_MEV: f64 = 0.510_998_950_0;
pub(crate) const NEWTON_G_MEV_MINUS_2: f64 = 6.708_830_746_231_458e-45;
pub(crate) const MEV_TO_INVERSE_SECONDS: f64 = 1.519_267_447_878_626e21;

const DEFAULT_SIMPSON_PANELS: usize = 256;
const DEFAULT_TAIL_E_FOLDS: f64 = 48.0;
const RHO_GAMMA_COEFFICIENT: f64 = PI * PI / 15.0;
const RHO_NEUTRINO_PAIR_COEFFICIENT: f64 = 7.0 * PI * PI / 120.0;
const RHO_THREE_NU_COEFFICIENT: f64 = 7.0 * PI * PI / 40.0;
const IDEAL_HIGH_T_EM_ENTROPY_COEFFICIENT: f64 = 11.0 * PI * PI / 45.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntropyOwner {
    /// Before instantaneous decoupling, the common EM+neutrino bath cools.
    CommonElectromagneticNeutrinoBath,
    /// Afterwards, EM entropy heats photons while neutrinos redshift alone.
    ElectromagneticBathAfterDecoupling,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum NeutrinoTemperatureAnchor {
    FiniteDecouplingTemperature(f64),
    IdealHighTemperatureEntropy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FlrwThermoError {
    InvalidInput { quantity: &'static str, value: f64 },
    InvalidPanelCount(usize),
    NonFiniteOutput { quantity: &'static str, value: f64 },
    NonPositiveOutput { quantity: &'static str, value: f64 },
    ElectronThermal(&'static str),
    Qed(QedEosError),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ElectromagneticEos {
    pub(crate) rho: f64,
    pub(crate) pressure: f64,
    pub(crate) entropy: f64,
    pub(crate) drho_dt: f64,
    pub(crate) ds_dt: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FlrwThermoState {
    pub(crate) t_gamma_mev: f64,
    pub(crate) t_nu_mev: f64,
    pub(crate) entropy_owner: EntropyOwner,
    pub(crate) electromagnetic: ElectromagneticEos,
    pub(crate) rho_neutrino: f64,
    pub(crate) rho_total: f64,
    pub(crate) pressure_total: f64,
    pub(crate) d_tgamma_d_lna: f64,
    pub(crate) h_mev: f64,
    pub(crate) h_inverse_seconds: f64,
}

/// One fixed-entropy-owner leg of the one-state ideal-FLRW system, with
/// `ln(a/a_initial)` as the independent variable and photon temperature as
/// the state.
///
/// The common-bath leg must stop at the instantaneous-decoupling event and the
/// integration must restart there with the electromagnetic-bath leg.  This
/// keeps solver stages and Jacobians on one smooth branch. Thermodynamics
/// errors are returned as non-finite derivatives so the solver's raw-failure
/// contract remains authoritative.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdealFlrwSystem {
    neutrino_temperature_anchor: NeutrinoTemperatureAnchor,
    entropy_owner: EntropyOwner,
    qed_model: FiniteTemperatureQed,
}

impl IdealFlrwSystem {
    pub(crate) fn common_bath_leg(t_dec_mev: f64) -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(
                t_dec_mev,
            ),
            entropy_owner: EntropyOwner::CommonElectromagneticNeutrinoBath,
            qed_model: FiniteTemperatureQed::Off,
        }
    }

    pub(crate) fn electromagnetic_bath_leg(t_dec_mev: f64) -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(
                t_dec_mev,
            ),
            entropy_owner: EntropyOwner::ElectromagneticBathAfterDecoupling,
            qed_model: FiniteTemperatureQed::Off,
        }
    }

    /// Ideal instantaneous-decoupling convention used by PRIMAT with
    /// incomplete decoupling and QED disabled.  The neutrino temperature is
    /// normalised to `T_nu/T_gamma -> 1` in the relativistic-electron limit,
    /// rather than at an arbitrary finite handoff temperature.
    pub(crate) fn ideal_high_temperature_instantaneous_decoupling() -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::IdealHighTemperatureEntropy,
            entropy_owner: EntropyOwner::ElectromagneticBathAfterDecoupling,
            qed_model: FiniteTemperatureQed::Off,
        }
    }

    pub(crate) fn common_bath_leg_with_qed(
        t_dec_mev: f64,
        qed_model: FiniteTemperatureQed,
    ) -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(
                t_dec_mev,
            ),
            entropy_owner: EntropyOwner::CommonElectromagneticNeutrinoBath,
            qed_model,
        }
    }

    pub(crate) fn electromagnetic_bath_leg_with_qed(
        t_dec_mev: f64,
        qed_model: FiniteTemperatureQed,
    ) -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(
                t_dec_mev,
            ),
            entropy_owner: EntropyOwner::ElectromagneticBathAfterDecoupling,
            qed_model,
        }
    }

    pub(crate) fn high_temperature_instantaneous_decoupling_with_qed(
        qed_model: FiniteTemperatureQed,
    ) -> Self {
        Self {
            neutrino_temperature_anchor: NeutrinoTemperatureAnchor::IdealHighTemperatureEntropy,
            entropy_owner: EntropyOwner::ElectromagneticBathAfterDecoupling,
            qed_model,
        }
    }

    pub(crate) fn thermo_state(
        &self,
        t_gamma_mev: f64,
    ) -> Result<FlrwThermoState, FlrwThermoError> {
        flrw_thermo_state_for_owner(
            t_gamma_mev,
            self.neutrino_temperature_anchor,
            self.entropy_owner,
            self.qed_model,
        )
    }
}

impl OdeSystem for IdealFlrwSystem {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, _ln_a: f64, y: &[f64], out: &mut [f64]) {
        out[0] = self
            .thermo_state(y[0])
            .map(|state| state.d_tgamma_d_lna)
            .unwrap_or(f64::NAN);
    }

    fn jacobian(&self, ln_a: f64, y: &[f64], out: &mut [f64]) {
        let relative_step = 1.0e-5;
        let step = (relative_step * y[0].abs()).max(1.0e-10);
        let mut plus = [0.0];
        let mut minus = [0.0];
        self.rhs(ln_a, &[y[0] + step], &mut plus);
        self.rhs(ln_a, &[y[0] - step], &mut minus);
        out[0] = (plus[0] - minus[0]) / (2.0 * step);
    }

    fn dfdt(&self, _ln_a: f64, _y: &[f64], out: &mut [f64]) {
        out[0] = 0.0;
    }
}

/// Mixed-approximation FLRW substrate for the first endpoint-consumed
/// electron-collision slice.
///
/// The electromagnetic sector uses the finite-electron-mass EOS and the
/// selected scalar QED pressure, while each neutrino/antineutrino pair is a
/// massless zero-chemical-potential Fermi--Dirac fluid.  Energy exchange is
/// the finite-vacuum-electron-mass, FD/Pauli tree-level CM collision moment.
/// Consequently this is still a thermal isotropic closure: it is not a
/// spectral Boltzmann evolution, does not include collision thermal masses or
/// radiative collision corrections, and is neither QKE nor a Standard-Model
/// precision-decoupling claim.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ThreeTemperatureFlrwSystem {
    qed_model: FiniteTemperatureQed,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ThreeTemperatureFlrwState {
    pub(crate) t_gamma_mev: f64,
    pub(crate) t_nue_mev: f64,
    pub(crate) t_nux_mev: f64,
    pub(crate) electromagnetic: ElectromagneticEos,
    /// Energy density of one neutrino plus antineutrino pair.
    pub(crate) rho_nue_pair: f64,
    /// Energy density of one heavy-flavour neutrino plus antineutrino pair.
    pub(crate) rho_nux_pair: f64,
    pub(crate) rho_total: f64,
    pub(crate) pressure_total: f64,
    /// Energy gained by the electron-flavour neutrino pair from the EM bath.
    pub(crate) q_nue_pair_mev5: f64,
    /// Energy gained by one heavy-flavour neutrino pair from the EM bath.
    pub(crate) q_nux_pair_mev5: f64,
    /// Total neutrino gain, including two degenerate heavy pairs.
    pub(crate) q_total_mev5: f64,
    pub(crate) d_tgamma_d_lna: f64,
    pub(crate) d_tnue_d_lna: f64,
    pub(crate) d_tnux_d_lna: f64,
    pub(crate) h_mev: f64,
    pub(crate) h_inverse_seconds: f64,
}

impl ThreeTemperatureFlrwSystem {
    pub(crate) const fn new(qed_model: FiniteTemperatureQed) -> Self {
        Self { qed_model }
    }

    pub(crate) fn thermo_state(
        &self,
        temperatures_mev: [f64; 3],
    ) -> Result<ThreeTemperatureFlrwState, FlrwThermoError> {
        three_temperature_flrw_state(
            temperatures_mev[0],
            temperatures_mev[1],
            temperatures_mev[2],
            self.qed_model,
        )
    }
}

/// Evaluate the three-temperature FLRW state without solver-owned state or
/// hidden calibration.  Positive finite temperatures are required and every
/// non-finite collision or thermodynamic result is returned raw as an error.
pub(crate) fn three_temperature_flrw_state(
    t_gamma_mev: f64,
    t_nue_mev: f64,
    t_nux_mev: f64,
    qed_model: FiniteTemperatureQed,
) -> Result<ThreeTemperatureFlrwState, FlrwThermoError> {
    let t_gamma_mev = require_positive_finite("T_gamma", t_gamma_mev)?;
    let t_nue_mev = require_positive_finite("T_nue", t_nue_mev)?;
    let t_nux_mev = require_positive_finite("T_nux", t_nux_mev)?;
    let electromagnetic = electromagnetic_eos_for_qed(t_gamma_mev, qed_model)?;

    let rho_nue_pair = require_output(
        "rho_nue_pair",
        RHO_NEUTRINO_PAIR_COEFFICIENT * t_nue_mev.powi(4),
    )?;
    let rho_nux_pair = require_output(
        "rho_nux_pair",
        RHO_NEUTRINO_PAIR_COEFFICIENT * t_nux_mev.powi(4),
    )?;
    let drho_nue_pair_dt = require_output(
        "drho_nue_pair_dT",
        4.0 * RHO_NEUTRINO_PAIR_COEFFICIENT * t_nue_mev.powi(3),
    )?;
    let drho_nux_pair_dt = require_output(
        "drho_nux_pair_dT",
        4.0 * RHO_NEUTRINO_PAIR_COEFFICIENT * t_nux_mev.powi(3),
    )?;
    let rho_total = require_output(
        "rho_total_3T",
        electromagnetic.rho + rho_nue_pair + 2.0 * rho_nux_pair,
    )?;
    let pressure_total = require_output(
        "pressure_total_3T",
        electromagnetic.pressure + rho_nue_pair / 3.0 + 2.0 * rho_nux_pair / 3.0,
    )?;
    let h_squared = (8.0 * PI * NEWTON_G_MEV_MINUS_2 / 3.0) * rho_total;
    let h_mev = require_output("H_3T", h_squared.sqrt())?;
    let h_inverse_seconds = require_output("H_3T_s^-1", h_mev * MEV_TO_INVERSE_SECONDS)?;

    let transfer = finite_mass_fd_electron_energy_transfer(t_gamma_mev, t_nue_mev, t_nux_mev)
        .map_err(FlrwThermoError::ElectronThermal)?;
    let q_nue_pair_mev5 = require_finite_output("Q_nue_pair", transfer.nue_pair_mev5)?;
    let q_nux_pair_mev5 = require_finite_output("Q_nux_pair", transfer.nux_pair_mev5)?;
    let q_total_mev5 = require_finite_output("Q_nu_total", transfer.total_mev5())?;

    // Positive Q means energy gained by neutrinos.  The equal and opposite
    // debit is applied to the electromagnetic sector exactly once.
    let d_tgamma_d_lna = require_finite_output(
        "dT_gamma_3T/dln(a)",
        (-3.0 * (electromagnetic.rho + electromagnetic.pressure) - q_total_mev5 / h_mev)
            / electromagnetic.drho_dt,
    )?;
    let d_tnue_d_lna = require_finite_output(
        "dT_nue/dln(a)",
        -t_nue_mev + q_nue_pair_mev5 / (h_mev * drho_nue_pair_dt),
    )?;
    let d_tnux_d_lna = require_finite_output(
        "dT_nux/dln(a)",
        -t_nux_mev + q_nux_pair_mev5 / (h_mev * drho_nux_pair_dt),
    )?;

    Ok(ThreeTemperatureFlrwState {
        t_gamma_mev,
        t_nue_mev,
        t_nux_mev,
        electromagnetic,
        rho_nue_pair,
        rho_nux_pair,
        rho_total,
        pressure_total,
        q_nue_pair_mev5,
        q_nux_pair_mev5,
        q_total_mev5,
        d_tgamma_d_lna,
        d_tnue_d_lna,
        d_tnux_d_lna,
        h_mev,
        h_inverse_seconds,
    })
}

impl OdeSystem for ThreeTemperatureFlrwSystem {
    fn dimension(&self) -> usize {
        3
    }

    fn state_is_valid(&self, state: &[f64]) -> bool {
        state.len() == 3 && state.iter().all(|value| value.is_finite() && *value > 0.0)
    }

    fn rhs(&self, _ln_a: f64, y: &[f64], out: &mut [f64]) {
        match self.thermo_state([y[0], y[1], y[2]]) {
            Ok(state) => {
                out[0] = state.d_tgamma_d_lna;
                out[1] = state.d_tnue_d_lna;
                out[2] = state.d_tnux_d_lna;
            }
            Err(_) => out.fill(f64::NAN),
        }
    }

    fn jacobian(&self, ln_a: f64, y: &[f64], out: &mut [f64]) {
        const RELATIVE_STEP: f64 = 1.0e-5;
        for column in 0..3 {
            let step = RELATIVE_STEP * y[column].abs();
            let mut plus_state = [y[0], y[1], y[2]];
            let mut minus_state = plus_state;
            plus_state[column] += step;
            minus_state[column] -= step;
            let mut plus = [0.0; 3];
            let mut minus = [0.0; 3];
            self.rhs(ln_a, &plus_state, &mut plus);
            self.rhs(ln_a, &minus_state, &mut minus);
            for row in 0..3 {
                out[row * 3 + column] = (plus[row] - minus[row]) / (2.0 * step);
            }
        }
    }

    fn dfdt(&self, _ln_a: f64, _y: &[f64], out: &mut [f64]) {
        out.fill(0.0);
    }
}

#[derive(Clone, Copy, Debug)]
struct ElectronPairEos {
    rho: f64,
    pressure: f64,
    drho_dt: f64,
    dpressure_dt: f64,
}

fn require_positive_finite(quantity: &'static str, value: f64) -> Result<f64, FlrwThermoError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(FlrwThermoError::InvalidInput { quantity, value });
    }
    Ok(value)
}

fn require_output(quantity: &'static str, value: f64) -> Result<f64, FlrwThermoError> {
    if !value.is_finite() {
        return Err(FlrwThermoError::NonFiniteOutput { quantity, value });
    }
    if value <= 0.0 {
        return Err(FlrwThermoError::NonPositiveOutput { quantity, value });
    }
    Ok(value)
}

fn require_finite_output(quantity: &'static str, value: f64) -> Result<f64, FlrwThermoError> {
    if !value.is_finite() {
        return Err(FlrwThermoError::NonFiniteOutput { quantity, value });
    }
    Ok(value)
}

/// Integrate the finite-mass pair EOS after `p = m_e sinh(theta)`.
///
/// This maps the square-root threshold at `E=m_e` to the regular endpoint
/// `theta=0`.  The upper endpoint satisfies `E/T = m_e/T + tail_e_folds`;
/// no integrand or returned thermodynamic quantity is clipped.
fn electron_pair_eos_with_quadrature(
    temperature: f64,
    panels: usize,
    tail_e_folds: f64,
) -> Result<ElectronPairEos, FlrwThermoError> {
    let temperature = require_positive_finite("T_gamma", temperature)?;
    let tail_e_folds = require_positive_finite("quadrature_tail_e_folds", tail_e_folds)?;
    if panels < 2 || !panels.is_multiple_of(2) {
        return Err(FlrwThermoError::InvalidPanelCount(panels));
    }

    let x = ELECTRON_MASS_MEV / temperature;
    let theta_max = (1.0 + tail_e_folds / x).acosh();
    let step = theta_max / panels as f64;
    let mut sums = [0.0; 4];

    for index in 0..=panels {
        let theta = index as f64 * step;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let epsilon = x * cosh;
        let exp_negative = (-epsilon).exp();
        let occupation = exp_negative / (1.0 + exp_negative);
        let thermal_response = occupation * (1.0 - occupation);
        let sinh2 = sinh * sinh;
        let sinh4 = sinh2 * sinh2;
        let x4 = x.powi(4);
        let x5 = x4 * x;
        let values = [
            x4 * sinh2 * cosh * cosh * occupation,
            x4 * sinh4 * occupation,
            x5 * sinh2 * cosh.powi(3) * thermal_response,
            x5 * sinh4 * cosh * thermal_response,
        ];
        let weight = if index == 0 || index == panels {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        for component in 0..4 {
            sums[component] += weight * values[component];
        }
    }
    for sum in &mut sums {
        *sum *= step / 3.0;
    }

    let t3 = temperature.powi(3);
    let t4 = t3 * temperature;
    let pair = ElectronPairEos {
        rho: 2.0 * t4 * sums[0] / (PI * PI),
        pressure: 2.0 * t4 * sums[1] / (3.0 * PI * PI),
        drho_dt: 2.0 * t3 * sums[2] / (PI * PI),
        dpressure_dt: 2.0 * t3 * sums[3] / (3.0 * PI * PI),
    };
    for (quantity, value) in [
        ("rho_e_pair", pair.rho),
        ("pressure_e_pair", pair.pressure),
        ("drho_e_pair_dT", pair.drho_dt),
        ("dpressure_e_pair_dT", pair.dpressure_dt),
    ] {
        if !value.is_finite() {
            return Err(FlrwThermoError::NonFiniteOutput { quantity, value });
        }
    }
    Ok(pair)
}

pub(crate) fn electromagnetic_eos(temperature: f64) -> Result<ElectromagneticEos, FlrwThermoError> {
    let temperature = require_positive_finite("T_gamma", temperature)?;
    let pair = electron_pair_eos_with_quadrature(
        temperature,
        DEFAULT_SIMPSON_PANELS,
        DEFAULT_TAIL_E_FOLDS,
    )?;
    let rho_gamma = RHO_GAMMA_COEFFICIENT * temperature.powi(4);
    let pressure_gamma = rho_gamma / 3.0;
    let drho_dt = 4.0 * rho_gamma / temperature + pair.drho_dt;
    let dpressure_dt = 4.0 * pressure_gamma / temperature + pair.dpressure_dt;
    let rho = rho_gamma + pair.rho;
    let pressure = pressure_gamma + pair.pressure;
    let entropy = (rho + pressure) / temperature;
    let ds_dt = (drho_dt + dpressure_dt) / temperature - entropy / temperature;

    Ok(ElectromagneticEos {
        rho: require_output("rho_em", rho)?,
        pressure: require_output("pressure_em", pressure)?,
        entropy: require_output("entropy_em", entropy)?,
        drho_dt: require_output("drho_em_dT", drho_dt)?,
        ds_dt: require_output("ds_em_dT", ds_dt)?,
    })
}

pub(crate) fn electromagnetic_eos_for_qed(
    temperature: f64,
    qed_model: FiniteTemperatureQed,
) -> Result<ElectromagneticEos, FlrwThermoError> {
    if qed_model == FiniteTemperatureQed::Off {
        return electromagnetic_eos(temperature);
    }
    let tree = electromagnetic_eos(temperature)?;
    let correction =
        qed_correction(temperature, ELECTRON_MASS_MEV, qed_model).map_err(FlrwThermoError::Qed)?;
    Ok(ElectromagneticEos {
        rho: require_output("rho_em_with_qed", tree.rho + correction.energy_density)?,
        pressure: require_output("pressure_em_with_qed", tree.pressure + correction.pressure)?,
        entropy: require_output(
            "entropy_em_with_qed",
            tree.entropy + correction.entropy_density,
        )?,
        drho_dt: require_output(
            "drho_em_with_qed_dT",
            tree.drho_dt + correction.denergy_density_dt,
        )?,
        ds_dt: require_output(
            "ds_em_with_qed_dT",
            tree.ds_dt + correction.dentropy_density_dt,
        )?,
    })
}

fn owned_temperature_derivative(
    t_gamma: f64,
    electromagnetic: ElectromagneticEos,
    entropy_owner: EntropyOwner,
) -> Result<f64, FlrwThermoError> {
    let (owned_entropy, owned_ds_dt) = match entropy_owner {
        EntropyOwner::CommonElectromagneticNeutrinoBath => {
            let rho_neutrino = RHO_THREE_NU_COEFFICIENT * t_gamma.powi(4);
            let neutrino_entropy = 4.0 * rho_neutrino / (3.0 * t_gamma);
            (
                electromagnetic.entropy + neutrino_entropy,
                electromagnetic.ds_dt + 3.0 * neutrino_entropy / t_gamma,
            )
        }
        EntropyOwner::ElectromagneticBathAfterDecoupling => {
            (electromagnetic.entropy, electromagnetic.ds_dt)
        }
    };
    let derivative = -3.0 * owned_entropy / owned_ds_dt;
    if !derivative.is_finite() {
        return Err(FlrwThermoError::NonFiniteOutput {
            quantity: "dT_gamma/dln(a)",
            value: derivative,
        });
    }
    Ok(derivative)
}

fn neutrino_temperature_for_owner(
    t_gamma: f64,
    anchor: NeutrinoTemperatureAnchor,
    em: ElectromagneticEos,
    entropy_owner: EntropyOwner,
    qed_model: FiniteTemperatureQed,
) -> Result<f64, FlrwThermoError> {
    match entropy_owner {
        EntropyOwner::CommonElectromagneticNeutrinoBath => Ok(t_gamma),
        EntropyOwner::ElectromagneticBathAfterDecoupling => match anchor {
            NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(t_dec) => {
                let t_dec = require_positive_finite("T_dec", t_dec)?;
                let entropy_at_decoupling = electromagnetic_eos_for_qed(t_dec, qed_model)?.entropy;
                let t_nu = t_dec * (em.entropy / entropy_at_decoupling).cbrt();
                require_output("T_nu", t_nu)
            }
            NeutrinoTemperatureAnchor::IdealHighTemperatureEntropy => {
                let entropy_coefficient = if qed_model == FiniteTemperatureQed::Off {
                    IDEAL_HIGH_T_EM_ENTROPY_COEFFICIENT
                } else {
                    high_temperature_em_entropy_coefficient(qed_model)
                };
                let t_nu = (em.entropy / entropy_coefficient).cbrt();
                require_output("T_nu", t_nu)
            }
        },
    }
}

fn flrw_thermo_state_for_owner(
    t_gamma: f64,
    anchor: NeutrinoTemperatureAnchor,
    entropy_owner: EntropyOwner,
    qed_model: FiniteTemperatureQed,
) -> Result<FlrwThermoState, FlrwThermoError> {
    let t_gamma = require_positive_finite("T_gamma", t_gamma)?;
    if let NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(t_dec) = anchor {
        require_positive_finite("T_dec", t_dec)?;
    }
    let electromagnetic = electromagnetic_eos_for_qed(t_gamma, qed_model)?;
    let t_nu =
        neutrino_temperature_for_owner(t_gamma, anchor, electromagnetic, entropy_owner, qed_model)?;
    let rho_neutrino = RHO_THREE_NU_COEFFICIENT * t_nu.powi(4);
    let pressure_neutrino = rho_neutrino / 3.0;
    let rho_total = electromagnetic.rho + rho_neutrino;
    let pressure_total = electromagnetic.pressure + pressure_neutrino;

    let d_tgamma_d_lna = owned_temperature_derivative(t_gamma, electromagnetic, entropy_owner)?;

    // Flat-FLRW Friedmann equation in the natural-unit convention above.
    let h_squared = (8.0 * PI * NEWTON_G_MEV_MINUS_2 / 3.0) * rho_total;
    let h_mev = require_output("H", h_squared.sqrt())?;
    let h_inverse_seconds = require_output("H_s^-1", h_mev * MEV_TO_INVERSE_SECONDS)?;

    Ok(FlrwThermoState {
        t_gamma_mev: t_gamma,
        t_nu_mev: t_nu,
        entropy_owner,
        electromagnetic,
        rho_neutrino: require_output("rho_nu", rho_neutrino)?,
        rho_total: require_output("rho_total", rho_total)?,
        pressure_total: require_output("pressure_total", pressure_total)?,
        d_tgamma_d_lna,
        h_mev,
        h_inverse_seconds,
    })
}

pub(crate) fn flrw_thermo_state(
    t_gamma: f64,
    t_dec: f64,
) -> Result<FlrwThermoState, FlrwThermoError> {
    let entropy_owner = if t_gamma >= t_dec {
        EntropyOwner::CommonElectromagneticNeutrinoBath
    } else {
        EntropyOwner::ElectromagneticBathAfterDecoupling
    };
    flrw_thermo_state_for_owner(
        t_gamma,
        NeutrinoTemperatureAnchor::FiniteDecouplingTemperature(t_dec),
        entropy_owner,
        FiniteTemperatureQed::Off,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::{OdeConfig, SolverKind, TerminalEvent, solve};

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs()
    }

    #[test]
    fn three_temperature_equilibrium_source_is_an_exact_null() {
        for qed_model in [
            FiniteTemperatureQed::Off,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        ] {
            let state = three_temperature_flrw_state(2.0, 2.0, 2.0, qed_model).unwrap();
            assert_eq!(state.t_gamma_mev.to_bits(), 2.0_f64.to_bits());
            assert_eq!(state.t_nue_mev.to_bits(), 2.0_f64.to_bits());
            assert_eq!(state.t_nux_mev.to_bits(), 2.0_f64.to_bits());
            assert_eq!(state.q_nue_pair_mev5.to_bits(), 0.0_f64.to_bits());
            assert_eq!(state.q_nux_pair_mev5.to_bits(), 0.0_f64.to_bits());
            assert_eq!(state.q_total_mev5.to_bits(), 0.0_f64.to_bits());
            assert_eq!(state.d_tnue_d_lna.to_bits(), (-2.0_f64).to_bits());
            assert_eq!(state.d_tnux_d_lna.to_bits(), (-2.0_f64).to_bits());
            assert_eq!(
                state.rho_total.to_bits(),
                (state.electromagnetic.rho + state.rho_nue_pair + 2.0 * state.rho_nux_pair)
                    .to_bits()
            );
            assert!(state.h_mev > 0.0);
            assert_eq!(
                state.h_inverse_seconds.to_bits(),
                (state.h_mev * MEV_TO_INVERSE_SECONDS).to_bits()
            );
        }
    }

    #[test]
    fn three_temperature_rhs_closes_the_combined_first_law() {
        for (qed_model, temperatures) in [
            (FiniteTemperatureQed::Off, [1.0, 0.83, 0.74]),
            (FiniteTemperatureQed::PrimatCompleteE2E3, [0.7, 0.61, 0.57]),
        ] {
            let state = three_temperature_flrw_state(
                temperatures[0],
                temperatures[1],
                temperatures[2],
                qed_model,
            )
            .unwrap();
            assert_eq!(
                state.q_total_mev5.to_bits(),
                (state.q_nue_pair_mev5 + 2.0 * state.q_nux_pair_mev5).to_bits()
            );
            let d_rho_d_lna = state.electromagnetic.drho_dt * state.d_tgamma_d_lna
                + 4.0 * state.rho_nue_pair / state.t_nue_mev * state.d_tnue_d_lna
                + 2.0 * 4.0 * state.rho_nux_pair / state.t_nux_mev * state.d_tnux_d_lna;
            let expected = -3.0 * (state.rho_total + state.pressure_total);
            assert!(
                (d_rho_d_lna - expected).abs() <= 3.0e-14 * expected.abs(),
                "qed={qed_model:?}: d_rho/dN={d_rho_d_lna:.17e}, expected={expected:.17e}"
            );
        }
    }

    #[test]
    fn three_temperature_invalid_states_fail_without_clipping() {
        let system = ThreeTemperatureFlrwSystem::new(FiniteTemperatureQed::Off);
        assert_eq!(
            system.thermo_state([1.0, -0.1, 0.9]).unwrap_err(),
            FlrwThermoError::InvalidInput {
                quantity: "T_nue",
                value: -0.1,
            }
        );
        let nan_error = system.thermo_state([1.0, 0.9, f64::NAN]).unwrap_err();
        match nan_error {
            FlrwThermoError::InvalidInput { quantity, value } => {
                assert_eq!(quantity, "T_nux");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert!(!system.state_is_valid(&[1.0, 0.0, 0.9]));
        let mut rhs = [0.0; 3];
        system.rhs(0.0, &[1.0, 0.0, 0.9], &mut rhs);
        assert!(rhs.into_iter().all(f64::is_nan));
    }

    #[test]
    fn transformed_quadrature_converges() {
        let coarse = electron_pair_eos_with_quadrature(0.5, 16, 48.0).unwrap();
        let medium = electron_pair_eos_with_quadrature(0.5, 32, 48.0).unwrap();
        let fine = electron_pair_eos_with_quadrature(0.5, 64, 48.0).unwrap();
        let coarse_change = (medium.rho - coarse.rho).abs();
        let fine_change = (fine.rho - medium.rho).abs();
        assert!(fine_change < coarse_change);
        assert!(fine_change / fine.rho < 1.0e-9);
    }

    #[test]
    fn eos_matches_independent_continuum_quadrature_references() {
        // These values were independently evaluated in the dimensionless
        // momentum variable y=p/T with scipy.integrate.quad (epsabs=1e-13,
        // epsrel=1e-13).  They are not generated by this theta/Simpson path.
        let references = [
            (
                0.005,
                4.112_335_167_120_566e-10,
                1.370_778_389_040_188_5e-10,
                1.096_622_711_232_150_9e-7,
                3.289_868_133_696_452_5e-7,
                6.579_736_267_392_905e-5,
            ),
            (
                0.02,
                1.052_757_815_038_867_7e-7,
                3.509_192_680_461_97e-8,
                7.018_385_415_425_324e-6,
                2.105_515_772_130_564e-5,
                1.052_757_886_065_282e-3,
            ),
            (
                0.05,
                4.139_086_653_323_757e-6,
                1.373_028_966_603_716_5e-6,
                1.102_423_123_985_494_6e-4,
                3.354_282_189_746_542_5e-4,
                6.708_564_379_493_085e-3,
            ),
            (
                0.1,
                8.286_707_098_987_887e-5,
                2.440_300_859_277_339_4e-5,
                1.072_700_795_826_522_5e-3,
                3.861_460_022_487_728_4e-3,
                3.861_460_022_487_728e-2,
            ),
            (
                0.2,
                2.123_441_832_108_308e-3,
                5.865_028_804_735_6e-4,
                1.354_972_356_290_933_9e-2,
                4.811_185_006_955_418e-2,
                2.405_592_503_477_709e-1,
            ),
            (
                0.5,
                1.073_033_525_517_558e-1,
                3.340_625_013_677_037e-2,
                2.814_192_053_770_523_4e-1,
                8.815_674_053_453_113e-1,
                1.763_134_810_690_622_6,
            ),
            (
                1.0,
                1.786_789_849_896_462_7,
                5.831_079_599_410_696e-1,
                2.369_897_809_837_532,
                7.193_372_934_740_34,
                7.193_372_934_740_34,
            ),
            (
                2.0,
                2.886_234_107_393_044_5e1,
                9.565_557_979_552_09,
                1.921_394_952_674_126_7e1,
                5.781_421_336_015_525e1,
                2.890_710_668_007_762e1,
            ),
            (
                5.0,
                1.130_345_926_338_910_7e3,
                3.764_231_648_251_340_5e2,
                3.013_538_182_328_09e2,
                9.044_959_645_003_262e2,
                1.808_991_929_000_652_6e2,
            ),
            (
                10.0,
                1.809_209_589_331_734e4,
                6.029_252_618_403_711e3,
                2.412_134_851_172_105_3e3,
                7.237_274_607_946_514e3,
                7.237_274_607_946_513e2,
            ),
        ];
        for (temperature, rho, pressure, entropy, drho_dt, ds_dt) in references {
            let actual = electromagnetic_eos(temperature).unwrap();
            assert!(relative_error(actual.rho, rho) < 3.0e-13);
            assert!(relative_error(actual.pressure, pressure) < 3.0e-13);
            assert!(relative_error(actual.entropy, entropy) < 3.0e-13);
            assert!(relative_error(actual.drho_dt, drho_dt) < 3.0e-13);
            assert!(relative_error(actual.ds_dt, ds_dt) < 3.0e-13);
        }
    }

    #[test]
    fn frozen_gl64_port_values_are_not_continuum_authority() {
        // Locked outputs from the legacy Python GL64 tree-level EOS.  This
        // regression intentionally demonstrates the known continuum bias;
        // it is not a parity acceptance tolerance for this implementation.
        let legacy_total_rho_at_point_one = 8.286_910_202_509_072e-5;
        let legacy_total_pressure_at_point_one = 2.440_300_524_378_856_4e-5;
        let continuum = electromagnetic_eos(0.1).unwrap();
        let rho_difference = relative_error(legacy_total_rho_at_point_one, continuum.rho);
        let pressure_difference =
            relative_error(legacy_total_pressure_at_point_one, continuum.pressure);
        assert!((rho_difference - 2.450_955_714_486e-5).abs() < 2.0e-15);
        assert!((pressure_difference - 1.372_365_549_556e-7).abs() < 2.0e-15);
    }

    #[test]
    fn default_quadrature_matches_refined_tail_and_resolution() {
        for temperature in [0.005, 0.1, 0.5, 2.0, 10.0] {
            let default = electron_pair_eos_with_quadrature(temperature, 256, 48.0).unwrap();
            let refined = electron_pair_eos_with_quadrature(temperature, 512, 56.0).unwrap();
            assert!(relative_error(default.rho, refined.rho) < 2.0e-12);
            assert!(relative_error(default.pressure, refined.pressure) < 2.0e-12);
            assert!(relative_error(default.drho_dt, refined.drho_dt) < 2.0e-12);
            assert!(relative_error(default.dpressure_dt, refined.dpressure_dt) < 2.0e-12);
        }
    }

    #[test]
    fn relativistic_and_photon_limits_are_recovered() {
        let hot = electromagnetic_eos(100.0).unwrap();
        let hot_expected_rho = (11.0 / 4.0) * RHO_GAMMA_COEFFICIENT * 100.0_f64.powi(4);
        assert!(relative_error(hot.rho, hot_expected_rho) < 2.0e-6);
        assert!(relative_error(hot.pressure, hot.rho / 3.0) < 3.0e-6);

        let cold = electromagnetic_eos(0.005).unwrap();
        let photon_rho = RHO_GAMMA_COEFFICIENT * 0.005_f64.powi(4);
        assert!(relative_error(cold.rho, photon_rho) < 1.0e-14);
        assert!(relative_error(cold.pressure, photon_rho / 3.0) < 1.0e-14);
        assert!(relative_error(-3.0 * cold.entropy / cold.ds_dt, -0.005) < 1.0e-14);
    }

    #[test]
    fn electron_pair_reaches_the_nonrelativistic_maxwell_boltzmann_limit() {
        let temperature = 0.01;
        let pair = electron_pair_eos_with_quadrature(temperature, 256, 48.0).unwrap();
        let number_density_leading = 4.0
            * (ELECTRON_MASS_MEV * temperature / (2.0 * PI)).powf(1.5)
            * (-ELECTRON_MASS_MEV / temperature).exp();
        let rho_leading = number_density_leading * (ELECTRON_MASS_MEV + 1.5 * temperature);
        let pressure_leading = number_density_leading * temperature;
        assert!(relative_error(pair.rho, rho_leading) < 0.04);
        assert!(relative_error(pair.pressure, pressure_leading) < 0.04);
    }

    #[test]
    fn analytic_temperature_derivative_matches_centered_difference() {
        let temperature = 0.7;
        let step = 1.0e-5;
        let eos = electromagnetic_eos(temperature).unwrap();
        let plus = electromagnetic_eos(temperature + step).unwrap();
        let minus = electromagnetic_eos(temperature - step).unwrap();
        let finite_difference = (plus.rho - minus.rho) / (2.0 * step);
        assert!(relative_error(eos.drho_dt, finite_difference) < 2.0e-9);
    }

    #[test]
    fn instantaneous_decoupling_changes_entropy_owner_continuously() {
        let t_dec = 2.0;
        let at_decoupling = flrw_thermo_state(t_dec, t_dec).unwrap();
        assert_eq!(at_decoupling.t_nu_mev, t_dec);
        assert_eq!(
            at_decoupling.entropy_owner,
            EntropyOwner::CommonElectromagneticNeutrinoBath
        );

        let cold = flrw_thermo_state(0.005, t_dec).unwrap();
        assert_eq!(
            cold.entropy_owner,
            EntropyOwner::ElectromagneticBathAfterDecoupling
        );
        let entropy_dec = electromagnetic_eos(t_dec).unwrap().entropy;
        let expected_t_nu = t_dec * (cold.electromagnetic.entropy / entropy_dec).cbrt();
        assert!(relative_error(cold.t_nu_mev, expected_t_nu) < 1.0e-14);
        assert!((cold.t_nu_mev / cold.t_gamma_mev - (4.0 / 11.0_f64).cbrt()).abs() < 0.002);
    }

    #[test]
    fn cooling_and_owned_comoving_entropy_hold_on_both_branches() {
        let t_dec = 2.0;
        for temperature in [10.0, 3.0, 2.0, 1.0, 0.1, 0.005] {
            let state = flrw_thermo_state(temperature, t_dec).unwrap();
            assert!(state.d_tgamma_d_lna < 0.0);
            let (owned_entropy, owned_derivative) = match state.entropy_owner {
                EntropyOwner::CommonElectromagneticNeutrinoBath => {
                    assert_eq!(state.t_nu_mev, temperature);
                    let s_nu = 4.0 * state.rho_neutrino / (3.0 * temperature);
                    (
                        state.electromagnetic.entropy + s_nu,
                        state.electromagnetic.ds_dt + 3.0 * s_nu / temperature,
                    )
                }
                EntropyOwner::ElectromagneticBathAfterDecoupling => {
                    assert!(state.t_nu_mev < state.t_gamma_mev);
                    (state.electromagnetic.entropy, state.electromagnetic.ds_dt)
                }
            };
            let ds_d_lna = owned_derivative * state.d_tgamma_d_lna;
            assert!(relative_error(ds_d_lna, -3.0 * owned_entropy) < 2.0e-15);
        }
    }

    #[test]
    fn friedmann_limit_has_standard_g_star() {
        let temperature = 100.0;
        let state = flrw_thermo_state(temperature, 2.0).unwrap();
        let rho_relativistic = PI * PI * 10.75 * temperature.powi(4) / 30.0;
        let expected_h = ((8.0 * PI * NEWTON_G_MEV_MINUS_2 / 3.0) * rho_relativistic).sqrt();
        assert!(relative_error(state.h_mev, expected_h) < 1.0e-6);
        assert_eq!(
            state.rho_total,
            state.electromagnetic.rho + state.rho_neutrino
        );
        assert!(state.pressure_total > 0.0);
        assert!(state.d_tgamma_d_lna < 0.0);
        assert_eq!(
            state.h_inverse_seconds,
            state.h_mev * MEV_TO_INVERSE_SECONDS
        );
        let independent_massless_h_at_one_mev = 0.677_345_743_346_034_1;
        assert!(
            relative_error(
                state.h_inverse_seconds / temperature.powi(2),
                independent_massless_h_at_one_mev,
            ) < 1.0e-6
        );
    }

    #[test]
    fn ideal_high_temperature_entropy_mode_matches_live_primat_blocks() {
        // PRIMAT v0.3.2 (commit
        // 21ff8f39fa18e3937e9fdf386cfa982361bfdfce), QED and incomplete
        // decoupling disabled.  Values were read from its public Plasma,
        // InstantaneousDecoupling, and StandardBackground APIs with the
        // CDM/Lambda amplitudes reduced below numerical relevance.
        let references = [
            (
                10.0,
                9.999_398_788_144_413,
                1.809_209_589_331_472_7e4,
                6.029_252_618_402_845e3,
                2.412_134_851_171_757_4e3,
                6.772_851_414_524_483e1,
            ),
            (
                1.0,
                9.940_690_848_188_369e-1,
                1.786_789_849_876_512_5,
                5.831_079_599_344_233e-1,
                2.369_897_809_810_935_6,
                6.712_615_452_153_449e-1,
            ),
            (
                0.1,
                7.632_489_731_303_93e-2,
                8.286_707_098_483_96e-5,
                2.440_300_859_115_771e-5,
                1.072_700_795_759_973e-3,
                4.284_167_098_400_032e-3,
            ),
            (
                0.005,
                3.568_829_277_518_042e-3,
                4.112_335_167_120_566e-10,
                1.370_778_389_040_188_5e-10,
                1.096_622_711_232_150_9e-7,
                9.470_799_585_748_04e-6,
            ),
        ];
        let system = IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling();
        for (temperature, t_nu, rho, pressure, entropy, h_per_second) in references {
            let state = system.thermo_state(temperature).unwrap();
            assert!(relative_error(state.t_nu_mev, t_nu) < 5.0e-9);
            assert!(relative_error(state.electromagnetic.rho, rho) < 2.0e-9);
            assert!(relative_error(state.electromagnetic.pressure, pressure) < 2.0e-9);
            assert!(relative_error(state.electromagnetic.entropy, entropy) < 2.0e-9);
            assert!(relative_error(state.h_inverse_seconds, h_per_second) < 5.0e-9);
        }
        let cold = system.thermo_state(0.005).unwrap();
        assert!(((cold.t_nu_mev / cold.t_gamma_mev).powi(3) - 4.0 / 11.0).abs() < 2.0e-14);
    }

    #[test]
    fn qed_off_is_bitwise_identical_to_the_f06_eos_and_flrw_path() {
        let explicit_off = IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(
            FiniteTemperatureQed::Off,
        );
        let f06 = IdealFlrwSystem::ideal_high_temperature_instantaneous_decoupling();
        for temperature in [0.005, 0.1, 0.5, 1.0, 10.0] {
            let direct = electromagnetic_eos(temperature).unwrap();
            let selected =
                electromagnetic_eos_for_qed(temperature, FiniteTemperatureQed::Off).unwrap();
            for (left, right) in [
                (direct.rho, selected.rho),
                (direct.pressure, selected.pressure),
                (direct.entropy, selected.entropy),
                (direct.drho_dt, selected.drho_dt),
                (direct.ds_dt, selected.ds_dt),
            ] {
                assert_eq!(left.to_bits(), right.to_bits());
            }

            let direct_state = f06.thermo_state(temperature).unwrap();
            let selected_state = explicit_off.thermo_state(temperature).unwrap();
            for (left, right) in [
                (direct_state.t_gamma_mev, selected_state.t_gamma_mev),
                (direct_state.t_nu_mev, selected_state.t_nu_mev),
                (direct_state.rho_neutrino, selected_state.rho_neutrino),
                (direct_state.rho_total, selected_state.rho_total),
                (direct_state.pressure_total, selected_state.pressure_total),
                (direct_state.d_tgamma_d_lna, selected_state.d_tgamma_d_lna),
                (direct_state.h_mev, selected_state.h_mev),
                (
                    direct_state.h_inverse_seconds,
                    selected_state.h_inverse_seconds,
                ),
            ] {
                assert_eq!(left.to_bits(), right.to_bits());
            }
        }
    }

    #[test]
    fn complete_qed_eos_is_positive_and_thermodynamically_closed() {
        for temperature in [0.005, 0.02, 0.1, 0.5, 1.0, 10.0] {
            let tree = electromagnetic_eos(temperature).unwrap();
            let corrected =
                electromagnetic_eos_for_qed(temperature, FiniteTemperatureQed::PrimatCompleteE2E3)
                    .unwrap();
            let correction = qed_correction(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatCompleteE2E3,
            )
            .unwrap();
            assert!(corrected.rho > 0.0);
            assert!(corrected.pressure > 0.0);
            assert!(corrected.entropy > 0.0);
            assert!(corrected.drho_dt > 0.0);
            assert!(corrected.ds_dt > 0.0);
            for (tree_value, corrected_value, delta) in [
                (tree.rho, corrected.rho, correction.energy_density),
                (tree.pressure, corrected.pressure, correction.pressure),
                (tree.entropy, corrected.entropy, correction.entropy_density),
            ] {
                // Direct addition is the contract.  At the cold end the
                // nonzero correction may legitimately round below one ulp
                // of the much larger tree-level total.
                assert_eq!(corrected_value.to_bits(), (tree_value + delta).to_bits());
            }
            assert!(
                relative_error(
                    (corrected.rho + corrected.pressure) / temperature,
                    corrected.entropy
                ) < 3.0e-15
            );
        }
    }

    #[test]
    fn qed_high_temperature_anchor_and_cold_reheating_ratio_are_consistent() {
        let model = FiniteTemperatureQed::PrimatCompleteE2E3;
        let system = IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(model);
        let hot = system.thermo_state(100.0).unwrap();
        assert!((hot.t_nu_mev / hot.t_gamma_mev - 1.0).abs() < 1.0e-6);
        let cold = system.thermo_state(0.005).unwrap();
        let alpha = crate::qed_eos::FINE_STRUCTURE_ALPHA;
        let expected_ratio_cubed = 11.0 / 4.0 - 25.0 * alpha / (8.0 * PI)
            + 10.0 * alpha.powf(1.5) * (PI / 3.0).sqrt() / PI.powi(2);
        let actual_ratio_cubed = (cold.t_gamma_mev / cold.t_nu_mev).powi(3);
        assert!(relative_error(actual_ratio_cubed, expected_ratio_cubed) < 2.0e-14);
        assert!(actual_ratio_cubed < 11.0 / 4.0);
    }

    #[test]
    fn qed_finite_decoupling_anchor_is_consistent_on_both_entropy_owners() {
        let model = FiniteTemperatureQed::PrimatCompleteE2E3;
        let t_dec = 2.0;
        let common = IdealFlrwSystem::common_bath_leg_with_qed(t_dec, model);
        let electromagnetic = IdealFlrwSystem::electromagnetic_bath_leg_with_qed(t_dec, model);
        let common_state = common.thermo_state(3.0).unwrap();
        assert_eq!(common_state.t_nu_mev.to_bits(), 3.0_f64.to_bits());
        let at_decoupling = common.thermo_state(t_dec).unwrap();
        let cold = electromagnetic.thermo_state(0.1).unwrap();
        let expected =
            t_dec * (cold.electromagnetic.entropy / at_decoupling.electromagnetic.entropy).cbrt();
        assert!(relative_error(cold.t_nu_mev, expected) < 2.0e-15);
        assert!(cold.t_nu_mev < cold.t_gamma_mev);
    }

    #[test]
    fn both_solvers_reach_the_complete_qed_flrw_event_with_entropy_closure() {
        let initial_temperature = 10.0;
        let final_temperature = 0.005;
        let model = FiniteTemperatureQed::PrimatCompleteE2E3;
        let system = IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(model);
        let final_event_fn = |_ln_a: f64, y: &[f64]| y[0] - final_temperature;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let config = OdeConfig {
            rtol: 2.0e-8,
            atol: vec![1.0e-11],
            h_init: 1.0e-4,
            h_min: 1.0e-13,
            h_max: 0.05,
            max_attempts: 20_000,
        };
        let initial_entropy = system
            .thermo_state(initial_temperature)
            .unwrap()
            .electromagnetic
            .entropy;
        let final_entropy = system
            .thermo_state(final_temperature)
            .unwrap()
            .electromagnetic
            .entropy;
        let expected_final_lna = (initial_entropy / final_entropy).ln() / 3.0;
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &system,
                (0.0, 12.0),
                &[initial_temperature],
                &config,
                Some(&final_event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!((result.y[0] - final_temperature).abs() < 1.0e-9);
            assert!((result.t - expected_final_lna).abs() < 2.0e-7);
            endpoints.push(result.t);
        }
        assert!((endpoints[0] - endpoints[1]).abs() < 2.0e-7);
    }

    #[test]
    fn both_solvers_match_the_independent_qed_off_three_temperature_endpoint() {
        // Frozen before this Rust regression from a separate Python
        // formulation: scipy.integrate.quad_vec finite-mass EM EOS plus an
        // independent plasma-frame/CM collision integral.  DOP853 and Radau
        // at rtol=3e-10 agreed to 5.4e-12 in N, 3.0e-13 MeV in either
        // neutrino temperature, and 5.6e-10 in thermal N_eff.  The anchors
        // below are their midpoints, not values fitted to either Rust solver.
        const EXPECTED_LN_A: f64 = 7.936_698_486_852_856;
        const EXPECTED_TNUE_MEV: f64 = 0.003_583_185_277_385_943;
        const EXPECTED_TNUX_MEV: f64 = 0.003_576_790_300_999_642;
        const EXPECTED_NEFF_THERMAL: f64 = 3.034_093_261_751_178;
        let initial_temperature = 10.0;
        let final_photon_temperature = 0.005;
        let system = ThreeTemperatureFlrwSystem::new(FiniteTemperatureQed::Off);
        let final_event_fn = |_ln_a: f64, y: &[f64]| y[0] - final_photon_temperature;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let config = OdeConfig {
            rtol: 2.0e-10,
            atol: vec![2.0e-13; 3],
            h_init: 1.0e-5,
            h_min: 1.0e-14,
            h_max: 0.025,
            max_attempts: 100_000,
        };

        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &system,
                (0.0, 12.0),
                &[initial_temperature; 3],
                &config,
                Some(&final_event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!(
                (result.y[0] - final_photon_temperature).abs() < 1.0e-10,
                "{kind:?}: {result:?}"
            );
            assert!(
                (result.t - EXPECTED_LN_A).abs() < 5.0e-7,
                "{kind:?}: {result:?}"
            );
            assert!(
                (result.y[1] - EXPECTED_TNUE_MEV).abs() < 5.0e-10,
                "{kind:?}: {result:?}"
            );
            assert!(
                (result.y[2] - EXPECTED_TNUX_MEV).abs() < 5.0e-10,
                "{kind:?}: {result:?}"
            );
            assert!(result.y[1] > result.y[2], "{kind:?}: {result:?}");
            let instantaneous_ratio = (4.0 / 11.0_f64).powf(4.0 / 3.0);
            let neff_thermal = ((result.y[1] / result.y[0]).powi(4)
                + 2.0 * (result.y[2] / result.y[0]).powi(4))
                / instantaneous_ratio;
            assert!(
                (neff_thermal - EXPECTED_NEFF_THERMAL).abs() < 2.0e-6,
                "{kind:?}: N_eff={neff_thermal:.17e}, result={result:?}"
            );
            endpoints.push((result.t, result.y));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 5.0e-7);
        assert!((endpoints[0].1[1] - endpoints[1].1[1]).abs() < 5.0e-10);
        assert!((endpoints[0].1[2] - endpoints[1].1[2]).abs() < 5.0e-10);
    }

    #[test]
    fn invalid_inputs_are_returned_without_clipping() {
        let nan_error = flrw_thermo_state(f64::NAN, 2.0).unwrap_err();
        match nan_error {
            FlrwThermoError::InvalidInput { quantity, value } => {
                assert_eq!(quantity, "T_gamma");
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(
            flrw_thermo_state(-0.1, 2.0).unwrap_err(),
            FlrwThermoError::InvalidInput {
                quantity: "T_gamma",
                value: -0.1
            }
        );
        assert_eq!(
            electron_pair_eos_with_quadrature(1.0, 63, 48.0).unwrap_err(),
            FlrwThermoError::InvalidPanelCount(63)
        );
    }

    #[test]
    fn both_solvers_reach_the_ideal_flrw_temperature_event() {
        let initial_temperature = 10.0;
        let decoupling_temperature = 2.0;
        let final_temperature = 0.005;
        let common_system = IdealFlrwSystem::common_bath_leg(decoupling_temperature);
        let electromagnetic_system =
            IdealFlrwSystem::electromagnetic_bath_leg(decoupling_temperature);
        let decoupling_event_fn = |_ln_a: f64, y: &[f64]| y[0] - decoupling_temperature;
        let decoupling_event = TerminalEvent {
            value: &decoupling_event_fn,
            direction: -1,
        };
        let final_event_fn = |_ln_a: f64, y: &[f64]| y[0] - final_temperature;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let config = OdeConfig {
            rtol: 2.0e-8,
            atol: vec![1.0e-11],
            h_init: 1.0e-4,
            h_min: 1.0e-13,
            h_max: 0.05,
            max_attempts: 20_000,
        };

        let initial = flrw_thermo_state(initial_temperature, decoupling_temperature).unwrap();
        let at_decoupling =
            flrw_thermo_state(decoupling_temperature, decoupling_temperature).unwrap();
        let final_state = flrw_thermo_state(final_temperature, decoupling_temperature).unwrap();
        let initial_common_entropy = initial.electromagnetic.entropy
            + 4.0 * initial.rho_neutrino / (3.0 * initial_temperature);
        let decoupling_common_entropy = at_decoupling.electromagnetic.entropy
            + 4.0 * at_decoupling.rho_neutrino / (3.0 * decoupling_temperature);
        let expected_pre_decoupling_ln_a =
            (initial_common_entropy / decoupling_common_entropy).ln() / 3.0;
        let expected_post_decoupling_delta_ln_a =
            (at_decoupling.electromagnetic.entropy / final_state.electromagnetic.entropy).ln()
                / 3.0;
        let expected_final_ln_a =
            expected_pre_decoupling_ln_a + expected_post_decoupling_delta_ln_a;

        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let at_branch = solve(
                kind,
                &common_system,
                (0.0, 5.0),
                &[initial_temperature],
                &config,
                Some(&decoupling_event),
            );
            assert_eq!(at_branch.failure, None, "{kind:?}: {at_branch:?}");
            assert!(at_branch.event_reached, "{kind:?}: {at_branch:?}");
            assert!((at_branch.y[0] - decoupling_temperature).abs() < 1.0e-9);
            assert!(
                (at_branch.t - expected_pre_decoupling_ln_a).abs() < 1.0e-7,
                "{kind:?}: {at_branch:?}"
            );

            let result = solve(
                kind,
                &electromagnetic_system,
                (at_branch.t, 12.0),
                &[decoupling_temperature],
                &config,
                Some(&final_event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!((result.y[0] - final_temperature).abs() < 1.0e-9);
            assert!(
                ((result.t - at_branch.t) - expected_post_decoupling_delta_ln_a).abs() < 1.0e-7,
                "{kind:?}: {result:?}"
            );
            assert!(
                (result.t - expected_final_ln_a).abs() < 2.0e-7,
                "{kind:?}: {result:?}"
            );
            endpoints.push(result.t);
        }
        assert!((endpoints[0] - endpoints[1]).abs() < 2.0e-7);
    }
}
