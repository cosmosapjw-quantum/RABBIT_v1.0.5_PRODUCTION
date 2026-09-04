//! First analytic `T_gamma`-tangent primitives for D-081R1F1.
//!
//! This module is deliberately limited to the moving incoming-electron
//! half-line quadrature and the QED-off electromagnetic equation of state.
//! It does not assemble collision, packed-RHS, solver, or trajectory tangents.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_kinematics::electron_half_line_rule;
use crate::flrw::{ELECTRON_MASS_MEV, ElectromagneticEos, electromagnetic_eos};

const DISCRETE_D2_RHO_SIMPSON_PANELS: usize = 256;
const CONTINUUM_D2_RHO_SIMPSON_PANELS: usize = 4096;
const D2_RHO_TAIL_E_FOLDS: f64 = 48.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum F10TgammaTangentError {
    InvalidInput,
    Foundation,
    Thermodynamics,
    NonFiniteOutput,
}

#[derive(Clone, Debug)]
pub(crate) struct F10ElectronHalfLineTgammaTangent {
    pub(crate) momentum: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    pub(crate) d_momentum_dt: Vec<f64>,
    pub(crate) d_weights_dt: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10ElectromagneticEosTgammaTangent {
    pub(crate) base: ElectromagneticEos,
    pub(crate) d_rho: f64,
    pub(crate) d_pressure: f64,
    pub(crate) d2_rho: f64,
}

pub(crate) fn electron_half_line_tgamma_tangent(
    order: usize,
    temperature_gamma_mev: f64,
) -> Result<F10ElectronHalfLineTgammaTangent, F10TgammaTangentError> {
    if order < 2 || !temperature_gamma_mev.is_finite() || temperature_gamma_mev <= 0.0 {
        return Err(F10TgammaTangentError::InvalidInput);
    }
    let (momentum, weights) = electron_half_line_rule(order, temperature_gamma_mev)
        .map_err(|_| F10TgammaTangentError::Foundation)?;
    if momentum.len() != order
        || weights.len() != order
        || momentum
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    let d_momentum_dt = momentum
        .iter()
        .map(|value| value / temperature_gamma_mev)
        .collect::<Vec<_>>();
    let d_weights_dt = weights
        .iter()
        .map(|value| value / temperature_gamma_mev)
        .collect::<Vec<_>>();
    if d_momentum_dt
        .iter()
        .chain(&d_weights_dt)
        .any(|value| !value.is_finite())
    {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    Ok(F10ElectronHalfLineTgammaTangent {
        momentum,
        weights,
        d_momentum_dt,
        d_weights_dt,
    })
}

pub(crate) fn electromagnetic_eos_tgamma_continuum_d2rho_dt2_reference(
    temperature_gamma_mev: f64,
) -> Result<f64, F10TgammaTangentError> {
    let ratio = ELECTRON_MASS_MEV / temperature_gamma_mev;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    let theta_max = (1.0 + D2_RHO_TAIL_E_FOLDS / ratio).acosh();
    if !theta_max.is_finite() || theta_max <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    let step = theta_max / CONTINUUM_D2_RHO_SIMPSON_PANELS as f64;
    let mut sum = 0.0_f64;
    for index in 0..=CONTINUUM_D2_RHO_SIMPSON_PANELS {
        let theta = index as f64 * step;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let momentum_over_temperature = ratio * sinh;
        let energy_over_temperature = ratio * cosh;
        let exponential_negative = (-energy_over_temperature).exp();
        let occupation = exponential_negative / (1.0 + exponential_negative);
        let blocking = occupation * (1.0 - occupation);
        let bracket = 3.0 * energy_over_temperature.powi(2)
            + ratio.powi(2) * (-2.0 + energy_over_temperature * (1.0 - 2.0 * occupation));
        let integrand =
            momentum_over_temperature.powi(2) * blocking * bracket * energy_over_temperature;
        if !integrand.is_finite() {
            return Err(F10TgammaTangentError::NonFiniteOutput);
        }
        let weight = if index == 0 || index == CONTINUUM_D2_RHO_SIMPSON_PANELS {
            1.0
        } else if index.is_multiple_of(2) {
            2.0
        } else {
            4.0
        };
        sum += weight * integrand;
    }
    let dimensionless_integral = sum * step / 3.0;
    let temperature_squared = temperature_gamma_mev.powi(2);
    let photon = 4.0 * PI.powi(2) * temperature_squared / 5.0;
    let electron = 2.0 * temperature_squared * dimensionless_integral / PI.powi(2);
    let result = photon + electron;
    if !result.is_finite() || result <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    Ok(result)
}

pub(crate) fn electromagnetic_eos_tgamma_discrete_d2rho_dt2(
    temperature_gamma_mev: f64,
) -> Result<f64, F10TgammaTangentError> {
    if !temperature_gamma_mev.is_finite() || temperature_gamma_mev <= 0.0 {
        return Err(F10TgammaTangentError::InvalidInput);
    }
    let ratio = ELECTRON_MASS_MEV / temperature_gamma_mev;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    let endpoint_argument = 1.0 + D2_RHO_TAIL_E_FOLDS / ratio;
    let theta_max = endpoint_argument.acosh();
    let endpoint_denominator = (endpoint_argument.powi(2) - 1.0).sqrt();
    let d_theta_max_dt = (D2_RHO_TAIL_E_FOLDS / ELECTRON_MASS_MEV) / endpoint_denominator;
    if !theta_max.is_finite()
        || theta_max <= 0.0
        || !d_theta_max_dt.is_finite()
        || d_theta_max_dt <= 0.0
    {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }

    let step = theta_max / DISCRETE_D2_RHO_SIMPSON_PANELS as f64;
    let d_step_dt = d_theta_max_dt / DISCRETE_D2_RHO_SIMPSON_PANELS as f64;
    let d_ratio_dt = -ratio / temperature_gamma_mev;
    let ratio_fourth = ratio.powi(4);
    let ratio_fifth = ratio_fourth * ratio;
    let mut weighted_sum = 0.0_f64;
    let mut weighted_tangent_sum = 0.0_f64;

    for index in 0..=DISCRETE_D2_RHO_SIMPSON_PANELS {
        let theta = index as f64 * step;
        let d_theta_dt = index as f64 * d_step_dt;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let epsilon = ratio * cosh;
        let d_epsilon_dt = d_ratio_dt * cosh + ratio * sinh * d_theta_dt;
        let exponential_negative = (-epsilon).exp();
        let occupation = exponential_negative / (1.0 + exponential_negative);
        let blocking = occupation * (1.0 - occupation);
        let d_blocking_dt = -blocking * (1.0 - 2.0 * occupation) * d_epsilon_dt;
        let geometric = sinh.powi(2) * cosh.powi(3);
        let d_geometric_dtheta = 2.0 * sinh * cosh.powi(4) + 3.0 * sinh.powi(3) * cosh.powi(2);
        let value = ratio_fifth * geometric * blocking;
        let tangent = 5.0 * ratio_fourth * d_ratio_dt * geometric * blocking
            + ratio_fifth * d_geometric_dtheta * d_theta_dt * blocking
            + ratio_fifth * geometric * d_blocking_dt;
        if !value.is_finite() || !tangent.is_finite() {
            return Err(F10TgammaTangentError::NonFiniteOutput);
        }
        let weight = if index == 0 || index == DISCRETE_D2_RHO_SIMPSON_PANELS {
            1.0
        } else if index.is_multiple_of(2) {
            2.0
        } else {
            4.0
        };
        weighted_sum += weight * value;
        weighted_tangent_sum += weight * tangent;
    }

    let dimensionless_sum = weighted_sum * step / 3.0;
    let d_dimensionless_sum_dt = d_step_dt * weighted_sum / 3.0 + step * weighted_tangent_sum / 3.0;
    let electron = 2.0
        * (3.0 * temperature_gamma_mev.powi(2) * dimensionless_sum
            + temperature_gamma_mev.powi(3) * d_dimensionless_sum_dt)
        / PI.powi(2);
    let photon = 4.0 * PI.powi(2) * temperature_gamma_mev.powi(2) / 5.0;
    let result = photon + electron;
    if !result.is_finite() || result <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    Ok(result)
}

pub(crate) fn electromagnetic_eos_tgamma_tangent(
    temperature_gamma_mev: f64,
) -> Result<F10ElectromagneticEosTgammaTangent, F10TgammaTangentError> {
    if !temperature_gamma_mev.is_finite() || temperature_gamma_mev <= 0.0 {
        return Err(F10TgammaTangentError::InvalidInput);
    }
    let base = electromagnetic_eos(temperature_gamma_mev)
        .map_err(|_| F10TgammaTangentError::Thermodynamics)?;
    let d_rho = base.drho_dt;
    let d_pressure = (base.rho + base.pressure) / temperature_gamma_mev;
    let d2_rho = electromagnetic_eos_tgamma_discrete_d2rho_dt2(temperature_gamma_mev)?;
    if !d_rho.is_finite()
        || d_rho <= 0.0
        || !d_pressure.is_finite()
        || d_pressure <= 0.0
        || !d2_rho.is_finite()
        || d2_rho <= 0.0
    {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    Ok(F10ElectromagneticEosTgammaTangent {
        base,
        d_rho,
        d_pressure,
        d2_rho,
    })
}
