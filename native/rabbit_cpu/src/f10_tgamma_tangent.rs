//! First analytic `T_gamma`-tangent primitives for D-081R1F1.
//!
//! This module is deliberately limited to the moving incoming-electron
//! half-line quadrature and the QED-off electromagnetic equation of state.
//! It does not assemble collision, packed-RHS, solver, or trajectory tangents.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_kinematics::electron_half_line_rule;
use crate::flrw::{
    ELECTRON_MASS_MEV, ElectromagneticEos, electromagnetic_eos,
};

const D2_RHO_SIMPSON_PANELS: usize = 4096;
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
        || momentum.iter().any(|value| !value.is_finite() || *value < 0.0)
        || weights.iter().any(|value| !value.is_finite() || *value <= 0.0)
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

fn electron_pair_d2rho_dt2(
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
    let step = theta_max / D2_RHO_SIMPSON_PANELS as f64;
    let mut sum = 0.0_f64;
    for index in 0..=D2_RHO_SIMPSON_PANELS {
        let theta = index as f64 * step;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let momentum_over_temperature = ratio * sinh;
        let energy_over_temperature = ratio * cosh;
        let exponential_negative = (-energy_over_temperature).exp();
        let occupation = exponential_negative / (1.0 + exponential_negative);
        let blocking = occupation * (1.0 - occupation);
        let bracket = 3.0 * energy_over_temperature.powi(2)
            + ratio.powi(2)
                * (-2.0 + energy_over_temperature * (1.0 - 2.0 * occupation));
        let integrand = momentum_over_temperature.powi(2)
            * blocking
            * bracket
            * energy_over_temperature;
        if !integrand.is_finite() {
            return Err(F10TgammaTangentError::NonFiniteOutput);
        }
        let weight = if index == 0 || index == D2_RHO_SIMPSON_PANELS {
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
    let d2_rho = electron_pair_d2rho_dt2(temperature_gamma_mev)?;
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
