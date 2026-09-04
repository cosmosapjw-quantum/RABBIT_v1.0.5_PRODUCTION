//! Combined analytic spectral-`c` JVP for D-081R1F0.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_tangent::F10SpectralTangent;
use crate::f10_combined_action::{
    F10CombinedActionConfig, F10CombinedActionError, F10CombinedActionMoments,
};
use crate::f10_electron_action::F10ElectronActionConfig;
use crate::f10_electron_action::c_jvp::{F10ElectronActionJvp, assemble_electron_action_c_jvp};
use crate::f10_self_action::F10SelfActionConfig;
use crate::f10_self_action::c_jvp::{F10SelfActionJvp, assemble_self_action_c_jvp};

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;
const TWO_PI_SQUARED: f64 = 2.0 * PI * PI;

#[derive(Clone, Debug)]
pub(crate) struct F10CombinedActionJvp {
    pub(crate) self_action: F10SelfActionJvp,
    pub(crate) electron_action: F10ElectronActionJvp,
    pub(crate) modal_total: Vec<f64>,
    pub(crate) native_total: Vec<f64>,
    pub(crate) moments: F10CombinedActionMoments,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) self_event_energy_residual: f64,
    pub(crate) self_event_energy_relative_residual: f64,
    pub(crate) charge_conjugation_residual: f64,
    pub(crate) mu_tau_residual: f64,
}

fn validate_inputs(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    tangent: &F10SpectralTangent,
    config: F10CombinedActionConfig,
) -> Result<(), F10CombinedActionError> {
    let spectral_size = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    if grid.order == 0
        || grid.nodes.len() != grid.order
        || grid.weights.len() != grid.order
        || !grid.y_max.is_finite()
        || grid.y_max <= 0.0
        || pair_cloglog.len() != spectral_size
        || tangent.logit_delta.len() != spectral_size
        || tangent.logit_modal.len() != spectral_size
        || !temperature_cm.is_finite()
        || temperature_cm <= 0.0
        || !temperature_gamma.is_finite()
        || temperature_gamma <= 0.0
    {
        return Err(F10CombinedActionError::InvalidInput);
    }
    if !config.matrix_roundoff_ulps.is_finite()
        || config.matrix_roundoff_ulps <= 0.0
        || !config.electron_mass_mev.is_finite()
        || config.electron_mass_mev <= 0.0
    {
        return Err(F10CombinedActionError::InvalidConfiguration);
    }
    Ok(())
}

fn add_arrays(left: &[f64], right: &[f64]) -> Result<Vec<f64>, F10CombinedActionError> {
    if left.len() != right.len() {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let result: Vec<f64> = left
        .iter()
        .zip(right)
        .map(|(&lhs, &rhs)| lhs + rhs)
        .collect();
    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

fn action_moments(
    grid: &F10ActionGrid,
    native: &[f64],
    temperature_cm: f64,
) -> Result<F10CombinedActionMoments, F10CombinedActionError> {
    let expected = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    if native.len() != expected {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let mut signed_number = 0.0_f64;
    let mut absolute_number = 0.0_f64;
    let mut signed_energy = 0.0_f64;
    let mut absolute_energy = 0.0_f64;
    for species in 0..SPECIES_COUNT {
        for node in 0..grid.order {
            let value = native[species * grid.order + node];
            let number_weight = grid.weights[node] * grid.nodes[node].powi(2);
            let energy_weight = number_weight * grid.nodes[node];
            signed_number += value * number_weight;
            absolute_number += value.abs() * number_weight;
            signed_energy += value * energy_weight;
            absolute_energy += value.abs() * energy_weight;
        }
    }
    let moments = F10CombinedActionMoments {
        signed_number_rate: temperature_cm.powi(3) * signed_number / TWO_PI_SQUARED,
        absolute_number_rate: temperature_cm.powi(3) * absolute_number / TWO_PI_SQUARED,
        signed_energy_rate: temperature_cm.powi(4) * signed_energy / TWO_PI_SQUARED,
        absolute_energy_rate: temperature_cm.powi(4) * absolute_energy / TWO_PI_SQUARED,
    };
    if [
        moments.signed_number_rate,
        moments.absolute_number_rate,
        moments.signed_energy_rate,
        moments.absolute_energy_rate,
    ]
    .into_iter()
    .all(f64::is_finite)
    {
        Ok(moments)
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

fn relative_max_difference(left: &[f64], right: &[f64]) -> Result<f64, F10CombinedActionError> {
    if left.is_empty() || left.len() != right.len() {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let scale = left
        .iter()
        .chain(right)
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let difference = left
        .iter()
        .zip(right)
        .map(|(&lhs, &rhs)| (lhs - rhs).abs())
        .fold(0.0_f64, f64::max);
    let residual = difference / scale;
    if residual.is_finite() {
        Ok(residual)
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

fn symmetry_residuals(
    native_total: &[f64],
    order: usize,
) -> Result<(f64, f64), F10CombinedActionError> {
    if order == 0 || native_total.len() != SPECIES_COUNT * order {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let mut charge_conjugation = 0.0_f64;
    for pair in 0..PAIR_COUNT {
        let particle_start = 2 * pair * order;
        let antiparticle_start = particle_start + order;
        charge_conjugation = charge_conjugation.max(relative_max_difference(
            &native_total[particle_start..particle_start + order],
            &native_total[antiparticle_start..antiparticle_start + order],
        )?);
    }
    let mut mu_pair = Vec::with_capacity(order);
    let mut tau_pair = Vec::with_capacity(order);
    for node in 0..order {
        mu_pair.push(0.5 * (native_total[2 * order + node] + native_total[3 * order + node]));
        tau_pair.push(0.5 * (native_total[4 * order + node] + native_total[5 * order + node]));
    }
    Ok((
        charge_conjugation,
        relative_max_difference(&mu_pair, &tau_pair)?,
    ))
}

pub(crate) fn assemble_combined_action_c_jvp(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    tangent: &F10SpectralTangent,
    config: F10CombinedActionConfig,
) -> Result<F10CombinedActionJvp, F10CombinedActionError> {
    validate_inputs(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
        tangent,
        config,
    )?;

    let self_action = assemble_self_action_c_jvp(
        grid,
        pair_cloglog,
        temperature_cm,
        tangent,
        F10SelfActionConfig {
            collision: config.collision,
            matrix_roundoff_ulps: config.matrix_roundoff_ulps,
        },
    )
    .map_err(|_| F10CombinedActionError::SelfAction)?;
    let electron_action = assemble_electron_action_c_jvp(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
        tangent,
        F10ElectronActionConfig {
            collision: config.collision,
            matrix_roundoff_ulps: config.matrix_roundoff_ulps,
            electron_mass_mev: config.electron_mass_mev,
        },
    )
    .map_err(|_| F10CombinedActionError::ElectronAction)?;

    let modal_total = add_arrays(&self_action.modal, &electron_action.modal)?;
    let native_total = add_arrays(&self_action.native, &electron_action.native)?;
    let moments = action_moments(grid, &native_total, temperature_cm)?;
    let neutrino_energy_transfer = electron_action.neutrino_energy_transfer;
    let electromagnetic_energy_transfer = electron_action.electromagnetic_energy_transfer;
    let first_law_scale = (neutrino_energy_transfer.abs() + electromagnetic_energy_transfer.abs())
        .max(f64::MIN_POSITIVE);
    let first_law_residual =
        (neutrino_energy_transfer + electromagnetic_energy_transfer).abs() / first_law_scale;
    let self_event_energy_residual = self_action.event_energy_residual;
    let self_event_energy_relative_residual =
        self_event_energy_residual.abs() / self_action.event_energy_absolute.max(f64::MIN_POSITIVE);
    let (charge_conjugation_residual, mu_tau_residual) =
        symmetry_residuals(&native_total, grid.order)?;

    let scalars = [
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        self_event_energy_residual,
        self_event_energy_relative_residual,
        charge_conjugation_residual,
        mu_tau_residual,
    ];
    if scalars.into_iter().any(|value| !value.is_finite()) {
        return Err(F10CombinedActionError::NonFiniteOutput);
    }

    Ok(F10CombinedActionJvp {
        self_action,
        electron_action,
        modal_total,
        native_total,
        moments,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        self_event_energy_residual,
        self_event_energy_relative_residual,
        charge_conjugation_residual,
        mu_tau_residual,
    })
}
