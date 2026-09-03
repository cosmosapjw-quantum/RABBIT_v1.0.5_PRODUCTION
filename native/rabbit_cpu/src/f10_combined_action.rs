//! Combined self-plus-electron collision action for the frozen D-081R1 comparator.
//!
//! This layer performs no collision quadrature of its own. It composes the
//! separately admitted six-species neutrino self action and finite-electron-
//! mass action at the same state, then reconstructs the frozen comparator's
//! total action and cross-sector diagnostics. Packed RHS construction,
//! derivatives, and ODE integration remain outside this node.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::{F10ActionGrid, decode_cloglog_to_logit};
use crate::f10_action_kinematics::F10CollisionConfig;
use crate::f10_action_spectral::modal_coefficients;
use crate::f10_electron_action::{
    F10_ELECTRON_MASS_MEV, F10ElectronAction, F10ElectronActionConfig,
    assemble_electron_action,
};
use crate::f10_self_action::{F10SelfAction, F10SelfActionConfig, assemble_self_action};

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;
const TWO_PI_SQUARED: f64 = 2.0 * PI * PI;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10CombinedActionConfig {
    pub(crate) collision: F10CollisionConfig,
    pub(crate) matrix_roundoff_ulps: f64,
    pub(crate) electron_mass_mev: f64,
}

impl Default for F10CombinedActionConfig {
    fn default() -> Self {
        Self {
            collision: F10CollisionConfig::default(),
            matrix_roundoff_ulps: 1024.0,
            electron_mass_mev: F10_ELECTRON_MASS_MEV,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10CombinedActionMoments {
    pub(crate) signed_number_rate: f64,
    pub(crate) absolute_number_rate: f64,
    pub(crate) signed_energy_rate: f64,
    pub(crate) absolute_energy_rate: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct F10CombinedAction {
    pub(crate) self_action: F10SelfAction,
    pub(crate) electron_action: F10ElectronAction,
    pub(crate) modal_total: Vec<f64>,
    pub(crate) native_total: Vec<f64>,
    pub(crate) moments: F10CombinedActionMoments,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) event_neutrino_h_rate: f64,
    pub(crate) node_neutrino_h_rate: f64,
    pub(crate) electromagnetic_h_rate: f64,
    pub(crate) entropy_production: f64,
    pub(crate) entropy_duality_residual: f64,
    pub(crate) self_event_energy_residual: f64,
    pub(crate) self_event_energy_absolute: f64,
    pub(crate) charge_conjugation_residual: f64,
    pub(crate) mu_tau_residual: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10CombinedActionError {
    InvalidInput,
    InvalidConfiguration,
    SelfAction,
    ElectronAction,
    DimensionOverflow,
    Foundation,
    NonFiniteOutput,
}

fn validate_inputs(
    grid: &F10ActionGrid,
    temperature_cm: f64,
    temperature_gamma: f64,
    config: F10CombinedActionConfig,
) -> Result<(), F10CombinedActionError> {
    if grid.order == 0
        || grid.nodes.len() != grid.order
        || grid.weights.len() != grid.order
        || !grid.y_max.is_finite()
        || grid.y_max <= 0.0
        || grid
            .nodes
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= grid.y_max)
        || grid
            .weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
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
    let output: Vec<f64> = left
        .iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value + right_value)
        .collect();
    if output.iter().all(|value| value.is_finite()) {
        Ok(output)
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

fn pair_modal_coefficients(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
) -> Result<Vec<f64>, F10CombinedActionError> {
    let expected = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    if pair_cloglog.len() != expected {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let mut coefficients = Vec::with_capacity(expected);
    for pair in 0..PAIR_COUNT {
        let start = pair * grid.order;
        let logits: Vec<f64> = pair_cloglog[start..start + grid.order]
            .iter()
            .map(|&coordinate| {
                decode_cloglog_to_logit(coordinate)
                    .map_err(|_| F10CombinedActionError::Foundation)
            })
            .collect::<Result<_, _>>()?;
        coefficients.extend(
            modal_coefficients(grid, &logits)
                .map_err(|_| F10CombinedActionError::Foundation)?,
        );
    }
    Ok(coefficients)
}

fn node_neutrino_h_rate(
    grid: &F10ActionGrid,
    modal_total: &[f64],
    pair_cloglog: &[f64],
) -> Result<f64, F10CombinedActionError> {
    let expected_modal = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    if modal_total.len() != expected_modal {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let coefficients = pair_modal_coefficients(grid, pair_cloglog)?;
    let mut result = 0.0_f64;
    for species in 0..SPECIES_COUNT {
        let pair = species / 2;
        for mode in 0..grid.order {
            result += modal_total[species * grid.order + mode]
                * coefficients[pair * grid.order + mode];
        }
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

fn symmetry_residuals(native_total: &[f64], order: usize) -> Result<(f64, f64), F10CombinedActionError> {
    if order == 0 || native_total.len() != SPECIES_COUNT * order {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let scale = native_total
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let mut cp_absolute = 0.0_f64;
    for pair in 0..PAIR_COUNT {
        for node in 0..order {
            cp_absolute = cp_absolute.max(
                (native_total[(2 * pair) * order + node]
                    - native_total[(2 * pair + 1) * order + node])
                    .abs(),
            );
        }
    }
    let mut mu_tau_absolute = 0.0_f64;
    for node in 0..order {
        let mu = 0.5 * (native_total[2 * order + node] + native_total[3 * order + node]);
        let tau = 0.5 * (native_total[4 * order + node] + native_total[5 * order + node]);
        mu_tau_absolute = mu_tau_absolute.max((mu - tau).abs());
    }
    let cp = cp_absolute / scale;
    let mu_tau = mu_tau_absolute / scale;
    if cp.is_finite() && mu_tau.is_finite() {
        Ok((cp, mu_tau))
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

pub(crate) fn assemble_combined_action(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    config: F10CombinedActionConfig,
) -> Result<F10CombinedAction, F10CombinedActionError> {
    validate_inputs(grid, temperature_cm, temperature_gamma, config)?;

    let self_action = assemble_self_action(
        grid,
        pair_cloglog,
        temperature_cm,
        F10SelfActionConfig {
            collision: config.collision,
            matrix_roundoff_ulps: config.matrix_roundoff_ulps,
        },
    )
    .map_err(|_| F10CombinedActionError::SelfAction)?;

    let electron_action = assemble_electron_action(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
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

    let whole_reaction_domain_rejections = self_action
        .whole_reaction_domain_rejections
        .checked_add(electron_action.whole_reaction_domain_rejections)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    let matrix_roundoff_corrections = self_action
        .matrix_roundoff_corrections
        .checked_add(electron_action.matrix_roundoff_corrections)
        .ok_or(F10CombinedActionError::DimensionOverflow)?;
    let largest_matrix_roundoff_correction = self_action
        .largest_matrix_roundoff_correction
        .max(electron_action.largest_matrix_roundoff_correction);

    let neutrino_energy_transfer = electron_action.neutrino_energy_transfer;
    let electromagnetic_energy_transfer = electron_action.electromagnetic_energy_transfer;
    let first_law_residual = (neutrino_energy_transfer + electromagnetic_energy_transfer).abs()
        / (neutrino_energy_transfer.abs() + electromagnetic_energy_transfer.abs())
            .max(f64::MIN_POSITIVE);

    let event_neutrino_h_rate = self_action.event_entropy_rate + electron_action.neutrino_h_rate;
    let node_neutrino_h_rate = node_neutrino_h_rate(grid, &modal_total, pair_cloglog)?;
    let electromagnetic_h_rate = electron_action.electromagnetic_h_rate;
    let entropy_production = -(event_neutrino_h_rate + electromagnetic_h_rate);
    let entropy_duality_residual = (node_neutrino_h_rate - event_neutrino_h_rate).abs()
        / (node_neutrino_h_rate.abs() + event_neutrino_h_rate.abs()).max(f64::MIN_POSITIVE);
    let self_event_energy_residual = self_action.event_energy_residual;
    let self_event_energy_absolute = self_action.event_energy_absolute;
    let (charge_conjugation_residual, mu_tau_residual) =
        symmetry_residuals(&native_total, grid.order)?;

    let scalars = [
        largest_matrix_roundoff_correction,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        event_neutrino_h_rate,
        node_neutrino_h_rate,
        electromagnetic_h_rate,
        entropy_production,
        entropy_duality_residual,
        self_event_energy_residual,
        self_event_energy_absolute,
        charge_conjugation_residual,
        mu_tau_residual,
    ];
    if scalars.into_iter().any(|value| !value.is_finite()) {
        return Err(F10CombinedActionError::NonFiniteOutput);
    }

    Ok(F10CombinedAction {
        self_action,
        electron_action,
        modal_total,
        native_total,
        moments,
        whole_reaction_domain_rejections,
        matrix_roundoff_corrections,
        largest_matrix_roundoff_correction,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        event_neutrino_h_rate,
        node_neutrino_h_rate,
        electromagnetic_h_rate,
        entropy_production,
        entropy_duality_residual,
        self_event_energy_residual,
        self_event_energy_absolute,
        charge_conjugation_residual,
        mu_tau_residual,
    })
}

#[cfg(test)]
impl PartialEq for F10CombinedAction {
    fn eq(&self, other: &Self) -> bool {
        self.self_action.modal == other.self_action.modal
            && self.self_action.native == other.self_action.native
            && self.self_action.row_modal == other.self_action.row_modal
            && self.self_action.row_native == other.self_action.row_native
            && self.self_action.moments.signed_number_rate
                == other.self_action.moments.signed_number_rate
            && self.self_action.moments.absolute_number_rate
                == other.self_action.moments.absolute_number_rate
            && self.self_action.moments.signed_energy_rate
                == other.self_action.moments.signed_energy_rate
            && self.self_action.moments.absolute_energy_rate
                == other.self_action.moments.absolute_energy_rate
            && self.self_action.whole_reaction_domain_rejections
                == other.self_action.whole_reaction_domain_rejections
            && self.self_action.matrix_roundoff_corrections
                == other.self_action.matrix_roundoff_corrections
            && self.self_action.largest_matrix_roundoff_correction
                == other.self_action.largest_matrix_roundoff_correction
            && self.self_action.event_entropy_rate == other.self_action.event_entropy_rate
            && self.self_action.node_entropy_rate == other.self_action.node_entropy_rate
            && self.self_action.entropy_duality_residual
                == other.self_action.entropy_duality_residual
            && self.self_action.event_energy_residual
                == other.self_action.event_energy_residual
            && self.self_action.event_energy_absolute
                == other.self_action.event_energy_absolute
            && self.electron_action == other.electron_action
            && self.modal_total == other.modal_total
            && self.native_total == other.native_total
            && self.moments == other.moments
            && self.whole_reaction_domain_rejections
                == other.whole_reaction_domain_rejections
            && self.matrix_roundoff_corrections == other.matrix_roundoff_corrections
            && self.largest_matrix_roundoff_correction
                == other.largest_matrix_roundoff_correction
            && self.neutrino_energy_transfer == other.neutrino_energy_transfer
            && self.electromagnetic_energy_transfer
                == other.electromagnetic_energy_transfer
            && self.first_law_residual == other.first_law_residual
            && self.event_neutrino_h_rate == other.event_neutrino_h_rate
            && self.node_neutrino_h_rate == other.node_neutrino_h_rate
            && self.electromagnetic_h_rate == other.electromagnetic_h_rate
            && self.entropy_production == other.entropy_production
            && self.entropy_duality_residual == other.entropy_duality_residual
            && self.self_event_energy_residual == other.self_event_energy_residual
            && self.self_event_energy_absolute == other.self_event_energy_absolute
            && self.charge_conjugation_residual == other.charge_conjugation_residual
            && self.mu_tau_residual == other.mu_tau_residual
    }
}
