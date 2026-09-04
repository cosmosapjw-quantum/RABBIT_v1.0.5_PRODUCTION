//! Static analytic spectral-`c` JVP for the admitted packed F10 RHS.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_tangent::{F10ActionTangentError, F10SpectralTangent};
use crate::f10_combined_action::F10CombinedActionMoments;
use crate::f10_combined_action_jvp::{F10CombinedActionJvp, assemble_combined_action_c_jvp};
use crate::f10_electron_action::F10ElectronActionMoments;
use crate::f10_electron_action::c_jvp::F10ElectronActionJvp;
use crate::f10_packed_rhs::{
    F10PackedRhs, F10PackedRhsConfig, F10PackedRhsError, evaluate_f10_packed_rhs,
};
use crate::f10_self_action::F10ActionMoments;
use crate::f10_self_action::c_jvp::F10SelfActionJvp;

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;

#[derive(Clone, Debug)]
pub(crate) struct F10PackedRhsJvp {
    pub(crate) base: F10PackedRhs,
    pub(crate) values: Vec<f64>,
    pub(crate) combined_action: F10CombinedActionJvp,
    pub(crate) delta_rho_neutrino_by_flavour: [f64; 3],
    pub(crate) delta_rho_neutrino: f64,
    pub(crate) delta_hubble_over_hubble: f64,
    pub(crate) delta_neutrino_energy_transfer: f64,
    pub(crate) delta_electromagnetic_energy_transfer: f64,
    pub(crate) first_law_tangent_residual: f64,
}

fn map_tangent_error(error: F10ActionTangentError) -> F10PackedRhsError {
    match error {
        F10ActionTangentError::InvalidInput => F10PackedRhsError::InvalidInput,
        F10ActionTangentError::DimensionOverflow => F10PackedRhsError::DimensionOverflow,
        F10ActionTangentError::Foundation => F10PackedRhsError::Chart,
        F10ActionTangentError::NonFiniteOutput => F10PackedRhsError::NonFiniteOutput,
    }
}

fn zero_combined_action_jvp(order: usize) -> Result<F10CombinedActionJvp, F10PackedRhsError> {
    let action_size = SPECIES_COUNT
        .checked_mul(order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    let self_action = F10SelfActionJvp {
        modal: vec![0.0; action_size],
        native: vec![0.0; action_size],
        moments: F10ActionMoments {
            signed_number_rate: 0.0,
            absolute_number_rate: 0.0,
            signed_energy_rate: 0.0,
            absolute_energy_rate: 0.0,
        },
        event_energy_residual: 0.0,
        event_energy_absolute: 0.0,
    };
    let electron_action = F10ElectronActionJvp {
        modal: vec![0.0; action_size],
        native: vec![0.0; action_size],
        moments: F10ElectronActionMoments {
            signed_number_rate: 0.0,
            absolute_number_rate: 0.0,
            signed_energy_rate: 0.0,
            absolute_energy_rate: 0.0,
        },
        neutrino_energy_transfer: 0.0,
        electromagnetic_energy_transfer: 0.0,
        first_law_residual: 0.0,
    };
    Ok(F10CombinedActionJvp {
        self_action,
        electron_action,
        modal_total: vec![0.0; action_size],
        native_total: vec![0.0; action_size],
        moments: F10CombinedActionMoments {
            signed_number_rate: 0.0,
            absolute_number_rate: 0.0,
            signed_energy_rate: 0.0,
            absolute_energy_rate: 0.0,
        },
        neutrino_energy_transfer: 0.0,
        electromagnetic_energy_transfer: 0.0,
        first_law_residual: 0.0,
        self_event_energy_residual: 0.0,
        self_event_energy_relative_residual: 0.0,
        charge_conjugation_residual: 0.0,
        mu_tau_residual: 0.0,
    })
}

pub(crate) fn evaluate_f10_packed_rhs_c_jvp(
    grid: &F10ActionGrid,
    ln_a: f64,
    packed_state: &[f64],
    direction_cloglog: &[f64],
    config: F10PackedRhsConfig,
) -> Result<F10PackedRhsJvp, F10PackedRhsError> {
    let base = evaluate_f10_packed_rhs(grid, ln_a, packed_state, config)?;
    let spectral_size = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    let state_size = spectral_size
        .checked_add(2)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    if packed_state.len() != state_size
        || direction_cloglog.len() != spectral_size
        || direction_cloglog.iter().any(|value| !value.is_finite())
    {
        return Err(F10PackedRhsError::InvalidInput);
    }

    if direction_cloglog.iter().all(|value| *value == 0.0) {
        return Ok(F10PackedRhsJvp {
            base,
            values: vec![0.0; state_size],
            combined_action: zero_combined_action_jvp(grid.order)?,
            delta_rho_neutrino_by_flavour: [0.0; PAIR_COUNT],
            delta_rho_neutrino: 0.0,
            delta_hubble_over_hubble: 0.0,
            delta_neutrino_energy_transfer: 0.0,
            delta_electromagnetic_energy_transfer: 0.0,
            first_law_tangent_residual: 0.0,
        });
    }

    let pair_cloglog = &packed_state[..spectral_size];
    let tangent = F10SpectralTangent::build(grid, pair_cloglog, direction_cloglog)
        .map_err(map_tangent_error)?;
    let temperature_cm = base.diagnostics.temperature_cm_mev;
    let temperature_gamma = base.diagnostics.temperature_gamma_mev;
    let combined_action = assemble_combined_action_c_jvp(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
        &tangent,
        config.combined_action,
    )
    .map_err(|_| F10PackedRhsError::Collision)?;

    let mut delta_rho_neutrino_by_flavour = [0.0_f64; PAIR_COUNT];
    let rho_prefactor = temperature_cm.powi(4) / (PI * PI);
    for (flavour, delta_rho) in delta_rho_neutrino_by_flavour.iter_mut().enumerate() {
        let mut integral = 0.0_f64;
        for node in 0..grid.order {
            integral += grid.weights[node]
                * grid.nodes[node].powi(3)
                * tangent.occupation_delta[flavour * grid.order + node];
        }
        *delta_rho = rho_prefactor * integral;
    }
    let delta_rho_neutrino = delta_rho_neutrino_by_flavour.into_iter().sum::<f64>();
    let delta_hubble_over_hubble = 0.5 * delta_rho_neutrino / base.diagnostics.rho_total;
    if !delta_rho_neutrino.is_finite() || !delta_hubble_over_hubble.is_finite() {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    let hubble = base.diagnostics.hubble_mev;
    let mut values = Vec::with_capacity(state_size);
    for flavour in 0..PAIR_COUNT {
        let particle = 2 * flavour;
        let antiparticle = particle + 1;
        for node in 0..grid.order {
            let index = flavour * grid.order + node;
            let coordinate = pair_cloglog[index];
            let exponential = coordinate.exp();
            let chain = (coordinate - exponential).exp();
            if !chain.is_finite() || chain <= 0.0 {
                return Err(F10PackedRhsError::Chart);
            }
            let delta_collision_rate = 0.5
                * (combined_action.native_total[particle * grid.order + node]
                    + combined_action.native_total[antiparticle * grid.order + node]);
            let derivative = delta_collision_rate / (hubble * chain)
                - base.values[index] * (delta_hubble_over_hubble + tangent.log_chain_delta[index]);
            if !derivative.is_finite() {
                return Err(F10PackedRhsError::NonFiniteOutput);
            }
            values.push(derivative);
        }
    }

    let base_qem = base.diagnostics.electromagnetic_energy_transfer;
    let delta_qem = combined_action.electromagnetic_energy_transfer;
    let delta_temperature_derivative = (delta_qem / hubble
        - (base_qem / hubble) * delta_hubble_over_hubble)
        / base.diagnostics.drho_electromagnetic_dt;
    let delta_elapsed_derivative = -(1.0 / hubble) * delta_hubble_over_hubble;
    values.push(delta_temperature_derivative);
    values.push(delta_elapsed_derivative);
    if values.len() != state_size || values.iter().any(|value| !value.is_finite()) {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    let delta_neutrino_energy_transfer = combined_action.neutrino_energy_transfer;
    let delta_electromagnetic_energy_transfer = combined_action.electromagnetic_energy_transfer;
    let first_law_tangent_residual = combined_action.first_law_residual;
    Ok(F10PackedRhsJvp {
        base,
        values,
        combined_action,
        delta_rho_neutrino_by_flavour,
        delta_rho_neutrino,
        delta_hubble_over_hubble,
        delta_neutrino_energy_transfer,
        delta_electromagnetic_energy_transfer,
        first_law_tangent_residual,
    })
}
