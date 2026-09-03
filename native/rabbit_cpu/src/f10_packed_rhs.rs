//! Retained order-60 packed-RHS adapter for the exact six-species F10 comparator.
//!
//! This layer composes the admitted combined collision action with the
//! tree-level finite-electron-mass FLRW thermodynamics. It evaluates one
//! static right-hand side only and does not call an ODE solver.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::{
    F10CombinedAction, F10CombinedActionConfig, assemble_combined_action,
};
use crate::flrw::{NEWTON_G_MEV_MINUS_2, electromagnetic_eos};

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10PackedRhsConfig {
    pub(crate) t_start_mev: f64,
    pub(crate) combined_action: F10CombinedActionConfig,
}

impl Default for F10PackedRhsConfig {
    fn default() -> Self {
        Self {
            t_start_mev: 10.0,
            combined_action: F10CombinedActionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct F10PackedRhsDiagnostics {
    pub(crate) temperature_cm_mev: f64,
    pub(crate) temperature_gamma_mev: f64,
    pub(crate) rho_neutrino_by_flavour: [f64; 3],
    pub(crate) rho_neutrino_total: f64,
    pub(crate) rho_electromagnetic: f64,
    pub(crate) pressure_electromagnetic: f64,
    pub(crate) drho_electromagnetic_dt: f64,
    pub(crate) rho_total: f64,
    pub(crate) hubble_mev: f64,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct F10PackedRhs {
    pub(crate) values: Vec<f64>,
    pub(crate) combined_action: F10CombinedAction,
    pub(crate) diagnostics: F10PackedRhsDiagnostics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10PackedRhsError {
    InvalidInput,
    InvalidConfiguration,
    DimensionOverflow,
    Grid,
    Chart,
    Collision,
    Thermodynamics,
    NonFiniteOutput,
    NonPositiveHubble,
}

fn validate_grid(grid: &F10ActionGrid) -> Result<(), F10PackedRhsError> {
    if grid.order < 8
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
        || grid.nodes.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(F10PackedRhsError::Grid);
    }
    Ok(())
}

fn validate_config(config: F10PackedRhsConfig) -> Result<(), F10PackedRhsError> {
    let collision = config.combined_action.collision;
    if !config.t_start_mev.is_finite()
        || config.t_start_mev <= 0.0
        || collision.incoming_polar_order < 2
        || collision.final_polar_order < 2
        || collision.final_azimuth_order != 4
        || collision.electron_radial_order < 2
        || !config.combined_action.matrix_roundoff_ulps.is_finite()
        || config.combined_action.matrix_roundoff_ulps <= 0.0
        || !config.combined_action.electron_mass_mev.is_finite()
        || config.combined_action.electron_mass_mev <= 0.0
    {
        return Err(F10PackedRhsError::InvalidConfiguration);
    }
    Ok(())
}

fn decode_chart(coordinates: &[f64]) -> Result<(Vec<f64>, Vec<f64>), F10PackedRhsError> {
    let mut occupations = Vec::with_capacity(coordinates.len());
    let mut chain = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        let exponential = coordinate.exp();
        if !exponential.is_finite() || exponential <= 0.0 {
            return Err(F10PackedRhsError::Chart);
        }
        let occupation = -(-exponential).exp_m1();
        let derivative = (coordinate - exponential).exp();
        if !occupation.is_finite()
            || !(0.0..1.0).contains(&occupation)
            || !derivative.is_finite()
            || derivative <= 0.0
        {
            return Err(F10PackedRhsError::Chart);
        }
        occupations.push(occupation);
        chain.push(derivative);
    }
    Ok((occupations, chain))
}

pub(crate) fn evaluate_f10_packed_rhs(
    grid: &F10ActionGrid,
    ln_a: f64,
    packed_state: &[f64],
    config: F10PackedRhsConfig,
) -> Result<F10PackedRhs, F10PackedRhsError> {
    validate_grid(grid)?;
    if !ln_a.is_finite() || packed_state.iter().any(|value| !value.is_finite()) {
        return Err(F10PackedRhsError::InvalidInput);
    }
    validate_config(config)?;

    let spectral_size = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    let state_size = spectral_size
        .checked_add(2)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    if packed_state.len() != state_size {
        return Err(F10PackedRhsError::InvalidInput);
    }

    let temperature_gamma = packed_state[spectral_size];
    let _elapsed_time = packed_state[spectral_size + 1];
    if temperature_gamma <= 0.0 {
        return Err(F10PackedRhsError::InvalidInput);
    }

    let temperature_cm = config.t_start_mev * (-ln_a).exp();
    if !temperature_cm.is_finite() || temperature_cm <= 0.0 {
        return Err(F10PackedRhsError::Thermodynamics);
    }

    let pair_cloglog = &packed_state[..spectral_size];
    let (occupations, chain) = decode_chart(pair_cloglog)?;
    let combined_action = assemble_combined_action(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
        config.combined_action,
    )
    .map_err(|_| F10PackedRhsError::Collision)?;

    let action_size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    if combined_action.native_total.len() != action_size
        || combined_action.modal_total.len() != action_size
    {
        return Err(F10PackedRhsError::Collision);
    }

    let mut rho_neutrino_by_flavour = [0.0_f64; PAIR_COUNT];
    let rho_prefactor = temperature_cm.powi(4) / (PI * PI);
    for flavour in 0..PAIR_COUNT {
        let mut integral = 0.0_f64;
        for node in 0..grid.order {
            integral += grid.weights[node]
                * grid.nodes[node].powi(3)
                * occupations[flavour * grid.order + node];
        }
        let density = rho_prefactor * integral;
        if !density.is_finite() || density <= 0.0 {
            return Err(F10PackedRhsError::Thermodynamics);
        }
        rho_neutrino_by_flavour[flavour] = density;
    }
    let rho_neutrino_total = rho_neutrino_by_flavour.into_iter().sum::<f64>();
    if !rho_neutrino_total.is_finite() || rho_neutrino_total <= 0.0 {
        return Err(F10PackedRhsError::Thermodynamics);
    }

    let electromagnetic =
        electromagnetic_eos(temperature_gamma).map_err(|_| F10PackedRhsError::Thermodynamics)?;
    let rho_total = rho_neutrino_total + electromagnetic.rho;
    let hubble_squared = (8.0 * PI * NEWTON_G_MEV_MINUS_2 / 3.0) * rho_total;
    if !rho_total.is_finite() || !hubble_squared.is_finite() {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }
    if rho_total <= 0.0 || hubble_squared <= 0.0 {
        return Err(F10PackedRhsError::NonPositiveHubble);
    }
    let hubble = hubble_squared.sqrt();
    if !hubble.is_finite() {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }
    if hubble <= 0.0 {
        return Err(F10PackedRhsError::NonPositiveHubble);
    }

    let mut values = Vec::with_capacity(state_size);
    for flavour in 0..PAIR_COUNT {
        let particle = 2 * flavour;
        let antiparticle = particle + 1;
        for node in 0..grid.order {
            let collision_rate = 0.5
                * (combined_action.native_total[particle * grid.order + node]
                    + combined_action.native_total[antiparticle * grid.order + node]);
            let denominator = hubble * chain[flavour * grid.order + node];
            let derivative = collision_rate / denominator;
            if !collision_rate.is_finite()
                || !denominator.is_finite()
                || denominator <= 0.0
                || !derivative.is_finite()
            {
                return Err(F10PackedRhsError::NonFiniteOutput);
            }
            values.push(derivative);
        }
    }

    let temperature_numerator = -3.0 * (electromagnetic.rho + electromagnetic.pressure)
        + combined_action.electromagnetic_energy_transfer / hubble;
    let temperature_derivative = temperature_numerator / electromagnetic.drho_dt;
    let elapsed_derivative = 1.0 / hubble;
    if !temperature_numerator.is_finite()
        || !temperature_derivative.is_finite()
        || !elapsed_derivative.is_finite()
    {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }
    values.push(temperature_derivative);
    values.push(elapsed_derivative);
    if values.len() != state_size || values.iter().any(|value| !value.is_finite()) {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    let diagnostics = F10PackedRhsDiagnostics {
        temperature_cm_mev: temperature_cm,
        temperature_gamma_mev: temperature_gamma,
        rho_neutrino_by_flavour,
        rho_neutrino_total,
        rho_electromagnetic: electromagnetic.rho,
        pressure_electromagnetic: electromagnetic.pressure,
        drho_electromagnetic_dt: electromagnetic.drho_dt,
        rho_total,
        hubble_mev: hubble,
        neutrino_energy_transfer: combined_action.neutrino_energy_transfer,
        electromagnetic_energy_transfer: combined_action.electromagnetic_energy_transfer,
        first_law_residual: combined_action.first_law_residual,
        whole_reaction_domain_rejections: combined_action.whole_reaction_domain_rejections,
        matrix_roundoff_corrections: combined_action.matrix_roundoff_corrections,
        largest_matrix_roundoff_correction: combined_action.largest_matrix_roundoff_correction,
    };

    if [
        diagnostics.temperature_cm_mev,
        diagnostics.temperature_gamma_mev,
        diagnostics.rho_neutrino_total,
        diagnostics.rho_electromagnetic,
        diagnostics.pressure_electromagnetic,
        diagnostics.drho_electromagnetic_dt,
        diagnostics.rho_total,
        diagnostics.hubble_mev,
        diagnostics.neutrino_energy_transfer,
        diagnostics.electromagnetic_energy_transfer,
        diagnostics.first_law_residual,
        diagnostics.largest_matrix_roundoff_correction,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    Ok(F10PackedRhs {
        values,
        combined_action,
        diagnostics,
    })
}
