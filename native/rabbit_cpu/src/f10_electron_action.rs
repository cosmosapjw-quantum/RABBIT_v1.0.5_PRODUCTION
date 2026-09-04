//! Finite-electron-mass collision action for the frozen D-081R1 comparator.
//!
//! This module assembles the twelve neutrino-electron/positron elastic events
//! and the three electron-positron pair events on the exact order-eight action
//! grid. It preserves the frozen Python comparator's quadrature, finite-mass
//! matrix elements, Pauli factor, species routing, family decomposition, and
//! neutrino/electromagnetic energy and H-functional ledgers. The combined
//! self-plus-electron action, packed RHS, derivatives, and solver remain
//! outside this node.

#![cfg_attr(not(test), allow(dead_code))]

pub(crate) mod c_jvp;

use core::f64::consts::PI;

use crate::f10_action_grid::{F10ActionGrid, decode_cloglog_to_logit};
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicBatch, F10KinematicInput, angular_rule,
    electron_half_line_rule, two_body_kinematics,
};
use crate::f10_action_spectral::{modal_basis, modal_coefficients, native_action};
use crate::f10_kernel_primitives::{
    F10ElectronCategory, F10ElectronEvent, F10EventMeasureInput, F10Flavour, F10InvariantProducts,
    F10KernelError, F10Species, f10_electron_events, f10_electron_matrix, f10_event_measure,
    stable_pauli_gain_minus_loss,
};

pub(crate) const F10_ELECTRON_MASS_MEV: f64 = 0.510_998_95;

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;
const ELASTIC_EVENT_COUNT: usize = 12;
const PAIR_EVENT_COUNT: usize = 3;
const ELECTRON_EVENT_COUNT: usize = 15;
const TWO_PI_SQUARED: f64 = 2.0 * PI * PI;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10ElectronActionConfig {
    pub(crate) collision: F10CollisionConfig,
    pub(crate) matrix_roundoff_ulps: f64,
    pub(crate) electron_mass_mev: f64,
}

impl Default for F10ElectronActionConfig {
    fn default() -> Self {
        Self {
            collision: F10CollisionConfig::default(),
            matrix_roundoff_ulps: 1024.0,
            electron_mass_mev: F10_ELECTRON_MASS_MEV,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10ElectronActionMoments {
    pub(crate) signed_number_rate: f64,
    pub(crate) absolute_number_rate: f64,
    pub(crate) signed_energy_rate: f64,
    pub(crate) absolute_energy_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct F10ElectronAction {
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) elastic_modal: Vec<f64>,
    pub(crate) pair_modal: Vec<f64>,
    pub(crate) elastic_native: Vec<f64>,
    pub(crate) pair_native: Vec<f64>,
    pub(crate) family_names: Vec<String>,
    pub(crate) family_modal: Vec<f64>,
    pub(crate) family_native: Vec<f64>,
    pub(crate) bath_energy_by_family: Vec<f64>,
    pub(crate) moments: F10ElectronActionMoments,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) elastic_domain_rejections: usize,
    pub(crate) pair_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) neutrino_h_rate: f64,
    pub(crate) electromagnetic_h_rate: f64,
    pub(crate) entropy_production: f64,
    pub(crate) node_neutrino_h_rate: f64,
    pub(crate) entropy_duality_residual: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10ElectronActionError {
    InvalidInput,
    InvalidConfiguration,
    DimensionOverflow,
    Foundation,
    Kinematics,
    Kernel,
    NonFiniteOutput,
}

impl From<F10KernelError> for F10ElectronActionError {
    fn from(_: F10KernelError) -> Self {
        Self::Kernel
    }
}

fn pair_index(species: F10Species) -> usize {
    match species.flavour() {
        F10Flavour::Electron => 0,
        F10Flavour::Muon => 1,
        F10Flavour::Tau => 2,
    }
}

fn species_index(species: F10Species) -> usize {
    match species {
        F10Species::NuE => 0,
        F10Species::AntiNuE => 1,
        F10Species::NuMu => 2,
        F10Species::AntiNuMu => 3,
        F10Species::NuTau => 4,
        F10Species::AntiNuTau => 5,
    }
}

fn flavour_name(flavour: F10Flavour) -> &'static str {
    match flavour {
        F10Flavour::Electron => "e",
        F10Flavour::Muon => "mu",
        F10Flavour::Tau => "tau",
    }
}

fn family_name(event: F10ElectronEvent) -> String {
    match event.category {
        F10ElectronCategory::ElasticMinus => {
            format!("{}:elastic_minus", event.target.name())
        }
        F10ElectronCategory::ElasticPlus => {
            format!("{}:elastic_plus", event.target.name())
        }
        F10ElectronCategory::Pair => {
            format!("{}:pair", flavour_name(event.target.flavour()))
        }
    }
}

fn strict_open_logit(value: f64) -> bool {
    if !value.is_finite() {
        return false;
    }
    let occupation = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    occupation.is_finite() && occupation > 0.0 && occupation < 1.0
}

fn decode_pair_logits(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
) -> Result<Vec<f64>, F10ElectronActionError> {
    let expected = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    if pair_cloglog.len() != expected || pair_cloglog.iter().any(|value| !value.is_finite()) {
        return Err(F10ElectronActionError::InvalidInput);
    }
    pair_cloglog
        .iter()
        .map(|coordinate| {
            decode_cloglog_to_logit(*coordinate).map_err(|_| F10ElectronActionError::Foundation)
        })
        .collect()
}

fn spectral_coefficients(
    grid: &F10ActionGrid,
    logits: &[f64],
) -> Result<Vec<f64>, F10ElectronActionError> {
    let mut coefficients = Vec::with_capacity(PAIR_COUNT * grid.order);
    for pair in 0..PAIR_COUNT {
        coefficients.extend(
            modal_coefficients(grid, &logits[pair * grid.order..(pair + 1) * grid.order])
                .map_err(|_| F10ElectronActionError::Foundation)?,
        );
    }
    Ok(coefficients)
}

fn interpolated_pair_logits(
    coefficients: &[f64],
    basis: &[f64],
    point_count: usize,
    order: usize,
) -> Result<Vec<f64>, F10ElectronActionError> {
    let expected_basis = point_count
        .checked_mul(order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    if basis.len() != expected_basis || coefficients.len() != PAIR_COUNT * order {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let mut values = vec![0.0; PAIR_COUNT * point_count];
    for pair in 0..PAIR_COUNT {
        for point in 0..point_count {
            let value = (0..order)
                .map(|mode| coefficients[pair * order + mode] * basis[point * order + mode])
                .sum::<f64>();
            if !strict_open_logit(value) {
                return Err(F10ElectronActionError::Foundation);
            }
            values[pair * point_count + point] = value;
        }
    }
    Ok(values)
}

fn invariant_products(batch: &F10KinematicBatch, index: usize) -> F10InvariantProducts {
    F10InvariantProducts {
        d12: batch.d12[index],
        d13: batch.d13[index],
        d14: batch.d14[index],
        d23: batch.d23[index],
        d24: batch.d24[index],
        d34: batch.d34[index],
    }
}

fn matrix_values(
    event: F10ElectronEvent,
    batch: &F10KinematicBatch,
    electron_mass: f64,
    roundoff_ulps: f64,
) -> Result<(Vec<f64>, usize, f64), F10ElectronActionError> {
    let mut values = Vec::with_capacity(batch.support.len());
    let mut corrections = 0_usize;
    let mut largest_correction = 0.0_f64;
    for index in 0..batch.support.len() {
        let matrix = f10_electron_matrix(
            event.target,
            event.category,
            invariant_products(batch, index),
            electron_mass,
            batch.support[index],
            roundoff_ulps,
        )?;
        if matrix.corrected {
            corrections = corrections
                .checked_add(1)
                .ok_or(F10ElectronActionError::DimensionOverflow)?;
            largest_correction = largest_correction.max(matrix.correction);
        }
        values.push(matrix.value);
    }
    Ok((values, corrections, largest_correction))
}

fn action_moments(
    grid: &F10ActionGrid,
    native: &[f64],
    temperature_cm: f64,
) -> Result<F10ElectronActionMoments, F10ElectronActionError> {
    if native.len() != SPECIES_COUNT * grid.order {
        return Err(F10ElectronActionError::InvalidInput);
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
    let moments = F10ElectronActionMoments {
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
        Err(F10ElectronActionError::NonFiniteOutput)
    }
}

fn node_neutrino_h_rate(
    modal: &[f64],
    coefficients: &[f64],
    order: usize,
) -> Result<f64, F10ElectronActionError> {
    if modal.len() != SPECIES_COUNT * order || coefficients.len() != PAIR_COUNT * order {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let mut result = 0.0_f64;
    for species in F10Species::ALL {
        let species_offset = species_index(species) * order;
        let pair_offset = pair_index(species) * order;
        result += (0..order)
            .map(|mode| modal[species_offset + mode] * coefficients[pair_offset + mode])
            .sum::<f64>();
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(F10ElectronActionError::NonFiniteOutput)
    }
}

fn validate_grid(grid: &F10ActionGrid) -> Result<(), F10ElectronActionError> {
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
    {
        return Err(F10ElectronActionError::InvalidInput);
    }
    Ok(())
}

pub(crate) fn assemble_electron_action(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    config: F10ElectronActionConfig,
) -> Result<F10ElectronAction, F10ElectronActionError> {
    validate_grid(grid)?;
    if !temperature_cm.is_finite()
        || temperature_cm <= 0.0
        || !temperature_gamma.is_finite()
        || temperature_gamma <= 0.0
    {
        return Err(F10ElectronActionError::InvalidInput);
    }
    if !config.matrix_roundoff_ulps.is_finite()
        || config.matrix_roundoff_ulps <= 0.0
        || !config.electron_mass_mev.is_finite()
        || config.electron_mass_mev <= 0.0
    {
        return Err(F10ElectronActionError::InvalidConfiguration);
    }

    let rule =
        angular_rule(config.collision).map_err(|_| F10ElectronActionError::InvalidConfiguration)?;
    let angular_size = rule
        .incoming_mu
        .len()
        .checked_mul(rule.final_mu.len())
        .and_then(|value| value.checked_mul(rule.azimuth.len()))
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let (electron_p2, electron_weights) =
        electron_half_line_rule(config.collision.electron_radial_order, temperature_gamma)
            .map_err(|_| F10ElectronActionError::InvalidConfiguration)?;
    let neutrino_p2: Vec<f64> = grid
        .nodes
        .iter()
        .map(|node| temperature_cm * node)
        .collect();
    let neutrino_weights: Vec<f64> = grid
        .weights
        .iter()
        .map(|weight| temperature_cm * weight)
        .collect();
    if neutrino_p2
        .iter()
        .chain(&neutrino_weights)
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(F10ElectronActionError::InvalidInput);
    }

    let logits = decode_pair_logits(grid, pair_cloglog)?;
    let coefficients = spectral_coefficients(grid, &logits)?;
    let native_basis =
        modal_basis(grid, &grid.nodes).map_err(|_| F10ElectronActionError::Foundation)?;

    let events = f10_electron_events();
    if events.len() != ELECTRON_EVENT_COUNT {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let (elastic_events, pair_events) = events.split_at(ELASTIC_EVENT_COUNT);
    if pair_events.len() != PAIR_EVENT_COUNT {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let family_names: Vec<String> = events.iter().copied().map(family_name).collect();

    let species_modal_size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let family_modal_size = ELECTRON_EVENT_COUNT
        .checked_mul(species_modal_size)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let mut modal = vec![0.0; species_modal_size];
    let mut elastic_modal = vec![0.0; species_modal_size];
    let mut pair_modal = vec![0.0; species_modal_size];
    let mut family_modal = vec![0.0; family_modal_size];
    let mut bath_energy_by_family = vec![0.0; ELECTRON_EVENT_COUNT];

    let mut elastic_domain_rejections = 0_usize;
    let pair_domain_rejections = 0_usize;
    let mut matrix_roundoff_corrections = 0_usize;
    let mut largest_matrix_roundoff_correction = 0.0_f64;
    let mut neutrino_energy_transfer = 0.0_f64;
    let mut electromagnetic_energy_transfer = 0.0_f64;
    let mut neutrino_h_rate = 0.0_f64;
    let mut electromagnetic_h_rate = 0.0_f64;

    for node_index in 0..grid.order {
        let y1 = grid.nodes[node_index];
        let p1 = temperature_cm * y1;
        let outer_weight =
            temperature_cm.powi(3) * grid.weights[node_index] * y1.powi(2) / TWO_PI_SQUARED;
        if !p1.is_finite() || p1 <= 0.0 || !outer_weight.is_finite() || outer_weight <= 0.0 {
            return Err(F10ElectronActionError::InvalidInput);
        }

        let elastic_batch = two_body_kinematics(F10KinematicInput {
            p1,
            p2_nodes: &electron_p2,
            p2_weights: &electron_weights,
            mass2: config.electron_mass_mev,
            mass3: 0.0,
            mass4: config.electron_mass_mev,
            config: config.collision,
        })
        .map_err(|_| F10ElectronActionError::Kinematics)?;
        let elastic_sample_count = elastic_batch.support.len();
        let expected_elastic_samples = electron_p2
            .len()
            .checked_mul(angular_size)
            .ok_or(F10ElectronActionError::DimensionOverflow)?;
        if elastic_sample_count != expected_elastic_samples {
            return Err(F10ElectronActionError::Kinematics);
        }

        let mut elastic_valid_positions = Vec::new();
        let mut elastic_y3 = Vec::new();
        let mut rejected_supported_samples = 0_usize;
        for sample in 0..elastic_sample_count {
            let y3 = elastic_batch.p3_magnitude[sample] / temperature_cm;
            let in_domain = elastic_batch.support[sample] && y3 > 0.0 && y3 < grid.y_max;
            if in_domain {
                elastic_valid_positions.push(sample);
                elastic_y3.push(y3);
            } else if elastic_batch.support[sample] {
                rejected_supported_samples = rejected_supported_samples
                    .checked_add(1)
                    .ok_or(F10ElectronActionError::DimensionOverflow)?;
            }
        }
        elastic_domain_rejections = elastic_domain_rejections
            .checked_add(
                rejected_supported_samples
                    .checked_mul(elastic_events.len())
                    .ok_or(F10ElectronActionError::DimensionOverflow)?,
            )
            .ok_or(F10ElectronActionError::DimensionOverflow)?;

        let elastic_valid_count = elastic_valid_positions.len();
        let elastic_basis3 =
            modal_basis(grid, &elastic_y3).map_err(|_| F10ElectronActionError::Foundation)?;
        let elastic_outgoing = interpolated_pair_logits(
            &coefficients,
            &elastic_basis3,
            elastic_valid_count,
            grid.order,
        )?;
        let mut elastic_measures = Vec::with_capacity(elastic_valid_count);
        for &sample in &elastic_valid_positions {
            elastic_measures.push(f10_event_measure(F10EventMeasureInput {
                p1,
                p2: elastic_batch.p2[sample],
                e2: elastic_batch.e2[sample],
                phase_space: elastic_batch.phase_space[sample],
                quadrature_weight: elastic_batch.quadrature_weight[sample],
                outer_weight,
            })?);
        }

        let elastic_rate_count = elastic_events
            .len()
            .checked_mul(elastic_sample_count)
            .ok_or(F10ElectronActionError::DimensionOverflow)?;
        let mut elastic_rates = vec![0.0; elastic_rate_count];
        for (event_index, &event) in elastic_events.iter().enumerate() {
            let (matrix, corrections, largest_correction) = matrix_values(
                event,
                &elastic_batch,
                config.electron_mass_mev,
                config.matrix_roundoff_ulps,
            )?;
            matrix_roundoff_corrections = matrix_roundoff_corrections
                .checked_add(corrections)
                .ok_or(F10ElectronActionError::DimensionOverflow)?;
            largest_matrix_roundoff_correction =
                largest_matrix_roundoff_correction.max(largest_correction);

            let species = event.target;
            let pair = pair_index(species);
            let u1 = logits[pair * grid.order + node_index];
            let rate_offset = event_index * elastic_sample_count;
            for (valid_index, &sample) in elastic_valid_positions.iter().enumerate() {
                let u2 = -elastic_batch.e2[sample] / temperature_gamma;
                let u3 = elastic_outgoing[pair * elastic_valid_count + valid_index];
                let u4 = -elastic_batch.e4[sample] / temperature_gamma;
                let pauli = stable_pauli_gain_minus_loss([u1, u2, u3, u4])?;
                let rate = elastic_measures[valid_index] * matrix[sample] * pauli;
                if !rate.is_finite() {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                elastic_rates[rate_offset + sample] = rate;

                let dqnu = rate * (p1 - elastic_batch.e3[sample]);
                let dqem = rate * (elastic_batch.e2[sample] - elastic_batch.e4[sample]);
                let dhnu = rate * (u1 - u3);
                let dhem = rate * (u2 - u4);
                if [dqnu, dqem, dhnu, dhem]
                    .into_iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                neutrino_energy_transfer += dqnu;
                electromagnetic_energy_transfer += dqem;
                neutrino_h_rate += dhnu;
                electromagnetic_h_rate += dhem;
                bath_energy_by_family[event_index] += dqem;
            }
        }

        for (event_index, &event) in elastic_events.iter().enumerate() {
            let rate_row = &elastic_rates
                [event_index * elastic_sample_count..(event_index + 1) * elastic_sample_count];
            let incoming_sum = rate_row.iter().sum::<f64>();
            let target = species_index(event.target);
            let family_offset = (event_index * SPECIES_COUNT + target) * grid.order;
            for mode in 0..grid.order {
                let incoming = incoming_sum * native_basis[node_index * grid.order + mode];
                let outgoing = elastic_valid_positions
                    .iter()
                    .enumerate()
                    .map(|(valid_index, &sample)| {
                        rate_row[sample] * elastic_basis3[valid_index * grid.order + mode]
                    })
                    .sum::<f64>();
                let contribution = incoming - outgoing;
                let target_index = target * grid.order + mode;
                modal[target_index] += contribution;
                elastic_modal[target_index] += contribution;
                family_modal[family_offset + mode] += contribution;
            }
        }

        let pair_batch = two_body_kinematics(F10KinematicInput {
            p1,
            p2_nodes: &neutrino_p2,
            p2_weights: &neutrino_weights,
            mass2: 0.0,
            mass3: config.electron_mass_mev,
            mass4: config.electron_mass_mev,
            config: config.collision,
        })
        .map_err(|_| F10ElectronActionError::Kinematics)?;
        let pair_sample_count = pair_batch.support.len();
        let expected_pair_samples = grid
            .order
            .checked_mul(angular_size)
            .ok_or(F10ElectronActionError::DimensionOverflow)?;
        if pair_sample_count != expected_pair_samples {
            return Err(F10ElectronActionError::Kinematics);
        }

        let pair_valid_positions: Vec<usize> = pair_batch
            .support
            .iter()
            .enumerate()
            .filter_map(|(index, support)| support.then_some(index))
            .collect();
        let mut pair_measures = Vec::with_capacity(pair_valid_positions.len());
        for &sample in &pair_valid_positions {
            pair_measures.push(f10_event_measure(F10EventMeasureInput {
                p1,
                p2: pair_batch.p2[sample],
                e2: pair_batch.e2[sample],
                phase_space: pair_batch.phase_space[sample],
                quadrature_weight: pair_batch.quadrature_weight[sample],
                outer_weight,
            })?);
        }

        let pair_rate_count = pair_events
            .len()
            .checked_mul(pair_sample_count)
            .ok_or(F10ElectronActionError::DimensionOverflow)?;
        let mut pair_rates = vec![0.0; pair_rate_count];
        for (pair_event_index, &event) in pair_events.iter().enumerate() {
            let (matrix, corrections, largest_correction) = matrix_values(
                event,
                &pair_batch,
                config.electron_mass_mev,
                config.matrix_roundoff_ulps,
            )?;
            matrix_roundoff_corrections = matrix_roundoff_corrections
                .checked_add(corrections)
                .ok_or(F10ElectronActionError::DimensionOverflow)?;
            largest_matrix_roundoff_correction =
                largest_matrix_roundoff_correction.max(largest_correction);

            let target = event.target;
            let partner = target.cp_partner();
            let target_pair = pair_index(target);
            let partner_pair = pair_index(partner);
            let u1 = logits[target_pair * grid.order + node_index];
            let rate_offset = pair_event_index * pair_sample_count;
            let family_index = ELASTIC_EVENT_COUNT + pair_event_index;
            for (valid_index, &sample) in pair_valid_positions.iter().enumerate() {
                let p2_index = sample / angular_size;
                let u2 = logits[partner_pair * grid.order + p2_index];
                let u3 = -pair_batch.e3[sample] / temperature_gamma;
                let u4 = -pair_batch.e4[sample] / temperature_gamma;
                let pauli = stable_pauli_gain_minus_loss([u1, u2, u3, u4])?;
                let rate = pair_measures[valid_index] * matrix[sample] * pauli;
                if !rate.is_finite() {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                pair_rates[rate_offset + sample] = rate;

                let dqnu = rate * (p1 + pair_batch.p2[sample]);
                let dqem = rate * (-pair_batch.e3[sample] - pair_batch.e4[sample]);
                let dhnu = rate * (u1 + u2);
                let dhem = rate * (-u3 - u4);
                if [dqnu, dqem, dhnu, dhem]
                    .into_iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                neutrino_energy_transfer += dqnu;
                electromagnetic_energy_transfer += dqem;
                neutrino_h_rate += dhnu;
                electromagnetic_h_rate += dhem;
                bath_energy_by_family[family_index] += dqem;
            }
        }

        for (pair_event_index, &event) in pair_events.iter().enumerate() {
            let rate_row = &pair_rates
                [pair_event_index * pair_sample_count..(pair_event_index + 1) * pair_sample_count];
            let incoming1_sum = rate_row.iter().sum::<f64>();
            let mut p2_sums = vec![0.0; grid.order];
            for (sample, &rate) in rate_row.iter().enumerate() {
                p2_sums[sample / angular_size] += rate;
            }
            let family_index = ELASTIC_EVENT_COUNT + pair_event_index;
            for (species, incoming1) in [(event.target, true), (event.target.cp_partner(), false)] {
                let target = species_index(species);
                let family_offset = (family_index * SPECIES_COUNT + target) * grid.order;
                for mode in 0..grid.order {
                    let contribution = if incoming1 {
                        incoming1_sum * native_basis[node_index * grid.order + mode]
                    } else {
                        (0..grid.order)
                            .map(|p2_index| {
                                p2_sums[p2_index] * native_basis[p2_index * grid.order + mode]
                            })
                            .sum::<f64>()
                    };
                    let target_index = target * grid.order + mode;
                    modal[target_index] += contribution;
                    pair_modal[target_index] += contribution;
                    family_modal[family_offset + mode] += contribution;
                }
            }
        }
    }

    let native = native_action(grid, &modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let elastic_native = native_action(grid, &elastic_modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let pair_native = native_action(grid, &pair_modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let mut family_native = vec![0.0; family_modal_size];
    for family in 0..ELECTRON_EVENT_COUNT {
        let start = family * species_modal_size;
        let converted = native_action(
            grid,
            &family_modal[start..start + species_modal_size],
            SPECIES_COUNT,
            temperature_cm,
        )
        .map_err(|_| F10ElectronActionError::Foundation)?;
        family_native[start..start + species_modal_size].copy_from_slice(&converted);
    }

    let moments = action_moments(grid, &native, temperature_cm)?;
    let node_neutrino_h_rate = node_neutrino_h_rate(&modal, &coefficients, grid.order)?;
    let first_law_residual = (neutrino_energy_transfer + electromagnetic_energy_transfer).abs()
        / (neutrino_energy_transfer.abs() + electromagnetic_energy_transfer.abs())
            .max(f64::MIN_POSITIVE);
    let entropy_production = -(neutrino_h_rate + electromagnetic_h_rate);
    let entropy_duality_residual = (node_neutrino_h_rate - neutrino_h_rate).abs()
        / (node_neutrino_h_rate.abs() + neutrino_h_rate.abs()).max(f64::MIN_POSITIVE);
    let whole_reaction_domain_rejections = elastic_domain_rejections
        .checked_add(pair_domain_rejections)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;

    let scalars = [
        largest_matrix_roundoff_correction,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        neutrino_h_rate,
        electromagnetic_h_rate,
        entropy_production,
        node_neutrino_h_rate,
        entropy_duality_residual,
    ];
    if modal
        .iter()
        .chain(&native)
        .chain(&elastic_modal)
        .chain(&pair_modal)
        .chain(&elastic_native)
        .chain(&pair_native)
        .chain(&family_modal)
        .chain(&family_native)
        .chain(&bath_energy_by_family)
        .any(|value| !value.is_finite())
        || scalars.into_iter().any(|value| !value.is_finite())
    {
        return Err(F10ElectronActionError::NonFiniteOutput);
    }

    Ok(F10ElectronAction {
        modal,
        native,
        elastic_modal,
        pair_modal,
        elastic_native,
        pair_native,
        family_names,
        family_modal,
        family_native,
        bath_energy_by_family,
        moments,
        whole_reaction_domain_rejections,
        elastic_domain_rejections,
        pair_domain_rejections,
        matrix_roundoff_corrections,
        largest_matrix_roundoff_correction,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        neutrino_h_rate,
        electromagnetic_h_rate,
        entropy_production,
        node_neutrino_h_rate,
        entropy_duality_residual,
    })
}
