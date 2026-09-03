//! Six-species neutrino self-collision action for the frozen D-081R1 comparator.
//!
//! This module assembles the 27 ordered members of the 24 reversible
//! neutrino-neutrino events on the exact order-eight action grid.  It preserves
//! the frozen Python comparator's target quadrature, strict spectral domain,
//! Pauli factor, weak matrix elements, four-leg signs, and R1--R9 physical-row
//! decomposition.  Electron collisions, the packed RHS, derivatives, and the
//! solver remain outside this node.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::{F10ActionGrid, decode_cloglog_to_logit};
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicBatch, F10KinematicInput, angular_rule,
    two_body_kinematics,
};
use crate::f10_action_spectral::{modal_basis, modal_coefficients, native_action};
use crate::f10_kernel_primitives::{
    F10EventMeasureInput, F10Flavour, F10InvariantProducts, F10KernelError, F10SelfCategory,
    F10SelfEvent, F10SelfKernel, F10Species, f10_event_measure, f10_self_events,
    f10_self_matrix, stable_pauli_gain_minus_loss,
};

const PAIR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;
const SELF_ROW_COUNT: usize = 9;
const TWO_PI_SQUARED: f64 = 2.0 * PI * PI;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10SelfActionConfig {
    pub(crate) collision: F10CollisionConfig,
    pub(crate) matrix_roundoff_ulps: f64,
}

impl Default for F10SelfActionConfig {
    fn default() -> Self {
        Self {
            collision: F10CollisionConfig::default(),
            matrix_roundoff_ulps: 1024.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10ActionMoments {
    pub(crate) signed_number_rate: f64,
    pub(crate) absolute_number_rate: f64,
    pub(crate) signed_energy_rate: f64,
    pub(crate) absolute_energy_rate: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct F10SelfAction {
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) row_modal: Vec<f64>,
    pub(crate) row_native: Vec<f64>,
    pub(crate) moments: F10ActionMoments,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
    pub(crate) event_entropy_rate: f64,
    pub(crate) node_entropy_rate: f64,
    pub(crate) entropy_duality_residual: f64,
    pub(crate) event_energy_residual: f64,
    pub(crate) event_energy_absolute: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10SelfActionError {
    InvalidInput,
    InvalidConfiguration,
    DimensionOverflow,
    Foundation,
    Kinematics,
    Kernel,
    NonFiniteOutput,
}

impl From<F10KernelError> for F10SelfActionError {
    fn from(_: F10KernelError) -> Self {
        Self::Kernel
    }
}

#[derive(Debug)]
struct MatrixCacheEntry {
    kernel: F10SelfKernel,
    coefficient_bits: u64,
    values: Vec<f64>,
    corrections: usize,
    largest_correction: f64,
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

fn self_event_row(
    event: F10SelfEvent,
    species: F10Species,
) -> Result<usize, F10SelfActionError> {
    let flavour = species.flavour();
    let electron = F10Flavour::Electron;
    let other = event
        .legs
        .into_iter()
        .map(F10Species::flavour)
        .find(|candidate| *candidate != flavour);
    let includes_electron = flavour == electron || other == Some(electron);
    let row = match event.category {
        F10SelfCategory::SameSignIdentical => {
            if flavour == electron {
                0
            } else {
                2
            }
        }
        F10SelfCategory::SamePairOppositeSignElastic => {
            if flavour == electron {
                1
            } else {
                3
            }
        }
        F10SelfCategory::DistinctSameSignElastic => {
            if other.is_none() {
                return Err(F10SelfActionError::InvalidInput);
            }
            if includes_electron { 6 } else { 2 }
        }
        F10SelfCategory::DistinctOppositeSignElastic => {
            if other.is_none() {
                return Err(F10SelfActionError::InvalidInput);
            }
            if includes_electron { 7 } else { 4 }
        }
        F10SelfCategory::PairConversion => {
            if other.is_none() {
                return Err(F10SelfActionError::InvalidInput);
            }
            if includes_electron { 8 } else { 5 }
        }
    };
    Ok(row)
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
) -> Result<Vec<f64>, F10SelfActionError> {
    let expected = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10SelfActionError::DimensionOverflow)?;
    if pair_cloglog.len() != expected || pair_cloglog.iter().any(|value| !value.is_finite()) {
        return Err(F10SelfActionError::InvalidInput);
    }
    pair_cloglog
        .iter()
        .map(|coordinate| {
            decode_cloglog_to_logit(*coordinate).map_err(|_| F10SelfActionError::Foundation)
        })
        .collect()
}

fn spectral_coefficients(
    grid: &F10ActionGrid,
    logits: &[f64],
) -> Result<Vec<f64>, F10SelfActionError> {
    let mut coefficients = Vec::with_capacity(PAIR_COUNT * grid.order);
    for pair in 0..PAIR_COUNT {
        coefficients.extend(
            modal_coefficients(grid, &logits[pair * grid.order..(pair + 1) * grid.order])
                .map_err(|_| F10SelfActionError::Foundation)?,
        );
    }
    Ok(coefficients)
}

fn interpolated_pair_logits(
    coefficients: &[f64],
    basis: &[f64],
    point_count: usize,
    order: usize,
) -> Result<Vec<f64>, F10SelfActionError> {
    let expected_basis = point_count
        .checked_mul(order)
        .ok_or(F10SelfActionError::DimensionOverflow)?;
    if basis.len() != expected_basis || coefficients.len() != PAIR_COUNT * order {
        return Err(F10SelfActionError::InvalidInput);
    }
    let mut values = vec![0.0; PAIR_COUNT * point_count];
    for pair in 0..PAIR_COUNT {
        for point in 0..point_count {
            let value = (0..order)
                .map(|mode| {
                    coefficients[pair * order + mode] * basis[point * order + mode]
                })
                .sum::<f64>();
            if !strict_open_logit(value) {
                return Err(F10SelfActionError::Foundation);
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

fn build_matrix_cache_entry(
    event: F10SelfEvent,
    batch: &F10KinematicBatch,
    roundoff_ulps: f64,
) -> Result<MatrixCacheEntry, F10SelfActionError> {
    let mut values = Vec::with_capacity(batch.support.len());
    let mut corrections = 0_usize;
    let mut largest_correction = 0.0_f64;
    for index in 0..batch.support.len() {
        let matrix = f10_self_matrix(
            event.kernel,
            event.coefficient,
            invariant_products(batch, index),
            batch.support[index],
            roundoff_ulps,
        )?;
        if matrix.corrected {
            corrections = corrections
                .checked_add(1)
                .ok_or(F10SelfActionError::DimensionOverflow)?;
            largest_correction = largest_correction.max(matrix.correction);
        }
        values.push(matrix.value);
    }
    Ok(MatrixCacheEntry {
        kernel: event.kernel,
        coefficient_bits: event.coefficient.to_bits(),
        values,
        corrections,
        largest_correction,
    })
}

fn matrix_cache_index(cache: &[MatrixCacheEntry], event: F10SelfEvent) -> Option<usize> {
    cache.iter().position(|entry| {
        entry.kernel == event.kernel && entry.coefficient_bits == event.coefficient.to_bits()
    })
}

fn action_moments(
    grid: &F10ActionGrid,
    native: &[f64],
    temperature: f64,
) -> Result<F10ActionMoments, F10SelfActionError> {
    if native.len() != SPECIES_COUNT * grid.order {
        return Err(F10SelfActionError::InvalidInput);
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
    let moments = F10ActionMoments {
        signed_number_rate: temperature.powi(3) * signed_number / TWO_PI_SQUARED,
        absolute_number_rate: temperature.powi(3) * absolute_number / TWO_PI_SQUARED,
        signed_energy_rate: temperature.powi(4) * signed_energy / TWO_PI_SQUARED,
        absolute_energy_rate: temperature.powi(4) * absolute_energy / TWO_PI_SQUARED,
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
        Err(F10SelfActionError::NonFiniteOutput)
    }
}

fn node_entropy_rate(
    modal: &[f64],
    coefficients: &[f64],
    order: usize,
) -> Result<f64, F10SelfActionError> {
    if modal.len() != SPECIES_COUNT * order || coefficients.len() != PAIR_COUNT * order {
        return Err(F10SelfActionError::InvalidInput);
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
        Err(F10SelfActionError::NonFiniteOutput)
    }
}

pub(crate) fn assemble_self_action(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    config: F10SelfActionConfig,
) -> Result<F10SelfAction, F10SelfActionError> {
    if !temperature_cm.is_finite() || temperature_cm <= 0.0 {
        return Err(F10SelfActionError::InvalidInput);
    }
    if !config.matrix_roundoff_ulps.is_finite() || config.matrix_roundoff_ulps <= 0.0 {
        return Err(F10SelfActionError::InvalidConfiguration);
    }
    let rule = angular_rule(config.collision).map_err(|_| F10SelfActionError::InvalidConfiguration)?;
    let angular_size = rule
        .incoming_mu
        .len()
        .checked_mul(rule.final_mu.len())
        .and_then(|value| value.checked_mul(rule.azimuth.len()))
        .ok_or(F10SelfActionError::DimensionOverflow)?;

    let logits = decode_pair_logits(grid, pair_cloglog)?;
    let coefficients = spectral_coefficients(grid, &logits)?;
    let native_basis = modal_basis(grid, &grid.nodes).map_err(|_| F10SelfActionError::Foundation)?;
    let p2_nodes: Vec<f64> = grid
        .nodes
        .iter()
        .map(|node| temperature_cm * node)
        .collect();
    let p2_weights: Vec<f64> = grid
        .weights
        .iter()
        .map(|weight| temperature_cm * weight)
        .collect();
    if p2_nodes.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || p2_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(F10SelfActionError::InvalidInput);
    }

    let events = f10_self_events();
    if events.len() != 27 {
        return Err(F10SelfActionError::InvalidInput);
    }
    let row_size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10SelfActionError::DimensionOverflow)?;
    let mut modal = vec![0.0; row_size];
    let mut row_modal = vec![0.0; SELF_ROW_COUNT * row_size];
    let mut whole_reaction_domain_rejections = 0_usize;
    let mut matrix_roundoff_corrections = 0_usize;
    let mut largest_matrix_roundoff_correction = 0.0_f64;
    let mut event_entropy_rate = 0.0_f64;
    let mut event_energy_residual = 0.0_f64;
    let mut event_energy_absolute = 0.0_f64;

    for node_index in 0..grid.order {
        let y1 = grid.nodes[node_index];
        let p1 = temperature_cm * y1;
        let outer_weight =
            temperature_cm.powi(3) * grid.weights[node_index] * y1.powi(2) / TWO_PI_SQUARED;
        if !p1.is_finite() || p1 <= 0.0 || !outer_weight.is_finite() || outer_weight <= 0.0 {
            return Err(F10SelfActionError::InvalidInput);
        }
        let batch = two_body_kinematics(F10KinematicInput {
            p1,
            p2_nodes: &p2_nodes,
            p2_weights: &p2_weights,
            mass2: 0.0,
            mass3: 0.0,
            mass4: 0.0,
            config: config.collision,
        })
        .map_err(|_| F10SelfActionError::Kinematics)?;
        let sample_count = batch.support.len();
        let expected_samples = grid
            .order
            .checked_mul(angular_size)
            .ok_or(F10SelfActionError::DimensionOverflow)?;
        if sample_count != expected_samples {
            return Err(F10SelfActionError::Kinematics);
        }

        let mut valid_positions = Vec::new();
        let mut y3_valid = Vec::new();
        let mut y4_valid = Vec::new();
        let mut rejected_samples = 0_usize;
        for sample in 0..sample_count {
            let y3 = batch.p3_magnitude[sample] / temperature_cm;
            let y4 = batch.p4_magnitude[sample] / temperature_cm;
            let in_domain = batch.support[sample]
                && y3 > 0.0
                && y3 < grid.y_max
                && y4 > 0.0
                && y4 < grid.y_max;
            if in_domain {
                valid_positions.push(sample);
                y3_valid.push(y3);
                y4_valid.push(y4);
            } else if batch.support[sample] {
                rejected_samples = rejected_samples
                    .checked_add(1)
                    .ok_or(F10SelfActionError::DimensionOverflow)?;
            }
        }
        whole_reaction_domain_rejections = whole_reaction_domain_rejections
            .checked_add(
                rejected_samples
                    .checked_mul(events.len())
                    .ok_or(F10SelfActionError::DimensionOverflow)?,
            )
            .ok_or(F10SelfActionError::DimensionOverflow)?;

        let valid_count = valid_positions.len();
        let basis3 = modal_basis(grid, &y3_valid).map_err(|_| F10SelfActionError::Foundation)?;
        let basis4 = modal_basis(grid, &y4_valid).map_err(|_| F10SelfActionError::Foundation)?;
        let outgoing3 =
            interpolated_pair_logits(&coefficients, &basis3, valid_count, grid.order)?;
        let outgoing4 =
            interpolated_pair_logits(&coefficients, &basis4, valid_count, grid.order)?;
        let mut measures = Vec::with_capacity(valid_count);
        for &sample in &valid_positions {
            measures.push(f10_event_measure(F10EventMeasureInput {
                p1,
                p2: batch.p2[sample],
                e2: batch.e2[sample],
                phase_space: batch.phase_space[sample],
                quadrature_weight: batch.quadrature_weight[sample],
                outer_weight,
            })?);
        }

        let rate_count = events
            .len()
            .checked_mul(sample_count)
            .ok_or(F10SelfActionError::DimensionOverflow)?;
        let mut rates = vec![0.0; rate_count];
        let mut matrix_cache = Vec::<MatrixCacheEntry>::new();
        for (event_index, &event) in events.iter().enumerate() {
            let cache_index = match matrix_cache_index(&matrix_cache, event) {
                Some(index) => index,
                None => {
                    matrix_cache.push(build_matrix_cache_entry(
                        event,
                        &batch,
                        config.matrix_roundoff_ulps,
                    )?);
                    matrix_cache.len() - 1
                }
            };
            let matrix = &matrix_cache[cache_index];
            matrix_roundoff_corrections = matrix_roundoff_corrections
                .checked_add(matrix.corrections)
                .ok_or(F10SelfActionError::DimensionOverflow)?;
            largest_matrix_roundoff_correction = largest_matrix_roundoff_correction
                .max(matrix.largest_correction);

            let [species1, species2, species3, species4] = event.legs;
            let u1 = logits[pair_index(species1) * grid.order + node_index];
            let rate_offset = event_index * sample_count;
            for (valid_index, &sample) in valid_positions.iter().enumerate() {
                let p2_index = sample / angular_size;
                let u2 = logits[pair_index(species2) * grid.order + p2_index];
                let u3 = outgoing3[pair_index(species3) * valid_count + valid_index];
                let u4 = outgoing4[pair_index(species4) * valid_count + valid_index];
                let pauli = stable_pauli_gain_minus_loss([u1, u2, u3, u4])?;
                let rate = measures[valid_index] * matrix.values[sample] * pauli;
                if !rate.is_finite() {
                    return Err(F10SelfActionError::NonFiniteOutput);
                }
                rates[rate_offset + sample] = rate;
                let affinity = u1 + u2 - u3 - u4;
                let energy_defect =
                    p1 + batch.p2[sample] - batch.e3[sample] - batch.e4[sample];
                let entropy_increment = rate * affinity;
                let energy_increment = rate * energy_defect;
                if !entropy_increment.is_finite() || !energy_increment.is_finite() {
                    return Err(F10SelfActionError::NonFiniteOutput);
                }
                event_entropy_rate += entropy_increment;
                event_energy_residual += energy_increment;
                event_energy_absolute += energy_increment.abs();
            }
        }

        for (event_index, &event) in events.iter().enumerate() {
            let rate_row = &rates[event_index * sample_count..(event_index + 1) * sample_count];
            let incoming1_sum = rate_row.iter().sum::<f64>();
            let mut p2_sums = vec![0.0; grid.order];
            for (sample, &rate) in rate_row.iter().enumerate() {
                p2_sums[sample / angular_size] += rate;
            }
            let mut leg_modes = [
                vec![0.0; grid.order],
                vec![0.0; grid.order],
                vec![0.0; grid.order],
                vec![0.0; grid.order],
            ];
            for mode in 0..grid.order {
                leg_modes[0][mode] =
                    incoming1_sum * native_basis[node_index * grid.order + mode];
                leg_modes[1][mode] = (0..grid.order)
                    .map(|p2_index| {
                        p2_sums[p2_index] * native_basis[p2_index * grid.order + mode]
                    })
                    .sum();
                leg_modes[2][mode] = valid_positions
                    .iter()
                    .enumerate()
                    .map(|(valid_index, &sample)| {
                        rate_row[sample] * basis3[valid_index * grid.order + mode]
                    })
                    .sum();
                leg_modes[3][mode] = valid_positions
                    .iter()
                    .enumerate()
                    .map(|(valid_index, &sample)| {
                        rate_row[sample] * basis4[valid_index * grid.order + mode]
                    })
                    .sum();
            }

            for (leg_index, (&species, values)) in
                event.legs.iter().zip(leg_modes.iter()).enumerate()
            {
                let sign = if leg_index < 2 { 1.0 } else { -1.0 };
                let target = species_index(species);
                let row = self_event_row(event, species)?;
                for mode in 0..grid.order {
                    let contribution = sign * values[mode];
                    modal[target * grid.order + mode] += contribution;
                    row_modal[(row * SPECIES_COUNT + target) * grid.order + mode] += contribution;
                }
            }
        }
    }

    let native = native_action(grid, &modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10SelfActionError::Foundation)?;
    let mut row_native = vec![0.0; SELF_ROW_COUNT * row_size];
    for row in 0..SELF_ROW_COUNT {
        let converted = native_action(
            grid,
            &row_modal[row * row_size..(row + 1) * row_size],
            SPECIES_COUNT,
            temperature_cm,
        )
        .map_err(|_| F10SelfActionError::Foundation)?;
        row_native[row * row_size..(row + 1) * row_size].copy_from_slice(&converted);
    }
    let moments = action_moments(grid, &native, temperature_cm)?;
    let node_entropy_rate = node_entropy_rate(&modal, &coefficients, grid.order)?;
    let entropy_duality_residual = (node_entropy_rate - event_entropy_rate).abs()
        / (node_entropy_rate.abs() + event_entropy_rate.abs()).max(f64::MIN_POSITIVE);

    let scalars = [
        largest_matrix_roundoff_correction,
        event_entropy_rate,
        node_entropy_rate,
        entropy_duality_residual,
        event_energy_residual,
        event_energy_absolute,
    ];
    if modal
        .iter()
        .chain(&native)
        .chain(&row_modal)
        .chain(&row_native)
        .any(|value| !value.is_finite())
        || scalars.into_iter().any(|value| !value.is_finite())
    {
        return Err(F10SelfActionError::NonFiniteOutput);
    }

    Ok(F10SelfAction {
        modal,
        native,
        row_modal,
        row_native,
        moments,
        whole_reaction_domain_rejections,
        matrix_roundoff_corrections,
        largest_matrix_roundoff_correction,
        event_entropy_rate,
        node_entropy_rate,
        entropy_duality_residual,
        event_energy_residual,
        event_energy_absolute,
    })
}
