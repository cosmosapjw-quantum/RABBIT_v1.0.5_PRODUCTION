//! Analytic spectral-`c` JVP for the admitted neutrino self action.

use super::*;
use crate::f10_action_tangent::F10SpectralTangent;
use crate::f10_kernel_primitives::pauli_logit_gradient;

#[derive(Clone, Debug)]
pub(crate) struct F10SelfActionJvp {
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) moments: F10ActionMoments,
    pub(crate) event_energy_residual: f64,
    pub(crate) event_energy_absolute: f64,
}

fn dot4(left: [f64; 4], right: [f64; 4]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

pub(crate) fn assemble_self_action_c_jvp(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    tangent: &F10SpectralTangent,
    config: F10SelfActionConfig,
) -> Result<F10SelfActionJvp, F10SelfActionError> {
    if !temperature_cm.is_finite() || temperature_cm <= 0.0 {
        return Err(F10SelfActionError::InvalidInput);
    }
    if !config.matrix_roundoff_ulps.is_finite() || config.matrix_roundoff_ulps <= 0.0 {
        return Err(F10SelfActionError::InvalidConfiguration);
    }
    let spectral_size = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10SelfActionError::DimensionOverflow)?;
    if pair_cloglog.len() != spectral_size
        || tangent.logit_delta.len() != spectral_size
        || tangent.logit_modal.len() != spectral_size
    {
        return Err(F10SelfActionError::InvalidInput);
    }

    let rule =
        angular_rule(config.collision).map_err(|_| F10SelfActionError::InvalidConfiguration)?;
    let angular_size = rule
        .incoming_mu
        .len()
        .checked_mul(rule.final_mu.len())
        .and_then(|value| value.checked_mul(rule.azimuth.len()))
        .ok_or(F10SelfActionError::DimensionOverflow)?;

    let logits = decode_pair_logits(grid, pair_cloglog)?;
    let coefficients = spectral_coefficients(grid, &logits)?;
    let native_basis =
        modal_basis(grid, &grid.nodes).map_err(|_| F10SelfActionError::Foundation)?;
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
    if p2_nodes
        .iter()
        .chain(&p2_weights)
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
        for sample in 0..sample_count {
            let y3 = batch.p3_magnitude[sample] / temperature_cm;
            let y4 = batch.p4_magnitude[sample] / temperature_cm;
            if batch.support[sample] && y3 > 0.0 && y3 < grid.y_max && y4 > 0.0 && y4 < grid.y_max {
                valid_positions.push(sample);
                y3_valid.push(y3);
                y4_valid.push(y4);
            }
        }

        let valid_count = valid_positions.len();
        let basis3 = modal_basis(grid, &y3_valid).map_err(|_| F10SelfActionError::Foundation)?;
        let basis4 = modal_basis(grid, &y4_valid).map_err(|_| F10SelfActionError::Foundation)?;
        let outgoing3 = interpolated_pair_logits(&coefficients, &basis3, valid_count, grid.order)?;
        let outgoing4 = interpolated_pair_logits(&coefficients, &basis4, valid_count, grid.order)?;
        let delta_outgoing3 = tangent
            .interpolate_all_pairs(&basis3, valid_count, grid.order)
            .map_err(|_| F10SelfActionError::Foundation)?;
        let delta_outgoing4 = tangent
            .interpolate_all_pairs(&basis4, valid_count, grid.order)
            .map_err(|_| F10SelfActionError::Foundation)?;

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
        let mut delta_rates = vec![0.0; rate_count];
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
            let [species1, species2, species3, species4] = event.legs;
            let pair1 = pair_index(species1);
            let pair2 = pair_index(species2);
            let pair3 = pair_index(species3);
            let pair4 = pair_index(species4);
            let u1 = logits[pair1 * grid.order + node_index];
            let du1 = tangent.logit_delta[pair1 * grid.order + node_index];
            let rate_offset = event_index * sample_count;
            for (valid_index, &sample) in valid_positions.iter().enumerate() {
                let p2_index = sample / angular_size;
                let u2 = logits[pair2 * grid.order + p2_index];
                let u3 = outgoing3[pair3 * valid_count + valid_index];
                let u4 = outgoing4[pair4 * valid_count + valid_index];
                let du2 = tangent.logit_delta[pair2 * grid.order + p2_index];
                let du3 = delta_outgoing3[pair3 * valid_count + valid_index];
                let du4 = delta_outgoing4[pair4 * valid_count + valid_index];
                let gradient = pauli_logit_gradient([u1, u2, u3, u4])?;
                let delta_pauli = dot4(gradient, [du1, du2, du3, du4]);
                let delta_rate = measures[valid_index] * matrix.values[sample] * delta_pauli;
                if !delta_rate.is_finite() {
                    return Err(F10SelfActionError::NonFiniteOutput);
                }
                delta_rates[rate_offset + sample] = delta_rate;
                let energy_defect = p1 + batch.p2[sample] - batch.e3[sample] - batch.e4[sample];
                let delta_energy = delta_rate * energy_defect;
                if !delta_energy.is_finite() {
                    return Err(F10SelfActionError::NonFiniteOutput);
                }
                event_energy_residual += delta_energy;
                event_energy_absolute += delta_energy.abs();
            }
        }

        for (event_index, &event) in events.iter().enumerate() {
            let rate_row =
                &delta_rates[event_index * sample_count..(event_index + 1) * sample_count];
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
                leg_modes[0][mode] = incoming1_sum * native_basis[node_index * grid.order + mode];
                leg_modes[1][mode] = (0..grid.order)
                    .map(|p2_index| p2_sums[p2_index] * native_basis[p2_index * grid.order + mode])
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
                for mode in 0..grid.order {
                    modal[target * grid.order + mode] += sign * values[mode];
                }
            }
        }
    }

    let native = native_action(grid, &modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10SelfActionError::Foundation)?;
    let moments = action_moments(grid, &native, temperature_cm)?;
    if modal.iter().chain(&native).any(|value| !value.is_finite())
        || !event_energy_residual.is_finite()
        || !event_energy_absolute.is_finite()
    {
        return Err(F10SelfActionError::NonFiniteOutput);
    }

    Ok(F10SelfActionJvp {
        modal,
        native,
        moments,
        event_energy_residual,
        event_energy_absolute,
    })
}
