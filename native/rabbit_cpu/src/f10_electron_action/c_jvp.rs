//! Analytic spectral-`c` JVP for the admitted neutrino-electron action.

use super::*;
use crate::f10_action_tangent::F10SpectralTangent;
use crate::f10_kernel_primitives::pauli_logit_gradient;

#[derive(Clone, Debug)]
pub(crate) struct F10ElectronActionJvp {
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) moments: F10ElectronActionMoments,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
}

fn dot4(left: [f64; 4], right: [f64; 4]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

pub(crate) fn assemble_electron_action_c_jvp(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    tangent: &F10SpectralTangent,
    config: F10ElectronActionConfig,
) -> Result<F10ElectronActionJvp, F10ElectronActionError> {
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
    let spectral_size = PAIR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    if pair_cloglog.len() != spectral_size
        || tangent.logit_delta.len() != spectral_size
        || tangent.logit_modal.len() != spectral_size
    {
        return Err(F10ElectronActionError::InvalidInput);
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

    let species_modal_size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let mut modal = vec![0.0; species_modal_size];
    let mut neutrino_energy_transfer = 0.0_f64;
    let mut electromagnetic_energy_transfer = 0.0_f64;

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
        for sample in 0..elastic_sample_count {
            let y3 = elastic_batch.p3_magnitude[sample] / temperature_cm;
            if elastic_batch.support[sample] && y3 > 0.0 && y3 < grid.y_max {
                elastic_valid_positions.push(sample);
                elastic_y3.push(y3);
            }
        }
        let elastic_valid_count = elastic_valid_positions.len();
        let elastic_basis3 =
            modal_basis(grid, &elastic_y3).map_err(|_| F10ElectronActionError::Foundation)?;
        let elastic_outgoing = interpolated_pair_logits(
            &coefficients,
            &elastic_basis3,
            elastic_valid_count,
            grid.order,
        )?;
        let delta_elastic_outgoing = tangent
            .interpolate_all_pairs(&elastic_basis3, elastic_valid_count, grid.order)
            .map_err(|_| F10ElectronActionError::Foundation)?;
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
        let mut delta_elastic_rates = vec![0.0; elastic_rate_count];
        for (event_index, &event) in elastic_events.iter().enumerate() {
            let (matrix, _, _) = matrix_values(
                event,
                &elastic_batch,
                config.electron_mass_mev,
                config.matrix_roundoff_ulps,
            )?;
            let species = event.target;
            let pair = pair_index(species);
            let u1 = logits[pair * grid.order + node_index];
            let du1 = tangent.logit_delta[pair * grid.order + node_index];
            let rate_offset = event_index * elastic_sample_count;
            for (valid_index, &sample) in elastic_valid_positions.iter().enumerate() {
                let u2 = -elastic_batch.e2[sample] / temperature_gamma;
                let u3 = elastic_outgoing[pair * elastic_valid_count + valid_index];
                let u4 = -elastic_batch.e4[sample] / temperature_gamma;
                let du3 = delta_elastic_outgoing[pair * elastic_valid_count + valid_index];
                let gradient = pauli_logit_gradient([u1, u2, u3, u4])?;
                let delta_pauli = dot4(gradient, [du1, 0.0, du3, 0.0]);
                let delta_rate = elastic_measures[valid_index] * matrix[sample] * delta_pauli;
                if !delta_rate.is_finite() {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                delta_elastic_rates[rate_offset + sample] = delta_rate;
                neutrino_energy_transfer += delta_rate * (p1 - elastic_batch.e3[sample]);
                electromagnetic_energy_transfer +=
                    delta_rate * (elastic_batch.e2[sample] - elastic_batch.e4[sample]);
            }
        }

        for (event_index, &event) in elastic_events.iter().enumerate() {
            let rate_row = &delta_elastic_rates
                [event_index * elastic_sample_count..(event_index + 1) * elastic_sample_count];
            let incoming_sum = rate_row.iter().sum::<f64>();
            let target = species_index(event.target);
            for mode in 0..grid.order {
                let incoming = incoming_sum * native_basis[node_index * grid.order + mode];
                let outgoing = elastic_valid_positions
                    .iter()
                    .enumerate()
                    .map(|(valid_index, &sample)| {
                        rate_row[sample] * elastic_basis3[valid_index * grid.order + mode]
                    })
                    .sum::<f64>();
                modal[target * grid.order + mode] += incoming - outgoing;
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
        let mut delta_pair_rates = vec![0.0; pair_rate_count];
        for (pair_event_index, &event) in pair_events.iter().enumerate() {
            let (matrix, _, _) = matrix_values(
                event,
                &pair_batch,
                config.electron_mass_mev,
                config.matrix_roundoff_ulps,
            )?;
            let target = event.target;
            let partner = target.cp_partner();
            let target_pair = pair_index(target);
            let partner_pair = pair_index(partner);
            let u1 = logits[target_pair * grid.order + node_index];
            let du1 = tangent.logit_delta[target_pair * grid.order + node_index];
            let rate_offset = pair_event_index * pair_sample_count;
            for (valid_index, &sample) in pair_valid_positions.iter().enumerate() {
                let p2_index = sample / angular_size;
                let u2 = logits[partner_pair * grid.order + p2_index];
                let u3 = -pair_batch.e3[sample] / temperature_gamma;
                let u4 = -pair_batch.e4[sample] / temperature_gamma;
                let du2 = tangent.logit_delta[partner_pair * grid.order + p2_index];
                let gradient = pauli_logit_gradient([u1, u2, u3, u4])?;
                let delta_pauli = dot4(gradient, [du1, du2, 0.0, 0.0]);
                let delta_rate = pair_measures[valid_index] * matrix[sample] * delta_pauli;
                if !delta_rate.is_finite() {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                delta_pair_rates[rate_offset + sample] = delta_rate;
                neutrino_energy_transfer += delta_rate * (p1 + pair_batch.p2[sample]);
                electromagnetic_energy_transfer +=
                    delta_rate * (-pair_batch.e3[sample] - pair_batch.e4[sample]);
            }
        }

        for (pair_event_index, &event) in pair_events.iter().enumerate() {
            let rate_row = &delta_pair_rates
                [pair_event_index * pair_sample_count..(pair_event_index + 1) * pair_sample_count];
            let incoming1_sum = rate_row.iter().sum::<f64>();
            let mut p2_sums = vec![0.0; grid.order];
            for (sample, &rate) in rate_row.iter().enumerate() {
                p2_sums[sample / angular_size] += rate;
            }
            for (species, incoming1) in [(event.target, true), (event.target.cp_partner(), false)] {
                let target = species_index(species);
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
                    modal[target * grid.order + mode] += contribution;
                }
            }
        }
    }

    let native = native_action(grid, &modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let moments = action_moments(grid, &native, temperature_cm)?;
    let first_law_residual = neutrino_energy_transfer + electromagnetic_energy_transfer;
    if modal
        .iter()
        .chain(&native)
        .any(|value| !value.is_finite())
        || [
            neutrino_energy_transfer,
            electromagnetic_energy_transfer,
            first_law_residual,
        ]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return Err(F10ElectronActionError::NonFiniteOutput);
    }

    Ok(F10ElectronActionJvp {
        modal,
        native,
        moments,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
    })
}
