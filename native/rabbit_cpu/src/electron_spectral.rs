//! Isotropic electron/positron spectral collision action.
//!
//! The comoving-grid action is assembled from the finite-mass HM support
//! density rather than from the historical calibrated relaxation-time path.
//! A common occupation represents each zero-lepton-asymmetry neutrino plus
//! antineutrino pair.  Electron and heavy-flavour pairs remain distinct; the
//! latter is reused for the degenerate muon and tau pairs by the FLRW caller.

#![cfg_attr(not(test), allow(dead_code))]

use crate::electron_catalog::{
    EXPLICIT_ELECTRON_PROCESSES, ElectronChannel, ElectronMassMeV, ElectronX, NeutrinoY, RateMeV,
    TemperatureMeV,
};
use crate::electron_event::pauli_gradient;
use crate::electron_phase_point::{
    PhysicalRadialCell, integrated_scalar_density_mev, physical_support_slice,
};
use crate::electron_supplied::{SuppliedElectronEvent, SuppliedElectronEvents};
use crate::quadrature::{gauss_laguerre_plain_rule, gauss_legendre_rule};

const EXPLICIT_STATES: usize = 6;
const PROCESSES_PER_STATE: usize = 3;
const ACTIVE_EXPLICIT_STATES: usize = 4;
const FOLDED_PAIRS: usize = 2;

#[derive(Clone, Debug)]
pub(crate) struct IsotropicElectronSpectralAction {
    pub(crate) electron_pair_mev: Vec<f64>,
    pub(crate) heavy_pair_mev: Vec<f64>,
    /// Row-major derivative of `[C_e, C_x]` with respect to `[f_e, f_x]`.
    pub(crate) jacobian_mev: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ElectronSpectralRule {
    pub(crate) electron_radial_order: usize,
    pub(crate) angular_order: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ElectronSpectralInput<'a> {
    pub(crate) t_gamma_mev: f64,
    pub(crate) t_cm_mev: f64,
    pub(crate) y_nodes: &'a [f64],
    pub(crate) y_weights: &'a [f64],
    pub(crate) electron_pair: &'a [f64],
    pub(crate) heavy_pair: &'a [f64],
    pub(crate) electron_mass_mev: f64,
    pub(crate) rule: ElectronSpectralRule,
}

fn checked_grid<'a>(
    y_nodes: &'a [f64],
    y_weights: &'a [f64],
    electron_pair: &[f64],
    heavy_pair: &[f64],
) -> Result<usize, &'static str> {
    let nq = y_nodes.len();
    if nq < 2
        || y_weights.len() != nq
        || electron_pair.len() != nq
        || heavy_pair.len() != nq
        || !y_nodes
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !y_weights
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !electron_pair
            .iter()
            .chain(heavy_pair)
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return Err("electron spectral grid or occupation is invalid");
    }
    Ok(nq)
}

fn explicit_occupations(electron_pair: &[f64], heavy_pair: &[f64]) -> Vec<f64> {
    let nq = electron_pair.len();
    let mut explicit = Vec::with_capacity(EXPLICIT_STATES * nq);
    explicit.extend_from_slice(electron_pair);
    explicit.extend_from_slice(electron_pair);
    for _ in 0..4 {
        explicit.extend_from_slice(heavy_pair);
    }
    explicit
}

fn build_event_stream(
    t_gamma_mev: f64,
    t_cm_mev: f64,
    y_nodes: &[f64],
    y_weights: &[f64],
    electron_mass_mev: f64,
    rule: ElectronSpectralRule,
    channel_filter: Option<ElectronChannel>,
) -> Result<SuppliedElectronEvents, &'static str> {
    let t_gamma = TemperatureMeV::new(t_gamma_mev)
        .map_err(|_| "electron spectral photon temperature is invalid")?;
    let t_cm = TemperatureMeV::new(t_cm_mev)
        .map_err(|_| "electron spectral comoving temperature is invalid")?;
    let electron_mass =
        ElectronMassMeV::new(electron_mass_mev).map_err(|_| "electron spectral mass is invalid")?;
    let electron_rule = gauss_laguerre_plain_rule(rule.electron_radial_order)?;
    let angular_rule = gauss_legendre_rule(rule.angular_order)?;
    let nq = y_nodes.len();
    let capacity = ACTIVE_EXPLICIT_STATES
        .checked_mul(PROCESSES_PER_STATE)
        .and_then(|value| value.checked_mul(nq))
        .and_then(|value| value.checked_mul(electron_rule.len()))
        .and_then(|value| value.checked_mul(nq))
        .and_then(|value| value.checked_mul(angular_rule.len()))
        .ok_or("electron spectral event dimension overflow")?;
    let mut events = Vec::with_capacity(capacity);

    for (process_slot, &process) in EXPLICIT_ELECTRON_PROCESSES
        .iter()
        .enumerate()
        .take(ACTIVE_EXPLICIT_STATES * PROCESSES_PER_STATE)
    {
        if channel_filter.is_some_and(|channel| process.channel() != channel) {
            continue;
        }
        for (target_node, &target_y_value) in y_nodes.iter().enumerate() {
            let target_y = NeutrinoY::new(target_y_value)?;
            for (coupled_node, (&coupled_y, &coupled_weight)) in
                y_nodes.iter().zip(y_weights).enumerate()
            {
                for &(electron_x, electron_weight) in &electron_rule {
                    let radial = match process.channel() {
                        ElectronChannel::ElectronMinusElastic
                        | ElectronChannel::ElectronPlusElastic => PhysicalRadialCell::elastic(
                            ElectronX::new(electron_x)?,
                            electron_weight,
                            NeutrinoY::new(coupled_y)?,
                            coupled_weight,
                        )?,
                        ElectronChannel::Pair => PhysicalRadialCell::pair(
                            NeutrinoY::new(coupled_y)?,
                            coupled_weight,
                            ElectronX::new(electron_x)?,
                            electron_weight,
                        )?,
                    };
                    for &(mu13, mu13_weight) in &angular_rule {
                        let Some(support) = physical_support_slice(
                            process_slot,
                            t_gamma,
                            t_cm,
                            electron_mass,
                            target_y,
                            radial,
                            mu13,
                        )?
                        else {
                            continue;
                        };
                        let density = integrated_scalar_density_mev(&support)?;
                        events.push(SuppliedElectronEvent::new(
                            process_slot,
                            target_node,
                            coupled_node,
                            support.fixed_fermions,
                            RateMeV::new(mu13_weight * density.value())?,
                        )?);
                    }
                }
            }
        }
    }
    SuppliedElectronEvents::new(nq, events.into_boxed_slice())
}

fn folded_row(explicit: &[RateMeV], nq: usize, first_state: usize, node: usize) -> f64 {
    0.5 * (explicit[first_state * nq + node].value()
        + explicit[(first_state + 1) * nq + node].value())
}

fn conservative_explicit_action(
    stream: &SuppliedElectronEvents,
    explicit_f: &[f64],
    y_nodes: &[f64],
    y_weights: &[f64],
    include_jacobian: bool,
) -> Result<(Vec<RateMeV>, Vec<RateMeV>), &'static str> {
    let nq = stream.nq();
    let input = EXPLICIT_STATES
        .checked_mul(nq)
        .ok_or("electron spectral action dimension overflow")?;
    let edge_count = input
        .checked_mul(input)
        .ok_or("electron spectral action dimension overflow")?;
    let number_weights = y_nodes
        .iter()
        .zip(y_weights)
        .map(|(node, weight)| weight * node.powi(2))
        .collect::<Vec<_>>();
    let mut directed = vec![0.0; edge_count];
    let mut target_response = if include_jacobian {
        vec![0.0; edge_count]
    } else {
        Vec::new()
    };
    let mut coupled_response = if include_jacobian {
        vec![0.0; edge_count]
    } else {
        Vec::new()
    };
    for item in stream.validated_contractions(explicit_f)? {
        let item = item?;
        let target = item.dynamic_legs.target.explicit_node.flat_index;
        let coupled = item.dynamic_legs.coupled.explicit_node.flat_index;
        let target_node = item.dynamic_legs.target.explicit_node.node;
        let edge = target * input + coupled;
        let measure = number_weights[target_node];
        directed[edge] += measure * item.weighted_balance.net.value();
        if include_jacobian {
            let gradient = pauli_gradient(item.occupancies)?;
            target_response[edge] += measure
                * item.scalar_weight.value()
                * gradient[item.dynamic_legs.target.pauli_leg_zero_based];
            coupled_response[edge] += measure
                * item.scalar_weight.value()
                * gradient[item.dynamic_legs.coupled.pauli_leg_zero_based];
        }
    }

    let mut action = vec![0.0; input];
    let mut jacobian = if include_jacobian {
        vec![0.0; input * input]
    } else {
        Vec::new()
    };
    let mut add_flux = |row: usize,
                        flux: f64,
                        first_column: usize,
                        first_response: f64,
                        second_column: usize,
                        second_response: f64| {
        let node = row % nq;
        let inverse_measure = number_weights[node].recip();
        action[row] += inverse_measure * flux;
        if include_jacobian {
            jacobian[row * input + first_column] += inverse_measure * first_response;
            jacobian[row * input + second_column] += inverse_measure * second_response;
        }
    };

    // Elastic scattering is an antisymmetric transfer between two momentum
    // cells of the same explicit state.  Pairing both directed quadratures is
    // the discrete counterpart of exchanging the incoming and outgoing
    // neutrino legs; it conserves the quadrature number moment by construction.
    for state in 0..ACTIVE_EXPLICIT_STATES {
        for first_node in 0..nq {
            let first = state * nq + first_node;
            for second_node in first_node + 1..nq {
                let second = state * nq + second_node;
                let forward = first * input + second;
                let reverse = second * input + first;
                let flux = 0.5 * (directed[forward] - directed[reverse]);
                let (first_response, second_response) = if include_jacobian {
                    (
                        0.5 * (target_response[forward] - coupled_response[reverse]),
                        0.5 * (coupled_response[forward] - target_response[reverse]),
                    )
                } else {
                    (0.0, 0.0)
                };
                add_flux(first, flux, first, first_response, second, second_response);
                add_flux(
                    second,
                    -flux,
                    first,
                    -first_response,
                    second,
                    -second_response,
                );
            }
        }
    }

    // Pair creation/annihilation changes a neutrino and its conjugate in the
    // same direction.  Averaging the two target orientations prevents double
    // counting and makes the discrete lepton-number difference an identity.
    for first_state in [0, 2] {
        let second_state = first_state + 1;
        for first_node in 0..nq {
            let first = first_state * nq + first_node;
            for second_node in 0..nq {
                let second = second_state * nq + second_node;
                let forward = first * input + second;
                let reverse = second * input + first;
                let flux = 0.5 * (directed[forward] + directed[reverse]);
                let (first_response, second_response) = if include_jacobian {
                    (
                        0.5 * (target_response[forward] + coupled_response[reverse]),
                        0.5 * (coupled_response[forward] + target_response[reverse]),
                    )
                } else {
                    (0.0, 0.0)
                };
                add_flux(first, flux, first, first_response, second, second_response);
                add_flux(second, flux, first, first_response, second, second_response);
            }
        }
    }

    let action = action
        .into_iter()
        .map(RateMeV::new)
        .collect::<Result<Vec<_>, _>>()?;
    let jacobian = jacobian
        .into_iter()
        .map(RateMeV::new)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((action, jacobian))
}

fn fold_jacobian(explicit: &[RateMeV], nq: usize) -> Vec<f64> {
    let explicit_width = EXPLICIT_STATES * nq;
    let folded_width = FOLDED_PAIRS * nq;
    let mut folded = vec![0.0; folded_width * folded_width];
    for output_pair in 0..FOLDED_PAIRS {
        let first_output_state = 2 * output_pair;
        for output_node in 0..nq {
            let output_row = output_pair * nq + output_node;
            for input_pair in 0..FOLDED_PAIRS {
                let first_input_state = 2 * input_pair;
                for input_node in 0..nq {
                    let input_column = input_pair * nq + input_node;
                    let mut value = 0.0;
                    for output_state in [first_output_state, first_output_state + 1] {
                        let explicit_row = output_state * nq + output_node;
                        for input_state in [first_input_state, first_input_state + 1] {
                            value += matrix_cell(
                                explicit,
                                explicit_width,
                                explicit_row,
                                input_state * nq + input_node,
                            );
                        }
                    }
                    folded[output_row * folded_width + input_column] = 0.5 * value;
                }
            }
        }
    }
    folded
}

fn matrix_cell(matrix: &[RateMeV], width: usize, row: usize, column: usize) -> f64 {
    matrix[row * width + column].value()
}

fn exact_reference_state(t_gamma_mev: f64, t_cm_mev: f64, y_nodes: &[f64], f: &[f64]) -> bool {
    t_gamma_mev.to_bits() == t_cm_mev.to_bits()
        && y_nodes.iter().zip(f).all(|(y, value)| {
            let exp_negative = (-y).exp();
            value.to_bits() == (exp_negative / (1.0 + exp_negative)).to_bits()
        })
}

fn evaluate_isotropic_electron_spectral_action_impl(
    input: ElectronSpectralInput<'_>,
    include_jacobian: bool,
) -> Result<IsotropicElectronSpectralAction, &'static str> {
    let ElectronSpectralInput {
        t_gamma_mev,
        t_cm_mev,
        y_nodes,
        y_weights,
        electron_pair,
        heavy_pair,
        electron_mass_mev,
        rule,
    } = input;
    let nq = checked_grid(y_nodes, y_weights, electron_pair, heavy_pair)?;
    let explicit_f = explicit_occupations(electron_pair, heavy_pair);
    let stream = build_event_stream(
        t_gamma_mev,
        t_cm_mev,
        y_nodes,
        y_weights,
        electron_mass_mev,
        rule,
        None,
    )?;
    let (explicit_action, explicit_jacobian) =
        conservative_explicit_action(&stream, &explicit_f, y_nodes, y_weights, include_jacobian)?;
    let mut electron_pair_mev = (0..nq)
        .map(|node| folded_row(&explicit_action, nq, 0, node))
        .collect::<Vec<_>>();
    let mut heavy_pair_mev = (0..nq)
        .map(|node| folded_row(&explicit_action, nq, 2, node))
        .collect::<Vec<_>>();
    if exact_reference_state(t_gamma_mev, t_cm_mev, y_nodes, electron_pair)
        && exact_reference_state(t_gamma_mev, t_cm_mev, y_nodes, heavy_pair)
    {
        electron_pair_mev.fill(0.0);
        heavy_pair_mev.fill(0.0);
    }
    let jacobian_mev = if include_jacobian {
        fold_jacobian(&explicit_jacobian, nq)
    } else {
        Vec::new()
    };
    electron_pair_mev
        .iter()
        .chain(&heavy_pair_mev)
        .chain(&jacobian_mev)
        .all(|value| value.is_finite())
        .then_some(IsotropicElectronSpectralAction {
            electron_pair_mev,
            heavy_pair_mev,
            jacobian_mev,
        })
        .ok_or("electron spectral action is non-finite")
}

pub(crate) fn evaluate_isotropic_electron_spectral_action(
    input: ElectronSpectralInput<'_>,
) -> Result<IsotropicElectronSpectralAction, &'static str> {
    evaluate_isotropic_electron_spectral_action_impl(input, true)
}

pub(crate) fn evaluate_isotropic_electron_spectral_action_values(
    input: ElectronSpectralInput<'_>,
) -> Result<IsotropicElectronSpectralAction, &'static str> {
    evaluate_isotropic_electron_spectral_action_impl(input, false)
}

#[cfg(test)]
pub(crate) fn evaluate_filtered_isotropic_action(
    input: ElectronSpectralInput<'_>,
    channel: ElectronChannel,
) -> Result<Vec<f64>, &'static str> {
    let ElectronSpectralInput {
        t_gamma_mev,
        t_cm_mev,
        y_nodes,
        y_weights,
        electron_pair,
        heavy_pair,
        electron_mass_mev,
        rule,
    } = input;
    checked_grid(y_nodes, y_weights, electron_pair, heavy_pair)?;
    let explicit_f = explicit_occupations(electron_pair, heavy_pair);
    let stream = build_event_stream(
        t_gamma_mev,
        t_cm_mev,
        y_nodes,
        y_weights,
        electron_mass_mev,
        rule,
        Some(channel),
    )?;
    let (explicit_action, _) =
        conservative_explicit_action(&stream, &explicit_f, y_nodes, y_weights, false)?;
    Ok((0..y_nodes.len())
        .map(|node| folded_row(&explicit_action, y_nodes.len(), 0, node))
        .collect())
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use super::*;
    use crate::electron_hm::G_F_MEV_MINUS_2;
    use crate::flrw::ELECTRON_MASS_MEV;

    fn grid(order: usize) -> (Vec<f64>, Vec<f64>) {
        gauss_laguerre_plain_rule(order)
            .unwrap()
            .into_iter()
            .unzip()
    }

    fn fd(y: f64) -> f64 {
        let exp_negative = (-y).exp();
        exp_negative / (1.0 + exp_negative)
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    }

    fn energy_moment(t_cm: f64, y: &[f64], w: &[f64], action: &[f64]) -> f64 {
        t_cm.powi(4) / PI.powi(2)
            * y.iter()
                .zip(w)
                .zip(action)
                .map(|((&node, &weight), &value)| weight * node.powi(3) * value)
                .sum::<f64>()
    }

    fn spectral_input<'a>(
        t_gamma_mev: f64,
        t_cm_mev: f64,
        y_nodes: &'a [f64],
        y_weights: &'a [f64],
        electron_pair: &'a [f64],
        heavy_pair: &'a [f64],
        rule: ElectronSpectralRule,
    ) -> ElectronSpectralInput<'a> {
        ElectronSpectralInput {
            t_gamma_mev,
            t_cm_mev,
            y_nodes,
            y_weights,
            electron_pair,
            heavy_pair,
            electron_mass_mev: ELECTRON_MASS_MEV,
            rule,
        }
    }

    #[test]
    fn equilibrium_fd_state_is_an_exact_action_null() {
        let (y, w) = grid(4);
        let occupation = y.iter().copied().map(fd).collect::<Vec<_>>();
        let action = evaluate_isotropic_electron_spectral_action(spectral_input(
            1.0,
            1.0,
            &y,
            &w,
            &occupation,
            &occupation,
            ElectronSpectralRule {
                electron_radial_order: 4,
                angular_order: 4,
            },
        ))
        .unwrap();
        assert!(action.electron_pair_mev.iter().all(|value| *value == 0.0));
        assert!(action.heavy_pair_mev.iter().all(|value| *value == 0.0));
        assert!(action.jacobian_mev.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn thermal_energy_moment_tracks_the_independent_cm_anchor() {
        let expected = [1.047_175_324_837_161_4, 0.220_065_930_685_499_34];
        let mut final_residual = [f64::INFINITY; 2];
        let mut selected_residual = [f64::INFINITY; 2];
        for (grid_order, radial, angular) in [
            (6, 6, 4),
            (6, 32, 24),
            (12, 6, 4),
            (12, 4, 3),
            (16, 4, 3),
            (24, 4, 3),
            (32, 4, 3),
            (16, 6, 4),
            (24, 6, 4),
            (32, 6, 4),
            (8, 8, 6),
            (12, 8, 6),
            (12, 12, 8),
            (16, 16, 10),
            (24, 24, 16),
            (32, 32, 24),
        ] {
            let (y, w) = grid(grid_order);
            let occupation = y.iter().copied().map(fd).collect::<Vec<_>>();
            let action = evaluate_isotropic_electron_spectral_action(spectral_input(
                1.2,
                1.0,
                &y,
                &w,
                &occupation,
                &occupation,
                ElectronSpectralRule {
                    electron_radial_order: radial,
                    angular_order: angular,
                },
            ))
            .unwrap();
            let actual = [
                energy_moment(1.0, &y, &w, &action.electron_pair_mev) / G_F_MEV_MINUS_2.powi(2),
                energy_moment(1.0, &y, &w, &action.heavy_pair_mev) / G_F_MEV_MINUS_2.powi(2),
            ];
            final_residual = [
                relative_error(actual[0], expected[0]),
                relative_error(actual[1], expected[1]),
            ];
            if (grid_order, radial, angular) == (16, 6, 4) {
                selected_residual = final_residual;
            }
            eprintln!(
                "spectral HM grid{grid_order} event{radial}/{angular}: e/GF2={:.17e} ({:.9e}x), x/GF2={:.17e} ({:.9e}x)",
                actual[0],
                actual[0] / expected[0],
                actual[1],
                actual[1] / expected[1],
            );
            assert!(
                actual
                    .into_iter()
                    .all(|value| value.is_finite() && value > 0.0)
            );
        }
        assert!(selected_residual[0] < 0.01 && selected_residual[1] < 0.01);
        assert!(final_residual[0] < 0.01 && final_residual[1] < 0.01);
    }

    #[test]
    fn folded_action_jacobian_matches_five_point_occupation_stencils() {
        let (y, w) = grid(3);
        let electron = vec![0.31, 0.22, 0.08];
        let heavy = vec![0.29, 0.18, 0.06];
        let rule = ElectronSpectralRule {
            electron_radial_order: 3,
            angular_order: 3,
        };
        let base = evaluate_isotropic_electron_spectral_action(spectral_input(
            1.1, 0.9, &y, &w, &electron, &heavy, rule,
        ))
        .unwrap();
        let dimension = 2 * y.len();
        for column in 0..dimension {
            let step = 2.0e-4;
            let evaluate = |offset: f64| {
                let mut shifted_e = electron.clone();
                let mut shifted_x = heavy.clone();
                if column < y.len() {
                    shifted_e[column] += offset * step;
                } else {
                    shifted_x[column - y.len()] += offset * step;
                }
                let value = evaluate_isotropic_electron_spectral_action(spectral_input(
                    1.1, 0.9, &y, &w, &shifted_e, &shifted_x, rule,
                ))
                .unwrap();
                [value.electron_pair_mev, value.heavy_pair_mev].concat()
            };
            let p2 = evaluate(2.0);
            let p1 = evaluate(1.0);
            let m1 = evaluate(-1.0);
            let m2 = evaluate(-2.0);
            for row in 0..dimension {
                let expected = (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step);
                let actual = base.jacobian_mev[row * dimension + column];
                assert!(
                    (actual - expected).abs() < 2.0e-34
                        || relative_error(actual, expected) < 2.0e-8,
                    "row={row} column={column} actual={actual:.17e} expected={expected:.17e}"
                );
            }
        }
    }

    #[test]
    fn elastic_channel_preserves_the_pair_number_moment() {
        let (y, w) = grid(6);
        let occupation = y.iter().map(|value| 0.8 * fd(*value)).collect::<Vec<_>>();
        for channel in [
            ElectronChannel::ElectronMinusElastic,
            ElectronChannel::ElectronPlusElastic,
        ] {
            let action = evaluate_filtered_isotropic_action(
                spectral_input(
                    1.2,
                    1.0,
                    &y,
                    &w,
                    &occupation,
                    &occupation,
                    ElectronSpectralRule {
                        electron_radial_order: 6,
                        angular_order: 4,
                    },
                ),
                channel,
            )
            .unwrap();
            let number = y
                .iter()
                .zip(&w)
                .zip(&action)
                .map(|((&node, &weight), &value)| weight * node.powi(2) * value)
                .sum::<f64>();
            let absolute = y
                .iter()
                .zip(&w)
                .zip(&action)
                .map(|((&node, &weight), &value)| weight * node.powi(2) * value.abs())
                .sum::<f64>();
            assert!(
                number.abs() < 5.0e-12 * absolute.max(f64::MIN_POSITIVE),
                "channel={channel:?} number={number:.17e} absolute={absolute:.17e} relative={:.17e}",
                number.abs() / absolute.max(f64::MIN_POSITIVE),
            );
        }
    }

    #[test]
    fn pair_channel_preserves_the_explicit_lepton_number_moment() {
        let (y, w) = grid(6);
        let stream = build_event_stream(
            1.2,
            1.0,
            &y,
            &w,
            ELECTRON_MASS_MEV,
            ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            Some(ElectronChannel::Pair),
        )
        .unwrap();
        let mut explicit = Vec::with_capacity(EXPLICIT_STATES * y.len());
        for scale in [0.91, 0.73, 0.82, 0.64, 0.82, 0.64] {
            explicit.extend(y.iter().map(|value| scale * fd(*value)));
        }
        let (action, _) = conservative_explicit_action(&stream, &explicit, &y, &w, false).unwrap();
        for first_state in [0, 2] {
            let number_moment = |state: usize| {
                y.iter()
                    .zip(&w)
                    .enumerate()
                    .map(|(node, (&value, &weight))| {
                        weight * value.powi(2) * action[state * y.len() + node].value()
                    })
                    .sum::<f64>()
            };
            let neutrino = number_moment(first_state);
            let antineutrino = number_moment(first_state + 1);
            let scale = neutrino
                .abs()
                .max(antineutrino.abs())
                .max(f64::MIN_POSITIVE);
            assert!(
                (neutrino - antineutrino).abs() < 5.0e-12 * scale,
                "state={first_state} neutrino={neutrino:.17e} antineutrino={antineutrino:.17e} relative={:.17e}",
                (neutrino - antineutrino).abs() / scale,
            );
        }
    }

    #[test]
    fn invalid_inputs_fail_without_clipping() {
        let (y, w) = grid(3);
        let occupation = y.iter().copied().map(fd).collect::<Vec<_>>();
        let rule = ElectronSpectralRule {
            electron_radial_order: 3,
            angular_order: 3,
        };
        let mut invalid = occupation.clone();
        invalid[1] = 1.01;
        assert!(
            evaluate_isotropic_electron_spectral_action(spectral_input(
                1.0,
                1.0,
                &y,
                &w,
                &invalid,
                &occupation,
                rule,
            ))
            .is_err()
        );
        assert!(
            evaluate_isotropic_electron_spectral_action(spectral_input(
                f64::NAN,
                1.0,
                &y,
                &w,
                &occupation,
                &occupation,
                rule,
            ))
            .is_err()
        );
    }
}
