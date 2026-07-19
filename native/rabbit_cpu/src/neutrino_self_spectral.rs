//! Classical diagonal neutrino self-collisions.
//!
//! This F10C module implements the frozen nine-row zero-asymmetry classical
//! diagonal catalogue on the electron-pair shape `E` and shared heavy-pair
//! shape `X`.  It integrates each physical event in centre-of-momentum
//! coordinates `(y1, y2, mu12, z3*)`, evaluates both off-grid outgoing logits
//! with the same nonnegative linear basis used for deposition, and therefore
//! conserves degeneracy-weighted number and energy event by event.
//!
//! The matrix-element coefficient `32` and elastic time-orientation factor
//! `1/2` give the global four-leg weak-form coefficient `16`.  Four identical
//! target-leg roles reproduce the corrected tagged coefficient `64`; two
//! target roles in a distinct-flavour family reproduce tagged coefficient
//! `32`.  Projection over the four equal heavy species therefore gives
//! `(4 identical + 2 distinct)/4 = 3/2` times the identical-only heavy action.
//! Opposite-sign elastic and pair-conversion rows use the independently
//! checked azimuth-averaged `K_t=(p1.p4)(p2.p3)` kernel.  EXEX elastic and
//! EEXX conversion remain separate because their affinities and null spaces
//! differ.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::electron_catalog::RateMeV;
use crate::electron_event::pauli_gradient;
use crate::electron_hm::G_F_MEV_MINUS_2;
use crate::quadrature::gauss_legendre_rule;

const SAME_SIGN_MATRIX_ELEMENT_COEFFICIENT: f64 = 32.0;
const ELASTIC_TIME_ORIENTATION_FACTOR: f64 = 0.5;
const GLOBAL_FOUR_LEG_BASE_COEFFICIENT: f64 =
    ELASTIC_TIME_ORIENTATION_FACTOR * SAME_SIGN_MATRIX_ELEMENT_COEFFICIENT;
const HEAVY_ROW3_IDENTICAL_FAMILIES: f64 = 4.0;
const HEAVY_ROW3_DISTINCT_FAMILIES: f64 = 2.0;
const HEAVY_SHARED_X_FOLD_DIVISOR: f64 = 4.0;
const HEAVY_SHARED_X_ROW3_PROJECTION: f64 =
    (HEAVY_ROW3_IDENTICAL_FAMILIES + HEAVY_ROW3_DISTINCT_FAMILIES) / HEAVY_SHARED_X_FOLD_DIVISOR;
#[cfg(test)]
const IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT: f64 = 64.0;
#[cfg(test)]
const DISTINCT_SAME_SIGN_TAGGED_COEFFICIENT: f64 = 32.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NeutrinoSelfSpectralRule {
    pub(crate) angular_order: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NeutrinoSelfSpectralInput<'a> {
    pub(crate) t_cm_mev: f64,
    pub(crate) y_nodes: &'a [f64],
    pub(crate) y_weights: &'a [f64],
    pub(crate) electron_pair_logit: &'a [f64],
    pub(crate) heavy_pair_logit: &'a [f64],
    pub(crate) rule: NeutrinoSelfSpectralRule,
}

#[derive(Clone, Debug)]
pub(crate) struct IsotropicNeutrinoSelfAction {
    pub(crate) electron_pair_mev: Vec<f64>,
    pub(crate) heavy_pair_mev: Vec<f64>,
    /// Row-major derivative of `[C_e, C_x]` with respect to logit state
    /// `[u_e, u_x]`.
    pub(crate) jacobian_logit_mev: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct InterpolationBracket {
    left: usize,
    right: usize,
    left_weight: f64,
    right_weight: f64,
}

#[derive(Clone, Copy, Debug)]
struct SelfEvent {
    first_incoming: usize,
    second_incoming: usize,
    first_outgoing: InterpolationBracket,
    second_outgoing: InterpolationBracket,
    s_weight_mev: f64,
    t_weight_mev: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvariantKernel {
    S,
    T,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpectralBank {
    Electron,
    Heavy,
}

impl SpectralBank {
    const fn index(self) -> usize {
        match self {
            Self::Electron => 0,
            Self::Heavy => 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FoldedChannel {
    leg_banks: [SpectralBank; 4],
    s_output_multipliers: [f64; 2],
    t_output_multipliers: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
struct FoldedState<'a> {
    logits: [&'a [f64]; 2],
    occupations: [&'a [f64]; 2],
    cell_measures: &'a [f64],
}

const EEEE: [SpectralBank; 4] = [SpectralBank::Electron; 4];
const XXXX: [SpectralBank; 4] = [SpectralBank::Heavy; 4];
const EXEX: [SpectralBank; 4] = [
    SpectralBank::Electron,
    SpectralBank::Heavy,
    SpectralBank::Electron,
    SpectralBank::Heavy,
];
const EEXX: [SpectralBank; 4] = [
    SpectralBank::Electron,
    SpectralBank::Electron,
    SpectralBank::Heavy,
    SpectralBank::Heavy,
];

const ROW1_KS_EEEE: FoldedChannel = FoldedChannel {
    leg_banks: EEEE,
    s_output_multipliers: [1.0, 0.0],
    t_output_multipliers: [0.0, 0.0],
};
const ROW3_KS_XXXX: FoldedChannel = FoldedChannel {
    leg_banks: XXXX,
    s_output_multipliers: [0.0, HEAVY_SHARED_X_ROW3_PROJECTION],
    t_output_multipliers: [0.0, 0.0],
};
const ROW7_KS_EXEX: FoldedChannel = FoldedChannel {
    leg_banks: EXEX,
    s_output_multipliers: [2.0, 1.0],
    t_output_multipliers: [0.0, 0.0],
};
const ROW2_KT_EEEE: FoldedChannel = FoldedChannel {
    leg_banks: EEEE,
    s_output_multipliers: [0.0, 0.0],
    t_output_multipliers: [2.0, 0.0],
};
const ROWS4_TO_6_KT_XXXX: FoldedChannel = FoldedChannel {
    leg_banks: XXXX,
    s_output_multipliers: [0.0, 0.0],
    t_output_multipliers: [0.0, 3.0],
};
const ROW8_KT_EXEX: FoldedChannel = FoldedChannel {
    leg_banks: EXEX,
    s_output_multipliers: [0.0, 0.0],
    t_output_multipliers: [2.0, 1.0],
};
const ROW9_KT_EEXX: FoldedChannel = FoldedChannel {
    leg_banks: EEXX,
    s_output_multipliers: [0.0, 0.0],
    t_output_multipliers: [2.0, 1.0],
};

const FULL_EEEE: FoldedChannel = FoldedChannel {
    leg_banks: EEEE,
    s_output_multipliers: ROW1_KS_EEEE.s_output_multipliers,
    t_output_multipliers: ROW2_KT_EEEE.t_output_multipliers,
};
const FULL_XXXX: FoldedChannel = FoldedChannel {
    leg_banks: XXXX,
    s_output_multipliers: ROW3_KS_XXXX.s_output_multipliers,
    t_output_multipliers: ROWS4_TO_6_KT_XXXX.t_output_multipliers,
};
const FULL_EXEX: FoldedChannel = FoldedChannel {
    leg_banks: EXEX,
    s_output_multipliers: ROW7_KS_EXEX.s_output_multipliers,
    t_output_multipliers: ROW8_KT_EXEX.t_output_multipliers,
};

fn occupation_from_logit(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

fn checked_input(input: NeutrinoSelfSpectralInput<'_>) -> Result<usize, &'static str> {
    let nq = input.y_nodes.len();
    if nq < 2
        || input.y_weights.len() != nq
        || input.electron_pair_logit.len() != nq
        || input.heavy_pair_logit.len() != nq
        || !input.t_cm_mev.is_finite()
        || input.t_cm_mev <= 0.0
        || !input
            .y_nodes
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !input.y_nodes.windows(2).all(|pair| pair[0] < pair[1])
        || !input
            .y_weights
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        || !input
            .electron_pair_logit
            .iter()
            .chain(input.heavy_pair_logit)
            .all(|value| value.is_finite())
    {
        return Err("neutrino self-scattering input is invalid");
    }
    Ok(nq)
}

fn interpolation_bracket(nodes: &[f64], value: f64) -> Option<InterpolationBracket> {
    if !value.is_finite() || value < nodes[0] || value > nodes[nodes.len() - 1] {
        return None;
    }
    match nodes.binary_search_by(|node| node.total_cmp(&value)) {
        Ok(index) => Some(InterpolationBracket {
            left: index,
            right: index,
            left_weight: 1.0,
            right_weight: 0.0,
        }),
        Err(right) if right > 0 && right < nodes.len() => {
            let left = right - 1;
            let right_weight = (value - nodes[left]) / (nodes[right] - nodes[left]);
            let left_weight = 1.0 - right_weight;
            (left_weight.is_finite()
                && right_weight.is_finite()
                && left_weight >= 0.0
                && right_weight >= 0.0)
                .then_some(InterpolationBracket {
                    left,
                    right,
                    left_weight,
                    right_weight,
                })
        }
        Err(_) => None,
    }
}

fn outgoing_energies(y1: f64, y2: f64, mu12: f64, z_star: f64) -> Option<(f64, f64)> {
    let total = y1 + y2;
    let boost_squared = (y1 - y2).powi(2) + 2.0 * y1 * y2 * (1.0 + mu12);
    if !boost_squared.is_finite() || boost_squared <= 0.0 {
        return None;
    }
    let boost = boost_squared.sqrt();
    let y3 = 0.5 * (total + boost * z_star);
    let y4 = 0.5 * (total - boost * z_star);
    (y3.is_finite() && y4.is_finite() && y3 > 0.0 && y4 > 0.0).then_some((y3, y4))
}

fn averaged_s_invariant_dimensionless(y1: f64, y2: f64, mu12: f64) -> f64 {
    let s = 2.0 * y1 * y2 * (1.0 - mu12);
    0.25 * s * s
}

fn azimuth_averaged_t_invariant_dimensionless(
    y1: f64,
    y2: f64,
    mu12: f64,
    z_star: f64,
) -> Option<f64> {
    let boost_squared = (y1 - y2).powi(2) + 2.0 * y1 * y2 * (1.0 + mu12);
    if !boost_squared.is_finite() || boost_squared <= 0.0 {
        return None;
    }
    let chi = (y1 - y2) / boost_squared.sqrt();
    let one_minus_chi_squared = 2.0 * y1 * y2 * (1.0 + mu12) / boost_squared;
    let s = 2.0 * y1 * y2 * (1.0 - mu12);
    let bracket =
        (1.0 + chi * z_star).powi(2) + 0.5 * one_minus_chi_squared * (1.0 - z_star * z_star);
    let invariant = s * s * bracket / 16.0;
    (chi.is_finite()
        && chi.abs() <= 1.0
        && one_minus_chi_squared.is_finite()
        && one_minus_chi_squared >= 0.0
        && bracket.is_finite()
        && bracket >= 0.0
        && invariant.is_finite()
        && invariant >= 0.0)
        .then_some(invariant)
}

fn build_event_stream(
    t_cm_mev: f64,
    y_nodes: &[f64],
    y_weights: &[f64],
    rule: NeutrinoSelfSpectralRule,
) -> Result<Vec<SelfEvent>, &'static str> {
    let angular_rule = gauss_legendre_rule(rule.angular_order)?;
    let radial_pairs = y_nodes
        .len()
        .checked_mul(y_nodes.len())
        .ok_or("neutrino self-scattering radial dimension overflow")?;
    let angular_pairs = angular_rule
        .len()
        .checked_mul(angular_rule.len())
        .ok_or("neutrino self-scattering angular dimension overflow")?;
    let capacity = radial_pairs
        .checked_mul(angular_pairs)
        .ok_or("neutrino self-scattering event dimension overflow")?;
    let prefactor = GLOBAL_FOUR_LEG_BASE_COEFFICIENT * G_F_MEV_MINUS_2.powi(2) * t_cm_mev.powi(5)
        / (512.0 * PI.powi(5));
    if !prefactor.is_finite() || prefactor <= 0.0 {
        return Err("neutrino self-scattering prefactor is invalid");
    }

    let mut events = Vec::with_capacity(capacity);
    for (first_incoming, (&y1, &w1)) in y_nodes.iter().zip(y_weights).enumerate() {
        for (second_incoming, (&y2, &w2)) in y_nodes.iter().zip(y_weights).enumerate() {
            for &(mu12, mu12_weight) in &angular_rule {
                for &(z_star, z_star_weight) in &angular_rule {
                    let Some((y3, y4)) = outgoing_energies(y1, y2, mu12, z_star) else {
                        continue;
                    };
                    let s_invariant = averaged_s_invariant_dimensionless(y1, y2, mu12);
                    let t_invariant =
                        azimuth_averaged_t_invariant_dimensionless(y1, y2, mu12, z_star)
                            .ok_or("neutrino self-scattering K_t invariant is invalid")?;
                    let (Some(first_outgoing), Some(second_outgoing)) = (
                        interpolation_bracket(y_nodes, y3),
                        interpolation_bracket(y_nodes, y4),
                    ) else {
                        // The finite radial domain is truncated symmetrically:
                        // either the entire physical event is retained or none
                        // of its legs is deposited.
                        continue;
                    };
                    let base_weight_mev =
                        prefactor * y1 * y2 * w1 * w2 * mu12_weight * z_star_weight;
                    let s_weight_mev = base_weight_mev * s_invariant;
                    let t_weight_mev = base_weight_mev * t_invariant;
                    if !s_weight_mev.is_finite()
                        || s_weight_mev <= 0.0
                        || !t_weight_mev.is_finite()
                        || t_weight_mev <= 0.0
                    {
                        return Err("neutrino self-scattering event weight is invalid");
                    }
                    events.push(SelfEvent {
                        first_incoming,
                        second_incoming,
                        first_outgoing,
                        second_outgoing,
                        s_weight_mev,
                        t_weight_mev,
                    });
                }
            }
        }
    }
    (!events.is_empty())
        .then_some(events)
        .ok_or("neutrino self-scattering grid retains no complete physical event")
}

fn interpolated_logit(logits: &[f64], bracket: InterpolationBracket) -> f64 {
    bracket.left_weight * logits[bracket.left] + bracket.right_weight * logits[bracket.right]
}

fn stable_gain_minus_loss(logits: [f64; 4], occupations: [f64; 4]) -> Result<f64, &'static str> {
    let [u1, u2, u3, u4] = logits;
    let [f1, f2, f3, f4] = occupations;
    let affinity = u1 + u2 - u3 - u4;
    let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
    let gain = (1.0 - f1) * (1.0 - f2) * f3 * f4;
    let net = if affinity >= 0.0 {
        loss * (-affinity).exp_m1()
    } else {
        -gain * affinity.exp_m1()
    };
    (affinity.is_finite() && loss.is_finite() && gain.is_finite() && net.is_finite())
        .then_some(net)
        .ok_or("neutrino self-scattering Pauli balance is non-finite")
}

fn bracket_outputs(bracket: InterpolationBracket, sign: f64) -> [(usize, f64); 2] {
    [
        (bracket.left, sign * bracket.left_weight),
        (bracket.right, sign * bracket.right_weight),
    ]
}

fn accumulate_folded_channel(
    state: FoldedState<'_>,
    events: &[SelfEvent],
    channel: FoldedChannel,
    include_jacobian: bool,
    actions: &mut [Vec<f64>; 2],
    jacobian: &mut [f64],
) -> Result<(), &'static str> {
    let nq = state.logits[0].len();
    let width = 2 * nq;
    for event in events {
        let banks = channel.leg_banks;
        let first_bank = banks[0].index();
        let second_bank = banks[1].index();
        let third_bank = banks[2].index();
        let fourth_bank = banks[3].index();
        let first_outgoing_logit =
            interpolated_logit(state.logits[third_bank], event.first_outgoing);
        let second_outgoing_logit =
            interpolated_logit(state.logits[fourth_bank], event.second_outgoing);
        let event_logits = [
            state.logits[first_bank][event.first_incoming],
            state.logits[second_bank][event.second_incoming],
            first_outgoing_logit,
            second_outgoing_logit,
        ];
        let event_occupations = event_logits.map(occupation_from_logit);
        if !event_occupations
            .iter()
            .all(|value| value.is_finite() && *value > 0.0 && *value < 1.0)
        {
            return Err("interpolated neutrino self-scattering occupation is invalid");
        }
        let net = stable_gain_minus_loss(event_logits, event_occupations)?;
        let mut weighted_net_by_bank = [0.0; 2];
        let mut weighted_gradient_by_bank = [0.0; 2];
        for bank in 0..2 {
            weighted_gradient_by_bank[bank] = event.s_weight_mev
                * channel.s_output_multipliers[bank]
                + event.t_weight_mev * channel.t_output_multipliers[bank];
            weighted_net_by_bank[bank] =
                RateMeV::new(weighted_gradient_by_bank[bank] * net)?.value();
        }
        let first_outgoing_outputs = bracket_outputs(event.first_outgoing, -1.0);
        let second_outgoing_outputs = bracket_outputs(event.second_outgoing, -1.0);
        let outputs = [
            (banks[0], event.first_incoming, 1.0),
            (banks[1], event.second_incoming, 1.0),
            (
                banks[2],
                first_outgoing_outputs[0].0,
                first_outgoing_outputs[0].1,
            ),
            (
                banks[2],
                first_outgoing_outputs[1].0,
                first_outgoing_outputs[1].1,
            ),
            (
                banks[3],
                second_outgoing_outputs[0].0,
                second_outgoing_outputs[0].1,
            ),
            (
                banks[3],
                second_outgoing_outputs[1].0,
                second_outgoing_outputs[1].1,
            ),
        ];
        for &(bank, row, coefficient) in &outputs {
            let bank = bank.index();
            actions[bank][row] +=
                coefficient * weighted_net_by_bank[bank] / state.cell_measures[row];
        }

        if include_jacobian {
            let gradient = pauli_gradient(event_occupations)?;
            let first_outgoing_response = event_occupations[2] * (1.0 - event_occupations[2]);
            let second_outgoing_response = event_occupations[3] * (1.0 - event_occupations[3]);
            let inputs = [
                (
                    banks[0],
                    event.first_incoming,
                    gradient[0]
                        * state.occupations[first_bank][event.first_incoming]
                        * (1.0 - state.occupations[first_bank][event.first_incoming]),
                ),
                (
                    banks[1],
                    event.second_incoming,
                    gradient[1]
                        * state.occupations[second_bank][event.second_incoming]
                        * (1.0 - state.occupations[second_bank][event.second_incoming]),
                ),
                (
                    banks[2],
                    event.first_outgoing.left,
                    gradient[2] * first_outgoing_response * event.first_outgoing.left_weight,
                ),
                (
                    banks[2],
                    event.first_outgoing.right,
                    gradient[2] * first_outgoing_response * event.first_outgoing.right_weight,
                ),
                (
                    banks[3],
                    event.second_outgoing.left,
                    gradient[3] * second_outgoing_response * event.second_outgoing.left_weight,
                ),
                (
                    banks[3],
                    event.second_outgoing.right,
                    gradient[3] * second_outgoing_response * event.second_outgoing.right_weight,
                ),
            ];
            for &(output_bank, row, output_coefficient) in &outputs {
                let output_bank = output_bank.index();
                for &(input_bank, column, input_derivative) in &inputs {
                    let input_bank = input_bank.index();
                    jacobian[(output_bank * nq + row) * width + input_bank * nq + column] +=
                        output_coefficient
                            * weighted_gradient_by_bank[output_bank]
                            * input_derivative
                            / state.cell_measures[row];
                }
            }
        }
    }
    Ok(())
}

fn evaluate_impl(
    input: NeutrinoSelfSpectralInput<'_>,
    include_jacobian: bool,
) -> Result<IsotropicNeutrinoSelfAction, &'static str> {
    let nq = checked_input(input)?;
    let events = build_event_stream(input.t_cm_mev, input.y_nodes, input.y_weights, input.rule)?;
    let electron_occupations = input
        .electron_pair_logit
        .iter()
        .copied()
        .map(occupation_from_logit)
        .collect::<Vec<_>>();
    let heavy_occupations = input
        .heavy_pair_logit
        .iter()
        .copied()
        .map(occupation_from_logit)
        .collect::<Vec<_>>();
    if !electron_occupations
        .iter()
        .chain(&heavy_occupations)
        .all(|value| value.is_finite() && *value > 0.0 && *value < 1.0)
    {
        return Err("neutrino self-scattering logit cannot represent an open occupation");
    }
    let logits = [input.electron_pair_logit, input.heavy_pair_logit];
    let occupations = [
        electron_occupations.as_slice(),
        heavy_occupations.as_slice(),
    ];
    let cell_measures = input
        .y_nodes
        .iter()
        .zip(input.y_weights)
        .map(|(node, weight)| node.powi(2) * weight / (2.0 * PI.powi(2)))
        .collect::<Vec<_>>();
    let mut actions = [vec![0.0; nq], vec![0.0; nq]];
    let mut jacobian_logit_mev = if include_jacobian {
        vec![0.0; 4 * nq * nq]
    } else {
        Vec::new()
    };
    let folded_state = FoldedState {
        logits,
        occupations,
        cell_measures: &cell_measures,
    };
    for channel in [FULL_EEEE, FULL_XXXX, FULL_EXEX, ROW9_KT_EEXX] {
        accumulate_folded_channel(
            folded_state,
            &events,
            channel,
            include_jacobian,
            &mut actions,
            &mut jacobian_logit_mev,
        )?;
    }
    if !actions
        .iter()
        .flat_map(|values| values.iter())
        .chain(&jacobian_logit_mev)
        .all(|value| value.is_finite())
    {
        return Err("neutrino self-scattering action is non-finite");
    }
    let [electron_pair_mev, heavy_pair_mev] = actions;
    Ok(IsotropicNeutrinoSelfAction {
        electron_pair_mev,
        heavy_pair_mev,
        jacobian_logit_mev,
    })
}

pub(crate) fn evaluate_isotropic_neutrino_self_action(
    input: NeutrinoSelfSpectralInput<'_>,
) -> Result<IsotropicNeutrinoSelfAction, &'static str> {
    evaluate_impl(input, true)
}

pub(crate) fn evaluate_isotropic_neutrino_self_action_values(
    input: NeutrinoSelfSpectralInput<'_>,
) -> Result<IsotropicNeutrinoSelfAction, &'static str> {
    evaluate_impl(input, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadrature::{gauss_laguerre_plain_rule, gauss_legendre_exponential_plain_rule};

    fn grid(order: usize) -> (Vec<f64>, Vec<f64>) {
        gauss_laguerre_plain_rule(order)
            .unwrap()
            .into_iter()
            .unzip()
    }

    fn compact_quadratic_legendre_rule(order: usize, maximum: f64) -> Vec<(f64, f64)> {
        gauss_legendre_rule(order)
            .unwrap()
            .into_iter()
            .map(|(coordinate, weight)| {
                let unit = 0.5 * (1.0 + coordinate);
                (maximum * unit.powi(2), weight * maximum * unit)
            })
            .collect()
    }

    fn exponential_legendre_rule(order: usize) -> Vec<(f64, f64)> {
        gauss_legendre_exponential_plain_rule(order, 3.0).unwrap()
    }

    fn input<'a>(
        temperature: f64,
        y: &'a [f64],
        w: &'a [f64],
        electron: &'a [f64],
        heavy: &'a [f64],
    ) -> NeutrinoSelfSpectralInput<'a> {
        input_with_angular_order(temperature, y, w, electron, heavy, 4)
    }

    fn input_with_angular_order<'a>(
        temperature: f64,
        y: &'a [f64],
        w: &'a [f64],
        electron: &'a [f64],
        heavy: &'a [f64],
        angular_order: usize,
    ) -> NeutrinoSelfSpectralInput<'a> {
        NeutrinoSelfSpectralInput {
            t_cm_mev: temperature,
            y_nodes: y,
            y_weights: w,
            electron_pair_logit: electron,
            heavy_pair_logit: heavy,
            rule: NeutrinoSelfSpectralRule { angular_order },
        }
    }

    fn evaluate_isolated_channel(
        input: NeutrinoSelfSpectralInput<'_>,
        kernel: InvariantKernel,
        channel: FoldedChannel,
        include_jacobian: bool,
    ) -> IsotropicNeutrinoSelfAction {
        let nq = checked_input(input).unwrap();
        let events =
            build_event_stream(input.t_cm_mev, input.y_nodes, input.y_weights, input.rule).unwrap();
        let channel = match kernel {
            InvariantKernel::S => FoldedChannel {
                leg_banks: channel.leg_banks,
                s_output_multipliers: channel.s_output_multipliers,
                t_output_multipliers: [0.0, 0.0],
            },
            InvariantKernel::T => FoldedChannel {
                leg_banks: channel.leg_banks,
                s_output_multipliers: [0.0, 0.0],
                t_output_multipliers: channel.t_output_multipliers,
            },
        };
        let electron_occupations = input
            .electron_pair_logit
            .iter()
            .copied()
            .map(occupation_from_logit)
            .collect::<Vec<_>>();
        let heavy_occupations = input
            .heavy_pair_logit
            .iter()
            .copied()
            .map(occupation_from_logit)
            .collect::<Vec<_>>();
        let logits = [input.electron_pair_logit, input.heavy_pair_logit];
        let occupations = [
            electron_occupations.as_slice(),
            heavy_occupations.as_slice(),
        ];
        let cell_measures = input
            .y_nodes
            .iter()
            .zip(input.y_weights)
            .map(|(node, weight)| node.powi(2) * weight / (2.0 * PI.powi(2)))
            .collect::<Vec<_>>();
        let mut actions = [vec![0.0; nq], vec![0.0; nq]];
        let mut jacobian = if include_jacobian {
            vec![0.0; 4 * nq * nq]
        } else {
            Vec::new()
        };
        let folded_state = FoldedState {
            logits,
            occupations,
            cell_measures: &cell_measures,
        };
        accumulate_folded_channel(
            folded_state,
            &events,
            channel,
            include_jacobian,
            &mut actions,
            &mut jacobian,
        )
        .unwrap();
        let [electron_pair_mev, heavy_pair_mev] = actions;
        IsotropicNeutrinoSelfAction {
            electron_pair_mev,
            heavy_pair_mev,
            jacobian_logit_mev: jacobian,
        }
    }

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    }

    fn independent_profile_logit(bank: SpectralBank, value: f64) -> f64 {
        match bank {
            SpectralBank::Electron => profile_logit(value, (3.0, 1.2, 0.25)),
            SpectralBank::Heavy => profile_logit(value, (6.0, 2.0, 0.25)),
        }
    }

    fn independent_occupation(logit: f64) -> f64 {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }

    fn independent_gain_minus_loss(logits: [f64; 4]) -> f64 {
        let [f1, f2, f3, f4] = logits.map(independent_occupation);
        (1.0 - f1) * (1.0 - f2) * f3 * f4 - f1 * f2 * (1.0 - f3) * (1.0 - f4)
    }

    fn independent_outgoing_energies(y1: f64, y2: f64, mu12: f64, z_star: f64) -> (f64, f64) {
        let total = y1 + y2;
        let boost = ((y1 - y2).powi(2) + 2.0 * y1 * y2 * (1.0 + mu12)).sqrt();
        (
            0.5 * (total + boost * z_star),
            0.5 * (total - boost * z_star),
        )
    }

    fn independent_invariant(
        kernel: InvariantKernel,
        y1: f64,
        y2: f64,
        mu12: f64,
        z_star: f64,
    ) -> f64 {
        let s = 2.0 * y1 * y2 * (1.0 - mu12);
        match kernel {
            InvariantKernel::S => s * s / 4.0,
            InvariantKernel::T => {
                let boost_squared = (y1 - y2).powi(2) + 2.0 * y1 * y2 * (1.0 + mu12);
                let chi = (y1 - y2) / boost_squared.sqrt();
                let q = 3.0 + 4.0 * chi * z_star - chi * chi - z_star * z_star
                    + 3.0 * chi * chi * z_star * z_star;
                s * s * q / 32.0
            }
        }
    }

    fn independent_tagged_target_action(
        target_y: f64,
        radial_rule: &[(f64, f64)],
        angular_rule: &[(f64, f64)],
        kernel: InvariantKernel,
        leg_banks: [SpectralBank; 4],
        tagged_coefficient: f64,
    ) -> f64 {
        let prefactor =
            tagged_coefficient * G_F_MEV_MINUS_2.powi(2) / (256.0 * PI.powi(3) * target_y);
        let mut collision = 0.0;
        for &(y2, w2) in radial_rule {
            for &(mu12, mu12_weight) in angular_rule {
                for &(z_star, z_star_weight) in angular_rule {
                    let (y3, y4) = independent_outgoing_energies(target_y, y2, mu12, z_star);
                    let event_logits = [
                        independent_profile_logit(leg_banks[0], target_y),
                        independent_profile_logit(leg_banks[1], y2),
                        independent_profile_logit(leg_banks[2], y3),
                        independent_profile_logit(leg_banks[3], y4),
                    ];
                    collision += w2
                        * y2
                        * mu12_weight
                        * z_star_weight
                        * independent_invariant(kernel, target_y, y2, mu12, z_star)
                        * independent_gain_minus_loss(event_logits);
                }
            }
        }
        prefactor * collision
    }

    fn tagged_massless_mb_loss(order: usize, target_y: f64, tagged_coefficient: f64) -> f64 {
        let radial_rule = gauss_laguerre_plain_rule(order).unwrap();
        let angular_rule = gauss_legendre_rule(4).unwrap();
        let prefactor =
            tagged_coefficient * G_F_MEV_MINUS_2.powi(2) / (256.0 * PI.powi(3) * target_y);
        let mut loss = 0.0;
        for &(y2, w2) in &radial_rule {
            for &(mu12, mu12_weight) in &angular_rule {
                let invariant = averaged_s_invariant_dimensionless(target_y, y2, mu12);
                for &(_, z_star_weight) in &angular_rule {
                    loss +=
                        w2 * y2 * mu12_weight * z_star_weight * invariant * (-target_y - y2).exp();
                }
            }
        }
        prefactor * loss
    }

    fn smooth_nonthermal_logit(value: f64) -> f64 {
        -value + 0.25 * (-0.5 * ((value - 3.0) / 1.2).powi(2)).exp()
    }

    fn profile_logit(value: f64, profile: (f64, f64, f64)) -> f64 {
        let (centre, width, amplitude) = profile;
        -value + amplitude * (-0.5 * ((value - centre) / width).powi(2)).exp()
    }

    #[derive(Clone, Copy, Debug)]
    struct AuxiliaryBasisAudit {
        basis_weak: f64,
        exact_all_weak: f64,
        exact_retained_weak: f64,
        number_relative: f64,
        energy_relative: f64,
    }

    fn auxiliary_basis_audit(
        state_nodes: &[f64],
        state_weights: &[f64],
        auxiliary_rule: &[(f64, f64)],
        profile: (f64, f64, f64),
    ) -> AuxiliaryBasisAudit {
        let angular_rule = gauss_legendre_rule(12).unwrap();
        let state_logits = state_nodes
            .iter()
            .copied()
            .map(|value| profile_logit(value, profile))
            .collect::<Vec<_>>();
        let cell_measures = state_nodes
            .iter()
            .zip(state_weights)
            .map(|(&node, &weight)| node.powi(2) * weight / (2.0 * PI.powi(2)))
            .collect::<Vec<_>>();
        let prefactor =
            GLOBAL_FOUR_LEG_BASE_COEFFICIENT * G_F_MEV_MINUS_2.powi(2) / (512.0 * PI.powi(5));
        let mut numerator = vec![0.0; state_nodes.len()];
        let mut exact_all_weak = 0.0;
        let mut exact_retained_weak = 0.0;

        for &(y1, w1) in auxiliary_rule {
            let first_incoming = interpolation_bracket(state_nodes, y1);
            let exact_u1 = profile_logit(y1, profile);
            for &(y2, w2) in auxiliary_rule {
                let second_incoming = interpolation_bracket(state_nodes, y2);
                let exact_u2 = profile_logit(y2, profile);
                for &(mu12, mu12_weight) in &angular_rule {
                    let invariant = averaged_s_invariant_dimensionless(y1, y2, mu12);
                    for &(z_star, z_star_weight) in &angular_rule {
                        let (y3, y4) = outgoing_energies(y1, y2, mu12, z_star).unwrap();
                        let exact_logits = [
                            exact_u1,
                            exact_u2,
                            profile_logit(y3, profile),
                            profile_logit(y4, profile),
                        ];
                        let exact_affinity =
                            exact_logits[0] + exact_logits[1] - exact_logits[2] - exact_logits[3];
                        let global_weight =
                            prefactor * y1 * y2 * w1 * w2 * mu12_weight * z_star_weight * invariant;
                        let exact_event_weak = global_weight
                            * stable_gain_minus_loss(
                                exact_logits,
                                exact_logits.map(occupation_from_logit),
                            )
                            .unwrap()
                            * exact_affinity;
                        exact_all_weak += exact_event_weak;

                        let (
                            Some(first_incoming),
                            Some(second_incoming),
                            Some(first_outgoing),
                            Some(second_outgoing),
                        ) = (
                            first_incoming,
                            second_incoming,
                            interpolation_bracket(state_nodes, y3),
                            interpolation_bracket(state_nodes, y4),
                        )
                        else {
                            continue;
                        };
                        exact_retained_weak += exact_event_weak;
                        let basis_logits = [
                            interpolated_logit(&state_logits, first_incoming),
                            interpolated_logit(&state_logits, second_incoming),
                            interpolated_logit(&state_logits, first_outgoing),
                            interpolated_logit(&state_logits, second_outgoing),
                        ];
                        let weighted_net = global_weight
                            * stable_gain_minus_loss(
                                basis_logits,
                                basis_logits.map(occupation_from_logit),
                            )
                            .unwrap();
                        for (bracket, sign) in [
                            (first_incoming, 1.0),
                            (second_incoming, 1.0),
                            (first_outgoing, -1.0),
                            (second_outgoing, -1.0),
                        ] {
                            for (index, coefficient) in bracket_outputs(bracket, sign) {
                                numerator[index] += coefficient * weighted_net;
                            }
                        }
                    }
                }
            }
        }

        let action = numerator
            .iter()
            .zip(&cell_measures)
            .map(|(&value, &cell)| value / cell)
            .collect::<Vec<_>>();
        let number = cell_measures
            .iter()
            .zip(&action)
            .map(|(&cell, &value)| cell * value)
            .sum::<f64>();
        let energy = cell_measures
            .iter()
            .zip(state_nodes)
            .zip(&action)
            .map(|((&cell, &node), &value)| cell * node * value)
            .sum::<f64>();
        let absolute_number = cell_measures
            .iter()
            .zip(&action)
            .map(|(&cell, &value)| cell * value.abs())
            .sum::<f64>();
        let absolute_energy = cell_measures
            .iter()
            .zip(state_nodes)
            .zip(&action)
            .map(|((&cell, &node), &value)| cell * node * value.abs())
            .sum::<f64>();
        let basis_weak = cell_measures
            .iter()
            .zip(&state_logits)
            .zip(&action)
            .map(|((&cell, &logit), &value)| cell * logit * value)
            .sum::<f64>();
        AuxiliaryBasisAudit {
            basis_weak,
            exact_all_weak,
            exact_retained_weak,
            number_relative: number.abs() / absolute_number.max(f64::MIN_POSITIVE),
            energy_relative: energy.abs() / absolute_energy.max(f64::MIN_POSITIVE),
        }
    }

    fn tagged_nonthermal_target_action(
        target_y: f64,
        radial_rule: &[(f64, f64)],
        angular_rule: &[(f64, f64)],
        tagged_coefficient: f64,
    ) -> f64 {
        let target_logit = smooth_nonthermal_logit(target_y);
        let prefactor =
            tagged_coefficient * G_F_MEV_MINUS_2.powi(2) / (256.0 * PI.powi(3) * target_y);
        let mut collision = 0.0;
        for &(y2, w2) in radial_rule {
            for &(mu12, mu12_weight) in angular_rule {
                let invariant = target_y.powi(2) * y2.powi(2) * (1.0 - mu12).powi(2);
                for &(z_star, z_star_weight) in angular_rule {
                    let (y3, y4) = outgoing_energies(target_y, y2, mu12, z_star).unwrap();
                    let event_logits = [
                        target_logit,
                        smooth_nonthermal_logit(y2),
                        smooth_nonthermal_logit(y3),
                        smooth_nonthermal_logit(y4),
                    ];
                    let event_occupations = event_logits.map(occupation_from_logit);
                    collision += w2
                        * y2
                        * mu12_weight
                        * z_star_weight
                        * invariant
                        * stable_gain_minus_loss(event_logits, event_occupations).unwrap();
                }
            }
        }
        prefactor * collision
    }

    #[test]
    fn common_affine_fd_logits_are_numerical_nulls_with_nonzero_response() {
        let (y, w) = grid(4);
        for (chemical_logit, inverse_temperature) in [(0.0, 1.0), (0.2, 1.1)] {
            let logit = y
                .iter()
                .map(|value| chemical_logit - inverse_temperature * value)
                .collect::<Vec<_>>();
            let action =
                evaluate_isotropic_neutrino_self_action(input(1.0, &y, &w, &logit, &logit))
                    .unwrap();
            let scale = action
                .jacobian_logit_mev
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            assert!(scale > 0.0);
            assert!(
                action
                    .electron_pair_mev
                    .iter()
                    .chain(&action.heavy_pair_mev)
                    .all(|value| value.abs() < 2.0e-13 * scale)
            );
        }
    }

    #[test]
    fn corrected_tagged_coefficients_reach_the_analytic_mb_losses() {
        assert_eq!(
            ELASTIC_TIME_ORIENTATION_FACTOR * SAME_SIGN_MATRIX_ELEMENT_COEFFICIENT,
            GLOBAL_FOUR_LEG_BASE_COEFFICIENT
        );
        assert_eq!(
            4.0 * GLOBAL_FOUR_LEG_BASE_COEFFICIENT,
            IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT
        );
        assert_eq!(
            2.0 * GLOBAL_FOUR_LEG_BASE_COEFFICIENT,
            DISTINCT_SAME_SIGN_TAGGED_COEFFICIENT
        );
        let target_y: f64 = 2.0;
        let identical_expected =
            G_F_MEV_MINUS_2.powi(2) * target_y * (-target_y).exp() * 8.0 / PI.powi(3);
        for (label, tagged_coefficient) in [
            ("identical", IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT),
            ("distinct", DISTINCT_SAME_SIGN_TAGGED_COEFFICIENT),
        ] {
            let expected =
                identical_expected * tagged_coefficient / IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT;
            for order in [4, 8, 12, 20] {
                let actual = tagged_massless_mb_loss(order, target_y, tagged_coefficient);
                let residual = relative_error(actual, expected);
                eprintln!(
                    "F10C CM {label} same-sign tagged MB loss order={order}: actual={actual:.17e} expected={expected:.17e} residual={residual:.17e}"
                );
                assert!(actual.is_finite() && actual > 0.0 && residual < 2.0e-12);
            }
        }
    }

    #[test]
    fn kt_azimuth_average_matches_explicit_phi_and_primary_q() {
        let points = [
            (0.7, 2.3, -0.6, -0.35),
            (1.4, 0.9, 0.2, 0.55),
            (4.2, 1.1, -0.15, 0.8),
            (2.0, 2.0, 0.65, -0.4),
        ];
        for (y1, y2, mu12, z_star) in points {
            let actual = azimuth_averaged_t_invariant_dimensionless(y1, y2, mu12, z_star).unwrap();
            let s = 2.0 * y1 * y2 * (1.0 - mu12);
            let boost_squared = (y1 - y2).powi(2) + 2.0 * y1 * y2 * (1.0 + mu12);
            let chi = (y1 - y2) / boost_squared.sqrt();
            let transverse = ((1.0 - chi * chi) * (1.0 - z_star * z_star)).sqrt();
            let phi_average = (0..64)
                .map(|index| {
                    let phi = 2.0 * PI * (index as f64 + 0.5) / 64.0;
                    let cosine = chi * z_star + transverse * phi.cos();
                    s * s * (1.0 + cosine).powi(2) / 16.0
                })
                .sum::<f64>()
                / 64.0;
            let q = 3.0 + 4.0 * chi * z_star - chi * chi - z_star * z_star
                + 3.0 * chi * chi * z_star * z_star;
            let primary_q = s * s * q / 32.0;
            assert!(actual >= 0.0 && phi_average >= 0.0 && primary_q >= 0.0);
            assert!(relative_error(actual, phi_average) < 3.0e-15);
            assert!(relative_error(actual, primary_q) < 3.0e-15);

            let z_rule = gauss_legendre_rule(32).unwrap();
            let integrated_t = z_rule
                .iter()
                .map(|&(z, weight)| {
                    weight * azimuth_averaged_t_invariant_dimensionless(y1, y2, mu12, z).unwrap()
                })
                .sum::<f64>();
            let integrated_s = z_rule
                .iter()
                .map(|&(_, weight)| weight * averaged_s_invariant_dimensionless(y1, y2, mu12))
                .sum::<f64>();
            assert!(relative_error(integrated_t / integrated_s, 1.0 / 3.0) < 3.0e-15);
        }
    }

    #[test]
    fn shared_heavy_row3_matches_an_explicit_six_family_projection() {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum HeavySpecies {
            NuMu,
            AntiNuMu,
            NuTau,
            AntiNuTau,
        }
        impl HeavySpecies {
            fn index(self) -> usize {
                match self {
                    Self::NuMu => 0,
                    Self::AntiNuMu => 1,
                    Self::NuTau => 2,
                    Self::AntiNuTau => 3,
                }
            }
        }

        // This species-level oracle deliberately does not call the production
        // evaluator and does not use its folded 3/2 constant.
        let families = [
            (HeavySpecies::NuMu, HeavySpecies::NuMu),
            (HeavySpecies::NuMu, HeavySpecies::NuTau),
            (HeavySpecies::NuTau, HeavySpecies::NuTau),
            (HeavySpecies::AntiNuMu, HeavySpecies::AntiNuMu),
            (HeavySpecies::AntiNuMu, HeavySpecies::AntiNuTau),
            (HeavySpecies::AntiNuTau, HeavySpecies::AntiNuTau),
        ];
        let mut tagged_by_species = [0.0_f64; 4];
        for (first, second) in families {
            if first == second {
                tagged_by_species[first.index()] += 64.0;
            } else {
                tagged_by_species[first.index()] += 32.0;
                tagged_by_species[second.index()] += 32.0;
            }
        }
        assert_eq!(tagged_by_species, [96.0; 4]);
        let folded_tagged = tagged_by_species.iter().sum::<f64>() / 4.0;
        let independently_enumerated_projection = folded_tagged / 64.0;
        assert_eq!(folded_tagged, 96.0);
        assert_eq!(
            independently_enumerated_projection,
            HEAVY_SHARED_X_ROW3_PROJECTION
        );

        let (y, w) = grid(4);
        let mut shared_logit = y.iter().map(|value| -*value).collect::<Vec<_>>();
        shared_logit[1] += 0.17;
        shared_logit[3] -= 0.09;
        let shared_input = input(1.0, &y, &w, &shared_logit, &shared_logit);
        let row1 = evaluate_isolated_channel(shared_input, InvariantKernel::S, ROW1_KS_EEEE, true);
        let row3 = evaluate_isolated_channel(shared_input, InvariantKernel::S, ROW3_KS_XXXX, true);
        for (&electron, &heavy) in row1.electron_pair_mev.iter().zip(&row3.heavy_pair_mev) {
            assert!(
                (heavy - independently_enumerated_projection * electron).abs()
                    < 2.0e-35_f64.max(2.0e-14 * heavy.abs())
            );
        }
        let nq = y.len();
        let width = 2 * nq;
        for row in 0..nq {
            for column in 0..nq {
                let ee = row1.jacobian_logit_mev[row * width + column];
                let ex = row1.jacobian_logit_mev[row * width + nq + column];
                let xe = row3.jacobian_logit_mev[(nq + row) * width + column];
                let xx = row3.jacobian_logit_mev[(nq + row) * width + nq + column];
                assert_eq!(ex, 0.0);
                assert_eq!(xe, 0.0);
                assert!(
                    (xx - independently_enumerated_projection * ee).abs()
                        < 2.0e-35_f64.max(2.0e-14 * xx.abs())
                );
            }
        }
    }

    #[test]
    fn explicit_six_species_enumerator_recovers_all_folded_row_factors() {
        #[derive(Clone, Copy, Debug)]
        enum Species {
            NuE,
            AntiNuE,
            NuMu,
            AntiNuMu,
            NuTau,
            AntiNuTau,
        }
        impl Species {
            fn index(self) -> usize {
                self as usize
            }

            fn bank(self) -> usize {
                match self {
                    Self::NuE | Self::AntiNuE => 0,
                    Self::NuMu | Self::AntiNuMu | Self::NuTau | Self::AntiNuTau => 1,
                }
            }
        }
        use Species::{AntiNuE as AE, AntiNuMu as AM, AntiNuTau as AT};
        use Species::{NuE as E, NuMu as M, NuTau as T};

        let rows: [(f64, Vec<[Species; 4]>); 9] = [
            (16.0, vec![[E, E, E, E], [AE, AE, AE, AE]]),
            (64.0, vec![[E, AE, E, AE]]),
            (
                16.0,
                vec![
                    [M, M, M, M],
                    [M, T, M, T],
                    [T, T, T, T],
                    [AM, AM, AM, AM],
                    [AM, AT, AM, AT],
                    [AT, AT, AT, AT],
                ],
            ),
            (64.0, vec![[M, AM, M, AM], [T, AT, T, AT]]),
            (16.0, vec![[M, AT, M, AT], [T, AM, T, AM]]),
            (32.0, vec![[M, AM, T, AT]]),
            (
                16.0,
                vec![
                    [E, M, E, M],
                    [E, T, E, T],
                    [AE, AM, AE, AM],
                    [AE, AT, AE, AT],
                ],
            ),
            (
                16.0,
                vec![
                    [E, AM, E, AM],
                    [E, AT, E, AT],
                    [AE, M, AE, M],
                    [AE, T, AE, T],
                ],
            ),
            (32.0, vec![[E, AE, M, AM], [E, AE, T, AT]]),
        ];
        let expected_multipliers = [
            ([1.0, 1.0, -1.0, -1.0], [0.0; 4]),
            ([2.0, 2.0, -2.0, -2.0], [0.0; 4]),
            ([0.0; 4], [1.5, 1.5, -1.5, -1.5]),
            ([0.0; 4], [2.0, 2.0, -2.0, -2.0]),
            ([0.0; 4], [0.5, 0.5, -0.5, -0.5]),
            ([0.0; 4], [0.5, 0.5, -0.5, -0.5]),
            ([2.0, 0.0, -2.0, 0.0], [0.0, 1.0, 0.0, -1.0]),
            ([2.0, 0.0, -2.0, 0.0], [0.0, 1.0, 0.0, -1.0]),
            ([2.0, 2.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0]),
        ];
        let expected_tagged = [
            [64.0, 64.0, 0.0, 0.0, 0.0, 0.0],
            [128.0, 128.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 96.0, 96.0, 96.0, 96.0],
            [0.0, 0.0, 128.0, 128.0, 128.0, 128.0],
            [0.0, 0.0, 32.0, 32.0, 32.0, 32.0],
            [0.0, 0.0, 32.0, 32.0, 32.0, 32.0],
            [64.0, 64.0, 32.0, 32.0, 32.0, 32.0],
            [64.0, 64.0, 32.0, 32.0, 32.0, 32.0],
            [64.0, 64.0, 32.0, 32.0, 32.0, 32.0],
        ];
        let mut derived = Vec::new();
        for ((global_coefficient, families), expected_tagged) in
            rows.into_iter().zip(expected_tagged)
        {
            let mut bank_legs = [[0.0_f64; 4]; 2];
            let mut tagged = [0.0_f64; 6];
            for family in families {
                for (leg, species) in family.into_iter().enumerate() {
                    bank_legs[species.bank()][leg] += global_coefficient;
                    tagged[species.index()] += global_coefficient;
                }
            }
            for (bank, legs) in bank_legs.iter_mut().enumerate() {
                let divisor = [2.0, 4.0][bank];
                for (value, sign) in legs.iter_mut().zip([1.0, 1.0, -1.0, -1.0]) {
                    *value *= sign / (divisor * 16.0);
                }
            }
            assert_eq!(tagged, expected_tagged);
            derived.push((bank_legs[0], bank_legs[1]));
        }
        assert_eq!(derived.as_slice(), &expected_multipliers);
        assert_eq!(ROW1_KS_EEEE.s_output_multipliers, [derived[0].0[0], 0.0]);
        assert_eq!(ROW2_KT_EEEE.t_output_multipliers, [derived[1].0[0], 0.0]);
        assert_eq!(ROW3_KS_XXXX.s_output_multipliers, [0.0, derived[2].1[0]]);
        assert_eq!(
            ROWS4_TO_6_KT_XXXX.t_output_multipliers,
            [0.0, derived[3].1[0] + derived[4].1[0] + derived[5].1[0]]
        );
        assert_eq!(
            ROW7_KS_EXEX.s_output_multipliers,
            [derived[6].0[0], derived[6].1[1]]
        );
        assert_eq!(
            ROW8_KT_EXEX.t_output_multipliers,
            [derived[7].0[0], derived[7].1[1]]
        );
        assert_eq!(
            ROW9_KT_EEXX.t_output_multipliers,
            [derived[8].0[0], -derived[8].1[2]]
        );
    }

    #[test]
    fn nonthermal_full_catalogue_conserves_weighted_number_and_energy() {
        let (y, w) = grid(6);
        let mut electron = y.iter().map(|value| -*value).collect::<Vec<_>>();
        let mut heavy = electron.clone();
        electron[1] += 0.2;
        electron[4] -= 0.1;
        heavy[2] -= 0.15;
        let action =
            evaluate_isotropic_neutrino_self_action_values(input(1.0, &y, &w, &electron, &heavy))
                .unwrap();
        let moments = [&action.electron_pair_mev, &action.heavy_pair_mev].map(|values| {
            let number = y
                .iter()
                .zip(&w)
                .zip(values)
                .map(|((&node, &weight), value)| weight * node.powi(2) * value)
                .sum::<f64>();
            let energy = y
                .iter()
                .zip(&w)
                .zip(values)
                .map(|((&node, &weight), value)| weight * node.powi(3) * value)
                .sum::<f64>();
            let absolute_number = y
                .iter()
                .zip(&w)
                .zip(values)
                .map(|((&node, &weight), value)| weight * node.powi(2) * value.abs())
                .sum::<f64>();
            let absolute_energy = y
                .iter()
                .zip(&w)
                .zip(values)
                .map(|((&node, &weight), value)| weight * node.powi(3) * value.abs())
                .sum::<f64>();
            (number, energy, absolute_number, absolute_energy)
        });
        let weighted_number = 2.0 * moments[0].0 + 4.0 * moments[1].0;
        let weighted_energy = 2.0 * moments[0].1 + 4.0 * moments[1].1;
        let absolute_number = 2.0 * moments[0].2 + 4.0 * moments[1].2;
        let absolute_energy = 2.0 * moments[0].3 + 4.0 * moments[1].3;
        assert!(weighted_number.abs() < 1.5e-12 * absolute_number.max(f64::MIN_POSITIVE));
        assert!(weighted_energy.abs() < 1.5e-12 * absolute_energy.max(f64::MIN_POSITIVE));
    }

    #[test]
    fn aggregated_topologies_match_the_explicit_row_sum() {
        let (y, w) = grid(4);
        let mut electron = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Electron, value))
            .collect::<Vec<_>>();
        let mut heavy = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Heavy, value))
            .collect::<Vec<_>>();
        electron[1] += 0.13;
        heavy[2] -= 0.07;
        let spectral_input = input(1.1, &y, &w, &electron, &heavy);
        let aggregated = evaluate_isotropic_neutrino_self_action(spectral_input).unwrap();
        let mut explicit = IsotropicNeutrinoSelfAction {
            electron_pair_mev: vec![0.0; y.len()],
            heavy_pair_mev: vec![0.0; y.len()],
            jacobian_logit_mev: vec![0.0; 4 * y.len() * y.len()],
        };
        for (kernel, channel) in [
            (InvariantKernel::S, ROW1_KS_EEEE),
            (InvariantKernel::S, ROW3_KS_XXXX),
            (InvariantKernel::S, ROW7_KS_EXEX),
            (InvariantKernel::T, ROW2_KT_EEEE),
            (InvariantKernel::T, ROWS4_TO_6_KT_XXXX),
            (InvariantKernel::T, ROW8_KT_EXEX),
            (InvariantKernel::T, ROW9_KT_EEXX),
        ] {
            let row = evaluate_isolated_channel(spectral_input, kernel, channel, true);
            for (sum, value) in explicit
                .electron_pair_mev
                .iter_mut()
                .zip(row.electron_pair_mev)
            {
                *sum += value;
            }
            for (sum, value) in explicit.heavy_pair_mev.iter_mut().zip(row.heavy_pair_mev) {
                *sum += value;
            }
            for (sum, value) in explicit
                .jacobian_logit_mev
                .iter_mut()
                .zip(row.jacobian_logit_mev)
            {
                *sum += value;
            }
        }
        for (label, actual, expected) in [
            (
                "electron action",
                aggregated.electron_pair_mev.as_slice(),
                explicit.electron_pair_mev.as_slice(),
            ),
            (
                "heavy action",
                aggregated.heavy_pair_mev.as_slice(),
                explicit.heavy_pair_mev.as_slice(),
            ),
            (
                "logit Jacobian",
                aggregated.jacobian_logit_mev.as_slice(),
                explicit.jacobian_logit_mev.as_slice(),
            ),
        ] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                let scale = actual.abs().max(expected.abs()).max(1.0e-35);
                assert!(
                    (actual - expected).abs() < 3.0e-14 * scale,
                    "{label}[{index}] actual={actual:.17e} expected={expected:.17e}"
                );
            }
        }
    }

    #[test]
    fn every_folded_channel_has_the_required_nulls_invariants_and_entropy_sign() {
        let (y, w) = grid(6);
        let mut electron = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Electron, value))
            .collect::<Vec<_>>();
        let mut heavy = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Heavy, value))
            .collect::<Vec<_>>();
        electron[1] += 0.11;
        heavy[3] -= 0.09;
        let channels = [
            ("row1", InvariantKernel::S, ROW1_KS_EEEE, true, true, false),
            ("row3", InvariantKernel::S, ROW3_KS_XXXX, true, true, false),
            ("row7", InvariantKernel::S, ROW7_KS_EXEX, true, false, false),
            ("row2", InvariantKernel::T, ROW2_KT_EEEE, true, true, false),
            (
                "rows4-6",
                InvariantKernel::T,
                ROWS4_TO_6_KT_XXXX,
                true,
                true,
                false,
            ),
            ("row8", InvariantKernel::T, ROW8_KT_EXEX, true, false, false),
            ("row9", InvariantKernel::T, ROW9_KT_EEXX, false, false, true),
        ];
        for (label, kernel, channel, bankwise_number, bankwise_energy, conversion) in channels {
            let action = evaluate_isolated_channel(
                input(1.0, &y, &w, &electron, &heavy),
                kernel,
                channel,
                false,
            );
            let moments = [
                (&electron, &action.electron_pair_mev),
                (&heavy, &action.heavy_pair_mev),
            ]
            .map(|(logits, values)| {
                let number = y
                    .iter()
                    .zip(&w)
                    .zip(values)
                    .map(|((&node, &weight), &value)| weight * node.powi(2) * value)
                    .sum::<f64>();
                let energy = y
                    .iter()
                    .zip(&w)
                    .zip(values)
                    .map(|((&node, &weight), &value)| weight * node.powi(3) * value)
                    .sum::<f64>();
                let absolute_number = y
                    .iter()
                    .zip(&w)
                    .zip(values)
                    .map(|((&node, &weight), &value)| weight * node.powi(2) * value.abs())
                    .sum::<f64>();
                let absolute_energy = y
                    .iter()
                    .zip(&w)
                    .zip(values)
                    .map(|((&node, &weight), &value)| weight * node.powi(3) * value.abs())
                    .sum::<f64>();
                let weak = y
                    .iter()
                    .zip(&w)
                    .zip(logits)
                    .zip(values)
                    .map(|(((&node, &weight), &logit), &value)| {
                        weight * node.powi(2) * logit * value
                    })
                    .sum::<f64>();
                (number, energy, absolute_number, absolute_energy, weak)
            });
            let weighted_number = 2.0 * moments[0].0 + 4.0 * moments[1].0;
            let weighted_energy = 2.0 * moments[0].1 + 4.0 * moments[1].1;
            let absolute_number = 2.0 * moments[0].2 + 4.0 * moments[1].2;
            let absolute_energy = 2.0 * moments[0].3 + 4.0 * moments[1].3;
            let entropy = -(2.0 * moments[0].4 + 4.0 * moments[1].4);
            assert!(
                weighted_number.abs() < 2.0e-12 * absolute_number.max(f64::MIN_POSITIVE),
                "{label}: weighted number={weighted_number:.17e}"
            );
            assert!(
                weighted_energy.abs() < 2.0e-12 * absolute_energy.max(f64::MIN_POSITIVE),
                "{label}: weighted energy={weighted_energy:.17e}"
            );
            assert!(entropy > 0.0, "{label}: entropy={entropy:.17e}");
            if bankwise_number {
                for bank in moments {
                    assert!(
                        bank.0.abs() < 2.0e-12 * bank.2.max(f64::MIN_POSITIVE),
                        "{label}: bank number={:.17e}",
                        bank.0
                    );
                }
            }
            if bankwise_energy {
                for bank in moments {
                    assert!(
                        bank.1.abs() < 2.0e-12 * bank.3.max(f64::MIN_POSITIVE),
                        "{label}: bank energy={:.17e}",
                        bank.1
                    );
                }
            }
            if conversion {
                assert!(moments[0].0.abs() > 1.0e-8 * moments[0].2);
                assert!(moments[1].0.abs() > 1.0e-8 * moments[1].2);
            }

            let elastic_e = y.iter().map(|&node| 0.23 - 1.07 * node).collect::<Vec<_>>();
            let elastic_x = y
                .iter()
                .map(|&node| -0.17 - 1.07 * node)
                .collect::<Vec<_>>();
            let (null_e, null_x) = if conversion {
                let common = y.iter().map(|&node| 0.05 - 1.07 * node).collect::<Vec<_>>();
                (common.clone(), common)
            } else {
                (elastic_e, elastic_x)
            };
            let null = evaluate_isolated_channel(
                input(1.0, &y, &w, &null_e, &null_x),
                kernel,
                channel,
                true,
            );
            let response_scale = null
                .jacobian_logit_mev
                .iter()
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            let action_scale = null
                .electron_pair_mev
                .iter()
                .chain(&null.heavy_pair_mev)
                .map(|value| value.abs())
                .fold(0.0, f64::max);
            assert!(response_scale > 0.0);
            assert!(
                action_scale < 3.0e-13 * response_scale,
                "{label}: null action={action_scale:.17e} response={response_scale:.17e}"
            );
        }

        let independent_intercepts = evaluate_isolated_channel(
            input(
                1.0,
                &y,
                &w,
                &y.iter().map(|&node| 0.25 - node).collect::<Vec<_>>(),
                &y.iter().map(|&node| -0.15 - node).collect::<Vec<_>>(),
            ),
            InvariantKernel::T,
            ROW9_KT_EEXX,
            false,
        );
        assert!(
            independent_intercepts
                .electron_pair_mev
                .iter()
                .chain(&independent_intercepts.heavy_pair_mev)
                .map(|value| value.abs())
                .fold(0.0, f64::max)
                > 0.0
        );
    }

    #[test]
    fn production_weak_action_converges_to_a_tagged_target_oracle() {
        let oracle_angular_rule = gauss_legendre_rule(24).unwrap();
        let mut residuals = Vec::new();
        let mut high_order_oracle = None;
        for order in [24, 32, 40, 48, 64] {
            let radial_rule = compact_quadratic_legendre_rule(order, 30.0);
            let (y, w): (Vec<_>, Vec<_>) = radial_rule.iter().copied().unzip();
            let logit = y
                .iter()
                .copied()
                .map(smooth_nonthermal_logit)
                .collect::<Vec<_>>();
            let production = evaluate_isolated_channel(
                input_with_angular_order(1.0, &y, &w, &logit, &logit, 12),
                InvariantKernel::S,
                ROW1_KS_EEEE,
                false,
            );
            let production_weak_action = y
                .iter()
                .zip(&w)
                .zip(&logit)
                .zip(&production.electron_pair_mev)
                .map(|(((&node, &weight), &test_value), collision)| {
                    weight * node.powi(2) * test_value * collision / (2.0 * PI.powi(2))
                })
                .sum::<f64>();
            let oracle_weak_action = y
                .iter()
                .zip(&w)
                .zip(&logit)
                .map(|((&node, &weight), &test_value)| {
                    weight
                        * node.powi(2)
                        * test_value
                        * tagged_nonthermal_target_action(
                            node,
                            &radial_rule,
                            &oracle_angular_rule,
                            IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT,
                        )
                        / (2.0 * PI.powi(2))
                })
                .sum::<f64>();
            let residual = relative_error(production_weak_action, oracle_weak_action);
            eprintln!(
                "F10C production/tagged weak oracle order={order}: production={production_weak_action:.17e} oracle={oracle_weak_action:.17e} residual={residual:.17e}"
            );
            assert!(production_weak_action.is_finite() && oracle_weak_action.is_finite());
            assert!(production_weak_action < 0.0 && oracle_weak_action < 0.0);
            residuals.push(residual);
            if order == 64 {
                high_order_oracle = Some(oracle_weak_action);
            }
        }
        assert!(residuals[1..].windows(2).all(|pair| pair[1] < pair[0]));
        assert!(residuals[residuals.len() - 1] < 2.1e-2);

        // The historical F10C0 endpoint's 16-node Gauss--Laguerre radial
        // representation is deliberately retained as a RED rather than being
        // promoted from a coincidental low-order comparison.  Refine only the
        // angular rule here and compare its full production weak form with the
        // independent high-order tagged target above.
        let (selected_y, selected_w) = grid(16);
        let selected_logit = selected_y
            .iter()
            .copied()
            .map(smooth_nonthermal_logit)
            .collect::<Vec<_>>();
        let selected = evaluate_isolated_channel(
            input_with_angular_order(
                1.0,
                &selected_y,
                &selected_w,
                &selected_logit,
                &selected_logit,
                24,
            ),
            InvariantKernel::S,
            ROW1_KS_EEEE,
            false,
        );
        let selected_weak_action = selected_y
            .iter()
            .zip(&selected_w)
            .zip(&selected_logit)
            .zip(&selected.electron_pair_mev)
            .map(|(((&node, &weight), &test_value), collision)| {
                weight * node.powi(2) * test_value * collision / (2.0 * PI.powi(2))
            })
            .sum::<f64>();
        let selected_residual = relative_error(selected_weak_action, high_order_oracle.unwrap());
        eprintln!(
            "F10C selected GL16/high-order tagged weak oracle: production={selected_weak_action:.17e} residual={selected_residual:.17e}"
        );
        assert!(selected_weak_action.is_finite() && selected_weak_action < 0.0);
        assert!((0.07..0.08).contains(&selected_residual));
    }

    #[test]
    fn every_folded_family_matches_an_independent_tagged_target_oracle() {
        let radial_rule = exponential_legendre_rule(48);
        let (y, w): (Vec<_>, Vec<_>) = radial_rule.iter().copied().unzip();
        let electron = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Electron, value))
            .collect::<Vec<_>>();
        let heavy = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Heavy, value))
            .collect::<Vec<_>>();
        let angular_rule = gauss_legendre_rule(24).unwrap();
        let cases = [
            (
                "row1-E",
                InvariantKernel::S,
                ROW1_KS_EEEE,
                SpectralBank::Electron,
                EEEE,
                64.0,
            ),
            (
                "row3-X",
                InvariantKernel::S,
                ROW3_KS_XXXX,
                SpectralBank::Heavy,
                XXXX,
                96.0,
            ),
            (
                "row7-E",
                InvariantKernel::S,
                ROW7_KS_EXEX,
                SpectralBank::Electron,
                EXEX,
                64.0,
            ),
            (
                "row7-X",
                InvariantKernel::S,
                ROW7_KS_EXEX,
                SpectralBank::Heavy,
                [
                    SpectralBank::Heavy,
                    SpectralBank::Electron,
                    SpectralBank::Heavy,
                    SpectralBank::Electron,
                ],
                32.0,
            ),
            (
                "row2-E",
                InvariantKernel::T,
                ROW2_KT_EEEE,
                SpectralBank::Electron,
                EEEE,
                128.0,
            ),
            (
                "rows4-6-X",
                InvariantKernel::T,
                ROWS4_TO_6_KT_XXXX,
                SpectralBank::Heavy,
                XXXX,
                192.0,
            ),
            (
                "row8-E",
                InvariantKernel::T,
                ROW8_KT_EXEX,
                SpectralBank::Electron,
                EXEX,
                64.0,
            ),
            (
                "row8-X",
                InvariantKernel::T,
                ROW8_KT_EXEX,
                SpectralBank::Heavy,
                [
                    SpectralBank::Heavy,
                    SpectralBank::Electron,
                    SpectralBank::Heavy,
                    SpectralBank::Electron,
                ],
                32.0,
            ),
            (
                "row9-E",
                InvariantKernel::T,
                ROW9_KT_EEXX,
                SpectralBank::Electron,
                EEXX,
                64.0,
            ),
            (
                "row9-X",
                InvariantKernel::T,
                ROW9_KT_EEXX,
                SpectralBank::Heavy,
                [
                    SpectralBank::Heavy,
                    SpectralBank::Heavy,
                    SpectralBank::Electron,
                    SpectralBank::Electron,
                ],
                32.0,
            ),
        ];
        for (label, kernel, channel, target_bank, tagged_banks, tagged_coefficient) in cases {
            let production = evaluate_isolated_channel(
                input_with_angular_order(1.0, &y, &w, &electron, &heavy, 12),
                kernel,
                channel,
                false,
            );
            let production_values = match target_bank {
                SpectralBank::Electron => &production.electron_pair_mev,
                SpectralBank::Heavy => &production.heavy_pair_mev,
            };
            let production_weak = y
                .iter()
                .zip(&w)
                .zip(production_values)
                .map(|((&node, &weight), &collision)| {
                    weight * node.powi(2) * independent_profile_logit(target_bank, node) * collision
                        / (2.0 * PI.powi(2))
                })
                .sum::<f64>();
            let tagged_weak = radial_rule
                .iter()
                .map(|&(node, weight)| {
                    weight
                        * node.powi(2)
                        * independent_profile_logit(target_bank, node)
                        * independent_tagged_target_action(
                            node,
                            &radial_rule,
                            &angular_rule,
                            kernel,
                            tagged_banks,
                            tagged_coefficient,
                        )
                        / (2.0 * PI.powi(2))
                })
                .sum::<f64>();
            let residual = relative_error(production_weak, tagged_weak);
            eprintln!(
                "F10C catalogue tagged oracle {label}: production={production_weak:.17e} tagged={tagged_weak:.17e} residual={residual:.17e}"
            );
            assert!(production_weak.is_finite() && tagged_weak.is_finite());
            assert!(
                production_weak * tagged_weak > 0.0,
                "{label}: production/tagged signs disagree"
            );
            assert!(residual < 3.0e-2, "{label}: residual={residual:.17e}");
        }
    }

    #[test]
    fn selected_angular_rule_has_a_nonthermal_entropy_ladder() {
        let (y, w) = grid(16);
        let logit = y
            .iter()
            .map(|value| -value + 0.25 * (-0.5 * ((value - 3.0) / 1.2).powi(2)).exp())
            .collect::<Vec<_>>();
        let mut values = Vec::new();
        for angular_order in [2, 4, 6, 8, 12, 16, 20, 24] {
            let action = evaluate_isolated_channel(
                input_with_angular_order(1.0, &y, &w, &logit, &logit, angular_order),
                InvariantKernel::S,
                ROW1_KS_EEEE,
                false,
            );
            let dissipation = -y
                .iter()
                .zip(&w)
                .zip(&logit)
                .zip(&action.electron_pair_mev)
                .map(|(((&node, &weight), &value), collision)| {
                    weight * node.powi(2) * value * collision / (2.0 * PI.powi(2))
                })
                .sum::<f64>()
                / G_F_MEV_MINUS_2.powi(2);
            eprintln!(
                "F10C CM same-nunu angular ladder order={angular_order}: normalized={dissipation:.17e}"
            );
            assert!(dissipation.is_finite() && dissipation > 0.0);
            values.push(dissipation);
        }
        assert!(relative_error(values[6], values[7]) < 5.0e-3);
        assert!(relative_error(values[4], values[7]) < 5.0e-3);
    }

    #[test]
    fn exponential_radial_rule_has_a_monotone_direct_ladder() {
        // This same-rule weak-action ladder is a necessary falsifier, not the
        // selection authority by itself.  The fixed endpoint order is chosen
        // only after the independent compact/exponential auxiliary rules also
        // bound the declared multi-profile state-basis envelope.
        let angular_rule = gauss_legendre_rule(24).unwrap();
        let reference_rule = compact_quadratic_legendre_rule(64, 30.0);
        let reference = reference_rule
            .iter()
            .map(|&(node, weight)| {
                weight
                    * node.powi(2)
                    * smooth_nonthermal_logit(node)
                    * tagged_nonthermal_target_action(
                        node,
                        &reference_rule,
                        &angular_rule,
                        IDENTICAL_SAME_SIGN_TAGGED_COEFFICIENT,
                    )
                    / (2.0 * PI.powi(2))
            })
            .sum::<f64>();
        let mut residuals = Vec::new();
        for order in [32, 40, 48, 64] {
            let radial_rule = exponential_legendre_rule(order);
            let (y, w): (Vec<_>, Vec<_>) = radial_rule.into_iter().unzip();
            let logit = y
                .iter()
                .copied()
                .map(smooth_nonthermal_logit)
                .collect::<Vec<_>>();
            let action = evaluate_isolated_channel(
                input_with_angular_order(1.0, &y, &w, &logit, &logit, 12),
                InvariantKernel::S,
                ROW1_KS_EEEE,
                false,
            );
            let weak_action = y
                .iter()
                .zip(&w)
                .zip(&logit)
                .zip(&action.electron_pair_mev)
                .map(|(((&node, &weight), &test_value), collision)| {
                    weight * node.powi(2) * test_value * collision / (2.0 * PI.powi(2))
                })
                .sum::<f64>();
            let residual = relative_error(weak_action, reference);
            eprintln!(
                "F10C exponential radial order={order}: weak={weak_action:.17e} reference={reference:.17e} residual={residual:.17e}"
            );
            assert!(weak_action.is_finite() && weak_action < 0.0);
            residuals.push(residual);
        }
        assert!(residuals.windows(2).all(|pair| pair[1] < pair[0]));
        assert!(residuals[2] < 9.0e-3);
        assert!(residuals[3] < 5.5e-3);
    }

    #[test]
    fn selected_exponential_n48_has_a_five_profile_auxiliary_basis_envelope() {
        let state_rule = exponential_legendre_rule(48);
        let (state_nodes, state_weights): (Vec<_>, Vec<_>) = state_rule.iter().copied().unzip();
        let auxiliary_rules = [
            ("compact30_N128", compact_quadratic_legendre_rule(128, 30.0)),
            (
                "exponential4_N128",
                gauss_legendre_exponential_plain_rule(128, 4.0).unwrap(),
            ),
        ];
        let profiles = [
            ("soft", (1.0, 0.5, 0.15)),
            ("baseline", (3.0, 1.2, 0.25)),
            ("mid", (6.0, 2.0, 0.25)),
            ("hard", (10.0, 3.0, 0.25)),
            ("negative", (3.0, 1.2, -0.25)),
        ];
        let mut maximum_residual = 0.0_f64;
        for (profile_name, profile) in profiles {
            let mut exact_references = Vec::with_capacity(auxiliary_rules.len());
            for (rule_name, auxiliary_rule) in &auxiliary_rules {
                let audit =
                    auxiliary_basis_audit(&state_nodes, &state_weights, auxiliary_rule, profile);
                let residual =
                    (audit.basis_weak - audit.exact_all_weak).abs() / audit.exact_all_weak.abs();
                let domain_fraction = (audit.exact_retained_weak - audit.exact_all_weak).abs()
                    / audit.exact_all_weak.abs();
                eprintln!(
                    "F10C1 N48 auxiliary profile={profile_name} rule={rule_name}: basis={:.17e} exact={:.17e} residual={residual:.17e} Nrel={:.17e} Erel={:.17e} entropy={:.17e} domain={domain_fraction:.17e}",
                    audit.basis_weak,
                    audit.exact_all_weak,
                    audit.number_relative,
                    audit.energy_relative,
                    -audit.basis_weak,
                );
                assert!(audit.basis_weak.is_finite() && audit.basis_weak < 0.0);
                assert!(audit.exact_all_weak.is_finite() && audit.exact_all_weak < 0.0);
                assert!(audit.exact_retained_weak.is_finite());
                assert!(residual < 2.1e-2);
                assert!(audit.number_relative < 5.0e-13);
                assert!(audit.energy_relative < 5.0e-13);
                assert!(-audit.basis_weak > 0.0);
                assert!(domain_fraction < 1.0e-4);
                maximum_residual = maximum_residual.max(residual);
                exact_references.push(audit.exact_all_weak);
            }
            assert!(relative_error(exact_references[0], exact_references[1]) < 3.0e-8);
        }
        assert!(maximum_residual < 2.1e-2);
    }

    #[test]
    fn action_has_the_massless_fifth_power_temperature_scaling() {
        let (y, w) = grid(4);
        let mut logit = y.iter().map(|value| -*value).collect::<Vec<_>>();
        logit[1] += 0.1;
        let first =
            evaluate_isotropic_neutrino_self_action_values(input(0.8, &y, &w, &logit, &logit))
                .unwrap();
        let second =
            evaluate_isotropic_neutrino_self_action_values(input(1.6, &y, &w, &logit, &logit))
                .unwrap();
        for (low_values, high_values) in [
            (&first.electron_pair_mev, &second.electron_pair_mev),
            (&first.heavy_pair_mev, &second.heavy_pair_mev),
        ] {
            for (low, high) in low_values.iter().zip(high_values) {
                if low.abs() > 1.0e-35 {
                    assert!(relative_error(*high / *low, 32.0) < 3.0e-13);
                }
            }
        }
    }

    #[test]
    fn every_folded_channel_jacobian_matches_five_point_stencils() {
        let (y, w) = grid(3);
        let mut electron = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Electron, value))
            .collect::<Vec<_>>();
        let mut heavy = y
            .iter()
            .map(|&value| independent_profile_logit(SpectralBank::Heavy, value))
            .collect::<Vec<_>>();
        electron[0] += 0.12;
        heavy[1] -= 0.08;
        let channels = [
            ("row1", InvariantKernel::S, ROW1_KS_EEEE),
            ("row3", InvariantKernel::S, ROW3_KS_XXXX),
            ("row7", InvariantKernel::S, ROW7_KS_EXEX),
            ("row2", InvariantKernel::T, ROW2_KT_EEEE),
            ("rows4-6", InvariantKernel::T, ROWS4_TO_6_KT_XXXX),
            ("row8", InvariantKernel::T, ROW8_KT_EXEX),
            ("row9", InvariantKernel::T, ROW9_KT_EEXX),
        ];
        let dimension = 2 * y.len();
        for (label, kernel, channel) in channels {
            let base = evaluate_isolated_channel(
                input(1.1, &y, &w, &electron, &heavy),
                kernel,
                channel,
                true,
            );
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
                    let action = evaluate_isolated_channel(
                        input(1.1, &y, &w, &shifted_e, &shifted_x),
                        kernel,
                        channel,
                        false,
                    );
                    [action.electron_pair_mev, action.heavy_pair_mev].concat()
                };
                let p2 = evaluate(2.0);
                let p1 = evaluate(1.0);
                let m1 = evaluate(-1.0);
                let m2 = evaluate(-2.0);
                for row in 0..dimension {
                    let expected =
                        (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step);
                    let actual = base.jacobian_logit_mev[row * dimension + column];
                    assert!(
                        (actual - expected).abs() < 3.0e-34
                            || relative_error(actual, expected) < 5.0e-8,
                        "{label}: row={row} column={column} actual={actual:.17e} expected={expected:.17e}"
                    );
                }
            }
        }
    }

    #[test]
    fn logit_jacobian_matches_five_point_stencils() {
        let (y, w) = grid(3);
        let mut electron = y.iter().map(|value| -*value).collect::<Vec<_>>();
        let mut heavy = electron.clone();
        electron[0] += 0.12;
        heavy[1] -= 0.08;
        let base =
            evaluate_isotropic_neutrino_self_action(input(1.1, &y, &w, &electron, &heavy)).unwrap();
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
                let action = evaluate_isotropic_neutrino_self_action_values(input(
                    1.1, &y, &w, &shifted_e, &shifted_x,
                ))
                .unwrap();
                [action.electron_pair_mev, action.heavy_pair_mev].concat()
            };
            let p2 = evaluate(2.0);
            let p1 = evaluate(1.0);
            let m1 = evaluate(-1.0);
            let m2 = evaluate(-2.0);
            for row in 0..dimension {
                let expected = (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step);
                let actual = base.jacobian_logit_mev[row * dimension + column];
                assert!(
                    (actual - expected).abs() < 2.0e-34
                        || relative_error(actual, expected) < 3.0e-8,
                    "row={row} column={column} actual={actual:.17e} expected={expected:.17e}"
                );
            }
        }
    }

    #[test]
    fn invalid_inputs_fail_without_clipping_or_tail_extrapolation() {
        let (y, w) = grid(3);
        let logit = y.iter().map(|value| -*value).collect::<Vec<_>>();
        let mut invalid = logit.clone();
        invalid[1] = f64::NAN;
        assert!(
            evaluate_isotropic_neutrino_self_action(input(1.0, &y, &w, &invalid, &logit)).is_err()
        );
        assert!(
            evaluate_isotropic_neutrino_self_action(input(f64::NAN, &y, &w, &logit, &logit))
                .is_err()
        );
    }
}
