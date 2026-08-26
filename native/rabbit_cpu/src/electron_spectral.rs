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
use crate::pauli_edge_step::{
    PauliEdge, PauliEdgeApplicationKind, PauliEdgeFailure, PauliEdgeFailureKind, PauliEdgeStep,
    PauliEdgeTopology,
};
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

#[derive(Clone, Debug)]
struct FoldedPauliEdge {
    bank: usize,
    edge: PauliEdge,
}

#[derive(Clone, Debug)]
pub(crate) struct IsotropicElectronPauliEdges {
    nq: usize,
    edges: Vec<FoldedPauliEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliSweepReport {
    pub(crate) edge_applications: usize,
    pub(crate) solved: usize,
    pub(crate) exact_stationary: usize,
    pub(crate) unresolved: usize,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) maximum_edge_iterations: usize,
    pub(crate) maximum_root_residual_ratio: f64,
    pub(crate) maximum_occupation_bracket_width: f64,
    pub(crate) maximum_flux_error_fraction: f64,
    pub(crate) maximum_root_error_bound: f64,
    pub(crate) maximum_occupation_error_bound: f64,
}

impl PauliSweepReport {
    fn empty() -> Self {
        Self {
            edge_applications: 0,
            solved: 0,
            exact_stationary: 0,
            unresolved: 0,
            nonlinear_iterations: 0,
            maximum_edge_iterations: 0,
            maximum_root_residual_ratio: 0.0,
            maximum_occupation_bracket_width: 0.0,
            maximum_flux_error_fraction: 0.0,
            maximum_root_error_bound: 0.0,
            maximum_occupation_error_bound: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliSweepFailure {
    pub(crate) kind: PauliEdgeFailureKind,
    pub(crate) edge_index: Option<usize>,
    pub(crate) partial_report: PauliSweepReport,
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

fn build_folded_pauli_edges(
    stream: &SuppliedElectronEvents,
    explicit_f: &[f64],
    y_nodes: &[f64],
    y_weights: &[f64],
) -> Result<IsotropicElectronPauliEdges, &'static str> {
    let nq = stream.nq();
    let input = EXPLICIT_STATES
        .checked_mul(nq)
        .ok_or("electron Pauli edge dimension overflow")?;
    let edge_count = input
        .checked_mul(input)
        .ok_or("electron Pauli edge dimension overflow")?;
    let number_weights = y_nodes
        .iter()
        .zip(y_weights)
        .map(|(node, weight)| weight * node.powi(2))
        .collect::<Vec<_>>();
    let mut directed_gain_coefficient = vec![0.0; edge_count];
    let mut directed_loss_coefficient = vec![0.0; edge_count];
    for item in stream.validated_contractions(explicit_f)? {
        let item = item?;
        let target = item.dynamic_legs.target.explicit_node.flat_index;
        let coupled = item.dynamic_legs.coupled.explicit_node.flat_index;
        let target_node = item.dynamic_legs.target.explicit_node.node;
        let edge = target * input + coupled;
        let measure = number_weights[target_node];
        let coefficients = item.dynamic_coefficients()?;
        directed_gain_coefficient[edge] += measure * coefficients.gain.value();
        directed_loss_coefficient[edge] += measure * coefficients.loss.value();
    }
    if directed_gain_coefficient
        .iter()
        .chain(&directed_loss_coefficient)
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("electron Pauli directed coefficient is invalid");
    }

    let elastic_coefficients = |state: usize, first_node: usize, second_node: usize| {
        let first = state * nq + first_node;
        let second = state * nq + second_node;
        let forward = first * input + second;
        let reverse = second * input + first;
        (
            0.5 * (directed_gain_coefficient[forward] + directed_loss_coefficient[reverse]),
            0.5 * (directed_loss_coefficient[forward] + directed_gain_coefficient[reverse]),
        )
    };
    let pair_coefficients = |first_state: usize, first_node: usize, second_node: usize| {
        let second_state = first_state + 1;
        let first = first_state * nq + first_node;
        let second = second_state * nq + second_node;
        let forward = first * input + second;
        let reverse = second * input + first;
        (
            0.5 * (directed_gain_coefficient[forward] + directed_gain_coefficient[reverse]),
            0.5 * (directed_loss_coefficient[forward] + directed_loss_coefficient[reverse]),
        )
    };

    let mut edges = Vec::with_capacity(2 * nq * nq);
    for (bank, first_state) in [0, 2].into_iter().enumerate() {
        for first_node in 0..nq {
            for second_node in first_node + 1..nq {
                let first = elastic_coefficients(first_state, first_node, second_node);
                let conjugate = elastic_coefficients(first_state + 1, first_node, second_node);
                let gain = 0.5 * (first.0 + conjugate.0);
                let loss = 0.5 * (first.1 + conjugate.1);
                if gain != 0.0 || loss != 0.0 {
                    edges.push(FoldedPauliEdge {
                        bank,
                        edge: PauliEdge::new(
                            PauliEdgeTopology::ElasticTransfer,
                            first_node,
                            second_node,
                            number_weights[first_node],
                            number_weights[second_node],
                            gain,
                            loss,
                        )?,
                    });
                }
            }
        }
        for first_node in 0..nq {
            for second_node in first_node..nq {
                let forward = pair_coefficients(first_state, first_node, second_node);
                let (gain, loss) = if first_node == second_node {
                    forward
                } else {
                    let transpose = pair_coefficients(first_state, second_node, first_node);
                    (
                        0.5 * (forward.0 + transpose.0),
                        0.5 * (forward.1 + transpose.1),
                    )
                };
                if gain != 0.0 || loss != 0.0 {
                    edges.push(FoldedPauliEdge {
                        bank,
                        edge: PauliEdge::new(
                            PauliEdgeTopology::PairSource,
                            first_node,
                            second_node,
                            number_weights[first_node],
                            number_weights[second_node],
                            gain,
                            loss,
                        )?,
                    });
                }
            }
        }
    }
    (!edges.is_empty())
        .then_some(IsotropicElectronPauliEdges { nq, edges })
        .ok_or("electron Pauli edge reconstruction is empty")
}

impl IsotropicElectronPauliEdges {
    fn record_edge_certificate(
        report: &mut PauliSweepReport,
        edge_report: PauliEdgeStep,
    ) -> Result<(), PauliEdgeFailureKind> {
        match edge_report.kind {
            Some(PauliEdgeApplicationKind::Solved) => {
                let finite_evidence = [
                    edge_report.extent,
                    edge_report.residual_abs,
                    edge_report.residual_scale,
                    edge_report.traffic_upper_bound_mev,
                    edge_report.flux_abs_error_mev,
                    edge_report.root_error_abs,
                    edge_report.occupation_error_abs,
                    edge_report.max_occupation_bracket_width,
                    edge_report.conditioning_lower_bound,
                ]
                .into_iter()
                .all(f64::is_finite);
                let consistent_evidence = (1..=96).contains(&edge_report.nonlinear_iterations)
                    && edge_report.residual_abs >= 0.0
                    && edge_report.residual_scale > 0.0
                    && edge_report.traffic_upper_bound_mev > 0.0
                    && edge_report.flux_abs_error_mev >= 0.0
                    && edge_report.flux_abs_error_mev <= edge_report.traffic_upper_bound_mev
                    && edge_report.root_error_abs >= edge_report.residual_abs
                    && edge_report.occupation_error_abs >= 0.0
                    && edge_report.occupation_error_abs <= 128.0 * f64::EPSILON
                    && edge_report.max_occupation_bracket_width >= 0.0
                    && (0.0..=1.0).contains(&edge_report.conditioning_lower_bound)
                    && edge_report.residual_abs
                        <= 128.0 * f64::EPSILON * edge_report.residual_scale + f64::from_bits(1);
                if !finite_evidence || !consistent_evidence {
                    return Err(PauliEdgeFailureKind::InvalidResidual);
                }
                report.edge_applications += 1;
                report.solved += 1;
                report.nonlinear_iterations += edge_report.nonlinear_iterations;
                report.maximum_edge_iterations = report
                    .maximum_edge_iterations
                    .max(edge_report.nonlinear_iterations);
                if edge_report.residual_scale.is_finite() && edge_report.residual_scale > 0.0 {
                    report.maximum_root_residual_ratio = report
                        .maximum_root_residual_ratio
                        .max(edge_report.residual_abs / edge_report.residual_scale);
                }
                report.maximum_occupation_bracket_width = report
                    .maximum_occupation_bracket_width
                    .max(edge_report.max_occupation_bracket_width);
                report.maximum_root_error_bound = report
                    .maximum_root_error_bound
                    .max(edge_report.root_error_abs);
                report.maximum_occupation_error_bound = report
                    .maximum_occupation_error_bound
                    .max(edge_report.occupation_error_abs);
                report.maximum_flux_error_fraction = report.maximum_flux_error_fraction.max(
                    edge_report.flux_abs_error_mev
                        / edge_report.traffic_upper_bound_mev.max(f64::from_bits(1)),
                );
            }
            Some(PauliEdgeApplicationKind::ExactStationary) => {
                report.edge_applications += 1;
                report.exact_stationary += 1;
            }
            None => report.edge_applications += 1,
        }
        Ok(())
    }

    fn checked_banks(&self, electron_pair: &[f64], heavy_pair: &[f64]) -> Result<(), &'static str> {
        if electron_pair.len() != self.nq
            || heavy_pair.len() != self.nq
            || electron_pair
                .iter()
                .chain(heavy_pair)
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("electron Pauli edge occupation bank is invalid");
        }
        Ok(())
    }

    pub(crate) fn action_values(
        &self,
        electron_pair: &[f64],
        heavy_pair: &[f64],
    ) -> Result<IsotropicElectronSpectralAction, &'static str> {
        self.checked_banks(electron_pair, heavy_pair)?;
        let mut action = vec![0.0; 2 * self.nq];
        for item in &self.edges {
            let bank_offset = item.bank * self.nq;
            let first = item.edge.first_node;
            let second = item.edge.second_node;
            let occupations = if item.bank == 0 {
                electron_pair
            } else {
                heavy_pair
            };
            let flux = item
                .edge
                .flux_mev(occupations[first], occupations[second])?;
            action[bank_offset + first] += flux / item.edge.first_measure;
            match item.edge.topology {
                PauliEdgeTopology::ElasticTransfer => {
                    action[bank_offset + second] -= flux / item.edge.second_measure;
                }
                PauliEdgeTopology::PairSource if first != second => {
                    action[bank_offset + second] += flux / item.edge.second_measure;
                }
                PauliEdgeTopology::PairSource => {}
            }
        }
        action
            .iter()
            .all(|value| value.is_finite())
            .then(|| IsotropicElectronSpectralAction {
                electron_pair_mev: action[..self.nq].to_vec(),
                heavy_pair_mev: action[self.nq..].to_vec(),
                jacobian_mev: Vec::new(),
            })
            .ok_or("electron Pauli edge action is non-finite")
    }

    pub(crate) fn transactional_step(
        &self,
        step_mev_inverse: f64,
        electron_pair: &[f64],
        heavy_pair: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, PauliSweepReport), PauliSweepFailure> {
        self.checked_banks(electron_pair, heavy_pair)
            .map_err(|_| PauliSweepFailure {
                kind: PauliEdgeFailureKind::InvalidInput,
                edge_index: None,
                partial_report: PauliSweepReport::empty(),
            })?;
        if !step_mev_inverse.is_finite() || step_mev_inverse < 0.0 {
            return Err(PauliSweepFailure {
                kind: PauliEdgeFailureKind::InvalidInput,
                edge_index: None,
                partial_report: PauliSweepReport::empty(),
            });
        }
        if step_mev_inverse == 0.0 {
            return Ok((
                electron_pair.to_vec(),
                heavy_pair.to_vec(),
                PauliSweepReport::empty(),
            ));
        }

        let mut electron_candidate = electron_pair.to_vec();
        let mut heavy_candidate = heavy_pair.to_vec();
        let mut report = PauliSweepReport::empty();
        let half_step = 0.5 * step_mev_inverse;
        for reverse in [false, true] {
            let apply = |item: &FoldedPauliEdge,
                         electron: &mut [f64],
                         heavy: &mut [f64]|
             -> Result<PauliEdgeStep, PauliEdgeFailure> {
                if item.bank == 0 {
                    item.edge.apply_implicit(half_step, electron)
                } else {
                    item.edge.apply_implicit(half_step, heavy)
                }
            };
            if reverse {
                for (edge_index, item) in self.edges.iter().enumerate().rev() {
                    match apply(item, &mut electron_candidate, &mut heavy_candidate) {
                        Ok(edge_report) => {
                            if let Err(kind) =
                                Self::record_edge_certificate(&mut report, edge_report)
                            {
                                return Err(PauliSweepFailure {
                                    kind,
                                    edge_index: Some(edge_index),
                                    partial_report: report,
                                });
                            }
                        }
                        Err(error) => {
                            report.unresolved +=
                                usize::from(error.kind == PauliEdgeFailureKind::UnresolvedFlux);
                            return Err(PauliSweepFailure {
                                kind: error.kind,
                                edge_index: Some(edge_index),
                                partial_report: report,
                            });
                        }
                    }
                }
            } else {
                for (edge_index, item) in self.edges.iter().enumerate() {
                    match apply(item, &mut electron_candidate, &mut heavy_candidate) {
                        Ok(edge_report) => {
                            if let Err(kind) =
                                Self::record_edge_certificate(&mut report, edge_report)
                            {
                                return Err(PauliSweepFailure {
                                    kind,
                                    edge_index: Some(edge_index),
                                    partial_report: report,
                                });
                            }
                        }
                        Err(error) => {
                            report.unresolved +=
                                usize::from(error.kind == PauliEdgeFailureKind::UnresolvedFlux);
                            return Err(PauliSweepFailure {
                                kind: error.kind,
                                edge_index: Some(edge_index),
                                partial_report: report,
                            });
                        }
                    }
                }
            }
        }
        self.checked_banks(&electron_candidate, &heavy_candidate)
            .map_err(|_| PauliSweepFailure {
                kind: PauliEdgeFailureKind::InvalidResidual,
                edge_index: None,
                partial_report: report,
            })?;
        Ok((electron_candidate, heavy_candidate, report))
    }
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
    let electron_pair_mev = (0..nq)
        .map(|node| folded_row(&explicit_action, nq, 0, node))
        .collect::<Vec<_>>();
    let heavy_pair_mev = (0..nq)
        .map(|node| folded_row(&explicit_action, nq, 2, node))
        .collect::<Vec<_>>();
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

pub(crate) fn reconstruct_isotropic_electron_pauli_edges(
    input: ElectronSpectralInput<'_>,
) -> Result<IsotropicElectronPauliEdges, &'static str> {
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
        None,
    )?;
    build_folded_pauli_edges(&stream, &explicit_f, y_nodes, y_weights)
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
    use super::*;
    use crate::electron_hm::G_F_MEV_MINUS_2;
    use crate::flrw::ELECTRON_MASS_MEV;
    use core::f64::consts::PI;

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
    fn unforced_fd_equilibrium_is_an_event_and_edge_null() {
        let (y, w) = grid(4);
        let occupation = y.iter().copied().map(fd).collect::<Vec<_>>();
        let input = spectral_input(
            1.0,
            1.0,
            &y,
            &w,
            &occupation,
            &occupation,
            ElectronSpectralRule {
                electron_radial_order: 4,
                angular_order: 3,
            },
        );
        let direct = evaluate_isotropic_electron_spectral_action(input).unwrap();
        let edges = reconstruct_isotropic_electron_pauli_edges(input).unwrap();
        let (zero_electron, zero_heavy, zero_report) = edges
            .transactional_step(0.0, &occupation, &occupation)
            .unwrap();
        assert_eq!(zero_electron, occupation);
        assert_eq!(zero_heavy, occupation);
        assert_eq!(zero_report, PauliSweepReport::empty());
        let reconstructed = edges.action_values(&occupation, &occupation).unwrap();
        let direct_l1 = direct
            .electron_pair_mev
            .iter()
            .chain(&direct.heavy_pair_mev)
            .map(|value| value.abs())
            .sum::<f64>();
        let reconstructed_l1 = reconstructed
            .electron_pair_mev
            .iter()
            .chain(&reconstructed.heavy_pair_mev)
            .map(|value| value.abs())
            .sum::<f64>();
        let mut net_l1 = 0.0;
        let mut traffic_l1 = 0.0;
        let mut maximum_normalized_flux: f64 = 0.0;
        let mut offenders = Vec::new();
        for item in &edges.edges {
            let bank = &occupation;
            let first = bank[item.edge.first_node];
            let second = bank[item.edge.second_node];
            let (gain_factor, loss_factor) = match item.edge.topology {
                PauliEdgeTopology::ElasticTransfer => {
                    ((1.0 - first) * second, first * (1.0 - second))
                }
                PauliEdgeTopology::PairSource => ((1.0 - first) * (1.0 - second), first * second),
            };
            let gain = item.edge.gain_coefficient_mev * gain_factor;
            let loss = item.edge.loss_coefficient_mev * loss_factor;
            let flux = item.edge.flux_mev(first, second).unwrap();
            let traffic = gain + loss;
            let normalized = flux.abs() / traffic.max(f64::MIN_POSITIVE);
            net_l1 += flux.abs();
            traffic_l1 += traffic;
            maximum_normalized_flux = maximum_normalized_flux.max(normalized);
            offenders.push((
                normalized,
                item.bank,
                item.edge.topology,
                item.edge.first_node,
                item.edge.second_node,
                flux,
                traffic,
            ));
        }
        offenders.sort_by(|left, right| right.0.total_cmp(&left.0));
        for offender in offenders.iter().take(10) {
            eprintln!(
                "raw DB offender normalized={:.17e} bank={} topology={:?} nodes={}/{} flux={:.17e} traffic={:.17e}",
                offender.0, offender.1, offender.2, offender.3, offender.4, offender.5, offender.6,
            );
        }
        let normalized_db = net_l1 / traffic_l1.max(f64::MIN_POSITIVE);
        eprintln!(
            "raw FD balance direct_L1={direct_l1:.17e} reconstructed_L1={reconstructed_l1:.17e} net_L1={net_l1:.17e} traffic_L1={traffic_l1:.17e} normalized_DB={normalized_db:.17e} max_edge={maximum_normalized_flux:.17e}",
        );
        assert!(
            normalized_db <= 1.0e-12,
            "raw reconstructed detailed balance failed"
        );
        assert!(
            maximum_normalized_flux <= 1.0e-12,
            "raw reconstructed edge detailed balance failed"
        );
        assert!(direct.jacobian_mev.iter().any(|value| *value != 0.0));
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
    fn unforced_equilibrium_jacobian_is_continuous() {
        let (y, w) = grid(4);
        let electron = y.iter().copied().map(fd).collect::<Vec<_>>();
        let heavy = electron.clone();
        let rule = ElectronSpectralRule {
            electron_radial_order: 4,
            angular_order: 3,
        };
        let base = evaluate_isotropic_electron_spectral_action(spectral_input(
            1.0, 1.0, &y, &w, &electron, &heavy, rule,
        ))
        .unwrap();
        let dimension = 2 * y.len();
        let step = 1.0e-6;
        for column in [0, y.len() + 1] {
            let evaluate = |direction: f64| {
                let mut shifted_electron = electron.clone();
                let mut shifted_heavy = heavy.clone();
                let target = if column < y.len() {
                    &mut shifted_electron[column]
                } else {
                    &mut shifted_heavy[column - y.len()]
                };
                let logit = target.ln() - (-*target).ln_1p();
                *target = 1.0 / (1.0 + (-(logit + direction * step)).exp());
                let action = evaluate_isotropic_electron_spectral_action(spectral_input(
                    1.0,
                    1.0,
                    &y,
                    &w,
                    &shifted_electron,
                    &shifted_heavy,
                    rule,
                ))
                .unwrap();
                [action.electron_pair_mev, action.heavy_pair_mev].concat()
            };
            let plus = evaluate(1.0);
            let minus = evaluate(-1.0);
            for row in 0..dimension {
                let centered_logit = (plus[row] - minus[row]) / (2.0 * step);
                let expected = base.jacobian_mev[row * dimension + column]
                    * if column < y.len() {
                        electron[column] * (1.0 - electron[column])
                    } else {
                        heavy[column - y.len()] * (1.0 - heavy[column - y.len()])
                    };
                assert!(
                    (centered_logit - expected).abs() <= 2.0e-34
                        || relative_error(centered_logit, expected) <= 2.0e-7,
                    "row={row} column={column} centered={centered_logit:.17e} expected={expected:.17e}",
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

    fn assert_action_reconstruction(
        reference: &IsotropicElectronSpectralAction,
        reconstructed: &IsotropicElectronSpectralAction,
    ) {
        for (label, expected, actual) in [
            (
                "electron",
                &reference.electron_pair_mev,
                &reconstructed.electron_pair_mev,
            ),
            (
                "heavy",
                &reference.heavy_pair_mev,
                &reconstructed.heavy_pair_mev,
            ),
        ] {
            let difference = expected
                .iter()
                .zip(actual)
                .map(|(left, right)| (left - right).abs())
                .sum::<f64>();
            let scale = expected
                .iter()
                .chain(actual)
                .map(|value| value.abs())
                .sum::<f64>()
                .max(f64::MIN_POSITIVE);
            assert!(
                difference <= 2.0e-11 * scale,
                "bank={label} difference={difference:.17e} scale={scale:.17e} relative={:.17e}",
                difference / scale,
            );
        }
    }

    fn focused_states(y: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let fd_state = y.iter().copied().map(fd).collect::<Vec<_>>();
        let alternating = |low: f64, high: f64| {
            y.iter()
                .enumerate()
                .map(|(node, value)| fd(*value) * if node.is_multiple_of(2) { low } else { high })
                .collect::<Vec<_>>()
        };
        let mut tail_electron = fd_state.clone();
        let mut tail_heavy = fd_state.clone();
        let last = y.len() - 1;
        tail_electron[last] = 1.0e-35;
        tail_heavy[last] = 1.0e-40;
        vec![
            (fd_state.clone(), fd_state.clone()),
            (alternating(0.91, 1.07), alternating(1.07, 0.91)),
            (alternating(0.73, 1.11), alternating(1.11, 0.73)),
            (tail_electron, tail_heavy),
            (
                vec![0.173, 0.631, 0.047, 0.812],
                vec![0.284, 0.519, 0.093, 0.741],
            ),
        ]
    }

    #[test]
    fn root_certificates_hold_across_legacy_small_step_scales() {
        let (y, w) = grid(4);
        let rule = ElectronSpectralRule {
            electron_radial_order: 4,
            angular_order: 3,
        };
        let states = focused_states(&y);
        let (electron, heavy) = &states[1];
        let input = spectral_input(1.15, 1.0, &y, &w, electron, heavy, rule);
        let edges = reconstruct_isotropic_electron_pauli_edges(input).unwrap();
        for exponent in [8_i32, 10, 12, 14, 16, 20, 24, 30] {
            let step = 2.0_f64.powi(-exponent);
            let (_, _, report) = edges
                .transactional_step(step, electron, heavy)
                .unwrap_or_else(|failure| panic!("h=2^-{exponent} failure={failure:?}"));
            eprintln!(
                "R3 tangent probe h={step:.17e} applications={} max_iterations={} max_residual_ratio={:.17e} max_occupation_width={:.17e}",
                report.edge_applications,
                report.maximum_edge_iterations,
                report.maximum_root_residual_ratio,
                report.maximum_occupation_bracket_width,
            );
            assert!(
                report.maximum_edge_iterations <= 4,
                "h=2^-{exponent} iterations={}",
                report.maximum_edge_iterations
            );
            assert_eq!(report.unresolved, 0, "h=2^-{exponent}");
            assert_eq!(report.exact_stationary, 0, "h=2^-{exponent}");
            assert!(report.solved > 0, "h=2^-{exponent}");
            assert_eq!(report.solved, report.edge_applications, "h=2^-{exponent}");
            assert!(
                report.maximum_occupation_error_bound <= 128.0 * f64::EPSILON,
                "h=2^-{exponent} occupation_error={:.17e}",
                report.maximum_occupation_error_bound,
            );
            assert!(
                [
                    report.maximum_root_residual_ratio,
                    report.maximum_occupation_bracket_width,
                    report.maximum_flux_error_fraction,
                    report.maximum_root_error_bound,
                    report.maximum_occupation_error_bound,
                ]
                .into_iter()
                .all(f64::is_finite),
                "h=2^-{exponent}"
            );
        }
    }

    #[test]
    fn subnormal_traffic_does_not_dilute_error_fraction() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 1.0, 1.0, 1.0e-310, 0.0).unwrap();
        let (_, edge_report) = edge.implicit_step(1.0, 0.5, 0.5).unwrap();
        assert_eq!(edge_report.kind, Some(PauliEdgeApplicationKind::Solved));
        assert!(edge_report.traffic_upper_bound_mev > 0.0);
        assert!(edge_report.traffic_upper_bound_mev < f64::MIN_POSITIVE);

        let expected_fraction = edge_report.flux_abs_error_mev
            / edge_report.traffic_upper_bound_mev.max(f64::from_bits(1));
        let mut sweep_report = PauliSweepReport::empty();
        IsotropicElectronPauliEdges::record_edge_certificate(&mut sweep_report, edge_report)
            .unwrap();

        assert_eq!(sweep_report.maximum_flux_error_fraction, expected_fraction);
    }

    #[test]
    fn malformed_solved_report_is_rejected_not_skipped() {
        let malformed = PauliEdgeStep {
            kind: Some(PauliEdgeApplicationKind::Solved),
            extent: 0.0,
            nonlinear_iterations: 1,
            residual_abs: f64::NAN,
            residual_scale: 1.0,
            traffic_upper_bound_mev: 1.0,
            flux_abs_error_mev: 0.0,
            root_error_abs: 0.0,
            occupation_error_abs: 0.0,
            max_occupation_bracket_width: 0.0,
            conditioning_lower_bound: 1.0,
        };
        let mut report = PauliSweepReport::empty();
        let result = IsotropicElectronPauliEdges::record_edge_certificate(&mut report, malformed);
        assert_eq!(result, Err(PauliEdgeFailureKind::InvalidResidual));
        assert_eq!(report, PauliSweepReport::empty());
    }

    #[test]
    fn sweep_report_counts_exact_stationary_without_nan_swallow() {
        let edges = IsotropicElectronPauliEdges {
            nq: 2,
            edges: vec![FoldedPauliEdge {
                bank: 0,
                edge: PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 1.0, 1.0, 0.0, 0.0)
                    .unwrap(),
            }],
        };
        let (_, _, report) = edges
            .transactional_step(0.25, &[0.2, 0.7], &[0.3, 0.6])
            .unwrap();
        assert_eq!(report.exact_stationary, 2);
        assert_eq!(report.solved, 0);
        assert_eq!(report.maximum_root_residual_ratio, 0.0);
        assert_eq!(report.maximum_occupation_bracket_width, 0.0);
        assert_eq!(report.maximum_occupation_error_bound, 0.0);
    }

    #[test]
    fn unresolved_edge_failure_is_transactional_and_observable() {
        let edges = IsotropicElectronPauliEdges {
            nq: 2,
            edges: vec![FoldedPauliEdge {
                bank: 0,
                edge: PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 1.0, 1.0, 1.0, 1.0)
                    .unwrap(),
            }],
        };
        let failure = edges
            .transactional_step(0.25, &[0.5, 0.5], &[0.3, 0.6])
            .unwrap_err();
        assert_eq!(failure.kind, PauliEdgeFailureKind::UnresolvedFlux);
        assert_eq!(failure.edge_index, Some(0));
        assert_eq!(failure.partial_report.unresolved, 1);
    }

    #[test]
    fn folded_edges_reconstruct_action_at_five_independent_states() {
        let (y, w) = grid(4);
        let rule = ElectronSpectralRule {
            electron_radial_order: 4,
            angular_order: 3,
        };
        let states = focused_states(&y);
        let (anchor_electron, anchor_heavy) = &states[0];
        let edges = reconstruct_isotropic_electron_pauli_edges(spectral_input(
            1.15,
            1.0,
            &y,
            &w,
            anchor_electron,
            anchor_heavy,
            rule,
        ))
        .unwrap();
        for (state, (electron, heavy)) in states.iter().enumerate() {
            let reference = evaluate_isotropic_electron_spectral_action_values(spectral_input(
                1.15, 1.0, &y, &w, electron, heavy, rule,
            ))
            .unwrap();
            let reconstructed = edges.action_values(electron, heavy).unwrap();
            assert_action_reconstruction(&reference, &reconstructed);
            eprintln!(
                "edge reconstruction state={state} action_L1={:.17e}",
                reconstructed
                    .electron_pair_mev
                    .iter()
                    .chain(&reconstructed.heavy_pair_mev)
                    .map(|value| value.abs())
                    .sum::<f64>()
            );
        }
    }

    #[test]
    fn folded_edges_are_boundary_inward_at_five_independent_states() {
        let (y, w) = grid(4);
        let rule = ElectronSpectralRule {
            electron_radial_order: 4,
            angular_order: 3,
        };
        let states = focused_states(&y);
        let (anchor_electron, anchor_heavy) = &states[0];
        let edges = reconstruct_isotropic_electron_pauli_edges(spectral_input(
            1.15,
            1.0,
            &y,
            &w,
            anchor_electron,
            anchor_heavy,
            rule,
        ))
        .unwrap();
        for (state, (electron, heavy)) in states.iter().enumerate() {
            for bank in 0..2 {
                for node in 0..y.len() {
                    let mut lower_electron = electron.clone();
                    let mut lower_heavy = heavy.clone();
                    let mut upper_electron = electron.clone();
                    let mut upper_heavy = heavy.clone();
                    if bank == 0 {
                        lower_electron[node] = 0.0;
                        upper_electron[node] = 1.0;
                    } else {
                        lower_heavy[node] = 0.0;
                        upper_heavy[node] = 1.0;
                    }
                    let lower = edges.action_values(&lower_electron, &lower_heavy).unwrap();
                    let upper = edges.action_values(&upper_electron, &upper_heavy).unwrap();
                    let lower_value = if bank == 0 {
                        lower.electron_pair_mev[node]
                    } else {
                        lower.heavy_pair_mev[node]
                    };
                    let upper_value = if bank == 0 {
                        upper.electron_pair_mev[node]
                    } else {
                        upper.heavy_pair_mev[node]
                    };
                    assert!(
                        lower_value >= 0.0,
                        "state={state} bank={bank} node={node} lower={lower_value:.17e}"
                    );
                    assert!(
                        upper_value <= 0.0,
                        "state={state} bank={bank} node={node} upper={upper_value:.17e}"
                    );
                }
            }
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
