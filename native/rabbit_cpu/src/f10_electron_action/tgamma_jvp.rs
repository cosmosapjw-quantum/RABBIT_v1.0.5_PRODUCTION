//! Pair-only analytic T_gamma tangent, at fixed neutrino grid and spectra.
//! The full electron-action thermal derivative is deliberately not exposed.

#![cfg_attr(not(test), allow(dead_code))]

use super::*;
use crate::f10_action_spectral::{interpolate, modal_coefficients};
use crate::f10_tgamma_kinematics::mapped_modal_basis_derivative;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10ElasticPauliLocationTangent {
    pub(crate) logits: [f64; 4],
    pub(crate) d_outgoing_logit: f64,
    pub(crate) base: f64,
    pub(crate) derivative: f64,
}

/// Differentiate one elastic Pauli factor at fixed incoming neutrino spectrum.
/// The outgoing neutrino logit moves only because its spectral query location
/// changes; electron/positron bath logits include both energy and temperature.
#[allow(clippy::too_many_arguments)]
pub(crate) fn elastic_pauli_location_tangent(
    grid: &F10ActionGrid,
    target_logits: &[f64],
    incoming_logit: f64,
    electron_energy: f64,
    d_electron_energy: f64,
    outgoing_bath_energy: f64,
    d_outgoing_bath_energy: f64,
    temperature_gamma: f64,
    outgoing_y: f64,
    d_outgoing_y: f64,
) -> Result<F10ElasticPauliLocationTangent, F10ElectronActionError> {
    if target_logits.len() != grid.order
        || !incoming_logit.is_finite()
        || !electron_energy.is_finite()
        || !d_electron_energy.is_finite()
        || !outgoing_bath_energy.is_finite()
        || !d_outgoing_bath_energy.is_finite()
        || !temperature_gamma.is_finite()
        || temperature_gamma <= 0.0
        || !outgoing_y.is_finite()
        || !d_outgoing_y.is_finite()
    {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let outgoing_logit = interpolate(grid, target_logits, &[outgoing_y])
        .map_err(|_| F10ElectronActionError::Foundation)?[0];
    let coefficients =
        modal_coefficients(grid, target_logits).map_err(|_| F10ElectronActionError::Foundation)?;
    let derivative_basis = mapped_modal_basis_derivative(grid, &[outgoing_y])
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let d_spectral_location = coefficients
        .iter()
        .zip(&derivative_basis)
        .map(|(coefficient, derivative)| coefficient * derivative)
        .sum::<f64>()
        * d_outgoing_y;
    let inverse_temperature = temperature_gamma.recip();
    let electron_logit = -electron_energy * inverse_temperature;
    let outgoing_bath_logit = -outgoing_bath_energy * inverse_temperature;
    let d_electron_logit =
        -d_electron_energy * inverse_temperature + electron_energy * inverse_temperature.powi(2);
    let d_outgoing_bath_logit = -d_outgoing_bath_energy * inverse_temperature
        + outgoing_bath_energy * inverse_temperature.powi(2);
    let logits = [
        incoming_logit,
        electron_logit,
        outgoing_logit,
        outgoing_bath_logit,
    ];
    let base = stable_pauli_gain_minus_loss(logits)?;
    let derivative = stable_pauli_jvp(
        logits,
        [
            0.0,
            d_electron_logit,
            d_spectral_location,
            d_outgoing_bath_logit,
        ],
    )?;
    if !d_spectral_location.is_finite() || !base.is_finite() || !derivative.is_finite() {
        return Err(F10ElectronActionError::NonFiniteOutput);
    }
    Ok(F10ElasticPauliLocationTangent {
        logits,
        d_outgoing_logit: d_spectral_location,
        base,
        derivative,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct F10PairActionTgammaJvp {
    pub(crate) base_modal: Vec<f64>,
    pub(crate) base_native: Vec<f64>,
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) family_names: Vec<String>,
    pub(crate) family_modal: Vec<f64>,
    pub(crate) family_native: Vec<f64>,
    pub(crate) energy_by_family: Vec<[f64; 2]>,
    pub(crate) measure_modal: Vec<f64>,
    pub(crate) matrix_modal: Vec<f64>,
    pub(crate) projection_modal: Vec<f64>,
    /// Geometry mask in outer-node/sample order, not repeated over flavours.
    pub(crate) support: Vec<bool>,
    /// Matrix-correction mask in outer-node/flavour/sample order.
    pub(crate) corrected: Vec<bool>,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    /// Absolute first-law residual divided by sum of absolute sector rates.
    pub(crate) first_law_residual: f64,
}

fn log_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        -(-value).exp().ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// d(G-L) = (G-L) d(log L) + G d(affinity).
/// G is evaluated from log probabilities, never as L*exp(affinity).
pub(crate) fn stable_pauli_jvp(
    logits: [f64; 4],
    direction: [f64; 4],
) -> Result<f64, F10ElectronActionError> {
    if logits.iter().chain(&direction).any(|x| !x.is_finite()) {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let [u1, u2, u3, u4] = logits;
    let [v1, v2, v3, v4] = direction;
    let phi = stable_pauli_gain_minus_loss(logits)?;
    let log_gain = log_sigmoid(-u1) + log_sigmoid(-u2) + log_sigmoid(u3) + log_sigmoid(u4);
    let gain = log_gain.exp();
    let d_log_loss = sigmoid(-u1) * v1 + sigmoid(-u2) * v2 - sigmoid(u3) * v3 - sigmoid(u4) * v4;
    let d_affinity = v3 + v4 - v1 - v2;
    let result = phi * d_log_loss + gain * d_affinity;
    if !result.is_finite() {
        return Err(F10ElectronActionError::NonFiniteOutput);
    }
    Ok(result)
}

pub(crate) fn assemble_pair_action_tgamma_jvp(
    grid: &F10ActionGrid,
    pair_cloglog: &[f64],
    temperature_cm: f64,
    temperature_gamma: f64,
    config: F10ElectronActionConfig,
) -> Result<F10PairActionTgammaJvp, F10ElectronActionError> {
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
        .and_then(|x| x.checked_mul(rule.azimuth.len()))
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let family_size = PAIR_EVENT_COUNT
        .checked_mul(size)
        .ok_or(F10ElectronActionError::DimensionOverflow)?;
    let logits = decode_pair_logits(grid, pair_cloglog)?;
    let basis = modal_basis(grid, &grid.nodes).map_err(|_| F10ElectronActionError::Foundation)?;
    let p2: Vec<f64> = grid.nodes.iter().map(|y| temperature_cm * y).collect();
    let weights: Vec<f64> = grid.weights.iter().map(|w| temperature_cm * w).collect();
    if p2
        .iter()
        .chain(&weights)
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err(F10ElectronActionError::InvalidInput);
    }
    let catalogue = f10_electron_events();
    if catalogue.len() != ELECTRON_EVENT_COUNT {
        return Err(F10ElectronActionError::InvalidConfiguration);
    }
    let events = &catalogue[ELASTIC_EVENT_COUNT..];
    let mut base_modal = vec![0.0_f64; size];
    let mut modal = vec![0.0_f64; size];
    let mut family_modal = vec![0.0_f64; family_size];
    let mut energy_by_family = vec![[0.0_f64; 2]; PAIR_EVENT_COUNT];
    let mut support = Vec::new();
    let mut corrected = Vec::new();

    for node in 0..grid.order {
        let p1 = temperature_cm * grid.nodes[node];
        let outer_weight =
            temperature_cm.powi(3) * grid.weights[node] * grid.nodes[node].powi(2) / TWO_PI_SQUARED;
        let batch = two_body_kinematics(F10KinematicInput {
            p1,
            p2_nodes: &p2,
            p2_weights: &weights,
            mass2: 0.0,
            mass3: config.electron_mass_mev,
            mass4: config.electron_mass_mev,
            config: config.collision,
        })
        .map_err(|_| F10ElectronActionError::Kinematics)?;
        let count = batch.support.len();
        if count
            != grid
                .order
                .checked_mul(angular_size)
                .ok_or(F10ElectronActionError::DimensionOverflow)?
        {
            return Err(F10ElectronActionError::Kinematics);
        }
        support.extend_from_slice(&batch.support);

        for (family, &event) in events.iter().enumerate() {
            let mut base_rates = vec![0.0_f64; count];
            let mut rates = vec![0.0_f64; count];
            let u1 = logits[pair_index(event.target) * grid.order + node];
            for sample in 0..count {
                let matrix = f10_electron_matrix(
                    event.target,
                    event.category,
                    invariant_products(&batch, sample),
                    config.electron_mass_mev,
                    batch.support[sample],
                    config.matrix_roundoff_ulps,
                )?;
                corrected.push(matrix.corrected);
                if !batch.support[sample] {
                    continue;
                }
                let measure = f10_event_measure(F10EventMeasureInput {
                    p1,
                    p2: batch.p2[sample],
                    e2: batch.e2[sample],
                    phase_space: batch.phase_space[sample],
                    quadrature_weight: batch.quadrature_weight[sample],
                    outer_weight,
                })?;
                let p2_node = sample / angular_size;
                let u2 = logits[pair_index(event.target.cp_partner()) * grid.order + p2_node];
                let u3 = -batch.e3[sample] / temperature_gamma;
                let u4 = -batch.e4[sample] / temperature_gamma;
                // Incoming spectra and geometry are fixed: only outgoing bath logits move.
                let direction = [
                    0.0,
                    0.0,
                    (batch.e3[sample] / temperature_gamma) / temperature_gamma,
                    (batch.e4[sample] / temperature_gamma) / temperature_gamma,
                ];
                let primal =
                    measure * matrix.value * stable_pauli_gain_minus_loss([u1, u2, u3, u4])?;
                let tangent =
                    measure * matrix.value * stable_pauli_jvp([u1, u2, u3, u4], direction)?;
                if !primal.is_finite() || !tangent.is_finite() {
                    return Err(F10ElectronActionError::NonFiniteOutput);
                }
                base_rates[sample] = primal;
                rates[sample] = tangent;
                energy_by_family[family][0] += tangent * (p1 + batch.p2[sample]);
                energy_by_family[family][1] += tangent * (-batch.e3[sample] - batch.e4[sample]);
            }

            let incoming1 = rates.iter().sum::<f64>();
            let base_incoming1 = base_rates.iter().sum::<f64>();
            let mut incoming2 = vec![0.0_f64; grid.order];
            let mut base_incoming2 = vec![0.0_f64; grid.order];
            for sample in 0..count {
                incoming2[sample / angular_size] += rates[sample];
                base_incoming2[sample / angular_size] += base_rates[sample];
            }
            for (species, first) in [(event.target, true), (event.target.cp_partner(), false)] {
                let row = species_index(species);
                for mode in 0..grid.order {
                    let value = if first {
                        incoming1 * basis[node * grid.order + mode]
                    } else {
                        (0..grid.order)
                            .map(|j| incoming2[j] * basis[j * grid.order + mode])
                            .sum::<f64>()
                    };
                    let base_value = if first {
                        base_incoming1 * basis[node * grid.order + mode]
                    } else {
                        (0..grid.order)
                            .map(|j| base_incoming2[j] * basis[j * grid.order + mode])
                            .sum::<f64>()
                    };
                    modal[row * grid.order + mode] += value;
                    base_modal[row * grid.order + mode] += base_value;
                    family_modal[family * size + row * grid.order + mode] += value;
                }
            }
        }
    }
    let native = native_action(grid, &modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let base_native = native_action(grid, &base_modal, SPECIES_COUNT, temperature_cm)
        .map_err(|_| F10ElectronActionError::Foundation)?;
    let family_native = native_action(
        grid,
        &family_modal,
        PAIR_EVENT_COUNT * SPECIES_COUNT,
        temperature_cm,
    )
    .map_err(|_| F10ElectronActionError::Foundation)?;
    let neutrino_energy_transfer = energy_by_family.iter().map(|q| q[0]).sum::<f64>();
    let electromagnetic_energy_transfer = energy_by_family.iter().map(|q| q[1]).sum::<f64>();
    let first_law_residual = (neutrino_energy_transfer + electromagnetic_energy_transfer).abs()
        / (neutrino_energy_transfer.abs() + electromagnetic_energy_transfer.abs())
            .max(f64::MIN_POSITIVE);
    if modal
        .iter()
        .chain(&native)
        .chain(&base_modal)
        .chain(&base_native)
        .chain(&family_modal)
        .chain(&family_native)
        .any(|x| !x.is_finite())
        || energy_by_family.iter().flatten().any(|x| !x.is_finite())
        || !first_law_residual.is_finite()
    {
        return Err(F10ElectronActionError::NonFiniteOutput);
    }
    Ok(F10PairActionTgammaJvp {
        base_modal,
        base_native,
        modal,
        native,
        family_names: events.iter().map(|&event| family_name(event)).collect(),
        family_modal,
        family_native,
        energy_by_family,
        measure_modal: vec![0.0; size],
        matrix_modal: vec![0.0; size],
        projection_modal: vec![0.0; size],
        support,
        corrected,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
    })
}
