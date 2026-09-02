//! Exact scalar collision-kernel primitives for the D-081R1 comparator lane.
//!
//! This module deliberately stops before collision-action assembly.  It freezes
//! the six-species catalogue, stable Pauli gain-minus-loss factor, event
//! measure, and weak matrix-element normalization used by the independent
//! Python F10 comparator.  D-081R1D may consume these primitives only after the
//! Python-generated fixture tests below pass.
#![cfg_attr(not(test), allow(dead_code))]

use crate::electron_hm::{G_F_MEV_MINUS_2, SIN2_THETA_W};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum F10Flavour {
    Electron,
    Muon,
    Tau,
}

impl F10Flavour {
    pub(crate) const ALL: [Self; 3] = [Self::Electron, Self::Muon, Self::Tau];
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum F10Species {
    NuE,
    AntiNuE,
    NuMu,
    AntiNuMu,
    NuTau,
    AntiNuTau,
}

impl F10Species {
    pub(crate) const ALL: [Self; 6] = [
        Self::NuE,
        Self::AntiNuE,
        Self::NuMu,
        Self::AntiNuMu,
        Self::NuTau,
        Self::AntiNuTau,
    ];

    pub(crate) const fn from_flavour(flavour: F10Flavour, anti: bool) -> Self {
        match (flavour, anti) {
            (F10Flavour::Electron, false) => Self::NuE,
            (F10Flavour::Electron, true) => Self::AntiNuE,
            (F10Flavour::Muon, false) => Self::NuMu,
            (F10Flavour::Muon, true) => Self::AntiNuMu,
            (F10Flavour::Tau, false) => Self::NuTau,
            (F10Flavour::Tau, true) => Self::AntiNuTau,
        }
    }

    pub(crate) const fn flavour(self) -> F10Flavour {
        match self {
            Self::NuE | Self::AntiNuE => F10Flavour::Electron,
            Self::NuMu | Self::AntiNuMu => F10Flavour::Muon,
            Self::NuTau | Self::AntiNuTau => F10Flavour::Tau,
        }
    }

    pub(crate) const fn is_antineutrino(self) -> bool {
        matches!(self, Self::AntiNuE | Self::AntiNuMu | Self::AntiNuTau)
    }

    pub(crate) const fn cp_partner(self) -> Self {
        Self::from_flavour(self.flavour(), !self.is_antineutrino())
    }

    pub(crate) const fn lepton_charge(self) -> i8 {
        if self.is_antineutrino() { -1 } else { 1 }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::NuE => "nu_e",
            Self::AntiNuE => "antinu_e",
            Self::NuMu => "nu_mu",
            Self::AntiNuMu => "antinu_mu",
            Self::NuTau => "nu_tau",
            Self::AntiNuTau => "antinu_tau",
        }
    }

    pub(crate) fn from_name(name: &str) -> Result<Self, F10KernelError> {
        match name {
            "nu_e" => Ok(Self::NuE),
            "antinu_e" => Ok(Self::AntiNuE),
            "nu_mu" => Ok(Self::NuMu),
            "antinu_mu" => Ok(Self::AntiNuMu),
            "nu_tau" => Ok(Self::NuTau),
            "antinu_tau" => Ok(Self::AntiNuTau),
            _ => Err(F10KernelError::UnknownSpecies),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10SelfCategory {
    SameSignIdentical,
    SamePairOppositeSignElastic,
    DistinctSameSignElastic,
    DistinctOppositeSignElastic,
    PairConversion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10SelfKernel {
    Ks,
    Kt,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10SelfEvent {
    pub(crate) legs: [F10Species; 4],
    pub(crate) category: F10SelfCategory,
    pub(crate) kernel: F10SelfKernel,
    pub(crate) coefficient: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10ElectronCategory {
    ElasticMinus,
    ElasticPlus,
    Pair,
}

impl F10ElectronCategory {
    pub(crate) fn from_name(name: &str) -> Result<Self, F10KernelError> {
        match name {
            "elastic_minus" => Ok(Self::ElasticMinus),
            "elastic_plus" => Ok(Self::ElasticPlus),
            "pair" => Ok(Self::Pair),
            _ => Err(F10KernelError::UnknownCategory),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct F10ElectronEvent {
    pub(crate) target: F10Species,
    pub(crate) category: F10ElectronCategory,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10InvariantProducts {
    pub(crate) d12: f64,
    pub(crate) d13: f64,
    pub(crate) d14: f64,
    pub(crate) d23: f64,
    pub(crate) d24: f64,
    pub(crate) d34: f64,
}

impl F10InvariantProducts {
    fn is_finite(self) -> bool {
        [self.d12, self.d13, self.d14, self.d23, self.d24, self.d34]
            .into_iter()
            .all(f64::is_finite)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10MatrixValue {
    pub(crate) value: f64,
    pub(crate) scale: f64,
    pub(crate) corrected: bool,
    pub(crate) correction: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10KernelError {
    NonFiniteInput,
    InvalidCoefficient,
    InvalidRoundoffBudget,
    MateriallyNegativeMatrix,
    InvalidMeasureDomain,
    UnknownSpecies,
    UnknownCategory,
}

pub(crate) fn f10_self_events() -> Vec<F10SelfEvent> {
    let mut events = Vec::with_capacity(27);

    for species in F10Species::ALL {
        events.push(F10SelfEvent {
            legs: [species; 4],
            category: F10SelfCategory::SameSignIdentical,
            kernel: F10SelfKernel::Ks,
            coefficient: 16.0,
        });
    }

    for flavour in F10Flavour::ALL {
        events.push(F10SelfEvent {
            legs: [
                F10Species::from_flavour(flavour, false),
                F10Species::from_flavour(flavour, true),
                F10Species::from_flavour(flavour, false),
                F10Species::from_flavour(flavour, true),
            ],
            category: F10SelfCategory::SamePairOppositeSignElastic,
            kernel: F10SelfKernel::Kt,
            coefficient: 64.0,
        });
    }

    for first_index in 0..F10Flavour::ALL.len() {
        let first = F10Flavour::ALL[first_index];
        for &second in &F10Flavour::ALL[first_index + 1..] {
            for anti in [false, true] {
                let a = F10Species::from_flavour(first, anti);
                let b = F10Species::from_flavour(second, anti);
                events.push(F10SelfEvent {
                    legs: [a, b, a, b],
                    category: F10SelfCategory::DistinctSameSignElastic,
                    kernel: F10SelfKernel::Ks,
                    coefficient: 16.0,
                });
            }
            for anti in [false, true] {
                let a = F10Species::from_flavour(first, anti);
                let b = F10Species::from_flavour(second, !anti);
                events.push(F10SelfEvent {
                    legs: [a, b, a, b],
                    category: F10SelfCategory::DistinctOppositeSignElastic,
                    kernel: F10SelfKernel::Kt,
                    coefficient: 16.0,
                });
            }
            for (source, destination) in [(first, second), (second, first)] {
                events.push(F10SelfEvent {
                    legs: [
                        F10Species::from_flavour(source, false),
                        F10Species::from_flavour(source, true),
                        F10Species::from_flavour(destination, false),
                        F10Species::from_flavour(destination, true),
                    ],
                    category: F10SelfCategory::PairConversion,
                    kernel: F10SelfKernel::Kt,
                    coefficient: 16.0,
                });
            }
        }
    }

    debug_assert_eq!(events.len(), 27);
    events
}

pub(crate) fn f10_electron_events() -> Vec<F10ElectronEvent> {
    let mut events = Vec::with_capacity(15);
    for target in F10Species::ALL {
        events.push(F10ElectronEvent {
            target,
            category: F10ElectronCategory::ElasticMinus,
        });
        events.push(F10ElectronEvent {
            target,
            category: F10ElectronCategory::ElasticPlus,
        });
    }
    for flavour in F10Flavour::ALL {
        events.push(F10ElectronEvent {
            target: F10Species::from_flavour(flavour, false),
            category: F10ElectronCategory::Pair,
        });
    }
    debug_assert_eq!(events.len(), 15);
    events
}

fn log_expit(value: f64) -> f64 {
    if value >= 0.0 {
        -(-value).exp().ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}

fn expit(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

/// Stable scalar form of
/// `(1-f1)(1-f2)f3f4 - f1f2(1-f3)(1-f4)`.
pub(crate) fn stable_pauli_gain_minus_loss(logits: [f64; 4]) -> Result<f64, F10KernelError> {
    if logits.into_iter().any(|value| !value.is_finite()) {
        return Err(F10KernelError::NonFiniteInput);
    }
    let [u1, u2, u3, u4] = logits;
    let affinity = u3 + u4 - u1 - u2;
    let log_loss = log_expit(u1) + log_expit(u2) + log_expit(-u3) + log_expit(-u4);
    let result = if affinity >= 0.0 {
        (log_loss + affinity).exp() * -(-affinity).exp_m1()
    } else {
        log_loss.exp() * affinity.exp_m1()
    };
    if result.is_finite() {
        Ok(result)
    } else {
        Err(F10KernelError::NonFiniteInput)
    }
}

pub(crate) fn pauli_logit_gradient(logits: [f64; 4]) -> Result<[f64; 4], F10KernelError> {
    if logits.into_iter().any(|value| !value.is_finite()) {
        return Err(F10KernelError::NonFiniteInput);
    }
    let [u1, u2, u3, u4] = logits;
    let [f1, f2, f3, f4] = [expit(u1), expit(u2), expit(u3), expit(u4)];
    let gain = (1.0 - f1) * (1.0 - f2) * f3 * f4;
    let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
    let gradient = [
        -f1 * gain - (1.0 - f1) * loss,
        -f2 * gain - (1.0 - f2) * loss,
        (1.0 - f3) * gain + f3 * loss,
        (1.0 - f4) * gain + f4 * loss,
    ];
    if gradient.into_iter().all(f64::is_finite) {
        Ok(gradient)
    } else {
        Err(F10KernelError::NonFiniteInput)
    }
}

fn checked_nonnegative_matrix(
    raw: f64,
    scale: f64,
    roundoff_ulps: f64,
) -> Result<F10MatrixValue, F10KernelError> {
    if !raw.is_finite() || !scale.is_finite() {
        return Err(F10KernelError::NonFiniteInput);
    }
    if !roundoff_ulps.is_finite() || roundoff_ulps <= 0.0 {
        return Err(F10KernelError::InvalidRoundoffBudget);
    }
    let nonnegative_scale = scale.abs().max(f64::MIN_POSITIVE);
    let tolerance = roundoff_ulps * f64::EPSILON * nonnegative_scale;
    if raw < -tolerance {
        return Err(F10KernelError::MateriallyNegativeMatrix);
    }
    let corrected = raw < 0.0;
    Ok(F10MatrixValue {
        value: if corrected { 0.0 } else { raw },
        scale: nonnegative_scale,
        corrected,
        correction: if corrected { -raw } else { 0.0 },
    })
}

pub(crate) fn f10_self_matrix(
    kernel: F10SelfKernel,
    coefficient: f64,
    invariants: F10InvariantProducts,
    support: bool,
    roundoff_ulps: f64,
) -> Result<F10MatrixValue, F10KernelError> {
    if !invariants.is_finite() {
        return Err(F10KernelError::NonFiniteInput);
    }
    if !coefficient.is_finite() || coefficient <= 0.0 {
        return Err(F10KernelError::InvalidCoefficient);
    }
    if !support {
        return Ok(F10MatrixValue {
            value: 0.0,
            scale: f64::MIN_POSITIVE,
            corrected: false,
            correction: 0.0,
        });
    }
    let contraction = match kernel {
        F10SelfKernel::Ks => invariants.d12 * invariants.d34,
        F10SelfKernel::Kt => invariants.d14 * invariants.d23,
    };
    let factor = coefficient * G_F_MEV_MINUS_2.powi(2);
    checked_nonnegative_matrix(
        factor * contraction,
        factor * contraction.abs(),
        roundoff_ulps,
    )
}

fn electron_couplings(target: F10Species) -> (f64, f64) {
    let left = if target.flavour() == F10Flavour::Electron {
        0.5 + SIN2_THETA_W
    } else {
        -0.5 + SIN2_THETA_W
    };
    (left, SIN2_THETA_W)
}

pub(crate) fn f10_electron_matrix(
    target: F10Species,
    category: F10ElectronCategory,
    invariants: F10InvariantProducts,
    electron_mass: f64,
    support: bool,
    roundoff_ulps: f64,
) -> Result<F10MatrixValue, F10KernelError> {
    if !invariants.is_finite() || !electron_mass.is_finite() || electron_mass < 0.0 {
        return Err(F10KernelError::NonFiniteInput);
    }
    if !support {
        return Ok(F10MatrixValue {
            value: 0.0,
            scale: f64::MIN_POSITIVE,
            corrected: false,
            correction: 0.0,
        });
    }

    let (mut left, mut right) = electron_couplings(target);
    match category {
        F10ElectronCategory::ElasticMinus if target.is_antineutrino() => {
            core::mem::swap(&mut left, &mut right);
        }
        F10ElectronCategory::ElasticPlus if !target.is_antineutrino() => {
            core::mem::swap(&mut left, &mut right);
        }
        F10ElectronCategory::Pair if target.is_antineutrino() => {
            core::mem::swap(&mut left, &mut right);
        }
        _ => {}
    }

    let ks = invariants.d12 * invariants.d34;
    let kt = invariants.d14 * invariants.d23;
    let ku = invariants.d13 * invariants.d24;
    let mass_squared = electron_mass * electron_mass;
    let interference_13 = mass_squared * invariants.d13;
    let interference_12 = mass_squared * invariants.d12;

    let terms = match category {
        F10ElectronCategory::ElasticMinus | F10ElectronCategory::ElasticPlus => [
            left * left * ks,
            right * right * kt,
            -left * right * interference_13,
        ],
        F10ElectronCategory::Pair => [
            left * left * kt,
            right * right * ku,
            left * right * interference_12,
        ],
    };
    let common = if category == F10ElectronCategory::Pair {
        128.0
    } else {
        64.0
    } * G_F_MEV_MINUS_2.powi(2);
    let raw = common * terms.into_iter().sum::<f64>();
    let scale = common * terms.into_iter().map(f64::abs).sum::<f64>();
    checked_nonnegative_matrix(raw, scale, roundoff_ulps)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10EventMeasureInput {
    pub(crate) p1: f64,
    pub(crate) p2: f64,
    pub(crate) e2: f64,
    pub(crate) phase_space: f64,
    pub(crate) quadrature_weight: f64,
    pub(crate) outer_weight: f64,
}

pub(crate) fn f10_event_measure(input: F10EventMeasureInput) -> Result<f64, F10KernelError> {
    let values = [
        input.p1,
        input.p2,
        input.e2,
        input.phase_space,
        input.quadrature_weight,
        input.outer_weight,
    ];
    if values.into_iter().any(|value| !value.is_finite()) {
        return Err(F10KernelError::NonFiniteInput);
    }
    if input.p1 <= 0.0
        || input.p2 < 0.0
        || input.e2 <= 0.0
        || input.phase_space < 0.0
        || input.quadrature_weight < 0.0
        || input.outer_weight < 0.0
    {
        return Err(F10KernelError::InvalidMeasureDomain);
    }
    let result =
        input.outer_weight * input.quadrature_weight * input.p2.powi(2) * input.phase_space
            / (input.e2 * 256.0 * core::f64::consts::PI.powi(4) * input.p1);
    if result.is_finite() && result >= 0.0 {
        Ok(result)
    } else {
        Err(F10KernelError::NonFiniteInput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../tests/fixtures/d081r1/collision_kernel_case.json");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("valid D-081R1C fixture")
    }

    fn bits(value: &Value) -> f64 {
        let encoded = value.as_str().expect("hex-bit string");
        let digits = encoded.strip_prefix("0x").expect("0x-prefixed bits");
        f64::from_bits(u64::from_str_radix(digits, 16).expect("valid f64 bits"))
    }

    fn bit_array<const N: usize>(value: &Value) -> [f64; N] {
        let source = value.as_array().expect("array");
        assert_eq!(source.len(), N);
        core::array::from_fn(|index| bits(&source[index]))
    }

    fn invariant_case(value: &Value, index: usize) -> F10InvariantProducts {
        F10InvariantProducts {
            d12: bits(&value["invariants"]["d12_bits"][index]),
            d13: bits(&value["invariants"]["d13_bits"][index]),
            d14: bits(&value["invariants"]["d14_bits"][index]),
            d23: bits(&value["invariants"]["d23_bits"][index]),
            d24: bits(&value["invariants"]["d24_bits"][index]),
            d34: bits(&value["invariants"]["d34_bits"][index]),
        }
    }

    fn scaled_residual(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    }

    #[test]
    fn fixture_binds_the_frozen_python_comparator() {
        let value = fixture();
        assert_eq!(value["schema"], "rabbit.d081r1.kernel_primitives.v1");
        assert_eq!(
            value["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(value["self_event_count"], 27);
        assert_eq!(value["electron_event_count"], 15);
        assert_eq!(
            bits(&value["constants"]["g_f_bits"]).to_bits(),
            G_F_MEV_MINUS_2.to_bits()
        );
        assert_eq!(
            bits(&value["constants"]["sin2_theta_w_bits"]).to_bits(),
            SIN2_THETA_W.to_bits()
        );
    }

    #[test]
    fn stable_pauli_factor_matches_python_fixture() {
        let value = fixture();
        for case in value["pauli_cases"].as_array().expect("Pauli cases") {
            let logits = bit_array::<4>(&case["logits_bits"]);
            let expected = bits(&case["expected_bits"]);
            let actual = stable_pauli_gain_minus_loss(logits).expect("finite Pauli factor");
            let tolerance = bits(&case["relative_tolerance_bits"]);
            assert!(
                scaled_residual(actual, expected) <= tolerance,
                "{}: actual={actual:.17e}, expected={expected:.17e}",
                case["name"]
            );
        }
    }

    #[test]
    fn detailed_balance_has_zero_value_and_nonzero_restoring_gradient() {
        let u1 = 0.7;
        let u2 = -0.2;
        let u3 = 1.1;
        let u4 = u1 + u2 - u3;
        let logits = [u1, u2, u3, u4];
        let collision = stable_pauli_gain_minus_loss(logits).unwrap();
        assert!(collision.abs() <= 32.0 * f64::EPSILON);

        let occupations = logits.map(expit);
        let gain =
            (1.0 - occupations[0]) * (1.0 - occupations[1]) * occupations[2] * occupations[3];
        let expected = [-gain, -gain, gain, gain];
        let gradient = pauli_logit_gradient(logits).unwrap();
        for (actual, reference) in gradient.into_iter().zip(expected) {
            assert!(scaled_residual(actual, reference) <= 64.0 * f64::EPSILON);
        }
    }

    #[test]
    fn self_matrices_match_python_fixture() {
        let value = fixture();
        let support = value["invariants"]["support"]
            .as_array()
            .expect("support array");
        for case in value["self_matrix_cases"].as_array().expect("self cases") {
            let kernel = match case["kernel"].as_str().unwrap() {
                "K_s" => F10SelfKernel::Ks,
                "K_t" => F10SelfKernel::Kt,
                other => panic!("unknown fixture kernel {other}"),
            };
            let coefficient = bits(&case["coefficient_bits"]);
            let expected = case["expected_value_bits"].as_array().unwrap();
            for index in 0..expected.len() {
                let actual = f10_self_matrix(
                    kernel,
                    coefficient,
                    invariant_case(&value, index),
                    support[index].as_bool().unwrap(),
                    1024.0,
                )
                .unwrap();
                let reference = bits(&expected[index]);
                assert!(scaled_residual(actual.value, reference) <= 32.0 * f64::EPSILON);
                assert!(!actual.corrected);
            }
        }
    }

    #[test]
    fn electron_matrices_match_python_fixture() {
        let value = fixture();
        let support = value["invariants"]["support"]
            .as_array()
            .expect("support array");
        let mass = bits(&value["constants"]["electron_mass_bits"]);
        for case in value["electron_matrix_cases"]
            .as_array()
            .expect("electron cases")
        {
            let target = F10Species::from_name(case["target"].as_str().unwrap()).unwrap();
            let category =
                F10ElectronCategory::from_name(case["category"].as_str().unwrap()).unwrap();
            let expected = case["expected_value_bits"].as_array().unwrap();
            for index in 0..expected.len() {
                let actual = f10_electron_matrix(
                    target,
                    category,
                    invariant_case(&value, index),
                    mass,
                    support[index].as_bool().unwrap(),
                    1024.0,
                )
                .unwrap();
                let reference = bits(&expected[index]);
                assert!(
                    scaled_residual(actual.value, reference) <= 64.0 * f64::EPSILON,
                    "{} / {} / {index}: actual={:.17e}, expected={reference:.17e}",
                    case["target"],
                    case["category"],
                    actual.value
                );
                assert!(!actual.corrected);
            }
        }
    }

    #[test]
    fn event_measure_matches_python_fixture() {
        let value = fixture();
        let case = &value["event_measure_case"];
        let actual = f10_event_measure(F10EventMeasureInput {
            p1: bits(&case["p1_bits"]),
            p2: bits(&case["p2_bits"]),
            e2: bits(&case["e2_bits"]),
            phase_space: bits(&case["phase_space_bits"]),
            quadrature_weight: bits(&case["quadrature_weight_bits"]),
            outer_weight: bits(&case["outer_weight_bits"]),
        })
        .unwrap();
        let expected = bits(&case["expected_bits"]);
        assert!(scaled_residual(actual, expected) <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn global_catalogues_preserve_unfurled_species_and_stoichiometry() {
        let self_events = f10_self_events();
        let electron_events = f10_electron_events();
        assert_eq!(self_events.len(), 27);
        assert_eq!(electron_events.len(), 15);

        let category_counts = [
            F10SelfCategory::SameSignIdentical,
            F10SelfCategory::SamePairOppositeSignElastic,
            F10SelfCategory::DistinctSameSignElastic,
            F10SelfCategory::DistinctOppositeSignElastic,
            F10SelfCategory::PairConversion,
        ]
        .map(|category| {
            self_events
                .iter()
                .filter(|event| event.category == category)
                .count()
        });
        assert_eq!(category_counts, [6, 3, 6, 6, 6]);

        for species in F10Species::ALL {
            let partner = species.cp_partner();
            assert_ne!(partner, species);
            assert_eq!(partner.cp_partner(), species);
            assert_eq!(F10Species::from_name(species.name()).unwrap(), species);
        }

        for event in self_events {
            let incoming_charge = event.legs[0].lepton_charge() + event.legs[1].lepton_charge();
            let outgoing_charge = event.legs[2].lepton_charge() + event.legs[3].lepton_charge();
            assert_eq!(incoming_charge, outgoing_charge);
        }

        assert_eq!(
            electron_events
                .iter()
                .filter(|event| event.category == F10ElectronCategory::Pair)
                .count(),
            3
        );
        assert!(
            electron_events
                .iter()
                .filter(|event| event.category == F10ElectronCategory::Pair)
                .all(|event| !event.target.is_antineutrino())
        );
    }

    #[test]
    fn roundoff_projection_is_typed_and_bounded() {
        let scale = 3.0;
        let tolerance = 1024.0 * f64::EPSILON * scale;
        let accepted = checked_nonnegative_matrix(-0.5 * tolerance, scale, 1024.0).unwrap();
        assert_eq!(accepted.value.to_bits(), 0.0_f64.to_bits());
        assert!(accepted.corrected);
        assert_eq!(accepted.correction.to_bits(), (0.5 * tolerance).to_bits());
        assert_eq!(
            checked_nonnegative_matrix(-2.0 * tolerance, scale, 1024.0),
            Err(F10KernelError::MateriallyNegativeMatrix)
        );
    }

    #[test]
    fn load_bearing_mutations_are_rejected_by_fixture_scale() {
        let value = fixture();
        let case = &value["electron_matrix_cases"][0];
        let expected = bits(&case["expected_value_bits"][0]);
        let invariants = invariant_case(&value, 0);
        let mass = bits(&value["constants"]["electron_mass_bits"]);
        let target = F10Species::from_name(case["target"].as_str().unwrap()).unwrap();
        let category = F10ElectronCategory::from_name(case["category"].as_str().unwrap()).unwrap();
        let correct = f10_electron_matrix(target, category, invariants, mass, true, 1024.0)
            .unwrap()
            .value;
        assert!(scaled_residual(correct, expected) <= 64.0 * f64::EPSILON);

        let missing_factor_two = 0.5 * correct;
        assert!(scaled_residual(missing_factor_two, expected) > 0.4);

        let (left, right) = electron_couplings(target);
        let ks = invariants.d12 * invariants.d34;
        let kt = invariants.d14 * invariants.d23;
        let wrong_sign_terms =
            left * left * ks + right * right * kt + left * right * mass.powi(2) * invariants.d13;
        let wrong_sign = 64.0 * G_F_MEV_MINUS_2.powi(2) * wrong_sign_terms;
        assert!(scaled_residual(wrong_sign, expected) > 1.0e-3);

        let wrong_measure = 2.0 * bits(&value["event_measure_case"]["expected_bits"]);
        let correct_measure = bits(&value["event_measure_case"]["expected_bits"]);
        assert!(scaled_residual(wrong_measure, correct_measure) > 0.4);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert_eq!(
            stable_pauli_gain_minus_loss([0.0, 0.0, f64::NAN, 0.0]),
            Err(F10KernelError::NonFiniteInput)
        );
        assert_eq!(
            f10_event_measure(F10EventMeasureInput {
                p1: 0.0,
                p2: 1.0,
                e2: 1.0,
                phase_space: 1.0,
                quadrature_weight: 1.0,
                outer_weight: 1.0,
            }),
            Err(F10KernelError::InvalidMeasureDomain)
        );
    }
}
