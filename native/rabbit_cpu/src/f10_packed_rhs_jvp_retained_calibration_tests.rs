//! Retained state-1200 calibration for the D-081R1F0 spectral-c JVP.
//!
//! This is the calibration lane.  It must complete before the preregistered
//! retained holdout oracle is generated or inspected.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::F10CombinedAction;
use crate::f10_packed_rhs::{F10PackedRhsConfig, evaluate_f10_packed_rhs};
use crate::f10_packed_rhs_jvp::evaluate_f10_packed_rhs_c_jvp;
use serde_json::Value;
use std::{env, fs};

const SPECTRAL_SIZE: usize = 180;
const STATE_SIZE: usize = 182;

fn bits(value: &Value) -> f64 {
    let encoded = value.as_str().expect("hex bit string");
    let digits = encoded.strip_prefix("0x").unwrap_or(encoded);
    f64::from_bits(u64::from_str_radix(digits, 16).expect("valid f64 bits"))
}

fn bit_array(value: &Value) -> Vec<f64> {
    value["bits"]
        .as_array()
        .expect("bit array")
        .iter()
        .map(bits)
        .collect()
}

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn maximum_absolute_difference(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn global_relative(actual: &[f64], expected: &[f64]) -> f64 {
    maximum_absolute_difference(actual, expected)
        / maximum_absolute(actual)
            .max(maximum_absolute(expected))
            .max(f64::MIN_POSITIVE)
}

fn scalar_relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
}

fn ordered_bits(value: f64) -> u64 {
    let raw = value.to_bits();
    if raw >> 63 == 0 {
        raw | (1_u64 << 63)
    } else {
        !raw
    }
}

fn maximum_local_relative(actual: &[f64], expected: &[f64]) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    let mut maximum_index = 0_usize;
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        let scale = left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
        let residual = (left - right).abs() / scale;
        if residual > maximum {
            maximum = residual;
            maximum_index = index;
        }
    }
    (maximum, maximum_index)
}

fn maximum_ulp(actual: &[f64], expected: &[f64]) -> (u64, usize) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0_u64;
    let mut maximum_index = 0_usize;
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        let distance = ordered_bits(left).abs_diff(ordered_bits(right));
        if distance > maximum {
            maximum = distance;
            maximum_index = index;
        }
    }
    (maximum, maximum_index)
}

fn assert_exact_array(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{label} bit mismatch at index {index}"
        );
    }
}

fn branch_signature(action: &F10CombinedAction) -> (usize, usize, u64) {
    (
        action.whole_reaction_domain_rejections,
        action.matrix_roundoff_corrections,
        action.largest_matrix_roundoff_correction.to_bits(),
    )
}

fn assert_below(value: f64, threshold: f64, label: &str) {
    assert!(
        value.is_finite() && value <= threshold,
        "{label} exceeded threshold: value={value:.17e}, threshold={threshold:.17e}"
    );
}

fn packed_block_residual(actual: &[f64], expected: &[f64]) -> (f64, f64, f64, f64) {
    assert_eq!(actual.len(), STATE_SIZE);
    assert_eq!(expected.len(), STATE_SIZE);
    let spectral = global_relative(&actual[..SPECTRAL_SIZE], &expected[..SPECTRAL_SIZE]);
    let temperature = scalar_relative(actual[SPECTRAL_SIZE], expected[SPECTRAL_SIZE]);
    let elapsed = scalar_relative(actual[SPECTRAL_SIZE + 1], expected[SPECTRAL_SIZE + 1]);
    (spectral.max(temperature).max(elapsed), spectral, temperature, elapsed)
}

fn fixture() -> Value {
    let path = env::var("D081R1F0_RETAINED_CALIBRATION_FIXTURE")
        .expect("D081R1F0_RETAINED_CALIBRATION_FIXTURE must identify the oracle");
    serde_json::from_str(&fs::read_to_string(path).expect("read retained calibration fixture"))
        .expect("valid retained calibration fixture")
}

#[test]
fn retained_state1200_c_only_jvp_matches_the_frozen_calibration_oracle() {
    let value = fixture();
    assert_eq!(value["schema"], "rabbit.d081r1f0.c_only_jvp_oracle.v1");
    assert_eq!(value["case"], "retained-calibration");
    assert_eq!(value["order"], 60);
    assert_eq!(
        value["metadata"]["retained_sha256"],
        "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
    );
    assert_eq!(value["contract_git_blob"], "ac7149fe5d5ec327cdc168d1eba7fe4a68ce3221");
    assert_eq!(value["python_tangent_git_blob"], "668f3fab76ffc3ad7f29335a79fcd5daf47d429e");
    assert_eq!(value["python_collision_jvp_git_blob"], "591a64702c58a2de265fb88636f186e2d1b7e019");
    assert_eq!(value["python_rhs_jvp_git_blob"], "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5");

    let grid = F10ActionGrid::affine_legendre(60, bits(&value["y_max_bits"])).unwrap();
    assert_exact_array(&grid.nodes, &bit_array(&value["grid_nodes"]), "GL60 nodes");
    assert_exact_array(
        &grid.weights,
        &bit_array(&value["grid_weights"]),
        "GL60 weights",
    );

    let state = bit_array(&value["packed_state"]);
    let direction = bit_array(&value["direction_cloglog"]);
    let ln_a = bits(&value["ln_a_bits"]);
    assert_eq!(state.len(), STATE_SIZE);
    assert_eq!(direction.len(), SPECTRAL_SIZE);
    let config = F10PackedRhsConfig::default();
    let result = evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &direction, config).unwrap();

    let thresholds = &value["frozen_thresholds"];
    let modal_cap = thresholds["retained_component_modal"].as_f64().unwrap();
    let packed_cap = thresholds["retained_packed_rhs_jvp"].as_f64().unwrap();
    let first_law_cap = thresholds["retained_first_law"].as_f64().unwrap();

    let expected_self_modal = bit_array(&value["collision"]["self_modal"]);
    let expected_electron_modal = bit_array(&value["collision"]["electron_modal"]);
    let expected_total_modal = bit_array(&value["collision"]["total_modal"]);
    let expected_self_native = bit_array(&value["collision"]["self_native"]);
    let expected_electron_native = bit_array(&value["collision"]["electron_native"]);
    let expected_total_native = bit_array(&value["collision"]["total_native"]);
    let expected_jvp = bit_array(&value["packed_rhs_jvp"]);

    let self_modal = global_relative(
        &result.combined_action.self_action.modal,
        &expected_self_modal,
    );
    let electron_modal = global_relative(
        &result.combined_action.electron_action.modal,
        &expected_electron_modal,
    );
    let total_modal = global_relative(
        &result.combined_action.modal_total,
        &expected_total_modal,
    );
    let self_native = global_relative(
        &result.combined_action.self_action.native,
        &expected_self_native,
    );
    let electron_native = global_relative(
        &result.combined_action.electron_action.native,
        &expected_electron_native,
    );
    let total_native = global_relative(
        &result.combined_action.native_total,
        &expected_total_native,
    );
    let (packed, spectral, temperature, elapsed) =
        packed_block_residual(&result.values, &expected_jvp);

    assert_below(self_modal, modal_cap, "retained self modal JVP");
    assert_below(electron_modal, modal_cap, "retained electron modal JVP");
    assert_below(total_modal, modal_cap, "retained total modal JVP");
    assert_below(packed, packed_cap, "retained packed-RHS JVP block maximum");

    assert_below(
        scalar_relative(
            result.delta_rho_neutrino,
            bits(&value["delta_rho_neutrino_bits"]),
        ),
        1.0e-7,
        "retained delta rho_nu",
    );
    assert_below(
        scalar_relative(
            result.delta_hubble_over_hubble,
            bits(&value["delta_hubble_over_hubble_bits"]),
        ),
        1.0e-7,
        "retained delta H/H",
    );
    assert_below(
        result.first_law_tangent_residual.abs(),
        first_law_cap,
        "Rust retained differentiated first law",
    );
    assert_below(
        bits(&value["collision"]["first_law_tangent_residual_bits"]).abs(),
        first_law_cap,
        "Python retained differentiated first law",
    );

    let self_number_ratio = result
        .combined_action
        .self_action
        .moments
        .signed_number_rate
        .abs()
        / result
            .combined_action
            .self_action
            .moments
            .absolute_number_rate
            .max(f64::MIN_POSITIVE);
    let self_energy_ratio = result
        .combined_action
        .self_action
        .moments
        .signed_energy_rate
        .abs()
        / result
            .combined_action
            .self_action
            .moments
            .absolute_energy_rate
            .max(f64::MIN_POSITIVE);
    assert_below(self_number_ratio, 2.0e-9, "retained self number tangent");
    assert_below(self_energy_ratio, 2.0e-9, "retained self energy tangent");
    assert_below(
        result.combined_action.charge_conjugation_residual,
        2.0e-9,
        "retained charge-conjugation tangent",
    );
    assert_below(
        result.combined_action.mu_tau_residual,
        2.0e-9,
        "retained mu/tau tangent",
    );

    let expected_branch = &value["collision"]["base_branch"];
    let expected_signature = (
        usize::try_from(
            expected_branch["whole_reaction_domain_rejections"]
                .as_u64()
                .unwrap(),
        )
        .unwrap(),
        usize::try_from(
            expected_branch["matrix_roundoff_corrections"]
                .as_u64()
                .unwrap(),
        )
        .unwrap(),
        bits(&expected_branch["largest_matrix_roundoff_correction_bits"]).to_bits(),
    );
    assert_eq!(branch_signature(&result.base.combined_action), expected_signature);

    let witnesses = value["centered_witnesses"].as_array().unwrap();
    assert_eq!(witnesses.len(), 1);
    let witness = &witnesses[0];
    assert!(witness["state_valid"].as_bool().unwrap());
    assert!(witness["same_support_and_correction_branch"].as_bool().unwrap());
    let epsilon = bits(&witness["epsilon_bits"]);
    let full_direction = bit_array(&value["direction_full"]);
    let plus_state: Vec<f64> = state
        .iter()
        .zip(&full_direction)
        .map(|(base, tangent)| base + epsilon * tangent)
        .collect();
    let minus_state: Vec<f64> = state
        .iter()
        .zip(&full_direction)
        .map(|(base, tangent)| base - epsilon * tangent)
        .collect();
    let plus = evaluate_f10_packed_rhs(&grid, ln_a, &plus_state, config).unwrap();
    let minus = evaluate_f10_packed_rhs(&grid, ln_a, &minus_state, config).unwrap();
    assert_eq!(branch_signature(&plus.combined_action), expected_signature);
    assert_eq!(branch_signature(&minus.combined_action), expected_signature);
    let finite_difference: Vec<f64> = plus
        .values
        .iter()
        .zip(&minus.values)
        .map(|(right, left)| (right - left) / (2.0 * epsilon))
        .collect();
    let (centered, centered_spectral, centered_temperature, centered_elapsed) =
        packed_block_residual(&finite_difference, &result.values);
    let centered_cap = thresholds["retained_centered_packed_rhs"].as_f64().unwrap();
    assert_below(centered, centered_cap, "retained centered packed-RHS witness");
    assert_below(
        witness["packed_residual"].as_f64().unwrap(),
        centered_cap,
        "Python retained centered packed-RHS witness",
    );

    let (packed_local, packed_local_index) =
        maximum_local_relative(&result.values, &expected_jvp);
    let (packed_ulp, packed_ulp_index) = maximum_ulp(&result.values, &expected_jvp);
    println!(
        "D081R1F0_RETAINED_CALIBRATION_PASS self_modal={self_modal:.17e} electron_modal={electron_modal:.17e} total_modal={total_modal:.17e} self_native={self_native:.17e} electron_native={electron_native:.17e} total_native={total_native:.17e} packed={packed:.17e} spectral={spectral:.17e} temperature={temperature:.17e} elapsed={elapsed:.17e} centered={centered:.17e} centered_spectral={centered_spectral:.17e} centered_temperature={centered_temperature:.17e} centered_elapsed={centered_elapsed:.17e} self_number={self_number_ratio:.17e} self_energy={self_energy_ratio:.17e} cp={:.17e} mu_tau={:.17e} packed_local={packed_local:.17e} packed_local_index={packed_local_index} packed_ulp={packed_ulp} packed_ulp_index={packed_ulp_index}",
        result.combined_action.charge_conjugation_residual,
        result.combined_action.mu_tau_residual,
    );
}
