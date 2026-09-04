//! Preregistered retained state-1200 holdout for the D-081R1F0 spectral-c JVP.
//!
//! The fixture is generated only after the order-eight and amended retained
//! calibration receipts are durable.  This test also checks the physical
//! mu/tau permutation covariance J(Sy) Sv = S J(y) v.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::F10CombinedAction;
use crate::f10_packed_rhs::{F10PackedRhsConfig, evaluate_f10_packed_rhs};
use crate::f10_packed_rhs_jvp::evaluate_f10_packed_rhs_c_jvp;
use serde_json::{Value, json};
use std::{env, fs};

const ORDER: usize = 60;
const SPECTRAL_SIZE: usize = 3 * ORDER;
const STATE_SIZE: usize = SPECTRAL_SIZE + 2;

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

fn maximum_absolute_difference(actual: &[f64], expected: &[f64]) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len());
    let mut maximum = 0.0_f64;
    let mut maximum_index = 0_usize;
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        let difference = (left - right).abs();
        if difference > maximum {
            maximum = difference;
            maximum_index = index;
        }
    }
    (maximum, maximum_index)
}

fn ordered_bits(value: f64) -> u64 {
    let raw = value.to_bits();
    if raw >> 63 == 0 {
        raw | (1_u64 << 63)
    } else {
        !raw
    }
}

#[derive(Clone, Copy, Debug)]
struct DifferenceStats {
    max_absolute: f64,
    max_absolute_index: usize,
    global_relative: f64,
    max_local_relative: f64,
    max_local_relative_index: usize,
    max_ulp: u64,
    max_ulp_index: usize,
}

fn difference_stats(actual: &[f64], expected: &[f64]) -> DifferenceStats {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let (max_absolute, max_absolute_index) = maximum_absolute_difference(actual, expected);
    let mut max_local_relative = 0.0_f64;
    let mut max_local_relative_index = 0_usize;
    let mut max_ulp = 0_u64;
    let mut max_ulp_index = 0_usize;
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(left.is_finite() && right.is_finite());
        let local_scale = left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
        let local_relative = (left - right).abs() / local_scale;
        let ulp = ordered_bits(left).abs_diff(ordered_bits(right));
        if local_relative > max_local_relative {
            max_local_relative = local_relative;
            max_local_relative_index = index;
        }
        if ulp > max_ulp {
            max_ulp = ulp;
            max_ulp_index = index;
        }
    }
    DifferenceStats {
        max_absolute,
        max_absolute_index,
        global_relative: max_absolute
            / maximum_absolute(actual)
                .max(maximum_absolute(expected))
                .max(f64::MIN_POSITIVE),
        max_local_relative,
        max_local_relative_index,
        max_ulp,
        max_ulp_index,
    }
}

fn stats_json(stats: DifferenceStats) -> Value {
    json!({
        "max_absolute": stats.max_absolute,
        "max_absolute_index": stats.max_absolute_index,
        "global_relative": stats.global_relative,
        "max_local_relative": stats.max_local_relative,
        "max_local_relative_index": stats.max_local_relative_index,
        "max_ulp": stats.max_ulp,
        "max_ulp_index": stats.max_ulp_index,
    })
}

fn scalar_relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
}

fn assert_below(value: f64, threshold: f64, label: &str) {
    assert!(
        value.is_finite() && value <= threshold,
        "{label} exceeded threshold: value={value:.17e}, threshold={threshold:.17e}"
    );
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

fn branch_json(action: &F10CombinedAction) -> Value {
    json!({
        "whole_reaction_domain_rejections": action.whole_reaction_domain_rejections,
        "matrix_roundoff_corrections": action.matrix_roundoff_corrections,
        "largest_matrix_roundoff_correction_bits": format!(
            "{:016x}",
            action.largest_matrix_roundoff_correction.to_bits()
        ),
    })
}

fn swap_three_blocks(values: &[f64], order: usize) -> Vec<f64> {
    assert!(values.len() == 3 * order || values.len() == 3 * order + 2);
    let mut swapped = values.to_vec();
    for node in 0..order {
        swapped.swap(order + node, 2 * order + node);
    }
    swapped
}

fn swap_six_blocks(values: &[f64], order: usize) -> Vec<f64> {
    assert_eq!(values.len(), 6 * order);
    let mut swapped = values.to_vec();
    for node in 0..order {
        swapped.swap(2 * order + node, 4 * order + node);
        swapped.swap(3 * order + node, 5 * order + node);
    }
    swapped
}

fn pair_average(values: &[f64], order: usize, pair: usize) -> Vec<f64> {
    assert!(pair < 3);
    assert_eq!(values.len(), 6 * order);
    (0..order)
        .map(|node| {
            0.5 * (values[(2 * pair) * order + node] + values[(2 * pair + 1) * order + node])
        })
        .collect()
}

fn pair_residual(mu: &[f64], tau: &[f64]) -> f64 {
    let numerator = maximum_absolute_difference(mu, tau).0;
    numerator / maximum_absolute(mu).max(maximum_absolute(tau))
}

#[test]
#[ignore = "preregistered retained state-1200 JVP holdout"]
fn retained_state1200_holdout_and_mu_tau_covariance() {
    let fixture_path = env::var("D081R1F0_RETAINED_HOLDOUT_FIXTURE")
        .expect("D081R1F0_RETAINED_HOLDOUT_FIXTURE must identify the oracle");
    let receipt_path = env::var("D081R1F0_RETAINED_HOLDOUT_RECEIPT")
        .expect("D081R1F0_RETAINED_HOLDOUT_RECEIPT must identify the receipt");
    let value: Value = serde_json::from_str(
        &fs::read_to_string(&fixture_path).expect("read retained holdout fixture"),
    )
    .expect("valid retained holdout fixture");

    assert_eq!(value["schema"], "rabbit.d081r1f0.c_only_jvp_oracle.v1");
    assert_eq!(value["case"], "retained-holdout");
    assert_eq!(value["order"], ORDER);
    assert_eq!(
        value["metadata"]["retained_sha256"],
        "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
    );
    assert_eq!(
        value["metadata"]["direction_definition"],
        "ve=0.25+cos(2phi); vmu=-0.15+sin(3phi); vtau=0.35cos(phi)-0.20sin(2phi); global L2 normalization"
    );
    assert_eq!(
        value["contract_git_blob"],
        "ac7149fe5d5ec327cdc168d1eba7fe4a68ce3221"
    );
    assert_eq!(
        value["python_tangent_git_blob"],
        "668f3fab76ffc3ad7f29335a79fcd5daf47d429e"
    );
    assert_eq!(
        value["python_collision_jvp_git_blob"],
        "591a64702c58a2de265fb88636f186e2d1b7e019"
    );
    assert_eq!(
        value["python_rhs_jvp_git_blob"],
        "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5"
    );

    let grid = F10ActionGrid::affine_legendre(ORDER, bits(&value["y_max_bits"])).unwrap();
    assert_exact_array(&grid.nodes, &bit_array(&value["grid_nodes"]), "GL60 nodes");
    assert_exact_array(
        &grid.weights,
        &bit_array(&value["grid_weights"]),
        "GL60 weights",
    );

    let state = bit_array(&value["packed_state"]);
    let direction = bit_array(&value["direction_cloglog"]);
    let full_direction = bit_array(&value["direction_full"]);
    let ln_a = bits(&value["ln_a_bits"]);
    assert_eq!(state.len(), STATE_SIZE);
    assert_eq!(direction.len(), SPECTRAL_SIZE);
    assert_eq!(full_direction.len(), STATE_SIZE);

    let config = F10PackedRhsConfig::default();
    let result = evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &direction, config).unwrap();
    let thresholds = &value["frozen_thresholds"];
    let modal_cap = thresholds["retained_component_modal"].as_f64().unwrap();
    let packed_cap = thresholds["retained_packed_rhs_jvp"].as_f64().unwrap();
    let first_law_cap = thresholds["retained_first_law"].as_f64().unwrap();
    let centered_cap = thresholds["retained_centered_packed_rhs"].as_f64().unwrap();
    assert_eq!(modal_cap.to_bits(), 1.0e-7_f64.to_bits());
    assert_eq!(packed_cap.to_bits(), 2.0e-4_f64.to_bits());
    assert_eq!(first_law_cap.to_bits(), 2.0e-9_f64.to_bits());
    assert_eq!(centered_cap.to_bits(), 2.0e-4_f64.to_bits());

    let expected_self_modal = bit_array(&value["collision"]["self_modal"]);
    let expected_electron_modal = bit_array(&value["collision"]["electron_modal"]);
    let expected_total_modal = bit_array(&value["collision"]["total_modal"]);
    let expected_self_native = bit_array(&value["collision"]["self_native"]);
    let expected_electron_native = bit_array(&value["collision"]["electron_native"]);
    let expected_total_native = bit_array(&value["collision"]["total_native"]);
    let expected_jvp = bit_array(&value["packed_rhs_jvp"]);

    let self_modal = difference_stats(
        &result.combined_action.self_action.modal,
        &expected_self_modal,
    );
    let electron_modal = difference_stats(
        &result.combined_action.electron_action.modal,
        &expected_electron_modal,
    );
    let total_modal = difference_stats(&result.combined_action.modal_total, &expected_total_modal);
    let self_native = difference_stats(
        &result.combined_action.self_action.native,
        &expected_self_native,
    );
    let electron_native = difference_stats(
        &result.combined_action.electron_action.native,
        &expected_electron_native,
    );
    let total_native = difference_stats(&result.combined_action.native_total, &expected_total_native);
    let packed = difference_stats(&result.values, &expected_jvp);

    assert_below(self_modal.global_relative, modal_cap, "holdout self modal JVP");
    assert_below(
        electron_modal.global_relative,
        modal_cap,
        "holdout electron modal JVP",
    );
    assert_below(total_modal.global_relative, modal_cap, "holdout total modal JVP");
    assert_below(packed.global_relative, packed_cap, "holdout packed-RHS JVP");

    let delta_rho_relative = scalar_relative(
        result.delta_rho_neutrino,
        bits(&value["delta_rho_neutrino_bits"]),
    );
    let delta_h_relative = scalar_relative(
        result.delta_hubble_over_hubble,
        bits(&value["delta_hubble_over_hubble_bits"]),
    );
    assert_below(delta_rho_relative, 1.0e-7, "holdout delta rho_nu");
    assert_below(delta_h_relative, 1.0e-7, "holdout delta H/H");
    assert_below(
        result.first_law_tangent_residual.abs(),
        first_law_cap,
        "Rust holdout differentiated first law",
    );
    assert_below(
        bits(&value["collision"]["first_law_tangent_residual_bits"]).abs(),
        first_law_cap,
        "Python holdout differentiated first law",
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
    assert_below(self_number_ratio, 2.0e-9, "holdout self number tangent");
    assert_below(self_energy_ratio, 2.0e-9, "holdout self energy tangent");
    assert_below(
        result.combined_action.charge_conjugation_residual,
        2.0e-9,
        "holdout charge-conjugation tangent",
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
    let centered_values: Vec<f64> = plus
        .values
        .iter()
        .zip(&minus.values)
        .map(|(right, left)| (right - left) / (2.0 * epsilon))
        .collect();
    let centered = difference_stats(&centered_values, &result.values);
    assert_below(
        centered.global_relative,
        centered_cap,
        "holdout centered packed-RHS witness",
    );
    assert_below(
        witness["packed_residual"].as_f64().unwrap(),
        centered_cap,
        "Python holdout centered packed-RHS witness",
    );

    let swapped_state = swap_three_blocks(&state, ORDER);
    let swapped_direction = swap_three_blocks(&direction, ORDER);
    let swapped_full_direction = swap_three_blocks(&full_direction, ORDER);
    assert_exact_array(
        &swap_three_blocks(&swapped_state, ORDER),
        &state,
        "state mu/tau swap involution",
    );
    assert_exact_array(
        &swap_three_blocks(&swapped_direction, ORDER),
        &direction,
        "direction mu/tau swap involution",
    );
    let mut reconstructed_swapped_full = swapped_direction.clone();
    reconstructed_swapped_full.extend_from_slice(&[0.0, 0.0]);
    assert_exact_array(
        &swapped_full_direction,
        &reconstructed_swapped_full,
        "swapped full direction",
    );

    let swapped_result = evaluate_f10_packed_rhs_c_jvp(
        &grid,
        ln_a,
        &swapped_state,
        &swapped_direction,
        config,
    )
    .unwrap();
    assert_eq!(
        branch_signature(&swapped_result.base.combined_action),
        expected_signature,
        "mu/tau-swapped base changed the support/correction branch"
    );

    let self_modal_covariance = difference_stats(
        &swapped_result.combined_action.self_action.modal,
        &swap_six_blocks(&result.combined_action.self_action.modal, ORDER),
    );
    let electron_modal_covariance = difference_stats(
        &swapped_result.combined_action.electron_action.modal,
        &swap_six_blocks(&result.combined_action.electron_action.modal, ORDER),
    );
    let total_modal_covariance = difference_stats(
        &swapped_result.combined_action.modal_total,
        &swap_six_blocks(&result.combined_action.modal_total, ORDER),
    );
    let self_native_covariance = difference_stats(
        &swapped_result.combined_action.self_action.native,
        &swap_six_blocks(&result.combined_action.self_action.native, ORDER),
    );
    let electron_native_covariance = difference_stats(
        &swapped_result.combined_action.electron_action.native,
        &swap_six_blocks(&result.combined_action.electron_action.native, ORDER),
    );
    let total_native_covariance = difference_stats(
        &swapped_result.combined_action.native_total,
        &swap_six_blocks(&result.combined_action.native_total, ORDER),
    );
    let packed_covariance = difference_stats(
        &swapped_result.values,
        &swap_three_blocks(&result.values, ORDER),
    );
    assert_below(
        self_modal_covariance.global_relative,
        modal_cap,
        "self-action mu/tau covariance",
    );
    assert_below(
        electron_modal_covariance.global_relative,
        modal_cap,
        "electron-action mu/tau covariance",
    );
    assert_below(
        total_modal_covariance.global_relative,
        modal_cap,
        "total-action mu/tau covariance",
    );
    assert_below(
        packed_covariance.global_relative,
        packed_cap,
        "packed-RHS mu/tau covariance",
    );

    let covariance_delta_rho = scalar_relative(
        swapped_result.delta_rho_neutrino,
        result.delta_rho_neutrino,
    );
    let covariance_delta_h = scalar_relative(
        swapped_result.delta_hubble_over_hubble,
        result.delta_hubble_over_hubble,
    );
    assert_below(covariance_delta_rho, 1.0e-7, "delta rho_nu covariance");
    assert_below(covariance_delta_h, 1.0e-7, "delta H/H covariance");
    assert_below(
        swapped_result.first_law_tangent_residual.abs(),
        first_law_cap,
        "swapped differentiated first law",
    );
    assert_below(
        swapped_result.combined_action.charge_conjugation_residual,
        2.0e-9,
        "swapped charge-conjugation tangent",
    );

    let swapped_plus_state: Vec<f64> = swapped_state
        .iter()
        .zip(&swapped_full_direction)
        .map(|(base, tangent)| base + epsilon * tangent)
        .collect();
    let swapped_minus_state: Vec<f64> = swapped_state
        .iter()
        .zip(&swapped_full_direction)
        .map(|(base, tangent)| base - epsilon * tangent)
        .collect();
    let swapped_plus =
        evaluate_f10_packed_rhs(&grid, ln_a, &swapped_plus_state, config).unwrap();
    let swapped_minus =
        evaluate_f10_packed_rhs(&grid, ln_a, &swapped_minus_state, config).unwrap();
    assert_eq!(
        branch_signature(&swapped_plus.combined_action),
        expected_signature,
        "swapped plus witness changed support/correction branch"
    );
    assert_eq!(
        branch_signature(&swapped_minus.combined_action),
        expected_signature,
        "swapped minus witness changed support/correction branch"
    );

    let original_mu = pair_average(&result.combined_action.native_total, ORDER, 1);
    let original_tau = pair_average(&result.combined_action.native_total, ORDER, 2);
    let swapped_mu = pair_average(&swapped_result.combined_action.native_total, ORDER, 1);
    let swapped_tau = pair_average(&swapped_result.combined_action.native_total, ORDER, 2);

    let payload = json!({
        "schema": "rabbit.d081r1f0.retained_holdout.v1",
        "classification": "PASS_WITH_RETAINED_STATE1200_C_ONLY_JVP_HOLDOUT_AND_MU_TAU_COVARIANCE_SCOPE",
        "identity": "J(Sy) Sv = S J(y) v",
        "fixture_case": "retained-holdout",
        "retained_state_sha256": value["metadata"]["retained_sha256"],
        "direction_definition": value["metadata"]["direction_definition"],
        "thresholds": {
            "component_modal": modal_cap,
            "packed_rhs": packed_cap,
            "first_law": first_law_cap,
            "centered_packed_rhs": centered_cap,
        },
        "cross_language": {
            "self_modal": stats_json(self_modal),
            "electron_modal": stats_json(electron_modal),
            "total_modal": stats_json(total_modal),
            "self_native_diagnostic": stats_json(self_native),
            "electron_native_diagnostic": stats_json(electron_native),
            "total_native_diagnostic": stats_json(total_native),
            "packed_rhs": stats_json(packed),
            "delta_rho_relative": delta_rho_relative,
            "delta_h_relative": delta_h_relative,
        },
        "conservation_and_symmetry": {
            "rust_first_law": result.first_law_tangent_residual,
            "python_first_law": bits(&value["collision"]["first_law_tangent_residual_bits"]),
            "self_number_ratio": self_number_ratio,
            "self_energy_ratio": self_energy_ratio,
            "charge_conjugation_residual": result.combined_action.charge_conjugation_residual,
            "raw_mu_tau_residual": pair_residual(&original_mu, &original_tau),
            "swapped_raw_mu_tau_residual": pair_residual(&swapped_mu, &swapped_tau),
        },
        "centered_witness": {
            "epsilon_bits": witness["epsilon_bits"],
            "rust": stats_json(centered),
            "python_packed_residual": witness["packed_residual"],
            "python_collision_modal_residual": witness["collision_modal_residual"],
        },
        "mu_tau_covariance": {
            "self_modal": stats_json(self_modal_covariance),
            "electron_modal": stats_json(electron_modal_covariance),
            "total_modal": stats_json(total_modal_covariance),
            "self_native_diagnostic": stats_json(self_native_covariance),
            "electron_native_diagnostic": stats_json(electron_native_covariance),
            "total_native_diagnostic": stats_json(total_native_covariance),
            "packed_rhs": stats_json(packed_covariance),
            "delta_rho_relative": covariance_delta_rho,
            "delta_h_relative": covariance_delta_h,
            "original_branch": branch_json(&result.base.combined_action),
            "swapped_branch": branch_json(&swapped_result.base.combined_action),
            "state_swap_is_involution": true,
            "direction_swap_is_involution": true,
        },
        "claim_ceiling": "static fixed-grid fixed-support spectral-c JVP holdout and mu/tau covariance only; no thermal input column, full Jacobian, solver, trajectory, endpoint, N_eff, performance, main integration, publication, or G-F10 movement",
    });
    fs::write(
        receipt_path,
        serde_json::to_string_pretty(&payload).unwrap() + "\n",
    )
    .expect("write retained holdout receipt");

    println!(
        "D081R1F0_RETAINED_HOLDOUT_PASS packed={:.17e} centered={:.17e} self_modal={:.17e} electron_modal={:.17e} total_modal={:.17e} covariance_self={:.17e} covariance_electron={:.17e} covariance_total={:.17e} covariance_packed={:.17e}",
        packed.global_relative,
        centered.global_relative,
        self_modal.global_relative,
        electron_modal.global_relative,
        total_modal.global_relative,
        self_modal_covariance.global_relative,
        electron_modal_covariance.global_relative,
        total_modal_covariance.global_relative,
        packed_covariance.global_relative,
    );
}
