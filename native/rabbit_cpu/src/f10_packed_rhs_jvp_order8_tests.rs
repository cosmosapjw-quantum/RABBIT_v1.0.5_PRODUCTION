//! Order-eight nonzero validation for the D-081R1F0 spectral-c JVP.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::{F10CombinedAction, F10CombinedActionConfig, assemble_combined_action};
use crate::f10_packed_rhs::{F10PackedRhsConfig, F10PackedRhsError, evaluate_f10_packed_rhs};
use crate::f10_packed_rhs_jvp::evaluate_f10_packed_rhs_c_jvp;
use serde_json::Value;
use std::{env, fs};

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

fn fixture() -> Value {
    let path = env::var("D081R1F0_ORDER8_FIXTURE")
        .expect("D081R1F0_ORDER8_FIXTURE must identify the generated oracle");
    serde_json::from_str(&fs::read_to_string(path).expect("read order-eight JVP fixture"))
        .expect("valid order-eight JVP fixture")
}

fn branch_signature(action: &F10CombinedAction) -> (usize, usize, u64) {
    (
        action.whole_reaction_domain_rejections,
        action.matrix_roundoff_corrections,
        action.largest_matrix_roundoff_correction.to_bits(),
    )
}

fn normalized(mut values: Vec<f64>) -> Vec<f64> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    assert!(norm.is_finite() && norm > 0.0);
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn assert_below(value: f64, threshold: f64, label: &str) {
    assert!(
        value.is_finite() && value <= threshold,
        "{label} exceeded threshold: value={value:.17e}, threshold={threshold:.17e}"
    );
}

#[test]
fn order8_nonzero_jvp_matches_the_frozen_python_oracle() {
    let value = fixture();
    assert_eq!(value["schema"], "rabbit.d081r1f0.c_only_jvp_oracle.v1");
    assert_eq!(value["case"], "order8");
    assert_eq!(value["order"], 8);
    assert_eq!(value["contract_git_blob"], "ac7149fe5d5ec327cdc168d1eba7fe4a68ce3221");
    assert_eq!(value["python_tangent_git_blob"], "668f3fab76ffc3ad7f29335a79fcd5daf47d429e");
    assert_eq!(value["python_collision_jvp_git_blob"], "591a64702c58a2de265fb88636f186e2d1b7e019");
    assert_eq!(value["python_rhs_jvp_git_blob"], "6bcff2bc5627c0af0ad4df61c908d09e62ffaba5");

    let order = 8_usize;
    let grid = F10ActionGrid::affine_legendre(order, bits(&value["y_max_bits"])).unwrap();
    let state = bit_array(&value["packed_state"]);
    let direction = bit_array(&value["direction_cloglog"]);
    let ln_a = bits(&value["ln_a_bits"]);
    let config = F10PackedRhsConfig::default();
    let result = evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &direction, config).unwrap();

    let thresholds = &value["frozen_thresholds"];
    let expected_self_modal = bit_array(&value["collision"]["self_modal"]);
    let expected_electron_modal = bit_array(&value["collision"]["electron_modal"]);
    let expected_total_modal = bit_array(&value["collision"]["total_modal"]);
    let expected_self_native = bit_array(&value["collision"]["self_native"]);
    let expected_electron_native = bit_array(&value["collision"]["electron_native"]);
    let expected_total_native = bit_array(&value["collision"]["total_native"]);
    let expected_jvp = bit_array(&value["packed_rhs_jvp"]);

    let self_modal_residual = global_relative(
        &result.combined_action.self_action.modal,
        &expected_self_modal,
    );
    let electron_modal_residual = global_relative(
        &result.combined_action.electron_action.modal,
        &expected_electron_modal,
    );
    let total_modal_residual = global_relative(
        &result.combined_action.modal_total,
        &expected_total_modal,
    );
    let self_native_residual = global_relative(
        &result.combined_action.self_action.native,
        &expected_self_native,
    );
    let electron_native_residual = global_relative(
        &result.combined_action.electron_action.native,
        &expected_electron_native,
    );
    let total_native_residual = global_relative(
        &result.combined_action.native_total,
        &expected_total_native,
    );
    let packed_residual = global_relative(&result.values, &expected_jvp);

    assert_below(
        self_modal_residual,
        thresholds["order8_self_modal"].as_f64().unwrap(),
        "self modal JVP",
    );
    assert_below(
        electron_modal_residual,
        thresholds["order8_electron_modal"].as_f64().unwrap(),
        "electron modal JVP",
    );
    assert_below(
        total_modal_residual,
        thresholds["order8_total_modal"].as_f64().unwrap(),
        "total modal JVP",
    );
    assert_below(
        packed_residual,
        thresholds["order8_packed_rhs_jvp"].as_f64().unwrap(),
        "packed RHS JVP",
    );

    let expected_delta_rho = bits(&value["delta_rho_neutrino_bits"]);
    let expected_delta_h = bits(&value["delta_hubble_over_hubble_bits"]);
    assert_below(
        scalar_relative(result.delta_rho_neutrino, expected_delta_rho),
        1.0e-9,
        "delta rho_nu",
    );
    assert_below(
        scalar_relative(result.delta_hubble_over_hubble, expected_delta_h),
        1.0e-9,
        "delta H/H",
    );

    let first_law_cap = thresholds["order8_first_law"].as_f64().unwrap();
    assert_below(
        result.first_law_tangent_residual.abs(),
        first_law_cap,
        "Rust differentiated first law",
    );
    assert_below(
        bits(&value["collision"]["first_law_tangent_residual_bits"]).abs(),
        first_law_cap,
        "Python differentiated first law",
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
    assert_below(self_number_ratio, 2.0e-9, "self number tangent");
    assert_below(self_energy_ratio, 2.0e-9, "self energy tangent");
    assert_below(
        result.combined_action.charge_conjugation_residual,
        2.0e-9,
        "charge-conjugation tangent",
    );
    assert_below(
        result.combined_action.mu_tau_residual,
        2.0e-9,
        "mu/tau tangent",
    );

    let expected_branch = &value["collision"]["base_branch"];
    assert_eq!(
        branch_signature(&result.base.combined_action),
        (
            usize::try_from(expected_branch["whole_reaction_domain_rejections"].as_u64().unwrap()).unwrap(),
            usize::try_from(expected_branch["matrix_roundoff_corrections"].as_u64().unwrap()).unwrap(),
            bits(&expected_branch["largest_matrix_roundoff_correction_bits"]).to_bits(),
        )
    );

    let full_direction = bit_array(&value["direction_full"]);
    let temperature_cm = bits(&value["temperature_cm_bits"]);
    let temperature_gamma = bits(&value["temperature_gamma_bits"]);
    let mut best_packed = f64::INFINITY;
    let mut best_collision = f64::INFINITY;
    for witness in value["centered_witnesses"].as_array().unwrap() {
        assert!(witness["state_valid"].as_bool().unwrap());
        assert!(witness["same_support_and_correction_branch"].as_bool().unwrap());
        let epsilon = bits(&witness["epsilon_bits"]);
        let plus_state: Vec<f64> = state
            .iter()
            .zip(&full_direction)
            .map(|(base, direction)| base + epsilon * direction)
            .collect();
        let minus_state: Vec<f64> = state
            .iter()
            .zip(&full_direction)
            .map(|(base, direction)| base - epsilon * direction)
            .collect();
        let plus_rhs = evaluate_f10_packed_rhs(&grid, ln_a, &plus_state, config).unwrap();
        let minus_rhs = evaluate_f10_packed_rhs(&grid, ln_a, &minus_state, config).unwrap();
        assert_eq!(branch_signature(&plus_rhs.combined_action), branch_signature(&result.base.combined_action));
        assert_eq!(branch_signature(&minus_rhs.combined_action), branch_signature(&result.base.combined_action));
        let packed_fd: Vec<f64> = plus_rhs
            .values
            .iter()
            .zip(&minus_rhs.values)
            .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
            .collect();
        best_packed = best_packed.min(global_relative(&packed_fd, &result.values));

        let plus_action = assemble_combined_action(
            &grid,
            &plus_state[..3 * order],
            temperature_cm,
            temperature_gamma,
            F10CombinedActionConfig::default(),
        )
        .unwrap();
        let minus_action = assemble_combined_action(
            &grid,
            &minus_state[..3 * order],
            temperature_cm,
            temperature_gamma,
            F10CombinedActionConfig::default(),
        )
        .unwrap();
        let modal_fd: Vec<f64> = plus_action
            .modal_total
            .iter()
            .zip(&minus_action.modal_total)
            .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
            .collect();
        best_collision = best_collision.min(global_relative(
            &modal_fd,
            &result.combined_action.modal_total,
        ));
    }
    assert_below(
        best_collision,
        thresholds["order8_centered_collision"].as_f64().unwrap(),
        "centered collision witness",
    );
    assert_below(
        best_packed,
        thresholds["order8_centered_packed_rhs"].as_f64().unwrap(),
        "centered packed-RHS witness",
    );

    println!(
        "D081R1F0_ORDER8_PASS self_modal={self_modal_residual:.17e} electron_modal={electron_modal_residual:.17e} total_modal={total_modal_residual:.17e} self_native={self_native_residual:.17e} electron_native={electron_native_residual:.17e} total_native={total_native_residual:.17e} packed={packed_residual:.17e} centered_collision={best_collision:.17e} centered_packed={best_packed:.17e}"
    );
}

#[test]
fn order8_jvp_is_linear_and_frozen_mutations_are_load_bearing() {
    let value = fixture();
    let grid = F10ActionGrid::affine_legendre(8, bits(&value["y_max_bits"])).unwrap();
    let state = bit_array(&value["packed_state"]);
    let direction = bit_array(&value["direction_cloglog"]);
    let ln_a = bits(&value["ln_a_bits"]);
    let config = F10PackedRhsConfig::default();
    let base = evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &direction, config).unwrap();

    let doubled: Vec<f64> = direction.iter().map(|value| 2.0 * value).collect();
    let doubled_result =
        evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &doubled, config).unwrap();
    let expected_doubled: Vec<f64> = base.values.iter().map(|value| 2.0 * value).collect();
    assert_below(
        global_relative(&doubled_result.values, &expected_doubled),
        5.0e-12,
        "J(2v)=2Jv",
    );

    let negated: Vec<f64> = direction.iter().map(|value| -value).collect();
    let negated_result =
        evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &negated, config).unwrap();
    let expected_negated: Vec<f64> = base.values.iter().map(|value| -value).collect();
    assert_below(
        global_relative(&negated_result.values, &expected_negated),
        5.0e-12,
        "J(-v)=-Jv",
    );

    let second = normalized(
        (0..direction.len())
            .map(|index| ((index % 7) as f64 - 3.0) / 7.0)
            .collect(),
    );
    let summed_direction: Vec<f64> = direction
        .iter()
        .zip(&second)
        .map(|(left, right)| left + right)
        .collect();
    let second_result =
        evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &second, config).unwrap();
    let summed_result = evaluate_f10_packed_rhs_c_jvp(
        &grid,
        ln_a,
        &state,
        &summed_direction,
        config,
    )
    .unwrap();
    let expected_sum: Vec<f64> = base
        .values
        .iter()
        .zip(&second_result.values)
        .map(|(left, right)| left + right)
        .collect();
    assert_below(
        global_relative(&summed_result.values, &expected_sum),
        5.0e-12,
        "J(v+w)=Jv+Jw",
    );

    let expected_total_modal = bit_array(&value["collision"]["total_modal"]);
    assert!(
        global_relative(
            &base.combined_action.self_action.modal,
            &expected_total_modal,
        ) > 1.0e-7,
        "omitting the electron tangent was not detected"
    );
    assert!(
        global_relative(
            &base.combined_action.electron_action.modal,
            &expected_total_modal,
        ) > 1.0e-7,
        "omitting the self tangent was not detected"
    );
    let expected_jvp = bit_array(&value["packed_rhs_jvp"]);
    let sign_mutant: Vec<f64> = base.values.iter().map(|value| -value).collect();
    let scale_mutant: Vec<f64> = base.values.iter().map(|value| 1.01 * value).collect();
    assert!(global_relative(&sign_mutant, &expected_jvp) > 5.0e-7);
    assert!(global_relative(&scale_mutant, &expected_jvp) > 5.0e-7);

    let mut flavour_mutant = base.values.clone();
    for node in 0..grid.order {
        flavour_mutant.swap(node, grid.order + node);
    }
    assert!(global_relative(&flavour_mutant, &expected_jvp) > 5.0e-7);
}

#[test]
fn order8_jvp_rejects_invalid_directions() {
    let value = fixture();
    let grid = F10ActionGrid::affine_legendre(8, bits(&value["y_max_bits"])).unwrap();
    let state = bit_array(&value["packed_state"]);
    let direction = bit_array(&value["direction_cloglog"]);
    let ln_a = bits(&value["ln_a_bits"]);
    let config = F10PackedRhsConfig::default();

    assert!(matches!(
        evaluate_f10_packed_rhs_c_jvp(
            &grid,
            ln_a,
            &state,
            &direction[..direction.len() - 1],
            config,
        ),
        Err(F10PackedRhsError::InvalidInput)
    ));
    let mut nonfinite = direction;
    nonfinite[0] = f64::NAN;
    assert!(matches!(
        evaluate_f10_packed_rhs_c_jvp(&grid, ln_a, &state, &nonfinite, config),
        Err(F10PackedRhsError::InvalidInput)
    ));
}
