//! Prospectively frozen unseen state-2000 admission for D-081R1E.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_packed_rhs::{F10PackedRhsConfig, evaluate_f10_packed_rhs};
use serde_json::{Value, json};
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
    let mut max_absolute = 0.0_f64;
    let mut max_absolute_index = 0_usize;
    let mut max_local_relative = 0.0_f64;
    let mut max_local_relative_index = 0_usize;
    let mut max_ulp = 0_u64;
    let mut max_ulp_index = 0_usize;

    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        assert!(observed.is_finite() && reference.is_finite());
        let absolute = (observed - reference).abs();
        let local_scale = observed.abs().max(reference.abs()).max(f64::MIN_POSITIVE);
        let local_relative = absolute / local_scale;
        let ulp = ordered_bits(observed).abs_diff(ordered_bits(reference));
        if absolute > max_absolute {
            max_absolute = absolute;
            max_absolute_index = index;
        }
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

fn assert_exact_slice(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            observed.to_bits(),
            reference.to_bits(),
            "{label} bit mismatch at index {index}"
        );
    }
}

fn assert_exact_sum(left: &[f64], right: &[f64], total: &[f64], label: &str) {
    assert_eq!(left.len(), right.len(), "{label} component length mismatch");
    assert_eq!(left.len(), total.len(), "{label} total length mismatch");
    for (index, ((&left_value, &right_value), &total_value)) in
        left.iter().zip(right).zip(total).enumerate()
    {
        assert_eq!(
            (left_value + right_value).to_bits(),
            total_value.to_bits(),
            "{label} component addition mismatch at index {index}"
        );
    }
}

fn pair_rate(native: &[f64], order: usize) -> Vec<f64> {
    assert_eq!(native.len(), 6 * order);
    let mut result = vec![0.0_f64; 3 * order];
    for flavour in 0..3 {
        for node in 0..order {
            result[flavour * order + node] = 0.5
                * (native[(2 * flavour) * order + node] + native[(2 * flavour + 1) * order + node]);
        }
    }
    result
}

fn chart_state(coordinates: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut occupations = Vec::with_capacity(coordinates.len());
    let mut chain = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        let exponential = coordinate.exp();
        let occupation = -(-exponential).exp_m1();
        let derivative = (coordinate - exponential).exp();
        assert!(
            exponential.is_finite()
                && exponential > 0.0
                && occupation.is_finite()
                && (0.0..1.0).contains(&occupation)
                && derivative.is_finite()
                && derivative > 0.0,
            "holdout state left the strict cloglog chart"
        );
        occupations.push(occupation);
        chain.push(derivative);
    }
    (occupations, chain)
}

fn retained_step_impact(
    actual: &[f64],
    expected: &[f64],
    state: &[f64],
    retained_step: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), state.len());
    let mut maximum = 0.0_f64;
    let mut maximum_index = 0_usize;
    for (index, ((&observed, &reference), &coordinate)) in
        actual.iter().zip(expected).zip(state).enumerate()
    {
        let local_scale = absolute_tolerance + relative_tolerance * coordinate.abs();
        let impact =
            retained_step * (observed - reference).abs() / local_scale.max(f64::MIN_POSITIVE);
        if impact > maximum {
            maximum = impact;
            maximum_index = index;
        }
    }
    (maximum, maximum_index)
}

fn spectral_decomposition_ratio(
    rhs: (&[f64], &[f64]),
    pair: (&[f64], &[f64]),
    chain: (&[f64], &[f64]),
    hubble: (f64, f64),
) -> f64 {
    let (actual_rhs, expected_rhs) = rhs;
    let (actual_pair, expected_pair) = pair;
    let (actual_chain, expected_chain) = chain;
    let (actual_hubble, expected_hubble) = hubble;
    assert_eq!(actual_rhs.len(), expected_rhs.len());
    assert_eq!(actual_rhs.len(), actual_pair.len());
    assert_eq!(actual_rhs.len(), expected_pair.len());
    assert_eq!(actual_rhs.len(), actual_chain.len());
    assert_eq!(actual_rhs.len(), expected_chain.len());
    assert!(actual_hubble.is_finite() && actual_hubble > 0.0);
    assert!(expected_hubble.is_finite() && expected_hubble > 0.0);

    let discrepancy_scale = difference_stats(actual_rhs, expected_rhs)
        .max_absolute
        .max(f64::MIN_POSITIVE);
    let mut maximum = 0.0_f64;
    for index in 0..actual_rhs.len() {
        let actual_denominator = actual_hubble * actual_chain[index];
        let expected_denominator = expected_hubble * expected_chain[index];
        let direct = actual_rhs[index] - expected_rhs[index];
        let collision = (actual_pair[index] - expected_pair[index]) / actual_denominator;
        let denominator = expected_pair[index] * (expected_denominator - actual_denominator)
            / (actual_denominator * expected_denominator);
        maximum = maximum.max((direct - collision - denominator).abs() / discrepancy_scale);
    }
    maximum
}

fn assert_scalar_hybrid(actual: f64, expected: f64, relative_tolerance: f64, label: &str) {
    assert!(actual.is_finite() && expected.is_finite());
    let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
    let floor = 1_048_576.0 * f64::EPSILON * scale;
    let allowed = floor + relative_tolerance * scale;
    assert!(
        (actual - expected).abs() <= allowed,
        "{label} mismatch: actual={actual:.17e}, expected={expected:.17e}, allowed={allowed:.17e}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "prospectively frozen unseen state-2000 holdout"]
    fn state2000_prospective_holdout_admission() {
        let fixture_path = env::var("D081R1E_HOLDOUT_FIXTURE")
            .expect("D081R1E_HOLDOUT_FIXTURE must identify the host-local fixture");
        let receipt_path = env::var("D081R1E_HOLDOUT_RECEIPT")
            .expect("D081R1E_HOLDOUT_RECEIPT must identify the durable receipt path");
        let value: Value = serde_json::from_str(
            &fs::read_to_string(&fixture_path).expect("read state-2000 holdout fixture"),
        )
        .expect("valid state-2000 holdout fixture");

        assert_eq!(
            value["schema"],
            "rabbit.d081r1e.state2000_holdout_metrology.v1"
        );
        assert_eq!(
            value["classification"],
            "UNSEEN_HOLDOUT_AFTER_PROSPECTIVE_CONTRACT"
        );
        assert_eq!(
            value["contract_commit"],
            "d98d725c6e252180a4108fb572e97b6a90c00887"
        );
        assert_eq!(
            value["state_git_blob"],
            "cfb17344ae166c01c2e5bcb14acae0d968e49477"
        );
        assert_eq!(
            value["state_sha256"],
            "780ad7c1388caec23f02012781717d43ffb85d96d4d501c40c504939e7c9a44d"
        );
        assert_eq!(
            value["python_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(
            value["trajectory_core_git_blob"],
            "465a73f0ce40f7149bebdc2d67103f388e2344d9"
        );
        assert_eq!(
            value["cargo_lock_git_blob"],
            "a1b5035da5c20712d1a2a4ab077da255ff94a014"
        );
        assert_eq!(value["numpy_version"], "2.4.4");
        assert_eq!(value["scipy_version"], "1.17.1");
        assert_eq!(value["order"], 60);

        let self_modal_cap = value["prospective_caps"]["self_modal_global_relative"]
            .as_f64()
            .expect("self modal cap");
        let electron_modal_cap = value["prospective_caps"]["electron_modal_global_relative"]
            .as_f64()
            .expect("electron modal cap");
        let total_modal_cap = value["prospective_caps"]["total_modal_global_relative"]
            .as_f64()
            .expect("total modal cap");
        let step_impact_cap = value["prospective_caps"]["maximum_step_impact"]
            .as_f64()
            .expect("step impact cap");
        let first_law_cap = value["prospective_caps"]["first_law_residual"]
            .as_f64()
            .expect("first-law cap");
        assert_eq!(self_modal_cap.to_bits(), 1.0e-7_f64.to_bits());
        assert_eq!(electron_modal_cap.to_bits(), 1.0e-7_f64.to_bits());
        assert_eq!(total_modal_cap.to_bits(), 1.0e-7_f64.to_bits());
        assert_eq!(step_impact_cap.to_bits(), 1.0e-3_f64.to_bits());
        assert_eq!(first_law_cap.to_bits(), 5.0e-13_f64.to_bits());

        let grid = F10ActionGrid::affine_legendre(60, 30.0).unwrap();
        let expected_nodes = bit_array(&value["arrays"]["grid_nodes"]);
        let expected_weights = bit_array(&value["arrays"]["grid_weights"]);
        assert_exact_slice(&grid.nodes, &expected_nodes, "GL60 nodes");
        assert_exact_slice(&grid.weights, &expected_weights, "GL60 weights");

        let state = bit_array(&value["arrays"]["packed_state"]);
        assert_eq!(state.len(), 182);
        let ln_a = bits(&value["ln_a_bits"]);
        let result =
            evaluate_f10_packed_rhs(&grid, ln_a, &state, F10PackedRhsConfig::default()).unwrap();
        assert_eq!(result.values.len(), 182);
        assert!(
            result.diagnostics.temperature_cm_mev.is_finite()
                && result.diagnostics.temperature_cm_mev > 0.0
                && result.diagnostics.temperature_gamma_mev.is_finite()
                && result.diagnostics.temperature_gamma_mev > 0.0
                && result.diagnostics.hubble_mev.is_finite()
                && result.diagnostics.hubble_mev > 0.0
        );

        let expected_occupation = bit_array(&value["arrays"]["occupation"]);
        let expected_chain = bit_array(&value["arrays"]["cloglog_chain"]);
        let (actual_occupation, actual_chain) = chart_state(&state[..180]);
        let occupation_stats = difference_stats(&actual_occupation, &expected_occupation);
        let chain_stats = difference_stats(&actual_chain, &expected_chain);

        let expected_self_modal = bit_array(&value["arrays"]["self_modal"]);
        let expected_electron_modal = bit_array(&value["arrays"]["electron_modal"]);
        let expected_total_modal = bit_array(&value["arrays"]["total_modal"]);
        let expected_self_native = bit_array(&value["arrays"]["self_native"]);
        let expected_electron_native = bit_array(&value["arrays"]["electron_native"]);
        let expected_total_native = bit_array(&value["arrays"]["total_native"]);
        let expected_pair = bit_array(&value["arrays"]["pair_rate"]);
        let expected_spectral = bit_array(&value["arrays"]["spectral_rhs"]);
        let expected_packed = bit_array(&value["arrays"]["packed_rhs"]);

        assert_exact_sum(
            &result.combined_action.self_action.modal,
            &result.combined_action.electron_action.modal,
            &result.combined_action.modal_total,
            "Rust modal",
        );
        assert_exact_sum(
            &result.combined_action.self_action.native,
            &result.combined_action.electron_action.native,
            &result.combined_action.native_total,
            "Rust native",
        );
        assert_exact_sum(
            &expected_self_modal,
            &expected_electron_modal,
            &expected_total_modal,
            "Python modal",
        );
        assert_exact_sum(
            &expected_self_native,
            &expected_electron_native,
            &expected_total_native,
            "Python native",
        );

        let self_modal_stats = difference_stats(
            &result.combined_action.self_action.modal,
            &expected_self_modal,
        );
        let electron_modal_stats = difference_stats(
            &result.combined_action.electron_action.modal,
            &expected_electron_modal,
        );
        let total_modal_stats =
            difference_stats(&result.combined_action.modal_total, &expected_total_modal);
        let self_native_stats = difference_stats(
            &result.combined_action.self_action.native,
            &expected_self_native,
        );
        let electron_native_stats = difference_stats(
            &result.combined_action.electron_action.native,
            &expected_electron_native,
        );
        let total_native_stats =
            difference_stats(&result.combined_action.native_total, &expected_total_native);
        let actual_pair = pair_rate(&result.combined_action.native_total, 60);
        let pair_stats = difference_stats(&actual_pair, &expected_pair);
        let spectral_stats = difference_stats(&result.values[..180], &expected_spectral);
        let packed_stats = difference_stats(&result.values, &expected_packed);

        assert!(
            self_modal_stats.global_relative <= self_modal_cap,
            "self modal holdout gate failed: {:.17e}",
            self_modal_stats.global_relative
        );
        assert!(
            electron_modal_stats.global_relative <= electron_modal_cap,
            "electron modal holdout gate failed: {:.17e}",
            electron_modal_stats.global_relative
        );
        assert!(
            total_modal_stats.global_relative <= total_modal_cap,
            "total modal holdout gate failed: {:.17e}",
            total_modal_stats.global_relative
        );

        let retained_step = bits(&value["retained_h_bits"]).abs();
        let retained_atol = 1.0e-9;
        let retained_rtol = 1.0e-6;
        let (maximum_step_impact, maximum_step_impact_index) = retained_step_impact(
            &result.values[..180],
            &expected_spectral,
            &state[..180],
            retained_step,
            retained_atol,
            retained_rtol,
        );
        assert!(
            maximum_step_impact <= step_impact_cap,
            "state-2000 step-impact gate failed: {maximum_step_impact:.17e} at index {maximum_step_impact_index}"
        );

        let expected_hubble = bits(&value["hubble_mev_bits"]);
        let decomposition_ratio = spectral_decomposition_ratio(
            (&result.values[..180], expected_spectral.as_slice()),
            (actual_pair.as_slice(), expected_pair.as_slice()),
            (actual_chain.as_slice(), expected_chain.as_slice()),
            (result.diagnostics.hubble_mev, expected_hubble),
        );
        assert!(
            decomposition_ratio <= 1.0e-9,
            "state-2000 spectral decomposition failed: {decomposition_ratio:.17e}"
        );

        let expected_first_law = bits(&value["first_law_residual_bits"]);
        assert!(
            result.diagnostics.first_law_residual.abs() <= first_law_cap,
            "Rust first-law holdout gate failed: {:.17e}",
            result.diagnostics.first_law_residual.abs()
        );
        assert!(
            expected_first_law.abs() <= first_law_cap,
            "Python first-law holdout gate failed: {:.17e}",
            expected_first_law.abs()
        );

        let expected_rejections = usize::try_from(
            value["support_and_roundoff"]["whole_reaction_domain_rejections"]
                .as_u64()
                .expect("whole-reaction rejection count"),
        )
        .unwrap();
        let expected_corrections = usize::try_from(
            value["support_and_roundoff"]["matrix_roundoff_corrections"]
                .as_u64()
                .expect("matrix correction count"),
        )
        .unwrap();
        assert_eq!(
            result.diagnostics.whole_reaction_domain_rejections,
            expected_rejections
        );
        assert_eq!(
            result.diagnostics.matrix_roundoff_corrections,
            expected_corrections
        );
        assert_eq!(
            result
                .diagnostics
                .largest_matrix_roundoff_correction
                .to_bits(),
            bits(&value["support_and_roundoff"]["largest_matrix_roundoff_correction_bits"])
                .to_bits()
        );

        assert_scalar_hybrid(
            result.values[180],
            bits(&value["temperature_rhs_bits"]),
            5.0e-8,
            "temperature RHS",
        );
        assert_scalar_hybrid(
            result.values[181],
            bits(&value["elapsed_rhs_bits"]),
            5.0e-10,
            "elapsed RHS",
        );

        let mut changed_elapsed = state.clone();
        changed_elapsed[181] = 9.876_543_21e18;
        let passive =
            evaluate_f10_packed_rhs(&grid, ln_a, &changed_elapsed, F10PackedRhsConfig::default())
                .unwrap();
        assert_exact_slice(&passive.values, &result.values, "passive elapsed-time RHS");

        let mut spectral_mutant = result.values[..180].to_vec();
        spectral_mutant[maximum_step_impact_index] += 2.0
            * (retained_atol + retained_rtol * state[maximum_step_impact_index].abs())
            / retained_step;
        assert!(
            retained_step_impact(
                &spectral_mutant,
                &expected_spectral,
                &state[..180],
                retained_step,
                retained_atol,
                retained_rtol,
            )
            .0 > 1.0,
            "state-2000 step-impact gate did not kill a two-budget mutation"
        );

        let source_head = env::var("GITHUB_SHA").unwrap_or_else(|_| "LOCAL".to_string());
        let receipt = json!({
            "schema": "rabbit.d081r1e.state2000_holdout_receipt.v1",
            "classification": "PASS_WITH_RETAINED_ORDER60_CONDITIONED_PACKED_RHS_SCOPE",
            "source_head": source_head,
            "contract_commit": value["contract_commit"].clone(),
            "holdout_authority_commit": value["holdout_authority_commit"].clone(),
            "historical_source_commit": value["historical_source_commit"].clone(),
            "state_git_blob": value["state_git_blob"].clone(),
            "state_sha256": value["state_sha256"].clone(),
            "python_comparator_git_blob": value["python_comparator_git_blob"].clone(),
            "trajectory_core_git_blob": value["trajectory_core_git_blob"].clone(),
            "cargo_lock_git_blob": value["cargo_lock_git_blob"].clone(),
            "generator_git_blob": value["generator_git_blob"].clone(),
            "runtime_identity": {
                "numpy": value["numpy_version"].clone(),
                "scipy": value["scipy_version"].clone(),
                "rust": "1.94.1",
            },
            "prospective_caps": value["prospective_caps"].clone(),
            "gated_metrics": {
                "self_modal_global_relative": self_modal_stats.global_relative,
                "electron_modal_global_relative": electron_modal_stats.global_relative,
                "total_modal_global_relative": total_modal_stats.global_relative,
                "maximum_step_impact": maximum_step_impact,
                "maximum_step_impact_index": maximum_step_impact_index,
                "rust_first_law_residual": result.diagnostics.first_law_residual.abs(),
                "python_first_law_residual": expected_first_law.abs(),
            },
            "structural_gates": {
                "gl60_y30_binary64_identity": "PASS",
                "strict_cloglog_chart": "PASS",
                "component_addition": "PASS",
                "pair_average_one_half": "PASS",
                "support_rejection_count_identity": "PASS",
                "matrix_correction_identity": "PASS",
                "passive_elapsed_time": "PASS",
                "two_budget_mutation_killed": "PASS",
                "spectral_error_decomposition": "PASS",
                "spectral_error_decomposition_ratio": decomposition_ratio,
            },
            "raw_diagnostics": {
                "occupation": stats_json(occupation_stats),
                "cloglog_chain": stats_json(chain_stats),
                "self_modal": stats_json(self_modal_stats),
                "electron_modal": stats_json(electron_modal_stats),
                "total_modal": stats_json(total_modal_stats),
                "self_native": stats_json(self_native_stats),
                "electron_native": stats_json(electron_native_stats),
                "total_native": stats_json(total_native_stats),
                "pair_rate": stats_json(pair_stats),
                "spectral_rhs": stats_json(spectral_stats),
                "packed_rhs": stats_json(packed_stats),
            },
            "claim_ceiling": "unseen state-2000 static retained packed-RHS metrology only; no JVP, Jacobian, diffsol, trajectory, endpoint, N_eff, publication authority, or G-F10-INDEPENDENT-FLRW movement",
        });
        fs::write(
            &receipt_path,
            serde_json::to_string_pretty(&receipt).expect("serialize holdout receipt") + "\n",
        )
        .expect("write holdout receipt");

        println!(
            "D081R1E_HOLDOUT_PASS self_modal={:.17e} electron_modal={:.17e} total_modal={:.17e} step_impact={:.17e} step_index={} first_law_rust={:.17e} first_law_python={:.17e} decomposition={:.17e}",
            self_modal_stats.global_relative,
            electron_modal_stats.global_relative,
            total_modal_stats.global_relative,
            maximum_step_impact,
            maximum_step_impact_index,
            result.diagnostics.first_law_residual.abs(),
            expected_first_law.abs(),
            decomposition_ratio,
        );
    }
}
