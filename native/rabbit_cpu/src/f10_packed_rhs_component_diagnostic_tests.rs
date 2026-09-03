//! Diagnostic-only decomposition of retained order-60 packed-RHS parity.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_packed_rhs::{F10PackedRhsConfig, evaluate_f10_packed_rhs};
use serde_json::Value;
use std::fs;

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

#[derive(Clone, Copy, Debug)]
struct DifferenceStats {
    max_absolute: f64,
    max_absolute_index: usize,
    max_local_relative: f64,
    max_local_relative_index: usize,
    global_relative: f64,
}

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn difference_stats(actual: &[f64], expected: &[f64]) -> DifferenceStats {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    let mut max_absolute = 0.0_f64;
    let mut max_absolute_index = 0_usize;
    let mut max_local_relative = 0.0_f64;
    let mut max_local_relative_index = 0_usize;
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        assert!(observed.is_finite() && reference.is_finite());
        let absolute = (observed - reference).abs();
        let scale = observed
            .abs()
            .max(reference.abs())
            .max(f64::MIN_POSITIVE);
        let local_relative = absolute / scale;
        if absolute > max_absolute {
            max_absolute = absolute;
            max_absolute_index = index;
        }
        if local_relative > max_local_relative {
            max_local_relative = local_relative;
            max_local_relative_index = index;
        }
    }
    DifferenceStats {
        max_absolute,
        max_absolute_index,
        max_local_relative,
        max_local_relative_index,
        global_relative: max_absolute
            / maximum_absolute(actual)
                .max(maximum_absolute(expected))
                .max(f64::MIN_POSITIVE),
    }
}

fn print_stats(name: &str, actual: &[f64], expected: &[f64]) -> DifferenceStats {
    let stats = difference_stats(actual, expected);
    println!(
        "COMPONENT[{name}] max_abs={:.17e} max_abs_index={} global_relative={:.17e} max_local_relative={:.17e} max_local_index={} actual_at_max_abs={:.17e} expected_at_max_abs={:.17e}",
        stats.max_absolute,
        stats.max_absolute_index,
        stats.global_relative,
        stats.max_local_relative,
        stats.max_local_relative_index,
        actual[stats.max_absolute_index],
        expected[stats.max_absolute_index],
    );
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "diagnostic-only retained component decomposition"]
    fn retained_component_modal_diagnostic() {
        let fixture_path = std::env::var("D081R1E_COMPONENT_FIXTURE")
            .expect("D081R1E_COMPONENT_FIXTURE must identify the host-local fixture");
        let value: Value = serde_json::from_str(
            &fs::read_to_string(fixture_path).expect("read component diagnostic fixture"),
        )
        .expect("valid component diagnostic fixture");
        assert_eq!(
            value["schema"],
            "rabbit.d081r1e.retained_component_diagnostic.v1"
        );
        assert_eq!(value["order"], 60);
        assert_eq!(
            value["retained_sha256"],
            "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
        );

        let state = bit_array(&value["arrays"]["packed_state"]);
        let ln_a = bits(&value["ln_a_bits"]);
        let grid = F10ActionGrid::affine_legendre(60, 30.0).unwrap();
        let result = evaluate_f10_packed_rhs(
            &grid,
            ln_a,
            &state,
            F10PackedRhsConfig::default(),
        )
        .unwrap();

        let expected_self_modal = bit_array(&value["arrays"]["self_modal"]);
        let expected_electron_modal = bit_array(&value["arrays"]["electron_modal"]);
        let expected_total_modal = bit_array(&value["arrays"]["total_modal"]);
        let expected_self_native = bit_array(&value["arrays"]["self_native"]);
        let expected_electron_native = bit_array(&value["arrays"]["electron_native"]);
        let expected_total_native = bit_array(&value["arrays"]["total_native"]);
        let expected_pair_rate = bit_array(&value["arrays"]["pair_rate"]);
        let expected_spectral = bit_array(&value["arrays"]["spectral_rhs"]);

        let self_modal = print_stats(
            "self_modal",
            &result.combined_action.self_action.modal,
            &expected_self_modal,
        );
        let electron_modal = print_stats(
            "electron_modal",
            &result.combined_action.electron_action.modal,
            &expected_electron_modal,
        );
        let total_modal = print_stats(
            "total_modal",
            &result.combined_action.modal_total,
            &expected_total_modal,
        );
        print_stats(
            "self_native",
            &result.combined_action.self_action.native,
            &expected_self_native,
        );
        print_stats(
            "electron_native",
            &result.combined_action.electron_action.native,
            &expected_electron_native,
        );
        print_stats(
            "total_native",
            &result.combined_action.native_total,
            &expected_total_native,
        );

        let order = 60_usize;
        let mut actual_pair_rate = vec![0.0_f64; 3 * order];
        for flavour in 0..3 {
            for node in 0..order {
                actual_pair_rate[flavour * order + node] = 0.5
                    * (result.combined_action.native_total[(2 * flavour) * order + node]
                        + result.combined_action.native_total
                            [(2 * flavour + 1) * order + node]);
            }
        }
        print_stats("pair_rate", &actual_pair_rate, &expected_pair_rate);
        let spectral = print_stats("spectral_rhs", &result.values[..180], &expected_spectral);

        let index = total_modal.max_absolute_index;
        let rust_self = result.combined_action.self_action.modal[index];
        let rust_electron = result.combined_action.electron_action.modal[index];
        let python_self = expected_self_modal[index];
        let python_electron = expected_electron_modal[index];
        let delta_self = rust_self - python_self;
        let delta_electron = rust_electron - python_electron;
        let delta_total = result.combined_action.modal_total[index] - expected_total_modal[index];
        println!(
            "TOTAL_MODAL_WORST index={index} species={} mode={} rust_self={:.17e} python_self={:.17e} delta_self={:.17e} rust_electron={:.17e} python_electron={:.17e} delta_electron={:.17e} delta_total={:.17e} decomposition_residual={:.17e}",
            index / order,
            index % order,
            rust_self,
            python_self,
            delta_self,
            rust_electron,
            python_electron,
            delta_electron,
            delta_total,
            delta_total - delta_self - delta_electron,
        );
        println!(
            "DOMINANCE self_modal_global={:.17e} electron_modal_global={:.17e} total_modal_global={:.17e}",
            self_modal.global_relative,
            electron_modal.global_relative,
            total_modal.global_relative,
        );

        let retained_h = bits(&value["retained_h_bits"]).abs();
        let mut maximum_step_impact = 0.0_f64;
        let mut maximum_step_impact_index = 0_usize;
        for index in 0..180 {
            let denominator = 1.0e-9 + 1.0e-6 * state[index].abs();
            let impact = retained_h * (result.values[index] - expected_spectral[index]).abs()
                / denominator;
            if impact > maximum_step_impact {
                maximum_step_impact = impact;
                maximum_step_impact_index = index;
            }
        }
        println!(
            "RETAINED_STEP_IMPACT max={:.17e} index={} h={:.17e} spectral_global_relative={:.17e}",
            maximum_step_impact,
            maximum_step_impact_index,
            retained_h,
            spectral.global_relative,
        );
    }
}
