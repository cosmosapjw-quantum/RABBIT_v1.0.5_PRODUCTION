//! RED-first admission tests for the D-081R1D4 combined collision action.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::{
    F10CombinedActionConfig, F10CombinedActionError, assemble_combined_action,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/fixtures/d081r1/full_collision_action_case.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("valid frozen D-081R1 full-action fixture")
}

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

fn named_case<'a>(value: &'a Value, name: &str) -> &'a Value {
    value["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .find(|case| case["name"] == name)
        .expect("named fixture case")
}

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn oracle_array_scale(value: &Value, name: &str) -> f64 {
    value["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .map(|case| maximum_absolute(&bit_array(&case["absolute_envelopes"][name])))
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn assert_hybrid_close(
    actual: &[f64],
    expected: &[f64],
    reference_scale: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let absolute_floor = 131_072.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
    let mut worst_ratio = 0.0_f64;
    let mut worst_index = 0_usize;
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        let allowed = absolute_floor + relative_tolerance * observed.abs().max(reference.abs());
        let difference = (observed - reference).abs();
        let ratio = difference / allowed.max(f64::MIN_POSITIVE);
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_index = index;
        }
    }
    assert!(
        worst_ratio <= 1.0,
        "hybrid parity failed at index {worst_index}: ratio={worst_ratio:.17e}, absolute_floor={absolute_floor:.17e}"
    );
}

fn assert_scalar_close(actual: f64, expected: f64, scale: f64, relative_tolerance: f64) {
    let absolute_floor = 262_144.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    let allowed = absolute_floor + relative_tolerance * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= allowed,
        "scalar mismatch: actual={actual:.17e}, expected={expected:.17e}, allowed={allowed:.17e}"
    );
}

fn scaled_difference(actual: &[f64], expected: &[f64], scale: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
        / scale.max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_full_action_contract_is_exact() {
        let value = fixture();
        assert_eq!(value["schema"], "rabbit.d081r1.full_collision_action.v1");
        assert_eq!(
            value["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(value["self_event_count"], 27);
        assert_eq!(value["electron_event_count"], 15);
        assert_eq!(value["species_order"].as_array().unwrap().len(), 6);
        assert_eq!(value["order"], 8);
        assert_eq!(bits(&value["y_max_bits"]).to_bits(), 8.0_f64.to_bits());
        assert_eq!(value["cases"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn combined_action_matches_every_frozen_python_case() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let modal_scale = oracle_array_scale(&value, "total_modal");
        let native_scale = oracle_array_scale(&value, "total_native");

        for case in value["cases"].as_array().expect("fixture cases") {
            let result = assemble_combined_action(
                &grid,
                &bit_array(&case["pair_cloglog"]),
                bits(&case["temperature_cm_bits"]),
                bits(&case["temperature_gamma_bits"]),
                F10CombinedActionConfig::default(),
            )
            .unwrap();

            assert_eq!(result.modal_total.len(), 6 * grid.order);
            assert_eq!(result.native_total.len(), 6 * grid.order);
            assert_hybrid_close(
                &result.modal_total,
                &bit_array(&case["arrays"]["total_modal"]),
                modal_scale,
                6.0e-9,
            );
            assert_hybrid_close(
                &result.native_total,
                &bit_array(&case["arrays"]["total_native"]),
                native_scale,
                6.0e-9,
            );
            assert_hybrid_close(
                &result.self_action.modal,
                &bit_array(&case["arrays"]["self_modal"]),
                oracle_array_scale(&value, "self_modal"),
                4.0e-9,
            );
            assert_hybrid_close(
                &result.electron_action.modal,
                &bit_array(&case["arrays"]["electron_modal"]),
                oracle_array_scale(&value, "electron_modal"),
                4.0e-9,
            );

            let expected_moments = &case["moments"]["total"];
            let moment_scale = result
                .moments
                .absolute_number_rate
                .max(result.moments.absolute_energy_rate)
                .max(1.0);
            assert_scalar_close(
                result.moments.signed_number_rate,
                bits(&expected_moments["signed_number_rate"]),
                moment_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.moments.absolute_number_rate,
                bits(&expected_moments["absolute_number_rate"]),
                moment_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.moments.signed_energy_rate,
                bits(&expected_moments["signed_energy_rate"]),
                moment_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.moments.absolute_energy_rate,
                bits(&expected_moments["absolute_energy_rate"]),
                moment_scale,
                6.0e-9,
            );

            assert_eq!(
                result.whole_reaction_domain_rejections,
                usize::try_from(case["whole_reaction_domain_rejections"].as_u64().unwrap())
                    .unwrap()
            );
            assert_eq!(
                result.matrix_roundoff_corrections,
                usize::try_from(case["matrix_roundoff_corrections"].as_u64().unwrap()).unwrap()
            );
            assert_scalar_close(
                result.largest_matrix_roundoff_correction,
                bits(&case["largest_matrix_roundoff_correction_bits"]),
                result.largest_matrix_roundoff_correction.abs().max(f64::MIN_POSITIVE),
                6.0e-9,
            );

            let diagnostics = &case["diagnostics"];
            let transfer_scale = result
                .neutrino_energy_transfer
                .abs()
                .max(result.electromagnetic_energy_transfer.abs())
                .max(1.0);
            assert_scalar_close(
                result.neutrino_energy_transfer,
                bits(&diagnostics["event_neutrino_energy_transfer"]),
                transfer_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.electromagnetic_energy_transfer,
                bits(&case["electron_bath_energy_transfer_bits"]),
                transfer_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.first_law_residual,
                bits(&diagnostics["first_law_residual"]),
                1.0,
                6.0e-9,
            );
            let h_scale = result
                .event_neutrino_h_rate
                .abs()
                .max(result.node_neutrino_h_rate.abs())
                .max(result.electromagnetic_h_rate.abs())
                .max(result.entropy_production.abs())
                .max(1.0);
            assert_scalar_close(
                result.event_neutrino_h_rate,
                bits(&diagnostics["event_neutrino_entropy_rate"]),
                h_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.node_neutrino_h_rate,
                bits(&diagnostics["node_neutrino_entropy_rate"]),
                h_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.electromagnetic_h_rate,
                bits(&diagnostics["electromagnetic_entropy_rate"]),
                h_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.entropy_production,
                bits(&diagnostics["entropy_production"]),
                h_scale,
                6.0e-9,
            );
            assert_scalar_close(
                result.entropy_duality_residual,
                bits(&diagnostics["entropy_duality_residual"]),
                1.0,
                6.0e-9,
            );
            assert_scalar_close(
                result.self_event_energy_residual,
                bits(&diagnostics["self_event_energy_residual"]),
                result.self_event_energy_absolute.max(1.0),
                6.0e-9,
            );
            assert_scalar_close(
                result.charge_conjugation_residual,
                bits(&diagnostics["charge_conjugation_residual"]),
                1.0,
                6.0e-9,
            );
            assert_scalar_close(
                result.mu_tau_residual,
                bits(&diagnostics["mu_tau_residual"]),
                1.0,
                6.0e-9,
            );
        }
    }

    #[test]
    fn component_addition_and_physical_ledgers_are_load_bearing() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let equilibrium_case = named_case(&value, "equilibrium");
        let thermal_case = named_case(&value, "thermal_split");
        let split_case = named_case(&value, "mu_tau_split");

        let build = |case: &Value| {
            assemble_combined_action(
                &grid,
                &bit_array(&case["pair_cloglog"]),
                bits(&case["temperature_cm_bits"]),
                bits(&case["temperature_gamma_bits"]),
                F10CombinedActionConfig::default(),
            )
            .unwrap()
        };
        let equilibrium = build(equilibrium_case);
        let thermal = build(thermal_case);
        let split = build(split_case);

        for index in 0..thermal.native_total.len() {
            assert_eq!(
                thermal.native_total[index].to_bits(),
                (thermal.self_action.native[index] + thermal.electron_action.native[index]).to_bits()
            );
            assert_eq!(
                thermal.modal_total[index].to_bits(),
                (thermal.self_action.modal[index] + thermal.electron_action.modal[index]).to_bits()
            );
        }
        assert_eq!(
            thermal.whole_reaction_domain_rejections,
            thermal.self_action.whole_reaction_domain_rejections
                + thermal.electron_action.whole_reaction_domain_rejections
        );
        assert_eq!(
            thermal.matrix_roundoff_corrections,
            thermal.self_action.matrix_roundoff_corrections
                + thermal.electron_action.matrix_roundoff_corrections
        );

        let thermal_scale = maximum_absolute(&thermal.native_total);
        assert!(maximum_absolute(&equilibrium.native_total) / thermal_scale < 1.0e-10);
        assert!(thermal.neutrino_energy_transfer > 0.0);
        assert!(thermal.electromagnetic_energy_transfer < 0.0);
        assert!(thermal.first_law_residual <= 5.0e-13);
        assert!(thermal.entropy_production >= -262_144.0 * f64::EPSILON);
        assert!(thermal.charge_conjugation_residual <= 5.0e-12);
        assert!(thermal.mu_tau_residual <= 5.0e-12);
        assert!(split.mu_tau_residual > 1.0e-8);

        let expected_total = bit_array(&thermal_case["arrays"]["total_native"]);
        assert!(
            scaled_difference(&thermal.self_action.native, &expected_total, thermal_scale) > 1.0e-3
        );
        assert!(
            scaled_difference(&thermal.electron_action.native, &expected_total, thermal_scale)
                > 1.0e-3
        );
        let wrong_sign: Vec<_> = thermal
            .self_action
            .native
            .iter()
            .zip(&thermal.electron_action.native)
            .map(|(self_value, electron_value)| self_value - electron_value)
            .collect();
        assert!(scaled_difference(&wrong_sign, &expected_total, thermal_scale) > 1.0e-3);
    }

    #[test]
    fn component_failures_propagate_without_a_partial_result() {
        let value = fixture();
        let case = named_case(&value, "thermal_split");
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let temperature_cm = bits(&case["temperature_cm_bits"]);
        let temperature_gamma = bits(&case["temperature_gamma_bits"]);

        assert_eq!(
            assemble_combined_action(
                &grid,
                &[0.0; 23],
                temperature_cm,
                temperature_gamma,
                F10CombinedActionConfig::default(),
            ),
            Err(F10CombinedActionError::SelfAction)
        );

        let mut nonfinite = bit_array(&case["pair_cloglog"]);
        nonfinite[0] = f64::NAN;
        assert_eq!(
            assemble_combined_action(
                &grid,
                &nonfinite,
                temperature_cm,
                temperature_gamma,
                F10CombinedActionConfig::default(),
            ),
            Err(F10CombinedActionError::SelfAction)
        );

        assert_eq!(
            assemble_combined_action(
                &grid,
                &bit_array(&case["pair_cloglog"]),
                temperature_cm,
                -temperature_gamma,
                F10CombinedActionConfig::default(),
            ),
            Err(F10CombinedActionError::InvalidInput)
        );

        let invalid_config = F10CombinedActionConfig {
            matrix_roundoff_ulps: 0.0,
            ..F10CombinedActionConfig::default()
        };
        assert_eq!(
            assemble_combined_action(
                &grid,
                &bit_array(&case["pair_cloglog"]),
                temperature_cm,
                temperature_gamma,
                invalid_config,
            ),
            Err(F10CombinedActionError::InvalidConfiguration)
        );
    }
}
