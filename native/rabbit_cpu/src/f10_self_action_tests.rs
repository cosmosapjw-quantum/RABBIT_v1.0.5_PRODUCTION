//! RED-first oracle admission tests for D-081R1D2 six-species self action.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_self_action::{F10SelfActionConfig, assemble_self_action};
use serde_json::Value;

const FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/full_collision_action_case.json");

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

fn oracle_row_scale(value: &Value) -> f64 {
    value["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .flat_map(|case| {
            case["self_rows"]
                .as_object()
                .expect("self row map")
                .values()
                .map(bit_array)
                .collect::<Vec<_>>()
        })
        .map(|row| maximum_absolute(&row))
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn assert_hybrid_close(
    actual: &[f64],
    expected: &[f64],
    reference_scale: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let absolute_floor = 32_768.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
    let mut worst_ratio = 0.0_f64;
    let mut worst_index = 0_usize;
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        let allowed = absolute_floor
            + relative_tolerance * observed.abs().max(reference.abs());
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

fn assert_moment_close(actual: f64, expected: f64, scale: f64) {
    let absolute_floor = 65_536.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    let allowed = absolute_floor + 3.0e-9 * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= allowed,
        "moment mismatch: actual={actual:.17e}, expected={expected:.17e}, allowed={allowed:.17e}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_fixture_contract_is_exact() {
        let value = fixture();
        assert_eq!(value["schema"], "rabbit.d081r1.full_collision_action.v1");
        assert_eq!(
            value["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(value["self_event_count"], 27);
        assert_eq!(value["species_order"].as_array().unwrap().len(), 6);
        assert_eq!(value["order"], 8);
        assert_eq!(bits(&value["y_max_bits"]).to_bits(), 8.0_f64.to_bits());
    }

    #[test]
    fn six_species_self_action_matches_every_frozen_python_case() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let modal_scale = oracle_array_scale(&value, "self_modal");
        let native_scale = oracle_array_scale(&value, "self_native");
        let row_scale = oracle_row_scale(&value);

        for case in value["cases"].as_array().expect("fixture cases") {
            let pair_cloglog = bit_array(&case["pair_cloglog"]);
            let temperature = bits(&case["temperature_cm_bits"]);
            let result = assemble_self_action(
                &grid,
                &pair_cloglog,
                temperature,
                F10SelfActionConfig::default(),
            )
            .unwrap();

            assert_eq!(result.modal.len(), 6 * grid.order);
            assert_eq!(result.native.len(), 6 * grid.order);
            assert_eq!(result.row_modal.len(), 9 * 6 * grid.order);
            assert_eq!(result.row_native.len(), 9 * 6 * grid.order);
            assert_hybrid_close(
                &result.modal,
                &bit_array(&case["arrays"]["self_modal"]),
                modal_scale,
                3.0e-9,
            );
            assert_hybrid_close(
                &result.native,
                &bit_array(&case["arrays"]["self_native"]),
                native_scale,
                3.0e-9,
            );

            let rows = case["self_rows"].as_object().expect("self row map");
            let row_size = 6 * grid.order;
            for row in 1..=9 {
                let key = row.to_string();
                let expected = bit_array(rows.get(&key).expect("named self row"));
                assert_hybrid_close(
                    &result.row_native[(row - 1) * row_size..row * row_size],
                    &expected,
                    row_scale,
                    4.0e-9,
                );
            }

            let expected_moments = &case["moments"]["self"];
            let moment_scale = result
                .moments
                .absolute_number_rate
                .abs()
                .max(result.moments.absolute_energy_rate.abs())
                .max(1.0);
            assert_moment_close(
                result.moments.signed_number_rate,
                bits(&expected_moments["signed_number_rate"]),
                moment_scale,
            );
            assert_moment_close(
                result.moments.absolute_number_rate,
                bits(&expected_moments["absolute_number_rate"]),
                moment_scale,
            );
            assert_moment_close(
                result.moments.signed_energy_rate,
                bits(&expected_moments["signed_energy_rate"]),
                moment_scale,
            );
            assert_moment_close(
                result.moments.absolute_energy_rate,
                bits(&expected_moments["absolute_energy_rate"]),
                moment_scale,
            );
            assert_moment_close(
                result.event_energy_residual,
                bits(&case["diagnostics"]["self_event_energy_residual"]),
                result.event_energy_absolute.max(1.0),
            );
        }
    }

    #[test]
    fn row_decomposition_conservation_entropy_and_symmetry_are_load_bearing() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let equilibrium_case = named_case(&value, "equilibrium");
        let thermal_case = named_case(&value, "thermal_split");
        let split_case = named_case(&value, "mu_tau_split");
        let configuration = F10SelfActionConfig::default();

        let equilibrium = assemble_self_action(
            &grid,
            &bit_array(&equilibrium_case["pair_cloglog"]),
            bits(&equilibrium_case["temperature_cm_bits"]),
            configuration,
        )
        .unwrap();
        let thermal = assemble_self_action(
            &grid,
            &bit_array(&thermal_case["pair_cloglog"]),
            bits(&thermal_case["temperature_cm_bits"]),
            configuration,
        )
        .unwrap();
        assert_eq!(
            equilibrium.modal.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            thermal.modal.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            equilibrium.native.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            thermal.native.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );

        let split = assemble_self_action(
            &grid,
            &bit_array(&split_case["pair_cloglog"]),
            bits(&split_case["temperature_cm_bits"]),
            configuration,
        )
        .unwrap();
        let row_size = 6 * grid.order;
        let reconstructed: Vec<f64> = (0..row_size)
            .map(|index| {
                (0..9)
                    .map(|row| split.row_native[row * row_size + index])
                    .sum()
            })
            .collect();
        assert_hybrid_close(
            &reconstructed,
            &split.native,
            oracle_row_scale(&value),
            4.0e-9,
        );

        for pair in [(0_usize, 1_usize), (2, 3), (4, 5)] {
            assert_hybrid_close(
                &split.native[pair.0 * grid.order..(pair.0 + 1) * grid.order],
                &split.native[pair.1 * grid.order..(pair.1 + 1) * grid.order],
                maximum_absolute(&split.native),
                2.0e-10,
            );
        }

        assert!(maximum_absolute(&split.native) > 0.0);
        assert!(
            split.moments.signed_number_rate.abs()
                <= 5.0e-9 * split.moments.absolute_number_rate.max(f64::MIN_POSITIVE)
        );
        assert!(
            split.moments.signed_energy_rate.abs()
                <= 5.0e-9 * split.moments.absolute_energy_rate.max(f64::MIN_POSITIVE)
        );
        assert!(
            split.event_energy_residual.abs()
                <= 5.0e-11 * split.event_energy_absolute.max(f64::MIN_POSITIVE)
        );
        assert!(split.event_entropy_rate <= 65_536.0 * f64::EPSILON);
        assert!(split.entropy_duality_residual <= 5.0e-9);
        assert!(split.node_entropy_rate.is_finite());
    }

    #[test]
    fn mu_tau_swap_is_equivariant_and_invalid_inputs_fail_closed() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let split_case = named_case(&value, "mu_tau_split");
        let temperature = bits(&split_case["temperature_cm_bits"]);
        let original_coordinates = bit_array(&split_case["pair_cloglog"]);
        let original = assemble_self_action(
            &grid,
            &original_coordinates,
            temperature,
            F10SelfActionConfig::default(),
        )
        .unwrap();
        let mut swapped_coordinates = original_coordinates.clone();
        for node in 0..grid.order {
            swapped_coordinates.swap(grid.order + node, 2 * grid.order + node);
        }
        let swapped = assemble_self_action(
            &grid,
            &swapped_coordinates,
            temperature,
            F10SelfActionConfig::default(),
        )
        .unwrap();
        let species_permutation = [0_usize, 1, 4, 5, 2, 3];
        for (observed_species, &reference_species) in species_permutation.iter().enumerate() {
            assert_hybrid_close(
                &swapped.native
                    [observed_species * grid.order..(observed_species + 1) * grid.order],
                &original.native
                    [reference_species * grid.order..(reference_species + 1) * grid.order],
                maximum_absolute(&original.native),
                5.0e-9,
            );
        }

        assert!(
            assemble_self_action(
                &grid,
                &[0.0; 23],
                temperature,
                F10SelfActionConfig::default(),
            )
            .is_err()
        );
        let mut nonfinite = original_coordinates;
        nonfinite[0] = f64::NAN;
        assert!(
            assemble_self_action(
                &grid,
                &nonfinite,
                temperature,
                F10SelfActionConfig::default(),
            )
            .is_err()
        );
        assert!(
            assemble_self_action(
                &grid,
                &bit_array(&split_case["pair_cloglog"]),
                -1.0,
                F10SelfActionConfig::default(),
            )
            .is_err()
        );
        let invalid_configuration = F10SelfActionConfig {
            matrix_roundoff_ulps: 0.0,
            ..F10SelfActionConfig::default()
        };
        assert!(
            assemble_self_action(
                &grid,
                &bit_array(&split_case["pair_cloglog"]),
                temperature,
                invalid_configuration,
            )
            .is_err()
        );
    }
}
