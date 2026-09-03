//! RED-first oracle admission tests for D-081R1D3 electron/positron action.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_electron_action::{
    F10ElectronActionConfig, F10ElectronActionError, F10_ELECTRON_MASS_MEV,
    assemble_electron_action,
};
use serde_json::Value;

const FULL_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/full_collision_action_case.json");
const METROLOGY_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/electron_collision_action_metrology.json");

fn full_fixture() -> Value {
    serde_json::from_str(FULL_FIXTURE).expect("valid frozen D-081R1 full-action fixture")
}

fn metrology_fixture() -> Value {
    serde_json::from_str(METROLOGY_FIXTURE)
        .expect("valid frozen D-081R1D3 electron-action metrology fixture")
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

fn oracle_family_scale(value: &Value) -> f64 {
    value["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .flat_map(|case| {
            case["electron_families"]
                .as_object()
                .expect("electron family map")
                .values()
                .map(bit_array)
                .collect::<Vec<_>>()
        })
        .map(|family| maximum_absolute(&family))
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn assert_hybrid_close(
    actual: &[f64],
    expected: &[f64],
    reference_scale: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let absolute_floor = 65_536.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
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
    let absolute_floor = 131_072.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    let allowed = absolute_floor + relative_tolerance * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= allowed,
        "scalar mismatch: actual={actual:.17e}, expected={expected:.17e}, allowed={allowed:.17e}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_electron_fixture_contract_is_exact() {
        let full = full_fixture();
        let metrology = metrology_fixture();
        assert_eq!(full["schema"], "rabbit.d081r1.full_collision_action.v1");
        assert_eq!(
            metrology["schema"],
            "rabbit.d081r1d3.electron_action_metrology.v1"
        );
        assert_eq!(
            metrology["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(
            metrology["full_collision_fixture_git_blob"],
            "c94d2e72a1f8300b7c20c9c793417a5c4a5fa302"
        );
        assert_eq!(metrology["elastic_event_count"], 12);
        assert_eq!(metrology["pair_event_count"], 3);
        assert_eq!(metrology["electron_event_count"], 15);
        assert_eq!(metrology["order"], 8);
        assert_eq!(bits(&metrology["y_max_bits"]).to_bits(), 8.0_f64.to_bits());
        assert_eq!(
            bits(&metrology["electron_mass_bits"]).to_bits(),
            F10_ELECTRON_MASS_MEV.to_bits()
        );
        assert_eq!(metrology["category_native_reconstruction_budget_ulps"], 256);
    }

    #[test]
    fn electron_action_matches_every_frozen_python_case() {
        let full = full_fixture();
        let metrology = metrology_fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let modal_scale = oracle_array_scale(&full, "electron_modal");
        let native_scale = oracle_array_scale(&full, "electron_native");
        let family_scale = oracle_family_scale(&full);

        for case in full["cases"].as_array().expect("full fixture cases") {
            let name = case["name"].as_str().expect("case name");
            let component = named_case(&metrology, name);
            let result = assemble_electron_action(
                &grid,
                &bit_array(&case["pair_cloglog"]),
                bits(&case["temperature_cm_bits"]),
                bits(&case["temperature_gamma_bits"]),
                F10ElectronActionConfig::default(),
            )
            .unwrap();

            assert_eq!(result.modal.len(), 6 * grid.order);
            assert_eq!(result.native.len(), 6 * grid.order);
            assert_eq!(result.elastic_modal.len(), 6 * grid.order);
            assert_eq!(result.pair_modal.len(), 6 * grid.order);
            assert_eq!(result.elastic_native.len(), 6 * grid.order);
            assert_eq!(result.pair_native.len(), 6 * grid.order);
            assert_eq!(result.family_names.len(), 15);
            assert_eq!(result.family_modal.len(), 15 * 6 * grid.order);
            assert_eq!(result.family_native.len(), 15 * 6 * grid.order);
            assert_eq!(result.bath_energy_by_family.len(), 15);

            assert_hybrid_close(
                &result.modal,
                &bit_array(&case["arrays"]["electron_modal"]),
                modal_scale,
                4.0e-9,
            );
            assert_hybrid_close(
                &result.native,
                &bit_array(&case["arrays"]["electron_native"]),
                native_scale,
                4.0e-9,
            );
            assert_hybrid_close(
                &result.elastic_modal,
                &bit_array(&component["category_arrays"]["elastic_modal"]),
                modal_scale,
                4.0e-9,
            );
            assert_hybrid_close(
                &result.pair_modal,
                &bit_array(&component["category_arrays"]["pair_modal"]),
                modal_scale,
                4.0e-9,
            );
            assert_hybrid_close(
                &result.elastic_native,
                &bit_array(&component["category_arrays"]["elastic_native"]),
                native_scale,
                5.0e-9,
            );
            assert_hybrid_close(
                &result.pair_native,
                &bit_array(&component["category_arrays"]["pair_native"]),
                native_scale,
                5.0e-9,
            );

            let expected_order = component["family_order"]
                .as_array()
                .expect("family order");
            let family_size = 6 * grid.order;
            for (index, encoded_name) in expected_order.iter().enumerate() {
                let family_name = encoded_name.as_str().expect("family name");
                assert_eq!(result.family_names[index], family_name);
                assert_hybrid_close(
                    &result.family_native[index * family_size..(index + 1) * family_size],
                    &bit_array(&case["electron_families"][family_name]),
                    family_scale,
                    5.0e-9,
                );
                assert_scalar_close(
                    result.bath_energy_by_family[index],
                    bits(&component["bath_energy_by_family"][family_name]),
                    result.bath_energy_by_family[index]
                        .abs()
                        .max(bits(&component["bath_energy_by_family"][family_name]).abs())
                        .max(f64::MIN_POSITIVE),
                    5.0e-9,
                );
            }

            let expected_moments = &component["moments"];
            let moment_scale = result
                .moments
                .absolute_number_rate
                .abs()
                .max(result.moments.absolute_energy_rate.abs())
                .max(1.0);
            assert_scalar_close(
                result.moments.signed_number_rate,
                bits(&expected_moments["signed_number_rate"]),
                moment_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.moments.absolute_number_rate,
                bits(&expected_moments["absolute_number_rate"]),
                moment_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.moments.signed_energy_rate,
                bits(&expected_moments["signed_energy_rate"]),
                moment_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.moments.absolute_energy_rate,
                bits(&expected_moments["absolute_energy_rate"]),
                moment_scale,
                5.0e-9,
            );

            assert_eq!(
                result.whole_reaction_domain_rejections,
                usize::try_from(
                    component["whole_reaction_domain_rejections"]
                        .as_u64()
                        .expect("electron rejection count"),
                )
                .expect("electron rejection count fits usize")
            );
            assert_eq!(
                result.elastic_domain_rejections,
                usize::try_from(
                    component["elastic_domain_rejections"]
                        .as_u64()
                        .expect("elastic rejection count"),
                )
                .expect("elastic rejection count fits usize")
            );
            assert_eq!(
                result.pair_domain_rejections,
                usize::try_from(
                    component["pair_domain_rejections"]
                        .as_u64()
                        .expect("pair rejection count"),
                )
                .expect("pair rejection count fits usize")
            );
            assert_eq!(
                result.matrix_roundoff_corrections,
                usize::try_from(
                    component["matrix_roundoff_corrections"]
                        .as_u64()
                        .expect("matrix correction count"),
                )
                .expect("matrix correction count fits usize")
            );
            let expected_largest = bits(&component["largest_matrix_roundoff_correction_bits"]);
            assert_scalar_close(
                result.largest_matrix_roundoff_correction,
                expected_largest,
                result
                    .largest_matrix_roundoff_correction
                    .abs()
                    .max(expected_largest.abs())
                    .max(f64::MIN_POSITIVE),
                5.0e-9,
            );

            let diagnostics = &component["diagnostics"];
            let transfer_scale = result
                .neutrino_energy_transfer
                .abs()
                .max(result.electromagnetic_energy_transfer.abs())
                .max(1.0);
            assert_scalar_close(
                result.neutrino_energy_transfer,
                bits(&diagnostics["neutrino_energy_transfer_bits"]),
                transfer_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.electromagnetic_energy_transfer,
                bits(&diagnostics["electromagnetic_energy_transfer_bits"]),
                transfer_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.first_law_residual,
                bits(&diagnostics["first_law_residual_bits"]),
                1.0,
                5.0e-9,
            );
            let h_scale = result
                .neutrino_h_rate
                .abs()
                .max(result.electromagnetic_h_rate.abs())
                .max(result.entropy_production.abs())
                .max(1.0);
            assert_scalar_close(
                result.neutrino_h_rate,
                bits(&diagnostics["neutrino_h_rate_bits"]),
                h_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.electromagnetic_h_rate,
                bits(&diagnostics["electromagnetic_h_rate_bits"]),
                h_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.entropy_production,
                bits(&diagnostics["entropy_production_bits"]),
                h_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.node_neutrino_h_rate,
                bits(&diagnostics["node_neutrino_h_rate_bits"]),
                h_scale,
                5.0e-9,
            );
            assert_scalar_close(
                result.entropy_duality_residual,
                bits(&diagnostics["entropy_duality_residual_bits"]),
                1.0,
                5.0e-9,
            );
        }
    }

    #[test]
    fn electron_component_physics_and_decomposition_are_load_bearing() {
        let full = full_fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let equilibrium_case = named_case(&full, "equilibrium");
        let thermal_case = named_case(&full, "thermal_split");
        let split_case = named_case(&full, "mu_tau_split");

        let equilibrium = assemble_electron_action(
            &grid,
            &bit_array(&equilibrium_case["pair_cloglog"]),
            bits(&equilibrium_case["temperature_cm_bits"]),
            bits(&equilibrium_case["temperature_gamma_bits"]),
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let thermal = assemble_electron_action(
            &grid,
            &bit_array(&thermal_case["pair_cloglog"]),
            bits(&thermal_case["temperature_cm_bits"]),
            bits(&thermal_case["temperature_gamma_bits"]),
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let thermal_scale = maximum_absolute(&thermal.native);
        assert!(maximum_absolute(&equilibrium.native) <= 1.0e-10 * thermal_scale);
        assert!(thermal.neutrino_energy_transfer > 0.0);
        assert!(thermal.electromagnetic_energy_transfer < 0.0);
        assert!(thermal.first_law_residual <= 5.0e-13);
        let entropy_scale = thermal.neutrino_h_rate.abs()
            + thermal.electromagnetic_h_rate.abs()
            + f64::MIN_POSITIVE;
        assert!(thermal.entropy_production >= -5.0e-13 * entropy_scale);
        assert!(thermal.entropy_duality_residual <= 5.0e-12);

        let reconstructed_modal: Vec<f64> = thermal
            .elastic_modal
            .iter()
            .zip(&thermal.pair_modal)
            .map(|(elastic, pair)| elastic + pair)
            .collect();
        assert_hybrid_close(
            &reconstructed_modal,
            &thermal.modal,
            maximum_absolute(&thermal.modal),
            5.0e-9,
        );
        let family_size = 6 * grid.order;
        let reconstructed_family_native: Vec<f64> = (0..family_size)
            .map(|index| {
                (0..15)
                    .map(|family| thermal.family_native[family * family_size + index])
                    .sum()
            })
            .collect();
        assert_hybrid_close(
            &reconstructed_family_native,
            &thermal.native,
            maximum_absolute(&thermal.native),
            6.0e-9,
        );
        let bath_sum = thermal.bath_energy_by_family.iter().sum::<f64>();
        assert_scalar_close(
            bath_sum,
            thermal.electromagnetic_energy_transfer,
            thermal.electromagnetic_energy_transfer.abs().max(f64::MIN_POSITIVE),
            6.0e-9,
        );

        let original_coordinates = bit_array(&split_case["pair_cloglog"]);
        let original = assemble_electron_action(
            &grid,
            &original_coordinates,
            bits(&split_case["temperature_cm_bits"]),
            bits(&split_case["temperature_gamma_bits"]),
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let mut swapped_coordinates = original_coordinates.clone();
        for node in 0..grid.order {
            swapped_coordinates.swap(grid.order + node, 2 * grid.order + node);
        }
        let swapped = assemble_electron_action(
            &grid,
            &swapped_coordinates,
            bits(&split_case["temperature_cm_bits"]),
            bits(&split_case["temperature_gamma_bits"]),
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let species_permutation = [0_usize, 1, 4, 5, 2, 3];
        for (observed_species, &reference_species) in species_permutation.iter().enumerate() {
            assert_hybrid_close(
                &swapped.native[observed_species * grid.order..(observed_species + 1) * grid.order],
                &original.native
                    [reference_species * grid.order..(reference_species + 1) * grid.order],
                maximum_absolute(&original.native),
                6.0e-9,
            );
        }
    }

    #[test]
    fn finite_electron_mass_temperature_split_and_fail_closed_semantics_are_load_bearing() {
        let full = full_fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let thermal_case = named_case(&full, "thermal_split");
        let coordinates = bit_array(&thermal_case["pair_cloglog"]);
        let t_cm = bits(&thermal_case["temperature_cm_bits"]);
        let t_gamma = bits(&thermal_case["temperature_gamma_bits"]);
        let admitted = assemble_electron_action(
            &grid,
            &coordinates,
            t_cm,
            t_gamma,
            F10ElectronActionConfig::default(),
        )
        .unwrap();

        let altered_mass = assemble_electron_action(
            &grid,
            &coordinates,
            t_cm,
            t_gamma,
            F10ElectronActionConfig {
                electron_mass_mev: 0.45,
                ..F10ElectronActionConfig::default()
            },
        )
        .unwrap();
        let mass_difference = admitted
            .native
            .iter()
            .zip(&altered_mass.native)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(mass_difference > 1.0e-8 * maximum_absolute(&admitted.native));

        let equal_temperature = assemble_electron_action(
            &grid,
            &coordinates,
            t_cm,
            t_cm,
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let temperature_difference = admitted
            .native
            .iter()
            .zip(&equal_temperature.native)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f64, f64::max);
        assert!(temperature_difference > 1.0e-8 * maximum_absolute(&admitted.native));

        assert_eq!(
            assemble_electron_action(
                &grid,
                &[0.0; 23],
                t_cm,
                t_gamma,
                F10ElectronActionConfig::default(),
            ),
            Err(F10ElectronActionError::InvalidInput)
        );
        let mut nonfinite = coordinates.clone();
        nonfinite[0] = f64::NAN;
        assert_eq!(
            assemble_electron_action(
                &grid,
                &nonfinite,
                t_cm,
                t_gamma,
                F10ElectronActionConfig::default(),
            ),
            Err(F10ElectronActionError::InvalidInput)
        );
        assert_eq!(
            assemble_electron_action(
                &grid,
                &coordinates,
                -1.0,
                t_gamma,
                F10ElectronActionConfig::default(),
            ),
            Err(F10ElectronActionError::InvalidInput)
        );
        assert_eq!(
            assemble_electron_action(
                &grid,
                &coordinates,
                t_cm,
                0.0,
                F10ElectronActionConfig::default(),
            ),
            Err(F10ElectronActionError::InvalidInput)
        );
        assert_eq!(
            assemble_electron_action(
                &grid,
                &coordinates,
                t_cm,
                t_gamma,
                F10ElectronActionConfig {
                    matrix_roundoff_ulps: 0.0,
                    ..F10ElectronActionConfig::default()
                },
            ),
            Err(F10ElectronActionError::InvalidConfiguration)
        );
        assert_eq!(
            assemble_electron_action(
                &grid,
                &coordinates,
                t_cm,
                t_gamma,
                F10ElectronActionConfig {
                    electron_mass_mev: 0.0,
                    ..F10ElectronActionConfig::default()
                },
            ),
            Err(F10ElectronActionError::InvalidConfiguration)
        );
    }
}
