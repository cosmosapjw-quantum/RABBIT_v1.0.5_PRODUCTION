//! RED-first admission tests for the D-081R1E retained packed RHS.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_packed_rhs::{F10PackedRhsConfig, F10PackedRhsError, evaluate_f10_packed_rhs};
use serde_json::Value;

const RETAINED_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/retained_packed_rhs_case.json");
const CONTROL_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/full_collision_action_case.json");

fn retained_fixture() -> Value {
    serde_json::from_str(RETAINED_FIXTURE).expect("valid retained packed-RHS fixture")
}

fn control_fixture() -> Value {
    serde_json::from_str(CONTROL_FIXTURE).expect("valid order-8 collision fixture")
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

fn maximum_absolute_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0_f64, f64::max)
}

#[track_caller]
fn assert_hybrid_close(
    actual: &[f64],
    expected: &[f64],
    reference_scale: f64,
    relative_tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());
    let absolute_floor = 1_048_576.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
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

#[track_caller]
fn assert_scalar_close(actual: f64, expected: f64, scale: f64, relative_tolerance: f64) {
    let absolute_floor = 1_048_576.0 * f64::EPSILON * scale.abs().max(f64::MIN_POSITIVE);
    let allowed = absolute_floor + relative_tolerance * actual.abs().max(expected.abs());
    assert!(
        (actual - expected).abs() <= allowed,
        "scalar mismatch: actual={actual:.17e}, expected={expected:.17e}, allowed={allowed:.17e}"
    );
}

fn control_state(case: &Value, elapsed: f64) -> (f64, Vec<f64>) {
    let t_cm = bits(&case["temperature_cm_bits"]);
    let t_gamma = bits(&case["temperature_gamma_bits"]);
    let mut state = bit_array(&case["pair_cloglog"]);
    state.push(t_gamma);
    state.push(elapsed);
    let ln_a = (10.0 / t_cm).ln();
    (ln_a, state)
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

fn chart_chain(state: &[f64], spectral_size: usize) -> Vec<f64> {
    state[..spectral_size]
        .iter()
        .map(|coordinate| {
            let exponential = coordinate.exp();
            (coordinate - exponential).exp()
        })
        .collect()
}

fn spectral_from_native(
    native: &[f64],
    state: &[f64],
    order: usize,
    hubble: f64,
    average_factor: f64,
    include_chain: bool,
    include_hubble: bool,
) -> Vec<f64> {
    let mut rate = pair_rate(native, order);
    if average_factor != 1.0 {
        for value in &mut rate {
            *value *= average_factor;
        }
    }
    let chain = chart_chain(state, 3 * order);
    rate.into_iter()
        .zip(chain)
        .map(|(value, q)| {
            let mut denominator = 1.0;
            if include_chain {
                denominator *= q;
            }
            if include_hubble {
                denominator *= hubble;
            }
            value / denominator
        })
        .collect()
}

fn scaled_difference(actual: &[f64], reference: &[f64]) -> f64 {
    maximum_absolute_difference(actual, reference)
        / maximum_absolute(actual)
            .max(maximum_absolute(reference))
            .max(f64::MIN_POSITIVE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_oracle_and_control_contracts_are_exact() {
        let retained = retained_fixture();
        let control = control_fixture();
        assert_eq!(retained["schema"], "rabbit.d081r1e.retained_packed_rhs.v1");
        assert_eq!(
            retained["d4_head"],
            "002086662bf2e553c78f4b247868cb1fd9e43f21"
        );
        assert_eq!(
            retained["d4_tree"],
            "d01ae7c0d3d9fbe8ce9513d054b835d3596f1de2"
        );
        assert_eq!(
            retained["retained_sha256"],
            "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
        );
        assert_eq!(retained["order"], 60);
        assert_eq!(retained["packed_state"]["shape"][0], 182);
        assert_eq!(retained["packed_rhs"]["shape"][0], 182);
        assert_eq!(control["schema"], "rabbit.d081r1.full_collision_action.v1");
        assert_eq!(control["order"], 8);
        assert_eq!(control["self_event_count"], 27);
        assert_eq!(control["electron_event_count"], 15);
    }

    #[test]
    fn retained_order60_packed_rhs_matches_the_frozen_python_oracle() {
        let value = retained_fixture();
        let grid = F10ActionGrid::affine_legendre(60, 30.0).unwrap();
        let state = bit_array(&value["packed_state"]);
        let ln_a = bits(&value["ln_a_bits"]);
        let result =
            evaluate_f10_packed_rhs(&grid, ln_a, &state, F10PackedRhsConfig::default()).unwrap();

        assert_eq!(result.values.len(), 182);
        assert_eq!(result.combined_action.native_total.len(), 360);
        assert_eq!(result.combined_action.modal_total.len(), 360);

        let expected_action_native = bit_array(&value["combined_action_native"]);
        let expected_action_modal = bit_array(&value["combined_action_modal"]);
        let action_scale = maximum_absolute(&bit_array(
            &value["absolute_envelopes"]["combined_action_native"],
        ));
        assert_hybrid_close(
            &result.combined_action.native_total,
            &expected_action_native,
            action_scale,
            5.0e-7,
        );
        assert_hybrid_close(
            &result.combined_action.modal_total,
            &expected_action_modal,
            maximum_absolute(&expected_action_modal),
            5.0e-7,
        );

        let expected_spectral = bit_array(&value["spectral_rhs"]);
        let spectral_scale =
            maximum_absolute(&bit_array(&value["absolute_envelopes"]["spectral_rhs"]));
        assert_hybrid_close(
            &result.values[..180],
            &expected_spectral,
            spectral_scale,
            5.0e-7,
        );
        assert_scalar_close(
            result.values[180],
            bits(&value["temperature_rhs_bits"]),
            bits(&value["temperature_rhs_bits"]).abs(),
            5.0e-8,
        );
        assert_scalar_close(
            result.values[181],
            bits(&value["elapsed_rhs_bits"]),
            bits(&value["elapsed_rhs_bits"]).abs(),
            5.0e-10,
        );

        let expected_rho = bit_array(&value["rho_neutrino_by_flavour"]);
        assert_hybrid_close(
            &result.diagnostics.rho_neutrino_by_flavour,
            &expected_rho,
            maximum_absolute(&expected_rho),
            2.0e-8,
        );
        for (actual, key) in [
            (
                result.diagnostics.rho_neutrino_total,
                "rho_neutrino_total_bits",
            ),
            (
                result.diagnostics.rho_electromagnetic,
                "rho_electromagnetic_bits",
            ),
            (
                result.diagnostics.pressure_electromagnetic,
                "pressure_electromagnetic_bits",
            ),
            (
                result.diagnostics.drho_electromagnetic_dt,
                "drho_electromagnetic_dt_bits",
            ),
            (result.diagnostics.rho_total, "rho_total_bits"),
            (result.diagnostics.hubble_mev, "hubble_mev_bits"),
            (result.diagnostics.neutrino_energy_transfer, "q_nu_bits"),
            (
                result.diagnostics.electromagnetic_energy_transfer,
                "q_em_bits",
            ),
        ] {
            let expected = bits(&value[key]);
            assert_scalar_close(actual, expected, expected.abs(), 2.0e-8);
        }
        assert_scalar_close(
            result.diagnostics.first_law_residual,
            bits(&value["first_law_residual_bits"]),
            1.0,
            5.0e-10,
        );
        assert_eq!(
            result.diagnostics.whole_reaction_domain_rejections,
            usize::try_from(
                value["support_and_roundoff_metrology"]["whole_reaction_domain_rejections"]
                    .as_u64()
                    .unwrap(),
            )
            .unwrap()
        );
        assert_eq!(
            result.diagnostics.matrix_roundoff_corrections,
            usize::try_from(
                value["support_and_roundoff_metrology"]["matrix_roundoff_corrections"]
                    .as_u64()
                    .unwrap(),
            )
            .unwrap()
        );

        let repeated =
            evaluate_f10_packed_rhs(&grid, ln_a, &state, F10PackedRhsConfig::default()).unwrap();
        assert_eq!(
            result
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            repeated
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn state_chart_and_passive_elapsed_contract_fail_closed() {
        let value = control_fixture();
        let case = named_case(&value, "thermal_split");
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let (ln_a, state) = control_state(case, 0.0);
        let result =
            evaluate_f10_packed_rhs(&grid, ln_a, &state, F10PackedRhsConfig::default()).unwrap();
        let mut changed_elapsed = state.clone();
        changed_elapsed[25] = 9.876_543_21e18;
        let passive =
            evaluate_f10_packed_rhs(&grid, ln_a, &changed_elapsed, F10PackedRhsConfig::default())
                .unwrap();
        assert_eq!(
            result
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            passive
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        assert_eq!(
            evaluate_f10_packed_rhs(&grid, ln_a, &state[..24], F10PackedRhsConfig::default(),),
            Err(F10PackedRhsError::InvalidInput)
        );
        for invalid_coordinate in [f64::NAN, f64::INFINITY, 1.0e3, -1.0e3] {
            let mut invalid = state.clone();
            invalid[0] = invalid_coordinate;
            assert_eq!(
                evaluate_f10_packed_rhs(&grid, ln_a, &invalid, F10PackedRhsConfig::default(),),
                Err(if invalid_coordinate.is_finite() {
                    F10PackedRhsError::Chart
                } else {
                    F10PackedRhsError::InvalidInput
                })
            );
        }
        let mut invalid_temperature = state.clone();
        invalid_temperature[24] = 0.0;
        assert_eq!(
            evaluate_f10_packed_rhs(
                &grid,
                ln_a,
                &invalid_temperature,
                F10PackedRhsConfig::default(),
            ),
            Err(F10PackedRhsError::InvalidInput)
        );
        let mut invalid_elapsed = state.clone();
        invalid_elapsed[25] = f64::NAN;
        assert_eq!(
            evaluate_f10_packed_rhs(&grid, ln_a, &invalid_elapsed, F10PackedRhsConfig::default(),),
            Err(F10PackedRhsError::InvalidInput)
        );
        assert_eq!(
            evaluate_f10_packed_rhs(&grid, f64::NAN, &state, F10PackedRhsConfig::default(),),
            Err(F10PackedRhsError::InvalidInput)
        );
        assert_eq!(
            evaluate_f10_packed_rhs(
                &grid,
                ln_a,
                &state,
                F10PackedRhsConfig {
                    t_start_mev: 0.0,
                    ..F10PackedRhsConfig::default()
                },
            ),
            Err(F10PackedRhsError::InvalidConfiguration)
        );
    }

    #[test]
    fn collision_chart_hubble_and_thermodynamic_mutations_are_killed() {
        let value = control_fixture();
        let thermal_case = named_case(&value, "thermal_split");
        let split_case = named_case(&value, "mu_tau_split");
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let (thermal_ln_a, thermal_state) = control_state(thermal_case, 0.0);
        let thermal = evaluate_f10_packed_rhs(
            &grid,
            thermal_ln_a,
            &thermal_state,
            F10PackedRhsConfig::default(),
        )
        .unwrap();
        let correct = &thermal.values[..24];
        let hubble = thermal.diagnostics.hubble_mev;

        let no_half = spectral_from_native(
            &thermal.combined_action.native_total,
            &thermal_state,
            8,
            hubble,
            2.0,
            true,
            true,
        );
        assert!(scaled_difference(&no_half, correct) > 0.5);

        let modal_as_native = spectral_from_native(
            &thermal.combined_action.modal_total,
            &thermal_state,
            8,
            hubble,
            1.0,
            true,
            true,
        );
        assert!(scaled_difference(&modal_as_native, correct) > 1.0e-4);

        let no_chain = spectral_from_native(
            &thermal.combined_action.native_total,
            &thermal_state,
            8,
            hubble,
            1.0,
            false,
            true,
        );
        assert!(scaled_difference(&no_chain, correct) > 1.0e-4);

        let no_hubble = spectral_from_native(
            &thermal.combined_action.native_total,
            &thermal_state,
            8,
            hubble,
            1.0,
            true,
            false,
        );
        assert!(scaled_difference(&no_hubble, correct) > 1.0e-4);

        let numerator = -3.0
            * (thermal.diagnostics.rho_electromagnetic
                + thermal.diagnostics.pressure_electromagnetic)
            + thermal.diagnostics.electromagnetic_energy_transfer / hubble;
        let wrong_qem = -3.0
            * (thermal.diagnostics.rho_electromagnetic
                + thermal.diagnostics.pressure_electromagnetic)
            - thermal.diagnostics.electromagnetic_energy_transfer / hubble;
        assert!(
            (wrong_qem / thermal.diagnostics.drho_electromagnetic_dt - thermal.values[24]).abs()
                > 1.0e-8 * thermal.values[24].abs().max(1.0)
        );
        assert!(
            (numerator - thermal.values[24]).abs() > 1.0e-8 * thermal.values[24].abs().max(1.0)
        );
        assert_eq!(thermal.values[25].to_bits(), (1.0 / hubble).to_bits());
        assert_ne!(thermal.values[25].to_bits(), (-1.0 / hubble).to_bits());

        let self_only = spectral_from_native(
            &thermal.combined_action.self_action.native,
            &thermal_state,
            8,
            hubble,
            1.0,
            true,
            true,
        );
        assert!(scaled_difference(&self_only, correct) > 0.5);

        let (split_ln_a, split_state) = control_state(split_case, 0.0);
        let split = evaluate_f10_packed_rhs(
            &grid,
            split_ln_a,
            &split_state,
            F10PackedRhsConfig::default(),
        )
        .unwrap();
        let electron_only = spectral_from_native(
            &split.combined_action.electron_action.native,
            &split_state,
            8,
            split.diagnostics.hubble_mev,
            1.0,
            true,
            true,
        );
        assert!(scaled_difference(&electron_only, &split.values[..24]) > 1.0e-4);
    }

    #[test]
    fn mu_tau_swap_covariance_survives_the_full_packed_rhs_map() {
        let value = control_fixture();
        let case = named_case(&value, "mu_tau_split");
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let (ln_a, state) = control_state(case, 0.0);
        let original =
            evaluate_f10_packed_rhs(&grid, ln_a, &state, F10PackedRhsConfig::default()).unwrap();
        let mut swapped_state = state.clone();
        for node in 0..8 {
            swapped_state.swap(8 + node, 16 + node);
        }
        let swapped =
            evaluate_f10_packed_rhs(&grid, ln_a, &swapped_state, F10PackedRhsConfig::default())
                .unwrap();

        let permutation = [0_usize, 2, 1];
        let scale = maximum_absolute(&original.values[..24]);
        for (observed_flavour, &reference_flavour) in permutation.iter().enumerate() {
            assert_hybrid_close(
                &swapped.values[observed_flavour * 8..(observed_flavour + 1) * 8],
                &original.values[reference_flavour * 8..(reference_flavour + 1) * 8],
                scale,
                1.0e-7,
            );
        }
        assert_scalar_close(
            swapped.values[24],
            original.values[24],
            original.values[24].abs(),
            1.0e-9,
        );
        assert_scalar_close(
            swapped.values[25],
            original.values[25],
            original.values[25].abs(),
            1.0e-9,
        );
        assert_scalar_close(
            swapped.diagnostics.rho_neutrino_by_flavour[1],
            original.diagnostics.rho_neutrino_by_flavour[2],
            original.diagnostics.rho_neutrino_total,
            1.0e-10,
        );
        assert_scalar_close(
            swapped.diagnostics.rho_neutrino_by_flavour[2],
            original.diagnostics.rho_neutrino_by_flavour[1],
            original.diagnostics.rho_neutrino_total,
            1.0e-10,
        );
    }
}
