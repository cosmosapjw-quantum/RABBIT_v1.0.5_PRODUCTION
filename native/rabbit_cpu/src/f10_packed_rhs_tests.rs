//! RED-first admission tests for the D-081R1E retained packed RHS.

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_spectral::modal_basis;
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

fn blockwise_relative_residual(actual: &[f64], expected: &[f64], reference_scale: f64) -> f64 {
    maximum_absolute_difference(actual, expected)
        / maximum_absolute(actual)
            .max(maximum_absolute(expected))
            .max(reference_scale)
            .max(f64::MIN_POSITIVE)
}

fn hybrid_worst_ratio(
    actual: &[f64],
    expected: &[f64],
    reference_scale: f64,
    relative_tolerance: f64,
) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let absolute_floor = 1_048_576.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
    actual
        .iter()
        .zip(expected)
        .map(|(&observed, &reference)| {
            let allowed = absolute_floor + relative_tolerance * observed.abs().max(reference.abs());
            (observed - reference).abs() / allowed.max(f64::MIN_POSITIVE)
        })
        .fold(0.0_f64, f64::max)
}

fn conditioned_native_worst_ratio(
    grid: &F10ActionGrid,
    actual_native: &[f64],
    expected_native: &[f64],
    actual_modal: &[f64],
    expected_modal: &[f64],
    temperature_cm: f64,
) -> f64 {
    assert!(temperature_cm.is_finite() && temperature_cm > 0.0);
    assert_eq!(actual_native.len(), expected_native.len());
    assert_eq!(actual_modal.len(), expected_modal.len());
    assert_eq!(actual_native.len(), actual_modal.len());
    assert_eq!(actual_native.len() % grid.order, 0);

    let rows = actual_native.len() / grid.order;
    let basis = modal_basis(grid, &grid.nodes).expect("valid frozen modal basis");
    assert_eq!(basis.len(), grid.order * grid.order);
    let normalization = temperature_cm.powi(3) / (2.0 * PI.powi(2));
    let operation_count = (8 * grid.order + 64) as f64;
    let gamma = operation_count * f64::EPSILON / (1.0 - operation_count * f64::EPSILON);
    let mut worst_ratio = 0.0_f64;

    for row in 0..rows {
        for node in 0..grid.order {
            let denominator = normalization * grid.nodes[node].powi(2);
            assert!(denominator.is_finite() && denominator > 0.0);
            let mut propagated_modal_error = 0.0_f64;
            let mut modal_magnitude = 0.0_f64;
            for mode in 0..grid.order {
                let basis_value = basis[node * grid.order + mode].abs();
                let actual = actual_modal[row * grid.order + mode];
                let expected = expected_modal[row * grid.order + mode];
                propagated_modal_error += (actual - expected).abs() * basis_value;
                modal_magnitude += (actual.abs() + expected.abs()) * basis_value;
            }
            propagated_modal_error /= denominator;
            let reconstructed_magnitude = modal_magnitude / denominator;
            let arithmetic_roundoff = gamma * reconstructed_magnitude;
            let index = row * grid.order + node;
            let implementation_floor = 1_048_576.0
                * f64::EPSILON
                * actual_native[index]
                    .abs()
                    .max(expected_native[index].abs())
                    .max(reconstructed_magnitude)
                    .max(f64::MIN_POSITIVE);
            let allowed = propagated_modal_error + arithmetic_roundoff + implementation_floor;
            let ratio = (actual_native[index] - expected_native[index]).abs()
                / allowed.max(f64::MIN_POSITIVE);
            worst_ratio = worst_ratio.max(ratio);
        }
    }
    worst_ratio
}

fn retained_step_impact_ratio(
    actual: &[f64],
    expected: &[f64],
    state: &[f64],
    retained_step: f64,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> (f64, usize) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), state.len());
    assert!(retained_step.is_finite() && retained_step > 0.0);
    assert!(absolute_tolerance.is_finite() && absolute_tolerance > 0.0);
    assert!(relative_tolerance.is_finite() && relative_tolerance > 0.0);

    let mut worst_ratio = 0.0_f64;
    let mut worst_index = 0_usize;
    for (index, ((&observed, &reference), &coordinate)) in
        actual.iter().zip(expected).zip(state).enumerate()
    {
        let local_scale = absolute_tolerance + relative_tolerance * coordinate.abs();
        let ratio =
            retained_step * (observed - reference).abs() / local_scale.max(f64::MIN_POSITIVE);
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_index = index;
        }
    }
    (worst_ratio, worst_index)
}

fn spectral_error_decomposition_worst_ratio(
    rhs: (&[f64], &[f64]),
    pair_rate: (&[f64], &[f64]),
    chain: (&[f64], &[f64]),
    hubble: (f64, f64),
) -> f64 {
    let (actual_rhs, expected_rhs) = rhs;
    let (actual_pair_rate, expected_pair_rate) = pair_rate;
    let (actual_chain, expected_chain) = chain;
    let (actual_hubble, expected_hubble) = hubble;

    assert_eq!(actual_rhs.len(), expected_rhs.len());
    assert_eq!(actual_rhs.len(), actual_pair_rate.len());
    assert_eq!(actual_rhs.len(), expected_pair_rate.len());
    assert_eq!(actual_rhs.len(), actual_chain.len());
    assert_eq!(actual_rhs.len(), expected_chain.len());
    assert!(actual_hubble.is_finite() && actual_hubble != 0.0);
    assert!(expected_hubble.is_finite() && expected_hubble != 0.0);

    let discrepancy_scale =
        maximum_absolute_difference(actual_rhs, expected_rhs).max(f64::MIN_POSITIVE);
    let mut worst_ratio = 0.0_f64;
    for index in 0..actual_rhs.len() {
        let actual_denominator = actual_hubble * actual_chain[index];
        let expected_denominator = expected_hubble * expected_chain[index];
        assert!(actual_denominator.is_finite() && actual_denominator != 0.0);
        assert!(expected_denominator.is_finite() && expected_denominator != 0.0);

        let direct_difference = actual_rhs[index] - expected_rhs[index];
        let collision_term =
            (actual_pair_rate[index] - expected_pair_rate[index]) / actual_denominator;
        let denominator_term = expected_pair_rate[index]
            * (expected_denominator - actual_denominator)
            / (actual_denominator * expected_denominator);
        let ratio =
            (direct_difference - collision_term - denominator_term).abs() / discrepancy_scale;
        worst_ratio = worst_ratio.max(ratio);
    }
    worst_ratio
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
        let modal_scale = maximum_absolute(&expected_action_modal);
        let modal_block_residual = blockwise_relative_residual(
            &result.combined_action.modal_total,
            &expected_action_modal,
            modal_scale,
        );
        let modal_local_forward_ratio = hybrid_worst_ratio(
            &result.combined_action.modal_total,
            &expected_action_modal,
            modal_scale,
            5.0e-7,
        );
        let retained_modal_cap = 1.0e-7;
        assert!(
            modal_block_residual <= retained_modal_cap,
            "modal block residual exceeded the prospectively frozen threshold: {modal_block_residual:.17e}"
        );

        let temperature_cm = bits(&value["temperature_cm_bits"]);
        let native_conditioned_ratio = conditioned_native_worst_ratio(
            &grid,
            &result.combined_action.native_total,
            &expected_action_native,
            &result.combined_action.modal_total,
            &expected_action_modal,
            temperature_cm,
        );
        let native_block_residual = blockwise_relative_residual(
            &result.combined_action.native_total,
            &expected_action_native,
            action_scale,
        );
        let native_local_forward_ratio = hybrid_worst_ratio(
            &result.combined_action.native_total,
            &expected_action_native,
            action_scale,
            5.0e-7,
        );
        assert!(
            native_conditioned_ratio <= 1.0,
            "native discrepancy is not explained by admitted modal error and reconstruction arithmetic: {native_conditioned_ratio:.17e}"
        );
        let mut native_mutant = result.combined_action.native_total.clone();
        native_mutant[300] += action_scale;
        assert!(
            conditioned_native_worst_ratio(
                &grid,
                &native_mutant,
                &expected_action_native,
                &result.combined_action.modal_total,
                &expected_action_modal,
                temperature_cm,
            ) > 1.0,
            "conditioned native gate did not kill an order-one native-only mutation"
        );
        eprintln!(
            "D081R1E_METROLOGY retained_modal_cap={retained_modal_cap:.17e} modal_block={modal_block_residual:.17e} modal_local_forward_ratio={modal_local_forward_ratio:.17e} native_block={native_block_residual:.17e} native_conditioned_ratio={native_conditioned_ratio:.17e} native_local_forward_ratio={native_local_forward_ratio:.17e}"
        );

        let expected_spectral = bit_array(&value["spectral_rhs"]);
        let spectral_scale =
            maximum_absolute(&bit_array(&value["absolute_envelopes"]["spectral_rhs"]));
        let spectral_block_residual =
            blockwise_relative_residual(&result.values[..180], &expected_spectral, spectral_scale);
        let spectral_local_forward_ratio = hybrid_worst_ratio(
            &result.values[..180],
            &expected_spectral,
            spectral_scale,
            5.0e-7,
        );

        let retained_h_values = bit_array(&value["retained_h"]);
        assert_eq!(retained_h_values.len(), 1);
        let retained_step = retained_h_values[0].abs();
        let retained_atol = 1.0e-9;
        let retained_rtol = 1.0e-6;
        let retained_impact_cap = 1.0e-3;
        let (step_impact_ratio, step_impact_index) = retained_step_impact_ratio(
            &result.values[..180],
            &expected_spectral,
            &state[..180],
            retained_step,
            retained_atol,
            retained_rtol,
        );

        let expected_pair_rate = bit_array(&value["pair_collision_rate"]);
        let actual_pair_rate = pair_rate(&result.combined_action.native_total, grid.order);
        let expected_chain = bit_array(&value["cloglog_chain_factor"]);
        let actual_chain = chart_chain(&state, 180);
        let decomposition_ratio = spectral_error_decomposition_worst_ratio(
            (&result.values[..180], expected_spectral.as_slice()),
            (actual_pair_rate.as_slice(), expected_pair_rate.as_slice()),
            (actual_chain.as_slice(), expected_chain.as_slice()),
            (
                result.diagnostics.hubble_mev,
                bits(&value["hubble_mev_bits"]),
            ),
        );
        assert!(
            decomposition_ratio <= 1.0e-9,
            "spectral discrepancy is not explained by collision, Hubble, and chart differences: {decomposition_ratio:.17e}"
        );
        assert!(
            step_impact_ratio <= retained_impact_cap,
            "spectral discrepancy consumes too much of the frozen retained-step local-error budget: {step_impact_ratio:.17e} at index {step_impact_index}"
        );

        let first_law_cap = 5.0e-13;
        let expected_first_law = bits(&value["first_law_residual_bits"]);
        assert!(
            result.diagnostics.first_law_residual.abs() <= first_law_cap,
            "Rust first-law residual exceeded the frozen threshold: {:.17e}",
            result.diagnostics.first_law_residual.abs(),
        );
        assert!(
            expected_first_law.abs() <= first_law_cap,
            "Python first-law residual exceeded the frozen threshold: {:.17e}",
            expected_first_law.abs(),
        );

        let mut spectral_mutant = result.values[..180].to_vec();
        spectral_mutant[step_impact_index] +=
            2.0 * (retained_atol + retained_rtol * state[step_impact_index].abs()) / retained_step;
        assert!(
            retained_step_impact_ratio(
                &spectral_mutant,
                &expected_spectral,
                &state[..180],
                retained_step,
                retained_atol,
                retained_rtol,
            )
            .0 > 1.0,
            "retained-step gate did not kill a two-budget spectral mutation"
        );
        eprintln!(
            "D081R1E_METROLOGY spectral_block={spectral_block_residual:.17e} spectral_local_forward_ratio={spectral_local_forward_ratio:.17e} retained_step={retained_step:.17e} retained_atol={retained_atol:.17e} retained_rtol={retained_rtol:.17e} retained_impact_cap={retained_impact_cap:.17e} step_impact_ratio={step_impact_ratio:.17e} step_impact_index={step_impact_index} decomposition_ratio={decomposition_ratio:.17e} first_law_cap={first_law_cap:.17e} rust_first_law={:.17e} python_first_law={:.17e}",
            result.diagnostics.first_law_residual.abs(),
            expected_first_law.abs(),
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
        let doubled_correct: Vec<f64> = correct.iter().map(|value| 2.0 * value).collect();
        assert_hybrid_close(
            &no_half,
            &doubled_correct,
            maximum_absolute(&doubled_correct),
            1.0e-13,
        );
        let factor_two_residual = scaled_difference(&no_half, correct);
        assert!(
            (factor_two_residual - 0.5).abs() <= 512.0 * f64::EPSILON,
            "removing the particle/antiparticle one-half factor must give residual one-half; got {factor_two_residual:.17e}"
        );

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
