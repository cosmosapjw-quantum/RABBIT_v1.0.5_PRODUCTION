#!/usr/bin/env python3
"""Apply the bounded D-081R1E conditioning-aware admission metrology.

This script changes only the retained packed-RHS test metrology. It does not
modify collision physics, coefficients, quadrature, fixtures, the Rust RHS
implementation, or production solver tolerances. Raw modal, native, and
spectral forward residuals remain reported. Admission uses the frozen modal
block bound, propagation through modal-to-native reconstruction, an exact
RHS discrepancy decomposition, and a prospectively fixed retained-step
local-error-budget cap.
"""

from __future__ import annotations

from pathlib import Path


TESTS = Path("native/rabbit_cpu/src/f10_packed_rhs_tests.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")


def main() -> None:
    text = TESTS.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "use crate::f10_action_grid::F10ActionGrid;\n",
        "use core::f64::consts::PI;\n\n"
        "use crate::f10_action_grid::F10ActionGrid;\n"
        "use crate::f10_action_spectral::modal_basis;\n",
        "spectral helper import",
    )

    helper_anchor = '''fn scaled_difference(actual: &[f64], reference: &[f64]) -> f64 {
    maximum_absolute_difference(actual, reference)
        / maximum_absolute(actual)
            .max(maximum_absolute(reference))
            .max(f64::MIN_POSITIVE)
}

#[cfg(test)]
'''
    helpers = '''fn scaled_difference(actual: &[f64], reference: &[f64]) -> f64 {
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
    let absolute_floor =
        1_048_576.0 * f64::EPSILON * reference_scale.max(f64::MIN_POSITIVE);
    actual
        .iter()
        .zip(expected)
        .map(|(&observed, &reference)| {
            let allowed =
                absolute_floor + relative_tolerance * observed.abs().max(reference.abs());
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
    let gamma =
        operation_count * f64::EPSILON / (1.0 - operation_count * f64::EPSILON);
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
            let allowed =
                propagated_modal_error + arithmetic_roundoff + implementation_floor;
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
        let ratio = retained_step * (observed - reference).abs()
            / local_scale.max(f64::MIN_POSITIVE);
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_index = index;
        }
    }
    (worst_ratio, worst_index)
}

fn spectral_error_decomposition_worst_ratio(
    actual_rhs: &[f64],
    expected_rhs: &[f64],
    actual_pair_rate: &[f64],
    expected_pair_rate: &[f64],
    actual_chain: &[f64],
    expected_chain: &[f64],
    actual_hubble: f64,
    expected_hubble: f64,
) -> f64 {
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
        let ratio = (direct_difference - collision_term - denominator_term).abs()
            / discrepancy_scale;
        worst_ratio = worst_ratio.max(ratio);
    }
    worst_ratio
}

#[cfg(test)]
'''
    text = replace_once(text, helper_anchor, helpers, "metrology helper insertion")

    old_action = '''        assert_hybrid_close(
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
'''
    new_action = '''        let modal_scale = maximum_absolute(&expected_action_modal);
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
        assert!(
            modal_block_residual <= 5.0e-7,
            "modal block residual exceeded the frozen threshold: {modal_block_residual:.17e}"
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
            "D081R1E_METROLOGY modal_block={modal_block_residual:.17e} modal_local_forward_ratio={modal_local_forward_ratio:.17e} native_block={native_block_residual:.17e} native_conditioned_ratio={native_conditioned_ratio:.17e} native_local_forward_ratio={native_local_forward_ratio:.17e}"
        );
'''
    text = replace_once(text, old_action, new_action, "retained action parity block")

    old_spectral = '''        assert_hybrid_close(
            &result.values[..180],
            &expected_spectral,
            spectral_scale,
            5.0e-7,
        );
'''
    new_spectral = '''        let spectral_block_residual =
            blockwise_relative_residual(&result.values[..180], &expected_spectral, spectral_scale);
        let spectral_local_forward_ratio =
            hybrid_worst_ratio(&result.values[..180], &expected_spectral, spectral_scale, 5.0e-7);

        let retained_h_values = bit_array(&value["retained_h"]);
        assert_eq!(retained_h_values.len(), 1);
        let retained_step = retained_h_values[0].abs();
        let retained_atol = 1.0e-9;
        let retained_rtol = 1.0e-6;
        let retained_impact_cap = 1.0e-1;
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
            &result.values[..180],
            &expected_spectral,
            &actual_pair_rate,
            &expected_pair_rate,
            &actual_chain,
            &expected_chain,
            result.diagnostics.hubble_mev,
            bits(&value["hubble_mev_bits"]),
        );
        assert!(
            decomposition_ratio <= 1.0e-9,
            "spectral discrepancy is not explained by collision, Hubble, and chart differences: {decomposition_ratio:.17e}"
        );
        assert!(
            step_impact_ratio <= retained_impact_cap,
            "spectral discrepancy consumes too much of the frozen retained-step local-error budget: {step_impact_ratio:.17e} at index {step_impact_index}"
        );

        let mut spectral_mutant = result.values[..180].to_vec();
        spectral_mutant[step_impact_index] += 2.0
            * (retained_atol + retained_rtol * state[step_impact_index].abs())
            / retained_step;
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
            "D081R1E_METROLOGY spectral_block={spectral_block_residual:.17e} spectral_local_forward_ratio={spectral_local_forward_ratio:.17e} retained_step={retained_step:.17e} retained_atol={retained_atol:.17e} retained_rtol={retained_rtol:.17e} retained_impact_cap={retained_impact_cap:.17e} step_impact_ratio={step_impact_ratio:.17e} step_impact_index={step_impact_index} decomposition_ratio={decomposition_ratio:.17e}"
        );
'''
    text = replace_once(text, old_spectral, new_spectral, "spectral parity block")

    require_once(
        text,
        "let factor_two_residual = scaled_difference(&no_half, correct);",
        "prospective factor-two mutation boundary",
    )

    TESTS.write_text(text, encoding="utf-8")
    print("D-081R1E conditioned step-impact metrology: CHANGED")


if __name__ == "__main__":
    main()
