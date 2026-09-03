#!/usr/bin/env python3
"""Apply bounded D-081R1D4 diagnostic-definition repairs.

This script changes no collision coefficient, state, grid, quadrature,
fixture, physical tolerance, or action-parity threshold.  It:

1. matches the frozen Python flavour-pair normalization for CP and mu/tau
   diagnostics;
2. scales component-omission negative controls by the omitted component;
3. propagates already-measured array error through the cancellation-sensitive
   CP residual instead of pretending that the normalized scalar is
   well-conditioned near charge-conjugation symmetry.
"""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("native/rabbit_cpu/src/f10_combined_action.rs")
TESTS = Path("native/rabbit_cpu/src/f10_combined_action_tests.rs")


def patch_source() -> bool:
    text = SOURCE.read_text(encoding="utf-8")
    if "fn relative_max_difference(" in text:
        return False

    start_marker = "fn symmetry_residuals(\n"
    end_marker = "\npub(crate) fn assemble_combined_action(\n"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise SystemExit("unexpected symmetry-residual function boundaries")

    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = '''fn relative_max_difference(
    left: &[f64],
    right: &[f64],
) -> Result<f64, F10CombinedActionError> {
    if left.is_empty() || left.len() != right.len() {
        return Err(F10CombinedActionError::InvalidInput);
    }
    let left_scale = left
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let right_scale = right
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let difference = left
        .iter()
        .zip(right)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0_f64, f64::max);
    let residual = difference / left_scale.max(right_scale).max(f64::MIN_POSITIVE);
    if residual.is_finite() {
        Ok(residual)
    } else {
        Err(F10CombinedActionError::NonFiniteOutput)
    }
}

fn symmetry_residuals(
    native_total: &[f64],
    order: usize,
) -> Result<(f64, f64), F10CombinedActionError> {
    if order == 0 || native_total.len() != SPECIES_COUNT * order {
        return Err(F10CombinedActionError::InvalidInput);
    }

    let mut charge_conjugation = 0.0_f64;
    for pair in 0..PAIR_COUNT {
        let particle_start = (2 * pair) * order;
        let antiparticle_start = particle_start + order;
        charge_conjugation = charge_conjugation.max(relative_max_difference(
            &native_total[particle_start..particle_start + order],
            &native_total[antiparticle_start..antiparticle_start + order],
        )?);
    }

    let mut mu_pair = Vec::with_capacity(order);
    let mut tau_pair = Vec::with_capacity(order);
    for node in 0..order {
        mu_pair.push(0.5 * (native_total[2 * order + node] + native_total[3 * order + node]));
        tau_pair.push(0.5 * (native_total[4 * order + node] + native_total[5 * order + node]));
    }
    let mu_tau = relative_max_difference(&mu_pair, &tau_pair)?;
    Ok((charge_conjugation, mu_tau))
}
'''
    SOURCE.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    return True


def patch_component_mutations(text: str) -> tuple[str, bool]:
    new_signature = "        let self_scale = maximum_absolute(&thermal.self_action.native);\n"
    if new_signature in text:
        return text, False

    start_marker = '        let expected_total = bit_array(&thermal_case["arrays"]["total_native"]);\n'
    end_marker = "    }\n\n    #[test]\n    fn component_failures_propagate_without_a_partial_result()"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise SystemExit("unexpected component-mutation test boundaries")

    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = '''        let self_scale = maximum_absolute(&thermal.self_action.native);
        let electron_scale = maximum_absolute(&thermal.electron_action.native);
        assert!(
            scaled_difference(
                &thermal.self_action.native,
                &thermal.native_total,
                electron_scale,
            ) > 0.5
        );
        assert!(
            scaled_difference(
                &thermal.electron_action.native,
                &thermal.native_total,
                self_scale,
            ) > 0.5
        );
        let wrong_sign: Vec<_> = thermal
            .self_action
            .native
            .iter()
            .zip(&thermal.electron_action.native)
            .map(|(self_value, electron_value)| self_value - electron_value)
            .collect();
        assert!(
            scaled_difference(
                &wrong_sign,
                &thermal.native_total,
                self_scale.max(electron_scale),
            ) > 0.5
        );
'''
    return text[:start] + replacement + text[end:], True


def patch_conditioned_cp_gate(text: str) -> tuple[str, bool]:
    helper_signature = "fn assert_conditioned_charge_conjugation_close(\n"
    changed = False
    if helper_signature not in text:
        marker = '''fn scaled_difference(actual: &[f64], expected: &[f64], scale: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
        / scale.max(f64::MIN_POSITIVE)
}
'''
        if text.count(marker) != 1:
            raise SystemExit("unexpected scaled_difference helper")
        helper = marker + '''
fn maximum_absolute_difference(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| (left_value - right_value).abs())
        .fold(0.0_f64, f64::max)
}

fn pair_charge_conjugation_residual(values: &[f64], order: usize, pair: usize) -> (f64, f64, f64) {
    assert!(order > 0);
    assert_eq!(values.len(), 6 * order);
    assert!(pair < 3);
    let particle_start = (2 * pair) * order;
    let antiparticle_start = particle_start + order;
    let particle = &values[particle_start..particle_start + order];
    let antiparticle = &values[antiparticle_start..antiparticle_start + order];
    let scale = maximum_absolute(particle)
        .max(maximum_absolute(antiparticle))
        .max(f64::MIN_POSITIVE);
    let difference = maximum_absolute_difference(particle, antiparticle);
    (difference / scale, difference, scale)
}

fn assert_conditioned_charge_conjugation_close(
    actual: &[f64],
    expected: &[f64],
    order: usize,
    stored_actual: f64,
    stored_expected: f64,
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), 6 * order);

    let mut recomputed_actual = 0.0_f64;
    let mut recomputed_expected = 0.0_f64;
    let mut maximum_propagated_bound = 0.0_f64;

    for pair in 0..3 {
        let particle_start = (2 * pair) * order;
        let antiparticle_start = particle_start + order;
        let actual_particle = &actual[particle_start..particle_start + order];
        let actual_antiparticle = &actual[antiparticle_start..antiparticle_start + order];
        let expected_particle = &expected[particle_start..particle_start + order];
        let expected_antiparticle = &expected[antiparticle_start..antiparticle_start + order];

        let (actual_residual, _actual_difference, actual_scale) =
            pair_charge_conjugation_residual(actual, order, pair);
        let (expected_residual, expected_difference, expected_scale) =
            pair_charge_conjugation_residual(expected, order, pair);
        recomputed_actual = recomputed_actual.max(actual_residual);
        recomputed_expected = recomputed_expected.max(expected_residual);

        let perturbation = maximum_absolute_difference(actual_particle, expected_particle)
            .max(maximum_absolute_difference(actual_antiparticle, expected_antiparticle));
        let propagated_bound = 2.0 * perturbation / actual_scale
            + expected_difference * perturbation / (actual_scale * expected_scale)
            + 262_144.0
                * f64::EPSILON
                * actual_residual.abs().max(expected_residual.abs()).max(1.0);
        maximum_propagated_bound = maximum_propagated_bound.max(propagated_bound);
        assert!(
            (actual_residual - expected_residual).abs() <= propagated_bound,
            "conditioned CP residual mismatch for pair {pair}: actual={actual_residual:.17e}, expected={expected_residual:.17e}, perturbation={perturbation:.17e}, propagated_bound={propagated_bound:.17e}"
        );
    }

    assert_eq!(stored_actual.to_bits(), recomputed_actual.to_bits());
    let fixture_roundoff = 64.0
        * f64::EPSILON
        * stored_expected.abs().max(recomputed_expected.abs()).max(1.0);
    assert!(
        (stored_expected - recomputed_expected).abs() <= fixture_roundoff,
        "frozen CP diagnostic does not match its frozen total array: stored={stored_expected:.17e}, recomputed={recomputed_expected:.17e}, allowed={fixture_roundoff:.17e}"
    );
    assert!(
        (stored_actual - stored_expected).abs() <= maximum_propagated_bound,
        "combined CP diagnostic exceeds the array-propagated condition bound: actual={stored_actual:.17e}, expected={stored_expected:.17e}, bound={maximum_propagated_bound:.17e}"
    );
}
'''
        text = text.replace(marker, helper)
        changed = True

    old_assertion = '''            assert_scalar_close(
                result.charge_conjugation_residual,
                bits(&diagnostics["charge_conjugation_residual"]),
                1.0,
                6.0e-9,
            );
'''
    new_assertion = '''            assert_conditioned_charge_conjugation_close(
                &result.native_total,
                &bit_array(&case["arrays"]["total_native"]),
                grid.order,
                result.charge_conjugation_residual,
                bits(&diagnostics["charge_conjugation_residual"]),
            );
'''
    if old_assertion in text:
        if text.count(old_assertion) != 1:
            raise SystemExit("unexpected charge-conjugation assertion count")
        text = text.replace(old_assertion, new_assertion)
        changed = True
    elif new_assertion not in text:
        raise SystemExit("neither old nor conditioned charge-conjugation gate is present")

    return text, changed


def patch_tests() -> bool:
    text = TESTS.read_text(encoding="utf-8")
    text, component_changed = patch_component_mutations(text)
    text, cp_changed = patch_conditioned_cp_gate(text)
    if component_changed or cp_changed:
        TESTS.write_text(text, encoding="utf-8")
    return component_changed or cp_changed


if __name__ == "__main__":
    changed = patch_source() | patch_tests()
    print("D-081R1D4 bounded definition repair:", "CHANGED" if changed else "NOOP")
