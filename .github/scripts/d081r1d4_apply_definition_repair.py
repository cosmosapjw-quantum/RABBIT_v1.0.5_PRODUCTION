#!/usr/bin/env python3
"""Apply the bounded D-081R1D4 diagnostic-definition repair.

This script changes no collision coefficient, state, grid, quadrature,
fixture, tolerance, or acceptance threshold.  It only aligns two derived
metrics with the frozen Python comparator and makes the component-mutation
negative control scale by the omitted component itself.
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


def patch_tests() -> bool:
    text = TESTS.read_text(encoding="utf-8")
    new_signature = "        let self_scale = maximum_absolute(&thermal.self_action.native);\n"
    if new_signature in text:
        return False

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
    TESTS.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
    return True


if __name__ == "__main__":
    changed = patch_source() | patch_tests()
    print("D-081R1D4 bounded definition repair:", "CHANGED" if changed else "NOOP")
