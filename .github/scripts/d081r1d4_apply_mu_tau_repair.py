#!/usr/bin/env python3
"""Apply the final bounded D-081R1D4 mu/tau metrology repair.

The frozen full-action arrays remain authoritative.  This script first applies
all earlier bounded definition repairs, then replaces only the ill-conditioned
native mu/tau scalar equality with an array-error-propagated consistency gate
and adds a distinct-state mu/tau swap-equivariance gate.  It changes no
collision coefficient, event catalogue, state, grid, quadrature, fixture,
array-parity tolerance, or physical acceptance threshold.
"""

from __future__ import annotations

import runpy
from pathlib import Path


BASE_REPAIR = Path(".github/scripts/d081r1d4_apply_definition_repair.py")
TESTS = Path("native/rabbit_cpu/src/f10_combined_action_tests.rs")


def apply_base_repairs() -> None:
    runpy.run_path(str(BASE_REPAIR), run_name="__main__")


def patch_conditioned_mu_tau_gate(text: str) -> tuple[str, bool]:
    helper_signature = "fn assert_conditioned_mu_tau_close(\n"
    changed = False

    if helper_signature not in text:
        marker = "\n#[cfg(test)]\nmod tests {\n"
        if text.count(marker) != 1:
            raise SystemExit("unexpected combined-action test module boundary")
        helper = r'''
fn pair_average(values: &[f64], order: usize, pair: usize) -> Vec<f64> {
    assert!(order > 0);
    assert_eq!(values.len(), 6 * order);
    assert!(pair < 3);
    let particle_start = (2 * pair) * order;
    let antiparticle_start = particle_start + order;
    (0..order)
        .map(|node| {
            0.5 * (values[particle_start + node] + values[antiparticle_start + node])
        })
        .collect()
}

fn mu_tau_residual_components(values: &[f64], order: usize) -> (f64, f64, f64) {
    let mu = pair_average(values, order, 1);
    let tau = pair_average(values, order, 2);
    let difference = maximum_absolute_difference(&mu, &tau);
    let scale = maximum_absolute(&mu)
        .max(maximum_absolute(&tau))
        .max(f64::MIN_POSITIVE);
    (difference / scale, difference, scale)
}

fn assert_conditioned_mu_tau_close(
    actual: &[f64],
    expected: &[f64],
    order: usize,
    stored_actual: f64,
    stored_expected: f64,
) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), 6 * order);

    let actual_mu = pair_average(actual, order, 1);
    let actual_tau = pair_average(actual, order, 2);
    let expected_mu = pair_average(expected, order, 1);
    let expected_tau = pair_average(expected, order, 2);
    let (actual_residual, _actual_difference, actual_scale) =
        mu_tau_residual_components(actual, order);
    let (expected_residual, expected_difference, expected_scale) =
        mu_tau_residual_components(expected, order);

    let mu_error = maximum_absolute_difference(&actual_mu, &expected_mu);
    let tau_error = maximum_absolute_difference(&actual_tau, &expected_tau);
    let numerator_perturbation = mu_error + tau_error;
    let scale_perturbation = mu_error.max(tau_error);
    let propagated_bound = numerator_perturbation / actual_scale
        + expected_difference * scale_perturbation / (actual_scale * expected_scale)
        + 262_144.0
            * f64::EPSILON
            * actual_residual.abs().max(expected_residual.abs()).max(1.0);

    assert_eq!(stored_actual.to_bits(), actual_residual.to_bits());
    let fixture_roundoff = 64.0
        * f64::EPSILON
        * stored_expected.abs().max(expected_residual.abs()).max(1.0);
    assert!(
        (stored_expected - expected_residual).abs() <= fixture_roundoff,
        "frozen mu/tau diagnostic does not match its frozen total array: stored={stored_expected:.17e}, recomputed={expected_residual:.17e}, allowed={fixture_roundoff:.17e}"
    );
    assert!(
        (actual_residual - expected_residual).abs() <= propagated_bound,
        "conditioned mu/tau residual exceeds the array-propagated bound: actual={actual_residual:.17e}, expected={expected_residual:.17e}, mu_error={mu_error:.17e}, tau_error={tau_error:.17e}, propagated_bound={propagated_bound:.17e}"
    );
}
'''
        text = text.replace(marker, "\n" + helper + marker)
        changed = True

    old_assertion = r'''            assert_scalar_close(
                result.mu_tau_residual,
                bits(&diagnostics["mu_tau_residual"]),
                1.0,
                6.0e-9,
            );
'''
    new_assertion = r'''            assert_conditioned_mu_tau_close(
                &result.native_total,
                &bit_array(&case["arrays"]["total_native"]),
                grid.order,
                result.mu_tau_residual,
                bits(&diagnostics["mu_tau_residual"]),
            );
'''
    if old_assertion in text:
        if text.count(old_assertion) != 1:
            raise SystemExit("unexpected mu/tau scalar assertion count")
        text = text.replace(old_assertion, new_assertion)
        changed = True
    elif new_assertion not in text:
        raise SystemExit("neither direct nor conditioned mu/tau gate is present")

    return text, changed


def patch_combined_swap_equivariance(text: str) -> tuple[str, bool]:
    signature = "        let swapped_combined = assemble_combined_action(\n"
    if signature in text:
        return text, False

    end_marker = "    }\n\n    #[test]\n    fn component_failures_propagate_without_a_partial_result()"
    if text.count(end_marker) != 1:
        raise SystemExit("unexpected combined component-test boundary")

    insertion = r'''

        let mut swapped_coordinates = bit_array(&split_case["pair_cloglog"]);
        for node in 0..grid.order {
            swapped_coordinates.swap(grid.order + node, 2 * grid.order + node);
        }
        let swapped_combined = assemble_combined_action(
            &grid,
            &swapped_coordinates,
            bits(&split_case["temperature_cm_bits"]),
            bits(&split_case["temperature_gamma_bits"]),
            F10CombinedActionConfig::default(),
        )
        .unwrap();
        let species_permutation = [0_usize, 1, 4, 5, 2, 3];
        let native_scale = maximum_absolute(&split.native_total);
        let modal_scale = maximum_absolute(&split.modal_total);
        for (observed_species, &reference_species) in species_permutation.iter().enumerate() {
            assert_hybrid_close(
                &swapped_combined.native_total
                    [observed_species * grid.order..(observed_species + 1) * grid.order],
                &split.native_total
                    [reference_species * grid.order..(reference_species + 1) * grid.order],
                native_scale,
                6.0e-9,
            );
            assert_hybrid_close(
                &swapped_combined.modal_total
                    [observed_species * grid.order..(observed_species + 1) * grid.order],
                &split.modal_total
                    [reference_species * grid.order..(reference_species + 1) * grid.order],
                modal_scale,
                6.0e-9,
            );
        }
        assert_scalar_close(
            swapped_combined.mu_tau_residual,
            split.mu_tau_residual,
            swapped_combined
                .mu_tau_residual
                .abs()
                .max(split.mu_tau_residual.abs())
                .max(f64::MIN_POSITIVE),
            6.0e-9,
        );
'''
    text = text.replace(end_marker, insertion + end_marker)
    return text, True


def main() -> None:
    apply_base_repairs()
    text = TESTS.read_text(encoding="utf-8")
    text, conditioned_changed = patch_conditioned_mu_tau_gate(text)
    text, swap_changed = patch_combined_swap_equivariance(text)
    if conditioned_changed or swap_changed:
        TESTS.write_text(text, encoding="utf-8")
    print(
        "D-081R1D4 bounded mu/tau repair:",
        "CHANGED" if conditioned_changed or swap_changed else "NOOP",
    )


if __name__ == "__main__":
    main()
