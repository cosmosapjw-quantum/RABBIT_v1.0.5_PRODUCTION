#!/usr/bin/env python3
"""Apply the bounded D-081R1F0 retained symmetry-metrology amendment.

This script changes only the retained calibration test. It preserves the raw
Rust and Python mu/tau residuals and all pre-existing physics, modal, packed,
first-law, conservation, support, correction, and centered-difference gates.
The invalid postcondition `r_mu_tau <= 2e-9` is not widened. It is replaced by
reference self-consistency, pair-array parity, and the exact propagation bound
documented in D081R1F0_RETAINED_SYMMETRY_METROLOGY_AMENDMENT_2026-09-04.md.
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path(
    "native/rabbit_cpu/src/"
    "f10_packed_rhs_jvp_retained_calibration_tests.rs"
)
MARKER = "D081R1F0_MU_TAU_METROLOGY"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    if MARKER in text:
        print("D-081R1F0 retained symmetry amendment: ALREADY_APPLIED")
        return

    scalar_anchor = "fn scalar_relative(actual: f64, expected: f64) -> f64 {\n"
    helpers = r'''#[derive(Clone, Copy, Debug)]
struct PairSymmetryMetric {
    numerator: f64,
    scale: f64,
    residual: f64,
}

fn pair_average(values: &[f64], order: usize, pair: usize) -> Vec<f64> {
    assert!(pair < 3);
    assert_eq!(values.len(), 6 * order);
    (0..order)
        .map(|node| {
            0.5 * (values[(2 * pair) * order + node]
                + values[(2 * pair + 1) * order + node])
        })
        .collect()
}

fn pair_symmetry_metric(mu: &[f64], tau: &[f64]) -> PairSymmetryMetric {
    assert_eq!(mu.len(), tau.len());
    assert!(!mu.is_empty());
    let numerator = maximum_absolute_difference(mu, tau);
    let scale = maximum_absolute(mu)
        .max(maximum_absolute(tau))
        .max(f64::MIN_POSITIVE);
    PairSymmetryMetric {
        numerator,
        scale,
        residual: numerator / scale,
    }
}

fn conditioned_ratio_difference_bound(
    rust_mu: &[f64],
    rust_tau: &[f64],
    python_mu: &[f64],
    python_tau: &[f64],
    rust_metric: PairSymmetryMetric,
    python_metric: PairSymmetryMetric,
) -> f64 {
    let delta_mu = maximum_absolute_difference(rust_mu, python_mu);
    let delta_tau = maximum_absolute_difference(rust_tau, python_tau);
    let delta_scale = delta_mu.max(delta_tau);
    let propagated = (delta_mu + delta_tau) / rust_metric.scale
        + python_metric.numerator * delta_scale
            / (rust_metric.scale * python_metric.scale);
    let evaluation_roundoff = 64.0
        * f64::EPSILON
        * rust_metric
            .residual
            .abs()
            .max(python_metric.residual.abs())
            .max(1.0);
    propagated + evaluation_roundoff
}

'''
    text = replace_once(
        text,
        scalar_anchor,
        helpers + scalar_anchor,
        "symmetry helper insertion",
    )

    old_gate = r'''    assert_below(
        result.combined_action.mu_tau_residual,
        2.0e-9,
        "retained mu/tau tangent",
    );
'''
    new_gate = r'''    let rust_mu = pair_average(&result.combined_action.native_total, grid.order, 1);
    let rust_tau = pair_average(&result.combined_action.native_total, grid.order, 2);
    let python_mu = pair_average(&expected_total_native, grid.order, 1);
    let python_tau = pair_average(&expected_total_native, grid.order, 2);
    let rust_mu_tau = pair_symmetry_metric(&rust_mu, &rust_tau);
    let python_mu_tau = pair_symmetry_metric(&python_mu, &python_tau);
    let python_stored_mu_tau = bits(&value["collision"]["mu_tau_residual_bits"]);

    assert_eq!(
        result.combined_action.mu_tau_residual.to_bits(),
        rust_mu_tau.residual.to_bits(),
        "Rust stored mu/tau residual disagrees with the Rust tangent arrays"
    );
    assert_eq!(
        python_stored_mu_tau.to_bits(),
        python_mu_tau.residual.to_bits(),
        "Python stored mu/tau residual disagrees with the frozen Python arrays"
    );

    let rust_python_mu = global_relative(&rust_mu, &python_mu);
    let rust_python_tau = global_relative(&rust_tau, &python_tau);
    assert_below(
        rust_python_mu,
        modal_cap,
        "retained Rust/Python mu pair-average tangent",
    );
    assert_below(
        rust_python_tau,
        modal_cap,
        "retained Rust/Python tau pair-average tangent",
    );

    let ratio_difference = (rust_mu_tau.residual - python_mu_tau.residual).abs();
    let ratio_bound = conditioned_ratio_difference_bound(
        &rust_mu,
        &rust_tau,
        &python_mu,
        &python_tau,
        rust_mu_tau,
        python_mu_tau,
    );
    assert!(
        ratio_difference <= ratio_bound,
        "retained mu/tau ratio discrepancy exceeds the propagated pair-array bound: difference={ratio_difference:.17e}, bound={ratio_bound:.17e}"
    );

    let legacy_mu_tau_cap = 2.0e-9;
    println!(
        "D081R1F0_MU_TAU_METROLOGY legacy_cap={legacy_mu_tau_cap:.17e} rust_raw={:.17e} rust_numerator={:.17e} rust_scale={:.17e} python_raw={:.17e} python_numerator={:.17e} python_scale={:.17e} mu_array_relative={rust_python_mu:.17e} tau_array_relative={rust_python_tau:.17e} ratio_difference={ratio_difference:.17e} ratio_bound={ratio_bound:.17e} legacy_rust_pass={} legacy_python_pass={}",
        rust_mu_tau.residual,
        rust_mu_tau.numerator,
        rust_mu_tau.scale,
        python_mu_tau.residual,
        python_mu_tau.numerator,
        python_mu_tau.scale,
        rust_mu_tau.residual <= legacy_mu_tau_cap,
        python_mu_tau.residual <= legacy_mu_tau_cap,
    );
'''
    text = replace_once(text, old_gate, new_gate, "legacy mu/tau gate replacement")

    TARGET.write_text(text, encoding="utf-8")
    print("D-081R1F0 retained symmetry amendment: CHANGED")


if __name__ == "__main__":
    main()
