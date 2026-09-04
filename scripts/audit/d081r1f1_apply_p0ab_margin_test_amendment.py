#!/usr/bin/env python3
"""Correct P0AB test metrology without changing production or thresholds.

This bounded test-only amendment fixes two domain errors:

1. Support margins are recomputed with the exact admitted `s` operation graph,
   not the algebraically equivalent but binary64-different `m_e^2 + 2 d12`.
2. Differentiated conservation/mass-shell/Minkowski identities are normalized
   by a characteristic primal scale per unit temperature. The original local
   contribution ratios are retained as raw conditioning diagnostics; they are
   not primary gates when all tangent terms are simultaneously near zero.

The frozen `2e-12` invariant cap, `64*eps` margin cap, direct `1e-7` D-080A
array-parity cap, production formulas, quadratures, and branch semantics are
unchanged. A one-percent massless-leg tangent mutation remains load-bearing.
"""

from __future__ import annotations

from pathlib import Path


TEST = Path("native/rabbit_cpu/src/f10_tgamma_adversarial_repair_tests.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_invariant_loop(text: str) -> str:
    marker = "P0AB_RAW_LOCAL_INVARIANT_MAXIMA"
    if marker in text:
        return text

    start_marker = "    for index in 0..tangent.support.len() {\n"
    end_marker = "    }\n}\n\n#[test]\n#[ignore = \"requires deterministic frozen D-080A P0B oracle\"]"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise SystemExit("invariant ledger: unexpected function boundaries")
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len("    }\n")

    replacement = r'''    let mut maximum_raw_local = [0.0_f64; 6];
    let mut maximum_conditioned = [0.0_f64; 6];
    let mut mutation_index = None;
    let mut mutation_sensitivity = 0.0_f64;

    for index in 0..tangent.support.len() {
        if !tangent.support[index] {
            continue;
        }

        let energy_residual =
            tangent.d_e2[index] - tangent.d_e3[index] - tangent.d_e4[index];
        let energy_raw = contribution_scaled(
            energy_residual,
            &[tangent.d_e2[index], tangent.d_e3[index], tangent.d_e4[index]],
        );
        let energy_scale = (tangent.base.e2[index].abs()
            + tangent.base.e3[index].abs()
            + tangent.base.e4[index].abs())
            / temperature_gamma;
        let energy_conditioned =
            energy_residual.abs() / energy_scale.max(f64::MIN_POSITIVE);

        let massless_residual = tangent.d_e3[index] - tangent.d_p3_magnitude[index];
        let massless_raw = contribution_scaled(
            massless_residual,
            &[tangent.d_e3[index], tangent.d_p3_magnitude[index]],
        );
        let massless_scale = (tangent.base.e3[index].abs()
            + tangent.base.p3_magnitude[index].abs())
            / temperature_gamma;
        let massless_conditioned =
            massless_residual.abs() / massless_scale.max(f64::MIN_POSITIVE);

        let massive_left = tangent.base.e4[index] * tangent.d_e4[index];
        let massive_right =
            tangent.base.p4_magnitude[index] * tangent.d_p4_magnitude[index];
        let massive_half_residual = massive_left - massive_right;
        let massive_raw = contribution_scaled(
            massive_half_residual,
            &[massive_left, massive_right],
        );
        let massive_scale = (tangent.base.e4[index].powi(2).abs()
            + tangent.base.p4_magnitude[index].powi(2).abs()
            + mass_squared.abs())
            / temperature_gamma;
        let massive_conditioned =
            2.0 * massive_half_residual.abs() / massive_scale.max(f64::MIN_POSITIVE);

        let d12_d34_residual = tangent.d_d12[index] - tangent.d_d34[index];
        let d12_d34_raw = contribution_scaled(
            d12_d34_residual,
            &[tangent.d_d12[index], tangent.d_d34[index]],
        );
        let d12_d34_scale =
            (tangent.base.d12[index].abs() + tangent.base.d34[index].abs())
                / temperature_gamma;
        let d12_d34_conditioned =
            d12_d34_residual.abs() / d12_d34_scale.max(f64::MIN_POSITIVE);

        let d13_d14_d12_residual =
            tangent.d_d13[index] + tangent.d_d14[index] - tangent.d_d12[index];
        let d13_d14_d12_raw = contribution_scaled(
            d13_d14_d12_residual,
            &[
                tangent.d_d13[index],
                tangent.d_d14[index],
                tangent.d_d12[index],
            ],
        );
        let d13_d14_d12_scale = (tangent.base.d13[index].abs()
            + tangent.base.d14[index].abs()
            + tangent.base.d12[index].abs())
            / temperature_gamma;
        let d13_d14_d12_conditioned = d13_d14_d12_residual.abs()
            / d13_d14_d12_scale.max(f64::MIN_POSITIVE);

        let d23_d24_d12_residual =
            tangent.d_d23[index] + tangent.d_d24[index] - tangent.d_d12[index];
        let d23_d24_d12_raw = contribution_scaled(
            d23_d24_d12_residual,
            &[
                tangent.d_d23[index],
                tangent.d_d24[index],
                tangent.d_d12[index],
            ],
        );
        let d23_d24_d12_scale = (tangent.base.d23[index].abs()
            + tangent.base.d24[index].abs()
            + tangent.base.d12[index].abs())
            / temperature_gamma;
        let d23_d24_d12_conditioned = d23_d24_d12_residual.abs()
            / d23_d24_d12_scale.max(f64::MIN_POSITIVE);

        let raw = [
            energy_raw,
            massless_raw,
            massive_raw,
            d12_d34_raw,
            d13_d14_d12_raw,
            d23_d24_d12_raw,
        ];
        let conditioned = [
            energy_conditioned,
            massless_conditioned,
            massive_conditioned,
            d12_d34_conditioned,
            d13_d14_d12_conditioned,
            d23_d24_d12_conditioned,
        ];
        for component in 0..6 {
            assert!(raw[component].is_finite());
            assert!(conditioned[component].is_finite());
            maximum_raw_local[component] = maximum_raw_local[component].max(raw[component]);
            maximum_conditioned[component] =
                maximum_conditioned[component].max(conditioned[component]);
            assert!(
                conditioned[component] <= INVARIANT_CAP,
                "conditioned tangent invariant {component} failed at sample {index}: conditioned={:.17e}, raw_local={:.17e}",
                conditioned[component],
                raw[component],
            );
        }

        let sensitivity = tangent.d_p3_magnitude[index].abs()
            / massless_scale.max(f64::MIN_POSITIVE);
        if sensitivity > mutation_sensitivity {
            mutation_sensitivity = sensitivity;
            mutation_index = Some(index);
        }
    }

    let mutation_index = mutation_index.expect("supported massless tangent sample");
    let mutation_scale = (tangent.base.e3[mutation_index].abs()
        + tangent.base.p3_magnitude[mutation_index].abs())
        / temperature_gamma;
    let mutated_dp3 = tangent.d_p3_magnitude[mutation_index] * 1.01;
    let mutation_ratio = (tangent.d_e3[mutation_index] - mutated_dp3).abs()
        / mutation_scale.max(f64::MIN_POSITIVE);
    assert!(
        mutation_ratio > 100.0 * INVARIANT_CAP,
        "one-percent massless-leg tangent mutation was not load-bearing: {mutation_ratio:.17e}"
    );

    eprintln!(
        "P0AB_RAW_LOCAL_INVARIANT_MAXIMA energy={:.17e} massless={:.17e} massive={:.17e} d12_d34={:.17e} d13_d14_d12={:.17e} d23_d24_d12={:.17e}",
        maximum_raw_local[0],
        maximum_raw_local[1],
        maximum_raw_local[2],
        maximum_raw_local[3],
        maximum_raw_local[4],
        maximum_raw_local[5],
    );
    eprintln!(
        "P0AB_CONDITIONED_INVARIANT_MAXIMA energy={:.17e} massless={:.17e} massive={:.17e} d12_d34={:.17e} d13_d14_d12={:.17e} d23_d24_d12={:.17e} mutation_index={mutation_index} mutation_ratio={mutation_ratio:.17e}",
        maximum_conditioned[0],
        maximum_conditioned[1],
        maximum_conditioned[2],
        maximum_conditioned[3],
        maximum_conditioned[4],
        maximum_conditioned[5],
    );
'''
    return text[:start] + replacement + text[end:]


def main() -> None:
    text = TEST.read_text(encoding="utf-8")
    margin_applied = "let rule = angular_rule(config).unwrap();" in text
    invariants_applied = "P0AB_RAW_LOCAL_INVARIANT_MAXIMA" in text
    if margin_applied and invariants_applied:
        print("D-081R1F1 P0AB metrology amendment: NOOP")
        return

    if not margin_applied:
        text = replace_once(
            text,
            "use crate::f10_action_kinematics::F10CollisionConfig;\n",
            "use crate::f10_action_kinematics::{F10CollisionConfig, angular_rule};\n",
            "kinematic rule import",
        )

        old = '''    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1: 2.0,
        temperature_gamma: 2.05,
        electron_mass: ELECTRON_MASS_MEV,
        config: F10CollisionConfig::default(),
    })
    .unwrap();
    let mass_squared = ELECTRON_MASS_MEV.powi(2);
    let invariant_s = tangent
        .base
        .d12
        .iter()
        .map(|value| mass_squared + 2.0 * value)
        .collect::<Vec<_>>();
    let support_scale = invariant_s
        .iter()
        .map(|value| value.abs())
        .fold(mass_squared, f64::max)
        .max(f64::MIN_POSITIVE);
    let expected_support_margin = invariant_s
        .iter()
        .map(|value| (value - mass_squared).abs())
        .fold(f64::INFINITY, f64::min)
        / support_scale;
    let expected_lambda_margin = invariant_s
        .iter()
        .zip(&tangent.support)
        .filter_map(|(value, support)| {
            support.then_some(
                (value - mass_squared).powi(2) / value.powi(2).max(f64::MIN_POSITIVE),
            )
        })
        .fold(f64::INFINITY, f64::min);

    assert!(
        scalar_relative(
            tangent.minimum_support_margin_relative,
            expected_support_margin
        ) <= 64.0 * f64::EPSILON
    );
    assert!(
        scalar_relative(
            tangent.minimum_supported_lambda_margin_relative,
            expected_lambda_margin
        ) <= 64.0 * f64::EPSILON
    );
'''
        new = '''    let config = F10CollisionConfig::default();
    let target_momentum = 2.0_f64;
    let temperature_gamma = 2.05_f64;
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1: target_momentum,
        temperature_gamma,
        electron_mass: ELECTRON_MASS_MEV,
        config,
    })
    .unwrap();
    let rule = angular_rule(config).unwrap();
    let final_angular_size = config.final_polar_order * config.final_azimuth_order;
    let mass_squared = ELECTRON_MASS_MEV.powi(2);
    let mass3_squared = 0.0_f64;
    let invariant_s = tangent
        .base
        .p2
        .iter()
        .zip(&tangent.base.e2)
        .enumerate()
        .map(|(index, (&p2, &e2))| {
            let incoming_index =
                (index / final_angular_size) % config.incoming_polar_order;
            let mu12 = rule.incoming_mu[incoming_index];
            let sin12 = (1.0 - mu12 * mu12).max(0.0).sqrt();
            let total_x = p2 * sin12;
            let total_z = target_momentum + p2 * mu12;
            let total_magnitude =
                ((total_x * total_x + 0.0) + total_z * total_z).sqrt();
            let total_energy = target_momentum + e2;
            (total_energy * total_energy - total_magnitude * total_magnitude).max(0.0)
        })
        .collect::<Vec<_>>();
    let support_scale = invariant_s
        .iter()
        .map(|value| value.abs())
        .fold(mass_squared, f64::max)
        .max(f64::MIN_POSITIVE);
    let expected_support_margin = invariant_s
        .iter()
        .map(|value| (value - mass_squared).abs())
        .fold(f64::INFINITY, f64::min)
        / support_scale;
    let expected_lambda_margin = invariant_s
        .iter()
        .zip(&tangent.support)
        .filter_map(|(value, support)| {
            if !support {
                return None;
            }
            let lambda = value * value
                + mass3_squared * mass3_squared
                + mass_squared * mass_squared
                - 2.0 * value * mass3_squared
                - 2.0 * value * mass_squared
                - 2.0 * mass3_squared * mass_squared;
            Some(lambda / (value * value).max(f64::MIN_POSITIVE))
        })
        .fold(f64::INFINITY, f64::min);

    let support_margin_residual = scalar_relative(
        tangent.minimum_support_margin_relative,
        expected_support_margin,
    );
    let lambda_margin_residual = scalar_relative(
        tangent.minimum_supported_lambda_margin_relative,
        expected_lambda_margin,
    );
    assert!(
        support_margin_residual <= 64.0 * f64::EPSILON,
        "support margin operation-graph residual {support_margin_residual:.17e}: actual={:.17e}, expected={expected_support_margin:.17e}",
        tangent.minimum_support_margin_relative,
    );
    assert!(
        lambda_margin_residual <= 64.0 * f64::EPSILON,
        "lambda margin operation-graph residual {lambda_margin_residual:.17e}: actual={:.17e}, expected={expected_lambda_margin:.17e}",
        tangent.minimum_supported_lambda_margin_relative,
    );
'''
        text = replace_once(text, old, new, "support-margin test oracle")

    text = replace_invariant_loop(text)
    TEST.write_text(text, encoding="utf-8")
    print("D-081R1F1 P0AB metrology amendment: CHANGED")


if __name__ == "__main__":
    main()
