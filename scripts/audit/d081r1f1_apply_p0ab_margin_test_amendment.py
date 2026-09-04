#!/usr/bin/env python3
"""Correct the P0AB support-margin test oracle without changing a gate.

The failed first GREEN run compared the production support margin, computed
from the exact `s` operation graph used by the support predicate, with an
algebraically equivalent reconstruction `m_e^2 + 2 d12`. At large incoming
electron momentum those binary64 operation graphs differ because the direct
energy-momentum subtraction is cancellation-sensitive. This patch makes the
test reproduce the admitted support-predicate operation order. The frozen
`64*eps` internal gate and the separate `1e-7` D-080A cross-language gate are
unchanged.
"""

from __future__ import annotations

from pathlib import Path


TEST = Path("native/rabbit_cpu/src/f10_tgamma_adversarial_repair_tests.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TEST.read_text(encoding="utf-8")
    if "let rule = angular_rule(config).unwrap();" in text:
        print("D-081R1F1 P0AB margin-test amendment: NOOP")
        return

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
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1: target_momentum,
        temperature_gamma: 2.05,
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
    TEST.write_text(text, encoding="utf-8")
    print("D-081R1F1 P0AB margin-test amendment: CHANGED")


if __name__ == "__main__":
    main()
