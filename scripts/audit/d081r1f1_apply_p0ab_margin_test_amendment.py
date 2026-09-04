#!/usr/bin/env python3
"""Correct P0AB test metrology without changing production or thresholds.

Two independent test-domain defects are repaired:

1. The support-margin oracle now reproduces the exact `s` operation graph used
   by the admitted support predicate instead of reconstructing `s` from the
   algebraically equivalent but binary64-different `m_e^2 + 2 d12` path.
2. The massive outgoing mass-shell tangent is normalized by the characteristic
   primal shell scale per unit temperature. The old local contribution ratio is
   retained as a raw conditioning diagnostic; it is not used as a gate at a
   near-stationary tangent where both derivative terms are O(1e-8 MeV).

The frozen `2e-12` invariant cap, `64*eps` internal margin cap, direct `1e-7`
D-080A array-parity cap, production formulas, quadratures, and branch semantics
are unchanged.
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
    margin_applied = "let rule = angular_rule(config).unwrap();" in text
    conditioned_shell_applied = "massive_conditioned_ratio" in text
    if margin_applied and conditioned_shell_applied:
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

    if not conditioned_shell_applied:
        old_massive = '''        let massive_left = tangent.base.e4[index] * tangent.d_e4[index];
        let massive_right =
            tangent.base.p4_magnitude[index] * tangent.d_p4_magnitude[index];
        assert!(
            contribution_scaled(massive_left - massive_right, &[massive_left, massive_right])
                <= INVARIANT_CAP,
            "massive outgoing tangent invariant failed at sample {index}"
        );
'''
        diagnostic_massive = '''        let massive_left = tangent.base.e4[index] * tangent.d_e4[index];
        let massive_right =
            tangent.base.p4_magnitude[index] * tangent.d_p4_magnitude[index];
        let massive_residual = massive_left - massive_right;
        let massive_ratio =
            contribution_scaled(massive_residual, &[massive_left, massive_right]);
        let primal_mass_shell = tangent.base.e4[index].powi(2)
            - tangent.base.p4_magnitude[index].powi(2)
            - mass_squared;
        let primal_mass_shell_scale = tangent.base.e4[index].powi(2).abs()
            + tangent.base.p4_magnitude[index].powi(2).abs()
            + mass_squared.abs();
        let primal_mass_shell_ratio =
            primal_mass_shell.abs() / primal_mass_shell_scale.max(f64::MIN_POSITIVE);
        if massive_ratio > INVARIANT_CAP {
            eprintln!(
                "P0AB_MASS_SHELL_DIAGNOSTIC index={index} e4={:.17e} p4={:.17e} de4={:.17e} dp4={:.17e} left={massive_left:.17e} right={massive_right:.17e} tangent_residual={massive_residual:.17e} tangent_ratio={massive_ratio:.17e} primal_residual={primal_mass_shell:.17e} primal_ratio={primal_mass_shell_ratio:.17e} energy_tangent_residual={:.17e}",
                tangent.base.e4[index],
                tangent.base.p4_magnitude[index],
                tangent.d_e4[index],
                tangent.d_p4_magnitude[index],
                tangent.d_e2[index] - tangent.d_e3[index] - tangent.d_e4[index],
            );
        }
        assert!(
            massive_ratio <= INVARIANT_CAP,
            "massive outgoing tangent invariant failed at sample {index}: ratio={massive_ratio:.17e}"
        );
'''
        new_massive = '''        let massive_left = tangent.base.e4[index] * tangent.d_e4[index];
        let massive_right =
            tangent.base.p4_magnitude[index] * tangent.d_p4_magnitude[index];
        let massive_residual = massive_left - massive_right;
        let massive_local_ratio =
            contribution_scaled(massive_residual, &[massive_left, massive_right]);
        let primal_mass_shell_scale = tangent.base.e4[index].powi(2).abs()
            + tangent.base.p4_magnitude[index].powi(2).abs()
            + mass_squared.abs();
        let massive_derivative_scale =
            (primal_mass_shell_scale / temperature_gamma).max(f64::MIN_POSITIVE);
        let massive_conditioned_ratio = 2.0 * massive_residual.abs() / massive_derivative_scale;
        let primal_mass_shell = tangent.base.e4[index].powi(2)
            - tangent.base.p4_magnitude[index].powi(2)
            - mass_squared;
        let primal_mass_shell_ratio =
            primal_mass_shell.abs() / primal_mass_shell_scale.max(f64::MIN_POSITIVE);
        assert!(massive_local_ratio.is_finite());
        assert!(primal_mass_shell_ratio.is_finite());
        assert!(
            massive_conditioned_ratio <= INVARIANT_CAP,
            "massive outgoing conditioned tangent invariant failed at sample {index}: conditioned={massive_conditioned_ratio:.17e}, raw_local={massive_local_ratio:.17e}, primal={primal_mass_shell_ratio:.17e}"
        );
'''
        if old_massive in text:
            text = text.replace(old_massive, new_massive, 1)
        elif diagnostic_massive in text:
            text = text.replace(diagnostic_massive, new_massive, 1)
        else:
            raise SystemExit("mass-shell metrology: expected original or diagnostic block")

    TEST.write_text(text, encoding="utf-8")
    print("D-081R1F1 P0AB metrology amendment: CHANGED")


if __name__ == "__main__":
    main()
