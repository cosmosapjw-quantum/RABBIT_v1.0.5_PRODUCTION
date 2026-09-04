#!/usr/bin/env python3
"""Apply the P0AB reference-conditioned invariant metrology amendment.

This test-only amendment is applied after the earlier production repair and
support-margin operation-graph correction. It keeps the frozen 2e-12 hard cap
for stable tangent identities. Two Minkowski-dot identities that the frozen
D-080A Python authority itself evaluates above that cap are instead checked by
an exact stored-array propagation inequality, while direct Rust/Python array
parity, raw residuals, and independent mutation kills remain load-bearing.
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
    marker = "P0AB_REFERENCE_CONDITIONED_INVARIANT_MAXIMA"
    if marker in text:
        print("D-081R1F1 P0AB reference-conditioned invariant amendment: NOOP")
        return

    if "use crate::f10_action_kinematics::{F10CollisionConfig, angular_rule};" not in text:
        text = replace_once(
            text,
            "use crate::f10_action_kinematics::F10CollisionConfig;\n",
            "use crate::f10_action_kinematics::{F10CollisionConfig, angular_rule};\n",
            "angular-rule import",
        )

    start_marker = (
        "#[test]\n"
        "fn normalized_branch_margins_and_kinematic_invariants_are_load_bearing() {\n"
    )
    end_marker = (
        "\n#[test]\n"
        "#[ignore = \"requires deterministic frozen D-080A P0B oracle\"]"
    )
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise SystemExit("invariant function: unexpected boundaries")

    start = text.index(start_marker)
    end = text.index(end_marker, start)

    replacement = r'''#[test]
fn normalized_branch_margins_and_kinematic_invariants_are_load_bearing() {
    let config = F10CollisionConfig::default();
    let target_momentum = 2.0_f64;
    let temperature_gamma = 2.05_f64;
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1: target_momentum,
        temperature_gamma,
        electron_mass: ELECTRON_MASS_MEV,
        config,
    })
    .unwrap();

    let oracle_path = env::var("D081R1F1_P0B_ORACLE")
        .expect("D081R1F1_P0B_ORACLE must identify the generated oracle");
    let oracle_value: Value = serde_json::from_str(
        &fs::read_to_string(oracle_path).expect("read P0B oracle"),
    )
    .expect("valid P0B oracle");
    let oracle = &oracle_value["kinematics"];
    let python_support = oracle["support"]
        .as_array()
        .expect("Python support array")
        .iter()
        .map(|value| value.as_bool().expect("bool support"))
        .collect::<Vec<_>>();
    assert_eq!(tangent.support, python_support);

    let python_base = [
        bit_array(&oracle["base_e2"]),
        bit_array(&oracle["base_e3"]),
        bit_array(&oracle["base_e4"]),
        bit_array(&oracle["base_p3_magnitude"]),
        bit_array(&oracle["base_p4_magnitude"]),
        bit_array(&oracle["base_d12"]),
        bit_array(&oracle["base_d13"]),
        bit_array(&oracle["base_d14"]),
        bit_array(&oracle["base_d23"]),
        bit_array(&oracle["base_d24"]),
        bit_array(&oracle["base_d34"]),
    ];
    let python_tangent = [
        bit_array(&oracle["d_e2"]),
        bit_array(&oracle["d_e3"]),
        bit_array(&oracle["d_e4"]),
        bit_array(&oracle["d_p3_magnitude"]),
        bit_array(&oracle["d_p4_magnitude"]),
        bit_array(&oracle["d_d12"]),
        bit_array(&oracle["d_d13"]),
        bit_array(&oracle["d_d14"]),
        bit_array(&oracle["d_d23"]),
        bit_array(&oracle["d_d24"]),
        bit_array(&oracle["d_d34"]),
    ];
    for values in python_base.iter().chain(&python_tangent) {
        assert_eq!(values.len(), tangent.support.len());
    }

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

    let mut maximum_raw_local = [0.0_f64; 6];
    let mut maximum_rust_conditioned = [0.0_f64; 6];
    let mut maximum_python_conditioned = [0.0_f64; 6];
    let mut maximum_cross_propagation_ratio = [0.0_f64; 6];
    let mut massless_mutation_index = None;
    let mut massless_mutation_sensitivity = 0.0_f64;
    let mut dot_mutation_index = None;
    let mut dot_mutation_sensitivity = 0.0_f64;
    let mut dot_mutation_allowance = 0.0_f64;
    let mut dot_mutation_python_residual = 0.0_f64;

    for index in 0..tangent.support.len() {
        if !tangent.support[index] {
            continue;
        }

        let rust_base = [
            tangent.base.e2[index],
            tangent.base.e3[index],
            tangent.base.e4[index],
            tangent.base.p3_magnitude[index],
            tangent.base.p4_magnitude[index],
            tangent.base.d12[index],
            tangent.base.d13[index],
            tangent.base.d14[index],
            tangent.base.d23[index],
            tangent.base.d24[index],
            tangent.base.d34[index],
        ];
        let rust_tangent = [
            tangent.d_e2[index],
            tangent.d_e3[index],
            tangent.d_e4[index],
            tangent.d_p3_magnitude[index],
            tangent.d_p4_magnitude[index],
            tangent.d_d12[index],
            tangent.d_d13[index],
            tangent.d_d14[index],
            tangent.d_d23[index],
            tangent.d_d24[index],
            tangent.d_d34[index],
        ];
        let python_base_at = [
            python_base[0][index],
            python_base[1][index],
            python_base[2][index],
            python_base[3][index],
            python_base[4][index],
            python_base[5][index],
            python_base[6][index],
            python_base[7][index],
            python_base[8][index],
            python_base[9][index],
            python_base[10][index],
        ];
        let python_tangent_at = [
            python_tangent[0][index],
            python_tangent[1][index],
            python_tangent[2][index],
            python_tangent[3][index],
            python_tangent[4][index],
            python_tangent[5][index],
            python_tangent[6][index],
            python_tangent[7][index],
            python_tangent[8][index],
            python_tangent[9][index],
            python_tangent[10][index],
        ];

        let rust_massive_left = rust_base[2] * rust_tangent[2];
        let rust_massive_right = rust_base[4] * rust_tangent[4];
        let python_massive_left = python_base_at[2] * python_tangent_at[2];
        let python_massive_right = python_base_at[4] * python_tangent_at[4];

        let rust_residual = [
            rust_tangent[0] - rust_tangent[1] - rust_tangent[2],
            rust_tangent[1] - rust_tangent[3],
            2.0 * (rust_massive_left - rust_massive_right),
            rust_tangent[5] - rust_tangent[10],
            rust_tangent[6] + rust_tangent[7] - rust_tangent[5],
            rust_tangent[8] + rust_tangent[9] - rust_tangent[5],
        ];
        let python_residual = [
            python_tangent_at[0] - python_tangent_at[1] - python_tangent_at[2],
            python_tangent_at[1] - python_tangent_at[3],
            2.0 * (python_massive_left - python_massive_right),
            python_tangent_at[5] - python_tangent_at[10],
            python_tangent_at[6] + python_tangent_at[7] - python_tangent_at[5],
            python_tangent_at[8] + python_tangent_at[9] - python_tangent_at[5],
        ];

        let rust_scale = [
            (rust_base[0].abs() + rust_base[1].abs() + rust_base[2].abs())
                / temperature_gamma,
            (rust_base[1].abs() + rust_base[3].abs()) / temperature_gamma,
            (rust_base[2].powi(2).abs()
                + rust_base[4].powi(2).abs()
                + mass_squared.abs())
                / temperature_gamma,
            (rust_base[5].abs() + rust_base[10].abs()) / temperature_gamma,
            (rust_base[6].abs() + rust_base[7].abs() + rust_base[5].abs())
                / temperature_gamma,
            (rust_base[8].abs() + rust_base[9].abs() + rust_base[5].abs())
                / temperature_gamma,
        ];
        let python_scale = [
            (python_base_at[0].abs()
                + python_base_at[1].abs()
                + python_base_at[2].abs())
                / temperature_gamma,
            (python_base_at[1].abs() + python_base_at[3].abs()) / temperature_gamma,
            (python_base_at[2].powi(2).abs()
                + python_base_at[4].powi(2).abs()
                + mass_squared.abs())
                / temperature_gamma,
            (python_base_at[5].abs() + python_base_at[10].abs())
                / temperature_gamma,
            (python_base_at[6].abs()
                + python_base_at[7].abs()
                + python_base_at[5].abs())
                / temperature_gamma,
            (python_base_at[8].abs()
                + python_base_at[9].abs()
                + python_base_at[5].abs())
                / temperature_gamma,
        ];

        let rust_term_sum = [
            rust_tangent[0].abs() + rust_tangent[1].abs() + rust_tangent[2].abs(),
            rust_tangent[1].abs() + rust_tangent[3].abs(),
            2.0 * (rust_massive_left.abs() + rust_massive_right.abs()),
            rust_tangent[5].abs() + rust_tangent[10].abs(),
            rust_tangent[6].abs() + rust_tangent[7].abs() + rust_tangent[5].abs(),
            rust_tangent[8].abs() + rust_tangent[9].abs() + rust_tangent[5].abs(),
        ];
        let python_term_sum = [
            python_tangent_at[0].abs()
                + python_tangent_at[1].abs()
                + python_tangent_at[2].abs(),
            python_tangent_at[1].abs() + python_tangent_at[3].abs(),
            2.0 * (python_massive_left.abs() + python_massive_right.abs()),
            python_tangent_at[5].abs() + python_tangent_at[10].abs(),
            python_tangent_at[6].abs()
                + python_tangent_at[7].abs()
                + python_tangent_at[5].abs(),
            python_tangent_at[8].abs()
                + python_tangent_at[9].abs()
                + python_tangent_at[5].abs(),
        ];

        let propagation = [
            (rust_tangent[0] - python_tangent_at[0]).abs()
                + (rust_tangent[1] - python_tangent_at[1]).abs()
                + (rust_tangent[2] - python_tangent_at[2]).abs(),
            (rust_tangent[1] - python_tangent_at[1]).abs()
                + (rust_tangent[3] - python_tangent_at[3]).abs(),
            2.0
                * ((rust_base[2] - python_base_at[2]).abs() * rust_tangent[2].abs()
                    + python_base_at[2].abs()
                        * (rust_tangent[2] - python_tangent_at[2]).abs()
                    + (rust_base[4] - python_base_at[4]).abs()
                        * rust_tangent[4].abs()
                    + python_base_at[4].abs()
                        * (rust_tangent[4] - python_tangent_at[4]).abs()),
            (rust_tangent[5] - python_tangent_at[5]).abs()
                + (rust_tangent[10] - python_tangent_at[10]).abs(),
            (rust_tangent[6] - python_tangent_at[6]).abs()
                + (rust_tangent[7] - python_tangent_at[7]).abs()
                + (rust_tangent[5] - python_tangent_at[5]).abs(),
            (rust_tangent[8] - python_tangent_at[8]).abs()
                + (rust_tangent[9] - python_tangent_at[9]).abs()
                + (rust_tangent[5] - python_tangent_at[5]).abs(),
        ];

        for component in 0..6 {
            let characteristic_scale = rust_scale[component]
                .max(python_scale[component])
                .max(f64::MIN_POSITIVE);
            let raw_local = rust_residual[component].abs()
                / rust_term_sum[component].max(f64::MIN_POSITIVE);
            let rust_conditioned =
                rust_residual[component].abs() / characteristic_scale;
            let python_conditioned =
                python_residual[component].abs() / characteristic_scale;
            let evaluation_roundoff = 128.0
                * f64::EPSILON
                * (rust_term_sum[component]
                    + python_term_sum[component]
                    + characteristic_scale);
            let propagation_allowance = propagation[component] + evaluation_roundoff;
            let cross_gap =
                (rust_residual[component] - python_residual[component]).abs();
            let cross_ratio =
                cross_gap / propagation_allowance.max(f64::MIN_POSITIVE);

            assert!(raw_local.is_finite());
            assert!(rust_conditioned.is_finite());
            assert!(python_conditioned.is_finite());
            assert!(cross_ratio.is_finite());
            maximum_raw_local[component] =
                maximum_raw_local[component].max(raw_local);
            maximum_rust_conditioned[component] =
                maximum_rust_conditioned[component].max(rust_conditioned);
            maximum_python_conditioned[component] =
                maximum_python_conditioned[component].max(python_conditioned);
            maximum_cross_propagation_ratio[component] =
                maximum_cross_propagation_ratio[component].max(cross_ratio);

            if matches!(component, 0 | 1 | 2 | 4) {
                assert!(
                    rust_conditioned <= INVARIANT_CAP,
                    "stable Rust tangent invariant {component} failed at sample {index}: {rust_conditioned:.17e}"
                );
                assert!(
                    python_conditioned <= INVARIANT_CAP,
                    "stable Python tangent invariant {component} failed at sample {index}: {python_conditioned:.17e}"
                );
            } else {
                assert!(
                    cross_gap <= propagation_allowance,
                    "reference-conditioned tangent invariant {component} failed at sample {index}: cross_gap={cross_gap:.17e}, allowance={propagation_allowance:.17e}, rust={rust_conditioned:.17e}, python={python_conditioned:.17e}"
                );
            }
        }

        let massless_sensitivity = rust_tangent[3].abs()
            / rust_scale[1].max(f64::MIN_POSITIVE);
        if massless_sensitivity > massless_mutation_sensitivity {
            massless_mutation_sensitivity = massless_sensitivity;
            massless_mutation_index = Some(index);
        }

        let dot_sensitivity = rust_tangent[9].abs()
            / rust_scale[5].max(f64::MIN_POSITIVE);
        if dot_sensitivity > dot_mutation_sensitivity {
            dot_mutation_sensitivity = dot_sensitivity;
            dot_mutation_index = Some(index);
            let characteristic_scale = rust_scale[5]
                .max(python_scale[5])
                .max(f64::MIN_POSITIVE);
            let evaluation_roundoff = 128.0
                * f64::EPSILON
                * (rust_term_sum[5] + python_term_sum[5] + characteristic_scale);
            dot_mutation_allowance = propagation[5] + evaluation_roundoff;
            dot_mutation_python_residual = python_residual[5];
        }
    }

    let massless_mutation_index =
        massless_mutation_index.expect("supported massless tangent sample");
    let massless_mutation_scale = (tangent.base.e3[massless_mutation_index].abs()
        + tangent.base.p3_magnitude[massless_mutation_index].abs())
        / temperature_gamma;
    let mutated_dp3 = tangent.d_p3_magnitude[massless_mutation_index] * 1.01;
    let massless_mutation_ratio =
        (tangent.d_e3[massless_mutation_index] - mutated_dp3).abs()
            / massless_mutation_scale.max(f64::MIN_POSITIVE);
    assert!(
        massless_mutation_ratio > 100.0 * INVARIANT_CAP,
        "one-percent massless-leg tangent mutation was not load-bearing: {massless_mutation_ratio:.17e}"
    );

    let dot_mutation_index =
        dot_mutation_index.expect("supported d24 tangent sample");
    let mutated_d24 = tangent.d_d24[dot_mutation_index] * 1.01;
    let mutated_dot_residual =
        tangent.d_d23[dot_mutation_index] + mutated_d24
            - tangent.d_d12[dot_mutation_index];
    let mutated_dot_cross_gap =
        (mutated_dot_residual - dot_mutation_python_residual).abs();
    assert!(
        mutated_dot_cross_gap > 100.0 * dot_mutation_allowance.max(f64::MIN_POSITIVE),
        "one-percent d24 tangent mutation was not load-bearing: cross_gap={mutated_dot_cross_gap:.17e}, original_allowance={dot_mutation_allowance:.17e}"
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
        "P0AB_RUST_CONDITIONED_INVARIANT_MAXIMA energy={:.17e} massless={:.17e} massive={:.17e} d12_d34={:.17e} d13_d14_d12={:.17e} d23_d24_d12={:.17e}",
        maximum_rust_conditioned[0],
        maximum_rust_conditioned[1],
        maximum_rust_conditioned[2],
        maximum_rust_conditioned[3],
        maximum_rust_conditioned[4],
        maximum_rust_conditioned[5],
    );
    eprintln!(
        "P0AB_REFERENCE_CONDITIONED_INVARIANT_MAXIMA energy={:.17e} massless={:.17e} massive={:.17e} d12_d34={:.17e} d13_d14_d12={:.17e} d23_d24_d12={:.17e}",
        maximum_python_conditioned[0],
        maximum_python_conditioned[1],
        maximum_python_conditioned[2],
        maximum_python_conditioned[3],
        maximum_python_conditioned[4],
        maximum_python_conditioned[5],
    );
    eprintln!(
        "P0AB_CROSS_IMPLEMENTATION_PROPAGATION_MAXIMA energy={:.17e} massless={:.17e} massive={:.17e} d12_d34={:.17e} d13_d14_d12={:.17e} d23_d24_d12={:.17e} massless_mutation_index={massless_mutation_index} massless_mutation_ratio={massless_mutation_ratio:.17e} dot_mutation_index={dot_mutation_index} dot_mutation_cross_gap={mutated_dot_cross_gap:.17e} dot_original_allowance={dot_mutation_allowance:.17e}",
        maximum_cross_propagation_ratio[0],
        maximum_cross_propagation_ratio[1],
        maximum_cross_propagation_ratio[2],
        maximum_cross_propagation_ratio[3],
        maximum_cross_propagation_ratio[4],
        maximum_cross_propagation_ratio[5],
    );
}
'''

    text = text[:start] + replacement + text[end:]
    TEST.write_text(text, encoding="utf-8")
    print("D-081R1F1 P0AB reference-conditioned invariant amendment: CHANGED")


if __name__ == "__main__":
    main()
