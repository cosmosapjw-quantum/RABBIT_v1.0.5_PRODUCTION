//! RED-first adversarial repair contract for D-081R1F1 P0A/P0B.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::{F10CollisionConfig, angular_rule};
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent, mapped_modal_basis_derivative,
};
use crate::f10_tgamma_tangent::{
    electromagnetic_eos_tgamma_continuum_d2rho_dt2_reference,
    electromagnetic_eos_tgamma_discrete_d2rho_dt2, electromagnetic_eos_tgamma_tangent,
};
use crate::flrw::electromagnetic_eos;
use serde_json::Value;
use std::{env, fs};

const ELECTRON_MASS_MEV: f64 = 0.510_998_95;
const INVARIANT_CAP: f64 = 2.0e-12;

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn global_relative(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
        / maximum_absolute(actual)
            .max(maximum_absolute(expected))
            .max(f64::MIN_POSITIVE)
}

fn scalar_relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
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

#[test]
fn discrete_eos_second_derivative_is_the_admitted_finite_sum_tangent() {
    for temperature in [0.5_f64, 2.05, 10.0] {
        let tangent = electromagnetic_eos_tgamma_tangent(temperature).unwrap();
        let discrete = electromagnetic_eos_tgamma_discrete_d2rho_dt2(temperature).unwrap();
        let continuum =
            electromagnetic_eos_tgamma_continuum_d2rho_dt2_reference(temperature).unwrap();
        assert_eq!(tangent.d2_rho.to_bits(), discrete.to_bits());

        let step = 1.0e-5 * temperature;
        let plus = electromagnetic_eos(temperature + step).unwrap().drho_dt;
        let minus = electromagnetic_eos(temperature - step).unwrap().drho_dt;
        let centered = (plus - minus) / (2.0 * step);
        assert!(
            scalar_relative(discrete, centered) <= 2.0e-7,
            "discrete EOS derivative at T={temperature:.17e} diverged from admitted primal centered witness"
        );
        assert!(
            scalar_relative(discrete, continuum) <= 2.0e-7,
            "discrete EOS derivative at T={temperature:.17e} diverged from continuum reference"
        );
    }
}

#[test]
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
    let oracle_value: Value =
        serde_json::from_str(&fs::read_to_string(oracle_path).expect("read P0B oracle"))
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
            let incoming_index = (index / final_angular_size) % config.incoming_polar_order;
            let mu12 = rule.incoming_mu[incoming_index];
            let sin12 = (1.0 - mu12 * mu12).max(0.0).sqrt();
            let total_x = p2 * sin12;
            let total_z = target_momentum + p2 * mu12;
            let total_magnitude = ((total_x * total_x + 0.0) + total_z * total_z).sqrt();
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
            let lambda =
                value * value + mass3_squared * mass3_squared + mass_squared * mass_squared
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
            (rust_base[0].abs() + rust_base[1].abs() + rust_base[2].abs()) / temperature_gamma,
            (rust_base[1].abs() + rust_base[3].abs()) / temperature_gamma,
            (rust_base[2].powi(2).abs() + rust_base[4].powi(2).abs() + mass_squared.abs())
                / temperature_gamma,
            (rust_base[5].abs() + rust_base[10].abs()) / temperature_gamma,
            (rust_base[6].abs() + rust_base[7].abs() + rust_base[5].abs()) / temperature_gamma,
            (rust_base[8].abs() + rust_base[9].abs() + rust_base[5].abs()) / temperature_gamma,
        ];
        let python_scale = [
            (python_base_at[0].abs() + python_base_at[1].abs() + python_base_at[2].abs())
                / temperature_gamma,
            (python_base_at[1].abs() + python_base_at[3].abs()) / temperature_gamma,
            (python_base_at[2].powi(2).abs()
                + python_base_at[4].powi(2).abs()
                + mass_squared.abs())
                / temperature_gamma,
            (python_base_at[5].abs() + python_base_at[10].abs()) / temperature_gamma,
            (python_base_at[6].abs() + python_base_at[7].abs() + python_base_at[5].abs())
                / temperature_gamma,
            (python_base_at[8].abs() + python_base_at[9].abs() + python_base_at[5].abs())
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
            python_tangent_at[0].abs() + python_tangent_at[1].abs() + python_tangent_at[2].abs(),
            python_tangent_at[1].abs() + python_tangent_at[3].abs(),
            2.0 * (python_massive_left.abs() + python_massive_right.abs()),
            python_tangent_at[5].abs() + python_tangent_at[10].abs(),
            python_tangent_at[6].abs() + python_tangent_at[7].abs() + python_tangent_at[5].abs(),
            python_tangent_at[8].abs() + python_tangent_at[9].abs() + python_tangent_at[5].abs(),
        ];

        let propagation = [
            (rust_tangent[0] - python_tangent_at[0]).abs()
                + (rust_tangent[1] - python_tangent_at[1]).abs()
                + (rust_tangent[2] - python_tangent_at[2]).abs(),
            (rust_tangent[1] - python_tangent_at[1]).abs()
                + (rust_tangent[3] - python_tangent_at[3]).abs(),
            2.0 * ((rust_base[2] - python_base_at[2]).abs() * rust_tangent[2].abs()
                + python_base_at[2].abs() * (rust_tangent[2] - python_tangent_at[2]).abs()
                + (rust_base[4] - python_base_at[4]).abs() * rust_tangent[4].abs()
                + python_base_at[4].abs() * (rust_tangent[4] - python_tangent_at[4]).abs()),
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
            let raw_local =
                rust_residual[component].abs() / rust_term_sum[component].max(f64::MIN_POSITIVE);
            let rust_conditioned = rust_residual[component].abs() / characteristic_scale;
            let python_conditioned = python_residual[component].abs() / characteristic_scale;
            let evaluation_roundoff = 128.0
                * f64::EPSILON
                * (rust_term_sum[component] + python_term_sum[component] + characteristic_scale);
            let propagation_allowance = propagation[component] + evaluation_roundoff;
            let cross_gap = (rust_residual[component] - python_residual[component]).abs();
            let cross_ratio = cross_gap / propagation_allowance.max(f64::MIN_POSITIVE);

            assert!(raw_local.is_finite());
            assert!(rust_conditioned.is_finite());
            assert!(python_conditioned.is_finite());
            assert!(cross_ratio.is_finite());
            maximum_raw_local[component] = maximum_raw_local[component].max(raw_local);
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

        let massless_sensitivity = rust_tangent[3].abs() / rust_scale[1].max(f64::MIN_POSITIVE);
        if massless_sensitivity > massless_mutation_sensitivity {
            massless_mutation_sensitivity = massless_sensitivity;
            massless_mutation_index = Some(index);
        }

        let dot_sensitivity = rust_tangent[9].abs() / rust_scale[5].max(f64::MIN_POSITIVE);
        if dot_sensitivity > dot_mutation_sensitivity {
            dot_mutation_sensitivity = dot_sensitivity;
            dot_mutation_index = Some(index);
            let characteristic_scale = rust_scale[5].max(python_scale[5]).max(f64::MIN_POSITIVE);
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
    let massless_mutation_ratio = (tangent.d_e3[massless_mutation_index] - mutated_dp3).abs()
        / massless_mutation_scale.max(f64::MIN_POSITIVE);
    assert!(
        massless_mutation_ratio > 100.0 * INVARIANT_CAP,
        "one-percent massless-leg tangent mutation was not load-bearing: {massless_mutation_ratio:.17e}"
    );

    let dot_mutation_index = dot_mutation_index.expect("supported d24 tangent sample");
    let mutated_d24 = tangent.d_d24[dot_mutation_index] * 1.01;
    let mutated_dot_residual =
        tangent.d_d23[dot_mutation_index] + mutated_d24 - tangent.d_d12[dot_mutation_index];
    let mutated_dot_cross_gap = (mutated_dot_residual - dot_mutation_python_residual).abs();
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

#[test]
#[ignore = "requires deterministic frozen D-080A P0B oracle"]
fn direct_python_d080a_basis_and_kinematic_arrays_match() {
    let path = env::var("D081R1F1_P0B_ORACLE")
        .expect("D081R1F1_P0B_ORACLE must identify the generated oracle");
    let value: Value = serde_json::from_str(&fs::read_to_string(path).expect("read P0B oracle"))
        .expect("valid P0B oracle");
    assert_eq!(
        value["schema"],
        "rabbit.d081r1f1.p0b_tgamma_kinematics_oracle.v1"
    );
    assert_eq!(
        value["d080a_blob"],
        "c585d5865fd68a90a04a76ab540b8437fba8cfce"
    );

    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let query = bit_array(&value["basis"]["query"]);
    let basis = mapped_modal_basis_derivative(&grid, &query).unwrap();
    assert!(global_relative(&basis, &bit_array(&value["basis"]["derivative"])) <= 2.0e-7);

    let kinematic = &value["kinematics"];
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1: bits(&kinematic["p1_bits"]),
        temperature_gamma: bits(&kinematic["temperature_bits"]),
        electron_mass: bits(&kinematic["electron_mass_bits"]),
        config: F10CollisionConfig::default(),
    })
    .unwrap();
    let expected_support = kinematic["support"]
        .as_array()
        .expect("support array")
        .iter()
        .map(|value| value.as_bool().expect("bool support"))
        .collect::<Vec<_>>();
    assert_eq!(tangent.support, expected_support);

    for (label, actual) in [
        ("base_p2", tangent.base.p2.as_slice()),
        ("base_e2", tangent.base.e2.as_slice()),
        ("base_e3", tangent.base.e3.as_slice()),
        ("base_e4", tangent.base.e4.as_slice()),
        ("base_p3_magnitude", tangent.base.p3_magnitude.as_slice()),
        ("base_p4_magnitude", tangent.base.p4_magnitude.as_slice()),
        ("base_phase_space", tangent.base.phase_space.as_slice()),
        (
            "base_quadrature_weight",
            tangent.base.quadrature_weight.as_slice(),
        ),
        ("base_d12", tangent.base.d12.as_slice()),
        ("base_d13", tangent.base.d13.as_slice()),
        ("base_d14", tangent.base.d14.as_slice()),
        ("base_d23", tangent.base.d23.as_slice()),
        ("base_d24", tangent.base.d24.as_slice()),
        ("base_d34", tangent.base.d34.as_slice()),
        ("d_p2", tangent.d_p2.as_slice()),
        ("d_e2", tangent.d_e2.as_slice()),
        ("d_e3", tangent.d_e3.as_slice()),
        ("d_e4", tangent.d_e4.as_slice()),
        ("d_p3_magnitude", tangent.d_p3_magnitude.as_slice()),
        ("d_p4_magnitude", tangent.d_p4_magnitude.as_slice()),
        ("d_phase_space", tangent.d_phase_space.as_slice()),
        (
            "d_quadrature_weight",
            tangent.d_quadrature_weight.as_slice(),
        ),
        ("d_d12", tangent.d_d12.as_slice()),
        ("d_d13", tangent.d_d13.as_slice()),
        ("d_d14", tangent.d_d14.as_slice()),
        ("d_d23", tangent.d_d23.as_slice()),
        ("d_d24", tangent.d_d24.as_slice()),
        ("d_d34", tangent.d_d34.as_slice()),
    ] {
        let expected = bit_array(&kinematic[label]);
        let residual = global_relative(actual, &expected);
        assert!(
            residual <= 1.0e-7,
            "{label} direct Python residual {residual:.17e} exceeded 1e-7"
        );
    }
    assert!(
        scalar_relative(
            tangent.minimum_support_margin_relative,
            bits(&kinematic["minimum_support_margin_relative_bits"]),
        ) <= 1.0e-7
    );
    assert!(
        scalar_relative(
            tangent.minimum_supported_lambda_margin_relative,
            bits(&kinematic["minimum_supported_lambda_margin_relative_bits"]),
        ) <= 1.0e-7
    );
}
