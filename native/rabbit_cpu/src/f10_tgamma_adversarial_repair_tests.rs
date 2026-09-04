//! RED-first adversarial repair contract for D-081R1F1 P0A/P0B.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::F10CollisionConfig;
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent,
    mapped_modal_basis_derivative,
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

fn contribution_scaled(residual: f64, terms: &[f64]) -> f64 {
    residual.abs()
        / terms
            .iter()
            .map(|value| value.abs())
            .sum::<f64>()
            .max(f64::MIN_POSITIVE)
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
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
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

    for index in 0..tangent.support.len() {
        if !tangent.support[index] {
            continue;
        }
        let energy = tangent.d_e2[index] - tangent.d_e3[index] - tangent.d_e4[index];
        assert!(
            contribution_scaled(
                energy,
                &[tangent.d_e2[index], tangent.d_e3[index], tangent.d_e4[index]],
            ) <= INVARIANT_CAP,
            "energy tangent invariant failed at sample {index}"
        );

        let massless = tangent.d_e3[index] - tangent.d_p3_magnitude[index];
        assert!(
            contribution_scaled(
                massless,
                &[tangent.d_e3[index], tangent.d_p3_magnitude[index]],
            ) <= INVARIANT_CAP,
            "massless outgoing tangent invariant failed at sample {index}"
        );

        let massive_left = tangent.base.e4[index] * tangent.d_e4[index];
        let massive_right =
            tangent.base.p4_magnitude[index] * tangent.d_p4_magnitude[index];
        assert!(
            contribution_scaled(massive_left - massive_right, &[massive_left, massive_right])
                <= INVARIANT_CAP,
            "massive outgoing tangent invariant failed at sample {index}"
        );

        for (label, residual, terms) in [
            (
                "d12-d34",
                tangent.d_d12[index] - tangent.d_d34[index],
                vec![tangent.d_d12[index], tangent.d_d34[index]],
            ),
            (
                "d13+d14-d12",
                tangent.d_d13[index] + tangent.d_d14[index] - tangent.d_d12[index],
                vec![
                    tangent.d_d13[index],
                    tangent.d_d14[index],
                    tangent.d_d12[index],
                ],
            ),
            (
                "d23+d24-d12",
                tangent.d_d23[index] + tangent.d_d24[index] - tangent.d_d12[index],
                vec![
                    tangent.d_d23[index],
                    tangent.d_d24[index],
                    tangent.d_d12[index],
                ],
            ),
        ] {
            assert!(
                contribution_scaled(residual, &terms) <= INVARIANT_CAP,
                "{label} tangent invariant failed at sample {index}"
            );
        }
    }
}

#[test]
#[ignore = "requires deterministic frozen D-080A P0B oracle"]
fn direct_python_d080a_basis_and_kinematic_arrays_match() {
    let path = env::var("D081R1F1_P0B_ORACLE")
        .expect("D081R1F1_P0B_ORACLE must identify the generated oracle");
    let value: Value = serde_json::from_str(&fs::read_to_string(path).expect("read P0B oracle"))
        .expect("valid P0B oracle");
    assert_eq!(value["schema"], "rabbit.d081r1f1.p0b_tgamma_kinematics_oracle.v1");
    assert_eq!(
        value["d080a_blob"],
        "c585d5865fd68a90a04a76ab540b8437fba8cfce"
    );

    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let query = bit_array(&value["basis"]["query"]);
    let basis = mapped_modal_basis_derivative(&grid, &query).unwrap();
    assert!(
        global_relative(&basis, &bit_array(&value["basis"]["derivative"])) <= 2.0e-7
    );

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
        ("base_quadrature_weight", tangent.base.quadrature_weight.as_slice()),
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
        ("d_quadrature_weight", tangent.d_quadrature_weight.as_slice()),
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
