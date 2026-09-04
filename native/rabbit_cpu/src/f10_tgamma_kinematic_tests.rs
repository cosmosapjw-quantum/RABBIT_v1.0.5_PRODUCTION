//! RED-first tests for mapped-basis and finite-mass elastic `T_gamma` tangents.
//!
//! The production APIs imported below were absent in the preserved P0B RED.
//! This committed suite now exercises the focused sibling implementation
//! directly, without a facade re-export that would create unused-import noise.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicBatch, F10KinematicInput, electron_half_line_rule,
    two_body_kinematics,
};
use crate::f10_action_spectral::modal_basis;
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent,
    mapped_modal_basis_derivative,
};

const ELECTRON_MASS_MEV: f64 = 0.510_998_95;

fn maximum_absolute(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max)
}

fn global_relative(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let difference = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    difference
        / maximum_absolute(actual)
            .max(maximum_absolute(expected))
            .max(f64::MIN_POSITIVE)
}

fn centered(plus: &[f64], minus: &[f64], epsilon: f64) -> Vec<f64> {
    assert_eq!(plus.len(), minus.len());
    plus.iter()
        .zip(minus)
        .map(|(upper, lower)| (upper - lower) / (2.0 * epsilon))
        .collect()
}

fn exact_bits(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "{label} bit mismatch at index {index}"
        );
    }
}

fn elastic_batch(
    p1: f64,
    temperature_gamma: f64,
    config: F10CollisionConfig,
) -> F10KinematicBatch {
    let (p2_nodes, p2_weights) =
        electron_half_line_rule(config.electron_radial_order, temperature_gamma).unwrap();
    two_body_kinematics(F10KinematicInput {
        p1,
        p2_nodes: &p2_nodes,
        p2_weights: &p2_weights,
        mass2: ELECTRON_MASS_MEV,
        mass3: 0.0,
        mass4: ELECTRON_MASS_MEV,
        config,
    })
    .unwrap()
}

#[test]
fn mapped_basis_derivative_matches_centered_y_witness() {
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let query = [0.25_f64, 1.25, 3.75, 7.25];
    let derivative = mapped_modal_basis_derivative(&grid, &query).unwrap();
    assert_eq!(derivative.len(), query.len() * grid.order);

    let epsilon = 1.0e-6_f64;
    let plus_query: Vec<f64> = query.iter().map(|value| value + epsilon).collect();
    let minus_query: Vec<f64> = query.iter().map(|value| value - epsilon).collect();
    let plus = modal_basis(&grid, &plus_query).unwrap();
    let minus = modal_basis(&grid, &minus_query).unwrap();
    let witness = centered(&plus, &minus, epsilon);

    assert!(
        global_relative(&derivative, &witness) <= 2.0e-7,
        "mapped basis derivative exceeded centered-witness gate"
    );
    for point in 0..query.len() {
        assert_eq!(derivative[point * grid.order].to_bits(), 0.0_f64.to_bits());
    }

    assert!(mapped_modal_basis_derivative(&grid, &[-1.0e-6]).is_err());
    assert!(mapped_modal_basis_derivative(&grid, &[8.0 + 1.0e-6]).is_err());
    assert!(mapped_modal_basis_derivative(&grid, &[f64::NAN]).is_err());
}

#[test]
fn elastic_kinematic_tangent_preserves_branch_and_matches_centered_witness() {
    let config = F10CollisionConfig::default();
    let p1 = 2.0_f64;
    let temperature = 2.05_f64;
    let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
        p1,
        temperature_gamma: temperature,
        electron_mass: ELECTRON_MASS_MEV,
        config,
    })
    .unwrap();
    let base = elastic_batch(p1, temperature, config);

    assert_eq!(tangent.base.shape, base.shape);
    assert_eq!(tangent.base.support, base.support);
    assert_eq!(tangent.support, base.support);
    exact_bits(&tangent.base.p2, &base.p2, "base p2");
    exact_bits(&tangent.base.e2, &base.e2, "base e2");
    exact_bits(&tangent.base.e3, &base.e3, "base e3");
    exact_bits(&tangent.base.e4, &base.e4, "base e4");
    exact_bits(&tangent.base.phase_space, &base.phase_space, "base phase space");
    exact_bits(
        &tangent.base.quadrature_weight,
        &base.quadrature_weight,
        "base quadrature weight",
    );

    assert!(tangent.minimum_support_margin.is_finite());
    assert!(tangent.minimum_support_margin > 0.0);
    assert!(tangent.minimum_lambda_margin.is_finite());
    assert!(tangent.minimum_lambda_margin > 0.0);

    let epsilon = 1.0e-5_f64;
    let plus = elastic_batch(p1, temperature + epsilon, config);
    let minus = elastic_batch(p1, temperature - epsilon, config);
    assert_eq!(plus.support, base.support, "plus support branch changed");
    assert_eq!(minus.support, base.support, "minus support branch changed");

    let comparisons: [(&str, &[f64], Vec<f64>); 15] = [
        ("p2", &tangent.d_p2, centered(&plus.p2, &minus.p2, epsilon)),
        ("e2", &tangent.d_e2, centered(&plus.e2, &minus.e2, epsilon)),
        ("e3", &tangent.d_e3, centered(&plus.e3, &minus.e3, epsilon)),
        ("e4", &tangent.d_e4, centered(&plus.e4, &minus.e4, epsilon)),
        (
            "p3 magnitude",
            &tangent.d_p3_magnitude,
            centered(&plus.p3_magnitude, &minus.p3_magnitude, epsilon),
        ),
        (
            "p4 magnitude",
            &tangent.d_p4_magnitude,
            centered(&plus.p4_magnitude, &minus.p4_magnitude, epsilon),
        ),
        (
            "phase space",
            &tangent.d_phase_space,
            centered(&plus.phase_space, &minus.phase_space, epsilon),
        ),
        (
            "quadrature weight",
            &tangent.d_quadrature_weight,
            centered(&plus.quadrature_weight, &minus.quadrature_weight, epsilon),
        ),
        ("d12", &tangent.d_d12, centered(&plus.d12, &minus.d12, epsilon)),
        ("d13", &tangent.d_d13, centered(&plus.d13, &minus.d13, epsilon)),
        ("d14", &tangent.d_d14, centered(&plus.d14, &minus.d14, epsilon)),
        ("d23", &tangent.d_d23, centered(&plus.d23, &minus.d23, epsilon)),
        ("d24", &tangent.d_d24, centered(&plus.d24, &minus.d24, epsilon)),
        ("d34", &tangent.d_d34, centered(&plus.d34, &minus.d34, epsilon)),
        (
            "support indicator tangent",
            &tangent.d_support_indicator,
            vec![0.0; base.support.len()],
        ),
    ];

    for (label, analytic, witness) in comparisons {
        let residual = global_relative(analytic, &witness);
        assert!(
            residual <= 1.0e-7,
            "{label} tangent residual {residual:.17e} exceeded 1e-7"
        );
    }

    assert!(
        evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
            p1: 0.0,
            temperature_gamma: temperature,
            electron_mass: ELECTRON_MASS_MEV,
            config,
        })
        .is_err()
    );
    assert!(
        evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
            p1,
            temperature_gamma: f64::NAN,
            electron_mass: ELECTRON_MASS_MEV,
            config,
        })
        .is_err()
    );
}
