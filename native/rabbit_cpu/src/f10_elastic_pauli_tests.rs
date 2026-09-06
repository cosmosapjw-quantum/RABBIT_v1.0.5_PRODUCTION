//! Elastic Pauli/location tangent tests. Action projection and energy weights remain open.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicInput, electron_half_line_rule, two_body_kinematics,
};
use crate::f10_action_spectral::interpolate;
use crate::f10_electron_action::tgamma_jvp::elastic_pauli_location_tangent;
use crate::f10_kernel_primitives::stable_pauli_gain_minus_loss;
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent,
};

const MASS: f64 = 0.510_998_95;

fn batch(p1: f64, temperature: f64) -> crate::f10_action_kinematics::F10KinematicBatch {
    let config = F10CollisionConfig::default();
    let (nodes, weights) =
        electron_half_line_rule(config.electron_radial_order, temperature).unwrap();
    two_body_kinematics(F10KinematicInput {
        p1,
        p2_nodes: &nodes,
        p2_weights: &weights,
        mass2: MASS,
        mass3: 0.0,
        mass4: MASS,
        config,
    })
    .unwrap()
}

fn relative(actual: &[f64], expected: &[f64]) -> f64 {
    let scale = actual
        .iter()
        .chain(expected)
        .map(|x| x.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn elastic_pauli_and_moving_location_match_centered_primal() {
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let target_logits: Vec<f64> = grid
        .nodes
        .iter()
        .map(|y| -0.35 - 0.8 * y + 0.015 * y * y)
        .collect();
    for (p1, temperature) in [(2.0, 2.05), (0.75, 0.4)] {
        let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
            p1,
            temperature_gamma: temperature,
            electron_mass: MASS,
            config: F10CollisionConfig::default(),
        })
        .unwrap();
        let base = &tangent.base;
        let envelope_h = temperature * 1.0e-3;
        let envelope_plus = batch(p1, temperature + envelope_h);
        let envelope_minus = batch(p1, temperature - envelope_h);
        assert_eq!(
            base.support, envelope_plus.support,
            "NONDIFFERENTIABLE_DISCRETE_EVENT"
        );
        assert_eq!(
            base.support, envelope_minus.support,
            "NONDIFFERENTIABLE_DISCRETE_EVENT"
        );
        let mut actual = Vec::new();
        let mut indices = Vec::new();
        for i in 0..base.support.len() {
            let y3 = base.p3_magnitude[i] / 2.0;
            let y3_plus = envelope_plus.p3_magnitude[i] / 2.0;
            let y3_minus = envelope_minus.p3_magnitude[i] / 2.0;
            if base.support[i]
                && [y3, y3_plus, y3_minus]
                    .into_iter()
                    .all(|query| query > 0.0 && query < grid.y_max)
            {
                let value = elastic_pauli_location_tangent(
                    &grid,
                    &target_logits,
                    -0.7,
                    base.e2[i],
                    tangent.d_e2[i],
                    base.e4[i],
                    tangent.d_e4[i],
                    temperature,
                    y3,
                    tangent.d_p3_magnitude[i] / 2.0,
                )
                .unwrap();
                assert_eq!(
                    value.base.to_bits(),
                    stable_pauli_gain_minus_loss(value.logits)
                        .unwrap()
                        .to_bits()
                );
                actual.push(value.derivative);
                indices.push(i);
            }
        }
        let mut best = f64::INFINITY;
        for factor in [1.0e-3, 3.0e-4, 1.0e-4] {
            let h = temperature * factor;
            let plus = batch(p1, temperature + h);
            let minus = batch(p1, temperature - h);
            assert_eq!(
                base.support, plus.support,
                "NONDIFFERENTIABLE_DISCRETE_EVENT"
            );
            assert_eq!(
                base.support, minus.support,
                "NONDIFFERENTIABLE_DISCRETE_EVENT"
            );
            let expected: Vec<f64> = indices
                .iter()
                .map(|&i| {
                    let yp = plus.p3_magnitude[i] / 2.0;
                    let ym = minus.p3_magnitude[i] / 2.0;
                    let u3p = interpolate(&grid, &target_logits, &[yp]).unwrap()[0];
                    let u3m = interpolate(&grid, &target_logits, &[ym]).unwrap()[0];
                    let pp = stable_pauli_gain_minus_loss([
                        -0.7,
                        -plus.e2[i] / (temperature + h),
                        u3p,
                        -plus.e4[i] / (temperature + h),
                    ])
                    .unwrap();
                    let pm = stable_pauli_gain_minus_loss([
                        -0.7,
                        -minus.e2[i] / (temperature - h),
                        u3m,
                        -minus.e4[i] / (temperature - h),
                    ])
                    .unwrap();
                    (pp - pm) / (2.0 * h)
                })
                .collect();
            best = best.min(relative(&actual, &expected));
        }
        assert!(best <= 1.0e-7, "Pauli/location residual: {best:.17e}");
    }
}

#[test]
fn elastic_pauli_location_rejects_invalid_shapes_and_coordinates() {
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    assert!(
        elastic_pauli_location_tangent(&grid, &[0.0; 7], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0)
            .is_err()
    );
    assert!(
        elastic_pauli_location_tangent(&grid, &[0.0; 8], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 9.0, 0.0)
            .is_err()
    );
}
