//! RED-first tests for the D-081R1F1 moving-rule and QED-off EOS tangents.
//!
//! This file is intentionally committed before `f10_tgamma_tangent` exists.
//! The dedicated RED workflow temporarily registers this test module and must
//! observe the contracted unresolved import before any production source is
//! added.

use crate::f10_action_kinematics::electron_half_line_rule;
use crate::flrw::electromagnetic_eos;
use crate::f10_tgamma_tangent::{
    electron_half_line_tgamma_tangent, electromagnetic_eos_tgamma_tangent,
};

fn maximum_relative(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&observed, &reference)| {
            (observed - reference).abs()
                / observed.abs().max(reference.abs()).max(f64::MIN_POSITIVE)
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn moving_half_line_rule_has_exact_linear_temperature_tangent() {
    let temperature = 2.05_f64;
    let tangent = electron_half_line_tgamma_tangent(48, temperature).unwrap();
    let (momentum, weights) = electron_half_line_rule(48, temperature).unwrap();

    assert_eq!(tangent.momentum.len(), 48);
    assert_eq!(tangent.weights.len(), 48);
    assert_eq!(tangent.d_momentum_dt.len(), 48);
    assert_eq!(tangent.d_weights_dt.len(), 48);

    for index in 0..48 {
        assert_eq!(tangent.momentum[index].to_bits(), momentum[index].to_bits());
        assert_eq!(tangent.weights[index].to_bits(), weights[index].to_bits());
        assert_eq!(
            tangent.d_momentum_dt[index].to_bits(),
            (momentum[index] / temperature).to_bits()
        );
        assert_eq!(
            tangent.d_weights_dt[index].to_bits(),
            (weights[index] / temperature).to_bits()
        );
    }

    let epsilon = 1.0e-5_f64;
    let (plus_momentum, plus_weights) =
        electron_half_line_rule(48, temperature + epsilon).unwrap();
    let (minus_momentum, minus_weights) =
        electron_half_line_rule(48, temperature - epsilon).unwrap();
    let centered_momentum: Vec<f64> = plus_momentum
        .iter()
        .zip(&minus_momentum)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect();
    let centered_weights: Vec<f64> = plus_weights
        .iter()
        .zip(&minus_weights)
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect();

    assert!(maximum_relative(&tangent.d_momentum_dt, &centered_momentum) <= 2.0e-10);
    assert!(maximum_relative(&tangent.d_weights_dt, &centered_weights) <= 2.0e-10);

    assert!(electron_half_line_tgamma_tangent(48, 0.0).is_err());
    assert!(electron_half_line_tgamma_tangent(48, f64::NAN).is_err());
}

#[test]
fn qed_off_eos_tangent_matches_the_admitted_primal_operator() {
    let temperature = 2.05_f64;
    let tangent = electromagnetic_eos_tgamma_tangent(temperature).unwrap();
    let base = electromagnetic_eos(temperature).unwrap();

    assert_eq!(tangent.base.rho.to_bits(), base.rho.to_bits());
    assert_eq!(tangent.base.pressure.to_bits(), base.pressure.to_bits());
    assert_eq!(tangent.base.drho_dt.to_bits(), base.drho_dt.to_bits());
    assert_eq!(tangent.d_rho.to_bits(), base.drho_dt.to_bits());
    assert_eq!(
        tangent.d_pressure.to_bits(),
        ((base.rho + base.pressure) / temperature).to_bits()
    );
    assert!(tangent.d2_rho.is_finite() && tangent.d2_rho > 0.0);

    let epsilon_ladder = [1.0e-2_f64, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4];
    let best = epsilon_ladder
        .into_iter()
        .map(|epsilon| {
            let plus = electromagnetic_eos(temperature + epsilon).unwrap();
            let minus = electromagnetic_eos(temperature - epsilon).unwrap();
            let centered = (plus.drho_dt - minus.drho_dt) / (2.0 * epsilon);
            (tangent.d2_rho - centered).abs()
                / tangent.d2_rho.abs().max(centered.abs()).max(f64::MIN_POSITIVE)
        })
        .fold(f64::INFINITY, f64::min);
    assert!(best <= 1.0e-7, "best EOS second-derivative witness was {best:.17e}");

    assert!(electromagnetic_eos_tgamma_tangent(0.0).is_err());
    assert!(electromagnetic_eos_tgamma_tangent(f64::INFINITY).is_err());
}
