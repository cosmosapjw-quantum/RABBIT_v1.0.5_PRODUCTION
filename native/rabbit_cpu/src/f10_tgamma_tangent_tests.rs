//! RED-first unit contract for the first D-081R1F1 thermal primitives.

use crate::f10_action_kinematics::electron_half_line_rule;
use crate::f10_tgamma_tangent::{
    F10TgammaTangentError, electromagnetic_eos_tgamma_tangent, electron_half_line_tgamma_tangent,
};
use crate::flrw::electromagnetic_eos;
use serde_json::Value;
use std::{env, fs};

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

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
}

fn fixture() -> Value {
    let path = env::var("D081R1F1_P0A_ORACLE")
        .expect("D081R1F1_P0A_ORACLE must identify the generated Python oracle");
    serde_json::from_str(&fs::read_to_string(path).expect("read P0A oracle"))
        .expect("valid P0A oracle")
}

#[test]
fn moving_half_line_tangent_matches_the_frozen_python_oracle() {
    let value = fixture();
    assert_eq!(
        value["schema"],
        "rabbit.d081r1f1.p0a_tgamma_primitives_oracle.v1"
    );
    assert_eq!(
        value["d080a_blob"],
        "c585d5865fd68a90a04a76ab540b8437fba8cfce"
    );
    let half_line = &value["half_line"];
    let order = usize::try_from(half_line["order"].as_u64().expect("order")).unwrap();
    let temperature = bits(&half_line["temperature_bits"]);
    let tangent = electron_half_line_tgamma_tangent(order, temperature).unwrap();

    for (label, actual, expected) in [
        (
            "momentum",
            tangent.momentum.as_slice(),
            bit_array(&half_line["momentum"]),
        ),
        (
            "weights",
            tangent.weights.as_slice(),
            bit_array(&half_line["weights"]),
        ),
        (
            "d_momentum_dt",
            tangent.d_momentum_dt.as_slice(),
            bit_array(&half_line["d_momentum_dt"]),
        ),
        (
            "d_weights_dt",
            tangent.d_weights_dt.as_slice(),
            bit_array(&half_line["d_weights_dt"]),
        ),
    ] {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&observed, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                relative_error(observed, reference) <= 1.0e-13,
                "{label}[{index}] observed={observed:.17e} expected={reference:.17e}"
            );
        }
    }

    let (primal_momentum, primal_weights) = electron_half_line_rule(order, temperature).unwrap();
    assert_eq!(tangent.momentum, primal_momentum);
    assert_eq!(tangent.weights, primal_weights);
    for index in 0..order {
        assert_eq!(
            tangent.d_momentum_dt[index].to_bits(),
            (tangent.momentum[index] / temperature).to_bits()
        );
        assert_eq!(
            tangent.d_weights_dt[index].to_bits(),
            (tangent.weights[index] / temperature).to_bits()
        );
    }
}

#[test]
fn qed_off_eos_tangent_matches_d080a_and_the_primal_eos() {
    let value = fixture();
    for case in value["eos"].as_array().expect("EOS cases") {
        let temperature = bits(&case["temperature_bits"]);
        let tangent = electromagnetic_eos_tgamma_tangent(temperature).unwrap();
        let primal = electromagnetic_eos(temperature).unwrap();

        assert_eq!(tangent.base.rho.to_bits(), primal.rho.to_bits());
        assert_eq!(tangent.base.pressure.to_bits(), primal.pressure.to_bits());
        assert_eq!(tangent.base.drho_dt.to_bits(), primal.drho_dt.to_bits());
        assert_eq!(tangent.d_rho.to_bits(), primal.drho_dt.to_bits());

        for (label, actual, expected) in [
            ("rho", tangent.base.rho, bits(&case["rho_bits"])),
            (
                "pressure",
                tangent.base.pressure,
                bits(&case["pressure_bits"]),
            ),
            ("d_rho", tangent.d_rho, bits(&case["d_rho_bits"])),
            (
                "d_pressure",
                tangent.d_pressure,
                bits(&case["d_pressure_bits"]),
            ),
            ("d2_rho", tangent.d2_rho, bits(&case["d2_rho_bits"])),
        ] {
            assert!(
                relative_error(actual, expected) <= 1.0e-7,
                "{label} at T={temperature:.17e}: actual={actual:.17e}, expected={expected:.17e}"
            );
        }

        let thermodynamic_identity = (tangent.base.rho + tangent.base.pressure) / temperature;
        assert!(relative_error(tangent.d_pressure, thermodynamic_identity) <= 64.0 * f64::EPSILON);
        assert!(tangent.d2_rho.is_finite() && tangent.d2_rho > 0.0);

        let step = 1.0e-5 * temperature;
        let plus = electromagnetic_eos(temperature + step).unwrap().drho_dt;
        let minus = electromagnetic_eos(temperature - step).unwrap().drho_dt;
        let centered = (plus - minus) / (2.0 * step);
        assert!(relative_error(tangent.d2_rho, centered) <= 2.0e-7);
    }
}

#[test]
fn thermal_primitive_inputs_fail_closed() {
    assert_eq!(
        electron_half_line_tgamma_tangent(1, 2.0).unwrap_err(),
        F10TgammaTangentError::InvalidInput
    );
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            electron_half_line_tgamma_tangent(48, temperature).unwrap_err(),
            F10TgammaTangentError::InvalidInput
        );
        assert_eq!(
            electromagnetic_eos_tgamma_tangent(temperature).unwrap_err(),
            F10TgammaTangentError::InvalidInput
        );
    }
}
