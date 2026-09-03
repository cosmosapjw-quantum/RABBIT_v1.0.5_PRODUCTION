//! RED-first admission tests for D-081R1D1 action foundations.

use crate::f10_action_grid::{F10ActionGrid, decode_cloglog_to_logit};
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicInput, angular_rule, electron_half_line_rule,
    two_body_kinematics,
};
use crate::f10_action_spectral::{
    interpolate, modal_basis, modal_coefficients, modal_product, native_action,
};
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/fixtures/d081r1/action_foundations_case.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("valid D-081R1D1 fixture")
}

fn bits(value: &Value) -> f64 {
    let encoded = value.as_str().expect("hex bit string");
    f64::from_bits(u64::from_str_radix(encoded, 16).expect("valid f64 bits"))
}

fn bit_array(value: &Value) -> Vec<f64> {
    value["bits"]
        .as_array()
        .expect("bit array")
        .iter()
        .map(bits)
        .collect()
}

fn scaled_residual(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
}

fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let block_scale = actual
        .iter()
        .chain(expected)
        .map(|value| value.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    // Keep the relative gate away from zero, while admitting only the
    // measured order-eight dot-product association floor near exact zero.
    let absolute_floor = 32.0 * f64::EPSILON * block_scale;
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(&left, &right)| {
            let absolute = (left - right).abs();
            if absolute <= absolute_floor {
                0.0
            } else {
                scaled_residual(left, right)
            }
        })
        .fold(0.0_f64, f64::max);
    assert!(
        maximum <= tolerance,
        "maximum scaled residual {maximum:.17e} exceeds {tolerance:.17e};          near-zero absolute floor is {absolute_floor:.17e}"
    );
}

fn batch_case<'a>(value: &'a Value, name: &str) -> &'a Value {
    value["kinematics"]["batches"]
        .as_array()
        .expect("kinematic batches")
        .iter()
        .find(|item| item["name"] == name)
        .expect("named kinematic batch")
}

fn batch_field<'a>(
    batch: &'a crate::f10_action_kinematics::F10KinematicBatch,
    name: &str,
) -> &'a [f64] {
    match name {
        "p2" => &batch.p2,
        "e2" => &batch.e2,
        "e3" => &batch.e3,
        "e4" => &batch.e4,
        "p3_magnitude" => &batch.p3_magnitude,
        "p4_magnitude" => &batch.p4_magnitude,
        "phase_space" => &batch.phase_space,
        "quadrature_weight" => &batch.quadrature_weight,
        "d12" => &batch.d12,
        "d13" => &batch.d13,
        "d14" => &batch.d14,
        "d23" => &batch.d23,
        "d24" => &batch.d24,
        "d34" => &batch.d34,
        other => panic!("unknown batch field {other}"),
    }
}

fn assert_batch_matches_fixture(
    actual: &crate::f10_action_kinematics::F10KinematicBatch,
    expected: &Value,
) {
    let expected_shape = expected["shape"].as_array().expect("shape");
    assert_eq!(
        actual.shape,
        [
            expected_shape[0].as_u64().unwrap() as usize,
            expected_shape[1].as_u64().unwrap() as usize,
            expected_shape[2].as_u64().unwrap() as usize,
            expected_shape[3].as_u64().unwrap() as usize,
        ]
    );
    assert_eq!(
        actual.support.iter().filter(|&&item| item).count(),
        expected["support_count"].as_u64().unwrap() as usize
    );
    for sample in expected["samples"].as_array().expect("samples") {
        let index = sample["flat_index"].as_u64().unwrap() as usize;
        assert_eq!(actual.support[index], sample["support"].as_bool().unwrap());
        for (name, encoded) in sample["values"].as_object().expect("sample values") {
            let reference = bits(encoded);
            let observed = batch_field(actual, name)[index];
            assert!(
                scaled_residual(observed, reference) <= 5.0e-11,
                "{name}[{index}] observed={observed:.17e} reference={reference:.17e}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_binds_frozen_python_and_shapes() {
        let value = fixture();
        assert_eq!(value["schema"], "rabbit.d081r1.action_foundations.v1");
        assert_eq!(
            value["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(value["order"], 8);
        assert_eq!(bits(&value["y_max_bits"]).to_bits(), 8.0_f64.to_bits());
        assert_eq!(value["grid"]["nodes"]["shape"], serde_json::json!([8]));
        assert_eq!(
            value["chart"]["pair_cloglog"]["shape"],
            serde_json::json!([3, 8])
        );
        assert_eq!(value["kinematics"]["batches"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn affine_grid_matches_python_fixture() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        assert_slice_close(&grid.nodes, &bit_array(&value["grid"]["nodes"]), 2.0e-13);
        assert_slice_close(
            &grid.weights,
            &bit_array(&value["grid"]["weights"]),
            5.0e-13,
        );
        assert!(grid.nodes.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(grid.weights.iter().all(|weight| *weight > 0.0));
    }

    #[test]
    fn cloglog_chart_roundtrip_matches_pair_logits() {
        let value = fixture();
        let coordinates = bit_array(&value["chart"]["pair_cloglog"]);
        let expected = bit_array(&value["chart"]["pair_logits"]);
        let actual: Vec<_> = coordinates
            .iter()
            .map(|&coordinate| decode_cloglog_to_logit(coordinate).unwrap())
            .collect();
        assert_slice_close(&actual, &expected, 3.0e-13);
    }

    #[test]
    fn mapped_basis_and_modal_coefficients_match_python() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let query = bit_array(&value["spectral"]["query"]);
        let basis = modal_basis(&grid, &query).unwrap();
        assert_slice_close(
            &basis,
            &bit_array(&value["spectral"]["modal_basis"]),
            8.0e-13,
        );
        let logits = bit_array(&value["chart"]["pair_logits"]);
        let expected = bit_array(&value["spectral"]["modal_coefficients"]);
        for row in 0..3 {
            let actual = modal_coefficients(&grid, &logits[row * 8..(row + 1) * 8]).unwrap();
            assert_slice_close(&actual, &expected[row * 8..(row + 1) * 8], 2.0e-12);
        }
    }

    #[test]
    fn interpolation_and_modal_product_match_python() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let query = bit_array(&value["spectral"]["query"]);
        let logits = bit_array(&value["chart"]["pair_logits"]);
        let expected_interpolation = bit_array(&value["spectral"]["interpolation"]);
        for row in 0..3 {
            let actual = interpolate(&grid, &logits[row * 8..(row + 1) * 8], &query).unwrap();
            assert_slice_close(
                &actual,
                &expected_interpolation[row * query.len()..(row + 1) * query.len()],
                3.0e-12,
            );
        }
        let rates = bit_array(&value["spectral"]["rates"]);
        let actual = modal_product(&grid, &rates, 2, &query).unwrap();
        assert_slice_close(
            &actual,
            &bit_array(&value["spectral"]["modal_product"]),
            3.0e-12,
        );
    }

    #[test]
    fn native_action_matches_python() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let modal = bit_array(&value["spectral"]["modal_seed"]);
        let temperature = bits(&value["temperature_cm_bits"]);
        let actual = native_action(&grid, &modal, 2, temperature).unwrap();
        assert_slice_close(
            &actual,
            &bit_array(&value["spectral"]["native_action"]),
            4.0e-12,
        );
    }

    #[test]
    fn angular_and_electron_rules_match_python() {
        let value = fixture();
        let config = F10CollisionConfig::default();
        let rule = angular_rule(config).unwrap();
        assert_slice_close(
            &rule.incoming_mu,
            &bit_array(&value["rules"]["incoming_mu"]),
            3.0e-13,
        );
        assert_slice_close(
            &rule.incoming_weights,
            &bit_array(&value["rules"]["incoming_weights"]),
            8.0e-13,
        );
        assert_slice_close(
            &rule.final_mu,
            &bit_array(&value["rules"]["final_mu"]),
            3.0e-13,
        );
        assert_slice_close(
            &rule.final_weights,
            &bit_array(&value["rules"]["final_weights"]),
            8.0e-13,
        );
        assert_slice_close(
            &rule.azimuth,
            &bit_array(&value["rules"]["azimuth"]),
            8.0e-16,
        );
        assert_slice_close(
            &rule.azimuth_weights,
            &bit_array(&value["rules"]["azimuth_weights"]),
            8.0e-16,
        );
        let (p2, weights) =
            electron_half_line_rule(48, bits(&value["temperature_gamma_bits"])).unwrap();
        assert_slice_close(&p2, &bit_array(&value["rules"]["electron_p2"]), 2.0e-12);
        assert_slice_close(
            &weights,
            &bit_array(&value["rules"]["electron_weights"]),
            3.0e-12,
        );
    }

    fn build_batch(name: &str) -> crate::f10_action_kinematics::F10KinematicBatch {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let p1 = bits(&value["kinematics"]["p1_bits"]);
        let t_cm = bits(&value["temperature_cm_bits"]);
        let t_gamma = bits(&value["temperature_gamma_bits"]);
        let mass = bits(&value["electron_mass_bits"]);
        let neutrino_p2: Vec<_> = grid.nodes.iter().map(|item| t_cm * item).collect();
        let neutrino_weights: Vec<_> = grid.weights.iter().map(|item| t_cm * item).collect();
        let (electron_p2, electron_weights) = electron_half_line_rule(48, t_gamma).unwrap();
        let (p2_nodes, p2_weights, masses) = match name {
            "self" => (&neutrino_p2[..], &neutrino_weights[..], [0.0, 0.0, 0.0]),
            "elastic" => (&electron_p2[..], &electron_weights[..], [mass, 0.0, mass]),
            "pair" => (&neutrino_p2[..], &neutrino_weights[..], [0.0, mass, mass]),
            other => panic!("unknown batch {other}"),
        };
        two_body_kinematics(F10KinematicInput {
            p1,
            p2_nodes,
            p2_weights,
            mass2: masses[0],
            mass3: masses[1],
            mass4: masses[2],
            config: F10CollisionConfig::default(),
        })
        .unwrap()
    }

    #[test]
    fn self_kinematics_matches_python_samples() {
        let value = fixture();
        assert_batch_matches_fixture(&build_batch("self"), batch_case(&value, "self"));
    }

    #[test]
    fn elastic_kinematics_matches_python_samples() {
        let value = fixture();
        assert_batch_matches_fixture(&build_batch("elastic"), batch_case(&value, "elastic"));
    }

    #[test]
    fn pair_kinematics_matches_python_samples() {
        let value = fixture();
        assert_batch_matches_fixture(&build_batch("pair"), batch_case(&value, "pair"));
    }

    #[test]
    fn foundations_fail_closed_and_mutations_are_detected() {
        let roundoff_actual = [-8.899_131_431_761_02e-16, 11.313_708_498_984_761];
        let roundoff_expected = [8.881_784_197_001_252e-16, 11.313_708_498_984_763];
        assert_slice_close(&roundoff_actual, &roundoff_expected, 2.0e-12);

        let material_mutation = std::panic::catch_unwind(|| {
            assert_slice_close(&[0.0, 1.0], &[0.0, 1.0 + 1.0e-8], 2.0e-12);
        });
        assert!(material_mutation.is_err());

        assert!(F10ActionGrid::affine_legendre(7, 8.0).is_err());
        assert!(F10ActionGrid::affine_legendre(8, -1.0).is_err());
        assert!(decode_cloglog_to_logit(f64::NAN).is_err());

        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        assert!(interpolate(&grid, &[0.0; 8], &[-0.1]).is_err());
        let wrong_nodes: Vec<_> = crate::quadrature::gauss_legendre_exponential_plain_rule(8, 3.0)
            .unwrap()
            .into_iter()
            .map(|item| item.0)
            .collect();
        assert!(
            wrong_nodes
                .iter()
                .zip(&grid.nodes)
                .map(|(left, right)| scaled_residual(*left, *right))
                .fold(0.0_f64, f64::max)
                > 1.0e-2
        );

        let value = fixture();
        let correct = build_batch("elastic");
        let massless = {
            let p1 = bits(&value["kinematics"]["p1_bits"]);
            let (p2, weights) =
                electron_half_line_rule(48, bits(&value["temperature_gamma_bits"])).unwrap();
            two_body_kinematics(F10KinematicInput {
                p1,
                p2_nodes: &p2,
                p2_weights: &weights,
                mass2: 0.0,
                mass3: 0.0,
                mass4: 0.0,
                config: F10CollisionConfig::default(),
            })
            .unwrap()
        };
        let index = batch_case(&value, "elastic")["samples"][0]["flat_index"]
            .as_u64()
            .unwrap() as usize;
        assert!(scaled_residual(correct.e2[index], massless.e2[index]) > 1.0e-5);
        assert!(
            two_body_kinematics(F10KinematicInput {
                p1: 0.0,
                p2_nodes: &[1.0],
                p2_weights: &[1.0],
                mass2: 0.0,
                mass3: 0.0,
                mass4: 0.0,
                config: F10CollisionConfig::default(),
            })
            .is_err()
        );
    }
}
