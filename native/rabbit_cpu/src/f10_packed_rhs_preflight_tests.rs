//! RED-first preflight tests for D-081R1E retained packed-RHS admission.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::{F10CollisionConfig, angular_rule, electron_half_line_rule};
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/fixtures/d081r1/retained_packed_rhs_case.json");
const INNER_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/f10_inner_quadrature_case.json",);

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("valid frozen D-081R1E retained fixture")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_packed_rhs_authority_is_exact() {
        let value = fixture();
        assert_eq!(value["schema"], "rabbit.d081r1e.retained_packed_rhs.v1");
        assert_eq!(value["d4_head"], "002086662bf2e553c78f4b247868cb1fd9e43f21");
        assert_eq!(value["d4_tree"], "d01ae7c0d3d9fbe8ce9513d054b835d3596f1de2");
        assert_eq!(
            value["python_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(
            value["trajectory_core_git_blob"],
            "465a73f0ce40f7149bebdc2d67103f388e2344d9"
        );
        assert_eq!(
            value["cargo_lock_git_blob"],
            "a1b5035da5c20712d1a2a4ab077da255ff94a014"
        );
        assert_eq!(
            value["retained_sha256"],
            "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
        );
        assert_eq!(value["numpy_version"], "2.4.4");
        assert_eq!(value["scipy_version"], "1.17.1");
        assert_eq!(value["order"], 60);
        assert_eq!(bits(&value["y_max_bits"]).to_bits(), 30.0_f64.to_bits());
        assert_eq!(value["packed_state"]["shape"][0], 182);
        assert_eq!(value["packed_rhs"]["shape"][0], 182);
        assert_eq!(value["combined_action_native"]["shape"][0], 6);
        assert_eq!(value["combined_action_native"]["shape"][1], 60);
    }

    #[test]
    fn retained_order60_grid_matches_frozen_numpy_binary64_operator() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(60, 30.0).unwrap();
        let expected_nodes = bit_array(&value["grid_nodes"]);
        let expected_weights = bit_array(&value["grid_weights"]);
        assert_eq!(expected_nodes.len(), 60);
        assert_eq!(expected_weights.len(), 60);

        let actual_node_bits: Vec<_> = grid.nodes.iter().map(|value| value.to_bits()).collect();
        let expected_node_bits: Vec<_> =
            expected_nodes.iter().map(|value| value.to_bits()).collect();
        let actual_weight_bits: Vec<_> = grid.weights.iter().map(|value| value.to_bits()).collect();
        let expected_weight_bits: Vec<_> = expected_weights
            .iter()
            .map(|value| value.to_bits())
            .collect();

        assert_eq!(
            actual_node_bits, expected_node_bits,
            "generic Rust GL60/Y30 nodes differ from the frozen NumPy operator"
        );
        assert_eq!(
            actual_weight_bits, expected_weight_bits,
            "generic Rust GL60/Y30 weights differ from the frozen NumPy operator"
        );
    }

    #[test]
    fn f10_inner_gl12_gl48_rules_match_frozen_numpy_binary64_operator() {
        let value: Value =
            serde_json::from_str(INNER_FIXTURE).expect("valid frozen inner quadrature fixture");
        assert_eq!(value["schema"], "rabbit.d081r1e.f10_inner_quadrature.v1");
        assert_eq!(value["numpy_version"], "2.4.4");
        assert_eq!(
            value["python_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );

        let angular = angular_rule(F10CollisionConfig::default()).unwrap();
        let expected_gl12_nodes = bit_array(&value["gl12_nodes"]);
        let expected_gl12_weights = bit_array(&value["gl12_weights"]);
        assert_eq!(
            angular
                .incoming_mu
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_nodes
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 incoming GL12 nodes differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .incoming_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 incoming GL12 weights differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .final_mu
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_nodes
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 final-state GL12 nodes differ from the frozen NumPy operator"
        );
        assert_eq!(
            angular
                .final_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_gl12_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 final-state GL12 weights differ from the frozen NumPy operator"
        );

        let temperature = bits(&value["temperature_probe_bits"]);
        let (electron_p2, electron_weights) = electron_half_line_rule(48, temperature).unwrap();
        let expected_p2 = bit_array(&value["electron_p2"]);
        let expected_electron_weights = bit_array(&value["electron_weights"]);
        assert_eq!(
            electron_p2
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_p2
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 GL48 electron momenta differ from the frozen NumPy operator"
        );
        assert_eq!(
            electron_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            expected_electron_weights
                .iter()
                .map(|item| item.to_bits())
                .collect::<Vec<_>>(),
            "F10 GL48 electron weights differ from the frozen NumPy operator"
        );
    }
}
