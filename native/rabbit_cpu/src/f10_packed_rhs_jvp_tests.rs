//! RED-first tests for D-081R1F0 Rust c-only packed-RHS JVP.
//!
//! The first commit intentionally names an absent production API. The sealed
//! workflow must record the resulting compile failure before implementation.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_packed_rhs::F10PackedRhsConfig;
use crate::f10_packed_rhs_jvp::evaluate_f10_packed_rhs_c_jvp;
use serde_json::Value;

const CONTROL_FIXTURE: &str =
    include_str!("../tests/fixtures/d081r1/full_collision_action_case.json");

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

fn named_case<'a>(value: &'a Value, name: &str) -> &'a Value {
    value["cases"]
        .as_array()
        .expect("fixture cases")
        .iter()
        .find(|case| case["name"] == name)
        .expect("named fixture case")
}

#[test]
fn zero_c_direction_is_bitwise_zero_without_changing_the_base_rhs() {
    let fixture: Value = serde_json::from_str(CONTROL_FIXTURE).expect("valid control fixture");
    let case = named_case(&fixture, "thermal_split");
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let temperature_cm = bits(&case["temperature_cm_bits"]);
    let temperature_gamma = bits(&case["temperature_gamma_bits"]);
    let mut state = bit_array(&case["pair_cloglog"]);
    state.push(temperature_gamma);
    state.push(0.0);
    let ln_a = (10.0 / temperature_cm).ln();
    let direction = vec![0.0; 3 * grid.order];

    let result = evaluate_f10_packed_rhs_c_jvp(
        &grid,
        ln_a,
        &state,
        &direction,
        F10PackedRhsConfig::default(),
    )
    .unwrap();

    assert_eq!(result.base.values.len(), 3 * grid.order + 2);
    assert_eq!(result.values.len(), 3 * grid.order + 2);
    assert!(result.values.iter().all(|value| value.to_bits() == 0));
    assert!(
        result
            .delta_rho_neutrino_by_flavour
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert_eq!(result.delta_rho_neutrino.to_bits(), 0);
    assert_eq!(result.delta_hubble_over_hubble.to_bits(), 0);
    assert_eq!(result.delta_neutrino_energy_transfer.to_bits(), 0);
    assert_eq!(result.delta_electromagnetic_energy_transfer.to_bits(), 0);
    assert_eq!(result.first_law_tangent_residual.to_bits(), 0);

    let combined = &result.combined_action;
    assert!(
        combined
            .modal_total
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(
        combined
            .native_total
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    for value in [
        combined.moments.signed_number_rate,
        combined.moments.absolute_number_rate,
        combined.moments.signed_energy_rate,
        combined.moments.absolute_energy_rate,
        combined.neutrino_energy_transfer,
        combined.electromagnetic_energy_transfer,
        combined.first_law_residual,
        combined.self_event_energy_residual,
        combined.self_event_energy_relative_residual,
        combined.charge_conjugation_residual,
        combined.mu_tau_residual,
    ] {
        assert_eq!(value.to_bits(), 0);
    }

    let self_action = &combined.self_action;
    assert!(self_action.modal.iter().all(|value| value.to_bits() == 0));
    assert!(self_action.native.iter().all(|value| value.to_bits() == 0));
    for value in [
        self_action.moments.signed_number_rate,
        self_action.moments.absolute_number_rate,
        self_action.moments.signed_energy_rate,
        self_action.moments.absolute_energy_rate,
        self_action.event_energy_residual,
        self_action.event_energy_absolute,
    ] {
        assert_eq!(value.to_bits(), 0);
    }

    let electron_action = &combined.electron_action;
    assert!(
        electron_action
            .modal
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(
        electron_action
            .native
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    for value in [
        electron_action.moments.signed_number_rate,
        electron_action.moments.absolute_number_rate,
        electron_action.moments.signed_energy_rate,
        electron_action.moments.absolute_energy_rate,
        electron_action.neutrino_energy_transfer,
        electron_action.electromagnetic_energy_transfer,
        electron_action.first_law_residual,
    ] {
        assert_eq!(value.to_bits(), 0);
    }
}
