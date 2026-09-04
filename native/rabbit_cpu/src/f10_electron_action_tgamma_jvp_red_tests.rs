//! RED-first contract for the electron collision-action `T_gamma` JVP.
//!
//! The production child module imported below is intentionally absent at the
//! frozen RED point. The eventual implementation must satisfy these structural
//! requirements before numerical oracle and centered-witness tests are added.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_electron_action::{F10ElectronActionConfig, tgamma_jvp::{
    F10ElectronActionTgammaJvpError, assemble_electron_action_tgamma_jvp,
}};

fn equilibrium_pair_cloglog(grid: &F10ActionGrid) -> Vec<f64> {
    let mut coordinates = Vec::with_capacity(3 * grid.order);
    for _flavour in 0..3 {
        for &y in &grid.nodes {
            let occupation = 1.0 / (1.0 + y.exp());
            let coordinate = (-(1.0 - occupation).ln()).ln();
            assert!(coordinate.is_finite());
            coordinates.push(coordinate);
        }
    }
    coordinates
}

fn all_bitwise_zero(values: &[f64]) -> bool {
    values.iter().all(|value| value.to_bits() == 0.0_f64.to_bits())
}

fn relative_component_residual(total: &[f64], components: &[&[f64]]) -> f64 {
    assert!(components.iter().all(|component| component.len() == total.len()));
    let mut maximum_difference = 0.0_f64;
    let mut maximum_scale = f64::MIN_POSITIVE;
    for index in 0..total.len() {
        let sum = components.iter().map(|component| component[index]).sum::<f64>();
        maximum_difference = maximum_difference.max((total[index] - sum).abs());
        maximum_scale = maximum_scale.max(total[index].abs()).max(sum.abs());
    }
    maximum_difference / maximum_scale
}

#[test]
fn p1_collision_thermal_jvp_exposes_load_bearing_component_and_pair_zero_structure() {
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let coordinates = equilibrium_pair_cloglog(&grid);
    let result = assemble_electron_action_tgamma_jvp(
        &grid,
        &coordinates,
        2.0,
        2.05,
        F10ElectronActionConfig::default(),
    )
    .unwrap();

    let species_modes = 6 * grid.order;
    for values in [
        result.modal.as_slice(),
        result.measure_modal.as_slice(),
        result.matrix_modal.as_slice(),
        result.pauli_modal.as_slice(),
        result.projection_modal.as_slice(),
        result.elastic_modal.as_slice(),
        result.pair_modal.as_slice(),
        result.pair_measure_modal.as_slice(),
        result.pair_matrix_modal.as_slice(),
        result.pair_projection_modal.as_slice(),
    ] {
        assert_eq!(values.len(), species_modes);
        assert!(values.iter().all(|value| value.is_finite()));
    }
    assert_eq!(result.native.len(), species_modes);
    assert!(result.native.iter().all(|value| value.is_finite()));

    assert!(
        relative_component_residual(
            &result.modal,
            &[
                &result.measure_modal,
                &result.matrix_modal,
                &result.pauli_modal,
                &result.projection_modal,
            ],
        ) <= 2.0e-12
    );
    assert!(
        relative_component_residual(
            &result.modal,
            &[&result.elastic_modal, &result.pair_modal],
        ) <= 2.0e-12
    );

    assert!(all_bitwise_zero(&result.pair_measure_modal));
    assert!(all_bitwise_zero(&result.pair_matrix_modal));
    assert!(all_bitwise_zero(&result.pair_projection_modal));

    assert_eq!(result.family_names.len(), 15);
    assert_eq!(result.family_modal.len(), 15 * species_modes);
    assert!(result.family_modal.iter().all(|value| value.is_finite()));
    assert_eq!(result.bath_energy_tangent_by_family.len(), 15);
    assert!(
        result
            .bath_energy_tangent_by_family
            .iter()
            .all(|value| value.is_finite())
    );

    for value in [
        result.rate_weight_neutrino_energy_tangent,
        result.rate_weight_electromagnetic_energy_tangent,
        result.kinematic_weight_neutrino_energy_tangent,
        result.kinematic_weight_electromagnetic_energy_tangent,
        result.neutrino_energy_transfer_tangent,
        result.electromagnetic_energy_transfer_tangent,
        result.first_law_tangent_residual,
        result.charge_conjugation_residual,
        result.mu_tau_residual,
        result.minimum_support_margin_relative,
        result.minimum_supported_lambda_margin_relative,
    ] {
        assert!(value.is_finite());
    }
    assert!(result.minimum_support_margin_relative > 0.0);
    assert!(result.minimum_supported_lambda_margin_relative > 0.0);
    assert!(result.first_law_tangent_residual <= 2.0e-9);
}

#[test]
fn p1_collision_thermal_jvp_fails_closed_on_invalid_input() {
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let coordinates = equilibrium_pair_cloglog(&grid);

    let mut short = coordinates.clone();
    short.pop();
    assert_eq!(
        assemble_electron_action_tgamma_jvp(
            &grid,
            &short,
            2.0,
            2.05,
            F10ElectronActionConfig::default(),
        )
        .unwrap_err(),
        F10ElectronActionTgammaJvpError::InvalidInput
    );

    assert_eq!(
        assemble_electron_action_tgamma_jvp(
            &grid,
            &coordinates,
            2.0,
            f64::NAN,
            F10ElectronActionConfig::default(),
        )
        .unwrap_err(),
        F10ElectronActionTgammaJvpError::InvalidInput
    );
}
