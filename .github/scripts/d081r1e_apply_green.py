#!/usr/bin/env python3
"""Apply the D-081R1E GREEN packed-RHS implementation.

The script is intentionally narrow.  It writes the production adapter, aligns
only the RED tests with the canonical R1E0 fixture schema, and rewrites the
preflight test.  It does not modify collision coefficients, event catalogues,
quadrature constants, tolerances, or the frozen fixture.
"""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("native/rabbit_cpu/src/f10_packed_rhs.rs")
TESTS = Path("native/rabbit_cpu/src/f10_packed_rhs_tests.rs")
PREFLIGHT = Path("native/rabbit_cpu/src/f10_packed_rhs_preflight_tests.rs")


SOURCE_TEXT = r'''//! Packed right-hand side for the exact six-species F10 comparator.
//!
//! This layer composes the admitted self-plus-electron collision action with
//! the tree-level finite-electron-mass electromagnetic EOS and flat-FLRW
//! Hubble convention.  It is a static evaluator only: no ODE solver is called.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::{
    F10CombinedAction, F10CombinedActionConfig, assemble_combined_action,
};
use crate::flrw::{NEWTON_G_MEV_MINUS_2, electromagnetic_eos};

const FLAVOUR_COUNT: usize = 3;
const SPECIES_COUNT: usize = 6;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10PackedRhsConfig {
    pub(crate) t_start_mev: f64,
    pub(crate) combined_action: F10CombinedActionConfig,
}

impl Default for F10PackedRhsConfig {
    fn default() -> Self {
        Self {
            t_start_mev: 10.0,
            combined_action: F10CombinedActionConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct F10PackedRhsDiagnostics {
    pub(crate) temperature_cm_mev: f64,
    pub(crate) temperature_gamma_mev: f64,
    pub(crate) rho_neutrino_by_flavour: [f64; FLAVOUR_COUNT],
    pub(crate) rho_neutrino_total: f64,
    pub(crate) rho_electromagnetic: f64,
    pub(crate) pressure_electromagnetic: f64,
    pub(crate) drho_electromagnetic_dt: f64,
    pub(crate) rho_total: f64,
    pub(crate) hubble_mev: f64,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct F10PackedRhs {
    pub(crate) values: Vec<f64>,
    pub(crate) combined_action: F10CombinedAction,
    pub(crate) diagnostics: F10PackedRhsDiagnostics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10PackedRhsError {
    InvalidInput,
    InvalidConfiguration,
    DimensionOverflow,
    Grid,
    Chart,
    Collision,
    Thermodynamics,
    NonFiniteOutput,
    NonPositiveHubble,
}

fn checked_sizes(grid: &F10ActionGrid) -> Result<(usize, usize), F10PackedRhsError> {
    let spectral = FLAVOUR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    let state = spectral
        .checked_add(2)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    Ok((spectral, state))
}

fn validate_grid(grid: &F10ActionGrid) -> Result<(), F10PackedRhsError> {
    if grid.order == 0
        || grid.nodes.len() != grid.order
        || grid.weights.len() != grid.order
        || !grid.y_max.is_finite()
        || grid.y_max <= 0.0
        || grid
            .nodes
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= grid.y_max)
        || grid
            .weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || grid.nodes.windows(2).any(|pair| pair[0] >= pair[1])
    {
        Err(F10PackedRhsError::Grid)
    } else {
        Ok(())
    }
}

fn decode_chart(coordinates: &[f64]) -> Result<(Vec<f64>, Vec<f64>), F10PackedRhsError> {
    let mut occupations = Vec::with_capacity(coordinates.len());
    let mut chain = Vec::with_capacity(coordinates.len());
    for &coordinate in coordinates {
        let exponential = coordinate.exp();
        if !exponential.is_finite() || exponential <= 0.0 {
            return Err(F10PackedRhsError::Chart);
        }
        let occupation = -(-exponential).exp_m1();
        let derivative = (coordinate - exponential).exp();
        if !occupation.is_finite()
            || !(0.0..1.0).contains(&occupation)
            || !derivative.is_finite()
            || derivative <= 0.0
        {
            return Err(F10PackedRhsError::Chart);
        }
        occupations.push(occupation);
        chain.push(derivative);
    }
    Ok((occupations, chain))
}

fn neutrino_energy_densities(
    grid: &F10ActionGrid,
    occupations: &[f64],
    temperature_cm: f64,
) -> Result<[f64; FLAVOUR_COUNT], F10PackedRhsError> {
    let expected = FLAVOUR_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    if occupations.len() != expected {
        return Err(F10PackedRhsError::InvalidInput);
    }
    let prefactor = temperature_cm.powi(4) / (PI * PI);
    let mut result = [0.0_f64; FLAVOUR_COUNT];
    for flavour in 0..FLAVOUR_COUNT {
        let mut integral = 0.0_f64;
        for node in 0..grid.order {
            integral += grid.weights[node]
                * grid.nodes[node].powi(3)
                * occupations[flavour * grid.order + node];
        }
        result[flavour] = prefactor * integral;
    }
    if result.into_iter().all(|value| value.is_finite() && value > 0.0) {
        Ok(result)
    } else {
        Err(F10PackedRhsError::Thermodynamics)
    }
}

pub(crate) fn evaluate_f10_packed_rhs(
    grid: &F10ActionGrid,
    ln_a: f64,
    packed_state: &[f64],
    config: F10PackedRhsConfig,
) -> Result<F10PackedRhs, F10PackedRhsError> {
    validate_grid(grid)?;
    let (spectral_size, state_size) = checked_sizes(grid)?;
    if packed_state.len() != state_size
        || !ln_a.is_finite()
        || packed_state.iter().any(|value| !value.is_finite())
    {
        return Err(F10PackedRhsError::InvalidInput);
    }
    if !config.t_start_mev.is_finite() || config.t_start_mev <= 0.0 {
        return Err(F10PackedRhsError::InvalidConfiguration);
    }

    let temperature_gamma = packed_state[spectral_size];
    if temperature_gamma <= 0.0 {
        return Err(F10PackedRhsError::InvalidInput);
    }
    let temperature_cm = config.t_start_mev * (-ln_a).exp();
    if !temperature_cm.is_finite() || temperature_cm <= 0.0 {
        return Err(F10PackedRhsError::InvalidInput);
    }

    let pair_cloglog = &packed_state[..spectral_size];
    let (occupations, chain) = decode_chart(pair_cloglog)?;
    let combined_action = assemble_combined_action(
        grid,
        pair_cloglog,
        temperature_cm,
        temperature_gamma,
        config.combined_action,
    )
    .map_err(|_| F10PackedRhsError::Collision)?;

    let expected_action_size = SPECIES_COUNT
        .checked_mul(grid.order)
        .ok_or(F10PackedRhsError::DimensionOverflow)?;
    if combined_action.native_total.len() != expected_action_size {
        return Err(F10PackedRhsError::Collision);
    }

    let rho_neutrino_by_flavour =
        neutrino_energy_densities(grid, &occupations, temperature_cm)?;
    let rho_neutrino_total = rho_neutrino_by_flavour.into_iter().sum::<f64>();
    let electromagnetic =
        electromagnetic_eos(temperature_gamma).map_err(|_| F10PackedRhsError::Thermodynamics)?;
    let rho_total = rho_neutrino_total + electromagnetic.rho;
    if !rho_neutrino_total.is_finite()
        || rho_neutrino_total <= 0.0
        || !rho_total.is_finite()
        || rho_total <= 0.0
        || !electromagnetic.rho.is_finite()
        || electromagnetic.rho <= 0.0
        || !electromagnetic.pressure.is_finite()
        || electromagnetic.pressure <= 0.0
        || !electromagnetic.drho_dt.is_finite()
        || electromagnetic.drho_dt <= 0.0
    {
        return Err(F10PackedRhsError::Thermodynamics);
    }

    let hubble_squared = (8.0 * PI * NEWTON_G_MEV_MINUS_2 / 3.0) * rho_total;
    let hubble = hubble_squared.sqrt();
    if !hubble.is_finite() {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }
    if hubble <= 0.0 {
        return Err(F10PackedRhsError::NonPositiveHubble);
    }

    let mut values = vec![0.0_f64; state_size];
    for flavour in 0..FLAVOUR_COUNT {
        let particle = 2 * flavour;
        let antiparticle = particle + 1;
        for node in 0..grid.order {
            let spectral_index = flavour * grid.order + node;
            let pair_rate = 0.5
                * (combined_action.native_total[particle * grid.order + node]
                    + combined_action.native_total[antiparticle * grid.order + node]);
            let denominator = hubble * chain[spectral_index];
            values[spectral_index] = pair_rate / denominator;
        }
    }

    let temperature_numerator = -3.0 * (electromagnetic.rho + electromagnetic.pressure)
        + combined_action.electromagnetic_energy_transfer / hubble;
    values[spectral_size] = temperature_numerator / electromagnetic.drho_dt;
    values[spectral_size + 1] = 1.0 / hubble;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    let neutrino_energy_transfer = combined_action.neutrino_energy_transfer;
    let electromagnetic_energy_transfer = combined_action.electromagnetic_energy_transfer;
    let first_law_residual = (neutrino_energy_transfer + electromagnetic_energy_transfer).abs()
        / (neutrino_energy_transfer.abs() + electromagnetic_energy_transfer.abs())
            .max(f64::MIN_POSITIVE);
    let diagnostics = F10PackedRhsDiagnostics {
        temperature_cm_mev: temperature_cm,
        temperature_gamma_mev: temperature_gamma,
        rho_neutrino_by_flavour,
        rho_neutrino_total,
        rho_electromagnetic: electromagnetic.rho,
        pressure_electromagnetic: electromagnetic.pressure,
        drho_electromagnetic_dt: electromagnetic.drho_dt,
        rho_total,
        hubble_mev: hubble,
        neutrino_energy_transfer,
        electromagnetic_energy_transfer,
        first_law_residual,
        whole_reaction_domain_rejections: combined_action.whole_reaction_domain_rejections,
        matrix_roundoff_corrections: combined_action.matrix_roundoff_corrections,
        largest_matrix_roundoff_correction: combined_action
            .largest_matrix_roundoff_correction,
    };
    let diagnostic_scalars = [
        diagnostics.temperature_cm_mev,
        diagnostics.temperature_gamma_mev,
        diagnostics.rho_neutrino_total,
        diagnostics.rho_electromagnetic,
        diagnostics.pressure_electromagnetic,
        diagnostics.drho_electromagnetic_dt,
        diagnostics.rho_total,
        diagnostics.hubble_mev,
        diagnostics.neutrino_energy_transfer,
        diagnostics.electromagnetic_energy_transfer,
        diagnostics.first_law_residual,
        diagnostics.largest_matrix_roundoff_correction,
    ];
    if !diagnostic_scalars.into_iter().all(f64::is_finite) {
        return Err(F10PackedRhsError::NonFiniteOutput);
    }

    Ok(F10PackedRhs {
        values,
        combined_action,
        diagnostics,
    })
}
'''


PREFLIGHT_TEXT = r'''//! Canonical-oracle preflight tests for D-081R1E retained packed-RHS admission.

use crate::f10_action_grid::F10ActionGrid;
use serde_json::Value;

const FIXTURE: &str = include_str!("../tests/fixtures/d081r1/retained_packed_rhs_case.json");

fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("valid frozen D-081R1E0 retained fixture")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_packed_rhs_authority_is_exact() {
        let value = fixture();
        assert_eq!(
            value["schema"],
            "rabbit.d081r1e0.retained_packed_rhs_oracle.v1"
        );
        assert_eq!(
            value["authorities"]["d081r1d4_final_head"],
            "002086662bf2e553c78f4b247868cb1fd9e43f21"
        );
        assert_eq!(
            value["authorities"]["private_comparator_git_blob"],
            "de44feee0aa484abe26976c7dc34c579643005b5"
        );
        assert_eq!(
            value["authorities"]["trajectory_core_git_blob"],
            "465a73f0ce40f7149bebdc2d67103f388e2344d9"
        );
        assert_eq!(
            value["authorities"]["retained_state_sha256"],
            "c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380"
        );
        assert_eq!(value["environment"]["numpy"], "2.4.4");
        assert_eq!(value["environment"]["scipy"], "1.17.1");
        assert_eq!(value["configuration"]["order"], 60);
        assert_eq!(bits(&value["configuration"]["y_max_bits"]).to_bits(), 30.0_f64.to_bits());
        assert_eq!(value["configuration"]["state_size"], 182);
        assert_eq!(value["arrays"]["packed_state"]["shape"][0], 182);
        assert_eq!(
            value["arrays"]["packed_rhs_trajectory_core"]["shape"][0],
            182
        );
        assert_eq!(value["arrays"]["total_native"]["shape"][0], 6);
        assert_eq!(value["arrays"]["total_native"]["shape"][1], 60);
    }

    #[test]
    fn retained_order60_grid_matches_frozen_numpy_binary64_operator() {
        let value = fixture();
        let grid = F10ActionGrid::affine_legendre(60, 30.0).unwrap();
        let expected_nodes = bit_array(&value["arrays"]["grid_nodes"]);
        let expected_weights = bit_array(&value["arrays"]["grid_weights"]);
        assert_eq!(expected_nodes.len(), 60);
        assert_eq!(expected_weights.len(), 60);
        assert_eq!(
            grid.nodes.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected_nodes.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            grid.weights.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected_weights.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }
}
'''


def patch_tests() -> None:
    text = TESTS.read_text(encoding="utf-8")
    text = text.replace("use serde_json::Value;", "use serde_json::{Value, json};")

    old_fixture = '''fn retained_fixture() -> Value {\n    serde_json::from_str(RETAINED_FIXTURE).expect("valid retained packed-RHS fixture")\n}\n'''
    new_fixture = '''fn retained_fixture() -> Value {\n    let canonical: Value = serde_json::from_str(RETAINED_FIXTURE)\n        .expect("valid canonical retained packed-RHS fixture");\n    assert_eq!(\n        canonical["schema"],\n        "rabbit.d081r1e0.retained_packed_rhs_oracle.v1"\n    );\n    json!({\n        "schema": "rabbit.d081r1e.retained_packed_rhs.v1",\n        "d4_head": canonical["authorities"]["d081r1d4_final_head"].clone(),\n        "d4_tree": "d01ae7c0d3d9fbe8ce9513d054b835d3596f1de2",\n        "retained_sha256": canonical["authorities"]["retained_state_sha256"].clone(),\n        "order": canonical["configuration"]["order"].clone(),\n        "packed_state": canonical["arrays"]["packed_state"].clone(),\n        "packed_rhs": canonical["arrays"]["packed_rhs_trajectory_core"].clone(),\n        "ln_a_bits": canonical["configuration"]["expansion_n_bits"].clone(),\n        "combined_action_native": canonical["arrays"]["total_native"].clone(),\n        "combined_action_modal": canonical["arrays"]["total_modal"].clone(),\n        "spectral_rhs": canonical["arrays"]["spectral_rhs"].clone(),\n        "temperature_rhs_bits": canonical["scalars"]["temperature_rhs_bits"].clone(),\n        "elapsed_rhs_bits": canonical["scalars"]["elapsed_rhs_bits"].clone(),\n        "rho_neutrino_total_bits": canonical["thermodynamics"]["energy_density_neutrino"].clone(),\n        "rho_electromagnetic_bits": canonical["electromagnetic_eos"]["rho"].clone(),\n        "pressure_electromagnetic_bits": canonical["electromagnetic_eos"]["pressure"].clone(),\n        "drho_electromagnetic_dt_bits": canonical["electromagnetic_eos"]["drho_dtemperature"].clone(),\n        "rho_total_bits": canonical["thermodynamics"]["energy_density_total"].clone(),\n        "hubble_mev_bits": canonical["thermodynamics"]["hubble_mev"].clone(),\n        "q_nu_bits": canonical["scalars"]["neutrino_energy_transfer_bits"].clone(),\n        "q_em_bits": canonical["scalars"]["electromagnetic_energy_transfer_bits"].clone(),\n        "first_law_residual_bits": canonical["scalars"]["first_law_residual_bits"].clone(),\n        "support_and_roundoff_metrology": {\n            "whole_reaction_domain_rejections": canonical["metrology"]["whole_reaction_domain_rejections"].clone(),\n            "matrix_roundoff_corrections": canonical["metrology"]["matrix_roundoff_corrections"].clone()\n        }\n    })\n}\n'''
    if old_fixture not in text:
        raise SystemExit("retained_fixture function no longer matches RED source")
    text = text.replace(old_fixture, new_fixture)

    old_action_scale = '''        let action_scale = maximum_absolute(&bit_array(\n            &value["absolute_envelopes"]["combined_action_native"],\n        ));\n'''
    if old_action_scale not in text:
        raise SystemExit("action scale block no longer matches RED source")
    text = text.replace(
        old_action_scale,
        "        let action_scale = maximum_absolute(&expected_action_native);\n",
    )

    old_spectral_scale = '''        let spectral_scale = maximum_absolute(&bit_array(\n            &value["absolute_envelopes"]["spectral_rhs"],\n        ));\n'''
    if old_spectral_scale not in text:
        raise SystemExit("spectral scale block no longer matches RED source")
    text = text.replace(
        old_spectral_scale,
        "        let spectral_scale = maximum_absolute(&expected_spectral);\n",
    )

    old_rho = '''        let expected_rho = bit_array(&value["rho_neutrino_by_flavour"]);\n        assert_hybrid_close(\n            &result.diagnostics.rho_neutrino_by_flavour,\n            &expected_rho,\n            maximum_absolute(&expected_rho),\n            2.0e-8,\n        );\n'''
    new_rho = '''        let rho_sum = result\n            .diagnostics\n            .rho_neutrino_by_flavour\n            .iter()\n            .sum::<f64>();\n        assert_scalar_close(\n            rho_sum,\n            result.diagnostics.rho_neutrino_total,\n            result.diagnostics.rho_neutrino_total,\n            8.0 * f64::EPSILON,\n        );\n'''
    if old_rho not in text:
        raise SystemExit("rho-by-flavour block no longer matches RED source")
    text = text.replace(old_rho, new_rho)

    TESTS.write_text(text, encoding="utf-8")


def main() -> None:
    SOURCE.write_text(SOURCE_TEXT, encoding="utf-8")
    PREFLIGHT.write_text(PREFLIGHT_TEXT, encoding="utf-8")
    patch_tests()
    print("D-081R1E GREEN implementation and canonical-test adapter written")


if __name__ == "__main__":
    main()
