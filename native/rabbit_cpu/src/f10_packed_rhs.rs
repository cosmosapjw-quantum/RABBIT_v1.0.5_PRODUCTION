//! Retained order-60 packed-RHS adapter for the exact six-species F10 comparator.
//!
//! RED-first placeholder: the API and failure surface are frozen before the
//! implementation is admitted. The GREEN implementation must compose the
//! already admitted combined collision action with the existing tree-level
//! finite-electron-mass FLRW thermodynamics. It must not call an ODE solver.

#![cfg_attr(not(test), allow(dead_code))]

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_combined_action::{F10CombinedAction, F10CombinedActionConfig};

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
    pub(crate) rho_neutrino_by_flavour: [f64; 3],
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
    NotImplemented,
}

pub(crate) fn evaluate_f10_packed_rhs(
    _grid: &F10ActionGrid,
    _ln_a: f64,
    _packed_state: &[f64],
    _config: F10PackedRhsConfig,
) -> Result<F10PackedRhs, F10PackedRhsError> {
    Err(F10PackedRhsError::NotImplemented)
}
