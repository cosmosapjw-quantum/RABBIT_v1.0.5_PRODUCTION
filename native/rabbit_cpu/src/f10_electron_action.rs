//! Electron/positron collision-action contract for D-081R1D3.
//!
//! The RED commit freezes the complete return and failure surface before the
//! finite-electron-mass elastic and pair assemblers are admitted.

#![cfg_attr(not(test), allow(dead_code))]

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::F10CollisionConfig;

pub(crate) const F10_ELECTRON_MASS_MEV: f64 = 0.510_998_95;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10ElectronActionConfig {
    pub(crate) collision: F10CollisionConfig,
    pub(crate) matrix_roundoff_ulps: f64,
    pub(crate) electron_mass_mev: f64,
}

impl Default for F10ElectronActionConfig {
    fn default() -> Self {
        Self {
            collision: F10CollisionConfig::default(),
            matrix_roundoff_ulps: 1024.0,
            electron_mass_mev: F10_ELECTRON_MASS_MEV,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct F10ElectronActionMoments {
    pub(crate) signed_number_rate: f64,
    pub(crate) absolute_number_rate: f64,
    pub(crate) signed_energy_rate: f64,
    pub(crate) absolute_energy_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct F10ElectronAction {
    pub(crate) modal: Vec<f64>,
    pub(crate) native: Vec<f64>,
    pub(crate) elastic_modal: Vec<f64>,
    pub(crate) pair_modal: Vec<f64>,
    pub(crate) elastic_native: Vec<f64>,
    pub(crate) pair_native: Vec<f64>,
    pub(crate) family_names: Vec<String>,
    pub(crate) family_modal: Vec<f64>,
    pub(crate) family_native: Vec<f64>,
    pub(crate) bath_energy_by_family: Vec<f64>,
    pub(crate) moments: F10ElectronActionMoments,
    pub(crate) whole_reaction_domain_rejections: usize,
    pub(crate) elastic_domain_rejections: usize,
    pub(crate) pair_domain_rejections: usize,
    pub(crate) matrix_roundoff_corrections: usize,
    pub(crate) largest_matrix_roundoff_correction: f64,
    pub(crate) neutrino_energy_transfer: f64,
    pub(crate) electromagnetic_energy_transfer: f64,
    pub(crate) first_law_residual: f64,
    pub(crate) neutrino_h_rate: f64,
    pub(crate) electromagnetic_h_rate: f64,
    pub(crate) entropy_production: f64,
    pub(crate) node_neutrino_h_rate: f64,
    pub(crate) entropy_duality_residual: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10ElectronActionError {
    InvalidInput,
    InvalidConfiguration,
    DimensionOverflow,
    Foundation,
    Kinematics,
    Kernel,
    NonFiniteOutput,
    NotImplemented,
}

pub(crate) fn assemble_electron_action(
    _grid: &F10ActionGrid,
    _pair_cloglog: &[f64],
    _temperature_cm: f64,
    _temperature_gamma: f64,
    _config: F10ElectronActionConfig,
) -> Result<F10ElectronAction, F10ElectronActionError> {
    Err(F10ElectronActionError::NotImplemented)
}
