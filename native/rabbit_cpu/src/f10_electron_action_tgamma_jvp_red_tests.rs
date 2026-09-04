//! Minimal absent-child-module RED for the electron collision-action
//! `T_gamma` JVP.
//!
//! The detailed component, conservation, branch, and failure contracts are
//! frozen in the P1 audit document. This file deliberately checks only that
//! the intended typed production API is absent, so no secondary type-inference
//! errors can contaminate the compiler RED witness.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_electron_action::{
    F10ElectronActionConfig,
    tgamma_jvp::{
        F10ElectronActionTgammaJvp, F10ElectronActionTgammaJvpError,
        assemble_electron_action_tgamma_jvp,
    },
};

type P1CollisionTgammaApi = fn(
    &F10ActionGrid,
    &[f64],
    f64,
    f64,
    F10ElectronActionConfig,
) -> Result<F10ElectronActionTgammaJvp, F10ElectronActionTgammaJvpError>;

#[test]
fn p1_collision_tgamma_jvp_child_module_api_is_absent_at_red() {
    let api: P1CollisionTgammaApi = assemble_electron_action_tgamma_jvp;
    let _ = api;
}
