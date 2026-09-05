//! Native CPU kernels for the bounded RABBIT Rust migration.

use pyo3::prelude::*;

mod born_freezeout;
mod born_weak;
mod electron_catalog;
mod electron_event;
mod electron_hm;
mod electron_thermal;
mod electron_thermal_fd;
mod f10_action_grid;
mod f10_action_kinematics;
mod f10_action_spectral;
mod f10_action_tangent;
mod f10_combined_action;
mod f10_combined_action_jvp;
mod f10_elastic_prefactor_tangent;
mod f10_electron_action;
mod f10_kernel_primitives;
mod f10_packed_rhs;
mod f10_packed_rhs_jvp;
mod f10_self_action;
mod f10_tgamma_kinematics;
mod f10_tgamma_tangent;
mod flrw;
mod isotropic_boltzmann;
mod minimal_bbn;
mod minimal_network;
mod ode;
mod pauli_edge_step;
mod qed_eos;
mod quadrature;
mod thermal_bbn;
mod thermal_weak;
mod tier1_lrs;

#[cfg(test)]
mod f10_action_foundations_tests;

#[cfg(test)]
mod f10_combined_action_tests;

#[cfg(test)]
mod f10_electron_action_tests;

#[cfg(test)]
mod f10_electron_action_tgamma_jvp_red_tests;

#[cfg(test)]
mod f10_elastic_d080b_direct_tests;

#[cfg(test)]
mod f10_elastic_prefactor_tests;

#[cfg(test)]
mod f10_packed_rhs_preflight_tests;

#[cfg(test)]
mod f10_packed_rhs_holdout_tests;

#[cfg(test)]
mod f10_packed_rhs_jvp_order8_tests;

#[cfg(test)]
mod f10_packed_rhs_jvp_retained_calibration_tests;

#[cfg(test)]
mod f10_packed_rhs_jvp_retained_holdout_tests;

#[cfg(test)]
mod f10_packed_rhs_jvp_tests;

#[cfg(test)]
mod f10_packed_rhs_tests;

#[cfg(test)]
mod f10_self_action_tests;

#[cfg(test)]
mod f10_tgamma_tangent_tests;

#[cfg(test)]
mod f10_tgamma_kinematic_tests;

#[cfg(test)]
mod f10_tgamma_adversarial_repair_tests;

#[cfg(test)]
mod electron_event_falsifiers;

#[cfg(test)]
mod electron_catalog_falsifiers;

#[cfg(test)]
mod electron_hm_falsifiers;

mod electron_phase_point;
#[cfg(test)]
mod electron_phase_point_falsifiers;

mod electron_supplied;
#[cfg(test)]
mod electron_supplied_falsifiers;

mod electron_response;
#[cfg(test)]
mod electron_response_falsifiers;

mod electron_spectral;
mod neutrino_self_spectral;

use tier1_lrs::NativeTier1Core;

#[pymodule]
fn _rabbit_cpu(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTier1Core>()?;
    Ok(())
}
