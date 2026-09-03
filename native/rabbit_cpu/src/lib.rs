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
mod f10_kernel_primitives;
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
mod f10_self_action_tests;

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
