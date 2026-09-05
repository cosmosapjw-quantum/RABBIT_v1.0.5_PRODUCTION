//! Fixed-branch elastic measure and weak-matrix temperature tangents.
//! These are prefactors only: Pauli, projection and action assembly are absent.
//! Primal values come from the admitted kernels without changing their algebra.

#![cfg_attr(not(test), allow(dead_code))]

use crate::electron_hm::{G_F_MEV_MINUS_2, SIN2_THETA_W};
use crate::f10_kernel_primitives::{
    F10ElectronCategory, F10EventMeasureInput, F10Flavour, F10InvariantProducts, F10KernelError,
    F10MatrixValue, F10Species, f10_electron_matrix, f10_event_measure,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10MeasureTangent {
    pub(crate) d_p2: f64,
    pub(crate) d_e2: f64,
    pub(crate) d_phase_space: f64,
    pub(crate) d_quadrature_weight: f64,
}

/// Analytic derivative at fixed target momentum and outer quadrature weight.
/// No division by p2, phase space or quadrature weight: their zeros are allowed.
pub(crate) fn event_measure_tangent(
    input: F10EventMeasureInput,
    tangent: F10MeasureTangent,
) -> Result<(f64, f64), F10KernelError> {
    let base = f10_event_measure(input)?;
    if [
        tangent.d_p2,
        tangent.d_e2,
        tangent.d_phase_space,
        tangent.d_quadrature_weight,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err(F10KernelError::NonFiniteInput);
    }
    let weighted_outer = input.outer_weight * input.quadrature_weight;
    let d_weighted_outer = input.outer_weight * tangent.d_quadrature_weight;
    let p2_squared = input.p2.powi(2);
    let d_p2_squared = 2.0 * input.p2 * tangent.d_p2;
    let d_numerator = (d_weighted_outer * p2_squared + weighted_outer * d_p2_squared)
        * input.phase_space
        + weighted_outer * p2_squared * tangent.d_phase_space;
    let denominator = input.e2 * 256.0 * core::f64::consts::PI.powi(4) * input.p1;
    let d_denominator = tangent.d_e2 * 256.0 * core::f64::consts::PI.powi(4) * input.p1;
    let derivative = (d_numerator - base * d_denominator) / denominator;
    if derivative.is_finite() {
        Ok((base, derivative))
    } else {
        Err(F10KernelError::NonFiniteInput)
    }
}

/// Fixed-mass elastic derivative with the primal CP/correction convention.
/// A corrected or unsupported primal branch has exactly zero tangent.
/// Branch identity across a finite witness is checked by the caller, not smoothed.
pub(crate) fn elastic_matrix_tangent(
    target: F10Species,
    category: F10ElectronCategory,
    invariants: F10InvariantProducts,
    tangent: F10InvariantProducts,
    electron_mass: f64,
    support: bool,
    roundoff_ulps: f64,
) -> Result<(F10MatrixValue, f64), F10KernelError> {
    if category == F10ElectronCategory::Pair {
        return Err(F10KernelError::UnknownCategory);
    }
    if [
        tangent.d12,
        tangent.d13,
        tangent.d14,
        tangent.d23,
        tangent.d24,
        tangent.d34,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err(F10KernelError::NonFiniteInput);
    }
    let base = f10_electron_matrix(
        target,
        category,
        invariants,
        electron_mass,
        support,
        roundoff_ulps,
    )?;
    if !support || base.corrected {
        return Ok((base, 0.0));
    }
    let mut left = if target.flavour() == F10Flavour::Electron {
        0.5 + SIN2_THETA_W
    } else {
        -0.5 + SIN2_THETA_W
    };
    let mut right = SIN2_THETA_W;
    if (category == F10ElectronCategory::ElasticMinus && target.is_antineutrino())
        || (category == F10ElectronCategory::ElasticPlus && !target.is_antineutrino())
    {
        core::mem::swap(&mut left, &mut right);
    }
    let d_ks = tangent.d12 * invariants.d34 + invariants.d12 * tangent.d34;
    let d_kt = tangent.d14 * invariants.d23 + invariants.d14 * tangent.d23;
    let d_interference = electron_mass * electron_mass * tangent.d13;
    let terms = [
        left * left * d_ks,
        right * right * d_kt,
        -left * right * d_interference,
    ];
    let derivative = 64.0 * G_F_MEV_MINUS_2.powi(2) * terms.into_iter().sum::<f64>();
    if derivative.is_finite() {
        Ok((base, derivative))
    } else {
        Err(F10KernelError::NonFiniteInput)
    }
}
