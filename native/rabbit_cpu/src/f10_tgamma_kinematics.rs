//! Analytic mapped-basis and finite-mass elastic `T_gamma` tangent primitives.
//!
//! These routines differentiate the admitted discrete F10 kinematic operator
//! while holding the angular quadrature, target-neutrino momentum, masses, and
//! support predicate fixed. A support or square-root branch crossing is not
//! smoothed into an ordinary derivative.

#![cfg_attr(not(test), allow(dead_code))]

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicBatch, F10KinematicInput, angular_rule,
    electron_half_line_rule, two_body_kinematics,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10TgammaKinematicError {
    InvalidInput,
    Foundation,
    NondifferentiableDiscreteEvent,
    NonFiniteOutput,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10ElasticTgammaInput {
    pub(crate) p1: f64,
    pub(crate) temperature_gamma: f64,
    pub(crate) electron_mass: f64,
    pub(crate) config: F10CollisionConfig,
}

#[derive(Clone, Debug)]
pub(crate) struct F10KinematicTgammaTangent {
    pub(crate) base: F10KinematicBatch,
    pub(crate) support: Vec<bool>,
    pub(crate) d_support_indicator: Vec<f64>,
    pub(crate) d_p2: Vec<f64>,
    pub(crate) d_e2: Vec<f64>,
    pub(crate) d_e3: Vec<f64>,
    pub(crate) d_e4: Vec<f64>,
    pub(crate) d_p3_magnitude: Vec<f64>,
    pub(crate) d_p4_magnitude: Vec<f64>,
    pub(crate) d_phase_space: Vec<f64>,
    pub(crate) d_quadrature_weight: Vec<f64>,
    pub(crate) d_d12: Vec<f64>,
    pub(crate) d_d13: Vec<f64>,
    pub(crate) d_d14: Vec<f64>,
    pub(crate) d_d23: Vec<f64>,
    pub(crate) d_d24: Vec<f64>,
    pub(crate) d_d34: Vec<f64>,
    pub(crate) minimum_support_margin: f64,
    pub(crate) minimum_lambda_margin: f64,
}

fn square(value: f64) -> f64 {
    value * value
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn norm_and_tangent(
    vector: [f64; 3],
    tangent: [f64; 3],
) -> Result<(f64, f64), F10TgammaKinematicError> {
    let magnitude = dot(vector, vector).sqrt();
    if !magnitude.is_finite() {
        return Err(F10TgammaKinematicError::NonFiniteOutput);
    }
    let derivative = if magnitude > 64.0 * f64::MIN_POSITIVE {
        dot(vector, tangent) / magnitude
    } else {
        0.0
    };
    if derivative.is_finite() {
        Ok((magnitude, derivative))
    } else {
        Err(F10TgammaKinematicError::NonFiniteOutput)
    }
}

#[allow(clippy::too_many_arguments)]
fn minkowski_dot_tangent(
    energy_a: f64,
    d_energy_a: f64,
    vector_a: [f64; 3],
    d_vector_a: [f64; 3],
    energy_b: f64,
    d_energy_b: f64,
    vector_b: [f64; 3],
    d_vector_b: [f64; 3],
) -> f64 {
    d_energy_a * energy_b + energy_a * d_energy_b
        - dot(d_vector_a, vector_b)
        - dot(vector_a, d_vector_b)
}

fn kallen(s: f64, mass3_squared: f64, mass4_squared: f64) -> f64 {
    square(s) + square(mass3_squared) + square(mass4_squared)
        - 2.0 * s * mass3_squared
        - 2.0 * s * mass4_squared
        - 2.0 * mass3_squared * mass4_squared
}

pub(crate) fn mapped_modal_basis_derivative(
    grid: &F10ActionGrid,
    query_y: &[f64],
) -> Result<Vec<f64>, F10TgammaKinematicError> {
    if grid.order == 0
        || grid.nodes.len() != grid.order
        || grid.weights.len() != grid.order
        || !grid.y_max.is_finite()
        || grid.y_max <= 0.0
        || query_y
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > grid.y_max)
    {
        return Err(F10TgammaKinematicError::InvalidInput);
    }
    let size = query_y
        .len()
        .checked_mul(grid.order)
        .ok_or(F10TgammaKinematicError::InvalidInput)?;
    let mut result = vec![0.0_f64; size];
    let dx_dy = 2.0 / grid.y_max;

    for (point, &query) in query_y.iter().enumerate() {
        let x = 2.0 * query / grid.y_max - 1.0;
        if grid.order == 1 {
            continue;
        }
        let mut p_previous = 1.0_f64;
        let mut dp_previous = 0.0_f64;
        let mut p_current = x;
        let mut dp_current = 1.0_f64;
        result[point * grid.order + 1] = (3.0 / grid.y_max).sqrt() * dp_current * dx_dy;

        for degree in 1..grid.order - 1 {
            let n = degree as f64;
            let p_next = ((2.0 * n + 1.0) * x * p_current - n * p_previous) / (n + 1.0);
            let dp_next =
                ((2.0 * n + 1.0) * (p_current + x * dp_current) - n * dp_previous) / (n + 1.0);
            let output_degree = degree + 1;
            result[point * grid.order + output_degree] =
                ((2 * output_degree + 1) as f64 / grid.y_max).sqrt() * dp_next * dx_dy;
            p_previous = p_current;
            dp_previous = dp_current;
            p_current = p_next;
            dp_current = dp_next;
        }
    }

    if result.iter().all(|value| value.is_finite()) {
        Ok(result)
    } else {
        Err(F10TgammaKinematicError::NonFiniteOutput)
    }
}

pub(crate) fn evaluate_elastic_tgamma_kinematic_tangent(
    input: F10ElasticTgammaInput,
) -> Result<F10KinematicTgammaTangent, F10TgammaKinematicError> {
    if !input.p1.is_finite()
        || input.p1 <= 0.0
        || !input.temperature_gamma.is_finite()
        || input.temperature_gamma <= 0.0
        || !input.electron_mass.is_finite()
        || input.electron_mass < 0.0
    {
        return Err(F10TgammaKinematicError::InvalidInput);
    }

    let (p2_nodes, p2_weights) =
        electron_half_line_rule(input.config.electron_radial_order, input.temperature_gamma)
            .map_err(|_| F10TgammaKinematicError::Foundation)?;
    let d_p2_nodes = p2_nodes
        .iter()
        .map(|value| value / input.temperature_gamma)
        .collect::<Vec<_>>();
    let d_p2_weights = p2_weights
        .iter()
        .map(|value| value / input.temperature_gamma)
        .collect::<Vec<_>>();
    let base = two_body_kinematics(F10KinematicInput {
        p1: input.p1,
        p2_nodes: &p2_nodes,
        p2_weights: &p2_weights,
        mass2: input.electron_mass,
        mass3: 0.0,
        mass4: input.electron_mass,
        config: input.config,
    })
    .map_err(|_| F10TgammaKinematicError::Foundation)?;
    let rule = angular_rule(input.config).map_err(|_| F10TgammaKinematicError::Foundation)?;
    let expected_size = base.shape.iter().product::<usize>();

    let mut d_p2 = Vec::with_capacity(expected_size);
    let mut d_e2 = Vec::with_capacity(expected_size);
    let mut d_e3 = Vec::with_capacity(expected_size);
    let mut d_e4 = Vec::with_capacity(expected_size);
    let mut d_p3_magnitude = Vec::with_capacity(expected_size);
    let mut d_p4_magnitude = Vec::with_capacity(expected_size);
    let mut d_phase_space = Vec::with_capacity(expected_size);
    let mut d_quadrature_weight = Vec::with_capacity(expected_size);
    let mut d_d12 = Vec::with_capacity(expected_size);
    let mut d_d13 = Vec::with_capacity(expected_size);
    let mut d_d14 = Vec::with_capacity(expected_size);
    let mut d_d23 = Vec::with_capacity(expected_size);
    let mut d_d24 = Vec::with_capacity(expected_size);
    let mut d_d34 = Vec::with_capacity(expected_size);
    let mut minimum_support_margin = f64::INFINITY;
    let mut minimum_lambda_margin = f64::INFINITY;
    let mut output_index = 0_usize;

    let energy1 = input.p1;
    let vector1 = [0.0_f64, 0.0, input.p1];
    let zero_vector = [0.0_f64; 3];
    let mass3_squared = 0.0_f64;
    let mass4_squared = square(input.electron_mass);
    let threshold_squared = mass4_squared;

    for (((&p2, &weight2), &dp2), &dweight2) in p2_nodes
        .iter()
        .zip(&p2_weights)
        .zip(&d_p2_nodes)
        .zip(&d_p2_weights)
    {
        let energy2 = p2.hypot(input.electron_mass);
        let denergy2 = p2 * dp2 / energy2;
        for (&mu12, &weight12) in rule.incoming_mu.iter().zip(&rule.incoming_weights) {
            let sin12 = (1.0 - square(mu12)).max(0.0).sqrt();
            let vector2 = [p2 * sin12, 0.0, p2 * mu12];
            let dvector2 = [dp2 * sin12, 0.0, dp2 * mu12];
            let total_energy = energy1 + energy2;
            let d_total_energy = denergy2;
            let total_vector = add(vector1, vector2);
            let d_total_vector = dvector2;
            let (total_magnitude, d_total_magnitude) =
                norm_and_tangent(total_vector, d_total_vector)?;
            let invariant_s_raw = square(total_energy) - square(total_magnitude);
            let d_invariant_s_raw =
                2.0 * total_energy * d_total_energy - 2.0 * total_magnitude * d_total_magnitude;
            let invariant_s = invariant_s_raw.max(0.0);
            let d_invariant_s = if invariant_s_raw > 0.0 {
                d_invariant_s_raw
            } else {
                0.0
            };
            let support = invariant_s > 0.0 && invariant_s > threshold_squared;
            minimum_support_margin =
                minimum_support_margin.min((invariant_s - threshold_squared).abs());
            let sqrt_s = if support { invariant_s.sqrt() } else { 1.0 };
            let d_sqrt_s = if support {
                d_invariant_s / (2.0 * sqrt_s)
            } else {
                0.0
            };
            let lambda = kallen(invariant_s, mass3_squared, mass4_squared);
            minimum_lambda_margin = minimum_lambda_margin.min(lambda.abs());
            let lambda_scale = square(invariant_s).max(1.0);
            let lambda_roundoff = 512.0 * f64::EPSILON * lambda_scale;
            if support && lambda < -lambda_roundoff {
                return Err(F10TgammaKinematicError::Foundation);
            }
            if support && lambda <= 0.0 {
                return Err(F10TgammaKinematicError::NondifferentiableDiscreteEvent);
            }
            let d_lambda = 2.0 * (invariant_s - mass3_squared - mass4_squared) * d_invariant_s;
            let sqrt_lambda = if support { lambda.sqrt() } else { 0.0 };
            let d_sqrt_lambda = if support {
                d_lambda / (2.0 * sqrt_lambda)
            } else {
                0.0
            };
            let k_star = if support {
                sqrt_lambda / (2.0 * sqrt_s)
            } else {
                0.0
            };
            let d_k_star = if support {
                d_sqrt_lambda / (2.0 * sqrt_s) - sqrt_lambda * d_sqrt_s / (2.0 * square(sqrt_s))
            } else {
                0.0
            };
            let energy3_star_numerator = invariant_s + mass3_squared - mass4_squared;
            let energy3_star = if support {
                energy3_star_numerator / (2.0 * sqrt_s)
            } else {
                0.0
            };
            let d_energy3_star = if support {
                d_invariant_s / (2.0 * sqrt_s)
                    - energy3_star_numerator * d_sqrt_s / (2.0 * square(sqrt_s))
            } else {
                0.0
            };
            let beta = if support {
                total_magnitude / total_energy
            } else {
                0.0
            };
            let d_beta = if support {
                d_total_magnitude / total_energy
                    - total_magnitude * d_total_energy / square(total_energy)
            } else {
                0.0
            };
            let gamma = if support { total_energy / sqrt_s } else { 1.0 };
            let d_gamma = if support {
                d_total_energy / sqrt_s - total_energy * d_sqrt_s / square(sqrt_s)
            } else {
                0.0
            };
            let (parallel, d_parallel) = if total_magnitude > 64.0 * f64::MIN_POSITIVE {
                (
                    scale(total_vector, total_magnitude.recip()),
                    subtract(
                        scale(d_total_vector, total_magnitude.recip()),
                        scale(total_vector, d_total_magnitude / square(total_magnitude)),
                    ),
                )
            } else {
                ([0.0, 0.0, 1.0], zero_vector)
            };
            let transverse_x = [parallel[2], 0.0, -parallel[0]];
            let d_transverse_x = [d_parallel[2], 0.0, -d_parallel[0]];
            let transverse_y = [0.0, 1.0, 0.0];

            for (&mu_star, &weight_star) in rule.final_mu.iter().zip(&rule.final_weights) {
                let sin_star = (1.0 - square(mu_star)).max(0.0).sqrt();
                for (&phi, &weight_phi) in rule.azimuth.iter().zip(&rule.azimuth_weights) {
                    if output_index >= base.support.len() || base.support[output_index] != support {
                        return Err(F10TgammaKinematicError::NondifferentiableDiscreteEvent);
                    }
                    let transverse_scale = k_star * sin_star;
                    let d_transverse_scale = d_k_star * sin_star;
                    let cos_phi = phi.cos();
                    let sin_phi = phi.sin();
                    let transverse = add(
                        scale(transverse_x, transverse_scale * cos_phi),
                        scale(transverse_y, transverse_scale * sin_phi),
                    );
                    let d_transverse = add(
                        add(
                            scale(d_transverse_x, transverse_scale * cos_phi),
                            scale(transverse_x, d_transverse_scale * cos_phi),
                        ),
                        scale(transverse_y, d_transverse_scale * sin_phi),
                    );
                    let parallel_argument = k_star * mu_star + beta * energy3_star;
                    let d_parallel_argument =
                        d_k_star * mu_star + d_beta * energy3_star + beta * d_energy3_star;
                    let p3_parallel = gamma * parallel_argument;
                    let d_p3_parallel = d_gamma * parallel_argument + gamma * d_parallel_argument;
                    let vector3 = add(transverse, scale(parallel, p3_parallel));
                    let d_vector3 = add(
                        d_transverse,
                        add(
                            scale(d_parallel, p3_parallel),
                            scale(parallel, d_p3_parallel),
                        ),
                    );
                    let energy3_argument = energy3_star + beta * k_star * mu_star;
                    let d_energy3_argument =
                        d_energy3_star + d_beta * k_star * mu_star + beta * d_k_star * mu_star;
                    let energy3 = gamma * energy3_argument;
                    let denergy3 = d_gamma * energy3_argument + gamma * d_energy3_argument;
                    let vector4 = subtract(total_vector, vector3);
                    let d_vector4 = subtract(d_total_vector, d_vector3);
                    let energy4 = total_energy - energy3;
                    let denergy4 = d_total_energy - denergy3;
                    let (_, dp3_magnitude) = norm_and_tangent(vector3, d_vector3)?;
                    let (_, dp4_magnitude) = norm_and_tangent(vector4, d_vector4)?;
                    let _phase_space = if support { k_star / sqrt_s } else { 0.0 };
                    let dphase_space = if support {
                        d_k_star / sqrt_s - k_star * d_sqrt_s / square(sqrt_s)
                    } else {
                        0.0
                    };
                    let _quadrature_weight = weight2 * weight12 * weight_star * weight_phi;
                    let dquadrature_weight = dweight2 * weight12 * weight_star * weight_phi;
                    let values = [
                        dp2,
                        denergy2,
                        denergy3,
                        denergy4,
                        dp3_magnitude,
                        dp4_magnitude,
                        dphase_space,
                        dquadrature_weight,
                        minkowski_dot_tangent(
                            energy1,
                            0.0,
                            vector1,
                            zero_vector,
                            energy2,
                            denergy2,
                            vector2,
                            dvector2,
                        ),
                        minkowski_dot_tangent(
                            energy1,
                            0.0,
                            vector1,
                            zero_vector,
                            energy3,
                            denergy3,
                            vector3,
                            d_vector3,
                        ),
                        minkowski_dot_tangent(
                            energy1,
                            0.0,
                            vector1,
                            zero_vector,
                            energy4,
                            denergy4,
                            vector4,
                            d_vector4,
                        ),
                        minkowski_dot_tangent(
                            energy2, denergy2, vector2, dvector2, energy3, denergy3, vector3,
                            d_vector3,
                        ),
                        minkowski_dot_tangent(
                            energy2, denergy2, vector2, dvector2, energy4, denergy4, vector4,
                            d_vector4,
                        ),
                        minkowski_dot_tangent(
                            energy3, denergy3, vector3, d_vector3, energy4, denergy4, vector4,
                            d_vector4,
                        ),
                    ];
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(F10TgammaKinematicError::NonFiniteOutput);
                    }
                    d_p2.push(values[0]);
                    d_e2.push(values[1]);
                    d_e3.push(values[2]);
                    d_e4.push(values[3]);
                    d_p3_magnitude.push(values[4]);
                    d_p4_magnitude.push(values[5]);
                    d_phase_space.push(values[6]);
                    d_quadrature_weight.push(values[7]);
                    d_d12.push(values[8]);
                    d_d13.push(values[9]);
                    d_d14.push(values[10]);
                    d_d23.push(values[11]);
                    d_d24.push(values[12]);
                    d_d34.push(values[13]);
                    output_index += 1;
                }
            }
        }
    }

    if output_index != expected_size
        || !minimum_support_margin.is_finite()
        || minimum_support_margin <= 0.0
        || !minimum_lambda_margin.is_finite()
        || minimum_lambda_margin <= 0.0
    {
        return Err(F10TgammaKinematicError::NonFiniteOutput);
    }

    Ok(F10KinematicTgammaTangent {
        support: base.support.clone(),
        d_support_indicator: vec![0.0; expected_size],
        base,
        d_p2,
        d_e2,
        d_e3,
        d_e4,
        d_p3_magnitude,
        d_p4_magnitude,
        d_phase_space,
        d_quadrature_weight,
        d_d12,
        d_d13,
        d_d14,
        d_d23,
        d_d24,
        d_d34,
        minimum_support_margin,
        minimum_lambda_margin,
    })
}
