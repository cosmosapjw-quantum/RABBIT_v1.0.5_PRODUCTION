//! Two-body quadrature and kinematic foundations for the exact F10 action lane.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::quadrature::gauss_legendre_rule;

#[derive(Clone, Copy, Debug)]
pub(crate) struct F10CollisionConfig {
    pub(crate) incoming_polar_order: usize,
    pub(crate) final_polar_order: usize,
    pub(crate) final_azimuth_order: usize,
    pub(crate) electron_radial_order: usize,
}

impl Default for F10CollisionConfig {
    fn default() -> Self {
        Self {
            incoming_polar_order: 12,
            final_polar_order: 12,
            final_azimuth_order: 4,
            electron_radial_order: 48,
        }
    }
}

impl F10CollisionConfig {
    fn validate(self) -> Result<(), &'static str> {
        if self.incoming_polar_order < 2
            || self.final_polar_order < 2
            || self.final_azimuth_order < 2
            || self.electron_radial_order < 2
        {
            return Err("collision quadrature orders must be at least two");
        }
        if self.final_azimuth_order != 4 {
            return Err("the exact F10 azimuth rule has four midpoint nodes");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct F10AngularRule {
    pub(crate) incoming_mu: Vec<f64>,
    pub(crate) incoming_weights: Vec<f64>,
    pub(crate) final_mu: Vec<f64>,
    pub(crate) final_weights: Vec<f64>,
    pub(crate) azimuth: Vec<f64>,
    pub(crate) azimuth_weights: Vec<f64>,
}

pub(crate) fn angular_rule(config: F10CollisionConfig) -> Result<F10AngularRule, &'static str> {
    config.validate()?;
    let incoming = gauss_legendre_rule(config.incoming_polar_order)?;
    let final_state = gauss_legendre_rule(config.final_polar_order)?;
    let azimuth_weight = 2.0 * PI / config.final_azimuth_order as f64;
    let azimuth: Vec<_> = (0..config.final_azimuth_order)
        .map(|index| (index as f64 + 0.5) * azimuth_weight)
        .collect();
    let azimuth_weights = vec![azimuth_weight; config.final_azimuth_order];
    if azimuth.iter().any(|value| !value.is_finite()) || !azimuth_weight.is_finite() {
        return Err("azimuth quadrature is non-finite");
    }
    Ok(F10AngularRule {
        incoming_mu: incoming.iter().map(|item| item.0).collect(),
        incoming_weights: incoming.iter().map(|item| item.1).collect(),
        final_mu: final_state.iter().map(|item| item.0).collect(),
        final_weights: final_state.iter().map(|item| item.1).collect(),
        azimuth,
        azimuth_weights,
    })
}

pub(crate) fn electron_half_line_rule(
    order: usize,
    temperature: f64,
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    if order < 2 || !temperature.is_finite() || temperature <= 0.0 {
        return Err("electron half-line rule input is invalid");
    }
    let mut momentum = Vec::with_capacity(order);
    let mut weights = Vec::with_capacity(order);
    for (coordinate, weight) in gauss_legendre_rule(order)? {
        let unit = 0.5 * (coordinate + 1.0);
        let one_minus = 1.0 - unit;
        let value = temperature * unit / one_minus;
        let mapped_weight = 0.5 * weight * temperature / one_minus.powi(2);
        if !value.is_finite() || value <= 0.0 || !mapped_weight.is_finite() || mapped_weight <= 0.0
        {
            return Err("electron half-line quadrature is invalid");
        }
        momentum.push(value);
        weights.push(mapped_weight);
    }
    Ok((momentum, weights))
}

pub(crate) struct F10KinematicInput<'a> {
    pub(crate) p1: f64,
    pub(crate) p2_nodes: &'a [f64],
    pub(crate) p2_weights: &'a [f64],
    pub(crate) mass2: f64,
    pub(crate) mass3: f64,
    pub(crate) mass4: f64,
    pub(crate) config: F10CollisionConfig,
}

#[derive(Clone, Debug)]
pub(crate) struct F10KinematicBatch {
    pub(crate) shape: [usize; 4],
    pub(crate) support: Vec<bool>,
    pub(crate) p2: Vec<f64>,
    pub(crate) e2: Vec<f64>,
    pub(crate) e3: Vec<f64>,
    pub(crate) e4: Vec<f64>,
    pub(crate) p3_magnitude: Vec<f64>,
    pub(crate) p4_magnitude: Vec<f64>,
    pub(crate) phase_space: Vec<f64>,
    pub(crate) quadrature_weight: Vec<f64>,
    pub(crate) d12: Vec<f64>,
    pub(crate) d13: Vec<f64>,
    pub(crate) d14: Vec<f64>,
    pub(crate) d23: Vec<f64>,
    pub(crate) d24: Vec<f64>,
    pub(crate) d34: Vec<f64>,
}

impl F10KinematicBatch {
    fn with_capacity(shape: [usize; 4]) -> Self {
        let size = shape.iter().product();
        Self {
            shape,
            support: Vec::with_capacity(size),
            p2: Vec::with_capacity(size),
            e2: Vec::with_capacity(size),
            e3: Vec::with_capacity(size),
            e4: Vec::with_capacity(size),
            p3_magnitude: Vec::with_capacity(size),
            p4_magnitude: Vec::with_capacity(size),
            phase_space: Vec::with_capacity(size),
            quadrature_weight: Vec::with_capacity(size),
            d12: Vec::with_capacity(size),
            d13: Vec::with_capacity(size),
            d14: Vec::with_capacity(size),
            d23: Vec::with_capacity(size),
            d24: Vec::with_capacity(size),
            d34: Vec::with_capacity(size),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        support: bool,
        p2: f64,
        e2: f64,
        e3: f64,
        e4: f64,
        p3_magnitude: f64,
        p4_magnitude: f64,
        phase_space: f64,
        quadrature_weight: f64,
        d12: f64,
        d13: f64,
        d14: f64,
        d23: f64,
        d24: f64,
        d34: f64,
    ) {
        self.support.push(support);
        self.p2.push(p2);
        self.e2.push(e2);
        self.e3.push(e3);
        self.e4.push(e4);
        self.p3_magnitude.push(p3_magnitude);
        self.p4_magnitude.push(p4_magnitude);
        self.phase_space.push(phase_space);
        self.quadrature_weight.push(quadrature_weight);
        self.d12.push(d12);
        self.d13.push(d13);
        self.d14.push(d14);
        self.d23.push(d23);
        self.d24.push(d24);
        self.d34.push(d34);
    }
}

fn square(value: f64) -> f64 {
    value * value
}

fn norm(vector: [f64; 3]) -> f64 {
    (square(vector[0]) + square(vector[1]) + square(vector[2])).sqrt()
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

fn kallen(s: f64, mass3_squared: f64, mass4_squared: f64) -> f64 {
    square(s) + square(mass3_squared) + square(mass4_squared)
        - 2.0 * s * mass3_squared
        - 2.0 * s * mass4_squared
        - 2.0 * mass3_squared * mass4_squared
}

fn minkowski_dot(energy_a: f64, vector_a: [f64; 3], energy_b: f64, vector_b: [f64; 3]) -> f64 {
    energy_a * energy_b - dot(vector_a, vector_b)
}

pub(crate) fn two_body_kinematics(
    input: F10KinematicInput<'_>,
) -> Result<F10KinematicBatch, &'static str> {
    input.config.validate()?;
    if !input.p1.is_finite() || input.p1 <= 0.0 {
        return Err("target momentum must be finite and positive");
    }
    if input.p2_nodes.is_empty() || input.p2_nodes.len() != input.p2_weights.len() {
        return Err("incoming radial rule has an invalid shape");
    }
    if input
        .p2_nodes
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || input
            .p2_weights
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || [input.mass2, input.mass3, input.mass4]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("two-body kinematic input is invalid");
    }

    let rule = angular_rule(input.config)?;
    let shape = [
        input.p2_nodes.len(),
        rule.incoming_mu.len(),
        rule.final_mu.len(),
        rule.azimuth.len(),
    ];
    let mut batch = F10KinematicBatch::with_capacity(shape);
    let energy1 = input.p1;
    let vector1 = [0.0, 0.0, input.p1];
    let mass3_squared = square(input.mass3);
    let mass4_squared = square(input.mass4);
    let threshold_squared = square(input.mass3 + input.mass4);

    for (&p2, &weight2) in input.p2_nodes.iter().zip(input.p2_weights) {
        let energy2 = p2.hypot(input.mass2);
        for (&mu12, &weight12) in rule.incoming_mu.iter().zip(&rule.incoming_weights) {
            let sin12 = (1.0 - square(mu12)).max(0.0).sqrt();
            let vector2 = [p2 * sin12, 0.0, p2 * mu12];
            let total_energy = energy1 + energy2;
            let total_vector = add(vector1, vector2);
            let total_magnitude = norm(total_vector);
            let invariant_s = (square(total_energy) - square(total_magnitude)).max(0.0);
            let support = invariant_s > 0.0 && invariant_s > threshold_squared;
            let sqrt_s = if support { invariant_s.sqrt() } else { 1.0 };
            let lambda = kallen(invariant_s, mass3_squared, mass4_squared);
            let lambda_scale = square(invariant_s).max(1.0);
            let lambda_roundoff = 512.0 * f64::EPSILON * lambda_scale;
            if support && lambda < -lambda_roundoff {
                return Err("negative Kallen discriminant on physical support");
            }
            let k_star = if support {
                lambda.max(0.0).sqrt() / (2.0 * sqrt_s)
            } else {
                0.0
            };
            let energy3_star = if support {
                (invariant_s + mass3_squared - mass4_squared) / (2.0 * sqrt_s)
            } else {
                input.mass3
            };
            let beta = if support {
                total_magnitude / total_energy
            } else {
                0.0
            };
            let gamma = if support { total_energy / sqrt_s } else { 1.0 };
            let parallel = if total_magnitude > 64.0 * f64::MIN_POSITIVE {
                scale(total_vector, total_magnitude.recip())
            } else {
                [0.0, 0.0, 1.0]
            };
            let transverse_x = [parallel[2], 0.0, -parallel[0]];
            let transverse_y = [0.0, 1.0, 0.0];

            for (&mu_star, &weight_star) in rule.final_mu.iter().zip(&rule.final_weights) {
                let sin_star = (1.0 - square(mu_star)).max(0.0).sqrt();
                for (&phi, &weight_phi) in rule.azimuth.iter().zip(&rule.azimuth_weights) {
                    let transverse_scale = k_star * sin_star;
                    let transverse = add(
                        scale(transverse_x, transverse_scale * phi.cos()),
                        scale(transverse_y, transverse_scale * phi.sin()),
                    );
                    let p3_parallel = gamma * (k_star * mu_star + beta * energy3_star);
                    let vector3 = add(transverse, scale(parallel, p3_parallel));
                    let energy3 = gamma * (energy3_star + beta * k_star * mu_star);
                    let vector4 = subtract(total_vector, vector3);
                    let energy4 = total_energy - energy3;
                    let p3_magnitude = norm(vector3);
                    let p4_magnitude = norm(vector4);
                    let shell3 = (square(energy3) - square(p3_magnitude) - mass3_squared).abs();
                    let shell4 = (square(energy4) - square(p4_magnitude) - mass4_squared).abs();
                    let shell_scale = square(total_energy).max(1.0);
                    if support && (shell3 > 2.0e-10 * shell_scale || shell4 > 2.0e-10 * shell_scale)
                    {
                        return Err("two-body boost violates a final mass shell");
                    }
                    if support && (energy3 <= 0.0 || energy4 <= 0.0) {
                        return Err("two-body boost returned nonpositive energy");
                    }

                    let quadrature_weight = weight2 * weight12 * weight_star * weight_phi;
                    let phase_space = if support { k_star / sqrt_s } else { 0.0 };
                    let values = [
                        energy2,
                        energy3,
                        energy4,
                        p3_magnitude,
                        p4_magnitude,
                        phase_space,
                        quadrature_weight,
                        minkowski_dot(energy1, vector1, energy2, vector2),
                        minkowski_dot(energy1, vector1, energy3, vector3),
                        minkowski_dot(energy1, vector1, energy4, vector4),
                        minkowski_dot(energy2, vector2, energy3, vector3),
                        minkowski_dot(energy2, vector2, energy4, vector4),
                        minkowski_dot(energy3, vector3, energy4, vector4),
                    ];
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err("two-body kinematic output is non-finite");
                    }
                    batch.push(
                        support, p2, values[0], values[1], values[2], values[3], values[4],
                        values[5], values[6], values[7], values[8], values[9], values[10],
                        values[11], values[12],
                    );
                }
            }
        }
    }

    Ok(batch)
}
