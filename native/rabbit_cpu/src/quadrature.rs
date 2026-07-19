//! Shared deterministic Gaussian quadrature rules for native physics blocks.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use nalgebra::{DMatrix, linalg::SymmetricEigen};

pub(crate) fn gauss_legendre_rule(order: usize) -> Result<Vec<(f64, f64)>, &'static str> {
    if order < 2 {
        return Err("quadrature order must be at least two");
    }
    let mut rule = Vec::with_capacity(order);
    for root_index in 0..order.div_ceil(2) {
        let mut root = (PI * (root_index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut derivative = f64::NAN;
        for _ in 0..64 {
            let mut previous = 1.0;
            let mut current = root;
            for degree in 2..=order {
                let next = ((2 * degree - 1) as f64 * root * current
                    - (degree - 1) as f64 * previous)
                    / degree as f64;
                previous = current;
                current = next;
            }
            derivative = order as f64 * (root * current - previous) / (root * root - 1.0);
            let update = current / derivative;
            root -= update;
            if update.abs() <= 8.0 * f64::EPSILON * root.abs().max(1.0) {
                break;
            }
        }
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        if !root.is_finite() || !weight.is_finite() || weight <= 0.0 {
            return Err("Gauss-Legendre rule did not converge");
        }
        if root.abs() <= 32.0 * f64::EPSILON {
            rule.push((0.0, weight));
        } else {
            rule.push((-root, weight));
            rule.push((root, weight));
        }
    }
    rule.sort_by(|left, right| left.0.total_cmp(&right.0));
    (rule.len() == order)
        .then_some(rule)
        .ok_or("Gauss-Legendre rule has the wrong size")
}

/// Gauss--Legendre nodes mapped to `(0, infinity)` with weights for plain
/// `dy`, using `y = -scale ln(1-t)` and `t` on `(0, 1)`.
pub(crate) fn gauss_legendre_exponential_plain_rule(
    order: usize,
    scale: f64,
) -> Result<Vec<(f64, f64)>, &'static str> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err("exponential quadrature scale must be positive and finite");
    }
    gauss_legendre_rule(order)?
        .into_iter()
        .map(|(coordinate, weight)| {
            let unit = 0.5 * (1.0 + coordinate);
            let one_minus_unit = 1.0 - unit;
            let node = -scale * one_minus_unit.ln();
            let plain_weight = 0.5 * scale * weight / one_minus_unit;
            if !node.is_finite() || node <= 0.0 || !plain_weight.is_finite() || plain_weight <= 0.0
            {
                return Err("exponential Gauss-Legendre rule is invalid");
            }
            Ok((node, plain_weight))
        })
        .collect()
}

/// Gauss--Laguerre nodes on `(0, infinity)` with weights for plain `dx`.
pub(crate) fn gauss_laguerre_plain_rule(order: usize) -> Result<Vec<(f64, f64)>, &'static str> {
    if order < 2 {
        return Err("quadrature order must be at least two");
    }
    let mut jacobi = DMatrix::<f64>::zeros(order, order);
    for index in 0..order {
        jacobi[(index, index)] = (2 * index + 1) as f64;
        if index + 1 < order {
            let off_diagonal = (index + 1) as f64;
            jacobi[(index, index + 1)] = off_diagonal;
            jacobi[(index + 1, index)] = off_diagonal;
        }
    }
    let eigensystem = SymmetricEigen::new(jacobi);
    let mut rule = Vec::with_capacity(order);
    for index in 0..order {
        let node = eigensystem.eigenvalues[index];
        let laguerre_weight = eigensystem.eigenvectors[(0, index)].powi(2);
        let plain_weight = laguerre_weight * node.exp();
        if !node.is_finite() || node <= 0.0 || !plain_weight.is_finite() || plain_weight <= 0.0 {
            return Err("Gauss-Laguerre rule is invalid");
        }
        rule.push((node, plain_weight));
    }
    rule.sort_by(|left, right| left.0.total_cmp(&right.0));
    Ok(rule)
}
