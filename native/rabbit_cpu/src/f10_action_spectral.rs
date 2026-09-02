//! Spectral operations for the exact six-species F10 action lane.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::f10_action_grid::F10ActionGrid;

fn validate_finite(values: &[f64]) -> Result<(), &'static str> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or("spectral input contains a non-finite value")
}

pub(crate) fn modal_basis(grid: &F10ActionGrid, query: &[f64]) -> Result<Vec<f64>, &'static str> {
    validate_finite(query)?;
    let mut basis = Vec::with_capacity(query.len() * grid.order);
    for &coordinate in query {
        if !(0.0..=grid.y_max).contains(&coordinate) {
            return Err("spectral query lies outside the frozen grid domain");
        }
        let mapped = 2.0 * coordinate / grid.y_max - 1.0;
        let mut previous = 1.0;
        basis.push((1.0 / grid.y_max).sqrt() * previous);
        if grid.order == 1 {
            continue;
        }
        let mut current = mapped;
        basis.push((3.0 / grid.y_max).sqrt() * current);
        for degree in 2..grid.order {
            let next = ((2 * degree - 1) as f64 * mapped * current
                - (degree - 1) as f64 * previous)
                / degree as f64;
            basis.push(((2 * degree + 1) as f64 / grid.y_max).sqrt() * next);
            previous = current;
            current = next;
        }
    }
    validate_finite(&basis)?;
    Ok(basis)
}

pub(crate) fn modal_coefficients(
    grid: &F10ActionGrid,
    values: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if values.len() != grid.order {
        return Err("spectral value count does not match the grid order");
    }
    validate_finite(values)?;
    let basis = modal_basis(grid, &grid.nodes)?;
    let mut coefficients = vec![0.0; grid.order];
    for (mode, coefficient) in coefficients.iter_mut().enumerate() {
        *coefficient = values
            .iter()
            .enumerate()
            .map(|(index, value)| grid.weights[index] * basis[index * grid.order + mode] * value)
            .sum();
    }
    validate_finite(&coefficients)?;
    Ok(coefficients)
}

pub(crate) fn interpolate(
    grid: &F10ActionGrid,
    values: &[f64],
    query: &[f64],
) -> Result<Vec<f64>, &'static str> {
    let coefficients = modal_coefficients(grid, values)?;
    let basis = modal_basis(grid, query)?;
    let mut interpolated = vec![0.0; query.len()];
    for (point, output) in interpolated.iter_mut().enumerate() {
        *output = coefficients
            .iter()
            .enumerate()
            .map(|(mode, coefficient)| coefficient * basis[point * grid.order + mode])
            .sum();
    }
    validate_finite(&interpolated)?;
    Ok(interpolated)
}

pub(crate) fn modal_product(
    grid: &F10ActionGrid,
    rates: &[f64],
    rows: usize,
    query: &[f64],
) -> Result<Vec<f64>, &'static str> {
    if rows == 0 || rates.len() != rows * query.len() {
        return Err("modal-product rate shape is invalid");
    }
    validate_finite(rates)?;
    let basis = modal_basis(grid, query)?;
    let mut output = vec![0.0; rows * grid.order];
    for row in 0..rows {
        for mode in 0..grid.order {
            output[row * grid.order + mode] = query
                .iter()
                .enumerate()
                .map(|(point, _)| {
                    rates[row * query.len() + point] * basis[point * grid.order + mode]
                })
                .sum();
        }
    }
    validate_finite(&output)?;
    Ok(output)
}

pub(crate) fn native_action(
    grid: &F10ActionGrid,
    modal: &[f64],
    rows: usize,
    temperature_cm: f64,
) -> Result<Vec<f64>, &'static str> {
    if rows == 0 || modal.len() != rows * grid.order {
        return Err("native-action modal shape is invalid");
    }
    if !temperature_cm.is_finite() || temperature_cm <= 0.0 {
        return Err("native-action temperature must be positive and finite");
    }
    validate_finite(modal)?;
    let basis = modal_basis(grid, &grid.nodes)?;
    let normalization = temperature_cm.powi(3) / (2.0 * PI.powi(2));
    let mut output = vec![0.0; rows * grid.order];
    for row in 0..rows {
        for node in 0..grid.order {
            let denominator = normalization * grid.nodes[node].powi(2);
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err("native-action normalization is invalid");
            }
            let reconstructed: f64 = (0..grid.order)
                .map(|mode| modal[row * grid.order + mode] * basis[node * grid.order + mode])
                .sum();
            output[row * grid.order + node] = reconstructed / denominator;
        }
    }
    validate_finite(&output)?;
    Ok(output)
}
