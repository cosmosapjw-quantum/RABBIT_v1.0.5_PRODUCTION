//! Analytic spectral-coordinate tangents for D-081R1F0.
//!
//! This module differentiates only the complementary-log-log chart and the
//! fixed-grid modal interpolation. It contains no collision coefficients,
//! support decisions, kinematics, finite differences, or solver logic.

#![cfg_attr(not(test), allow(dead_code))]

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_action_spectral::modal_coefficients;

const PAIR_COUNT: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum F10ActionTangentError {
    InvalidInput,
    DimensionOverflow,
    Foundation,
    NonFiniteOutput,
}

#[derive(Clone, Debug)]
pub(crate) struct F10SpectralTangent {
    pub(crate) occupation_delta: Vec<f64>,
    pub(crate) logit_delta: Vec<f64>,
    pub(crate) log_chain_delta: Vec<f64>,
    pub(crate) logit_modal: Vec<f64>,
}

impl F10SpectralTangent {
    pub(crate) fn build(
        grid: &F10ActionGrid,
        pair_cloglog: &[f64],
        direction_cloglog: &[f64],
    ) -> Result<Self, F10ActionTangentError> {
        let expected = PAIR_COUNT
            .checked_mul(grid.order)
            .ok_or(F10ActionTangentError::DimensionOverflow)?;
        if grid.order == 0
            || pair_cloglog.len() != expected
            || direction_cloglog.len() != expected
            || pair_cloglog
                .iter()
                .chain(direction_cloglog)
                .any(|value| !value.is_finite())
        {
            return Err(F10ActionTangentError::InvalidInput);
        }

        let mut occupation_delta = Vec::with_capacity(expected);
        let mut logit_delta = Vec::with_capacity(expected);
        let mut log_chain_delta = Vec::with_capacity(expected);
        for (&coordinate, &direction) in pair_cloglog.iter().zip(direction_cloglog) {
            let exponential = coordinate.exp();
            let occupation = -(-exponential).exp_m1();
            let chain = (coordinate - exponential).exp();
            if !exponential.is_finite()
                || exponential <= 0.0
                || !occupation.is_finite()
                || !(0.0..1.0).contains(&occupation)
                || !chain.is_finite()
                || chain <= 0.0
            {
                return Err(F10ActionTangentError::Foundation);
            }
            if direction == 0.0 {
                occupation_delta.push(0.0);
                logit_delta.push(0.0);
                log_chain_delta.push(0.0);
                continue;
            }
            let delta_occupation = chain * direction;
            let delta_logit = exponential * direction / occupation;
            let delta_log_chain = (1.0 - exponential) * direction;
            if [delta_occupation, delta_logit, delta_log_chain]
                .into_iter()
                .any(|value| !value.is_finite())
            {
                return Err(F10ActionTangentError::NonFiniteOutput);
            }
            occupation_delta.push(delta_occupation);
            logit_delta.push(delta_logit);
            log_chain_delta.push(delta_log_chain);
        }

        let mut logit_modal = Vec::with_capacity(expected);
        for pair in 0..PAIR_COUNT {
            logit_modal.extend(
                modal_coefficients(
                    grid,
                    &logit_delta[pair * grid.order..(pair + 1) * grid.order],
                )
                .map_err(|_| F10ActionTangentError::Foundation)?,
            );
        }
        if logit_modal.iter().any(|value| !value.is_finite()) {
            return Err(F10ActionTangentError::NonFiniteOutput);
        }

        Ok(Self {
            occupation_delta,
            logit_delta,
            log_chain_delta,
            logit_modal,
        })
    }

    pub(crate) fn interpolate_all_pairs(
        &self,
        basis: &[f64],
        point_count: usize,
        order: usize,
    ) -> Result<Vec<f64>, F10ActionTangentError> {
        let expected_basis = point_count
            .checked_mul(order)
            .ok_or(F10ActionTangentError::DimensionOverflow)?;
        if basis.len() != expected_basis || self.logit_modal.len() != PAIR_COUNT * order {
            return Err(F10ActionTangentError::InvalidInput);
        }
        let mut result = vec![0.0; PAIR_COUNT * point_count];
        for pair in 0..PAIR_COUNT {
            for point in 0..point_count {
                let value = (0..order)
                    .map(|mode| self.logit_modal[pair * order + mode] * basis[point * order + mode])
                    .sum::<f64>();
                if !value.is_finite() {
                    return Err(F10ActionTangentError::NonFiniteOutput);
                }
                result[pair * point_count + point] = value;
            }
        }
        Ok(result)
    }
}
