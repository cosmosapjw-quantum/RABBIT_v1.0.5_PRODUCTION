//! Exact grid and state-chart foundations for the six-species F10 comparator.

#![cfg_attr(not(test), allow(dead_code))]

use crate::quadrature::gauss_legendre_rule;

#[derive(Clone, Debug)]
pub(crate) struct F10ActionGrid {
    pub(crate) order: usize,
    pub(crate) y_max: f64,
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

impl F10ActionGrid {
    pub(crate) fn affine_legendre(order: usize, y_max: f64) -> Result<Self, &'static str> {
        if order < 8 {
            return Err("F10 action grid order must be at least eight");
        }
        if !y_max.is_finite() || y_max <= 0.0 {
            return Err("F10 action grid upper bound must be positive and finite");
        }

        let scale = 0.5 * y_max;
        let rule = gauss_legendre_rule(order)?;
        let mut nodes = Vec::with_capacity(order);
        let mut weights = Vec::with_capacity(order);
        for (coordinate, weight) in rule {
            let node = scale * (1.0 + coordinate);
            let mapped_weight = scale * weight;
            if !node.is_finite()
                || !(0.0..=y_max).contains(&node)
                || !mapped_weight.is_finite()
                || mapped_weight <= 0.0
            {
                return Err("mapped F10 action grid is invalid");
            }
            nodes.push(node);
            weights.push(mapped_weight);
        }
        if nodes.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("mapped F10 action grid is not strictly increasing");
        }

        Ok(Self {
            order,
            y_max,
            nodes,
            weights,
        })
    }
}

pub(crate) fn decode_cloglog_to_logit(coordinate: f64) -> Result<f64, &'static str> {
    if !coordinate.is_finite() {
        return Err("cloglog coordinate must be finite");
    }
    let exponential = coordinate.exp();
    if !exponential.is_finite() || exponential <= 0.0 {
        return Err("cloglog exponential is outside the strict chart");
    }
    let occupation = -(-exponential).exp_m1();
    if !occupation.is_finite() || !(0.0..1.0).contains(&occupation) {
        return Err("cloglog occupation must remain strictly between zero and one");
    }
    let logit = occupation.ln() - (-occupation).ln_1p();
    logit
        .is_finite()
        .then_some(logit)
        .ok_or("decoded logit is non-finite")
}
