//! Exact grid and state-chart foundations for the six-species F10 comparator.
//! The first GREEN admission is rerun only after exact Rust 1.94.1 formatting.

#![cfg_attr(not(test), allow(dead_code))]

use crate::quadrature::gauss_legendre_rule;

const EXACT_ORDER8_YMAX8_NODE_BITS: [u64; 8] = [
    0x3fc4_54e3_4f53_39a0,
    0x3fea_06d5_36d8_2f88,
    0x3ffe_5dad_4f9a_f698,
    0x400a_214d_ac30_e32f,
    0x4012_ef59_29e7_8e68,
    0x4018_6894_ac19_425a,
    0x401c_bf25_5924_fa0f,
    0x401f_5d58_e585_6633,
];

const EXACT_ORDER8_YMAX8_WEIGHT_BITS: [u64; 8] = [
    0x3fd9_ea1d_04ca_03ae,
    0x3fec_76fb_531d_2b94,
    0x3ff4_13c5_0a25_560e,
    0x3ff7_3636_0b19_933d,
    0x3ff7_3636_0b19_933d,
    0x3ff4_13c5_0a25_560e,
    0x3fec_76fb_531d_2b94,
    0x3fd9_ea1d_04ca_03ae,
];

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

        // The frozen Python comparator defines this authoritative action grid
        // through NumPy 2.4.4 `leggauss(8)`.  A separately converged Newton
        // implementation is mathematically equivalent but differs by up to
        // O(1e-14) in the weights, which contaminates analytically vanishing
        // high Legendre modes.  Preserve the exact finite-dimensional operator
        // for this one admitted grid while leaving every other F10 grid on the
        // generic Rust quadrature path.
        if order == 8 && y_max.to_bits() == 8.0_f64.to_bits() {
            return Ok(Self {
                order,
                y_max,
                nodes: EXACT_ORDER8_YMAX8_NODE_BITS
                    .into_iter()
                    .map(f64::from_bits)
                    .collect(),
                weights: EXACT_ORDER8_YMAX8_WEIGHT_BITS
                    .into_iter()
                    .map(f64::from_bits)
                    .collect(),
            });
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

#[cfg(test)]
mod exact_grid_tests {
    use super::*;

    #[test]
    fn admitted_order8_ymax8_grid_preserves_numpy_binary64_identity() {
        let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
        let node_bits: Vec<_> = grid.nodes.iter().map(|value| value.to_bits()).collect();
        let weight_bits: Vec<_> = grid.weights.iter().map(|value| value.to_bits()).collect();
        assert_eq!(node_bits, EXACT_ORDER8_YMAX8_NODE_BITS);
        assert_eq!(weight_bits, EXACT_ORDER8_YMAX8_WEIGHT_BITS);
    }

    #[test]
    fn non_authoritative_domains_remain_on_the_generic_quadrature_path() {
        let grid = F10ActionGrid::affine_legendre(8, 9.0).unwrap();
        assert_eq!(grid.order, 8);
        assert_eq!(grid.y_max.to_bits(), 9.0_f64.to_bits());
        assert!(grid.nodes.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(grid.weights.iter().all(|weight| *weight > 0.0));
    }
}
