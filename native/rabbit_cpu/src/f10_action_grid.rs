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

const EXACT_ORDER60_YMAX30_NODE_BITS: [u64; 60] = [
    0x3f8843d7fcb95200,
    0x3faff1dc417e63d0,
    0x3fc39b7428544188,
    0x3fd22cecabda9522,
    0x3fdd1616a0ee246c,
    0x3fe540d96553c750,
    0x3fed333f7c537378,
    0x3ff32e60e05f4209,
    0x3ff85b86c8f47e68,
    0x3ffe1d7ef6bf7826,
    0x4002382818bb529f,
    0x4005a7cea46aa929,
    0x40095b540c13b836,
    0x400d502a663dd3d2,
    0x4010c1cb5ad6ffdc,
    0x4012f95965eb7683,
    0x40154d37a489bea8,
    0x4017bbcadca61db4,
    0x401a436565c9dbd5,
    0x401ce2485199515e,
    0x401f96a4a0455b17,
    0x40212f4e4009b8c1,
    0x40229c224b8ff72a,
    0x402410d2ab3b7b54,
    0x40258c5e2acd0566,
    0x40270dbedac1352f,
    0x402893eac515a74a,
    0x402a1dd4a4d53d46,
    0x402baa6c9fece55c,
    0x402d38a102c8a2d2,
    0x402ec75efd375d2f,
    0x40302ac9b0098d52,
    0x4030f115ad95615c,
    0x4031b60a9d752c5b,
    0x40327920929f6569,
    0x403339d0ea997d4d,
    0x4033f796aa624256,
    0x4034b1eeda38046a,
    0x40356858dffb239f,
    0x40361a56d7eea93a,
    0x4036c76deb99aba8,
    0x40376f26a68d890a,
    0x4038110d48d67894,
    0x4038acb216dd9056,
    0x403941a9a685225f,
    0x4039cf8d294a4009,
    0x403a55fab3384586,
    0x403ad4957e7d88fa,
    0x403b4b062b72aadb,
    0x403bb8fafce895ac,
    0x403c1e281094087e,
    0x403c7a479370b81a,
    0x403ccd19f1fa0bdf,
    0x403d1666041d6464,
    0x403d55f934d561c6,
    0x403d8ba7a57c476e,
    0x403db74c4d5095ab,
    0x403dd8c917af577d,
    0x403df00711df40ce,
    0x403dfcf7850068d6,
];

const EXACT_ORDER60_YMAX30_WEIGHT_BITS: [u64; 60] = [
    0x3f9f21bfbb49c64a,
    0x3fb218cd51aad074,
    0x3fbc609911fe7425,
    0x3fc34a9347450587,
    0x3fc8579287ef0a09,
    0x3fcd53c7d0fd9106,
    0x3fd11de0608d0ff1,
    0x3fd3860d009ab5d8,
    0x3fd5e0c06e5f7694,
    0x3fd82c5aab8ecf79,
    0x3fda67462bde05aa,
    0x3fdc8ff8e963971a,
    0x3fdea4f57302e052,
    0x3fe05265f97b5c8f,
    0x3fe1470d962915e3,
    0x3fe22fc8b74c56ee,
    0x3fe30bf6bf583ec8,
    0x3fe3daffba422ab2,
    0x3fe49c54c65b428d,
    0x3fe54f7076e905a0,
    0x3fe5f3d730381b5e,
    0x3fe689177ce9721b,
    0x3fe70eca5c3e91bf,
    0x3fe78493892ef6bb,
    0x3fe7ea21ba154ea1,
    0x3fe83f2ed8c897ce,
    0x3fe8838032fa63af,
    0x3fe8b6e6a2b8d43b,
    0x3fe8d93eaef85d1a,
    0x3fe8ea70a40ed039,
    0x3fe8ea70a40ed039,
    0x3fe8d93eaef85d1a,
    0x3fe8b6e6a2b8d43b,
    0x3fe8838032fa63af,
    0x3fe83f2ed8c897ce,
    0x3fe7ea21ba154ea1,
    0x3fe78493892ef6bb,
    0x3fe70eca5c3e91bf,
    0x3fe689177ce9721b,
    0x3fe5f3d730381b5e,
    0x3fe54f7076e905a0,
    0x3fe49c54c65b428d,
    0x3fe3daffba422ab2,
    0x3fe30bf6bf583ec8,
    0x3fe22fc8b74c56ee,
    0x3fe1470d962915e3,
    0x3fe05265f97b5c8f,
    0x3fdea4f57302e052,
    0x3fdc8ff8e963971a,
    0x3fda67462bde05aa,
    0x3fd82c5aab8ecf79,
    0x3fd5e0c06e5f7694,
    0x3fd3860d009ab5d8,
    0x3fd11de0608d0ff1,
    0x3fcd53c7d0fd9106,
    0x3fc8579287ef0a09,
    0x3fc34a9347450587,
    0x3fbc609911fe7425,
    0x3fb218cd51aad074,
    0x3f9f21bfbb49c64a,
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

        // D-081R1E: preserve the frozen NumPy 2.4.4 binary64
        // finite-dimensional operator at the provenance-locked retained grid.
        if order == 60 && y_max.to_bits() == 30.0_f64.to_bits() {
            return Ok(Self {
                order,
                y_max,
                nodes: EXACT_ORDER60_YMAX30_NODE_BITS
                    .into_iter()
                    .map(f64::from_bits)
                    .collect(),
                weights: EXACT_ORDER60_YMAX30_WEIGHT_BITS
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
