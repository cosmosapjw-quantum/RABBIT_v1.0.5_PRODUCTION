//! Allocation-free Tier-1 LRS/FLRW RHS pack.
//!
//! This module preserves the frozen Python/JAX physics contract while fusing
//! the finite-mass EOS, analytic characteristic reconstruction, stress,
//! monopole, geometry, temperature, and Hubble work into one native call.

use std::f64::consts::PI;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const M_E_MEV: f64 = 0.510_998_950_0;
const G_N_MEV_MINUS_2: f64 = 6.708_83e-45;
const MEV_TO_INV_S: f64 = 1.519_267_447e21;
const T_DEC_MEV: f64 = 2.0;
const EOS_CUTOFF_X: f64 = 50.0;
const RHO_GAMMA_PREFACTOR: f64 = PI * PI / 15.0;
// Nearest f64 to (5 alpha / 4 pi) * (1 - 0.20 sqrt(alpha / pi)) for
// alpha^{-1}=137.035999084.
const QED_SCALE: f64 = f64::from_bits(0x3f678e701cf90d2e);

const AUX_H_MEV: usize = 0;
const AUX_H_INV_S: usize = 1;
const AUX_T_NU: usize = 2;
const AUX_PI_PLUS: usize = 3;
const AUX_RHO: usize = 4;
const AUX_PRESSURE: usize = 5;
const AUX_ENTROPY: usize = 6;
const AUX_DS_DT: usize = 7;
const AUX_DRHO_DT: usize = 8;

// Exact f64 bit patterns from numpy.polynomial.laguerre.laggauss(64) at
// the frozen R-01 base commit. They define the port-fidelity EOS functional.
#[rustfmt::skip]
const EOS_GL_NODES_BITS: [u64; 64] = [
    0x3f96f42fd97c44c6, 0x3fbe3d46e66081ee, 0x3fd2955a339bea54, 0x3fe141d52d6b7156,
    0x3febae62159963e6, 0x3ff448f76028a0e5, 0x3ffbf73ba7bec8f8, 0x40027196f913845b,
    0x40078718bd767bd6, 0x400d3ceef51ce08c, 0x4011ca0030348668, 0x401546a7bd960aa1,
    0x401914fdb3c08b87, 0x401d35a07b619dff, 0x4020d49ef4c8d619, 0x40233849f32b3741,
    0x4025c638933ada05, 0x40287edafd35ad87, 0x402b62aa9a4c9e4a, 0x402e722a948b3fb6,
    0x4030d6f43277c428, 0x40328b3e38a38a55, 0x403456455dd2ae4d, 0x40363861d2c7715f,
    0x403831f264a56c68, 0x403a435cf618d582, 0x403c6d0f06a35e5e, 0x403eaf7e49fd40bc,
    0x40408594a8e750c6, 0x4041c04c263639b9, 0x4043082eede04e99, 0x40445d8c040e5a8d,
    0x4045c0b8a9b7fd35, 0x40473210f91e9bb9, 0x4048b1f89863b962, 0x404a40db8625fe5a,
    0x404bdf2f04eccd37, 0x404d8d72ab411de3, 0x404f4c319fb58fc8, 0x40508e0204f404f5,
    0x40517ec861eb7742, 0x405278c7acf53c5b, 0x40537c653609bf82, 0x40548a1143dddacb,
    0x4055a248ccc2266e, 0x4056c5978fe237f9, 0x4057f49aaa4b7bdc, 0x40593003ccf35e7a,
    0x405a789d47172381, 0x405bcf4f2d139b35, 0x405d352603001f22, 0x405eab5b82506d9b,
    0x406019b12ed110ea, 0x4060e77b30ddad32, 0x4061c019923d2576, 0x4062a4da7272142d,
    0x40639764f7ad37ba, 0x406499e012acea91, 0x4065af32c3b538fd, 0x4066db7669af5f1a,
    0x406824d61b3e0f35, 0x4069958141e1a9f2, 0x406b4104ee59919c, 0x406d59e812940bcc,
];

#[rustfmt::skip]
const EOS_GL_WEIGHTS_BITS: [u64; 64] = [
    0x3faccd2c2c4ef0c1, 0x3fbe785b24f1fe08, 0x3fc428d797e9a6cf, 0x3fc5722e8709dc59,
    0x3fc3a110fe28f690, 0x3fbfccf372ba1285, 0x3fb720ac4c2eafdd, 0x3fae73de4fbaaccd,
    0x3fa23dc5290f98e5, 0x3f93f2ac36dbab85, 0x3f83f4732b81ae23, 0x3f72492b30366e9d,
    0x3f5eb9d4047b435d, 0x3f47ae0093826e39, 0x3f30be5e991a2cd8, 0x3f15b9686b15b1df,
    0x3ef9dac6ca31e708, 0x3edc36c108188213, 0x3ebc373691d7ce1e, 0x3e99d7a575b81dcc,
    0x3e75a8c9a5fcd41e, 0x3e5098da5c6410f1, 0x3e273b36f91f1000, 0x3dfdaa26d79e3388,
    0x3dd141935c3d6a69, 0x3da2440a7dabbbbb, 0x3d7190687eb966f1, 0x3d3ea26a8fcd5642,
    0x3d082f49ef0ae281, 0x3cd14007838703fd, 0x3c962f677c7bc4f9, 0x3c59a9da1f5eb371,
    0x3c1aa2a3171400c1, 0x3bd8bc09870dc58c, 0x3b947dbd0fd0a2d5, 0x3b4e30ec5025dede,
    0x3b03b570a1cb554c, 0x3ab6b6fb12618bb4, 0x3a67037d731faed8, 0x3a14675108a26289,
    0x39bf80440a96f561, 0x39650e6f60100cc2, 0x39083a0e902ebd90, 0x38a7d4133c5d75d4,
    0x3843e26c7e822ec6, 0x37dbebb37212b00d, 0x3770562cab384feb, 0x36ff868a24c2cf0d,
    0x3688c6dd7c93b2ae, 0x360f48509a60ead6, 0x358f3754189e743c, 0x350828b71b22b40c,
    0x347c5ce42673e222, 0x33e89743d05d733f, 0x334e7c873133a843, 0x32a9f5bb2ebf4187,
    0x31fcdd48dc2affb7, 0x31439db9f3d2d671, 0x307dd2c46f8c99c2, 0x2fa6607187649614,
    0x2ebb78b905fec288, 0x2db44431a6d076a0, 0x2c83f4665c9bf703, 0x2b076514909d5409,
];

#[derive(Clone, Copy)]
struct RayConst {
    weight: f64,
    log_x0: f64,
    log_one_plus_x0: f64,
    sign: f64,
}

#[derive(Clone, Copy)]
struct EosState {
    rho: f64,
    pressure: f64,
    entropy: f64,
    drho_dt: f64,
    ds_dt: f64,
    dt_d_n: f64,
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn contiguous<'a, 'py>(array: &'a PyReadonlyArray1<'py, f64>, name: &str) -> PyResult<&'a [f64]> {
    array
        .as_slice()
        .map_err(|_| value_error(format!("{name} must be a C-contiguous float64 array")))
}

fn contiguous_mut<'a, 'py>(
    array: &'a mut PyReadwriteArray1<'py, f64>,
    name: &str,
) -> PyResult<&'a mut [f64]> {
    array.as_slice_mut().map_err(|_| {
        value_error(format!(
            "{name} must be a writable C-contiguous float64 array"
        ))
    })
}

fn require_len(values: &[f64], expected: usize, name: &str) -> PyResult<()> {
    if values.len() != expected {
        return Err(value_error(format!(
            "{name} must have length {expected}, got {}",
            values.len()
        )));
    }
    Ok(())
}

fn require_finite(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(value_error(format!("{name} contains a non-finite value")));
    }
    Ok(())
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn eos_at(temperature: f64) -> Result<EosState, &'static str> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("T_gamma must be positive and finite");
    }

    let x = M_E_MEV / temperature;
    let mut i21 = 0.0;
    let mut i03 = 0.0;
    let mut di21_dx = 0.0;
    let mut di03_dx = 0.0;

    if x <= EOS_CUTOFF_X {
        let exp_x = x.exp();
        for index in 0..EOS_GL_NODES_BITS.len() {
            let u = f64::from_bits(EOS_GL_NODES_BITS[index]);
            let weight = f64::from_bits(EOS_GL_WEIGHTS_BITS[index]);
            let energy = u + x;
            let momentum = (u * u + 2.0 * u * x).sqrt();
            let distribution = 1.0 / (exp_x + (-u).exp());
            let energy_sq = energy * energy;
            let momentum_cubed = momentum * momentum * momentum;

            i21 += weight * energy_sq * momentum * distribution;
            i03 += weight * momentum_cubed * distribution;
            di21_dx += weight
                * (2.0 * energy * momentum * distribution
                    + u * energy_sq * distribution / momentum
                    - energy_sq * momentum * exp_x * distribution * distribution);
            di03_dx += weight
                * (3.0 * u * momentum * distribution
                    - momentum_cubed * exp_x * distribution * distribution);
        }
    }

    let t2 = temperature * temperature;
    let t3 = t2 * temperature;
    let t4 = t2 * t2;
    let rho_gamma = RHO_GAMMA_PREFACTOR * t4;
    let drho_gamma_dt = 4.0 * rho_gamma / temperature;
    let rho_e = 2.0 * t4 * i21 / (PI * PI);
    let pressure_e = 2.0 * t4 * i03 / (3.0 * PI * PI);
    let drho_e_dt = 2.0 * t3 * (4.0 * i21 - x * di21_dx) / (PI * PI);
    let dpressure_e_dt = 2.0 * t3 * (4.0 * i03 - x * di03_dx) / (3.0 * PI * PI);

    let rho = rho_gamma + (1.0 + QED_SCALE) * rho_e;
    let pressure = rho_gamma / 3.0 + pressure_e + QED_SCALE * rho_e / 3.0;
    let drho_dt = drho_gamma_dt + (1.0 + QED_SCALE) * drho_e_dt;
    let dpressure_dt = drho_gamma_dt / 3.0 + dpressure_e_dt + QED_SCALE * drho_e_dt / 3.0;
    let entropy = (rho + pressure) / temperature;
    let ds_dt =
        (drho_dt + dpressure_dt) / temperature - (rho + pressure) / (temperature * temperature);
    let dt_d_n = -3.0 * entropy / ds_dt;

    let values = [rho, pressure, entropy, drho_dt, ds_dt, dt_d_n];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("EOS produced a non-finite output");
    }
    if rho <= 0.0 || pressure <= 0.0 || entropy <= 0.0 || drho_dt <= 0.0 || ds_dt <= 0.0 {
        return Err("EOS produced a non-positive thermodynamic quantity");
    }

    Ok(EosState {
        rho,
        pressure,
        entropy,
        drho_dt,
        ds_dt,
        dt_d_n,
    })
}

fn g_star_s(temperature: f64, entropy: f64) -> f64 {
    45.0 * entropy / (2.0 * PI * PI * temperature * temperature * temperature)
}

fn fermi_dirac_nonnegative(argument: f64) -> f64 {
    let exp_negative = (-argument).exp();
    exp_negative / (1.0 + exp_negative)
}

/// Frozen, LRS-only native Tier-1 pack.
#[pyclass(frozen, module = "_rabbit_cpu")]
pub(crate) struct NativeTier1Core {
    rays: Vec<RayConst>,
    q_nodes: Vec<f64>,
    n_eff: f64,
    f_nu: f64,
    ablate_hubble_anisotropy: bool,
    g_star_s_dec: f64,
}

#[pymethods]
impl NativeTier1Core {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new<'py>(
        mu0: PyReadonlyArray1<'py, f64>,
        w0: PyReadonlyArray1<'py, f64>,
        x0: PyReadonlyArray1<'py, f64>,
        signs: PyReadonlyArray1<'py, f64>,
        q_nodes: PyReadonlyArray1<'py, f64>,
        q_weights: PyReadonlyArray1<'py, f64>,
        n_eff: f64,
        f_nu: f64,
        ablate_hubble_anisotropy: bool,
    ) -> PyResult<Self> {
        let mu0 = contiguous(&mu0, "mu0")?;
        let w0 = contiguous(&w0, "w0")?;
        let x0 = contiguous(&x0, "X0")?;
        let signs = contiguous(&signs, "signs")?;
        let q_nodes = contiguous(&q_nodes, "q_nodes")?;
        let q_weights = contiguous(&q_weights, "q_weights")?;

        let n_mu = mu0.len();
        if n_mu == 0 || q_nodes.is_empty() {
            return Err(value_error("ray and momentum grids must be non-empty"));
        }
        require_len(w0, n_mu, "w0")?;
        require_len(x0, n_mu, "X0")?;
        require_len(signs, n_mu, "signs")?;
        require_len(q_weights, q_nodes.len(), "q_weights")?;
        require_finite(mu0, "mu0")?;
        require_finite(w0, "w0")?;
        require_finite(x0, "X0")?;
        require_finite(signs, "signs")?;
        require_finite(q_nodes, "q_nodes")?;
        require_finite(q_weights, "q_weights")?;
        if !n_eff.is_finite() || n_eff <= 0.0 {
            return Err(value_error("N_eff must be positive and finite"));
        }
        if !f_nu.is_finite() || !(0.0..=1.0).contains(&f_nu) {
            return Err(value_error("f_nu must be finite and in [0, 1]"));
        }

        let mut rays = Vec::with_capacity(n_mu);
        for index in 0..n_mu {
            let mu = mu0[index];
            let weight = w0[index];
            let ray_x0 = x0[index];
            let sign = signs[index];
            if mu.abs() >= 1.0 || weight <= 0.0 || ray_x0 < 0.0 {
                return Err(value_error(
                    "ray nodes, weights, or X0 are outside their domain",
                ));
            }
            let expected_sign = if mu == 0.0 { 0.0 } else { mu.signum() };
            if sign != expected_sign {
                return Err(value_error("signs must equal sign(mu0)"));
            }
            let expected_x0 = mu * mu / (1.0 - mu * mu);
            let x0_error = (ray_x0 - expected_x0).abs() / expected_x0.abs().max(1.0);
            if x0_error > 1.0e-12 {
                return Err(value_error("X0 is inconsistent with mu0"));
            }
            rays.push(RayConst {
                weight,
                log_x0: if ray_x0 == 0.0 {
                    f64::NEG_INFINITY
                } else {
                    ray_x0.ln()
                },
                log_one_plus_x0: ray_x0.ln_1p(),
                sign,
            });
        }
        if q_nodes
            .iter()
            .zip(q_weights.iter())
            .any(|(&node, &weight)| node <= 0.0 || weight <= 0.0)
        {
            return Err(value_error("momentum nodes and weights must be positive"));
        }
        if q_nodes.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(value_error("q_nodes must be strictly increasing"));
        }

        let eos_dec = eos_at(T_DEC_MEV).map_err(value_error)?;
        let g_star_s_dec = g_star_s(T_DEC_MEV, eos_dec.entropy);
        if !g_star_s_dec.is_finite() || g_star_s_dec <= 0.0 {
            return Err(value_error("decoupling entropy normalization is invalid"));
        }

        Ok(Self {
            rays,
            q_nodes: q_nodes.to_vec(),
            n_eff,
            f_nu,
            ablate_hubble_anisotropy,
            g_star_s_dec,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_into<'py>(
        &self,
        state: PyReadonlyArray1<'py, f64>,
        mut derivative: PyReadwriteArray1<'py, f64>,
        mut monopole: PyReadwriteArray1<'py, f64>,
        mut aux: PyReadwriteArray1<'py, f64>,
        mut ray_work: PyReadwriteArray1<'py, f64>,
    ) -> PyResult<()> {
        let state = contiguous(&state, "state")?;
        let derivative = contiguous_mut(&mut derivative, "derivative")?;
        let monopole = contiguous_mut(&mut monopole, "monopole")?;
        let aux = contiguous_mut(&mut aux, "aux")?;
        let ray_work = contiguous_mut(&mut ray_work, "ray_work")?;
        require_len(state, 4, "state")?;
        require_len(derivative, 4, "derivative")?;
        require_len(monopole, self.q_nodes.len(), "monopole")?;
        require_len(aux, 9, "aux")?;
        require_len(ray_work, 5 * self.rays.len(), "ray_work")?;
        require_finite(state, "state")?;

        let sigma_plus = state[0];
        let sigma_minus = state[1];
        let accumulated_shear = state[2];
        let temperature = state[3];
        if sigma_minus != 0.0 {
            return Err(value_error(
                "rust_native is LRS-only: Sigma_minus must equal zero",
            ));
        }
        if temperature <= 0.0 {
            return Err(value_error("T_gamma must be positive"));
        }
        let sigma_squared = sigma_plus * sigma_plus;
        let omega = 1.0 - sigma_squared;
        if !omega.is_finite() || omega <= 0.0 {
            return Err(value_error(
                "Omega=1-Sigma_squared must be positive and finite",
            ));
        }

        let n_mu = self.rays.len();
        let (mu, remainder) = ray_work.split_at_mut(n_mu);
        let (intensity, remainder) = remainder.split_at_mut(n_mu);
        let (jacobian, remainder) = remainder.split_at_mut(n_mu);
        let (exp_2i, energy_weight) = remainder.split_at_mut(n_mu);

        let mut pi_plus_sum = 0.0;
        for index in 0..n_mu {
            let ray = self.rays[index];
            let z = ray.log_x0 + 6.0 * accumulated_shear;
            let mu_squared = sigmoid(z);
            let ray_mu = ray.sign * mu_squared.sqrt();
            let ray_i = 0.25 * (softplus(z) - ray.log_one_plus_x0) - 0.5 * accumulated_shear;
            let ray_j = (-6.0 * ray_i).exp();
            let ray_exp_2i = (2.0 * ray_i).exp();
            let ray_energy_weight = ray_j * (-8.0 * ray_i).exp();
            let p2 = 0.5 * (3.0 * mu_squared - 1.0);

            let values = [ray_mu, ray_i, ray_j, ray_exp_2i, ray_energy_weight];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(value_error(
                    "characteristic reconstruction produced a non-finite value",
                ));
            }
            mu[index] = ray_mu;
            intensity[index] = ray_i;
            jacobian[index] = ray_j;
            exp_2i[index] = ray_exp_2i;
            energy_weight[index] = ray_energy_weight;
            pi_plus_sum += ray.weight * p2 * ray_energy_weight;
        }
        let pi_plus = self.f_nu * pi_plus_sum;
        if !pi_plus.is_finite() {
            return Err(value_error("anisotropic stress is non-finite"));
        }

        for (q_index, &q) in self.q_nodes.iter().enumerate() {
            let mut angular_sum = 0.0;
            for ray_index in 0..n_mu {
                let argument = q * exp_2i[ray_index];
                if !argument.is_finite() {
                    return Err(value_error("monopole FD argument is non-finite"));
                }
                angular_sum += self.rays[ray_index].weight
                    * jacobian[ray_index]
                    * fermi_dirac_nonnegative(argument);
            }
            let value = 0.5 * angular_sum;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(value_error("monopole lies outside [0, 1] or is non-finite"));
            }
            monopole[q_index] = value;
        }

        let eos = eos_at(temperature).map_err(value_error)?;
        let t_nu = if temperature >= T_DEC_MEV {
            temperature
        } else {
            let current_g_star_s = g_star_s(temperature, eos.entropy);
            let ratio = current_g_star_s / self.g_star_s_dec;
            if !ratio.is_finite() || ratio <= 0.0 {
                return Err(value_error(
                    "neutrino entropy ratio is not positive and finite",
                ));
            }
            temperature * ratio.cbrt()
        };
        if !t_nu.is_finite() || t_nu <= 0.0 {
            return Err(value_error("T_nu is not positive and finite"));
        }

        let rho_nu = self.n_eff * (7.0 / 8.0) * RHO_GAMMA_PREFACTOR * t_nu * t_nu * t_nu * t_nu;
        let hubble_omega = if self.ablate_hubble_anisotropy {
            1.0
        } else {
            omega
        };
        let h_squared = (8.0 * PI * G_N_MEV_MINUS_2 / 3.0) * (eos.rho + rho_nu) / hubble_omega;
        if !h_squared.is_finite() || h_squared <= 0.0 {
            return Err(value_error("H_squared is not positive and finite"));
        }
        let h_mev = h_squared.sqrt();
        let h_inv_s = h_mev * MEV_TO_INV_S;
        if !h_mev.is_finite() || h_mev <= 0.0 || !h_inv_s.is_finite() || h_inv_s <= 0.0 {
            return Err(value_error("H is not positive and finite"));
        }

        let damping = -(1.0 - sigma_squared);
        derivative[0] = damping * sigma_plus + pi_plus;
        derivative[1] = 0.0;
        derivative[2] = sigma_plus;
        derivative[3] = eos.dt_d_n;
        require_finite(derivative, "derivative")?;

        aux[AUX_H_MEV] = h_mev;
        aux[AUX_H_INV_S] = h_inv_s;
        aux[AUX_T_NU] = t_nu;
        aux[AUX_PI_PLUS] = pi_plus;
        aux[AUX_RHO] = eos.rho;
        aux[AUX_PRESSURE] = eos.pressure;
        aux[AUX_ENTROPY] = eos.entropy;
        aux[AUX_DS_DT] = eos.ds_dt;
        aux[AUX_DRHO_DT] = eos.drho_dt;
        require_finite(aux, "aux")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_gl64_bits_are_positive() {
        for (&node, &weight) in EOS_GL_NODES_BITS.iter().zip(EOS_GL_WEIGHTS_BITS.iter()) {
            assert!(f64::from_bits(node) > 0.0);
            assert!(f64::from_bits(weight) > 0.0);
        }
    }

    #[test]
    fn photon_limit_has_exact_cooling_law() {
        let eos = eos_at(0.005).expect("low-temperature EOS");
        let expected_rho = RHO_GAMMA_PREFACTOR * 0.005_f64.powi(4);
        assert_eq!(eos.rho, expected_rho);
        assert_eq!(eos.pressure, expected_rho / 3.0);
        assert_eq!(eos.dt_d_n, -0.005);
    }

    #[test]
    fn qed_total_scale_is_positive() {
        let scale = std::hint::black_box(QED_SCALE);
        assert!(scale > 0.0);
        assert!(scale < 0.01);
    }

    #[test]
    fn stable_characteristic_primitives_have_exact_center_limit() {
        assert_eq!(sigmoid(f64::NEG_INFINITY), 0.0);
        assert_eq!(softplus(f64::NEG_INFINITY), 0.0);
        assert_eq!(softplus(0.0), 2.0_f64.ln());
    }

    #[test]
    fn eos_is_finite_across_frozen_temperature_grid() {
        for temperature in [0.005, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0] {
            let eos = eos_at(temperature).expect("EOS grid point");
            assert!(eos.rho > 0.0);
            assert!(eos.pressure > 0.0);
            assert!(eos.entropy > 0.0);
            assert!(eos.drho_dt > 0.0);
            assert!(eos.ds_dt > 0.0);
            assert!(eos.dt_d_n < 0.0);
        }
    }
}
