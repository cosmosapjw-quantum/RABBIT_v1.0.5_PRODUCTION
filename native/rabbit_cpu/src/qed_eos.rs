//! Finite-temperature QED corrections to the isotropic electromagnetic EOS.
//!
//! This module implements the zero-chemical-potential pressure convention of
//! PRIMAT-Main: the log-independent and exchange pieces at `O(e^2)`, followed
//! by the ring contribution at `O(e^3)`.  Energy and entropy corrections are
//! derived from that one pressure function; no thermal-mass substitution is
//! made in the ideal-gas EOS.

#![cfg_attr(not(test), allow(dead_code))]

use std::f64::consts::PI;
use std::sync::OnceLock;

pub(crate) const FINE_STRUCTURE_ALPHA: f64 = 1.0 / 137.035_999_084;

const DEFAULT_SINGLE_INTEGRAL_PANELS: usize = 256;
const DEFAULT_EXCHANGE_ORDER: usize = 48;
const DEFAULT_TAIL_E_FOLDS: f64 = 48.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FiniteTemperatureQed {
    Off,
    /// The historically common PRIMAT approximation `dPa + dPe3`.
    /// This omits the log-dependent exchange part of `O(e^2)`.
    PrimatLeadingE2E3,
    /// The recommended PRIMAT-Main scalar pressure `dPa + dPb + dPe3`.
    PrimatCompleteE2E3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum QedEosError {
    InvalidInput { quantity: &'static str, value: f64 },
    InvalidQuadratureOrder(usize),
    NonFiniteOutput { quantity: &'static str, value: f64 },
    NegativeRingBase(f64),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PressureDerivatives {
    pub(crate) pressure: f64,
    pub(crate) dpressure_dt: f64,
    pub(crate) d2pressure_dt2: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct QedThermoCorrection {
    pub(crate) e2_log_independent: PressureDerivatives,
    pub(crate) e2_exchange: PressureDerivatives,
    pub(crate) e3_ring: PressureDerivatives,
    pub(crate) pressure: f64,
    pub(crate) dpressure_dt: f64,
    pub(crate) d2pressure_dt2: f64,
    pub(crate) energy_density: f64,
    pub(crate) denergy_density_dt: f64,
    pub(crate) entropy_density: f64,
    pub(crate) dentropy_density_dt: f64,
}

#[derive(Clone, Copy, Debug)]
struct IntegralDerivatives {
    value: f64,
    first: f64,
    second: f64,
}

impl PressureDerivatives {
    fn from_dimensionless(temperature: f64, x: f64, value: f64, first: f64, second: f64) -> Self {
        Self {
            pressure: temperature.powi(4) * value,
            dpressure_dt: temperature.powi(3) * (4.0 * value - x * first),
            d2pressure_dt2: temperature.powi(2) * (12.0 * value - 6.0 * x * first + x * x * second),
        }
    }
}

fn require_positive_finite(quantity: &'static str, value: f64) -> Result<f64, QedEosError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(QedEosError::InvalidInput { quantity, value });
    }
    Ok(value)
}

fn fermi_occupation(energy: f64) -> f64 {
    let exp_negative = (-energy).exp();
    exp_negative / (1.0 + exp_negative)
}

/// Evaluate `I01`, `I2,-1`, and their first two derivatives with respect to
/// `x=m_e/T`.  Derivatives are taken at fixed dimensionless momentum before
/// the regularising substitution `p=x sinh(theta)` is applied.
fn fermi_integrals_with_quadrature(
    x: f64,
    panels: usize,
    tail_e_folds: f64,
) -> Result<(IntegralDerivatives, IntegralDerivatives), QedEosError> {
    let x = require_positive_finite("m_e/T", x)?;
    let tail_e_folds = require_positive_finite("quadrature_tail_e_folds", tail_e_folds)?;
    if panels < 2 || !panels.is_multiple_of(2) {
        return Err(QedEosError::InvalidQuadratureOrder(panels));
    }

    let theta_max = (1.0 + tail_e_folds / x).acosh();
    let step = theta_max / panels as f64;
    let mut sums = [0.0; 6];
    for index in 0..=panels {
        let theta = index as f64 * step;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let momentum = x * sinh;
        let energy = x * cosh;
        let dp_dtheta = x * cosh;
        let occupation = fermi_occupation(energy);
        let response = occupation * (1.0 - occupation);
        let response_prime = response * (1.0 - 2.0 * occupation);
        let energy_x = x / energy;
        let energy_xx = momentum * momentum / energy.powi(3);
        let momentum2 = momentum * momentum;

        let a_energy_first = -response / energy - occupation / energy.powi(2);
        let a_energy_second = response_prime / energy
            + 2.0 * response / energy.powi(2)
            + 2.0 * occupation / energy.powi(3);
        let b_energy_first = occupation - energy * response;
        let b_energy_second = -2.0 * response + energy * response_prime;
        let values = [
            momentum2 * occupation / energy,
            momentum2 * a_energy_first * energy_x,
            momentum2 * (a_energy_second * energy_x * energy_x + a_energy_first * energy_xx),
            energy * occupation,
            b_energy_first * energy_x,
            b_energy_second * energy_x * energy_x + b_energy_first * energy_xx,
        ];
        let weight = if index == 0 || index == panels {
            1.0
        } else if index % 2 == 0 {
            2.0
        } else {
            4.0
        };
        for component in 0..6 {
            sums[component] += weight * values[component] * dp_dtheta;
        }
    }
    for sum in &mut sums {
        *sum *= step / 3.0;
    }
    if sums.iter().any(|value| !value.is_finite()) {
        return Err(QedEosError::NonFiniteOutput {
            quantity: "QED Fermi integral",
            value: f64::NAN,
        });
    }
    Ok((
        IntegralDerivatives {
            value: sums[0],
            first: sums[1],
            second: sums[2],
        },
        IntegralDerivatives {
            value: sums[3],
            first: sums[4],
            second: sums[5],
        },
    ))
}

fn gauss_legendre_nodes_weights(order: usize) -> Result<Vec<(f64, f64)>, QedEosError> {
    if order < 2 {
        return Err(QedEosError::InvalidQuadratureOrder(order));
    }
    let mut nodes_weights = Vec::with_capacity(order);
    let roots = order.div_ceil(2);
    for root_index in 0..roots {
        let mut root = (PI * (root_index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let derivative = loop {
            let mut p_nm2 = 1.0;
            let mut p_nm1 = root;
            for degree in 2..=order {
                let p_n = ((2 * degree - 1) as f64 * root * p_nm1 - (degree - 1) as f64 * p_nm2)
                    / degree as f64;
                p_nm2 = p_nm1;
                p_nm1 = p_n;
            }
            let derivative = order as f64 * (root * p_nm1 - p_nm2) / (root * root - 1.0);
            let next = root - p_nm1 / derivative;
            if (next - root).abs() <= 4.0 * f64::EPSILON * next.abs().max(1.0) {
                root = next;
                break derivative;
            }
            root = next;
        };
        let weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        if root_index == order - 1 - root_index {
            nodes_weights.push((0.0, weight));
        } else {
            nodes_weights.push((-root, weight));
            nodes_weights.push((root, weight));
        }
    }
    nodes_weights.sort_by(|left, right| left.0.total_cmp(&right.0));
    if nodes_weights.len() != order
        || nodes_weights
            .iter()
            .any(|(node, weight)| !node.is_finite() || !weight.is_finite() || *weight <= 0.0)
    {
        return Err(QedEosError::NonFiniteOutput {
            quantity: "Gauss-Legendre rule",
            value: f64::NAN,
        });
    }
    Ok(nodes_weights)
}

static DEFAULT_EXCHANGE_RULE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();

/// Evaluate the dimensionless log-dependent exchange integral and its first
/// two `x` derivatives.  On the symmetric triangle use
/// `u=p1+p2`, `v=p1-p2=u z^2`; the Jacobian supplies `2 u z`, so no node ever
/// samples the integrable logarithmic diagonal at `z=0`.
fn exchange_integral_with_quadrature(
    x: f64,
    order: usize,
    tail_e_folds: f64,
) -> Result<IntegralDerivatives, QedEosError> {
    let x = require_positive_finite("m_e/T", x)?;
    let tail_e_folds = require_positive_finite("quadrature_tail_e_folds", tail_e_folds)?;
    let generated_rule;
    let rule = if order == DEFAULT_EXCHANGE_ORDER {
        DEFAULT_EXCHANGE_RULE
            .get_or_init(|| gauss_legendre_nodes_weights(DEFAULT_EXCHANGE_ORDER).unwrap())
            .as_slice()
    } else {
        generated_rule = gauss_legendre_nodes_weights(order)?;
        generated_rule.as_slice()
    };
    let momentum_tail = ((x + tail_e_folds).powi(2) - x * x).sqrt();
    let theta_max = momentum_tail.asinh();
    let mut sums = [0.0; 3];

    for (theta_node, theta_weight) in rule {
        let theta = 0.5 * theta_max * (theta_node + 1.0);
        let u = theta.sinh();
        let du_dtheta = theta.cosh();
        let mapped_theta_weight = 0.5 * theta_max * theta_weight;
        for (z_node, z_weight) in rule {
            let z = 0.5 * (z_node + 1.0);
            let mapped_z_weight = 0.5 * z_weight;
            let z2 = z * z;
            let momentum1 = 0.5 * u * (1.0 + z2);
            let momentum2 = 0.5 * u * (1.0 - z2);
            let energy1 = momentum1.hypot(x);
            let energy2 = momentum2.hypot(x);
            let occupation1 = fermi_occupation(energy1);
            let occupation2 = fermi_occupation(energy2);
            let response1 = occupation1 * (1.0 - occupation1);
            let response2 = occupation2 * (1.0 - occupation2);
            let complement1 = 1.0 - occupation1;
            let complement2 = 1.0 - occupation2;
            let jacobian = 2.0 * u * z * du_dtheta;
            let logarithm = -2.0 * z.ln();
            let kernel = momentum1 * momentum2 / (energy1 * energy2)
                * logarithm
                * occupation1
                * occupation2
                * jacobian;

            let log_first1 = -x * (complement1 / energy1 + energy1.recip().powi(2));
            let log_first2 = -x * (complement2 / energy2 + energy2.recip().powi(2));
            let log_second1 = -complement1 / energy1
                - energy1.recip().powi(2)
                - x * x * response1 / energy1.powi(2)
                + x * x * complement1 / energy1.powi(3)
                + 2.0 * x * x / energy1.powi(4);
            let log_second2 = -complement2 / energy2
                - energy2.recip().powi(2)
                - x * x * response2 / energy2.powi(2)
                + x * x * complement2 / energy2.powi(3)
                + 2.0 * x * x / energy2.powi(4);
            let log_first = log_first1 + log_first2;
            let values = [
                kernel,
                kernel * log_first,
                kernel * (log_first * log_first + log_second1 + log_second2),
            ];
            let weight = mapped_theta_weight * mapped_z_weight;
            for component in 0..3 {
                sums[component] += weight * values[component];
            }
        }
    }
    if sums.iter().any(|value| !value.is_finite()) {
        return Err(QedEosError::NonFiniteOutput {
            quantity: "QED exchange integral",
            value: f64::NAN,
        });
    }
    Ok(IntegralDerivatives {
        value: sums[0],
        first: sums[1],
        second: sums[2],
    })
}

fn qed_correction_with_quadrature(
    temperature: f64,
    electron_mass: f64,
    model: FiniteTemperatureQed,
    single_panels: usize,
    exchange_order: usize,
    tail_e_folds: f64,
) -> Result<QedThermoCorrection, QedEosError> {
    if model == FiniteTemperatureQed::Off {
        return Ok(QedThermoCorrection::default());
    }
    let temperature = require_positive_finite("T_gamma", temperature)?;
    let electron_mass = require_positive_finite("m_e", electron_mass)?;
    let x = electron_mass / temperature;
    let (a, b) = fermi_integrals_with_quadrature(x, single_panels, tail_e_folds)?;

    let e2_prefactor = FINE_STRUCTURE_ALPHA / PI;
    let e2_value = e2_prefactor * (-2.0 * a.value / 3.0 - 2.0 * a.value.powi(2) / PI.powi(2));
    let e2_first = e2_prefactor * (-2.0 * a.first / 3.0 - 4.0 * a.value * a.first / PI.powi(2));
    let e2_second = e2_prefactor
        * (-2.0 * a.second / 3.0 - 4.0 * (a.first.powi(2) + a.value * a.second) / PI.powi(2));
    let e2_log_independent =
        PressureDerivatives::from_dimensionless(temperature, x, e2_value, e2_first, e2_second);

    let ring_sum = a.value + b.value;
    if ring_sum < 0.0 {
        return Err(QedEosError::NegativeRingBase(ring_sum));
    }
    let (e3_value, e3_first, e3_second) = if ring_sum == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        let e3_prefactor =
            FINE_STRUCTURE_ALPHA.powf(1.5) * (4.0 / 3.0) * (2.0 * PI).sqrt() / PI.powi(3);
        (
            e3_prefactor * ring_sum.powf(1.5),
            e3_prefactor * 1.5 * ring_sum.sqrt() * (a.first + b.first),
            e3_prefactor
                * (0.75 * (a.first + b.first).powi(2) / ring_sum.sqrt()
                    + 1.5 * ring_sum.sqrt() * (a.second + b.second)),
        )
    };
    let e3_ring =
        PressureDerivatives::from_dimensionless(temperature, x, e3_value, e3_first, e3_second);

    let e2_exchange = if model == FiniteTemperatureQed::PrimatCompleteE2E3 {
        let exchange = exchange_integral_with_quadrature(x, exchange_order, tail_e_folds)?;
        let prefactor = FINE_STRUCTURE_ALPHA / PI.powi(3);
        let value = prefactor * x * x * exchange.value;
        let first = prefactor * (2.0 * x * exchange.value + x * x * exchange.first);
        let second =
            prefactor * (2.0 * exchange.value + 4.0 * x * exchange.first + x * x * exchange.second);
        PressureDerivatives::from_dimensionless(temperature, x, value, first, second)
    } else {
        PressureDerivatives::default()
    };

    let pressure = e2_log_independent.pressure + e2_exchange.pressure + e3_ring.pressure;
    let dpressure_dt =
        e2_log_independent.dpressure_dt + e2_exchange.dpressure_dt + e3_ring.dpressure_dt;
    let d2pressure_dt2 =
        e2_log_independent.d2pressure_dt2 + e2_exchange.d2pressure_dt2 + e3_ring.d2pressure_dt2;
    let correction = QedThermoCorrection {
        e2_log_independent,
        e2_exchange,
        e3_ring,
        pressure,
        dpressure_dt,
        d2pressure_dt2,
        energy_density: temperature * dpressure_dt - pressure,
        denergy_density_dt: temperature * d2pressure_dt2,
        entropy_density: dpressure_dt,
        dentropy_density_dt: d2pressure_dt2,
    };
    let values = [
        correction.pressure,
        correction.dpressure_dt,
        correction.d2pressure_dt2,
        correction.energy_density,
        correction.denergy_density_dt,
        correction.entropy_density,
        correction.dentropy_density_dt,
    ];
    if let Some(value) = values.into_iter().find(|value| !value.is_finite()) {
        return Err(QedEosError::NonFiniteOutput {
            quantity: "finite-temperature QED correction",
            value,
        });
    }
    Ok(correction)
}

pub(crate) fn qed_correction(
    temperature: f64,
    electron_mass: f64,
    model: FiniteTemperatureQed,
) -> Result<QedThermoCorrection, QedEosError> {
    qed_correction_with_quadrature(
        temperature,
        electron_mass,
        model,
        DEFAULT_SINGLE_INTEGRAL_PANELS,
        DEFAULT_EXCHANGE_ORDER,
        DEFAULT_TAIL_E_FOLDS,
    )
}

/// Relativistic-limit electromagnetic entropy coefficient `s/T^3` for the
/// chosen scalar QED convention.  The exchange term vanishes as `x^2`.
pub(crate) fn high_temperature_em_entropy_coefficient(model: FiniteTemperatureQed) -> f64 {
    if model == FiniteTemperatureQed::Off {
        return 11.0 * PI * PI / 45.0;
    }
    let f_infinity = -5.0 * PI * FINE_STRUCTURE_ALPHA / 72.0
        + (2.0 / 9.0) * (PI / 3.0).sqrt() * FINE_STRUCTURE_ALPHA.powf(1.5);
    11.0 * PI * PI / 45.0 + 4.0 * f_infinity
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELECTRON_MASS_MEV: f64 = 0.510_998_950_0;

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / expected.abs().max(1.0e-300)
    }

    #[test]
    fn gauss_legendre_rule_integrates_polynomials() {
        let rule = gauss_legendre_nodes_weights(16).unwrap();
        for power in 0_i32..=30 {
            let actual: f64 = rule
                .iter()
                .map(|(node, weight)| weight * node.powi(power))
                .sum();
            let expected = if power % 2 == 0 {
                2.0 / (power + 1) as f64
            } else {
                0.0
            };
            assert!((actual - expected).abs() < 2.0e-14);
        }
    }

    #[test]
    fn fermi_integrals_match_independent_adaptive_anchors() {
        let (a, b) = fermi_integrals_with_quadrature(1.0, 256, 48.0).unwrap();
        assert!(relative_error(a.value, 0.542_873_833_239_153) < 2.0e-13);
        assert!(relative_error(b.value, 0.876_347_374_241_896_5) < 2.0e-13);
    }

    #[test]
    fn pressure_terms_match_independent_primat_formulation() {
        // scipy.integrate.quad/dblquad evaluation of the public PRIMAT
        // formulas, with the Rust electron mass supplied explicitly.
        let references = [
            (0.1, -2.842_143_455_479_594_4e-9, 2.848_723_756_478_649e-10),
            (0.2, -4.482_918_167_365_504e-7, 6.366_325_104_940_779e-8),
            (1.0, -1.332_437_398_710_869_1e-3, 1.336_288_724_564_464_7e-4),
            (10.0, -1.585_910_175_453_058_6e1, 1.416_743_675_181_026_3),
        ];
        for (temperature, e2, e3) in references {
            let correction = qed_correction(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatLeadingE2E3,
            )
            .unwrap();
            assert!(relative_error(correction.e2_log_independent.pressure, e2) < 3.0e-11);
            assert!(relative_error(correction.e3_ring.pressure, e3) < 3.0e-11);
            assert_eq!(correction.e2_exchange, PressureDerivatives::default());
        }
    }

    #[test]
    fn exchange_pressure_derivatives_match_independent_dblquad_anchors() {
        // scipy.integrate.dblquad on the public PRIMAT dPb integrand, followed
        // by five-point temperature stencils.  This path shares neither the
        // Rust triangle mapping nor its analytic x derivatives.
        let references = [
            (
                0.2,
                2.016_422_563_112_374_3e-8,
                6.931_412_962_13e-7,
                1.764_081_55e-5,
            ),
            (
                1.0,
                1.772_055_038_424_537_4e-5,
                4.830_520_384_28e-5,
                6.753_485_61e-5,
            ),
            (
                10.0,
                3.035_413_285_796_833_5e-3,
                6.171_474_607_21e-4,
                6.228_277_91e-5,
            ),
        ];
        for (temperature, expected, expected_first, expected_second) in references {
            let correction = qed_correction(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatCompleteE2E3,
            )
            .unwrap();
            assert!(
                relative_error(correction.e2_exchange.pressure, expected) < 8.0e-6,
                "T={temperature}: actual={:.16e}, expected={expected:.16e}",
                correction.e2_exchange.pressure
            );
            assert!(
                relative_error(correction.e2_exchange.dpressure_dt, expected_first) < 1.0e-5,
                "T={temperature}: dP/dT actual={:.16e}, expected={expected_first:.16e}",
                correction.e2_exchange.dpressure_dt
            );
            assert!(
                relative_error(correction.e2_exchange.d2pressure_dt2, expected_second) < 2.0e-5,
                "T={temperature}: d2P/dT2 actual={:.16e}, expected={expected_second:.16e}",
                correction.e2_exchange.d2pressure_dt2
            );
        }
    }

    #[test]
    fn pressure_derivatives_match_five_point_differences() {
        for model in [
            FiniteTemperatureQed::PrimatLeadingE2E3,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        ] {
            for temperature in [0.1, 0.5, 1.0, 10.0] {
                let step = 2.0e-4 * temperature;
                let pressure = |offset: f64| {
                    qed_correction(temperature + offset, ELECTRON_MASS_MEV, model)
                        .unwrap()
                        .pressure
                };
                let m2 = pressure(-2.0 * step);
                let m1 = pressure(-step);
                let center = pressure(0.0);
                let p1 = pressure(step);
                let p2 = pressure(2.0 * step);
                let first = (m2 - 8.0 * m1 + 8.0 * p1 - p2) / (12.0 * step);
                let second =
                    (-p2 + 16.0 * p1 - 30.0 * center + 16.0 * m1 - m2) / (12.0 * step * step);
                let analytic = qed_correction(temperature, ELECTRON_MASS_MEV, model).unwrap();
                assert!(relative_error(analytic.dpressure_dt, first) < 2.0e-8);
                assert!(relative_error(analytic.d2pressure_dt2, second) < 2.0e-7);
            }
        }
    }

    #[test]
    fn quadratures_converge_in_value_and_derivatives() {
        for temperature in [0.1, 0.5, 1.0, 10.0] {
            let medium = qed_correction_with_quadrature(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatCompleteE2E3,
                256,
                48,
                48.0,
            )
            .unwrap();
            let fine = qed_correction_with_quadrature(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatCompleteE2E3,
                512,
                96,
                56.0,
            )
            .unwrap();
            assert!(relative_error(medium.pressure, fine.pressure) < 2.0e-7);
            assert!(relative_error(medium.dpressure_dt, fine.dpressure_dt) < 2.0e-7);
            assert!(relative_error(medium.d2pressure_dt2, fine.d2pressure_dt2) < 3.0e-7);
            assert!(
                relative_error(medium.e2_exchange.pressure, fine.e2_exchange.pressure) < 3.0e-7
            );
            assert!(
                relative_error(
                    medium.e2_exchange.dpressure_dt,
                    fine.e2_exchange.dpressure_dt
                ) < 3.0e-7
            );
            assert!(
                relative_error(
                    medium.e2_exchange.d2pressure_dt2,
                    fine.e2_exchange.d2pressure_dt2
                ) < 3.0e-7
            );
        }
    }

    #[test]
    fn thermodynamic_identities_signs_and_limits_hold() {
        for temperature in [0.005, 0.02, 0.1, 0.5, 1.0, 10.0, 100.0] {
            let correction = qed_correction(
                temperature,
                ELECTRON_MASS_MEV,
                FiniteTemperatureQed::PrimatCompleteE2E3,
            )
            .unwrap();
            assert!(correction.e2_log_independent.pressure < 0.0);
            assert!(correction.e2_exchange.pressure > 0.0);
            assert!(correction.e3_ring.pressure > 0.0);
            assert!(correction.pressure < 0.0);
            assert!(correction.energy_density < 0.0);
            assert!(correction.entropy_density < 0.0);
            assert_eq!(
                correction.energy_density,
                temperature * correction.dpressure_dt - correction.pressure
            );
            assert_eq!(correction.entropy_density, correction.dpressure_dt);
            assert_eq!(
                correction.denergy_density_dt,
                temperature * correction.d2pressure_dt2
            );
            assert_eq!(correction.dentropy_density_dt, correction.d2pressure_dt2);
        }

        let hot = qed_correction(
            1.0e4,
            ELECTRON_MASS_MEV,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        )
        .unwrap();
        let f_infinity = -5.0 * PI * FINE_STRUCTURE_ALPHA / 72.0
            + (2.0 / 9.0) * (PI / 3.0).sqrt() * FINE_STRUCTURE_ALPHA.powf(1.5);
        assert!(relative_error(hot.pressure / 1.0e16, f_infinity) < 2.0e-6);
        assert!(relative_error(hot.energy_density / 1.0e16, 3.0 * f_infinity) < 3.0e-6);

        let cold = qed_correction(
            0.005,
            ELECTRON_MASS_MEV,
            FiniteTemperatureQed::PrimatCompleteE2E3,
        )
        .unwrap();
        assert!(cold.pressure != 0.0);
        assert!(cold.pressure.abs() / 0.005_f64.powi(4) < 1.0e-40);
    }

    #[test]
    fn off_is_an_exact_zero_without_validating_unused_inputs() {
        assert_eq!(
            qed_correction(f64::NAN, f64::NAN, FiniteTemperatureQed::Off).unwrap(),
            QedThermoCorrection::default()
        );
    }
}
