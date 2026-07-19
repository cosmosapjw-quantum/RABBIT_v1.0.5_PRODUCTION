//! Complete finite-temperature radiative correction to the n<->p weak rates.
//!
//! This is the Brown--Sawyer/Pitrou et al. `L_CCRTh` block used by PRIMAT
//! v0.3.2: true thermal photons, the crossed-process differential-
//! bremsstrahlung correction, and the principal-value-free `L1 + L2+3`
//! virtual/medium block.  Only their complete signed sum is a runtime model.
//! The direct evaluator is deterministic; the repeated-run path builds one
//! private in-memory table for the leading-QED instantaneous-decoupling
//! profile and locally interpolates it.  No on-disk cache is an authority.

#![cfg_attr(not(test), allow(dead_code))]

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::born_weak::fermi_coulomb_factor;
use crate::flrw::IdealFlrwSystem;
use crate::qed_eos::FiniteTemperatureQed;

pub(crate) const MEV_TO_KELVIN: f64 = 11_604_518_121.550_083;
pub(crate) const THERMAL_RADIATIVE_FLOOR_KELVIN: f64 = 158_489_319.246_111_1;
pub(crate) const THERMAL_RADIATIVE_MAXIMUM_MEV: f64 = 10.0;
const THERMAL_TABLE_POINT_COUNT: usize = 57;
const DEFAULT_DIRECT_QUADRATURE_ORDER: usize = 64;
const MIN_DIRECT_QUADRATURE_ORDER: usize = 8;
const MAX_DIRECT_QUADRATURE_ORDER: usize = 128;
const ROOT_ITERATION_LIMIT: usize = 64;
const FINE_STRUCTURE_CONSTANT: f64 = 1.0 / 137.035_999_084;
const ELECTRON_MASS_MEV: f64 = 0.510_998_950_0;
const NEUTRON_PROTON_MASS_DIFFERENCE_MEV: f64 = 1.293_332_36;
const ELECTRON_ENERGY_OFFSET: f64 = 1.0e-3;
const PHOTON_ENERGY_FLOOR: f64 = 1.0e-3;
const ENERGY_DIFFERENCE_FLOOR: f64 = 1.0e-3;
const REAL_PHOTON_ELECTRON_PANELS: usize = 8;
const PROFILE_RELATIVE_TOLERANCE: f64 = 2.0e-10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DirectionalThermalRadiativeComponents {
    pub(crate) true_photon: f64,
    pub(crate) differential_bremsstrahlung: f64,
    pub(crate) medium_one_dimensional: f64,
    pub(crate) medium_two_dimensional: f64,
}

impl DirectionalThermalRadiativeComponents {
    pub(crate) fn total(self) -> f64 {
        self.true_photon
            + self.differential_bremsstrahlung
            + self.medium_one_dimensional
            + self.medium_two_dimensional
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ThermalRadiativeRawPair {
    pub(crate) neutron_to_proton: f64,
    pub(crate) proton_to_neutron: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ThermalRadiativeError {
    InvalidTemperature {
        field: &'static str,
        raw_value_mev: f64,
    },
    InvalidQuadratureOrder {
        raw_order: usize,
        minimum: usize,
        maximum: usize,
    },
    InvalidDirection {
        raw_direction: i8,
    },
    QuadratureRootDidNotConverge {
        order: usize,
        root_index: usize,
        raw_root: f64,
    },
    ProfileMismatch {
        photon_temperature_mev: f64,
        provided_neutrino_temperature_mev: f64,
        expected_neutrino_temperature_mev: f64,
        relative_residual: f64,
    },
    TemperatureAboveTable {
        raw_value_mev: f64,
        maximum_mev: f64,
    },
    BackgroundFailure {
        photon_temperature_mev: f64,
    },
    NonFiniteComponent {
        component: &'static str,
        direction: &'static str,
        photon_temperature_mev: f64,
        raw_value: f64,
    },
    NonPositiveProfileTableValue {
        direction: &'static str,
        photon_temperature_mev: f64,
        raw_value: f64,
    },
}

#[derive(Debug, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn new() -> Self {
        Self {
            sum: 0.0,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }
}

#[derive(Debug, Clone)]
struct ThermalRadiativeTable {
    temperature_kelvin: Vec<f64>,
    neutron_to_proton: Vec<f64>,
    proton_to_neutron: Vec<f64>,
}

static COMPLETE_THERMAL_TABLE: OnceLock<Result<ThermalRadiativeTable, ThermalRadiativeError>> =
    OnceLock::new();

pub(crate) fn expected_profile_neutrino_temperature(
    photon_temperature_mev: f64,
) -> Result<f64, ThermalRadiativeError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    IdealFlrwSystem::high_temperature_instantaneous_decoupling_with_qed(
        FiniteTemperatureQed::PrimatLeadingE2E3,
    )
    .thermo_state(photon_temperature_mev)
    .map(|state| state.t_nu_mev)
    .map_err(|_| ThermalRadiativeError::BackgroundFailure {
        photon_temperature_mev,
    })
}

/// Complete profile-specific raw correction used by the repeated-run path.
///
/// The returned values are dimensionless phase-space integrals.  The caller
/// must divide by the unchanged F08C `F_n * tau_n` normalization.
pub(crate) fn complete_thermal_radiative_raw(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
) -> Result<ThermalRadiativeRawPair, ThermalRadiativeError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    validate_positive_temperature("neutrino_temperature_mev", neutrino_temperature_mev)?;
    if photon_temperature_mev > THERMAL_RADIATIVE_MAXIMUM_MEV {
        return Err(ThermalRadiativeError::TemperatureAboveTable {
            raw_value_mev: photon_temperature_mev,
            maximum_mev: THERMAL_RADIATIVE_MAXIMUM_MEV,
        });
    }

    let expected = expected_profile_neutrino_temperature(photon_temperature_mev)?;
    let relative_residual = (neutrino_temperature_mev - expected) / expected;
    if relative_residual.abs() > PROFILE_RELATIVE_TOLERANCE {
        return Err(ThermalRadiativeError::ProfileMismatch {
            photon_temperature_mev,
            provided_neutrino_temperature_mev: neutrino_temperature_mev,
            expected_neutrino_temperature_mev: expected,
            relative_residual,
        });
    }

    let temperature_kelvin = photon_temperature_mev * MEV_TO_KELVIN;
    if temperature_kelvin < THERMAL_RADIATIVE_FLOOR_KELVIN {
        return Ok(ThermalRadiativeRawPair {
            neutron_to_proton: 0.0,
            proton_to_neutron: 0.0,
        });
    }

    let table = COMPLETE_THERMAL_TABLE
        .get_or_init(build_complete_thermal_table)
        .as_ref()
        .map_err(Clone::clone)?;
    let pair = ThermalRadiativeRawPair {
        neutron_to_proton: interpolate_local_cubic(
            &table.temperature_kelvin,
            &table.neutron_to_proton,
            temperature_kelvin,
        ),
        proton_to_neutron: interpolate_positive_log_log_cubic(
            &table.temperature_kelvin,
            &table.proton_to_neutron,
            temperature_kelvin,
        ),
    };
    validate_component(
        "interpolated_complete",
        "neutron_to_proton",
        photon_temperature_mev,
        pair.neutron_to_proton,
    )?;
    validate_component(
        "interpolated_complete",
        "proton_to_neutron",
        photon_temperature_mev,
        pair.proton_to_neutron,
    )?;
    Ok(pair)
}

fn build_complete_thermal_table() -> Result<ThermalRadiativeTable, ThermalRadiativeError> {
    let maximum_kelvin = THERMAL_RADIATIVE_MAXIMUM_MEV * MEV_TO_KELVIN;
    let log_minimum = THERMAL_RADIATIVE_FLOOR_KELVIN.ln();
    let log_span = maximum_kelvin.ln() - log_minimum;
    let mut temperature_kelvin = Vec::with_capacity(THERMAL_TABLE_POINT_COUNT);
    let mut neutron_to_proton = Vec::with_capacity(THERMAL_TABLE_POINT_COUNT);
    let mut proton_to_neutron = Vec::with_capacity(THERMAL_TABLE_POINT_COUNT);

    for index in 0..THERMAL_TABLE_POINT_COUNT {
        let fraction = index as f64 / (THERMAL_TABLE_POINT_COUNT - 1) as f64;
        let temperature_k = if index == 0 {
            THERMAL_RADIATIVE_FLOOR_KELVIN
        } else if index + 1 == THERMAL_TABLE_POINT_COUNT {
            maximum_kelvin
        } else {
            (log_minimum + fraction * log_span).exp()
        };
        let temperature_mev = temperature_k / MEV_TO_KELVIN;
        let neutrino_temperature_mev = expected_profile_neutrino_temperature(temperature_mev)?;
        let neutron = direct_directional_components(
            temperature_mev,
            neutrino_temperature_mev,
            1,
            DEFAULT_DIRECT_QUADRATURE_ORDER,
        )?
        .total();
        let proton = direct_directional_components(
            temperature_mev,
            neutrino_temperature_mev,
            -1,
            DEFAULT_DIRECT_QUADRATURE_ORDER,
        )?
        .total();
        validate_component("complete", "neutron_to_proton", temperature_mev, neutron)?;
        validate_component("complete", "proton_to_neutron", temperature_mev, proton)?;
        if proton <= 0.0 {
            return Err(ThermalRadiativeError::NonPositiveProfileTableValue {
                direction: "proton_to_neutron",
                photon_temperature_mev: temperature_mev,
                raw_value: proton,
            });
        }
        temperature_kelvin.push(temperature_k);
        neutron_to_proton.push(neutron);
        proton_to_neutron.push(proton);
    }

    Ok(ThermalRadiativeTable {
        temperature_kelvin,
        neutron_to_proton,
        proton_to_neutron,
    })
}

fn direct_directional_components(
    photon_temperature_mev: f64,
    neutrino_temperature_mev: f64,
    direction: i8,
    quadrature_order: usize,
) -> Result<DirectionalThermalRadiativeComponents, ThermalRadiativeError> {
    validate_positive_temperature("photon_temperature_mev", photon_temperature_mev)?;
    validate_positive_temperature("neutrino_temperature_mev", neutrino_temperature_mev)?;
    if direction != 1 && direction != -1 {
        return Err(ThermalRadiativeError::InvalidDirection {
            raw_direction: direction,
        });
    }
    let nodes = gauss_legendre_unit_nodes(quadrature_order)?;
    let temperature_kelvin = photon_temperature_mev * MEV_TO_KELVIN;
    if temperature_kelvin < THERMAL_RADIATIVE_FLOOR_KELVIN {
        return Ok(DirectionalThermalRadiativeComponents {
            true_photon: 0.0,
            differential_bremsstrahlung: 0.0,
            medium_one_dimensional: 0.0,
            medium_two_dimensional: 0.0,
        });
    }
    let x = ELECTRON_MASS_MEV / photon_temperature_mev;
    let z_nu = ELECTRON_MASS_MEV / neutrino_temperature_mev;
    let true_photon = integrate_real_photon_rectangle(&nodes, x, z_nu, direction, false);
    let differential_bremsstrahlung =
        integrate_real_photon_rectangle(&nodes, x, z_nu, direction, true);
    let medium_one_dimensional = integrate_medium_one(&nodes, x, z_nu, direction);
    let medium_two_dimensional = integrate_medium_two_three(&nodes, x, z_nu, direction);
    let components = DirectionalThermalRadiativeComponents {
        true_photon,
        differential_bremsstrahlung,
        medium_one_dimensional,
        medium_two_dimensional,
    };
    for (component, value) in [
        ("true_photon", components.true_photon),
        (
            "differential_bremsstrahlung",
            components.differential_bremsstrahlung,
        ),
        ("medium_one_dimensional", components.medium_one_dimensional),
        ("medium_two_dimensional", components.medium_two_dimensional),
        ("complete", components.total()),
    ] {
        validate_component(
            component,
            direction_name(direction),
            photon_temperature_mev,
            value,
        )?;
    }
    Ok(components)
}

fn integrate_real_photon_rectangle(
    nodes: &[(f64, f64)],
    x: f64,
    z_nu: f64,
    direction: i8,
    differential_bremsstrahlung: bool,
) -> f64 {
    let maximum = (20.0 / x).max(10.0);
    let electron_span = ((maximum - 1.0) / ELECTRON_ENERGY_OFFSET).ln();
    let signed_direction = f64::from(direction);
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let mut electron_breakpoints = Vec::with_capacity(REAL_PHOTON_ELECTRON_PANELS + 2);
    for panel in 0..=REAL_PHOTON_ELECTRON_PANELS {
        electron_breakpoints.push(panel as f64 / REAL_PHOTON_ELECTRON_PANELS as f64);
    }
    if (1.0 + ELECTRON_ENERGY_OFFSET..maximum).contains(&q) {
        electron_breakpoints.push(((q - 1.0) / ELECTRON_ENERGY_OFFSET).ln() / electron_span);
        electron_breakpoints.sort_by(f64::total_cmp);
    }

    let mut sum = CompensatedSum::new();
    for electron_panel in electron_breakpoints.windows(2) {
        let electron_width = electron_panel[1] - electron_panel[0];
        for &(electron_node, electron_weight) in nodes {
            let electron_coordinate = electron_panel[0] + electron_width * electron_node;
            let electron_offset =
                ELECTRON_ENERGY_OFFSET * (electron_span * electron_coordinate).exp();
            let electron_energy = 1.0 + electron_offset;
            let electron_jacobian = electron_width * electron_span * electron_offset;

            // Both Chitilde transitions and the explicit soft subtractions lie
            // on k=|E +/- s*q|.  Making those lines inner-panel boundaries
            // prevents a tensor rule from aliasing the narrow sign-changing
            // low-temperature support while preserving every signed term.
            let mut photon_breakpoints = [
                PHOTON_ENERGY_FLOOR,
                (electron_energy - signed_direction * q).abs(),
                (electron_energy + signed_direction * q).abs(),
                maximum,
            ];
            photon_breakpoints.sort_by(f64::total_cmp);
            for photon_panel in photon_breakpoints.windows(2) {
                let photon_lower = photon_panel[0].max(PHOTON_ENERGY_FLOOR);
                let photon_upper = photon_panel[1].min(maximum);
                if photon_upper <= photon_lower {
                    continue;
                }
                let photon_log_width = (photon_upper / photon_lower).ln();
                for &(photon_node, photon_weight) in nodes {
                    let photon_energy = photon_lower * (photon_log_width * photon_node).exp();
                    let photon_jacobian = photon_log_width * photon_energy;
                    let integrand = if differential_bremsstrahlung {
                        differential_bremsstrahlung_integrand(
                            electron_energy,
                            photon_energy,
                            x,
                            z_nu,
                            direction,
                        )
                    } else {
                        true_photon_integrand(electron_energy, photon_energy, x, z_nu, direction)
                    };
                    sum.add(
                        electron_weight
                            * photon_weight
                            * electron_jacobian
                            * photon_jacobian
                            * integrand,
                    );
                }
            }
        }
    }
    sum.sum
}

fn true_photon_integrand(
    electron_energy: f64,
    photon_energy: f64,
    x: f64,
    z_nu: f64,
    direction: i8,
) -> f64 {
    let momentum = (electron_energy * electron_energy - 1.0).sqrt();
    let beta = momentum / electron_energy;
    let forward_stat = fermi_stat(direction, 1, beta);
    let reverse_stat = fermi_stat(direction, -1, beta);
    let forward_fd = fermi_dirac_from_exponent(-electron_energy * x);
    let reverse_fd = fermi_dirac_from_exponent(electron_energy * x);
    let a = thermal_a(electron_energy, photon_energy);
    let b = thermal_b(electron_energy);
    let forward_even = chitilde(electron_energy - photon_energy, z_nu, direction)
        + chitilde(electron_energy + photon_energy, z_nu, direction)
        - 2.0 * chitilde(electron_energy, z_nu, direction);
    let reverse_even = chitilde(-electron_energy + photon_energy, z_nu, direction)
        + chitilde(-electron_energy - photon_energy, z_nu, direction)
        - 2.0 * chitilde(-electron_energy, z_nu, direction);
    let forward_odd = chitilde(electron_energy - photon_energy, z_nu, direction)
        - chitilde(electron_energy + photon_energy, z_nu, direction);
    let reverse_odd = chitilde(-electron_energy + photon_energy, z_nu, direction)
        - chitilde(-electron_energy - photon_energy, z_nu, direction);
    let term_one =
        a * (forward_fd * forward_stat * forward_even + reverse_fd * reverse_stat * reverse_even);
    let term_two = photon_energy
        * b
        * (forward_fd * forward_stat * forward_odd + reverse_fd * reverse_stat * reverse_odd);
    FINE_STRUCTURE_CONSTANT / (2.0 * PI) * bose_einstein(x * photon_energy) / photon_energy
        * (term_one - term_two)
}

fn differential_bremsstrahlung_integrand(
    electron_energy: f64,
    photon_energy: f64,
    x: f64,
    z_nu: f64,
    direction: i8,
) -> f64 {
    let signed_direction = f64::from(direction);
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let momentum = (electron_energy * electron_energy - 1.0).sqrt();
    let beta = momentum / electron_energy;
    let base = thermal_a(electron_energy, photon_energy);
    let shift = photon_energy * thermal_b(electron_energy);
    let f_plus = base + shift;
    let f_minus = base - shift;

    let mut forward = f_plus * chitilde(electron_energy + photon_energy, z_nu, direction);
    let forward_gap = (electron_energy - signed_direction * q).abs();
    if photon_energy < forward_gap {
        forward -= f_plus
            * fermi_dirac_from_exponent(z_nu * (electron_energy - signed_direction * q))
            * (forward_gap - photon_energy).powi(2);
    }
    forward *= fermi_dirac_from_exponent(-electron_energy * x) * fermi_stat(direction, 1, beta);

    let mut reverse = f_minus * chitilde(-electron_energy + photon_energy, z_nu, direction);
    let reverse_gap = (electron_energy + signed_direction * q).abs();
    if photon_energy < reverse_gap {
        // Phys. Rep. B48--B49 requires F+ in this soft subtraction.
        reverse -= f_plus
            * fermi_dirac_from_exponent(z_nu * (-electron_energy - signed_direction * q))
            * (reverse_gap - photon_energy).powi(2);
    }
    reverse *= fermi_dirac_from_exponent(electron_energy * x) * fermi_stat(direction, -1, beta);

    FINE_STRUCTURE_CONSTANT / (2.0 * PI * photon_energy) * (forward + reverse)
}

fn integrate_medium_one(nodes: &[(f64, f64)], x: f64, z_nu: f64, direction: i8) -> f64 {
    let energy_maximum = (150.0 / x).max(25.0);
    let momentum_maximum = (energy_maximum * energy_maximum - 1.0).sqrt();
    let log_span = momentum_maximum.ln_1p();
    let mut sum = CompensatedSum::new();
    for &(node, weight) in nodes {
        let momentum = (log_span * node).exp_m1();
        let jacobian = log_span * (1.0 + momentum);
        let electron_energy = momentum.hypot(1.0);
        let integrand = -FINE_STRUCTURE_CONSTANT * PI / (3.0 * x * x)
            * (chi(electron_energy, x, z_nu, direction)
                + chi(-electron_energy, x, z_nu, direction));
        sum.add(weight * jacobian * integrand);
    }
    sum.sum
}

fn integrate_medium_two_three(nodes: &[(f64, f64)], x: f64, z_nu: f64, direction: i8) -> f64 {
    let half = (15.0 / x).max(10.0);
    let difference_span = (half / ENERGY_DIFFERENCE_FLOOR).ln();
    let outer_upper = 2.0 + half;
    let mut sum = CompensatedSum::new();
    for &(difference_node, difference_weight) in nodes {
        let difference = ENERGY_DIFFERENCE_FLOOR * (difference_span * difference_node).exp();
        let difference_jacobian = difference_span * difference;
        let outer_lower = (2.0 + difference).max(2.002);
        let outer_width = outer_upper - outer_lower;
        if outer_width <= 0.0 {
            continue;
        }
        for &(sum_node, sum_weight) in nodes {
            let energy_sum = outer_lower + outer_width * sum_node;
            let common_weight =
                difference_weight * sum_weight * difference_jacobian * outer_width * 0.5;
            let positive = medium_two_integrand(
                0.5 * (energy_sum + difference),
                0.5 * (energy_sum - difference),
                x,
                z_nu,
                direction,
            );
            let negative = medium_two_integrand(
                0.5 * (energy_sum - difference),
                0.5 * (energy_sum + difference),
                x,
                z_nu,
                direction,
            );
            sum.add(common_weight * (positive + negative));
        }
    }
    sum.sum
}

fn medium_two_integrand(
    first_energy: f64,
    second_energy: f64,
    x: f64,
    z_nu: f64,
    direction: i8,
) -> f64 {
    let first_momentum = (first_energy * first_energy - 1.0).sqrt();
    let second_momentum = (second_energy * second_energy - 1.0).sqrt();
    let first_rapidity = first_momentum.asinh();
    let second_rapidity = second_momentum.asinh();
    let l_factor = log_cosh_plus_one(first_rapidity + second_rapidity)
        - log_cosh_plus_one(first_rapidity - second_rapidity);
    let log_ratio_squared = 2.0
        * ((first_momentum + second_momentum).ln() - (first_momentum - second_momentum).abs().ln());
    let fd_second = fermi_dirac_from_exponent(second_energy * x);
    let dfd_second = -x * fd_second * (1.0 - fd_second);
    let chi_sum = chi(first_energy, x, z_nu, direction) + chi(-first_energy, x, z_nu, direction);
    let first_momentum_squared = first_momentum * first_momentum;
    let second_momentum_squared = second_momentum * second_momentum;

    let first_bracket = dfd_second * second_momentum / first_momentum * first_energy.powi(2)
        / second_energy
        * (first_energy + second_energy)
        + fd_second * first_energy.powi(2) / (first_momentum * second_momentum)
            * (second_energy + first_energy / second_energy.powi(2));
    let second_bracket = dfd_second
        * (second_momentum_squared * first_energy / second_energy
            * (first_momentum_squared.recip() + 2.0)
            - first_energy.powi(2) * second_momentum / first_momentum * l_factor)
        + fd_second
            * (first_energy / (first_momentum_squared * second_energy.powi(2))
                * (second_energy.powi(2) + 2.0 * first_momentum_squared + 1.0)
                - (first_energy.powi(2) + second_energy.powi(2)) / (first_energy + second_energy)
                - first_energy.powi(2) * second_energy / (first_momentum * second_momentum)
                    * l_factor);
    let term = -0.25 * log_ratio_squared.powi(2) * first_bracket
        + log_ratio_squared * second_bracket
        - fd_second
            * (4.0 * first_energy * second_momentum / first_momentum
                + 2.0 * second_energy * l_factor);
    FINE_STRUCTURE_CONSTANT / (2.0 * PI) * chi_sum * term
}

fn thermal_a(electron_energy: f64, photon_energy: f64) -> f64 {
    let momentum = (electron_energy * electron_energy - 1.0).sqrt();
    (2.0 * electron_energy.powi(2) + photon_energy.powi(2)) * (2.0 * momentum.asinh())
        - 4.0 * momentum * electron_energy
}

fn thermal_b(electron_energy: f64) -> f64 {
    let momentum = (electron_energy * electron_energy - 1.0).sqrt();
    4.0 * electron_energy * momentum.asinh() - 4.0 * momentum
}

fn chi(electron_energy: f64, x: f64, z_nu: f64, direction: i8) -> f64 {
    let signed_direction = f64::from(direction);
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let neutrino_energy = electron_energy - signed_direction * q;
    fermi_dirac_from_exponent(z_nu * neutrino_energy)
        * fermi_dirac_from_exponent(-x * electron_energy)
        * neutrino_energy.powi(2)
}

fn chitilde(energy: f64, z_nu: f64, direction: i8) -> f64 {
    let signed_direction = f64::from(direction);
    let q = NEUTRON_PROTON_MASS_DIFFERENCE_MEV / ELECTRON_MASS_MEV;
    let neutrino_energy = energy - signed_direction * q;
    // Saturating the negative Fermi-Dirac tail to one is the physical limit;
    // PRIMAT's |arg|>300 zeroing is treated as an external numerical limit.
    fermi_dirac_from_exponent(z_nu * neutrino_energy) * neutrino_energy.powi(2)
}

fn fermi_stat(direction: i8, electron_sign: i8, beta: f64) -> f64 {
    if direction * electron_sign > 0 {
        fermi_coulomb_factor(beta)
    } else {
        1.0
    }
}

fn bose_einstein(exponent: f64) -> f64 {
    exponent.exp_m1().recip()
}

fn fermi_dirac_from_exponent(exponent: f64) -> f64 {
    if exponent.is_nan() {
        return f64::NAN;
    }
    if exponent == f64::INFINITY {
        return 0.0;
    }
    if exponent == f64::NEG_INFINITY {
        return 1.0;
    }
    if exponent >= 0.0 {
        let inverse = (-exponent).exp();
        inverse / (1.0 + inverse)
    } else {
        1.0 / (1.0 + exponent.exp())
    }
}

fn log_cosh_plus_one(value: f64) -> f64 {
    let half = 0.5 * value.abs();
    2.0 * (half + (-2.0 * half).exp().ln_1p()) - 2.0_f64.ln()
}

fn interpolate_local_quadratic(x: &[f64], y: &[f64], query: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(x.len() >= 3);
    let mut low = 0;
    let mut high = x.len() - 1;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if x[middle] <= query {
            low = middle;
        } else {
            high = middle;
        }
    }
    let start = if low == 0 {
        0
    } else if low + 2 >= x.len() {
        x.len() - 3
    } else {
        low - 1
    };
    let (x0, x1, x2) = (x[start], x[start + 1], x[start + 2]);
    let l0 = (query - x1) * (query - x2) / ((x0 - x1) * (x0 - x2));
    let l1 = (query - x0) * (query - x2) / ((x1 - x0) * (x1 - x2));
    let l2 = (query - x0) * (query - x1) / ((x2 - x0) * (x2 - x1));
    y[start] * l0 + y[start + 1] * l1 + y[start + 2] * l2
}

fn interpolate_positive_log_log_cubic(x: &[f64], y: &[f64], query: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(x.len() >= 4);
    debug_assert!(query > 0.0);
    debug_assert!(x.iter().all(|coordinate| *coordinate > 0.0));
    debug_assert!(y.iter().all(|value| *value > 0.0));
    let mut low = 0;
    let mut high = x.len() - 1;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if x[middle] <= query {
            low = middle;
        } else {
            high = middle;
        }
    }
    let start = if low <= 1 {
        0
    } else if low + 2 >= x.len() {
        x.len() - 4
    } else {
        low - 1
    };
    let query = query.ln();
    let mut interpolated_log = 0.0;
    let stencil_x = &x[start..start + 4];
    let stencil_y = &y[start..start + 4];
    for (point, (&point_x, &point_y)) in stencil_x.iter().zip(stencil_y).enumerate() {
        let point_x = point_x.ln();
        let mut basis = 1.0;
        for (other, &other_x) in stencil_x.iter().enumerate() {
            if point != other {
                let other_x = other_x.ln();
                basis *= (query - other_x) / (point_x - other_x);
            }
        }
        interpolated_log += point_y.ln() * basis;
    }
    interpolated_log.exp()
}

fn interpolate_local_cubic(x: &[f64], y: &[f64], query: f64) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(x.len() >= 4);
    let mut low = 0;
    let mut high = x.len() - 1;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if x[middle] <= query {
            low = middle;
        } else {
            high = middle;
        }
    }
    let start = if low <= 1 {
        0
    } else if low + 2 >= x.len() {
        x.len() - 4
    } else {
        low - 1
    };
    let mut interpolated = 0.0;
    let stencil_x = &x[start..start + 4];
    let stencil_y = &y[start..start + 4];
    for (point, (&point_x, &point_y)) in stencil_x.iter().zip(stencil_y).enumerate() {
        let mut basis = 1.0;
        for (other, &other_x) in stencil_x.iter().enumerate() {
            if point != other {
                basis *= (query - other_x) / (point_x - other_x);
            }
        }
        interpolated += point_y * basis;
    }
    interpolated
}

fn validate_positive_temperature(
    field: &'static str,
    raw_value_mev: f64,
) -> Result<(), ThermalRadiativeError> {
    if !raw_value_mev.is_finite() || raw_value_mev <= 0.0 {
        return Err(ThermalRadiativeError::InvalidTemperature {
            field,
            raw_value_mev,
        });
    }
    Ok(())
}

fn validate_component(
    component: &'static str,
    direction: &'static str,
    photon_temperature_mev: f64,
    raw_value: f64,
) -> Result<(), ThermalRadiativeError> {
    if !raw_value.is_finite() {
        return Err(ThermalRadiativeError::NonFiniteComponent {
            component,
            direction,
            photon_temperature_mev,
            raw_value,
        });
    }
    Ok(())
}

fn direction_name(direction: i8) -> &'static str {
    if direction == 1 {
        "neutron_to_proton"
    } else {
        "proton_to_neutron"
    }
}

fn gauss_legendre_unit_nodes(order: usize) -> Result<Vec<(f64, f64)>, ThermalRadiativeError> {
    if !(MIN_DIRECT_QUADRATURE_ORDER..=MAX_DIRECT_QUADRATURE_ORDER).contains(&order) {
        return Err(ThermalRadiativeError::InvalidQuadratureOrder {
            raw_order: order,
            minimum: MIN_DIRECT_QUADRATURE_ORDER,
            maximum: MAX_DIRECT_QUADRATURE_ORDER,
        });
    }
    let mut nodes = vec![(0.0, 0.0); order];
    let root_count = order.div_ceil(2);
    let order_f64 = order as f64;
    for root_index in 0..root_count {
        let phase = PI * (root_index as f64 + 0.75) / (order_f64 + 0.5);
        let mut root = phase.cos();
        let mut converged = false;
        for _ in 0..ROOT_ITERATION_LIMIT {
            let (polynomial, derivative) = legendre_polynomial_and_derivative(order, root);
            let update = polynomial / derivative;
            root -= update;
            if update.abs() <= 8.0 * f64::EPSILON * (1.0 + root.abs()) {
                converged = true;
                break;
            }
        }
        if !converged || !root.is_finite() {
            return Err(ThermalRadiativeError::QuadratureRootDidNotConverge {
                order,
                root_index,
                raw_root: root,
            });
        }
        let (_, derivative) = legendre_polynomial_and_derivative(order, root);
        let full_weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        let unit_weight = 0.5 * full_weight;
        nodes[root_index] = (0.5 * (1.0 - root), unit_weight);
        nodes[order - 1 - root_index] = (0.5 * (1.0 + root), unit_weight);
    }
    Ok(nodes)
}

fn legendre_polynomial_and_derivative(order: usize, coordinate: f64) -> (f64, f64) {
    let mut previous = 1.0;
    let mut current = coordinate;
    for degree in 2..=order {
        let degree_f64 = degree as f64;
        let next = ((2.0 * degree_f64 - 1.0) * coordinate * current
            - (degree_f64 - 1.0) * previous)
            / degree_f64;
        previous = current;
        current = next;
    }
    let derivative =
        order as f64 * (coordinate * current - previous) / (coordinate * coordinate - 1.0);
    (current, derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_distribution_limits_and_log_identity_are_finite() {
        assert_eq!(fermi_dirac_from_exponent(1.0e4), 0.0);
        assert_eq!(fermi_dirac_from_exponent(-1.0e4), 1.0);
        assert!(bose_einstein(0.1).is_finite());
        for value in [0.0, 1.0, 20.0, 1000.0] {
            assert!(log_cosh_plus_one(value).is_finite());
        }
        assert_eq!(log_cosh_plus_one(0.0).to_bits(), 2.0_f64.ln().to_bits());
    }

    #[test]
    fn invalid_direct_quadrature_order_fails_closed() {
        assert!(matches!(
            gauss_legendre_unit_nodes(1),
            Err(ThermalRadiativeError::InvalidQuadratureOrder { .. })
        ));
        assert!(matches!(
            direct_directional_components(1.0, 1.0, 0, 32),
            Err(ThermalRadiativeError::InvalidDirection { raw_direction: 0 })
        ));
    }

    #[test]
    fn local_quadratic_interpolation_reproduces_knots() {
        let x = [1.0, 2.0, 4.0, 8.0];
        let y = [1.0, 4.0, 16.0, 64.0];
        for (coordinate, expected) in x.into_iter().zip(y) {
            assert_eq!(interpolate_local_quadratic(&x, &y, coordinate), expected);
            assert!(
                (interpolate_positive_log_log_cubic(&x, &y, coordinate) / expected - 1.0).abs()
                    < 2.0e-15
            );
            assert_eq!(interpolate_local_cubic(&x, &y, coordinate), expected);
        }
    }

    #[test]
    fn f08d_direct_complete_sum_matches_frozen_external_envelopes_and_refines() {
        // These broad white-box envelopes were frozen from independent C and
        // SciPy probes before any Rust F08D numerical value was read.  The
        // flat Tnu/Tgamma profile is diagnostic-only, not a cosmological run.
        let rows = [
            (
                2.0e8,
                1,
                5.993_377_680_674_609e-7,
                5.993_377_680_674_609e-7 * 0.85,
                5.993_377_680_674_609e-7 * 1.15,
            ),
            (2.0e8, -1, 1.171_420_408_307_136_3e-37, -1.0e-35, 1.0e-35),
            (
                1.0e9,
                1,
                1.854_915e-4,
                1.854_915e-4 * 0.97,
                1.854_915e-4 * 1.03,
            ),
            (
                1.0e9,
                -1,
                2.822_955_660_681_275e-10,
                2.822_955_660_681_275e-10 * 0.2,
                2.822_955_660_681_275e-10 * 5.0,
            ),
            (
                1.0e10,
                1,
                -1.678_494_1,
                -1.678_494_1 * 1.01,
                -1.678_494_1 * 0.99,
            ),
            (
                1.0e10,
                -1,
                0.486_649_22,
                0.486_649_22 * 0.99,
                0.486_649_22 * 1.01,
            ),
        ];
        let mut failures = Vec::new();
        for (temperature_kelvin, direction, reference, lower, upper) in rows {
            let photon_temperature_mev = temperature_kelvin / MEV_TO_KELVIN;
            let neutrino_temperature_mev = 0.7138 * photon_temperature_mev;
            let components_64 = direct_directional_components(
                photon_temperature_mev,
                neutrino_temperature_mev,
                direction,
                64,
            )
            .expect("order-64 direct F08D evaluation");
            let components_128 = direct_directional_components(
                photon_temperature_mev,
                neutrino_temperature_mev,
                direction,
                128,
            )
            .expect("order-128 direct F08D evaluation");
            let order_64 = components_64.total();
            let order_128 = components_128.total();
            let refinement_ceiling = 0.5 * (upper - lower);
            println!(
                "F08D direct T_K={temperature_kelvin:.1e} direction={direction}: reference={reference:.17e}; order64={components_64:?}, total={order_64:.17e}; order128={components_128:?}, total={order_128:.17e}"
            );
            if !(lower..=upper).contains(&order_128) {
                failures.push(format!(
                    "T_K={temperature_kelvin:.1e} direction={direction}: order128={order_128:.17e}, reference={reference:.17e}, envelope=[{lower:.17e},{upper:.17e}]"
                ));
            }
            if (order_128 - order_64).abs() > refinement_ceiling {
                failures.push(format!(
                    "T_K={temperature_kelvin:.1e} direction={direction}: order64={order_64:.17e}, order128={order_128:.17e}, refinement ceiling={refinement_ceiling:.17e}"
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn f08d_direct_covered_subterms_match_independent_dblquad_and_refine() {
        // The two externally sampled p->n decompositions are far below
        // scipy.dblquad's default absolute error scale, so only their frozen
        // complete-sum absolute/factor contracts are authoritative.  The
        // n->p row below has resolved subterms and supports relative checks.
        let rows: [(f64, i8, [f64; 4]); 1] = [(
            2.0e8,
            1,
            [
                2.207_975_994_127_729_5e-5,
                3.831_193_071_581_839_6e-7,
                -2.186_354_148_036_496_5e-5,
                -3.053_127_076_929_569_4e-18,
            ],
        )];
        for (temperature_kelvin, direction, reference) in rows {
            let photon_temperature_mev = temperature_kelvin / MEV_TO_KELVIN;
            let neutrino_temperature_mev = 0.7138 * photon_temperature_mev;
            let coarse = direct_directional_components(
                photon_temperature_mev,
                neutrino_temperature_mev,
                direction,
                64,
            )
            .expect("order-64 covered-subterm evaluation");
            let refined = direct_directional_components(
                photon_temperature_mev,
                neutrino_temperature_mev,
                direction,
                128,
            )
            .expect("order-128 covered-subterm evaluation");
            let coarse_values = [
                coarse.true_photon,
                coarse.differential_bremsstrahlung,
                coarse.medium_one_dimensional,
                coarse.medium_two_dimensional,
            ];
            let refined_values = [
                refined.true_photon,
                refined.differential_bremsstrahlung,
                refined.medium_one_dimensional,
                refined.medium_two_dimensional,
            ];
            for component in 0..4 {
                let ceiling = match component {
                    0 | 2 => 1.0e-3 * reference[component].abs(),
                    1 => 0.15 * reference[component].abs(),
                    3 => (0.2 * reference[component].abs()).max(1.0e-12),
                    _ => unreachable!(),
                };
                assert!(
                    (refined_values[component] - reference[component]).abs() <= ceiling,
                    "T_K={temperature_kelvin:.1e} direction={direction} component={component}: refined={:.17e}, reference={:.17e}, ceiling={ceiling:.17e}",
                    refined_values[component],
                    reference[component]
                );
                assert!(
                    (refined_values[component] - coarse_values[component]).abs() <= ceiling,
                    "T_K={temperature_kelvin:.1e} direction={direction} component={component}: coarse={:.17e}, refined={:.17e}, ceiling={ceiling:.17e}",
                    coarse_values[component],
                    refined_values[component]
                );
            }
        }
    }

    #[test]
    fn f08d_private_table_all_interval_midpoints_track_refined_direct_values() {
        // Fixed before inspecting any midpoint result: every one of the 56
        // log-temperature intervals must agree with an order-128 direct
        // evaluation to 1% or 1e-9 in raw phase-space units, whichever is
        // larger.  The absolute floor covers genuine zero crossings without
        // turning a relative singularity into a false interpolation failure.
        let table = build_complete_thermal_table().expect("F08D private table build");
        let mut maximum_scaled_difference: f64 = 0.0;
        for interval in 0..table.temperature_kelvin.len() - 1 {
            let temperature_kelvin = (table.temperature_kelvin[interval]
                * table.temperature_kelvin[interval + 1])
                .sqrt();
            let photon_temperature_mev = temperature_kelvin / MEV_TO_KELVIN;
            let neutrino_temperature_mev =
                expected_profile_neutrino_temperature(photon_temperature_mev)
                    .expect("leading-QED profile midpoint");
            for (direction, tabulated) in [
                (1, &table.neutron_to_proton),
                (-1, &table.proton_to_neutron),
            ] {
                let interpolated = if direction == 1 {
                    interpolate_local_cubic(
                        &table.temperature_kelvin,
                        tabulated,
                        temperature_kelvin,
                    )
                } else {
                    interpolate_positive_log_log_cubic(
                        &table.temperature_kelvin,
                        tabulated,
                        temperature_kelvin,
                    )
                };
                let direct = direct_directional_components(
                    photon_temperature_mev,
                    neutrino_temperature_mev,
                    direction,
                    128,
                )
                .expect("refined direct midpoint")
                .total();
                let ceiling = (0.01 * direct.abs()).max(1.0e-9);
                let difference = (interpolated - direct).abs();
                maximum_scaled_difference = maximum_scaled_difference.max(difference / ceiling);
                assert!(
                    difference <= ceiling,
                    "interval={interval} T_K={temperature_kelvin:.17e} direction={direction}: interpolated={interpolated:.17e}, direct128={direct:.17e}, ceiling={ceiling:.17e}"
                );
            }
        }
        println!(
            "F08D private-table midpoint maximum normalized interpolation difference={maximum_scaled_difference:.17e}"
        );
    }

    #[test]
    fn profile_mismatch_and_low_temperature_clamp_fail_closed_without_a_table() {
        let cold_photon_temperature = 0.005;
        let cold_neutrino_temperature =
            expected_profile_neutrino_temperature(cold_photon_temperature)
                .expect("leading-QED profile");
        let cold =
            complete_thermal_radiative_raw(cold_photon_temperature, cold_neutrino_temperature)
                .expect("cold clamp");
        assert_eq!(cold.neutron_to_proton.to_bits(), 0.0_f64.to_bits());
        assert_eq!(cold.proton_to_neutron.to_bits(), 0.0_f64.to_bits());

        assert!(matches!(
            complete_thermal_radiative_raw(1.0, 0.8),
            Err(ThermalRadiativeError::ProfileMismatch { .. })
        ));
        assert!(matches!(
            complete_thermal_radiative_raw(10.1, 10.1),
            Err(ThermalRadiativeError::TemperatureAboveTable { .. })
        ));
    }
}
