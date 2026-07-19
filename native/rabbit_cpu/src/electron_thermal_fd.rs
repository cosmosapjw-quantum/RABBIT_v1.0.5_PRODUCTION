//! Direct finite-electron-mass thermal collision moment.
//!
//! This module integrates the preserved Hannestad--Madsen physical point
//! density over the isotropic target momentum, the two radial momenta, and
//! the two remaining angular coordinates.  It is deliberately a slow
//! construction oracle for F09B.  The endpoint path instead evaluates the
//! same invariant two-body phase space in the centre-of-momentum frame.  That
//! independent coordinate system avoids the HM support solve and needs only a
//! bounded tensor rule; no fitted suppression factor or lookup table enters.
//!
//! The thermal distributions have zero chemical potential.  The public
//! claim ceiling remains an isotropic tree-level electron collision moment:
//! no neutrino-neutrino reactions, spectral transport, thermal collision
//! masses, QKE, or precision-decoupling authority is implied.

#![allow(dead_code)]

use core::f64::consts::PI;
use std::sync::OnceLock;

use crate::electron_catalog::{
    ElectronChannel, ElectronMassMeV, ElectronX, NeutrinoY, TemperatureMeV,
};
use crate::electron_phase_point::{
    PhysicalRadialCell, integrated_scalar_density_mev, physical_support_slice,
};
use crate::electron_thermal::ElectronThermalTransfer;
use crate::quadrature::{gauss_laguerre_plain_rule, gauss_legendre_rule};

const ELECTRON_PAIR_FIRST_SLOT: usize = 0;
const HEAVY_PAIR_FIRST_SLOT: usize = 6;
const EXPLICIT_ROWS_PER_PAIR: usize = 6;
const CM_RADIAL_ORDER: usize = 6;
const CM_ANGULAR_ORDER: usize = 4;
const CM_AZIMUTH_ORDER: usize = 8;
const CM_PHASE_PREFACTOR: f64 = 1.0 / (512.0 * PI * PI * PI * PI * PI * PI);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ThermalStatistics {
    FermiDirac,
    MaxwellBoltzmann,
}

#[derive(Clone, Copy, Debug)]
struct DirectThermalRule {
    radial_order: usize,
    angular_order: usize,
}

#[derive(Clone, Copy, Debug)]
struct CmThermalRule<'a> {
    radial: &'a [(f64, f64)],
    angular: &'a [(f64, f64)],
    azimuth_order: usize,
}

fn require_temperature(value: f64) -> Result<TemperatureMeV, &'static str> {
    TemperatureMeV::new(value).map_err(|_| "thermal collision temperature is invalid")
}

fn fermi_dirac(energy_over_temperature: f64) -> f64 {
    let exp_negative = (-energy_over_temperature).exp();
    exp_negative / (1.0 + exp_negative)
}

fn thermal_gain_loss(
    channel: ElectronChannel,
    energies_mev: [f64; 4],
    t_gamma_mev: f64,
    t_nu_mev: f64,
    statistics: ThermalStatistics,
) -> Result<(f64, f64, f64), &'static str> {
    let [e1, e2, e3, e4] = energies_mev;
    if !energies_mev
        .into_iter()
        .all(|energy| energy.is_finite() && energy >= 0.0)
    {
        return Err("thermal collision energy is invalid");
    }
    let (gain, loss, logarithmic_ratio) = match (channel, statistics) {
        (
            ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic,
            ThermalStatistics::FermiDirac,
        ) => {
            let [f1, f2, f3, f4] = [
                fermi_dirac(e1 / t_nu_mev),
                fermi_dirac(e2 / t_gamma_mev),
                fermi_dirac(e3 / t_nu_mev),
                fermi_dirac(e4 / t_gamma_mev),
            ];
            (
                (1.0 - f1) * (1.0 - f2) * f3 * f4,
                f1 * f2 * (1.0 - f3) * (1.0 - f4),
                (e1 - e3) * (t_nu_mev.recip() - t_gamma_mev.recip()),
            )
        }
        (ElectronChannel::Pair, ThermalStatistics::FermiDirac) => {
            let [f1, f2, f3, f4] = [
                fermi_dirac(e1 / t_nu_mev),
                fermi_dirac(e2 / t_nu_mev),
                fermi_dirac(e3 / t_gamma_mev),
                fermi_dirac(e4 / t_gamma_mev),
            ];
            (
                (1.0 - f1) * (1.0 - f2) * f3 * f4,
                f1 * f2 * (1.0 - f3) * (1.0 - f4),
                (e1 + e2) * (t_nu_mev.recip() - t_gamma_mev.recip()),
            )
        }
        (
            ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic,
            ThermalStatistics::MaxwellBoltzmann,
        ) => (
            (-e3 / t_nu_mev - e4 / t_gamma_mev).exp(),
            (-e1 / t_nu_mev - e2 / t_gamma_mev).exp(),
            (e1 - e3) * (t_nu_mev.recip() - t_gamma_mev.recip()),
        ),
        (ElectronChannel::Pair, ThermalStatistics::MaxwellBoltzmann) => (
            (-(e3 + e4) / t_gamma_mev).exp(),
            (-(e1 + e2) / t_nu_mev).exp(),
            (e1 + e2) * (t_nu_mev.recip() - t_gamma_mev.recip()),
        ),
    };
    let net = if logarithmic_ratio >= 0.0 {
        gain * -(-logarithmic_ratio).exp_m1()
    } else {
        loss * logarithmic_ratio.exp_m1()
    };
    [gain, loss, logarithmic_ratio, net]
        .into_iter()
        .all(f64::is_finite)
        .then_some((gain, loss, net))
        .ok_or("thermal collision Pauli balance is non-finite")
}

fn direct_pair_energy_transfer(
    first_process_slot: usize,
    t_gamma_mev: f64,
    t_nu_mev: f64,
    electron_mass_mev: f64,
    statistics: ThermalStatistics,
    rule: DirectThermalRule,
) -> Result<f64, &'static str> {
    if t_gamma_mev == t_nu_mev {
        return Ok(0.0);
    }
    let t_gamma = require_temperature(t_gamma_mev)?;
    let t_nu = require_temperature(t_nu_mev)?;
    let electron_mass = ElectronMassMeV::new(electron_mass_mev)
        .map_err(|_| "thermal collision electron mass is invalid")?;
    let radial_rule = gauss_laguerre_plain_rule(rule.radial_order)?;
    let angular_rule = gauss_legendre_rule(rule.angular_order)?;
    let mut total_mev5 = 0.0;

    for process_slot in first_process_slot..first_process_slot + EXPLICIT_ROWS_PER_PAIR {
        let channel = crate::electron_catalog::EXPLICIT_ELECTRON_PROCESSES[process_slot].channel();
        for &(target_y_value, target_weight) in &radial_rule {
            let target_y = NeutrinoY::new(target_y_value)?;
            let p1_mev = target_y_value * t_nu_mev;
            let target_energy_measure_mev4 =
                t_nu_mev.powi(3) * target_y_value.powi(2) * target_weight / (2.0 * PI * PI)
                    * p1_mev;
            for &(second_value, second_weight) in &radial_rule {
                for &(third_value, third_weight) in &radial_rule {
                    let radial = match channel {
                        ElectronChannel::ElectronMinusElastic
                        | ElectronChannel::ElectronPlusElastic => PhysicalRadialCell::elastic(
                            ElectronX::new(second_value)?,
                            second_weight,
                            NeutrinoY::new(third_value)?,
                            third_weight,
                        )?,
                        ElectronChannel::Pair => PhysicalRadialCell::pair(
                            NeutrinoY::new(second_value)?,
                            second_weight,
                            ElectronX::new(third_value)?,
                            third_weight,
                        )?,
                    };
                    for &(mu13, mu13_weight) in &angular_rule {
                        let Some(support) = physical_support_slice(
                            process_slot,
                            t_gamma,
                            t_nu,
                            electron_mass,
                            target_y,
                            radial,
                            mu13,
                        )?
                        else {
                            continue;
                        };
                        let (_, _, net) = thermal_gain_loss(
                            channel,
                            support.energies_mev,
                            t_gamma_mev,
                            t_nu_mev,
                            statistics,
                        )?;
                        if net == 0.0 {
                            continue;
                        }
                        let integrated_density = integrated_scalar_density_mev(&support)?;
                        total_mev5 += target_energy_measure_mev4
                            * mu13_weight
                            * integrated_density.value()
                            * net;
                    }
                }
            }
        }
    }
    total_mev5
        .is_finite()
        .then_some(total_mev5)
        .ok_or("direct thermal collision moment is non-finite")
}

fn direct_electron_energy_transfer(
    t_gamma_mev: f64,
    t_nue_mev: f64,
    t_nux_mev: f64,
    electron_mass_mev: f64,
    statistics: ThermalStatistics,
    rule: DirectThermalRule,
) -> Result<ElectronThermalTransfer, &'static str> {
    let transfer = ElectronThermalTransfer {
        nue_pair_mev5: direct_pair_energy_transfer(
            ELECTRON_PAIR_FIRST_SLOT,
            t_gamma_mev,
            t_nue_mev,
            electron_mass_mev,
            statistics,
            rule,
        )?,
        nux_pair_mev5: direct_pair_energy_transfer(
            HEAVY_PAIR_FIRST_SLOT,
            t_gamma_mev,
            t_nux_mev,
            electron_mass_mev,
            statistics,
            rule,
        )?,
    };
    [
        transfer.nue_pair_mev5,
        transfer.nux_pair_mev5,
        transfer.total_mev5(),
    ]
    .into_iter()
    .all(f64::is_finite)
    .then_some(transfer)
    .ok_or("direct electron thermal transfer is non-finite")
}

#[derive(Clone, Copy, Debug)]
struct CmFinalState {
    incoming_energies_mev: [f64; 2],
    outgoing_energies_mev: [f64; 2],
    s_mev2: f64,
}

/// Independent plasma-frame to centre-of-momentum construction.
fn cm_final_state(
    p1_mev: f64,
    p2_mev: f64,
    incoming_mu: f64,
    cos_star: f64,
    phi_star: f64,
    incoming_masses_mev: [f64; 2],
    outgoing_masses_mev: [f64; 2],
) -> Result<CmFinalState, &'static str> {
    let [m1, m2] = incoming_masses_mev;
    let [m3, m4] = outgoing_masses_mev;
    let e1 = p1_mev.hypot(m1);
    let e2 = p2_mev.hypot(m2);
    let p2x = p2_mev * (1.0 - incoming_mu * incoming_mu).sqrt();
    let p2z = p2_mev * incoming_mu;
    let total_px = p2x;
    let total_pz = p1_mev + p2z;
    let total_energy = e1 + e2;
    let s_mev2 = total_energy * total_energy - total_px * total_px - total_pz * total_pz;
    if !s_mev2.is_finite() || s_mev2 <= 0.0 {
        return Err("CM invariant mass is invalid");
    }
    let root_s = s_mev2.sqrt();
    let beta_x = total_px / total_energy;
    let beta_z = total_pz / total_energy;
    let beta2 = beta_x * beta_x + beta_z * beta_z;
    let gamma = total_energy / root_s;

    let (axis_x, axis_z) = if beta2 == 0.0 {
        (0.0, 1.0)
    } else {
        let beta_dot_p1 = beta_z * p1_mev;
        let coefficient = (gamma - 1.0) * beta_dot_p1 / beta2 - gamma * e1;
        let p1_star_x = coefficient * beta_x;
        let p1_star_z = p1_mev + coefficient * beta_z;
        let norm = p1_star_x.hypot(p1_star_z);
        if !norm.is_finite() || norm <= 0.0 {
            return Err("CM incoming axis is invalid");
        }
        (p1_star_x / norm, p1_star_z / norm)
    };

    let lambda = s_mev2 * s_mev2 + m3.powi(4) + m4.powi(4)
        - 2.0 * s_mev2 * m3 * m3
        - 2.0 * s_mev2 * m4 * m4
        - 2.0 * m3 * m3 * m4 * m4;
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("CM final-state Kallen function is invalid");
    }
    let momentum_star = lambda.sqrt() / (2.0 * root_s);
    let e3_star = (s_mev2 + m3 * m3 - m4 * m4) / (2.0 * root_s);
    let e4_star = (s_mev2 + m4 * m4 - m3 * m3) / (2.0 * root_s);
    let sin_star = (1.0 - cos_star * cos_star).sqrt();
    // ez is the incoming-p1 direction in CM; ex=(ez_z,0,-ez_x).
    let qx = momentum_star * (sin_star * phi_star.cos() * axis_z + cos_star * axis_x);
    let qz = momentum_star * (-sin_star * phi_star.cos() * axis_x + cos_star * axis_z);
    let beta_dot_q = beta_x * qx + beta_z * qz;
    let e3 = gamma * (e3_star + beta_dot_q);
    let e4 = gamma * (e4_star - beta_dot_q);
    let result = CmFinalState {
        incoming_energies_mev: [e1, e2],
        outgoing_energies_mev: [e3, e4],
        s_mev2,
    };
    [e1, e2, e3, e4, s_mev2]
        .into_iter()
        .all(|value| value.is_finite() && value > 0.0)
        .then_some(result)
        .ok_or("CM energies are invalid")
}

fn stable_loss_minus_gain(loss: f64, gain: f64, log_loss_over_gain: f64) -> f64 {
    if log_loss_over_gain >= 0.0 {
        loss * -(-log_loss_over_gain).exp_m1()
    } else {
        gain * log_loss_over_gain.exp_m1()
    }
}

fn flavour_couplings(electron_flavour: bool) -> (f64, f64) {
    let g_r = 2.0 * crate::electron_hm::SIN2_THETA_W;
    let g_l = if electron_flavour {
        1.0 + g_r
    } else {
        -1.0 + g_r
    };
    (g_l, g_r)
}

fn elastic_amplitude_over_gf2(s: f64, t: f64, u: f64, mass: f64, g_l: f64, g_r: f64) -> f64 {
    8.0 * (g_l * g_l * (s - mass * mass).powi(2)
        + g_r * g_r * (mass * mass - u).powi(2)
        + 2.0 * g_l * g_r * mass * mass * t)
}

fn pair_amplitude_over_gf2(s: f64, t: f64, u: f64, mass: f64, g_l: f64, g_r: f64) -> f64 {
    8.0 * (g_l * g_l * (mass * mass - u).powi(2)
        + g_r * g_r * (mass * mass - t).powi(2)
        + 2.0 * g_l * g_r * mass * mass * s)
}

fn production_radial_rule() -> &'static Vec<(f64, f64)> {
    static RULE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    RULE.get_or_init(|| {
        gauss_laguerre_plain_rule(CM_RADIAL_ORDER)
            .expect("fixed CM radial quadrature must construct")
    })
}

fn production_angular_rule() -> &'static Vec<(f64, f64)> {
    static RULE: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    RULE.get_or_init(|| {
        gauss_legendre_rule(CM_ANGULAR_ORDER).expect("fixed CM angular quadrature must construct")
    })
}

fn cm_pair_components_over_gf2(
    t_gamma_mev: f64,
    t_nu_mev: f64,
    electron_flavour: bool,
    electron_mass_mev: f64,
    statistics: ThermalStatistics,
) -> Result<[f64; 2], &'static str> {
    cm_pair_components_over_gf2_with_rule(
        t_gamma_mev,
        t_nu_mev,
        electron_flavour,
        electron_mass_mev,
        statistics,
        CmThermalRule {
            radial: production_radial_rule(),
            angular: production_angular_rule(),
            azimuth_order: CM_AZIMUTH_ORDER,
        },
    )
}

fn cm_pair_components_over_gf2_with_rule(
    t_gamma_mev: f64,
    t_nu_mev: f64,
    electron_flavour: bool,
    electron_mass_mev: f64,
    statistics: ThermalStatistics,
    rule: CmThermalRule<'_>,
) -> Result<[f64; 2], &'static str> {
    if t_gamma_mev == t_nu_mev {
        return Ok([0.0, 0.0]);
    }
    require_temperature(t_gamma_mev)?;
    require_temperature(t_nu_mev)?;
    ElectronMassMeV::new(electron_mass_mev).map_err(|_| "CM electron mass is invalid")?;
    if rule.radial.len() < 2 || rule.angular.len() < 2 || rule.azimuth_order < 2 {
        return Err("CM thermal collision quadrature rule is invalid");
    }
    let (g_l, g_r) = flavour_couplings(electron_flavour);
    let phi_weight = 2.0 * PI / rule.azimuth_order as f64;
    let mut elastic = 0.0;
    let mut pair = 0.0;

    for &(radial_1, radial_weight_1) in rule.radial {
        for &(radial_2, radial_weight_2) in rule.radial {
            for &(incoming_mu, incoming_weight) in rule.angular {
                // nu/antinu scattering on e-/e+.  The two crossed chiral
                // orderings each occur twice in the explicit six-row pair.
                let p1 = t_nu_mev * radial_1;
                let p2 = t_gamma_mev * radial_2;
                let elastic_radial_weight =
                    t_nu_mev * radial_weight_1 * t_gamma_mev * radial_weight_2;
                for &(cos_star, final_weight) in rule.angular {
                    for phi_index in 0..rule.azimuth_order {
                        let phi_star =
                            2.0 * PI * (phi_index as f64 + 0.5) / rule.azimuth_order as f64;
                        let final_state = cm_final_state(
                            p1,
                            p2,
                            incoming_mu,
                            cos_star,
                            phi_star,
                            [0.0, electron_mass_mev],
                            [0.0, electron_mass_mev],
                        )?;
                        let [e1, e2] = final_state.incoming_energies_mev;
                        let [e3, e4] = final_state.outgoing_energies_mev;
                        let s = final_state.s_mev2;
                        let root_s = s.sqrt();
                        let momentum_star = (s - electron_mass_mev.powi(2)) / (2.0 * root_s);
                        let t =
                            -(s - electron_mass_mev.powi(2)).powi(2) / (2.0 * s) * (1.0 - cos_star);
                        let u = 2.0 * electron_mass_mev.powi(2) - s - t;
                        let amplitude = 2.0
                            * elastic_amplitude_over_gf2(s, t, u, electron_mass_mev, g_l, g_r)
                            + 2.0
                                * elastic_amplitude_over_gf2(s, t, u, electron_mass_mev, g_r, g_l);
                        let statistical = match statistics {
                            ThermalStatistics::MaxwellBoltzmann => {
                                (-e1 / t_nu_mev - e2 / t_gamma_mev).exp()
                                    - (-e3 / t_nu_mev - e4 / t_gamma_mev).exp()
                            }
                            ThermalStatistics::FermiDirac => {
                                let [f1, f2, f3, f4] = [
                                    fermi_dirac(e1 / t_nu_mev),
                                    fermi_dirac(e2 / t_gamma_mev),
                                    fermi_dirac(e3 / t_nu_mev),
                                    fermi_dirac(e4 / t_gamma_mev),
                                ];
                                let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
                                let gain = f3 * f4 * (1.0 - f1) * (1.0 - f2);
                                stable_loss_minus_gain(
                                    loss,
                                    gain,
                                    (e3 - e1) * (t_nu_mev.recip() - t_gamma_mev.recip()),
                                )
                            }
                        };
                        let phase = p1.powi(2) * p2.powi(2) / (e1 * e2)
                            * (momentum_star / root_s)
                            * CM_PHASE_PREFACTOR;
                        elastic += elastic_radial_weight
                            * incoming_weight
                            * final_weight
                            * phi_weight
                            * phase
                            * amplitude
                            * 0.5
                            * (e3 - e1)
                            * statistical;
                    }
                }

                // Crossed e-+e+ -> nu+nubar coordinates avoid a pair
                // threshold cut; detailed balance supplies either direction.
                let pair_scale = t_gamma_mev.max(t_nu_mev);
                let p1 = pair_scale * radial_1;
                let p2 = pair_scale * radial_2;
                let pair_radial_weight = pair_scale.powi(2) * radial_weight_1 * radial_weight_2;
                for &(cos_star, final_weight) in rule.angular {
                    for phi_index in 0..rule.azimuth_order {
                        let phi_star =
                            2.0 * PI * (phi_index as f64 + 0.5) / rule.azimuth_order as f64;
                        let final_state = cm_final_state(
                            p1,
                            p2,
                            incoming_mu,
                            cos_star,
                            phi_star,
                            [electron_mass_mev, electron_mass_mev],
                            [0.0, 0.0],
                        )?;
                        let [e1, e2] = final_state.incoming_energies_mev;
                        let [e3, e4] = final_state.outgoing_energies_mev;
                        let s = final_state.s_mev2;
                        let root_s = s.sqrt();
                        let momentum_star = 0.5 * root_s;
                        let incoming_star = (s / 4.0 - electron_mass_mev.powi(2)).sqrt();
                        let t =
                            electron_mass_mev.powi(2) - s / 2.0 + root_s * incoming_star * cos_star;
                        let u = 2.0 * electron_mass_mev.powi(2) - s - t;
                        let amplitude =
                            pair_amplitude_over_gf2(s, t, u, electron_mass_mev, g_l, g_r);
                        let statistical = match statistics {
                            ThermalStatistics::MaxwellBoltzmann => {
                                (-(e1 + e2) / t_gamma_mev).exp() - (-(e3 + e4) / t_nu_mev).exp()
                            }
                            ThermalStatistics::FermiDirac => {
                                let [f1, f2, f3, f4] = [
                                    fermi_dirac(e1 / t_gamma_mev),
                                    fermi_dirac(e2 / t_gamma_mev),
                                    fermi_dirac(e3 / t_nu_mev),
                                    fermi_dirac(e4 / t_nu_mev),
                                ];
                                let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
                                let gain = f3 * f4 * (1.0 - f1) * (1.0 - f2);
                                stable_loss_minus_gain(
                                    loss,
                                    gain,
                                    (e3 + e4) * (t_nu_mev.recip() - t_gamma_mev.recip()),
                                )
                            }
                        };
                        let phase = p1.powi(2) * p2.powi(2) / (e1 * e2)
                            * (momentum_star / root_s)
                            * CM_PHASE_PREFACTOR;
                        pair += pair_radial_weight
                            * incoming_weight
                            * final_weight
                            * phi_weight
                            * phase
                            * amplitude
                            * (e3 + e4)
                            * statistical;
                    }
                }
            }
        }
    }
    [elastic, pair]
        .into_iter()
        .all(f64::is_finite)
        .then_some([elastic, pair])
        .ok_or("CM thermal collision components are non-finite")
}

/// Endpoint-consumed finite-mass, FD/Pauli electron thermal source.
pub(crate) fn finite_mass_fd_electron_energy_transfer(
    t_gamma_mev: f64,
    t_nue_mev: f64,
    t_nux_mev: f64,
) -> Result<ElectronThermalTransfer, &'static str> {
    let electron = cm_pair_components_over_gf2(
        t_gamma_mev,
        t_nue_mev,
        true,
        crate::flrw::ELECTRON_MASS_MEV,
        ThermalStatistics::FermiDirac,
    )?;
    let heavy = cm_pair_components_over_gf2(
        t_gamma_mev,
        t_nux_mev,
        false,
        crate::flrw::ELECTRON_MASS_MEV,
        ThermalStatistics::FermiDirac,
    )?;
    let gf2 = crate::electron_hm::G_F_MEV_MINUS_2.powi(2);
    let transfer = ElectronThermalTransfer {
        nue_pair_mev5: gf2 * (electron[0] + electron[1]),
        nux_pair_mev5: gf2 * (heavy[0] + heavy[1]),
    };
    [
        transfer.nue_pair_mev5,
        transfer.nux_pair_mev5,
        transfer.total_mev5(),
    ]
    .into_iter()
    .all(f64::is_finite)
    .then_some(transfer)
    .ok_or("finite-mass FD electron transfer is non-finite")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electron_catalog::EXPLICIT_ELECTRON_PROCESSES;
    use crate::electron_hm::G_F_MEV_MINUS_2;
    use crate::electron_thermal::massless_mb_electron_energy_transfer;
    use crate::flrw::ELECTRON_MASS_MEV;

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    }

    #[test]
    fn direct_fd_operator_has_an_exact_equal_temperature_null() {
        let transfer = direct_electron_energy_transfer(
            2.0,
            2.0,
            2.0,
            ELECTRON_MASS_MEV,
            ThermalStatistics::FermiDirac,
            DirectThermalRule {
                radial_order: 4,
                angular_order: 4,
            },
        )
        .unwrap();
        assert_eq!(transfer.nue_pair_mev5, 0.0);
        assert_eq!(transfer.nux_pair_mev5, 0.0);
        assert_eq!(transfer.total_mev5(), 0.0);
    }

    #[test]
    fn direct_fd_operator_is_signed_by_temperature_ordering() {
        let rule = DirectThermalRule {
            radial_order: 4,
            angular_order: 4,
        };
        let heating = direct_electron_energy_transfer(
            2.0,
            1.9,
            1.8,
            ELECTRON_MASS_MEV,
            ThermalStatistics::FermiDirac,
            rule,
        )
        .unwrap();
        let cooling = direct_electron_energy_transfer(
            1.8,
            1.9,
            2.0,
            ELECTRON_MASS_MEV,
            ThermalStatistics::FermiDirac,
            rule,
        )
        .unwrap();
        assert!(heating.nue_pair_mev5 > 0.0 && heating.nux_pair_mev5 > 0.0);
        assert!(cooling.nue_pair_mev5 < 0.0 && cooling.nux_pair_mev5 < 0.0);
    }

    #[test]
    fn direct_operator_rejects_invalid_mass_and_orders() {
        for invalid_mass in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                direct_electron_energy_transfer(
                    2.0,
                    1.9,
                    1.9,
                    invalid_mass,
                    ThermalStatistics::FermiDirac,
                    DirectThermalRule {
                        radial_order: 4,
                        angular_order: 4,
                    },
                )
                .is_err()
            );
        }
        assert!(
            direct_electron_energy_transfer(
                2.0,
                1.9,
                1.9,
                ELECTRON_MASS_MEV,
                ThermalStatistics::FermiDirac,
                DirectThermalRule {
                    radial_order: 1,
                    angular_order: 4,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn direct_massless_mb_operator_reaches_the_analytic_limit() {
        let expected = massless_mb_electron_energy_transfer(2.0, 1.9, 1.9).unwrap();
        for (order, tolerance) in [(24, 1.3e-2), (32, 4.0e-3)] {
            let actual = direct_electron_energy_transfer(
                2.0,
                1.9,
                1.9,
                0.0,
                ThermalStatistics::MaxwellBoltzmann,
                DirectThermalRule {
                    radial_order: order,
                    angular_order: order,
                },
            )
            .unwrap();
            eprintln!(
                "massless MB order={order}: nue={:.17e} ({:.9e}x), nux={:.17e} ({:.9e}x)",
                actual.nue_pair_mev5,
                actual.nue_pair_mev5 / expected.nue_pair_mev5,
                actual.nux_pair_mev5,
                actual.nux_pair_mev5 / expected.nux_pair_mev5,
            );
            assert!(relative_error(actual.nue_pair_mev5, expected.nue_pair_mev5) < tolerance);
            assert!(relative_error(actual.nux_pair_mev5, expected.nux_pair_mev5) < tolerance);
        }
    }

    #[test]
    fn direct_finite_mass_fd_operator_tracks_independent_cm_anchors() {
        let expected_e_over_gf2 = 1.047_175_324_837_161_4;
        let expected_x_over_gf2 = 0.220_065_930_685_499_34;
        for (order, tolerance) in [(24, 1.8e-2), (32, 3.0e-3)] {
            let actual = direct_electron_energy_transfer(
                1.2,
                1.0,
                1.0,
                ELECTRON_MASS_MEV,
                ThermalStatistics::FermiDirac,
                DirectThermalRule {
                    radial_order: order,
                    angular_order: order,
                },
            )
            .unwrap();
            let e_over_gf2 = actual.nue_pair_mev5 / G_F_MEV_MINUS_2.powi(2);
            let x_over_gf2 = actual.nux_pair_mev5 / G_F_MEV_MINUS_2.powi(2);
            eprintln!(
                "finite-mass FD order={order}: e/GF2={e_over_gf2:.17e} ({:.9e}x), x/GF2={x_over_gf2:.17e} ({:.9e}x)",
                e_over_gf2 / expected_e_over_gf2,
                x_over_gf2 / expected_x_over_gf2,
            );
            assert!(relative_error(e_over_gf2, expected_e_over_gf2) < tolerance);
            assert!(relative_error(x_over_gf2, expected_x_over_gf2) < tolerance);
        }
    }

    #[test]
    fn cm_production_rule_matches_independent_python_cm_components() {
        // Independent NumPy/SciPy Golub-Welsch plus plasma-frame CM boost;
        // no Rust HM support/root routine or catalogue helper was reused.
        let electron = cm_pair_components_over_gf2(
            1.2,
            1.0,
            true,
            ELECTRON_MASS_MEV,
            ThermalStatistics::FermiDirac,
        )
        .unwrap();
        let heavy = cm_pair_components_over_gf2(
            1.2,
            1.0,
            false,
            ELECTRON_MASS_MEV,
            ThermalStatistics::FermiDirac,
        )
        .unwrap();
        for (actual, expected) in electron
            .into_iter()
            .zip([0.143_302_741_614_643_32, 0.904_073_641_384_939_5])
            .chain(
                heavy
                    .into_iter()
                    .zip([0.030_934_584_632_570_22, 0.189_173_620_885_246_17]),
            )
        {
            assert!(relative_error(actual, expected) < 3.0e-13);
        }

        // A separate radial-24/angular-16/azimuth-24 Python calculation gives
        // the totals below.  This bounds the fixed endpoint rule's remaining
        // quadrature error without presenting the production rule as continuum truth.
        assert!(relative_error(electron.iter().sum(), 1.047_175_324_837_161_4) < 3.0e-4);
        assert!(relative_error(heavy.iter().sum(), 0.220_065_930_685_499_34) < 3.0e-4);
    }

    #[test]
    fn production_polar_four_tracks_polar_six_over_the_endpoint_envelope() {
        let radial = gauss_laguerre_plain_rule(CM_RADIAL_ORDER).unwrap();
        let polar_four = gauss_legendre_rule(4).unwrap();
        let polar_six = gauss_legendre_rule(6).unwrap();
        for (t_gamma, t_nu) in [
            (10.0, 9.99),
            (2.0, 1.9),
            (1.0, 0.9),
            (0.5, 0.4),
            (0.2, 0.15),
            (0.05, 0.04),
        ] {
            for electron_flavour in [true, false] {
                let order_four = cm_pair_components_over_gf2_with_rule(
                    t_gamma,
                    t_nu,
                    electron_flavour,
                    ELECTRON_MASS_MEV,
                    ThermalStatistics::FermiDirac,
                    CmThermalRule {
                        radial: &radial,
                        angular: &polar_four,
                        azimuth_order: CM_AZIMUTH_ORDER,
                    },
                )
                .unwrap()
                .iter()
                .sum::<f64>();
                let order_six = cm_pair_components_over_gf2_with_rule(
                    t_gamma,
                    t_nu,
                    electron_flavour,
                    ELECTRON_MASS_MEV,
                    ThermalStatistics::FermiDirac,
                    CmThermalRule {
                        radial: &radial,
                        angular: &polar_six,
                        azimuth_order: CM_AZIMUTH_ORDER,
                    },
                )
                .unwrap()
                .iter()
                .sum::<f64>();
                assert!(
                    relative_error(order_four, order_six) < 2.0e-5,
                    "Tgamma={t_gamma} Tnu={t_nu} electron={electron_flavour} order4={order_four:.17e} order6={order_six:.17e}"
                );
            }
        }
    }

    #[test]
    fn cm_event_construction_conserves_energy_for_elastic_and_pair_channels() {
        for (incoming_masses, outgoing_masses) in [
            ([0.0, ELECTRON_MASS_MEV], [0.0, ELECTRON_MASS_MEV]),
            ([ELECTRON_MASS_MEV; 2], [0.0; 2]),
        ] {
            for (p1, p2, incoming_mu, cos_star, phi_star) in [
                (0.13, 0.77, -0.81, 0.37, 0.19),
                (1.7, 0.42, 0.23, -0.91, 2.4),
                (4.1, 2.3, 0.94, 0.02, 5.8),
            ] {
                let state = cm_final_state(
                    p1,
                    p2,
                    incoming_mu,
                    cos_star,
                    phi_star,
                    incoming_masses,
                    outgoing_masses,
                )
                .unwrap();
                let incoming_energy = state.incoming_energies_mev.iter().sum::<f64>();
                let outgoing_energy = state.outgoing_energies_mev.iter().sum::<f64>();
                assert!(relative_error(incoming_energy, outgoing_energy) < 3.0e-15);
                assert!(state.s_mev2 > (outgoing_masses[0] + outgoing_masses[1]).powi(2));
            }
        }
    }

    #[test]
    fn electron_catalog_preserves_elastic_number_and_pair_lepton_number() {
        for process in EXPLICIT_ELECTRON_PROCESSES {
            match process.channel() {
                ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => {
                    assert_eq!(process.input(), process.target());
                }
                ElectronChannel::Pair => {
                    assert_eq!(process.input(), process.target().conjugate());
                    let target_lepton_number = if process.target().is_antineutrino() {
                        -1
                    } else {
                        1
                    };
                    let input_lepton_number = if process.input().is_antineutrino() {
                        -1
                    } else {
                        1
                    };
                    assert_eq!(target_lepton_number + input_lepton_number, 0);
                }
            }
        }
    }

    #[test]
    fn finite_electron_mass_source_has_cold_suppression_beyond_t9_scaling() {
        for electron_flavour in [true, false] {
            let warm = cm_pair_components_over_gf2(
                0.5,
                0.4,
                electron_flavour,
                ELECTRON_MASS_MEV,
                ThermalStatistics::FermiDirac,
            )
            .unwrap()
            .iter()
            .sum::<f64>()
                / 0.5_f64.powi(9);
            let cold = cm_pair_components_over_gf2(
                0.05,
                0.04,
                electron_flavour,
                ELECTRON_MASS_MEV,
                ThermalStatistics::FermiDirac,
            )
            .unwrap()
            .iter()
            .sum::<f64>()
                / 0.05_f64.powi(9);
            assert!(warm > 0.0 && cold > 0.0);
            assert!(cold / warm < 3.0e-4, "cold/T^9={cold}, warm/T^9={warm}");
        }
    }

    #[test]
    fn cm_massless_mb_limit_matches_the_analytic_elastic_and_pair_terms() {
        for electron_flavour in [true, false] {
            let actual = cm_pair_components_over_gf2(
                1.2,
                1.0,
                electron_flavour,
                0.0,
                ThermalStatistics::MaxwellBoltzmann,
            )
            .unwrap();
            let (g_l, g_r) = flavour_couplings(electron_flavour);
            let coupling = g_l * g_l + g_r * g_r;
            let expected = [
                56.0 * coupling / PI.powi(5) * 1.2_f64.powi(4) * (1.2 - 1.0),
                32.0 * coupling / PI.powi(5) * (1.2_f64.powi(9) - 1.0),
            ];
            assert!(relative_error(actual[0], expected[0]) < 3.0e-7);
            assert!(relative_error(actual[1], expected[1]) < 2.0e-7);
        }
    }

    #[test]
    fn endpoint_cm_source_has_exact_null_sign_entropy_and_multiplicity() {
        let null = finite_mass_fd_electron_energy_transfer(2.0, 2.0, 2.0).unwrap();
        assert_eq!(null.nue_pair_mev5, 0.0);
        assert_eq!(null.nux_pair_mev5, 0.0);
        assert_eq!(null.total_mev5(), 0.0);

        for (t_gamma, t_nue, t_nux) in [(2.0, 1.9, 1.8), (1.8, 1.9, 2.0)] {
            let transfer = finite_mass_fd_electron_energy_transfer(t_gamma, t_nue, t_nux).unwrap();
            let entropy = transfer.nue_pair_mev5 * (t_nue.recip() - t_gamma.recip())
                + 2.0 * transfer.nux_pair_mev5 * (t_nux.recip() - t_gamma.recip());
            assert!(entropy >= 0.0);
            assert_eq!(
                transfer.total_mev5(),
                transfer.nue_pair_mev5 + 2.0 * transfer.nux_pair_mev5
            );
        }
    }
}
