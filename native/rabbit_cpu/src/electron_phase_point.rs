#![allow(dead_code)]

use core::f64::consts::PI;

use crate::electron_catalog::{
    EXPLICIT_ELECTRON_PROCESSES, ElectronChannel, ElectronMassMeV, ElectronX,
    ExplicitElectronProcess, FourMomentumMeV, NeutrinoY, RateMeV, TemperatureMeV,
};
use crate::electron_hm::author_hm_w_dimensionless;

#[derive(Clone, Copy, Debug)]
enum RadialTopology {
    Elastic(ElectronX, f64, NeutrinoY, f64),
    Pair(NeutrinoY, f64, ElectronX, f64),
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhysicalRadialCell(RadialTopology);
impl PhysicalRadialCell {
    pub(crate) fn elastic(
        x2: ElectronX,
        w2: f64,
        y3: NeutrinoY,
        w3: f64,
    ) -> Result<Self, &'static str> {
        valid_radial([x2.value(), w2, y3.value(), w3])
            .then_some(Self(RadialTopology::Elastic(x2, w2, y3, w3)))
            .ok_or("elastic radial values are outside the open domain")
    }

    pub(crate) fn pair(
        y2: NeutrinoY,
        w2: f64,
        x3_minus: ElectronX,
        w3: f64,
    ) -> Result<Self, &'static str> {
        valid_radial([y2.value(), w2, x3_minus.value(), w3])
            .then_some(Self(RadialTopology::Pair(y2, w2, x3_minus, w3)))
            .ok_or("pair radial values are outside the open domain")
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhysicalSupportSlice {
    pub(crate) process: ExplicitElectronProcess,
    pub(crate) electron_mass: ElectronMassMeV,
    pub(crate) energies_mev: [f64; 4],
    pub(crate) momentum_magnitudes_mev: [f64; 4],
    pub(crate) radial_differentials_mev: [f64; 2],
    pub(crate) fixed_fermions: [f64; 2],
    pub(crate) support_polynomial_mev4: [f64; 3],
    pub(crate) support_center_radius: [f64; 2],
    theta_center_halfspan: [f64; 2],
    pub(crate) mu13: f64,
    support_k_mev2: f64,
    support_p_mev2: f64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhysicalPointDensity {
    pub(crate) raw_momenta_mev: [[f64; 4]; 4],
    pub(crate) four_momenta: [FourMomentumMeV; 4],
    pub(crate) theta: f64,
    pub(crate) mu12: f64,
    pub(crate) mu13: f64,
    pub(crate) beta: f64,
    pub(crate) support_d_mev4: f64,
    pub(crate) mu_1i: [f64; 4],
    pub(crate) hm_dimensionless: f64,
    pub(crate) scalar_density_mev: RateMeV,
    pub(crate) p2_densities_mev: [RateMeV; 2],
}
fn valid_radial(values: [f64; 4]) -> bool {
    values.into_iter().all(|v| v.is_finite() && v > 0.0)
}

fn open_unit(value: f64) -> bool {
    value.is_finite() && value > -1.0 && value < 1.0
}
fn fd_ratio(value: f64) -> f64 {
    1.0 / (value.exp() + 1.0)
}

fn physical_momentum(raw: [f64; 4]) -> Result<FourMomentumMeV, &'static str> {
    FourMomentumMeV::new(raw[0], raw[1], raw[2], raw[3])
}

pub(crate) fn checked_nonnegative_hm(value: f64) -> Result<f64, &'static str> {
    (value.is_finite() && value >= 0.0)
        .then_some(value)
        .ok_or("HM value is negative or non-finite")
}

pub(crate) fn physical_support_slice(
    process_slot: usize,
    t_gamma: TemperatureMeV,
    t_cm: TemperatureMeV,
    electron_mass: ElectronMassMeV,
    target_y: NeutrinoY,
    radial: PhysicalRadialCell,
    mu13: f64,
) -> Result<Option<PhysicalSupportSlice>, &'static str> {
    let process = EXPLICIT_ELECTRON_PROCESSES
        .get(process_slot)
        .copied()
        .ok_or("process slot is outside the catalogue")?;
    if target_y.value() <= 0.0 || !open_unit(mu13) {
        return Err("target node or mu13 is outside the open domain");
    }
    let (tg, tc, mass) = (t_gamma.value(), t_cm.value(), electron_mass.value());
    let p1 = target_y.value() * tc;
    let (p2, p3, e2, e3, radial_differentials_mev, fixed_energy) =
        match (process.channel(), radial.0) {
            (
                ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic,
                RadialTopology::Elastic(x2, w2, y3, w3),
            ) => {
                let (p2, p3) = (x2.value() * tg, y3.value() * tc);
                let e2 = p2.hypot(mass);
                (p2, p3, e2, p3, [tg * w2, tc * w3], e2)
            }
            (ElectronChannel::Pair, RadialTopology::Pair(y2, w2, x3_minus, w3)) => {
                let (p2, p3) = (y2.value() * tc, x3_minus.value() * tg);
                let e3 = p3.hypot(mass);
                (p2, p3, p2, e3, [tc * w2, tg * w3], e3)
            }
            _ => return Err("radial topology does not match the process"),
        };
    let e1 = p1;
    let e4 = e1 + e2 - e3;
    if ![p1, p2, p3, e2, e3, e4].into_iter().all(f64::is_finite)
        || !radial_differentials_mev.into_iter().all(f64::is_finite)
    {
        return Err("radial shell arithmetic is non-finite");
    }
    if e4 <= mass {
        return Ok(None);
    }
    let p4_squared = (e4 - mass) * (e4 + mass);
    if !p4_squared.is_finite() || p4_squared <= 0.0 {
        return Err("fourth-leg momentum is non-finite");
    }
    let p4 = p4_squared.sqrt();
    let k = p1 * p1 + p2 * p2 + p3 * p3 - 2.0 * p1 * p3 * mu13 - p4 * p4;
    let p = 2.0 * p2 * (p1 - p3 * mu13);
    let base = 4.0 * p2 * p2 * p3 * p3 * (1.0 - mu13 * mu13);
    if ![k, p, base].into_iter().all(f64::is_finite) || base <= 0.0 {
        return Err("support inputs are non-finite or degenerate");
    }
    let (a, b, c) = (-base - p * p, -2.0 * k * p, base - k * k);
    if ![a, b, c].into_iter().all(f64::is_finite) || a >= 0.0 {
        return Err("support polynomial is invalid");
    }
    let disc = b * b - 4.0 * a * c;
    if !disc.is_finite() {
        return Err("support discriminant is non-finite");
    }
    if disc <= 0.0 {
        return Ok(None);
    }
    let support_center_radius = [-b / (2.0 * a), disc.sqrt() / (-2.0 * a)];
    if !support_center_radius.into_iter().all(f64::is_finite) || support_center_radius[1] <= 0.0 {
        return Err("support roots are invalid");
    }
    let [center, radius] = support_center_radius;
    let lower_mu12 = (center - radius).max(-1.0);
    let upper_mu12 = (center + radius).min(1.0);
    if ![lower_mu12, upper_mu12].into_iter().all(f64::is_finite) {
        return Err("physical support intersection is non-finite");
    }
    if lower_mu12 >= upper_mu12 {
        return Ok(None);
    }
    // mu12=center+radius*cos(theta) decreases over theta in [0, pi].
    // The min/max operations above are the exact intersection of the HM
    // quadratic support with the physical angular domain, not a numerical
    // support cutoff.  The final clamp removes endpoint roundoff only.
    let upper_cos = ((upper_mu12 - center) / radius).clamp(-1.0, 1.0);
    let lower_cos = ((lower_mu12 - center) / radius).clamp(-1.0, 1.0);
    let theta_min = upper_cos.acos();
    let theta_max = lower_cos.acos();
    let theta_center_halfspan = [0.5 * (theta_min + theta_max), 0.5 * (theta_max - theta_min)];
    if !theta_center_halfspan.into_iter().all(f64::is_finite) || theta_center_halfspan[1] <= 0.0 {
        return Err("physical theta support is invalid");
    }
    Ok(Some(PhysicalSupportSlice {
        process,
        electron_mass,
        energies_mev: [e1, e2, e3, e4],
        momentum_magnitudes_mev: [p1, p2, p3, p4],
        radial_differentials_mev,
        fixed_fermions: [fd_ratio(fixed_energy / tg), fd_ratio(e4 / tg)],
        support_polynomial_mev4: [a, b, c],
        support_center_radius,
        theta_center_halfspan,
        mu13,
        support_k_mev2: k,
        support_p_mev2: p,
    }))
}

pub(crate) fn physical_point_density(
    support: &PhysicalSupportSlice,
    u_theta: f64,
) -> Result<PhysicalPointDensity, &'static str> {
    if !open_unit(u_theta) {
        return Err("u_theta is outside the open domain");
    }
    let [e1, e2, e3, e4] = support.energies_mev;
    let [p1, p2, p3, p4] = support.momentum_magnitudes_mev;
    let [a, _, _] = support.support_polynomial_mev4;
    let [center, radius] = support.support_center_radius;
    let [theta_center, theta_halfspan] = support.theta_center_halfspan;
    let theta = theta_center + theta_halfspan * u_theta;
    let mu12 = center + radius * theta.cos();
    let mu13 = support.mu13;
    // On the quadratic-root map D=-a*r^2*sin^2(theta).  Evaluate that
    // factorisation directly: the expanded polynomial loses its sign through
    // cancellation for high-momentum cells near a support endpoint.
    let support_d_mev4 = (-a) * radius * radius * theta.sin().powi(2);
    if !open_unit(mu12) || !support_d_mev4.is_finite() || support_d_mev4 <= 0.0 {
        return Err("transformed support point is invalid");
    }
    let (sin12, sin13) = ((1.0 - mu12 * mu12).sqrt(), (1.0 - mu13 * mu13).sqrt());
    let beta_numerator = support.support_k_mev2 + support.support_p_mev2 * mu12;
    let beta = support_d_mev4.sqrt().atan2(beta_numerator);
    if !beta.is_finite() || beta <= 0.0 || beta >= PI {
        return Err("beta angle is outside the open domain");
    }
    let (cos_beta_value, sin_beta) = (beta.cos(), beta.sin());
    let (p3x, p3y) = (p3 * sin13 * cos_beta_value, p3 * sin13 * sin_beta);
    let raw_momenta_mev = [
        [e1, 0.0, 0.0, p1],
        [e2, p2 * sin12, 0.0, p2 * mu12],
        [e3, p3x, p3y, p3 * mu13],
        [e4, p2 * sin12 - p3x, -p3y, p1 + p2 * mu12 - p3 * mu13],
    ];
    let four_momenta = [
        physical_momentum(raw_momenta_mev[0])?,
        physical_momentum(raw_momenta_mev[1])?,
        physical_momentum(raw_momenta_mev[2])?,
        physical_momentum(raw_momenta_mev[3])?,
    ];
    let hm_dimensionless = checked_nonnegative_hm(author_hm_w_dimensionless(
        support.process,
        four_momenta,
        support.electron_mass,
    ))?;
    let [dp2, dp3] = support.radial_differentials_mev;
    let measure_mev =
        2.0 / (2.0 * PI).powi(4) / (2.0 * e1) * p2.powi(2) * dp2 / (2.0 * e2) * p3.powi(2) * dp3
            / (2.0 * e3)
            * theta_halfspan
            / (-a).sqrt();
    let scalar_density_mev = RateMeV::new(measure_mev * hm_dimensionless)?;
    let coupled_mu = match support.process.channel() {
        ElectronChannel::Pair => mu12,
        ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => mu13,
    };
    let coupled_p2 = 0.5 * (3.0 * coupled_mu * coupled_mu - 1.0);
    let coupled_density = RateMeV::new(scalar_density_mev.value() * coupled_p2)?;
    let p2_densities_mev = [scalar_density_mev, coupled_density];
    let mu14 = raw_momenta_mev[3][3] / p4;
    if !mu14.is_finite() {
        return Err("fourth-leg direction is non-finite");
    }
    Ok(PhysicalPointDensity {
        raw_momenta_mev,
        four_momenta,
        theta,
        mu12,
        mu13,
        beta,
        support_d_mev4,
        mu_1i: [1.0, mu12, mu13, mu14],
        hm_dimensionless,
        scalar_density_mev,
        p2_densities_mev,
    })
}

/// Integrate the scalar HM point density over the remaining support angle.
///
/// At fixed radial shell and `mu13`, every HM invariant product is at most
/// quadratic in `mu12 = center + radius*cos(theta)`.  The delta-function
/// Jacobian removes the accompanying `sin(theta)`, so the remaining scalar
/// density is `A + B*cos(theta) + C*cos(theta)^2`.  Three interior samples and
/// the analytic first three cosine moments therefore perform this angular
/// integral exactly, including shells whose quadratic support intersects only
/// part of the physical `mu12` interval.
fn integrated_density_mev(
    support: &PhysicalSupportSlice,
    mut density: impl FnMut(&PhysicalPointDensity) -> Result<f64, &'static str>,
) -> Result<RateMeV, &'static str> {
    let [theta_center, theta_halfspan] = support.theta_center_halfspan;
    let theta_min = theta_center - theta_halfspan;
    let theta_max = theta_center + theta_halfspan;
    let i0 = theta_max - theta_min;
    let i1 = theta_max.sin() - theta_min.sin();
    let i2 = 0.5 * i0 + 0.25 * ((2.0 * theta_max).sin() - (2.0 * theta_min).sin());
    if ![i0, i1, i2].into_iter().all(f64::is_finite) || i0 <= 0.0 {
        return Err("integrated angular moments are invalid");
    }

    let sample_u = [-0.5, 0.0, 0.5];
    let mut cosines = [0.0; 3];
    let mut densities_per_theta = [0.0; 3];
    for (index, u_theta) in sample_u.into_iter().enumerate() {
        let theta = theta_center + theta_halfspan * u_theta;
        cosines[index] = theta.cos();
        let point = physical_point_density(support, u_theta)?;
        densities_per_theta[index] = density(&point)? / theta_halfspan;
    }

    let mut integrated = 0.0;
    for index in 0..3 {
        let other_a = (index + 1) % 3;
        let other_b = (index + 2) % 3;
        let denominator = (cosines[index] - cosines[other_a]) * (cosines[index] - cosines[other_b]);
        let weight = (i2 - (cosines[other_a] + cosines[other_b]) * i1
            + cosines[other_a] * cosines[other_b] * i0)
            / denominator;
        integrated += weight * densities_per_theta[index];
    }
    if !integrated.is_finite() || integrated < 0.0 {
        return Err("integrated scalar HM density is negative or non-finite");
    }
    RateMeV::new(integrated)
}

pub(crate) fn integrated_scalar_density_mev(
    support: &PhysicalSupportSlice,
) -> Result<RateMeV, &'static str> {
    integrated_density_mev(support, |point| Ok(point.scalar_density_mev.value()))
}
