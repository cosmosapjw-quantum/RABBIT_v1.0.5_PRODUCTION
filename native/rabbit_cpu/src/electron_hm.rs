#![allow(dead_code)]
use crate::electron_catalog::{
    ElectronChannel, ElectronMassMeV, ExplicitElectronProcess, FourMomentumMeV, NeutrinoBank,
    RateMeV, TemperatureMeV,
};

pub(crate) const SIN2_THETA_W: f64 = 0.231_22;
pub(crate) const G_F_MEV_MINUS_2: f64 = 1.166_378_8e-11;

pub(crate) fn author_hm_w_over_gf2_mev4(
    process: ExplicitElectronProcess,
    momenta: [FourMomentumMeV; 4],
    electron_mass: ElectronMassMeV,
) -> f64 {
    let [p1, p2, p3, p4] = momenta;
    let (chi12, chi13, chi14) = (p1.chi(p2), p1.chi(p3), p1.chi(p4));
    let (chi23, chi24, chi34) = (p2.chi(p3), p2.chi(p4), p3.chi(p4));
    let g_r = 2.0 * SIN2_THETA_W;
    let g_l = match process.target().bank() {
        NeutrinoBank::Nux => -1.0 + g_r,
        NeutrinoBank::Nue | NeutrinoBank::Nuebar => 1.0 + g_r,
    };
    let (g_l, g_r) =
        if process.channel() != ElectronChannel::Pair && process.target().is_antineutrino() {
            (g_r, g_l)
        } else {
            (g_l, g_r)
        };
    let mass_squared = electron_mass.value() * electron_mass.value();
    32.0 * match (process.channel(), process.target().is_antineutrino()) {
        (ElectronChannel::ElectronMinusElastic, _) => {
            g_l * g_l * chi12 * chi34 + g_r * g_r * chi14 * chi23 - g_l * g_r * mass_squared * chi13
        }
        (ElectronChannel::ElectronPlusElastic, _) => {
            g_l * g_l * chi14 * chi23 + g_r * g_r * chi12 * chi34 - g_l * g_r * mass_squared * chi13
        }
        (ElectronChannel::Pair, false) => {
            g_l * g_l * chi14 * chi23 + g_r * g_r * chi13 * chi24 + g_l * g_r * mass_squared * chi12
        }
        (ElectronChannel::Pair, true) => {
            g_l * g_l * chi13 * chi24 + g_r * g_r * chi14 * chi23 + g_l * g_r * mass_squared * chi12
        }
    }
}

pub(crate) fn author_hm_w_dimensionless(
    process: ExplicitElectronProcess,
    momenta: [FourMomentumMeV; 4],
    electron_mass: ElectronMassMeV,
) -> f64 {
    G_F_MEV_MINUS_2.powi(2) * author_hm_w_over_gf2_mev4(process, momenta, electron_mass)
}

pub(crate) fn dimensionless_hm_event_factor_mev(w: f64, temperature: TemperatureMeV) -> RateMeV {
    RateMeV::new(temperature.value() * G_F_MEV_MINUS_2.powi(2) * w).expect("finite HM event")
}
