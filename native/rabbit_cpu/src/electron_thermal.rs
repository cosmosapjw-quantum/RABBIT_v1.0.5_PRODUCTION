//! Bounded massless Maxwell-Boltzmann electron thermal source.
//!
//! This is the electron-only part of Escudero (2019), Eqs. (2.12)-(2.13),
//! evaluated per neutrino-antineutrino pair.  The heavy-flavour value is for
//! one pair; [`ElectronThermalTransfer::total_mev5`] includes the two
//! degenerate muon- and tau-flavour pairs.
//! The published analytic form is evaluated with this repository's existing
//! Hannestad--Madsen tree-level constants (`G_F` and `sin^2(theta_W)`), rather
//! than silently substituting the paper's rounded numerical convention.
//!
//! Claim ceiling: this analytic source is massless and Maxwell-Boltzmann.  It
//! includes neither finite electron mass nor Fermi-Dirac/Pauli corrections,
//! neutrino-neutrino transfer, spectral distortions, or a precision-BBN claim.

#![allow(dead_code)]

use crate::electron_hm::{G_F_MEV_MINUS_2, SIN2_THETA_W};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ElectronThermalTransfer {
    /// Energy transferred from the electromagnetic bath to one nue+nuebar pair.
    pub(crate) nue_pair_mev5: f64,
    /// Energy transferred from the electromagnetic bath to one nux+nuxbar pair.
    pub(crate) nux_pair_mev5: f64,
}

impl ElectronThermalTransfer {
    /// Total over one electron-flavour pair and two degenerate heavy pairs.
    pub(crate) fn total_mev5(self) -> f64 {
        self.nue_pair_mev5 + 2.0 * self.nux_pair_mev5
    }
}

fn massless_mb_kernel_mev9(t1_mev: f64, t2_mev: f64) -> f64 {
    // Factor T1^9-T2^9 before evaluation.  Grouping the quotient into
    // symmetric pairs also makes swapping T1/T2 change only the leading sign.
    let t1_2 = t1_mev * t1_mev;
    let t2_2 = t2_mev * t2_mev;
    let t1_4 = t1_2 * t1_2;
    let t2_4 = t2_2 * t2_2;
    let t1_6 = t1_4 * t1_2;
    let t2_6 = t2_4 * t2_2;
    let t1_8 = t1_4 * t1_4;
    let t2_8 = t2_4 * t2_4;
    let t1_t2 = t1_mev * t2_mev;
    let t1_t2_2 = t1_t2 * t1_t2;
    let t1_t2_3 = t1_t2_2 * t1_t2;
    let t1_t2_4 = t1_t2_2 * t1_t2_2;
    let ninth_power_quotient = t1_8
        + t2_8
        + t1_t2 * (t1_6 + t2_6)
        + t1_t2_2 * (t1_4 + t2_4)
        + t1_t2_3 * (t1_2 + t2_2)
        + t1_t2_4;
    (t1_mev - t2_mev) * (32.0 * ninth_power_quotient + 56.0 * t1_t2_4)
}

/// Electron-only energy transfer from the electromagnetic bath, in MeV^5.
///
/// Positive output heats the named neutrino pair.  Inputs are raw MeV
/// temperatures and deliberately fail instead of being floored or clamped.
pub(crate) fn massless_mb_electron_energy_transfer(
    t_gamma_mev: f64,
    t_nue_mev: f64,
    t_nux_mev: f64,
) -> Result<ElectronThermalTransfer, &'static str> {
    if ![t_gamma_mev, t_nue_mev, t_nux_mev]
        .into_iter()
        .all(|temperature| temperature.is_finite() && temperature > 0.0)
    {
        return Err("electron thermal temperatures must be positive and finite");
    }

    let common = G_F_MEV_MINUS_2.powi(2) / std::f64::consts::PI.powi(5);
    let sin2 = SIN2_THETA_W;
    let nue_coupling = 1.0 + 4.0 * sin2 + 8.0 * sin2 * sin2;
    let nux_coupling = 1.0 - 4.0 * sin2 + 8.0 * sin2 * sin2;
    let transfer = ElectronThermalTransfer {
        nue_pair_mev5: common * nue_coupling * massless_mb_kernel_mev9(t_gamma_mev, t_nue_mev),
        nux_pair_mev5: common * nux_coupling * massless_mb_kernel_mev9(t_gamma_mev, t_nux_mev),
    };
    [
        transfer.nue_pair_mev5,
        transfer.nux_pair_mev5,
        transfer.total_mev5(),
    ]
    .into_iter()
    .all(f64::is_finite)
    .then_some(transfer)
    .ok_or("electron thermal transfer is non-finite")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, relative_tolerance: f64) {
        let scale = actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE);
        assert!(
            (actual - expected).abs() <= relative_tolerance * scale,
            "actual={actual:.17e}, expected={expected:.17e}"
        );
    }

    #[test]
    fn equal_temperatures_are_an_exact_null() {
        let transfer = massless_mb_electron_energy_transfer(2.0, 2.0, 2.0).unwrap();
        assert_eq!(transfer.nue_pair_mev5, 0.0);
        assert_eq!(transfer.nux_pair_mev5, 0.0);
        assert_eq!(transfer.total_mev5(), 0.0);
    }

    #[test]
    fn matches_independent_decimal_anchors_at_two_and_one_point_nine_mev() {
        // Python Decimal(80 digits), using the unfactored published polynomial
        // and an 80-digit literal for pi; no Rust helper or rounded f64
        // intermediate was reused to generate these anchors.
        let transfer = massless_mb_electron_energy_transfer(2.0, 1.9, 1.9).unwrap();
        assert_close(transfer.nue_pair_mev5, 7.557_056_394_839_505e-21, 3.0e-15);
        assert_close(transfer.nux_pair_mev5, 1.615_183_352_763_455e-21, 3.0e-15);
        assert_close(
            massless_mb_kernel_mev9(2.0, 1.9),
            7_225.669_831_072,
            3.0e-15,
        );
    }

    #[test]
    fn sign_and_antisymmetry_follow_the_temperature_ordering() {
        let heating = massless_mb_electron_energy_transfer(2.0, 1.9, 1.8).unwrap();
        let cooling = massless_mb_electron_energy_transfer(1.9, 2.0, 2.0).unwrap();
        assert!(heating.nue_pair_mev5 > 0.0 && heating.nux_pair_mev5 > 0.0);
        assert!(cooling.nue_pair_mev5 < 0.0 && cooling.nux_pair_mev5 < 0.0);

        let forward = massless_mb_electron_energy_transfer(2.0, 1.9, 1.9).unwrap();
        let reverse = massless_mb_electron_energy_transfer(1.9, 2.0, 2.0).unwrap();
        assert_close(reverse.nue_pair_mev5, -forward.nue_pair_mev5, 1.0e-15);
        assert_close(reverse.nux_pair_mev5, -forward.nux_pair_mev5, 1.0e-15);
    }

    #[test]
    fn common_temperature_rescaling_is_ninth_order() {
        let scale: f64 = 1.7;
        let base = massless_mb_electron_energy_transfer(2.0, 1.9, 1.8).unwrap();
        let scaled =
            massless_mb_electron_energy_transfer(2.0 * scale, 1.9 * scale, 1.8 * scale).unwrap();
        let expected = scale.powi(9);
        assert_close(scaled.nue_pair_mev5 / base.nue_pair_mev5, expected, 3.0e-15);
        assert_close(scaled.nux_pair_mev5 / base.nux_pair_mev5, expected, 3.0e-15);
    }

    #[test]
    fn electron_to_heavy_pair_ratio_matches_the_independent_coupling_anchor() {
        let transfer = massless_mb_electron_energy_transfer(2.0, 1.9, 1.9).unwrap();
        assert_close(
            transfer.nue_pair_mev5 / transfer.nux_pair_mev5,
            4.678_760_700_393_525,
            2.0e-15,
        );
        assert_close(
            transfer.total_mev5(),
            transfer.nue_pair_mev5 + 2.0 * transfer.nux_pair_mev5,
            0.0,
        );
    }

    #[test]
    fn electron_transfer_has_nonnegative_thermal_entropy_production() {
        for (t_gamma, t_nue, t_nux) in [
            (2.0, 1.9, 1.8),
            (1.8, 1.9, 2.0),
            (2.0, 2.0, 1.7),
            (2.0, 2.3, 1.7),
            (1.0, 1.0, 1.0),
        ] {
            let transfer = massless_mb_electron_energy_transfer(t_gamma, t_nue, t_nux).unwrap();
            let entropy_production_mev4 = transfer.nue_pair_mev5
                * (t_nue.recip() - t_gamma.recip())
                + 2.0 * transfer.nux_pair_mev5 * (t_nux.recip() - t_gamma.recip());
            assert!(
                entropy_production_mev4 >= 0.0,
                "negative entropy production at ({t_gamma}, {t_nue}, {t_nux}): {entropy_production_mev4}"
            );
        }
    }

    #[test]
    fn invalid_raw_inputs_and_nonfinite_outputs_fail_without_clamping() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(massless_mb_electron_energy_transfer(invalid, 1.0, 1.0).is_err());
            assert!(massless_mb_electron_energy_transfer(1.0, invalid, 1.0).is_err());
            assert!(massless_mb_electron_energy_transfer(1.0, 1.0, invalid).is_err());
        }
        assert!(massless_mb_electron_energy_transfer(f64::MAX, 1.0, 1.0).is_err());
    }
}
