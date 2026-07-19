mod frozen {
    use crate::electron_catalog::*;
    use ElectronChannel::{ElectronMinusElastic as Em, ElectronPlusElastic as Ep, Pair};
    use ExplicitNeutrino::{Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar};
    use NeutrinoBank::{Nue as BNue, Nuebar as BNuebar, Nux};

    const STATES: [ExplicitNeutrino; 6] = [Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar];
    const CHANNELS: [ElectronChannel; 3] = [Em, Ep, Pair];
    const BANKS: [NeutrinoBank; 3] = [BNue, BNuebar, Nux];

    #[test]
    fn checked_coordinates_units_and_metric_are_unambiguous() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(TemperatureMeV::new(bad).is_err());
            assert!(NeutrinoY::new(bad).is_err());
            assert!(ElectronX::new(bad).is_err());
            assert!(MomentumMeV::new(bad).is_err());
            assert!(ElectronMassMeV::new(bad).is_err());
            assert!(RateMeV::new(bad).is_err());
            for axis in 0..4 {
                let mut components = [0.0; 4];
                components[axis] = bad;
                let [e, x, y, z] = components;
                assert!(FourMomentumMeV::new(e, x, y, z).is_err());
            }
        }
        assert!(TemperatureMeV::new(0.0).is_err() && TemperatureMeV::new(-1.0).is_err());
        assert!(NeutrinoY::new(-1.0).is_err() && ElectronX::new(-1.0).is_err());
        assert!(MomentumMeV::new(-1.0).is_err() && ElectronMassMeV::new(-1.0).is_err());
        assert!(NeutrinoY::new(0.0).is_ok());
        assert!(ElectronX::new(0.0).is_ok());
        assert!(MomentumMeV::new(0.0).is_ok());
        assert!(ElectronMassMeV::new(0.0).is_ok());
        for signed in [-2.0, 0.0, 3.0] {
            assert_eq!(RateMeV::new(signed).unwrap().value(), signed);
        }
        let temperature = TemperatureMeV::new(3.0).unwrap();
        let (y, x) = (NeutrinoY::new(2.0).unwrap(), ElectronX::new(2.0).unwrap());
        let neutrino_p = y.momentum(temperature).value();
        let electron_p = x.momentum(temperature).value();
        assert_eq!(neutrino_p, 6.0);
        assert_eq!(electron_p, 6.0);
        assert_eq!(fd_reference(y), 1.0 / (2.0_f64.exp() + 1.0));
        let p = FourMomentumMeV::new(5.0, 1.0, 2.0, 3.0).unwrap();
        let q = FourMomentumMeV::new(7.0, 4.0, -1.0, 2.0).unwrap();
        assert_eq!(p.chi(q), 5.0 * 7.0 - (1.0 * 4.0 - 2.0 + 3.0 * 2.0));
    }

    #[test]
    fn catalogues_are_exact_unique_and_structural() {
        assert_eq!(EXPLICIT_ELECTRON_PROCESSES.len(), 18);
        let state_banks = [BNue, BNuebar, Nux, Nux, Nux, Nux];
        for (i, target) in STATES.into_iter().enumerate() {
            assert_eq!(target.conjugate(), STATES[i ^ 1]);
            assert_eq!(target.bank(), state_banks[i]);
            assert_eq!(target.is_antineutrino(), i % 2 == 1);
            for (j, channel) in CHANNELS.into_iter().enumerate() {
                let row = EXPLICIT_ELECTRON_PROCESSES[3 * i + j];
                assert_eq!((row.target(), row.channel()), (target, channel));
                let input = if channel == Pair {
                    target.conjugate()
                } else {
                    target
                };
                assert_eq!(row.input(), input);
            }
        }
        for (i, row) in EXPLICIT_ELECTRON_PROCESSES.iter().enumerate() {
            let key = (row.target(), row.channel());
            for other in &EXPLICIT_ELECTRON_PROCESSES[i + 1..] {
                assert_ne!(key, (other.target(), other.channel()));
            }
        }
        assert_eq!(FOLDED_ELECTRON_CHANNELS.len(), 9);
        let pair_inputs = [BNuebar, BNue, Nux];
        for (i, bank) in BANKS.into_iter().enumerate() {
            for (j, channel) in CHANNELS.into_iter().enumerate() {
                let row = FOLDED_ELECTRON_CHANNELS[3 * i + j];
                assert_eq!((row.target(), row.channel()), (bank, channel));
                let input = if channel == Pair {
                    pair_inputs[i]
                } else {
                    bank
                };
                assert_eq!(row.input(), input);
                assert_eq!(row.is_cp_average(), bank == Nux);
            }
        }
        for (i, row) in FOLDED_ELECTRON_CHANNELS.iter().enumerate() {
            let key = (row.target(), row.channel());
            for other in &FOLDED_ELECTRON_CHANNELS[i + 1..] {
                assert_ne!(key, (other.target(), other.channel()));
            }
        }
    }

    #[test]
    fn structural_fold_uses_four_only_in_readout() {
        let lifted = lift_three_to_six([1.0, 2.0, 3.0]);
        assert_eq!(lifted, [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
        let projected = project_six_to_three([1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
        assert_eq!(projected, [1.0, 2.0, 15.0]);
        assert_eq!(conserved_readout(projected), 1.0 + 2.0 + 4.0 * 15.0);
    }
}
