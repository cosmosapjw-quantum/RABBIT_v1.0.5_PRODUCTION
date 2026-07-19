mod frozen {
    use crate::electron_catalog::{
        EXPLICIT_ELECTRON_PROCESSES, ElectronChannel, ExplicitElectronProcess, RateMeV,
    };
    use crate::electron_event::{
        neutrino_leg_direction, pauli_balance, pauli_directional_derivative, pauli_gradient,
        weighted_event_gain_loss_mev, weighted_neutrino_event_jvp_mev,
    };
    const BOUNDARIES: [[f64; 4]; 6] = [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
    ];
    const SEEDS: [([f64; 4], [f64; 4]); 5] = [
        ([0.21, 0.34, 0.47, 0.58], [0.17, -0.23, 0.31, -0.19]),
        ([0.12, 0.73, 0.44, 0.65], [-0.11, 0.29, -0.37, 0.41]),
        ([0.82, 0.25, 0.63, 0.16], [0.21, 0.13, -0.27, -0.33]),
        ([0.39, 0.56, 0.22, 0.77], [-0.32, 0.18, 0.24, -0.14]),
        ([0.68, 0.14, 0.75, 0.33], [0.09, -0.31, 0.16, 0.27]),
    ];
    const EPSILONS: [f64; 4] = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6];
    fn relative(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1.0e-300)
    }
    fn raw(occupancies: [f64; 4]) -> (f64, f64) {
        let [f1, f2, f3, f4] = occupancies;
        (
            (1.0 - f1) * (1.0 - f2) * f3 * f4,
            f1 * f2 * (1.0 - f3) * (1.0 - f4),
        )
    }
    fn independent_six(occupancies: [f64; 4]) -> f64 {
        let [f1, f2, f3, f4] = occupancies;
        let positive_cubic = f1 * f2 * (f3 + f4);
        let negative_cubic = f3 * f4 * (f1 + f2);
        f3 * f4 - negative_cubic + positive_cubic - f1 * f2
    }
    fn symbolic_gradient([f1, f2, f3, f4]: [f64; 4]) -> [f64; 4] {
        [
            -f2 + f2 * f3 + f2 * f4 - f3 * f4,
            -f1 + f1 * f3 + f1 * f4 - f3 * f4,
            f4 + f1 * f2 - f1 * f4 - f2 * f4,
            f3 + f1 * f2 - f1 * f3 - f2 * f3,
        ]
    }

    fn dot(left: [f64; 4], right: [f64; 4]) -> f64 {
        left.into_iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn fd(energy: f64, temperature: f64) -> f64 {
        1.0 / ((energy / temperature).exp() + 1.0)
    }

    #[test]
    fn pauli_gain_loss_matches_six_monomial_and_boundaries() {
        for occupancies in BOUNDARIES.into_iter().chain(SEEDS.map(|seed| seed.0)) {
            let actual = pauli_balance(occupancies).unwrap();
            let (gain, loss) = raw(occupancies);
            assert!(relative(actual.gain, gain) <= 1.0e-15);
            assert!(relative(actual.loss, loss) <= 1.0e-15);
            let scale = gain.max(loss).max(1.0e-300);
            assert!((actual.net - (gain - loss)).abs() / scale <= 1.0e-15);
            assert!((actual.net - independent_six(occupancies)).abs() / scale <= 1.0e-15);
            assert!((0.0..=1.0).contains(&actual.gain));
            assert!((0.0..=1.0).contains(&actual.loss));
        }
        let pure_gain = pauli_balance([0.0, 0.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            (pure_gain.gain, pure_gain.loss, pure_gain.net),
            (1.0, 0.0, 1.0)
        );
        let pure_loss = pauli_balance([1.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            (pure_loss.gain, pure_loss.loss, pure_loss.net),
            (0.0, 1.0, -1.0)
        );
        for invalid in [
            [f64::NAN, 0.2, 0.3, 0.4],
            [0.2, f64::INFINITY, 0.3, 0.4],
            [-1.0e-16, 0.2, 0.3, 0.4],
            [0.2, 1.000_000_000_000_000_2, 0.3, 0.4],
        ] {
            assert!(pauli_balance(invalid).is_err());
            assert!(pauli_gradient(invalid).is_err());
        }
    }

    #[test]
    fn pauli_directional_derivative_matches_five_seed_ladder() {
        for (occupancies, direction) in SEEDS {
            let gradient = pauli_gradient(occupancies).unwrap();
            let expected_gradient = symbolic_gradient(occupancies);
            for (actual, expected) in gradient.into_iter().zip(expected_gradient) {
                assert!(relative(actual, expected) <= 1.0e-15);
            }
            let analytic = pauli_directional_derivative(occupancies, direction).unwrap();
            let expected = dot(expected_gradient, direction);
            assert!(relative(analytic, expected) <= 1.0e-15);
            assert!(analytic.abs() > 1.0e-8);
            for epsilon in EPSILONS {
                let plus = std::array::from_fn(|i| occupancies[i] + epsilon * direction[i]);
                let minus = std::array::from_fn(|i| occupancies[i] - epsilon * direction[i]);
                let centered = (pauli_balance(plus).unwrap().net
                    - pauli_balance(minus).unwrap().net)
                    / (2.0 * epsilon);
                assert!(relative(centered, analytic) <= 1.0e-5);
            }
        }
        assert!(pauli_directional_derivative(SEEDS[0].0, [f64::NAN, 0.0, 0.0, 0.0]).is_err());
        assert!(
            pauli_directional_derivative([1.0, 1.0, 0.0, 0.0], [f64::MAX, f64::MAX, 0.0, 0.0],)
                .is_err()
        );
        assert!(pauli_directional_derivative([1.1, 0.2, 0.3, 0.4], [0.0; 4]).is_err());
    }

    fn check_physical_balance(energies: [f64; 4]) {
        let occupancies = energies.map(|energy| fd(energy, 1.3));
        let balance = pauli_balance(occupancies).unwrap();
        assert!((balance.gain - balance.loss).abs() / balance.gain.max(balance.loss) <= 1.0e-10);
    }

    fn weighted_topology(
        process: ExplicitElectronProcess,
        occupancies: [f64; 4],
        target_delta: f64,
        coupled_delta: f64,
        weight: f64,
    ) -> f64 {
        weighted_neutrino_event_jvp_mev(
            process,
            occupancies,
            target_delta,
            coupled_delta,
            RateMeV::new(weight).unwrap(),
        )
        .unwrap()
        .value()
    }

    #[test]
    fn topology_aware_weighted_event_jvp_is_pointwise_db_consistent() {
        let seed = [0.21, 0.34, 0.47, 0.58];
        for process in EXPLICIT_ELECTRON_PROCESSES {
            let direction = neutrino_leg_direction(process, 0.17, -0.23).unwrap();
            let expected = match process.channel() {
                ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => {
                    [0.17, 0.0, -0.23, 0.0]
                }
                ElectronChannel::Pair => [0.17, -0.23, 0.0, 0.0],
            };
            assert_eq!(direction, expected);
            let analytic = dot(symbolic_gradient(seed), expected);
            assert!(
                relative(weighted_topology(process, seed, 0.17, -0.23, 1.0), analytic) <= 1.0e-15
            );
            if process.channel() == ElectronChannel::Pair {
                assert!((symbolic_gradient(seed)[1] * direction[1]).abs() > 1.0e-8);
            }
        }
        let electron_mass = 0.510_998_950_0;
        let incoming = 2.3;
        let outgoing = incoming / (1.0 + incoming * (1.0 - 0.27) / electron_mass);
        check_physical_balance([
            incoming,
            electron_mass,
            outgoing,
            electron_mass + incoming - outgoing,
        ]);
        check_physical_balance([1.7; 4]);

        let pair = EXPLICIT_ELECTRON_PROCESSES
            .into_iter()
            .find(|process| process.channel() == ElectronChannel::Pair)
            .unwrap();
        let raw_balance = pauli_balance(seed).unwrap();
        assert!(raw_balance.gain != 0.0 && raw_balance.loss != 0.0 && raw_balance.net != 0.0);
        let unit = weighted_event_gain_loss_mev(seed, RateMeV::new(1.0).unwrap()).unwrap();
        assert_eq!(unit.gain.value(), raw_balance.gain);
        assert_eq!(unit.loss.value(), raw_balance.loss);
        assert_eq!(unit.net.value(), raw_balance.net);
        let weighted = weighted_event_gain_loss_mev(seed, RateMeV::new(-2.0).unwrap()).unwrap();
        assert_eq!(weighted.gain.value(), -2.0 * raw_balance.gain);
        assert_eq!(weighted.loss.value(), -2.0 * raw_balance.loss);
        assert_eq!(weighted.net.value(), -2.0 * raw_balance.net);
        let raw_jvp = dot(symbolic_gradient(seed), [0.17, -0.23, 0.0, 0.0]);
        assert!(raw_jvp != 0.0);
        assert_eq!(
            weighted_topology(pair, seed, 0.17, -0.23, -2.0),
            -2.0 * raw_jvp
        );

        let elastic = EXPLICIT_ELECTRON_PROCESSES
            .into_iter()
            .find(|process| process.channel() == ElectronChannel::ElectronMinusElastic)
            .unwrap();
        let pure =
            weighted_event_gain_loss_mev([1.0, 1.0, 0.0, 0.0], RateMeV::new(1.0).unwrap()).unwrap();
        assert_eq!(pure.net.value(), -1.0);
        assert_eq!(
            weighted_topology(elastic, [1.0, 1.0, 0.0, 0.0], 1.0, 0.0, 1.0),
            -1.0
        );
        assert!(neutrino_leg_direction(pair, f64::INFINITY, 0.0).is_err());
        assert!(
            weighted_neutrino_event_jvp_mev(pair, seed, 0.0, f64::NAN, RateMeV::new(1.0).unwrap(),)
                .is_err()
        );
        assert!(
            weighted_event_gain_loss_mev([0.2, -0.1, 0.3, 0.4], RateMeV::new(1.0).unwrap(),)
                .is_err()
        );
        assert!(RateMeV::new(f64::NAN).is_err());
        assert!(RateMeV::new(f64::INFINITY).is_err());
    }
}
