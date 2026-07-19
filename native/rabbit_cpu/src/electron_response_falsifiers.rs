mod frozen {
    use crate::electron_catalog::RateMeV;
    use crate::electron_event::pauli_gradient;
    use crate::electron_response::GeneralizedExplicitSixResponseMeV;
    use crate::electron_response::{ExplicitSixJ0, explicit_six_action_j0};
    use crate::electron_supplied::evaluate_explicit_action;
    use crate::electron_supplied::{SuppliedElectronEvent, SuppliedElectronEvents};

    const BAD_F6: &str = "explicit occupations have invalid dimension or value";
    const BAD_DIRECTION: &str = "explicit response direction has invalid dimension or value";
    const DIMENSION_OVERFLOW: &str = "explicit response dimension overflow";
    fn event(args: (usize, usize, usize, [f64; 2], f64)) -> SuppliedElectronEvent {
        let (slot, target, coupled, fixed, weight) = args;
        SuppliedElectronEvent::new(slot, target, coupled, fixed, RateMeV::new(weight).unwrap())
            .unwrap()
    }
    fn stream(nq: usize, entries: Vec<SuppliedElectronEvent>) -> SuppliedElectronEvents {
        SuppliedElectronEvents::new(nq, entries.into_boxed_slice()).unwrap()
    }
    fn fd_stream() -> SuppliedElectronEvents {
        stream(
            3,
            (0..18)
                .map(|slot| {
                    let target = (slot / 3 + slot % 3) % 3;
                    let coupled = (target + 1 + (slot / 3) % 2) % 3;
                    let fixed = [0.19 + 0.013 * slot as f64, 0.71 - 0.011 * slot as f64];
                    let sign = 1.0 - 2.0 * u8::from(matches!(slot, 2 | 7 | 15)) as f64;
                    let weight = sign * (0.5 + 0.03125 * slot as f64);
                    event((slot, target, coupled, fixed, weight))
                })
                .collect(),
        )
    }
    fn matrix_action(matrix: &[RateMeV], direction: &[f64]) -> Vec<f64> {
        matrix
            .chunks_exact(direction.len())
            .map(|row| {
                row.iter()
                    .zip(direction)
                    .map(|(entry, delta)| entry.value() * delta)
                    .sum()
            })
            .collect()
    }
    fn centered_action(
        supplied: &SuppliedElectronEvents,
        f6: &[f64],
        direction: &[f64],
        epsilon: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let shifted = |sign: f64| {
            f6.iter()
                .zip(direction)
                .map(|(f, delta)| f + sign * epsilon * delta)
                .collect::<Vec<_>>()
        };
        let (plus, minus) = (shifted(1.0), shifted(-1.0));
        let plus_action = evaluate_explicit_action(supplied, &plus).unwrap();
        let minus_action = evaluate_explicit_action(supplied, &minus).unwrap();
        let factor = 0.5 / epsilon;
        let channel = plus_action
            .channel_contributions
            .iter()
            .zip(minus_action.channel_contributions.iter())
            .map(|(plus, minus)| (plus.net.value() - minus.net.value()) * factor)
            .collect();
        let total = plus_action
            .total_action
            .iter()
            .zip(minus_action.total_action.iter())
            .map(|(plus, minus)| (plus.value() - minus.value()) * factor)
            .collect();
        (channel, total)
    }
    fn relative_max(analytic: &[f64], finite_difference: &[f64]) -> f64 {
        let (mut difference, mut scale) = (0.0_f64, 1.0e-300_f64);
        for (analytic, finite_difference) in analytic.iter().zip(finite_difference) {
            difference = difference.max((analytic - finite_difference).abs());
            scale = scale.max(analytic.abs()).max(finite_difference.abs());
        }
        difference / scale
    }
    fn assert_rates(actual: &[RateMeV], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.value().to_bits(), expected.to_bits());
        }
    }
    fn zero_j0() -> ExplicitSixJ0 {
        let zero = RateMeV::new(0.0).unwrap();
        ExplicitSixJ0 {
            nq: 1,
            channel_row_major: vec![zero; 108],
            total_row_major: vec![zero; 36],
        }
    }
    #[test]
    fn j0_matches_five_seeded_channel_and_total_action_ladders() {
        let supplied = fd_stream();
        for seed in 0..5 {
            let f6: Vec<f64> = (0..18)
                .map(|index| 0.16 + 0.025 * ((3 * index + 2 * seed) % 17) as f64)
                .collect();
            let direction: Vec<f64> = (0..18)
                .map(|index| 0.04 * (((5 * index + 7 * seed) % 13) as isize - 6) as f64 / 6.0)
                .collect();
            let j0 = explicit_six_action_j0(&supplied, &f6).unwrap();
            let channel = matrix_action(&j0.channel_row_major, &direction);
            let total = matrix_action(&j0.total_row_major, &direction);
            assert!(channel.iter().any(|value| *value != 0.0) && total.iter().any(|v| *v != 0.0));
            for epsilon in [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
                let (fd_channel, fd_total) = centered_action(&supplied, &f6, &direction, epsilon);
                assert!(
                    fd_channel
                        .iter()
                        .chain(&fd_total)
                        .all(|value| value.is_finite())
                );
                assert!(fd_channel.iter().any(|v| *v != 0.0) && fd_total.iter().any(|v| *v != 0.0));
                assert!(relative_max(&channel, &fd_channel) <= 2.0e-8);
                assert!(relative_max(&total, &fd_total) <= 2.0e-8);
            }
        }
    }
    #[test]
    fn j0_preserves_channel_reduction_sparsity_pair_coupling_and_signed_multiplicity() {
        let mut f6 = [0.25; 12];
        (f6[1], f6[3]) = (0.6, 0.55);
        let supplied = stream(
            2,
            vec![
                event((0, 0, 1, [0.2, 0.8], 2.0)),
                event((0, 0, 1, [0.2, 0.8], 2.0)),
                event((1, 0, 0, [0.3, 0.7], -1.5)),
                event((2, 0, 1, [0.4, 0.6], -0.75)),
            ],
        );
        let j0 = explicit_six_action_j0(&supplied, &f6).unwrap();
        let mut expected = vec![0.0; 36 * 12];
        let elastic = pauli_gradient([0.25, 0.2, 0.6, 0.8]).unwrap();
        for _ in 0..2 {
            expected[0] += 2.0 * elastic[0];
            expected[1] += 2.0 * elastic[2];
        }
        let same = pauli_gradient([0.25, 0.3, 0.25, 0.7]).unwrap();
        expected[2 * 12] += -1.5 * same[0];
        expected[2 * 12] += -1.5 * same[2];
        let pair = pauli_gradient([0.25, 0.55, 0.4, 0.6]).unwrap();
        expected[4 * 12] += -0.75 * pair[0];
        expected[4 * 12 + 3] += -0.75 * pair[1];
        assert_rates(&j0.channel_row_major, &expected);
        let mut total = vec![0.0_f64; 12 * 12];
        for column in 0..12 {
            total[column] =
                (expected[column] + expected[2 * 12 + column]) + expected[4 * 12 + column];
        }
        assert_rates(&j0.total_row_major, &total);
        let direction: Vec<f64> = (0..12).map(|index| (index as f64 - 5.0) / 7.0).collect();
        let applied = j0.apply(&direction).unwrap();
        let manual_channel = matrix_action(&j0.channel_row_major, &direction);
        let manual_total = matrix_action(&j0.total_row_major, &direction);
        assert_rates(&applied.channel_rows, &manual_channel);
        assert_rates(&applied.total_rows, &manual_total);
        let big = 2.0_f64.powi(54);
        let f6: Vec<_> = [0.25, 1.0].into_iter().cycle().take(12).collect();
        let supplied = stream(
            2,
            vec![
                event((0, 0, 1, [0.0, 1.0], -big)),
                event((2, 0, 1, [0.0, 0.0], -1.0)),
                event((1, 0, 1, [0.0, 1.0], big)),
            ],
        );
        let j0 = explicit_six_action_j0(&supplied, &f6).unwrap();
        let (em, ep, pair): (f64, f64, f64) = (
            j0.channel_row_major[0].value(),
            j0.channel_row_major[2 * 12].value(),
            j0.channel_row_major[4 * 12].value(),
        );
        assert_eq!([em, pair, ep], [big, 1.0, -big]);
        let (zero_bits, one_bits) = (0.0_f64.to_bits(), 1.0_f64.to_bits());
        assert_eq!(((em + pair) + ep).to_bits(), zero_bits);
        assert_eq!(((em + ep) + pair).to_bits(), one_bits);
        assert_eq!(j0.total_row_major[0].value().to_bits(), one_bits);
    }
    #[test]
    fn j0_rejects_invalid_states_directions_and_overflow() {
        let empty = stream(1, Vec::new());
        for bad in [
            -f64::EPSILON,
            1.0 + f64::EPSILON,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut f6 = [0.2; 6];
            f6[0] = bad;
            assert_eq!(explicit_six_action_j0(&empty, &f6).err(), Some(BAD_F6));
        }
        for f6 in [vec![0.2; 5], vec![0.2; 7]] {
            assert_eq!(explicit_six_action_j0(&empty, &f6).err(), Some(BAD_F6));
        }
        let j0 = explicit_six_action_j0(&empty, &[0.2; 6]).unwrap();
        for direction in [
            vec![0.0; 5],
            vec![0.0; 7],
            vec![f64::NAN; 6],
            vec![f64::INFINITY; 6],
            vec![f64::NEG_INFINITY; 6],
        ] {
            assert_eq!(j0.apply(&direction).err(), Some(BAD_DIRECTION));
        }
        let huge = stream(usize::MAX / 18, Vec::new());
        let overflow = explicit_six_action_j0(&huge, &[]).err();
        assert_eq!(overflow, Some(DIMENSION_OVERFLOW));
        let mut malformed = zero_j0();
        malformed.channel_row_major.pop();
        assert!(malformed.apply(&[0.0; 6]).is_err());
        let mut malformed = zero_j0();
        malformed.total_row_major.pop();
        assert!(malformed.apply(&[0.0; 6]).is_err());
        let mut product = zero_j0();
        product.channel_row_major[0] = RateMeV::new(f64::MAX).unwrap();
        assert!(product.apply(&[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).is_err());
        let mut sum = zero_j0();
        sum.total_row_major[0] = RateMeV::new(f64::MAX).unwrap();
        sum.total_row_major[1] = RateMeV::new(f64::MAX).unwrap();
        assert!(sum.apply(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]).is_err());
    }
    fn generalized(
        supplied: &SuppliedElectronEvents,
        f6: &[f64],
        weights: &[[RateMeV; 2]],
    ) -> Result<GeneralizedExplicitSixResponseMeV, &'static str> {
        GeneralizedExplicitSixResponseMeV::from_leg_weights(supplied, f6, weights)
    }
    fn assert_scaled(actual: &[RateMeV], expected: &[RateMeV], factor: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.value().to_bits(), (factor * e.value()).to_bits());
        }
    }
    fn generalized_fixture() -> SuppliedElectronEvents {
        stream(
            2,
            vec![
                event((0, 0, 1, [0.2, 0.8], 17.0)),
                event((1, 0, 0, [0.3, 0.7], -19.0)),
                event((2, 0, 1, [0.4, 0.6], 23.0)),
            ],
        )
    }
    #[test]
    fn generalized_equal_weights_reproduce_j0_and_common_factor_scales() {
        let f6 = [0.25; 12];
        let supplied = generalized_fixture();
        let j0 = explicit_six_action_j0(&supplied, &f6).unwrap();
        for factor in [1.0, 2.0] {
            let weights =
                [17.0, -19.0, 23.0].map(|value| [RateMeV::new(factor * value).unwrap(); 2]);
            let r: GeneralizedExplicitSixResponseMeV =
                generalized(&supplied, &f6, &weights).unwrap();
            assert_eq!(r.nq, j0.nq);
            assert_scaled(&r.channel_row_major, &j0.channel_row_major, factor);
            assert_scaled(&r.total_row_major, &j0.total_row_major, factor);
        }
    }
    #[test]
    fn generalized_unequal_target_and_coupled_weights_match_event_algebra() {
        let f6 = [0.25; 12];
        let supplied = generalized_fixture();
        let weights = [[2.0, -3.0], [5.0, -7.0], [11.0, -13.0]]
            .map(|pair| pair.map(|value| RateMeV::new(value).unwrap()));
        let response = generalized(&supplied, &f6, &weights).unwrap();
        let mut channel = vec![0.0; 36 * 12];
        let elastic = pauli_gradient([f6[0], 0.2, f6[1], 0.8]).unwrap();
        (channel[0], channel[1]) = (2.0 * elastic[0], -3.0 * elastic[2]);
        let same = pauli_gradient([f6[0], 0.3, f6[0], 0.7]).unwrap();
        channel[2 * 12] = (5.0 * same[0]) + (-7.0 * same[2]);
        let pair = pauli_gradient([f6[0], f6[3], 0.4, 0.6]).unwrap();
        (channel[4 * 12], channel[4 * 12 + 3]) = (11.0 * pair[0], -13.0 * pair[1]);
        let mut total = vec![0.0; 12 * 12];
        for column in 0..12 {
            total[column] = (channel[column] + channel[2 * 12 + column]) + channel[4 * 12 + column];
        }
        assert_eq!(response.nq, 2);
        assert_rates(&response.channel_row_major, &channel);
        assert_rates(&response.total_row_major, &total);
    }
    #[test]
    fn generalized_weights_reject_bad_length_nonfinite_and_overflow() {
        const BAD_LENGTH: &str = "explicit generalized response leg weights have invalid length";
        const FINITE: &str = "value is outside the finite domain";
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(RateMeV::new(value).err(), Some(FINITE));
        }
        let empty = stream(1, Vec::new());
        assert_eq!(generalized(&empty, &[0.2; 6], &[]).unwrap().nq, 1);
        let short_stream = stream(usize::MAX / 18, vec![event((0, 0, 0, [0.2, 0.8], 1.0))]);
        let short = generalized(&short_stream, &[], &[]);
        assert_eq!(short.err(), Some(BAD_LENGTH));
        let long_stream = stream(usize::MAX / 18, Vec::new());
        let long = generalized(&long_stream, &[], &[[RateMeV::new(1.0).unwrap(); 2]]);
        assert_eq!(long.err(), Some(BAD_LENGTH));
        let (large, negative, zero) = (
            RateMeV::new(0.75 * f64::MAX).unwrap(),
            RateMeV::new(-0.75 * f64::MAX).unwrap(),
            RateMeV::new(0.0).unwrap(),
        );
        let duplicate = stream(
            1,
            vec![
                event((0, 0, 0, [1.0, 0.0], 1.0)),
                event((0, 0, 0, [1.0, 0.0], 1.0)),
            ],
        );
        let channel_overflow = generalized(&duplicate, &[0.0; 6], &[[negative, zero]; 2]);
        assert_eq!(channel_overflow.err(), Some(FINITE));
        let canonical = stream(
            1,
            vec![
                event((2, 0, 0, [0.0, 0.0], 1.0)),
                event((0, 0, 0, [1.0, 0.0], 1.0)),
                event((1, 0, 0, [1.0, 0.0], 1.0)),
            ],
        );
        let canonical_overflow = generalized(
            &canonical,
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            &[[large, zero], [negative, zero], [negative, zero]],
        );
        assert_eq!(canonical_overflow.err(), Some(FINITE));
    }
}
