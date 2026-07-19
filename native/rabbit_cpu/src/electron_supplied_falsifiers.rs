mod frozen {
    use crate::electron_catalog::{
        EXPLICIT_ELECTRON_PROCESSES, ElectronChannel, ExplicitNeutrino, RateMeV,
    };
    use crate::electron_event::{WeightedGainLossMeV, weighted_event_gain_loss_mev};
    use crate::electron_supplied::{
        ExplicitSixAction, SuppliedContraction, SuppliedElectronEvent, SuppliedElectronEvents,
        evaluate_explicit_action,
    };
    fn event(
        slot: usize,
        target: usize,
        coupled: usize,
        fixed: [f64; 2],
        weight: f64,
    ) -> SuppliedElectronEvent {
        SuppliedElectronEvent::new(slot, target, coupled, fixed, RateMeV::new(weight).unwrap())
            .unwrap()
    }
    fn stream(nq: usize, events: Vec<SuppliedElectronEvent>) -> SuppliedElectronEvents {
        SuppliedElectronEvents::new(nq, events.into_boxed_slice()).unwrap()
    }
    fn state_index(state: ExplicitNeutrino) -> usize {
        EXPLICIT_ELECTRON_PROCESSES
            .chunks_exact(3)
            .position(|rows| rows[0].target() == state)
            .unwrap()
    }
    #[test]
    fn supplied_stream_rejects_invalid_dimensions_indices_and_values() {
        for nq in [1, 4] {
            let empty = stream(nq, Vec::new());
            let f6 = vec![0.25; 6 * nq];
            let action: ExplicitSixAction = evaluate_explicit_action(&empty, &f6).unwrap();
            assert_eq!(action.channel_contributions.len(), 18 * nq);
            assert_eq!(action.total_action.len(), 6 * nq);
            assert!(
                action
                    .channel_contributions
                    .iter()
                    .all(|entry: &WeightedGainLossMeV| {
                        entry.gain.value() == 0.0
                            && entry.loss.value() == 0.0
                            && entry.net.value() == 0.0
                    })
            );
            assert!(
                action
                    .total_action
                    .iter()
                    .all(|entry: &RateMeV| entry.value() == 0.0)
            );
            let mut invalid = f6;
            invalid[0] = f64::NAN;
            assert!(evaluate_explicit_action(&empty, &invalid).is_err());
        }
        assert!(SuppliedElectronEvents::new(0, Vec::new().into_boxed_slice()).is_err());
        assert!(SuppliedElectronEvents::new(usize::MAX, Vec::new().into_boxed_slice()).is_err());
        assert!(
            SuppliedElectronEvent::new(18, 0, 0, [0.2, 0.8], RateMeV::new(1.0).unwrap()).is_err()
        );
        for fixed in [[f64::NAN, 0.5], [-0.1, 0.5], [0.5, 1.1]] {
            assert!(
                SuppliedElectronEvent::new(0, 0, 0, fixed, RateMeV::new(1.0).unwrap()).is_err()
            );
        }
        assert!(RateMeV::new(f64::NAN).is_err());
        for (target, coupled) in [(1, 0), (0, 1)] {
            let events = vec![event(0, target, coupled, [0.2, 0.8], 1.0)];
            assert!(SuppliedElectronEvents::new(1, events.into_boxed_slice()).is_err());
        }
        let valid = stream(1, vec![event(0, 0, 0, [0.2, 0.8], 1.0)]);
        assert!(evaluate_explicit_action(&valid, &[0.2; 5]).is_err());
        assert!(evaluate_explicit_action(&valid, &[0.2; 7]).is_err());
        for bad in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut f6 = [0.2; 6];
            f6[0] = bad;
            assert!(evaluate_explicit_action(&valid, &f6).is_err());
        }
    }
    #[test]
    fn explicit_action_preserves_signed_gain_loss_rows_and_multiplicity() {
        let f6 = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (target, fixed, weight, expected) in [
            (0, [0.0, 1.0], 1.0, 1.0),
            (1, [1.0, 0.0], 1.0, -1.0),
            (1, [1.0, 0.0], -2.0, 2.0),
        ] {
            let action: ExplicitSixAction = evaluate_explicit_action(
                &stream(2, vec![event(0, target, 1 - target, fixed, weight)]),
                &f6,
            )
            .unwrap();
            for (position, value) in action.total_action.iter().enumerate() {
                let value: &RateMeV = value;
                assert_eq!(
                    value.value(),
                    if position == target { expected } else { 0.0 }
                );
            }
            let channel = &action.channel_contributions[target];
            assert_eq!(channel.net.value(), expected);
            assert_eq!(channel.gain.value() - channel.loss.value(), expected);
        }
        let events = vec![
            event(0, 0, 1, [0.0, 1.0], 0.5),
            event(0, 0, 1, [0.0, 1.0], 0.5),
            event(2, 0, 1, [1.0, 1.0], -1.25),
        ];
        let supplied = stream(2, events);
        assert_eq!(supplied.events().len(), 3);
        let action: ExplicitSixAction = evaluate_explicit_action(&supplied, &f6).unwrap();
        for (index, actual) in action.channel_contributions.iter().enumerate() {
            let actual: &WeightedGainLossMeV = actual;
            let expected = match index {
                0 => [1.0, 0.0, 1.0],
                4 => [-1.25, 0.0, -1.25],
                _ => [0.0; 3],
            };
            assert_eq!(
                [actual.gain.value(), actual.loss.value(), actual.net.value()],
                expected
            );
        }
        for (index, actual) in action.total_action.iter().enumerate() {
            let actual: &RateMeV = actual;
            assert_eq!(actual.value(), if index == 0 { -0.25 } else { 0.0 });
        }
        let huge = stream(
            2,
            vec![
                event(0, 0, 1, [0.0, 1.0], f64::MAX),
                event(0, 0, 1, [0.0, 1.0], f64::MAX),
            ],
        );
        assert!(evaluate_explicit_action(&huge, &f6).is_err());
    }
    #[test]
    fn all_process_slots_materialize_exact_dynamic_leg_contract() {
        let nq = 3;
        let f6: Vec<f64> = (0..18)
            .map(|flat| 0.11 + 0.09 * (flat / nq) as f64 + 0.04 * (flat % nq) as f64)
            .collect();
        for (slot, process) in EXPLICIT_ELECTRON_PROCESSES.iter().copied().enumerate() {
            let target_state_index = slot / 3;
            let channel_index = slot % 3;
            let target_node = (target_state_index + channel_index) % nq;
            let coupled_node = (target_node + 1 + target_state_index % 2) % nq;
            let fixed = [0.19 + 0.013 * slot as f64, 0.71 - 0.011 * slot as f64];
            let magnitude = 0.5 + 0.03125 * slot as f64;
            let weight = magnitude
                * if [2, 7, 15].contains(&slot) {
                    -1.0
                } else {
                    1.0
                };
            let supplied_event = event(slot, target_node, coupled_node, fixed, weight);
            assert_eq!(supplied_event.process().unwrap(), process);
            let contraction: SuppliedContraction = supplied_event.contraction(nq, &f6).unwrap();
            let input_state_index = state_index(process.input());
            let (occupancies, coupled_leg) = match process.channel() {
                ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => (
                    [
                        f6[target_state_index * nq + target_node],
                        fixed[0],
                        f6[target_state_index * nq + coupled_node],
                        fixed[1],
                    ],
                    2,
                ),
                ElectronChannel::Pair => (
                    [
                        f6[target_state_index * nq + target_node],
                        f6[input_state_index * nq + coupled_node],
                        fixed[0],
                        fixed[1],
                    ],
                    1,
                ),
            };
            assert_eq!(contraction.process_slot, slot);
            assert_eq!(contraction.process, process);
            assert_eq!(contraction.occupancies, occupancies);
            assert_eq!(contraction.scalar_weight.value(), weight);
            assert_eq!(
                contraction.weighted_balance,
                weighted_event_gain_loss_mev(occupancies, RateMeV::new(weight).unwrap()).unwrap()
            );
            let target = contraction.dynamic_legs.target;
            assert_eq!(target.explicit_node.state, process.target());
            assert_eq!(target.explicit_node.node, target_node);
            assert_eq!(
                target.explicit_node.flat_index,
                target_state_index * nq + target_node
            );
            assert_eq!(target.pauli_leg_zero_based, 0);
            let coupled = contraction.dynamic_legs.coupled;
            assert_eq!(coupled.explicit_node.state, process.input());
            assert_eq!(coupled.explicit_node.node, coupled_node);
            assert_eq!(
                coupled.explicit_node.flat_index,
                input_state_index * nq + coupled_node
            );
            assert_eq!(coupled.pauli_leg_zero_based, coupled_leg);
        }
    }
}
