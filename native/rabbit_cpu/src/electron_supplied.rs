#![allow(dead_code)]

use crate::electron_catalog::{
    EXPLICIT_ELECTRON_PROCESSES, ExplicitElectronProcess, ExplicitNeutrino, RateMeV,
};
use crate::electron_event::{
    DynamicGainLossCoefficientsMeV, WeightedGainLossMeV,
    weighted_dynamic_gain_loss_coefficients_mev, weighted_event_gain_loss_mev,
};

pub(crate) struct SuppliedElectronEvent {
    process_slot: usize,
    target_node: usize,
    coupled_node: usize,
    fixed_fermions: [f64; 2],
    scalar_weight: RateMeV,
}

impl SuppliedElectronEvent {
    pub(crate) fn new(
        process_slot: usize,
        target_node: usize,
        coupled_node: usize,
        fixed_fermions: [f64; 2],
        scalar_weight: RateMeV,
    ) -> Result<Self, &'static str> {
        if !valid_occupations(&fixed_fermions) {
            return Err("fixed occupations must be finite and within [0, 1]");
        }
        let event = Self {
            process_slot,
            target_node,
            coupled_node,
            fixed_fermions,
            scalar_weight,
        };
        event.process()?;
        Ok(event)
    }

    pub(crate) fn process(&self) -> Result<ExplicitElectronProcess, &'static str> {
        EXPLICIT_ELECTRON_PROCESSES
            .get(self.process_slot)
            .copied()
            .ok_or("process slot is outside the accepted catalogue")
    }

    pub(crate) fn contraction(
        &self,
        nq: usize,
        f6: &[f64],
    ) -> Result<SuppliedContraction, &'static str> {
        checked_f6(nq, f6)?;
        self.materialize(nq, f6)
    }

    fn materialize(&self, nq: usize, f6: &[f64]) -> Result<SuppliedContraction, &'static str> {
        let process = self.process()?;
        let target = self::explicit_node(process.target(), self.target_node, nq)?;
        let coupled = self::explicit_node(process.input(), self.coupled_node, nq)?;
        let [fixed_a, fixed_b] = self.fixed_fermions;
        let (target_f, coupled_f) = (f6[target.flat_index], f6[coupled.flat_index]);
        let elastic = target.state == coupled.state;
        let occupancies = if elastic {
            [target_f, fixed_a, coupled_f, fixed_b]
        } else {
            [target_f, coupled_f, fixed_a, fixed_b]
        };
        let weighted_balance = weighted_event_gain_loss_mev(occupancies, self.scalar_weight)?;
        Ok(SuppliedContraction {
            process_slot: self.process_slot,
            process,
            dynamic_legs: DynamicNeutrinoLegs {
                target: DynamicNeutrinoLeg {
                    explicit_node: target,
                    pauli_leg_zero_based: 0,
                },
                coupled: DynamicNeutrinoLeg {
                    explicit_node: coupled,
                    pauli_leg_zero_based: if elastic { 2 } else { 1 },
                },
            },
            occupancies,
            scalar_weight: self.scalar_weight,
            weighted_balance,
        })
    }
}

pub(crate) struct SuppliedElectronEvents {
    nq: usize,
    events: Box<[SuppliedElectronEvent]>,
}

impl SuppliedElectronEvents {
    pub(crate) fn new(
        nq: usize,
        events: Box<[SuppliedElectronEvent]>,
    ) -> Result<Self, &'static str> {
        (nq > 0
            && nq.checked_mul(6).is_some()
            && nq.checked_mul(18).is_some()
            && events.iter().all(|event| {
                event.process().is_ok() && event.target_node < nq && event.coupled_node < nq
            }))
        .then_some(Self { nq, events })
        .ok_or("supplied event stream is outside the accepted domain")
    }

    pub(crate) fn nq(&self) -> usize {
        self.nq
    }

    pub(crate) fn events(&self) -> &[SuppliedElectronEvent] {
        &self.events
    }

    pub(crate) fn validated_contractions<'a>(
        &'a self,
        f6: &'a [f64],
    ) -> Result<
        impl ExactSizeIterator<Item = Result<SuppliedContraction, &'static str>> + 'a,
        &'static str,
    > {
        checked_f6(self.nq, f6)?;
        Ok(self
            .events
            .iter()
            .map(move |event| event.materialize(self.nq, f6)))
    }
}

pub(crate) struct ExplicitNode {
    pub(crate) state: ExplicitNeutrino,
    pub(crate) node: usize,
    pub(crate) flat_index: usize,
}

pub(crate) struct DynamicNeutrinoLeg {
    pub(crate) explicit_node: ExplicitNode,
    pub(crate) pauli_leg_zero_based: usize,
}

pub(crate) struct DynamicNeutrinoLegs {
    pub(crate) target: DynamicNeutrinoLeg,
    pub(crate) coupled: DynamicNeutrinoLeg,
}

pub(crate) struct SuppliedContraction {
    pub(crate) process_slot: usize,
    pub(crate) process: ExplicitElectronProcess,
    pub(crate) dynamic_legs: DynamicNeutrinoLegs,
    pub(crate) occupancies: [f64; 4],
    pub(crate) scalar_weight: RateMeV,
    pub(crate) weighted_balance: WeightedGainLossMeV,
}

impl SuppliedContraction {
    pub(crate) fn dynamic_coefficients(
        &self,
    ) -> Result<DynamicGainLossCoefficientsMeV, &'static str> {
        weighted_dynamic_gain_loss_coefficients_mev(
            self.process,
            self.occupancies,
            self.scalar_weight,
        )
    }
}

pub(crate) struct ExplicitSixAction {
    pub(crate) channel_contributions: Box<[WeightedGainLossMeV]>,
    pub(crate) total_action: Box<[RateMeV]>,
}

fn explicit_state_index(state: ExplicitNeutrino) -> Result<usize, &'static str> {
    EXPLICIT_ELECTRON_PROCESSES
        .chunks_exact(3)
        .position(|processes| processes[0].target() == state)
        .ok_or("explicit state is absent from catalogue")
}

fn explicit_node(
    state: ExplicitNeutrino,
    node: usize,
    nq: usize,
) -> Result<ExplicitNode, &'static str> {
    if node >= nq {
        return Err("node is outside the supplied grid");
    }
    let state_index = explicit_state_index(state)?;
    let flat_index = state_index
        .checked_mul(nq)
        .and_then(|offset| offset.checked_add(node))
        .ok_or("explicit node index overflow")?;
    Ok(ExplicitNode {
        state,
        node,
        flat_index,
    })
}

fn valid_occupations(values: &[f64]) -> bool {
    values
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
}

fn checked_f6(nq: usize, f6: &[f64]) -> Result<(), &'static str> {
    let expected = nq
        .checked_mul(6)
        .ok_or("explicit state dimension overflow")?;
    if nq == 0 || f6.len() != expected || !valid_occupations(f6) {
        return Err("explicit occupations have invalid dimension or value");
    }
    Ok(())
}

pub(crate) fn evaluate_explicit_action(
    stream: &SuppliedElectronEvents,
    f6: &[f64],
) -> Result<ExplicitSixAction, &'static str> {
    let nq = stream.nq();
    checked_f6(nq, f6)?;
    let zero = RateMeV::new(0.0)?;
    let empty = WeightedGainLossMeV {
        gain: zero,
        loss: zero,
        net: zero,
    };
    let mut channel_contributions = vec![empty; 18 * nq].into_boxed_slice();
    let mut total_action = vec![zero; 6 * nq].into_boxed_slice();
    for event in stream.events() {
        let contraction = event.materialize(nq, f6)?;
        let channel =
            contraction.process_slot * nq + contraction.dynamic_legs.target.explicit_node.node;
        let old = channel_contributions[channel];
        let add = contraction.weighted_balance;
        channel_contributions[channel] = WeightedGainLossMeV {
            gain: RateMeV::new(old.gain.value() + add.gain.value())?,
            loss: RateMeV::new(old.loss.value() + add.loss.value())?,
            net: RateMeV::new(old.net.value() + add.net.value())?,
        };
        let row = contraction.dynamic_legs.target.explicit_node.flat_index;
        total_action[row] = RateMeV::new(total_action[row].value() + add.net.value())?;
    }
    Ok(ExplicitSixAction {
        channel_contributions,
        total_action,
    })
}
