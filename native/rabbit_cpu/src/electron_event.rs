#![allow(dead_code)]

use crate::electron_catalog::{ElectronChannel, ExplicitElectronProcess, RateMeV};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliBalance {
    pub(crate) gain: f64,
    pub(crate) loss: f64,
    pub(crate) net: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WeightedGainLossMeV {
    pub(crate) gain: RateMeV,
    pub(crate) loss: RateMeV,
    pub(crate) net: RateMeV,
}

fn checked_occupancies(occupancies: [f64; 4]) -> Result<[f64; 4], &'static str> {
    occupancies
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
        .then_some(occupancies)
        .ok_or("occupancies must be finite and within [0, 1]")
}

fn checked_direction(direction: [f64; 4]) -> Result<[f64; 4], &'static str> {
    direction
        .into_iter()
        .all(f64::is_finite)
        .then_some(direction)
        .ok_or("direction must be finite")
}

pub(crate) fn pauli_balance(occupancies: [f64; 4]) -> Result<PauliBalance, &'static str> {
    let [f1, f2, f3, f4] = checked_occupancies(occupancies)?;
    let gain = (1.0 - f1) * (1.0 - f2) * f3 * f4;
    let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
    let net = f3 * f4 - f1 * f2 + f1 * f2 * f3 + f1 * f2 * f4 - f1 * f3 * f4 - f2 * f3 * f4;
    (gain.is_finite() && loss.is_finite() && net.is_finite())
        .then_some(PauliBalance { gain, loss, net })
        .ok_or("Pauli balance is non-finite")
}

pub(crate) fn pauli_gradient(occupancies: [f64; 4]) -> Result<[f64; 4], &'static str> {
    let [f1, f2, f3, f4] = checked_occupancies(occupancies)?;
    let gradient = [
        -f2 + f2 * f3 + f2 * f4 - f3 * f4,
        -f1 + f1 * f3 + f1 * f4 - f3 * f4,
        f4 + f1 * f2 - f1 * f4 - f2 * f4,
        f3 + f1 * f2 - f1 * f3 - f2 * f3,
    ];
    gradient
        .into_iter()
        .all(f64::is_finite)
        .then_some(gradient)
        .ok_or("Pauli gradient is non-finite")
}

pub(crate) fn pauli_directional_derivative(
    occupancies: [f64; 4],
    direction: [f64; 4],
) -> Result<f64, &'static str> {
    let gradient = pauli_gradient(occupancies)?;
    let direction = checked_direction(direction)?;
    let result = gradient
        .into_iter()
        .zip(direction)
        .map(|(coefficient, delta)| coefficient * delta)
        .sum::<f64>();
    result
        .is_finite()
        .then_some(result)
        .ok_or("directional derivative is non-finite")
}

pub(crate) fn neutrino_leg_direction(
    process: ExplicitElectronProcess,
    target_delta: f64,
    coupled_delta: f64,
) -> Result<[f64; 4], &'static str> {
    checked_direction(match process.channel() {
        ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => {
            [target_delta, 0.0, coupled_delta, 0.0]
        }
        ElectronChannel::Pair => [target_delta, coupled_delta, 0.0, 0.0],
    })
}

pub(crate) fn weighted_event_gain_loss_mev(
    occupancies: [f64; 4],
    weight: RateMeV,
) -> Result<WeightedGainLossMeV, &'static str> {
    let balance = pauli_balance(occupancies)?;
    Ok(WeightedGainLossMeV {
        gain: RateMeV::new(weight.value() * balance.gain)?,
        loss: RateMeV::new(weight.value() * balance.loss)?,
        net: RateMeV::new(weight.value() * balance.net)?,
    })
}

pub(crate) fn weighted_neutrino_event_jvp_mev(
    process: ExplicitElectronProcess,
    occupancies: [f64; 4],
    target_delta: f64,
    coupled_delta: f64,
    weight: RateMeV,
) -> Result<RateMeV, &'static str> {
    let direction = neutrino_leg_direction(process, target_delta, coupled_delta)?;
    let derivative = pauli_directional_derivative(occupancies, direction)?;
    RateMeV::new(weight.value() * derivative)
}
