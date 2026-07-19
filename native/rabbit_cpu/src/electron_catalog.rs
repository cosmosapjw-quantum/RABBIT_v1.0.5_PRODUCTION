// R-02A stages crate-private primitives for later collision consumers.
#![allow(dead_code)]
macro_rules! checked_scalar {
    ($name:ident, $valid:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub(crate) struct $name(f64);
        impl $name {
            pub(crate) fn new(value: f64) -> Result<Self, &'static str> {
                (value.is_finite() && ($valid)(value))
                    .then_some(Self(value))
                    .ok_or("value is outside the finite domain")
            }
            pub(crate) const fn value(self) -> f64 {
                self.0
            }
        }
    };
}
checked_scalar!(TemperatureMeV, |v: f64| v > 0.0);
checked_scalar!(NeutrinoY, |v: f64| v >= 0.0);
checked_scalar!(ElectronX, |v: f64| v >= 0.0);
checked_scalar!(MomentumMeV, |v: f64| v >= 0.0);
checked_scalar!(ElectronMassMeV, |v: f64| v >= 0.0);
checked_scalar!(RateMeV, |_v: f64| true);
impl NeutrinoY {
    pub(crate) fn momentum(self, temperature: TemperatureMeV) -> MomentumMeV {
        MomentumMeV::new(self.0 * temperature.0).expect("finite y*T_cm")
    }
}
impl ElectronX {
    pub(crate) fn momentum(self, temperature: TemperatureMeV) -> MomentumMeV {
        MomentumMeV::new(self.0 * temperature.0).expect("finite x*T_gamma")
    }
}
pub(crate) fn fd_reference(y: NeutrinoY) -> f64 {
    1.0 / (y.0.exp() + 1.0)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FourMomentumMeV([f64; 4]);
impl FourMomentumMeV {
    pub(crate) fn new(e: f64, x: f64, y: f64, z: f64) -> Result<Self, &'static str> {
        let components = [e, x, y, z];
        let valid = components.into_iter().all(f64::is_finite);
        valid
            .then_some(Self(components))
            .ok_or("four-momentum must be finite")
    }
    pub(crate) fn chi(self, other: Self) -> f64 {
        let ([e, x, y, z], [oe, ox, oy, oz]) = (self.0, other.0);
        e * oe - x * ox - y * oy - z * oz
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NeutrinoBank {
    Nue,
    Nuebar,
    Nux,
}
macro_rules! explicit_neutrinos {
    ($( $state:ident => ($conjugate:ident, $bank:ident, $anti:literal) ),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(crate) enum ExplicitNeutrino { $( $state, )+ }
        impl ExplicitNeutrino {
            pub(crate) const fn conjugate(self) -> Self {
                match self { $( Self::$state => Self::$conjugate, )+ }
            }
            pub(crate) const fn bank(self) -> NeutrinoBank {
                match self { $( Self::$state => NeutrinoBank::$bank, )+ }
            }
            pub(crate) const fn is_antineutrino(self) -> bool {
                match self { $( Self::$state => $anti, )+ }
            }
        }
    };
}
explicit_neutrinos! {
    Nue => (Nuebar, Nue, false),
    Nuebar => (Nue, Nuebar, true),
    Numu => (Numubar, Nux, false),
    Numubar => (Numu, Nux, true),
    Nutau => (Nutaubar, Nux, false),
    Nutaubar => (Nutau, Nux, true),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ElectronChannel {
    ElectronMinusElastic,
    ElectronPlusElastic,
    Pair,
}
const fn explicit_input(target: ExplicitNeutrino, channel: ElectronChannel) -> ExplicitNeutrino {
    match channel {
        ElectronChannel::Pair => target.conjugate(),
        ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic => target,
    }
}
const fn folded_input(target: NeutrinoBank, channel: ElectronChannel) -> NeutrinoBank {
    match (target, channel) {
        (NeutrinoBank::Nue, ElectronChannel::Pair) => NeutrinoBank::Nuebar,
        (NeutrinoBank::Nuebar, ElectronChannel::Pair) => NeutrinoBank::Nue,
        (NeutrinoBank::Nux, ElectronChannel::Pair) => NeutrinoBank::Nux,
        (target, ElectronChannel::ElectronMinusElastic | ElectronChannel::ElectronPlusElastic) => {
            target
        }
    }
}
macro_rules! process_descriptor {
    ($name:ident($target:ty), $input:ident $(, $cp_target:path)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub(crate) struct $name($target, ElectronChannel);
        impl $name {
            pub(crate) const fn target(self) -> $target {
                self.0
            }
            pub(crate) const fn input(self) -> $target {
                $input(self.0, self.1)
            }
            pub(crate) const fn channel(self) -> ElectronChannel {
                self.1
            }
            $(
                pub(crate) const fn is_cp_average(self) -> bool {
                    matches!(self.0, $cp_target)
                }
            )?
        }
    };
}
process_descriptor!(ExplicitElectronProcess(ExplicitNeutrino), explicit_input);
process_descriptor!(
    FoldedElectronChannel(NeutrinoBank),
    folded_input,
    NeutrinoBank::Nux
);
use ElectronChannel::{ElectronMinusElastic as Em, ElectronPlusElastic as Ep, Pair};
use ExplicitNeutrino::{Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar};
use NeutrinoBank::{Nue as BNue, Nuebar as BNuebar, Nux};
macro_rules! state_major_catalog {
    ($descriptor:ident; $( $target:ident ),+ $(,)?) => {
        [$(
            $descriptor($target, Em),
            $descriptor($target, Ep),
            $descriptor($target, Pair),
        )+]
    };
}
pub(crate) const EXPLICIT_ELECTRON_PROCESSES: [ExplicitElectronProcess; 18] =
    state_major_catalog!(ExplicitElectronProcess; Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar);
pub(crate) const FOLDED_ELECTRON_CHANNELS: [FoldedElectronChannel; 9] =
    state_major_catalog!(FoldedElectronChannel; BNue, BNuebar, Nux);
pub(crate) const fn lift_three_to_six(values: [f64; 3]) -> [f64; 6] {
    let [nue, nuebar, nux] = values;
    [nue, nuebar, nux, nux, nux, nux]
}
pub(crate) const fn project_six_to_three(values: [f64; 6]) -> [f64; 3] {
    let [nue, nuebar, numu, numubar, nutau, nutaubar] = values;
    [nue, nuebar, (numu + numubar + nutau + nutaubar) / 4.0]
}
pub(crate) const fn conserved_readout(values: [f64; 3]) -> f64 {
    values[0] + values[1] + 4.0 * values[2]
}
