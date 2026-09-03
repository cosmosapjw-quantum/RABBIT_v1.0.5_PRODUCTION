//! Isotropic classical-neutrino transport on a comoving grid.
//!
//! The sole momentum coordinate is `q = a p`, with `a=1` at the initial
//! state.  For massless collisionless particles the Liouville equation is
//! exactly `df(q)/dN = 0`, `N=ln(a)`.  The discretized distribution contributes
//! to the Friedmann density and therefore to an endpoint-consumed cosmic-time
//! state; it is not a detached diagnostic.  F10B adds the finite-mass
//! electron/positron action in bounded logit occupation coordinates.  F10C
//! adds the first classical diagonal neutrino self-collision slice: identical
//! same-flavour neutrino--neutrino and antineutrino--antineutrino scattering.
//! Cross-flavour and neutrino--antineutrino channels, flavour coherence,
//! anisotropy, and QKE remain absent.

#![cfg_attr(not(test), allow(dead_code))]

use core::f64::consts::PI;

use crate::electron_spectral::{
    ElectronSpectralInput, ElectronSpectralRule, IsotropicElectronSpectralAction,
    evaluate_isotropic_electron_spectral_action,
    evaluate_isotropic_electron_spectral_action_values,
};
use crate::flrw::{
    ELECTRON_MASS_MEV, MEV_TO_INVERSE_SECONDS, NEWTON_G_MEV_MINUS_2, electromagnetic_eos,
};
use crate::neutrino_self_spectral::{
    IsotropicNeutrinoSelfAction, NeutrinoSelfSpectralInput, NeutrinoSelfSpectralRule,
    evaluate_isotropic_neutrino_self_action, evaluate_isotropic_neutrino_self_action_values,
};
use crate::ode::OdeSystem;
use crate::quadrature::{gauss_laguerre_plain_rule, gauss_legendre_exponential_plain_rule};

const T_GAMMA_INDEX: usize = 0;
const ELAPSED_SECONDS_INDEX: usize = 1;
const OCCUPATION_START: usize = 2;
const NEUTRINO_FLAVOUR_PAIRS: f64 = 3.0;

#[derive(Clone, Debug)]
pub(crate) struct ComovingMomentumGrid {
    nodes_mev: Vec<f64>,
    weights_mev: Vec<f64>,
}

impl ComovingMomentumGrid {
    pub(crate) fn gauss_laguerre(order: usize, scale_mev: f64) -> Result<Self, &'static str> {
        if !scale_mev.is_finite() || scale_mev <= 0.0 {
            return Err("comoving-momentum scale must be positive and finite");
        }
        let rule = gauss_laguerre_plain_rule(order)?;
        let (nodes_mev, weights_mev): (Vec<_>, Vec<_>) = rule
            .into_iter()
            .map(|(node, weight)| (scale_mev * node, scale_mev * weight))
            .unzip();
        Ok(Self {
            nodes_mev,
            weights_mev,
        })
    }

    fn selected_decoupling(scale_mev: f64) -> Result<Self, &'static str> {
        if !scale_mev.is_finite() || scale_mev <= 0.0 {
            return Err("selected comoving-momentum scale is invalid");
        }
        // Fixed positive half-line rule selected by the F10C1 direct ladder
        // and independent multi-profile auxiliary-basis envelope.  The map is
        // y=-3 ln(1-t), t in [0,1), with no user-imposed integration cutoff;
        // its finite extreme nodes still define the interpolation support.
        let (nodes_mev, weights_mev): (Vec<_>, Vec<_>) =
            gauss_legendre_exponential_plain_rule(48, 3.0)?
                .into_iter()
                .map(|(node, plain_weight)| (scale_mev * node, scale_mev * plain_weight))
                .unzip();
        Ok(Self {
            nodes_mev,
            weights_mev,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes_mev.len()
    }

    pub(crate) fn zero_chemical_potential_fd(
        &self,
        reference_temperature_mev: f64,
    ) -> Result<Vec<f64>, &'static str> {
        if !reference_temperature_mev.is_finite() || reference_temperature_mev <= 0.0 {
            return Err("reference neutrino temperature must be positive and finite");
        }
        Ok(self
            .nodes_mev
            .iter()
            .map(|momentum| {
                let exp_negative = (-momentum / reference_temperature_mev).exp();
                exp_negative / (1.0 + exp_negative)
            })
            .collect())
    }

    pub(crate) fn pair_moments(
        &self,
        ln_a: f64,
        occupation: &[f64],
    ) -> Result<IsotropicPairMoments, &'static str> {
        if !ln_a.is_finite()
            || occupation.len() != self.len()
            || occupation
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("collisionless distribution state is invalid");
        }
        let mut comoving_number_integral = 0.0;
        let mut comoving_energy_integral = 0.0;
        for ((momentum, weight), value) in
            self.nodes_mev.iter().zip(&self.weights_mev).zip(occupation)
        {
            comoving_number_integral += weight * momentum.powi(2) * value;
            comoving_energy_integral += weight * momentum.powi(3) * value;
        }
        let phase_prefactor = PI.powi(2).recip();
        let moments = IsotropicPairMoments {
            number_density_mev3: phase_prefactor * (-3.0 * ln_a).exp() * comoving_number_integral,
            energy_density_mev4: phase_prefactor * (-4.0 * ln_a).exp() * comoving_energy_integral,
        };
        [moments.number_density_mev3, moments.energy_density_mev4]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
            .then_some(moments)
            .ok_or("collisionless moments are non-finite")
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IsotropicPairMoments {
    pub(crate) number_density_mev3: f64,
    pub(crate) energy_density_mev4: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CollisionlessIsotropicFlrwState {
    pub(crate) pair_moments: IsotropicPairMoments,
    pub(crate) rho_neutrino_total_mev4: f64,
    pub(crate) rho_total_mev4: f64,
    pub(crate) h_inverse_seconds: f64,
    pub(crate) d_tgamma_d_lna: f64,
    pub(crate) d_elapsed_seconds_d_lna: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct CollisionlessIsotropicFlrwSystem {
    grid: ComovingMomentumGrid,
}

impl CollisionlessIsotropicFlrwSystem {
    pub(crate) const fn new(grid: ComovingMomentumGrid) -> Self {
        Self { grid }
    }

    pub(crate) fn initial_fd_state(
        &self,
        t_gamma_mev: f64,
        t_nu_mev: f64,
    ) -> Result<Vec<f64>, &'static str> {
        if !t_gamma_mev.is_finite() || t_gamma_mev <= 0.0 {
            return Err("initial photon temperature must be positive and finite");
        }
        let occupation = self.grid.zero_chemical_potential_fd(t_nu_mev)?;
        let mut state = Vec::with_capacity(self.dimension());
        state.extend([t_gamma_mev, 0.0]);
        state.extend(occupation);
        Ok(state)
    }

    pub(crate) fn physical_state(
        &self,
        ln_a: f64,
        state: &[f64],
    ) -> Result<CollisionlessIsotropicFlrwState, &'static str> {
        if !self.state_is_valid(state) {
            return Err("collisionless FLRW state is invalid");
        }
        let t_gamma_mev = state[T_GAMMA_INDEX];
        let electromagnetic =
            electromagnetic_eos(t_gamma_mev).map_err(|_| "electromagnetic EOS failed")?;
        let pair_moments = self.grid.pair_moments(ln_a, &state[OCCUPATION_START..])?;
        let rho_neutrino_total_mev4 = NEUTRINO_FLAVOUR_PAIRS * pair_moments.energy_density_mev4;
        let rho_total_mev4 = electromagnetic.rho + rho_neutrino_total_mev4;
        let h_mev = (8.0 * PI * NEWTON_G_MEV_MINUS_2 * rho_total_mev4 / 3.0).sqrt();
        let h_inverse_seconds = h_mev * MEV_TO_INVERSE_SECONDS;
        let d_tgamma_d_lna =
            -3.0 * (electromagnetic.rho + electromagnetic.pressure) / electromagnetic.drho_dt;
        let d_elapsed_seconds_d_lna = h_inverse_seconds.recip();
        let result = CollisionlessIsotropicFlrwState {
            pair_moments,
            rho_neutrino_total_mev4,
            rho_total_mev4,
            h_inverse_seconds,
            d_tgamma_d_lna,
            d_elapsed_seconds_d_lna,
        };
        (result.rho_total_mev4.is_finite()
            && result.rho_total_mev4 > 0.0
            && result.h_inverse_seconds.is_finite()
            && result.h_inverse_seconds > 0.0
            && result.d_tgamma_d_lna.is_finite()
            && result.d_tgamma_d_lna < 0.0
            && result.d_elapsed_seconds_d_lna.is_finite()
            && result.d_elapsed_seconds_d_lna > 0.0)
            .then_some(result)
            .ok_or("collisionless FLRW output is invalid")
    }
}

impl OdeSystem for CollisionlessIsotropicFlrwSystem {
    fn dimension(&self) -> usize {
        OCCUPATION_START + self.grid.len()
    }

    fn state_is_valid(&self, state: &[f64]) -> bool {
        state.len() == self.dimension()
            && state[T_GAMMA_INDEX].is_finite()
            && state[T_GAMMA_INDEX] > 0.0
            && state[ELAPSED_SECONDS_INDEX].is_finite()
            && state[ELAPSED_SECONDS_INDEX] >= 0.0
            && state[OCCUPATION_START..]
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    }

    fn rhs(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        match self.physical_state(ln_a, state) {
            Ok(value) => {
                out.fill(0.0);
                out[T_GAMMA_INDEX] = value.d_tgamma_d_lna;
                out[ELAPSED_SECONDS_INDEX] = value.d_elapsed_seconds_d_lna;
            }
            Err(_) => out.fill(f64::NAN),
        }
    }

    fn jacobian(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let Ok(value) = self.physical_state(ln_a, state) else {
            out.fill(f64::NAN);
            return;
        };
        let dimension = self.dimension();
        let step = 1.0e-5 * state[T_GAMMA_INDEX];
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[T_GAMMA_INDEX] += step;
        minus[T_GAMMA_INDEX] -= step;
        let mut plus_rhs = vec![0.0; dimension];
        let mut minus_rhs = vec![0.0; dimension];
        self.rhs(ln_a, &plus, &mut plus_rhs);
        self.rhs(ln_a, &minus, &mut minus_rhs);
        out[T_GAMMA_INDEX * dimension + T_GAMMA_INDEX] =
            (plus_rhs[T_GAMMA_INDEX] - minus_rhs[T_GAMMA_INDEX]) / (2.0 * step);

        let electromagnetic = electromagnetic_eos(state[T_GAMMA_INDEX])
            .expect("validated temperature must have an electromagnetic EOS");
        out[ELAPSED_SECONDS_INDEX * dimension + T_GAMMA_INDEX] =
            -0.5 * value.d_elapsed_seconds_d_lna * electromagnetic.drho_dt / value.rho_total_mev4;
        let redshift = (-4.0 * ln_a).exp() / PI.powi(2);
        for index in 0..self.grid.len() {
            let pair_derivative =
                redshift * self.grid.weights_mev[index] * self.grid.nodes_mev[index].powi(3);
            out[ELAPSED_SECONDS_INDEX * dimension + OCCUPATION_START + index] =
                -0.5 * value.d_elapsed_seconds_d_lna * NEUTRINO_FLAVOUR_PAIRS * pair_derivative
                    / value.rho_total_mev4;
        }
    }

    fn dfdt(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        match self.physical_state(ln_a, state) {
            Ok(value) => {
                out[ELAPSED_SECONDS_INDEX] =
                    2.0 * value.d_elapsed_seconds_d_lna * value.rho_neutrino_total_mev4
                        / value.rho_total_mev4;
            }
            Err(_) => out.fill(f64::NAN),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct IsotropicBoltzmannFlrwState {
    pub(crate) t_cm_mev: f64,
    pub(crate) electron_pair_occupation: Vec<f64>,
    pub(crate) heavy_pair_occupation: Vec<f64>,
    pub(crate) electron_pair_moments: IsotropicPairMoments,
    pub(crate) heavy_pair_moments: IsotropicPairMoments,
    pub(crate) rho_neutrino_total_mev4: f64,
    pub(crate) rho_total_mev4: f64,
    pub(crate) pressure_total_mev4: f64,
    pub(crate) h_mev: f64,
    pub(crate) h_inverse_seconds: f64,
    pub(crate) q_electron_pair_mev5: f64,
    pub(crate) q_heavy_pair_mev5: f64,
    pub(crate) q_total_mev5: f64,
    pub(crate) d_tgamma_d_lna: f64,
    pub(crate) d_elapsed_seconds_d_lna: f64,
    pub(crate) electron_action: IsotropicElectronSpectralAction,
    pub(crate) neutrino_self_action: IsotropicNeutrinoSelfAction,
}

#[derive(Clone, Debug)]
pub(crate) struct IsotropicBoltzmannFlrwSystem {
    grid: ComovingMomentumGrid,
    reference_temperature_mev: f64,
    electron_rule: ElectronSpectralRule,
    neutrino_self_rule: NeutrinoSelfSpectralRule,
}

impl IsotropicBoltzmannFlrwSystem {
    pub(crate) fn new(
        grid: ComovingMomentumGrid,
        reference_temperature_mev: f64,
        electron_rule: ElectronSpectralRule,
        neutrino_self_rule: NeutrinoSelfSpectralRule,
    ) -> Result<Self, &'static str> {
        if !reference_temperature_mev.is_finite() || reference_temperature_mev <= 0.0 {
            return Err("spectral reference temperature must be positive and finite");
        }
        let laguerre = ComovingMomentumGrid::gauss_laguerre(grid.len(), reference_temperature_mev)?;
        let selected = ComovingMomentumGrid::selected_decoupling(reference_temperature_mev)?;
        let matches_laguerre =
            grid.nodes_mev == laguerre.nodes_mev && grid.weights_mev == laguerre.weights_mev;
        let matches_selected =
            grid.nodes_mev == selected.nodes_mev && grid.weights_mev == selected.weights_mev;
        if !matches_laguerre && !matches_selected {
            return Err("spectral grid must use the declared reference temperature");
        }
        Ok(Self {
            grid,
            reference_temperature_mev,
            electron_rule,
            neutrino_self_rule,
        })
    }

    fn electron_start(&self) -> usize {
        OCCUPATION_START
    }

    fn heavy_start(&self) -> usize {
        OCCUPATION_START + self.grid.len()
    }

    fn logits<'a>(&self, state: &'a [f64]) -> (&'a [f64], &'a [f64]) {
        let heavy_start = self.heavy_start();
        (
            &state[self.electron_start()..heavy_start],
            &state[heavy_start..],
        )
    }

    fn occupation_from_logit(logit: f64) -> f64 {
        if logit >= 0.0 {
            1.0 / (1.0 + (-logit).exp())
        } else {
            let exponential = logit.exp();
            exponential / (1.0 + exponential)
        }
    }

    fn occupation_values(&self, state: &[f64]) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
        let (electron_logit, heavy_logit) = self.logits(state);
        let electron = electron_logit
            .iter()
            .copied()
            .map(Self::occupation_from_logit)
            .collect::<Vec<_>>();
        let heavy = heavy_logit
            .iter()
            .copied()
            .map(Self::occupation_from_logit)
            .collect::<Vec<_>>();
        electron
            .iter()
            .chain(&heavy)
            .all(|value| value.is_finite() && *value > 0.0 && *value < 1.0)
            .then_some((electron, heavy))
            .ok_or("spectral logit state cannot represent a strict occupation")
    }

    pub(crate) fn initial_fd_state(&self, t_gamma_mev: f64) -> Result<Vec<f64>, &'static str> {
        if t_gamma_mev.to_bits() != self.reference_temperature_mev.to_bits() {
            return Err("spectral initial photon and reference temperatures must match");
        }
        let logit = self
            .grid
            .nodes_mev
            .iter()
            .map(|momentum| -momentum / self.reference_temperature_mev)
            .collect::<Vec<_>>();
        let mut state = Vec::with_capacity(self.dimension());
        state.extend([t_gamma_mev, 0.0]);
        state.extend_from_slice(&logit);
        state.extend_from_slice(&logit);
        Ok(state)
    }

    fn collision_energy_moment(&self, ln_a: f64, action_mev: &[f64]) -> f64 {
        (-4.0 * ln_a).exp() / PI.powi(2)
            * self
                .grid
                .nodes_mev
                .iter()
                .zip(&self.grid.weights_mev)
                .zip(action_mev)
                .map(|((&momentum, &weight), &value)| weight * momentum.powi(3) * value)
                .sum::<f64>()
    }

    fn physical_state_impl(
        &self,
        ln_a: f64,
        state: &[f64],
        include_jacobian: bool,
    ) -> Result<IsotropicBoltzmannFlrwState, &'static str> {
        if !ln_a.is_finite() || !self.state_is_valid(state) {
            return Err("spectral FLRW state is invalid");
        }
        let t_gamma_mev = state[T_GAMMA_INDEX];
        let t_cm_mev = self.reference_temperature_mev * (-ln_a).exp();
        let (electron_pair, heavy_pair) = self.occupation_values(state)?;
        let y_nodes = self
            .grid
            .nodes_mev
            .iter()
            .map(|value| value / self.reference_temperature_mev)
            .collect::<Vec<_>>();
        let y_weights = self
            .grid
            .weights_mev
            .iter()
            .map(|value| value / self.reference_temperature_mev)
            .collect::<Vec<_>>();
        let spectral_input = ElectronSpectralInput {
            t_gamma_mev,
            t_cm_mev,
            y_nodes: &y_nodes,
            y_weights: &y_weights,
            electron_pair: &electron_pair,
            heavy_pair: &heavy_pair,
            electron_mass_mev: ELECTRON_MASS_MEV,
            rule: self.electron_rule,
        };
        let electron_action = if include_jacobian {
            evaluate_isotropic_electron_spectral_action(spectral_input)?
        } else {
            evaluate_isotropic_electron_spectral_action_values(spectral_input)?
        };
        let (electron_logit, heavy_logit) = self.logits(state);
        let neutrino_self_input = NeutrinoSelfSpectralInput {
            t_cm_mev,
            y_nodes: &y_nodes,
            y_weights: &y_weights,
            electron_pair_logit: electron_logit,
            heavy_pair_logit: heavy_logit,
            rule: self.neutrino_self_rule,
        };
        let neutrino_self_action = if include_jacobian {
            evaluate_isotropic_neutrino_self_action(neutrino_self_input)?
        } else {
            evaluate_isotropic_neutrino_self_action_values(neutrino_self_input)?
        };
        let electron_pair_moments = self.grid.pair_moments(ln_a, &electron_pair)?;
        let heavy_pair_moments = self.grid.pair_moments(ln_a, &heavy_pair)?;
        let rho_neutrino_total_mev4 = electron_pair_moments.energy_density_mev4
            + 2.0 * heavy_pair_moments.energy_density_mev4;
        let electromagnetic =
            electromagnetic_eos(t_gamma_mev).map_err(|_| "electromagnetic EOS failed")?;
        let rho_total_mev4 = electromagnetic.rho + rho_neutrino_total_mev4;
        let pressure_total_mev4 = electromagnetic.pressure + rho_neutrino_total_mev4 / 3.0;
        let h_mev = (8.0 * PI * NEWTON_G_MEV_MINUS_2 * rho_total_mev4 / 3.0).sqrt();
        let h_inverse_seconds = h_mev * MEV_TO_INVERSE_SECONDS;
        let q_electron_pair_mev5 =
            self.collision_energy_moment(ln_a, &electron_action.electron_pair_mev);
        let q_heavy_pair_mev5 = self.collision_energy_moment(ln_a, &electron_action.heavy_pair_mev);
        let q_total_mev5 = q_electron_pair_mev5 + 2.0 * q_heavy_pair_mev5;
        let d_tgamma_d_lna = (-3.0 * (electromagnetic.rho + electromagnetic.pressure)
            - q_total_mev5 / h_mev)
            / electromagnetic.drho_dt;
        let d_elapsed_seconds_d_lna = h_inverse_seconds.recip();
        let result = IsotropicBoltzmannFlrwState {
            t_cm_mev,
            electron_pair_occupation: electron_pair,
            heavy_pair_occupation: heavy_pair,
            electron_pair_moments,
            heavy_pair_moments,
            rho_neutrino_total_mev4,
            rho_total_mev4,
            pressure_total_mev4,
            h_mev,
            h_inverse_seconds,
            q_electron_pair_mev5,
            q_heavy_pair_mev5,
            q_total_mev5,
            d_tgamma_d_lna,
            d_elapsed_seconds_d_lna,
            electron_action,
            neutrino_self_action,
        };
        [
            result.t_cm_mev,
            result.rho_neutrino_total_mev4,
            result.rho_total_mev4,
            result.pressure_total_mev4,
            result.h_mev,
            result.h_inverse_seconds,
            result.q_electron_pair_mev5,
            result.q_heavy_pair_mev5,
            result.q_total_mev5,
            result.d_tgamma_d_lna,
            result.d_elapsed_seconds_d_lna,
        ]
        .into_iter()
        .all(f64::is_finite)
        .then_some(result)
        .ok_or("spectral FLRW output is non-finite")
    }

    pub(crate) fn physical_state(
        &self,
        ln_a: f64,
        state: &[f64],
    ) -> Result<IsotropicBoltzmannFlrwState, &'static str> {
        self.physical_state_impl(ln_a, state, true)
    }

    fn write_rhs(&self, value: &IsotropicBoltzmannFlrwState, out: &mut [f64]) {
        out.fill(0.0);
        out[T_GAMMA_INDEX] = value.d_tgamma_d_lna;
        out[ELAPSED_SECONDS_INDEX] = value.d_elapsed_seconds_d_lna;
        let inverse_hubble = value.h_mev.recip();
        for node in 0..self.grid.len() {
            let electron = value.electron_pair_occupation[node];
            let heavy = value.heavy_pair_occupation[node];
            let electron_collision = value.electron_action.electron_pair_mev[node]
                + value.neutrino_self_action.electron_pair_mev[node];
            let heavy_collision = value.electron_action.heavy_pair_mev[node]
                + value.neutrino_self_action.heavy_pair_mev[node];
            out[self.electron_start() + node] =
                electron_collision * inverse_hubble / (electron * (1.0 - electron));
            out[self.heavy_start() + node] =
                heavy_collision * inverse_hubble / (heavy * (1.0 - heavy));
        }
    }

    fn write_jacobian(
        &self,
        physical: &IsotropicBoltzmannFlrwState,
        ln_a: f64,
        state: &[f64],
        out: &mut [f64],
    ) {
        let dimension = self.dimension();

        // Only the photon-temperature column changes the kinematic event
        // stream.  The occupation block below uses the exact Pauli/event
        // response and the analytic rank-one Friedmann coupling.
        let temperature_step = 1.0e-5 * state[T_GAMMA_INDEX];
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[T_GAMMA_INDEX] += temperature_step;
        minus[T_GAMMA_INDEX] -= temperature_step;
        let mut plus_rhs = vec![0.0; dimension];
        let mut minus_rhs = vec![0.0; dimension];
        self.rhs(ln_a, &plus, &mut plus_rhs);
        self.rhs(ln_a, &minus, &mut minus_rhs);
        for row in 0..dimension {
            out[row * dimension + T_GAMMA_INDEX] =
                (plus_rhs[row] - minus_rhs[row]) / (2.0 * temperature_step);
        }

        let folded_dimension = 2 * self.grid.len();
        let redshift = (-4.0 * ln_a).exp() / PI.powi(2);
        let electromagnetic = match electromagnetic_eos(state[T_GAMMA_INDEX]) {
            Ok(value) => value,
            Err(_) => {
                out.fill(f64::NAN);
                return;
            }
        };
        for folded_column in 0..folded_dimension {
            let pair = folded_column / self.grid.len();
            let node = folded_column % self.grid.len();
            let multiplicity = if pair == 0 { 1.0 } else { 2.0 };
            let input_occupation = if pair == 0 {
                physical.electron_pair_occupation[node]
            } else {
                physical.heavy_pair_occupation[node]
            };
            let input_response = input_occupation * (1.0 - input_occupation);
            let state_column = OCCUPATION_START + folded_column;
            let energy_weight =
                redshift * self.grid.weights_mev[node] * self.grid.nodes_mev[node].powi(3);
            let d_rho_total = multiplicity * energy_weight * input_response;
            let d_log_hubble = 0.5 * d_rho_total / physical.rho_total_mev4;
            let mut d_q_total = 0.0;
            for folded_row in 0..folded_dimension {
                let row_pair = folded_row / self.grid.len();
                let row_node = folded_row % self.grid.len();
                let row_multiplicity = if row_pair == 0 { 1.0 } else { 2.0 };
                let row_energy_weight = redshift
                    * self.grid.weights_mev[row_node]
                    * self.grid.nodes_mev[row_node].powi(3);
                let electron_collision_derivative = physical.electron_action.jacobian_mev
                    [folded_row * folded_dimension + folded_column];
                let self_collision_logit_derivative = physical
                    .neutrino_self_action
                    .jacobian_logit_mev[folded_row * folded_dimension + folded_column];
                let output_occupation = if row_pair == 0 {
                    physical.electron_pair_occupation[row_node]
                } else {
                    physical.heavy_pair_occupation[row_node]
                };
                let output_response = output_occupation * (1.0 - output_occupation);
                let (electron_collision, self_collision) = if row_pair == 0 {
                    (
                        physical.electron_action.electron_pair_mev[row_node],
                        physical.neutrino_self_action.electron_pair_mev[row_node],
                    )
                } else {
                    (
                        physical.electron_action.heavy_pair_mev[row_node],
                        physical.neutrino_self_action.heavy_pair_mev[row_node],
                    )
                };
                let collision = electron_collision + self_collision;
                let collision_logit_derivative = electron_collision_derivative * input_response
                    + self_collision_logit_derivative;
                // Only electron/positron collisions exchange energy with the
                // electromagnetic plasma.  The self-collision weak form has a
                // vanishing total neutrino energy moment event by event.
                d_q_total += row_multiplicity
                    * row_energy_weight
                    * electron_collision_derivative
                    * input_response;
                let state_row = OCCUPATION_START + folded_row;
                let collision_rhs = collision / physical.h_mev;
                let mut derivative = (collision_logit_derivative / physical.h_mev
                    - collision_rhs * d_log_hubble)
                    / output_response;
                if folded_row == folded_column {
                    derivative -= collision_rhs / output_response * (1.0 - 2.0 * output_occupation);
                }
                out[state_row * dimension + state_column] = derivative;
            }
            out[T_GAMMA_INDEX * dimension + state_column] = (-d_q_total / physical.h_mev
                + physical.q_total_mev5 / physical.h_mev * d_log_hubble)
                / electromagnetic.drho_dt;
            out[ELAPSED_SECONDS_INDEX * dimension + state_column] =
                -physical.d_elapsed_seconds_d_lna * d_log_hubble;
        }
    }
}

impl OdeSystem for IsotropicBoltzmannFlrwSystem {
    fn dimension(&self) -> usize {
        OCCUPATION_START + 2 * self.grid.len()
    }

    fn state_is_valid(&self, state: &[f64]) -> bool {
        state.len() == self.dimension()
            && state[T_GAMMA_INDEX].is_finite()
            && state[T_GAMMA_INDEX] > 0.0
            && state[ELAPSED_SECONDS_INDEX].is_finite()
            && state[ELAPSED_SECONDS_INDEX] >= 0.0
            && state[OCCUPATION_START..]
                .iter()
                .all(|value| value.is_finite())
    }

    fn rhs(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        let Ok(value) = self.physical_state_impl(ln_a, state, false) else {
            out.fill(f64::NAN);
            return;
        };
        self.write_rhs(&value, out);
    }

    fn jacobian(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        let Ok(physical) = self.physical_state(ln_a, state) else {
            out.fill(f64::NAN);
            return;
        };
        self.write_jacobian(&physical, ln_a, state, out);
    }

    fn rhs_and_jacobian(&self, ln_a: f64, state: &[f64], f_out: &mut [f64], jac_out: &mut [f64]) {
        let Ok(physical) = self.physical_state_impl(ln_a, state, true) else {
            f_out.fill(f64::NAN);
            jac_out.fill(f64::NAN);
            return;
        };
        self.write_rhs(&physical, f_out);
        jac_out.fill(0.0);
        self.write_jacobian(&physical, ln_a, state, jac_out);
    }

    fn dfdt(&self, ln_a: f64, state: &[f64], out: &mut [f64]) {
        let step = 1.0e-5;
        let dimension = self.dimension();
        let mut plus_one = vec![0.0; dimension];
        let mut minus_one = vec![0.0; dimension];
        self.rhs(ln_a + step, state, &mut plus_one);
        self.rhs(ln_a - step, state, &mut minus_one);
        for row in 0..dimension {
            out[row] = (plus_one[row] - minus_one[row]) / (2.0 * step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::electron_hm::G_F_MEV_MINUS_2;
    use crate::ode::{OdeConfig, SolverKind, TerminalEvent, solve};
    use std::time::Instant;

    const ZETA_THREE: f64 = 1.202_056_903_159_594_2;

    fn relative_error(actual: f64, expected: f64) -> f64 {
        (actual - expected).abs() / actual.abs().max(expected.abs()).max(f64::MIN_POSITIVE)
    }

    #[test]
    fn fd_number_and_energy_moments_converge_without_a_tail_cutoff() {
        let temperature: f64 = 2.0;
        let expected_number = 3.0 * ZETA_THREE / (2.0 * PI.powi(2)) * temperature.powi(3);
        let expected_energy = 7.0 * PI.powi(2) / 120.0 * temperature.powi(4);
        let mut previous = [f64::INFINITY; 2];
        for order in [12, 24, 36] {
            let grid = ComovingMomentumGrid::gauss_laguerre(order, temperature).unwrap();
            let occupation = grid.zero_chemical_potential_fd(temperature).unwrap();
            let moments = grid.pair_moments(0.0, &occupation).unwrap();
            let residuals = [
                relative_error(moments.number_density_mev3, expected_number),
                relative_error(moments.energy_density_mev4, expected_energy),
            ];
            assert!(residuals[0] < previous[0] && residuals[1] < previous[1]);
            previous = residuals;
        }
        assert!(previous[0] < 4.0e-12 && previous[1] < 6.0e-12);
    }

    #[test]
    fn selected_exponential_grid_resolves_fd_moments_and_is_constructor_locked() {
        let temperature: f64 = 2.0;
        let grid = ComovingMomentumGrid::selected_decoupling(temperature).unwrap();
        assert_eq!(grid.len(), 48);
        assert!(grid.nodes_mev.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(grid.weights_mev.iter().all(|weight| *weight > 0.0));
        let occupation = grid.zero_chemical_potential_fd(temperature).unwrap();
        let moments = grid.pair_moments(0.0, &occupation).unwrap();
        let expected_number = 3.0 * ZETA_THREE / (2.0 * PI.powi(2)) * temperature.powi(3);
        let expected_energy = 7.0 * PI.powi(2) / 120.0 * temperature.powi(4);
        assert!(relative_error(moments.number_density_mev3, expected_number) < 1.0e-5);
        assert!(relative_error(moments.energy_density_mev4, expected_energy) < 1.0e-5);

        let system = IsotropicBoltzmannFlrwSystem::new(
            grid.clone(),
            temperature,
            ElectronSpectralRule {
                electron_radial_order: 4,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 4 },
        )
        .unwrap();
        assert_eq!(system.grid.nodes_mev, grid.nodes_mev);
        assert!(
            IsotropicBoltzmannFlrwSystem::new(
                grid,
                1.01 * temperature,
                ElectronSpectralRule {
                    electron_radial_order: 4,
                    angular_order: 4,
                },
                NeutrinoSelfSpectralRule { angular_order: 4 },
            )
            .is_err()
        );
    }

    #[test]
    fn selected_exponential_grid_keeps_the_electron_thermal_anchor() {
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::selected_decoupling(1.0).unwrap(),
            1.0,
            ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 4 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(1.0).unwrap();
        state[T_GAMMA_INDEX] = 1.2;
        let physical = system.physical_state_impl(0.0, &state, false).unwrap();
        let actual = [
            physical.q_electron_pair_mev5 / G_F_MEV_MINUS_2.powi(2),
            physical.q_heavy_pair_mev5 / G_F_MEV_MINUS_2.powi(2),
        ];
        let expected = [1.047_175_324_837_161_4, 0.220_065_930_685_499_34];
        eprintln!(
            "F10C exponential electron thermal anchor: e/GF2={:.17e} residual={:.17e}; x/GF2={:.17e} residual={:.17e}",
            actual[0],
            relative_error(actual[0], expected[0]),
            actual[1],
            relative_error(actual[1], expected[1]),
        );
        assert!(actual.into_iter().zip(expected).all(|(value, anchor)| {
            value.is_finite() && value > 0.0 && relative_error(value, anchor) < 0.01
        }));
    }

    #[test]
    fn arbitrary_distribution_has_exact_comoving_number_and_energy_scaling() {
        let grid = ComovingMomentumGrid::gauss_laguerre(16, 1.0).unwrap();
        let occupation = grid
            .nodes_mev
            .iter()
            .map(|momentum| 0.2 + 0.3 * (-momentum).exp())
            .collect::<Vec<_>>();
        let first = grid.pair_moments(0.3, &occupation).unwrap();
        let second = grid.pair_moments(2.7, &occupation).unwrap();
        assert!(
            relative_error(
                first.number_density_mev3 * (3.0_f64 * 0.3).exp(),
                second.number_density_mev3 * (3.0_f64 * 2.7).exp(),
            ) < 3.0e-15
        );
        assert!(
            relative_error(
                first.energy_density_mev4 * (4.0_f64 * 0.3).exp(),
                second.energy_density_mev4 * (4.0_f64 * 2.7).exp(),
            ) < 3.0e-15
        );
    }

    #[test]
    fn analytic_jacobian_and_explicit_time_derivative_match_five_point_stencils() {
        let grid = ComovingMomentumGrid::gauss_laguerre(8, 1.0).unwrap();
        let system = CollisionlessIsotropicFlrwSystem::new(grid);
        let mut state = vec![1.1, 0.2];
        state.extend((0..system.grid.len()).map(|index| 0.25 + 0.01 * index as f64));
        let ln_a = 0.4;
        let dimension = system.dimension();
        let mut analytic = vec![0.0; dimension * dimension];
        system.jacobian(ln_a, &state, &mut analytic);
        for column in 0..dimension {
            let step = if column >= OCCUPATION_START {
                2.0e-3
            } else {
                2.0e-6 * state[column].abs().max(1.0)
            };
            let mut p2 = state.clone();
            let mut p1 = state.clone();
            let mut m1 = state.clone();
            let mut m2 = state.clone();
            p2[column] += 2.0 * step;
            p1[column] += step;
            m1[column] -= step;
            m2[column] -= 2.0 * step;
            let mut fp2 = vec![0.0; dimension];
            let mut fp1 = vec![0.0; dimension];
            let mut fm1 = vec![0.0; dimension];
            let mut fm2 = vec![0.0; dimension];
            system.rhs(ln_a, &p2, &mut fp2);
            system.rhs(ln_a, &p1, &mut fp1);
            system.rhs(ln_a, &m1, &mut fm1);
            system.rhs(ln_a, &m2, &mut fm2);
            for row in 0..dimension {
                let expected =
                    (-fp2[row] + 8.0 * fp1[row] - 8.0 * fm1[row] + fm2[row]) / (12.0 * step);
                let actual = analytic[row * dimension + column];
                let scale = actual.abs().max(expected.abs()).max(1.0e-12);
                let residual = (actual - expected).abs() / scale;
                assert!(
                    (actual - expected).abs() < 2.0e-12 || residual < 3.0e-6,
                    "row={row} column={column} actual={actual:.17e} expected={expected:.17e} residual={residual:.17e}"
                );
            }
        }
        let step = 2.0e-5;
        let mut p2 = vec![0.0; dimension];
        let mut p1 = vec![0.0; dimension];
        let mut m1 = vec![0.0; dimension];
        let mut m2 = vec![0.0; dimension];
        system.rhs(ln_a + 2.0 * step, &state, &mut p2);
        system.rhs(ln_a + step, &state, &mut p1);
        system.rhs(ln_a - step, &state, &mut m1);
        system.rhs(ln_a - 2.0 * step, &state, &mut m2);
        let expected = (-p2[1] + 8.0 * p1[1] - 8.0 * m1[1] + m2[1]) / (12.0 * step);
        let mut analytic_time = vec![0.0; dimension];
        system.dfdt(ln_a, &state, &mut analytic_time);
        assert!(relative_error(analytic_time[1], expected) < 2.0e-10);
    }

    #[test]
    fn invalid_grid_distribution_and_solver_state_fail_without_clipping() {
        assert!(ComovingMomentumGrid::gauss_laguerre(1, 1.0).is_err());
        assert!(ComovingMomentumGrid::gauss_laguerre(8, f64::NAN).is_err());
        let system = CollisionlessIsotropicFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(8, 1.0).unwrap(),
        );
        let mut state = system.initial_fd_state(2.0, 2.0).unwrap();
        state[OCCUPATION_START + 2] = 1.01;
        let config = OdeConfig {
            rtol: 1.0e-8,
            atol: vec![1.0e-11; system.dimension()],
            h_init: 1.0e-5,
            h_min: 1.0e-14,
            h_max: 0.1,
            max_attempts: 100,
        };
        let result = solve(SolverKind::Bdf, &system, (0.0, 1.0), &state, &config, None);
        assert_eq!(result.failure.as_deref(), Some("invalid_initial_state"));
        assert_eq!(result.y, state);
    }

    #[test]
    fn classical_spectral_rhs_closes_the_combined_first_law() {
        let reference_temperature = 1.0;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(4, reference_temperature).unwrap(),
            reference_temperature,
            ElectronSpectralRule {
                electron_radial_order: 4,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 4 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(reference_temperature).unwrap();
        state[T_GAMMA_INDEX] = 1.2;
        state[system.electron_start() + 1] += 0.04;
        state[system.heavy_start() + 2] -= 0.03;
        let ln_a = 0.0;
        let physical = system.physical_state(ln_a, &state).unwrap();
        assert!(physical.electron_pair_moments.energy_density_mev4 > 0.0);
        assert!(physical.heavy_pair_moments.energy_density_mev4 > 0.0);
        assert!(physical.q_electron_pair_mev5 > 0.0);
        assert!(physical.q_heavy_pair_mev5 > 0.0);
        assert!(
            physical
                .neutrino_self_action
                .electron_pair_mev
                .iter()
                .chain(&physical.neutrino_self_action.heavy_pair_mev)
                .any(|value| *value != 0.0)
        );
        let self_energy_residual = system
            .collision_energy_moment(ln_a, &physical.neutrino_self_action.electron_pair_mev)
            + 2.0
                * system
                    .collision_energy_moment(ln_a, &physical.neutrino_self_action.heavy_pair_mev);
        let self_energy_scale = (-4.0 * ln_a).exp() / PI.powi(2)
            * system
                .grid
                .nodes_mev
                .iter()
                .zip(&system.grid.weights_mev)
                .enumerate()
                .map(|(node, (&momentum, &weight))| {
                    weight
                        * momentum.powi(3)
                        * (physical.neutrino_self_action.electron_pair_mev[node].abs()
                            + 2.0 * physical.neutrino_self_action.heavy_pair_mev[node].abs())
                })
                .sum::<f64>();
        assert!(self_energy_residual.abs() < 8.0e-13 * self_energy_scale);
        let electromagnetic = electromagnetic_eos(state[T_GAMMA_INDEX]).unwrap();
        let d_rho_em = electromagnetic.drho_dt * physical.d_tgamma_d_lna;
        let d_rho_nu =
            -4.0 * physical.rho_neutrino_total_mev4 + physical.q_total_mev5 / physical.h_mev;
        let expected = -3.0 * (physical.rho_total_mev4 + physical.pressure_total_mev4);
        assert!(relative_error(d_rho_em + d_rho_nu, expected) < 3.0e-15);
        let mut rhs = vec![0.0; system.dimension()];
        system.rhs(ln_a, &state, &mut rhs);
        for node in 0..system.grid.len() {
            for (start, occupation, collision) in [
                (
                    system.electron_start(),
                    physical.electron_pair_occupation[node],
                    physical.electron_action.electron_pair_mev[node]
                        + physical.neutrino_self_action.electron_pair_mev[node],
                ),
                (
                    system.heavy_start(),
                    physical.heavy_pair_occupation[node],
                    physical.electron_action.heavy_pair_mev[node]
                        + physical.neutrino_self_action.heavy_pair_mev[node],
                ),
            ] {
                let mapped = occupation * (1.0 - occupation) * rhs[start + node];
                let expected_collision = collision / physical.h_mev;
                assert!(
                    (mapped - expected_collision).abs() < 2.0e-13
                        || relative_error(mapped, expected_collision) < 3.0e-15
                );
            }
        }
    }

    #[test]
    fn spectral_logit_state_preserves_open_occupations_and_fails_raw_on_underflow() {
        let reference_temperature = 1.0;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(4, reference_temperature).unwrap(),
            reference_temperature,
            ElectronSpectralRule {
                electron_radial_order: 4,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 4 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(reference_temperature).unwrap();
        let physical = system.physical_state(0.0, &state).unwrap();
        assert!(
            physical
                .electron_pair_occupation
                .iter()
                .chain(&physical.heavy_pair_occupation)
                .all(|value| *value > 0.0 && *value < 1.0)
        );
        state[system.electron_start()] = -1_000.0;
        assert!(system.state_is_valid(&state));
        assert!(system.physical_state(0.0, &state).is_err());
        let mut rhs = vec![0.0; system.dimension()];
        system.rhs(0.0, &state, &mut rhs);
        assert!(rhs.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn classical_spectral_flrw_jacobian_and_time_derivative_match_stencils() {
        let reference_temperature = 1.0;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(2, reference_temperature).unwrap(),
            reference_temperature,
            ElectronSpectralRule {
                electron_radial_order: 2,
                angular_order: 2,
            },
            NeutrinoSelfSpectralRule { angular_order: 2 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(reference_temperature).unwrap();
        state[T_GAMMA_INDEX] = 1.1;
        state[ELAPSED_SECONDS_INDEX] = 0.2;
        state[system.electron_start()] *= 0.97;
        state[system.heavy_start() + 1] *= 1.03;
        let ln_a = 0.1;
        let dimension = system.dimension();
        let mut analytic = vec![0.0; dimension * dimension];
        system.jacobian(ln_a, &state, &mut analytic);
        for column in 0..dimension {
            let step = if column == ELAPSED_SECONDS_INDEX {
                2.0e-5
            } else if column == T_GAMMA_INDEX {
                2.0e-5 * state[column]
            } else {
                2.0e-5 * state[column].abs().max(1.0)
            };
            let evaluate = |offset: f64| {
                let mut shifted = state.clone();
                shifted[column] += offset * step;
                let mut value = vec![0.0; dimension];
                system.rhs(ln_a, &shifted, &mut value);
                value
            };
            let p2 = evaluate(2.0);
            let p1 = evaluate(1.0);
            let m1 = evaluate(-1.0);
            let m2 = evaluate(-2.0);
            for row in 0..dimension {
                let expected = (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step);
                let actual = analytic[row * dimension + column];
                assert!(
                    (actual - expected).abs() < 2.0e-9 || relative_error(actual, expected) < 3.0e-5,
                    "row={row} column={column} actual={actual:.17e} expected={expected:.17e}"
                );
            }
        }
        let step = 2.0e-5;
        let evaluate_time = |offset: f64| {
            let mut value = vec![0.0; dimension];
            system.rhs(ln_a + offset * step, &state, &mut value);
            value
        };
        let p2 = evaluate_time(2.0);
        let p1 = evaluate_time(1.0);
        let m1 = evaluate_time(-1.0);
        let m2 = evaluate_time(-2.0);
        let mut analytic_time = vec![0.0; dimension];
        system.dfdt(ln_a, &state, &mut analytic_time);
        for row in 0..dimension {
            let expected = (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step);
            assert!(
                (analytic_time[row] - expected).abs() < 2.0e-9
                    || relative_error(analytic_time[row], expected) < 3.0e-4,
                "time row={row} actual={:.17e} expected={expected:.17e}",
                analytic_time[row],
            );
        }
    }

    #[test]
    fn fused_rhs_and_jacobian_matches_separate_calls_bitwise() {
        let reference_temperature = 1.0;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(2, reference_temperature).unwrap(),
            reference_temperature,
            ElectronSpectralRule {
                electron_radial_order: 2,
                angular_order: 2,
            },
            NeutrinoSelfSpectralRule { angular_order: 2 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(reference_temperature).unwrap();
        state[T_GAMMA_INDEX] = 1.1;
        state[ELAPSED_SECONDS_INDEX] = 0.2;
        state[system.electron_start()] *= 0.97;
        state[system.heavy_start() + 1] *= 1.03;
        let ln_a = 0.1;
        let dimension = system.dimension();

        let mut f_fused = vec![0.0; dimension];
        let mut j_fused = vec![0.0; dimension * dimension];
        system.rhs_and_jacobian(ln_a, &state, &mut f_fused, &mut j_fused);

        let mut f_sep = vec![0.0; dimension];
        system.rhs(ln_a, &state, &mut f_sep);
        let mut j_sep = vec![0.0; dimension * dimension];
        system.jacobian(ln_a, &state, &mut j_sep);

        for i in 0..dimension {
            assert_eq!(
                f_fused[i].to_bits(),
                f_sep[i].to_bits(),
                "rhs mismatch at index {i}: fused={:.17e} separate={:.17e}",
                f_fused[i],
                f_sep[i]
            );
        }
        for k in 0..dimension * dimension {
            assert_eq!(
                j_fused[k].to_bits(),
                j_sep[k].to_bits(),
                "jacobian mismatch at index {k}: fused={:.17e} separate={:.17e}",
                j_fused[k],
                j_sep[k]
            );
        }
    }

    #[test]
    fn selected_spectral_rule_has_a_bounded_explicit_lna_derivative_step_ladder() {
        let reference_temperature = 1.0;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(6, reference_temperature).unwrap(),
            reference_temperature,
            ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 12 },
        )
        .unwrap();
        let mut state = system.initial_fd_state(reference_temperature).unwrap();
        state[T_GAMMA_INDEX] = 1.2;
        state[system.electron_start() + 1] += 0.04;
        state[system.heavy_start() + 2] -= 0.03;
        let ln_a = 0.15;
        let derivative = |step: f64| {
            let dimension = system.dimension();
            let evaluate = |offset: f64| {
                let mut value = vec![0.0; dimension];
                system.rhs(ln_a + offset * step, &state, &mut value);
                value
            };
            let p2 = evaluate(2.0);
            let p1 = evaluate(1.0);
            let m1 = evaluate(-1.0);
            let m2 = evaluate(-2.0);
            (0..dimension)
                .map(|row| (-p2[row] + 8.0 * p1[row] - 8.0 * m1[row] + m2[row]) / (12.0 * step))
                .collect::<Vec<_>>()
        };
        let ladders = [4.0e-5, 2.0e-5, 1.0e-5].map(derivative);
        let finest = &ladders[2];
        for (step, candidate) in [4.0e-5, 2.0e-5].into_iter().zip(&ladders[..2]) {
            for (label, start, end) in [
                ("photon", T_GAMMA_INDEX, T_GAMMA_INDEX + 1),
                ("elapsed", ELAPSED_SECONDS_INDEX, ELAPSED_SECONDS_INDEX + 1),
                ("occupation", OCCUPATION_START, system.dimension()),
            ] {
                let difference = candidate[start..end]
                    .iter()
                    .zip(&finest[start..end])
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f64, f64::max);
                let scale = candidate[start..end]
                    .iter()
                    .chain(&finest[start..end])
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max)
                    .max(1.0e-30);
                let residual = difference / scale;
                eprintln!(
                    "F10C explicit-N support ladder step={step:.1e} block={label} residual={residual:.17e}"
                );
                assert!(
                    residual < 3.0e-4,
                    "step={step:.1e} block={label} residual={residual:.17e}"
                );
            }
        }
    }

    #[test]
    fn both_solvers_consume_the_classical_collision_action_to_an_flrw_event() {
        const F10C2_FULL_CATALOGUE_N48_BDF_LN_A: f64 = 7.936_693_339_485_084;
        const F10C2_FULL_CATALOGUE_N48_BDF_NEFF: f64 = 3.034_035_983_584_399_5;
        let initial_temperature = 10.0;
        let final_temperature = 0.005;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::selected_decoupling(initial_temperature).unwrap(),
            initial_temperature,
            ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 12 },
        )
        .unwrap();
        let initial = system.initial_fd_state(initial_temperature).unwrap();
        let event_fn = |_ln_a: f64, state: &[f64]| state[T_GAMMA_INDEX] - final_temperature;
        let event = TerminalEvent {
            value: &event_fn,
            direction: -1,
        };
        let mut atol = vec![2.0e-10; system.dimension()];
        atol[T_GAMMA_INDEX] = 2.0e-9;
        atol[ELAPSED_SECONDS_INDEX] = 2.0e-5;
        let config = OdeConfig {
            rtol: 2.0e-7,
            atol,
            h_init: 1.0e-5,
            h_min: 1.0e-12,
            h_max: 0.04,
            max_attempts: 100_000,
        };
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let leg_start = Instant::now();
            let result = solve(kind, &system, (0.0, 10.0), &initial, &config, Some(&event));
            let leg_wall = leg_start.elapsed();
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!(system.state_is_valid(&result.y));
            assert!((result.y[T_GAMMA_INDEX] - final_temperature).abs() < 2.0e-8);
            assert!(result.y[ELAPSED_SECONDS_INDEX] > 0.0);
            assert!(
                result.y[OCCUPATION_START..]
                    .iter()
                    .zip(&initial[OCCUPATION_START..])
                    .any(|(final_value, initial_value)| final_value.to_bits()
                        != initial_value.to_bits())
            );
            let physical = system.physical_state(result.t, &result.y).unwrap();
            assert!(physical.rho_neutrino_total_mev4 > 0.0);
            assert!(
                physical
                    .neutrino_self_action
                    .electron_pair_mev
                    .iter()
                    .chain(&physical.neutrino_self_action.heavy_pair_mev)
                    .any(|value| *value != 0.0)
            );
            let rho_gamma = PI.powi(2) / 15.0 * final_temperature.powi(4);
            let n_eff =
                8.0 / 7.0 * (11.0_f64 / 4.0).powf(4.0 / 3.0) * physical.rho_neutrino_total_mev4
                    / rho_gamma;
            let (initial_electron, initial_heavy) = system.occupation_values(&initial).unwrap();
            let maximum_resolved_relative_distortion = physical
                .electron_pair_occupation
                .iter()
                .chain(&physical.heavy_pair_occupation)
                .zip(initial_electron.iter().chain(&initial_heavy))
                .filter(|(_, initial_value)| **initial_value > 1.0e-8)
                .map(|(final_value, initial_value)| {
                    (final_value - initial_value).abs() / initial_value
                })
                .fold(0.0_f64, f64::max);
            eprintln!(
                "F10C {kind:?}: N={:.17e} elapsed={:.17e}s Neff={n_eff:.17e} max_df/f(f0>1e-8)={maximum_resolved_relative_distortion:.17e} q={:.17e} steps={}/{} wall={:.6}s rhs={} dfdt={}",
                result.t,
                result.y[ELAPSED_SECONDS_INDEX],
                physical.q_total_mev5,
                result.accepted,
                result.rejected,
                leg_wall.as_secs_f64(),
                result.rhs_evaluations,
                result.dfdt_evaluations,
            );
            if kind == SolverKind::Bdf {
                let self_electron_energy = system.collision_energy_moment(
                    result.t,
                    &physical.neutrino_self_action.electron_pair_mev,
                );
                let self_heavy_energy = system.collision_energy_moment(
                    result.t,
                    &physical.neutrino_self_action.heavy_pair_mev,
                );
                eprintln!(
                    "F10C2_BDF_BLOCKS Tcm={:.17e} H={:.17e} rho_nu={:.17e} n_e={:.17e} rho_e={:.17e} n_x={:.17e} rho_x={:.17e} Qe_e={:.17e} Qe_x={:.17e} Qe_total={:.17e} Qnunu_e={self_electron_energy:.17e} Qnunu_x={self_heavy_energy:.17e}",
                    physical.t_cm_mev,
                    physical.h_inverse_seconds,
                    physical.rho_neutrino_total_mev4,
                    physical.electron_pair_moments.number_density_mev3,
                    physical.electron_pair_moments.energy_density_mev4,
                    physical.heavy_pair_moments.number_density_mev3,
                    physical.heavy_pair_moments.energy_density_mev4,
                    physical.q_electron_pair_mev5,
                    physical.q_heavy_pair_mev5,
                    physical.q_total_mev5,
                );
                for (node, (((&momentum, &weight), &electron), &heavy)) in system
                    .grid
                    .nodes_mev
                    .iter()
                    .zip(&system.grid.weights_mev)
                    .zip(&physical.electron_pair_occupation)
                    .zip(&physical.heavy_pair_occupation)
                    .enumerate()
                {
                    eprintln!(
                        "F10C2_BDF_SPECTRUM node={node} y={:.17e} weight={:.17e} f_e={electron:.17e} f_x={heavy:.17e}",
                        momentum / system.reference_temperature_mev,
                        weight / system.reference_temperature_mev,
                    );
                }
            }
            endpoints.push((result, n_eff));
        }
        assert!((endpoints[0].0.t - endpoints[1].0.t).abs() < 2.0e-5);
        for (index, (bdf, rodas)) in endpoints[0].0.y.iter().zip(&endpoints[1].0.y).enumerate() {
            let ceiling = 3.0e-5 * bdf.abs().max(rodas.abs()).max(1.0e-8);
            assert!(
                (bdf - rodas).abs() < ceiling,
                "index={index} bdf={bdf:.17e} rodas={rodas:.17e} difference={:.17e} ceiling={ceiling:.17e}",
                (bdf - rodas).abs(),
            );
        }
        assert!((endpoints[0].0.t - F10C2_FULL_CATALOGUE_N48_BDF_LN_A).abs() < 3.0e-7);
        assert!((endpoints[0].1 - F10C2_FULL_CATALOGUE_N48_BDF_NEFF).abs() < 3.0e-7);
    }

    #[test]
    #[ignore = "diagnostic: h_max fairness sweep, run manually with --ignored"]
    fn rodas5p_hmax_sensitivity_diagnostic() {
        // The headline test `both_solvers_consume_the_classical_collision_action_to_an_flrw_event`
        // pins h_max=0.04, which is a near-binding step cap for Rodas5P (mean accepted
        // step there is ~0.0293, so some steps get clamped) but is only a
        // `set_stop_time` chunk window for BDF (solve_bdf), not a step-size cap. That is
        // an inherent fairness asymmetry between the two legs at the same h_max. This
        // diagnostic reruns Rodas5P alone at larger h_max to quantify how much the cap
        // suppresses Rodas's step-count advantage. It is pure measurement: no production
        // default changes, and it does not touch the frozen BDF anchor used by the
        // headline test. Raising h_max also perturbs BDF's own chunking behavior via
        // `set_stop_time`, so the production h_max=0.04 cannot be changed unilaterally
        // from this data alone without re-anchoring BDF; that remains an owner-level
        // decision this diagnostic merely informs.
        const F10C2_FULL_CATALOGUE_N48_BDF_LN_A: f64 = 7.936_693_339_485_084;
        let initial_temperature = 10.0;
        let final_temperature = 0.005;
        let system = IsotropicBoltzmannFlrwSystem::new(
            ComovingMomentumGrid::selected_decoupling(initial_temperature).unwrap(),
            initial_temperature,
            ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            NeutrinoSelfSpectralRule { angular_order: 12 },
        )
        .unwrap();
        let initial = system.initial_fd_state(initial_temperature).unwrap();
        let event_fn = |_ln_a: f64, state: &[f64]| state[T_GAMMA_INDEX] - final_temperature;
        let event = TerminalEvent {
            value: &event_fn,
            direction: -1,
        };
        let mut atol = vec![2.0e-10; system.dimension()];
        atol[T_GAMMA_INDEX] = 2.0e-9;
        atol[ELAPSED_SECONDS_INDEX] = 2.0e-5;
        for h_max in [0.04_f64, 0.08, 0.16] {
            let config = OdeConfig {
                rtol: 2.0e-7,
                atol: atol.clone(),
                h_init: 1.0e-5,
                h_min: 1.0e-12,
                h_max,
                max_attempts: 100_000,
            };
            let start = Instant::now();
            let result = solve(
                SolverKind::Rodas5P,
                &system,
                (0.0, 10.0),
                &initial,
                &config,
                Some(&event),
            );
            let wall = start.elapsed().as_secs_f64();
            assert_eq!(result.failure, None, "h_max={h_max}: {result:?}");
            assert!(result.event_reached, "h_max={h_max}: {result:?}");
            assert!(result.t.is_finite());
            eprintln!(
                "B3_HMAX h_max={h_max:.3} steps={}/{} wall={wall:.3}s N={:.17e} rhs={} dfdt={} \
                 drift_vs_bdf_anchor={:.3e}",
                result.accepted,
                result.rejected,
                result.t,
                result.rhs_evaluations,
                result.dfdt_evaluations,
                (result.t - F10C2_FULL_CATALOGUE_N48_BDF_LN_A).abs(),
            );
        }
    }

    #[test]
    fn both_solvers_consume_the_collisionless_distribution_to_the_flrw_event() {
        // Independent SciPy quad_vec EOS plus continuum FD density, frozen
        // before this Rust comparison. DOP853 and Radau at rtol=3e-11 agreed
        // to 1.5e-13 in N and 1.6e-7 seconds in elapsed time.
        const EXTERNAL_LN_A: f64 = 7.938_042_640_441_81;
        const EXTERNAL_ELAPSED_SECONDS: f64 = 52_796.035_152_103_475;
        let initial_temperature = 10.0;
        let final_temperature = 0.005;
        let system = CollisionlessIsotropicFlrwSystem::new(
            ComovingMomentumGrid::gauss_laguerre(24, initial_temperature).unwrap(),
        );
        let initial = system
            .initial_fd_state(initial_temperature, initial_temperature)
            .unwrap();
        let initial_em = electromagnetic_eos(initial_temperature).unwrap();
        let final_em = electromagnetic_eos(final_temperature).unwrap();
        let expected_ln_a = (initial_em.entropy / final_em.entropy).ln() / 3.0;
        let final_event_fn = |_ln_a: f64, state: &[f64]| state[0] - final_temperature;
        let final_event = TerminalEvent {
            value: &final_event_fn,
            direction: -1,
        };
        let mut atol = vec![2.0e-13; system.dimension()];
        atol[ELAPSED_SECONDS_INDEX] = 1.0e-7;
        let config = OdeConfig {
            rtol: 2.0e-10,
            atol,
            h_init: 1.0e-5,
            h_min: 1.0e-14,
            h_max: 0.025,
            max_attempts: 100_000,
        };
        let mut endpoints = Vec::new();
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &system,
                (0.0, 12.0),
                &initial,
                &config,
                Some(&final_event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!((result.t - expected_ln_a).abs() < 5.0e-7);
            assert!((result.t - EXTERNAL_LN_A).abs() < 5.0e-8);
            assert!(result.y[ELAPSED_SECONDS_INDEX] > 0.0);
            assert!(
                (result.y[ELAPSED_SECONDS_INDEX] - EXTERNAL_ELAPSED_SECONDS).abs() < 2.0e-3,
                "{kind:?}: {result:?}"
            );
            assert_eq!(
                result.y[OCCUPATION_START..]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                initial[OCCUPATION_START..]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let physical = system.physical_state(result.t, &result.y).unwrap();
            let t_nu = initial_temperature * (-result.t).exp();
            let expected_energy = 7.0 * PI.powi(2) / 120.0 * t_nu.powi(4);
            let expected_number = 3.0 * ZETA_THREE / (2.0 * PI.powi(2)) * t_nu.powi(3);
            assert!(
                relative_error(physical.pair_moments.energy_density_mev4, expected_energy,)
                    < 4.0e-10
            );
            assert!(
                relative_error(physical.pair_moments.number_density_mev3, expected_number,)
                    < 2.0e-9
            );
            eprintln!(
                "F10A {kind:?}: N={:.17e} elapsed={:.17e}s Tnu={t_nu:.17e} steps={}/{}",
                result.t, result.y[ELAPSED_SECONDS_INDEX], result.accepted, result.rejected,
            );
            endpoints.push((result.t, result.y[ELAPSED_SECONDS_INDEX]));
        }
        assert!((endpoints[0].0 - endpoints[1].0).abs() < 5.0e-7);
        assert!(relative_error(endpoints[0].1, endpoints[1].1) < 2.0e-7);
    }
}

// This evidence exporter is deliberately test-only and crate-private.  It is
// compiled into a separately hash-reviewed test binary and has no production
// dispatch surface.  Its sole entry point stays ignored so ordinary test runs
// cannot create physical comparison output.
#[cfg(test)]
mod f10_independent_evidence {
    use super::*;
    use crate::electron_hm::{G_F_MEV_MINUS_2, SIN2_THETA_W};
    use crate::ode::{OdeConfig, OdeResult, OdeSystem, SolverKind, TerminalEvent, solve};
    use serde_json::{Map, Value, json};
    use std::cell::{Cell, RefCell};
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Read, Write};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    const SCHEMA_ID: &str = "rabbit.f10.independent.raw.v1";
    const GRID_HASH_SCHEMA_ID: &str = "rabbit.f10.nonphysical-grid-hashes.v1";
    const GRID_DOMAIN: &[u8] = b"rabbit.f10.grid.v1\0";
    const EXP48_GRID_SHA256: &str =
        "6e34b889e1c3f39e202ec143f7975b10f4d315283a540f4db407945e9e2a86a6";
    const EXP64_GRID_SHA256: &str =
        "204f093affff0c9b8c406b644126b3cf2cff7b187cb96c2c21a7b15ca265ca4c";
    const TAIL_INTERVAL_LABEL: &str = "tail_correction_interval_around_primary_finite_estimate";
    const GK_ABSCISSAE: [f64; 8] = [
        0.9914553711208126,
        0.9491079123427585,
        0.8648644233597691,
        0.7415311855993945,
        0.5860872354676911,
        0.4058451513773972,
        0.2077849550078985,
        0.0,
    ];
    const GK_WEIGHTS: [f64; 8] = [
        0.02293532201052922,
        0.06309209262997855,
        0.1047900103222502,
        0.1406532597155259,
        0.1690047266392679,
        0.1903505780647854,
        0.2044329400752989,
        0.2094821410847278,
    ];
    const G_WEIGHTS: [f64; 4] = [
        0.1294849661688697,
        0.2797053914892767,
        0.3818300505051189,
        0.4179591836734694,
    ];
    const RUN_CELL_ID: &str = "__run__";
    const GRID_CELL_ID: &str = "__grid__";
    const CONTRACT_HASHES: [&str; 5] = [
        "fb9d0308633319557c8b78c7f2f9305a6ff7fc2c5edcc870e5aff8ca314a264a",
        "c34fc9ec5e2239cb3898e54be302a195a0e4b884edbef1be167bbaa1ec98e38a",
        "213d5db53e7b386f8c972b555ef8be3745d615b48e94d8406f5a42210bc49a7e",
        "8c49dd6af6d06ebebaa5ee9ea773a379f2e5af2d189f478ab156cf1bfd70eada",
        "6c4519976b51d496a2411918d5fb781ac0447d24e87474be3b4f4a67f607f2e9",
    ];
    const CHECKPOINT_TARGETS: [f64; 14] = [
        10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005,
    ];
    const REPLAY_TARGETS: [f64; 12] =
        [5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01];
    const RABBIT_CELL_IDS: [&str; 5] = [
        "R-EXP48-COARSE",
        "R-EXP48-NOMINAL",
        "R-EXP48-TIGHT",
        "R-EXP64-NOMINAL",
        "R-EXP64-TIGHT",
    ];
    const FORT_CELL_IDS: [&str; 12] = [
        "fort-gl30-coarse",
        "fort-gl30-nominal",
        "fort-gl30-tight",
        "fort-gl40-coarse",
        "fort-gl40-nominal",
        "fort-gl40-tight",
        "fort-gl50-coarse",
        "fort-gl50-nominal",
        "fort-gl50-tight",
        "fort-nc100-coarse",
        "fort-nc100-nominal",
        "fort-nc100-tight",
    ];

    unsafe extern "C" {
        fn fegetround() -> std::os::raw::c_int;
    }

    fn fail<T>(message: impl Into<String>) -> Result<T, String> {
        Err(message.into())
    }

    fn assert_binary64_environment() -> Result<Value, String> {
        // FE_TONEAREST is zero on the frozen glibc/x86_64 target.  This probe
        // is read-only: the exporter never changes a process FP mode.
        if unsafe { fegetround() } != 0 {
            return fail("active rounding mode is not nearest");
        }
        let minimum = std::hint::black_box(f64::from_bits(1));
        if 0.0_f64.next_up().to_bits() != 1
            || (minimum * std::hint::black_box(1.0)).to_bits() != 1
            || (std::hint::black_box(0.0) + minimum).to_bits() != 1
            || (-0.0_f64).to_bits() != 0x8000_0000_0000_0000
            || f64::EPSILON.to_bits() != (2.0_f64.powi(-52)).to_bits()
        {
            return fail("binary64 gradual-subnormal or signed-zero probe failed");
        }
        Ok(json!({
            "format": "IEEE-754-binary64",
            "rounding": "nearest-ties-to-even",
            "minimum_positive_subnormal_bits": "0x0000000000000001",
            "signed_negative_zero_bits": "0x8000000000000000",
            "gradual_subnormals": true,
            "ftz_daz_observed": false
        }))
    }

    fn required_env(name: &str) -> Result<String, String> {
        std::env::var(name).map_err(|_| format!("missing required environment variable {name}"))
    }

    fn absolute_existing_env(name: &str) -> Result<PathBuf, String> {
        let value = PathBuf::from(required_env(name)?);
        if !value.is_absolute() {
            return Err(format!("{name} must be absolute"));
        }
        value
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize {name}: {error}"))
    }

    fn validate_sha256(value: &str) -> Result<(), String> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return fail(format!("invalid lowercase SHA-256 {value:?}"));
        }
        Ok(())
    }

    fn sha256_bytes_via_locked_tool(tool: &Path, bytes: &[u8]) -> Result<String, String> {
        let mut child = Command::new(tool)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("cannot execute locked sha256sum: {error}"))?;
        child
            .stdin
            .take()
            .ok_or("locked sha256sum stdin unavailable")?
            .write_all(bytes)
            .map_err(|error| format!("cannot write locked sha256sum stdin: {error}"))?;
        let output = child
            .wait_with_output()
            .map_err(|error| format!("cannot wait for locked sha256sum: {error}"))?;
        if !output.status.success() || !output.stderr.is_empty() {
            return fail("locked sha256sum failed or wrote stderr");
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|_| "locked sha256sum stdout is not UTF-8".to_string())?;
        let mut fields = stdout.split_whitespace();
        let digest = fields
            .next()
            .ok_or("locked sha256sum returned no digest")?
            .to_string();
        if fields.next() != Some("-") || fields.next().is_some() {
            return fail("locked sha256sum stdout shape mismatch");
        }
        validate_sha256(&digest)?;
        Ok(digest)
    }

    fn read_bytes(path: &Path) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|mut file| file.read_to_end(&mut bytes))
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        Ok(bytes)
    }

    fn sha256_file(tool: &Path, path: &Path) -> Result<String, String> {
        sha256_bytes_via_locked_tool(tool, &read_bytes(path)?)
    }

    fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
        value
            .as_object()
            .ok_or_else(|| format!("{label} must be a JSON object"))
    }

    fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
        value
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing string field {key}"))
    }

    fn bool_field(value: &Value, key: &str) -> Result<bool, String> {
        value
            .get(key)
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("missing boolean field {key}"))
    }

    fn string_array(value: &Value, key: &str) -> Result<Vec<String>, String> {
        value
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("missing array field {key}"))?
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("{key} contains a non-string"))
            })
            .collect()
    }

    struct OutputAuthority {
        approval_path: PathBuf,
        approval_sha256: String,
        approval: Value,
        fort_manifest_path: PathBuf,
        fort_manifest_sha256: String,
        fort_complete_bundle_sha256: String,
        sha256sum_path: PathBuf,
        sha256sum_binary_sha256: String,
        rust_source_sha256: String,
        rust_binary_sha256: String,
        schema_path: PathBuf,
        schema_sha256: String,
        orchestrator_path: PathBuf,
        orchestrator_sha256: String,
        raw_output_path: PathBuf,
        run_id: String,
    }

    fn load_output_authority() -> Result<OutputAuthority, String> {
        let sha256sum_path = absolute_existing_env("F10_SHA256SUM_PATH")?;
        let expected_tool_sha = required_env("F10_SHA256SUM_SHA256")?;
        validate_sha256(&expected_tool_sha)?;
        let sha256sum_binary_sha256 = sha256_file(&sha256sum_path, &sha256sum_path)?;
        if sha256sum_binary_sha256 != expected_tool_sha {
            return fail("locked sha256sum binary hash mismatch");
        }

        let approval_path = absolute_existing_env("F10_STAGE2_APPROVAL_PATH")?;
        let expected_approval_sha = required_env("F10_STAGE2_APPROVAL_SHA256")?;
        validate_sha256(&expected_approval_sha)?;
        let approval_bytes = read_bytes(&approval_path)?;
        let approval_sha256 = sha256_bytes_via_locked_tool(&sha256sum_path, &approval_bytes)?;
        if approval_sha256 != expected_approval_sha {
            return fail("stage-2 approval hash mismatch");
        }
        let approval: Value = serde_json::from_slice(&approval_bytes)
            .map_err(|error| format!("invalid stage-2 approval JSON: {error}"))?;
        object(&approval, "stage-2 approval")?;
        if !bool_field(&approval, "permit_output")? {
            return fail("stage-2 approval does not permit output");
        }
        if string_field(&approval, "solver_mode")? != "BDF_ONLY" {
            return fail("stage-2 approval solver mode is not BDF_ONLY");
        }
        for (index, expected) in CONTRACT_HASHES.iter().enumerate() {
            let key = format!("contract_sha256_v{}", index + 4);
            if string_field(&approval, &key)? != *expected {
                return fail(format!("stage-2 contract hash mismatch for {key}"));
            }
        }
        if string_array(&approval, "rabbit_cell_ids")?
            != RABBIT_CELL_IDS.map(str::to_string).to_vec()
        {
            return fail("stage-2 RABBIT cell order mismatch");
        }

        let source_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/isotropic_boltzmann.rs")
            .canonicalize()
            .map_err(|error| format!("cannot canonicalize Rust source: {error}"))?;
        let binary_path = std::env::current_exe()
            .and_then(|path| path.canonicalize())
            .map_err(|error| format!("cannot resolve current test binary: {error}"))?;
        let rust_source_sha256 = sha256_file(&sha256sum_path, &source_path)?;
        let rust_binary_sha256 = sha256_file(&sha256sum_path, &binary_path)?;
        if string_field(&approval, "rust_source_sha256")? != rust_source_sha256
            || string_field(&approval, "rust_binary_sha256")? != rust_binary_sha256
        {
            return fail("stage-2 Rust source or binary hash mismatch");
        }

        let schema_path = absolute_existing_env("F10_SCHEMA_PATH")?;
        let orchestrator_path = absolute_existing_env("F10_ORCHESTRATOR_PATH")?;
        let schema_sha256 = sha256_file(&sha256sum_path, &schema_path)?;
        let orchestrator_sha256 = sha256_file(&sha256sum_path, &orchestrator_path)?;
        if string_field(&approval, "schema_sha256")? != schema_sha256
            || string_field(&approval, "orchestrator_sha256")? != orchestrator_sha256
            || string_field(&approval, "sha256sum_path")? != sha256sum_path.to_string_lossy()
            || string_field(&approval, "sha256sum_binary_sha256")? != sha256sum_binary_sha256
        {
            return fail("stage-2 schema/orchestrator/hash-tool authority mismatch");
        }

        let fort_manifest_path = absolute_existing_env("F10_FORT_BUNDLE_MANIFEST_PATH")?;
        if fort_manifest_path == approval_path {
            return fail("Fort bundle manifest must be distinct from stage-2 approval");
        }
        let expected_fort_manifest_sha = required_env("F10_FORT_BUNDLE_MANIFEST_SHA256")?;
        validate_sha256(&expected_fort_manifest_sha)?;
        let fort_bytes = read_bytes(&fort_manifest_path)?;
        let fort_manifest_sha256 = sha256_bytes_via_locked_tool(&sha256sum_path, &fort_bytes)?;
        if fort_manifest_sha256 != expected_fort_manifest_sha {
            return fail("frozen Fort bundle manifest hash mismatch");
        }
        let fort: Value = serde_json::from_slice(&fort_bytes)
            .map_err(|error| format!("invalid Fort bundle manifest JSON: {error}"))?;
        if !bool_field(&fort, "immutable")?
            || !string_array(&fort, "omitted_cells")?.is_empty()
            || !string_array(&fort, "hidden_cells")?.is_empty()
        {
            return fail("Fort bundle is not complete and immutable");
        }
        let cells = fort
            .get("cells")
            .and_then(Value::as_array)
            .ok_or("Fort bundle cells missing")?;
        if cells.len() != FORT_CELL_IDS.len() {
            return fail("Fort bundle cell count mismatch");
        }
        for (record, expected_id) in cells.iter().zip(FORT_CELL_IDS) {
            if string_field(record, "cell_id")? != expected_id
                || string_field(record, "status")? != "success"
            {
                return fail("Fort bundle cell order or success status mismatch");
            }
        }
        let fort_complete_bundle_sha256 =
            string_field(&fort, "complete_bundle_sha256")?.to_string();
        validate_sha256(&fort_complete_bundle_sha256)?;

        let raw_output_path = PathBuf::from(required_env("F10_RABBIT_RAW_OUTPUT_PATH")?);
        if !raw_output_path.is_absolute() || raw_output_path.exists() {
            return fail("RABBIT raw output path must be absolute and nonexistent");
        }
        if string_field(&approval, "rabbit_raw_output_path")? != raw_output_path.to_string_lossy() {
            return fail("stage-2 RABBIT raw output path mismatch");
        }
        let run_id = required_env("F10_RUN_ID")?;
        if string_field(&approval, "run_id")? != run_id {
            return fail("stage-2 run ID mismatch");
        }

        Ok(OutputAuthority {
            approval_path,
            approval_sha256,
            approval,
            fort_manifest_path,
            fort_manifest_sha256,
            fort_complete_bundle_sha256,
            sha256sum_path,
            sha256sum_binary_sha256,
            rust_source_sha256,
            rust_binary_sha256,
            schema_path,
            schema_sha256,
            orchestrator_path,
            orchestrator_sha256,
            raw_output_path,
            run_id,
        })
    }

    #[derive(Clone)]
    struct EvidenceGrid {
        id: &'static str,
        order: usize,
        grid: ComovingMomentumGrid,
        y_nodes: Vec<f64>,
        plain_dy_weights: Vec<f64>,
        canonical_sha256: String,
    }

    fn canonical_grid_bytes(order: usize, y_nodes: &[f64], weights: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(GRID_DOMAIN.len() + 32 + 16 * y_nodes.len());
        bytes.extend_from_slice(GRID_DOMAIN);
        bytes.extend_from_slice(&(order as u64).to_be_bytes());
        bytes.extend_from_slice(&3.0_f64.to_bits().to_be_bytes());
        bytes.extend_from_slice(&10.0_f64.to_bits().to_be_bytes());
        bytes.extend_from_slice(&(y_nodes.len() as u64).to_be_bytes());
        for (&node, &weight) in y_nodes.iter().zip(weights) {
            bytes.extend_from_slice(&node.to_bits().to_be_bytes());
            bytes.extend_from_slice(&weight.to_bits().to_be_bytes());
        }
        bytes
    }

    fn build_evidence_grid(order: usize, sha256sum_path: &Path) -> Result<EvidenceGrid, String> {
        if !matches!(order, 48 | 64) {
            return fail("evidence grid order is not EXP48 or EXP64");
        }
        let rule = gauss_legendre_exponential_plain_rule(order, 3.0)
            .map_err(|error| format!("cannot construct evidence grid: {error}"))?;
        if rule.len() != order {
            return fail("evidence grid order mismatch");
        }
        let (y_nodes, plain_dy_weights): (Vec<_>, Vec<_>) = rule.into_iter().unzip();
        if y_nodes
            .iter()
            .zip(&plain_dy_weights)
            .any(|(&node, &weight)| {
                !node.is_finite() || node <= 0.0 || !weight.is_finite() || weight <= 0.0
            })
            || !y_nodes.windows(2).all(|pair| pair[0] < pair[1])
        {
            return fail("evidence grid is not finite, positive, and monotone");
        }
        let grid = ComovingMomentumGrid {
            nodes_mev: y_nodes.iter().map(|value| 10.0 * value).collect(),
            weights_mev: plain_dy_weights.iter().map(|value| 10.0 * value).collect(),
        };
        if order == 48 {
            let selected = ComovingMomentumGrid::selected_decoupling(10.0)
                .map_err(|error| format!("cannot construct selected grid: {error}"))?;
            if grid.nodes_mev.len() != selected.nodes_mev.len()
                || grid
                    .nodes_mev
                    .iter()
                    .zip(&selected.nodes_mev)
                    .any(|(left, right)| left.to_bits() != right.to_bits())
                || grid
                    .weights_mev
                    .iter()
                    .zip(&selected.weights_mev)
                    .any(|(left, right)| left.to_bits() != right.to_bits())
            {
                return fail("private EXP48 grid differs bitwise from selected_decoupling(10)");
            }
        }
        let canonical_sha256 = sha256_bytes_via_locked_tool(
            sha256sum_path,
            &canonical_grid_bytes(order, &y_nodes, &plain_dy_weights),
        )?;
        let id = if order == 48 { "EXP48" } else { "EXP64" };
        Ok(EvidenceGrid {
            id,
            order,
            grid,
            y_nodes,
            plain_dy_weights,
            canonical_sha256,
        })
    }

    fn expected_grid_sha256(order: usize) -> Result<&'static str, String> {
        match order {
            48 => Ok(EXP48_GRID_SHA256),
            64 => Ok(EXP64_GRID_SHA256),
            _ => fail("evidence grid order is not EXP48 or EXP64"),
        }
    }

    fn evidence_grid(order: usize, authority: &OutputAuthority) -> Result<EvidenceGrid, String> {
        let grid = build_evidence_grid(order, &authority.sha256sum_path)?;
        if grid.canonical_sha256 != expected_grid_sha256(order)? {
            return fail(format!("frozen {} canonical grid hash mismatch", grid.id));
        }
        let grid_hashes = authority
            .approval
            .get("grid_sha256")
            .and_then(Value::as_object)
            .ok_or("stage-2 grid_sha256 object missing")?;
        if grid_hashes.get(grid.id).and_then(Value::as_str) != Some(&grid.canonical_sha256) {
            return fail(format!("stage-2 {} canonical grid hash mismatch", grid.id));
        }
        Ok(grid)
    }

    fn nonphysical_grid_hash_artifact(sha256sum_path: &Path) -> Result<Value, String> {
        let exp48 = build_evidence_grid(48, sha256sum_path)?;
        let exp64 = build_evidence_grid(64, sha256sum_path)?;
        Ok(json!({
            "schema_id": GRID_HASH_SCHEMA_ID,
            "hash_domain": String::from_utf8(GRID_DOMAIN.to_vec())
                .map_err(|_| "grid hash domain is not UTF-8")?,
            "scale_parameter": 3.0,
            "reference_temperature_mev": 10.0,
            "grid_sha256": {
                "EXP48": exp48.canonical_sha256,
                "EXP64": exp64.canonical_sha256
            }
        }))
    }

    #[derive(Clone, Copy)]
    struct CellSpec {
        id: &'static str,
        grid_order: usize,
        tolerance_id: &'static str,
        rtol: f64,
        state_atol: f64,
        tgamma_atol_mev: f64,
        elapsed_atol_seconds: f64,
    }

    fn five_cells() -> [CellSpec; 5] {
        [
            CellSpec {
                id: RABBIT_CELL_IDS[0],
                grid_order: 48,
                tolerance_id: "coarse",
                rtol: 1.0e-6,
                state_atol: 1.0e-9,
                tgamma_atol_mev: 1.0e-8,
                elapsed_atol_seconds: 1.0e-4,
            },
            CellSpec {
                id: RABBIT_CELL_IDS[1],
                grid_order: 48,
                tolerance_id: "nominal",
                rtol: 2.0e-7,
                state_atol: 2.0e-10,
                tgamma_atol_mev: 2.0e-9,
                elapsed_atol_seconds: 2.0e-5,
            },
            CellSpec {
                id: RABBIT_CELL_IDS[2],
                grid_order: 48,
                tolerance_id: "tight",
                rtol: 5.0e-8,
                state_atol: 5.0e-11,
                tgamma_atol_mev: 5.0e-10,
                elapsed_atol_seconds: 5.0e-6,
            },
            CellSpec {
                id: RABBIT_CELL_IDS[3],
                grid_order: 64,
                tolerance_id: "nominal",
                rtol: 2.0e-7,
                state_atol: 2.0e-10,
                tgamma_atol_mev: 2.0e-9,
                elapsed_atol_seconds: 2.0e-5,
            },
            CellSpec {
                id: RABBIT_CELL_IDS[4],
                grid_order: 64,
                tolerance_id: "tight",
                rtol: 5.0e-8,
                state_atol: 5.0e-11,
                tgamma_atol_mev: 5.0e-10,
                elapsed_atol_seconds: 5.0e-6,
            },
        ]
    }

    fn system_for_grid(grid: &EvidenceGrid) -> IsotropicBoltzmannFlrwSystem {
        IsotropicBoltzmannFlrwSystem {
            grid: grid.grid.clone(),
            reference_temperature_mev: 10.0,
            electron_rule: ElectronSpectralRule {
                electron_radial_order: 6,
                angular_order: 4,
            },
            neutrino_self_rule: NeutrinoSelfSpectralRule { angular_order: 12 },
        }
    }

    fn config_for_cell(system: &IsotropicBoltzmannFlrwSystem, cell: CellSpec) -> OdeConfig {
        let mut atol = vec![cell.state_atol; system.dimension()];
        atol[T_GAMMA_INDEX] = cell.tgamma_atol_mev;
        atol[ELAPSED_SECONDS_INDEX] = cell.elapsed_atol_seconds;
        OdeConfig {
            rtol: cell.rtol,
            atol,
            h_init: 1.0e-5,
            h_min: 1.0e-12,
            h_max: 0.04,
            max_attempts: 100_000,
        }
    }

    struct CountingSystem<'a, S> {
        inner: &'a S,
        rhs_calls: Cell<u64>,
        jacobian_calls: Cell<u64>,
    }

    impl<'a, S> CountingSystem<'a, S> {
        fn new(inner: &'a S) -> Self {
            Self {
                inner,
                rhs_calls: Cell::new(0),
                jacobian_calls: Cell::new(0),
            }
        }
    }

    impl<S: OdeSystem> OdeSystem for CountingSystem<'_, S> {
        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn state_is_valid(&self, state: &[f64]) -> bool {
            self.inner.state_is_valid(state)
        }

        fn rhs(&self, t: f64, state: &[f64], out: &mut [f64]) {
            self.rhs_calls.set(self.rhs_calls.get() + 1);
            self.inner.rhs(t, state, out);
        }

        fn jacobian(&self, t: f64, state: &[f64], out: &mut [f64]) {
            self.jacobian_calls.set(self.jacobian_calls.get() + 1);
            self.inner.jacobian(t, state, out);
        }

        fn dfdt(&self, t: f64, state: &[f64], out: &mut [f64]) {
            self.inner.dfdt(t, state, out);
        }
    }

    #[derive(Clone)]
    struct EventInvocation {
        invocation_index: usize,
        n: f64,
        state: Vec<f64>,
        event_value_mev: f64,
        rhs_calls: u64,
        jacobian_calls: u64,
        state_valid: bool,
    }

    struct SolveEvidence {
        result: OdeResult,
        trace: Vec<EventInvocation>,
        rhs_calls: u64,
        jacobian_calls: u64,
    }

    fn solve_target(
        system: &IsotropicBoltzmannFlrwSystem,
        initial: &[f64],
        config: &OdeConfig,
        target: f64,
    ) -> Result<SolveEvidence, String> {
        if !target.is_finite() || target <= 0.0 || target >= 10.0 {
            return fail("invalid replay target");
        }
        let counting = CountingSystem::new(system);
        let trace = RefCell::new(Vec::new());
        let event_fn = |n: f64, state: &[f64]| {
            let event_value = state[T_GAMMA_INDEX] - target;
            let mut records = trace.borrow_mut();
            let invocation_index = records.len();
            records.push(EventInvocation {
                invocation_index,
                n,
                state: state.to_vec(),
                event_value_mev: event_value,
                rhs_calls: counting.rhs_calls.get(),
                jacobian_calls: counting.jacobian_calls.get(),
                state_valid: counting.state_is_valid(state),
            });
            event_value
        };
        let event = TerminalEvent {
            value: &event_fn,
            direction: -1,
        };
        let result = solve(
            SolverKind::Bdf,
            &counting,
            (0.0, 10.0),
            initial,
            config,
            Some(&event),
        );
        let trace = trace.into_inner();
        let prefix_len = result
            .accepted
            .checked_add(1)
            .ok_or("accepted-prefix length overflow")?;
        if result.failure.is_some()
            || !result.event_reached
            || !system.state_is_valid(&result.y)
            || result.accepted + result.rejected > config.max_attempts
            || trace.len() < prefix_len
            || trace.first().is_none_or(|record| {
                record.invocation_index != 0
                    || record.n.to_bits() != 0.0_f64.to_bits()
                    || record
                        .state
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                        != initial
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>()
            })
            || trace[..prefix_len]
                .iter()
                .any(|record| !record.n.is_finite() || !record.state_valid)
            || trace[..prefix_len]
                .windows(2)
                .any(|pair| pair[1].n <= pair[0].n)
            || trace[..prefix_len].windows(2).any(|pair| {
                pair[1].rhs_calls < pair[0].rhs_calls
                    || pair[1].jacobian_calls < pair[0].jacobian_calls
            })
            || counting.jacobian_calls.get() != result.jacobian_evaluations as u64
        {
            return fail(format!(
                "BDF accepted-prefix invariant failed at target {target}"
            ));
        }
        let event_cap = deterministic_max(&[1.0e-12, 64.0 * f64::EPSILON * target])?;
        if (result.y[T_GAMMA_INDEX] - target).abs() > event_cap {
            return fail(format!("BDF event residual exceeds cap at target {target}"));
        }
        Ok(SolveEvidence {
            result,
            trace,
            rhs_calls: counting.rhs_calls.get(),
            jacobian_calls: counting.jacobian_calls.get(),
        })
    }

    fn compare_prefix(terminal: &SolveEvidence, replay: &SolveEvidence) -> Result<usize, String> {
        let replay_len = replay.result.accepted + 1;
        let terminal_len = terminal.result.accepted + 1;
        if replay_len > terminal_len {
            return fail("replay prefix is longer than terminal authority prefix");
        }
        for (index, (left, right)) in replay.trace[..replay_len]
            .iter()
            .zip(&terminal.trace[..replay_len])
            .enumerate()
        {
            if left.n.to_bits() != right.n.to_bits()
                || left.state.len() != right.state.len()
                || left
                    .state
                    .iter()
                    .zip(&right.state)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                || left.rhs_calls != right.rhs_calls
                || left.jacobian_calls != right.jacobian_calls
                || left.state_valid != right.state_valid
            {
                return fail(format!(
                    "bitwise accepted-prefix mismatch at invocation {index}"
                ));
            }
        }
        Ok(replay_len)
    }

    fn pairwise(values: &[f64]) -> Result<f64, String> {
        if values.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite deterministic pairwise input".into());
        }
        fn inner(values: &[f64]) -> f64 {
            match values.len() {
                0 => 0.0,
                1 => values[0],
                len => {
                    let middle = len / 2;
                    inner(&values[..middle]) + inner(&values[middle..])
                }
            }
        }
        Ok(inner(values))
    }

    fn deterministic_max(values: &[f64]) -> Result<f64, String> {
        let mut retained = 0.0;
        for &candidate in values {
            if !candidate.is_finite() || candidate < 0.0 {
                return Err("invalid deterministic maximum candidate".into());
            }
            if candidate > retained {
                retained = candidate;
            }
        }
        Ok(retained)
    }

    fn f64_bits(value: f64) -> String {
        format!("0x{:016x}", value.to_bits())
    }

    fn f64_bits_vec(values: &[f64]) -> Vec<String> {
        values.iter().map(|&value| f64_bits(value)).collect()
    }

    fn pairwise_weighted(
        weights: &[f64],
        nodes: &[f64],
        field: &[f64],
        power: i32,
    ) -> Result<(f64, f64), String> {
        if weights.len() != nodes.len() || nodes.len() != field.len() {
            return Err("weighted moment shape mismatch".into());
        }
        let terms = weights
            .iter()
            .zip(nodes)
            .zip(field)
            .map(|((&weight, &node), &value)| weight * node.powi(power) * value)
            .collect::<Vec<_>>();
        let absolute = terms.iter().map(|value| value.abs()).collect::<Vec<_>>();
        Ok((pairwise(&terms)?, pairwise(&absolute)?))
    }

    fn ulp(value: f64) -> f64 {
        if value == 0.0 {
            f64::from_bits(1)
        } else if value > 0.0 {
            value.next_up() - value
        } else {
            value - value.next_down()
        }
    }

    fn guarded_ratio(numerator: f64, denominator: f64, scale: f64) -> Result<f64, String> {
        if !numerator.is_finite()
            || !denominator.is_finite()
            || !scale.is_finite()
            || numerator < 0.0
            || denominator < 0.0
            || scale <= 0.0
        {
            return Err("invalid guarded-ratio input".into());
        }
        let zero_guard = 128.0 * f64::EPSILON * scale;
        if denominator > zero_guard {
            Ok(numerator / denominator)
        } else if numerator <= zero_guard {
            Ok(0.0)
        } else {
            Err("guarded ratio has active numerator on a null denominator".into())
        }
    }

    #[derive(Clone, Copy)]
    struct GkPanel {
        kronrod: f64,
        gauss: f64,
        resabs: f64,
        embedded: f64,
    }

    #[derive(Default)]
    struct Neumaier {
        sum: f64,
        correction: f64,
    }

    impl Neumaier {
        fn add(&mut self, value: f64) -> Result<(), String> {
            if !value.is_finite() {
                return fail("nonfinite Neumaier input");
            }
            let updated = self.sum + value;
            if self.sum.abs() >= value.abs() {
                self.correction += (self.sum - updated) + value;
            } else {
                self.correction += (value - updated) + self.sum;
            }
            self.sum = updated;
            if !self.sum.is_finite() || !self.correction.is_finite() {
                return fail("nonfinite Neumaier state");
            }
            Ok(())
        }

        fn total(&self) -> Result<f64, String> {
            let total = self.sum + self.correction;
            if !total.is_finite() {
                return fail("nonfinite Neumaier total");
            }
            Ok(total)
        }
    }

    struct GkRun {
        estimate: f64,
        embedded: f64,
        calls: u64,
        leaves: u64,
        maximum_depth: usize,
    }

    struct GkState {
        calls: u64,
        leaves: u64,
        maximum_depth: usize,
        estimate: Neumaier,
        embedded: Neumaier,
    }

    fn gk_panel<F>(
        integrand: &mut F,
        left: f64,
        right: f64,
        calls: &mut u64,
        call_cap: u64,
    ) -> Result<GkPanel, String>
    where
        F: FnMut(f64) -> Result<f64, String>,
    {
        if !left.is_finite() || !right.is_finite() || right < left {
            return fail("invalid GK panel bounds");
        }
        if right.to_bits() == left.to_bits() {
            return Ok(GkPanel {
                kronrod: 0.0,
                gauss: 0.0,
                resabs: 0.0,
                embedded: 0.0,
            });
        }
        if call_cap.saturating_sub(*calls) < 15 {
            return fail("GK call cap exhausted before panel evaluation");
        }
        let midpoint = left + 0.5 * (right - left);
        let half_width = 0.5 * (right - left);
        if midpoint == left || midpoint == right || !half_width.is_finite() || half_width <= 0.0 {
            return fail("GK panel midpoint is not representable");
        }
        let centre = integrand(midpoint)?;
        let mut pairs = [(0.0, 0.0); 7];
        for index in 0..7 {
            pairs[index].0 = integrand(midpoint - half_width * GK_ABSCISSAE[index])?;
            pairs[index].1 = integrand(midpoint + half_width * GK_ABSCISSAE[index])?;
        }
        *calls += 15;
        if !centre.is_finite()
            || pairs.iter().any(|&(left_value, right_value)| {
                !left_value.is_finite() || !right_value.is_finite()
            })
        {
            return fail("nonfinite GK integrand value");
        }

        let mut kronrod_terms = Vec::with_capacity(8);
        let mut resabs_terms = Vec::with_capacity(8);
        let mut gauss_terms = Vec::with_capacity(4);
        for index in 0..7 {
            let pair = pairs[index].0 + pairs[index].1;
            let absolute_pair = pairs[index].0.abs() + pairs[index].1.abs();
            kronrod_terms.push(GK_WEIGHTS[index] * pair);
            resabs_terms.push(GK_WEIGHTS[index] * absolute_pair);
            match index {
                1 => gauss_terms.push(G_WEIGHTS[0] * pair),
                3 => gauss_terms.push(G_WEIGHTS[1] * pair),
                5 => gauss_terms.push(G_WEIGHTS[2] * pair),
                _ => {}
            }
        }
        kronrod_terms.push(GK_WEIGHTS[7] * centre);
        resabs_terms.push(GK_WEIGHTS[7] * centre.abs());
        gauss_terms.push(G_WEIGHTS[3] * centre);
        let kronrod = half_width * pairwise(&kronrod_terms)?;
        let gauss = half_width * pairwise(&gauss_terms)?;
        let resabs = half_width.abs() * pairwise(&resabs_terms)?;
        let embedded = (kronrod - gauss).abs();
        if !kronrod.is_finite()
            || !gauss.is_finite()
            || !resabs.is_finite()
            || !embedded.is_finite()
        {
            return fail("nonfinite GK panel result");
        }
        Ok(GkPanel {
            kronrod,
            gauss,
            resabs,
            embedded,
        })
    }

    // The adaptive Gauss-Kronrod recursion threads its full frozen control
    // surface explicitly; a parameter struct would only relabel the same
    // ten values (D-049 validation-only lint fix).
    #[allow(clippy::too_many_arguments)]
    fn gk_recurse<F>(
        integrand: &mut F,
        left: f64,
        right: f64,
        depth: usize,
        budget: f64,
        panel: GkPanel,
        mandatory_min_depth: usize,
        maximum_depth: usize,
        call_cap: u64,
        state: &mut GkState,
    ) -> Result<(), String>
    where
        F: FnMut(f64) -> Result<f64, String>,
    {
        if !budget.is_finite() || budget < 0.0 {
            return fail("invalid GK panel budget");
        }
        if depth >= mandatory_min_depth && panel.embedded <= budget {
            state.estimate.add(panel.kronrod)?;
            state.embedded.add(panel.embedded)?;
            state.leaves += 1;
            if depth > state.maximum_depth {
                state.maximum_depth = depth;
            }
            return Ok(());
        }
        if depth >= maximum_depth {
            return fail("GK maximum depth exhausted");
        }
        let midpoint = left + 0.5 * (right - left);
        if midpoint == left || midpoint == right {
            return fail("GK subdivision midpoint is not representable");
        }
        let child_budget = 0.5 * budget;
        let left_panel = gk_panel(integrand, left, midpoint, &mut state.calls, call_cap)?;
        gk_recurse(
            integrand,
            left,
            midpoint,
            depth + 1,
            child_budget,
            left_panel,
            mandatory_min_depth,
            maximum_depth,
            call_cap,
            state,
        )?;
        let right_panel = gk_panel(integrand, midpoint, right, &mut state.calls, call_cap)?;
        gk_recurse(
            integrand,
            midpoint,
            right,
            depth + 1,
            child_budget,
            right_panel,
            mandatory_min_depth,
            maximum_depth,
            call_cap,
            state,
        )
    }

    fn finish_gk_run<F>(
        integrand: &mut F,
        root: GkPanel,
        root_budget: f64,
        mandatory_min_depth: usize,
        maximum_depth: usize,
        call_cap: u64,
        root_calls: u64,
    ) -> Result<GkRun, String>
    where
        F: FnMut(f64) -> Result<f64, String>,
    {
        let mut state = GkState {
            calls: root_calls,
            leaves: 0,
            maximum_depth: 0,
            estimate: Neumaier::default(),
            embedded: Neumaier::default(),
        };
        gk_recurse(
            integrand,
            0.0,
            1.0,
            0,
            root_budget,
            root,
            mandatory_min_depth,
            maximum_depth,
            call_cap,
            &mut state,
        )?;
        if state.leaves == 0 || state.calls > call_cap {
            return fail("invalid completed GK work counters");
        }
        Ok(GkRun {
            estimate: state.estimate.total()?,
            embedded: state.embedded.total()?,
            calls: state.calls,
            leaves: state.leaves,
            maximum_depth: state.maximum_depth,
        })
    }

    fn em_finite_integrand(
        tgamma_mev: f64,
        cutoff_mev: f64,
        secondary: bool,
        s: f64,
    ) -> Result<f64, String> {
        if !tgamma_mev.is_finite()
            || tgamma_mev <= 0.0
            || !cutoff_mev.is_finite()
            || cutoff_mev <= 0.0
            || !s.is_finite()
            || !(0.0..=1.0).contains(&s)
        {
            return fail("invalid finite EM-transform input");
        }
        if s == 0.0 {
            return Ok(0.0);
        }
        let (momentum, jacobian) = if secondary {
            (cutoff_mev * s * s, 2.0 * cutoff_mev * s)
        } else {
            (cutoff_mev * s, cutoff_mev)
        };
        let energy = (momentum * momentum + ELECTRON_MASS_MEV * ELECTRON_MASS_MEV).sqrt();
        let boltzmann = (-energy / tgamma_mev).exp();
        let denominator = 1.0 + boltzmann;
        let f_one_minus_f = boltzmann / (denominator * denominator);
        let prefactor = 2.0 / (PI.powi(2) * tgamma_mev.powi(2));
        let value = prefactor * momentum.powi(2) * energy.powi(2) * f_one_minus_f * jacobian;
        if !value.is_finite() || value <= 0.0 {
            return fail("finite EM-transform interior value is not strictly positive");
        }
        Ok(value)
    }

    struct EmDerivativeCertificate {
        d1: f64,
        d2: f64,
        d_direct: f64,
        e1: f64,
        e2: f64,
        b_tail: f64,
        cutoff_mev: f64,
        q_mass_over_temperature: f64,
        r_cutoff_over_temperature: f64,
        certificate: f64,
        primary_calls: u64,
        secondary_calls: u64,
        primary_leaves: u64,
        secondary_leaves: u64,
        primary_depth: usize,
        secondary_depth: usize,
    }

    fn em_derivative_certificate(
        tgamma_mev: f64,
        d_native_mev3: f64,
    ) -> Result<EmDerivativeCertificate, String> {
        if !tgamma_mev.is_finite()
            || tgamma_mev <= 0.0
            || !d_native_mev3.is_finite()
            || d_native_mev3 <= 0.0
        {
            return fail("invalid native EM derivative certificate input");
        }
        let cutoff_mev = ELECTRON_MASS_MEV + 128.0 * tgamma_mev;
        let photon_derivative = 4.0 * PI.powi(2) * tgamma_mev.powi(3) / 15.0;
        let mut primary = |s| em_finite_integrand(tgamma_mev, cutoff_mev, false, s);
        let mut secondary = |s| em_finite_integrand(tgamma_mev, cutoff_mev, true, s);
        let mut primary_root_calls = 0;
        let primary_root = gk_panel(&mut primary, 0.0, 1.0, &mut primary_root_calls, 65_536)?;
        let mut secondary_root_calls = 0;
        let secondary_root = gk_panel(&mut secondary, 0.0, 1.0, &mut secondary_root_calls, 65_536)?;
        let dtrial_primary = photon_derivative + primary_root.kronrod;
        let dtrial_secondary = photon_derivative + secondary_root.kronrod;
        let scale0 = deterministic_max(&[
            d_native_mev3.abs(),
            dtrial_primary.abs(),
            dtrial_secondary.abs(),
            tgamma_mev.powi(3),
        ])?;
        let primary_budget =
            deterministic_max(&[5.0e-13 * scale0, 128.0 * f64::EPSILON * primary_root.resabs])?;
        let secondary_budget = deterministic_max(&[
            5.0e-13 * scale0,
            128.0 * f64::EPSILON * secondary_root.resabs,
        ])?;
        let primary_run = finish_gk_run(
            &mut primary,
            primary_root,
            primary_budget,
            0,
            24,
            65_536,
            primary_root_calls,
        )?;
        let secondary_run = finish_gk_run(
            &mut secondary,
            secondary_root,
            secondary_budget,
            1,
            24,
            65_536,
            secondary_root_calls,
        )?;
        let d1 = photon_derivative + primary_run.estimate;
        let d2 = photon_derivative + secondary_run.estimate;
        let b_tail = 1.0e-18 * tgamma_mev.powi(3);
        let d_direct = d1 + 0.5 * b_tail;
        let scale = deterministic_max(&[
            d_native_mev3.abs(),
            d1.abs(),
            d2.abs(),
            d_direct.abs(),
            tgamma_mev.powi(3),
        ])?;
        let dual_difference = (d1 - d2).abs();
        let certificate = pairwise(&[
            primary_run.embedded,
            secondary_run.embedded,
            dual_difference,
            b_tail,
        ])?;
        if d1 <= 0.0
            || d2 <= 0.0
            || d_direct <= 0.0
            || primary_run.embedded > 1.0e-12 * scale
            || secondary_run.embedded > 1.0e-12 * scale
            || dual_difference > 3.0e-12 * scale
            || b_tail > 1.0e-18 * scale
            || certificate > 5.0e-12 * scale
            || (d_native_mev3 - d_direct).abs() > 8.0 * certificate + 5.0e-10 * scale
        {
            return fail("V7 finite-plus-tail EM derivative certificate failed");
        }
        Ok(EmDerivativeCertificate {
            d1,
            d2,
            d_direct,
            e1: primary_run.embedded,
            e2: secondary_run.embedded,
            b_tail,
            cutoff_mev,
            q_mass_over_temperature: ELECTRON_MASS_MEV / tgamma_mev,
            r_cutoff_over_temperature: cutoff_mev / tgamma_mev,
            certificate,
            primary_calls: primary_run.calls,
            secondary_calls: secondary_run.calls,
            primary_leaves: primary_run.leaves,
            secondary_leaves: secondary_run.leaves,
            primary_depth: primary_run.maximum_depth,
            secondary_depth: secondary_run.maximum_depth,
        })
    }

    fn full_first_law_residual(
        rho_neutrino_mev4: f64,
        total_block_q_mev5: f64,
        h_mev: f64,
        d_rho_em_d_n_mev4: f64,
        rho_total_mev4: f64,
        pressure_total_mev4: f64,
    ) -> Result<f64, String> {
        if [
            rho_neutrino_mev4,
            total_block_q_mev5,
            h_mev,
            d_rho_em_d_n_mev4,
            rho_total_mev4,
            pressure_total_mev4,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || h_mev <= 0.0
            || rho_neutrino_mev4 <= 0.0
            || rho_total_mev4 <= 0.0
        {
            return fail("invalid full first-law input");
        }
        let d_rho_neutrino = -4.0 * rho_neutrino_mev4 + total_block_q_mev5 / h_mev;
        let enthalpy_term = 3.0 * (rho_total_mev4 + pressure_total_mev4);
        guarded_ratio(
            (d_rho_neutrino + d_rho_em_d_n_mev4 + enthalpy_term).abs(),
            (d_rho_neutrino + d_rho_em_d_n_mev4).abs() + enthalpy_term.abs(),
            rho_total_mev4,
        )
    }

    fn native_checkpoint(
        system: &IsotropicBoltzmannFlrwSystem,
        grid: &EvidenceGrid,
        checkpoint_index: usize,
        solve_role: &str,
        target: f64,
        n: f64,
        state: &[f64],
    ) -> Result<Value, String> {
        if !n.is_finite() || !system.state_is_valid(state) {
            return Err("invalid checkpoint state".into());
        }
        let event_cap = deterministic_max(&[1.0e-12, 64.0 * f64::EPSILON * target])?;
        let event_residual = (state[T_GAMMA_INDEX] - target).abs();
        if event_residual > event_cap {
            return fail("checkpoint event residual exceeds the frozen cap");
        }
        let physical = system
            .physical_state_impl(n, state, false)
            .map_err(|error| format!("checkpoint physical state failed: {error}"))?;
        let mut rhs = vec![0.0; system.dimension()];
        system.rhs(n, state, &mut rhs);
        if rhs.iter().any(|value| !value.is_finite()) {
            return fail("checkpoint RHS is nonfinite");
        }
        let electromagnetic = electromagnetic_eos(state[T_GAMMA_INDEX])
            .map_err(|_| "checkpoint electromagnetic EOS failed".to_string())?;
        let em_certificate =
            em_derivative_certificate(state[T_GAMMA_INDEX], electromagnetic.drho_dt)?;
        let spectra = [
            physical.electron_pair_occupation.clone(),
            physical.heavy_pair_occupation.clone(),
            physical.heavy_pair_occupation.clone(),
        ];
        let mut i2 = [0.0; 4];
        let mut i3 = [0.0; 4];
        let mut i2_abs = [0.0; 3];
        let mut i3_abs = [0.0; 3];
        for flavour in 0..3 {
            (i2[flavour], i2_abs[flavour]) =
                pairwise_weighted(&grid.plain_dy_weights, &grid.y_nodes, &spectra[flavour], 2)?;
            (i3[flavour], i3_abs[flavour]) =
                pairwise_weighted(&grid.plain_dy_weights, &grid.y_nodes, &spectra[flavour], 3)?;
        }
        i2[3] = pairwise(&i2[..3])?;
        i3[3] = pairwise(&i3[..3])?;
        let t_cm = 10.0 * (-n).exp();
        if t_cm.to_bits() != physical.t_cm_mev.to_bits() {
            return fail("checkpoint Tcm differs from the native physical path");
        }
        let phase = PI.powi(2).recip();
        let number = i2.map(|value| t_cm.powi(3) * phase * value);
        let energy = i3.map(|value| t_cm.powi(4) * phase * value);
        let native_number = [
            physical.electron_pair_moments.number_density_mev3,
            physical.heavy_pair_moments.number_density_mev3,
            physical.heavy_pair_moments.number_density_mev3,
            physical.electron_pair_moments.number_density_mev3
                + 2.0 * physical.heavy_pair_moments.number_density_mev3,
        ];
        let native_energy = [
            physical.electron_pair_moments.energy_density_mev4,
            physical.heavy_pair_moments.energy_density_mev4,
            physical.heavy_pair_moments.energy_density_mev4,
            physical.rho_neutrino_total_mev4,
        ];
        let gamma_m = (grid.order as f64 * f64::EPSILON) / (1.0 - grid.order as f64 * f64::EPSILON);
        let mut number_residual = [0.0; 4];
        let mut number_cap = [0.0; 4];
        let mut energy_residual = [0.0; 4];
        let mut energy_cap = [0.0; 4];
        for flavour in 0..4 {
            let number_sum_abs = if flavour < 3 {
                t_cm.powi(3) * phase * i2_abs[flavour]
            } else {
                pairwise(&number[..3])?
            };
            let energy_sum_abs = if flavour < 3 {
                t_cm.powi(4) * phase * i3_abs[flavour]
            } else {
                pairwise(&energy[..3])?
            };
            number_residual[flavour] = (native_number[flavour] - number[flavour]).abs();
            energy_residual[flavour] = (native_energy[flavour] - energy[flavour]).abs();
            let number_ulp_scale =
                deterministic_max(&[native_number[flavour].abs(), number[flavour].abs()])?;
            let energy_ulp_scale =
                deterministic_max(&[native_energy[flavour].abs(), energy[flavour].abs()])?;
            number_cap[flavour] = gamma_m * number_sum_abs + 16.0 * ulp(number_ulp_scale);
            energy_cap[flavour] = gamma_m * energy_sum_abs + 16.0 * ulp(energy_ulp_scale);
            if number_residual[flavour] > number_cap[flavour]
                || energy_residual[flavour] > energy_cap[flavour]
            {
                return fail("checkpoint native moment recomputation mismatch");
            }
        }

        let electron_action = [
            physical.electron_action.electron_pair_mev.clone(),
            physical.electron_action.heavy_pair_mev.clone(),
            physical.electron_action.heavy_pair_mev.clone(),
        ];
        let self_action = [
            physical.neutrino_self_action.electron_pair_mev.clone(),
            physical.neutrino_self_action.heavy_pair_mev.clone(),
            physical.neutrino_self_action.heavy_pair_mev.clone(),
        ];
        let total_action = std::array::from_fn::<_, 3, _>(|flavour| {
            electron_action[flavour]
                .iter()
                .zip(&self_action[flavour])
                .map(|(electron, self_value)| electron + self_value)
                .collect::<Vec<_>>()
        });
        let action_blocks = [&electron_action, &self_action, &total_action];
        let mut number_transfer = [[0.0; 4]; 3];
        let mut energy_transfer = [[0.0; 4]; 3];
        for (block, actions) in action_blocks.iter().enumerate() {
            for flavour in 0..3 {
                number_transfer[block][flavour] = t_cm.powi(3)
                    * phase
                    * pairwise_weighted(
                        &grid.plain_dy_weights,
                        &grid.y_nodes,
                        &actions[flavour],
                        2,
                    )?
                    .0;
                energy_transfer[block][flavour] = t_cm.powi(4)
                    * phase
                    * pairwise_weighted(
                        &grid.plain_dy_weights,
                        &grid.y_nodes,
                        &actions[flavour],
                        3,
                    )?
                    .0;
            }
            number_transfer[block][3] = pairwise(&number_transfer[block][..3])?;
            energy_transfer[block][3] = pairwise(&energy_transfer[block][..3])?;
        }
        let self_number_scale = (0..3)
            .map(|flavour| {
                pairwise_weighted(
                    &grid.plain_dy_weights,
                    &grid.y_nodes,
                    &self_action[flavour]
                        .iter()
                        .map(|value| value.abs())
                        .collect::<Vec<_>>(),
                    2,
                )
                .map(|result| t_cm.powi(3) * phase * result.0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let self_energy_scale = (0..3)
            .map(|flavour| {
                pairwise_weighted(
                    &grid.plain_dy_weights,
                    &grid.y_nodes,
                    &self_action[flavour]
                        .iter()
                        .map(|value| value.abs())
                        .collect::<Vec<_>>(),
                    3,
                )
                .map(|result| t_cm.powi(4) * phase * result.0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let self_number_residual = guarded_ratio(
            number_transfer[1][3].abs(),
            pairwise(&self_number_scale)?,
            physical.h_mev * number[3],
        )?;
        let self_energy_residual = guarded_ratio(
            energy_transfer[1][3].abs(),
            pairwise(&self_energy_scale)?,
            physical.h_mev * energy[3],
        )?;
        if self_number_residual > 1.0e-10 || self_energy_residual > 1.0e-10 {
            return fail("checkpoint self-collision conservation residual failed");
        }

        let rho_gamma = PI.powi(2) / 15.0 * state[T_GAMMA_INDEX].powi(4);
        let n_eff = 8.0 / 7.0 * (11.0_f64 / 4.0).powf(4.0 / 3.0) * energy[3] / rho_gamma;
        let native_n_eff =
            8.0 / 7.0 * (11.0_f64 / 4.0).powf(4.0 / 3.0) * physical.rho_neutrino_total_mev4
                / rho_gamma;
        let n_eff_residual = (n_eff - native_n_eff).abs();
        let n_eff_ulp_scale = deterministic_max(&[n_eff.abs(), native_n_eff.abs()])?;
        let n_eff_cap = gamma_m * n_eff.abs() + 32.0 * ulp(n_eff_ulp_scale);
        if n_eff_residual > n_eff_cap {
            return fail("checkpoint native N_eff recomputation mismatch");
        }
        let adiabatic_d_tgamma =
            -3.0 * (electromagnetic.rho + electromagnetic.pressure) / electromagnetic.drho_dt;
        let electron_d_tgamma = physical.d_tgamma_d_lna - adiabatic_d_tgamma;
        let d_rho_em = electromagnetic.drho_dt * physical.d_tgamma_d_lna;
        let first_law_residual = full_first_law_residual(
            physical.rho_neutrino_total_mev4,
            energy_transfer[2][3],
            physical.h_mev,
            d_rho_em,
            physical.rho_total_mev4,
            physical.pressure_total_mev4,
        )?;
        let q_em_electron = physical.h_mev * electromagnetic.drho_dt * electron_d_tgamma;
        let electron_exchange_residual = guarded_ratio(
            (q_em_electron + energy_transfer[0][3]).abs(),
            q_em_electron.abs() + energy_transfer[0][3].abs(),
            physical.h_mev * physical.rho_total_mev4,
        )?;
        if first_law_residual > 1.0e-8 || electron_exchange_residual > 1.0e-8 {
            return fail("checkpoint exchange or first-law residual failed");
        }

        let mut record = json!({
            "checkpoint_index": checkpoint_index,
            "solve_role": solve_role,
            "target_tgamma_mev": target,
            "event_residual_mev": event_residual,
            "event_residual_cap_mev": event_cap,
            "n": n,
            "n_bits": f64_bits(n),
            "x_me_over_tcm": ELECTRON_MASS_MEV / t_cm,
            "z_tgamma_over_tcm": state[T_GAMMA_INDEX] / t_cm,
            "t_cm_mev": t_cm,
            "t_gamma_mev": state[T_GAMMA_INDEX],
            "elapsed_seconds": state[ELAPSED_SECONDS_INDEX],
            "h_mev": physical.h_mev,
            "h_inverse_seconds": physical.h_inverse_seconds,
            "state": state,
            "state_bits": f64_bits_vec(state),
            "rhs_dstate_dN": rhs,
            "spectra_e_mu_tau": spectra,
            "native_moments_e_mu_tau_total": {
                "i2": i2,
                "i3": i3,
                "number_pair_mev3": number,
                "energy_pair_mev4": energy
            },
            "n_eff": n_eff,
            "actions_mev_e_mu_tau_total_by_block": {
                "electron": electron_action,
                "self": self_action,
                "total": total_action
            },
            "number_transfer_mev4_e_mu_tau_total_by_block": {
                "electron": number_transfer[0],
                "self": number_transfer[1],
                "total": number_transfer[2]
            },
            "energy_transfer_mev5_e_mu_tau_total_by_block": {
                "electron": energy_transfer[0],
                "self": energy_transfer[1],
                "total": energy_transfer[2]
            },
            "rho_em_mev4": electromagnetic.rho,
            "pressure_em_mev4": electromagnetic.pressure,
            "drho_em_dtgamma_native_mev3": electromagnetic.drho_dt,
            "rho_total_mev4": physical.rho_total_mev4,
            "pressure_total_mev4": physical.pressure_total_mev4,
            "d_tgamma_dN_full": physical.d_tgamma_d_lna,
            "d_tgamma_dN_electron": electron_d_tgamma,
            "native_recomputation_checks": {
                "number_residual": number_residual,
                "number_cap": number_cap,
                "energy_residual": energy_residual,
                "energy_cap": energy_cap,
                "n_eff_residual": n_eff_residual,
                "n_eff_cap": n_eff_cap,
                "self_number_residual": self_number_residual,
                "self_energy_residual": self_energy_residual,
                "electron_exchange_residual": electron_exchange_residual,
                "first_law_residual": first_law_residual
            },
            "mu_tau_construction": "e_mu_tau_equals_E_X_X_and_total_equals_E_plus_2X",
            "off_diagonal_slot_count": 0,
            "commutator_exact_zero": true
        });
        let record_object = record
            .as_object_mut()
            .ok_or("checkpoint JSON is not an object")?;
        for (key, value) in [
            ("em_D1", Value::from(em_certificate.d1)),
            ("em_D2", Value::from(em_certificate.d2)),
            ("em_D_direct", Value::from(em_certificate.d_direct)),
            ("em_E1", Value::from(em_certificate.e1)),
            ("em_E2", Value::from(em_certificate.e2)),
            ("em_B_tail", Value::from(em_certificate.b_tail)),
            ("em_cutoff_P_mev", Value::from(em_certificate.cutoff_mev)),
            (
                "em_q_mass_over_temperature",
                Value::from(em_certificate.q_mass_over_temperature),
            ),
            (
                "em_R_cutoff_over_temperature",
                Value::from(em_certificate.r_cutoff_over_temperature),
            ),
            (
                "tail_correction_interval_low",
                Value::from(em_certificate.d1),
            ),
            (
                "tail_correction_interval_high",
                Value::from(em_certificate.d1 + em_certificate.b_tail),
            ),
            ("em_certificate", Value::from(em_certificate.certificate)),
            (
                "em_primary_calls",
                Value::from(em_certificate.primary_calls),
            ),
            (
                "em_secondary_calls",
                Value::from(em_certificate.secondary_calls),
            ),
            (
                "em_primary_leaves",
                Value::from(em_certificate.primary_leaves),
            ),
            (
                "em_secondary_leaves",
                Value::from(em_certificate.secondary_leaves),
            ),
            (
                "em_primary_depth",
                Value::from(em_certificate.primary_depth),
            ),
            (
                "em_secondary_depth",
                Value::from(em_certificate.secondary_depth),
            ),
            ("tail_interval_label", Value::from(TAIL_INTERVAL_LABEL)),
        ] {
            record_object.insert(key.into(), value);
        }
        Ok(record)
    }

    fn assert_finite_json(value: &Value) -> Result<(), String> {
        match value {
            Value::Null => fail("null/nonfinite JSON value is forbidden"),
            Value::Array(items) => {
                for item in items {
                    assert_finite_json(item)?;
                }
                Ok(())
            }
            Value::Object(items) => {
                for item in items.values() {
                    assert_finite_json(item)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    struct JsonlWriter {
        writer: BufWriter<File>,
        sequence: u64,
        run_id: String,
    }

    impl JsonlWriter {
        fn create(path: &Path, run_id: &str) -> Result<Self, String> {
            let file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|error| format!("cannot create raw JSONL {}: {error}", path.display()))?;
            Ok(Self {
                writer: BufWriter::new(file),
                sequence: 0,
                run_id: run_id.to_string(),
            })
        }

        fn record(&mut self, record_type: &str, cell_id: &str, body: Value) -> Result<(), String> {
            let mut body = body
                .as_object()
                .cloned()
                .ok_or("JSONL record body must be an object")?;
            for reserved in ["schema_id", "record_type", "sequence", "run_id", "cell_id"] {
                if body.contains_key(reserved) {
                    return fail(format!("JSONL body contains reserved key {reserved}"));
                }
            }
            let mut record = Map::new();
            record.insert("schema_id".into(), Value::String(SCHEMA_ID.into()));
            record.insert("record_type".into(), Value::String(record_type.into()));
            record.insert("sequence".into(), Value::from(self.sequence));
            record.insert("run_id".into(), Value::String(self.run_id.clone()));
            record.insert("cell_id".into(), Value::String(cell_id.into()));
            record.append(&mut body);
            let value = Value::Object(record);
            assert_finite_json(&value)?;
            serde_json::to_writer(&mut self.writer, &value)
                .map_err(|error| format!("cannot serialize JSONL record: {error}"))?;
            self.writer
                .write_all(b"\n")
                .map_err(|error| format!("cannot terminate JSONL record: {error}"))?;
            self.sequence += 1;
            Ok(())
        }

        fn flush(&mut self) -> Result<(), String> {
            self.writer
                .flush()
                .map_err(|error| format!("cannot flush JSONL output: {error}"))
        }
    }

    fn emit_grid(writer: &mut JsonlWriter, grid: &EvidenceGrid) -> Result<(), String> {
        writer.record(
            "grid",
            GRID_CELL_ID,
            json!({
                "grid_id": grid.id,
                "order": grid.order,
                "map": "y=-3 ln(1-t)",
                "scale_parameter": 3.0,
                "reference_temperature_mev": 10.0,
                "canonical_grid_sha256": grid.canonical_sha256,
                "y_nodes": grid.y_nodes,
                "y_node_bits": f64_bits_vec(&grid.y_nodes),
                "plain_dy_weights": grid.plain_dy_weights,
                "plain_dy_weight_bits": f64_bits_vec(&grid.plain_dy_weights),
                "q_nodes_mev": grid.grid.nodes_mev,
                "dq_weights_mev": grid.grid.weights_mev
            }),
        )
    }

    fn emit_trace(
        writer: &mut JsonlWriter,
        cell_id: &str,
        solve_role: &str,
        target: f64,
        evidence: &SolveEvidence,
    ) -> Result<(), String> {
        let accepted_prefix_len = evidence.result.accepted + 1;
        for record in &evidence.trace {
            let classification = if record.invocation_index == 0 {
                "initial"
            } else if record.invocation_index < accepted_prefix_len {
                "accepted_endpoint"
            } else {
                "event_bisection_probe"
            };
            writer.record(
                "event_trace",
                cell_id,
                json!({
                    "solve_role": solve_role,
                    "target_tgamma_mev": target,
                    "invocation_index": record.invocation_index,
                    "classification": classification,
                    "n": record.n,
                    "n_bits": f64_bits(record.n),
                    "state": record.state,
                    "state_bits": f64_bits_vec(&record.state),
                    "event_value_mev": record.event_value_mev,
                    "rhs_calls": record.rhs_calls,
                    "jacobian_calls": record.jacobian_calls,
                    "state_valid": record.state_valid
                }),
            )?;
        }
        Ok(())
    }

    fn emit_stats(
        writer: &mut JsonlWriter,
        cell_id: &str,
        solve_role: &str,
        target: f64,
        evidence: &SolveEvidence,
    ) -> Result<(), String> {
        writer.record(
            "solver_stats",
            cell_id,
            json!({
                "solve_role": solve_role,
                "target_tgamma_mev": target,
                "accepted": evidence.result.accepted,
                "rejected": evidence.result.rejected,
                "jacobian_evaluations": evidence.result.jacobian_evaluations,
                "linear_setups": evidence.result.linear_setups,
                "rhs_calls": evidence.rhs_calls,
                "jacobian_calls": evidence.jacobian_calls,
                "event_reached": evidence.result.event_reached,
                "failure": evidence.result.failure.clone().unwrap_or_default()
            }),
        )
    }

    fn run_cell(
        writer: &mut JsonlWriter,
        cell: CellSpec,
        grid: &EvidenceGrid,
    ) -> Result<(), String> {
        if cell.grid_order != grid.order {
            return fail("cell/grid order mismatch");
        }
        let system = system_for_grid(grid);
        let initial = system
            .initial_fd_state(10.0)
            .map_err(|error| format!("cannot construct initial state: {error}"))?;
        let config = config_for_cell(&system, cell);
        writer.record(
            "cell_start",
            cell.id,
            json!({
                "grid_id": grid.id,
                "tolerance_id": cell.tolerance_id,
                "solver": "BDF",
                "rtol": cell.rtol,
                "state_atol": cell.state_atol,
                "tgamma_atol_mev": cell.tgamma_atol_mev,
                "elapsed_atol_seconds": cell.elapsed_atol_seconds,
                "h_init": config.h_init,
                "h_min": config.h_min,
                "h_max": config.h_max,
                "max_attempts": config.max_attempts,
                "initial_state_bits": f64_bits_vec(&initial)
            }),
        )?;
        writer.record(
            "checkpoint",
            cell.id,
            native_checkpoint(
                &system,
                grid,
                0,
                "initial_without_solve",
                10.0,
                0.0,
                &initial,
            )?,
        )?;
        writer.flush()?;

        // The terminal solve is deliberately first and is the sole authority
        // source for endpoint cost.  Replays below always restart from these
        // exact initial bits and change only the decreasing event target.
        let terminal = solve_target(&system, &initial, &config, 0.005)?;
        emit_trace(writer, cell.id, "authority_terminal", 0.005, &terminal)?;

        for (target_index, &target) in REPLAY_TARGETS.iter().enumerate() {
            let replay = solve_target(&system, &initial, &config, target)?;
            emit_trace(writer, cell.id, "checkpoint_prefix_replay", target, &replay)?;
            let prefix_len = compare_prefix(&terminal, &replay)?;
            writer.record(
                "prefix_comparison",
                cell.id,
                json!({
                    "target_tgamma_mev": target,
                    "replay_prefix_len": prefix_len,
                    "terminal_prefix_len_compared": prefix_len,
                    "n_bits_equal": true,
                    "state_bits_equal": true,
                    "rhs_counters_equal": true,
                    "jacobian_counters_equal": true,
                    "verdict": "PASS"
                }),
            )?;
            emit_stats(writer, cell.id, "checkpoint_prefix_replay", target, &replay)?;
            writer.record(
                "checkpoint",
                cell.id,
                native_checkpoint(
                    &system,
                    grid,
                    target_index + 1,
                    "checkpoint_prefix_replay",
                    target,
                    replay.result.t,
                    &replay.result.y,
                )?,
            )?;
        }

        writer.record(
            "checkpoint",
            cell.id,
            native_checkpoint(
                &system,
                grid,
                CHECKPOINT_TARGETS.len() - 1,
                "authority_terminal",
                0.005,
                terminal.result.t,
                &terminal.result.y,
            )?,
        )?;
        emit_stats(writer, cell.id, "authority_terminal", 0.005, &terminal)?;
        writer.record(
            "cell_complete",
            cell.id,
            json!({
                "all_targets_present_once": true,
                "all_prefixes_bitwise_equal": true,
                "authority_stats_source": "0.005_MeV_authority_terminal_only",
                "cell_verdict": "PASS"
            }),
        )?;
        writer.flush()
    }

    fn run_authorized_export() -> Result<(), String> {
        let floating_environment = assert_binary64_environment()?;
        // All hashes, permission bits, the complete Fort bundle, and schema
        // identities are checked before grid/system construction or any RHS.
        let authority = load_output_authority()?;
        let exp48 = evidence_grid(48, &authority)?;
        let exp64 = evidence_grid(64, &authority)?;
        let implementation_bundle_sha256 =
            string_field(&authority.approval, "implementation_bundle_sha256")?.to_string();
        validate_sha256(&implementation_bundle_sha256)?;
        let mut writer = JsonlWriter::create(&authority.raw_output_path, &authority.run_id)?;
        writer.record(
            "run_header",
            RUN_CELL_ID,
            json!({
                "contract_sha256_v4": CONTRACT_HASHES[0],
                "contract_sha256_v5": CONTRACT_HASHES[1],
                "contract_sha256_v6": CONTRACT_HASHES[2],
                "contract_sha256_v7": CONTRACT_HASHES[3],
                "contract_sha256_v8": CONTRACT_HASHES[4],
                "implementation_bundle_sha256": implementation_bundle_sha256,
                "stage2_approval_path": authority.approval_path,
                "stage2_approval_sha256": authority.approval_sha256,
                "fort_bundle_manifest_path": authority.fort_manifest_path,
                "fort_bundle_manifest_sha256": authority.fort_manifest_sha256,
                "fort_complete_bundle_sha256": authority.fort_complete_bundle_sha256,
                "rust_source_sha256": authority.rust_source_sha256,
                "rust_binary_sha256": authority.rust_binary_sha256,
                "schema_path": authority.schema_path,
                "schema_sha256": authority.schema_sha256,
                "orchestrator_path": authority.orchestrator_path,
                "orchestrator_sha256": authority.orchestrator_sha256,
                "sha256sum_path": authority.sha256sum_path,
                "sha256sum_binary_sha256": authority.sha256sum_binary_sha256,
                "floating_environment": floating_environment,
                "constants": {
                    "electron_mass_mev": ELECTRON_MASS_MEV,
                    "fermi_constant_mev_minus_2": G_F_MEV_MINUS_2,
                    "sin2_theta_w": SIN2_THETA_W,
                    "newton_g_mev_minus_2": NEWTON_G_MEV_MINUS_2,
                    "mev_to_inverse_seconds": MEV_TO_INVERSE_SECONDS
                },
                "physical_model_switches": {
                    "flat_flrw": true,
                    "massless_diagonal_neutrinos": true,
                    "zero_lepton_asymmetry": true,
                    "electron_collisions": true,
                    "neutrino_self_collisions": true,
                    "qed": false,
                    "qke": false,
                    "coherence": false,
                    "solver": "BDF_ONLY"
                },
                "checkpoint_targets_mev": CHECKPOINT_TARGETS
            }),
        )?;
        emit_grid(&mut writer, &exp48)?;
        emit_grid(&mut writer, &exp64)?;
        writer.flush()?;

        for cell in five_cells() {
            let grid = if cell.grid_order == 48 {
                &exp48
            } else {
                &exp64
            };
            if let Err(error) = run_cell(&mut writer, cell, grid) {
                let _ = writer.record(
                    "failure",
                    cell.id,
                    json!({
                        "stage": "rabbit_cell",
                        "solve_role": "unknown_or_partial",
                        "target_tgamma_mev": 0.0,
                        "reason": error,
                        "partial_output_retained": true
                    }),
                );
                let _ = writer.flush();
                return fail(format!("RABBIT evidence cell {} failed", cell.id));
            }
        }
        writer.record(
            "run_complete",
            RUN_CELL_ID,
            json!({
                "cell_count": RABBIT_CELL_IDS.len(),
                "all_five_cells_complete": true,
                "physical_output_authorized_by_stage2_hash": true,
                "fort_bundle_frozen_before_rabbit": true,
                "run_verdict": "PASS"
            }),
        )?;
        writer.flush()
    }

    fn gk_panel_sample_points(left: f64, right: f64) -> Vec<f64> {
        let midpoint = left + 0.5 * (right - left);
        let half_width = 0.5 * (right - left);
        let mut points = vec![midpoint];
        for &abscissa in &GK_ABSCISSAE[..7] {
            points.push(midpoint - half_width * abscissa);
            points.push(midpoint + half_width * abscissa);
        }
        points
    }

    fn adversarial_gk_integrand(s: f64) -> Result<f64, String> {
        if !s.is_finite() || !(0.0..=1.0).contains(&s) {
            return fail("invalid adversarial GK input");
        }
        let u = 2.0 * s - 1.0;
        let mut product = u * u;
        for &abscissa in &GK_ABSCISSAE[..7] {
            product *= (u - abscissa).powi(2) * (u + abscissa).powi(2);
        }
        Ok(2.0 * (1.0 + product))
    }

    fn displayed_em_tail_bound(q: f64) -> Result<(f64, f64), String> {
        if !q.is_finite() || q < 0.0 {
            return fail("invalid nonnegative tail coordinate");
        }
        let r = q + 128.0;
        let polynomial = 2.0 * r.powi(4)
            + 16.0 * r.powi(3)
            + 96.0 * r.powi(2)
            + 384.0 * r
            + 768.0
            + q.powi(2) * (2.0 * r.powi(2) + 8.0 * r + 16.0);
        Ok((polynomial, (-(q + 64.0)).exp() * polynomial / 4.0))
    }

    #[test]
    fn f10_nonphysical_grid_hashes_are_exact() {
        assert_binary64_environment().expect("binary64 environment");
        let tool = Path::new("/usr/bin/sha256sum");
        let artifact = nonphysical_grid_hash_artifact(tool).expect("grid hash artifact");
        println!(
            "F10_GRID_HASH_ARTIFACT={}",
            serde_json::to_string(&artifact).expect("serialize grid hash artifact")
        );
        let hashes = artifact["grid_sha256"]
            .as_object()
            .expect("grid hash object");
        assert_eq!(hashes["EXP48"].as_str(), Some(EXP48_GRID_SHA256));
        assert_eq!(hashes["EXP64"].as_str(), Some(EXP64_GRID_SHA256));
    }

    #[test]
    fn f10_nonphysical_gk_controller_contract_branches() {
        assert_binary64_environment().expect("binary64 environment");

        let mut adversarial = adversarial_gk_integrand;
        let mut adversarial_calls = 0;
        let adversarial_root = gk_panel(&mut adversarial, 0.0, 1.0, &mut adversarial_calls, 65_536)
            .expect("adversarial root panel");
        assert_eq!(adversarial_calls, 15);
        assert_eq!(adversarial_root.kronrod.to_bits(), 2.0_f64.to_bits());
        assert_eq!(adversarial_root.gauss.to_bits(), 2.0_f64.to_bits());
        assert_eq!(adversarial_root.embedded.to_bits(), 0.0_f64.to_bits());
        let root_kronrod_bits = adversarial_root.kronrod.to_bits();
        let root_gauss_bits = adversarial_root.gauss.to_bits();
        let root_embedded_bits = adversarial_root.embedded.to_bits();
        let root_only = finish_gk_run(
            &mut adversarial,
            adversarial_root,
            0.0,
            0,
            24,
            65_536,
            adversarial_calls,
        )
        .expect("equality accepts the zero embedded estimate");
        let exact_fraction_oracle = f64::from_bits(0x4000_0000_0077_d2db);
        assert_eq!(root_only.estimate.to_bits(), 2.0_f64.to_bits());
        assert_ne!(
            root_only.estimate.to_bits(),
            exact_fraction_oracle.to_bits()
        );

        let mut adversarial_forced = adversarial_gk_integrand;
        let mut adversarial_forced_calls = 0;
        let adversarial_forced_root = gk_panel(
            &mut adversarial_forced,
            0.0,
            1.0,
            &mut adversarial_forced_calls,
            65_536,
        )
        .expect("forced adversarial root panel");
        let forced_run = finish_gk_run(
            &mut adversarial_forced,
            adversarial_forced_root,
            1.0e-14,
            1,
            24,
            65_536,
            adversarial_forced_calls,
        )
        .expect("mandatory split must resolve the adversarial polynomial");
        let forced_tolerance =
            8.0 * forced_run.embedded + 128.0 * f64::EPSILON * exact_fraction_oracle.abs();
        assert!((forced_run.estimate - exact_fraction_oracle).abs() <= forced_tolerance);
        assert!(forced_run.calls > 15 && forced_run.leaves >= 2 && forced_run.maximum_depth >= 1);
        assert_ne!(root_only.estimate.to_bits(), forced_run.estimate.to_bits());

        let mut equality_integrand = |x: f64| Ok(x.exp());
        let mut equality_calls = 0;
        let equality_root = gk_panel(
            &mut equality_integrand,
            0.0,
            1.0,
            &mut equality_calls,
            65_536,
        )
        .expect("equality root panel");
        let equality_budget = equality_root.embedded;
        let equality_run = finish_gk_run(
            &mut equality_integrand,
            equality_root,
            equality_budget,
            0,
            24,
            65_536,
            equality_calls,
        )
        .expect("embedded equality branch");
        assert_eq!(equality_run.calls, 15);
        assert_eq!(equality_run.leaves, 1);
        assert_eq!(equality_run.maximum_depth, 0);

        let evaluation_trace = RefCell::new(Vec::new());
        let mut ordered_constant = |x: f64| {
            evaluation_trace.borrow_mut().push(x);
            Ok(1.0)
        };
        let mut ordered_calls = 0;
        let ordered_root = gk_panel(&mut ordered_constant, 0.0, 1.0, &mut ordered_calls, 65_536)
            .expect("ordered root panel");
        let ordered_run = finish_gk_run(
            &mut ordered_constant,
            ordered_root,
            f64::MAX,
            1,
            24,
            65_536,
            ordered_calls,
        )
        .expect("mandatory one-level subdivision");
        assert_eq!(ordered_run.calls, 45);
        assert_eq!(ordered_run.leaves, 2);
        assert_eq!(ordered_run.maximum_depth, 1);
        let trace = evaluation_trace.borrow();
        let root_points = gk_panel_sample_points(0.0, 1.0);
        assert_eq!(trace[..15], root_points[..]);
        assert!(trace[15..30].iter().all(|&x| x < 0.5));
        assert!(trace[30..45].iter().all(|&x| x > 0.5));

        let zero_width_evaluations = Cell::new(0_u64);
        let mut zero_width = |_: f64| {
            zero_width_evaluations.set(zero_width_evaluations.get() + 1);
            Ok(1.0)
        };
        let mut zero_width_calls = 0;
        let zero_panel = gk_panel(&mut zero_width, 0.5, 0.5, &mut zero_width_calls, 0)
            .expect("zero-width panel");
        assert_eq!(zero_panel.kronrod.to_bits(), 0.0_f64.to_bits());
        assert_eq!(zero_panel.gauss.to_bits(), 0.0_f64.to_bits());
        assert_eq!(zero_panel.resabs.to_bits(), 0.0_f64.to_bits());
        assert_eq!(zero_panel.embedded.to_bits(), 0.0_f64.to_bits());
        assert_eq!(zero_width_calls, 0);
        assert_eq!(zero_width_evaluations.get(), 0);

        let mut bounded = |_: f64| Ok(1.0);
        let mut bounded_calls = 0;
        assert_eq!(
            gk_panel(&mut bounded, 1.0, 0.0, &mut bounded_calls, 65_536)
                .err()
                .expect("reversed bounds must reject"),
            "invalid GK panel bounds"
        );
        assert_eq!(bounded_calls, 0);

        let mut depth_integrand = |_: f64| Ok(1.0);
        let mut depth_calls = 0;
        let depth_root = gk_panel(&mut depth_integrand, 0.0, 1.0, &mut depth_calls, 65_536)
            .expect("depth root panel");
        assert_eq!(
            finish_gk_run(
                &mut depth_integrand,
                depth_root,
                0.0,
                1,
                0,
                65_536,
                depth_calls,
            )
            .err()
            .expect("depth exhaustion must reject"),
            "GK maximum depth exhausted"
        );

        let capped_evaluations = Cell::new(0_u64);
        let mut capped = |_: f64| {
            capped_evaluations.set(capped_evaluations.get() + 1);
            Ok(1.0)
        };
        let mut capped_calls = 0;
        assert_eq!(
            gk_panel(&mut capped, 0.0, 1.0, &mut capped_calls, 14)
                .err()
                .expect("call cap must reject before evaluation"),
            "GK call cap exhausted before panel evaluation"
        );
        assert_eq!(capped_calls, 0);
        assert_eq!(capped_evaluations.get(), 0);

        let mut nonfinite = |_: f64| Ok(f64::NAN);
        let mut nonfinite_calls = 0;
        assert_eq!(
            gk_panel(&mut nonfinite, 0.0, 1.0, &mut nonfinite_calls, 65_536)
                .err()
                .expect("nonfinite integrand must reject"),
            "nonfinite GK integrand value"
        );
        assert_eq!(nonfinite_calls, 15);

        let cutoff = 12.0_f64;
        let exact_exponential = 1.0 - (-cutoff).exp();
        let primary_evaluations = Cell::new(0_u64);
        let secondary_evaluations = Cell::new(0_u64);
        let mut primary = |s: f64| {
            primary_evaluations.set(primary_evaluations.get() + 1);
            Ok(cutoff * (-cutoff * s).exp())
        };
        let mut secondary = |s: f64| {
            secondary_evaluations.set(secondary_evaluations.get() + 1);
            Ok(2.0 * cutoff * s * (-cutoff * s * s).exp())
        };
        let mut primary_calls = 0;
        let primary_root = gk_panel(&mut primary, 0.0, 1.0, &mut primary_calls, 65_536)
            .expect("primary exponential transform");
        let mut secondary_calls = 0;
        let secondary_root = gk_panel(&mut secondary, 0.0, 1.0, &mut secondary_calls, 65_536)
            .expect("secondary exponential transform");
        let primary_run = finish_gk_run(
            &mut primary,
            primary_root,
            1.0e-12,
            0,
            24,
            65_536,
            primary_calls,
        )
        .expect("primary exponential run");
        let secondary_run = finish_gk_run(
            &mut secondary,
            secondary_root,
            1.0e-12,
            1,
            24,
            65_536,
            secondary_calls,
        )
        .expect("secondary forced exponential run");
        assert!((primary_run.estimate - exact_exponential).abs() <= 1.0e-12);
        assert!((secondary_run.estimate - exact_exponential).abs() <= 1.0e-12);
        assert!((primary_run.estimate - secondary_run.estimate).abs() <= 2.0e-12);
        assert_eq!(primary_evaluations.get(), primary_run.calls);
        assert_eq!(secondary_evaluations.get(), secondary_run.calls);
        assert!(secondary_run.maximum_depth >= 1);

        const EXACT_FRACTION_NUMERATOR: &str = "19685918893639194464452448367780047320884800172344097481579344457433385619488528681029067169754881774235738859296689979550272106686236893645157905454124965918755875847818352925791973464803287949432635804266828183498641658432948168132908132632071808725564854092279411536990106812115653784075817547599140214982330796705992601658970213928587761729819873110003776542071965424315756746490430757967458921809849928011623319562425365387237968261439331825996123";
        const EXACT_FRACTION_DENOMINATOR: &str = "9842959429656787573772398083360525213633788470409790471919535770244079266815986945706020897759378143617756918355765798301235802965197396253878046259854801111797575264905496416641325657882508783950597668102576949989214816717347552541807091103964053976227077831042748675469202355169385616275351261946601273582735803406482020450281876577531275499100171825368836939424003970798394277866775482166564222615429756768325148388916776453284668751986247807795200";
        let node_bits: Vec<String> = GK_ABSCISSAE
            .iter()
            .map(|node| format!("{:016x}", node.to_bits()))
            .collect();
        let artifact = json!({
            "schema_id": "rabbit.f10.rust-gk-artifact.v1",
            "node_bits": node_bits,
            "root_kronrod_bits": format!("{root_kronrod_bits:016x}"),
            "root_gauss_bits": format!("{root_gauss_bits:016x}"),
            "root_embedded_bits": format!("{root_embedded_bits:016x}"),
            "forced_estimate_bits": format!("{:016x}", forced_run.estimate.to_bits()),
            "forced_embedded_bits": format!("{:016x}", forced_run.embedded.to_bits()),
            "exact_target_bits": "400000000077d2db",
            "exact_fraction_numerator": EXACT_FRACTION_NUMERATOR,
            "exact_fraction_denominator": EXACT_FRACTION_DENOMINATOR,
            "equality": true,
            "forced": true,
            "zero": true,
            "reversed": true,
            "left_first": true,
            "depth": true,
            "call_cap": true,
            "nonfinite": true,
            "dual_transform": true
        });
        println!(
            "F10_RUST_GK_ARTIFACT={}",
            serde_json::to_string(&artifact).expect("serialize Rust GK artifact")
        );
    }

    #[test]
    fn f10_nonphysical_em_certificate_covers_every_checkpoint() {
        assert_binary64_environment().expect("binary64 environment");
        for tgamma_mev in CHECKPOINT_TARGETS {
            let native = electromagnetic_eos(tgamma_mev)
                .expect("native electromagnetic EOS")
                .drho_dt;
            let certificate =
                em_derivative_certificate(tgamma_mev, native).expect("V7 EM certificate");
            assert!(certificate.primary_calls > 0 && certificate.primary_calls <= 65_536);
            assert!(certificate.secondary_calls > 0 && certificate.secondary_calls <= 65_536);
            assert_eq!(certificate.primary_calls % 15, 0);
            assert_eq!(certificate.secondary_calls % 15, 0);
            assert!(certificate.primary_leaves > 0);
            assert!(certificate.secondary_leaves >= 2);
            assert!(certificate.primary_depth <= 24);
            assert!((1..=24).contains(&certificate.secondary_depth));
            assert_eq!(
                certificate.b_tail.to_bits(),
                (1.0e-18 * tgamma_mev.powi(3)).to_bits()
            );
            assert!(certificate.d_direct >= certificate.d1);

            let cutoff = ELECTRON_MASS_MEV + 128.0 * tgamma_mev;
            let maximum_energy = (cutoff * cutoff + ELECTRON_MASS_MEV * ELECTRON_MASS_MEV).sqrt();
            assert!(maximum_energy / tgamma_mev < 252.0);
            let q = ELECTRON_MASS_MEV / tgamma_mev;
            let r = q + 128.0;
            let (tail_polynomial, derived_tail_ratio) =
                displayed_em_tail_bound(q).expect("physical q tail bound");
            assert!(tail_polynomial < 4.2 * r.powi(4));
            assert!(derived_tail_ratio < 4.53e-20);

            let panel_width = 2.0_f64.powi(-24);
            let mut samples = vec![0.0, 1.0];
            samples.extend(gk_panel_sample_points(0.0, 1.0));
            samples.extend(gk_panel_sample_points(0.0, panel_width));
            samples.extend(gk_panel_sample_points(1.0 - panel_width, 1.0));
            for secondary in [false, true] {
                for &s in &samples {
                    let value = em_finite_integrand(tgamma_mev, cutoff, secondary, s)
                        .expect("representable finite EM transform");
                    if s == 0.0 {
                        assert_eq!(value.to_bits(), 0);
                    } else {
                        assert!(value.is_finite() && value > 0.0);
                    }
                }
            }
            let rejected_half_line_s =
                1.0 - 0.5 * panel_width + 0.5 * panel_width * GK_ABSCISSAE[0];
            let rejected_half_line_p =
                ELECTRON_MASS_MEV * rejected_half_line_s / (1.0 - rejected_half_line_s);
            let rejected_half_line_energy =
                (rejected_half_line_p.powi(2) + ELECTRON_MASS_MEV.powi(2)).sqrt();
            assert!(rejected_half_line_p > cutoff);
            assert_eq!(
                (-(rejected_half_line_energy / tgamma_mev)).exp().to_bits(),
                0
            );
        }

        let (q0_polynomial, q0_ratio) = displayed_em_tail_bound(0.0).expect("q=0 tail bound");
        assert!(q0_polynomial < 4.2 * 128.0_f64.powi(4));
        assert!(q0_ratio < 4.53e-20);
        let q0_monotone_envelope = 1.05 * (-64.0_f64).exp() * 128.0_f64.powi(4);
        assert!(q0_monotone_envelope < 4.53e-20);
        let log_derivative_upper = -1.0_f64 + 4.0 / 128.0;
        assert_eq!(log_derivative_upper.to_bits(), (-31.0_f64 / 32.0).to_bits());
        assert!(log_derivative_upper < 0.0);
        for q in [0.0, 0.25, 1.0, 4.0, 16.0, 64.0, 128.0, 512.0, 2048.0] {
            let r = q + 128.0;
            let (polynomial, ratio) = displayed_em_tail_bound(q).expect("q>=0 tail bound");
            assert!(polynomial < 4.2 * r.powi(4));
            assert!(ratio <= q0_ratio);
            assert!(-1.0 + 4.0 / r <= log_derivative_upper);
            let monotone_envelope = 1.05 * (-(q + 64.0)).exp() * r.powi(4);
            assert!(ratio <= monotone_envelope);
            assert!(monotone_envelope <= q0_monotone_envelope);
        }
    }

    #[test]
    fn f10_nonphysical_first_law_consumes_total_block_q() {
        let total = full_first_law_residual(4.0, 8.0, 2.0, -30.0, 10.0, 4.0)
            .expect("total-block first law");
        let electron_only = full_first_law_residual(4.0, 6.0, 2.0, -30.0, 10.0, 4.0)
            .expect("electron-only negative control");
        assert_eq!(total.to_bits(), 0.0_f64.to_bits());
        assert!(electron_only > 0.0);
    }

    #[test]
    #[ignore = "requires exact stage-2 output authority and a frozen complete Fort bundle"]
    fn export_f10_independent_bdf_evidence() {
        if let Err(error) = run_authorized_export() {
            panic!("F10 independent BDF evidence export failed: {error}");
        }
    }
}
