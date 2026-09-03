//! Positivity-preserving local steps for conservative Pauli collision edges.
//!
//! The state variable is the raw occupation, not a clipped or projected
//! surrogate.  Each edge carries non-negative forward and reverse
//! coefficients.  A backward-Euler extent solve is one dimensional,
//! bracketed by the exact Pauli capacities.  The Pauli-box algebra supports
//! arbitrarily stiff finite steps, while finite-precision certification may
//! still fail closed.  The caller owns operator splitting across edges and
//! commits only a completely validated candidate.

#![cfg_attr(not(test), allow(dead_code))]

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    static FLUX_VALUE_EVALUATIONS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliEdgeTopology {
    /// `j -> i` minus `i -> j`; the weighted number sum is invariant.
    ElasticTransfer,
    /// Bath-driven pair creation minus annihilation; both nodes move together.
    PairSource,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PauliEdge {
    pub(crate) topology: PauliEdgeTopology,
    pub(crate) first_node: usize,
    pub(crate) second_node: usize,
    pub(crate) first_measure: f64,
    pub(crate) second_measure: f64,
    pub(crate) gain_coefficient_mev: f64,
    pub(crate) loss_coefficient_mev: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliFluxResolution {
    Resolved,
    ExactZero,
    UnresolvedForCertificate,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PauliFluxEvaluation {
    pub(crate) net_mev: f64,
    pub(crate) traffic_upper_bound_mev: f64,
    pub(crate) abs_error_bound_mev: f64,
    pub(crate) resolution: PauliFluxResolution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliEdgeApplicationKind {
    Solved,
    ExactStationary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliEdgeFailureKind {
    InvalidInput,
    UnresolvedFlux,
    UncertainPhysicalBracket,
    CertificateUnattainableAtStep,
    StateMapUnresolved,
    StagnatedInterval,
    InvalidResidual,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PauliEdgeFailure {
    pub(crate) kind: PauliEdgeFailureKind,
    message: &'static str,
}

impl PauliEdgeFailure {
    fn new(kind: PauliEdgeFailureKind, message: &'static str) -> Self {
        Self { kind, message }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliEdgeStep {
    /// None means the requested step was exactly zero and no edge was applied.
    pub(crate) kind: Option<PauliEdgeApplicationKind>,
    pub(crate) extent: f64,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) residual_abs: f64,
    pub(crate) residual_scale: f64,
    pub(crate) traffic_upper_bound_mev: f64,
    pub(crate) flux_abs_error_mev: f64,
    pub(crate) root_error_abs: f64,
    pub(crate) occupation_error_abs: f64,
    pub(crate) max_occupation_bracket_width: f64,
    pub(crate) conditioning_lower_bound: f64,
}

fn valid_occupation(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const PRODUCT_ABS_ERROR_FACTOR: f64 = 8.0 * f64::EPSILON;
const DIFFERENCE_ABS_ERROR_FACTOR: f64 = 2.0 * f64::EPSILON;

#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn new(lo: f64, hi: f64) -> Result<Self, PauliEdgeFailure> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge interval is non-finite or unordered",
            ));
        }
        Ok(Self { lo, hi })
    }

    fn point(value: f64) -> Result<Self, PauliEdgeFailure> {
        Self::new(value, value)
    }

    fn is_zero(self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }

    fn add(self, rhs: Self) -> Result<Self, PauliEdgeFailure> {
        if rhs.is_zero() {
            return Ok(self);
        }
        if self.is_zero() {
            return Ok(rhs);
        }
        Self::new((self.lo + rhs.lo).next_down(), (self.hi + rhs.hi).next_up())
    }

    fn sub(self, rhs: Self) -> Result<Self, PauliEdgeFailure> {
        if rhs.is_zero() {
            return Ok(self);
        }
        Self::new((self.lo - rhs.hi).next_down(), (self.hi - rhs.lo).next_up())
    }

    fn mul_nonnegative(self, rhs: Self) -> Result<Self, PauliEdgeFailure> {
        if self.lo < 0.0 || rhs.lo < 0.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge interval multiplication requires non-negative factors",
            ));
        }
        if self.is_zero() || rhs.is_zero() {
            return Self::point(0.0);
        }
        let lo_product = self.lo * rhs.lo;
        let hi_product = self.hi * rhs.hi;
        let lo = if lo_product == 0.0 {
            0.0
        } else {
            lo_product.next_down().max(0.0)
        };
        let hi = if hi_product == 0.0 {
            MIN_SUBNORMAL
        } else {
            hi_product.next_up()
        };
        Self::new(lo, hi)
    }

    fn div_positive(self, rhs: f64) -> Result<Self, PauliEdgeFailure> {
        if !rhs.is_finite() || rhs <= 0.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge interval divisor is not finite and positive",
            ));
        }
        if self.is_zero() {
            return Ok(self);
        }
        Self::new((self.lo / rhs).next_down(), (self.hi / rhs).next_up())
    }

    fn scale_nonnegative(self, rhs: f64) -> Result<Self, PauliEdgeFailure> {
        if !rhs.is_finite() || rhs < 0.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge interval scale is not finite and non-negative",
            ));
        }
        if rhs == 0.0 || self.is_zero() {
            return Self::point(0.0);
        }
        Self::new((self.lo * rhs).next_down(), (self.hi * rhs).next_up())
    }

    fn complement_unit(self) -> Result<Self, PauliEdgeFailure> {
        if self.lo < 0.0 || self.hi > 1.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::StateMapUnresolved,
                "Pauli edge interval leaves the Fermi box",
            ));
        }
        if self.lo == 0.0 && self.hi == 0.0 {
            return Self::point(1.0);
        }
        if self.lo == 1.0 && self.hi == 1.0 {
            return Self::point(0.0);
        }
        Self::new((1.0 - self.hi).next_down(), (1.0 - self.lo).next_up())
    }
}

#[derive(Clone, Copy, Debug)]
struct PauliFluxInterval {
    net_mev: Interval,
    traffic_upper_bound_mev: f64,
}

#[derive(Clone, Copy, Debug)]
struct RootResidualInterval {
    residual: Interval,
    root_error_abs: f64,
    occupation_error_abs: f64,
}

#[derive(Clone, Copy, Debug)]
enum DirectProduct {
    ExactZero,
    Representable { value: f64, abs_error: f64 },
    Unresolved,
}

fn direct_three_factor_product(coefficient: f64, factors: [f64; 2]) -> DirectProduct {
    if coefficient == 0.0 || factors.into_iter().any(|factor| factor == 0.0) {
        return DirectProduct::ExactZero;
    }
    let first = coefficient * factors[0];
    if !first.is_finite() || first == 0.0 {
        return DirectProduct::Unresolved;
    }
    let value = first * factors[1];
    if !value.is_finite() || value == 0.0 {
        return DirectProduct::Unresolved;
    }
    DirectProduct::Representable {
        value,
        abs_error: PRODUCT_ABS_ERROR_FACTOR * value.abs() + 4.0 * MIN_SUBNORMAL,
    }
}

fn zero_step_report() -> PauliEdgeStep {
    PauliEdgeStep {
        kind: None,
        extent: 0.0,
        nonlinear_iterations: 0,
        residual_abs: 0.0,
        residual_scale: f64::from_bits(1),
        traffic_upper_bound_mev: 0.0,
        flux_abs_error_mev: 0.0,
        root_error_abs: 0.0,
        occupation_error_abs: 0.0,
        max_occupation_bracket_width: 0.0,
        conditioning_lower_bound: 0.0,
    }
}

fn exact_stationary_report() -> PauliEdgeStep {
    PauliEdgeStep {
        kind: Some(PauliEdgeApplicationKind::ExactStationary),
        ..zero_step_report()
    }
}

fn edge_factors(
    topology: PauliEdgeTopology,
    first_occupation: f64,
    second_occupation: f64,
) -> Result<([f64; 2], [f64; 2]), &'static str> {
    if !valid_occupation(first_occupation) || !valid_occupation(second_occupation) {
        return Err("Pauli edge occupation is outside [0, 1]");
    }
    Ok(match topology {
        PauliEdgeTopology::ElasticTransfer => (
            [1.0 - first_occupation, second_occupation],
            [first_occupation, 1.0 - second_occupation],
        ),
        PauliEdgeTopology::PairSource => (
            [1.0 - first_occupation, 1.0 - second_occupation],
            [first_occupation, second_occupation],
        ),
    })
}

fn stable_nonnegative_product_difference(
    gain_coefficient: f64,
    gain_factors: [f64; 2],
    loss_coefficient: f64,
    loss_factors: [f64; 2],
) -> Result<f64, &'static str> {
    let log_product = |coefficient: f64, factors: [f64; 2]| -> Result<Option<f64>, &'static str> {
        if !coefficient.is_finite()
            || coefficient < 0.0
            || factors
                .into_iter()
                .any(|factor| !factor.is_finite() || factor < 0.0)
        {
            return Err("Pauli edge gain or loss is invalid");
        }
        if coefficient == 0.0 || factors.into_iter().any(|factor| factor == 0.0) {
            return Ok(None);
        }
        Ok(Some(coefficient.ln() + factors[0].ln() + factors[1].ln()))
    };
    let signed_representable_exp = |sign: f64, log_abs: f64| -> Result<f64, &'static str> {
        if !log_abs.is_finite() || log_abs > f64::MAX.ln() {
            return Err("Pauli edge affinity difference is non-finite");
        }
        let value = sign * log_abs.exp();
        value
            .is_finite()
            .then_some(value)
            .ok_or("Pauli edge affinity difference is non-finite")
    };

    let log_gain = log_product(gain_coefficient, gain_factors)?;
    let log_loss = log_product(loss_coefficient, loss_factors)?;
    match (log_gain, log_loss) {
        (None, None) => Ok(0.0),
        (Some(log_gain), None) => signed_representable_exp(1.0, log_gain),
        (None, Some(log_loss)) => signed_representable_exp(-1.0, log_loss),
        (Some(log_gain), Some(log_loss)) => {
            let (sign, maximum, scaled_difference) = if log_gain >= log_loss {
                (1.0, log_gain, -(log_loss - log_gain).exp_m1())
            } else {
                (-1.0, log_loss, -(log_gain - log_loss).exp_m1())
            };
            if scaled_difference == 0.0 {
                return Ok(0.0);
            }
            signed_representable_exp(sign, maximum + scaled_difference.ln())
        }
    }
}

fn residual_scale(extent: f64, step_traffic_upper: f64) -> f64 {
    extent.abs().max(step_traffic_upper).max(MIN_SUBNORMAL)
}

#[derive(Clone, Copy, Debug)]
struct RootResidual {
    value: f64,
    derivative: f64,
    scale: f64,
    flux: PauliFluxEvaluation,
    certified: RootResidualInterval,
}

fn occupation_bracket_is_certified(
    lower: f64,
    upper: f64,
    first_measure: f64,
    second_measure: f64,
) -> bool {
    let extent_width = (upper - lower).abs();
    let first_width = extent_width / first_measure;
    let second_width = extent_width / second_measure;
    first_width.max(second_width) <= 128.0 * f64::EPSILON
}

fn maximum_occupation_bracket_width(
    lower: f64,
    upper: f64,
    first_measure: f64,
    second_measure: f64,
) -> f64 {
    let extent_width = (upper - lower).abs();
    (extent_width / first_measure).max(extent_width / second_measure)
}

fn exhausted_interval_failure_kind(
    lower: f64,
    upper: f64,
    occupation_width: f64,
    all_candidates_rejected: bool,
) -> Option<PauliEdgeFailureKind> {
    (all_candidates_rejected
        && lower.is_finite()
        && upper.is_finite()
        && lower < upper
        && lower.next_up().to_bits() == upper.to_bits()
        && occupation_width.is_finite()
        && occupation_width > 128.0 * f64::EPSILON)
        .then_some(PauliEdgeFailureKind::CertificateUnattainableAtStep)
}

impl PauliEdge {
    pub(crate) fn new(
        topology: PauliEdgeTopology,
        first_node: usize,
        second_node: usize,
        first_measure: f64,
        second_measure: f64,
        gain_coefficient_mev: f64,
        loss_coefficient_mev: f64,
    ) -> Result<Self, &'static str> {
        if (topology == PauliEdgeTopology::ElasticTransfer && first_node == second_node)
            || !first_measure.is_finite()
            || first_measure <= 0.0
            || !second_measure.is_finite()
            || second_measure <= 0.0
            || !gain_coefficient_mev.is_finite()
            || gain_coefficient_mev < 0.0
            || !loss_coefficient_mev.is_finite()
            || loss_coefficient_mev < 0.0
        {
            return Err("Pauli edge is outside the accepted physical domain");
        }

        Ok(Self {
            topology,
            first_node,
            second_node,
            first_measure,
            second_measure,
            gain_coefficient_mev,
            loss_coefficient_mev,
        })
    }

    #[inline]
    pub(crate) fn flux_mev(
        &self,
        first_occupation: f64,
        second_occupation: f64,
    ) -> Result<f64, &'static str> {
        #[cfg(test)]
        FLUX_VALUE_EVALUATIONS.with(|count| count.set(count.get() + 1));
        let (gain_factors, loss_factors) =
            edge_factors(self.topology, first_occupation, second_occupation)?;
        stable_nonnegative_product_difference(
            self.gain_coefficient_mev,
            gain_factors,
            self.loss_coefficient_mev,
            loss_factors,
        )
    }

    pub(crate) fn certified_flux_evaluation(
        &self,
        first_occupation: f64,
        second_occupation: f64,
    ) -> Result<PauliFluxEvaluation, PauliEdgeFailure> {
        let (gain_factors, loss_factors) =
            edge_factors(self.topology, first_occupation, second_occupation).map_err(
                |message| PauliEdgeFailure::new(PauliEdgeFailureKind::InvalidInput, message),
            )?;
        let gain = direct_three_factor_product(self.gain_coefficient_mev, gain_factors);
        let loss = direct_three_factor_product(self.loss_coefficient_mev, loss_factors);
        match (gain, loss) {
            (DirectProduct::ExactZero, DirectProduct::ExactZero) => Ok(PauliFluxEvaluation {
                net_mev: 0.0,
                traffic_upper_bound_mev: 0.0,
                abs_error_bound_mev: 0.0,
                resolution: PauliFluxResolution::ExactZero,
            }),
            (DirectProduct::Unresolved, _) | (_, DirectProduct::Unresolved) => {
                let value_flux =
                    self.flux_mev(first_occupation, second_occupation)
                        .map_err(|message| {
                            PauliEdgeFailure::new(PauliEdgeFailureKind::InvalidResidual, message)
                        })?;
                Ok(PauliFluxEvaluation {
                    net_mev: value_flux,
                    traffic_upper_bound_mev: 0.0,
                    abs_error_bound_mev: f64::INFINITY,
                    resolution: PauliFluxResolution::UnresolvedForCertificate,
                })
            }
            (gain, loss) => {
                let unpack = |product: DirectProduct| match product {
                    DirectProduct::ExactZero => (0.0, 0.0),
                    DirectProduct::Representable { value, abs_error } => (value, abs_error),
                    DirectProduct::Unresolved => unreachable!("handled before unpacking"),
                };
                let (gain_value, gain_error) = unpack(gain);
                let (loss_value, loss_error) = unpack(loss);
                let net_mev = gain_value - loss_value;
                let difference_error = DIFFERENCE_ABS_ERROR_FACTOR
                    * (gain_value.abs() + loss_value.abs())
                    + 2.0 * MIN_SUBNORMAL;
                let abs_error_bound_mev = gain_error + loss_error + difference_error;
                let traffic_upper_bound_mev =
                    gain_value.abs() + loss_value.abs() + gain_error + loss_error;
                Ok(PauliFluxEvaluation {
                    net_mev,
                    traffic_upper_bound_mev,
                    abs_error_bound_mev,
                    resolution: if net_mev.abs() > abs_error_bound_mev {
                        PauliFluxResolution::Resolved
                    } else {
                        PauliFluxResolution::UnresolvedForCertificate
                    },
                })
            }
        }
    }

    fn initial_flux_resolution(
        &self,
        first_occupation: f64,
        second_occupation: f64,
    ) -> Result<PauliFluxResolution, PauliEdgeFailure> {
        Ok(self
            .certified_flux_evaluation(first_occupation, second_occupation)?
            .resolution)
    }

    fn occupations_at_extent(
        &self,
        initial: [f64; 2],
        extent: f64,
    ) -> Result<[f64; 2], &'static str> {
        let first = initial[0] + extent / self.first_measure;
        let second_sign = match self.topology {
            PauliEdgeTopology::ElasticTransfer => -1.0,
            PauliEdgeTopology::PairSource => 1.0,
        };
        let second = initial[1] + second_sign * extent / self.second_measure;
        (valid_occupation(first) && valid_occupation(second))
            .then_some([first, second])
            .ok_or("Pauli edge extent leaves the Fermi box")
    }

    fn affine_occupation_intervals(
        &self,
        initial: [f64; 2],
        extent: f64,
    ) -> Result<[Interval; 2], PauliEdgeFailure> {
        let extent_interval = Interval::point(extent)?;
        let first =
            Interval::point(initial[0])?.add(extent_interval.div_positive(self.first_measure)?)?;
        let second_quotient = extent_interval.div_positive(self.second_measure)?;
        let second = match self.topology {
            PauliEdgeTopology::ElasticTransfer => {
                Interval::point(initial[1])?.sub(second_quotient)?
            }
            PauliEdgeTopology::PairSource => Interval::point(initial[1])?.add(second_quotient)?,
        };
        if [first, second]
            .into_iter()
            .any(|occupation| occupation.lo < 0.0 || occupation.hi > 1.0)
        {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::StateMapUnresolved,
                "Pauli edge exact affine state cannot be enclosed in the Fermi box",
            ));
        }
        Ok([first, second])
    }

    fn certified_flux_interval(
        &self,
        occupations: [Interval; 2],
    ) -> Result<PauliFluxInterval, PauliEdgeFailure> {
        let [first, second] = occupations;
        if [first, second]
            .into_iter()
            .any(|occupation| occupation.lo < 0.0 || occupation.hi > 1.0)
        {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::StateMapUnresolved,
                "Pauli edge interval leaves the Fermi box",
            ));
        }
        let first_complement = first.complement_unit()?;
        let second_complement = second.complement_unit()?;
        let (gain_factors, loss_factors) = match self.topology {
            PauliEdgeTopology::ElasticTransfer => {
                ([first_complement, second], [first, second_complement])
            }
            PauliEdgeTopology::PairSource => {
                ([first_complement, second_complement], [first, second])
            }
        };
        let product =
            |coefficient: f64, factors: [Interval; 2]| -> Result<Interval, PauliEdgeFailure> {
                Interval::point(coefficient)?
                    .mul_nonnegative(factors[0])?
                    .mul_nonnegative(factors[1])
            };
        let gain = product(self.gain_coefficient_mev, gain_factors)?;
        let loss = product(self.loss_coefficient_mev, loss_factors)?;
        let traffic_upper_bound_mev = gain.add(loss)?.hi;
        if !traffic_upper_bound_mev.is_finite() {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge interval traffic is non-finite",
            ));
        }
        Ok(PauliFluxInterval {
            net_mev: gain.sub(loss)?,
            traffic_upper_bound_mev,
        })
    }

    fn residual_interval(
        &self,
        initial: [f64; 2],
        extent: f64,
        step_mev_inverse: f64,
    ) -> Result<RootResidualInterval, PauliEdgeFailure> {
        let occupations = self.affine_occupation_intervals(initial, extent)?;
        let flux = self.certified_flux_interval(occupations)?;
        let residual =
            Interval::point(extent)?.sub(flux.net_mev.scale_nonnegative(step_mev_inverse)?)?;
        let root_error_abs = residual.lo.abs().max(residual.hi.abs());
        let minimum_measure = self.first_measure.min(self.second_measure);
        let occupation_error_abs = if root_error_abs == 0.0 {
            0.0
        } else {
            (root_error_abs / minimum_measure).next_up()
        };
        if !root_error_abs.is_finite()
            || !occupation_error_abs.is_finite()
            || !flux.traffic_upper_bound_mev.is_finite()
        {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidResidual,
                "Pauli edge exact-real residual interval is invalid",
            ));
        }
        Ok(RootResidualInterval {
            residual,
            root_error_abs,
            occupation_error_abs,
        })
    }

    fn extent_bounds(&self, initial: [f64; 2]) -> Result<(f64, f64), PauliEdgeFailure> {
        let product_upper = |measure: f64, occupation: Interval| {
            Interval::point(measure)?
                .mul_nonnegative(occupation)
                .map(|value| value.hi)
        };
        let first = Interval::point(initial[0])?;
        let second = Interval::point(initial[1])?;
        let first_down = product_upper(self.first_measure, first)?;
        let first_up = product_upper(self.first_measure, first.complement_unit()?)?;
        let (lower, upper) = match self.topology {
            PauliEdgeTopology::ElasticTransfer => (
                -first_down.min(product_upper(
                    self.second_measure,
                    second.complement_unit()?,
                )?),
                first_up.min(product_upper(self.second_measure, second)?),
            ),
            PauliEdgeTopology::PairSource => (
                -first_down.min(product_upper(self.second_measure, second)?),
                first_up.min(product_upper(
                    self.second_measure,
                    second.complement_unit()?,
                )?),
            ),
        };
        if !lower.is_finite() || !upper.is_finite() || lower > 0.0 || upper < 0.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::StateMapUnresolved,
                "Pauli edge physical-capacity interval is invalid",
            ));
        }
        Ok((lower, upper))
    }

    fn flux_derivative_by_extent(&self, occupations: [f64; 2]) -> f64 {
        // LEMMA-R1C-UNIQUE-PHYSICAL-ROOT.  For PairSource both partial
        // derivatives of J and both df/dxi are non-positive/positive,
        // respectively.  For ElasticTransfer, dJ/df1 <= 0, dJ/df2 >= 0,
        // df1/dxi > 0, and df2/dxi < 0.  Therefore dJ/dxi <= 0 for both
        // topologies and r'(xi) = 1 - h*dJ/dxi >= 1 for h >= 0.
        let [first, second] = occupations;
        match self.topology {
            PauliEdgeTopology::ElasticTransfer => {
                let derivative_first = -self.gain_coefficient_mev * second
                    - self.loss_coefficient_mev * (1.0 - second);
                let derivative_second =
                    self.gain_coefficient_mev * (1.0 - first) + self.loss_coefficient_mev * first;
                derivative_first / self.first_measure - derivative_second / self.second_measure
            }
            PauliEdgeTopology::PairSource => {
                let derivative_first = -self.gain_coefficient_mev * (1.0 - second)
                    - self.loss_coefficient_mev * second;
                let derivative_second =
                    -self.gain_coefficient_mev * (1.0 - first) - self.loss_coefficient_mev * first;
                derivative_first / self.first_measure + derivative_second / self.second_measure
            }
        }
    }

    fn try_certify_solved(
        &self,
        initial: [f64; 2],
        extent: f64,
        nonlinear_iterations: usize,
        lower: f64,
        upper: f64,
        current: RootResidual,
    ) -> Result<Option<([f64; 2], PauliEdgeStep)>, PauliEdgeFailure> {
        if current.flux.resolution != PauliFluxResolution::Resolved
            || current.value.abs() > 128.0 * f64::EPSILON * current.scale + MIN_SUBNORMAL
        {
            return Ok(None);
        }
        if current.certified.occupation_error_abs > 128.0 * f64::EPSILON {
            return Ok(None);
        }
        let candidate = self
            .occupations_at_extent(initial, extent)
            .map_err(|message| {
                PauliEdgeFailure::new(PauliEdgeFailureKind::InvalidResidual, message)
            })?;
        Ok(Some((
            candidate,
            PauliEdgeStep {
                kind: Some(PauliEdgeApplicationKind::Solved),
                extent,
                nonlinear_iterations,
                residual_abs: current.value.abs(),
                residual_scale: current.scale,
                traffic_upper_bound_mev: current.flux.traffic_upper_bound_mev,
                flux_abs_error_mev: current.flux.abs_error_bound_mev,
                root_error_abs: current.certified.root_error_abs,
                occupation_error_abs: current.certified.occupation_error_abs,
                max_occupation_bracket_width: maximum_occupation_bracket_width(
                    lower,
                    upper,
                    self.first_measure,
                    self.second_measure,
                ),
                conditioning_lower_bound: (current.flux.net_mev.abs()
                    - current.flux.abs_error_bound_mev)
                    .max(0.0)
                    / current.flux.traffic_upper_bound_mev.max(MIN_SUBNORMAL),
            },
        )))
    }

    pub(crate) fn implicit_step(
        &self,
        step_mev_inverse: f64,
        first_occupation: f64,
        second_occupation: f64,
    ) -> Result<([f64; 2], PauliEdgeStep), PauliEdgeFailure> {
        self.implicit_step_with_limit(step_mev_inverse, first_occupation, second_occupation, 96)
    }

    fn implicit_step_with_limit(
        &self,
        step_mev_inverse: f64,
        first_occupation: f64,
        second_occupation: f64,
        max_iterations: usize,
    ) -> Result<([f64; 2], PauliEdgeStep), PauliEdgeFailure> {
        if !step_mev_inverse.is_finite() || step_mev_inverse < 0.0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidInput,
                "Pauli edge step must be finite and non-negative",
            ));
        }
        if max_iterations == 0 {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::IterationLimit,
                "Pauli edge implicit root did not converge",
            ));
        }
        let initial = [first_occupation, second_occupation];
        if !initial.into_iter().all(valid_occupation) {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidInput,
                "Pauli edge initial occupation is outside [0, 1]",
            ));
        }
        if step_mev_inverse == 0.0 {
            return Ok((initial, zero_step_report()));
        }
        match self.initial_flux_resolution(initial[0], initial[1])? {
            PauliFluxResolution::ExactZero => return Ok((initial, exact_stationary_report())),
            PauliFluxResolution::UnresolvedForCertificate => {
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::UnresolvedFlux,
                    "Pauli edge initial flux is unresolved for certification",
                ));
            }
            PauliFluxResolution::Resolved => {}
        }

        let (lower_capacity, upper_capacity) = self.extent_bounds(initial)?;
        let residual = |extent: f64| -> Result<RootResidual, PauliEdgeFailure> {
            let occupations = self
                .occupations_at_extent(initial, extent)
                .map_err(|message| {
                    PauliEdgeFailure::new(PauliEdgeFailureKind::InvalidResidual, message)
                })?;
            let flux = self.certified_flux_evaluation(occupations[0], occupations[1])?;
            let step_flux = step_mev_inverse * flux.net_mev;
            let value = extent - step_flux;
            let derivative = 1.0 - step_mev_inverse * self.flux_derivative_by_extent(occupations);
            let scale = residual_scale(extent, step_mev_inverse * flux.traffic_upper_bound_mev);
            let certified = self.residual_interval(initial, extent, step_mev_inverse)?;
            if !value.is_finite()
                || !derivative.is_finite()
                || derivative < 1.0
                || !scale.is_finite()
            {
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::InvalidResidual,
                    "Pauli edge implicit residual is invalid",
                ));
            }
            Ok(RootResidual {
                value,
                derivative,
                scale,
                flux,
                certified,
            })
        };

        // The outward-rounded capacity bounds enclose the physical interval.
        // Pauli inward signs give r(lower) <= 0 and r(upper) >= 0 analytically;
        // never evaluate a rounded outer endpoint as a physical state.
        let mut lower = lower_capacity;
        let mut upper = upper_capacity;
        let mut lower_is_sign_certified = false;
        let mut upper_is_sign_certified = false;
        let mut saw_state_map_rejection = false;

        let mut extent = 0.0;
        for iteration in 1..=max_iterations {
            let bracket_before = (lower.to_bits(), upper.to_bits());
            let current = residual(extent)?;
            saw_state_map_rejection |=
                current.certified.occupation_error_abs > 128.0 * f64::EPSILON;
            if let Some(solved) =
                self.try_certify_solved(initial, extent, iteration, lower, upper, current)?
            {
                return Ok(solved);
            }
            let current_is_lower = current.certified.residual.hi <= 0.0;
            let current_is_upper = current.certified.residual.lo >= 0.0;
            if current_is_lower {
                lower = extent;
                lower_is_sign_certified = true;
            } else if current_is_upper {
                upper = extent;
                upper_is_sign_certified = true;
            }
            let midpoint = 0.5 * (lower + upper);
            if (midpoint.to_bits() == lower.to_bits() && !lower_is_sign_certified)
                || (midpoint.to_bits() == upper.to_bits() && !upper_is_sign_certified)
            {
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::StateMapUnresolved,
                    "Pauli edge outer capacity leaves no valid interior candidate",
                ));
            }
            let midpoint_certificate = residual(midpoint)?;
            saw_state_map_rejection |=
                midpoint_certificate.certified.occupation_error_abs > 128.0 * f64::EPSILON;
            if let Some(solved) = self.try_certify_solved(
                initial,
                midpoint,
                iteration,
                lower,
                upper,
                midpoint_certificate,
            )? {
                return Ok(solved);
            }
            let occupation_width = maximum_occupation_bracket_width(
                lower,
                upper,
                self.first_measure,
                self.second_measure,
            );
            if lower < upper && lower.next_up().to_bits() == upper.to_bits() {
                if !lower_is_sign_certified || !upper_is_sign_certified {
                    return Err(PauliEdgeFailure::new(
                        PauliEdgeFailureKind::StateMapUnresolved,
                        "Pauli edge outer capacity leaves no sign-certified adjacent bracket",
                    ));
                }
                let lower_certificate = residual(lower)?;
                saw_state_map_rejection |=
                    lower_certificate.certified.occupation_error_abs > 128.0 * f64::EPSILON;
                if let Some(solved) = self.try_certify_solved(
                    initial,
                    lower,
                    iteration,
                    lower,
                    upper,
                    lower_certificate,
                )? {
                    return Ok(solved);
                }
                let upper_certificate = residual(upper)?;
                saw_state_map_rejection |=
                    upper_certificate.certified.occupation_error_abs > 128.0 * f64::EPSILON;
                if let Some(solved) = self.try_certify_solved(
                    initial,
                    upper,
                    iteration,
                    lower,
                    upper,
                    upper_certificate,
                )? {
                    return Ok(solved);
                }
                if let Some(kind) =
                    exhausted_interval_failure_kind(lower, upper, occupation_width, true)
                {
                    return Err(PauliEdgeFailure::new(
                        kind,
                        "Pauli edge adjacent extents cannot attain the occupation certificate",
                    ));
                }
                if saw_state_map_rejection {
                    return Err(PauliEdgeFailure::new(
                        PauliEdgeFailureKind::StateMapUnresolved,
                        "Pauli edge adjacent candidates lack an occupation-map certificate",
                    ));
                }
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::UncertainPhysicalBracket,
                    "Pauli edge adjacent bracket lacks a pointwise root certificate",
                ));
            }
            let newton = extent - current.value / current.derivative;
            let next_extent = if !current_is_lower && !current_is_upper {
                midpoint
            } else if newton > lower && newton < upper && newton.is_finite() {
                newton
            } else {
                midpoint
            };
            if iteration < max_iterations
                && next_extent.to_bits() == extent.to_bits()
                && bracket_before == (lower.to_bits(), upper.to_bits())
            {
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::StagnatedInterval,
                    "Pauli edge extent and root bracket stagnated",
                ));
            }
            extent = next_extent;
        }
        if occupation_bracket_is_certified(lower, upper, self.first_measure, self.second_measure) {
            Err(PauliEdgeFailure::new(
                if saw_state_map_rejection {
                    PauliEdgeFailureKind::StateMapUnresolved
                } else {
                    PauliEdgeFailureKind::UncertainPhysicalBracket
                },
                "Pauli edge floating bracket lacks a pointwise occupation certificate",
            ))
        } else {
            Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::IterationLimit,
                "Pauli edge implicit root did not converge",
            ))
        }
    }

    pub(crate) fn apply_implicit(
        &self,
        step_mev_inverse: f64,
        occupations: &mut [f64],
    ) -> Result<PauliEdgeStep, PauliEdgeFailure> {
        if self.first_node >= occupations.len() || self.second_node >= occupations.len() {
            return Err(PauliEdgeFailure::new(
                PauliEdgeFailureKind::InvalidInput,
                "Pauli edge node is outside the occupation bank",
            ));
        }
        let initial = [occupations[self.first_node], occupations[self.second_node]];
        let (candidate, report) = self.implicit_step(step_mev_inverse, initial[0], initial[1])?;
        if self.first_node == self.second_node {
            if self.topology != PauliEdgeTopology::PairSource
                || candidate[0].to_bits() != candidate[1].to_bits()
            {
                return Err(PauliEdgeFailure::new(
                    PauliEdgeFailureKind::InvalidResidual,
                    "same-node Pauli pair step is inconsistent",
                ));
            }
            occupations[self.first_node] = candidate[0];
        } else {
            occupations[self.first_node] = candidate[0];
            occupations[self.second_node] = candidate[1];
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relative(left: f64, right: f64) -> f64 {
        (left - right).abs() / left.abs().max(right.abs()).max(f64::MIN_POSITIVE)
    }

    fn assert_root_certificate(report: PauliEdgeStep) {
        assert!(
            report.residual_abs <= 128.0 * f64::EPSILON * report.residual_scale + f64::MIN_POSITIVE
        );
        assert!(
            report.occupation_error_abs <= 128.0 * f64::EPSILON,
            "occupation_error_abs={:.17e} bracket_width={:.17e}",
            report.occupation_error_abs,
            report.max_occupation_bracket_width,
        );
        assert!(report.nonlinear_iterations <= 96);
    }

    fn assert_root_certificate_against_golden(
        report: PauliEdgeStep,
        golden: f64,
        min_measure: f64,
    ) {
        let actual = (report.extent - golden).abs() / min_measure;
        assert!(actual <= 128.0 * f64::EPSILON);
        assert!(report.occupation_error_abs >= actual);
        assert!(report.occupation_error_abs <= 128.0 * f64::EPSILON);
        assert_root_certificate(report);
    }

    fn assert_local_golden_ladder() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        for (exponent, golden_bits) in [
            (8_i32, 0xbf6e_4ad0_bfc2_b909),
            (14, 0xbf0f_8e82_450e_ec7e),
            (20, 0xbeaf_93c8_2712_ec39),
            (30, 0xbe0f_93dd_9299_eefb),
        ] {
            let (_, report) = edge
                .implicit_step(2.0_f64.powi(-exponent), 0.23, 0.79)
                .unwrap();
            assert!(report.nonlinear_iterations <= 4, "h=2^-{exponent}");
            assert_eq!(report.kind, Some(PauliEdgeApplicationKind::Solved));
            assert_root_certificate_against_golden(
                report,
                f64::from_bits(golden_bits),
                edge.first_measure.min(edge.second_measure),
            );
        }
    }

    #[test]
    fn root_cap_is_an_error_not_a_midpoint_success() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::ElasticTransfer,
            0,
            1,
            1.0,
            1.0,
            10.0,
            1.0,
        )
        .unwrap();
        assert_eq!(
            edge.implicit_step_with_limit(1.0, 0.2, 0.8, 1)
                .unwrap_err()
                .kind,
            PauliEdgeFailureKind::IterationLimit
        );
    }

    #[test]
    fn affine_state_rounding_cannot_produce_false_solved_bracket() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            0.125,
            2.0_f64.powi(-27),
            262_144.0,
            0.03125,
        )
        .unwrap();

        let failure = edge
            .implicit_step(4.0, 1.0 - 2.0_f64.powi(-30), 0.25)
            .unwrap_err();
        assert!(
            matches!(
                failure.kind,
                PauliEdgeFailureKind::UncertainPhysicalBracket
                    | PauliEdgeFailureKind::UnresolvedFlux
                    | PauliEdgeFailureKind::StagnatedInterval
            ),
            "unexpected failure kind: {failure:?}"
        );
    }

    #[test]
    fn power_of_two_state_map_rounding_never_false_solves() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            2.0_f64.powi(8),
            2.0_f64.powi(-36),
            2.0_f64.powi(26),
            2.0_f64.powi(-14),
        )
        .unwrap();
        let result = edge.implicit_step(
            2.0_f64.powi(-36),
            1.0 - 2.0_f64.powi(-40),
            1.0 - 2.0_f64.powi(-6),
        );
        let golden = f64::from_bits(0xbcce_ff07_e8a3_8d5c);
        let known_bad = 0xbcce_ff08_07bf_a264_u64;
        let min_measure = 2.0_f64.powi(-36);
        match result {
            Ok((_, report)) => {
                assert_ne!(report.extent.to_bits(), known_bad);
                let actual_occupation_error = (report.extent - golden).abs() / min_measure;
                assert!(actual_occupation_error <= 128.0 * f64::EPSILON);
                assert!(report.occupation_error_abs >= actual_occupation_error);
            }
            Err(error) => assert!(matches!(
                error.kind,
                PauliEdgeFailureKind::UncertainPhysicalBracket
                    | PauliEdgeFailureKind::CertificateUnattainableAtStep
                    | PauliEdgeFailureKind::StateMapUnresolved
                    | PauliEdgeFailureKind::StagnatedInterval
            )),
        }
    }

    #[test]
    fn repeated_extent_fails_immediately_as_stagnated_interval() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::ElasticTransfer,
            0,
            1,
            2.0_f64.powi(-30),
            2.0_f64.powi(-30),
            2.0_f64.powi(-20),
            2.0_f64.powi(-20),
        )
        .unwrap();
        let failure = edge.implicit_step(1.0, 1.0 / 8.0, 1.0 / 4.0).unwrap_err();
        assert_eq!(failure.kind, PauliEdgeFailureKind::StagnatedInterval);
    }

    #[test]
    fn adjacent_uncertifiable_root_interval_has_typed_outcome() {
        let lower = -1.0_f64;
        let upper = lower.next_up();
        assert_eq!(
            exhausted_interval_failure_kind(lower, upper, 129.0 * f64::EPSILON, true,),
            Some(PauliEdgeFailureKind::CertificateUnattainableAtStep)
        );
    }

    #[test]
    fn resolved_direct_certificate_does_not_call_log_value_path() {
        FLUX_VALUE_EVALUATIONS.with(|count| count.set(0));
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let evaluation = edge.certified_flux_evaluation(0.23, 0.79).unwrap();
        assert_eq!(evaluation.resolution, PauliFluxResolution::Resolved);
        assert_eq!(FLUX_VALUE_EVALUATIONS.with(Cell::get), 0);
    }

    #[test]
    fn exact_balance_is_explicitly_blocked_until_r1e() {
        // BLOCKED_EQUILIBRIUM_STEP_SEMANTICS: PR-ODE-R1E owns any future
        // CertifiedFrozen evolution semantics. PR1 must remain fail-closed.
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 1.0, 1.0, 1.0, 1.0).unwrap();
        let failure = edge.implicit_step(0.25, 0.5, 0.5).unwrap_err();
        assert_eq!(failure.kind, PauliEdgeFailureKind::UnresolvedFlux);
    }

    #[test]
    fn tiny_extent_uses_occupation_scaled_certificate() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            1.0e-20,
            2.0e-20,
            3.0e-30,
            1.0e-30,
        )
        .unwrap();
        let (candidate, report) = edge.implicit_step(1.0, 0.2, 0.7).unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        assert!(report.extent.abs() < 1.0e-20);
        assert_root_certificate(report);
    }

    #[test]
    fn successful_edge_step_carries_a_residual_certificate() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0).unwrap();
        let (candidate, report) = edge.implicit_step(0.25, 0.23, 0.79).unwrap();
        let residual = report.extent - 0.25 * edge.flux_mev(candidate[0], candidate[1]).unwrap();
        assert!(residual.abs() <= report.root_error_abs);
        assert_root_certificate(report);
    }

    #[test]
    fn flux_is_inward_on_every_topological_boundary() {
        for topology in [
            PauliEdgeTopology::ElasticTransfer,
            PauliEdgeTopology::PairSource,
        ] {
            let edge = PauliEdge::new(topology, 0, 1, 2.0, 3.0, 7.0, 11.0).unwrap();
            for other in [0.0, 0.23, 1.0] {
                assert!(edge.flux_mev(0.0, other).unwrap() >= 0.0);
                assert!(edge.flux_mev(1.0, other).unwrap() <= 0.0);
                match topology {
                    PauliEdgeTopology::ElasticTransfer => {
                        assert!(edge.flux_mev(other, 0.0).unwrap() <= 0.0);
                        assert!(edge.flux_mev(other, 1.0).unwrap() >= 0.0);
                    }
                    PauliEdgeTopology::PairSource => {
                        assert!(edge.flux_mev(other, 0.0).unwrap() >= 0.0);
                        assert!(edge.flux_mev(other, 1.0).unwrap() <= 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn flux_derivative_is_nonpositive_on_physical_box() {
        for topology in [
            PauliEdgeTopology::ElasticTransfer,
            PauliEdgeTopology::PairSource,
        ] {
            for measures in [
                [2.0_f64.powi(-20), 3.0],
                [2.0, 5.0],
                [7.0, 2.0_f64.powi(20)],
            ] {
                for coefficients in [
                    [0.0, 0.0],
                    [2.0_f64.powi(-30), 11.0],
                    [13.0, 2.0_f64.powi(20)],
                ] {
                    let edge = PauliEdge::new(
                        topology,
                        0,
                        1,
                        measures[0],
                        measures[1],
                        coefficients[0],
                        coefficients[1],
                    )
                    .unwrap();
                    for first in [0.0, 0.125, 0.5, 0.875, 1.0] {
                        for second in [0.0, 0.25, 0.75, 1.0] {
                            let derivative = edge.flux_derivative_by_extent([first, second]);
                            assert!(derivative <= 0.0, "dJ/dxi={derivative:.17e}");
                            for step in [0.0, 2.0_f64.powi(-20), 1.0, 2.0_f64.powi(20)] {
                                assert!(1.0 - step * derivative >= 1.0);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn log_scaled_difference_survives_extreme_tail_and_near_balance() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::ElasticTransfer,
            0,
            1,
            1.0,
            1.0,
            1.0,
            1.0 + 2.0e-12,
        )
        .unwrap();
        let flux = edge.flux_mev(0.5, 0.5).unwrap();
        assert!(flux < 0.0);
        assert!(relative(flux, -5.0e-13) < 5.0e-5);

        let tail = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            1.0,
            1.0,
            3.0e-280,
            2.0e-300,
        )
        .unwrap();
        let tail_flux = tail.flux_mev(1.0e-20, 1.0e-30).unwrap();
        assert!(tail_flux.is_finite() && tail_flux > 0.0);
    }

    #[test]
    fn nonzero_step_rejects_positive_traffic_zero_net_as_unresolved() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 1.0, 1.0, 1.0, 1.0).unwrap();
        assert!(edge.implicit_step(0.25, 0.5, 0.5).is_err());
    }

    #[test]
    fn both_zero_products_are_explicit_exact_stationary() {
        let edge = PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 1.0, 1.0, 0.0, 0.0).unwrap();
        let (_, report) = edge.implicit_step(0.25, 0.2, 0.7).unwrap();
        assert_eq!(report.residual_scale, f64::from_bits(1));
    }

    #[test]
    fn zero_step_is_distinct_from_nonzero_exact_stationary() {
        let edge = PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 1.0, 1.0, 0.0, 0.0).unwrap();
        let (_, zero_step) = edge.implicit_step(0.0, 0.2, 0.7).unwrap();
        let (_, exact_stationary) = edge.implicit_step(0.25, 0.2, 0.7).unwrap();
        assert_ne!(zero_step, exact_stationary);
    }

    #[test]
    fn current_newton_iterate_closes_without_full_bracket_collapse() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0).unwrap();
        let (_, report) = edge.implicit_step(2.0_f64.powi(-8), 0.23, 0.79).unwrap();
        assert!(report.nonlinear_iterations <= 4);
        assert!(report.max_occupation_bracket_width > 128.0 * f64::EPSILON);
        assert_root_certificate(report);
    }

    #[test]
    fn root_certificates_hold_across_extreme_step_scales() {
        assert_local_golden_ladder();
    }

    #[test]
    fn reported_interval_covers_high_precision_root_on_power_fixtures() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            2.0_f64.powi(8),
            2.0_f64.powi(-36),
            2.0_f64.powi(26),
            2.0_f64.powi(-14),
        )
        .unwrap();
        if let Ok((_, report)) = edge.implicit_step(
            2.0_f64.powi(-36),
            1.0 - 2.0_f64.powi(-40),
            1.0 - 2.0_f64.powi(-6),
        ) {
            assert_root_certificate_against_golden(
                report,
                f64::from_bits(0xbcce_ff07_e8a3_8d5c),
                2.0_f64.powi(-36),
            );
        }
        assert_local_golden_ladder();
    }

    #[test]
    fn positive_interval_underflow_encloses_zero_and_positive_upper() {
        let product = Interval::point(MIN_SUBNORMAL)
            .unwrap()
            .mul_nonnegative(Interval::point(0.5).unwrap())
            .unwrap();
        assert_eq!(product.lo, 0.0);
        assert!(product.hi > 0.0);
    }

    #[test]
    fn every_solved_path_carries_true_root_interval() {
        let fixtures = [
            (
                PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0)
                    .unwrap(),
                0.25,
                [0.23, 0.79],
            ),
            (
                PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap(),
                2.0_f64.powi(-14),
                [0.23, 0.79],
            ),
        ];
        for (edge, step, initial) in fixtures {
            let (_, report) = edge.implicit_step(step, initial[0], initial[1]).unwrap();
            assert_eq!(report.kind, Some(PauliEdgeApplicationKind::Solved));
            let certified = edge
                .residual_interval(initial, report.extent, step)
                .unwrap();
            assert_eq!(
                report.root_error_abs.to_bits(),
                certified.root_error_abs.to_bits()
            );
            assert_eq!(
                report.occupation_error_abs.to_bits(),
                certified.occupation_error_abs.to_bits()
            );
        }
    }

    #[test]
    fn uncertain_flux_cannot_be_hidden_by_bracket_collapse() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 1.0, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(
            edge.implicit_step(0.25, 0.5, 0.5).unwrap_err().kind,
            PauliEdgeFailureKind::UnresolvedFlux
        );
    }

    #[test]
    fn physical_capacity_bracket_uses_analytic_inward_signs() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            1.0,
            1.0,
            1.0e-308,
            2.0e-308,
        )
        .unwrap();
        let (_, report) = edge.implicit_step(1.0, 0.5, 0.5).unwrap();
        assert_eq!(report.kind, Some(PauliEdgeApplicationKind::Solved));
        assert_root_certificate(report);
    }

    #[test]
    fn direct_product_certificate_resolves_well_conditioned_flux() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0).unwrap();
        let evaluation = edge.certified_flux_evaluation(0.23, 0.79).unwrap();
        assert_eq!(evaluation.resolution, PauliFluxResolution::Resolved);
        assert!(evaluation.traffic_upper_bound_mev > evaluation.abs_error_bound_mev);
    }

    #[test]
    fn near_balance_is_unresolved_when_net_is_below_arithmetic_bound() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::ElasticTransfer,
            0,
            1,
            1.0,
            1.0,
            1.0,
            1.0_f64.next_up(),
        )
        .unwrap();
        let evaluation = edge.certified_flux_evaluation(0.5, 0.5).unwrap();
        assert_eq!(
            evaluation.resolution,
            PauliFluxResolution::UnresolvedForCertificate
        );
        assert!(evaluation.net_mev.abs() <= evaluation.abs_error_bound_mev);
    }

    #[test]
    fn extreme_log_only_flux_remains_value_available_but_uncertified() {
        let edge = PauliEdge::new(
            PauliEdgeTopology::PairSource,
            0,
            1,
            1.0,
            1.0,
            3.0e-280,
            2.0e-300,
        )
        .unwrap();
        assert!(edge.flux_mev(1.0e-20, 1.0e-30).unwrap().is_finite());
        assert_eq!(
            edge.certified_flux_evaluation(1.0e-20, 1.0e-30)
                .unwrap()
                .resolution,
            PauliFluxResolution::UnresolvedForCertificate
        );
    }

    #[test]
    fn resolved_direct_and_value_only_flux_agree_within_the_reported_bound() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let evaluation = edge.certified_flux_evaluation(0.23, 0.79).unwrap();
        assert_eq!(evaluation.resolution, PauliFluxResolution::Resolved);
        assert!(
            (edge.flux_mev(0.23, 0.79).unwrap() - evaluation.net_mev).abs()
                <= evaluation.abs_error_bound_mev
        );
    }

    #[test]
    fn stiff_elastic_step_preserves_box_number_and_backward_euler_residual() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0).unwrap();
        let initial = [1.0e-35, 1.0 - 2.0e-15];
        let invariant = 2.0 * initial[0] + 5.0 * initial[1];
        let step_mev_inverse = 8.0;
        let (candidate, report) = edge
            .implicit_step(step_mev_inverse, initial[0], initial[1])
            .unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        assert!(report.extent > 0.0);
        assert!(relative(2.0 * candidate[0] + 5.0 * candidate[1], invariant) < 3.0e-16);
        let residual =
            report.extent - step_mev_inverse * edge.flux_mev(candidate[0], candidate[1]).unwrap();
        assert!(residual.abs() <= report.root_error_abs);
        assert_root_certificate(report);
    }

    #[test]
    fn pair_step_preserves_box_cp_difference_and_backward_euler_residual() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let initial = [0.23, 0.79];
        let invariant = 2.0 * initial[0] - 5.0 * initial[1];
        let step_mev_inverse = 2.0_f64.powi(-8);
        let (candidate, report) = edge
            .implicit_step(step_mev_inverse, initial[0], initial[1])
            .unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        let residual = (2.0 * candidate[0] - 5.0 * candidate[1] - invariant).abs();
        assert!(residual <= 8.0 * f64::EPSILON * 5.0);
        let backward_euler =
            report.extent - step_mev_inverse * edge.flux_mev(candidate[0], candidate[1]).unwrap();
        assert!(backward_euler.abs() <= report.root_error_abs);
        assert_root_certificate(report);
    }

    #[test]
    fn stiff_pair_step_without_state_map_certificate_fails_closed() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let failure = edge.implicit_step(8.0, 1.0e-35, 2.0e-31).unwrap_err();
        assert_eq!(failure.kind, PauliEdgeFailureKind::StateMapUnresolved);
    }

    #[test]
    fn same_node_pair_step_and_invalid_input_are_transactional() {
        let edge = PauliEdge::new(PauliEdgeTopology::PairSource, 0, 0, 3.0, 3.0, 5.0, 2.0).unwrap();
        let mut occupation = [0.2];
        edge.apply_implicit(4.0, &mut occupation).unwrap();
        assert!(occupation[0] > 0.2 && occupation[0] < 1.0);

        let before = occupation;
        assert!(edge.apply_implicit(f64::NAN, &mut occupation).is_err());
        assert_eq!(occupation, before);
        assert!(
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 0, 1.0, 1.0, 1.0, 1.0,).is_err()
        );
    }
}
