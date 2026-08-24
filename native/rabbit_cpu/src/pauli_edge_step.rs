//! Positivity-preserving local steps for conservative Pauli collision edges.
//!
//! The state variable is the raw occupation, not a clipped or projected
//! surrogate.  Each edge carries non-negative forward and reverse
//! coefficients.  A backward-Euler extent solve is one dimensional,
//! bracketed by the exact Pauli capacities, and therefore remains inside the
//! Fermi box for arbitrarily stiff finite steps.  The caller owns operator
//! splitting across edges and commits only a completely validated candidate.

#![cfg_attr(not(test), allow(dead_code))]

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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PauliEdgeStep {
    pub(crate) extent: f64,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) residual_abs: f64,
    pub(crate) residual_scale: f64,
    pub(crate) max_occupation_bracket_width: f64,
}

fn valid_occupation(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
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

fn residual_is_certified(extent: f64, step_flux: f64, residual: f64) -> bool {
    let scale = extent.abs().max(step_flux.abs()).max(f64::MIN_POSITIVE);
    residual.abs() <= 128.0 * f64::EPSILON * scale + f64::MIN_POSITIVE
}

fn residual_scale(extent: f64, step_flux: f64) -> f64 {
    extent.abs().max(step_flux.abs()).max(f64::MIN_POSITIVE)
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
        if !valid_occupation(first_occupation) || !valid_occupation(second_occupation) {
            return Err("Pauli edge occupation is outside [0, 1]");
        }
        let (gain_factors, loss_factors) = match self.topology {
            PauliEdgeTopology::ElasticTransfer => (
                [1.0 - first_occupation, second_occupation],
                [first_occupation, 1.0 - second_occupation],
            ),
            PauliEdgeTopology::PairSource => (
                [1.0 - first_occupation, 1.0 - second_occupation],
                [first_occupation, second_occupation],
            ),
        };
        stable_nonnegative_product_difference(
            self.gain_coefficient_mev,
            gain_factors,
            self.loss_coefficient_mev,
            loss_factors,
        )
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

    fn extent_bounds(&self, initial: [f64; 2]) -> (f64, f64) {
        let first_down = self.first_measure * initial[0];
        let first_up = self.first_measure * (1.0 - initial[0]);
        match self.topology {
            PauliEdgeTopology::ElasticTransfer => (
                -first_down.min(self.second_measure * (1.0 - initial[1])),
                first_up.min(self.second_measure * initial[1]),
            ),
            PauliEdgeTopology::PairSource => (
                -first_down.min(self.second_measure * initial[1]),
                first_up.min(self.second_measure * (1.0 - initial[1])),
            ),
        }
    }

    fn flux_derivative_by_extent(&self, occupations: [f64; 2]) -> f64 {
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

    pub(crate) fn implicit_step(
        &self,
        step_mev_inverse: f64,
        first_occupation: f64,
        second_occupation: f64,
    ) -> Result<([f64; 2], PauliEdgeStep), &'static str> {
        self.implicit_step_with_limit(step_mev_inverse, first_occupation, second_occupation, 96)
    }

    fn implicit_step_with_limit(
        &self,
        step_mev_inverse: f64,
        first_occupation: f64,
        second_occupation: f64,
        max_iterations: usize,
    ) -> Result<([f64; 2], PauliEdgeStep), &'static str> {
        if !step_mev_inverse.is_finite() || step_mev_inverse < 0.0 {
            return Err("Pauli edge step must be finite and non-negative");
        }
        if max_iterations == 0 {
            return Err("Pauli edge implicit root did not converge");
        }
        let initial = [first_occupation, second_occupation];
        if !initial.into_iter().all(valid_occupation) {
            return Err("Pauli edge initial occupation is outside [0, 1]");
        }
        let initial_flux = self.flux_mev(initial[0], initial[1])?;
        if step_mev_inverse == 0.0 || initial_flux == 0.0 {
            return Ok((initial, PauliEdgeStep::default()));
        }

        let (lower_capacity, upper_capacity) = self.extent_bounds(initial);
        let residual = |extent: f64| -> Result<(f64, f64, f64), &'static str> {
            let occupations = self.occupations_at_extent(initial, extent)?;
            let flux = self.flux_mev(occupations[0], occupations[1])?;
            let step_flux = step_mev_inverse * flux;
            let value = extent - step_flux;
            let derivative = 1.0 - step_mev_inverse * self.flux_derivative_by_extent(occupations);
            if !value.is_finite() || !derivative.is_finite() || derivative < 1.0 {
                return Err("Pauli edge implicit residual is invalid");
            }
            Ok((value, derivative, step_flux))
        };

        // Capacity endpoints can round one ulp outside [0,1] when converted
        // back from weighted extent. Move only the root bracket one
        // representable value inward; no candidate occupation is clipped.
        let mut lower = if lower_capacity < 0.0 {
            lower_capacity.next_up()
        } else {
            lower_capacity
        };
        let mut upper = if upper_capacity > 0.0 {
            upper_capacity.next_down()
        } else {
            upper_capacity
        };
        let (lower_residual, _, lower_step_flux) = residual(lower)?;
        let (upper_residual, _, upper_step_flux) = residual(upper)?;
        let lower_tolerance =
            128.0 * f64::EPSILON * residual_scale(lower, lower_step_flux) + f64::MIN_POSITIVE;
        let upper_tolerance =
            128.0 * f64::EPSILON * residual_scale(upper, upper_step_flux) + f64::MIN_POSITIVE;
        if lower_residual > lower_tolerance || upper_residual < -upper_tolerance {
            return Err("Pauli edge implicit root is not bracketed by physical capacities");
        }

        let mut extent = 0.0;
        for iteration in 1..=max_iterations {
            let (value, derivative, _) = residual(extent)?;
            if value <= 0.0 {
                lower = extent;
            } else {
                upper = extent;
            }
            let midpoint = 0.5 * (lower + upper);
            let (midpoint_residual, _, midpoint_step_flux) = residual(midpoint)?;
            let occupation_width = maximum_occupation_bracket_width(
                lower,
                upper,
                self.first_measure,
                self.second_measure,
            );
            if occupation_bracket_is_certified(
                lower,
                upper,
                self.first_measure,
                self.second_measure,
            ) && residual_is_certified(midpoint, midpoint_step_flux, midpoint_residual)
            {
                let candidate = self.occupations_at_extent(initial, midpoint)?;
                return Ok((
                    candidate,
                    PauliEdgeStep {
                        extent: midpoint,
                        nonlinear_iterations: iteration,
                        residual_abs: midpoint_residual.abs(),
                        residual_scale: residual_scale(midpoint, midpoint_step_flux),
                        max_occupation_bracket_width: occupation_width,
                    },
                ));
            }
            let newton = extent - value / derivative;
            extent = if newton > lower && newton < upper && newton.is_finite() {
                newton
            } else {
                midpoint
            };
        }
        Err("Pauli edge implicit root did not converge")
    }

    pub(crate) fn apply_implicit(
        &self,
        step_mev_inverse: f64,
        occupations: &mut [f64],
    ) -> Result<PauliEdgeStep, &'static str> {
        if self.first_node >= occupations.len() || self.second_node >= occupations.len() {
            return Err("Pauli edge node is outside the occupation bank");
        }
        let initial = [occupations[self.first_node], occupations[self.second_node]];
        let (candidate, report) = self.implicit_step(step_mev_inverse, initial[0], initial[1])?;
        if self.first_node == self.second_node {
            if self.topology != PauliEdgeTopology::PairSource
                || candidate[0].to_bits() != candidate[1].to_bits()
            {
                return Err("same-node Pauli pair step is inconsistent");
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
        assert!(report.max_occupation_bracket_width <= 128.0 * f64::EPSILON);
        assert!(report.nonlinear_iterations <= 96);
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
            edge.implicit_step_with_limit(1.0, 0.2, 0.8, 1).unwrap_err(),
            "Pauli edge implicit root did not converge"
        );
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
        assert!(relative(residual.abs(), report.residual_abs) <= 4.0 * f64::EPSILON);
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
        assert!(relative(residual.abs(), report.residual_abs) <= 4.0 * f64::EPSILON);
        assert_root_certificate(report);
    }

    #[test]
    fn stiff_pair_step_preserves_box_cp_difference_and_backward_euler_residual() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let initial = [1.0e-35, 2.0e-31];
        let invariant = 2.0 * initial[0] - 5.0 * initial[1];
        let step_mev_inverse = 8.0;
        let (candidate, report) = edge
            .implicit_step(step_mev_inverse, initial[0], initial[1])
            .unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        assert!(report.extent > 0.0);
        let residual = (2.0 * candidate[0] - 5.0 * candidate[1] - invariant).abs();
        assert!(residual <= 8.0 * f64::EPSILON * 5.0);
        let backward_euler =
            report.extent - step_mev_inverse * edge.flux_mev(candidate[0], candidate[1]).unwrap();
        assert!(relative(backward_euler.abs(), report.residual_abs) <= 4.0 * f64::EPSILON);
        assert_root_certificate(report);
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
