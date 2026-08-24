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
    let gain = gain_coefficient * gain_factors[0] * gain_factors[1];
    let loss = loss_coefficient * loss_factors[0] * loss_factors[1];
    if !gain.is_finite() || gain < 0.0 || !loss.is_finite() || loss < 0.0 {
        return Err("Pauli edge gain or loss is invalid");
    }
    if gain == 0.0 || loss == 0.0 {
        return Ok(gain - loss);
    }
    let direct = gain - loss;
    let scale = gain.max(loss);
    if direct.abs() > 1.0e-6 * scale {
        return Ok(direct);
    }

    // Near detailed balance, evaluate the affinity before subtraction.  This
    // avoids destroying a small physical net by cancelling two positive rates.
    let log_gain = gain_coefficient.ln() + gain_factors[0].ln() + gain_factors[1].ln();
    let log_loss = loss_coefficient.ln() + loss_factors[0].ln() + loss_factors[1].ln();
    let stable = if log_gain >= log_loss {
        log_gain.exp() * -((log_loss - log_gain).exp_m1())
    } else {
        log_loss.exp() * (log_gain - log_loss).exp_m1()
    };
    stable
        .is_finite()
        .then_some(stable)
        .ok_or("Pauli edge affinity difference is non-finite")
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
        if !step_mev_inverse.is_finite() || step_mev_inverse < 0.0 {
            return Err("Pauli edge step must be finite and non-negative");
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
        let residual = |extent: f64| -> Result<(f64, f64), &'static str> {
            let occupations = self.occupations_at_extent(initial, extent)?;
            let flux = self.flux_mev(occupations[0], occupations[1])?;
            let value = extent - step_mev_inverse * flux;
            let derivative = 1.0 - step_mev_inverse * self.flux_derivative_by_extent(occupations);
            if !value.is_finite() || !derivative.is_finite() || derivative < 1.0 {
                return Err("Pauli edge implicit residual is invalid");
            }
            Ok((value, derivative))
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
        let lower_residual = residual(lower)?.0;
        let upper_residual = residual(upper)?.0;
        let bracket_scale = lower.abs().max(upper.abs()).max(1.0);
        let bracket_tolerance = 64.0 * f64::EPSILON * bracket_scale;
        if lower_residual > bracket_tolerance || upper_residual < -bracket_tolerance {
            return Err("Pauli edge implicit root is not bracketed by physical capacities");
        }

        let mut extent = 0.0;
        let mut iterations = 0;
        for iteration in 0..32 {
            iterations = iteration + 1;
            let (value, derivative) = residual(extent)?;
            if value <= 0.0 {
                lower = extent;
            } else {
                upper = extent;
            }
            let width = upper - lower;
            if width <= 16.0 * f64::EPSILON * upper.abs().max(lower.abs()).max(1.0) {
                extent = 0.5 * (lower + upper);
                break;
            }
            let newton = extent - value / derivative;
            extent = if newton > lower && newton < upper && newton.is_finite() {
                newton
            } else {
                0.5 * (lower + upper)
            };
        }
        if iterations == 32 {
            extent = 0.5 * (lower + upper);
        }
        let candidate = self.occupations_at_extent(initial, extent)?;
        Ok((
            candidate,
            PauliEdgeStep {
                extent,
                nonlinear_iterations: iterations,
            },
        ))
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
    fn affinity_difference_resolves_near_cancellation_and_extreme_tail() {
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
    fn stiff_elastic_step_preserves_box_and_weighted_number() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::ElasticTransfer, 0, 1, 2.0, 5.0, 3.0, 7.0).unwrap();
        let initial = [1.0e-35, 1.0 - 2.0e-15];
        let invariant = 2.0 * initial[0] + 5.0 * initial[1];
        let (candidate, report) = edge.implicit_step(1.0e30, initial[0], initial[1]).unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        assert!(report.extent > 0.0 && report.nonlinear_iterations <= 32);
        assert!(relative(2.0 * candidate[0] + 5.0 * candidate[1], invariant) < 3.0e-16);
    }

    #[test]
    fn stiff_pair_step_preserves_box_and_cp_difference() {
        let edge =
            PauliEdge::new(PauliEdgeTopology::PairSource, 0, 1, 2.0, 5.0, 13.0, 17.0).unwrap();
        let initial = [1.0e-35, 2.0e-31];
        let invariant = 2.0 * initial[0] - 5.0 * initial[1];
        let (candidate, report) = edge.implicit_step(1.0e30, initial[0], initial[1]).unwrap();
        assert!(candidate.into_iter().all(valid_occupation));
        assert!(report.extent > 0.0 && report.nonlinear_iterations <= 32);
        let residual = (2.0 * candidate[0] - 5.0 * candidate[1] - invariant).abs();
        assert!(residual <= 8.0 * f64::EPSILON * 5.0);
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
