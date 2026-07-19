//! Native stiff-solver substrate for the Rust-first BBN endpoint.
//!
//! BDF is delegated to pinned `diffsol`; Rodas5P preserves the corrected
//! nonautonomous SciML convention used by the frozen NumPy oracle.
#![cfg_attr(not(test), allow(dead_code))]

use diffsol::{
    NalgebraLU, NalgebraMat, OdeBuilder, OdeSolverMethod, OdeSolverStopReason, VectorHost,
};
use nalgebra::{DMatrix, DVector};

const GAMMA: f64 = 0.211_937_563_194_290_14;
const C: [f64; 8] = [
    0.0,
    0.635_812_689_582_870_4,
    0.409_579_839_339_753_5,
    0.976_930_672_506_071_6,
    0.428_840_360_955_866_4,
    1.0,
    1.0,
    1.0,
];
const D: [f64; 8] = [
    GAMMA,
    -0.423_875_126_388_580_27,
    -0.338_462_712_623_592_4,
    1.804_645_287_288_273_4,
    2.325_825_639_765_069,
    0.0,
    0.0,
    0.0,
];
#[rustfmt::skip]
const A: [[f64; 8]; 8] = [
    [0.0; 8],
    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.849394379747939, 0.45842242204463923, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-6.954028509809101, 2.489845061869568, -10.358996098473584, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.8029986275628964, 0.5072464736228206, -0.3988312541770524, -0.04721187230404641, 0.0, 0.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 0.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 0.0, 0.0],
    [-7.502846399306121, 2.561846144803919, -11.627539656261098, -0.18268767659942256, 0.030198172008377946, 1.0, 1.0, 0.0],
];
#[rustfmt::skip]
const G: [[f64; 8]; 8] = [
    [0.0; 8],
    [-14.155112264123755, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-17.97296035885952, -2.859693295451294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [147.12150275711716, -1.41221402718213, 71.68940251302358, 0.0, 0.0, 0.0, 0.0, 0.0],
    [165.43517024871676, -0.4592823456491126, 42.90938336958603, -5.961986721573306, 0.0, 0.0, 0.0, 0.0],
    [24.854864614690072, -3.0009227002832186, 47.4931110020768, 5.5814197821558125, -0.6610691825249471, 0.0, 0.0, 0.0],
    [30.91273214028599, -3.1208243349937974, 77.79954646070892, 34.28646028294783, -19.097331116725623, -28.087943162872662, 0.0, 0.0],
    [37.80277123390563, -3.2571969029072276, 112.26918849496327, 66.9347231244047, -40.06618937091002, -54.66780262877968, -9.48861652309627, 0.0],
];
const B: [f64; 8] = [
    -7.502_846_399_306_121,
    2.561_846_144_803_919,
    -11.627_539_656_261_098,
    -0.182_687_676_599_422_56,
    0.030_198_172_008_377_946,
    1.0,
    1.0,
    1.0,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SolverKind {
    Bdf,
    Rodas5P,
}

#[derive(Clone, Debug)]
pub(crate) struct OdeConfig {
    pub(crate) rtol: f64,
    pub(crate) atol: Vec<f64>,
    pub(crate) h_init: f64,
    pub(crate) h_min: f64,
    pub(crate) h_max: f64,
    pub(crate) max_attempts: usize,
}

impl OdeConfig {
    fn validate(&self, n: usize) -> Result<(), &'static str> {
        if !self.rtol.is_finite() || self.rtol <= 0.0 {
            return Err("rtol must be positive and finite");
        }
        if self.atol.len() != n || self.atol.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return Err("atol must contain one positive finite value per state");
        }
        if !self.h_init.is_finite()
            || !self.h_min.is_finite()
            || !self.h_max.is_finite()
            || self.h_min <= 0.0
            || self.h_init < self.h_min
            || self.h_max < self.h_init
            || self.max_attempts == 0
        {
            return Err("invalid step-size or attempt budget");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OdeResult {
    pub(crate) t: f64,
    pub(crate) y: Vec<f64>,
    pub(crate) accepted: usize,
    pub(crate) rejected: usize,
    pub(crate) jacobian_evaluations: usize,
    pub(crate) linear_setups: usize,
    pub(crate) event_reached: bool,
    pub(crate) failure: Option<String>,
    pub(crate) rhs_evaluations: usize,
    pub(crate) dfdt_evaluations: usize,
}

pub(crate) struct TerminalEvent<'a> {
    pub(crate) value: &'a dyn Fn(f64, &[f64]) -> f64,
    pub(crate) direction: i8,
}

pub(crate) trait OdeSystem {
    fn dimension(&self) -> usize;
    fn state_is_valid(&self, _state: &[f64]) -> bool {
        true
    }
    fn rhs(&self, t: f64, y: &[f64], out: &mut [f64]);
    /// Return the same matrix bits for the same system and bitwise-identical
    /// finite `(t, y)` within one solve. Diagnostic call counting is permitted.
    fn jacobian(&self, t: f64, y: &[f64], out_row_major: &mut [f64]);
    fn dfdt(&self, t: f64, y: &[f64], out: &mut [f64]);
    /// Fused evaluation of `rhs` and `jacobian` at the same `(t, y)`. The
    /// default forwards to the two separate calls; systems whose rhs and
    /// jacobian share expensive intermediate state may override this to
    /// compute both in a single pass. Overrides MUST return bitwise-identical
    /// output to the separate calls.
    fn rhs_and_jacobian(&self, t: f64, y: &[f64], f_out: &mut [f64], jac_out: &mut [f64]) {
        self.rhs(t, y, f_out);
        self.jacobian(t, y, jac_out);
    }
}

struct CountingSystem<'a, S: OdeSystem> {
    inner: &'a S,
    rhs_calls: std::cell::Cell<usize>,
    dfdt_calls: std::cell::Cell<usize>,
}

impl<'a, S: OdeSystem> CountingSystem<'a, S> {
    fn new(inner: &'a S) -> Self {
        Self {
            inner,
            rhs_calls: std::cell::Cell::new(0),
            dfdt_calls: std::cell::Cell::new(0),
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
    fn rhs(&self, t: f64, y: &[f64], out: &mut [f64]) {
        self.rhs_calls.set(self.rhs_calls.get() + 1);
        self.inner.rhs(t, y, out);
    }
    fn jacobian(&self, t: f64, y: &[f64], out: &mut [f64]) {
        self.inner.jacobian(t, y, out); // inner.jacobian's internal self.rhs calls are NOT counted here — intended
    }
    fn dfdt(&self, t: f64, y: &[f64], out: &mut [f64]) {
        self.dfdt_calls.set(self.dfdt_calls.get() + 1);
        self.inner.dfdt(t, y, out);
    }
    fn rhs_and_jacobian(&self, t: f64, y: &[f64], f_out: &mut [f64], jac_out: &mut [f64]) {
        self.rhs_calls.set(self.rhs_calls.get() + 1);
        self.inner.rhs_and_jacobian(t, y, f_out, jac_out);
    }
}

fn exact_jacobian_cache_key_matches(
    cached_t_bits: u64,
    cached_state_bits: &[u64],
    t: f64,
    state: &[f64],
) -> bool {
    t.is_finite()
        && cached_t_bits == t.to_bits()
        && cached_state_bits.len() == state.len()
        && state
            .iter()
            .zip(cached_state_bits)
            .all(|(value, bits)| value.is_finite() && value.to_bits() == *bits)
}

fn apply_dense_row_major_jacobian(jacobian: &[f64], n: usize, vector: &[f64], out: &mut [f64]) {
    for i in 0..n {
        out[i] = (0..n).map(|j| jacobian[i * n + j] * vector[j]).sum();
    }
}

pub(crate) fn solve<S: OdeSystem>(
    kind: SolverKind,
    system: &S,
    t_span: (f64, f64),
    y0: &[f64],
    config: &OdeConfig,
    event: Option<&TerminalEvent<'_>>,
) -> OdeResult {
    let counting = CountingSystem::new(system);
    let mut result = solve_inner(kind, &counting, t_span, y0, config, event);
    result.rhs_evaluations = counting.rhs_calls.get();
    result.dfdt_evaluations = counting.dfdt_calls.get();
    result
}

fn solve_inner<S: OdeSystem>(
    kind: SolverKind,
    system: &S,
    t_span: (f64, f64),
    y0: &[f64],
    config: &OdeConfig,
    event: Option<&TerminalEvent<'_>>,
) -> OdeResult {
    let invalid = || OdeResult {
        t: t_span.0,
        y: y0.to_vec(),
        accepted: 0,
        rejected: 0,
        jacobian_evaluations: 0,
        linear_setups: 0,
        event_reached: false,
        failure: Some("invalid_input".into()),
        rhs_evaluations: 0,
        dfdt_evaluations: 0,
    };
    if system.dimension() != y0.len()
        || y0.is_empty()
        || y0.iter().any(|x| !x.is_finite())
        || !t_span.0.is_finite()
        || !t_span.1.is_finite()
        || t_span.1 <= t_span.0
        || config.validate(y0.len()).is_err()
        || event.is_some_and(|item| !(-1..=1).contains(&item.direction))
    {
        return invalid();
    }
    if !system.state_is_valid(y0) {
        return failed(t_span.0, y0, 0, 0, 0, "invalid_initial_state");
    }
    let initial_event = event.map(|item| (item.value)(t_span.0, y0));
    if initial_event.is_some_and(|value| !value.is_finite()) {
        return failed(t_span.0, y0, 0, 0, 0, "nonfinite_event");
    }
    if initial_event == Some(0.0) {
        return OdeResult {
            t: t_span.0,
            y: y0.to_vec(),
            accepted: 0,
            rejected: 0,
            jacobian_evaluations: 0,
            linear_setups: 0,
            event_reached: true,
            failure: None,
            rhs_evaluations: 0,
            dfdt_evaluations: 0,
        };
    }
    let mut rhs_probe = vec![0.0; y0.len()];
    system.rhs(t_span.0, y0, &mut rhs_probe);
    if rhs_probe.iter().any(|value| !value.is_finite()) {
        return failed(t_span.0, y0, 0, 0, 0, "nonfinite_rhs");
    }
    match kind {
        SolverKind::Bdf => solve_bdf(system, t_span, y0, config, event, initial_event),
        SolverKind::Rodas5P => solve_rodas5p(system, t_span, y0, config, event, initial_event),
    }
}

fn solve_bdf<S: OdeSystem>(
    system: &S,
    t_span: (f64, f64),
    y0: &[f64],
    config: &OdeConfig,
    event: Option<&TerminalEvent<'_>>,
    mut previous_event: Option<f64>,
) -> OdeResult {
    type M = NalgebraMat<f64>;
    type LS = NalgebraLU<f64>;
    let n = y0.len();
    let initial = y0.to_vec();
    let jacobian_evaluations = std::cell::Cell::new(0_usize);
    let jacobian_cache = std::cell::RefCell::new(None::<(u64, Vec<u64>, Vec<f64>)>);
    let nonfinite_rhs = std::cell::Cell::new(false);
    let nonfinite_jacobian = std::cell::Cell::new(false);
    let problem = OdeBuilder::<M>::new()
        .t0(t_span.0)
        .h0(config.h_init)
        .rtol(config.rtol)
        .atol(config.atol.clone())
        .rhs_implicit(
            |x, _p, t, out| {
                system.rhs(t, x.as_slice(), out.as_mut_slice());
                if out.as_slice().iter().any(|value| !value.is_finite()) {
                    nonfinite_rhs.set(true);
                }
            },
            |x, _p, t, v, out| {
                let state = x.as_slice();
                let cache_hit = jacobian_cache.borrow().as_ref().is_some_and(
                    |(cached_t_bits, cached_state_bits, _)| {
                        exact_jacobian_cache_key_matches(
                            *cached_t_bits,
                            cached_state_bits,
                            t,
                            state,
                        )
                    },
                );
                if cache_hit {
                    let cached = jacobian_cache.borrow();
                    let jacobian = &cached
                        .as_ref()
                        .expect("a cache hit must retain its Jacobian")
                        .2;
                    apply_dense_row_major_jacobian(jacobian, n, v.as_slice(), out.as_mut_slice());
                } else {
                    jacobian_cache.borrow_mut().take();
                    jacobian_evaluations.set(jacobian_evaluations.get() + 1);
                    let mut jacobian = vec![0.0; n * n];
                    system.jacobian(t, state, &mut jacobian);
                    let finite_jacobian = jacobian.iter().all(|value| value.is_finite());
                    if !finite_jacobian {
                        nonfinite_jacobian.set(true);
                    }
                    apply_dense_row_major_jacobian(&jacobian, n, v.as_slice(), out.as_mut_slice());
                    if finite_jacobian
                        && t.is_finite()
                        && state.iter().all(|value| value.is_finite())
                    {
                        *jacobian_cache.borrow_mut() = Some((
                            t.to_bits(),
                            state.iter().map(|value| value.to_bits()).collect(),
                            jacobian,
                        ));
                    }
                }
            },
        )
        .init(
            move |_p, _t, out| {
                out.as_mut_slice().copy_from_slice(&initial);
            },
            n,
        )
        .build();
    let Ok(mut problem) = problem else {
        return failed(t_span.0, y0, 0, 0, 0, "bdf_build");
    };
    problem.ode_options.min_timestep = config.h_min;
    problem.ode_options.max_error_test_failures = config.max_attempts;
    problem.ode_options.max_nonlinear_solver_failures = config.max_attempts;
    let mut solver = match problem.bdf::<LS>() {
        Ok(solver) => solver,
        Err(_) => {
            let reason = if nonfinite_rhs.get() {
                "nonfinite_rhs"
            } else if nonfinite_jacobian.get() {
                "nonfinite_jacobian"
            } else {
                "bdf_initialise"
            };
            return failed_with_setups(t_span.0, y0, 0, 0, jacobian_evaluations.get(), 0, reason);
        }
    };
    loop {
        let current_t = solver.state().t;
        let chunk_end = (current_t + config.h_max).min(t_span.1);
        if solver.set_stop_time(chunk_end).is_err() {
            let stats = solver.get_statistics();
            let state = solver.state();
            return failed_with_setups(
                state.t,
                state.y.as_slice(),
                stats.number_of_steps,
                stats.number_of_error_test_failures + stats.number_of_nonlinear_solver_fails,
                jacobian_evaluations.get(),
                stats.number_of_linear_solver_setups,
                "bdf_stop_time",
            );
        }
        loop {
            let before = solver.get_statistics();
            let attempts_before = before.number_of_steps
                + before.number_of_error_test_failures
                + before.number_of_nonlinear_solver_fails;
            if attempts_before >= config.max_attempts {
                let state = solver.state();
                return failed_with_setups(
                    state.t,
                    state.y.as_slice(),
                    before.number_of_steps,
                    before.number_of_error_test_failures + before.number_of_nonlinear_solver_fails,
                    jacobian_evaluations.get(),
                    before.number_of_linear_solver_setups,
                    "max_attempts",
                );
            }
            let step_start = solver.state().t;
            let step = solver.step();
            let stats = solver.get_statistics();
            let attempts = stats.number_of_steps
                + stats.number_of_error_test_failures
                + stats.number_of_nonlinear_solver_fails;
            let (state_t, raw) = {
                let state = solver.state();
                (state.t, state.y.as_slice().to_vec())
            };
            if nonfinite_rhs.get() || nonfinite_jacobian.get() {
                return failed_with_setups(
                    state_t,
                    &raw,
                    stats.number_of_steps,
                    stats.number_of_error_test_failures + stats.number_of_nonlinear_solver_fails,
                    jacobian_evaluations.get(),
                    stats.number_of_linear_solver_setups,
                    if nonfinite_rhs.get() {
                        "nonfinite_rhs"
                    } else {
                        "nonfinite_jacobian"
                    },
                );
            }
            if !system.state_is_valid(&raw) {
                return failed_with_setups(
                    state_t,
                    &raw,
                    stats.number_of_steps,
                    stats.number_of_error_test_failures + stats.number_of_nonlinear_solver_fails,
                    jacobian_evaluations.get(),
                    stats.number_of_linear_solver_setups,
                    "invalid_solution_state",
                );
            }
            if attempts > config.max_attempts {
                return failed_with_setups(
                    state_t,
                    &raw,
                    stats.number_of_steps,
                    stats.number_of_error_test_failures + stats.number_of_nonlinear_solver_fails,
                    jacobian_evaluations.get(),
                    stats.number_of_linear_solver_setups,
                    "max_attempts",
                );
            }
            let reason = match step {
                Ok(reason) => reason,
                Err(_) => {
                    return failed_with_setups(
                        state_t,
                        &raw,
                        stats.number_of_steps,
                        stats.number_of_error_test_failures
                            + stats.number_of_nonlinear_solver_fails,
                        jacobian_evaluations.get(),
                        stats.number_of_linear_solver_setups,
                        "bdf_solve",
                    );
                }
            };
            if matches!(reason, OdeSolverStopReason::RootFound(..)) {
                return failed_with_setups(
                    state_t,
                    &raw,
                    stats.number_of_steps,
                    stats.number_of_error_test_failures + stats.number_of_nonlinear_solver_fails,
                    jacobian_evaluations.get(),
                    stats.number_of_linear_solver_setups,
                    "unexpected_root",
                );
            }
            if let (Some(item), Some(event_left)) = (event, previous_event) {
                let event_right = (item.value)(state_t, &raw);
                if !event_right.is_finite() {
                    return failed_with_setups(
                        state_t,
                        &raw,
                        stats.number_of_steps,
                        stats.number_of_error_test_failures
                            + stats.number_of_nonlinear_solver_fails,
                        jacobian_evaluations.get(),
                        stats.number_of_linear_solver_setups,
                        "nonfinite_event",
                    );
                }
                if event_fires(event_left, event_right, item.direction) {
                    let mut lo = step_start;
                    let mut hi = state_t;
                    let start_positive = event_left > 0.0;
                    let mut event_y = raw.clone();
                    for _ in 0..60 {
                        if event_interval_small(lo, hi, step_start) {
                            break;
                        }
                        let mid = 0.5 * (lo + hi);
                        let Ok(mid_y) = solver.interpolate(mid) else {
                            return failed_with_setups(
                                state_t,
                                &raw,
                                stats.number_of_steps,
                                stats.number_of_error_test_failures
                                    + stats.number_of_nonlinear_solver_fails,
                                jacobian_evaluations.get(),
                                stats.number_of_linear_solver_setups,
                                "event_refinement",
                            );
                        };
                        if !system.state_is_valid(mid_y.as_slice()) {
                            return failed_with_setups(
                                state_t,
                                &raw,
                                stats.number_of_steps,
                                stats.number_of_error_test_failures
                                    + stats.number_of_nonlinear_solver_fails,
                                jacobian_evaluations.get(),
                                stats.number_of_linear_solver_setups,
                                "invalid_event_state",
                            );
                        }
                        let mid_event = (item.value)(mid, mid_y.as_slice());
                        if !mid_event.is_finite() {
                            return failed_with_setups(
                                state_t,
                                &raw,
                                stats.number_of_steps,
                                stats.number_of_error_test_failures
                                    + stats.number_of_nonlinear_solver_fails,
                                jacobian_evaluations.get(),
                                stats.number_of_linear_solver_setups,
                                "nonfinite_event",
                            );
                        }
                        if (mid_event > 0.0) == start_positive {
                            lo = mid;
                        } else {
                            hi = mid;
                            event_y = mid_y.as_slice().to_vec();
                        }
                    }
                    let event_failure = (!system.state_is_valid(&event_y))
                        .then(|| "invalid_event_state".to_string());
                    return OdeResult {
                        t: hi,
                        y: event_y,
                        accepted: stats.number_of_steps,
                        rejected: stats.number_of_error_test_failures
                            + stats.number_of_nonlinear_solver_fails,
                        jacobian_evaluations: jacobian_evaluations.get(),
                        linear_setups: stats.number_of_linear_solver_setups,
                        event_reached: true,
                        failure: event_failure,
                        rhs_evaluations: 0,
                        dfdt_evaluations: 0,
                    };
                }
                previous_event = Some(event_right);
            }
            if reason == OdeSolverStopReason::TstopReached {
                if t_span.1 - state_t <= 8.0 * f64::EPSILON * t_span.1.abs().max(1.0) {
                    return OdeResult {
                        t: state_t,
                        y: raw,
                        accepted: stats.number_of_steps,
                        rejected: stats.number_of_error_test_failures
                            + stats.number_of_nonlinear_solver_fails,
                        jacobian_evaluations: jacobian_evaluations.get(),
                        linear_setups: stats.number_of_linear_solver_setups,
                        event_reached: false,
                        failure: None,
                        rhs_evaluations: 0,
                        dfdt_evaluations: 0,
                    };
                }
                break;
            }
        }
    }
}

fn event_fires(previous: f64, current: f64, direction: i8) -> bool {
    match direction {
        0 => current == 0.0 || previous * current < 0.0,
        value if value < 0 => previous > 0.0 && current <= 0.0,
        _ => previous < 0.0 && current >= 0.0,
    }
}

fn event_interval_small(lo: f64, hi: f64, start: f64) -> bool {
    hi - lo <= 1.0e-15 * start.abs().max(1.0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RodasAttemptError {
    LinearSolve,
    NonFinite(&'static str),
}

impl RodasAttemptError {
    fn reason(self) -> &'static str {
        match self {
            Self::LinearSolve => "linear_solve",
            Self::NonFinite(reason) => reason,
        }
    }
}

struct LinearizationPoint {
    jac: DMatrix<f64>,
    partial_t: DVector<f64>,
    f0: DVector<f64>,
}

fn rodas_attempt<S: OdeSystem>(
    system: &S,
    t: f64,
    y: &DVector<f64>,
    h: f64,
    point: &LinearizationPoint,
    config: &OdeConfig,
) -> Result<(DVector<f64>, f64), RodasAttemptError> {
    let n = y.len();
    let w = DMatrix::identity(n, n) * (1.0 / (GAMMA * h)) - &point.jac;
    let lu = w.lu();
    let partial_t = &point.partial_t;
    let mut stages = Vec::<DVector<f64>>::with_capacity(8);
    for i in 0..8 {
        let mut stage_y = y.clone();
        let mut coupling = DVector::zeros(n);
        for j in 0..i {
            stage_y.axpy(A[i][j], &stages[j], 1.0);
            coupling.axpy(G[i][j] / h, &stages[j], 1.0);
        }
        let rhs_vector = if i == 0 {
            point.f0.clone()
        } else {
            let mut rhs_values = vec![0.0; n];
            system.rhs(t + C[i] * h, stage_y.as_slice(), &mut rhs_values);
            if rhs_values.iter().any(|value| !value.is_finite()) {
                return Err(RodasAttemptError::NonFinite("nonfinite_rhs"));
            }
            DVector::from_vec(rhs_values)
        };
        let rhs = rhs_vector + coupling + partial_t * (h * D[i]);
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(RodasAttemptError::NonFinite("nonfinite_stage_rhs"));
        }
        let stage = lu.solve(&rhs).ok_or(RodasAttemptError::LinearSolve)?;
        if stage.iter().any(|value| !value.is_finite()) {
            return Err(RodasAttemptError::NonFinite("nonfinite_stage"));
        }
        stages.push(stage);
    }
    let mut candidate = y.clone();
    for i in 0..8 {
        candidate.axpy(B[i], &stages[i], 1.0);
    }
    if candidate.iter().any(|value| !value.is_finite()) {
        return Err(RodasAttemptError::NonFinite("nonfinite_candidate"));
    }
    let error = weighted_norm(&stages[7], &candidate, config);
    if !error.is_finite() {
        return Err(RodasAttemptError::NonFinite("nonfinite_error_norm"));
    }
    Ok((candidate, error))
}

fn solve_rodas5p<S: OdeSystem>(
    system: &S,
    t_span: (f64, f64),
    y0: &[f64],
    config: &OdeConfig,
    event: Option<&TerminalEvent<'_>>,
    mut previous_event: Option<f64>,
) -> OdeResult {
    let n = y0.len();
    let mut t = t_span.0;
    let mut y = DVector::from_column_slice(y0);
    let mut h = config.h_init.min(t_span.1 - t);
    let mut accepted = 0;
    let mut rejected = 0;
    let mut previous_error = 1.0;
    let mut jacobian_evaluations = 0;
    let mut linear_setups = 0;
    let mut point: Option<LinearizationPoint> = None;
    for _ in 0..config.max_attempts {
        h = h.min(t_span.1 - t);
        linear_setups += 1;
        if point.is_none() {
            let mut jac_values = vec![0.0; n * n];
            let mut f0_values = vec![0.0; n];
            system.rhs_and_jacobian(t, y.as_slice(), &mut f0_values, &mut jac_values);
            jacobian_evaluations += 1;
            if jac_values.iter().any(|value| !value.is_finite()) {
                return failed_with_setups(
                    t,
                    y.as_slice(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    "nonfinite_jacobian",
                );
            }
            let jac = DMatrix::from_row_slice(n, n, &jac_values);
            let mut partial_t = vec![0.0; n];
            system.dfdt(t, y.as_slice(), &mut partial_t);
            if partial_t.iter().any(|value| !value.is_finite()) {
                return failed_with_setups(
                    t,
                    y.as_slice(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    "nonfinite_dfdt",
                );
            }
            if f0_values.iter().any(|value| !value.is_finite()) {
                return failed_with_setups(
                    t,
                    y.as_slice(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    "nonfinite_rhs",
                );
            }
            point = Some(LinearizationPoint {
                jac,
                partial_t: DVector::from_vec(partial_t),
                f0: DVector::from_vec(f0_values),
            });
        }
        let current_point = point.as_ref().expect("linearization point is set above");
        let attempt = match rodas_attempt(system, t, &y, h, current_point, config) {
            Ok(attempt) => attempt,
            Err(error) => {
                return failed_with_setups(
                    t,
                    y.as_slice(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    error.reason(),
                );
            }
        };
        let error = attempt.1;
        if error <= 1.0 {
            let step_start = t;
            let start_y = y.clone();
            let candidate = attempt.0;
            if !system.state_is_valid(candidate.as_slice()) {
                rejected += 1;
                if h <= config.h_min {
                    return failed_with_setups(
                        t,
                        y.as_slice(),
                        accepted,
                        rejected,
                        jacobian_evaluations,
                        linear_setups,
                        "invalid_candidate_state",
                    );
                }
                h = (0.2 * h).max(config.h_min);
                continue;
            }
            t += h;
            y = candidate;
            accepted += 1;
            if let (Some(item), Some(event_left)) = (event, previous_event) {
                let event_right = (item.value)(t, y.as_slice());
                if !event_right.is_finite() {
                    return failed_with_setups(
                        t,
                        y.as_slice(),
                        accepted,
                        rejected,
                        jacobian_evaluations,
                        linear_setups,
                        "nonfinite_event",
                    );
                }
                if event_fires(event_left, event_right, item.direction) {
                    let mut lo = 0.0;
                    let mut hi = h;
                    let start_positive = event_left > 0.0;
                    let mut event_offset = h;
                    let mut event_y = y.clone();
                    for _ in 0..60 {
                        if event_interval_small(lo, hi, step_start) {
                            break;
                        }
                        let mid = 0.5 * (lo + hi);
                        linear_setups += 1;
                        let Ok((mid_y, _)) =
                            rodas_attempt(system, step_start, &start_y, mid, current_point, config)
                        else {
                            return failed_with_setups(
                                t,
                                y.as_slice(),
                                accepted,
                                rejected,
                                jacobian_evaluations,
                                linear_setups,
                                "event_refinement",
                            );
                        };
                        if !system.state_is_valid(mid_y.as_slice()) {
                            return failed_with_setups(
                                t,
                                y.as_slice(),
                                accepted,
                                rejected,
                                jacobian_evaluations,
                                linear_setups,
                                "invalid_event_state",
                            );
                        }
                        let mid_event = (item.value)(step_start + mid, mid_y.as_slice());
                        if !mid_event.is_finite() {
                            return failed_with_setups(
                                t,
                                y.as_slice(),
                                accepted,
                                rejected,
                                jacobian_evaluations,
                                linear_setups,
                                "nonfinite_event",
                            );
                        }
                        if (mid_event > 0.0) == start_positive {
                            lo = mid;
                        } else {
                            hi = mid;
                            event_offset = mid;
                            event_y = mid_y;
                        }
                    }
                    return OdeResult {
                        t: step_start + event_offset,
                        y: event_y.as_slice().to_vec(),
                        accepted,
                        rejected,
                        jacobian_evaluations,
                        linear_setups,
                        event_reached: true,
                        failure: (!system.state_is_valid(event_y.as_slice()))
                            .then(|| "invalid_event_state".to_string()),
                        rhs_evaluations: 0,
                        dfdt_evaluations: 0,
                    };
                }
                previous_event = Some(event_right);
            }
            if t >= t_span.1 {
                return OdeResult {
                    t,
                    y: y.as_slice().to_vec(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    event_reached: false,
                    failure: None,
                    rhs_evaluations: 0,
                    dfdt_evaluations: 0,
                };
            }
            point = None;
            h = next_h(h, error, previous_error, config);
            previous_error = error.max(1.0e-30);
        } else {
            rejected += 1;
            if h <= config.h_min {
                return failed_with_setups(
                    t,
                    y.as_slice(),
                    accepted,
                    rejected,
                    jacobian_evaluations,
                    linear_setups,
                    "h_min",
                );
            }
            h = next_h(h, error, previous_error, config);
            if h < config.h_min {
                h = config.h_min;
            }
        }
    }
    failed_with_setups(
        t,
        y.as_slice(),
        accepted,
        rejected,
        jacobian_evaluations,
        linear_setups,
        "max_attempts",
    )
}

fn weighted_norm(error: &DVector<f64>, y: &DVector<f64>, config: &OdeConfig) -> f64 {
    let sum = error
        .iter()
        .zip(y.iter())
        .zip(config.atol.iter())
        .map(|((&error, &value), &atol)| (error / (atol + config.rtol * value.abs())).powi(2))
        .sum::<f64>();
    (sum / y.len() as f64).sqrt()
}

fn next_h(h: f64, error: f64, previous: f64, config: &OdeConfig) -> f64 {
    let factor = if error < 1.0e-30 {
        6.0
    } else {
        (0.9 * error.powf(-0.2) * previous.powf(0.04)).clamp(0.2, 6.0)
    };
    (h * factor).clamp(config.h_min, config.h_max)
}

fn failed(
    t: f64,
    y: &[f64],
    accepted: usize,
    rejected: usize,
    jacobians: usize,
    reason: &str,
) -> OdeResult {
    failed_with_setups(t, y, accepted, rejected, jacobians, jacobians, reason)
}

fn failed_with_setups(
    t: f64,
    y: &[f64],
    accepted: usize,
    rejected: usize,
    jacobian_evaluations: usize,
    linear_setups: usize,
    reason: &str,
) -> OdeResult {
    OdeResult {
        t,
        y: y.to_vec(),
        accepted,
        rejected,
        jacobian_evaluations,
        linear_setups,
        event_reached: false,
        failure: Some(reason.into()),
        rhs_evaluations: 0,
        dfdt_evaluations: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nonautonomous;

    impl OdeSystem for Nonautonomous {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = -y[0] + t.sin();
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = -1.0;
        }
        fn dfdt(&self, t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = t.cos();
        }
    }

    fn config() -> OdeConfig {
        OdeConfig {
            rtol: 1.0e-8,
            atol: vec![1.0e-10],
            h_init: 1.0e-3,
            h_min: 1.0e-14,
            h_max: 0.1,
            max_attempts: 10_000,
        }
    }

    #[test]
    fn both_solvers_match_nonautonomous_exact_endpoint() {
        let exact = 0.5 * (1.0_f64.sin() - 1.0_f64.cos() + (-1.0_f64).exp());
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(kind, &Nonautonomous, (0.0, 1.0), &[0.0], &config(), None);
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(
                (result.t - 1.0).abs() <= f64::EPSILON,
                "{kind:?}: {result:?}"
            );
            assert!((result.y[0] - exact).abs() < 2.0e-7, "{kind:?}: {result:?}");
            assert!(result.accepted > 0);
            assert!(result.jacobian_evaluations > 0);
            assert!(result.linear_setups > 0);
        }
    }

    struct StiffTracking;

    impl OdeSystem for StiffTracking {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = -1_000.0 * (y[0] - t.cos()) - t.sin();
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = -1_000.0;
        }
        fn dfdt(&self, t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = -1_000.0 * t.sin() - t.cos();
        }
    }

    #[test]
    fn both_solvers_match_stiff_scalar_exact_solution() {
        let cfg = OdeConfig {
            rtol: 1.0e-8,
            atol: vec![1.0e-10],
            h_init: 1.0e-5,
            h_min: 1.0e-14,
            h_max: 0.05,
            max_attempts: 20_000,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(kind, &StiffTracking, (0.0, 1.0), &[1.0], &cfg, None);
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(
                (result.y[0] - 1.0_f64.cos()).abs() < 2.0e-7,
                "{kind:?}: {result:?}"
            );
        }
    }

    type JacobianKeyLog = std::cell::RefCell<Vec<(u64, Vec<u64>)>>;

    struct Robertson<'a> {
        jacobian_keys: Option<&'a JacobianKeyLog>,
    }

    impl OdeSystem for Robertson<'_> {
        fn dimension(&self) -> usize {
            3
        }
        fn rhs(&self, _t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
            out[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
            out[2] = 3.0e7 * y[1] * y[1];
        }
        fn jacobian(&self, t: f64, y: &[f64], out: &mut [f64]) {
            if let Some(keys) = self.jacobian_keys {
                keys.borrow_mut()
                    .push((t.to_bits(), y.iter().map(|value| value.to_bits()).collect()));
            }
            out.copy_from_slice(&[
                -0.04,
                1.0e4 * y[2],
                1.0e4 * y[1],
                0.04,
                -1.0e4 * y[2] - 6.0e7 * y[1],
                -1.0e4 * y[1],
                0.0,
                6.0e7 * y[1],
                0.0,
            ]);
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out.fill(0.0);
        }
    }

    #[test]
    fn both_solvers_match_robertson_reference_and_preserve_mass() {
        let cfg = OdeConfig {
            rtol: 1.0e-7,
            atol: vec![1.0e-10, 1.0e-12, 1.0e-10],
            h_init: 1.0e-6,
            h_min: 1.0e-14,
            h_max: 1.0,
            max_attempts: 100_000,
        };
        // Independent SciPy 1.17.1 Radau/BDF/LSODA/DOP853 integrations agree
        // on these t=40 values (rtol=2.5e-14, component atol down to 1e-18).
        let reference = [
            0.715_827_068_719_406_8,
            9.185_534_764_557_829e-6,
            0.284_163_745_745_828_6,
        ];
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let system = Robertson {
                jacobian_keys: None,
            };
            let result = solve(kind, &system, (0.0, 40.0), &[1.0, 0.0, 0.0], &cfg, None);
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(
                result.y.iter().all(|value| *value >= -1.0e-12),
                "{kind:?}: {result:?}"
            );
            assert!(
                (result.y.iter().sum::<f64>() - 1.0).abs() < 2.0e-9,
                "{kind:?}: {result:?}"
            );
            for (actual, expected) in result.y.iter().zip(reference) {
                assert!((actual - expected).abs() < 2.0e-6, "{kind:?}: {result:?}");
            }
        }
    }

    #[test]
    fn bdf_reuses_full_jacobian_across_dense_columns() {
        let keys = std::cell::RefCell::new(Vec::new());
        let system = Robertson {
            jacobian_keys: Some(&keys),
        };
        let cfg = OdeConfig {
            rtol: 1.0e-7,
            atol: vec![1.0e-10, 1.0e-12, 1.0e-10],
            h_init: 1.0e-6,
            h_min: 1.0e-14,
            h_max: 1.0,
            max_attempts: 100_000,
        };
        let result = solve(
            SolverKind::Bdf,
            &system,
            (0.0, 40.0),
            &[1.0, 0.0, 0.0],
            &cfg,
            None,
        );
        assert_eq!(result.failure, None, "{result:?}");
        assert!((result.y.iter().sum::<f64>() - 1.0).abs() < 2.0e-9);
        let keys = keys.into_inner();
        assert_eq!(result.jacobian_evaluations, keys.len());
        assert!(result.jacobian_evaluations > 0);
        assert!(result.linear_setups > 0);
        for (index, key) in keys.iter().enumerate() {
            assert!(
                !keys[..index].contains(key),
                "full Jacobian recomputed for the same bitwise point: {key:?}"
            );
        }
    }

    #[test]
    fn jacobian_cache_key_requires_one_finite_bitwise_point() {
        let state = [0.0_f64, 1.0];
        let bits = state
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert!(exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            1.0,
            &state,
        ));
        assert!(!exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            f64::from_bits(1.0_f64.to_bits() + 1),
            &state,
        ));
        assert!(!exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            1.0,
            &[-0.0, 1.0],
        ));
        assert!(!exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            1.0,
            &[0.0, f64::from_bits(1.0_f64.to_bits() + 1)],
        ));
        assert!(!exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            1.0,
            &[0.0],
        ));
        assert!(!exact_jacobian_cache_key_matches(
            1.0_f64.to_bits(),
            &bits,
            1.0,
            &[0.0, f64::NAN],
        ));
    }

    #[test]
    fn invalid_tolerance_preserves_raw_initial_state() {
        let mut cfg = config();
        cfg.atol[0] = 0.0;
        let result = solve(
            SolverKind::Rodas5P,
            &Nonautonomous,
            (0.0, 1.0),
            &[0.25],
            &cfg,
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("invalid_input"));
        assert_eq!(result.y, vec![0.25]);
        assert_eq!(result.accepted, 0);
        assert_eq!(result.rejected, 0);
    }

    #[test]
    fn invalid_dimension_and_nonfinite_state_preserve_raw_input() {
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let wrong_dimension = solve(
                kind,
                &Nonautonomous,
                (0.0, 1.0),
                &[0.25, 0.5],
                &OdeConfig {
                    atol: vec![1.0e-10; 2],
                    ..config()
                },
                None,
            );
            assert_eq!(wrong_dimension.failure.as_deref(), Some("invalid_input"));
            assert_eq!(wrong_dimension.y, vec![0.25, 0.5]);

            let nonfinite = solve(
                kind,
                &Nonautonomous,
                (0.0, 1.0),
                &[f64::NAN],
                &config(),
                None,
            );
            assert_eq!(nonfinite.failure.as_deref(), Some("invalid_input"));
            assert!(nonfinite.y[0].is_nan());
            assert_eq!((nonfinite.accepted, nonfinite.rejected), (0, 0));
        }
    }

    struct OnlyZeroPhysical;

    impl OdeSystem for OnlyZeroPhysical {
        fn dimension(&self) -> usize {
            1
        }
        fn state_is_valid(&self, state: &[f64]) -> bool {
            state.len() == 1 && state[0] == 0.0
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    fn fixed_domain_config(step: f64) -> OdeConfig {
        OdeConfig {
            rtol: 1.0e6,
            atol: vec![1.0e6],
            h_init: step,
            h_min: step,
            h_max: step,
            max_attempts: 10,
        }
    }

    #[test]
    fn state_domain_rejects_raw_initial_and_solver_specific_candidates() {
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let initial = solve(
                kind,
                &OnlyZeroPhysical,
                (0.0, 0.2),
                &[1.0],
                &fixed_domain_config(0.1),
                None,
            );
            assert_eq!(initial.failure.as_deref(), Some("invalid_initial_state"));
            assert_eq!(initial.y, vec![1.0]);
            assert_eq!((initial.accepted, initial.rejected), (0, 0));
            assert_eq!(
                (initial.jacobian_evaluations, initial.linear_setups),
                (0, 0)
            );
        }

        let bdf = solve(
            SolverKind::Bdf,
            &OnlyZeroPhysical,
            (0.0, 0.2),
            &[0.0],
            &fixed_domain_config(0.1),
            None,
        );
        assert_eq!(bdf.failure.as_deref(), Some("invalid_solution_state"));
        assert!(bdf.t > 0.0);
        assert!(bdf.y[0] > 0.0);
        assert!(bdf.accepted > 0);

        let rodas = solve(
            SolverKind::Rodas5P,
            &OnlyZeroPhysical,
            (0.0, 0.2),
            &[0.0],
            &fixed_domain_config(0.1),
            None,
        );
        assert_eq!(rodas.failure.as_deref(), Some("invalid_candidate_state"));
        assert_eq!(rodas.t, 0.0);
        assert_eq!(rodas.y, vec![0.0]);
        assert_eq!((rodas.accepted, rodas.rejected), (0, 1));
    }

    struct EndpointOnlyLinear;

    impl OdeSystem for EndpointOnlyLinear {
        fn dimension(&self) -> usize {
            1
        }
        fn state_is_valid(&self, state: &[f64]) -> bool {
            state.len() == 1 && (state[0].abs() < 1.0e-12 || state[0] >= 9.0e-5)
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn both_event_refiners_reject_nonphysical_interpolated_states() {
        let event_fn = |_t: f64, state: &[f64]| state[0] - 5.0e-5;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 1,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &EndpointOnlyLinear,
                (0.0, 1.0),
                &[0.0],
                &fixed_domain_config(1.0),
                Some(&event),
            );
            assert_eq!(
                result.failure.as_deref(),
                Some("invalid_event_state"),
                "{kind:?}: {result:?}"
            );
            assert!(!result.event_reached);
            assert_eq!(result.accepted, 1);
        }
    }

    #[test]
    fn bdf_attempt_budget_returns_last_raw_state() {
        let mut cfg = config();
        cfg.h_max = 0.01;
        cfg.max_attempts = 1;
        let result = solve(
            SolverKind::Bdf,
            &Nonautonomous,
            (0.0, 1.0),
            &[0.0],
            &cfg,
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("max_attempts"));
        assert!(result.t > 0.0 && result.t <= cfg.h_max);
        assert!(result.y[0].is_finite());
        assert!(result.accepted + result.rejected <= cfg.max_attempts);
    }

    struct Nonfinite;

    impl OdeSystem for Nonfinite {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = f64::NAN;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn both_solvers_nonfinite_rhs_fail_loud_with_raw_state() {
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(kind, &Nonfinite, (0.0, 1.0), &[1.0], &config(), None);
            assert_eq!(result.failure.as_deref(), Some("nonfinite_rhs"));
            assert_eq!(result.y, vec![1.0]);
            assert_eq!((result.accepted, result.rejected), (0, 0));
        }
    }

    struct NonfiniteJacobian;

    impl OdeSystem for NonfiniteJacobian {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = f64::NAN;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn both_solvers_nonfinite_jacobian_fail_loud_with_raw_state() {
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &NonfiniteJacobian,
                (0.0, 1.0),
                &[1.0],
                &config(),
                None,
            );
            assert_eq!(result.failure.as_deref(), Some("nonfinite_jacobian"));
            assert_eq!(result.y, vec![1.0]);
            assert_eq!(result.accepted, 0);
        }
    }

    struct NonfiniteTimeDerivative;

    impl OdeSystem for NonfiniteTimeDerivative {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = f64::NAN;
        }
    }

    #[test]
    fn rodas_nonfinite_time_derivative_fails_loud_with_raw_state() {
        let result = solve(
            SolverKind::Rodas5P,
            &NonfiniteTimeDerivative,
            (0.0, 1.0),
            &[1.0],
            &config(),
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("nonfinite_dfdt"));
        assert_eq!(result.y, vec![1.0]);
        assert_eq!((result.accepted, result.rejected), (0, 0));
    }

    struct SingularFirstRodasMatrix;

    impl OdeSystem for SingularFirstRodasMatrix {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0 / (GAMMA * 0.1);
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn rodas_singular_linear_system_fails_loud_with_raw_state() {
        let cfg = OdeConfig {
            rtol: 1.0e-8,
            atol: vec![1.0e-10],
            h_init: 0.1,
            h_min: 0.1,
            h_max: 0.1,
            max_attempts: 10,
        };
        let result = solve(
            SolverKind::Rodas5P,
            &SingularFirstRodasMatrix,
            (0.0, 1.0),
            &[2.0],
            &cfg,
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("linear_solve"));
        assert_eq!(result.t, 0.0);
        assert_eq!(result.y, vec![2.0]);
        assert_eq!((result.accepted, result.rejected), (0, 0));
        assert_eq!((result.jacobian_evaluations, result.linear_setups), (1, 1));
    }

    #[test]
    fn rodas_error_rejection_at_step_floor_preserves_raw_state() {
        let cfg = OdeConfig {
            rtol: 1.0e-16,
            atol: vec![1.0e-16],
            h_init: 0.1,
            h_min: 0.1,
            h_max: 0.1,
            max_attempts: 10,
        };
        let result = solve(
            SolverKind::Rodas5P,
            &Nonautonomous,
            (0.0, 1.0),
            &[0.0],
            &cfg,
            None,
        );
        assert_eq!(result.failure.as_deref(), Some("h_min"));
        assert_eq!(result.t, 0.0);
        assert_eq!(result.y, vec![0.0]);
        assert_eq!((result.accepted, result.rejected), (0, 1));
    }

    struct ExponentialDecay;

    impl OdeSystem for ExponentialDecay {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = -3.0 * y[0];
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = -3.0;
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn both_solvers_refine_downward_terminal_event() {
        let event_fn = |_t: f64, y: &[f64]| y[0] - 0.1;
        let event = TerminalEvent {
            value: &event_fn,
            direction: -1,
        };
        let mut cfg = config();
        cfg.rtol = 1.0e-9;
        cfg.atol[0] = 1.0e-12;
        let exact = 10.0_f64.ln() / 3.0;
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &ExponentialDecay,
                (0.0, 5.0),
                &[1.0],
                &cfg,
                Some(&event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!((result.t - exact).abs() < 1.0e-8, "{kind:?}: {result:?}");
            assert!((result.y[0] - 0.1).abs() < 1.0e-8, "{kind:?}: {result:?}");
        }
    }

    struct Oscillator;

    impl OdeSystem for Oscillator {
        fn dimension(&self) -> usize {
            2
        }
        fn rhs(&self, _t: f64, y: &[f64], out: &mut [f64]) {
            let omega = 2.0 * std::f64::consts::PI;
            out[0] = y[1];
            out[1] = -omega * omega * y[0];
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            let omega = 2.0 * std::f64::consts::PI;
            out.copy_from_slice(&[0.0, 1.0, -omega * omega, 0.0]);
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out.fill(0.0);
        }
    }

    #[test]
    fn both_solvers_refine_upward_event_after_skipping_downward_crossing() {
        let event_fn = |_t: f64, y: &[f64]| y[0] - 0.5;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 1,
        };
        let cfg = OdeConfig {
            rtol: 1.0e-10,
            atol: vec![1.0e-12; 2],
            h_init: 1.0e-3,
            h_min: 1.0e-14,
            h_max: 0.02,
            max_attempts: 20_000,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &Oscillator,
                (0.0, 1.0),
                &[1.0, 0.0],
                &cfg,
                Some(&event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!(
                (result.t - 5.0 / 6.0).abs() < 1.0e-3,
                "{kind:?}: {result:?}"
            );
        }
    }

    #[test]
    fn both_solvers_ignore_a_crossing_in_the_wrong_direction() {
        let event_fn = |_t: f64, y: &[f64]| y[0] - 0.5;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 1,
        };
        let cfg = OdeConfig {
            rtol: 1.0e-10,
            atol: vec![1.0e-12; 2],
            h_init: 1.0e-3,
            h_min: 1.0e-14,
            h_max: 0.02,
            max_attempts: 20_000,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &Oscillator,
                (0.0, 0.3),
                &[1.0, 0.0],
                &cfg,
                Some(&event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(!result.event_reached, "{kind:?}: {result:?}");
            assert!(
                (result.t - 0.3).abs() <= f64::EPSILON,
                "{kind:?}: {result:?}"
            );
            assert!(
                result.y[0] < 0.5,
                "crossing was not exercised: {kind:?}: {result:?}"
            );
        }
    }

    #[test]
    fn invalid_event_direction_preserves_raw_initial_state() {
        let event_fn = |_t: f64, y: &[f64]| y[0] - 0.1;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 2,
        };
        let result = solve(
            SolverKind::Bdf,
            &ExponentialDecay,
            (0.0, 1.0),
            &[0.25],
            &config(),
            Some(&event),
        );
        assert_eq!(result.failure.as_deref(), Some("invalid_input"));
        assert_eq!(result.y, vec![0.25]);
        assert!(!result.event_reached);
        assert_eq!((result.accepted, result.rejected), (0, 0));
    }

    #[test]
    fn both_solvers_stop_on_an_initial_terminal_event() {
        let event_fn = |_t: f64, y: &[f64]| y[0] - 1.0;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 0,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &ExponentialDecay,
                (0.0, 1.0),
                &[1.0],
                &config(),
                Some(&event),
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert_eq!(result.t, 0.0);
            assert_eq!(result.y, vec![1.0]);
            assert_eq!((result.accepted, result.rejected), (0, 0));
            assert_eq!((result.jacobian_evaluations, result.linear_setups), (0, 0));
        }
    }

    #[test]
    fn both_solvers_detect_directionless_zero_at_step_endpoint() {
        let event_fn = |t: f64, _y: &[f64]| t - 0.1;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 0,
        };
        let cfg = OdeConfig {
            rtol: 1.0e6,
            atol: vec![1.0e6],
            h_init: 0.1,
            h_min: 1.0e-14,
            h_max: 0.1,
            max_attempts: 100,
        };
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(kind, &Nonautonomous, (0.0, 0.5), &[0.0], &cfg, Some(&event));
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert!(result.event_reached, "{kind:?}: {result:?}");
            assert!(
                (result.t - 0.1).abs() <= f64::EPSILON,
                "{kind:?}: {result:?}"
            );
        }
    }

    struct RefinementSingularAtHalfStep;

    impl OdeSystem for RefinementSingularAtHalfStep {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
        fn jacobian(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 1.0 / (GAMMA * 0.05);
        }
        fn dfdt(&self, _t: f64, _y: &[f64], out: &mut [f64]) {
            out[0] = 0.0;
        }
    }

    #[test]
    fn rodas_event_refinement_failure_returns_last_raw_endpoint() {
        let event_fn = |t: f64, _y: &[f64]| t - 0.075;
        let event = TerminalEvent {
            value: &event_fn,
            direction: 0,
        };
        let cfg = OdeConfig {
            rtol: 1.0e6,
            atol: vec![1.0e6],
            h_init: 0.1,
            h_min: 1.0e-14,
            h_max: 0.1,
            max_attempts: 100,
        };
        let result = solve(
            SolverKind::Rodas5P,
            &RefinementSingularAtHalfStep,
            (0.0, 0.5),
            &[2.0],
            &cfg,
            Some(&event),
        );
        assert_eq!(result.failure.as_deref(), Some("event_refinement"));
        assert!(!result.event_reached);
        assert_eq!(result.t, 0.1);
        assert_eq!(result.y, vec![2.0]);
        assert_eq!((result.accepted, result.rejected), (1, 0));
        assert_eq!(result.jacobian_evaluations, 1);
        assert_eq!(result.linear_setups, 2);
    }

    #[test]
    fn rodas_has_fifth_order_nonautonomous_convergence() {
        let exact = 0.5 * (1.0_f64.sin() - 1.0_f64.cos() + (-1.0_f64).exp());
        let mut errors = Vec::new();
        for steps in [8, 16, 32, 64] {
            let h = 1.0 / steps as f64;
            let cfg = OdeConfig {
                rtol: 1.0e6,
                atol: vec![1.0e6],
                h_init: h,
                h_min: h,
                h_max: h,
                max_attempts: steps as usize + 1,
            };
            let result = solve(
                SolverKind::Rodas5P,
                &Nonautonomous,
                (0.0, 1.0),
                &[0.0],
                &cfg,
                None,
            );
            assert_eq!(result.failure, None, "{result:?}");
            errors.push((result.y[0] - exact).abs());
        }
        let order = (errors[2] / errors[3]).ln() / 2.0_f64.ln();
        assert!(order >= 4.5, "order={order}, errors={errors:?}");
    }

    #[test]
    fn both_solvers_repeat_bit_for_bit_on_the_same_host() {
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let first = solve(kind, &Nonautonomous, (0.0, 1.0), &[0.0], &config(), None);
            let second = solve(kind, &Nonautonomous, (0.0, 1.0), &[0.0], &config(), None);
            assert_eq!(first.t.to_bits(), second.t.to_bits(), "{kind:?}");
            assert_eq!(
                first
                    .y
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                second
                    .y
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{kind:?}"
            );
            assert_eq!(first.accepted, second.accepted, "{kind:?}");
            assert_eq!(first.rejected, second.rejected, "{kind:?}");
            assert_eq!(
                first.jacobian_evaluations, second.jacobian_evaluations,
                "{kind:?}"
            );
            assert_eq!(first.linear_setups, second.linear_setups, "{kind:?}");
            assert_eq!(first.event_reached, second.event_reached, "{kind:?}");
            assert_eq!(first.failure, second.failure, "{kind:?}");
            assert_eq!(first.rhs_evaluations, second.rhs_evaluations, "{kind:?}");
            assert_eq!(first.dfdt_evaluations, second.dfdt_evaluations, "{kind:?}");
        }
    }

    #[test]
    fn solver_counters_report_trait_level_rhs_and_dfdt_calls() {
        let rodas = solve(
            SolverKind::Rodas5P,
            &Nonautonomous,
            (0.0, 1.0),
            &[0.0],
            &config(),
            None,
        );
        assert_eq!(rodas.failure, None, "{rodas:?}");
        let attempts = rodas.accepted + rodas.rejected;
        assert_eq!(
            rodas.dfdt_evaluations, rodas.jacobian_evaluations,
            "{rodas:?}"
        ); // both = number of linearization points
        assert_eq!(rodas.rhs_evaluations, 1 + 8 * attempts, "{rodas:?}");
        assert!(rodas.rhs_evaluations > 0);
        eprintln!(
            "T0_COUNTERS Rodas5P: accepted={} rejected={} attempts={attempts} rhs_evaluations={} dfdt_evaluations={}",
            rodas.accepted, rodas.rejected, rodas.rhs_evaluations, rodas.dfdt_evaluations
        );

        let bdf = solve(
            SolverKind::Bdf,
            &Nonautonomous,
            (0.0, 1.0),
            &[0.0],
            &config(),
            None,
        );
        assert_eq!(bdf.failure, None, "{bdf:?}");
        assert!(bdf.rhs_evaluations > 0);
        assert_eq!(bdf.dfdt_evaluations, 0);
        eprintln!(
            "T0_COUNTERS Bdf: accepted={} rejected={} rhs_evaluations={} dfdt_evaluations={}",
            bdf.accepted, bdf.rejected, bdf.rhs_evaluations, bdf.dfdt_evaluations
        );
    }

    #[test]
    fn default_rhs_and_jacobian_matches_separate_calls() {
        // Nonautonomous does not override rhs_and_jacobian, so it uses the
        // trait default. This pins the contract: default fused output must
        // equal separate rhs()+jacobian() calls, bitwise.
        let t = 0.37;
        let y = [0.42];

        let mut f_fused = [0.0];
        let mut j_fused = [0.0];
        Nonautonomous.rhs_and_jacobian(t, &y, &mut f_fused, &mut j_fused);

        let mut f_sep = [0.0];
        Nonautonomous.rhs(t, &y, &mut f_sep);
        let mut j_sep = [0.0];
        Nonautonomous.jacobian(t, &y, &mut j_sep);

        assert_eq!(f_fused[0].to_bits(), f_sep[0].to_bits());
        assert_eq!(j_fused[0].to_bits(), j_sep[0].to_bits());
    }

    struct CubicDecayKeyLogger {
        jac_keys: JacobianKeyLog,
        dfdt_keys: JacobianKeyLog,
    }

    impl OdeSystem for CubicDecayKeyLogger {
        fn dimension(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, y: &[f64], out: &mut [f64]) {
            out[0] = -y[0] * y[0] * y[0];
        }
        fn jacobian(&self, t: f64, y: &[f64], out: &mut [f64]) {
            self.jac_keys
                .borrow_mut()
                .push((t.to_bits(), y.iter().map(|value| value.to_bits()).collect()));
            out[0] = -3.0 * y[0] * y[0];
        }
        fn dfdt(&self, t: f64, y: &[f64], out: &mut [f64]) {
            self.dfdt_keys
                .borrow_mut()
                .push((t.to_bits(), y.iter().map(|value| value.to_bits()).collect()));
            out[0] = 0.0;
        }
    }

    #[test]
    fn rodas_reuses_jacobian_and_dfdt_at_an_unchanged_point() {
        let system = CubicDecayKeyLogger {
            jac_keys: std::cell::RefCell::new(Vec::new()),
            dfdt_keys: std::cell::RefCell::new(Vec::new()),
        };
        let event_fn = |_t: f64, y: &[f64]| y[0] - 1.0;
        let event = TerminalEvent {
            value: &event_fn,
            direction: -1,
        };
        let cfg = OdeConfig {
            rtol: 1.0e-6,
            atol: vec![1.0e-9],
            h_init: 1.0,
            h_min: 1.0e-14,
            h_max: 1.0,
            max_attempts: 10_000,
        };
        let result = solve(
            SolverKind::Rodas5P,
            &system,
            (0.0, 5.0),
            &[2.0],
            &cfg,
            Some(&event),
        );
        assert_eq!(result.failure, None, "{result:?}");
        assert!(
            result.rejected >= 1,
            "guard: expected a rejection: {result:?}"
        );
        assert!(
            result.event_reached,
            "guard: expected event bisection: {result:?}"
        );

        let jac_keys = system.jac_keys.into_inner();
        let dfdt_keys = system.dfdt_keys.into_inner();
        let unique_jac: std::collections::HashSet<_> = jac_keys.iter().cloned().collect();
        let unique_dfdt: std::collections::HashSet<_> = dfdt_keys.iter().cloned().collect();
        assert_eq!(
            jac_keys.len(),
            unique_jac.len(),
            "jacobian recomputed at an already-linearized point: {jac_keys:?}"
        );
        assert_eq!(
            dfdt_keys.len(),
            unique_dfdt.len(),
            "dfdt recomputed at an already-linearized point: {dfdt_keys:?}"
        );
        assert_eq!(
            jac_keys, dfdt_keys,
            "jacobian and dfdt must be evaluated at exactly the same set of points, in the same order"
        );
    }
}
