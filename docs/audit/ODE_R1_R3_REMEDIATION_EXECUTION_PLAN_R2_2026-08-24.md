# ODE R1–R3 Temporal Blocker Remediation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` for the single-writer implementation and a fresh-context reviewer after each PR. Use `superpowers:test-driven-development` for every behavior change. Do not dispatch parallel production-code writers.

**Goal:** replace the current false-certifying/local-cap failure surface with an explicit, condition-aware Pauli edge certificate, then measure the reconstructed sweep's operator consistency and temporal order in a dimensionless, representable window.

**Architecture:** keep the current conservative Pauli edge decomposition and fail-closed transaction boundary. Separate value-only flux evaluation from certificate-bearing flux evaluation; the latter must expose gross traffic, an absolute arithmetic-error bound, and a resolution class. Repair the local root certificate before replacing the assertionless R3 probe with four independent, rate-scaled tests.

**Tech Stack:** Rust 1.94.1, IEEE-754 binary64, the existing `rabbit_cpu` crate, `cargo test/check/clippy/fmt`; no new runtime dependency.

**Spec:** `docs/audit/ODE_R2_R3_TEMPORAL_BLOCKER_REAUDIT_REQUEST_2026-08-24.md`, amended by the findings and rulings embedded in this plan.

## Global Constraints

- Exact ancestry at planning time: `external-audit/ode-r2-r3-temporal-blocker-20260824` at the externally audited `31f9be0` lineage.
- The active `OdeSystem::rhs` path remains unchanged. This plan does not promote the reconstructed sweep into the active ODE driver.
- QKE, new JAX forward development, public-production claims, endpoint claims, R4–R11, clipping, projection, tolerance widening, equilibrium anchor forcing, and a cap-only `96 -> 128` change are FORBIDDEN.
- Rust AOT remains the implementation target; SciPy/BDF remains the temporary number-of-record; JAX remains a frozen local oracle.
- Do not add a registry, manifest, readiness wrapper, feature flag, or general telemetry subsystem. New fields are allowed only when they directly carry the edge/root certificate required to retire this blocker.
- Preserve raw failures. A branch that reaches a more precise typed failure is acceptable; a silent green is not.
- Every implementation PR must include the repository cost line from `bbn_codex_anti_drift_cost_effective_policy.md`.
- One designated writer owns production code. Reviewers are read-only except for isolated review artifacts.
- Tests are written first and observed failing for the intended reason before production code is changed.

---

## 1. External-audit findings adopted as authority

The supplied static audit read the complete `pauli_edge_step.rs` and the relevant `electron_spectral.rs` sections. It did not compile or execute the branch. Its source-confirmed findings are adopted:

1. The Newton iterate becomes a bracket endpoint, fails the strict-interior test, and the algorithm degenerates into full-bracket bisection. The `91, 93, 95` iteration sequence and the predicted `97 > 96` at `h=2^-14` are one dyadic scaling signature, not three independent observations.
2. `initial_flux == 0.0` is a second non-zero-step bypass. It returns `PauliEdgeStep::default()`, after which `0.0/0.0` becomes `NaN` and is swallowed by `f64::max`.
3. `pauli_sweep_tangent_converges_to_unforced_action` has no assertions and discards the candidate state. Increasing the cap would turn a loud failure into a silent false green.
4. The current certificate bounds the root of the computed flux, not the exact-real flux defined by the binary64 inputs. Its scale uses `h|J|` rather than gross gain/loss traffic, so near balance it requests precision below the evaluation noise floor.
5. The current raw-`h` ladder is below the step-doubling representability floor. The next temporal ladder must move upward in physical `h`, while remaining small in a dimensionless stiffness variable.
6. For `D(h)=||Phi_h-Phi_{h/2}^2||`, a method of order `p` gives a log2 ratio of `p+1`; a first-order sweep therefore gives approximately `2`, not `1`.
7. The ladder scale is `eta_J=h max_e |dJ_e/dxi|`. The capacity scale `eta_cap=h max_i |F_i|/d_i` is a safety gate, not the accuracy scale.
8. A stable closed-form quadratic backward-Euler root is a later candidate, but it does not remove flux-conditioning error and is not the first implementation step.

### Design ruling beyond the static audit

The audit sketches an affinity-based `delta_J`. Standard-library `ln`/`expm1` accuracy is not a portable rigorous interval contract. Therefore the immediate certificate-bearing path will use direct non-negative products with a conservative IEEE-754 absolute-error bound whenever those products are representable. The existing log/`expm1` path remains available for value-only evaluation, but a log-only result is `UnresolvedForCertificate` until independently bounded. This avoids converting an approximate error estimate into a false theorem.

---

## 2. Revised status before implementation

```text
R1_FAIL_CLOSED_SEMANTICS                PASS
R1_FALSE_CERTIFICATE_CHANNEL            BLOCKED
R1_ROOT_CERTIFICATE_COMPLETENESS        BLOCKED
R1_ROOT_CERTIFICATE_SOUNDNESS           BLOCKED
R2_EDGE_RECONSTRUCTION                  VALIDATED_FOCUSED
R2_RAW_DB_MAGNITUDE_BELOW_F64_FLOOR     FORBIDDEN
R2_CONDITIONED_DB_SENSITIVITY           NOT_YET_EVALUATED
R3_TEST_HAS_ASSERTIONS                  BLOCKED
R3_PROBE_WINDOW_RESOLVABLE              BLOCKED
R3_TEMPORAL_CONSISTENCY                 NOT_YET_EVALUATED
R4-R11                                  FORBIDDEN
```

The implementation is a three-PR DAG:

```text
PR-ODE-R1C  explicit outcomes + bounded flux certificate + root completeness
    |
    v
PR-ODE-R2C  condition-aware detailed-balance gate + negative control
    |
    v
PR-ODE-R3C  rate-scaled root/tangent/step-doubling/order-bias tests
    |
    +--> VALIDATED_FIRST_ORDER_SWEEP: stop and request promotion adjudication
    +--> BLOCKED_*: preserve evidence and stop
```

No PR may skip its predecessor.

---

## 3. File map

### Production files

- `native/rabbit_cpu/src/pauli_edge_step.rs`
  - Owns value-only flux evaluation.
  - Owns certificate-bearing flux evaluation and arithmetic error bounds.
  - Owns typed edge outcomes/failures.
  - Owns the backward-Euler extent solve and root certificate.

- `native/rabbit_cpu/src/electron_spectral.rs`
  - Owns folded-edge assembly.
  - Owns transactional sweep composition and aggregate report semantics.
  - Must delete the assertionless R3 test.
  - May expose only the minimal internal methods required by the test child module.

### Test-only file

- Create `native/rabbit_cpu/src/electron_spectral_temporal_tests.rs`
  - Owns the R3 scale census, representability gates, norms, and the four temporal tests.
  - Is included from `electron_spectral.rs` with:
    ```rust
    #[cfg(test)]
    #[path = "electron_spectral_temporal_tests.rs"]
    mod temporal_tests;
    ```
  - Must not become a runtime module or public API.

### Durable evidence

- Do not create one audit note per test.
- Store raw command logs under the current `.agent-harness/runs/<RUN_ID>/` evidence tree.
- Use the PR body for the cost line and concise verdict.
- Create a single final remediation result note only after PR-ODE-R3C is adjudicated.

---

## 4. Preflight: immutable baseline and reproduction

Run before editing:

```bash
git fetch origin
git switch plan/ode-r1-r3-remediation-v2-20260824
git pull --ff-only
git status --short
git rev-parse HEAD
git merge-base HEAD origin/external-audit/ode-r2-r3-temporal-blocker-20260824
python3 .agent-harness/scripts/build_context_pack.py

cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_spectral::tests::unforced_fd_equilibrium_is_an_event_and_edge_null -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::pauli_sweep_tangent_converges_to_unforced_action -- --exact --nocapture
```

Expected baseline:

- all format/check/clippy/R1/R2 focused commands pass;
- the final R3 command fails at `h=2^-14` with `Pauli edge implicit root did not converge`;
- successful prefixes report maximum iterations `91, 93, 95`;
- the worktree remains clean.

Stop if ancestry, output, or worktree state differs. Record the discrepancy; do not adapt the plan silently.

---

# PR-ODE-R1C — Explicit outcomes, bounded flux certificate, and root completeness

**Named blocker moved:** false-certificate channel, certificate soundness, and the `h=2^-14` local root-cap failure.

**Expected net line budget:** 160–320 lines. More than 320 requires deletion or a written explanation in the PR cost line.

## Task 1: Make zero, stationary, solved, and unresolved states non-interchangeable

**Files:**
- Modify: `native/rabbit_cpu/src/pauli_edge_step.rs`
- Modify: `native/rabbit_cpu/src/electron_spectral.rs`
- Test in the same modules; do not create a generic reporting framework.

**Interfaces produced:**

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliFluxResolution {
    Resolved,
    ExactZero,
    UnresolvedForCertificate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliEdgeStep {
    pub(crate) kind: PauliEdgeApplicationKind,
    pub(crate) extent: f64,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) residual_abs: f64,
    pub(crate) residual_scale: f64,
    pub(crate) flux_abs_error_bound_mev: f64,
    pub(crate) root_error_bound: f64,
    pub(crate) max_occupation_error_bound: f64,
    pub(crate) max_occupation_bracket_width: f64,
    pub(crate) conditioning_lower_bound: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PauliEdgeFailureKind {
    InvalidInput,
    UnresolvedFlux,
    UncertainPhysicalBracket,
    InvalidResidual,
    IterationLimit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PauliEdgeFailure {
    pub(crate) kind: PauliEdgeFailureKind,
    pub(crate) flux: Option<PauliFluxEvaluation>,
}
```

`PauliSweepReport` must add, at minimum:

```rust
pub(crate) solved_edges: usize,
pub(crate) exact_stationary_edges: usize,
pub(crate) unresolved_edges: usize,
pub(crate) maximum_flux_error_fraction: f64,
pub(crate) maximum_root_error_bound: f64,
```

A failed transactional sweep must return a typed failure carrying the partial report and failing edge identity. The input slices remain immutable and no candidate is returned.

- [ ] **Step 1: Write failing edge-level tests**

Add tests with these exact behavioral contracts:

```rust
#[test]
fn nonzero_step_rejects_positive_traffic_zero_net_as_unresolved() {
    // A=B=1 and f1=f2=0.5: both positive products are equal in binary64.
    // This is not allowed to masquerade as an exact stationary certificate.
}

#[test]
fn both_zero_products_are_explicit_exact_stationary() {
    // A=B=0: the exact-real gain and loss are both zero.
    // A non-zero step returns unchanged occupations with kind ExactStationary.
}

#[test]
fn zero_step_is_distinct_from_nonzero_exact_stationary() {
    // The caller-visible outcome/report must distinguish the explicit zero-step bypass.
}
```

Run:

```bash
cargo test --lib pauli_edge_step::tests::nonzero_step_rejects_positive_traffic_zero_net_as_unresolved -- --exact --nocapture
cargo test --lib pauli_edge_step::tests::both_zero_products_are_explicit_exact_stationary -- --exact --nocapture
```

Expected: FAIL because the typed outcomes do not exist and `initial_flux == 0.0` currently returns a default success.

- [ ] **Step 2: Write failing sweep aggregation tests**

Add:

```rust
#[test]
fn sweep_report_counts_exact_stationary_without_nan_swallow() {
    // Aggregate one ExactStationary and one Solved application.
    // Every floating report field is finite.
    // solved_edges==1, exact_stationary_edges==1, unresolved_edges==0.
}

#[test]
fn unresolved_edge_failure_is_transactional_and_observable() {
    // Build a minimal folded edge bank with one unresolved positive-traffic edge.
    // The returned failure has kind UnresolvedFlux, unresolved_edges==1,
    // and the caller's input banks are bitwise unchanged.
}
```

Expected: FAIL under the current default report and `f64::max` path.

- [ ] **Step 3: Implement only the outcome plumbing**

Rules:

- remove `Default` as a semantic success path for `PauliEdgeStep`;
- retain a zero-step fast return at the sweep boundary, but represent it separately from an applied edge;
- an exact zero is accepted only when both non-negative products are provably zero because a coefficient or factor is exactly zero;
- positive gain and loss products that compare equal are `UnresolvedForCertificate`, not `ExactZero`;
- compute residual ratios only for `Solved` applications with `residual_scale > 0.0`;
- reject any non-finite aggregate field.

Run all tests from Steps 1–2 and the pre-existing `pauli_edge_step::tests`.

- [ ] **Step 4: Commit**

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs native/rabbit_cpu/src/electron_spectral.rs
git commit -m "ODE-R1C: expose stationary and unresolved Pauli edges"
```

## Task 2: Add a bounded certificate-bearing flux path

**Files:**
- Modify: `native/rabbit_cpu/src/pauli_edge_step.rs`

**Consumes:** the types from Task 1.

**Produces:**

```rust
pub(crate) fn flux_mev(
    &self,
    first_occupation: f64,
    second_occupation: f64,
) -> Result<f64, PauliEdgeFailure>;
```

This remains the value-only robust path and may use the existing log/`expm1` implementation.

```rust
pub(crate) fn certified_flux_evaluation(
    &self,
    first_occupation: f64,
    second_occupation: f64,
) -> Result<PauliFluxEvaluation, PauliEdgeFailure>;
```

This is the only path the root solver and temporal certificates may use.

- [ ] **Step 1: Write failing arithmetic-bound tests**

```rust
#[test]
fn direct_product_certificate_resolves_well_conditioned_flux() {
    // Use normal, non-zero, power-of-two inputs.
    // resolution==Resolved, abs_error_bound>0,
    // abs(net_mev)>abs_error_bound_mev,
    // traffic_upper_bound_mev>=abs(net_mev).
}

#[test]
fn near_balance_is_unresolved_when_net_is_below_arithmetic_bound() {
    // Positive gross traffic, |net| <= delta_J.
    // resolution==UnresolvedForCertificate.
}

#[test]
fn extreme_log_only_flux_remains_value_available_but_uncertified() {
    // Preserve the existing extreme-tail finite value check.
    // The value-only path succeeds; the certificate path is unresolved
    // if direct products underflow or overflow.
}

#[test]
fn resolved_direct_and_value_only_flux_agree_within_the_reported_bound() {
    // |J_direct-J_log| <= delta_J + 32*eps*traffic_upper.
}
```

Expected: FAIL because no certificate-bearing evaluator exists.

- [ ] **Step 2: Implement conservative direct-product bounds**

Use exact-zero detection before multiplication. For positive inputs, compute each three-factor product directly. If a positive product overflows, or underflows to zero, the certificate path is unresolved; do not assign an invented finite relative error.

Use these named constants and document the IEEE-754 reasoning:

```rust
const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const PRODUCT_ABS_ERROR_FACTOR: f64 = 8.0 * f64::EPSILON;
const DIFFERENCE_ABS_ERROR_FACTOR: f64 = 2.0 * f64::EPSILON;
```

For each representable product `p_hat`:

```rust
product_error =
    PRODUCT_ABS_ERROR_FACTOR * p_hat.abs()
    + 4.0 * MIN_SUBNORMAL;
```

For `net_hat = gain_hat - loss_hat`:

```rust
difference_error =
    DIFFERENCE_ABS_ERROR_FACTOR * (gain_hat.abs() + loss_hat.abs())
    + 2.0 * MIN_SUBNORMAL;

delta_j = gain_error + loss_error + difference_error;
traffic_upper =
    gain_hat.abs() + loss_hat.abs() + gain_error + loss_error;
```

Classification:

```rust
if both_exact_products_zero {
    ExactZero
} else if net_hat.abs() > delta_j {
    Resolved
} else {
    UnresolvedForCertificate
}
```

Define the conservative conditioning lower bound:

```rust
kappa_lower =
    ((net_hat.abs() - delta_j).max(0.0)
      / traffic_upper.max(MIN_SUBNORMAL));
```

Do not use the log/`expm1` result as a certificate when the direct bound is unavailable.

- [ ] **Step 3: Run focused and legacy flux tests**

```bash
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
```

Expected: PASS. The existing extreme-tail value test remains green without implying a root certificate.

- [ ] **Step 4: Commit**

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs
git commit -m "ODE-R1C: bound certificate-bearing Pauli fluxes"
```

## Task 3: Certify the current Newton iterate and the true-root distance bound

**Files:**
- Modify: `native/rabbit_cpu/src/pauli_edge_step.rs`
- Modify: `native/rabbit_cpu/src/electron_spectral.rs`

**Consumes:** `certified_flux_evaluation`.

**Produces:** a successful `PauliEdgeStep` only when both solver residual and exact-real root-distance bounds close.

- [ ] **Step 1: Write failing root-completeness tests**

```rust
#[test]
fn current_newton_iterate_closes_without_full_bracket_collapse() {
    // The previously failing representative edge returns Solved.
    // nonlinear_iterations <= 4.
}

#[test]
fn root_certificates_hold_across_extreme_step_scales() {
    // Same deterministic edge/state for h=2^-8, 2^-10, ..., 2^-30.
    // Every resolved solve has nonlinear_iterations <= 4,
    // finite root_error_bound,
    // max_occupation_error_bound <= 128*eps,
    // and a candidate inside [0,1].
}

#[test]
fn uncertain_flux_cannot_be_hidden_by_bracket_collapse() {
    // A positive-traffic unresolved flux returns UnresolvedFlux,
    // not a midpoint candidate.
}

#[test]
fn physical_capacity_bracket_uses_flux_error_intervals() {
    // Bracket signs are accepted only when the true residual intervals
    // prove lower <= 0 <= upper.
}
```

Expected: FAIL under midpoint-only termination.

- [ ] **Step 2: Change residual evaluation**

At each extent return:

```rust
struct ResidualEvaluation {
    value: f64,                    // xi - h*J_hat
    derivative: f64,               // 1 - h*dJ/dxi
    flux: PauliFluxEvaluation,
}
```

Define:

```rust
residual_scale =
    max(abs(xi), h * flux.traffic_upper_bound_mev, MIN_SUBNORMAL);

root_error_bound =
    abs(value) + h * flux.abs_error_bound_mev;

max_occupation_error_bound =
    root_error_bound / min(first_measure, second_measure);
```

A current iterate is certified only if:

```rust
abs(value) <= 128*eps*residual_scale + MIN_SUBNORMAL
&& max_occupation_error_bound <= 128*eps
&& flux.resolution == Resolved
```

For `ExactZero` at the initial state, return `ExactStationary`. For `UnresolvedForCertificate`, return `UnresolvedFlux`.

Bracket proof:

```text
lower_value + h*lower_delta_J <= 0
upper_value - h*upper_delta_J >= 0
```

If either inequality cannot be proved, return `UncertainPhysicalBracket`.

- [ ] **Step 3: Check the current iterate before midpoint fallback**

At the top of each iteration:

1. evaluate and certify the current iterate;
2. if certified, return it;
3. update the bracket from its residual sign;
4. retain bracket+midpoint certification only as a fallback;
5. propose Newton inside the open bracket, otherwise use midpoint.

Do not change `max_iterations=96`.

- [ ] **Step 4: Run the former R3 raw ladder as a root-robustness fixture**

Rename it away from an R3 claim:

```text
root_certificates_hold_across_legacy_small_step_scales
```

It may verify root closure, but it must not claim tangent convergence or order.

Run:

```bash
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
```

Expected:

- all rungs, including `2^-14`, close;
- maximum edge iterations are independent of `h` to within the frozen `<=4` gate;
- no tolerance or iteration-cap change appears in the diff.

- [ ] **Step 5: Adversarial mutation check, then revert the mutation**

Temporarily disable the current-iterate success branch without changing tolerances. Re-run the `2^-14` legacy fixture and confirm the old non-convergence returns. Revert the temporary mutation and verify a clean worktree.

Record the command and result in `.agent-harness`; do not commit the mutation.

- [ ] **Step 6: Commit**

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs native/rabbit_cpu/src/electron_spectral.rs
git commit -m "ODE-R1C: certify Newton iterates with flux error bounds"
```

## PR-ODE-R1C acceptance

Run:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
cargo test --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
```

PASS requires:

```text
unresolved edges are explicit
no NaN aggregate is swallowed
no default step is interpreted as a solved edge
root error includes h*delta_J
residual scale includes gross traffic
legacy h=2^-14 closes with max iterations <=4
max_iterations remains 96
no active ODE path change
```

Open a PR against `plan/ode-r1-r3-remediation-v2-20260824`. Do not start PR-ODE-R2C until a fresh-context reviewer approves the type semantics, arithmetic bound, bracket inequalities, and transactionality.

---

# PR-ODE-R2C — Condition-aware detailed balance with a sensitivity control

**Named blocker moved:** the existing `<=1e-12` raw-DB test can pass at its own floating-point floor or if every flux is spuriously zero.

**Expected net line budget:** 60–140 lines, primarily tests and a small test-only metric helper.

## Task 4: Replace the vacuous magnitude claim without weakening edge reconstruction

**Files:**
- Modify: `native/rabbit_cpu/src/electron_spectral.rs`
- Keep the five-state action reconstruction and boundary-inward tests.

**Test-only metric:**

```rust
struct ConditionedDetailedBalance {
    resolved_excess_l1: f64,
    traffic_upper_l1: f64,
    normalized_resolved_excess: f64,
    unresolved_near_balance_edges: usize,
    maximum_error_fraction: f64,
    maximum_resolved_edge_ratio: f64,
}
```

Per edge:

```rust
resolved_excess =
    (abs(net_mev) - abs_error_bound_mev).max(0.0);

error_fraction =
    abs_error_bound_mev
    / traffic_upper_bound_mev.max(MIN_SUBNORMAL);
```

- [ ] **Step 1: Write a failing base-state test**

Rename the current equilibrium test to:

```text
unforced_fd_equilibrium_is_conditioned_near_balance
```

Assertions:

```text
all value-only edge evaluations remain finite
maximum_error_fraction <= 64*eps
normalized_resolved_excess <= 1e-12
maximum_resolved_edge_ratio <= 1e-12
the report prints unresolved_near_balance_edges explicitly
```

Do not assert that a `1e-15` net ratio is physically validated. That magnitude claim is FORBIDDEN.

- [ ] **Step 2: Write the required negative control**

```rust
#[test]
fn conditioned_detailed_balance_detects_gain_perturbation() {
    // Select the edge with the largest traffic_upper_bound.
    // Multiply its gain coefficient by 1.0 + 1.0e-10.
    // Require a resolved imbalance and:
    // maximum_resolved_edge_ratio >= 1.0e-11.
}
```

Expected before implementing the conditioned metric: FAIL.

The negative control is load-bearing. A flux evaluator that returns zero or an infinite error bound for every edge must fail this test.

- [ ] **Step 3: Implement the test-only metric and run R2 tests**

```bash
cargo test --lib electron_spectral::tests::unforced_fd_equilibrium_is_conditioned_near_balance -- --exact --nocapture
cargo test --lib electron_spectral::tests::conditioned_detailed_balance_detects_gain_perturbation -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_are_boundary_inward_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::invalid_inputs_fail_without_clipping -- --exact --nocapture
```

- [ ] **Step 4: Commit**

```bash
git add native/rabbit_cpu/src/electron_spectral.rs
git commit -m "ODE-R2C: make detailed-balance sensitivity explicit"
```

## PR-ODE-R2C acceptance and claim ceiling

PASS permits only:

```text
R2_EDGE_RECONSTRUCTION              VALIDATED_FOCUSED
R2_CONDITIONED_DB_COMPATIBILITY     VALIDATED_FOCUSED
R2_DB_SENSITIVITY_CONTROL           VALIDATED
R2_RAW_DB_MAGNITUDE_BELOW_F64_FLOOR FORBIDDEN
```

It does not validate continuum convergence, endpoint behavior, or the active ODE driver.

---

# PR-ODE-R3C — Rate-scaled and representable temporal contract

**Named blocker moved:** the assertionless, dimensionful R3 probe and its roundoff-floor ladder.

**Expected net line budget:** 140–280 test lines and at most 40 production lines. If production growth exceeds 40 lines, move helper logic into the test-only child module.

## Task 5: Create a deterministic scale census without changing the fixture

**Files:**
- Modify: `native/rabbit_cpu/src/electron_spectral.rs`
- Create: `native/rabbit_cpu/src/electron_spectral_temporal_tests.rs`
- Modify visibility in `pauli_edge_step.rs` only if the child test module cannot read the analytic edge derivative through an existing method.

Use the same deterministic R3 state:

```text
grid order 4
electron rule 4/3
T_gamma=1.15 MeV
T_cm=1.0 MeV
electron bank=alternating 0.91/1.07 FD
heavy bank=alternating 1.07/0.91 FD
```

Do not replace the state to obtain a green result.

**Definitions:**

```text
lambda_J   = max_e |dJ_e/dxi|                         [MeV]
lambda_cap = max_i |F_i| / min(f_i, 1-f_i)           [MeV]
eta_J      = h * lambda_J
eta_cap    = h * lambda_cap
```

The fixed target ladder is:

```text
eta_J in {2^-4, 2^-5, 2^-6, 2^-7, 2^-8}
h_k = eta_J_k / lambda_J
```

Safety gate:

```text
eta_cap(h_max) <= 1/16
```

If the same fixture fails this gate, stop with `BLOCKED_TEMPORAL_WINDOW_UNSAFE`; do not clip the state or silently shrink individual rungs.

Define two 99.9% carrying sets:

- `S_action`: the smallest component set carrying at least 99.9% of
  `sum_i (w_i y_i^2 |F_i|)` across both banks.
- `E_flux`: the smallest edge set carrying at least 99.9% of
  `sum_e |J_e|`.

Conditioning gate on every `e in E_flux`:

```text
kappa_e = |J_e| / traffic_upper_e
kappa_e >= (2^10 * eps) / eta_J_max
```

For `eta_J_max=2^-4`, the threshold is approximately `3.64e-12`.

- [ ] **Step 1: Add a failing scale-census test**

```rust
#[test]
fn rate_scaled_probe_window_is_safe_and_conditioned() {
    // Assert finite positive lambda_J and lambda_cap.
    // Assert eta_cap(h_max)<=1/16.
    // Assert the E_flux conditioning gate.
    // Print lambda_J, lambda_cap, realised h range, eta_J, eta_cap,
    // action/edge carrying-set sizes, and minimum kappa.
}
```

Expected: FAIL because the scale census does not exist.

- [ ] **Step 2: Implement the test-only census**

The only production-facing addition allowed is the minimal analytic derivative access required to evaluate `lambda_J`. Do not add runtime configuration or report schemas.

Run the test. If it fails a physical gate, preserve the output and stop PR-ODE-R3C as a blocked branch.

## Task 6: Replace the assertionless R3 test with four asserted tests

**Files:**
- Delete the old `pauli_sweep_tangent_converges_to_unforced_action` body from `electron_spectral.rs`.
- Add the four tests to `electron_spectral_temporal_tests.rs`.
- Add a private ordered-sweep helper only if necessary:
  ```rust
  enum PauliSweepOrdering {
      ForwardThenReverse,
      ReverseThenForward,
  }
  ```
  The production `transactional_step` remains a wrapper selecting `ForwardThenReverse`.

**Shared definitions:**

```text
Phi_h(f)       = one transactional sweep
Phi_half2(f)   = Phi_{h/2}(Phi_{h/2}(f))
T(h)           = ||(Phi_h(f)-f)/h - F(f)||
D(h)           = ||Phi_h(f)-Phi_half2(f)||
B(h)           = ||Phi_h^FR(f)-Phi_h^RF(f)||
```

Record three norms independently:

```text
component max over S_action
quadrature-number L1 = sum w_i y_i^2 |delta_i|
quadrature-energy L1 = sum w_i y_i^3 |delta_i|
```

Define local binary64 spacing conservatively:

```rust
fn local_ulp(value: f64) -> f64 {
    (value.next_up() - value)
        .abs()
        .max((value - value.next_down()).abs())
}
```

Resolution gate for every component in `S_action` at the finest rung:

```text
D_i(h_min) / local_ulp(f_i) >= 2^10
```

Otherwise stop with `BLOCKED_TEMPORAL_WINDOW_NOT_RESOLVED`. Do not widen the acceptance bands.

- [ ] **Step 1: Root coverage**

```rust
#[test]
fn root_certificates_cover_rate_scaled_ladder() {
    // Every rung completes.
    // unresolved_edges==0.
    // exact_stationary_edges==0 on this non-equilibrium fixture.
    // all reported bounds are finite.
    // maximum_edge_iterations<=4.
    // every occupation remains in [0,1].
}
```

- [ ] **Step 2: Operator consistency**

```rust
#[test]
fn sweep_tangent_matches_reconstructed_edge_action() {
    // Compute T(h) on all rungs.
    // Adjacent resolved log2 ratios have median in [0.8,1.2].
    // No resolved adjacent ratio is below 0.6.
    // This is a consistency claim, not a method-order claim.
}
```

A half-step double count producing `2F` gives a slope near zero and must fail.

- [ ] **Step 3: Method order**

```rust
#[test]
fn sweep_step_doubling_measures_order() {
    // Compute D(h), assert the D/ulp resolution gate,
    // and fit adjacent log2[D(h)/D(h/2)].
}
```

Classification:

```text
1.8 <= median_slope <= 2.2
    -> VALIDATED_FIRST_ORDER_SWEEP

median_slope >= 2.8
    -> SECOND_ORDER_CANDIDATE_ONLY
       (requires separate adjoint/symmetry evidence)

median_slope < 1.7, non-monotone resolved norms, or norm disagreement
    -> BLOCKED_TEMPORAL_CONSISTENCY
```

Do not classify a slope near `2` as second order.

- [ ] **Step 4: Edge-order bias**

```rust
#[test]
fn sweep_edge_order_bias_decreases_cubically_when_resolved() {
    // Compute B(h).
    // If B(h_min) is above the same 2^10-ulp floor,
    // require median log2 ratio in [2.6,3.4].
    // If below the floor, report INCONCLUSIVE_BELOW_RESOLUTION
    // without using this row as a promotion gate.
}
```

- [ ] **Step 5: Run the exact R3 suite**

```bash
cargo test --lib electron_spectral::temporal_tests::rate_scaled_probe_window_is_safe_and_conditioned -- --exact --nocapture
cargo test --lib electron_spectral::temporal_tests::root_certificates_cover_rate_scaled_ladder -- --exact --nocapture
cargo test --lib electron_spectral::temporal_tests::sweep_tangent_matches_reconstructed_edge_action -- --exact --nocapture
cargo test --lib electron_spectral::temporal_tests::sweep_step_doubling_measures_order -- --exact --nocapture
cargo test --lib electron_spectral::temporal_tests::sweep_edge_order_bias_decreases_cubically_when_resolved -- --exact --nocapture
```

All four property tests must contain load-bearing assertions. A test that only prints is a blocker.

- [ ] **Step 6: Commit**

```bash
git add \
  native/rabbit_cpu/src/electron_spectral.rs \
  native/rabbit_cpu/src/electron_spectral_temporal_tests.rs \
  native/rabbit_cpu/src/pauli_edge_step.rs
git commit -m "ODE-R3C: measure the Pauli sweep in a resolved rate window"
```

## PR-ODE-R3C final verification

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_spectral::tests -- --nocapture
cargo test --lib electron_spectral::temporal_tests -- --nocapture
```

A fresh-context reviewer must inspect:

```text
- no initial_flux==0 default success
- no 0/0 aggregate ratio
- no computed-flux residual presented as a true-root bound without delta_J
- no log-only result used as a certificate
- no cap-only fix
- no raw-h temporal ladder
- no assertionless test
- step-doubling exponent interpreted as p+1
- same deterministic R3 state retained
- no active ODE-driver promotion
- no R4-R11 claim
```

---

## 5. Decision gates after PR-ODE-R3C

### Case A — first-order contract validates

```text
R1_FALSE_CERTIFICATE_CHANNEL       VALIDATED
R1_ROOT_CERTIFICATE_COMPLETENESS   VALIDATED_FOCUSED
R1_ROOT_CERTIFICATE_SOUNDNESS      VALIDATED_FOR_RESOLVED_DIRECT_PRODUCTS
R2_CONDITIONED_DB_COMPATIBILITY    VALIDATED_FOCUSED
R3_OPERATOR_CONSISTENCY            VALIDATED_FOCUSED
R3_TEMPORAL_ORDER                  VALIDATED_FIRST_ORDER_SWEEP
R4-R11                             FORBIDDEN_PENDING_ADJUDICATION
```

Stop. Do not promote the active ODE driver in the same PR. Prepare a separate integration spec.

### Case B — flux conditioning blocks root or R3

Status:

```text
BLOCKED_FLUX_CONDITIONING
```

Do not implement the closed-form root; it solves the wrong problem. The next work unit is improved arithmetic or higher-precision/reference evaluation for the action-carrying unresolved edges.

### Case C — roots remain expensive but flux bounds are resolved

If all soundness gates pass but any edge needs more than four iterations, consider optional PR-ODE-R1Q:

- stable closed-form quadratic backward-Euler root;
- iterative path retained under `#[cfg(test)]` for cross-validation only;
- physical-bracket root selection;
- residual plus `delta_J` remains the verification certificate;
- no claim that the quadratic formula improves flux conditioning.

### Case D — first-order accuracy is valid but insufficient

Only after an endpoint-consumed cost/accuracy study may an exact Riccati edge-flow/Strang candidate be designed. It is not part of this plan.

---

## 6. Required PR cost line

Paste this into every PR body and fill it from the actual diff and executed commands:

```text
added_lines:
deleted_lines:
net_lines:
files_touched:
token_use_exact: UNAVAILABLE
token_use_basis: exact token counter not exposed by the active harness
runtime_behavior_changed: yes/no
physics_behavior_changed: yes/no
known_blocker_reduced: yes/no
blocker_movement_ratio: 0.00..1.00
validation_strengthened: yes/no
cost_effectiveness_verdict: ACCEPT / ACCEPT_WITH_LIMITS / FAILURE_MODE_RELOCATION / NO_PROGRESS / DRIFT
```

For these focused PRs, a segment-only local result must be labeled `focused local solver/operator evidence`, never endpoint progress.

---

## 7. Completion definition

This plan is complete only when one of the following durable outcomes exists:

1. PR-ODE-R3C is independently reviewed and records `VALIDATED_FIRST_ORDER_SWEEP`; or
2. a typed `BLOCKED_FLUX_CONDITIONING`, `BLOCKED_TEMPORAL_WINDOW_UNSAFE`, `BLOCKED_TEMPORAL_WINDOW_NOT_RESOLVED`, or `BLOCKED_TEMPORAL_CONSISTENCY` result is preserved with exact command output and no downstream claim.

A green cargo test produced by removing assertions, increasing only the iteration cap, widening tolerances, replacing the fixture, or accepting an unresolved zero flux is a failure of the plan.
