# ODE R1–R3 Temporal Blocker Remediation v2

> **Execution rule:** single production-code writer, TDD, one fresh-context reviewer per PR. Implement PRs sequentially; stop on any typed blocker.

**Goal:** make the local Pauli edge solve honest and complete for resolved binary64 fluxes, replace the vacuous R2 detailed-balance claim with a sensitivity-controlled gate, then measure the sweep in a dimensionless, representable temporal window.

**Authority:** `docs/audit/ODE_R2_R3_TEMPORAL_BLOCKER_REAUDIT_REQUEST_2026-08-24.md`, the attached 2026-08-24 static audit, `AGENTS.md`, the Type-I anti-drift guardrails, and the cost-effective policy.

## 1. Adopted findings

The external audit confirms:

1. The root loop certifies only the midpoint. Once Newton becomes a bracket endpoint, the strict-interior test forces bisection. The `91,93,95` sequence predicts `97>96` at `h=2^-14`.
2. `initial_flux==0.0` is a second non-zero-step bypass. It returns `PauliEdgeStep::default()`; aggregation then swallows `0/0=NaN`.
3. The R3 test has no assertions and discards the post-step state. Raising the cap would create a silent false green.
4. The certificate bounds the root of the computed flux, not the exact-real flux of the binary64 inputs. Its scale uses `h|J|` rather than gross traffic.
5. The raw `h=2^-8...2^-14 MeV^-1` ladder is below the step-doubling resolution floor.
6. For `D(h)=||Phi_h-Phi_{h/2}^2||`, the fitted log2 ratio is `p+1`; first order gives `2`.
7. Use `eta_J=h max_e|dJ_e/dxi|` for the ladder. Use `eta_cap=h max_i|F_i|/min(f_i,1-f_i)` only as a safety gate.

### Design ruling

Do not turn the audit's approximate `ln`-error estimate into a theorem. Keep the current log/`expm1` path for value-only evaluation. Certificate-bearing evaluation uses direct non-negative products with a conservative IEEE-754 absolute-error bound when representable; otherwise it returns `UnresolvedForCertificate`.

## 2. Current status

```text
R1_FAIL_CLOSED_SEMANTICS             PASS
R1_FALSE_CERTIFICATE_CHANNEL         BLOCKED
R1_CERTIFICATE_COMPLETENESS          BLOCKED
R1_CERTIFICATE_SOUNDNESS             BLOCKED
R2_EDGE_RECONSTRUCTION               VALIDATED_FOCUSED
R2_RAW_DB_BELOW_F64_FLOOR            FORBIDDEN
R2_CONDITIONED_DB_SENSITIVITY        NOT_YET_EVALUATED
R3_TEST_HAS_ASSERTIONS               BLOCKED
R3_WINDOW_RESOLVABLE                 BLOCKED
R3_TEMPORAL_CONSISTENCY              NOT_YET_EVALUATED
R4-R11                               FORBIDDEN
```

## 3. Non-goals and invariants

- No active `OdeSystem::rhs` change.
- No QKE, JAX runtime development, endpoint/public-production claim, R4–R11, clipping, projection, equilibrium forcing, tolerance widening, or cap-only `96->128`.
- No new registry, manifest, feature flag, readiness wrapper, or per-test audit note.
- Preserve raw failures. A typed blocked result is valid; a silent green is not.
- Every PR reports added/deleted/net lines, exact token availability, blocker movement, validation, and cost-effectiveness.

## 4. PR DAG

```text
PR-ODE-R1C  explicit outcomes + bounded flux certificate + root completeness
   -> PR-ODE-R2C  conditioned DB gate + 1e-10 gain perturbation
   -> PR-ODE-R3C  rate-scaled root/tangent/step-doubling/order-bias tests
   -> STOP: validated first order or typed blocker
```

Implementation files:

```text
native/rabbit_cpu/src/pauli_edge_step.rs
native/rabbit_cpu/src/electron_spectral.rs
native/rabbit_cpu/src/electron_spectral_temporal_tests.rs  # new, cfg(test)
```

## 5. Immutable preflight

```bash
git fetch origin
git switch plan/ode-r1-r3-remediation-v2-20260824
git pull --ff-only
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

Expected: all but the final command pass; R3 prints `91,93,95` and fails at `2^-14`. Any divergence is `BASELINE_DIVERGENCE`; stop without editing.

# PR-ODE-R1C

**Blockers:** false certificate, computed-flux-only soundness, midpoint-only completeness.

**Line budget:** net 160–320; no new dependency.

## Task R1C-1 — explicit outcomes

Add the following minimal contract:

```rust
enum PauliFluxResolution { Resolved, ExactZero, UnresolvedForCertificate }

struct PauliFluxEvaluation {
    net_mev: f64,
    traffic_upper_bound_mev: f64,
    abs_error_bound_mev: f64,
    resolution: PauliFluxResolution,
}

enum PauliEdgeApplicationKind { Solved, ExactStationary }

enum PauliEdgeFailureKind {
    InvalidInput,
    UnresolvedFlux,
    UncertainPhysicalBracket,
    InvalidResidual,
    IterationLimit,
}
```

`PauliEdgeStep` carries: kind, extent, iterations, residual/scale, flux error, root error, occupation error, bracket width, and conditioning lower bound.

`PauliSweepReport` adds: solved, exact-stationary, unresolved counts; maximum flux-error fraction; maximum root-error bound. A failed sweep carries the partial report and failing edge identity. Input banks remain bitwise unchanged.

Write and observe RED:

```text
nonzero_step_rejects_positive_traffic_zero_net_as_unresolved
both_zero_products_are_explicit_exact_stationary
zero_step_is_distinct_from_nonzero_exact_stationary
sweep_report_counts_exact_stationary_without_nan_swallow
unresolved_edge_failure_is_transactional_and_observable
```

Rules:

- remove `Default` as an applied-edge success;
- exact stationary requires both exact products to be zero because a coefficient/factor is exactly zero;
- positive equal products are unresolved;
- ratio aggregation runs only for solved edges with positive finite scale;
- every aggregate field is finite.

Commit:

```bash
git commit -am "ODE-R1C: expose stationary and unresolved Pauli edges"
```

## Task R1C-2 — bounded flux evaluation

Keep `flux_mev` value-only. Add `certified_flux_evaluation`.

Use:

```rust
const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const PRODUCT_ABS_ERROR_FACTOR: f64 = 8.0 * f64::EPSILON;
const DIFFERENCE_ABS_ERROR_FACTOR: f64 = 2.0 * f64::EPSILON;
```

For each representable three-factor product:

```text
product_error = 8 eps |p_hat| + 4 MIN_SUBNORMAL
difference_error = 2 eps (|G_hat|+|L_hat|) + 2 MIN_SUBNORMAL
delta_J = gain_error + loss_error + difference_error
traffic_upper = |G_hat|+|L_hat|+gain_error+loss_error
```

Classify exact zero only when both products are provably zero; resolved only when `|J_hat|>delta_J`; otherwise unresolved. Positive-product underflow/overflow is unresolved for certificates. Define:

```text
kappa_lower=max(|J_hat|-delta_J,0)/max(traffic_upper,MIN_SUBNORMAL)
```

Write and observe RED:

```text
direct_product_certificate_resolves_well_conditioned_flux
near_balance_is_unresolved_when_net_is_below_arithmetic_bound
extreme_log_only_flux_remains_value_available_but_uncertified
resolved_direct_and_value_only_flux_agree_within_the_reported_bound
```

Run `cargo test --lib pauli_edge_step::tests -- --nocapture` and the frozen event falsifiers.

Commit:

```bash
git commit -am "ODE-R1C: bound certificate-bearing Pauli fluxes"
```

## Task R1C-3 — root certificate

At candidate `xi` compute:

```text
r_hat = xi - h J_hat
scale = max(|xi|, h traffic_upper, MIN_SUBNORMAL)
root_error = |r_hat| + h delta_J
occupation_error = root_error/min(m1,m2)
```

Accept the current Newton iterate before midpoint fallback only when:

```text
|r_hat| <= 128 eps scale + MIN_SUBNORMAL
occupation_error <= 128 eps
flux resolution == Resolved
```

At the initial state, `ExactZero` returns explicit `ExactStationary`; unresolved returns `UnresolvedFlux`.

True-root bracket proof:

```text
lower_r_hat + h lower_delta_J <= 0
upper_r_hat - h upper_delta_J >= 0
```

Otherwise return `UncertainPhysicalBracket`. Keep `max_iterations=96`.

Write and observe RED:

```text
current_newton_iterate_closes_without_full_bracket_collapse
root_certificates_hold_across_extreme_step_scales
uncertain_flux_cannot_be_hidden_by_bracket_collapse
physical_capacity_bracket_uses_flux_error_intervals
```

Rename the old raw ladder to `root_certificates_hold_across_legacy_small_step_scales`. It is root robustness only. Require every rung through `2^-30` to close in at most four iterations.

Mutation check: temporarily disable only current-iterate acceptance; verify the old `2^-14` failure returns; revert and record the evidence.

Commit:

```bash
git commit -am "ODE-R1C: certify Newton iterates with flux error bounds"
```

R1C gate:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
cargo test --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
```

Fresh reviewer must approve outcome semantics, arithmetic bounds, bracket inequalities, transactionality, unchanged cap/tolerance, and no active-driver diff.

# PR-ODE-R2C

Replace the raw `1e-15` claim with a condition-aware test-only metric:

```text
resolved_excess=max(|J_hat|-delta_J,0)
normalized_excess=sum(resolved_excess)/sum(traffic_upper)
error_fraction=delta_J/traffic_upper
```

Base test `unforced_fd_equilibrium_is_conditioned_near_balance` requires:

```text
all value fluxes finite
max error_fraction <=64 eps
normalized_excess <=1e-12
max resolved edge ratio <=1e-12
unresolved-near-balance count printed explicitly
```

Negative control `conditioned_detailed_balance_detects_gain_perturbation` selects the maximum-traffic edge, multiplies gain by `1+1e-10`, and requires a resolved edge ratio `>=1e-11`.

Retain the five-state reconstruction, boundary-inward, and no-clipping tests.

Claim ceiling:

```text
R2_EDGE_RECONSTRUCTION            VALIDATED_FOCUSED
R2_CONDITIONED_DB_COMPATIBILITY   VALIDATED_FOCUSED
R2_DB_SENSITIVITY_CONTROL         VALIDATED
R2_RAW_DB_BELOW_F64_FLOOR         FORBIDDEN
```

# PR-ODE-R3C

Create `electron_spectral_temporal_tests.rs` under `#[cfg(test)]`; delete the assertionless R3 test. Keep the same grid/rules/temperatures/alternating state.

Compute:

```text
lambda_J=max_e|dJ_e/dxi|
lambda_cap=max_i |F_i|/min(f_i,1-f_i)
eta_J=h lambda_J
eta_cap=h lambda_cap
eta_J ladder={2^-4,2^-5,2^-6,2^-7,2^-8}
h_k=eta_J_k/lambda_J
```

Require `eta_cap(h_max)<=1/16`.

Define:

- `S_action`: smallest component set carrying 99.9% of `sum w y^2 |F|`;
- `E_flux`: smallest edge set carrying 99.9% of `sum |J|`.

On `E_flux`, require:

```text
kappa=|J|/traffic_upper >= (2^10 eps)/eta_J_max ~=3.64e-12
```

Four asserted tests:

```text
rate_scaled_probe_window_is_safe_and_conditioned
root_certificates_cover_rate_scaled_ladder
sweep_tangent_matches_reconstructed_edge_action
sweep_step_doubling_measures_order
sweep_edge_order_bias_decreases_cubically_when_resolved
```

Quantities:

```text
T(h)=||(Phi_h-f)/h-F(f)||        expected log2 ratio ~1
D(h)=||Phi_h-Phi_{h/2}^2||       expected ratio ~2 for first order
B(h)=||Phi_h^FR-Phi_h^RF||       expected ratio ~3 when resolved
```

Use component max, quadrature-number L1, and quadrature-energy L1. At the finest rung, every component in `S_action` must satisfy:

```text
D_i/local_ulp(f_i) >=2^10
```

Classify:

```text
median slope(D) in [1.8,2.2] -> VALIDATED_FIRST_ORDER_SWEEP
median slope(D) >=2.8        -> SECOND_ORDER_CANDIDATE_ONLY
slope(D)<1.7/non-monotone/norm disagreement -> BLOCKED_TEMPORAL_CONSISTENCY
```

If the window is unsafe, unresolved, or ill-conditioned, preserve the typed result and stop; do not change the state or tolerance.

## 6. Deferred alternatives

- Closed-form quadratic backward-Euler root is allowed only if flux bounds are resolved but iterations remain `>4`. It retains the iterative path for test cross-validation and does not claim to solve conditioning.
- Exact Riccati edge flow/Strang is deferred until first-order accuracy is measurable and an endpoint-consumed cost study shows need.

## 7. Completion and PR cost line

Stop after R3 adjudication. No active-driver promotion in the same PR.

```text
added_lines:
deleted_lines:
net_lines:
files_touched:
token_use_exact: UNAVAILABLE
token_use_basis: exact counter not exposed
runtime_behavior_changed: yes/no
physics_behavior_changed: yes/no
known_blocker_reduced: yes/no
blocker_movement_ratio: 0.00..1.00
validation_strengthened: yes/no
cost_effectiveness_verdict: ACCEPT / ACCEPT_WITH_LIMITS / FAILURE_MODE_RELOCATION / NO_PROGRESS / DRIFT
```

Completion is either independently reviewed `VALIDATED_FIRST_ORDER_SWEEP` or a durable typed blocker with exact command output. A green obtained by cap/tolerance changes, missing assertions, fixture replacement, clipping, or unresolved-zero acceptance is failure.
