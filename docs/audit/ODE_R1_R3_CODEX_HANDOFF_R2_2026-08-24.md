# Codex Handoff — ODE R1C Certified Pauli Edge Outcomes

Copy the prompt below into a fresh Codex session. This handoff deliberately starts only the first implementation PR. Do not execute PR-ODE-R2C or PR-ODE-R3C in the same implementation run.

---

You are the single production-code writer for the next RABBIT ODE remediation PR.

## Mission

Implement **PR-ODE-R1C only** from:

`docs/audit/ODE_R1_R3_REMEDIATION_EXECUTION_PLAN_R2_2026-08-24.md`

The PR must close:

```text
R1_FALSE_CERTIFICATE_CHANNEL
R1_ROOT_CERTIFICATE_COMPLETENESS
R1_ROOT_CERTIFICATE_SOUNDNESS
```

It must not claim or implement R2C, R3C, active ODE integration, endpoint validation, R4–R11, QKE, public production support, or a second-order method.

## Canonical repository and branches

```text
repository:
  cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION

planning branch:
  plan/ode-r1-r3-remediation-v2-20260824

evidence parent:
  external-audit/ode-r2-r3-temporal-blocker-20260824

implementation branch to create:
  fix/ode-r1c-certified-edge-outcomes-20260824

PR base:
  plan/ode-r1-r3-remediation-v2-20260824
```

Do not work on `main`. Do not merge anything.

## Mandatory read order

Read these files completely, in this order:

1. `AGENTS.md`
2. `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
3. `bbn_codex_anti_drift_cost_effective_policy.md`
4. `docs/audit/ODE_R2_R3_TEMPORAL_BLOCKER_REAUDIT_REQUEST_2026-08-24.md`
5. `docs/audit/ODE_R1_R3_REMEDIATION_EXECUTION_PLAN_R2_2026-08-24.md`

The execution plan already incorporates the attached external static audit. Do not invent a different remediation from memory.

## Non-negotiable design rulings

1. `initial_flux == 0.0` is not a generic success path.
2. A non-zero-step edge application must be classified as:
   - resolved and solved;
   - provably exact stationary because both exact products are zero; or
   - unresolved and failed.
3. `PauliEdgeStep::default()` must not encode a successful applied edge.
4. No `0.0/0.0 -> NaN -> f64::max` aggregation path may remain.
5. The root certificate must include:
   ```text
   root_error_bound = |computed residual| + h * flux_abs_error_bound
   ```
6. The residual scale must include gross traffic:
   ```text
   max(|xi|, h * traffic_upper_bound, MIN_SUBNORMAL)
   ```
7. The current Newton iterate is checked for certification before midpoint fallback.
8. The iteration cap remains `96`.
9. Do not widen `128*eps`, clip occupations, replace the fixture, force equilibrium, or add a feature flag.
10. The existing log/`expm1` evaluation may remain value-only. It cannot provide a certificate when the direct product bound is unavailable.
11. The direct-product error constants and classification rules in the plan are exact plan requirements. If you find a mathematical contradiction in them, record a `PLAN_DEFECT` with a minimal counterexample and stop before production implementation; do not silently substitute another formula.
12. The active `OdeSystem::rhs` path is untouched.

## Workflow

Use an isolated worktree. Use TDD. One behavior per test. Observe RED before writing production code.

```bash
git fetch origin
git worktree add ../rabbit-ode-r1c \
  -b fix/ode-r1c-certified-edge-outcomes-20260824 \
  origin/plan/ode-r1-r3-remediation-v2-20260824
cd ../rabbit-ode-r1c

python3 .agent-harness/scripts/build_context_pack.py
git status --short
git rev-parse HEAD
```

If you spawn any reviewer/subagent, follow the repository admission protocol exactly. Do not use parallel production-code writers.

## Immutable preflight

From `native/rabbit_cpu` run:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_spectral::tests::unforced_fd_equilibrium_is_an_event_and_edge_null -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::pauli_sweep_tangent_converges_to_unforced_action -- --exact --nocapture
```

Expected:

```text
format/check/clippy/R1/R2 focused tests: PASS
R3 command: FAIL at h=2^-14
successful prefix maximum iterations: 91, 93, 95
failure: Pauli edge implicit root did not converge
```

If this exact baseline is not reproduced, stop and report `BASELINE_DIVERGENCE`. Do not modify source.

## Execute the plan task-by-task

Implement only Tasks 1–3 under `PR-ODE-R1C`.

### Task 1 — explicit outcomes and aggregation

Write the failing tests first:

```text
nonzero_step_rejects_positive_traffic_zero_net_as_unresolved
both_zero_products_are_explicit_exact_stationary
zero_step_is_distinct_from_nonzero_exact_stationary
sweep_report_counts_exact_stationary_without_nan_swallow
unresolved_edge_failure_is_transactional_and_observable
```

Then implement the exact interfaces and report fields specified in the plan.

Commit:

```bash
git commit -m "ODE-R1C: expose stationary and unresolved Pauli edges"
```

### Task 2 — bounded certificate-bearing flux

Write the failing tests first:

```text
direct_product_certificate_resolves_well_conditioned_flux
near_balance_is_unresolved_when_net_is_below_arithmetic_bound
extreme_log_only_flux_remains_value_available_but_uncertified
resolved_direct_and_value_only_flux_agree_within_the_reported_bound
```

Implement the direct non-negative product bound and resolution classification exactly as specified. Do not add a dependency.

Commit:

```bash
git commit -m "ODE-R1C: bound certificate-bearing Pauli fluxes"
```

### Task 3 — root completeness and soundness

Write the failing tests first:

```text
current_newton_iterate_closes_without_full_bracket_collapse
root_certificates_hold_across_extreme_step_scales
uncertain_flux_cannot_be_hidden_by_bracket_collapse
physical_capacity_bracket_uses_flux_error_intervals
```

Implement:

```text
current-iterate certification
gross-traffic residual scale
|r_hat| + h*delta_J root bound
occupation error bound
true-residual interval bracket check
typed unresolved/bracket/iteration failures
```

Rename the old raw ladder test to:

```text
root_certificates_hold_across_legacy_small_step_scales
```

It is a root-robustness fixture only. It must not retain a tangent/order claim.

Commit:

```bash
git commit -m "ODE-R1C: certify Newton iterates with flux error bounds"
```

## Required mutation check

After the normal implementation is green:

1. temporarily disable only the current-iterate success branch;
2. run the legacy `h=2^-14` fixture;
3. confirm the old non-convergence returns;
4. revert the mutation;
5. verify `git status --short` contains no mutation residue.

Store the command/output in the current `.agent-harness` evidence directory. Never commit the mutation.

## Required verification

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
cargo test --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
```

Also inspect the diff and prove:

```text
max_iterations is still 96
128*eps is not widened
no clipping/projection was added
no active ODE-driver call site changed
no R2C/R3C code was added
no assertionless replacement test exists
```

## Fresh-context review

After all implementation commits and tests:

- mint one read-only fresh-context reviewer assignment through `.agent-harness`;
- reviewer reads the plan, the changed files, and the focused test output;
- reviewer checks type semantics, arithmetic-bound reasoning, bracket inequalities, transactionality, and claim ceiling;
- fix load-bearing findings before push;
- do not let the reviewer edit production code.

## Push and PR

Only after verification and review:

```bash
git push -u origin fix/ode-r1c-certified-edge-outcomes-20260824
```

Open a PR against:

```text
plan/ode-r1-r3-remediation-v2-20260824
```

Do not merge it.

The PR body must include:

```text
scope
exact base and head SHA
changed files
commands run with pass/fail
mutation-check result
remaining risks
claim statuses
the required cost line
```

Use this claim ceiling:

```text
R1_FAIL_CLOSED_SEMANTICS              VALIDATED
R1_FALSE_CERTIFICATE_CHANNEL          VALIDATED only if all explicit-outcome tests pass
R1_ROOT_CERTIFICATE_COMPLETENESS      VALIDATED_FOCUSED only if the legacy ladder closes <=4 iterations
R1_ROOT_CERTIFICATE_SOUNDNESS         VALIDATED_FOR_RESOLVED_DIRECT_PRODUCTS only
R2_CONDITIONED_DB_SENSITIVITY         NOT_YET_EVALUATED
R3_TEMPORAL_CONSISTENCY               NOT_YET_EVALUATED
ACTIVE_ODE_DRIVER_INTEGRATION         FORBIDDEN
R4-R11                                FORBIDDEN
```

Required cost line:

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

## Stop conditions

Stop immediately and push only a blocked evidence branch, not a green PR, if any of these occurs:

```text
BASELINE_DIVERGENCE
PLAN_DEFECT
BLOCKED_FLUX_CONDITIONING
UncertainPhysicalBracket on action-carrying resolved fixtures
legacy h=2^-14 still needs >4 iterations
a required test can only pass by widening tolerance or raising the cap
fresh reviewer finds an unresolved load-bearing certificate defect
```

Do not continue into PR-ODE-R2C or PR-ODE-R3C.

## Final response contract

Return exactly:

```text
status: COMPLETE | BLOCKED
branch:
base_sha:
head_sha:
commits:
files_changed:
commands_run:
tests_passed:
tests_failed:
mutation_check:
review_result:
claim_status:
cost_line:
pr_url:
remaining_blocker:
worktree_clean: yes/no
```

Do not claim a push, PR, validation, or clean worktree without re-reading the remote branch/PR and running `git status --short`.
