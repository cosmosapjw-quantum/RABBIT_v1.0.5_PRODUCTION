# Codex Handoff — PR-ODE-R1C

Paste this into a fresh Codex session. Execute **PR-ODE-R1C only**; do not start R2C/R3C.

---

Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`

Base: `plan/ode-r1-r3-remediation-v2-20260824`

Create: `fix/ode-r1c-certified-edge-outcomes-20260824`

PR base: `plan/ode-r1-r3-remediation-v2-20260824`

## Mission

Implement Tasks `R1C-1` through `R1C-3` in:

`docs/audit/ODE_R1_R3_REMEDIATION_EXECUTION_PLAN_R2_2026-08-24.md`

Close only:

```text
R1_FALSE_CERTIFICATE_CHANNEL
R1_CERTIFICATE_COMPLETENESS
R1_CERTIFICATE_SOUNDNESS
```

Do not touch active `OdeSystem::rhs`, raise the 96 cap, widen `128*eps`, clip/project, replace the fixture, add QKE/JAX runtime work, or claim R2C/R3C/endpoint/public production/R4–R11.

## Mandatory read order

1. `AGENTS.md`
2. `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
3. `bbn_codex_anti_drift_cost_effective_policy.md`
4. `docs/audit/ODE_R2_R3_TEMPORAL_BLOCKER_REAUDIT_REQUEST_2026-08-24.md`
5. `docs/audit/ODE_R1_R3_REMEDIATION_EXECUTION_PLAN_R2_2026-08-24.md`

Use one production-code writer, TDD, and one fresh read-only reviewer. Follow `.agent-harness` admission rules if any agent is spawned.

## Worktree and baseline

```bash
git fetch origin
git worktree add ../rabbit-ode-r1c \
  -b fix/ode-r1c-certified-edge-outcomes-20260824 \
  origin/plan/ode-r1-r3-remediation-v2-20260824
cd ../rabbit-ode-r1c
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

Expected: all but the last pass; the last reports `91,93,95` and fails at `h=2^-14`. Otherwise stop as `BASELINE_DIVERGENCE`.

## TDD sequence

### Commit 1 — explicit outcomes

RED tests:

```text
nonzero_step_rejects_positive_traffic_zero_net_as_unresolved
both_zero_products_are_explicit_exact_stationary
zero_step_is_distinct_from_nonzero_exact_stationary
sweep_report_counts_exact_stationary_without_nan_swallow
unresolved_edge_failure_is_transactional_and_observable
```

Implement explicit resolved/exact-stationary/unresolved outcomes, typed failure, finite aggregation, and transactional partial-report failure. `PauliEdgeStep::default()` must not represent an applied success.

```bash
git commit -am "ODE-R1C: expose stationary and unresolved Pauli edges"
```

### Commit 2 — bounded flux

RED tests:

```text
direct_product_certificate_resolves_well_conditioned_flux
near_balance_is_unresolved_when_net_is_below_arithmetic_bound
extreme_log_only_flux_remains_value_available_but_uncertified
resolved_direct_and_value_only_flux_agree_within_the_reported_bound
```

Keep log/`expm1` value-only. Implement the plan's direct-product absolute-error bound and `UnresolvedForCertificate`; add no dependency.

```bash
git commit -am "ODE-R1C: bound certificate-bearing Pauli fluxes"
```

### Commit 3 — root certificate

RED tests:

```text
current_newton_iterate_closes_without_full_bracket_collapse
root_certificates_hold_across_extreme_step_scales
uncertain_flux_cannot_be_hidden_by_bracket_collapse
physical_capacity_bracket_uses_flux_error_intervals
```

Implement exactly:

```text
scale=max(|xi|,h*traffic_upper,MIN_SUBNORMAL)
root_error=|r_hat|+h*delta_J
occupation_error=root_error/min(m1,m2)
current-iterate certification before midpoint fallback
true-residual interval bracket proof
```

Keep cap `96`. Rename the old test to `root_certificates_hold_across_legacy_small_step_scales`; it is root robustness only and must close through `2^-30` in `<=4` iterations.

```bash
git commit -am "ODE-R1C: certify Newton iterates with flux error bounds"
```

## Mutation and verification

Temporarily disable current-iterate acceptance, reproduce the old `2^-14` failure, revert, and store evidence under `.agent-harness`.

Then run:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
cargo test --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
```

Fresh reviewer checks outcome semantics, error-bound derivation, bracket signs, transactionality, unchanged cap/tolerance, and no active-driver/R2C/R3C diff. Fix load-bearing findings before push.

## Push, PR, and stop

```bash
git push -u origin fix/ode-r1c-certified-edge-outcomes-20260824
```

Open a PR against the planning branch. Do not merge. Include exact base/head, changed files, commands/results, mutation result, remaining risks, claim ceiling, and the required cost line.

Stop as BLOCKED rather than adapting if:

```text
PLAN_DEFECT
BLOCKED_FLUX_CONDITIONING
UncertainPhysicalBracket on action-carrying resolved fixtures
legacy ladder needs >4 iterations
a test needs cap/tolerance/fixture/clipping changes
review leaves a load-bearing defect
```

Return:

```text
status:
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
worktree_clean:
```

Verify the remote branch/PR and `git status --short` before claiming completion.
