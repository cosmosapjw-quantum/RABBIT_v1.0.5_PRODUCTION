# Codex Handoff — Close PR #1 Under the Dynamic Audit Contract

Copy this prompt into a **fresh Codex session**. Execute only the correction of existing PR #1. Do not implement R1E, R2C, R3C, R4+, endpoint, or active-driver promotion in this run.

---

Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Existing PR: `#1`  
Remote implementation branch: `fix/ode-r1c-certified-edge-outcomes-20260824`  
Required starting HEAD: `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`  
Contract branch: `external-audit/pr1-r1c-dynamic-compiled-contract-20260825`

## Mission

Close the current P0 false-`Solved` and all PR-#1-local P1 findings by executing work units `PR1-CLOSURE-A` and `PR1-CLOSURE-B` from:

```text
.codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json#audit_compiled_exec_plan
```

The primary deliverable is not a plausible patch. It is an implementation whose every known P0/P1 failure mode fails closed under the compiled tests, invariants, mutations, and fresh-context review.

## Non-interactive policy

```text
DO NOT ASK USER QUESTIONS.
DO NOT GUESS ACROSS A SPECIFICATION BOUNDARY.
Search repository code, tests, contract files, and versioned decisions first.
If authority is missing or contradictory, stop BLOCKED_BY_UNRESOLVED_SPEC with exact evidence.
Do not weaken tests, cap, tolerance, fixture, golden root, or claim ceiling.
Do not suppress failures or replace them with clipping/projection.
```

## Bootstrap

```bash
git fetch origin

test "$(git rev-parse origin/fix/ode-r1c-certified-edge-outcomes-20260824)" = \
  "50d3bc5b8093bc33e9311f94505c5ee0711ce51b" || { echo BASE_DRIFT; exit 2; }

git worktree add ../rabbit-pr1-closure \
  -b local/pr1-r1c-closure \
  origin/fix/ode-r1c-certified-edge-outcomes-20260824
cd ../rabbit-pr1-closure

# Materialize the immutable contract from the external-audit branch.
mkdir -p .codex/audit/pr1-r1c-closure docs/audit
for p in \
  .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json \
  .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py; do
  git show origin/external-audit/pr1-r1c-dynamic-compiled-contract-20260825:$p > $p
done

git show origin/external-audit/pr1-r1c-dynamic-compiled-contract-20260825:docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md \
  > docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md

python3 -m json.tool .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json >/dev/null
python3 .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py \
  | tee /tmp/pr1-r1c-baseline-falsifier.json
```

Expected baseline: exit `0`, containing all three labels:

```text
P0_FALSE_SOLVED
P1_STAGNATED_INTERVAL
P1_EQUILIBRIUM_ABORT
```

Any mismatch is `REPRODUCER_DIVERGENCE`; do not edit production code.

## Required read order

```text
1. AGENTS.md
2. docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md
3. bbn_codex_anti_drift_cost_effective_policy.md
4. docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md
5. .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json
   - p0_p1_threat_catalogue
   - audit_compiled_exec_plan
   - invariant_test_matrix
   - final_independent_audit_contract
   - evidence_bundle_schema
   - dynamic_evidence
```

## Execution rule

Use one production-code writer. Use TDD: write each required regression, run and observe the specified RED, implement the smallest correction, then run GREEN. Do not dispatch a reviewer from the implementation context.

### Commit 1 — P0 exact-real interval certificate

Implement `PR1-CLOSURE-A` exactly.

Required RED fixture:

```rust
let edge = PauliEdge::new(
    PauliEdgeTopology::PairSource,
    0,
    1,
    2.0_f64.powi(8),
    2.0_f64.powi(-36),
    2.0_f64.powi(26),
    2.0_f64.powi(-14),
).unwrap();
let result = edge.implicit_step(
    2.0_f64.powi(-36),
    1.0 - 2.0_f64.powi(-40),
    1.0 - 2.0_f64.powi(-6),
);
let golden = f64::from_bits(0xbcceff07e8a38d5c);
match result {
    Ok((_, report)) => {
        assert_ne!(report.extent.to_bits(), 0xbcceff0807bfa264);
        assert!((report.extent - golden).abs() / 2.0_f64.powi(-36)
            <= 128.0 * f64::EPSILON);
        assert!(report.root_error_abs >= (report.extent - golden).abs());
    }
    Err(error) => assert!(matches!(
        error.kind,
        PauliEdgeFailureKind::UncertainPhysicalBracket
            | PauliEdgeFailureKind::CertificateUnattainableAtStep
            | PauliEdgeFailureKind::StateMapUnresolved
    )),
}
```

The production fix must use outward-rounded certificate-only intervals through:

```text
xi/measure -> initial +/- quotient -> 1-f -> nonnegative products
-> gain-loss flux interval -> h*flux interval -> residual interval
```

The returned candidate remains the normal point binary64 state. Newton derivative is proposal-only. Every `Solved` constructor must call one common interval-acceptance helper.

Commit:

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs
git commit -m "ODE-R1C: enclose affine-state Pauli root certificates"
```

### Commit 2 — P1 robustness and evidence integrity

Implement `PR1-CLOSURE-B` exactly.

Required fallback fixture:

```rust
PauliEdgeTopology::ElasticTransfer
m1=m2=2^-30
A=B=2^-20 MeV
h=1 MeV^-1
f=(1/8,1/4)
```

It must return `StagnatedInterval` on the first repeated extent, not after 96 identical evaluations.

Add `CertificateUnattainableAtStep`; move log/`expm1` work to the unresolved arm; replace numeric endpoint proof with analytic capacity signs; make report anomalies hard failures; aggregate maximum occupation error; update the module claim.

Extend `.github/workflows/harness.yml` with all commands listed in the plan. Do not delete existing checks.

Commit:

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs \
        native/rabbit_cpu/src/electron_spectral.rs \
        .github/workflows/harness.yml
git commit -m "ODE-R1C: fail fast on uncertifiable Pauli steps"
```

## Required verification at final SHA

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --lib pauli_edge_step::tests -- --nocapture
cargo test --locked --lib electron_spectral::tests::sweep_report_counts_exact_stationary_without_nan_swallow -- --exact --nocapture
cargo test --locked --lib electron_spectral::tests::unresolved_edge_failure_is_transactional_and_observable -- --exact --nocapture
cargo test --locked --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
cargo test --locked --lib electron_event_falsifiers::frozen -- --nocapture
cd ../..
python3 .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py
# The old model remains a baseline falsifier; the Rust regression must now reject/cover it.
git diff --check origin/fix/ode-r1c-certified-edge-outcomes-20260824...HEAD
git status --short
```

Run the plan's mutations. Preserve RED and restored-GREEN logs. Generate an evidence bundle satisfying the `evidence_bundle_schema` object in `AUDIT_COMPILED_PACKAGE.json`; every command record must name the final commit SHA.

## Fresh-context reviewer gate

Start a new context containing only:

```text
base SHA
final SHA
base..final diff
compiled package
verification/mutation logs
unresolved blockers
```

Do not include implementation reasoning. Reviewer edits are forbidden. First pass outputs only `P0/P1/P2/P3`, exact file/symbol, violated invariant, reproducer, and evidence. Fix P0/P1 through the implementation context, regenerate all evidence, then rerun a fresh review.

Pass only when:

```text
P0=0
P1=0
unmapped threats=0
missing evidence=0
unresolved spec boundaries=0
contract changes=0
new relevant-suite failures=0
```

## Push and PR update

```bash
git push origin HEAD:fix/ode-r1c-certified-edge-outcomes-20260824
```

Replace PR #1 body with the exact live base/head, three changed-source/workflow files plus any approved tests, command results, mutation evidence, fresh review verdict, claim ceiling, remaining R1E blocker, and cost line. Do not merge.

## Stop boundaries

```text
BASE_DRIFT
REPRODUCER_DIVERGENCE
BLOCKED_BY_UNRESOLVED_SPEC
BLOCKED_TRUE_ROOT_INTERVAL
BLOCKED_UNCERTIFIED_SUCCESS_PATH
CertificateUnattainableAtStep
StagnatedInterval
BLOCKED_REPORT_INTEGRITY
BLOCKED_INCOMPLETE_CI
STALE_EVIDENCE
CONTRACT_AMENDMENT_REQUIRED
```

A typed blocker is an acceptable result. A green produced by cap/tolerance/fixture/golden changes is failure.

## Final response schema

```text
status:
base_sha:
final_sha:
commits:
changed_files:
red_green_tests:
mutation_checks:
required_commands:
full_relevant_suite_delta:
evidence_bundle_path:
fresh_review_P0:
fresh_review_P1:
claim_ceiling:
unresolved_blockers:
pr_url:
pr_body_updated:
worktree_clean:
```
