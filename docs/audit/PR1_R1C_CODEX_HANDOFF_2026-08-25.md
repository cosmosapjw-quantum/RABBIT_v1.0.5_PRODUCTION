# Codex Handoff — Close PR #1 Under Compiled Contract v2

Copy this entire prompt into a **fresh Codex implementation session**. Execute
only the correction of existing PR #1. Do not implement `PR-ODE-R1E`, R2C,
R3C, R4+, endpoint work, or active-driver promotion.

---

Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Existing PR: `#1`  
Implementation branch: `fix/ode-r1c-certified-edge-outcomes-20260824`  
Required implementation HEAD: `50d3bc5b8093bc33e9311f94505c5ee0711ce51b`  
Contract branch: `external-audit/pr1-r1c-dynamic-compiled-contract-v2-20260825`  
Frozen contract release commit: `635939939295122b05e6e086b87ee1a128f9afcd`

## 1. Mission

Amend PR #1 by executing exactly:

```text
PR1-CLOSURE-A  exact-real interval authority for every Solved result
PR1-CLOSURE-B  fallback/taxonomy/hot-path/report/CI/evidence closure
```

The deliverable is not a plausible patch. Every catalogue P0/P1 that is assigned
to PR #1 must be mechanically closed. The three deliberately deferred P1 items
must remain explicit typed blockers and must not be implemented in this run.

```text
must_close_in_pr1:
  P0-001 P0-002 P0-003
  P1-002 P1-003 P1-004 P1-005 P1-006 P1-007

deferred_typed_blocker:
  P1-001 exact detailed-balance semantics
  P1-008 accumulated frozen-error budget
  P1-009 local-to-global scientific claim firewall
```

Do **not** require total `P1=0`. Require zero open must-close P1 findings and
zero deferred items lacking their exact blocker.

## 2. Non-interactive policy

```text
DO NOT ASK USER QUESTIONS.
DO NOT GUESS ACROSS A SPECIFICATION BOUNDARY.
Search code, tests, frozen contract files, and versioned decisions first.
If authority is absent or contradictory, stop BLOCKED_BY_UNRESOLVED_SPEC.
Do not edit the contract package.
Do not weaken cap, tolerance, fixtures, golden bits, or assertions.
Do not replace failures with clipping, projection, or output truncation.
Do not start PR-ODE-R1E, R2C, or R3C.
```

All paths in the contract are repository-rooted. `/src/**` means only the
repository-root Python package and does not match `/native/...`.

## 3. Bootstrap and contract materialization

```bash
set -euo pipefail

git fetch origin

test "$(git rev-parse origin/fix/ode-r1c-certified-edge-outcomes-20260824)" = \
  "50d3bc5b8093bc33e9311f94505c5ee0711ce51b" \
  || { echo BASE_DRIFT; exit 2; }

git worktree add ../rabbit-pr1-closure-v2 \
  -b local/pr1-r1c-closure-v2 \
  origin/fix/ode-r1c-certified-edge-outcomes-20260824
cd ../rabbit-pr1-closure-v2

CONTRACT_COMMIT=635939939295122b05e6e086b87ee1a128f9afcd
mkdir -p .codex/audit/pr1-r1c-closure docs/audit

for path in \
  .codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json \
  .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json \
  .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py \
  .codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch \
  docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md; do
  git show "$CONTRACT_COMMIT:$path" > "$path"
done

python3 -m json.tool \
  .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json >/dev/null
python3 -m json.tool \
  .codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json >/dev/null

python3 - <<'PY'
import json, subprocess
from pathlib import Path
manifest = json.loads(Path(
    '.codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json'
).read_text())
for item in manifest['components']:
    path = item['path'].lstrip('/')
    actual = subprocess.check_output(['git', 'hash-object', path], text=True).strip()
    assert actual == item['blob_sha'], (path, actual, item['blob_sha'])
print('CONTRACT_COMPONENT_IDENTITIES_OK')
PY
```

Any identity mismatch is `CONTRACT_AMENDMENT_REQUIRED`. Do not continue.

## 4. Mandatory read order

```text
1. AGENTS.md
2. docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md
3. bbn_codex_anti_drift_cost_effective_policy.md
4. docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md
5. .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json
   - threat_partition
   - p0_p1_threat_catalogue
   - mathematical_contract
   - invariant_test_matrix
   - audit_compiled_exec_plan
   - final_independent_audit_contract
   - evidence_bundle_schema
```

## 5. Baseline evidence — do not edit production yet

### PRE-001 — audit-model package consistency

```bash
python3 .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py \
  --baseline-audit-model \
  | tee /tmp/pr1-r1c-baseline-audit-model.json

grep -q 'P0_FALSE_SOLVED' /tmp/pr1-r1c-baseline-audit-model.json
grep -q 'P1_STAGNATED_INTERVAL' /tmp/pr1-r1c-baseline-audit-model.json
grep -q 'P1_EQUILIBRIUM_ABORT' /tmp/pr1-r1c-baseline-audit-model.json
```

This checks the released audit model and exact-real golden data only. It is not a
Rust oracle. Failure is `AUDIT_MODEL_DIVERGENCE`.

### PRE-002B — exact-head Rust known-bad observation

```bash
patch=.codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch
git apply "$patch"
trap 'git apply -R .codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch >/dev/null 2>&1 || true' EXIT
(
  cd native/rabbit_cpu
  cargo test --locked --lib \
    pauli_edge_step::tests::audit_baseline_power_fixture_returns_known_bad_bits \
    -- --exact --nocapture
)
git apply -R "$patch"
trap - EXIT
git diff --exit-code -- native/rabbit_cpu/src/pauli_edge_step.rs
```

Expected: Rust at exactly `50d3bc5` returns `0xbcceff0807bfa264` as `Solved`.
Mismatch is `REPRODUCER_MODEL_DIVERGENCE`. Do not edit production code.

### PRE-003 — existing focused suite

```bash
cd native/rabbit_cpu
cargo test --locked --lib pauli_edge_step::tests -- --nocapture
cd ../..
```

Failure is `BASELINE_DIVERGENCE`.

## 6. TDD rule

For each behavior:

```text
1. add the required regression first;
2. run the exact test and observe the intended RED;
3. record the RED log;
4. implement the smallest production correction;
5. run GREEN;
6. commit;
7. never modify the test to fit the implementation.
```

After adding the primary future-contract regression, it must fail against the
old source. If it passes immediately, stop `BASELINE_RED_NOT_OBSERVED`.

Use one production-code writer. Do not create a reviewer from this context.

# PR1-CLOSURE-A — exact-real root interval

Allowed production path:

```text
/native/rabbit_cpu/src/pauli_edge_step.rs
```

Forbidden:

```text
/native/rabbit_cpu/Cargo.toml
/src/**
/data/**
/tests/fixtures/**
cap 96
128*f64::EPSILON
golden and known-bad bits
```

## A1. Primary RED regression

Add `power_of_two_state_map_rounding_never_false_solves` with exactly:

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
    )),
}
```

Run and observe RED before production changes.

## A2. Exact interval primitives

Create private certificate-only interval operations. Use outward rounding for
every operation. Do not reuse these intervals as returned physical states.

Required conceptual API:

```rust
#[derive(Clone, Copy, Debug)]
struct Interval { lo: f64, hi: f64 }

impl Interval {
    fn point(value: f64) -> Result<Self, PauliEdgeFailure>;
    fn add(self, rhs: Self) -> Result<Self, PauliEdgeFailure>;
    fn sub(self, rhs: Self) -> Result<Self, PauliEdgeFailure>;
    fn mul_nonnegative(self, rhs: Self) -> Result<Self, PauliEdgeFailure>;
    fn div_positive(self, rhs: f64) -> Result<Self, PauliEdgeFailure>;
    fn scale_nonnegative(self, rhs: f64) -> Result<Self, PauliEdgeFailure>;
    fn complement_unit(self) -> Result<Self, PauliEdgeFailure>;
}
```

Rules:

```text
add: [next_down(a.lo+b.lo), next_up(a.hi+b.hi)]
sub: [next_down(a.lo-b.hi), next_up(a.hi-b.lo)]
div by positive m: outward-round each endpoint
nonnegative multiply: [next_down(a.lo*b.lo), next_up(a.hi*b.hi)]
1-[lo,hi]: [next_down(1-hi), next_up(1-lo)]
positive underflow: include zero and a positive upper bound; never ExactZero
nonfinite or unordered interval: typed failure
```

Required pipeline:

```text
Interval::point(xi)
 -> xi/m_i
 -> initial +/- quotient
 -> 1-f
 -> gain and loss products
 -> gain-loss flux interval
 -> h*flux interval
 -> Interval::point(xi) - h*flux interval
```

Required signatures:

```rust
fn affine_occupation_intervals(
    &self, initial: [f64; 2], extent: f64
) -> Result<[Interval; 2], PauliEdgeFailure>;

fn certified_flux_interval(
    &self, occupations: [Interval; 2]
) -> Result<PauliFluxInterval, PauliEdgeFailure>;

fn residual_interval(
    &self, initial: [f64; 2], extent: f64, h: f64
) -> Result<RootResidualInterval, PauliEdgeFailure>;
```

For residual interval `[lo,hi]`:

```text
root_error_abs = max(abs(lo), abs(hi))
occupation_error_abs = root_error_abs/min(m1,m2)
```

## A3. One authority for every Solved path

Create one helper, for example:

```rust
fn try_certify_solved(...) -> Result<Option<PauliEdgeStep>, PauliEdgeFailure>
```

Both current-iterate and midpoint return paths must call it. A source census must
find no other direct `PauliEdgeApplicationKind::Solved` constructor.

The point binary64 state and `flux_derivative_by_extent` are proposal-only.
They may not authorize success.

## A4. Physical-root theorem and bracket

Add the two-topology sign derivation from
`LEMMA-R1C-UNIQUE-PHYSICAL-ROOT` as a code comment near the derivative. Add:

```text
pauli_edge_step::tests::flux_derivative_is_nonpositive_on_physical_box
```

Test both topologies on a deterministic grid of Pauli-box states, nonnegative
coefficients, and positive measures. Assert `dJ/dxi <= 0.0` and therefore
`1-h*dJ/dxi >= 1.0` for nonnegative test steps.

The capacity bounds establish an analytic outer enclosure of the unique physical
root. Do not require a flux evaluation at a boundary occupation. If capacity
arithmetic is outward-rounded, keep its endpoints as an outer root enclosure;
only sign-certified interior point evaluations may tighten it. If an outer
capacity endpoint prevents a valid interior candidate, fail `StateMapUnresolved`
rather than evaluating outside the Pauli box.

## A5. External golden coverage

Generalize the test helper so a supplied golden root is checked in occupation
units:

```rust
fn assert_root_certificate_against_golden(
    report: PauliEdgeStep,
    golden: f64,
    min_measure: f64,
) {
    let actual = (report.extent - golden).abs() / min_measure;
    assert!(actual <= 128.0 * f64::EPSILON);
    assert!(report.occupation_error_abs >= actual);
    assert!(report.occupation_error_abs <= 128.0 * f64::EPSILON);
}
```

Apply it to the primary fixture when `Solved`, and to the local PairSource
ladder:

```text
m=(2,5), A=13, B=17, f=(0.23,0.79)
2^-8   0xbf6e4ad0bfc2b909
2^-14  0xbf0f8e82450eec7e
2^-20  0xbeaf93c82712ec39
2^-30  0xbe0f93dd9299eefb
```

The local ladder must remain `Solved`; typed failure is not accepted there.

## A6. Mutations

Run and preserve evidence for each temporary mutation:

```text
restore the old direct-product-only residual bound
round one interval operation inward
bypass try_certify_solved on the current path
bypass try_certify_solved on the midpoint path
```

Each must make the appropriate P0 regression RED. Restore and rerun GREEN.

Commit only after GREEN:

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs
git commit -m "ODE-R1C: enclose affine-state Pauli root certificates"
```

# PR1-CLOSURE-B — P1 robustness and evidence

Allowed paths:

```text
/native/rabbit_cpu/src/pauli_edge_step.rs
/native/rabbit_cpu/src/electron_spectral.rs
/.github/workflows/harness.yml
```

## B1. Stagnated interval

Use only this fixture:

```text
ElasticTransfer
m1=m2=2^-30
A=B=2^-20 MeV
h=1 MeV^-1
f=(1/8,1/4)
```

Add `StagnatedInterval`. Return it on the first bitwise-repeated next extent with
an unchanged bracket. Do not wait for the 96 cap.

## B2. CertificateUnattainableAtStep

Do not gate this outcome with `h*delta_J/min(m)` or `eta_J`. Those are diagnostics,
not universal theorems.

Emit `CertificateUnattainableAtStep` only after:

```text
1. the unique physical root remains enclosed;
2. current, midpoint, lower, and upper acceptance were attempted;
3. lower and upper are adjacent binary64 extents;
4. no representable extent lies between them;
5. occupation bracket width remains >128 eps;
6. no candidate passed try_certify_solved.
```

Add `adjacent_uncertifiable_root_interval_has_typed_outcome`.

A repeated non-adjacent interval is `StagnatedInterval`, not unattainable.

## B3. Hot path

Move `flux_mev` and all log/`exp`/`expm1` work into the unresolved direct-product
arm. Add a test-only counter or injectable hook proving a resolved direct path
does not call it. Remove the hook outside `#[cfg(test)]`.

## B4. Report integrity

Make `record_edge_certificate` return `Result`. A `Solved` report with nonfinite,
nonpositive, or internally inconsistent evidence is a hard error, not a skipped
aggregate.

Add:

```text
maximum_occupation_error_bound
```

to `PauliSweepReport`.

Run the two existing sweep-level B1 regressions and add
`malformed_solved_report_is_rejected_not_skipped`.

## B5. Non-degeneration gate

After B1–B4, the following remain mandatory:

```text
local golden ladder: all Solved and externally covered
sweep legacy ladder:
  unresolved == 0
  exact_stationary == 0
  solved == edge_applications
  maximum_occupation_error_bound <= 128 eps
  all aggregate fields finite
```

If established resolved fixtures become generic typed failures, stop
`BLOCKED_CERTIFICATE_OVERCONSERVATISM`.

## B6. CI and evidence

Extend `native-r1c` without deleting existing checks. It must run:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --lib pauli_edge_step::tests -- --nocapture
cargo test --locked --lib electron_spectral::tests::sweep_report_counts_exact_stationary_without_nan_swallow -- --exact --nocapture
cargo test --locked --lib electron_spectral::tests::unresolved_edge_failure_is_transactional_and_observable -- --exact --nocapture
cargo test --locked --lib electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales -- --exact --nocapture
```

The audit-model Python program is not a final-SHA job.

Update the PR body with exact live base/head, file count, tests, mutations,
fresh-review verdict, remaining deferred blockers, claim ceiling, and cost line.

Commit:

```bash
git add native/rabbit_cpu/src/pauli_edge_step.rs \
        native/rabbit_cpu/src/electron_spectral.rs \
        .github/workflows/harness.yml
git commit -m "ODE-R1C: fail fast on uncertifiable Pauli steps"
```

## 7. Required verification at final SHA

Do not run the frozen Python mirror as proof of the implementation.

```bash
set -euo pipefail
FINAL_SHA=$(git rev-parse HEAD)
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --locked --lib pauli_edge_step::tests -- --nocapture
cargo test --locked --lib \
  electron_spectral::tests::sweep_report_counts_exact_stationary_without_nan_swallow \
  -- --exact --nocapture
cargo test --locked --lib \
  electron_spectral::tests::unresolved_edge_failure_is_transactional_and_observable \
  -- --exact --nocapture
cargo test --locked --lib \
  electron_spectral::tests::root_certificates_hold_across_legacy_small_step_scales \
  -- --exact --nocapture
cargo test --locked --lib electron_event_falsifiers::frozen -- --nocapture
cd ../..
git diff --check origin/fix/ode-r1c-certified-edge-outcomes-20260824...HEAD
git status --short
printf '%s\n' "$FINAL_SHA"
```

Run all contract mutations and preserve RED/restored-GREEN logs. Every command
record must include `executed_at_commit=$FINAL_SHA`, exit code, environment, and
stdout SHA-256.

Generate an evidence bundle matching
`rabbit-pr1-r1c-evidence-bundle/v2` in the compiled package.

## 8. Deferred P1 gates

PR #1 does not pass by removing these blockers. It passes by preserving their
precise typed status:

```text
P1-001 BLOCKED_EQUILIBRIUM_STEP_SEMANTICS
P1-008 BLOCKED_FROZEN_ERROR_BUDGET
P1-009 no forbidden scientific/global claims
```

Do not add `CertifiedFrozen` here.

## 9. Fresh-context reviewer

Use a new context containing only:

```text
base SHA
final SHA
base..final diff
frozen contract release and manifest
Rust verification/mutation logs
evidence bundle
deferred typed blockers
```

Do not provide implementation reasoning. Reviewer edits are forbidden. First
pass is classification-only.

Pass condition:

```text
must_close_p0_open = 0
must_close_p1_open = 0
deferred_without_required_typed_blocker = 0
deferred_claim_ceiling_violations = 0
new_uncatalogued_p0_p1 = 0
unmapped_threats = 0
missing_evidence = 0
unresolved_spec_boundaries = 0
contract_changes = 0
new_relevant_suite_failures = 0
```

Allow at most **two** fresh review rounds. Each round may be followed by one fix
cycle and complete evidence regeneration. If must-close P0/P1 findings remain
after round two, stop `BLOCKED_REVIEW_NONCONVERGENCE`.

## 10. Push and PR update

Only after the reviewer gate passes:

```bash
git push origin HEAD:fix/ode-r1c-certified-edge-outcomes-20260824
```

Update PR #1 body to the exact live SHA and evidence. Do not merge.

## 11. Stop states

```text
BASE_DRIFT
AUDIT_MODEL_DIVERGENCE
REPRODUCER_MODEL_DIVERGENCE
BASELINE_DIVERGENCE
BASELINE_RED_NOT_OBSERVED
BLOCKED_BY_UNRESOLVED_SPEC
BLOCKED_TRUE_ROOT_INTERVAL
BLOCKED_UNCERTIFIED_SUCCESS_PATH
StateMapUnresolved
CertificateUnattainableAtStep
StagnatedInterval
BLOCKED_REPORT_INTEGRITY
BLOCKED_INCOMPLETE_CI
BLOCKED_CERTIFICATE_OVERCONSERVATISM
STALE_EVIDENCE
CONTRACT_AMENDMENT_REQUIRED
BLOCKED_REVIEW_NONCONVERGENCE
```

A typed blocker is a valid result. A green produced by changing cap, tolerance,
fixtures, golden values, or the contract is failure.

## 12. Final response schema

```text
status:
contract_release_commit:
base_sha:
final_sha:
commits:
changed_files:
baseline_rust_known_bad_observed:
red_green_tests:
golden_root_checks:
mutation_checks:
required_commands:
full_relevant_suite_delta:
evidence_bundle_path:
fresh_review_rounds:
must_close_P0_open:
must_close_P1_open:
deferred_typed_blockers:
claim_ceiling:
unresolved_blockers:
pr_url:
pr_body_updated:
worktree_clean:
```
