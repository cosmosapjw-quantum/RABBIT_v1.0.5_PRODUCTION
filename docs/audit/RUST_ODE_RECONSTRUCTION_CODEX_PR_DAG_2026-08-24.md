# Codex-Executable PR DAG — Rust ODE Collision Reconstruction

Date: 2026-08-24 (Asia/Seoul)  
Source evidence branch: `external-audit/ode-rust-reconstruction-complete-20260824`  
Source evidence head: `5689f2889163c3cf939a2d83c66075910d1948ff`  
Clean code base: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Governing audit: `docs/audit/RUST_ODE_RECONSTRUCTION_THIRD_PARTY_REAUDIT_2026-08-24.md`

This is the normative human-readable execution plan. The machine-readable twin is `.codex/plans/rust_ode_reconstruction_pr_dag_20260824.json`.

## 0. Non-interactive execution law

Codex must not ask the user a question. Resolve every ordinary choice exactly as specified below. When a required identity or acceptance condition fails, stop with the listed machine status instead of guessing.

```text
BASE_DRIFT                         source/base blob mismatch
BLOCKED_ROOT_CERTIFICATE           Pauli edge solve cannot certify residual
BLOCKED_RAW_DETAILED_BALANCE       unforced equilibrium residual exceeds gate
BLOCKED_EDGE_RECONSTRUCTION        edge action does not match direct action
BLOCKED_TEMPORAL_CONSISTENCY       tangent/order gate fails
BLOCKED_ENERGY_DERIVATIVE          finite-step energy derivative disagrees
BLOCKED_ADAPTIVE_CONTROLLER        deterministic accept/reject gate fails
BLOCKED_SAME_PHYSICS_REFERENCE     independent reference disagrees
BLOCKED_SELF_COLLISION             self-collision bounded step unavailable
BLOCKED_SPLIT_PREFIX               composed short prefix fails
BLOCKED_FULL_REGRESSION            fmt/check/clippy/test failure
BLOCKED_ENDPOINT                   endpoint or convergence gate fails
BLOCKED_PERFORMANCE                endpoint performance target not met
```

Never respond to a failed gate by widening a tolerance, clipping an occupation, forcing an equilibrium result, skipping a test, changing physics, adding a wrapper, or starting a later PR.

## 1. Immutable execution policy

### 1.1 Source firewall

The evidence branch contains large archives and must never be merged. Executable branches begin from the clean code base and import only the files/symbols named by the active PR.

Expected source blobs on the evidence head:

```text
native/rabbit_cpu/src/electron_event.rs              039e0a8d897367d656e602fadf9af0afb265214a
native/rabbit_cpu/src/electron_event_falsifiers.rs   6d66d62de54fd1591cd0e7da9db6f10b6305e580
native/rabbit_cpu/src/electron_supplied.rs           6cc913d73b7a9bf0ef6f87b7de30728f9deeb65f
native/rabbit_cpu/src/electron_spectral.rs           4457d7332f3b2763f19790334094ed0b883d9ca2
native/rabbit_cpu/src/isotropic_boltzmann.rs         918207debeeb27d005c593f8d10893c83bbb4434
native/rabbit_cpu/src/lib.rs                         25dab9b461fd2cce05767300d17456bae2b014e4
native/rabbit_cpu/src/pauli_edge_step.rs             5b6be3565cc63f4f8a4baf3a25ce4a1a574fc15b
```

### 1.2 One-PR rule

- Execute one PR at a time.
- A PR starts only after every dependency is merged into the clean integration branch.
- Do not run later validation to compensate for an earlier failed focused gate.
- No new manifest, claim ledger, readiness file, figure, benchmark framework, or telemetry subsystem.
- Tests remain in existing Rust modules unless this plan explicitly authorizes one new module.
- Current-head line numbers are advisory. Symbol names and code snippets are authoritative after prior PRs shift lines.

### 1.3 Common branch pattern

Use these exact names:

```bash
INTEGRATION_BRANCH=codex/ode-rust-reconstruction-certified
PR_BRANCH_PREFIX=codex/ode-rust-r
```

The clean integration branch is created once:

```bash
git fetch origin --prune
git cat-file -e 78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b^{commit}
git cat-file -e origin/external-audit/ode-rust-reconstruction-complete-20260824^{commit}
git worktree add ../rabbit-ode-rust-certified \
  -b codex/ode-rust-reconstruction-certified \
  78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b
cd ../rabbit-ode-rust-certified
git status --porcelain=v1
```

Expected final command output: empty. Otherwise stop `BASE_DRIFT`.

### 1.4 Common Rust commands

Run from `native/rabbit_cpu` unless stated otherwise:

```bash
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib <focused-test-name> -- --exact --nocapture
```

At the end of every PR run:

```bash
git diff --check
git status --short
git diff --stat HEAD~1..HEAD
```

Record added/deleted/net lines and exact commands in the PR body. `token_use_exact` is `UNAVAILABLE` unless the runtime exposes an exact counter.

## 2. Dependency DAG

```text
PRE-00 clean source firewall
  -> R1 certified Pauli edge root
  -> R2 unforced detailed balance and serial edge assembly
  -> R3 multi-state reconstruction + temporal consistency
  -> R4 isolated energy-closed electron substep
  -> R5 adaptive electron collision controller
  -> R6 same-physics selected-grid short-prefix oracle
  -> R7 bounded neutrino self-collision step
  -> R8 composed expansion/electron/self short prefix
  -> R9 full regression + clean code-only promotion
  -> R10 endpoint/convergence authority
  -> R11 endpoint-consumed performance
```

Do not parallelize R1-R10. R11 is forbidden until R10 passes.

---

# PRE-00 — Clean source firewall

Type: preflight, no PR, no commit.

## Goal

Verify exact evidence identity and guarantee that no archive or audit-only path enters executable history.

## Commands

```bash
SOURCE_REF=origin/external-audit/ode-rust-reconstruction-complete-20260824
BASE=78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

test "$(git rev-parse "$SOURCE_REF")" = 5689f2889163c3cf939a2d83c66075910d1948ff
for pair in \
  'native/rabbit_cpu/src/electron_event.rs 039e0a8d897367d656e602fadf9af0afb265214a' \
  'native/rabbit_cpu/src/electron_event_falsifiers.rs 6d66d62de54fd1591cd0e7da9db6f10b6305e580' \
  'native/rabbit_cpu/src/electron_supplied.rs 6cc913d73b7a9bf0ef6f87b7de30728f9deeb65f' \
  'native/rabbit_cpu/src/electron_spectral.rs 4457d7332f3b2763f19790334094ed0b883d9ca2' \
  'native/rabbit_cpu/src/isotropic_boltzmann.rs 918207debeeb27d005c593f8d10893c83bbb4434' \
  'native/rabbit_cpu/src/lib.rs 25dab9b461fd2cce05767300d17456bae2b014e4' \
  'native/rabbit_cpu/src/pauli_edge_step.rs 5b6be3565cc63f4f8a4baf3a25ce4a1a574fc15b'
do
  set -- $pair
  test "$(git rev-parse "$SOURCE_REF:$1")" = "$2" || exit 91
done

test -z "$(git status --porcelain=v1)"
```

Exit 91 or any dirty status means `BASE_DRIFT`. Do not continue.

Forbidden checkout paths:

```text
external_audit/**
docs/audit/RUST_ODE_COLLISION_RECONSTRUCTION_2026-08-24.md
.agent-harness/runs/**
native/rabbit_cpu/target/**
**/__pycache__/**
```

---

# PR R1 — Certified Pauli edge root and stable gain/loss primitive

PR title: `R1: certify Pauli edge implicit roots`  
Branch: `codex/ode-rust-r1-edge-root-certificate`  
Base: clean integration branch  
Commit: `fix(ode): certify Pauli edge implicit roots`

## Scope and write set

```text
native/rabbit_cpu/src/electron_event.rs
native/rabbit_cpu/src/electron_event_falsifiers.rs
native/rabbit_cpu/src/electron_supplied.rs
native/rabbit_cpu/src/pauli_edge_step.rs
native/rabbit_cpu/src/lib.rs
```

No other file may change.

## Import step

```bash
git switch -c codex/ode-rust-r1-edge-root-certificate
SOURCE_REF=origin/external-audit/ode-rust-reconstruction-complete-20260824
for path in \
  native/rabbit_cpu/src/electron_event.rs \
  native/rabbit_cpu/src/electron_event_falsifiers.rs \
  native/rabbit_cpu/src/electron_supplied.rs \
  native/rabbit_cpu/src/pauli_edge_step.rs \
  native/rabbit_cpu/src/lib.rs
do
  git show "$SOURCE_REF:$path" > "$path"
done
```

## Required code changes

### R1.1 Replace false-success iteration handling

In `PauliEdge::implicit_step` (`pauli_edge_step.rs`, source-head lines 187-274):

1. Extract the nonlinear solve into a private helper with an explicit `max_iterations: usize` argument. Production calls it with `96`. Unit tests may call it with smaller limits.
2. Delete the branch that returns a midpoint merely because the iteration count equals the cap.
3. After every proposed extent, compute a root certificate.
4. If neither residual nor occupation-width certificate passes before the cap, return exactly:

```rust
Err("Pauli edge implicit root did not converge")
```

### R1.2 Use two independent certificates

Add private helpers with these semantics:

```rust
fn residual_is_certified(extent: f64, step_flux: f64, residual: f64) -> bool {
    let scale = extent.abs().max(step_flux.abs()).max(f64::MIN_POSITIVE);
    residual.abs() <= 128.0 * f64::EPSILON * scale + f64::MIN_POSITIVE
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
```

A solve succeeds when either:

- residual certificate passes and candidate occupations are valid; or
- occupation-bracket certificate passes and the midpoint residual certificate also passes.

Do not use `.max(1.0)` on the dimensional extent or residual scale.

### R1.3 Certify the physical bracket

Replace the old bracket check with:

```text
lower residual must be <= its local residual tolerance
upper residual must be >= minus its local residual tolerance
```

If not, return exactly:

```rust
Err("Pauli edge implicit root is not bracketed by physical capacities")
```

### R1.4 Extend the step report without adding a telemetry subsystem

Extend the existing `PauliEdgeStep` only:

```rust
pub(crate) struct PauliEdgeStep {
    pub(crate) extent: f64,
    pub(crate) nonlinear_iterations: usize,
    pub(crate) residual_abs: f64,
    pub(crate) residual_scale: f64,
    pub(crate) max_occupation_bracket_width: f64,
}
```

`Default` remains all zeros. Every nontrivial successful step stores the final certificate values.

### R1.5 Evaluate gain/loss in a log-scaled difference

Replace `stable_nonnegative_product_difference` with a log-scaled implementation:

```text
log_gain = log(coefficient) + log(factor_1) + log(factor_2)
log_loss = log(coefficient) + log(factor_1) + log(factor_2)
m = max(log_gain, log_loss)
net = exp(m) * (exp(log_gain-m) - exp(log_loss-m))
```

Rules:

- a zero coefficient or factor represents log `-infinity` without calling `ln(0)`;
- both zero returns `0.0`;
- one zero returns the representable signed nonzero side;
- non-finite positive inputs return the existing physical-domain error;
- do not compute and subtract the two full products before deciding the near-balance path.

## Required exact tests

Add or replace tests in `pauli_edge_step.rs`:

```text
root_cap_is_an_error_not_a_midpoint_success
tiny_extent_uses_occupation_scaled_certificate
successful_edge_step_carries_a_residual_certificate
stiff_elastic_step_preserves_box_number_and_backward_euler_residual
stiff_pair_step_preserves_box_cp_difference_and_backward_euler_residual
log_scaled_difference_survives_extreme_tail_and_near_balance
```

Assertions for every nontrivial successful step:

```rust
assert!(report.residual_abs <= 128.0 * f64::EPSILON * report.residual_scale + f64::MIN_POSITIVE);
assert!(report.max_occupation_bracket_width <= 128.0 * f64::EPSILON);
assert!(report.nonlinear_iterations <= 96);
```

The cap test calls the private solver with `max_iterations=1` on a case whose first Newton proposal is not certified and asserts the exact error string. Do not alter production constants to construct the test.

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib pauli_edge_step::tests -- --nocapture
cargo test --lib electron_event_falsifiers::frozen -- --nocapture
```

## Acceptance

- no success after iteration exhaustion;
- no unit extent floor;
- all existing invariant/transaction tests remain green;
- exact test command has zero ignored, failed, or filtered-in unexpected tests;
- diff touches only the five allowed files.

Otherwise stop `BLOCKED_ROOT_CERTIFICATE`.

Claim ceiling: `VALIDATED LOCAL EDGE ROOT PRIMITIVE`; no collision action, prefix, or endpoint claim.

---

# PR R2 — Serial folded edge assembly and unforced detailed balance

PR title: `R2: validate unforced electron edge detailed balance`  
Branch: `codex/ode-rust-r2-electron-edge-balance`  
Base: merged R1  
Commit: `feat(collisions): validate unforced electron Pauli edges`

## Scope and write set

```text
native/rabbit_cpu/src/electron_spectral.rs
```

`electron_event.rs`, `electron_supplied.rs`, and `pauli_edge_step.rs` are read-only in this PR.

## Import and serial baseline

Import the evidence version of `electron_spectral.rs`, then restore the serial `build_event_stream` implementation from clean base `78f5...`. Do not import `build_event_task`, `build_event_stream_with_workers`, `available_parallelism`, scoped threads, or the ignored performance benchmark.

```bash
git switch -c codex/ode-rust-r2-electron-edge-balance
SOURCE_REF=origin/external-audit/ode-rust-reconstruction-complete-20260824
git show "$SOURCE_REF:native/rabbit_cpu/src/electron_spectral.rs" \
  > native/rabbit_cpu/src/electron_spectral.rs
# Replace source-head lines 103-277 with the serial build_event_stream from BASE.
git show 78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b:native/rabbit_cpu/src/electron_spectral.rs \
  > /tmp/electron_spectral.base.rs
```

Copy the complete base `build_event_stream` symbol from `/tmp/electron_spectral.base.rs`; do not hand-rewrite its physics loops.

## Required code changes

### R2.1 Remove all exact-anchor result forcing

Delete:

```text
IsotropicElectronPauliEdges.exact_equilibrium
IsotropicElectronPauliEdges.anchor_electron_pair
IsotropicElectronPauliEdges.anchor_heavy_pair
IsotropicElectronPauliEdges::is_exact_anchor
exact_reference_state
all electron_pair_mev.fill(0.0) / heavy_pair_mev.fill(0.0)
all exact-anchor early returns in transactional_step
```

Keep only the `step_mev_inverse == 0.0` no-op.

### R2.2 Always evaluate real edge fluxes

`action_values` must iterate over every reconstructed edge for every valid occupation state. `transactional_step` must apply the certified R1 edge solver even at FD equilibrium.

### R2.3 Keep edge coefficients state-independent after construction

The dynamic coefficient extraction may validate the construction anchor, but the stored coefficient must not retain or read dynamic occupations. Add a test that constructs edges once and calls `action_values` on at least five distinct valid states without rebuilding.

### R2.4 Add an unforced detailed-balance certificate

Inside the existing test module, calculate at `T_gamma == T_cm` and FD occupations:

```text
net_L1 = sum over edges |edge flux|
traffic_L1 = sum over edges (positive gain term + positive loss term)
normalized_DB = net_L1 / max(traffic_L1, MIN_POSITIVE)
```

Use the actual reconstructed edges, not the production action after any branch. Freeze:

```text
normalized_DB <= 1e-12
max individual |flux| / max(gain+loss, MIN_POSITIVE) <= 1e-12
```

If the current event pairing does not satisfy these bounds, do not reintroduce a shortcut. Stop `BLOCKED_RAW_DETAILED_BALANCE` and record the ten largest offending edges by topology/bank/node in test output. No later PR starts.

### R2.5 Check Jacobian continuity at equilibrium

At the same FD anchor:

1. evaluate the unforced analytic folded Jacobian;
2. perturb one electron and one heavy logit by `±1e-6` through the occupation map;
3. compare the centered derivative of the unforced action;
4. require relative error `<=2e-7` or absolute error `<=2e-34`.

This test replaces the old assertion that a forced-zero action may coexist with a nonzero Jacobian.

## Required exact tests

```text
unforced_fd_equilibrium_is_an_event_and_edge_null
unforced_equilibrium_jacobian_is_continuous
folded_edges_reconstruct_action_at_five_independent_states
folded_edges_are_boundary_inward_at_five_independent_states
invalid_inputs_fail_without_clipping
```

Use grids/rules:

```text
unit grid: order 4, event 4/3
cross-check grid: order 6, event 6/4
states: FD; 0.91/1.07 alternating; 0.73/1.11 alternating; high-q 1e-35/1e-40; deterministic pseudo-random values in [1e-8,1-1e-8]
```

Do not use randomness at runtime. Store the pseudo-random vector literals in the test.

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib electron_spectral::tests::unforced_fd_equilibrium_is_an_event_and_edge_null -- --exact --nocapture
cargo test --lib electron_spectral::tests::unforced_equilibrium_jacobian_is_continuous -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_reconstruct_action_at_five_independent_states -- --exact --nocapture
cargo test --lib electron_spectral::tests::folded_edges_are_boundary_inward_at_five_independent_states -- --exact --nocapture
```

Acceptance: all tests pass without exact-anchor branches or zero filling. Otherwise `BLOCKED_RAW_DETAILED_BALANCE` or `BLOCKED_EDGE_RECONSTRUCTION`.

Claim ceiling: `VALIDATED SERIAL ELECTRON EDGE ASSEMBLY AT FOCUSED STATES`.

---

# PR R3 — Temporal consistency, edge-order sensitivity, and derivative identity

PR title: `R3: certify electron Pauli sweep consistency`  
Branch: `codex/ode-rust-r3-edge-sweep-consistency`  
Base: merged R2  
Commit: `test(collisions): certify Pauli sweep consistency`

## Scope

```text
native/rabbit_cpu/src/electron_spectral.rs
native/rabbit_cpu/src/pauli_edge_step.rs   # report aggregation only; no solver formula change
```

## Required code

Extend `PauliSweepReport` with only:

```rust
pub(crate) maximum_root_residual_ratio: f64,
pub(crate) maximum_occupation_bracket_width: f64,
```

Aggregate R1 certificates from each edge. Do not add per-edge logs or a new report type.

Add a test-only helper that runs the same edge list in:

```text
forward-only order
reverse-only order
forward-half + reverse-half production order
```

Production remains forward-half + reverse-half.

## Required tests

### R3.1 Tangent consistency

For fixed reconstructed edges and state `f`, run `h in [2^-8,2^-10,2^-12,2^-14] MeV^-1` and compute

```text
D_h = (sweep_h(f)-f)/h
E_h = weighted L1(D_h - action_values(f)) / weighted L1(action_values(f))
```

Require:

```text
E_(h/2) < 0.60 * E_h for the final two pairs
final E_h <= 2e-5
```

If not, stop `BLOCKED_TEMPORAL_CONSISTENCY`.

### R3.2 Step-doubling observed order

For `h in [2^-6,2^-7,2^-8]`, compare one `h` step with two `h/2` steps in logit and moment norms. Compute observed order

```text
p = log2(error(h)/error(h/2))
```

Freeze the implementation as first order unless every tested state yields `p>=1.8`. Do not label the forward/reverse sweep second order merely because its ordering is symmetric.

Acceptance minimum:

```text
0.8 <= median observed p <= 1.3
error decreases monotonically
all root certificates pass
all occupations remain strict on strict inputs
```

If measured order is consistently `>=1.8`, record `method_order=2`; otherwise record `method_order=1`. This value controls R5's step-size exponent.

### R3.3 Edge-order sensitivity

Require forward-only vs reverse-only discrepancy to converge to zero with `h`; no fixed hard equality is required. Production forward/reverse result must lie within the componentwise envelope of the two first-order orderings for the final two `h` values. Failure means `BLOCKED_TEMPORAL_CONSISTENCY`.

### R3.4 Exact invariants after complete sweep

For elastic-only edge sets: weighted number residual `<=256 eps * invariant scale`.  
For pair-only edge sets: weighted CP-difference residual `<=256 eps * invariant scale`.

## Exact test names

```text
pauli_sweep_tangent_converges_to_unforced_action
pauli_sweep_step_doubling_has_a_measured_order
pauli_sweep_edge_order_error_converges
pauli_sweep_preserves_complete_edge_set_invariants
```

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib electron_spectral::tests::pauli_sweep_tangent_converges_to_unforced_action -- --exact --nocapture
cargo test --lib electron_spectral::tests::pauli_sweep_step_doubling_has_a_measured_order -- --exact --nocapture
cargo test --lib electron_spectral::tests::pauli_sweep_edge_order_error_converges -- --exact --nocapture
cargo test --lib electron_spectral::tests::pauli_sweep_preserves_complete_edge_set_invariants -- --exact --nocapture
```

Claim ceiling: `VALIDATED LOCAL FIRST-ORDER OR MEASURED-ORDER ELECTRON EDGE SWEEP`.

---

# PR R4 — Isolated energy-closed electron collision substep

PR title: `R4: isolate the energy-closed electron collision substep`  
Branch: `codex/ode-rust-r4-electron-substep`  
Base: merged R3  
Commit: `feat(ode): isolate electron collision reconstruction`

## Scope

```text
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

Import only these concepts from the evidence version, not the full file blindly:

```text
ElectronCollisionReconstruction
electromagnetic_temperature_for_density
reconstruct_electron_collision_substep
```

## Required refactor

### R4.1 Add one collision-free thermodynamic snapshot

Inside `IsotropicBoltzmannFlrwSystem`, add a private struct/helper in the same file:

```rust
struct SpectralThermodynamicSnapshot {
    t_gamma_mev: f64,
    t_cm_mev: f64,
    electron_pair_occupation: Vec<f64>,
    heavy_pair_occupation: Vec<f64>,
    electron_pair_moments: IsotropicPairMoments,
    heavy_pair_moments: IsotropicPairMoments,
    rho_neutrino_total_mev4: f64,
    electromagnetic_rho_mev4: f64,
    h_mev: f64,
}
```

`thermodynamic_snapshot(ln_a,state)` performs no electron event build and no neutrino-self action build. `physical_state_impl` reuses it. `reconstruct_electron_collision_substep` uses it and builds the electron edge stream exactly once.

### R4.2 Keep operator semantics explicit

The substep is collision-only at frozen `ln_a`:

```text
elapsed seconds unchanged
T_cm unchanged
H frozen only for converting delta_N to delta_t
occupations updated by certified edge sweep
T_gamma reconstructed from opposite neutrino energy change
expansion and self-collision excluded
```

### R4.3 Reject energy residual, do not merely report it

After EOS inversion, require

```rust
let total_scale = electromagnetic_before.rho
    + rho_neutrino_before_mev4
    + electromagnetic_after.rho
    + rho_neutrino_after_mev4;
let allowed = 256.0 * f64::EPSILON * total_scale.max(f64::MIN_POSITIVE);
if total_energy_residual_mev4.abs() > allowed {
    return Err("electron collision reconstruction energy residual is too large");
}
```

### R4.4 Validate the infinitesimal energy derivative independently

At decreasing `delta_N`, require

```text
(rho_nu_after-rho_nu_before)/(delta_N/H)
    -> collision_energy_moment(unforced electron action)
```

The electromagnetic change is not the reference because it is constructed from the neutrino change.

## Required tests

```text
thermodynamic_snapshot_builds_no_collision_action
reconstructed_electron_substep_is_transactional_and_strict
reconstructed_electron_substep_energy_residual_is_rejected
reconstructed_electron_substep_converges_to_action_energy_derivative
zero_step_is_bitwise_identity
```

Test grid/rule: grid order 4 and 6; electron rule 4/3 and 6/4. Include the manually small tail and one ordinary non-equilibrium state.

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib isotropic_boltzmann::tests::thermodynamic_snapshot_builds_no_collision_action -- --exact --nocapture
cargo test --lib isotropic_boltzmann::tests::reconstructed_electron_substep_is_transactional_and_strict -- --exact --nocapture
cargo test --lib isotropic_boltzmann::tests::reconstructed_electron_substep_energy_residual_is_rejected -- --exact --nocapture
cargo test --lib isotropic_boltzmann::tests::reconstructed_electron_substep_converges_to_action_energy_derivative -- --exact --nocapture
```

Failure of derivative convergence is `BLOCKED_ENERGY_DERIVATIVE`.

Claim ceiling: `VALIDATED ISOLATED ELECTRON COLLISION SUBSTEP ON SMALL GRIDS`.

---

# PR R5 — Deterministic adaptive electron collision controller

PR title: `R5: add adaptive Pauli electron collision steps`  
Branch: `codex/ode-rust-r5-adaptive-electron-step`  
Base: merged R4  
Commit: `feat(ode): adapt electron collision substeps`

## Scope

```text
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

No generic solver framework or registry.

## Required types

Add in the same file:

```rust
pub(crate) struct ElectronCollisionStepConfig {
    pub(crate) logit_rtol: f64,       // default 1e-6
    pub(crate) logit_atol: f64,       // default 1e-8
    pub(crate) moment_rtol: f64,      // default 1e-8
    pub(crate) temperature_rtol: f64, // default 1e-10
    pub(crate) min_delta_ln_a: f64,   // default 1e-12
    pub(crate) max_rejections: usize, // default 12
}

pub(crate) struct AdaptiveElectronCollisionStep {
    pub(crate) candidate_state: Vec<f64>,
    pub(crate) accepted_delta_ln_a: f64,
    pub(crate) suggested_delta_ln_a: f64,
    pub(crate) error_norm: f64,
    pub(crate) rejections: usize,
    pub(crate) sweep: PauliSweepReport,
}
```

Validate every field; no silent coercion.

## Algorithm

For requested `h`:

1. compute one full R4 step `F_h`;
2. compute two half steps `F_(h/2) o F_(h/2)` rebuilding coefficients at the half-step state;
3. compare the full and two-half results;
4. accept the two-half result only if `error_norm <= 1`;
5. otherwise reduce `h` and retry transactionally.

Error norm is the maximum of:

```text
max logit component error / (logit_atol + logit_rtol*max(|u_full|,|u_half|))
electron energy moment relative error / moment_rtol
heavy energy moment relative error / moment_rtol
T_gamma relative error / temperature_rtol
```

Do not use occupation absolute tolerances to hide high-q errors. Both candidates are already Pauli-bounded algebraically.

Step factor uses R3's measured method order `p`:

```text
factor = clamp(0.2, 2.0, 0.9 * error_norm^(-1/(p+1)))
error_norm == 0 -> factor 2.0
```

At `max_rejections` or `h < min_delta_ln_a`, return exact errors:

```text
electron collision adaptive rejection budget exhausted
electron collision adaptive step fell below minimum
```

## Required tests

```text
adaptive_electron_step_rejects_then_accepts_without_mutating_input
adaptive_electron_step_is_deterministic_bitwise
adaptive_electron_step_error_decreases_with_requested_step
adaptive_electron_step_preserves_box_energy_and_elapsed_time
adaptive_electron_step_fails_at_minimum_step
```

Determinism test runs the same input five times and compares all candidate-state bits and scalar report bits.

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib isotropic_boltzmann::tests::adaptive_electron_step -- --nocapture
```

Any non-determinism, mutation after rejection, or tolerance widening is `BLOCKED_ADAPTIVE_CONTROLLER`.

Claim ceiling: `VALIDATED ADAPTIVE ISOLATED ELECTRON COLLISION STEP ON SMALL GRIDS`.

---

# PR R6 — Same-physics selected-grid short-prefix oracle

PR title: `R6: validate the selected-grid electron collision prefix`  
Branch: `codex/ode-rust-r6-selected-prefix-oracle`  
Base: merged R5  
Commit: `test(ode): validate selected-grid electron collision prefix`

## Scope

```text
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

Test-only helper code may live in the existing `#[cfg(test)]` module. No public API.

## Required same-physics reference

Implement a test-only `FrozenElectronCollisionOde` that evolves the **same unforced Rust electron action** in physical collision time `tau [MeV^-1]`:

```text
df_i/dtau = C_i(f,T_gamma)
drho_em/dtau = -d rho_nu/dtau
elapsed time fixed
T_cm and ln_a fixed
neutrino self-collision absent
```

Use logit state internally, with `du/dtau=C/[f(1-f)]`, only as a tight reference. Fail on occupation underflow; do not floor or clip. Integrate with both `ode::solve(Bdf,...)` and `ode::solve(Rodas5P,...)` at tolerances at least 100 times tighter than R5's accepted target. Both reference solvers must succeed and agree before comparison to R5.

## Production-shaped fixtures

1. selected exponential grid, 48 nodes;
2. reference temperature `1.0 MeV`;
3. `T_gamma=1.15 MeV`, `ln_a=0`;
4. electron rule `6/4`;
5. heavy/electron logits from FD plus deterministic perturbations on the final six nodes, including occupations below `1e-30` but representable;
6. requested `delta_N` ladder `[1e-6, 5e-7, 2.5e-7]`.

Also run one reduced grid-8 fixture for faster diagnostics.

## Comparison gates

For the accepted R5 result versus the tight reference:

```text
all occupations strict and finite
max logit error <= 5e-5
electron energy moment relative error <= 2e-7
heavy energy moment relative error <= 2e-7
T_gamma relative error <= 2e-9
total-energy residual <= 512 eps * total energy scale
BDF/Rodas reference state agreement <= half of each candidate threshold
```

Errors must decrease monotonically over the `delta_N` ladder.

## Required tests

```text
selected48_reference_solvers_agree_for_electron_collision_only
selected48_adaptive_pauli_step_matches_same_physics_reference
selected48_high_q_tail_remains_strict_without_clipping
selected48_prefix_error_decreases_under_step_refinement
```

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --lib isotropic_boltzmann::tests::selected48_reference_solvers_agree_for_electron_collision_only -- --exact --nocapture
cargo test --release --lib isotropic_boltzmann::tests::selected48_adaptive_pauli_step_matches_same_physics_reference -- --exact --nocapture
cargo test --release --lib isotropic_boltzmann::tests::selected48_high_q_tail_remains_strict_without_clipping -- --exact --nocapture
cargo test --release --lib isotropic_boltzmann::tests::selected48_prefix_error_decreases_under_step_refinement -- --exact --nocapture
```

Failure is `BLOCKED_SAME_PHYSICS_REFERENCE`. Do not start composition.

Claim ceiling: `VALIDATED ELECTRON-ONLY SELECTED-GRID SHORT PREFIX`; still no full FLRW prefix or endpoint.

---

# PR R7 — Bounded neutrino self-collision substep

PR title: `R7: bound the neutrino self-collision substep`  
Branch: `codex/ode-rust-r7-self-collision-step`  
Base: merged R6  
Commit: `feat(ode): bound neutrino self-collision steps`

## Scope

```text
native/rabbit_cpu/src/neutrino_self_spectral.rs
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

Do not approximate the four-leg self-collision action by the two-node electron edge type.

## Required implementation

Use the existing stable event affinity in `neutrino_self_spectral.rs` and add a collision-only logit system at frozen `ln_a`, `T_cm`, `T_gamma`, and H. Advance it through existing Rust `Rodas5P` with step doubling:

```text
one full self step
versus two half self steps
accepted result = two-half result
strict logit/occupation validation after every solve
no electromagnetic temperature change
no elapsed-time change
```

The self action already deposits complete events symmetrically. Preserve the event stream and its number/energy weak-form identities. Do not add projection after the solve.

Default controller:

```text
logit_rtol=1e-7
logit_atol=1e-9
moment_rtol=1e-10
min_delta_ln_a=1e-12
max_rejections=12
```

## Required tests

```text
self_collision_step_is_strict_and_transactional
self_collision_step_preserves_degeneracy_weighted_number
self_collision_step_preserves_total_neutrino_energy
self_collision_step_matches_tight_bdf_reference
self_collision_step_does_not_change_tgamma_or_elapsed_time
self_collision_step_fails_before_logit_underflow
```

Run on grid 6 and selected 48. Selected-48 may be release-only but not ignored.

If strict occupations or invariants fail, stop `BLOCKED_SELF_COLLISION`; do not omit self-collisions from R8.

Claim ceiling: `VALIDATED ISOLATED SELF-COLLISION STEP`.

---

# PR R8 — Adaptive composed FLRW short prefix

PR title: `R8: compose the bounded FLRW collision operators`  
Branch: `codex/ode-rust-r8-composed-prefix`  
Base: merged R7  
Commit: `feat(ode): compose bounded FLRW collision steps`

## Scope

```text
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

No Python API, endpoint driver, or performance optimization.

## Operator definition

Define three explicit operators over one `delta_N`:

```text
A: expansion/EM adiabatic evolution + elapsed time; occupations fixed
B: R5 adaptive electron collision + EM energy exchange; elapsed time fixed
C: R7 adaptive self collision; T_gamma and elapsed time fixed
```

Initial promoted composition is first-order Lie:

```text
Phi_h = A_h -> B_h -> C_h
```

Do not call it Strang or second order. Control its local error with one full composed step versus two composed half steps; accept the two-half result. Alternate ordering is test-only for splitting-error diagnosis.

## Expansion operator

Reuse existing EOS/Friedmann formulas. Integrate only the two scalar variables `T_gamma` and elapsed seconds over `delta_N` with occupations fixed. Use existing Rust Rodas5P with tight internal tolerances; no duplicate physics formula outside `isotropic_boltzmann.rs`.

## Composed error norm

Maximum of:

```text
logit norm from R5/R7
T_gamma relative error at 1e-9
elapsed-time relative error at 1e-9
electron/heavy energy moments at 1e-7
total first-law residual at 1e-9 relative per accepted step
```

## Frozen short-prefix case

```text
grid=selected48
T_gamma_initial=T_cm_initial=10 MeV
ln_a_initial=0
prefix_delta_N=1e-3 first, then 1e-2 after the 1e-3 gate passes
electron rule=6/4
self rule angular=4
initial occupations=zero-chemical-potential FD
```

## Required raw history

Store in an existing test-local vector, not a new telemetry module:

```text
ln_a
T_gamma
elapsed seconds
min electron occupation
max electron occupation
min heavy occupation
max heavy occupation
energy moments
per-step first-law residual
accepted/rejected composed attempts
```

No clipping or output sanitization.

## Required tests

```text
composed_prefix_1e3_is_strict_energy_closed_and_deterministic
composed_prefix_1e2_is_strict_energy_closed_and_deterministic
composed_prefix_converges_under_global_step_halving
composed_prefix_split_order_bias_decreases
composed_prefix_failure_is_transactional
```

## Validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --lib isotropic_boltzmann::tests::composed_prefix_1e3_is_strict_energy_closed_and_deterministic -- --exact --nocapture
cargo test --release --lib isotropic_boltzmann::tests::composed_prefix_1e2_is_strict_energy_closed_and_deterministic -- --exact --nocapture
cargo test --release --lib isotropic_boltzmann::tests::composed_prefix_converges_under_global_step_halving -- --exact --nocapture
```

Failure is `BLOCKED_SPLIT_PREFIX`.

Claim ceiling: `VALIDATED SELECTED-GRID COMPOSED SHORT PREFIX`; not endpoint authority.

---

# PR R9 — Full regression and clean code-only promotion

PR title: `R9: validate the certified Rust collision reconstruction`  
Branch: `codex/ode-rust-r9-full-regression`  
Base: merged R8  
Commit: `test(native): validate certified collision reconstruction`

## Scope

Production/source files changed by R1-R8 plus, only if needed:

```text
.github/workflows/native-rust.yml
```

No audit archive or run packet.

## Required local validation

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
cargo test --release --lib
```

From repository root, after building/installing the extension through the existing documented path:

```bash
pytest -q tests/test_native_runtime.py tests/test_native_isotropic_boltzmann.py --tb=short
pytest -m 'production and not slow' -q --tb=line
```

If either named Python test file does not exist, search `tests/` for current native and isotropic test modules and record the exact resolved files. Do not invent a new wrapper merely to satisfy the names.

## Required CI

Add `.github/workflows/native-rust.yml` only if no existing workflow runs the exact Rust commands. Use shell `rustup toolchain install 1.94.1 --profile minimal` rather than an unpinned third-party toolchain action. Pin `actions/checkout` to the repository's existing full SHA. Jobs are read-only.

CI commands:

```text
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --lib
```

## Clean-history audit

```bash
git diff --name-only 78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b...HEAD \
  | grep -E '^(external_audit/|\.agent-harness/runs/|native/rabbit_cpu/target/)' \
  && exit 92 || true
```

Any match is `BLOCKED_FULL_REGRESSION`.

Claim ceiling: `VALIDATED CODE-ONLY RUST SHORT-PREFIX SUBSTRATE`.

---

# PR R10 — Endpoint and convergence authority

PR title: `R10: run the certified Rust decoupling endpoint gate`  
Branch: `codex/ode-rust-r10-endpoint-gate`  
Base: merged R9  
Commit: `test(physics): gate the Rust decoupling endpoint`

## Preconditions

R1-R9 all green. No unresolved xfail or ignored selected-grid correctness test. No endpoint code begins before this.

## Frozen endpoint ladder

```text
initial T_gamma=T_cm=10 MeV
terminal T_gamma<=0.01 MeV
grid orders=32,48,64 using the same exponential-map family
selected production candidate=48
electron rules=(4/3),(6/4),(8/6)
self angular orders=3,4,6
raw histories retained
no QKE, no flavour coherence, no anisotropy
```

Run in increasing cost:

```text
32 / 4-3 / self-3
48 / 6-4 / self-4
64 / 8-6 / self-6
```

Do not tune after seeing results.

## Gates

```text
terminal event reached exactly once
all accepted occupations strict and finite
T_gamma positive and monotone decreasing
elapsed time finite and monotone increasing
per-step first-law residual <=1e-8 relative
cumulative first-law residual <=3e-7 relative
32->48 and 48->64 energy-moment convergence monotone
48->64 N_eff difference <=2e-3
48->64 weak-rate QoI difference <=2e-3 relative at freezeout checkpoints
no adaptive minimum-step or rejection-budget failure
same initial case repeated twice is bitwise deterministic on one host
```

The `N_eff` gate is internal convergence, not an external Standard Model anchor.

## Validation order

```bash
# focused endpoint test first
cargo test --release --lib isotropic_boltzmann::tests::certified_endpoint_ladder -- --exact --nocapture
# only after it passes
cargo test --release --lib
pytest -m gold -q --tb=short
```

No public-production claim. Failure is `BLOCKED_ENDPOINT` and terminates the DAG.

Claim ceiling: `VALIDATED INTERNAL NO-QKE ISOTROPIC RUST ENDPOINT FOR THE FROZEN LADDER`.

---

# PR R11 — Endpoint-consumed performance and deterministic worker policy

PR title: `R11: reduce certified Rust endpoint collision cost`  
Branch: `codex/ode-rust-r11-endpoint-performance`  
Base: merged R10  
Commit: `perf(native): reduce certified endpoint collision cost`

## Scope

```text
native/rabbit_cpu/src/electron_spectral.rs
native/rabbit_cpu/src/isotropic_boltzmann.rs
```

Only now may the parallel event construction from the evidence branch be reconsidered.

## Required worker policy

- explicit `worker_count: usize` in an existing configuration struct;
- default `1` for deterministic correctness tests;
- production candidate `min(configured, available_parallelism, task_count)`;
- no thread creation inside the innermost edge contraction;
- preserve deterministic task order and bitwise action identity;
- avoid nested oversubscription.

Prefer caching temperature-independent quadrature/task topology and reusing allocations before adding threads.

## Same-case benchmark

Benchmark the exact R10 selected-48 endpoint on the same host with workers `1,2,4,8` where available. Record median of five complete endpoint runs after one warm-up. Segment timings are supplementary only.

Acceptance:

```text
endpoint observables and raw history predicates unchanged
bitwise equality for worker count changes where accumulation order is fixed
otherwise numerical differences <=0.1 of R10 convergence tolerances
selected worker endpoint wall improves >=10% or code is reverted
peak RSS does not increase >15%
```

Failure is `BLOCKED_PERFORMANCE`; keep the serial certified path and revert the optimization.

Claim ceiling: `VALIDATED ENDPOINT-CONSUMED PERFORMANCE IMPROVEMENT`.

---

## 3. Universal PR body template

Every PR body uses exactly:

```text
source_base:
dependency_prs:
blocker_target:
files_changed:
added_lines:
deleted_lines:
net_lines_total:
token_use_exact:
token_use_basis:
runtime_behavior_changed:
physics_behavior_changed:
raw_state_preserved:
commands_run:
passed:
failed:
skipped:
claim_ceiling:
remaining_blockers:
cost_effectiveness_verdict:
```

No completion claim is allowed when any required command was skipped.

## 4. Final stop rule

The DAG is complete only after R10 passes. R11 is optional performance work. A failure in R1-R10 leaves the project at the last merged claim ceiling and must not be bypassed by a larger run, different solver, looser tolerance, or manually adjusted output.
