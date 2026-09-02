# D-081R1 Exact Rust F10 Comparator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and statically admit an exact Rust implementation of the frozen 182-state D-080F Python comparator without invoking a trajectory solver.

**Architecture:** Add a new `F10ComparatorSystem` beside the legacy folded `IsotropicBoltzmannFlrwSystem`. Reuse the existing Rust ODE and low-level physics substrate only through explicit semantic gates, and compare compact frozen fixtures against the independent Python oracle.

**Tech Stack:** Rust 1.94.1, `nalgebra 0.35.0`, existing RABBIT collision modules, `serde_json 1.0.150`, offline `Cargo.lock` vendor; Python oracle artifacts only as test data.

**Spec:** `docs/superpowers/specs/2026-09-02-d081r1-rust-f10-comparator-design.md`

## Global Constraints

- Preserve D-080F physics, state meaning, grid, domain, collision catalogue, tolerances, support semantics, and failure behaviour.
- State order is exactly `c_e[0:n], c_mu[0:n], c_tau[0:n], T_gamma, elapsed_time`.
- The first qualifying case is `n=60`, `y_max=30`, state size `182`.
- Use the strict complementary-log-log chart; no clipping, floors, projection, or state repair.
- Do not call `ode::solve` in D-081R1.
- Do not add a new numerical library or modify dependency versions.
- All Cargo validation must pass with `--locked --offline` against the D-081R0 exact vendor.

---

### Task 1: Commit the compact oracle fixture and parser

**Files:**
- Create: `native/rabbit_cpu/tests/fixtures/d081r1/oracle_case.json`
- Create: `native/rabbit_cpu/src/f10_oracle_fixture.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: `native/rabbit_cpu/src/f10_oracle_fixture.rs`

**Interfaces:**
- Produces: `F10OracleCase::load() -> Result<F10OracleCase, String>` for test-only static admission.

- [ ] **Step 1: Generate the compact JSON from the frozen D-081R0 oracle artifact**

Include source hashes, nodes, weights, packed state, base RHS, columns `0,59,60,119,120,179,180,181`, four directions and their `Jv`, diagnostics, and support signature. The generation script must verify the oracle artifact SHA-256 before writing JSON.

- [ ] **Step 2: Write a failing fixture-shape test**

```rust
#[test]
fn oracle_case_has_exact_d080f_shape_and_hashes() {
    let case = F10OracleCase::load().unwrap();
    assert_eq!(case.order, 60);
    assert_eq!(case.y_max.to_bits(), 30.0_f64.to_bits());
    assert_eq!(case.state.len(), 182);
    assert_eq!(case.base_rhs.len(), 182);
    assert_eq!(case.selected_columns.len(), 8);
    assert_eq!(case.private_comparator_git_blob, "de44feee0aa484abe26976c7dc34c579643005b5");
}
```

- [ ] **Step 3: Run the test and confirm RED**

```bash
cargo test --locked --offline f10_oracle_fixture::tests::oracle_case_has_exact_d080f_shape_and_hashes -- --exact
```

Expected: compile failure because `F10OracleCase` does not exist.

- [ ] **Step 4: Implement the minimal test-only parser**

Use existing `serde_json`; reject missing fields, duplicate selected-column indices, nonfinite numbers, wrong shapes, or hash mismatches.

- [ ] **Step 5: Run the test and commit**

```bash
cargo test --locked --offline f10_oracle_fixture::tests -- --nocapture
git add native/rabbit_cpu/tests/fixtures/d081r1/oracle_case.json native/rabbit_cpu/src/f10_oracle_fixture.rs native/rabbit_cpu/src/lib.rs
git commit -m "test(d081r1): freeze compact Python oracle fixture"
```

### Task 2: Implement exact state layout, chart, and affine grid

**Files:**
- Create: `native/rabbit_cpu/src/f10_comparator.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Produces: `F10Grid::affine_legendre(order, y_max)`, `F10Layout`, `decode_cloglog_state`.

- [ ] **Step 1: Write failing layout/chart/grid tests**

```rust
#[test]
fn layout_is_three_flavours_then_temperature_and_elapsed() {
    let layout = F10Layout::new(60).unwrap();
    assert_eq!(layout.dimension(), 182);
    assert_eq!(layout.electron(), 0..60);
    assert_eq!(layout.muon(), 60..120);
    assert_eq!(layout.tau(), 120..180);
    assert_eq!(layout.t_gamma(), 180);
    assert_eq!(layout.elapsed(), 181);
}

#[test]
fn affine_grid_matches_frozen_python_nodes_and_weights() {
    let case = F10OracleCase::load().unwrap();
    let grid = F10Grid::affine_legendre(60, 30.0).unwrap();
    assert_blockwise_ulp_close(&grid.nodes, &case.nodes, 16);
    assert_blockwise_ulp_close(&grid.weights, &case.weights, 32);
}
```

- [ ] **Step 2: Confirm RED**

Run `cargo test --locked --offline f10_comparator::tests -- --nocapture` and require missing-type failures.

- [ ] **Step 3: Implement layout and grid**

Map `quadrature::gauss_legendre_rule` using

```rust
let y = 0.5 * y_max * (x + 1.0);
let w = 0.5 * y_max * w_x;
```

Implement stable chart decoding:

```rust
let e = c.exp();
let f = -(-e).exp_m1();
let df_dc = (c - e).exp();
```

Reject `f <= 0`, `f >= 1`, nonfinite `f`, or nonpositive/nonfinite chain factors.

- [ ] **Step 4: Add mutation tests**

Kill swapped temperature/elapsed indices, exponential-grid substitution, logit decoding, and a two-bank layout.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests -- --nocapture
git add native/rabbit_cpu/src/f10_comparator.rs native/rabbit_cpu/src/lib.rs
git commit -m "feat(d081r1): add exact F10 state chart and affine grid"
```

### Task 3: Implement primal thermodynamics and packed RHS shell

**Files:**
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Produces: `F10ComparatorSystem::rhs_static`, `F10Diagnostics` with moments, Hubble, and energy-transfer ledger.

- [ ] **Step 1: Write a failing no-collision thermodynamic test**

Freeze occupations and set a test collision provider to zero. Compare neutrino moments, electromagnetic EOS, `H`, photon cooling, and elapsed row with the compact oracle components.

- [ ] **Step 2: Confirm RED**

Run the exact test and require `rhs_static` absence.

- [ ] **Step 3: Implement the thermodynamic shell**

Use `T_cm = 10.0 * exp(-N)`, three flavour-pair moment sums, existing constants only after exact-value assertions, and state outputs in the frozen ordering.

- [ ] **Step 4: Add strict failure tests**

Reject wrong length, nonfinite `N`, nonpositive `T_gamma`, saturated cloglog occupations, and nonfinite EOS output without modifying the input state.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests::thermodynamic_shell -- --nocapture
git commit -am "feat(d081r1): add exact F10 thermodynamic RHS shell"
```

### Task 4: Add unfurled three-flavour collision action

**Files:**
- Modify: `native/rabbit_cpu/src/electron_spectral.rs`
- Modify: `native/rabbit_cpu/src/neutrino_self_spectral.rs`
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: corresponding module tests

**Interfaces:**
- Produces: unfurled six-species or three-pair action values and diagnostic event ledgers without folding `mu` and `tau`.

- [ ] **Step 1: Write failing symmetry and antisymmetry tests**

Require `mu=tau` to reproduce the folded action and require a nonzero `mu-tau` antisymmetric perturbation to remain observable.

- [ ] **Step 2: Confirm RED**

The existing folded API must fail the antisymmetric test.

- [ ] **Step 3: Expose minimal unfurled low-level output**

Reuse event contractions and matrix elements, but return separate electron, muon, and tau banks. Do not change the legacy folded functions.

- [ ] **Step 4: Add catalogue and conservation tests**

Check event counts, signs, multiplicities, CP symmetry, flavour symmetry, weighted number conservation for self collisions, and differentiated neutrino/electromagnetic first law.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline electron_spectral::tests -- --nocapture
cargo test --locked --offline neutrino_self_spectral::tests -- --nocapture
cargo test --locked --offline f10_comparator::tests::collision -- --nocapture
git commit -am "feat(d081r1): expose unfurled F10 collision action"
```

### Task 5: Close primal RHS parity

**Files:**
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Consumes: compact oracle fixture and unfurled collision action.
- Produces: admitted `rhs_static` at the retained D-080F state.

- [ ] **Step 1: Write the failing retained-state parity test**

Compute block-scaled residuals separately for spectral, `T_gamma`, and elapsed rows. Record all three and fail if any exceeds its prospectively frozen bound.

- [ ] **Step 2: Confirm RED**

Run the exact test and preserve the first measured discrepancy by component.

- [ ] **Step 3: Repair semantics only**

Classify discrepancies into chart/grid, event catalogue, matrix normalisation, interpolation, thermodynamics, or output assembly. Do not widen thresholds.

- [ ] **Step 4: Kill primal mutations**

Pauli sign, omitted pair block, omitted self block, flavour swap, multiplicity, grid map, and energy-transfer sign must all fail.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests::retained_rhs_parity -- --exact --nocapture
git commit -am "feat(d081r1): close retained Rust Python RHS parity"
```

### Task 6: Implement analytic JVP and selected Jacobian columns

**Files:**
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Produces: `jvp_static` and selected Jacobian columns.

- [ ] **Step 1: Write failing selected-column tests**

Compare columns `0,59,60,119,120,179,180,181`; require column 181 to be exact zero.

- [ ] **Step 2: Confirm RED**

Run exact tests before implementation.

- [ ] **Step 3: Implement spectral and thermal tangents**

Use the exact cloglog chain, unfurled event tangent, Hubble rank-one terms, analytic moving-kinematics `T_gamma` column, and passive elapsed column.

- [ ] **Step 4: Add column mutations**

Kill transpose, block swap, missing Hubble, missing heat-capacity derivative, reversed `Q_em,T`, and nonzero elapsed column.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests::selected_jacobian_columns -- --exact --nocapture
git commit -am "feat(d081r1): add exact F10 selected Jacobian columns"
```

### Task 7: Close mixed and holdout Jv parity

**Files:**
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Produces: full `jvp_static` admission over four frozen directions.

- [ ] **Step 1: Write four failing direction tests**

Use the D-080F contribution-scaled block metric and preserve the legacy forward-relative residual diagnostically.

- [ ] **Step 2: Confirm RED**

Record first failures without changing the directions or threshold.

- [ ] **Step 3: Implement remaining JVP assembly**

Ensure action linearity, exact state ordering, and no hidden re-evaluation through finite differences.

- [ ] **Step 4: Add contribution-scale mutation**

A 1% selected-column mutation and a `mu-tau` fold must fail the four-direction gate.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests::frozen_direction_parity -- --nocapture
git commit -am "feat(d081r1): close mixed Rust Python JVP parity"
```

### Task 8: Assemble and certify the full static matrix

**Files:**
- Modify: `native/rabbit_cpu/src/f10_comparator.rs`
- Test: `native/rabbit_cpu/src/f10_comparator.rs`

**Interfaces:**
- Produces: `jacobian_static` row-major `182 x 182` matrix and `rhs_and_jacobian_static`.

- [ ] **Step 1: Write failing assembly and fused-evaluation tests**

Require exact matrix shape, selected columns, `Jv`, zero final input column, and bitwise equality between fused and separate calls.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Implement columnwise or batched assembly**

Reuse fixed-state geometry without global monkeypatching. Prepared state must be immutable and keyed by exact finite `(N,Y)` bits.

- [ ] **Step 4: Record time and memory as diagnostics only**

No production speed claim is permitted in D-081R1.

- [ ] **Step 5: Run and commit**

```bash
cargo test --locked --offline f10_comparator::tests::full_static_matrix -- --exact --nocapture
git commit -am "feat(d081r1): assemble exact Rust F10 static Jacobian"
```

### Task 9: Add diagnostic PyO3 surface and evidence workflow

**Files:**
- Modify: `native/rabbit_cpu/src/lib.rs`
- Create: `native/rabbit_cpu/src/f10_python.rs`
- Create: `.github/workflows/d081r1_rust_static_admission.yml`
- Create: `docs/audit/BD635_D081R1_RUST_STATIC_ADMISSION_RESULT_2026-09-02.md`

**Interfaces:**
- Produces: test/research-only Python methods `rhs`, `jvp`, `selected_columns`, and `diagnostics`; no solver method.

- [ ] **Step 1: Write a failing Python-surface smoke test**

Require exact state size and typed errors; do not accept Python callbacks into Rust physics.

- [ ] **Step 2: Implement minimal read-only diagnostic class**

- [ ] **Step 3: Run all static admission tests offline**

```bash
cargo fmt --all -- --check
cargo check --release --locked --offline
cargo clippy --all-targets --all-features --locked --offline -- -D warnings
cargo test --locked --lib --offline f10_comparator -- --nocapture
```

- [ ] **Step 4: Generate receipt and mutation ledger**

Bind source/tree, toolchain, lock, oracle hashes, every residual, and claim ceiling.

- [ ] **Step 5: Commit**

```bash
git add native/rabbit_cpu .github/workflows/d081r1_rust_static_admission.yml docs/audit/BD635_D081R1_RUST_STATIC_ADMISSION_RESULT_2026-09-02.md
git commit -m "ci(d081r1): certify static Rust F10 comparator parity"
```

### Task 10: Independent closeout and D-081R2 handoff

**Files:**
- Create: `docs/audit/BD636_D081R1_DUAL_AUDIT_2026-09-02.md`
- Create: `docs/superpowers/plans/2026-09-02-d081r2-rust-diffsol-integration.md`

- [ ] **Step 1: Run PHYS-MATH audit**

Check charts, signs, units, conservation, equilibrium, flavour modes, support boundaries, and limits.

- [ ] **Step 2: Run PHYS-MATH-CODE audit**

Check exact path identity, failure preservation, fixture provenance, mutation sensitivity, cache immutability, and offline reproducibility.

- [ ] **Step 3: Stop on any P0 or unresolved P1 affecting semantics**

- [ ] **Step 4: If static admission passes, write the D-081R2 plan**

D-081R2 may implement `OdeSystem` and call existing diffsol only after this gate.

- [ ] **Step 5: Commit the closeout**

```bash
git add docs/audit/BD636_D081R1_DUAL_AUDIT_2026-09-02.md docs/superpowers/plans/2026-09-02-d081r2-rust-diffsol-integration.md
git commit -m "docs(d081r1): close static Rust comparator admission"
```
