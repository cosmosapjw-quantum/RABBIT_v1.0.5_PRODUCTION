# D-081R1D Full Six-Species Collision Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble and statically admit the exact three-pair/six-species Rust collision action against the frozen private Python comparator, without invoking an ODE solver.

**Architecture:** Treat the frozen Python full action as the sole positive oracle. Rematerialize any missing affine-grid, spectral, and kinematic foundations in the remote Rust lineage through compact parity fixtures, then implement self and electron actions as separate reviewable modules before combining their conservation and diagnostic ledgers. The legacy folded two-bank action is retained only as a negative control.

**Tech Stack:** Rust 1.94.1, existing exact offline `Cargo.lock` vendor, NumPy 2.4.4 and SciPy 1.17.1 for oracle generation, existing RABBIT scalar kernel primitives, no new dependency.

**Spec:** `docs/audit/BD636_D081R1D_FULL_ACTION_ORACLE_CONTRACT_2026-09-03.md`

## Global constraints

- Base exactly on D-081R1C final head `5e4adf6af6ae1983ba0a272074503a7d6f469a46`.
- Preserve the Python comparator Git blob `de44feee0aa484abe26976c7dc34c579643005b5`.
- Preserve the three-pair state chart and explicit six-species action ordering.
- Preserve affine Gauss–Legendre order 8 on `[0,8]` for implementation fixtures.
- Use order-60 `[0,30]` retained state only after order-8 admission; it is a holdout.
- Do not use the legacy two-bank folded action as a positive oracle.
- Do not call `ode::solve`, build a Jacobian, benchmark solver speed, or change dependencies.
- Every Cargo command must use `--locked --offline` with Rust 1.94.1.
- No tolerance or threshold may be widened after a Rust full-action output exists.

---

### Task 1: Freeze deterministic full-action Python fixtures

**Files:**
- Create: `native/rabbit_cpu/tests/fixtures/d081r1/generate_full_collision_action_case.py`
- Create: `native/rabbit_cpu/tests/fixtures/d081r1/full_collision_action_case.json`
- Create: `docs/audit/artifacts/d081r1d0/full_action_oracle_receipt.json`

**Interfaces:**
- Produces: schema `rabbit.d081r1.full_collision_action.v1` containing `equilibrium`, `thermal_split`, and `mu_tau_split` cases.

- [ ] **Step 1: Write the generator against the frozen Python comparator**

Generate exact binary64 bit patterns for grid, three pair-cloglog rows, six-species native/modal self/electron/total actions, nine self rows, fifteen electron families, energy ledgers, support/correction signatures, diagnostics, moments, and absolute envelopes.

- [ ] **Step 2: Execute it twice and require byte identity**

```bash
python3 native/rabbit_cpu/tests/fixtures/d081r1/generate_full_collision_action_case.py
cp native/rabbit_cpu/tests/fixtures/d081r1/full_collision_action_case.json /tmp/full-action.first.json
python3 native/rabbit_cpu/tests/fixtures/d081r1/generate_full_collision_action_case.py
cmp /tmp/full-action.first.json native/rabbit_cpu/tests/fixtures/d081r1/full_collision_action_case.json
```

- [ ] **Step 3: Run Python-only physical checks**

Require exact schema and source identity, three cases, 27/15 catalogue counts, finite diagnostics, equilibrium first-law closure, positive thermal restoring energy transfer, and nonzero `mu_tau_residual` in the split case.

- [ ] **Step 4: Commit the fixture and receipt**

```bash
git add native/rabbit_cpu/tests/fixtures/d081r1 docs/audit/artifacts/d081r1d0
git commit -m "test(d081r1d0): freeze full six-species Python action oracle"
```

### Task 2: Rematerialize exact grid and spectral foundations

**Files:**
- Create: `native/rabbit_cpu/src/f10_action_grid.rs`
- Create: `native/rabbit_cpu/src/f10_action_spectral.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: module-local Rust tests

**Interfaces:**
- Produces: `F10ActionGrid`, `F10Spectra`, `modal_basis`, `modal_coefficients`, `modal_product`, `native_action`.

- [ ] **Step 1: Write RED tests from the full-action fixture**

Check order, nodes, weights, native cloglog-to-logit conversion, mapped orthonormal basis, modal coefficients, interpolation, and native/modal reconstruction.

- [ ] **Step 2: Confirm RED before implementation**

```bash
cargo +1.94.1 test --locked --offline f10_action_grid::tests -- --nocapture
cargo +1.94.1 test --locked --offline f10_action_spectral::tests -- --nocapture
```

- [ ] **Step 3: Implement the minimal exact foundations**

Use existing quadrature only after node/weight parity. Keep strict-open chart semantics; no clipping or projection.

- [ ] **Step 4: Kill grid and basis mutations**

Exponential-grid substitution, missing mapped-basis normalization, flavour-bank fold, and out-of-domain interpolation must fail.

- [ ] **Step 5: Verify and commit**

```bash
cargo +1.94.1 fmt --all -- --check
cargo +1.94.1 test --locked --offline f10_action_ -- --nocapture
git commit -am "feat(d081r1d1): add exact action grid and spectral foundations"
```

### Task 3: Rematerialize exact kinematic batches

**Files:**
- Create: `native/rabbit_cpu/src/f10_action_kinematics.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: module-local Rust tests

**Interfaces:**
- Produces: massless self, finite-electron elastic, and pair-annihilation kinematic batches with support, measure ingredients, six invariant products, and correction signatures.

- [ ] **Step 1: Add compact selected-point kinematic fixture fields and RED tests**

Freeze selected supported and rejected points for self, elastic, and pair branches.

- [ ] **Step 2: Confirm RED**

Require missing-type or parity failures.

- [ ] **Step 3: Implement the same fixed angular/radial rules**

Preserve GL12/GL48 and four midpoint azimuth nodes. Return typed errors for nonfinite input and materially invalid support.

- [ ] **Step 4: Kill kinematic mutations**

Wrong Källén sign, omitted electron mass, wrong boost direction, and event-measure denominator must fail.

- [ ] **Step 5: Verify and commit**

```bash
cargo +1.94.1 test --locked --offline f10_action_kinematics::tests -- --nocapture
git commit -am "feat(d081r1d1): add exact action kinematics"
```

### Task 4: Implement the 27-event self action

**Files:**
- Create: `native/rabbit_cpu/src/f10_self_action.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: module-local Rust tests

**Interfaces:**
- Produces: six-species modal/native self action, nine row actions, support/correction ledger, event entropy and energy residuals.

- [ ] **Step 1: Write RED parity tests for all three order-8 cases**

Compare native action, modal action, all nine rows, support/correction signatures, signed/absolute moments, and diagnostics against the fixture.

- [ ] **Step 2: Confirm RED**

```bash
cargo +1.94.1 test --locked --offline f10_self_action::tests -- --nocapture
```

- [ ] **Step 3: Implement `R=W*M*C` and `(+,+,-,-)` modal routing**

Use only D-081R1C-admitted scalar primitives. Preserve pair-conversion orientations independently.

- [ ] **Step 4: Add physical and mutation gates**

Require weighted number/energy conservation, nonnegative entropy-production core within frozen roundoff allowance, CP/mu-tau symmetry on symmetric cases, nonzero mu-tau response in the split case, and kills for leg sign, coefficient, orientation, omitted family, and row swap.

- [ ] **Step 5: Verify and commit**

```bash
cargo +1.94.1 test --locked --offline f10_self_action::tests -- --nocapture
git commit -am "feat(d081r1d2): implement exact six-species self action"
```

### Task 5: Implement electron elastic and pair actions

**Files:**
- Create: `native/rabbit_cpu/src/f10_electron_action.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Test: module-local Rust tests

**Interfaces:**
- Produces: six-species modal/native electron action, fifteen family actions, independent `Q_nu/Q_em`, entropy, support, and correction ledgers.

- [ ] **Step 1: Write RED parity tests**

Compare the full action, every family, bath-energy map, diagnostics, and moments for all order-8 cases.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Implement elastic and pair routing**

Route elastic target differences and both pair incoming species explicitly. Never repair multiplicity after the sum.

- [ ] **Step 4: Add physical and mutation gates**

Require elastic target-number conservation, pair lepton-asymmetry conservation, first-law closure, thermal restoring sign, CP/mu-tau symmetry, and kills for omitted elastic/pair families, interference sign, CP partner, measure, basis, and antineutrino multiplicity.

- [ ] **Step 5: Verify and commit**

```bash
cargo +1.94.1 test --locked --offline f10_electron_action::tests -- --nocapture
git commit -am "feat(d081r1d3): implement exact electron and pair actions"
```

### Task 6: Combine total action and execute order-60 holdout

**Files:**
- Create: `native/rabbit_cpu/src/f10_full_action.rs`
- Modify: `native/rabbit_cpu/src/lib.rs`
- Create: `.github/workflows/d081r1d_full_action_admission.yml`
- Create: `docs/audit/BD637_D081R1D_FULL_ACTION_RESULT_2026-09-03.md`

**Interfaces:**
- Produces: `evaluate_f10_full_action` and a machine receipt; no RHS or solver API.

- [ ] **Step 1: Write combined RED tests**

Require `total=self+electron`, native/modal reconstruction, decomposition identity, all conservation/symmetry gates, and typed failures.

- [ ] **Step 2: Confirm RED**

- [ ] **Step 3: Implement the total facade**

No global monkeypatching and no hidden fallback to the legacy folded action.

- [ ] **Step 4: Execute untouched order-60 holdout**

Recover the exact retained state by SHA-256 and compare action/moments/support/first-law values with a separately generated frozen Python holdout. Do not alter order-8 thresholds after seeing it.

- [ ] **Step 5: Run exact offline closeout and commit**

```bash
cargo +1.94.1 fmt --all -- --check
cargo +1.94.1 check --release --locked --offline
cargo +1.94.1 clippy --all-targets --all-features --locked --offline -- -D warnings
cargo +1.94.1 test --locked --offline f10_ -- --nocapture
```

Publish source/tree/toolchain/lock/oracle hashes, forward and cancellation-aware residuals, all physical ledgers, mutation results, and claim ceiling. Remove the write-enabled workflow from the final tree.

## Completion boundary

D-081R1D is complete only when Tasks 1–6 pass. Its completion opens retained-state packed RHS parity. It does not open solver work directly.
