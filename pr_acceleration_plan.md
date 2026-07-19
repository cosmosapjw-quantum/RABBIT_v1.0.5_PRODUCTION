# PR Acceleration Plan

Date: 2026-06-02

## 2026-06-17 Status Update

This plan is historical for BD279/BD280 ordering. The PR-2 parity blocker below
has since been exercised on current head:

- BD490:
  `diagnostic_outputs/bd490_pr_b_collision_on_parity_current_head/bd490_split_pairwise_cold_endpoint_current_head.json`
  reached the cold endpoint and passed LRS/non-LRS parity, but failed the
  `N_eff_3T` floor/band check with both rows high (`N_eff_3T ~= 3.11496`).
- BD491:
  `diagnostic_outputs/bd491_pr_b_thermal_collision_on_split_current_head/bd491_q4_thermal_start_lrs_nonlrs_collision_on_parity_current_head.json`
  reached the cold endpoint and passed the controlled PR-B pair:
  `default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`,
  LRS `N_eff_3T=3.0348008780946367`, non-LRS
  `N_eff_3T=3.0348087179727026`, and
  `delta.N_eff_3T=7.839878065851735e-06`.

This closes the current-head q4 thermal-start controlled PR-B floor/parity
blocker, not the whole programme. Remaining blockers include nonzero-shear/ell
convergence, high-q evidence before structured-solve claims, component-wall
residual attribution, and broader setting coverage. No optimization default is
turned on by this document update.

BD490/BD491 are not a single-knob ablation, and BD491 top-level `passed=false`
still prevents treating it as full readiness/convergence validation. Use the
controlled PR-B pair/floor object as scoped evidence only.

This plan converts the BD279/BD280 reconciliation into at most six PR-sized
changes. It preserves:

- QKE out of scope;
- no public-production claim;
- CPU-JAX plus in-tree Rodas5P/AP65 as the target path;
- phase-2 BE/BDF2/Newton corrector unless directly falsified;
- raw negative/nonfinite evidence;
- `N_eff_3T` as a proxy until physically pinned.

No PR below is allowed to add a standalone readiness/manifest/figure/hash gate
unless it deletes or consolidates obsolete plumbing and moves a named runtime
physics, solver, or performance blocker.

## Historical PR-1: Add AP65 Solver A/B Telemetry And Prototype Block/Low-Rank Endpoint Wiring

**Current status (2026-06-17):** conditional and no longer first for q4.
BD491-style q4 evidence keeps dense outer linear solve below the phase2/payload
and residual-attribution targets. Revisit this item only if high-q W/J evidence
or a captured AP65-like system shows the linear solve is a dominant bucket.

**Historical blocker moved:** dense AP65 host linear solve and missing memory
attribution.

**Historical rationale:** low-rank/Woodbury and block-sparse pieces already pass
algebraic tests, but AP65 endpoint still uses dense `W=I/(gamma*h)-J` and LU.
This rationale is superseded as a first-next step by BD491 q4 wall evidence; it
is now conditional high-q/large-WJ work, not the next default route.

**Files likely touched:**

- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- `src/rabbit/jax/solver_jax_rodas5p.py`
- `src/rabbit/jax/linear_solve_strategies.py`
- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `tests/test_j04_jax_rodas5p.py`
- new focused AP65 solver-policy test

**Tests added:**

- dense-LU vs block/low-rank stage residual parity on a captured or synthetic
  AP65-like system;
- AP65 q4 row A/B smoke marked slow or integration;
- artifact memory-field smoke for `ru_maxrss`, `/proc/self/status:VmHWM`, and
  optional `tracemalloc`.

**Code deleted/consolidated:**

- remove or demote stale "not routed yet" comments once a real opt-in route
  exists;
- consolidate duplicate solver-policy metadata fields into one policy summary
  if safe.

**Acceptance criteria:**

- existing low-rank and block-sparse unit tests still pass;
- q4 dense-vs-prototype run has endpoint observables within declared tolerance;
- artifact includes memory fields without changing raw observables;
- no default public/production promotion.

**Rollback criteria:**

- stage residual parity fails;
- endpoint `Yp`, `D/H`, `N_eff_3T`, `Sigma_H`, or raw diagnostics drift outside
  tolerance;
- memory instrumentation changes solver behavior or hides raw evidence.

**Expected risk:** high.

**Runtime impact category:** medium to slow.

## PR-2: Resolve Controlled LRS/Non-LRS FLRW-Limit `N_eff_3T` Parity

**Blocker moved:** BD279 `N_eff_3T ~= 2.994` path-dependent parity gap.

**Current status (2026-06-17):** RESOLVED for the current-head q4 thermal-start
controlled pair by BD491. The equal-temperature/current-head pair in BD490
already passed LRS/non-LRS parity but failed the floor high; the BD491
thermal-start pair passed both parity and floor. Keep this section as historical
design context and do not treat it as the next open implementation PR.

**Files likely touched:**

- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- `src/rabbit/transport/augmented_typeI_weak_network.py`
- `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`
- new `tests/test_lrs_nonlrs_flrw_parity.py`

**Tests added:**

- controlled LRS-radial vs non-LRS zero-shear pair with identical:
  `h_max`, `N_span`, q grid, source-composition policy, restart/handoff policy,
  weak correction level, and initialization metadata;
- optional `h_max` sensitivity probe only if the first pair remains split.

**Code deleted/consolidated:**

- consolidate older ad hoc parity/probe surfaces into the new parity pair;
- remove duplicated metadata-only parity assertions that do not run a physics
  row.

**Acceptance criteria:**

- either `N_eff_3T`, `Yp`, `D/H`, `T_nu_e/T_std`, and `T_nu_x/T_std` agree
  within declared tolerance, or the residual gap is sharply attributed to one
  named code path/setting;
- all rows serialize init/source/projection/restart policy explicitly;
- no clipping/truncation of negative/nonfinite evidence.

**Rollback criteria:**

- parity pair requires broad new gate plumbing;
- controlled pair still silently falls back to fixed `Xn0` or nonzero `A0`;
- source-policy metadata is ambiguous.

**Expected risk:** medium-high.

**Runtime impact category:** slow but bounded.

**Resolution evidence:** BD491 satisfies the acceptance criterion for the
thermal-start controlled q4 pair with `delta.N_eff_3T=7.839878065851735e-06`
and both endpoint rows classified as `full_bbn_completed`. This is not evidence
for QKE, public-production support, high-q convergence, or nonzero-shear
anisotropic transport.

## PR-3: Land Physics Invariant Tests And Fence Collisional `ell_max=2` Claims

**Blocker moved:** weak physics/test gates and overbroad PSTF exactness claims.

**Files likely touched:**

- `tests/test_three_temperature_closure_invariants.py`
- `tests/test_augmented_collision_bridge.py`
- `src/rabbit/config/conventions.py`
- `src/rabbit/config/grids.py`
- `docs/audit/v4_derivations/typeI_augmented_pstf_noqke.md`

**Tests added:**

- 3T closure invariant tests already introduced in this audit pass:
  Python/JAX `N_eff_3T` consistency, positive heavy-bank heat response, and
  equal-temperature nu-nu neutrality;
- no-projection FLRW collision source invariant if the API supports disabling
  projection safely;
- occupation-source to `dA` energy-moment invariant using `f(1-f) q^3`;
- branch agreement for heavy-bank degeneracy if per-species functions appear.

**Code deleted/consolidated:**

- rewrite or remove collisional-overbroad `ell_max=2 exact` text;
- demote exactness to collisionless/free-streaming regime and document fixed
  diagonal three-mode S2 as current approximation.

**Acceptance criteria:**

- fast invariant suite passes;
- no new broad readiness gate added;
- config/docs no longer imply generic collisional ell/m convergence.

**Rollback criteria:**

- new invariant is too implementation-specific and cannot fail on a real
  physics regression;
- doc/config changes weaken an actually valid collisionless convention.

**Expected risk:** medium.

**Runtime impact category:** cheap.

## PR-4: Extract AP65 RHS Initialization/Phase Modules And Fail Closed

**Blocker moved:** 19,678-line RHS god-module and silent default class.

**Files likely touched:**

- `src/rabbit/validation/augmented_continuous_ap65_rhs.py`
- new modules under `src/rabbit/validation/ap65/` or equivalent:
  - `anchor_init.py`
  - `phase1_prerun.py`
  - `phase2_corrector.py`
  - `restart_handoff.py`
  - `host_config.py`
- `tests/test_augmented_continuous_ap65_rhs.py`

**Tests added:**

- characterization tests for standard anchor initialization;
- fail-closed test: production/endpoint mode without explicit init/source
  policy raises;
- phase-1/phase-2 handoff metadata parity.

**Code deleted/consolidated:**

- duplicated restart/default construction;
- silent fallback paths for standard-anchor rows;
- dead local helper copies where replaced by extracted modules.

**Acceptance criteria:**

- endpoint dry-run metadata unchanged except clearer module provenance;
- focused tests pass;
- original RHS file LOC materially drops;
- public behavior is preserved unless a fallback was explicitly proven wrong.

**Rollback criteria:**

- endpoint row behavior changes without an explicit physics reason;
- imports become circular;
- extraction blocks follow-up solver/parity work.

**Expected risk:** medium-high.

**Runtime impact category:** cheap tests plus one medium smoke.

## PR-5: Split Span Ladder Runtime, Case Generation, And Artifact I/O

**Blocker moved:** 13,359-line span-ladder god-module and brittle count locks.

**Files likely touched:**

- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
- new modules under `src/rabbit/validation/ap65/` or equivalent:
  - `endpoint_matrix.py`
  - `runtime_rows.py`
  - `artifact_io.py`
  - `resolution_summary.py`
- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`
- `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`

**Tests added:**

- matrix coverage properties rather than only `len(cases)==98`;
- first/last label smoke retained only as a compatibility tripwire;
- artifact IO roundtrip preserving raw observable fields and claim-boundary
  fields.

**Code deleted/consolidated:**

- duplicate dry-run and endpoint-matrix wrappers;
- obsolete readiness/witness/bundle plumbing adjacent to this runner;
- noisy nested summary fields if consumers are not using them.

**Acceptance criteria:**

- default endpoint matrix still covers q4/q5/q9/q10, zero/small/milliscale
  shear, and a3x5/a4x7 classes;
- raw `Yp`, `D/H`, `N_eff_3T`, `Sigma_H`, failure rows, and no-QKE/public flags
  are preserved;
- net validation/script LOC decreases or responsibility boundaries become
  demonstrably smaller.

**Rollback criteria:**

- artifact consumers break without a compatibility layer;
- deleted fields include raw physics evidence;
- refactor becomes claim-gate churn rather than runtime simplification.

**Expected risk:** medium.

**Runtime impact category:** cheap to medium.

## PR-6: Teff And Evidence Plumbing Cleanup After Call-Graph Check

**Blocker moved:** deprecated import-reachable surfaces and validation plumbing
that slow development.

**Files likely touched:**

- `src/rabbit/transport/teff_collision_bridge.py`
- `src/rabbit/weak/teff_correction.py`
- `src/rabbit/jax/teff_correction_jax.py`
- `src/rabbit/config/backend_capabilities.py`
- `src/rabbit/config/feature_capabilities.py`
- readiness/publication/figure/witness validation modules
- Teff-related tests and registry sync scripts

**Tests added/kept:**

- keep `enable_teff=True` rejection tests;
- call-graph import test proving active no-QKE endpoint path does not import
  Teff runtime modules;
- registry/doc regeneration tests if capability tables change.

**Code deleted/consolidated:**

- delete or quarantine Teff runtime trio after import tests;
- consolidate figure/readiness/manifest/bundle witnesses into one provenance
  utility or remove if no active consumer exists;
- remove tests that only assert non-ready scaffolding and no physics behavior.

**Acceptance criteria:**

- active endpoint path imports and tests pass without Teff modules;
- public runtime still rejects Teff and QKE overclaims;
- generated docs/capability tables are regenerated if registries change;
- net obsolete plumbing decreases.

**Rollback criteria:**

- active transport/weak/collision code still depends on Teff helpers;
- deleting a witness removes a claim firewall with no replacement;
- generated docs drift.

**Expected risk:** medium-high.

**Runtime impact category:** cheap.

## Ordering Notes

1. PR-2 is no longer the next open current-head blocker for the q4 thermal-start
   controlled pair; use BD491 as the evidence baseline.
2. PR-1 solver A/B is still conditional on high-q/WJ evidence; q4 `[62,62]`
   does not justify structured-solve priority by itself.
3. PR-3 can proceed in parallel because it is cheap invariant hardening and
   claim fencing.
4. PR-4 should precede PR-5 because the span ladder depends on RHS interfaces.
5. PR-6 should wait until call-graph evidence is explicit.
6. The next blocker-bearing work should target nonzero-shear/ell convergence,
   component-wall residual attribution on the BD491-style endpoint, and only
   then performance defaults.
7. No language rewrite should be considered without a post-BD491 profile proving
   a stable residual kernel above the project threshold.
