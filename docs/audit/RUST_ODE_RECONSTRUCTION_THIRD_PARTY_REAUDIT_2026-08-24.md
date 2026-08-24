# Third-Party Adversarial Re-Audit — Rust ODE Collision Reconstruction

Date: 2026-08-24 (Asia/Seoul)  
Audited branch: `external-audit/ode-rust-reconstruction-complete-20260824`  
Audited branch head: `5689f2889163c3cf939a2d83c66075910d1948ff`  
Candidate development base: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Review mode: independent third-party, adversarial, read-only  
Fresh numerical execution by this re-audit: **NONE**  
CI/status checks on audited head: **NONE PRESENT**

## 1. Executive verdict

```text
research prototype value              ACCEPT
negative-result / causal value        ACCEPT
mergeable production solver slice     REJECT
endpoint or cold-wall authority       NOT EARNED
same-physics parity authority         NOT EARNED
claimed blocker_movement_ratio=0.50   REJECT AS OVERSTATED
third-party blocker_movement ceiling  0.25
```

The update contains a serious and potentially valuable idea: reconstruct the finite-mass electron collision action as non-negative gain/loss edges, solve each edge inside exact Pauli capacities, and close the electromagnetic energy change transactionally. It also correctly keeps public-production, endpoint, and solver-promotion claims closed.

It is not yet a trustworthy solver component. The most serious defect is executable, not rhetorical: `PauliEdge::implicit_step` may exhaust its nonlinear iteration cap and return the bracket midpoint as success without checking the backward-Euler residual. The retained focused evidence reports `maximum_edge_iterations = 32`, exactly the hard cap used by this code. Therefore at least one executed edge may have entered the unverified-success branch. Positivity alone does not certify that the intended implicit equation was solved.

The second critical issue is that the advertised exact equilibrium null is forced by a bitwise special case. The unforced event/edge action is computed and then overwritten with zero at the exact FD anchor; the edge sweep also returns without applying any edge. This does not validate detailed balance. It can make the returned RHS discontinuous at the bitwise anchor and can make the returned Jacobian inconsistent with the actual branched RHS.

The reconstruction is also not on an active endpoint call path. It is crate-private, electron-collision-only, exercised by focused tests, and not composed with expansion, elapsed time, neutrino self-collisions, adaptive acceptance/rejection, or a same-physics reference. The correct claim is therefore:

> IMPLEMENTED FOCUSED RECONSTRUCTION PROTOTYPE WITH A CONCRETE ROOT-CERTIFICATION DEFECT AND OPEN PHYSICS/COMPOSITION GATES.

## 2. Review boundary

This re-audit inspected the committed source, retained P0 discriminator/result, reconstruction evidence note, worktree patch/inventory, tests, and repository workflow surface. It did not unpack or execute the 544 MB temporary snapshot, clone the repository locally, or rerun Cargo/Python commands. The GitHub head has no combined status checks and no workflow run associated with it. Previously recorded focused runs remain evidence only at their original scope.

The audit branch contains a 544,174,962-byte compressed temporary snapshot split across seven Git blobs, including build products, a Python environment, Git metadata, and linked-worktree state. That archive is useful for forensic custody but must never be used as a code-integration base.

## 3. What survives adversarial review

The following are credible within a narrow scope:

1. The P0 Python mapped collision path exhibits a high-q raw-tail problem and node-local gain/loss reconstruction fails on that mapped operator.
2. The Rust event coefficients are explicitly separated into non-negative production/destruction coefficients for the tested finite-mass event stream.
3. The folded elastic and pair edge algebra can preserve the stated weighted number or CP-difference invariant for one edge.
4. Candidate updates are formed on copies and committed only after validation, so ordinary error returns are transactional.
5. The segment-level event construction has a reproducible serial/parallel bitwise-equality test for the tested configuration.
6. The authors explicitly state that expansion/self-collision composition, adaptive control, same-physics parity, endpoint authority, and solver promotion remain open.

These strengths justify continued work. They do not override the failures below.

## 4. Adversarial findings

### RR-001 — Critical — unverified nonlinear root returned as success

**Current-head location**

- `native/rabbit_cpu/src/pauli_edge_step.rs:187-274`
- symbol: `PauliEdge::implicit_step`
- decisive branch:

```rust
for iteration in 0..32 { ... }
if iterations == 32 {
    extent = 0.5 * (lower + upper);
}
let candidate = self.occupations_at_extent(initial, extent)?;
Ok((candidate, ...))
```

**Failure mechanism**

When the loop reaches 32 iterations, the function returns the midpoint without checking

```text
r(xi) = xi - delta_t * J(f(xi))
```

or proving that the occupation-space bracket is sufficiently narrow. The evidence note reports `max_edge_iters=32` in both a tail sweep and a solver substep. Thus the cap branch is not hypothetical.

**Impact**

- false solver success;
- unknown local truncation defect;
- possible systematic dependence on edge ordering;
- energy closure may conceal an incorrect neutrino update by adjusting the electromagnetic bath to it.

**Required disposition**

`BLOCKER-ROOT-CERTIFICATE`; first code PR. No further physics promotion is admissible before it passes.

### RR-002 — Critical — root tolerances use a unit-scale floor on a tiny dimensional extent

**Current-head location**

- `native/rabbit_cpu/src/pauli_edge_step.rs:225-257`
- code:

```rust
let bracket_scale = lower.abs().max(upper.abs()).max(1.0);
let bracket_tolerance = 64.0 * f64::EPSILON * bracket_scale;
...
width <= 16.0 * f64::EPSILON * upper.abs().max(lower.abs()).max(1.0)
```

**Failure mechanism**

`xi` is a weighted extent and can be many orders of magnitude below one. The `max(1.0)` floor turns both tests into an absolute tolerance near `1e-14` in extent units. A bracket can be declared converged while its induced occupation uncertainty is enormous relative to a `1e-35` tail.

**Required remedy**

Certify residual relative to `max(|xi|, |delta_t J|, MIN_POSITIVE)` and certify bracket width after dividing by both cell measures, in occupation units. A unit floor is appropriate only after conversion to the dimensionless occupation interval.

### RR-003 — Critical — exact equilibrium is a forced branch, not a validated null

**Current-head locations**

- `native/rabbit_cpu/src/electron_spectral.rs:558-579`, `is_exact_anchor`
- `native/rabbit_cpu/src/electron_spectral.rs:594-626`, `action_values`
- `native/rabbit_cpu/src/electron_spectral.rs:640-650`, `transactional_step`
- `native/rabbit_cpu/src/electron_spectral.rs:732-790`, `exact_reference_state` and forced `fill(0.0)`

**Failure mechanism**

At bitwise-identical FD state and equal temperatures:

- the production action is computed, then overwritten with zero;
- the reconstructed edge action returns zero without evaluating edges;
- the edge step returns unchanged state with zero applications;
- the Jacobian is retained from the unforced action.

This proves only that the branch predicate works. It does not prove event-level microreversibility or aggregate detailed balance. If the unforced residual is nonzero, the RHS is discontinuous at the exact bit pattern and the Jacobian is not the derivative of the branched function.

**Required disposition**

`BLOCKER-DETAILED-BALANCE`. The shortcut must not be counted as validation. First expose and measure the unforced residual; then either prove it is within a prospectively fixed bound or repair the event/edge coefficients before removing the branch.

### RR-004 — High — the sweep has no consistency or order certificate

**Current-head location**

- `native/rabbit_cpu/src/electron_spectral.rs:652-709`
- symbol: `IsotropicElectronPauliEdges::transactional_step`

The method applies backward-Euler edge maps with half step in forward order and half step in reverse order. Reversing edge order does not make backward Euler self-adjoint. No test establishes first-order consistency, second-order symmetry, or a valid step-doubling estimator.

Required tests:

```text
(candidate(h)-f)/h -> action(f) as h -> 0
one_step(h) vs two_steps(h/2)
observed order on at least three h values
edge-order sensitivity at fixed h
```

Until those exist, the method is a positivity-preserving map, not a certified time integrator.

### RR-005 — High — P0 and P1 do not implement identical physics

P0 diagnoses the Python comoving-to-thermal interpolation path using the simplified deterministic collision reference. P1 constructs a different Rust finite-electron-mass direct-comoving event stream. The latter may be scientifically superior, but it is not a demonstrated repair of the exact former operator.

The P0 node-local reconstruction failure also does not uniquely imply that a two-node local edge decomposition is the only valid remedy; the composed interpolation operator is nonlocal in the comoving basis by construction.

Required claim correction:

```text
old: known blocker reduced; ratio 0.50
new: alternative finite-mass positivity candidate implemented; ratio <=0.25
```

A same-physics reference must be bound before using the word parity or replacement.

### RR-006 — High — total-energy residual is largely tautological

**Current-head location**

- `native/rabbit_cpu/src/isotropic_boltzmann.rs:625-725`
- symbol: `reconstruct_electron_collision_substep`

The method computes the neutrino energy after the edge sweep and solves

```text
rho_em(T_new) = rho_em(T_old) - (rho_nu,new - rho_nu,old)
```

so global energy closure is imposed algebraically. This is a useful transaction invariant, but a near-zero final residual does not independently validate the edge flux, pair/elastic rates, weak matrix element, or time discretization.

Required independent checks:

- as `h -> 0`, `Delta rho_nu / Delta t` converges to the energy moment of the existing unforced electron action;
- equilibrium unforced event action vanishes without a branch;
- electron energy loss inferred from event kinematics agrees with the bath update to the selected order;
- EOS inversion residual is explicitly bounded before success is returned.

### RR-007 — High — focused tail tests are not production-shaped

The key transactional test uses a four-node Gauss-Laguerre grid and manually sets logits to `-81.5` and `-92`. P0 implicated nodes near `q=70-81` on a 24-node Python grid; the retained Rust selected grid has 48 nodes. No selected-grid 48-state reconstruction test or retained-physics short-prefix test exists.

A tiny synthetic tail is useful for unit coverage but cannot establish production-grid conditioning, edge counts, root convergence, or QoI behavior.

### RR-008 — High — action reconstruction is not an independent oracle

`folded_pauli_edges_reconstruct_action_and_are_boundary_inward` compares two implementations assembled from the same event stream, same coefficients, same symmetrization, and same floating-point inputs. This is a structural self-consistency test, not an independent physics reference.

Required independent axes:

1. direct event contraction vs edge action at multiple states;
2. higher-precision or separately accumulated selected events;
3. tight collision-only ODE integration of the same unforced action vs the edge step;
4. production-shaped moment and weak-rate QoIs.

### RR-009 — High — reconstruction is not on an active endpoint path

**Current-head locations**

- `native/rabbit_cpu/src/isotropic_boltzmann.rs:625-725`, crate-private candidate method
- `native/rabbit_cpu/src/lib.rs:1-47`, no Python/public registration

The candidate is called by tests, not by `ode::solve`, the Python extension API, or an endpoint driver. Existing `OdeSystem::rhs` still evolves logits continuously by dividing the summed electron+self collision by `f(1-f)`.

Therefore no solver, endpoint, or repeated-run production blocker has moved yet.

### RR-010 — High — neutrino self-collision positivity remains unresolved

`neutrino_self_spectral.rs` uses a stable affinity for the instantaneous action, but the active ODE path still maps the total collision action through division by `f(1-f)`. The electron-only Pauli edge step cannot be composed honestly until the self-collision operator has either:

- a capacity-preserving event extent update; or
- a separately certified strict-logit step with no tail underflow/slaving and exact discrete invariants.

Do not silently omit self-collisions in a promoted endpoint run.

### RR-011 — High — no adaptive acceptance/rejection contract

The candidate freezes `delta_t = delta_ln_a/H` at the beginning of the collision substep and commits the result after one sweep. There is no one-step/two-half-step comparison, error norm, accepted/rejected history, minimum step, or failure reason. Coefficient variation through `T_gamma`, `T_cm`, and occupations is uncontrolled.

A positivity-preserving map can still be quantitatively wrong. Adaptive control is mandatory before prefix integration.

### RR-012 — Medium — electron substep is not isolated and repeats expensive work

`reconstruct_electron_collision_substep` first calls `physical_state_impl(..., false)`, which constructs both electron and self-collision actions, then reconstructs the electron event stream/edges again. This couples the electron operator to unrelated self-collision failure and duplicates event construction.

Required refactor: one collision-free thermodynamic snapshot helper returning occupations, moments, EOS, rho, and H; the electron operator then builds its own action/edges exactly once.

### RR-013 — Medium — bitwise equilibrium anchor is fragile

Equivalent equilibrium states produced through a different algebraic path or one-ulp perturbation do not take the special branch. The solver therefore sees qualitatively different semantics based on representation bits rather than a physical contract. This is especially dangerous when the Jacobian uses the unforced path.

### RR-014 — Medium — parallel benchmark is segment-only and the runtime policy is uncontrolled

`build_event_stream` chooses `available_parallelism()` and spawns scoped threads each time the event stream is built. The measured speedup is for one event-build/action segment, uses best-of-three timing, and has no endpoint or solver-context measurement. Repeated thread creation can oversubscribe a solver, test runner, or outer parallel loop.

Keep serial deterministic construction as the correctness baseline. Add explicit worker count, reuse, or caching only after the active prefix passes and only with a same-case wall benchmark.

### RR-015 — High — validation authority is absent on the published head

The publication commit ran packaging checks only. The evidence note states that no full Rust suite, full Python suite, gold test, endpoint, package, JAX, or Diffrax command was run. GitHub reports no status checks and no associated workflow run for the audited head.

Before any code PR can be called complete, execute at minimum:

```bash
cd native/rabbit_cpu
cargo fmt --all -- --check
cargo check --release
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
cargo test --release --lib
```

Then run the exact focused commands specified per PR. Historical logs do not satisfy this requirement.

### RR-016 — Medium — evidence custody should not become code history

The evidence packet includes 11,806 regular files and a 544 MB compressed snapshot. Do not merge, cherry-pick, or rebase that archive into the production branch. All executable PRs must begin from the clean code base and import only reviewed source/test hunks.

## 5. Current blocker map

| Blocker | Severity | Present state | Exit condition |
|---|---:|---|---|
| B-R1 edge root certificate | critical | false success possible; cap observed | every success carries residual + occupation-width certificate; cap returns error |
| B-R2 raw detailed balance | critical | exact null forced by branch | unforced event/edge action satisfies frozen residual and Jacobian continuity |
| B-R3 temporal consistency | high | no order or tangent proof | small-step tangent and step-halving order pass |
| B-R4 representative selected-grid test | high | grid-4 synthetic only | selected-48 physical short-prefix discriminator passes |
| B-R5 same-physics reference | high | common-mode/internal anchors only | tight independent integration of identical action agrees on state and QoIs |
| B-R6 adaptive electron operator | high | one frozen step, no rejection | deterministic step-doubling accept/reject contract passes |
| B-R7 self-collision positivity | high | instantaneous action only | compatible bounded self-collision step and invariant tests pass |
| B-R8 split composition | high | absent | expansion/electron/self split has convergence and raw history |
| B-R9 full regression | high | not run on published head | fmt/check/clippy/debug+release/full focused gates green |
| B-R10 endpoint authority | critical | not attempted/earned | prospectively frozen collision-on endpoint and cold-wall gates pass |
| B-R11 endpoint performance | deferred | segment speedup only | same-physics endpoint wall/memory improves without correctness loss |

## 6. Constructive recommendation

Do not discard the reconstruction. Preserve the non-negative coefficient and capacity-bracketed edge ideas, but treat the current branch as a research source, not a merge source. Repair correctness in this order:

```text
root certificate
  -> unforced detailed balance
  -> multi-state action reconstruction
  -> temporal consistency and independent energy derivative
  -> isolated adaptive electron step
  -> selected-grid same-physics short prefix
  -> self-collision positivity
  -> split composition
  -> full regression
  -> endpoint authority
  -> endpoint performance
```

The detailed implementation contract is in:

- `docs/audit/RUST_ODE_RECONSTRUCTION_CODEX_PR_DAG_2026-08-24.md`
- `.codex/plans/rust_ode_reconstruction_pr_dag_20260824.json`

Those files are normative for low-cost Codex execution. Symbol anchors are authoritative; current-head line ranges are advisory and must be re-resolved after each stacked PR.

## 7. Final third-party disposition

```text
RUST_EDGE_COEFFICIENT_IDEA            SURVIVES
CURRENT_EDGE_ROOT_SOLVER              FAIL
CURRENT_EXACT_EQUILIBRIUM_CLAIM       FAIL
CURRENT_TIME_INTEGRATOR_CLAIM         NOT ESTABLISHED
CURRENT_ELECTRON_SUBSTEP              IMPLEMENTED PROTOTYPE
CURRENT_SELF_COLLISION_COMPOSITION    ABSENT
CURRENT_ENDPOINT/PROMOTION AUTHORITY  NONE
RECOMMENDED NEXT PR                   R1_EDGE_ROOT_CERTIFICATE
```
