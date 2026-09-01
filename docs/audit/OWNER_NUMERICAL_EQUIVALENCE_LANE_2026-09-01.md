# Owner Authorization — Sealed Numerical-Equivalence Lane

Date: 2026-09-01
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`
Tracking issue: #3
Base branch at authorization: `plan/ode-r1-r3-remediation-v2-20260824`
Base commit: `f6e42c29f3186926b65544d9147e9b6eab248b4e`

## 1. Decision

The owner authorizes a limited numerical-equivalence lane for the private,
structurally independent full-spectral FLRW comparator governed by
`G-F10-INDEPENDENT-FLRW`.

This authorization changes only the class of numerical method that may satisfy
the reopen conditions attached to the D-071 instrument closure. It does not
change the gate grade, reinterpret the retained r4 measurement, or authorize a
scientific endpoint claim.

The old instrument remains closed on its current measurement. The underlying
physical question remains open.

## 2. Authorized candidate methods

Exactly the following replacement classes are admissible:

1. a supplied analytic Jacobian derived from the private comparator equations;
2. an AD Jacobian of the private comparator RHS, with derivative provenance;
3. a matrix-free JVP/Newton–Krylov solve using the identical comparator RHS;
4. a hybrid analytic/AD block Jacobian with prospectively declared omitted
   blocks and independently bounded omission error.

A finite-difference-factor reset or cap may be used only as a causal diagnostic
control. It is not, by itself, a qualifying endpoint method.

No production Rust or frozen JAX derivative implementation may be copied into
the independent comparator. Shared numerical libraries are allowed; shared
project derivative code is not.

## 3. Frozen physical-equivalence boundary

A qualifying candidate must preserve all of the following:

- collision catalogue, coefficients, multiplicities, and physical model;
- constants, units, state layout, species ordering, and coordinate transforms;
- radial and angular grids, quadrature rules, and domain;
- RHS meaning, accepted-state semantics, and failure preservation;
- `rtol`, vector or scalar `atol`, step bounds, and attempt budget;
- event function, direction, refinement, terminal target, and wall budget;
- checkpoint state, moments, conservation and exchange ledgers, tail records,
  and endpoint observables;
- private-comparator status, no-QKE scope, and existing claim ceiling.

Changing any item above creates a new scientific method and leaves this lane.

## 4. Prospectively sealed contract

Before the first qualifying output byte, the implementation run must freeze:

### 4.1 Identity

- source commit and tree;
- complete changed-path manifest and byte digests;
- environment, dependency, compiler, BLAS, Python, and SciPy identities;
- exact commands, working directory, CPU/thread settings, and wall budget.

### 4.2 Equation-to-derivative map

For every Jacobian or JVP block, record:

- governing RHS terms and state indices;
- analytic, AD, or matrix-free construction route;
- sparsity/block assumptions;
- intentionally omitted derivative terms, if any;
- an a priori omission bound and the observable it can affect.

### 4.3 Local equivalence tests

Use preregistered states covering:

- equilibrium and near-equilibrium states;
- asymmetric flavour states;
- the collision-dominated creep epoch that stalled r4;
- the transition out of the collision-dominated epoch;
- a late-time weak-collision state.

At those states test:

- directional derivatives against independently chosen centered differences;
- analytic versus AD or complex-step witnesses where available;
- scaled column and block residuals;
- Newton correction and linear residual agreement;
- conservation, exchange, and symmetry identities of the linearized operator.

The test tolerances must be frozen before candidate output is inspected.

### 4.4 Stalled-phase discriminator

The first executable scientific discriminator must cover the phase that stalled
in the retained order-60/domain-holdout run. It must record at least:

- accepted and rejected steps;
- BDF order and step-size histories;
- Newton iterations and failures;
- Jacobian/JVP evaluations and linear setups;
- raw state and conservation/exchange residuals;
- progress per evaluation and per wall second;
- an end-to-end completion projection with explicit uncertainty.

The candidate is killed before a full endpoint run unless the sealed projection
fits inside the existing wall budget with a prospectively declared margin.
Hardware-only or budget-only rescue is not admissible.

### 4.5 Trajectory and endpoint equivalence

A surviving candidate must then satisfy the frozen trajectory-prefix,
checkpoint, spectral-moment, tail, conservation, exchange, event, endpoint, and
holdout predicates. No band may be refitted after output.

### 4.6 Adversarial sensitivity

The evidence package must kill designated mutations in:

- collision or exchange sign;
- normalization or multiplicity;
- state-index permutation;
- omitted Jacobian block;
- stale Jacobian reuse;
- derivative scaling;
- event direction or terminal interpretation.

A mutation that cannot change a relevant predicate is not an admissible test.

## 5. Explicit prohibitions

This authorization does not permit:

- tolerance widening or output-led threshold selection;
- clipping, projection, state repair, hidden floors, or failed-state removal;
- grid, domain, catalogue, coefficient, endpoint, or event changes;
- production Rust/JAX derivative reuse inside the independent comparator;
- larger wall budgets, faster hardware, or more cores as the reopening method;
- generic performance claims without same-physics/same-tolerance/same-output
  evidence;
- public dispatch, Type-I/F-11 work, QKE, inference, or publication claims;
- describing same-host deterministic replay as independent corroboration.

## 6. Required repository integration

The next repository-local execution must atomically synchronize this owner
record into all controlling machine-readable surfaces, including:

- the frozen decision ledger;
- the decision log;
- `GATE_REGISTRY.json` reopen conditions and status basis;
- the generated status board;
- the context index and generated context pack;
- any validation-ledger row required by the harness.

The integration must leave `G-F10-INDEPENDENT-FLRW` at FAIL. Authorization alone
is not evidence and cannot move a gate.

## 7. Ordered implementation DAG

1. Merge this owner-authorization record without gate movement.
2. Synchronize all registry/context surfaces in one harness-validated change.
3. Freeze the numerical-equivalence contract before implementation or output.
4. Implement exactly one derivative/linear-solve candidate.
5. Run and adjudicate the stalled-phase discriminator.
6. Run the full endpoint/holdout programme only if the discriminator passes.
7. Perform independent review and a separate gate-reconsideration decision.

## 8. Claim boundary

What this record establishes: `SPECIFIED` authorization for a bounded method
class and its validation contract.

What it does not establish: `IMPLEMENTED` or `VALIDATED` numerical equivalence,
a completed endpoint, independent scientific corroboration, or any gate pass.

## 9. Cost line

```text
added_lines: 196
removed_lines: 0
net_lines: 196
files_touched: 1
token_use_exact: UNAVAILABLE
token_use_basis: the GitHub connector exposes no exact token counter
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: governance deadlock only
blocker_movement_ratio: 0.25
validation_strengthened: specified, not yet executed
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```
