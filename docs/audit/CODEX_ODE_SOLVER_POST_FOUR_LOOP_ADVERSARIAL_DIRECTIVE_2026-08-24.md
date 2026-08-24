# Codex Development Directive: RABBIT ODE Solver After the Four-Loop Audit

Date: 2026-08-24 (Asia/Seoul)  
Document status: **ADVERSARIAL DEVELOPMENT DIRECTIVE**  
Source evidence branch: `external-audit/ode-four-loop-complete-20260823`  
Source evidence head: `9c05a65eaa5fbb86bec5d131c0d300689217de16`  
Audited code head: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Disposition: **ACCEPT AS A NEGATIVE RESEARCH RESULT / REJECT AS A SOLVER REMEDIATION**

This document is for Codex implementation planning. It does not reopen a gate,
amend the canonical claim ledger, authorize public production, or validate a new
solver. It was produced by adversarially re-adjudicating the retained source,
raw trajectory, ledgers, reports, and exact-head comparison. No fresh numerical
execution belongs to this document.

`complete` in the source branch name means that the four-loop evidence packet is
closed. It does **not** mean that the ODE solver, collision discretization,
endpoint physics, or production path is complete.

## 0. Mandatory read order

Before changing code, read:

1. `AGENTS.md`
2. `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
3. `bbn_codex_anti_drift_cost_effective_policy.md`
4. `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/FINAL_EXTERNAL_AUDIT_REPORT.md`
5. `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/results/A-IF-REVIEW2.json`
6. this document

The source evidence branch is an evidence archive, not the next production base.
Do not merge it wholesale into `main`. Start future executable work from the
current canonical development base, then verify whether the two relevant source
blobs still match the audited versions:

- `src/rabbit/collisions/dynamic_collision_driver.py` blob
  `0164b9262244e07770b74afa7db57b27811b8e5f`
- `src/rabbit/collisions/dynamic_collision_core.py` blob
  `49126d52cff4a150e017a72d98635afbfbd9b7cd`

If either blob differs, stop and localize the semantic changes before reusing any
conclusion below.

## 1. Executive verdict

The four-loop work succeeded at one important task: it prevented a false fix from
being admitted. The C1 candidate did not change the collision-on numerical
trajectory. It changed admission and postprocessing so that the same trajectory
was rejected rather than clipped and reported. Exact HEAD and C1 retained the
same configuration, solver statistics, and raw stored occupations; C1 therefore
acted as a falsification instrument, not a numerical remedy.

The strongest new result is narrowly stated:

> For the frozen `n_q=24`, collision-on, direct-occupation Radau trajectory with
> `rtol=1e-8`, scalar `atol=1e-10`, `max_step=0.5`, and
> `T_gamma_stop=0.01 MeV`, accepted stored high-momentum occupations leave the
> Pauli interval. The first rejection occurs near the initial time and the
> largest negative excursion is about `5.49e-15`. This makes strict pointwise
> raw-state admission incompatible with that coordinate/error-control pair.

This does **not** establish any of the following:

- that the continuum neutrino distribution becomes negative;
- that logit coordinates are already validated;
- that Patankar, exponential, projection, or entropy methods will close the
  endpoint blocker;
- that scalar tolerance is the sole cause;
- that the collision operator is boundary-inward after discretization;
- that the observed tail error is negligible for every weak-rate or final-state
  quantity of interest;
- that C2 or C3 should start immediately.

The correct next action is causal discrimination, not another broad candidate.

## 2. What the audit established reliably

### 2.1 Surviving positive conclusions

- C1 was correctly rejected and not committed to the retained code tree.
- Raw negative occupations, failed runs, rejected code, exact-head comparators,
  and environment limits were preserved rather than hidden.
- The real collision-on trajectory, not only synthetic fakes, falsified C1.
- The collisionless control did not show the same stored-state violation.
- Exact HEAD passed the pre-existing focused module while C1 failed the three
  real collision-on tests.
- The selected five gold failures reproduced on exact HEAD and therefore were
  not caused by C1.
- The scientific and public-promotion gates remain closed.

### 2.2 Hard ceiling

The audit did not run an independent high-precision collision-on reference, did
not separate collision-discretization error from time-integration error, did not
validate a positivity-preserving method, and did not execute a new endpoint or
D-071 trajectory. Its negative result is strong; its positive remedy authority
is absent.

## 3. Adversarial findings that must guide development

### A-01. C1 repaired admission semantics, not the ODE trajectory — critical

C1 and exact HEAD produced the same raw trajectory. Therefore the experiment
proved that current postprocessing launders a raw-state violation, but it did not
move the underlying integrator or collision dynamics. Any report that calls C1 a
solver fix is false.

Development consequence: do not revive C1 with a different epsilon, a narrower
exception, a final-only check, or a tolerance chosen after seeing the tail.

### A-02. Causal identification is incomplete — critical

At least four explanations remain live:

1. scalar absolute error control permits direct-`f` tail overshoot;
2. the discretized/interpolated collision field is not inward on the Pauli
   boundary;
3. gain-loss cancellation, finite-difference Jacobian probes, or floating-point
   accumulation destroys the tail sign;
4. the raw pointwise violation is tiny for some moments but not certified for the
   actual weak-rate quantities of interest.

The final report acknowledges this residual uncertainty but still places a
positivity-coordinate implementation next. That ordering is premature. A large
logit or Patankar candidate can fail for a different reason while leaving the
causal question unanswered.

### A-03. The accepted-state, evaluator-probe, and reported-state domains are mixed — critical

The current direct-occupation driver:

- clips state occupations before every RHS evaluation;
- clips the resampled evaluator input;
- clips the final accepted state before moments are formed;
- records `max_clip_excursion`, but only rejects a missed endpoint or a sufficiently
  negative **clipped** spectral moment;
- floors Hubble with `max(H, 1e-100)` inside the RHS.

Thus a raw accepted state can leave the physical domain, be mapped back into the
domain for collision evaluation, and still generate an endpoint observable. The
existing `max_clip_excursion < 1e-6` test is a bulk-collapse guard, not a Pauli
admission proof.

Future contracts must distinguish:

- **physical accepted state**: may support scientific observables;
- **internal solver/Jacobian probe**: may be rejected or handled by a typed
  recoverable mechanism, but must not silently redefine physics;
- **reported endpoint state**: must be formed only from an admitted accepted
  state and must retain the raw state and terminal certificate.

Do not solve this by adding another result wrapper. Consolidate the distinction in
existing solver/result surfaces and delete ambiguous fields when implementation
authority is earned.

### A-04. Existing logit code is not an admissible shortcut — critical

`dynamic_collision_core.flrw_dynamic_collision_rhs` already maps an inverse-logit
variable to `0<f<1`, but it also:

- divides by `f(1-f)`;
- freezes the tail through `fnn_floor` and `np.where(..., 0.0)`;
- retains an Hubble floor;
- has not been validated as the same comoving collision-on endpoint path.

Reusing it unchanged would replace clipping with tail slaving. It would not answer
the four-loop finding.

There is also a sign-convention hazard. Choose exactly one convention and name it
correctly:

\[
 u = \log\frac{f}{1-f},\qquad
 f=\frac{1}{1+e^{-u}},\qquad
 \frac{du}{dN}=\frac{C}{Hf(1-f)},
\]

or

\[
 \ell = \log\frac{1-f}{f}=-u,\qquad
 f=\frac{1}{1+e^{\ell}},\qquad
 \frac{d\ell}{dN}=-\frac{C}{Hf(1-f)}.
\]

The current `q + A` variable is the second convention even where prose calls it
“logit”. A future patch must not mix these signs.

### A-05. “Energy conservation by construction” is an algebraic discrete invariant — high

Feeding the plasma the negative of the same discrete neutrino `df/dN` moment makes
the two discrete exchange terms cancel. This is useful and should be preserved.
It does not independently validate:

- the collision kernel;
- the comoving-to-thermal interpolation;
- the quadrature tail;
- weak-rate accuracy;
- the physical continuum first law beyond the implemented discrete system.

Keep the test, but classify it as a discrete exchange-identity test rather than an
external or independent physics validation.

### A-06. Test names and assertions overstate authority — high

The current focused tests contain several semantic mismatches:

- “recovers analytic `N_eff=3` exactly” is tested with a broad absolute band,
  while another test pins a discretized value near `2.9934`;
- `max_clip_excursion < 1e-6` is described as proof that clipping did not repair
  an unphysical state, although the four-loop raw trajectory contains many
  negative tail entries and still satisfies that threshold;
- collision-on `3.00 <= N_eff <= 3.15` is a smoke envelope, not external
  validation;
- the exchange-identity test is not an independent collision-physics oracle.

Do not open a separate cosmetic PR for these labels. Correct them in the first
accepted executable PR or in a net-deflation consolidation.

### A-07. The active and inactive solver debts are mixed — high

The NumPy Rodas event state machine has real false-success defects, and Rust has
failure-taxonomy/counter/certificate gaps. They do not all deserve equal priority.
The current frozen collision-on discriminator uses SciPy Radau, Rust AOT is the
active retained implementation target, and JAX is frozen.

C2 may start only if NumPy Rodas remains necessary as an active reference for the
chosen next implementation. Otherwise freeze it with an explicit unsupported
contract rather than spend another loop repairing a non-authoritative lane.

### A-08. Green subset tests cannot offset the failed physical discriminator — high

The measured production-not-slow selection passed, but it excluded the slow
collision-on test that falsified C1. The full and gold runs were red, with only a
subset paired against exact HEAD. Future work must use this order:

1. static/local causal discriminator;
2. focused real collision-on short-prefix falsifier;
3. complete focused module;
4. paired exact-head comparison;
5. production-not-slow;
6. gold/full only after the changed physical path passes.

Do not spend another hour-scale full-suite run on a candidate already falsified at
steps 1–3.

### A-09. Cost accounting was ambiguous — medium

The implementation discussion treated the production-only net change (`+39`) as
meeting a `<=40` cap, while the independent review treated the total production
plus test change (`+185`) as violating the same cap. Future contracts must state
the denominator before editing. Count production, tests, docs, and committed
artifacts, consistent with the repository cost policy. Never redefine the cap
post hoc.

### A-10. Evidence preservation became repository inflation — medium

The audit branch preserves extensive raw logs, materialized temporary trees,
archives, harness copies, and rejected postimages. That is acceptable as a sealed
external-audit branch, but it is not an executable-development pattern and must
not be merged wholesale into the canonical branch.

Future loops may retain one compact raw result and one concise adjudication. Do
not commit temporary worktrees, repeated materializations, or duplicated harness
packages. If a large durable archive is required, keep one compressed artifact and
one checksum manifest on an evidence-only branch or release asset.

## 4. Physics and numerical conventions for the next work unit

This ODE surface uses internal natural units `c = hbar = k_B = 1`.

- `N = ln a`: dimensionless independent variable.
- `q` and `Y`: dimensionless momentum coordinates.
- `f`: dimensionless fermion occupation, physical interval `0 <= f <= 1`.
- `T_gamma`, `H`, and collision rate `C=df/dt`: MeV.
- `df/dN = C/H`: dimensionless per e-fold.
- energy densities and per-e-fold exchange sources: MeV^4.
- sign convention: `Q > 0` transfers energy into neutrinos.

The coupled first-law signs are

\[
\frac{d\rho_\nu}{dN}=-4\rho_\nu+\frac{Q}{H},\qquad
\frac{d\rho_{em}}{dN}=-3(\rho_{em}+P_{em})-\frac{Q}{H}.
\]

For a node-local gain-loss representation,

\[
 C_i(f)=(1-f_i)G_i(f_{\neg i})-f_iL_i(f_{\neg i}),
 \qquad G_i\ge0,\quad L_i\ge0,
\]

implies the Pauli-boundary conditions

\[
 C_i\big|_{f_i=0}=G_i\ge0,\qquad
 C_i\big|_{f_i=1}=-L_i\le0.
\]

If `G_i`, `L_i`, and `H` are frozen over one local interval, the exact update is

\[
 f_i(N+\Delta N)=f_{i,*}+
 \bigl[f_i(N)-f_{i,*}\bigr]e^{-\lambda_i\Delta N},
\]

where

\[
 f_{i,*}=\frac{G_i}{G_i+L_i},\qquad
 \lambda_i=\frac{G_i+L_i}{H}.
\]

This update preserves `[0,1]` when the gain-loss premises hold. It is a diagnostic
and possible algorithmic primitive, not yet a validated production scheme.

## 5. The only authorized next research unit

Name:

`P0_COLLISION_TAIL_INWARDNESS_AND_CAUSAL_DISCRIMINATOR`

### 5.1 Objective

Determine whether the first observed high-q negative tail is caused primarily by:

- a non-inward discrete collision field;
- catastrophic gain-loss cancellation or summation error;
- direct-`f` time-integration/error-scaling overshoot;
- or an unresolved combination.

P0 is read-only with respect to production source. It must not implement logit,
Patankar, projection, C2, C3, a new solver backend, or a full endpoint run.

### 5.2 Frozen case and evidence

Use exactly the retained case:

```text
n_q = 24
collisions = true
method = SciPy Radau
rtol = 1e-8
atol = 1e-10 (scalar baseline)
max_step = 0.5
T_gamma_stop = 0.01 MeV
first implicated bank/node = f_nue[22]
implicated tail = nue nodes 18..23, nux nodes 19..23
```

Extract the raw exact-head trajectory without merging the evidence branch:

```bash
git show \
  external-audit/ode-four-loop-complete-20260823:.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/artifacts/raw_trajectory_exact_head_collision_on.json \
  > /tmp/rabbit_raw_trajectory_exact_head_collision_on.json
sha256sum /tmp/rabbit_raw_trajectory_exact_head_collision_on.json
```

Expected SHA-256:

```text
2da11255f25761e4b7e7ea330eda2c4948013e4d1df00c5e127720a90f9a678e
```

Use the last stored state before the first violation and the first violating
state. Do not reconstruct a D-071 restart from logs.

### 5.3 Required discriminator measurements

#### P0-A. Pauli-boundary inwardness and gain-loss reconstruction

For each implicated node at the last valid physical state:

1. hold all other state entries fixed;
2. evaluate the node collision field with `f_i=0` and `f_i=1`;
3. define `G_i=C_i(0)` and `L_i=-C_i(1)`;
4. require sign-certified `G_i>=0` and `L_i>=0`;
5. test whether the actual field is reconstructed by
   `(1-f_i)G_i-f_iL_i` at the retained `f_i` and at one interior probe;
6. repeat the selected worst node with an independent accumulation/precision
   axis.

Use an error interval based on the difference between ordinary and independent
accumulation, not an epsilon chosen after seeing the sign. If the interval crosses
zero, report `INCONCLUSIVE`; do not round it into an inward result.

The exact 2-to-2 Pauli statistical factor is gain minus loss before quadrature,
but the full interpolated, bank-folded implementation must be tested rather than
assumed to inherit that property.

#### P0-B. Cancellation and stiffness diagnosis

For every selected node report:

```text
q
f_previous
f_first_invalid
G
L
C
(|G|+|L|)/max(|C|, smallest_positive_scale)
(G+L)/H
Delta_N_to_first_invalid
lambda_Delta_N
```

A large cancellation ratio is evidence against a naive `C/[f(1-f)]` logit RHS.
It is not by itself evidence for a specific replacement.

#### P0-C. Frozen-coefficient positivity prediction

Using the measured `G`, `L`, `H`, and the observed accepted-step `Delta_N`, compute
the exact frozen-coefficient gain-loss update above. Compare it with the next raw
accepted state.

Interpretation:

- frozen update physical, raw Radau state negative: integrator/scaling or
  coefficient-variation mechanism remains live;
- frozen update itself nonphysical: the extracted gain-loss premise or collision
  discretization failed;
- disagreement dominated by an uncertified precision interval: stop inconclusive.

Do not treat this local model as an endpoint solver benchmark.

#### P0-D. Raw tail impact on quantities of interest

Without clipping, report the signed contribution of all invalid tail entries to:

- number moment;
- energy moment;
- each available weak-rate or collision-rate quantity that consumes the same
  spectrum;
- final `N_eff` difference between raw signed and clipped postprocessing, labeled
  diagnostic only.

This measurement cannot legalize negative occupations. It decides whether the
future numerical contract must control pointwise `f`, weak-rate QoIs, moments, or
all of them.

### 5.4 P0 output and cost boundary

P0 may produce:

- one compact machine-readable result;
- one concise adjudication;
- at most one focused regression added to an existing test module if the result
  becomes a durable invariant.

P0 may not add a new registry, gate, readiness wrapper, manifest framework,
telemetry class, figure package, or duplicated harness. Temporary probe code
should remain outside the repository unless it becomes the single durable
regression.

Recommended cost ceiling:

```text
production source changed: 0 files
committed probe/test files: <= 1
committed report files: <= 1
net committed lines: <= 160 total
full endpoint runs: 0
full default suite runs: 0
candidate implementations: 0
```

## 6. Decision table after P0

| P0 result | Next admissible route | Forbidden inference |
|---|---|---|
| boundary inwardness fails or `G/L` sign fails | repair collision discretization/interpolation/statistical assembly first; test detailed balance, boundary signs, discrete exchange and required invariants | do not blame the ODE solver or hide the defect with a transform |
| boundary inwardness passes; cancellation is severe | derive a factorized gain-loss/affinity form and a positivity-preserving local update; naive logit with `fnn_floor` remains forbidden | do not divide an already cancelled `C` by tiny `f(1-f)` |
| boundary inwardness passes; frozen update is physical; direct-`f` step overshoots | test one bounded positivity-preserving or physically scaled short-prefix integrator; retained implementation target is Rust | do not claim endpoint movement from a local step |
| pointwise defect has small moment impact but weak-rate impact is unbounded or absent | derive and validate the missing weak-rate/tail certificate before choosing an error norm | do not equate energy convergence with kinetic accuracy |
| precision interval or model decomposition is inconclusive | `STOP_INCONCLUSIVE`; improve only the discriminator | do not start C1, C2, C3, logit, Patankar, or a solver swap |

## 7. Conditional implementation rules

Only after P0 selects a route:

### 7.1 If the collision field is not boundary-inward

The first executable PR must repair the discrete collision field, not the solver.
Required evidence:

- equilibrium detailed-balance residual with an independent accumulation axis;
- `C_i(0)>=0` and `C_i(1)<=0` at representative physical states;
- exact or bounded discrete invariants appropriate to each channel;
- total first-law exchange with the correct electron-pair accounting;
- no input or output clipping used to make the test pass.

### 7.2 If a positivity-preserving integrator is selected

Prefer a gain-loss exponential, Patankar-type, or entropy-compatible update whose
Pauli invariance follows from its algebra. A logit variable is admissible only if:

- gain/loss or affinity is evaluated without catastrophic subtraction;
- no `fnn_floor` freezes the tail;
- zero-flux/equilibrium branches are explicit and continuous;
- temperature and Hubble domains are typed separately from occupation domains;
- the method has an order/consistency test and a raw short-prefix comparator;
- the accepted implementation is placed in the active Rust path rather than
  growing another Python/JAX production twin.

SciPy may remain the temporary number-of-record. JAX remains a frozen local oracle.

### 7.3 If error scaling is selected

A vector or physics-weighted norm must be specified before the run and tied to
explicit pointwise and QoI goals. It may not be selected merely because it removes
the observed negative entries. Pointwise positivity still requires an invariant
mechanism or a proof that every accepted step remains inside the Pauli box.

## 8. Explicit deferrals

Until P0 completes:

- C1 reimplementation: **FORBIDDEN**
- C2 NumPy Rodas state-machine repair: **DEFERRED** unless required by P0
- C3 slow-manifold/D-071 work: **FORBIDDEN**
- generic Jacobian/sparse/Krylov tuning: **FORBIDDEN AS BLOCKER CLAIM**
- JAX/Diffrax forward development: **FORBIDDEN**
- full endpoint or hour-scale sweep: **FORBIDDEN**
- public production, publication readiness, and QKE: **FORBIDDEN**
- tolerance widening, post-hoc epsilon admission, clipping, projection after the
  solve, or returned-value flooring: **FORBIDDEN**

Rust event/failure/counter debt remains real but should be bundled only with an
active retained path and should modify existing result surfaces rather than add
standalone telemetry plumbing.

## 9. Claim-language corrections for future PRs

Use these labels precisely:

- the four-loop packet: `VALIDATED NEGATIVE RESULT`;
- C1 code: `IMPLEMENTED EXPERIMENT / REJECTED / NOT ADMITTED`;
- current direct-`f` endpoint observable: `DIAGNOSTIC, FORMED AFTER CLIPPING`;
- `max_clip_excursion < 1e-6`: `BULK-STABILITY GUARD`, not positivity validation;
- collision-on `N_eff` band: `INTERNAL SMOKE ENVELOPE`;
- same-`df` plasma/neutrino cancellation: `DISCRETE EXCHANGE IDENTITY`;
- logit/Patankar/entropy route: `PROPOSED` until executed and independently
  falsified;
- endpoint, independent-FLRW, scientific, and public-production authority:
  `NOT EARNED`.

## 10. Required closeout record for the next Codex run

Report only:

```text
base_commit:
relevant_source_blob_identity:
production_files_changed:
test_files_changed:
report_files_changed:
added_lines:
deleted_lines:
net_lines_total:
token_use_exact:
token_use_basis:
frozen_case_identity:
boundary_inwardness_verdict:
gain_loss_reconstruction_verdict:
cancellation_diagnosis:
frozen_update_verdict:
raw_tail_qoi_impact:
selected_route:
blocker_movement_ratio:
claim_ceiling:
stop_reason:
```

No additional gate board is needed.

## 11. Ready-to-paste Codex task

```text
Read AGENTS.md,
docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md,
bbn_codex_anti_drift_cost_effective_policy.md,
.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/FINAL_EXTERNAL_AUDIT_REPORT.md,
.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/results/A-IF-REVIEW2.json,
and docs/audit/CODEX_ODE_SOLVER_POST_FOUR_LOOP_ADVERSARIAL_DIRECTIVE_2026-08-24.md.

Execute only P0_COLLISION_TAIL_INWARDNESS_AND_CAUSAL_DISCRIMINATOR.

Do not edit production source. Do not implement logit, Patankar, projection,
C2, C3, a solver swap, a new backend, or a full endpoint run. Extract the exact
retained collision-on raw trajectory by git-show and verify its SHA-256. Use the
last valid state and first invalid state. At the implicated nue/nux tail nodes,
test Pauli-boundary inwardness, reconstruct a gain-loss form, quantify
cancellation, compare the exact frozen-coefficient positivity update with the
next accepted raw state, and report raw-versus-clipped number, energy, weak-rate,
and N_eff effects.

Use an independent accumulation/precision axis for the worst node. If a sign
interval crosses zero, return STOP_INCONCLUSIVE. Do not tune a tolerance after
seeing the result. Preserve raw signed values. Produce at most one compact result,
one concise adjudication, and one durable regression in an existing test module.
No new manifest, gate, telemetry framework, figure, archive copy, or harness
package.

End by selecting exactly one conditional route from the directive's decision
table, or STOP_INCONCLUSIVE. Do not claim endpoint, scientific, public-production,
QKE, Rust-promotion, or D-071 movement.
```

## 12. Final audit disposition

The four-loop programme should be credited for finding and rejecting a false
repair. It should not be credited with selecting the next solver. The most useful
next result is not another endpoint number; it is a sign-certified causal answer
to whether the discrete collision field points inward at the Pauli boundary and
whether the observed tail failure is created by collision assembly, cancellation,
or direct-`f` integration.

Until that answer exists:

```text
C1 = REJECTED
C2 = DEFERRED
C3 = FORBIDDEN
POSITIVITY METHOD = UNSELECTED
G-F10-INDEPENDENT-FLRW = FAIL
G-HARNESS-INTEGRITY = FAIL
SCIENTIFIC/PUBLIC/ENDPOINT PROMOTION = NOT EARNED
```
