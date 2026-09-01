# BD622 D-078 — Analytic-Jacobian Mathematics/Physics and Dual Coding Research Result

Date: 2026-09-01  
Canonical parent: `ae3f6776bd6fc5bc84bca72d251dc0db1bba7da5`  
Canonical parent tree: `45592a70165439947af3918f2ed5d7c9baa7c70d`  
Research branch: `research/d078-analytic-jacobian-research-loop-20260901`  
Contract: `docs/audit/BD622_D078_ANALYTIC_JACOBIAN_RESEARCH_CONTRACT_2026-09-01.md`  
VigilODE reference: `8d0c79184e09efb5bdadc24a6315c60a71a44264` / tree `acd94364cf69f19d782619fc6c75554cb0754208`  
Classification: **METHOD MECHANISM VALIDATED; PHYSICAL BLOCKER UNRESOLVED; NO GATE MOVEMENT**

## 1. Executive result

D-078 completed the requested ordered loop:

1. **mathematics/physics research:** derive the exact piecewise Jacobian and
   JVP of the private comparator's logit-state push-forward, classify the floor
   kink, recover the equilibrium similarity transform, and rank plausible stall
   mechanisms;
2. **coding research loop 1:** implement that map as a research-only module and
   specify it first with focused tests;
3. **coding research loop 2:** adapt VigilODE's projected-candidate/true-residual
   discipline into a fail-closed directional-derivative certificate without
   copying its solver;
4. **plot/mutation loop:** generate manufactured chain-amplification and
   residual-ladder plots, kill sign and scaling mutants, and narrow the claim
   where binary64 finite differences cease to be a competent witness.

The strongest result is structural, not physical: the exact transformed
Jacobian is now unambiguous and locally testable, and an uncertified finite-
difference sample can no longer masquerade as a valid derivative. The retained
D-069/r4 stall is **not** diagnosed or repaired by this branch because no
private-comparator physical state, trajectory, endpoint, or BDF run was
executed.

`G-F10-INDEPENDENT-FLRW` remains `FAIL`. D-071's old-instrument/current-
measurement closure is unchanged.

## 2. Current DAG state

| Node | State after D-078 | Evidence / limitation |
|---|---|---|
| D-071 retained instrument measurement | CLOSED ON CURRENT MEASUREMENT | unchanged historical evidence; no predicate evaluated |
| D-077 numerical-equivalence authority | COMPLETE | merged before D-078; gate unchanged |
| Exact logit push-forward derivation | VALIDATED AS ALGEBRA | symbolic derivation, matrix/JVP tests, equilibrium and floor-branch tests |
| True directional certificate | VALIDATED ON MANUFACTURED FUNCTIONS | correct JVP accepted; sign and 1% scale mutants rejected; branch/domain outcomes typed |
| Binary64 tail finite-difference witness | VALID ONLY IN A BOUNDED WINDOW | interior and moderate tail certify; near-floor and clamped tail do not |
| Occupation-space private-comparator Jacobian/JVP | NOT IMPLEMENTED | next equation-to-code task |
| Static physical-state derivative suite | BLOCKED | requires independent occupation-space derivative implementation and frozen states |
| Stalled-phase discriminator | BLOCKED | must follow static derivative admission; not run here |
| Full endpoint/holdout programme | BLOCKED | requires discriminator pass inside existing wall budget |
| Gate reconsideration | BLOCKED | requires complete evidence package and later adjudication |

Allowed next work is one bounded equation-to-code implementation of the
occupation-space derivative plus static physical-state certification. A
trajectory, solver swap, tolerance change, preconditioner campaign, or endpoint
run remains premature.

## 3. CLEAN_CONTEXT and reconstructed contracts

### 3.1 Load-bearing current mismatch

The retained base GL48/Y24 phase completed, whereas the density-matched
order-60/Y30 domain holdout entered a long post-drop progress plateau and
exhausted the frozen wall budget. The retained report contains no step-size,
rejection, BDF-order, Newton, error-norm, or linear-solve history. Consequently
several mechanisms remain observationally equivalent in the old evidence.

### 3.2 Physical and variable contract

For each occupation component,

\[
 0<f_i<1,\qquad z_i=\log\frac{f_i}{1-f_i},\qquad
 D_i=f_i(1-f_i).
\]

Let the existing private-comparator occupation law be

\[
 \frac{df}{dN}=F(f,T_\gamma,N),
\]

and let the retained trajectory map use

\[
 E_i=D_i^{\rm eff}=\max(D_i,\delta),\qquad
 \delta=10^{-12},\qquad
 \frac{dz_i}{dN}=G_i=\frac{F_i}{E_i}.
\]

`f`, `z`, `D`, and `E` are dimensionless. `G` and `F` have the inherited
per-e-fold rate dimension, so every Jacobian entry is per e-fold. No natural-
unit substitution or physical coefficient enters this derivation.

### 3.3 Code contract

The authoritative physical RHS remains in
`src/rabbit/decoupling/_independent_noqke.py`; state assembly and BDF evolution
remain in `scripts/audit/_trajectory_core.py`; the retained r4 driver remains
`scripts/audit/d069_independent_trajectory_r4.py`.

D-078 edits none of those paths. Its additions are research-only:

- `scripts/audit/_d078_logit_linearization.py`;
- `scripts/audit/_d078_tangent_certificate.py`;
- `scripts/audit/d078_research_probe.py`;
- `tests/test_d078_jacobian_research.py`;
- the pre-output contract, generated structural artifacts, and this result.

### 3.4 Observable contract

D-078 observables are derivative residuals, branch classifications, chain
amplification, mutation discrimination, and algebraic limits. They are not
cosmological observables and carry no endpoint, `N_eff`, likelihood, or
independent-validation authority.

## 4. Mathematics/physics research result

### 4.1 Exact piecewise transformed Jacobian

Let

\[
 D=\operatorname{diag}(D_i),\qquad
 E=\operatorname{diag}(E_i),\qquad
 J_f=\frac{\partial F}{\partial f}.
\]

Away from the floor kink `D_i=delta`, direct differentiation gives

\[
 \boxed{
 J_z
 =E^{-1}J_fD
 -\operatorname{diag}\!\left[
  \frac{F_i\,\partial_{z_i}E_i}{E_i^2}
 \right]
 }
\]

with

\[
 \partial_{z_i}E_i=
 \begin{cases}
 D_i(1-2f_i),&D_i>\delta,\\
 0,&D_i<\delta.
 \end{cases}
\]

For a logit direction `v`, the exact JVP contract is

\[
 \boxed{
 J_zv=E^{-1}J_f(Dv)
 -F\odot\frac{(\partial_zE)\odot v}{E^2}
 }.
\]

For an auxiliary variable `a` that does not redefine the chart or floor,

\[
 \partial_aG=E^{-1}\partial_aF.
\]

The input-side chain in `J_f D` is the **raw** logistic derivative `D`, even
when the output row is clamped by `E=delta`. Replacing the input-side `D` by
`E` would differentiate a different vector field and is explicitly tested as a
forbidden implementation drift.

### 4.2 Equilibrium and known limits

At exact equilibrium `F=0` with no active floor,

\[
 J_z=D^{-1}J_fD.
\]

This is a similarity transform. It preserves the eigenvalues of `J_f`, so the
logit chart alone cannot explain the retained stall. A viable diagnosis must
involve at least one non-equilibrium or numerical ingredient: nonzero tail
`F`, small chain factors, a floor transition, finite-difference cancellation,
or BDF/controller/linear-solve behaviour.

At `D_i=delta`, `G` is continuous but not differentiable. There is no unique
classical Jacobian at the kink. The research implementation returns a typed
`LinearizationKinkError`; it does not silently pick the left or right branch.

### 4.3 Symbolic and high-precision verification

Wolfram Language independently differentiated `F(sigmoid(z))/D(z)`, recovered
the diagonal correction, and located the two `delta=10^-12` transitions at
approximately

\[
 z_\delta=\pm 27.6310211159.
\]

A 50-digit manufactured relaxation witness then separated formula correctness
from binary64 metrology:

| State | Branch | Exact derivative | 50-digit centered derivative at `epsilon=10^-12` |
|---|---|---:|---:|
| `z=0` | raw | `-1` | `-1.0000000000000000000000001667` |
| `z=20` | raw | `-3.8813215632783222e8` | `-3.8813215632783222e8` |
| `z=z_delta-0.011` | raw | `-7.9124822301950371e11` | `-7.9124822301950371e11` |
| `z=30` | clamped | `-9.3576229688384233e-2` | `-9.3576229688384233e-2` |

Thus the binary64 near-floor and clamped-tail failures below do not falsify the
analytic map. They falsify the stronger claim that an ordinary binary64
central-difference ladder is an adequate derivative oracle everywhere in the
tail.

### 4.4 Hypothesis triage

| Rank | Hypothesis | D-078 disposition |
|---|---|---|
| Top | Tail chain amplification makes numerical BDF Jacobian columns cancellation-sensitive. | Mechanism survives; no physical diagnosis yet. |
| Top | Floor-branch crossings invalidate otherwise plausible finite-difference columns. | Mechanism demonstrated and typed. |
| Hold | Occupation-space collision Jacobian is itself nearly singular at the retained stalled epoch. | Unmeasured; requires static physical states. |
| Hold | BDF controller/event/linear-solve bookkeeping, rather than derivative quality, dominates the stall. | Unmeasured because old evidence lacks counters. |
| Downranked | Generic RHS cost alone explains the stall. | Does not explain base completion versus holdout plateau by itself. |
| Rejected | Faster hardware, larger budget, or a finite-difference-factor reset is a qualifying reopening method. | Excluded by D-071/D-077. |

### 4.5 Selected numerical-equivalence method

The first admissible implementation candidate remains:

> independently derive the private comparator's occupation-space analytic
> Jacobian or JVP, apply the exact piecewise push-forward above, and supply it
> to the existing SciPy BDF path without changing the RHS, state, grid,
> tolerance, event, endpoint, observable, wall budget, or failure semantics.

AD is deferred because the current branch-heavy NumPy/interpolation/quadrature
RHS would require a second differentiable RHS implementation before it produced
evidence. JFNK remains a later fallback because it additionally requires
nonlinear forcing, preconditioning, iteration policy, and true-residual
certification.

## 5. Coding research loop 1 — exact push-forward

`_d078_logit_linearization.py` implements:

- strict occupation-domain and finite-value checks;
- raw, effective, active-floor, and kink masks;
- exact matrix push-forward;
- exact JVP push-forward with an explicit `J_f(Dv)` input contract;
- auxiliary columns divided by `E` exactly once;
- immutable result arrays;
- typed refusal at the floor kink.

Focused tests establish:

1. matrix agreement with centered differences away from the floor;
2. exact equilibrium similarity and eigenvalue recovery;
3. correct raw-input/effective-output chain handling on a clamped row;
4. typed kink refusal;
5. matrix/JVP identity;
6. auxiliary-column normalization.

No physical comparator import, event contraction, or integration path is
present in this module.

## 6. Coding research loop 2 — VigilODE-adapted verification

VigilODE was not copied. The useful reference pattern was narrower:

- a projected or cheap residual can nominate a candidate but cannot certify it;
- the original operator's true residual must be recomputed;
- nonfinite values, breakdown, singular factors, and exhausted limits remain
  explicit failures;
- iterations, matvecs, residuals, and preconditioner actions are report data,
  not hidden implementation details.

D-078 translates that discipline into a derivative certificate:

- centered finite differences are independent witnesses, not the derivative
  implementation;
- one favourable `epsilon` cannot certify a JVP;
- a minimum valid-sample count and consecutive same-branch pass count are
  frozen inputs;
- branch crossings, domain exits, and nonfinite evaluations are typed sample
  outcomes;
- sign and scaling mutations must fail;
- callbacks receive copies, so failed probes are transactional with respect to
  caller state.

This is an application of VigilODE's verification semantics to RABBIT's
blocker. It is not a transplant of GMRES, Rodas coefficients, preconditioners,
or tolerance policy.

## 7. Execution and repair chronology

Four bounded workflow attempts were preserved in GitHub Actions:

| Run | Outcome | Interpretation |
|---|---|---|
| `33513579360` | FAIL, `11 passed / 1 failed` | test input requested one valid sample but retained the default two-consecutive-pass contract; test made explicit, production rule not weakened |
| `33513873110` | scientific checks PASS, evidence commit step FAIL | Git porcelain collapsed five untracked files to their directory; path guard changed to enumerate files, no result changed |
| `33514057720` | SUCCESS | 12 focused tests, probe, receipt checks, mutation kills, generated evidence commit |
| `33514573717` | SUCCESS | deterministic rerun, 12 focused tests, identical evidence, one-day plot artifact for hostile readback |

The final successful artifact bundle has SHA-256
`6ecc512dfb108ac30257d936328fb488fe5e394df944a3190ec26bd46c6aa793`.
The two failed attempts are orchestration evidence, not scientific failures,
and neither was relabelled as success.

## 8. Plot-generated result and adversarial reading

The generated PNGs and exact plotted series are retained under
`docs/audit/artifacts/d078/`.

| Case | `z` | `1/E` | Result | Best normalized residual | Key reading |
|---|---:|---:|---|---:|---|
| interior | `0` | `4` | CERTIFIED | `6.551204e-12` | broad second-order descent followed by a roundoff floor; 11 consecutive passing samples |
| moderate tail | `20` | `4.851652e8` | CERTIFIED | `1.189747e-5` | only a narrow `epsilon~10^-2--10^-3` window passes; smaller steps worsen sharply from cancellation |
| near floor | `27.6200` | `9.891499e11` | UNRESOLVED | `7.785922e-4` | two large steps cross the floor branch; same-branch binary64 samples never enter the frozen band |
| clamped tail | `30` | `1e12` | UNRESOLVED | `1.137558e-3` | no floor crossing, yet occupation subtraction is too cancellation-dominated for the binary64 oracle |

Mutation curves separate by orders of magnitude:

| Direction | Result | Best normalized residual |
|---|---|---:|
| correct | CERTIFIED | `1.538698e-10` |
| sign mutation | UNRESOLVED | `1.998335` |
| 1% scale mutation | UNRESOLVED | `8.25e-3` |

The chain-amplification plot is symmetric about `z=0`; raw amplification grows
exponentially in both tails, while `1/E` plateaus at `10^12` beyond the two
floor transitions. This confirms that the floor bounds the transformed RHS
amplification but introduces a nonsmooth derivative boundary; it does not make
ordinary finite differences uniformly reliable.

The session's local raster inspection runtime returned an internal client
error after the workflow artifact was downloaded. Therefore the adversarial
scientific reading above is based on the exact plotted arrays, retained PNG
bytes, plot-generation source, machine receipt, and independent 50-digit
Wolfram witnesses rather than on an unrecorded visual impression. This limits
only typography/legibility review, not the mathematical reading of the plotted
curves.

## 9. PHYS-MATH AUDIT

| Check | Verdict | Basis / limitation |
|---|---|---|
| Definitions and chart | PASS | strict `0<f<1`; `z`, raw `D`, effective `E`, and floor branch separated |
| Sign | PASS | quotient rule gives the negative diagonal correction; sign mutant killed |
| Normalization | PASS | raw chain on `J_fD`; effective chain only on output denominator; auxiliary division once |
| Units/dimensions | PASS | all chart factors dimensionless; Jacobian retains per-e-fold rate dimension |
| Equilibrium limit | PASS | exact similarity transform and eigenvalue equality |
| Interior finite difference | PASS | matrix and JVP tests; manufactured residual ladder |
| Floor-active branch | PASS WITH LIMIT | derivative formula is exact away from kink; binary64 finite-difference oracle is weak in deep tail |
| Floor kink | PASS AS REFUSAL | no classical derivative; typed error rather than silent branch choice |
| Positivity/domain | PASS | strict occupation domain; no clipping or projection introduced |
| Special counterexample | PASS | clamped row proves raw input chain cannot be replaced by the floor |
| Physical-stall implication | NOT ESTABLISHED | no physical state or integrator measurement in D-078 |

### PHYS-MATH ranked findings

- **P0: none** in the frozen research scope.
- **P1:** the effective-floor map is nondifferentiable on `D=delta`; a future
  physical state touching the kink cannot receive an ordinary supplied
  Jacobian without an explicit generalized/one-sided contract.
- **P1:** binary64 centered differences are not a universal oracle near or below
  the floor; physical admission must use occupation-space analytic identities
  as primary evidence and branch-safe finite differences only as witnesses.
- **P2:** auxiliary `T_gamma`, time, and any explicit `N` derivatives remain to
  be mapped from the actual occupation RHS.
- **P3:** no independent second implementation of the full event derivative is
  present yet.

## 10. PHYS-MATH-CODE AUDIT

| Check | Verdict | Basis / limitation |
|---|---|---|
| Equation-to-code map | PASS for transform | every algebraic factor appears once and is separately named/tested |
| Actual physical code path | DELIBERATELY DISCONNECTED | research-only modules do not alter or call the private comparator |
| Branch reality | PASS | raw, clamped, and exact-kink states are distinct outcomes |
| Transactionality | PASS | hostile callback test cannot mutate caller state/direction |
| False-success resistance | PASS in focused scope | consecutive true residuals required; sign and scale mutants killed |
| Numerical sensitivity | PASS AS CHARACTERISATION | epsilon-window collapse is retained, not tuned away |
| Regression sufficiency | PASS for mechanism, not solver | 12 tests cover the frozen transform/certificate contract |
| SciPy BDF integration | NOT IMPLEMENTED | no `jac=` supplied; no trajectory authority |
| Sparsity/preconditioner | NOT DESIGNED | must follow measured physical event graph, not precede it |
| Performance claim | FORBIDDEN | no same-output stalled-phase or endpoint measurement |

### PHYS-MATH-CODE ranked findings

- **P0: none** in the implemented research-only path.
- **P1:** the occupation-space event Jacobian/JVP is absent, so the current work
  cannot change the retained BDF instrument.
- **P1:** no frozen physical state from the stalled collision epoch has yet been
  admitted to the derivative suite.
- **P1:** the old r4 evidence lacks Newton/order/step/rejection/linear-solve
  counters, so derivative quality and controller failure remain confounded.
- **P2:** no static conservation, exchange, flavour symmetry, or equilibrium
  null checks have been applied to an actual analytic collision derivative.
- **P2:** no sparsity pattern, factorization cost, or Jacobian reuse policy is
  justified yet.
- **P3:** the one-day artifact is review transport only and must not become a
  permanent authority surface.

## 11. Plot-based CRAG adversarial verdict

### C — Correctness

**Survives, bounded.** The exact transform passes matrix, JVP, equilibrium,
clamped-branch, auxiliary-column, and mutation tests. High-precision witnesses
explain the two binary64 non-certifications without changing the formula.

### R — Retrieval and external consistency

**Survives.** The supplied-derivative and true-residual strategy is consistent
with the primary stiff/nonlinear-solver literature recorded in the contract.
VigilODE independently exemplifies candidate-trigger versus true-residual
certification and explicit failure reporting. Neither source is treated as
evidence about RABBIT's physical stall.

### A — Augmentation across regimes

**Claim narrowed.** The certificate works in the interior and a moderate tail,
but ordinary binary64 centered differences are not adequate near or below the
floor. Future physical testing must stratify directions by floor branch, record
active/kink masks, and add analytic conservation/exchange identities plus a
higher-precision or occupation-space witness where binary64 loses separation.

### G — Generated prediction

The next physical probe predicts one of three discriminating outcomes:

1. an admitted analytic occupation JVP sharply lowers Newton/step failures in
   the stalled phase, supporting H1/H2;
2. it passes local tests but does not change stalled-phase work, shifting weight
   to controller/event or physical conditioning H3/H4;
3. it fails static conservation/exchange or branch tests, killing the candidate
   before any trajectory.

### Final claim classification

- **Surviving:** exact piecewise logit Jacobian/JVP formula; typed kink; true
  directional certificate; mutation sensitivity; analytic-Jacobian-first
  candidate selection.
- **Narrowed:** binary64 finite differences are useful only in a measured,
  same-branch epsilon window.
- **Rejected:** logit coordinates alone diagnose the stall; the floor cures
  conditioning; one favourable epsilon certifies a derivative; a factor reset,
  hardware gain, or larger budget reopens the gate.

## 12. Applied fixes, deferred fixes, and forbidden changes

### Applied in D-078

1. exact piecewise push-forward with raw/effective-chain separation;
2. typed floor-kink refusal;
3. transactional multi-epsilon true-residual certificate;
4. branch/domain/nonfinite sample classification;
5. sign and 1% scaling mutation controls;
6. deterministic structural plots and machine receipt;
7. two orchestration defects preserved and repaired without altering scientific
   thresholds or output.

### Deferred until the next sealed contract

1. independently derive the private occupation-space event Jacobian/JVP;
2. map `T_gamma` and auxiliary columns;
3. freeze physical equilibrium, asymmetric, stalled-epoch, transition, and
   late-time states;
4. test linearized conservation, exchange, symmetry, and null identities;
5. infer sparsity/block structure from that measured derivative graph;
6. wire a passing Jacobian into a separate stalled-phase BDF instrument with
   full Newton/order/step/rejection telemetry.

### Forbidden now

- changing the RHS, collision catalogue, state chart, floor, grid, tolerance,
  event, endpoint, observable, wall budget, or failure semantics;
- copying production Rust/JAX derivative code;
- implementing AD by rewriting the RHS before the analytic candidate is tested;
- introducing JFNK/preconditioning before a certified JVP and measured block
  graph exist;
- running the full endpoint before the stalled-phase discriminator passes;
- gate, precision, performance, public-runtime, F-11, Type-I, QKE, inference,
  or publication claims.

## 13. Updated DAG and recommended next step

### Newly completed

- derivative method selection under D-077;
- exact logit push-forward derivation and implementation;
- manufactured matrix/JVP/equilibrium/floor tests;
- VigilODE-derived true-residual certificate;
- plot/mutation adversarial mechanism audit.

### Newly opened

- a bounded **occupation-space analytic event-Jacobian/JVP derivation** task.

### Still blocked

- physical-state derivative admission;
- stalled-phase BDF discriminator;
- endpoint/holdout;
- evidence package and gate reconsideration.

### Recommended next step — exactly one

Freeze a new pre-output implementation contract that maps every term of the
private comparator's occupation-space collision and expansion RHS to an
independently derived analytic Jacobian/JVP, then execute it only on a
prospectively fixed static state set. The admission criteria must include:

- directional residuals in occupation space before logit push-forward;
- equilibrium null and asymmetric-flavour states;
- linearized number/energy exchange and coupled first-law identities;
- floor-active/kink masks after push-forward;
- event-by-event sign, normalization, multiplicity, state-index, omitted-block,
  and scaling mutants;
- no call to `solve_ivp` and no trajectory output.

Only after that static package passes should a separate instrument supply the
Jacobian to the unchanged BDF path and measure the actual stalled phase.

## 14. Short reason

The blocker cannot be responsibly attacked by changing the integrator first:
D-078 shows that the derivative map is exact but its binary64 witness becomes
regime-dependent, so the next information-gaining move is to certify the real
occupation-space derivative before allowing it to influence one physical BDF
step.

## 15. Cost and authority line

```text
runtime_behavior_changed: no
physics_behavior_changed: no
private_comparator_or_trajectory_path_changed: no
physical_output_generated: no
gate_movement: no
blocker_movement: method selected; transform and verifier closed; physical stall unresolved
validation_strengthened: yes, with focused tests, true-residual ladder, plots, high-precision witness, and mutations
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```
