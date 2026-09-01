# BD622 D-078 — Analytic-Jacobian Research Contract

Date: 2026-09-01  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Canonical parent: `ae3f6776bd6fc5bc84bca72d251dc0db1bba7da5`  
Canonical parent tree: `45592a70165439947af3918f2ed5d7c9baa7c70d`  
VigilODE reference commit/tree: `8d0c79184e09efb5bdadc24a6315c60a71a44264` / `acd94364cf69f19d782619fc6c75554cb0754208`  
Decision authority: D-077  
Status at freeze: **PRE-OUTPUT RESEARCH CONTRACT; NO RABBIT TRAJECTORY AUTHORITY**

## 1. Purpose and exact DAG position

D-071 closed the old order-60/domain-holdout instrument on its retained
measurement without evaluating a scientific predicate. D-077 subsequently
allowed a bounded numerical-equivalence lane while keeping
`G-F10-INDEPENDENT-FLRW` at `FAIL`.

This D-078 slice performs, in order:

1. a mathematics/physics research loop on the private comparator's logit-state
   push-forward;
2. a first coding research loop implementing and testing only the exact
   piecewise differential map;
3. a second coding research loop adapting VigilODE's *verification pattern*,
   not its solver implementation, into a true directional-derivative
   certificate;
4. structural toy plots and adversarial mutations that cannot be confused with
   a RABBIT trajectory, checkpoint, endpoint, or gate result.

The next scientific node remains the separately sealed stalled-phase
instrument. It is not executed in this slice.

## 2. Reconstructed contract

### 2.1 Physical contract

The private comparator evolves three flavour-pair occupation spectra on its own
quadrature grid, plus photon temperature and elapsed time. Let

\[
 f_i\in(0,1),\qquad z_i=\log\frac{f_i}{1-f_i},\qquad
 D_i=\frac{\partial f_i}{\partial z_i}=f_i(1-f_i).
\]

The occupation-space collision and expansion law is written abstractly as

\[
 \frac{d f}{dN}=F(f,T_\gamma,N),
\]

where the actual `F` remains the existing private comparator RHS and no
collision coefficient, event, grid, tolerance, state meaning, or failure
semantics may change here.

The retained trajectory core uses

\[
 D_i^{\mathrm{eff}}=\max(D_i,\delta),\qquad \delta=10^{-12},
 \qquad
 \frac{dz_i}{dN}=\frac{F_i}{D_i^{\mathrm{eff}}}.
\]

All quantities in the differential map are dimensionless except the inherited
RHS scale; division by `D_eff` changes no physical dimensions.

### 2.2 Code contract

Load-bearing existing paths:

- `src/rabbit/decoupling/_independent_noqke.py`: private independent
  occupation-space collision operator and transforms;
- `scripts/audit/_trajectory_core.py`: logit-state assembly and SciPy BDF
  integration;
- `scripts/audit/d069_independent_trajectory_r4.py`: retained r4 driver and
  base/domain-holdout phases;
- `.agent-harness/runs/run-20260729-f10-d069-trajectory-r4/`: retained failure
  evidence.

D-078 may add research-only modules and focused tests. It may not edit any path
above, call `solve_ivp`, or generate a physical trajectory.

### 2.3 Observable contract

This slice may report only:

- algebraic equality between the exact piecewise push-forward and finite
  differences on manufactured functions;
- directional-derivative residuals versus perturbation size;
- chain-factor amplification and floor-branch location;
- whether designated sign and scaling mutations are detected;
- whether a certificate fails closed on domain or branch crossings.

It may not report completion time, trajectory agreement, endpoint agreement,
physical-prefix agreement, `N_eff`, a gate grade, or independent scientific
corroboration.

## 3. Exact transformed Jacobian and JVP

Write `E=diag(D_eff)`, `D=diag(D_i)`, and let `J_f=partial F/partial f`.
Away from the floor kink `D_i=delta`,

\[
 J_z
 =E^{-1}J_fD
 -\operatorname{diag}\!\left(
   \frac{F_i\,\partial_{z_i}D_i^{\mathrm{eff}}}
        {(D_i^{\mathrm{eff}})^2}
 \right),
\]

with

\[
 \partial_{z_i}D_i^{\mathrm{eff}}=
 \begin{cases}
 D_i(1-2f_i), & D_i>\delta,\\
 0, & D_i<\delta.
 \end{cases}
\]

For a logit-space direction `v`,

\[
 J_zv=E^{-1}J_f(Dv)
 -F\odot\frac{\partial_zD^{\mathrm{eff}}\odot v}
                  {(D^{\mathrm{eff}})^2}.
\]

For an auxiliary variable `a` that does not redefine the logit chart,

\[
 \partial_a z'=E^{-1}\partial_aF.
\]

At an exact equilibrium `F=0` with no active floor,

\[
 J_z=D^{-1}J_fD,
\]

so `J_z` is similar to `J_f` and has the same eigenvalues. Therefore the logit
chart alone is not an admissible explanation of the r4 stall. The potentially
load-bearing combination is nonzero tail disequilibrium, small `D_eff`, the
piecewise floor, and a numerically estimated Jacobian.

At `D_i=delta` the map is continuous but not differentiable. D-078 must return a
typed uncertified outcome rather than silently selecting either derivative.

## 4. Mathematics/physics hypothesis register

| ID | Hypothesis | Predicted signature | Kill test | Disposition at freeze |
|---|---|---|---|---|
| H1 | Numerical BDF Jacobian cancellation is amplified in tail logit rows. | Directional residual loses its clean epsilon window as `1/D_eff` grows. | Exact analytic push-forward remains wrong against central differences away from branch crossings. | TOP |
| H2 | The floor kink, rather than tail amplification alone, corrupts selected columns. | Perturbations that cross `D=delta` are non-smooth while same-branch perturbations remain differentiable. | Same-branch and crossing ladders are indistinguishable. | TOP |
| H3 | The physical collision Jacobian itself is singular or nearly singular in the stalled epoch. | Analytic occupation-space linearization remains ill-conditioned before the logit map. | Later occupation-space singular-value probe is benign. | HOLD; requires physical states |
| H4 | Solver bookkeeping/event handling causes the apparent stall. | Step/rejection/Newton histories implicate event or controller logic rather than linearization. | Instrumented stalled-phase run shows linear-solve domination. | HOLD; retained r4 lacks counters |
| H5 | Generic collision-RHS cost alone explains the result. | Base and holdout have comparable progress per evaluation and differ mainly in per-call wall. | Retained base completes while the holdout enters a long progress plateau. | DOWNRANKED |
| H6 | Larger hardware or wall budget is sufficient. | Completion projection fits the existing method after modest scaling. | D-071's order-of-magnitude evaluation-count miss. | REJECTED by D-071/D-077 |

No D-078 toy result may promote H1 or H2 to a diagnosis of the retained physical
run. It can only validate the mechanism and the tests needed to discriminate it.

## 5. Method selection

### Selected first implementation candidate

**An independently derived analytic occupation Jacobian or occupation JVP,
followed by the exact piecewise logit push-forward above, supplied to the
existing SciPy BDF path.**

Reason:

- it preserves the existing RHS, state, BDF integrator, tolerances, event,
  endpoint, observables, wall budget, and failure semantics;
- SciPy BDF already accepts a supplied Jacobian, so the eventual integration
  delta can remain narrow;
- the derivation can be mapped event-by-event to the private comparator rather
  than copied from production Rust/JAX code;
- the local derivative can be certified before any trajectory output.

### Deferred alternatives

- **AD Jacobian:** deferred because the current private comparator is
  branch-heavy NumPy with interpolation and quadrature. A differentiable
  parallel rewrite would create a second RHS and a new provenance/independence
  problem before it supplied evidence.
- **JFNK/Newton--Krylov:** retained as a fallback after an analytic JVP exists.
  It additionally requires nonlinear forcing, preconditioning, iteration caps,
  and true-residual certification, so it is not the minimal first change.
- **finite-difference factor reset/cap:** diagnostic control only, exactly as
  D-077 requires.

## 6. External research provenance

The literature loop used the following primary numerical-analysis anchors:

- Knoll and Keyes, *Jacobian-free Newton--Krylov methods: a survey of
  approaches and applications*, J. Comput. Phys. 193 (2004) 357--397;
- Pernice and Walker, *NITSOL: a Newton iterative solver for nonlinear
  systems*, SIAM J. Sci. Comput. 19 (1998) 302--318;
- Brown and Saad, *Hybrid Krylov methods for nonlinear systems of equations*,
  SIAM J. Sci. Stat. Comput. 11 (1990) 450--481;
- Curtis, Powell and Reid, sparse-Jacobian estimation, J. Inst. Math. Appl. 13
  (1974) 117--119;
- Hindmarsh et al., *SUNDIALS*, ACM TOMS 31 (2005) 363--396.

These sources motivate supplied derivatives, true residuals, and explicit
preconditioning/cost accounting. They do not diagnose RABBIT's retained stall.

A Wolfram Language check independently differentiated `F(sigmoid(z))/D(z)`,
located the `delta=1e-12` floor transitions near
`z=+-27.6310211159`, and confirmed divergent `1/D` tail amplification. A toy
relaxation model reproduced the diagonal correction term and the derivative
change on the clamped branch. This is symbolic mechanism verification only.

## 7. VigilODE-derived second-loop pattern

VigilODE is used as a reference implementation for verification discipline,
not as donor physics or a donor solver. At the pinned reference tree:

- `crates/rodas5p-krylov/src/common.rs` computes the true residual from the
  original operator;
- `gmres_givens.rs` treats the projected residual only as a candidate trigger;
- each candidate is rechecked against the true residual;
- projected candidates that fail the true check are counted and rejected;
- nonfinite values, singular triangular factors, breakdown, and exhausted
  Arnoldi limits are explicit failures;
- the final report is marked converged only after a final true-residual check.

D-078 translates that pattern into a directional-derivative certificate:

1. a finite-difference residual is evidence, not an automatic success;
2. at least two consecutive same-branch epsilon samples must satisfy the frozen
   bound;
3. domain exits, floor-branch crossings, nonfinite values, and insufficient
   samples are typed non-success outcomes;
4. input arrays are never mutated;
5. sign and scale mutants must be killed.

No VigilODE Krylov implementation, coefficient, tolerance, or solver policy is
copied into RABBIT.

## 8. Frozen research acceptance criteria

The focused research suite must establish all of the following:

1. matrix push-forward agrees with central differences away from a floor kink;
2. JVP output equals the matrix push-forward applied to the same direction;
3. exact equilibrium yields the similarity transform;
4. the clamped branch uses zero denominator derivative but retains the raw
   occupation chain in `J_f D`;
5. exact floor-kink states are refused;
6. auxiliary columns divide by `D_eff` exactly once;
7. the directional certificate accepts a correct manufactured JVP;
8. sign and 1% scaling mutants are refused;
9. branch crossing, strict-domain exit, and nonfinite output are not success;
10. state and direction inputs remain byte-identical.

Plots must show:

- raw and floored chain amplification versus logit;
- central-difference residual ladders at interior, tail, near-floor, and
  clamped states;
- correct versus sign- and scale-mutated JVP residual ladders.

## 9. Explicit prohibitions

This branch must not:

- edit the private collision RHS, trajectory core, r4 driver, retained run, or
  gate registry;
- call the physical comparator integrator;
- change any physical coefficient, state layout, grid, domain, tolerance,
  event, endpoint, observable, wall budget, or failure semantics;
- copy production Rust/JAX derivative code;
- claim that the retained r4 stall has been diagnosed or repaired;
- claim endpoint, independent-validation, performance, publication, F-11,
  Type-I, QKE, inference, or public-production authority.

## 10. Ordered execution and stop rule

1. Commit this contract and failing focused tests.
2. Implement the exact push-forward and typed certificate in research-only
   modules.
3. Run focused tests and the structural probe.
4. Read the generated plots and perform PHYS-MATH, PHYS-MATH-CODE, and CRAG
   audits.
5. Apply only bounded corrections exposed by those audits.
6. Open a Draft PR with no gate movement.
7. Stop before any physical-state Jacobian, stalled-phase discriminator, or
   trajectory run.

A P0 formula, branch, domain, or false-success defect returns the DAG to this
contract rather than being patched around in the future instrument.
