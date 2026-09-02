# BD-630 — D-080E production-order Jacobian profile result

## Final classification

```text
FIXED_STATE_REUSE_VALIDATED
EXPLICIT_FULL_BUILD_MEASUREMENT_ADMISSIBLE
SOLVER_CALLBACK_NOT_YET_ADMITTED
G-F10-INDEPENDENT-FLRW = FAIL (unchanged)
```

D-080E validates a research-only, same-physics fixed-state reuse path for the
D-079 spectral RHS derivative and measures its construction cost at order 8 and
on the exact retained order-60 state.  It does not execute a 180-column
production matrix build and does not call an ODE solver.

## Frozen authority

```text
D-080D predecessor:
985fdacc63ce09b93eef4fc77d87ade12fdff284

frozen comparator blob:
de44feee0aa484abe26976c7dc34c579643005b5

D-079 collision-JVP blob:
591a64702c58a2de265fb88636f186e2d1b7e019

D-079 RHS-JVP blob:
6bcff2bc5627c0af0ad4df61c908d09e62ffaba5

D-080C T_gamma RHS-column blob:
c18feacbd57c9519af14504027b7d465758eb1ef

D-080D static-Jacobian blob:
c577fefaf7a83443a7531e59a283c4f15e8815e1

validated reuse facade blob:
0913b3be5ad66af27ae7115deb603c88556cd6b4

preserved first candidate core blob:
915196691eb166f5624d413a46d314b32faacfe6

profile generator blob:
27ec71adbca78741bf844a46ec5e0d8646cab1be
```

The principal profile workflow was run at
`be2f7c7ba4ad50d9895008cfb3079280b1ed4d8d`.  A second exact readback run at
`3c214bb024e685bc5cdff45381972665a5a347b4` independently regenerated the
profile, emitted the complete machine receipt into the Actions log and passed
all internal SHA-256 checks.

## Mathematical contract

The comparator uses natural units

\[
\hbar=c=k_B=1
\]

and the packed state

\[
Y=(c_e,c_\mu,c_\tau,T_\gamma,t_{\rm elapsed}).
\]

For a fixed physical state, fixed temperatures, fixed grid and quadrature, and
one unchanged discrete support/matrix-correction branch, each event contribution
can be written

\[
{\cal I}_e=W_e{\cal M}_eC_e(u).
\]

The event kinematics, support, quadrature measure and weak matrix element do not
depend on the spectral tangent direction.  Thus

\[
D_c{\cal I}_e[v]
 =W_e{\cal M}_e
 \sum_i\frac{\partial C_e}{\partial u_i}D_cu_i[v],
\]

and

\[
D_cF[a v+b w]=aD_cF[v]+bD_cF[w].
\]

A stateless Wolfram Language evaluation returned exact zero for:

- fixed-factor cache residual;
- tangent-direction linearity residual;
- basis-column assembly residual;
- stable Pauli-JVP residual.

This is formula-level corroboration rather than a repository-native Wolfram
replay.

## Same-physics fairness contract

Both arms used identical:

- equations and collision catalogue;
- state and tangent directions;
- quadrature orders and momentum grid;
- matrix-roundoff policy;
- packed output ordering;
- tolerances and physical constants.

No collision approximation, tolerance change or reduced output was introduced.

Arm S independently invoked the frozen D-079 path for each direction.  Arm P
computed the primal state once and retained direction-independent kinematics,
weak matrices and mapped-Legendre basis values.  The Pauli tangent, spectral
interpolation and modal contractions were still evaluated separately for each
direction.

## RED–GREEN and cache-lifetime defect

The first workflow failed as intended because the implementation module was
absent.  The first GREEN candidate then failed a cache-policy ablation.  Its
matrix cache used retained kinematic-batch identity as part of the key, while a
policy allowed matrix caching after disabling kinematic retention.  Python could
reuse the object ID of a deallocated batch, allowing an unrelated cached matrix
to be returned.

The final public facade fails closed when

```text
cache_kinematics = false
cache_matrices   = true
```

is requested.  The unsafe first candidate is preserved as a separate auditable
core rather than silently rewritten.  All final cache-policy ablations return
the same mathematical derivative.

## Order-8 results

All reported residuals against the frozen serial D-079 oracle were exactly zero
at binary64 output precision.

| Directions | Serial [s] | Prepared total [s] | Total speedup | Marginal speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.2857848135 | 0.1558157530 | 1.83412 | 3.38749 |
| 2 | 0.5663067810 | 0.2402639425 | 2.35702 | 3.35484 |
| 4 | 1.1302117330 | 0.4102165685 | 2.75516 | 3.33327 |
| 8 | 2.2630200260 | 0.7490324475 | 3.02126 | 3.34065 |

Two-direction cache ablations, all with zero same-physics residual:

| Policy | Preparation + application [s] |
|---|---:|
| full reuse | 0.2416998180 |
| no matrix cache | 0.2610792950 |
| no modal-basis cache | 0.4185229230 |
| no fixed-state caches | 0.4552620220 |

At order 8, mapped-Legendre basis reuse is the largest isolated benefit among
the tested cache families.  Matrix reuse is beneficial but less dominant.

## Exact retained order-60 result

```text
fixture SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380

order:          60
spectral size: 180
state size:    182
y_max:          30
T_cm:            8.497022351366393 MeV
T_gamma:         8.497129004420698 MeV
```

Measured using two deterministic spectral directions:

| Quantity | Result |
|---|---:|
| serial time per direction | 16.702108392 s |
| prepared-state construction | 1.813036950 s |
| prepared marginal time per direction | 2.2777218345 s |
| measured marginal speedup | 7.33281305 |
| analytic T_gamma column | 13.208925162 s |
| same-physics residual | 0.0 |
| differentiated first-law residual | 9.172736325446216e-17 |
| T_gamma base-RHS reconstruction residual | 0.0 |

Linear projections to a 180-column spectral build are:

| Route | Projected time |
|---|---:|
| frozen serial D-079 | 3019.588435722 s |
| prepared fixed-state reuse | 425.011892322 s |
| projected speedup | 7.10471516 |

These are projections from two measured directions, not an executed
\(182\times182\) matrix build.

## Memory and cache profile

```text
estimated NumPy cache payload: 294,865,920 bytes
explicit 182x182 matrix:           264,992 bytes
process RSS before:             122,630,144 bytes
process RSS after preparation:  423,743,488 bytes
process RSS after application:  426,467,328 bytes
```

The matrix itself is negligible; retained event data dominate memory.
The cache contained 180 kinematic batches, 180 self-matrix entries, 900
electron-matrix entries and 181 modal-basis entries.  It recorded:

```text
kinematic hits/misses: 1,140 / 180
matrix hits/misses:   10,620 / 1,080
modal hits/misses:    37,635 / 181
modal evictions:       0
```

The subsequent analytic T_gamma-column evaluation added 600 kinematic hits,
5,940 matrix hits and 9,327 modal-basis hits with zero new misses.  This is
direct evidence that preparation covered the fixed-state object families used
by both the spectral and thermal derivative paths.

The NumPy payload estimate excludes Python object, dictionary and key overhead;
process RSS is therefore the broader memory witness.

## PHYS-MATH audit

Verdict: `PASS_WITH_FIXED_STATE_AND_BRANCH_SCOPE`.

Passed:

- spectral directional linearity;
- stable Pauli JVP;
- fixed-factor product rule;
- unchanged natural-unit dimensions;
- exact order-8 and retained order-60 agreement;
- differentiated first-law closure.

Open P1:

- a support or matrix-correction branch change invalidates ordinary reuse and
  requires a fresh prepared state;
- performance has been measured at one physical state and one CI host;
- the projected 180-column cost has not been directly executed.

## PHYS-MATH-CODE audit

Verdict:
`REUSE_CORRECT; DIRECT_PRODUCTION_MATRIX_MEASUREMENT_OPEN; SOLVER_ADMISSION_CLOSED`.

Passed:

- genuine RED implementation-absent failure;
- exact source/blob pinning;
- serial-oracle and D-080D order-8 matrix equivalence;
- retained order-60 equivalence;
- cache-policy ablations;
- nested/concurrent patch rejection;
- unsafe cache-lifetime policy rejection;
- deterministic plot and receipt checksum audit.

Open P1:

- private-helper monkeypatching is not a thread-safe production API;
- the prepared object does not yet deep-freeze every state/grid array against
  external mutation;
- kinematic and matrix caches have no explicit byte cap;
- the direction loop remains serial rather than a true batched event tape;
- timing samples are too sparse for a portable benchmark claim.

## Plot-driven CRAG verdict

The generated plots show monotonic amortisation with direction count, order-8
cache-ablation separation and a large order-60 projected construction reduction.
The numerical arrays and SHA-256 manifest passed.  The session-local image
runtime failed during independent ZIP rendering, so manuscript-scale typography
and print legibility are not certified.

- Correctness: survives; all numerical outputs match the frozen derivative.
- Retrieval: consistent with direct-Jacobian neutrino-decoupling work and
  structured kinetic-kernel reuse, but the literature does not validate this
  private implementation.
- Augmentation: order-8 direction scaling and one exact retained order-60 state
  pass; multi-state and full-180-column evidence remain absent.
- Generation: the profile predicts that a direct full build is feasible within
  the declared construction budget, but this prediction must be tested.

Surviving claim: fixed-state same-physics reuse is correct and materially lowers
marginal derivative cost on the tested host.

Narrowed claim: approximately sevenfold production-order construction speedup is
only a one-host projection from two directions.

Rejected claim: portable performance, BDF speedup, stalled-prefix completion or
endpoint improvement.

## Route decision and next node

All predeclared conditions for

```text
EXPLICIT_FULL_BUILD_MEASUREMENT_ADMISSIBLE
```

were met:

- same-physics residual below `5e-11`;
- projected prepared time below 900 s;
- estimated cache below 2 GiB;
- projected speedup above 1.5.

The next node is **D-080F: directly execute and audit the production-order
\(182\times182\) Jacobian build** on the exact retained state.  Before the build,
D-080F must deep-freeze or fingerprint the prepared arrays and impose explicit
cache-memory accounting.  It must then verify selected basis columns against
the frozen serial oracle, mixed directions against the independent JVP, the
analytic T_gamma column, the exact-zero elapsed column, first-law closure, wall
time and peak memory.

Only D-080F success may open a paired D-081A `solve_ivp(BDF, jac=...)`
experiment.  No solver claim is opened by D-080E.

## Workflow and artifacts

```text
GREEN correctness workflow:
33588987799  SUCCESS

principal two-arm profile:
33589488214  SUCCESS
artifact 9831265718
artifact SHA-256:
12fe8fb4f68929924b181729b78bfa6ba19cf3573863a1a134cc973b2217db40

exact readback profile:
33590046742  SUCCESS
artifact 9831436608
artifact SHA-256:
03d3c79c512f973db38e25351508fdd6d21275d136d6966ec5f2b50e7df53dce
```
