# BD-629 — D-080E production-order Jacobian construction profile

## Status and claim ceiling

D-080E is a same-physics construction-cost study after the D-080D static
Jacobian admission.  It may establish that direction-independent data can be
reused at a fixed state and may measure single-host wall time and memory.  It
must not claim an ODE-solver improvement, a completed trajectory, an endpoint,
`N_eff`, a portable speedup, movement of `G-F10-INDEPENDENT-FLRW`, or
release/publication readiness.

## Frozen physical contract

The comparator uses natural units

\[
\hbar=c=k_B=1,
\]

and packed state ordering

\[
Y=(c_e,c_\mu,c_\tau,T_\gamma,t_{\rm elapsed}).
\]

For a fixed state, fixed \(T_{\rm cm}\), fixed \(T_\gamma\), fixed quadrature
configuration, and one unchanged discrete support/matrix-correction branch, an
event contribution has the form

\[
{\cal I}_e=W_e\,{\cal M}_e\,C_e(u).
\]

For a spectral cloglog direction \(v_c\), the kinematic measure \(W_e\), weak
matrix element \({\cal M}_e\), and support predicate are independent of the
tangent direction.  Therefore

\[
D_c{\cal I}_e[v_c]
 =W_e{\cal M}_e
 \sum_{i=1}^4\frac{\partial C_e}{\partial u_i}\,D_cu_i[v_c].
\]

The map is linear:

\[
D_cF[a v+b w]=aD_cF[v]+bD_cF[w].
\]

No cache is valid across a state, temperature, grid, quadrature, electron-mass,
comparator-blob, or discrete-branch change.

## Frozen two-arm comparison

### Arm S — serial frozen D-079

Each direction calls the admitted D-079 oracle independently.  The primal
collision action, kinematics, matrix elements, interpolation basis and
thermodynamics may be recomputed.

### Arm P — prepared fixed-state reuse

The primal action and thermodynamics are computed once.  The following exact
objects may be retained and reused:

- two-body kinematic batches and their support masks;
- weak matrix-element arrays;
- mapped-Legendre basis values at repeated interpolation coordinates;
- base collision action, Hubble rate, chart factor and thermodynamic rows.

The Pauli tangent, moving spectral values, modal tangent contractions and RHS
chain rule remain direction dependent and are evaluated for every direction.
The first implementation is serial over the direction axis; it is not yet a
true vectorized event tape.

Both arms must use the same equations, collision catalogue, quadrature,
roundoff policy, state, directions, output ordering and tolerances.  No reduced
collision model or approximate Jacobian update is admitted in D-080E.

## Cache-lifetime invariant

The matrix cache uses the identity of a retained kinematic batch as part of its
key.  Consequently matrix caching requires kinematic retention.  The policy

```text
cache_kinematics = false
cache_matrices   = true
```

is invalid and must fail closed.

The first GREEN candidate did not enforce this dependency.  Python reused a
deallocated batch object ID, allowing an unrelated matrix array to be returned.
The cache-ablation test produced an order-unity spectral discrepancy and killed
the candidate.  The repaired public facade rejects the unsafe policy and keeps
the original candidate as an auditable core implementation.

## Required evidence

1. Genuine RED run with the implementation module absent.
2. Exact agreement with the frozen serial D-079 oracle.
3. Exact order-8 matrix agreement with D-080D.
4. Fixed-state cache hit growth with no new kinematic/matrix/basis misses after
   preparation.
5. Cache-policy ablations that preserve the mathematical result.
6. Fail-closed nested/concurrent patch and unsafe-policy tests.
7. Exact retained order-60 state with source SHA-256.
8. Two-arm wall-time samples for 1, 2, 4 and 8 directions at order 8.
9. Retained order-60 marginal cost, cache memory, and linear 180-column
   projection, explicitly labelled as a projection rather than an executed
   matrix build.
10. Differentiated first-law residual and exact comparator/source identities.
11. Diagnostic plots read adversarially; no manuscript-figure claim.

## Routing rule

`EXPLICIT_FULL_BUILD_MEASUREMENT_ADMISSIBLE` opens only a direct production
matrix-build measurement.  It requires same-physics residual below
\(5\times10^{-11}\), projected prepared time no larger than 900 s, estimated
cache no larger than 2 GiB, and projected speedup at least 1.5 on the measured
host.

Otherwise, a correct cache with tolerable memory is classified
`REUSE_VALID_TRUE_BATCHING_OR_MATRIX_FREE_DECISION_REQUIRED`.  Neither result
opens a solver callback by itself.

## Methodological context

The receipt structure follows the VIGILODE principle of frozen identities and
same-input two-arm comparison.  SciSpace triage found direct-Jacobian neutrino
decoupling work and structured/fast kinetic collision methods as methodological
precedents.  They motivate measuring Jacobian construction cost but do not
validate this private discretization, cache key, or support semantics.
