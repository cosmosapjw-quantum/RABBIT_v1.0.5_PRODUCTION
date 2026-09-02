# BD633 — D-080F retained production-order full-Jacobian result

**Date:** 2026-09-02  
**Classification:** `EXECUTED_RETAINED_PRODUCTION_ORDER_JACOBIAN`  
**Route decision:** `EXPLICIT_CALLBACK_CANDIDATE`  
**Scientific gate movement:** none

## 1. Scope and exact lineage

D-080F answers one narrow question left open by D-080E:

> Can the exact retained order-60 private-comparator state produce a complete
> fixed-support `182 x 182` static Jacobian inside the prospectively frozen
> time and memory budgets while preserving the admitted D-079/D-080C
> derivatives?

The answer is **yes on the measured CI host and retained state**, subject to
all scope limitations below.

Frozen identities used by the qualifying v2 workflow:

```text
private comparator blob:
de44feee0aa484abe26976c7dc34c579643005b5

D-079 RHS-JVP blob:
6bcff2bc5627c0af0ad4df61c908d09e62ffaba5

D-080C T_gamma RHS-column blob:
c18feacbd57c9519af14504027b7d465758eb1ef

D-080E prepared facade/core blobs:
0913b3be5ad66af27ae7115deb603c88556cd6b4
915196691eb166f5624d413a46d314b32faacfe6

retained-state source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

retained-state SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

The measured evidence commit is

```text
0bd79b3bf4869199824127980b936fbae655ab23
```

with tree

```text
c3c8178fc75d8f2c7bc0be491cddb4926b91303b.
```

## 2. Physical and numerical contract

The packed state is

```text
Y = (c_e[0:60], c_mu[0:60], c_tau[0:60], T_gamma, elapsed_time),
```

and the static Jacobian is

```text
J = [ J_c | J_Tgamma | 0_elapsed ].
```

The comparator uses natural units `hbar=c=k_B=1`.  Spectral coefficients are
dimensionless, `T_gamma` is measured in MeV, and stored elapsed time has units
`MeV^-1`.  Output blocks therefore retain heterogeneous units and are never
combined under one unscaled Euclidean residual.

The retained state has

```text
order                         60
y_max                         30
spectral columns              180
full state size               182
T_cm                          8.497022351366393 MeV
T_gamma                       8.497129004420698 MeV
```

No physical equation, collision event, matrix element, quadrature rule,
support predicate, state chart, tolerance, trajectory endpoint, or wall
budget was changed by D-080F.

## 3. Executed production-order construction

The complete matrix was actually assembled; this is no longer a D-080E linear
projection.

```text
matrix shape                  182 x 182
matrix raw payload            264,992 bytes
preparation time              32.475363648 s
measured full-build time      415.61379038300004 s
D-080E prepared projection    425.011892322 s
actual / projection           0.9778874377193199
D-080E serial projection      3019.588435722 s
serial projection / actual    7.2653711344355605
```

The measured build is below the frozen `900 s` budget.  The `7.27` ratio is a
comparison with the earlier serial **projection**, not a measured BDF or
trajectory speedup.

Matrix digests:

```text
content SHA-256:
714b83107c9302b55cd09a9d2c7d2dc260a5b7d1e9289265e6a844766ed388f8

.npy-file SHA-256:
3594e861cc800bbfa2feedc4ec90f7de1f69e283a450484bdb600345caa0d4ae
```

The elapsed-time input column is exact zero.

## 4. Prepared-state integrity and memory

The sealed prepared state contained 4,002 NumPy arrays.  All were read-only,
the semantic fingerprint was unchanged across the build, and no cache entry
or miss appeared during column application.

```text
fingerprint SHA-256:
ab0a2af7e1ec75e7acdcbb0665742aa218a13a98092306eb2e5bd844c9f440f8

cache-entry delta             0
cache-miss delta              0
cache unique bytes            294,865,920
unique array bytes            294,992,656
```

Measured process memory:

```text
initial RSS/HWM               114,970,624 / 114,970,624 bytes
after seal RSS/HWM            428,658,688 / 428,658,688 bytes
after build RSS/HWM           435,425,280 / 437,039,104 bytes
```

The peak high-water mark is about `417 MiB`, below the frozen `2 GiB` cap.
The dominant memory cost is the retained event/cache substrate, not the dense
matrix itself.

## 5. Preserved failed run and metrology repair

### 5.1 First qualifying attempt

Workflow `33592265710` completed the full matrix construction but stopped
before artifact publication because the legacy dense-direction
forward-relative metric returned

```text
9.098989500537355e-10 > 5e-10.
```

The failure is preserved.  It was not silently rewritten as success.

### 5.2 Independent localization

Workflow `33593370075` rebuilt the same matrix and compared frozen serial
basis columns `0, 59, 60, 119, 120, 179`.  Every selected column agreed
exactly in every output block.  Temperature and elapsed-row dense-direction
residuals were at approximately `10^-15`; the discrepancy was confined to the
spectral action after summing 180 signed column contributions.

This rejected wrong basis columns, cache mutation, cache growth, thermal-row
mismatch, and elapsed-row mismatch as explanations.

### 5.3 Cancellation-aware action metrology

BD632 prospectively replaced only the ill-conditioned dense-action
normalization.  For each native-dimensional block `B`,

```text
R_B = ||(Jv)_B - r_B||
      / max(||(Jv)_B||, ||r_B||, || |J_B| |v| ||, tiny),
```

with the total residual equal to the maximum over spectral, `T_gamma`, and
elapsed-output blocks.

The numerical threshold remained `5e-10`.  Selected basis-column validation
remained unchanged.  For a basis direction `e_j`, the contribution scale is
exactly `|J[:,j]|`, so the amended metric does not weaken column admission.

Because the amendment followed a failed run, two previously unseen holdout
directions were frozen before the v2 rerun.

Maximum action residuals were

```text
original direction 0         6.075961640389264e-14
original direction 1         6.891185357751888e-14
holdout direction 0          6.794647284848686e-14
holdout direction 1          1.0432572279192872e-13
```

The maximum selected serial-column residual was exactly `0.0`.

For transparency, the maximum legacy forward-relative residual across the
four directions was retained as

```text
1.7231101759072105e-9.
```

The contribution-scaled result is a backward-stability statement; it is not
an operator-norm or future Newton-step error bound.

## 6. Original-RHS witnesses

Two mixed spectral/thermal directions were also compared directly with
centered differences of the unchanged packed RHS.

```text
direction 0 best residual     2.2341039938949794e-7
direction 1 best residual     1.3119142426559677e-7
all samples same branch       true
```

The narrow epsilon window does not show a clean monotone second-order region;
large perturbations approach or cross support boundaries and small
perturbations are roundoff/cancellation limited.  D-080F therefore uses the
prospectively frozen same-branch and minimum-residual conditions, not a
post-hoc slope gate.  The recorded non-monotonic ladders remain diagnostic
evidence.

## 7. Wolfram verification

Stateless Wolfram Language checks returned the exact identities

```text
|sum_j J_ij v_j| <= sum_j |J_ij v_j|,
|J| |e_j| = |J[:,j]|,
forward cancellation scaling       eta/delta,
contribution-scaled scaling         eta/(2-delta).
```

Separate construction checks returned

```text
state size at order 60             182
spectral columns                   180
matrix bytes                       264992
det(I-gamma J)-det(I-gamma A)      0
```

These are formula-level corroborations from a stateless plugin evaluation,
not a repository-native Wolfram replay.  Executable values come from the
GitHub Actions receipts.

## 8. Literature context

SciSpace retrieval identifies Froustey, Pitrou and Volpe, JCAP 12 (2020) 015,
DOI `10.1088/1475-7516/2020/12/015`, as the closest direct precedent.  Their
neutrino-decoupling calculation combines full collision physics with direct
computation of the differential-system Jacobian and reports a large
integration acceleration when also using averaged flavour oscillations.
That result motivates testing a direct Jacobian but does not predict RABBIT's
callback cost or validate this discretization.

Akita and Yamaguchi, JCAP 08 (2020) 012, DOI
`10.1088/1475-7516/2020/08/012`, provide an independent precision
momentum-dependent neutrino-decoupling calculation including finite-temperature
QED corrections.  The frozen RABBIT comparator used here does not contain the
same QED scope, so no direct endpoint parity is inferred.

General JFNK literature confirms that matrix-free storage savings do not by
themselves guarantee robustness; useful Newton-Krylov performance normally
depends on adequate, often physics-based, preconditioning.  D-080F therefore
selects the explicit callback as the next candidate without closing a future
matrix-free route.

## 9. PHYS-MATH audit

**Verdict:** `PASS_WITH_FIXED_STATE_FIXED_BRANCH_SCOPE`.

Closed items:

- state and RHS ordering;
- natural-unit dimensions;
- exact passive elapsed-time input column;
- exact selected basis columns;
- four-direction action-backward equivalence, including two holdouts;
- same-branch original-RHS witnesses;
- differentiated first-law and thermal-column authority inherited from
  D-079–D-080C;
- measured time and memory budgets.

Open P1:

1. The ordinary derivative is not defined across a discrete support or
   matrix-correction branch change.
2. The backward-stable action metric does not bound forward error under strong
   cancellation; the paired Newton/BDF experiment is decisive.
3. Only one retained physical state and six serial basis columns were checked
   independently.
4. The retained state is a restart point; it does not preserve the historical
   BDF order/history or the finite-difference-factor ratchet.

Open P2:

- no retained genuine late weak-collision state;
- no cross-host timing or memory replication;
- no finite-temperature-QED extension;
- no independent second full-matrix implementation.

## 10. PHYS-MATH-CODE audit

**Verdict:** `PRODUCTION_ORDER_STATIC_MATRIX_VALIDATED; SOLVER_ADMISSION_OPEN`.

Genuinely fixed:

- D-080E timing projection was replaced by an executed order-60 full build;
- the prepared-state seal now checks read-only arrays, topology/fingerprint,
  cache growth and cache misses;
- complete-cache bytes have a hard cap;
- six flavour-boundary basis columns and four dense directions are audited;
- two unseen holdouts prevent direct refitting to the first failed directions;
- exact matrix bytes and deterministic artifact checksums are recorded.

Remaining P1:

1. The production integration path still calls SciPy BDF without `jac=`.
2. The prepared implementation uses private-helper interception and is not a
   concurrent/thread-safe public runtime API.
3. A fresh matrix costs about seven minutes at one state; actual Jacobian
   refresh frequency may erase any solver benefit.
4. `state_1200` restart resets BDF history and the prior finite-difference
   factor state, so it cannot by itself reproduce the historical multiyear
   creep mechanism.
5. SciPy result counters alone are insufficient; detailed accepted/rejected
   steps, order, step size, Newton failures, matrix builds and true residuals
   require an instrumented BDF subclass or equivalent wrapper.

## 11. Plot-driven CRAG audit

The deterministic package contains:

- construction timing;
- memory accounting;
- legacy versus cancellation-aware action metrology;
- integrity and residual gates;
- original-RHS epsilon ladders.

**Correctness.** Numeric arrays and receipt fields agree with the gate values.
The action-metrology plot must be read with the distinction between legacy
forward-relative and contribution-scaled backward residuals; neither may be
silently substituted for the other.

**Retrieval.** Direct-Jacobian neutrino-decoupling literature supports the
methodological sequence but not the RABBIT numerical result.

**Augmented.** Two new holdout directions survived.  Multi-state and
cross-host replication remain absent.

**Generation.** The plots support opening a bounded explicit-callback
experiment.  They do not predict a BDF speedup because the seven-minute
Jacobian refresh cost is not yet combined with `njev`, `nlu`, rejection or
Newton data.

The local artifact-inspection client returned `ClientError` during an
additional independent extraction attempt.  GitHub Actions nevertheless
verified every generated file against `SHA256SUMS` and checked the matrix
shape, bytes, hash and exact-zero final column before publication.  Visual
single-column/double-column publication legibility remains ungraded.

## 12. Claim boundary

Allowed:

> On the exact retained order-60 fixed-support state, a sealed prepared path
> built the complete `182 x 182` static Jacobian in 415.6 s with about 417 MiB
> peak HWM, exact agreement on six selected serial columns, and a maximum
> four-direction contribution-scaled action residual of `1.04e-13`.  This
> admits an explicit Jacobian callback as a candidate for a separate paired
> stiff-prefix experiment.

Not allowed:

- BDF or trajectory speedup;
- recovery of the historical stalled integration;
- globally smooth derivative authority;
- endpoint or `N_eff` agreement;
- `G-F10-INDEPENDENT-FLRW` movement;
- release, production-runtime or publication authority.

## 13. Next admissible node — D-081A

The next step is not a full endpoint run.  It is a separately preregistered,
bounded paired BDF stiff-prefix discriminator restarted from the exact retained
state.

Required arms:

```text
A: unchanged SciPy BDF without jac
B: identical SciPy BDF with an analytic jac(t,y) callback
```

The callback must rebuild or safely refresh the Jacobian at every state where
SciPy requests it; the fixed `state_1200` matrix cannot be reused after the
state changes without a separate stale-Jacobian contract.

Both arms must preserve the same equations, state, grid, tolerances,
first-step/max-step policy, prefix endpoint, accepted-state semantics and
failure behaviour.  Required telemetry includes RHS evaluations, Jacobian
evaluations, LU factorizations, accepted/rejected steps, BDF order and step
size, Newton iterations/failures, Jacobian build times, cache/RSS, support
branch changes, true nonlinear/linear residuals on the original operator,
conservation and blockwise trajectory differences.

Because a restart resets historical BDF state, D-081A may claim only local
stiff-prefix discrimination.  Reproduction or removal of the original
long-horizon ratchet requires a later run from the original initial condition
or a prospectively justified state-history reconstruction.
