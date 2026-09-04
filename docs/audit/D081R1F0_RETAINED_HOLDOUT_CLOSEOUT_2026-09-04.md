# D-081R1F0 retained spectral-c JVP holdout closeout

Date: 2026-09-04  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Branch: `research/d081r1f0-rust-c-only-jvp-20260904`  
Base branch: `research/d081r1e-retained-packed-rhs-20260903`  
Base commit: `8cef907e704149340774214f4da1bd28b79608e9`

## 1. Closeout classification

```text
PASS_WITH_RETAINED_STATE1200_C_ONLY_JVP_HOLDOUT_AND_MU_TAU_COVARIANCE_SCOPE
```

This classification is limited to the static, fixed-grid, fixed-support analytic
Jacobian-vector product for spectral complementary-log-log (`c`) directions.
It includes the preregistered retained `state_1200` holdout and the physically
correct mu/tau permutation covariance test.

It does **not** establish a thermal input column, full square Jacobian, solver
integration, trajectory completion, endpoint agreement, `N_eff`, performance,
publication readiness, main integration, or movement of
`G-F10-INDEPENDENT-FLRW`.

## 2. Exact lineage and durable evidence

```text
holdout trigger commit:
c4c8375f08533bd69205f15a465a10a1a218cf94

holdout trigger tree:
e377f1c27329a560ef8e708e58dc65dac48b8d2d

holdout workflow run:
33844270726

receipt publication commit:
066b616c6275b1069a6e0b6bee8c34444de167a2

receipt publication tree:
6d2ca545c3d3ae2b1c7db967e5e894d5636a915e

tested source tree without receipts:
70217e98ca03fb96b57173fab219d577a4eeb1ee
```

Durable receipts:

```text
docs/audit/artifacts/d081r1f0/order8_verification_receipt.json
  blob: b14288ad944786e2f4792e0793da5b2d936929ef

docs/audit/artifacts/d081r1f0/retained_calibration_amended_receipt.json
  blob: 2ab3b55b83087b67693955c11b866b200bd373ab

docs/audit/artifacts/d081r1f0/retained_holdout_receipt.json
  blob: 01a87b4227fbe11e83412a5899a1eead69fbda3c

docs/audit/artifacts/d081r1f0/symbolic_symmetry_audit_receipt.json
  blob: ba8243404c5932e73b9bd8a5ca380f04a62c41d8
```

The holdout fixture was generated only after its direction, test source,
thresholds, covariance identity, calibration predecessor, and workflow were
committed. Its SHA-256 is

```text
efc16d5ebdb2a17e4289d85f6f1745feb09b72591f1ac2f5e38023bb67bf0b25
```

## 3. Retained holdout results

The exact order-60 holdout passed with the following global-relative residuals:

```text
Rust/Python self modal JVP:       1.3271592057749718e-14
Rust/Python electron modal JVP:   3.6816521833306234e-15
Rust/Python total modal JVP:      9.5597176074558773e-15
Rust/Python packed-RHS JVP:       2.0703220300037675e-16
centered packed-RHS witness:      3.9004601891915251e-9
```

The frozen admission caps remained unchanged:

```text
component modal:                  1.0e-7
packed RHS:                       2.0e-4
centered packed RHS:              2.0e-4
first-law tangent:                2.0e-9
```

The differentiated conservation diagnostics were

```text
Rust first-law residual:          0
Python first-law residual:        0
self-number tangent ratio:        9.062975120796111e-16
self-energy tangent ratio:        4.515211284400525e-15
charge-conjugation residual:      7.314521521223062e-11
```

The support/correction branch was unchanged under the original direction,
centered witnesses, and mu/tau-swapped state and direction:

```text
whole-reaction domain rejections: 14494584
matrix roundoff corrections:      0
largest matrix correction:        0
```

## 4. Mu/tau covariance

The tested physical identity was

\[
J(Sy)\,Sv=S\,J(y)v,
\]

where `S` exchanges the mu and tau flavour blocks. The preregistered direction
was intentionally non-symmetric, so a raw equality of the mu and tau outputs
was neither assumed nor used as an admission gate.

The covariance residuals were

```text
self modal covariance:            1.3475590209670513e-15
electron modal covariance:        0
total modal covariance:           9.7612245765977443e-16
packed-RHS covariance:            1.1666494316195391e-22
delta-rho covariance:             1.5494196423186407e-16
delta-H/H covariance:             0
```

The raw asymmetric-lane mu/tau ratio was approximately `1.3514`; this is an
expected diagnostic for the non-symmetric holdout direction, not a symmetry
failure. The covariance identity above is the relevant physical statement.

## 5. Preserved legacy failure and amendment

The original retained symmetric-direction calibration required
`r_mu_tau <= 2e-9`. It failed in Rust at
`2.5305688270469514e-9`; the frozen Python authority independently produced
`2.5237504535839513e-9` and therefore also failed that scalar cap.

The failure remains preserved. The threshold was not widened. Before holdout
unblinding, the calibration contract was amended to require stored/recomputed
self-consistency, Rust/Python pair-array parity, and a propagated conditioning
bound. The observed Rust/Python scalar difference

```text
6.8183734630001737e-12
```

was below the conservative propagated bound

```text
1.9980962279218883e-11.
```

Physical flavour symmetry was then tested prospectively by the permutation
covariance holdout in section 4.

## 6. CAS and formal-tool status

Fresh Wolfram Context and Wolfram Language evaluator requests were blocked by
upstream HTTP 502. This is an external-service blocker and supplies no new
Wolfram receipt.

The fallback audit was executed in the holdout workflow with

```text
SymPy 1.14.0
mpmath 1.3.0
mpmath precision: 80 decimal digits
```

and classified

```text
PASS_AUXILIARY_SYMPY_MPMATH_AUDIT.
```

It verified the algebraic ratio identity, the propagated ratio bound, the
permutation derivative identity, and an explicit counterexample showing that
`Sv=v` alone does not imply `J(y)v=S J(y)v` when `Sy != y`.

Executable probes found no installed `octave`, `octave-cli`, `sage`,
`Singular`, `lean`, `lake`, or `elan` on the runner. Their absence is an
environment/tool-availability result, not a failed mathematical check.

The SymPy/mpmath receipt is corroborative only. Repository equations, the
frozen Python comparator, Rust execution, and durable receipts remain the
scientific authorities.

## 7. Executed regression substrate

The holdout workflow also passed:

- Rust 1.94.1 release check with the exact 174-package offline vendor;
- `cargo fmt --all -- --check` after bounded formatting of the holdout test;
- the full order-eight nonzero JVP test set;
- exact-zero spectral-direction JVP;
- retained primal packed-RHS preflight and regressions;
- Pauli/support/correction inherited regressions;
- `cargo clippy --all-targets --all-features -- -D warnings`.

## 8. Remaining closeout conditions

This document is a bounded human-authored closeout commit, also used to trigger
the ordinary PR harness after the receipt commit was created by
`github-actions[bot]` and its automatic harness run ended with
`action_required` before creating jobs.

The feature PR remains Draft. Merge remains forbidden until:

1. the ordinary PR harness is terminal green on the human-authored closeout
   head;
2. independent PHYS-MATH and PHYS-MATH-CODE reviews find no P0/P1 issue that
   invalidates the admitted scope;
3. the PR readback agrees with the exact head, tree, receipt blobs, and claim
   ceiling recorded here.

## 9. Next admissible node

The next implementation node is a Rust `T_gamma` input-direction JVP, using the
frozen D-080C static-column authority. At fixed `c`, `T_cm`, and `N`, it must
implement

\[
\frac{H_T}{H}=\frac{\chi_\gamma}{2\rho_{\rm total}},
\]

\[
F_{c,T}=\frac{P_T}{Hq}-F_c\frac{H_T}{H},
\]

\[
F_{t,T}=-\frac1H\frac{H_T}{H},
\]

and the complete photon-temperature quotient derivative, including collision,
Hubble-feedback, EOS-numerator, and heat-capacity-denominator terms. The
elapsed-time input column remains exactly zero.

The thermal collision derivative must preserve the existing moving-electron
kinematics, finite-mass phase space, interpolation, support predicates, and
matrix-correction branch. Blockwise dimensional residuals and original-RHS
centered witnesses remain mandatory.
