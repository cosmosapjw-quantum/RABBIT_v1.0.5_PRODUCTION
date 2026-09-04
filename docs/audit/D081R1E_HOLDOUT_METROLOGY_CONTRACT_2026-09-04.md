# D-081R1E unseen retained-state metrology contract

Status: `PROSPECTIVELY_FROZEN_BEFORE_HOLDOUT_OUTPUT`

## Scope

This contract adjudicates cross-language floating-point reduction differences
for the retained order-60 packed RHS. It changes no collision equation, event
catalogue, quadrature, grid, state, support policy, matrix-roundoff policy,
thermodynamic formula, chart, or solver tolerance.

The calibration state was `state_1200.npz`. The holdout state is
`state_2000.npz`. Its Rust/Python collision and RHS outputs were not evaluated
before the thresholds below and the exact holdout byte identity were committed.

## Authorities

```text
repository:
cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION

R1E implementation predecessor:
22240bba2f4c4c02ec2eedd4f131a8fffd3be5e2

historical source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

holdout path:
.agent-harness/runs/run-20260805-f10-v3-campaign/
v3a_r2/domain/state_2000.npz

holdout Git blob:
cfb17344ae166c01c2e5bcb14acae0d968e49477

holdout bytes:
2103

holdout SHA-256:
780ad7c1388caec23f02012781717d43ffb85d96d4d501c40c504939e7c9a44d

Python comparator blob:
de44feee0aa484abe26976c7dc34c579643005b5

trajectory-core blob:
465a73f0ce40f7149bebdc2d67103f388e2344d9

Cargo.lock blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014

authority-only workflow:
33804815829

authority-only job:
100812636614
```

The authority-only run verified the historical commit, path, Git blob, byte
count and SHA-256 before any collision or RHS evaluation. Its optional archive
structure step failed because bare runner Python lacked NumPy; this is a CI
orchestration failure after the byte authority was printed, not a physics or
identity failure. The holdout evaluation workflow must install exact NumPy
2.4.4 and recheck archive structure before opening the holdout output.

## Calibration evidence

At `state_1200.npz`, after exact F10 GL12/GL48 and GL60/Y30 operator identity:

```text
self-modal global relative difference:     3.2216170899877647e-9
electron-modal global relative difference: 4.1934221790331675e-10
total-modal global relative difference:    2.7454257901540250e-9
spectral-RHS global relative difference:   3.7276436876653520e-5
maximum one-step solver-scaled impact:      8.9605267987536170e-6
```

The solver-scaled impact is

\[
R_{\rm step}
=
\max_{i<180}
\frac{|h_{\rm retained}|\,|F_i^{\rm Rust}-F_i^{\rm Python}|}
{10^{-9}+10^{-6}|Y_i|}.
\]

The modal global metric for component \(X\) is

\[
R_{X,\rm modal}
=
\frac{\|\widehat C_X^{\rm Rust}-\widehat C_X^{\rm Python}\|_\infty}
{\max(\|\widehat C_X^{\rm Rust}\|_\infty,
      \|\widehat C_X^{\rm Python}\|_\infty,
      s_{\rm tiny})}.
\]

## Prospectively frozen holdout gates

The unseen `state_2000.npz` must satisfy all of the following:

```text
self_modal_global_relative     <= 1.0e-7
electron_modal_global_relative <= 1.0e-7
total_modal_global_relative    <= 1.0e-7
maximum_step_impact             <= 1.0e-3 local tolerance units
first_law_residual              <= 5.0e-13
```

The modal cap is stricter than the existing `5e-7` full-action elementwise
hybrid tolerance. The step-impact cap allocates at most 0.1% of one local
solver tolerance unit to the cross-language RHS difference at the retained
historical step size.

The following remain hard structural gates:

- exact GL60/Y30 and F10 GL12/GL48 binary64 identities;
- state size and ordering `(c_e,c_mu,c_tau,T_gamma,t)`;
- strict `0<f<1` with no clipping;
- finite positive `T_cm`, `T_gamma`, and `H`;
- exact passive stored-elapsed-time semantics;
- support/rejection and matrix-correction branch identity;
- component addition and pair averaging by one half;
- first-law, flavour-covariance, mutation, and transactional-failure gates.

## Recorded but non-gating diagnostics

The following MUST be retained in the receipt but are not independent
admission gates at a near-cancellation state:

- per-entry local forward relative differences;
- native-action global and local forward differences;
- pair-rate global and local forward differences;
- spectral-RHS global and local forward differences;
- ULP distances across unlike reduction associations.

They are not deleted or hidden. They are interpreted together with the modal
backward-scale metric, the solver-scaled impact, and the physical invariants.

## Decision rule

```text
PASS:
  all prospective holdout gates and all structural gates pass.
  Then replace only the ill-conditioned retained elementwise admission gate
  by the frozen modal-global plus solver-step-impact contract. Preserve raw
  forward diagnostics in the durable receipt.

FAIL:
  any prospective or structural gate fails.
  Do not widen thresholds. Open a separate deterministic/pairwise/compensated
  reduction diagnostic before changing production summation.
```

## Claim ceiling

A holdout PASS supports only a cross-language retained static packed-RHS
admission under the frozen finite-dimensional operator and tolerance policy.
It does not establish a JVP, Jacobian, diffsol trajectory, speedup, endpoint,
`N_eff`, publication authority, or movement of `G-F10-INDEPENDENT-FLRW`.
