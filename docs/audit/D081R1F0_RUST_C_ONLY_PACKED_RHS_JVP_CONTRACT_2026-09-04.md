# D-081R1F0 — Rust c-only packed-RHS analytic JVP contract

Date: 2026-09-04  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Canonical parent: `8cef907e704149340774214f4da1bd28b79608e9`  
Canonical parent tree: `189e100de980fdbbe654e579d83c939cbdb1cef1`  
D-081R1E holdout receipt head: `1fa6afc024921dbad89811892b579b42ce5e2288`  
Frozen Python comparator blob: `de44feee0aa484abe26976c7dc34c579643005b5`  
Python c-only JVP sources: `668f3fab76ffc3ad7f29335a79fcd5daf47d429e`, `591a64702c58a2de265fb88636f186e2d1b7e019`, `6bcff2bc5627c0af0ad4df61c908d09e62ffaba5`  
Status: **PRE-OUTPUT RED-FIRST CONTRACT; STATIC DERIVATIVE ONLY**

## 1. DAG position and objective

D-081R1E admitted the Rust order-60 packed right-hand side on the retained
calibration state and one prospectively frozen unseen state. D-081R1F0 is the
first derivative node above that result. It ports the already-derived D-079
spectral-`c` event JVP into the Rust production collision path and pushes it
through the admitted packed RHS.

The input direction changes only the three pair-cloglog spectral blocks. The
input values of `ln_a`, `T_gamma`, and elapsed time are fixed. Their output rows
may nevertheless have induced tangents through the collision action and Hubble
factor.

This node does not add a finite-difference production path and does not build a
dense matrix.

## 2. Frozen physical and numerical boundary

The following must remain semantically unchanged from the canonical parent:

- all self and electron reaction catalogues and coefficients;
- four-leg signs, target routing, charge conjugation, and flavour ordering;
- finite-electron-mass kinematics, support masks, and domain-rejection policy;
- matrix elements and matrix-roundoff correction policy;
- GL60/Y30, inner GL12/GL48, and all non-authoritative quadrature routes;
- modal basis, modal contraction, and modal-to-native reconstruction;
- FLRW thermodynamics, electromagnetic equation of state, and Hubble law;
- packed-state order, units, signs, and failure semantics;
- the D-081R1E calibration fixture and unseen holdout receipt.

Permitted changes are derivative-only modules, derivative tests, derivative
fixtures/receipts, module registration, and bounded visibility changes required
to reuse the admitted base operators. A refitted coefficient, changed support
mask, clipped state, projected direction, or modified base RHS leaves this
contract.

## 3. State, chart, and units

The packed state is

\[
y=(c_{ai},T_\gamma,t),\qquad a\in\{e,\mu,\tau\},
\]

with dimensionless complementary-log-log coordinate

\[
f(c)=1-e^{-e^c},\qquad q(c)=\frac{df}{dc}=e^{c-e^c}.
\]

For a spectral direction `v`,

\[
\delta f=qv,
\qquad
\delta u=\frac{e^c}{f}v,
\qquad
\delta\log q=(1-e^c)v,
\]

where `u=log(f/(1-f))`. No endpoint floor, clipping, or projection is allowed.
Occupations, `c`, `u`, and `ln_a` are dimensionless. Collision actions and the
Hubble rate have MeV, `T_gamma` has MeV, and elapsed time has MeV^-1.

## 4. Exact event tangent

For one ordered event with logits `(u1,u2,u3,u4)`, the admitted Pauli factor is

\[
P=(1-f_1)(1-f_2)f_3f_4-f_1f_2(1-f_3)(1-f_4).
\]

The Rust source already exposes its exact logit gradient. D-081R1F0 must use

\[
\delta P=\sum_{r=1}^4\frac{\partial P}{\partial u_r}\,\delta u_r.
\]

Equivalently, with `G` and `L` the gain and loss products and
`a=u3+u4-u1-u2`,

\[
\delta P=P\,\delta\log L+G\,\delta a.
\]

At fixed `T_cm` and `T_gamma`, the kinematic measure, matrix element, support
mask, quadrature weights, and basis functions have zero tangent. Each event
therefore obeys

\[
\delta R=W\,|\mathcal M|^2\,\delta P.
\]

The signed four-leg assembly and the admitted modal/native maps are then applied
linearly to `delta R`.

## 5. Packed-RHS push-forward

For a pair-averaged native collision action `C`,

\[
g=\frac{dc}{dN}=\frac{C}{Hq}.
\]

At fixed `ln_a` and `T_gamma`,

\[
\delta\rho_\nu=
\frac{T_{\rm cm}^4}{\pi^2}
\sum_{a,i}w_i y_i^3\,\delta f_{ai},
\qquad
\frac{\delta H}{H}=\frac12\frac{\delta\rho_\nu}{\rho_{\rm tot}},
\]

and

\[
\boxed{
\delta g=
\frac{\delta C}{Hq}
-g\left(\frac{\delta H}{H}+\delta\log q\right)
}.
\]

For the photon-temperature and elapsed-time output rows,

\[
\delta\!\left(\frac{dT_\gamma}{dN}\right)=
\frac{1}{d\rho_{\rm EM}/dT_\gamma}
\left[
\frac{\delta Q_{\rm EM}}{H}
-rac{Q_{\rm EM}}{H}\frac{\delta H}{H}
\right],
\]

\[
\delta\!\left(\frac{dt}{dN}\right)
=-\frac1H\frac{\delta H}{H}.
\]

These are induced output tangents. They are not `T_gamma`, `ln_a`, or elapsed
input columns.

## 6. Prospectively frozen directions

No RNG is permitted.

### 6.1 Order-8 calibration direction

Use the exact D-079 construction. For `x_i` linearly spaced from `-1` to `1`,

\[
v_e=0.3+x_i,\qquad
v_\mu=-0.2+x_i^2,\qquad
v_\tau=-0.2+x_i^2,
\]

followed by one global Euclidean normalization.

### 6.2 Retained order-60 calibration direction

For `phi_i=pi i/(n-1)`,

\[
v_e=\cos\phi_i,\qquad
v_\mu=\sin\phi_i,\qquad
v_\tau=\sin\phi_i,
\]

followed by one global Euclidean normalization.

### 6.3 Retained order-60 unseen holdout direction

For the same `phi_i`,

\[
\begin{aligned}
v_e &= 0.25+\cos(2\phi_i),\\
v_\mu &= -0.15+\sin(3\phi_i),\\
v_\tau &= 0.35\cos\phi_i-0.20\sin(2\phi_i),
\end{aligned}
\]

followed by one global Euclidean normalization. The holdout output must not be
generated or inspected until the implementation, thresholds, and workflow are
committed.

## 7. Prospectively frozen gates

Raw absolute, local-relative, norm-relative, and ULP diagnostics must always be
reported even when a conditioned gate is used.

### 7.1 Structural gates

- zero direction produces bitwise-zero derivative arrays and scalar tangents;
- linearity residual for `J(a v+b w)=aJv+bJw` is at most `5e-12` in a
  scale-aware infinity norm;
- the elapsed-time input coordinate is passive and absent from this direction;
- self and electron tangent actions add to the total tangent exactly in the
  implementation's declared addition order;
- the particle/antiparticle pair average retains the factor `1/2`;
- a sign mutation, a 1% scale mutation, omission of the electron tangent, and a
  flavour permutation are each killed by the inherited D-079 discriminators;
- the original D-081R1E packed-RHS suites remain GREEN.

### 7.2 Order-8 analytic cross-language gates

- self modal global relative residual `<= 1e-7`;
- electron modal global relative residual `<= 1e-7`;
- total modal global relative residual `<= 1e-7`;
- packed-RHS JVP global relative residual `<= 5e-7`;
- first-law tangent residual in each implementation `<= 2e-11`;
- best centered-difference collision witness `<= 2e-6`;
- best centered-difference packed-RHS witness `<= 3e-6`.

The centered ladder is `3e-4, 1e-4, 3e-5`. Production evaluation must not use
this ladder.

### 7.3 Retained order-60 analytic cross-language gates

For both the calibration and unseen holdout directions:

- self modal global relative residual `<= 1e-7`;
- electron modal global relative residual `<= 1e-7`;
- total modal global relative residual `<= 1e-7`;
- packed-RHS JVP global relative residual `<= 2e-4`;
- first-law tangent residual in each implementation `<= 2e-9`;
- differentiated self-number and self-energy moments are each bounded by
  `2e-9` times the larger of the tangent-action norm and unity;
- CP and mu/tau residuals in the symmetric calibration lane are each
  `<= 2e-9`;
- the local centered witness at epsilon `3e-6` has packed-RHS residual
  `<= 2e-4` and remains inside the strict chart on both sides.

A failed or unresolved centered witness is classified
`REFERENCE_DERIVATIVE_UNRESOLVED`; its threshold may not be widened after output
inspection.

## 8. Required provenance and runtime identity

The slow lane must recover without copying into the source tree:

- historical source commit `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`;
- `state_1200.npz` blob from
  `.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz`;
- state SHA-256
  `c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380`;
- `ln_a=0.16286930247517223`, order `60`, and `y_max=30`;
- NumPy `2.4.4`, SciPy `1.17.1`, Rust `1.94.1`;
- Cargo.lock blob `a1b5035da5c20712d1a2a4ab077da255ff94a014`;
- the exact 174-package offline Rust vendor used by D-081R1E.

The generated Python oracle fixture must contain binary64 bit strings,
direction definitions, authority identities, all gate values, and raw
metrology. It must be generated twice and compared byte-for-byte.

## 9. RED-first execution order

1. Commit this contract before generating derivative output.
2. Add a test that names the intended Rust API while the API is absent.
3. Run the exact Rust CI and retain the compile failure as the RED receipt.
4. Implement chart, event, action, and packed-RHS tangents without changing the
   base operators.
5. Run focused order-8 tests, then the retained calibration lane.
6. Only after both are GREEN, generate and execute the unseen holdout once.
7. Publish a durable receipt and a Draft PR for independent review.

## 10. Claim ceiling

A successful D-081R1F0 establishes only a static analytic Rust JVP for spectral
`c` input directions on the admitted packed RHS. It does not establish:

- `T_gamma`, `ln_a/T_cm`, or elapsed-time input columns;
- a dense or sparse full Jacobian;
- a preconditioner, Newton policy, or diffsol callback;
- BDF integration, accepted-step histories, trajectories, endpoints, or
  `N_eff`;
- speedup, memory advantage, or same-budget solver superiority;
- movement of `G-F10-INDEPENDENT-FLRW`;
- QKE, Bianchi, inference, or publication authority.

The next admissible node is D-081R1F1 for the thermal/explicit-expansion input
columns. Only after D-081R1F0 and D-081R1F1 pass may D-081R1G assemble and admit
the full production Jacobian or matrix-free callback.
