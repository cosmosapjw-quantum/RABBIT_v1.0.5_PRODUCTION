# D-081R1F1 P0A/P0B adversarial repair contract

**Date:** 2026-09-04  
**Base commit/tree:** `9929596e1cf0351aba6e8e2b501ced89d1c80fb1` / `80eacd19ce2a66b943925b95d334144ec952f79e`  
**Status:** `CONTRACT_FROZEN_REPAIR_IMPLEMENTATION_ABSENT_RED_REQUIRED`

## Scope

This node corrects and strengthens only the already integrated P0A/P0B thermal primitives. It must not implement the electron collision-action `T_gamma` JVP, the packed-RHS thermal column, retained calibration, holdout, solver, trajectory, endpoint, performance, or gate movement.

## Immutable authorities

```text
R1F1 integration commit/tree:
9929596e1cf0351aba6e8e2b501ced89d1c80fb1
80eacd19ce2a66b943925b95d334144ec952f79e

R1F1 contract blob:
58eeaf38b9f4edd4c60a01d22d2e101a33b71812

P0A source/test blobs:
0217768ba62b02d14a3c52ae68208fafc8c8d46c
f2642cf9a699290e2c97a7d6e8a12b980e5f8919

P0B source/test blobs:
aa5092ad4d1a37365a8a1354a6cce0f81c3c4412
14db5662106ad432d8236a97beccace85c2f6075

D-080A source blob:
c585d5865fd68a90a04a76ab540b8437fba8cfce

Cargo.lock blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014
```

## Repair A — exact derivative of the admitted finite EOS algorithm

The admitted primal Rust electromagnetic EOS uses a 256-panel transformed-Simpson sum with `tail_e_folds=48`. Production `d2_rho` must be the analytic derivative of that finite real-arithmetic algorithm, including the moving endpoint, step, and nodes. The existing 4096-panel expression remains available only as a continuum-reference diagnostic.

For

\[
D_e(T)=\frac{2T^3}{\pi^2}S(T),\qquad
S(T)=\frac{h(T)}3\sum_{i=0}^{N}c_iA_i(T),\qquad N=256,
\]

\[
x=m_e/T,\quad z=1+48/x,\quad
\theta_{\max}=\operatorname{arcosh}z,\quad h=\theta_{\max}/N,\quad\theta_i=ih,
\]

\[
A_i=x^5\sinh^2\theta_i\cosh^3\theta_i f_i(1-f_i),\qquad
f_i=(e^{x\cosh\theta_i}+1)^{-1},
\]

the implementation must evaluate

\[
(D_e)_T=\frac{2}{\pi^2}\left(3T^2S+T^3S_T\right)
\]

with the full moving-node chain rule. The photon contribution is `4*pi^2*T^2/5`.

Prospective gates:

```text
discrete analytic vs centered derivative of admitted primal drho/dT: <= 2e-7
discrete analytic vs 4096-panel continuum reference:                <= 2e-7
discrete analytic vs frozen D-080A reference:                        <= 1e-7
```

No gate may be widened after output is observed.

## Repair B — normalized branch margins

Retain the existing raw dimensionful minima and add

\[
m_s=\frac{\min|s-m_e^2|}{\max(\max|s|,m_e^2,\mathrm{tiny})},
\]

\[
m_\lambda=\min_{\mathrm{support}}\frac{\lambda}{\max(s^2,\mathrm{tiny})}.
\]

These normalized quantities must match the frozen Python D-080A definitions to relative residual `<=1e-7`.

## Repair C — direct D-080A oracle

A deterministic Python generator must emit the mapped-basis derivative, support mask, all primal kinematic arrays, all fourteen nontrivial tangent arrays, and both normalized margins. It must be run twice in one execution capsule and produce byte-identical JSON. Rust/Python array parity gates are `2e-7` for the basis derivative and `1e-7` for kinematic arrays and normalized margins.

## Repair D — tangent invariants

For every supported sample, require contribution-scaled residual `<=2e-12` for

\[
E_{2,T}-E_{3,T}-E_{4,T}=0,
\]

\[
E_{3,T}-|\mathbf p_3|_T=0,
\]

\[
E_4E_{4,T}-|\mathbf p_4||\mathbf p_4|_T=0,
\]

\[
(d_{12})_T-(d_{34})_T=0,
\]

\[
(d_{13})_T+(d_{14})_T-(d_{12})_T=0,
\]

\[
(d_{23})_T+(d_{24})_T-(d_{12})_T=0.
\]

## Repair E — clean replay

The final committed source must be checked out by exact `github.sha`, with no source mutation. The replay must regenerate P0A and P0B Python oracles twice, run focused P0A/P0B tests, inherited R1F0 zero-JVP and packed-RHS preflight tests, release check, and strict Clippy.

## Claim ceiling

A PASS establishes only corrected P0A/P0B primitive semantics and reproducibility. It does not establish collision-action or packed-RHS `T_gamma` JVP, arbitrary-direction JVP, a dense Jacobian, solver behavior, trajectory, endpoint, `N_eff`, performance, publication readiness, merge authority, or `G-F10-INDEPENDENT-FLRW` movement.
