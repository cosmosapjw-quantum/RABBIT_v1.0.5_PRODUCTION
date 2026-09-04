# D-081R1F1-P1 — electron collision-action `T_gamma` JVP contract

**Date:** 2026-09-05  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Branch:** `research/d081r1f1-p1-collision-tgamma-jvp-red-20260905-r1`  
**Exact base commit/tree:** `8c6fcf68186040313b6fd7407ff027205cfa1df8` / `054ca8eb440009de8b27c4c9a20325eb82413e6b`  
**Status:** `CONTRACT_FROZEN_PRODUCTION_ABSENT_RED_PENDING`

## 1. Admitted object

This node may implement only the fixed-support derivative of the admitted
finite-electron-mass electron/positron collision action with respect to the
photon/electron bath temperature:

\[
K_T(c,T_{\rm cm},T_\gamma)
=\frac{\partial C_e}{\partial T_\gamma}
\]

at fixed neutrino cloglog coordinates `c`, fixed `T_cm`, fixed event catalogue,
fixed quadrature configuration, and fixed support/domain/matrix-correction
branch.

The output is a collision-action column per MeV. It is not the packed-RHS
`T_gamma` column, an arbitrary-direction JVP, a dense Jacobian, a solver
callback, a trajectory, or a performance result.

## 2. Frozen authority

Every workflow and oracle must fail closed unless these identities match:

```text
base canonical commit/tree:
8c6fcf68186040313b6fd7407ff027205cfa1df8
054ca8eb440009de8b27c4c9a20325eb82413e6b

R1F1 parent contract blob:
58eeaf38b9f4edd4c60a01d22d2e101a33b71812

P0AB repair receipt blob:
1206d9c9c73b4ab7166939cad6b5c8a038532e7e

P0AB clean replay receipt blob:
d549e270b20b6c9972e1dcf62a232f019621896e

P0A repaired source blob:
8d731d9f88076375b73ac4eca2a6ac54e98e66c6

P0B repaired source blob:
88717ed6206cae78599e805de79cf1b3f046443a

Rust primal electron-action blob:
f1c25926100ae6904a95b180ac50a2f05cdcfeaf

Python D-080B source blob:
78489c43f3046db09d8ba2d96070124ed7b0aa91

frozen private comparator blob:
de44feee0aa484abe26976c7dc34c579643005b5

Cargo.lock blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014
```

## 3. Event-rate derivative

For one event/sample, write

\[
R=W\,\mathcal M\,\Phi,
\]

where `W` is the complete event measure, `M` the weak matrix element, and
`Phi` the fermionic gain-minus-loss factor. On one unchanged discrete branch,

\[
\boxed{
R_T=W_T\mathcal M\Phi+W\mathcal M_T\Phi+W\mathcal M\Phi_T
}.
\]

The implementation and receipt must retain the three rate components
`measure`, `matrix`, and `Pauli` separately.

For an outgoing moving modal projection,

\[
-R\phi_n(y_3),
\]

\[
\boxed{
\partial_T[-R\phi_n(y_3)]
=-R_T\phi_n(y_3)-R\phi_n'(y_3)y_{3,T}
}.
\]

The second term is the independent `projection` component.

## 4. Elastic channel

The admitted event measure has the form

\[
W=C\,w_2p_2^2\frac{\Phi_{\rm ps}}{E_2}.
\]

Its derivative is evaluated by the explicit product rule

\[
\begin{aligned}
W_T=C\Big[&w_{2,T}p_2^2\frac{\Phi_{\rm ps}}{E_2}
+2w_2p_2p_{2,T}\frac{\Phi_{\rm ps}}{E_2}\\
&+w_2p_2^2\frac{(\Phi_{\rm ps})_T}{E_2}
-w_2p_2^2\Phi_{\rm ps}\frac{E_{2,T}}{E_2^2}\Big].
\end{aligned}
\]

For matrix primitives

\[
K_s=d_{12}d_{34},\qquad K_t=d_{14}d_{23},
\]

\[
(K_s)_T=(d_{12})_Td_{34}+d_{12}(d_{34})_T,
\]

\[
(K_t)_T=(d_{14})_Td_{23}+d_{14}(d_{23})_T.
\]

The finite-mass interference derivative is

\[
(m_e^2d_{13})_T=m_e^2(d_{13})_T.
\]

Flavour and particle/antiparticle coupling exchanges must occur in exactly the
same places as in the primal operator.

For an electron or positron logit

\[
u_e=-E/T_\gamma,
\]

\[
\boxed{(u_e)_T=-E_T/T_\gamma+E/T_\gamma^2}.
\]

For a moving outgoing neutrino interpolation,

\[
y_{3,T}=|\mathbf p_3|_T/T_{\rm cm},
\]

\[
(u_{\nu,3})_T=\sum_n c_n\phi_n'(y_3)y_{3,T}.
\]

## 5. Stable Pauli derivative

Let

\[
\Phi=L\operatorname{expm1}(a),
\]

\[
L=f_1f_2(1-f_3)(1-f_4),\qquad
a=u_3+u_4-u_1-u_2.
\]

The production tangent shall use the stable affinity form

\[
\boxed{
\Phi_T=L_T\operatorname{expm1}(a)+Le^aa_T
}
\]

or its algebraically identical log-derivative form. A direct partial-gradient
form may be retained only as an independent diagnostic, especially near
detailed balance and occupation tails.

## 6. Pair-channel exact-zero structure

At fixed neutrino grid, fixed masses, and fixed `T_cm`, the pair-channel
kinematic batch, measure, weak matrix element, and projection locations have no
direct `T_gamma` dependence:

\[
W_T=0,\qquad \mathcal M_T=0,\qquad y_{3,T}=0.
\]

Only the outgoing electron/positron logits vary:

\[
(u_{e^\pm})_T=E_{e^\pm}/T_\gamma^2.
\]

The pair-channel `measure`, `matrix`, and `projection` tangent arrays must be
bitwise zero and must be protected by mutations.

## 7. Energy ledgers and differentiated first law

For elastic events,

\[
w_\nu=E_1-E_3,\qquad w_{\rm em}=E_2-E_4.
\]

Four-momentum conservation implies

\[
w_\nu+w_{\rm em}=0,
\]

\[
(w_\nu)_T+(w_{\rm em})_T=-E_{3,T}+E_{2,T}-E_{4,T}=0.
\]

The implementation must preserve separately:

```text
rate-weight Q_nu,T
rate-weight Q_em,T
kinematic-weight Q_nu,T
kinematic-weight Q_em,T
total Q_nu,T
total Q_em,T
first-law tangent residual
```

For pair events the energy weights and their derivatives are fixed and sum to
zero. The differentiated first law must be checked event-family-wise and
globally.

## 8. Required result semantics

The intended child module is:

```rust
crate::f10_electron_action::tgamma_jvp
```

The API shall expose a result containing at least:

```text
base primal electron action
modal and native total tangent
measure / matrix / Pauli / projection modal and native components
elastic / pair modal and native components
pair measure / matrix / projection exact-zero diagnostics
15 family names and family tangent rows
Q_nu,T and Q_em,T by family and component
rate-weight and kinematic-weight ledgers
first-law tangent residual
charge-conjugation and mu/tau diagnostics
support/domain/matrix-correction branch signature
normalized support and supported-lambda margins
```

Errors must distinguish invalid input/configuration, dimension overflow,
foundation/kinematic/kernel failure, nondifferentiable discrete branch, and
nonfinite output.

## 9. Frozen gates

No threshold may be selected after Rust P1 output is inspected.

```text
component reconstruction:                 <= 2e-12
D-080B modal/native component parity:      <= 1e-7
Q_nu,T and Q_em,T parity:                  <= 1e-7
first-law tangent residual:                <= 2e-9
order-8 centered witness:                  <= 2e-6
branch signatures:                         exact identity
pair measure/matrix/projection tangents:   bitwise zero
```

Raw forward residuals, local relative residuals, ULP distances, worst indices,
rejection counts, correction counts, and branch margins remain in receipts.

## 10. Required mutations

At minimum kill:

```text
freeze electron nodes
freeze electron weights
omit phase-space tangent
omit matrix tangent
omit electron-FD tangent
omit moving-neutrino interpolation tangent
omit mapped-basis projection tangent
reverse Q_em,T
omit kinematic energy-weight tangent
make pair kinematics move with T_gamma
remove particle/antiparticle routing
swap flavour couplings
inject support or matrix-correction branch change
one-percent scale mutation in each nonzero component
```

## 11. Execution order

```text
contract and absent-module RED
pair-channel Pauli-only GREEN
elastic measure/matrix GREEN
elastic Pauli/projection GREEN
energy ledgers and branch signature GREEN
direct Python D-080B order-8 oracle
centered ladders and mutation battery
only then packed-RHS T_gamma push-forward
```

Retained `state_1200` and holdout `state_2000` must not be accessed in P1.

## 12. Claim ceiling

The maximum P1 classification is

```text
PASS_WITH_ORDER8_FIXED_BRANCH_ELECTRON_COLLISION_TGAMMA_JVP_SCOPE
```

It does not admit the packed-RHS thermal column, arbitrary-direction JVP, full
Jacobian, solver callback, trajectory, endpoint, `N_eff`, performance,
publication readiness, merge, or movement of `G-F10-INDEPENDENT-FLRW`.
