# BD622 D-079 — Cloglog Event-JVP Static Physical Contract

Date: 2026-09-01  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Canonical parent: `4fd19764a0acfc60955b8ed819158e599867321f`  
Canonical parent tree: `343e7b1397e09f55c9ea376400323ee1c18f56d6`  
Comparator blob: `de44feee0aa484abe26976c7dc34c579643005b5`  
Decision authority: D-077 and D-078  
Status: **PRE-OUTPUT STATIC SPECTRAL-COLUMN CONTRACT; NO TRAJECTORY AUTHORITY**

## 1. Exact DAG position

D-078 closed the generic transformed-Jacobian research loop without calling the
private comparator. Inspection of the merged source showed that the real
comparator does not evolve a logit. It evolves the complementary-log-log chart

\[
 c=\log[-\log(1-f)],\qquad 0<f<1.
\]

D-079 therefore differentiates the actual frozen event-quadrature path with
respect to the three spectral `c` blocks. It does not yet provide columns for
`T_gamma`, explicit `N/T_cm`, or elapsed time. It does not call `solve_ivp` or
alter the BDF path.

## 2. Frozen physical and numerical boundary

The following remain byte-identical to the canonical parent:

- reaction catalogues and coefficients;
- target-leg convention and Pauli gain-minus-loss convention;
- two-body kinematics and support/domain rules;
- matrix elements and roundoff correction policy;
- radial/angular quadrature and `y_max`;
- modal interpolation and Galerkin-Petrov reconstruction;
- thermodynamics, Hubble law, electromagnetic equation of state;
- state layout, units, signs, failure semantics, and the private status of the comparator.

The new code imports those functions and differentiates only occupation/logit
dependence. A changed support mask, projected occupation, refitted coefficient,
or finite-difference factor would leave this contract.

## 3. Conventions, variables, and dimensions

The comparator uses natural-unit MeV conventions. Occupations, `c`, logits,
and e-fold `N` are dimensionless. Collision actions and the Hubble rate have
MeV, equivalent to inverse time. `T_gamma` and `T_cm` have MeV; elapsed time has
MeV^-1.

The chart and its differential are

\[
 f(c)=1-e^{-e^c},\qquad
 q(c)=\frac{df}{dc}=e^{c-e^c},
\]

\[
 u(c)=\log\frac{f}{1-f},\qquad
 \frac{du}{dc}=\frac{e^c}{f},\qquad
 \delta\log q=(1-e^c)\,\delta c.
\]

These are smooth on the strict-open finite-`c` domain. D-079 performs no floor,
clip, or endpoint projection.

## 4. Exact Pauli-factor tangent

For one event `1+2 <-> 3+4`, define

\[
 P=(1-f_1)(1-f_2)f_3f_4-f_1f_2(1-f_3)(1-f_4)
   =G-L,
\]

\[
 a=u_3+u_4-u_1-u_2,
 \qquad L=f_1f_2(1-f_3)(1-f_4),
 \qquad G=Le^a.
\]

The exact directional derivative is evaluated as

\[
 \boxed{\delta P=P\,\delta\log L+G\,\delta a},
\]

where

\[
 \delta\log L=(1-f_1)\delta u_1+(1-f_2)\delta u_2
 -f_3\delta u_3-f_4\delta u_4,
\]

\[
 \delta a=\delta u_3+\delta u_4-\delta u_1-\delta u_2.
\]

At detailed balance, `a=0` and `P=0`, so

\[
 \delta P=G\,\delta a.
\]

This form avoids treating a cancellation-small base collision factor as a
zero tangent.

## 5. Event-quadrature JVP

At fixed temperatures the kinematic measure, matrix element, support mask, and
basis functions have zero `c` tangent. For every retained quadrature event,

\[
 \delta R_e=W_e\,|\mathcal M_e|^2\,\delta P_e.
\]

The signed four-leg assembly, row/flavour assignment, modal contraction, and
native action map are then applied to `delta R_e` exactly as they are to the
base event rate. This gives

\[
 \delta C_{\rm self},\qquad
 \delta C_e,\qquad
 \delta C=\delta C_{\rm self}+\delta C_e,
\]

and the electron-bath transfer tangent `delta Q_em`.

Load-bearing implementation paths:

- `scripts/audit/_d079_tangent_primitives.py`;
- `scripts/audit/_d079_collision_jvp.py`;
- `scripts/audit/_d079_rhs_jvp.py`.

The private source file is not edited.

## 6. C-only full static RHS push-forward

For pair-averaged collision action `C`,

\[
 g=\frac{dc}{dN}=\frac{C}{Hq}.
\]

At fixed `T_cm` and `T_gamma`,

\[
 \boxed{
 \delta g=\frac{\delta C}{Hq}
 -g\left(\frac{\delta H}{H}+\delta\log q\right)}.
\]

The neutrino energy-density tangent and Hubble tangent are

\[
 \delta\rho_\nu=
 \frac{2T_{\rm cm}^4}{2\pi^2}
 \sum_{a,i}w_i y_i^3\,\delta f_{ai},
 \qquad
 \frac{\delta H}{H}=\frac12\frac{\delta\rho_\nu}{\rho_{\rm tot}}.
\]

For the photon-temperature and elapsed-time rows,

\[
 \delta\!\left(\frac{dT_\gamma}{dN}\right)=
 \frac{1}{d\rho_{\rm EM}/dT_\gamma}
 \left[
 \frac{\delta Q_{\rm em}}{H}
 -\frac{Q_{\rm em}}{H}\frac{\delta H}{H}
 \right],
\]

\[
 \delta\!\left(\frac{dt}{dN}\right)
 =-\frac1H\frac{\delta H}{H}.
\]

These are induced output-row tangents from spectral columns. They are not
`T_gamma`, `N`, or time input columns.

## 7. Mandatory static checks

D-079 is admissible only if all of the following hold:

1. exact chart derivatives agree with a resolved centered witness;
2. Pauli JVP agrees with centered evaluation and recovers the detailed-balance tangent;
3. full collision JVP agrees with the unchanged original collision path;
4. full static RHS JVP agrees with the unchanged original RHS path;
5. self-collision number and energy tangent moments remain null within scale-aware bounds;
6. `delta Q_nu + delta Q_em` satisfies the differentiated first law;
7. CP and mu/tau symmetric state/direction limits are recovered;
8. sign, 1% scale, flavour-index swap, and omitted-electron-block mutations are rejected;
9. exact retained `creep_1200` bytes remain strict-open under a local centered direction and pass the original-RHS discriminator.

## 8. Retained-state provenance

The slow lane must recover, without copying into the source tree,

- diagnosis branch commit: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`;
- `state_1200.npz` SHA-256:
  `c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380`;
- `N=0.16286930247517223`, order `60`, `y_max=30`.

The previous direct-JVP failure is retained as a negative control: its
forward, norm-scaled Arnoldi rule generated a nonlocal step and left the strict
domain. D-079 neither overwrites nor reinterprets that receipt.

## 9. Independent symbolic and literature checks

Wolfram exact differentiation returned zero residual for the chart/RHS quotient
rule and for the compact Pauli tangent identity. The detailed-balance reduction
was obtained without numerical fitting.

A SciSpace search identified direct differential-system Jacobians as an
established acceleration strategy in precision neutrino-decoupling work,
including Froustey, Pitrou, and Volpe, JCAP 12 (2020) 015,
DOI `10.1088/1475-7516/2020/12/015`. This is methodological background only;
it is not authority for RABBIT coefficients or derivatives.

## 10. Explicit non-authority

D-079 does not establish:

- `T_gamma`, explicit `N/T_cm`, or elapsed-time input columns;
- a dense full Jacobian, sparsity pattern, preconditioner, or Newton policy;
- BDF integration, a physical prefix, trajectory, endpoint, or completion time;
- same-budget performance or solver improvement;
- movement of `G-F10-INDEPENDENT-FLRW` from `FAIL`;
- public backend, QKE, Bianchi, inference, or publication authority.

The next admissible node after a successful D-079 closeout is the remaining
thermal/explicit-time columns and a full-state static Jacobian/JVP certificate.
Only after that node passes may a separately sealed BDF `jac` instrument be
considered.
