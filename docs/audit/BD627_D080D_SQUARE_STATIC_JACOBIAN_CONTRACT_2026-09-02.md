# BD627 — D-080D Square Static-Jacobian Contract

**Date:** 2026-09-02  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Branch:** `research/d080d-square-static-jacobian-20260902`  
**Stack base:** D-080C head `718f5c839635f322f5f2e900f057709862a03a93`  
**Frozen comparator Git blob:** `de44feee0aa484abe26976c7dc34c579643005b5`  
**D-079 RHS-JVP Git blob:** `6bcff2bc5627c0af0ad4df61c908d09e62ffaba5`  
**D-080C T-gamma RHS-column Git blob:** `c18feacbd57c9519af14504027b7d465758eb1ef`

## 1. Purpose and authority boundary

D-080D assembles the complete **fixed-support static square Jacobian** of the frozen private no-QKE comparator RHS in the exact packed-state ordering

```text
Y = (c_e[0:n], c_mu[0:n], c_tau[0:n], T_gamma, elapsed_time).
```

The operator is

```text
J_static = [ J_c | J_Tgamma | 0_elapsed ],
```

where:

- `J_c` is assembled from the already admitted D-079 exact spectral-direction RHS JVP by applying it to every spectral basis vector;
- `J_Tgamma` is the already admitted D-080C full static `T_gamma` RHS column;
- the stored elapsed-time input column is exact structural zero because the static RHS does not depend on the accumulated elapsed-time state.

D-080D is research-only. It does **not** authorize:

- differentiation through a discrete support or matrix-correction branch change;
- a BDF, Newton, JFNK, or nonlinear-solver call;
- trajectory completion or endpoint agreement;
- a speedup, memory, scaling, or conditioning claim;
- an updated `N_eff` result;
- movement of `G-F10-INDEPENDENT-FLRW`;
- public-production, release, or publication authority.

## 2. State, RHS, units, and dimensions

The independent variable is

```text
N = log(a),
T_cm(N) = T_start exp(-N).
```

The comparator uses natural units

```text
hbar = c = k_B = 1.
```

The packed state and RHS dimensions are:

| Block | State dimension | RHS dimension |
|---|---:|---:|
| `c_e,c_mu,c_tau` | 1 | 1 |
| `T_gamma` | MeV | MeV |
| `elapsed_time` | MeV^-1 | MeV^-1 |

After differentiation, Jacobian blocks carry the corresponding output/input dimension ratios. Consequently a raw full-matrix norm or condition number is not physically invariant under admissible unit rescalings. All validation residuals must separately normalize the spectral, photon-temperature, and elapsed-output blocks before taking their maximum.

## 3. Mathematical construction

Let

```text
F(Y;N) = dY/dN.
```

For a full direction

```text
v = (v_c, v_T, v_t),
```

the exact static directional derivative is assembled independently of the explicit matrix as

```text
D F[Y] v
  = D_c F[Y] v_c
  + v_T partial_Tgamma F[Y]
  + 0 * v_t.
```

The explicit spectral block is then defined columnwise by

```text
J_c[:,j] = D_c F[Y] e_j,
```

and the full matrix by

```text
J_static[:,0:3n] = J_c,
J_static[:,3n]   = J_Tgamma,
J_static[:,3n+1] = 0.
```

The explicit-matrix and independent-directional paths must agree:

```text
J_static v = D F[Y] v.
```

Both must also agree with centered differences of the unchanged original packed RHS on an unchanged discrete branch.

## 4. Passive elapsed-time block

With active variables `x=(c,T_gamma)` and the elapsed accumulator `t`, the matrix has block form

```text
J = [[A, 0],
     [b^T, 0]].
```

Therefore

```text
J e_elapsed = 0,
chi_J(lambda) = -lambda chi_A(lambda)
```

under the Wolfram Language convention `CharacteristicPolynomial[M,lambda]=det(M-lambda I)`.

For a BDF/Newton coefficient `gamma`, the corresponding static matrix identity is

```text
det(I - gamma J) = det(I - gamma A).
```

This proves only that the passive accumulator does not add a determinant obstruction. It does not prove useful conditioning, Newton convergence, or a performance benefit.

## 5. Fixed-branch differentiability domain

The ordinary Jacobian is admitted only while all of the following remain unchanged under the validation perturbation:

- collision support masks;
- interpolation-domain membership;
- matrix-roundoff correction masks;
- strict-open occupation domain `0<f<1`;
- finite positive `T_gamma`;
- the frozen reaction catalogue and quadrature configuration.

A support change is a discrete event, not an ordinary derivative residual. Such a state must be classified separately rather than absorbed into a tolerance.

## 6. Required test matrix

D-080D must pass:

1. exact state/RHS ordering and shape checks;
2. base-RHS reconstruction against the unchanged packed RHS;
3. exact zero elapsed-time input column;
4. basis-column assembly consistency;
5. explicit `J v` versus the independent combined JVP;
6. centered original-RHS ladders for two mixed spectral/temperature directions;
7. exact equilibrium;
8. a controlled weak-collision static state;
9. an exact retained order-60 stiff state, at least through the independent combined directional operator;
10. differentiated first-law and equilibrium restoring-sign inheritance from D-080C;
11. deterministic receipts, plots, and checksums.

The order-60 retained-state gate does not require explicit assembly of all 182 columns. That computation is deferred until its cost and reuse strategy are measured rather than guessed.

## 7. Mandatory adversarial mutations

The original-RHS witness must reject, at minimum:

- matrix transpose;
- electron/muon spectral-column swap;
- omission of the `T_gamma` column;
- reversal of the `T_gamma` column sign;
- injection of a nonzero elapsed-time column;
- exchange of `T_gamma` and elapsed-time input columns;
- exchange of `T_gamma` and elapsed-time output rows.

## 8. Validation thresholds

For the committed order-8 probes:

- explicit matrix versus independent direct JVP: `< 1e-11`;
- thermal centered ladder: best block residual `< 8e-5`;
- equilibrium and weak-tail centered ladders: best block residual `< 4e-4`;
- retained order-60 combined-direction residual: `< 6e-3`;
- base and column assembly residuals: `< 5e-13`;
- exact elapsed column and null action: exactly zero;
- centered-difference slope: between `1.5` and `2.5`;
- every designated mutation: `> max(1e-5, 30 times correct residual)`.

## 9. Claim ceiling

The strongest admissible claim after all gates pass is:

> The explicit fixed-support static square Jacobian of the frozen private comparator has been assembled at the tested order-8 thermal, equilibrium, and controlled weak-collision states, and its action agrees with an independently assembled exact directional derivative and centered differences of the unchanged original RHS. The passive elapsed-time input column is exact structural zero. An exact retained order-60 stiff state has passed a combined-direction discriminator, but an explicit production-order matrix and any solver-level benefit remain unestablished.

## 10. Next admissible node

The next node is not automatic BDF integration. It is a production-order construction/admission study that must determine whether an explicit order-60 Jacobian can be built and reused economically, or whether a batched tangent/JFNK lane is required. Only after that measurement may an optional same-physics BDF-Jacobian experiment be authorized.
