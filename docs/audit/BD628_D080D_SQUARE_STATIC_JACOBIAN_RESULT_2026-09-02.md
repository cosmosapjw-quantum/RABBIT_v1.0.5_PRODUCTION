# BD628 — D-080D Square Static-Jacobian Result and Dual Audit

**Date:** 2026-09-02  
**Branch:** `research/d080d-square-static-jacobian-20260902`  
**Scientific workflow:** `33579946767` — **SUCCESS**  
**Evidence commit:** `0bdac06b1e0d8c63818ea1206e48f53f4692bd83`  
**Evidence tree:** `d547ec6de2fca720f52ff0eacc0174e4a013ff40`

## 1. Result classification

```text
EXPLICIT_SQUARE_STATIC_JACOBIAN
```

D-080D closes the fixed-support static assembly node. It does not close a production-order construction-cost node or a solver-admission node.

## 2. TDD and execution history

The work was performed contract-first:

1. A RED test suite specified state ordering, square shape, exact passive column, matrix/direct-JVP agreement, original-RHS ladders, multi-regime checks, and adversarial mutations.
2. Workflow `33579299677` failed for the intended reason: `scripts.audit._d080d_static_jacobian` did not exist. Predecessor identities, dependency installation, and test compilation passed.
3. The minimal assembly implementation was added.
4. GREEN workflow `33579525817` passed `5 passed, 1 deselected`.
5. Deterministic matrices, receipts, plots, exact retained-state recovery, the slow discriminator, checksum validation, and the Wolfram receipt were added.
6. Scientific workflow `33579946767` passed every step and committed the deterministic evidence.

No frozen comparator, trajectory, production solver, tolerance, reaction catalogue, quadrature, endpoint, or gate-registry path was modified.

## 3. Explicit matrix results

At order 8 the packed state has

```text
3*8 + 2 = 26
```

coordinates. Explicit `26 x 26` matrices and `25 x 25` active blocks were assembled for:

- thermal split: `T_cm=2.0 MeV`, `T_gamma=2.05 MeV`, `y_max=8`;
- exact equilibrium: `T_cm=T_gamma=2.0 MeV`, `y_max=8`;
- controlled weak-tail state: `T_cm=0.45 MeV`, `T_gamma=0.50 MeV`, `y_max=10`.

For all three matrices:

```text
base reconstruction residual       = 0
column assembly residual            = 0
elapsed-time input-column norm      = 0
elapsed null-action norm            = 0
Newton active-block residual        = 0
Newton final-column residual        = 0
```

The maximum residual between the explicit matrix action and the independently assembled combined directional JVP over all committed regimes was

```text
2.056051346578248e-15.
```

## 4. Original-RHS directional ladders

### 4.1 Thermal mixed direction 1

| epsilon | dimension-aware residual |
|---:|---:|
| `3e-3` | `1.7109286544179555e-4` |
| `1e-3` | `1.9010061902804180e-5` |
| `3e-4` | `1.7109142251978609e-6` |
| `1e-4` | `1.9010912874814737e-7` |

Fitted convergence slope:

```text
1.9999892675490227
```

### 4.2 Thermal mixed direction 2

| epsilon | dimension-aware residual |
|---:|---:|
| `3e-3` | `5.252208168305251e-5` |
| `1e-3` | `5.835577650040430e-6` |
| `3e-4` | `5.252140163750890e-7` |
| `1e-4` | `5.835278247229578e-8` |

Fitted convergence slope:

```text
2.000020241192658
```

Both ladders stayed on the same discrete branch and show the expected second-order centered-difference convergence.

### 4.3 Exact equilibrium

```text
matrix/direct-JVP residual: 6.421533698495348e-16
best original-RHS residual:  4.821570363351343e-8
convergence slope:            2.000002393
```

Inherited equilibrium energy-transfer checks are

```text
dQ_nu/dT_gamma = +8.606919856943374e-20
dQ_em/dT_gamma = -8.606919856943374e-20
first-law tangent residual = 0
```

### 4.4 Controlled weak-tail state

```text
matrix/direct-JVP residual: 2.056051346578248e-15
best original-RHS residual: 4.672064299535787e-8
convergence slope:           2.000011765
```

This is a controlled static probe, not retained late-time trajectory evidence.

## 5. Exact retained stiff-state discriminator

The workflow recovered the exact order-60, `y_max=30` state from:

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

fixture:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

The independent combined spectral/`T_gamma` directional derivative was compared with the original packed RHS at `epsilon=2e-4`:

```text
combined block residual:  2.789203857384459e-7
spectral residual:        2.789203857384459e-7
T_gamma-row residual:     1.9970409899604735e-9
elapsed-row residual:     1.8886516744448004e-12
same T_gamma branch:      true
```

No explicit `182 x 182` retained-state matrix was assembled. This distinction is deliberate and load-bearing.

## 6. Adversarial mutation ledger

Residuals against the best unchanged-original-RHS witness:

| Mutation | Residual |
|---|---:|
| transpose | `1.0326770292350311` |
| swap electron/muon spectral column | `1.576920545328798e-1` |
| omit `T_gamma` column | `4.6198910580522196e-2` |
| flip `T_gamma` column | `9.239772647652596e-2` |
| inject nonzero elapsed column | `7.107517439287529e-2` |
| swap `T_gamma`/elapsed input columns | `7.107517439287529e-2` |
| swap `T_gamma`/elapsed output rows | `1.0073227926835113` |

The correct residual is `1.9010912874814737e-7`; every mutation is separated by many orders of magnitude.

## 7. Wolfram symbolic closure

Stateless Wolfram Language evaluation returned exact zeros for:

```text
column assembly residual      {0,0,0}
elapsed input-column residual {0,0,0}
Newton determinant residual   0
characteristic factor residual 0
active action residual        {0,0}
```

For

```text
J = [[A,0],[b^T,0]],
```

this independently confirms

```text
J e_elapsed = 0,
chi_J(lambda) = -lambda chi_A(lambda),
det(I-gamma J) = det(I-gamma A).
```

The receipt is explicitly labelled a stateless plugin evaluation, not a repository-native Wolfram replay.

## 8. Literature comparison from SciSpace

SciSpace retrieval identified Froustey, Pitrou and Volpe, *Neutrino decoupling including flavour oscillations and primordial nucleosynthesis*, JCAP 12 (2020) 015, DOI `10.1088/1475-7516/2020/12/015`, as the closest methodological precedent: that work reports direct computation of the differential-system Jacobian together with averaged oscillations to accelerate stiff neutrino-decoupling integration.

This supports the project ordering

```text
collision derivative -> full RHS columns -> square Jacobian -> solver experiment,
```

but does not validate the present implementation or establish a speedup.

SciSpace also retrieved Hannestad, Hansen, Tram and Wong, *Active-sterile neutrino oscillations in the early Universe with full collision terms*, JCAP 08 (2015) 019, DOI `10.1088/1475-7516/2015/08/019`. Its emphasis on failures caused by collision approximations supports the present insistence that any future optimization be tested against the same original collision operator, tolerance, and output rather than a reduced surrogate.

## 9. PHYS-MATH audit

### P0

None found within the fixed-support static scope.

### P1

1. The ordinary Jacobian is not defined across a discrete support or matrix-correction branch transition; no generalized derivative has been admitted.
2. A full raw-matrix condition number is not an invariant diagnostic because the state blocks carry different dimensions and the passive elapsed column enforces structural singularity.
3. The retained order-60 test validates a combined direction, not every basis column or an explicit production-order matrix.
4. The controlled weak-tail state is not a genuine retained late-time trajectory state.

### P2

1. No second independent implementation assembles the entire square matrix.
2. Only two deterministic mixed directions are used in the explicit order-8 original-RHS ladder.
3. No production-order matrix-construction cost, cache-reuse rate, or memory profile has been measured.

### Verdict

```text
PASS_WITH_SCOPE
```

The mathematical assembly and structural-zero claims are supported. Conditioning, support-crossing, and solver conclusions are not.

## 10. PHYS-MATH-CODE audit

### Equation-to-code mapping

- spectral basis columns: `evaluate_c_only_rhs_jvp`;
- full thermal input column: `evaluate_tgamma_rhs_column`;
- independent mixed-direction oracle: `evaluate_static_rhs_direction_jvp`;
- explicit assembly: `assemble_static_jacobian`;
- unchanged original-RHS witness: `evaluate_static_rhs_from_packed_state`;
- passive-block exposure only: `static_newton_matrix`.

### P0

None found in the committed test and workflow paths.

### P1

1. `assemble_static_jacobian` recomputes the expensive collision JVP independently for each spectral basis vector. The implementation is correct but not yet a viable production-order construction strategy.
2. The current trajectory core calls `solve_ivp(method="BDF")` without `jac=`, so the D-080D matrix is not connected to the solver.
3. SciPy BDF accepts an explicit/callable dense or sparse Jacobian, whereas the currently scalable admitted object is a directional JVP; a matrix-free JVP cannot simply be passed to the existing call without a solver-path change.
4. No same-physics paired trajectory has compared internal numerical differentiation against the analytic matrix.

### P2

1. The explicit order-8 matrix is intentionally dense in storage; no sparsity claim is made.
2. Generated plots and arrays are checksum-gated, but no manuscript-figure typography claim is made.
3. Dependency versions are pinned, but wheel hashes are not.

### Verdict

```text
STATIC_OPERATOR_VALIDATED; PRODUCTION_CONSTRUCTION_AND_SOLVER_ADMISSION_OPEN
```

## 11. Plot-driven adversarial interpretation

The deterministic evidence contains:

- `thermal_directional_residual_ladders.png`;
- `thermal_block_residuals.png`;
- `regime_directional_residuals.png`;
- `mutation_kills.png`;
- `jacobian_structure.png`;
- `column_profiles.png`.

The underlying arrays and machine receipt show:

- second-order convergence in all reported ladders;
- exact structural zero of the elapsed input column;
- matrix/direct-JVP agreement at approximately machine precision;
- strong rejection of transpose, index, sign, and passive-column mutations;
- blockwise rather than dimensionally mixed normalization.

These are scientific diagnostic figures, not yet publication figures. No single-column or double-column legibility claim is made.

## 12. What is genuinely closed

- exact packed ordering;
- full order-8 fixed-support square assembly in three regimes;
- explicit matrix versus independent directional derivative;
- unchanged-original-RHS directional ladders;
- exact passive elapsed-time column;
- symbolic active/passive block identities;
- exact retained order-60 combined-direction discriminator;
- deterministic matrices, plots, receipts, and checksums.

## 13. What remains open

- explicit production-order matrix construction;
- batched/multi-direction tangent evaluation;
- genuine retained late weak-collision state;
- support-crossing semantics;
- solver callback integration;
- BDF/Newton telemetry and same-physics paired runs;
- performance, endpoint, holdout, and gate claims.

## 14. Next DAG node

The correct next node is:

```text
D-080E — production-order Jacobian construction and admission study.
```

It must first measure the cost of one D-079 spectral JVP, project the full 180-column order-60 cost, inspect reuse opportunities in kinematics/matrix elements/base action, and test a batched tangent kernel against the admitted single-direction oracle. An explicit order-60 matrix or BDF callback is authorized only if this study demonstrates a bounded same-physics construction cost.

## 15. One-line conclusion

D-080D closes the mathematics and order-8 explicit assembly of the fixed-support square static Jacobian; the remaining blocker is no longer correctness of the columns but scalable production-order construction and honest solver admission.
