# BD632 — D-080F cancellation-aware equivalence-metrology amendment

**Date:** 2026-09-02  
**Scope:** metrology repair after a preserved fail-closed run; no equation, grid, tolerance, state, event, support, quadrature, cache, wall-budget, or acceptance-threshold change.

## 1. Preserved failed measurement

The first qualifying attempt, workflow `33592265710`, completed the retained order-60 matrix construction but stopped before artifact publication with

```text
SAME_PHYSICS_EQUIVALENCE_FAILED
```

because the existing dense-direction forward-relative metric reported

```text
maximum prepared-action residual = 9.098989500537355e-10
```

against the prospectively frozen `5e-10` threshold.  That failure remains historical evidence and is not overwritten.

## 2. Independent localization

Diagnostic workflow `33593370075` rebuilt the same `182 x 182` matrix and decomposed the discrepancy without changing the physics path.

The six frozen serial basis columns

```text
0, 59, 60, 119, 120, 179
```

were identical to the non-prepared D-079 oracle in every output block: all reported differences and relative residuals were exactly zero.

Two independent dense spectral directions gave:

```text
direction 0:
  spectral difference norm = 5.729964598796764e-4
  spectral action norm     = 5.521880146442882e5
  forward-relative residual= 1.0376836231250886e-9
  T_gamma-row residual     = 1.8334734443577126e-15
  elapsed-row residual     = 5.085973804365285e-16

direction 1:
  spectral difference norm = 6.445955441189437e-4
  spectral action norm     = 9.780067696228243e5
  forward-relative residual= 6.590910861011251e-10
  T_gamma-row residual     = 4.00325341881832e-15
  elapsed-row residual     = 1.1490036352110576e-16
```

The prepared seal was unchanged and cache miss/entry deltas were zero.  The measured matrix-construction time in that diagnostic was `421.46595270600005 s`.

This evidence rejects a wrong basis column, state mutation, cache growth, thermal-row mismatch, or elapsed-row mismatch as the explanation.  The residual is confined to the floating-point association of many spectral column contributions in a dense matrix action.

## 3. Amended action metric

For an explicit Jacobian `J`, direction `v`, and independently evaluated directional witness `r`, define the contribution vector

```text
c = |J| |v|.
```

For each native-dimensional output block `B`, the action residual is

```text
R_B = ||(Jv)_B - r_B||
      / max(||(Jv)_B||, ||r_B||, ||c_B||, tiny).
```

The total residual is

```text
R_action = max(R_spectral, R_Tgamma, R_elapsed).
```

The spectral block uses the Euclidean norm.  The scalar rows use absolute values.  Heterogeneous dimensions are never mixed.

The contribution scale is a standard backward-error scale for a matrix-vector product.  It is necessary when large signed column contributions cancel in the forward result.  It does not weaken basis-column admission: for `v=e_j`,

```text
|J| |e_j| = |J[:,j]|,
```

so the denominator reduces to the ordinary column magnitude.  The six selected serial-column gate therefore remains unchanged and continues to use the original block-relative metric.

## 4. Frozen gates retained

No numerical threshold is changed.

```text
maximum cancellation-aware prepared-action residual <= 5e-10
maximum selected serial-column residual              <= 5e-10
original-RHS mixed-direction best residual            < 6e-3
measured full-build wall time                         <= 900 s
unique cache bytes                                    <= 2 GiB
```

The legacy forward-relative dense-direction residual is retained in the receipt as a diagnostic and cannot itself move the route.

## 5. Anti-refitting holdout

Because this metrology amendment follows a failed run, the rerun must evaluate the amended action metric not only on the two original D-080F directions but also on two newly frozen holdout directions that were not inspected in workflows `33592265710` or `33593370075`.  The route uses the maximum across all four directions.

Synthetic RED tests were committed before implementation and demonstrated:

1. ordinary forward relative error can false-fail a near-cancelling action;
2. the contribution-scaled metric remains small for the same exact columns;
3. a basis direction reduces to the original column-relative scale;
4. a one-percent column mutation remains strongly visible;
5. invalid shapes and nonfinite values fail closed.

The initial RED workflow `33594219783` failed for the intended reason—absence of the implementation—and the subsequent GREEN run passed all four specifications.

## 6. Wolfram role

A stateless Wolfram Language check verifies the scalar triangle bound

```text
|sum_j J_ij v_j| <= sum_j |J_ij v_j|,
```

the basis-direction reduction, and the contrasting near-cancellation scalings

```text
forward-relative      ~ eta/delta,
contribution-scaled   ~ eta/(2-delta).
```

This is formula-level corroboration only.  The order-60 values require executable evidence.

## 7. Claim ceiling

A successful rerun may establish only that, on the frozen retained state and one CI host, an explicitly built order-60 matrix passes selected-column equality and cancellation-aware matrix-action equivalence under a sealed prepared-state contract.

It still cannot establish solver convergence, trajectory completion, speedup of BDF, endpoint or `N_eff` agreement, F10 gate movement, or publication readiness.
