# BD626 — D-080B Full Static `T_gamma` Collision-Column Result

**Date:** 2026-09-02  
**Branch:** `research/d080b-tgamma-collision-column-20260902`  
**Scientific workflow:** `33536621144` — **SUCCESS**  
**Evidence commit:** `f612277c9e1a4681bdf121fc0b9986204afd484e`  
**Evidence tree:** `e0badbb574d3e03ad0b4c80c9181b553fd6acdca`  
**Artifact digest:** `sha256:f4d7cb9aceac0f5834a7634efb9a53f5bec2105069bc0054bf201b69846e65de`

## 1. Result classification

```text
FULL_STATIC_TGAMMA_COLLISION_COLUMN
```

The frozen private comparator's complete static electron/positron collision action has been differentiated with respect to `T_gamma` on unchanged support branches.  The implementation includes moving electron quadrature, exact relativistic kinematics, weak matrix elements, Pauli blocking, moving outgoing-neutrino interpolation/projection, pair annihilation, flavour routing, and differentiated neutrino/electromagnetic energy ledgers.

It does not construct a full RHS column or call an integrator.

## 2. Exact execution record

The authoritative workflow performed:

1. exact Git-object checks for the comparator and predecessor tangent primitives;
2. frozen research dependency installation;
3. source/test compilation;
4. focused non-slow derivative tests;
5. deterministic plot and JSON receipt generation;
6. fail-closed receipt and mutation audit;
7. exact retained-state recovery and SHA-256 verification;
8. retained stiff-state derivative discriminator;
9. one-day direct artifact upload;
10. deterministic generated-evidence commit.

Observed test results:

```text
small-grid/equilibrium suite: 2 passed, 1 deselected
retained-state slow suite:   1 passed, 2 deselected
```

Retained fixture:

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

## 3. Wolfram symbolic closure

A stateless Wolfram Language evaluation returned exact zero residuals for:

- event product rule `D_T(W M C)`;
- electron-logit temperature derivative;
- elastic detailed-balance positive energy-transfer identity;
- pair detailed-balance positive energy-transfer identity;
- differentiated event first law.

The resulting equilibrium sign gates are

```text
elastic:
(dC/dT_gamma)(E1-E3) = G (E4-E2)^2/T_gamma^2 >= 0

pair:
(dC/dT_gamma)(E1+E2) = G (E3+E4)^2/T_gamma^2 > 0
```

on nonzero-energy support.  The corresponding electromagnetic energy-transfer tangent has the opposite sign.

The retained receipt explicitly states that this was a stateless plugin evaluation, not a repository-native Wolfram replay.

## 4. Small-grid thermal-split result

Configuration:

```text
order                         8
y_max                         8
electron radial order         8
incoming polar order          2
final polar order             2
azimuth order                 4
T_cm                          2.00 MeV
T_gamma                       2.05 MeV
```

### 4.1 Collision-action residual ladder

| `epsilon_Tgamma` [MeV] | scaled residual |
|---:|---:|
| `1.0e-2` | `6.9647025e-3` |
| `3.0e-3` | `6.3039020e-4` |
| `1.0e-3` | `7.0078409e-5` |
| `3.0e-4` | `6.3074147e-6` |
| `1.0e-4` | `7.0083195e-7` |

The ratios are consistent with the expected pre-roundoff second-order convergence of a centered difference.  No support or matrix-correction branch changed on the ladder.

### 4.2 Energy-transfer residual ladder

| `epsilon_Tgamma` [MeV] | scaled residual |
|---:|---:|
| `1.0e-2` | `1.3447577e-4` |
| `3.0e-3` | `1.2104305e-5` |
| `1.0e-3` | `1.3449371e-6` |
| `3.0e-4` | `1.2104519e-7` |
| `1.0e-4` | `1.3451422e-8` |

This ladder also follows second-order centered behaviour through the tested range.

### 4.3 Conservation and symmetry

```text
maximum differentiated first-law residual:
5.904140785354764e-17

CP residual:
8.511669472522492e-15

mu/tau residual:
7.176505633695463e-15

base-action reconstruction residual:
0.0

analytic component-sum residual:
3.6697417739742726e-15
```

### 4.4 Support margins

```text
minimum normalized support margin:
2.6697082136230193e-3

minimum normalized Kallen margin:
2.598020334735334e-1
```

These values establish a nonzero local margin only for the tested state and quadrature.  They do not prove global differentiability over a trajectory.

## 5. Exact-equilibrium result

The equilibrium Fermi-Dirac state had collision-action norm

```text
5.870417396613988e-35
```

and the temperature column satisfied

```text
dQ_nu/dT_gamma = +8.606919856943374e-20
dQ_em/dT_gamma = -8.606919856943374e-20
```

with the differentiated first law closing to machine precision.

This is the expected restoring sign: increasing the electromagnetic bath temperature relative to the fixed neutrino/comoving sector transfers energy into neutrinos.

## 6. Component decomposition

Norms relative to the total analytic collision column are:

| component | `||component||/||total||` |
|---|---:|
| moving measure | `3.9443661e-2` |
| moving matrix | `4.2255233e-2` |
| Pauli/occupation | `6.8034543e-1` |
| moving projection | `2.4670549e-1` |
| elastic total | `1.0275293` |
| pair total | `2.6321183e-1` |

The elastic norm exceeding one is not an error.  It shows that elastic and pair contributions partially cancel in the final vector.  The plot therefore supports a decomposition claim, not a ranking by independent positive fractions.

The differentiated energy ledgers split as follows:

| component | `dQ_nu/dT_gamma` | `dQ_em/dT_gamma` |
|---|---:|---:|
| moving energy weight | `7.6623184e-23` | `-7.6623184e-23` |
| moving matrix | `6.1168925e-22` | `-6.1168925e-22` |
| moving measure | `6.0459111e-22` | `-6.0459111e-22` |
| Pauli/occupation | `1.0064456e-19` | `-1.0064456e-19` |

Pauli response dominates the energy-transfer derivative at this state, but measure, matrix, and moving-weight terms are nonzero and load-bearing for a full state-vector derivative.

## 7. Mutation audit

Residuals against the best original-comparator centered witness are:

| mutation | scaled residual |
|---|---:|
| flip Pauli tangent | `1.3606906` |
| omit moving measure | `3.9444313e-2` |
| omit moving matrix | `4.2255886e-2` |
| omit moving projection | `2.4670602e-1` |
| omit elastic block | `1.0275293` |
| omit pair block | `2.5615987e-1` |

Every designated mutation is separated by much more than the best correct residual `7.0083e-7`.  No tested omitted term can be absorbed by the declared numerical tolerance.

## 8. Plot-based CRAG readback

### Correctness

- Both residual ladders exhibit the expected approximately quadratic decrease.
- Analytic and centered energy-transfer curves converge to equal-and-opposite values.
- Component sums reconstruct the total column to roundoff.
- All mandatory mutations are visibly separated from the correct witness.

### Retrieval

The literature supports direct Jacobian evaluation and full-collision/conservation checks as appropriate stiff-neutrino-transport methodology.  It does not independently validate this exact RABBIT quadrature tangent; the original comparator path and centered witness remain the direct reference.

### Augmented scope

The claim survives at:

- exact equilibrium;
- one thermal-split small-grid state;
- one provenance-locked retained stiff-region state.

It has not yet been augmented over a temperature grid, all quadrature orders, a late weak-collision state, or an endpoint trajectory.

### Generation

The plot decomposition predicts that a full RHS temperature column which omits Hubble feedback or the EOS second derivative can pass the collision-only tests while still fail the complete RHS centered difference.  Those terms must therefore be tested in D-080C rather than inferred from D-080B.

### Claim classification

```text
SURVIVES:
full static fixed-support T_gamma collision-action column on tested states

NARROWED:
local differentiability, not global trajectory differentiability

REJECTED:
frozen-kinematics or occupation-only T_gamma derivative

REJECTED:
full RHS, square Jacobian, solver-stall resolution, or speedup claim
```

The PNG files were generated deterministically from the exact arrays above.  The scientific readback is based on those arrays and the plotting source.  Raster typography at journal print size was not separately certified in this run.

## 9. PHYS-MATH audit

### PASS

- event product rule;
- Pauli/logit signs;
- moving quadrature scaling;
- finite-mass elastic kinematics;
- pair-channel structural dependence;
- energy-transfer sign convention;
- differentiated first law;
- natural-unit dimensions;
- detailed-balance restoring sign;
- strict-open occupation and same-branch requirement.

### P1 findings

1. The result is piecewise differentiable; support or matrix-projection boundary states have no single ordinary derivative under the discrete algorithm.
2. Local support margins are not a global trajectory guarantee.
3. The late weak-collision regime has not been admitted.
4. The full RHS quotient and thermodynamic feedback have not been assembled.

### P2 findings

1. The centered ladder has not reached its roundoff upturn; this is acceptable for formula admission but leaves the optimal metrology window incompletely mapped.
2. Finite-temperature QED corrections are outside the frozen comparator.
3. No independent second collision implementation was used.

### P0 findings

None in the frozen D-080B scope.

## 10. PHYS-MATH-CODE audit

### Genuinely fixed

- the D-079 spectral block and D-080A moving-kinematics primitive are combined in the actual frozen event path;
- the pair and elastic channel dependencies are treated separately rather than forced through one false rule;
- the full product-rule decomposition is explicit and machine-reconstructable;
- the original collision action is reconstructed exactly;
- energy and symmetry ledgers accompany the state-vector tangent;
- all centered comparisons are branch-checked;
- a verifier bug that initially included continuous `T_gamma` in the discrete branch signature was diagnosed and removed;
- the retained-state discriminator uses exact provenance and SHA-256.

### P1 findings

1. Research code imports private comparator helpers and therefore requires exact object pinning.
2. The implementation duplicates part of the event assembly for differentiation, creating future drift risk if the frozen comparator changes.
3. No full RHS column or square-Jacobian API exists.
4. No solver-level Newton, order, rejection, step-size, or linear-residual telemetry has been exercised.
5. Only one retained physical direction (`T_gamma`) and one retained state are tested.

### P2 findings

1. Dependency versions are frozen in the dedicated workflow but wheel hashes are not.
2. The failed one-shot branch-signature repair attempts are process evidence, not scientific failures; the final tree contains no repair helper.
3. Plot raster presentation was not given a publication-figure audit.

### P0 findings

None in the final D-080B tree.

## 11. Execution-history honesty

The following failed runs are retained:

1. Intended RED: implementation file absent after tests/workflow were committed.
2. First GREEN attempt: the branch signature incorrectly contained continuous `T_gamma`, making every centered perturbation appear to cross a branch.
3. Two bounded repair-orchestration attempts failed before changing scientific evidence: one exact-text matcher and one staged-path assertion.

The final scientific workflow passed only after the discrete branch signature was made independent of the continuously varied parameter.  No physical tolerance, test threshold, collision coefficient, or support rule was weakened to obtain PASS.

## 12. Claim ceiling and gate status

`G-F10-INDEPENDENT-FLRW` remains **FAIL**.  D-080B supplies derivative evidence only.

Allowed:

> The complete static fixed-support `T_gamma` derivative of the frozen private electron collision action is implemented and certified on the tested equilibrium, thermal-split, and retained stiff states.

Not allowed:

- complete state-space Jacobian;
- BDF/JFNK execution;
- wall-time or speedup;
- stalled-phase completion;
- endpoint or holdout agreement;
- `N_eff` result;
- independent scientific corroboration;
- public-production or publication claim.

## 13. Next DAG node — D-080C

D-080C must construct the complete static RHS `T_gamma` column for

```text
Y = (c[3n], T_gamma, t_elapsed).
```

With

```text
F_c = P_pair/(H q),
F_gamma = [-3(rho_em+p_em) + Q_em/H]/chi_gamma,
F_t = 1/H,
chi_gamma = d rho_em/dT_gamma,
```

and fixed `c,T_cm`, the required formulas are

```text
dH/H = (1/2) chi_gamma dT_gamma/rho_total,

dF_c/dT_gamma
  = (dP_pair/dT_gamma)/(H q)
    - F_c (dH/H)/dT_gamma,

dN_gamma/dT_gamma
  = -3[chi_gamma + dp_em/dT_gamma]
    + (dQ_em/dT_gamma)/H
    - (Q_em/H) (dH/H)/dT_gamma,

dF_gamma/dT_gamma
  = (dN_gamma/dT_gamma)/chi_gamma
    - F_gamma (d chi_gamma/dT_gamma)/chi_gamma,

dF_t/dT_gamma
  = -(1/H) (dH/H)/dT_gamma.
```

D-080C must test:

- exact equilibrium;
- thermal split;
- retained stiff state;
- late weak-collision state;
- original packed-RHS centered ladders;
- omitted collision-column mutation;
- omitted Hubble-feedback mutation;
- omitted EOS-second-derivative mutation;
- energy-transfer sign mutation;
- state-index swap;
- structurally zero elapsed-time input column.

Only after D-080C and a multi-regime square-operator assembly pass may a separate stalled-phase BDF-Jacobian instrument be considered.
