# BD628 — D-080C Full Static `T_gamma` RHS-Column Result

**Date:** 2026-09-02  
**Branch:** `research/d080c-tgamma-full-rhs-column-20260902`  
**Predecessor D-080B head:** `382711d2f7e6e59342390e3189096ebe3a4dc455`  
**Scientific workflow:** `33566702784` — **SUCCESS**  
**Workflow-tested head:** `a44288dbbb85babd7179dd231aa6a595eb4af30b`  
**Deterministic evidence commit:** `e05e05e9a4c7c70253441d697b06004e1ac3c037`  
**Evidence tree:** `64c83cec9b0f71f7c8ccb9010998901fbd72feef`  
**Artifact ID:** `9823417669`  
**Artifact SHA-256:** `2ad041d8bb494957f526b83a47ff7bbdbfdabbc1e569f3f577e2df490318b62d`

## 1. Result classification

D-080C is classified as

```text
FULL_STATIC_TGAMMA_RHS_COLUMN
```

for the frozen private no-QKE comparator on a fixed differentiable support and
matrix-correction branch.

The admitted object is the complete static input column

```text
partial F(c_e,c_mu,c_tau,T_gamma,t_elapsed; N) / partial T_gamma
```

of the original packed RHS at fixed neutrino cloglog coordinates, fixed
`T_cm`, and fixed independent variable `N=log(a)`.

This result does **not** authorize a square Jacobian, BDF/JFNK/Newton use, a
trajectory, stalled-prefix completion, an endpoint, `N_eff`, a speedup, gate
movement, release authority, or publication authority.

## 2. Frozen authority and code-path reality

The workflow verified the following Git-object identities before any numerical
claim:

```text
private comparator:
de44feee0aa484abe26976c7dc34c579643005b5

D-079 original-static-RHS helper:
6bcff2bc5627c0af0ad4df61c908d09e62ffaba5

D-080A EOS/kinematics primitive:
c585d5865fd68a90a04a76ab540b8437fba8cfce

D-080B full collision-column implementation:
78489c43f3046db09d8ba2d96070124ed7b0aa91
```

The D-080C implementation calls the admitted D-080B collision tangent and the
same frozen thermodynamic and packed-RHS paths used by the primal comparator.
No production solver, tolerance, reaction catalogue, trajectory driver,
endpoint, or gate path was modified.

## 3. Mathematical result

### 3.1 Conventions and units

The state is

```text
Y = (c_e[1:n], c_mu[1:n], c_tau[1:n], T_gamma, t_elapsed),
```

with

```text
c = log(-log(1-f)),     0 < f < 1.
```

The comparator uses natural units

```text
hbar = c = k_B = 1.
```

Hence `T_gamma`, `T_cm`, energies, masses, and `H` have MeV units.  The three
blocks of the `T_gamma` input column have different dimensions:

- spectral rows: MeV^-1;
- photon-temperature row: dimensionless;
- elapsed-time output row: MeV^-2.

They are therefore compared with separately dimensionless residuals rather
than a single dimensional Euclidean norm.

### 3.2 Hubble feedback

At fixed `c` and `T_cm`, only the electromagnetic energy density changes under
a `T_gamma` variation.  From

```text
H^2 = (8 pi G_N/3) rho_total
```

one obtains

```text
H_T/H = chi_gamma/(2 rho_total),
chi_gamma = partial rho_em/partial T_gamma.
```

### 3.3 Spectral rows

For CP-paired collision action `P` and cloglog chain `q=df/dc`,

```text
F_c = P/(H q),
F_c,T = P_T/(H q) - F_c (H_T/H).
```

The first term is the D-080B collision-action tangent; the second is Hubble
feedback.  There is no `q_T` term because the input column holds `c` fixed.

### 3.4 Photon-temperature row

Let

```text
N_gamma = -3(rho_em+p_em) + Q_em/H,
F_gamma = N_gamma/chi_gamma.
```

Then

```text
N_gamma,T
  = -3(chi_gamma+p_em,T)
    + Q_em,T/H
    - (Q_em/H)(H_T/H),
```

and

```text
F_gamma,T
  = N_gamma,T/chi_gamma
    - F_gamma chi_gamma,T/chi_gamma,
chi_gamma,T = partial^2 rho_em/partial T_gamma^2.
```

The implementation records the expansion/EOS, collision, Hubble, and
heat-capacity-denominator terms independently.

### 3.5 Elapsed-time row and input column

For the elapsed-time output equation,

```text
F_t = 1/H,
F_t,T = -(1/H)(H_T/H) < 0.
```

The elapsed-time **input** column is exactly

```text
partial F/partial t_elapsed = 0.
```

This was checked both by direct evaluation at `t_elapsed=+-10^40` and by an
exact all-zero constructed column.

## 4. Stateless Wolfram verification

A stateless Wolfram Language evaluation returned

```text
RHS quotient residuals:                  {0,0,0,0}
elapsed-time input column:               {0,0,0}
differentiated first-law residual:       0
restoring sign from Q_em,T < 0:          True
massless rho=a T^4 second derivative:    0
```

The four quotient residuals correspond to the spectral row, photon-temperature
numerator, photon-temperature quotient, and elapsed-time output row.  This is
formula-level corroboration and is not described as a repository-native
Wolfram replay.

## 5. SciSpace literature positioning

The SciSpace search identified the closest peer-reviewed precedents as
precision neutrino-decoupling calculations that combine full collision terms,
plasma thermodynamics, and direct differential-system Jacobians or stiff
integration:

- Froustey, Pitrou & Volpe, JCAP 12 (2020) 015,
  DOI `10.1088/1475-7516/2020/12/015`;
- Akita & Yamaguchi, JCAP 08 (2020) 012,
  DOI `10.1088/1475-7516/2020/08/012`;
- Hannestad et al., JCAP 08 (2015) 019,
  DOI `10.1088/1475-7516/2015/08/019`.

These papers support direct full-collision/plasma-thermodynamic treatment as a
methodological direction.  They do not validate RABBIT's private quadrature,
branch semantics, or packed-RHS implementation; those are tested against the
frozen original code path here.

## 6. TDD execution record

### 6.1 Intended RED

The initial test specification was committed before the implementation.  The
RED workflow passed predecessor identity and environment setup and failed all
four non-slow specifications only because
`scripts.audit._d080c_tgamma_rhs` did not yet exist.

### 6.2 First GREEN candidate and verifier failure

The first implementation reproduced the original packed RHS and matched the
finite-difference derivative, but the mutation audit used one Euclidean norm
across all output rows.  The elapsed-time output row has MeV^-2 units and a
magnitude near `10^20`, so it dominated that dimensional norm and hid omitted
spectral/collision terms.

This was a verifier-metric error, not evidence that the physical derivative was
wrong.  The implementation was left unchanged.  The test and probe were
repaired to compare the spectral block, photon-temperature row, and elapsed
output row separately and to take the maximum of their dimensionless
residuals.

### 6.3 Final GREEN

The repaired non-slow suite returned

```text
4 passed, 1 deselected
```

and the exact retained-state slow lane returned

```text
1 passed, 4 deselected.
```

## 7. Quantitative static results

### 7.1 Thermal-split state

```text
order              = 8
y_max               = 8
T_cm                = 2.00 MeV
T_gamma             = 2.05 MeV
```

The original packed RHS was reconstructed with residual `0.0`, and the seven
atomic analytic components reconstructed the complete input column with
residual `0.0`.

| epsilon_Tgamma [MeV] | block residual | spectral | T_gamma row | elapsed row |
|---:|---:|---:|---:|---:|
| 1.0e-2 | 1.7290336493e-3 | 1.7290336493e-3 | 1.1318965811e-5 | 2.2183906560e-5 |
| 3.0e-3 | 1.5573993591e-4 | 1.5573993591e-4 | 1.0186391386e-6 | 1.9965671286e-6 |
| 1.0e-3 | 1.7305676739e-5 | 1.7305676739e-5 | 1.1318130290e-7 | 2.2184106328e-7 |
| 3.0e-4 | 1.5575237583e-6 | 1.5575237583e-6 | 1.0187282544e-8 | 1.9964809157e-8 |
| 1.0e-4 | **1.7305993362e-7** | **1.7305993362e-7** | 1.1325469224e-9 | 2.2164519637e-9 |

All perturbations remained on the same discrete branch.  Wolfram evaluation of
the first two pairwise convergence exponents gave

```text
1.9993229082402455
1.9999348071979512
```

with the later intervals continuing the same quadratic trend.

The Hubble logarithmic tangent was

```text
H_T/H = 0.5346295229727147 MeV^-1 > 0.
```

### 7.2 Manufactured weak-collision-tail probe

```text
order              = 8
y_max               = 10
T_cm                = 0.45 MeV
T_gamma             = 0.50 MeV
```

| epsilon_Tgamma [MeV] | block residual |
|---:|---:|
| 1.0e-3 | 2.4550462639e-4 |
| 3.0e-4 | 2.2097257721e-5 |
| 1.0e-4 | 2.4552687621e-6 |
| 3.0e-5 | 2.2097580203e-7 |
| 1.0e-5 | **2.4549561368e-8** |

The first two pairwise convergence exponents were

```text
1.9999307853063972
1.9999933623637904.
```

All perturbations remained on the same branch.  This state is explicitly a
manufactured controlled static probe, not retained late-trajectory evidence.

### 7.3 Exact equilibrium

At `T_cm=T_gamma=2 MeV` with exact Fermi-Dirac spectra,

```text
dQ_nu/dT_gamma  = +8.606919856943374e-20
dQ_em/dT_gamma  = -8.606919856943374e-20
first-law residual = 0.0
H_T/H              = 0.5231017159993603 MeV^-1
F_t,T              = -2.9682069952782854e20 MeV^-2
```

The collision tangent therefore has the restoring detailed-balance sign, and
the elapsed-time output tangent has the sign required by increasing `H`.

### 7.4 Exact retained stiff state

The workflow recovered the exact fixture

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

and passed the order-60, `y_max=30` full packed-RHS `T_gamma` discriminator on
the unchanged branch.

## 8. Component readback

Component magnitudes are reported within their own native output block, not
across dimensionally different rows.

| component | absolute component / native-block total |
|---|---:|
| spectral collision | 1.0220033564 |
| spectral Hubble feedback | 0.0222049817 |
| T expansion/EOS | 2.1875259650 |
| T collision | 0.4952643600 |
| T Hubble feedback | 0.0117819353 |
| T heat-capacity derivative | 1.6710083897 |
| elapsed-row Hubble term | 1.0 |

Values larger than one are not fractions.  They expose cancellations among
load-bearing terms.  In particular, the photon-temperature row would be badly
wrong if either the expansion/EOS numerator derivative or the heat-capacity
denominator derivative were omitted.

## 9. Mutation audit

Residual against the best thermal original-RHS centered witness:

| mutation | block-scaled residual |
|---|---:|
| omit collision contribution | 1.0220033536 |
| omit all Hubble feedback | 1.0000000000 |
| omit heat-capacity derivative | 0.6256095619 |
| reverse Q_em,T sign | 0.9905287200 |

The correct residual is `1.7305993362e-7`.  Every designated mutation is
separated from the admitted column by more than six orders of magnitude.

The elapsed-time input-column mutation is separately killed by the exact-zero
contract and by direct independence of the packed RHS from the stored elapsed
value.

## 10. Plot-driven CRAG readback

### Correctness

The plotted arrays show monotone second-order convergence for both the thermal
split and manufactured weak-tail probes.  The separately scaled row plot shows
that the spectral block controls the aggregate error rather than the enormous
dimensional magnitude of the elapsed-time row.

### Retrieval

The plots reproduce exactly the values retained in the machine receipt.  The
quadratic guide is supported by the independently evaluated exponents near two.
No visual claim relies on an unrecorded plotting transformation.

### Augmented checks

The same qualitative convergence survives:

- a different temperature regime;
- a different `y_max`;
- equilibrium sign and conservation checks;
- an exact retained order-60 stiff state;
- omission and sign mutations.

### Generation / prediction

The evidence predicts that a future square-Jacobian audit must use the same
blockwise dimensional scaling.  A raw full-vector norm will systematically
underdiagnose errors in the spectral and photon-temperature blocks whenever the
elapsed-time row dominates.

### Claim status

```text
SURVIVING, WITH NARROWED FIGURE CLAIM
```

The diagnostic semantic claims survive.  The PNGs are not promoted to
publication figures.  The current chat runtime failed to provide an independent
raster decode for 90 mm / 180 mm print-size inspection, so pixel-level overlap,
font-size, and reduced-width legibility remain `MINOR REVISE` rather than
certified.  The data-level and generator-level audit is complete.

## 11. PHYS-MATH AUDIT

### PASS

1. **Definitions:** the differentiated coordinate is `T_gamma`; `c`, `T_cm`,
   and `N` are fixed.
2. **Signs:** `Q_em` uses the frozen comparator sign; equilibrium gives
   `Q_nu,T>0`, `Q_em,T<0`.
3. **Hubble quotient:** `H_T/H=chi_gamma/(2 rho_total)` follows from the
   Friedmann square root.
4. **Photon-temperature quotient:** numerator and heat-capacity denominator
   derivatives are both present.
5. **Elapsed rows:** the output tangent is negative and the input column is
   exactly zero.
6. **Dimensions:** spectral, temperature, and elapsed rows are not mixed in a
   dimensional norm.
7. **Known limit:** `rho=aT^4` gives `rho_TT=12aT^2` exactly.
8. **Conservation:** differentiated neutrino/electromagnetic energy exchange
   closes at equilibrium and in the retained collision audit.
9. **Strict occupation domain:** inherited D-080B strict-open cloglog semantics
   are unchanged.
10. **Fixed-branch honesty:** centered witnesses are admitted only when both
    perturbed signatures equal the analytic branch signature.

### Remaining findings

- **P1:** ordinary differentiability is local to a fixed support and
  matrix-correction branch; no derivative is admitted at branch crossings.
- **P1:** the low-temperature state is manufactured, not a retained late
  trajectory state.
- **P2:** finite-temperature QED plasma corrections are outside the frozen
  comparator and therefore outside this derivative.
- **P2:** only one exact retained stiff state and one `T_gamma` direction are
  tested at order 60.

No P0 PHYS-MATH finding remains.

## 12. PHYS-MATH-CODE AUDIT

### PASS

1. Exact predecessor Git objects are checked before execution.
2. The analytic column is compared with centered differences of the original
   packed RHS, not a rewritten primal proxy.
3. The primal packed RHS is reconstructed exactly.
4. Seven atomic components reconstruct the analytic column exactly.
5. Non-slow equilibrium/thermal/weak-tail tests pass.
6. The exact retained order-60 state is checksum-gated and passes.
7. Hubble, heat-capacity, collision, and sign mutations are independently
   killed.
8. The elapsed stored value is demonstrated not to affect the RHS.
9. Deterministic JSON and plot evidence is committed and artifact-addressed.
10. No production solver path is changed.

### Remaining findings

- **P1:** D-080C imports private audit helpers and therefore depends on exact
  blob pins; future comparator changes require deliberate re-admission.
- **P1:** a full square static Jacobian has not been assembled, so column
  ordering, matrix shape, and multi-direction action are not yet certified.
- **P1:** no BDF/Newton telemetry exists; solver-stall reduction remains
  completely untested.
- **P2:** the derivative assembly restates parts of the primal thermodynamic
  quotient, creating a controlled but real drift risk.
- **P2:** there is no independent second implementation of the full column.
- **P2:** dependency versions are pinned, but wheel hashes are not.
- **P3:** the GitHub runner reports the upstream Node 20 deprecation warning for
  pinned actions; it does not affect the scientific result.

No P0 PHYS-MATH-CODE finding remains.

## 13. Ranked risk ledger

| Priority | Risk | Current disposition |
|---|---|---|
| P0 | Incorrect full static `T_gamma` RHS formula | closed for tested fixed branches |
| P0 | Dimensional residual hides missing physics | closed by blockwise metric |
| P1 | Support/matrix branch crossing | open; must remain fail-closed |
| P1 | Square-Jacobian index/order error | open; D-080D target |
| P1 | No genuine retained late weak-collision state | open |
| P1 | Solver/Jacobian interaction unknown | open; integrator still forbidden |
| P1 | Private-helper drift | controlled by exact Git-object pins |
| P2 | Sparse state/direction coverage | open |
| P2 | No independent implementation | open |
| P2 | No wheel-hash lock | open |
| P3 | Diagnostic PNG print polish | minor revise |
| P3 | Node runtime warning | infrastructure-only |

## 14. Updated DAG state

```text
D-079  spectral c-input RHS JVP              MERGED / ADMITTED
D-080A moving T_gamma kinematics + EOS       MERGED / ADMITTED
D-080B full static collision-action column   MERGED / ADMITTED
D-080C full static original-RHS column        VALIDATED ON BRANCH
D-080D full square static Jacobian            OPENED / NOT STARTED
D-081  stalled-phase BDF-Jacobian probe       FORBIDDEN UNTIL D-080D
```

## 15. Next admissible step: D-080D

Construct the square static operator

```text
J_static = [ J_c   J_Tgamma   0_elapsed ],
```

with output and input ordering

```text
(c_e, c_mu, c_tau, T_gamma, t_elapsed).
```

D-080D must certify:

1. exact shape `(3n+2,3n+2)` and block/index manifest;
2. equality between explicit matrix action and the existing JVP path;
3. reconstruction from every basis column;
4. random and adversarial directional centered differences of the original
   packed RHS;
5. equilibrium, thermal-split, exact retained stiff, and genuine retained late
   weak-collision states;
6. the exact zero elapsed-input column;
7. blockwise dimension-aware residuals;
8. state-index swap, missing Hubble, missing heat-capacity, sign, transpose, and
   zero-column mutations;
9. conditioning and sparsity diagnostics without any solver call.

Only after D-080D closes may D-081 attach the admitted operator to a separate
stalled-prefix BDF/Jacobian instrument.

## 16. Claim ceiling

Allowed:

> The frozen private comparator now has a validated analytic static
> `T_gamma` input column for its complete packed RHS on tested fixed-support
> branches, including collision, Hubble, electromagnetic heat-capacity, and
> elapsed-time-row effects.

Not allowed:

- complete analytic square Jacobian;
- globally smooth Jacobian across support crossings;
- improved BDF/Newton convergence;
- stalled-prefix completion or wall-time reduction;
- endpoint, holdout, or `N_eff` agreement;
- movement of the formal F10 gate;
- production or publication readiness.

## 17. Cost and effectiveness

```text
production physics paths modified:       0
research derivative modules added:       1
focused test modules added:              1
deterministic diagnostic plots:          5
machine receipts/summaries:               2 + 1 symbolic receipt
exact retained state checks:             1
formal gate movement:                    no
solver calls:                            0
blocker reduction:                       full static thermal RHS column closed
cost-effectiveness verdict:              ACCEPT_WITH_LIMITS
```
