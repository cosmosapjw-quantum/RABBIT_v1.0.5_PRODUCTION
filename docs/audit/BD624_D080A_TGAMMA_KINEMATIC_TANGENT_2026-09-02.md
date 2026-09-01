# BD624 / D-080A — T-gamma moving-kinematics and electromagnetic-EOS tangent

Date: 2026-09-02  
Status: `PARTIAL_STATIC_ADMISSION`  
Claim ceiling: smooth `T_gamma` quadrature, elastic-kinematics, mapped-basis, and QED-off electromagnetic-EOS tangent only

## 1. Exact predecessor and scope

This record is stacked on the D-079 static c-only derivative head
`c09d361249bd1d7a64212e1f8a092c2b9f9f5740`.

The frozen predecessor objects are:

- private comparator Git blob:
  `src/rabbit/decoupling/_independent_noqke.py`
  = `de44feee0aa484abe26976c7dc34c579643005b5`;
- D-079 collision JVP Git blob:
  `scripts/audit/_d079_collision_jvp.py`
  = `591a64702c58a2de265fb88636f186e2d1b7e019`;
- D-079 RHS JVP Git blob:
  `scripts/audit/_d079_rhs_jvp.py`
  = `6bcff2bc5627c0af0ad4df61c908d09e62ffaba5`.

The actual trajectory state is

```text
(c_e[order], c_mu[order], c_tau[order], T_gamma, elapsed_time)
```

with

```text
T_cm(N) = T_start exp(-N).
```

Consequently, `T_cm` is not a state coordinate and does not contribute a
square-Jacobian input column. The missing non-spectral columns are the
`T_gamma` input column and the structurally zero elapsed-time input column.

No production runtime, comparator, collision catalogue, tolerance, grid,
event, endpoint, or gate registry was changed.

## 2. Load-bearing code-path correction

A prior working model treated `T_gamma` as changing electron occupations while
leaving the elastic collision geometry fixed. The actual comparator does not
have that dependency structure.

Its incoming-electron half-line quadrature is temperature scaled:

\[
 p_2=T_\gamma\frac{r}{1-r},\qquad
 w_2=T_\gamma\frac{w_r}{(1-r)^2}.
\]

Therefore a `T_gamma` perturbation changes all of the following in the elastic
`nu+e -> nu+e` channels:

1. incoming electron momentum and quadrature measure;
2. incoming electron energy;
3. total invariant mass and the Kallen function;
4. boosted outgoing momenta and energies;
5. matrix-element Minkowski dot products;
6. outgoing-neutrino interpolation locations;
7. the phase-space factor;
8. the discrete support decision if a threshold is crossed.

A frozen-kinematics `T_gamma` derivative would therefore differentiate a
different numerical operator. D-080A explicitly differentiates the moving
quadrature and smooth fixed-support kinematics before any collision assembly is
attempted.

## 3. Conventions and dimensions

The private comparator uses natural units internally:

\[
\hbar=c=k_B=1.
\]

Momenta, masses, energies, and temperatures have dimension MeV. Hence

- `dp/dT`, `dE/dT`, and `dw/dT` for the linearly scaled one-dimensional
  quadrature weight are dimensionless;
- `ds/dT` and the derivative of a Minkowski dot product have dimension MeV;
- `d lambda/dT` has dimension MeV^3;
- `d(k/sqrt(s))/dT` has dimension MeV^-1.

The metric-sign convention entering the comparator dot products is the usual
energy-first `E_a E_b - p_a dot p_b` convention. No spacetime metric or Bianchi
geometry is involved in this flat-FLRW comparator stage.

## 4. Analytic temperature tangent

For a fixed Gauss--Legendre coordinate `r`,

\[
\frac{\partial p_2}{\partial T_\gamma}=\frac{p_2}{T_\gamma},
\qquad
\frac{\partial w_2}{\partial T_\gamma}=\frac{w_2}{T_\gamma}.
\]

For

\[
E_2=\sqrt{p_2^2+m_e^2},
\]

\[
\frac{\partial E_2}{\partial T_\gamma}
 =\frac{p_2^2}{T_\gamma E_2}.
\]

With a massless incoming neutrino of momentum `p1` and incoming polar cosine
`mu`,

\[
s=m_e^2+2p_1(E_2-p_2\mu),
\]

so

\[
\frac{\partial s}{\partial T_\gamma}
 =2p_1\left(
 \frac{p_2^2}{T_\gamma E_2}-\frac{p_2\mu}{T_\gamma}
 \right).
\]

For final masses `m3,m4`,

\[
\lambda=s^2+m_3^4+m_4^4
 -2sm_3^2-2sm_4^2-2m_3^2m_4^2,
\]

\[
\frac{\partial\lambda}{\partial T_\gamma}
 =2(s-m_3^2-m_4^2)\frac{\partial s}{\partial T_\gamma},
\]

and on a smooth supported branch

\[
k_*=\frac{\sqrt\lambda}{2\sqrt s},
\qquad
\frac{1}{k_*}\frac{\partial k_*}{\partial T_\gamma}
 =\frac{1}{2\lambda}\frac{\partial\lambda}{\partial T_\gamma}
 -\frac{1}{2s}\frac{\partial s}{\partial T_\gamma}.
\]

The implementation propagates these derivatives through the exact boost used
by the comparator and through every returned matrix-element dot product.

The boolean support mask is not differentiated. A perturbation that changes
that mask is a typed non-smooth state and cannot be admitted by widening the
finite-difference tolerance.

## 5. Electromagnetic EOS tangent

For the QED-off photon plus equilibrium electron/positron plasma, the primal
comparator supplies

\[
\chi_\gamma=\frac{\partial\rho_{\rm em}}{\partial T_\gamma}.
\]

D-080A retains that exact primal heat-capacity path and uses the zero-chemical-
potential identity

\[
\frac{\partial p_{\rm em}}{\partial T_\gamma}
 =\frac{\rho_{\rm em}+p_{\rm em}}{T_\gamma}.
\]

It also evaluates

\[
\frac{\partial^2\rho_{\rm em}}{\partial T_\gamma^2}
\]

by differentiating the comparator's dimensionless integral for
`drho_dtemperature`. This second derivative is needed later for the
`T_gamma` evolution row.

## 6. Wolfram closure

The stateless Wolfram evaluation of
`scripts/audit/d080a_symbolic_checks.wl` returned exact zero for:

- `dp2`;
- `dE2`;
- `ds`;
- `d lambda`;
- `dk_*`;
- half-line quadrature-weight scaling;
- the Fermi--Dirac temperature derivative;
- the second electromagnetic-energy integrand derivative;
- the complete quotient rule for the future `T_gamma` RHS row.

The machine-readable transcription is
`docs/audit/artifacts/d080a/wolfram_symbolic_receipt.json`.
This was a stateless plugin evaluation, not a native Wolfram invocation inside
GitHub Actions.

## 7. TDD and executable result

The RED run first verified all predecessor object IDs and then failed only
because the new implementation module did not exist.

The GREEN implementation adds:

- `modal_basis_derivative` for moving outgoing-neutrino interpolation points;
- the exact temperature tangent of the incoming-electron half-line rule;
- the full fixed-support elastic kinematic tangent;
- QED-off electromagnetic EOS first and second temperature derivatives;
- a structurally zero elapsed-time input-column contract.

Focused result:

```text
4 passed
```

The deterministic receipt at the final evidence commit reports:

| quantity | value |
|---|---:|
| best moving-kinematics residual | `3.682855771233863e-11` |
| best EOS residual | `1.5358912625172127e-09` |
| minimum normalized support margin | `2.6758977636325144e-03` |
| minimum normalized Kallen margin | `8.332592638827941e-01` |
| all finite-difference samples retained support | `true` |
| frozen-p2 mutation residual | `1.0` |
| flipped-E2-sign mutation residual | `1.9999999999980516` |
| omitted-weight-scale mutation residual | `1.0` |

The epsilon ladder is

```text
1e-2, 3e-3, 1e-3, 3e-4, 1e-4 MeV.
```

The kinematic residual sequence is

```text
1.8575648641e-7
1.6718010755e-8
1.8576485403e-9
1.6715382950e-10
3.6828557712e-11
```

and the EOS residual sequence is

```text
1.5338130382e-5
1.3804485373e-6
1.5338326485e-7
1.3805076418e-8
1.5358912625e-9.
```

Both show the expected centered-difference second-order regime over the tested
window; the last kinematic point begins to deviate mildly from ideal scaling,
consistent with the approach to roundoff.

## 8. Plot-driven CRAG audit

### Correctness

The plotted residual ladders decrease by approximately the square of the
step-size ratio throughout the pre-roundoff window. The three load-bearing
mutations are separated from the correct derivative by order-unity residuals.

### Retrieval

The symbolic chain rule agrees with the stateless Wolfram calculation. The
numerical strategy is consistent with the neutrino-decoupling literature in
which direct system Jacobians are used to control stiff integration, but this
record does not borrow an external collision formula or infer correctness from
speedup literature.

### Augmentation

The current plot probes one elastic configuration:

```text
T_gamma = 2.05 MeV
p1      = 3.2 MeV
m_e     = 0.51099895 MeV
angular orders = (2,2,4)
electron radial order = 8
```

It does not establish uniform validity over the complete momentum grid,
thermal history, event catalogue, or retained stalled-region state.

### Generation

The result predicts that a correct full `T_gamma` collision column must include
at least three separately visible contributions:

1. moving electron quadrature and phase-space measure;
2. moving matrix-element invariants and outgoing interpolation points;
3. Fermi--Dirac blocking derivatives.

Omitting any one is expected to produce a finite-difference residual rather
than a harmless performance difference.

### Claim classification

- surviving: smooth moving-kinematics and QED-off EOS tangent at the tested
  fixed-support state;
- narrowed: no uniform-support or all-event claim;
- rejected: frozen elastic kinematics as a qualifying `T_gamma` derivative;
- rejected: any solver, completion-time, endpoint, or gate claim.

## 9. PHYS-MATH audit

### PASS

- derivative signs for `p2`, `E2`, `s`, Kallen, and boost chains;
- natural-unit dimensions;
- QED-off thermodynamic identity;
- centered-difference convergence;
- fixed-support branch honesty;
- finite output and strict positive temperature/mass domains.

### P1

- only the elastic `mass3=0, mass4=m_e` kinematic branch is implemented;
- support differentiability is demonstrated at one state, not uniformly;
- the complete electron matrix-element derivative is not yet assembled;
- Pauli occupation and moving interpolation tangents are not yet combined into
  an event rate.

### P2

- the massless-electron and high-temperature analytic limits are not separate
  regression fixtures;
- QED corrections are outside this comparator and therefore outside the
  derivative;
- no equilibrium/common-temperature full-collision null has been evaluated.

## 10. PHYS-MATH-CODE audit

### Genuinely fixed

- the hidden moving-quadrature dependency has been reconstructed from the
  actual executable path;
- RED-first tests were observed before implementation;
- the implementation is frozen to exact predecessor Git objects;
- deterministic plots and receipts are committed;
- dependency versions are frozen in the dedicated workflow;
- provenance uses the Git blob object ID rather than an incorrect raw-file
  SHA-1.

### Remaining P1

- no `T_gamma` derivative of the collision action exists yet;
- no `T_gamma` derivative of the full RHS exists yet;
- no retained physical-state test exists for this column;
- only one target momentum and temperature were probed;
- the implementation duplicates the primal boost algebra and therefore needs
  an equation-to-code drift test whenever the comparator blob is intentionally
  changed.

### Remaining P2

- the research workflow uses private comparator functions and types;
- raster layout was generated and byte-preserved, but the scientific verdict
  comes from the exact plotted arrays and receipt rather than from typography;
- no wall-time or memory claim is admitted.

## 11. Updated DAG

```text
D-077 equivalence-lane authority                    CLOSED
D-078 generic logit/cloglog derivative contract     CLOSED
D-079 c-only physical collision/RHS JVP              DRAFT PR / STATIC PASS
D-080A-1 moving electron quadrature + kinematics     PASS at one static state
D-080A-2 electromagnetic EOS tangent                 PASS at one static state
D-080B full T_gamma collision-action column          OPEN
D-080C full T_gamma RHS column                       BLOCKED by D-080B
D-080D square static Jacobian admission              BLOCKED
D-081 stalled-phase BDF discriminator                FORBIDDEN NOW
endpoint/holdout/gate reconsideration                FORBIDDEN NOW
```

## 12. Next admissible step

The next single step is **D-080B: assemble the full `T_gamma` collision-action
column on static states**.

It must combine, channel by channel:

- moving quadrature and kinematics from this record;
- electron matrix-element tangents;
- electron Fermi--Dirac logit tangents;
- moving outgoing-neutrino modal interpolation;
- pair-annihilation blocking tangents;
- flavour and neutrino/antineutrino multiplicities;
- neutrino/electromagnetic energy-transfer tangent ledgers.

Admission requires:

1. same-support finite-difference ladders;
2. collision-output and energy-ledger agreement;
3. first-law tangent closure;
4. sign, normalization, moving-measure, matrix, interpolation, multiplicity,
   flavour-row, and omitted-event mutation kills;
5. at least equilibrium, thermal-split, and retained-stiff fixtures.

Only after D-080B passes may the EOS/Hubble/quotient terms be combined into the
full RHS column.
