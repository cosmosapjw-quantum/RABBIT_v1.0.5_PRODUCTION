# BD627 — D-080C Full Static `T_gamma` RHS-Column Contract

**Date:** 2026-09-02  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Predecessor D-080B head:** `382711d2f7e6e59342390e3189096ebe3a4dc455`  
**Frozen comparator Git blob:** `de44feee0aa484abe26976c7dc34c579643005b5`  
**D-079 static-RHS helper blob:** `6bcff2bc5627c0af0ad4df61c908d09e62ffaba5`  
**D-080A EOS/kinematics blob:** `c585d5865fd68a90a04a76ab540b8437fba8cfce`  
**D-080B collision-column blob:** `78489c43f3046db09d8ba2d96070124ed7b0aa91`

## 1. Purpose and authority boundary

D-080C constructs and certifies the complete **static input column**

```text
partial F(Y,N) / partial T_gamma
```

of the frozen private no-QKE comparator's original packed RHS, with

```text
Y = (c_e[1:n], c_mu[1:n], c_tau[1:n], T_gamma, t_elapsed).
```

The derivative is taken at fixed neutrino cloglog coordinates `c`, fixed
`T_cm`, and fixed independent variable `N=log(a)`.  It composes the admitted
D-080B collision-action column with the exact Hubble and electromagnetic-EOS
quotient rules already used by the primal packed RHS.

This is a research-only derivative path.  It does not modify the frozen primal
collision operator, trajectory driver, solver configuration, tolerances,
reaction catalogue, grid, endpoint, or gate registry.

This contract does **not** authorize:

- a full square Jacobian;
- BDF/JFNK/Newton use;
- a trajectory or stalled-prefix completion;
- an endpoint or `N_eff` result;
- a speedup or performance claim;
- movement of `G-F10-INDEPENDENT-FLRW` from `FAIL`;
- public-production or publication authority.

## 2. Conventions, state, and units

The cloglog chart is

```text
c = log(-log(1-f)),     0 < f < 1,
q(c) = df/dc = exp(c-exp(c)).
```

`T_cm(N)=T_start exp(-N)` is explicit in the independent variable and is not a
state coordinate.  `t_elapsed` is carried as an output accumulator but the RHS
has no explicit dependence on its value.

The comparator uses natural units

```text
hbar = c = k_B = 1.
```

Thus `T_gamma`, `T_cm`, momenta, masses, energies, and `H` have units MeV.
For the packed `T_gamma` input column:

- spectral rows `partial(dc/dN)/partial T_gamma` have units MeV^-1;
- the photon-temperature row `partial(dT_gamma/dN)/partial T_gamma` is dimensionless;
- the elapsed-time output row `partial(dt/dN)/partial T_gamma` has units MeV^-2.

These blocks must never be collapsed into one unscaled Euclidean residual.
Validation uses separately dimensionless spectral, temperature, and elapsed-row
residuals and reports their maximum.

## 3. Primal packed RHS

Let `P(c,T_cm,T_gamma)` denote the CP-paired collision action, `Q_em` the
comparator's electromagnetic-bath energy-transfer rate, and

```text
chi_gamma = partial rho_em / partial T_gamma.
```

The original packed RHS is

```text
F_c     = P / (H q),
F_gamma = [-3(rho_em+p_em) + Q_em/H] / chi_gamma,
F_t     = 1/H.
```

At fixed `c` and `T_cm`, the neutrino energy density and `q(c)` are fixed.
Only the electromagnetic density changes the Hubble rate.

## 4. Hubble derivative

From

```text
H^2 = (8 pi G_N / 3) rho_total
```

one obtains

```text
H_T/H = chi_gamma / (2 rho_total).
```

`H_T/H` has units MeV^-1 and must be positive for a positive electromagnetic
heat capacity.

## 5. Spectral rows

The exact spectral quotient derivative is

```text
F_c,T = P_T/(H q) - F_c (H_T/H).
```

The first term is the admitted D-080B collision column after CP pairing.  The
second term is Hubble feedback.  The cloglog chain factor has no `T_gamma`
derivative because the input column holds `c` fixed.

## 6. Photon-temperature row

Define

```text
N_gamma = -3(rho_em+p_em) + Q_em/H.
```

Then

```text
N_gamma,T
  = -3(chi_gamma + p_em,T)
    + Q_em,T/H
    - (Q_em/H)(H_T/H),
```

and

```text
F_gamma,T
  = N_gamma,T/chi_gamma
    - F_gamma chi_gamma,T/chi_gamma,

chi_gamma,T = partial^2 rho_em / partial T_gamma^2.
```

The implementation exposes four independently mutable temperature-row pieces:

1. expansion/EOS numerator derivative;
2. collision energy-transfer derivative;
3. Hubble feedback inside `Q_em/H`;
4. heat-capacity denominator derivative.

## 7. Elapsed-time row and input column

For the elapsed-time output row,

```text
F_t,T = -(1/H)(H_T/H).
```

For the elapsed-time **input** column,

```text
partial F / partial t_elapsed = 0
```

exactly.  This is a structural zero, not a small numerical quantity.

## 8. Fixed-branch differentiability

The D-080B collision derivative is ordinary only while the finite quadrature
operator remains on the same discrete support/domain/matrix-correction branch.
Every centered-difference witness must verify the branch signature at both
`T_gamma+epsilon` and `T_gamma-epsilon`.

No claim is made at support crossings.  A future solver-facing implementation
must detect or otherwise account for such piecewise differentiability rather
than silently treating the branch as globally smooth.

## 9. Required test states

The minimum static admission set is:

1. exact equilibrium (`T_cm=T_gamma` and Fermi-Dirac spectra);
2. thermal-split distorted spectra;
3. the exact retained order-60 `creep_1200` state;
4. a clearly labelled manufactured low-temperature weak-collision-tail state.

The fourth state is not claimed to be retained trajectory evidence.  It is a
controlled static regime probe until a genuine late retained trajectory state
is available.

## 10. Original-operator witness

Finite differences must call the original static packed RHS directly:

```text
[F(Y with T_gamma+epsilon)-F(Y with T_gamma-epsilon)]/(2 epsilon).
```

A reimplementation of the primal RHS is not an independent witness.

The correct analytic column must show a stable centered-difference ladder and
remain on the same branch.  Correctness is evaluated blockwise because the
three output blocks have different dimensions.

## 11. Mandatory mutations

At minimum, the following mutants must be killed against the best original-RHS
centered witness:

- omit the collision contribution;
- omit all Hubble-feedback contributions;
- omit `chi_gamma,T` from the denominator derivative;
- reverse the electromagnetic energy-transfer tangent;
- swap the photon-temperature and elapsed-time indices;
- make the elapsed-time input column nonzero.

Mutation thresholds must be stated relative to the correct blockwise residual,
not only to an arbitrary absolute floor.

## 12. Invariants and limits

Required checks include:

- exact base-RHS reconstruction;
- exact atomic-component reconstruction;
- positive `H_T/H`;
- negative `F_t,T`;
- equilibrium restoring collision sign:
  `Q_nu,T>0`, `Q_em,T<0`;
- differentiated first-law closure;
- exact elapsed-time input structural zero;
- finite output and strict-open occupation semantics;
- massless-radiation EOS limit `rho proportional T^4`, so
  `rho_TT=12 a T^2`.

## 13. Symbolic evidence

The stateless Wolfram check evaluates the quotient identities, elapsed-time
structural zero, differentiated first law, restoring sign implication, and the
massless EOS second derivative.  It is formula-level corroboration only and is
not described as a repository-native Wolfram replay.

## 14. Literature positioning

The closest peer-reviewed numerical precedents found in the SciSpace search are
precision neutrino-decoupling calculations that combine full collision terms,
plasma thermodynamics, and direct differential-system Jacobians, notably:

- Froustey, Pitrou & Volpe, JCAP 12 (2020) 015,
  DOI `10.1088/1475-7516/2020/12/015`;
- Akita & Yamaguchi, JCAP 08 (2020) 012,
  DOI `10.1088/1475-7516/2020/08/012`;
- Hannestad et al., JCAP 08 (2015) 019,
  DOI `10.1088/1475-7516/2015/08/019`.

These works motivate direct Jacobian/full-collision treatment.  They do not
validate RABBIT's private finite quadrature, support semantics, or packed RHS;
those require the exact code-path and original-operator witnesses above.

## 15. DAG exit condition

D-080C is complete only after:

- all non-slow static tests pass;
- deterministic plot/JSON evidence passes a fail-closed audit;
- the exact retained order-60 state is recovered by SHA-256 and passes;
- PHYS-MATH and PHYS-MATH-CODE ledgers contain no P0 finding;
- the claim ceiling remains static-RHS-column only.

The next admissible node is D-080D: assemble the full square static Jacobian
from the D-079 spectral block, this `T_gamma` column, and the exact zero elapsed
input column, then validate matrix-vector identity and original-RHS finite
differences in multiple regimes.  Integrator work remains forbidden until that
node closes.
