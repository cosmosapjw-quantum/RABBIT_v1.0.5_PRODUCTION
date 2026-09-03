# BD625 — D-080B Full Static `T_gamma` Collision-Column Contract

**Date:** 2026-09-02  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Predecessor head:** `14cda90236d5885988ac0af7d036f147bb9855c9`  
**Frozen comparator Git blob:** `de44feee0aa484abe26976c7dc34c579643005b5`  
**D-079 tangent primitive blob:** `668f3fab76ffc3ad7f29335a79fcd5daf47d429e`  
**D-080A kinematic primitive blob:** `c585d5865fd68a90a04a76ab540b8437fba8cfce`

## 1. Purpose and authority boundary

D-080B differentiates the complete **static electron/positron collision action** of the frozen private no-QKE comparator with respect to the electromagnetic bath temperature `T_gamma`, while holding the neutrino complementary-log-log spectra and `T_cm` fixed.

The differentiated operator is the same finite-dimensional quadrature/Galerkin operator used by the primal comparator.  It retains:

- the frozen reaction catalogue and flavour routing;
- Pauli gain-minus-loss signs;
- finite electron mass;
- the `T_gamma`-scaled half-line electron quadrature;
- exact two-body kinematics on a fixed discrete support branch;
- channel-dependent weak matrix elements;
- moving outgoing-neutrino interpolation points;
- modal/Galerkin reconstruction;
- neutrino and electromagnetic energy-transfer ledgers;
- the comparator's strict-open occupation and fail-closed semantics.

This contract does **not** authorize a full RHS `T_gamma` column, a square Jacobian, a `solve_ivp` call, BDF `jac=`, a trajectory, an endpoint, a performance claim, or gate movement.

## 2. State, independent variable, and units

The trajectory state is

```text
Y = (c_e[1:n], c_mu[1:n], c_tau[1:n], T_gamma, t_elapsed),
```

where

```text
c = log(-log(1-f)),
0 < f < 1.
```

The independent variable is `N = log(a)`.  The comoving temperature is explicit,

```text
T_cm(N) = T_start exp(-N),
```

and is not a state coordinate.  The elapsed-time input column is therefore structurally zero.

The private comparator uses natural units

```text
hbar = c = k_B = 1.
```

Consequently:

- `T_gamma`, `T_cm`, momenta, energies, and `m_e` have units MeV;
- `G_F` has units MeV^-2;
- the primal collision action has its comparator-defined MeV rate dimension;
- differentiation with respect to `T_gamma` removes one power of MeV;
- all cloglog, logit, occupation, support masks, and branch signatures are dimensionless.

## 3. Event-product derivative

For one quadrature contribution, write

```text
I_event = W * M * C,
```

where `W` is the phase-space/quadrature measure, `M` is the weak matrix element, and `C` is the Pauli gain-minus-loss factor.  On one differentiable support branch,

```text
dI_event/dT_gamma
  = (dW/dT_gamma) M C
  + W (dM/dT_gamma) C
  + W M (dC/dT_gamma).
```

For outgoing-neutrino modal projection, an additional moving-basis contribution appears:

```text
D_T [rate * phi_j(y_out(T))]
  = (D_T rate) phi_j(y_out)
  + rate phi'_j(y_out) D_T y_out.
```

The implementation reports four native-action components:

1. moving measure;
2. moving matrix element;
3. Pauli/occupation tangent;
4. moving projection.

Their sum must reproduce the total analytic column to roundoff.

## 4. Pauli tangent and detailed balance

For

```text
G = (1-f1)(1-f2) f3 f4,
L = f1 f2 (1-f3)(1-f4),
C = G-L,
a = u3+u4-u1-u2,
u_i = log(f_i/(1-f_i)),
```

the stable exact tangent is

```text
dC = C d(log L) + G da.
```

At detailed balance, `C=0`, but

```text
dC = G da
```

is generally nonzero.  A zero derivative merely because the primal collision value vanishes is forbidden.

For an equilibrium electron/positron logit

```text
u_e = -E_e/T_gamma,
```

where both `E_e` and `T_gamma` may vary,

```text
du_e/dT_gamma
  = -(1/T_gamma) dE_e/dT_gamma
    + E_e/T_gamma^2.
```

## 5. `T_gamma`-scaled incoming-electron quadrature

For a fixed Gauss-Legendre coordinate `r` on the half-line map,

```text
p_2 = T_gamma r/(1-r),
w_2 = T_gamma w_r/(1-r)^2.
```

Thus

```text
dp_2/dT_gamma = p_2/T_gamma,
dw_2/dT_gamma = w_2/T_gamma.
```

With

```text
E_2 = sqrt(p_2^2 + m_e^2),
```

```text
dE_2/dT_gamma = p_2^2/(T_gamma E_2).
```

The derivative is propagated through:

- `s` and the Kallen discriminant;
- center-of-momentum energies and momentum;
- the Lorentz boost;
- final energies and momenta;
- phase space;
- the six Minkowski dot products;
- finite-mass interference terms;
- matrix-roundoff projection on its unchanged branch;
- outgoing-neutrino spectral locations.

## 6. Elastic and pair channel split

### 6.1 Elastic `nu + e^± <-> nu + e^±`

The incoming electron quadrature and the entire kinematic path move with `T_gamma`.  All four product-rule components are active.

At exact detailed balance and with the comparator's gain convention,

```text
(dC/dT_gamma) (E1-E3)
  = G (E4-E2)^2/T_gamma^2 >= 0.
```

The corresponding electromagnetic transfer tangent is nonpositive, and the event-level first law must close.

### 6.2 Pair `nu + antinu <-> e^- + e^+`

In the frozen comparator, the incoming neutrino momentum rule is scaled by `T_cm`; therefore the pair kinematics, measure, and matrix element are independent of `T_gamma` at fixed `T_cm`.  Only the final electron/positron logits and Pauli factor vary.

At detailed balance,

```text
(dC/dT_gamma) (E1+E2)
  = G (E3+E4)^2/T_gamma^2 > 0
```

on nonzero-energy support.

### 6.3 Neutrino self interactions

The self-interaction block contains no electromagnetic-bath object and has a structural zero `T_gamma` column.

## 7. Energy-transfer contract

Let positive `Q_nu` denote energy transferred into neutrinos and `Q_em` denote the comparator's electromagnetic-bath transfer.  The primal event construction obeys

```text
Q_nu + Q_em = 0.
```

The differentiated ledger must obey

```text
dQ_nu/dT_gamma + dQ_em/dT_gamma = 0
```

on every admitted fixed-support branch.

The implementation separates rate-derivative and kinematic-weight contributions.  The first-law residual is normalized by

```text
max(|dQ_nu/dT_gamma| + |dQ_em/dT_gamma|, tiny).
```

## 8. Branch and nondifferentiability semantics

A branch signature contains only discrete choices:

- support masks;
- outgoing-neutrino domain masks;
- matrix-roundoff correction masks;
- grid and quadrature orders;
- fixed `T_cm` and `m_e` identities.

The continuous differentiated value `T_gamma` must not be included in the signature.  Otherwise every nonzero centered perturbation would be falsely labelled a branch crossing.

A centered comparison is admissible only if both perturbed evaluations have the same branch signature as the base.  A support or clipping change is classified as a nondifferentiable discrete state, not as finite-difference noise and not as a reason to widen tolerances.

## 9. Admission fixtures

D-080B requires:

1. **Exact equilibrium:** `T_gamma=T_cm` and all pair spectra are Fermi-Dirac.
2. **Thermal-split small-grid state:** unequal `T_gamma` and `T_cm`, flavour-asymmetric spectral perturbation, low quadrature order for bounded audit cost.
3. **Retained stiff/creep state:** the exact `state_1200.npz` bytes from diagnosis commit `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`, SHA-256 `c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380`.

## 10. Required tests and mutation kills

A qualifying result must pass:

- exact predecessor Git-object checks;
- compile and import checks;
- equilibrium collision-null and restoring-sign tests;
- same-branch centered-difference ladders for the native collision action;
- the same ladder for `dQ_nu/dT_gamma` and `dQ_em/dT_gamma`;
- differentiated first-law closure;
- CP and mu/tau symmetry residuals;
- retained stiff-state local discriminator;
- exact base-action reconstruction;
- exact component-sum reconstruction.

Mandatory mutations are:

- Pauli tangent sign flip;
- omitted moving measure;
- omitted moving matrix element;
- omitted moving interpolation/projection;
- omitted elastic block;
- omitted pair block.

Each mutation must separate from the original comparator-centered witness by its prospectively declared threshold.

## 11. Literature role

The literature establishes the broader methodological context, not the RABBIT-specific derivative:

- Froustey, Pitrou, and Volpe, *Neutrino decoupling including flavour oscillations and primordial nucleosynthesis*, JCAP 12 (2020) 015, DOI `10.1088/1475-7516/2020/12/015`: direct evaluation of the differential-system Jacobian is part of an efficient full-collision calculation.
- Hannestad, Tram, and Wong, *Active-sterile neutrino oscillations in the early Universe with full collision terms*, JCAP 11 (2015) 035, arXiv:`1506.05266`: full collision terms expose failures in approximate scattering, momentum redistribution, and Pauli blocking.
- Blaschke and Cirigliano, *Neutrino quantum kinetic equations: The collision term*, Phys. Rev. D 94 (2016) 033009, arXiv:`1605.09383`: collision-term formalism and matrix-valued kinetic structure.
- Akita and Yamaguchi, *A precision calculation of relic neutrino decoupling*, JCAP 08 (2020) 012, DOI `10.1088/1475-7516/2020/08/012`: precision neutrino-decoupling numerics and controlled collision treatment.

None of these sources supplies the exact derivative of RABBIT's private half-line quadrature, finite event catalogue, support masks, and modal interpolation.  Those bytes and their tests remain the direct authority.

## 12. Claim ceiling

After this contract passes, the strongest allowed claim is:

> The complete static fixed-support derivative of the frozen private comparator's electron collision action with respect to `T_gamma` is implemented and locally certified at equilibrium, a thermal-split state, and one retained stiff-region state.

The following remain forbidden:

- full RHS `T_gamma` column;
- full square Jacobian or `LinearOperator`;
- BDF/JFNK integration;
- stalled-phase completion or speedup;
- endpoint, holdout, or `N_eff` result;
- independent scientific corroboration;
- movement of `G-F10-INDEPENDENT-FLRW`;
- public-production or publication authority.
