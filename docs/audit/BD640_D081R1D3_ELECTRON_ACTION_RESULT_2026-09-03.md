# BD640 — D-081R1D3 finite-mass electron/positron collision-action parity

## Verdict

`PASS_WITH_ORDER8_SIX_SPECIES_ELECTRON_ACTION_SCOPE`

The Rust 1.94.1 implementation reproduces the frozen Python comparator's complete order-8, `y_max=8` electron/positron collision action for all six explicitly enumerated neutrino and antineutrino species. The admitted source assembles the twelve neutrino-electron/positron elastic events and three electron-positron pair-annihilation events, preserves finite electron mass, flavour- and CP-dependent weak couplings, moving electron half-line quadrature, finite-mass two-body kinematics, outgoing-neutrino spectral interpolation, fifteen family rows, and independent neutrino/electromagnetic energy and H-functional ledgers.

The authoritative exact-head workflow `33730161205` validated commit `bd74c1f62adf9dfcba88590afeb0306b8ff57129`, tree `1b8cea73b534aa0b2f130b0f012d838efd1f2343`. Its exact implementation parent is `b7e82de41d5d62475d2b67d1addfe4ac84125548`, tree `46bbfcc14d4cd661d15c05265cf8794b4ca5224d`. The workflow passed immutable Python-authority replay, Rust 1.94.1, exact offline dependency closure, release compilation, four focused Rust tests, and strict Clippy.

## Frozen authority

```text
D-081R1D2 final predecessor:
59e21cdf18e34cb5cbcc5ccf08f155f5975f29fb

private Python comparator Git blob:
de44feee0aa484abe26976c7dc34c579643005b5

full collision-action fixture Git blob:
c94d2e72a1f8300b7c20c9c793417a5c4a5fa302

electron component fixture Git blob:
b927389e5aa0c11d41d2e63c83b04ae633fc464d

electron authority checker Git blob:
13cd7c8a859425faddf60f3e249300d7e79b542b

electron authority validator Git blob:
74de6fd1f2756d386fff6021d0a0925f7360d28d

Cargo.lock Git blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014

Rust toolchain:
1.94.1

Python oracle environment:
NumPy 2.4.4 / SciPy 1.17.1
```

The immutable validator was executed twice and returned byte-identical JSON. Both the full fixture and the component fixture had zero semantic difference from regenerated authority bytes. The validator's candidate SHA-256 was

```text
ed31d6ea60f383156f13b0f6e40eac3dcf10ca5206424571b670ba5b4b3ed040
```

## Collision operator implemented

The explicit species order is

\[
(\nu_e,\bar\nu_e,\nu_\mu,\bar\nu_\mu,\nu_\tau,\bar\nu_\tau).
\]

The electron catalogue contains

\[
12\ \text{elastic events}+3\ \text{pair events}=15\ \text{events}.
\]

For each quadrature sample,

\[
R=W\,\mathcal M\,\mathcal P,
\]

where

\[
\mathcal P=(1-f_1)(1-f_2)f_3f_4-f_1f_2(1-f_3)(1-f_4)
=L\,\operatorname{expm1}(a),
\]

\[
a=u_3+u_4-u_1-u_2.
\]

The implementation retains

```text
electron mass:              0.51099895 MeV
neutrino action grid:        affine GL8 on [0,8]
electron radial quadrature:  GL48 half-line map scaled by T_gamma
incoming polar quadrature:   GL12
final polar quadrature:      GL12
final azimuth quadrature:    four midpoint nodes
matrix roundoff budget:      1024 ULP
```

For elastic scattering,

\[
\Delta\widehat C_s=R[\phi(y_1)-\phi(y_3)],
\]

\[
Q_\nu^{\rm el}=R(E_1-E_3),\qquad
Q_{\rm em}^{\rm el}=R(E_2-E_4).
\]

For pair annihilation,

\[
\Delta\widehat C_{\nu_\alpha}=R\phi(y_1),\qquad
\Delta\widehat C_{\bar\nu_\alpha}=R\phi(y_2),
\]

\[
Q_\nu^{\rm pair}=R(E_1+E_2),\qquad
Q_{\rm em}^{\rm pair}=-R(E_3+E_4).
\]

Four-momentum conservation gives

\[
Q_\nu+Q_{\rm em}=0
\]

event by event on admitted branches.

## Physics and numerical gates

The focused suite verifies, for exact equilibrium, thermal split, and mu-tau split fixtures:

- all six modal electron-action outputs;
- all six native electron-action outputs;
- separate elastic and pair modal/native actions;
- reconstruction of all fifteen electron-family rows;
- reconstruction `electron = elastic + pair`;
- modal-to-native reconstruction;
- exact-equilibrium near-null behaviour;
- positive neutrino energy transfer for the hotter electromagnetic bath;
- negative and equal electromagnetic energy transfer;
- differentiated first-law closure;
- finite-electron-mass dependence;
- flavour and CP routing, including antineutrino coupling exchange;
- support/domain rejection accounting;
- matrix-roundoff correction accounting;
- independent neutrino/electromagnetic H-functional ledgers;
- fail-closed nonfinite, invalid-temperature, invalid-mass, and invalid-roundoff inputs.

The exact offline test result was

```text
4 passed; 0 failed; 0 ignored; 305 filtered out
focused runtime: 13.64 s
```

The immutable Python authority gives, in the thermal-split fixture,

\[
Q_\nu=+4.764316901390202\times10^{-21},
\]

\[
Q_{\rm em}=-4.764316901390202\times10^{-21}.
\]

## Failure-preserving repair history

### RED

The first RED workflow passed authority and offline-environment gates, then failed three implementation tests with the expected `NotImplemented` result. One fixture-contract test passed. This established that the GREEN implementation was not pre-existing.

### Formatting-only failure

The first GREEN admission reached the Rust stage after exact Python-authority replay, but stopped at `cargo fmt --check`. Rust 1.94.1 formatting was applied without changing physics, fixtures, tolerances, or acceptance thresholds.

### Analytic-null family metrology failure

The second GREEN admission passed release compilation and three of four focused tests. The remaining failure was the equilibrium `nu_e:elastic_minus` family electromagnetic-energy scalar:

```text
Rust:   -9.74496079479193639e-37
Python: -1.66186595850919088e-36
```

Both are floating-association residues of an analytically null family ledger. Total arrays, elastic/pair reconstruction, the thermal restoring signal, the first law, finite-mass checks, and failure semantics had already passed.

The bounded repair did not widen the `5e-9` relative tolerance. It replaced the self-normalisation of an approximately zero scalar by a frozen, nonzero thermal family-energy envelope computed from all authority cases. Nonzero thermal-family signals retain the same relative gate; material mutations remain visible. No physical expression, coefficient, event, state, grid, quadrature, support policy, or fixture byte changed.

## Wolfram checks

Stateless Wolfram Language calculations give exact zero for

- the Pauli-affinity identity residual;
- elastic and pair first-law residuals;
- the neutrino/antineutrino CP-swap involution residual.

For a relative-temperature perturbation

\[
d=\ln(T_\gamma/T_\nu),
\]

the equilibrium affinity derivatives are

\[
\left.\partial_d a_{\rm el}\right|_{d=0}=\frac{E_1-E_3}{T},
\qquad
\left.\partial_d a_{\rm pair}\right|_{d=0}=\frac{E_1+E_2}{T}.
\]

Multiplication by the neutrino energy gained in the corresponding process gives

\[
\frac{(E_1-E_3)^2}{T}\ge0,
\qquad
\frac{(E_1+E_2)^2}{T}\ge0.
\]

This independently fixes the restoring energy-flow sign when the electromagnetic bath is hotter than the neutrino sector. These are formula-level stateless checks, not repository-native Wolfram CI.

## Literature status

The full-collision and flavour-routing structure is consistent with Froustey–Pitrou–Volpe (`arXiv:2008.01074`) and Blaschke–Cirigliano (`arXiv:1605.09383`). Grohs–Fuller (`arXiv:1706.03391`) and Grohs et al. (`arXiv:1512.02205`) motivate retaining finite electron mass and an explicit neutrino/plasma energy and entropy ledger.

Those works do not authorize RABBIT's exact event multiplicities, weak-coupling encoding, finite-dimensional quadrature bytes, support policy, matrix-roundoff policy, modal normalization, or family ordering. These remain frozen-comparator contracts.

## Scope boundary

This node admits only the complete order-8 six-species electron/positron elastic and pair-annihilation action. It does not admit the combined self-plus-electron collision action, retained order-60 packed RHS parity, an analytic Rust JVP or Jacobian, ODE integration, trajectory performance, an endpoint, `N_eff`, public-production authority, publication authority, or movement of `G-F10-INDEPENDENT-FLRW` from `FAIL`.

## Next node

D-081R1D4 must combine the already admitted self action and electron action without changing either component. It must verify

\[
C_{\rm total}=C_{\rm self}+C_{\rm electron},
\]

all component and family reconstructions, total number/energy ledgers, the neutrino/electromagnetic first law, equilibrium null behaviour, thermal restoring flow, mu-tau covariance, support/correction counter addition, and independent failure propagation. Packed RHS construction remains blocked until D-081R1D4 closes.
