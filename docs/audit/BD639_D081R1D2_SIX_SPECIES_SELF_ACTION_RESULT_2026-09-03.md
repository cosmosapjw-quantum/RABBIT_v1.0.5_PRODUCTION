# BD639 — D-081R1D2 exact six-species neutrino self-action parity

## Verdict

`PASS_WITH_ORDER8_SIX_SPECIES_SELF_ACTION_SCOPE`

The Rust 1.94.1 implementation reproduces the frozen Python comparator's complete order-8, `y_max=8` neutrino self-collision action for all six explicitly enumerated neutrino and antineutrino species. The validated source tree assembles the 27 global reversible self events, applies the frozen four-leg `(+,+,-,-)` routing, produces modal and native actions, reconstructs all nine physical-family rows, preserves weighted number and energy conservation, retains the event-level H-rate/entropy-production sign, and passes the mu–tau exchange and invalid-input gates.

The authoritative exact-head workflow `33713911950` validated commit `06d4c6b78c2b3749f0e60c7984f677c90137befb`, tree `4981c9c5508922b3f9156843200c78493c367f94`. It passed four focused Rust tests, release compilation, formatting, strict Clippy, byte-identical component-fixture regeneration, exact offline dependency closure, and final D-081R1D1 ancestry.

## Frozen authority

```text
D-081R1D1 final predecessor:
3c84da625cf6829637045bcec61bacd5227a535a

private Python comparator Git blob:
de44feee0aa484abe26976c7dc34c579643005b5

full collision-action fixture Git blob:
c94d2e72a1f8300b7c20c9c793417a5c4a5fa302

self-only metrology fixture Git blob:
4a15f26e35f210fb666e27e0f09b40a5a975b280

self-only metrology fixture SHA-256:
9e96e1c74434de420af4a04649248b4992f308b8da90cacb3a1f75f1cf38ee86

Cargo.lock Git blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014

Rust toolchain:
1.94.1

Python oracle environment:
NumPy 2.4.4 / SciPy 1.17.1
```

## Collision operator implemented

For each global event with species legs `(s1,s2,s3,s4)`, the scalar event rate is

\[
R_{e,q}=W_{e,q}\,\mathcal M_{e,q}\,\mathcal P_{e,q},
\]

where the stable Pauli factor is

\[
\mathcal P
=(1-f_1)(1-f_2)f_3f_4
-f_1f_2(1-f_3)(1-f_4)
=L\left(e^a-1\right),
\]

\[
a=u_3+u_4-u_1-u_2.
\]

The modal weak-form routing is

\[
\Delta\widehat C_{s_1,n}\mathrel{+}=R\phi_n(y_1),\qquad
\Delta\widehat C_{s_2,n}\mathrel{+}=R\phi_n(y_2),
\]

\[
\Delta\widehat C_{s_3,n}\mathrel{-}=R\phi_n(y_3),\qquad
\Delta\widehat C_{s_4,n}\mathrel{-}=R\phi_n(y_4).
\]

The catalogue contains 27 global events rather than the separate 48 target-directed diagnostic rows. Ordered pair-conversion orientations are both retained. The six species are never folded into a shared heavy-flavour bank.

## Physics and numerical gates

The focused suite verifies the following against the frozen full-action oracle:

- all six modal self-action outputs;
- all six native self-action outputs;
- reconstruction of the nine physical-family rows;
- modal-to-native reconstruction;
- equilibrium and thermal-split self-action equality at fixed neutrino spectra and `T_cm`;
- a nonzero mu–tau antisymmetric response;
- mu–tau swap equivariance;
- weighted number and energy conservation;
- event H-rate and nonnegative entropy-production interpretation;
- support/domain rejection accounting;
- matrix roundoff correction accounting;
- transactional failure on nonfinite or invalid inputs;
- load-bearing routing, row, normalization, and symmetry mutations.

The exact offline test result was

```text
4 passed; 0 failed; 0 ignored; 301 filtered out
focused runtime: 2.70 s
```

## Root-cause-preserving metrology repair

The first D-081R1D2 run failed only because a self-only Rust rejection counter was compared with the frozen full-collision counter, which includes the electron block. The action arrays and conservation checks had already passed up to that assertion.

The frozen Python comparator gives

| case | self-only rejections | combined rejections | electron contribution by difference |
|---|---:|---:|---:|
| equilibrium | 217,728 | 672,960 | 455,232 |
| thermal split | 217,728 | 677,520 | 459,792 |
| mu–tau split | 217,728 | 672,960 | 455,232 |

The bounded repair does not remove or weaken the assertion. It freezes a component-specific fixture by calling the frozen Python `_assemble_self` authority directly, leaves the original combined fixture unchanged, and checks the decomposition

\[
N_{\rm combined}=N_{\rm self}+N_{\rm electron}.
\]

No physical coefficient, event catalogue, state, grid, quadrature, tolerance, roundoff budget, or acceptance threshold changed.

## Entropy convention

The accumulated quantity

\[
\dot H_e=R\,(u_1+u_2-u_3-u_4)
=-W\mathcal M L\,a(e^a-1)
\]

is the fermionic H-functional rate. Since

\[
a(e^a-1)\ge0,
\]

one has

\[
\dot H_e\le0,\qquad \dot S_e=-\dot H_e\ge0.
\]

This node checks the event-level sign and its projected oracle duality. It does not claim an unconditional trajectory-level discrete H-theorem under arbitrary truncation or time integration.

## Literature status

Direct full-collision and flavour-routing methodology is consistent with Froustey–Pitrou–Volpe (`arXiv:2008.01074`) and Blaschke–Cirigliano (`arXiv:1605.09383`). Hannestad et al. (`arXiv:1506.05266`) and Bennett et al. (`arXiv:2012.02726`) motivate retaining Pauli blocking, momentum redistribution, conservation tests, and a separate discretisation-convergence budget. These works do not authorize RABBIT's exact event multiplicities, coefficients, finite-domain support policy, quadrature bytes, or normalization; those remain frozen-oracle contracts.

## Scope boundary

This node admits only the complete order-8 six-species neutrino self action. It does not admit electron/positron elastic or pair-annihilation action, the combined collision action, retained order-60 packed RHS parity, an analytic Rust JVP or Jacobian, ODE integration, trajectory performance, an endpoint, `N_eff`, public-production authority, publication authority, or movement of `G-F10-INDEPENDENT-FLRW` from `FAIL`.

## Next node

D-081R1D3 must implement only the 12 neutrino-electron/positron elastic events and three electron-positron pair-annihilation events. It must preserve finite electron mass, flavour- and CP-dependent weak couplings, moving electron half-line quadrature, outgoing spectral interpolation, fifteen family rows, neutrino and electromagnetic energy-transfer ledgers, and the differentiated first law. Exact equilibrium, thermal split, mu–tau split, component-specific metrology, and mandatory matrix/routing/sign mutations must pass before the combined action or packed RHS is opened.
