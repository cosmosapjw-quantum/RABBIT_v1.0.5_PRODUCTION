# BD638 — D-081R1D1 exact Rust action-foundation parity

## Verdict

`PASS_WITH_ORDER8_SELECTED_KINEMATIC_SCOPE`

Exact Rust 1.94.1 with the frozen 174-package offline Cargo vendor reproduced the
frozen Python comparator's order-8 affine Gauss–Legendre grid on `[0,8]`, strict
cloglog chart, mapped orthonormal Legendre basis, modal coefficients,
interpolation, modal products, native reconstruction, angular/electron rules, and
selected self, elastic, and pair kinematic batches. The authoritative exact-head
workflow `33702547853` passed 11/11 focused tests, release check, formatting, and
strict Clippy.

## Root-cause-preserving repair

The initial spectral failures were not a basis, flattening, or physics error. The
generic Rust Newton–Legendre rule differed from NumPy 2.4.4 `leggauss(8)` by at
most `8.881784197001252e-16` in mapped nodes and
`2.936539900133539e-14` in mapped weights. For profiles that are exactly degree
one in the mapped coordinate, those differences contaminated analytically zero
higher modes and their reconstructed interpolation.

The bounded repair freezes the Python comparator's exact binary64 node and weight
bits only for `(order=8, y_max=8)`. Every other grid remains on the generic Rust
quadrature path. No physical coefficient, collision catalogue, tolerance,
acceptance threshold, state convention, or gate changed.

## Mathematical interpretation

The thermal and mu/tau-split fixture logits are degree-one functions of the mapped
coordinate. Their Legendre coefficients for modes `n>=2` vanish exactly in the
continuum. Therefore the observed high-mode signal was numerical-operator identity
drift, not physical spectral structure. The repair makes the finite-dimensional Rust
comparator use the same frozen quadrature operator as its Python authority.

## Scope boundary

This node admits reusable foundations only. It does not assemble `W*M*C`, route a
six-species collision action, evaluate the packed RHS, construct a JVP/Jacobian,
call an ODE solver, establish performance, produce an endpoint or `N_eff`, or move
`G-F10-INDEPENDENT-FLRW` from `FAIL`.

## Literature status

Direct reading of arXiv:2008.01074 confirms the standard full collision
gain-minus-loss structure and direct differential-system Jacobian strategy.
arXiv:1605.09383 supplies the flavour/spin collision-term framework, while
arXiv:1506.05266 and arXiv:2012.02726 motivate retaining full collision physics and
separating implementation parity from discretisation/convergence uncertainty.
These papers do not authorize RABBIT's exact grid bytes, event catalogue, or
normalization; those remain frozen-oracle contracts.

## Next node

D-081R1D2 must implement only the complete 27-event six-species neutrino self action.
It must preserve `(+,+,-,-)` leg routing and compare six native/modal outputs, all
nine physical-family rows, support/correction signatures, weighted number and energy
conservation, entropy production, CP/mu-tau symmetries, the nonzero mu-tau
antisymmetric response, and mandatory mutations against the frozen Python full-action
fixture. Electron processes, packed RHS, Jacobian, and solver work remain blocked.
