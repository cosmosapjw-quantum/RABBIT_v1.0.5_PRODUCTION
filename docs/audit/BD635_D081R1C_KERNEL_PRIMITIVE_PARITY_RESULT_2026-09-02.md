# BD635 — D-081R1C exact collision-kernel primitive parity

## Verdict

`PASS_WITH_SELECTED_SCALAR_PRIMITIVE_SCOPE`

The Rust implementation matches a deterministic fixture generated directly from the
frozen private Python comparator for five stable Pauli cases, two self-matrix
normalisations, six electron elastic/pair matrix cases, and one event-measure point.
The exact Rust 1.94.1 / Cargo.lock offline vendor was used. Ten focused tests passed.

## Physics contract

The admitted Pauli factor is

```text
(1-f1)(1-f2)f3f4 - f1 f2 (1-f3)(1-f4),
```

evaluated through a cancellation-resistant logit affinity. At detailed balance its
value is zero but its restoring gradient is `(-G,-G,+G,+G)`.

Self matrices use the frozen `K_s=d12*d34` and `K_t=d14*d23`
contractions with coefficients from the exact 27-event global catalogue. Electron
matrices preserve finite electron mass, flavour-dependent weak couplings, CP exchange,
the elastic factor `64 G_F^2`, and pair factor `128 G_F^2`. Event measure parity covers
the frozen `256*pi^4*p1*E2` denominator.

## Direct arXiv reading

The primary methodological checks were read from arXiv:2008.01074,
arXiv:1605.09383, arXiv:1506.05266, and arXiv:2012.02726. They support the
gain-minus-loss/Pauli structure, spin-summed matrix elements, identical-particle
multiplicities, explicit antiparticle treatment, energy-exchange conservation, and
numerical-convergence separation. They do not validate the RABBIT discretisation;
the frozen Python fixture is the implementation oracle.

## Deliberate exclusions

No angular/radial collision integral was assembled in Rust. No modal routing,
retained-state RHS, analytic JVP/Jacobian, PyO3 comparator surface, ODE solver,
performance result, endpoint, or `N_eff` was produced. The legacy folded two-bank
action remains a negative-control substrate, not the six-species oracle.

## Next node

D-081R1D must assemble the full six-species self, elastic, and pair collision action
from these admitted primitives and compare the action, moments, support signature,
CP/flavour symmetries, weighted number/energy conservation, and first-law ledger with
frozen Python outputs before the packed RHS is opened.
