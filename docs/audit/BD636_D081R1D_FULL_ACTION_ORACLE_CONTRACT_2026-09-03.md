# BD636 — D-081R1D full six-species collision-action oracle contract

## Status

`PROSPECTIVE_PRE_IMPLEMENTATION_CONTRACT`

This contract supersedes the positive-oracle clause in Task 4 of
`docs/superpowers/plans/2026-09-02-d081r1-rust-f10-comparator.md`.
The legacy two-bank Rust action is a negative-control and regression substrate;
it is not the authority for the exact three-pair/six-species F10 comparator.
The sole positive implementation oracle is the frozen private Python comparator
at Git blob `de44feee0aa484abe26976c7dc34c579643005b5`.

D-081R1D contains no ODE solve, Jacobian, trajectory, endpoint, performance,
`N_eff`, production, publication, or F10-gate authority.

## Frozen predecessor

- D-081R1C head: `5e4adf6af6ae1983ba0a272074503a7d6f469a46`
- D-081R1C tree: `8f29a8331e3e060e18cfa61dd1ccb4baee84ab8f`
- Rust: `1.94.1`
- Cargo.lock Git blob: `a1b5035da5c20712d1a2a4ab077da255ff94a014`
- Exact offline vendor artifact: `9838506413`
- Exact offline vendor outer SHA-256:
  `391b45bbf6446dc9144bc39321723eda18240d7c8751f45887f9492ffc66f78a`

## Mathematical operator

For each event `e` and quadrature point `q`, the scalar signed rate is

```text
R[e,q] = W[e,q] * M[e,q] * C[e,q],
```

where:

- `W` is the frozen target-leg invariant phase-space measure;
- `M` is the admitted nonnegative weak matrix element;
- `C` is the admitted Pauli gain-minus-loss factor
  `(1-f1)(1-f2)f3f4 - f1 f2(1-f3)(1-f4)`.

The exact global self-event catalogue has 27 members and the exact global
electron catalogue has 15 members. The target-directed catalogues with 48 and
18 rows count a different representation and must not be substituted silently.

### Self modal routing

For event legs `(s1,s2,s3,s4)`,

```text
Delta C_hat[s1,n] += R * phi_n(y1)
Delta C_hat[s2,n] += R * phi_n(y2)
Delta C_hat[s3,n] -= R * phi_n(y3)
Delta C_hat[s4,n] -= R * phi_n(y4)
```

with signs `(+,+,-,-)` exactly.

### Electron elastic routing

For target species `s`,

```text
Delta C_hat[s,n] += R * (phi_n(y1) - phi_n(y3)).
```

The neutrino and electromagnetic energy-transfer ledgers are computed
independently from the same event rates:

```text
Q_nu += R * (E1 - E3)
Q_em += R * (E2 - E4).
```

### Pair routing

For `nu_a + antinu_a <-> e- + e+`, the two incoming modal contributions are
routed separately to `target` and `cp_partner(target)`. A posterior multiplicity
factor is forbidden.

## State and grid

The comparator state represented by the oracle fixtures has three independent
pair spectra and not six independent particle/antiparticle spectra:

```text
(c_e[0:n], c_mu[0:n], c_tau[0:n], T_gamma, elapsed_time).
```

The collision action itself is reported on the explicit six-species order

```text
(nu_e, antinu_e, nu_mu, antinu_mu, nu_tau, antinu_tau).
```

The first implementation fixtures use affine Gauss–Legendre order 8 on `[0,8]`.
The retained order-60 `[0,30]` state is a later holdout and cannot be used to tune
R1D thresholds or routing.

## Prospectively frozen oracle cases

1. `equilibrium`
   - `T_cm = T_gamma = 2.0 MeV`;
   - all three pair logits equal `-y`.
2. `thermal_split`
   - `T_cm = 2.0 MeV`, `T_gamma = 2.05 MeV`;
   - all three pair logits equal `-y`.
3. `mu_tau_split`
   - `T_cm = T_gamma = 2.0 MeV`;
   - electron logit `-y`;
   - muon/tau logits `-y +/- 0.02*(1-2y/y_max)`.

The third case is the positive test that the exact three-pair target preserves a
nonzero `mu-tau` antisymmetric direction. The legacy two-bank fold is expected
to erase this direction and is retained only as a negative control.

## Fixture payload

For every case, the deterministic Python fixture must contain exact binary64
bit patterns for:

- grid nodes and weights;
- three pair cloglog rows;
- six-species native `self`, `electron`, and `total` actions;
- six-species modal `self`, `electron`, and `total` actions;
- all nine self-row native actions;
- all fifteen electron-family native actions;
- `Q_em` and every electron-family bath-energy contribution;
- whole-reaction domain rejection count;
- matrix roundoff correction count and largest correction;
- the full diagnostics mapping;
- signed and absolute number/energy moments of self, electron, and total action;
- per-array absolute-contribution envelopes used for cancellation-aware parity.

The generator must run twice byte-identically and bind the Python comparator Git
blob before the fixture can be committed.

## Cross-language parity metrics

For scalar primitive values without cancellation, use exact-bit or the
predeclared ULP/relative threshold from D-081R1C.

For a signed action block `B`, use both:

```text
forward = ||B_rust - B_python|| / max(||B_rust||, ||B_python||, tiny)
```

and the primary cancellation-aware scale

```text
backward = ||B_rust - B_python|| /
           max(||B_rust||, ||B_python||, ||B_absolute_envelope||, tiny).
```

The backward metric is not an operator-norm or trajectory-error bound. Both
values remain in the receipt.

Thresholds must be frozen in the first Rust RED test before inspecting a Rust
full-action result. No threshold widening is allowed after output.

## Required physical gates

Before packed-RHS work opens, all of the following must pass:

- native/modal reconstruction;
- exact species ordering;
- exact event-family decomposition;
- support/rejection and matrix-correction signatures;
- CP equality on CP-symmetric fixtures;
- mu/tau equality on symmetric fixtures;
- nonzero mu/tau antisymmetric response on `mu_tau_split`;
- self weighted number conservation;
- self weighted energy conservation;
- electron elastic target-number conservation;
- pair lepton-asymmetry conservation;
- `Q_nu + Q_em` first-law closure;
- nonnegative event entropy-production core, within a prospectively frozen
  quadrature/Galerkin roundoff allowance;
- fail-closed nonfinite, domain, support, and materially-negative-matrix outcomes.

## Mandatory mutation kills

At least these mutations must fail:

1. one of the self leg signs `(+,+,-,-)`;
2. CP-partner routing;
3. same-sign identical coefficient;
4. one pair-conversion orientation;
5. one self family omission;
6. one elastic family omission;
7. one pair family omission;
8. finite-electron-mass interference sign;
9. event-measure denominator;
10. mapped-basis normalization;
11. mu/tau output-row swap;
12. antineutrino pair multiplicity.

## Decomposition and stop conditions

D-081R1D is split into independently reviewable nodes:

- `R1D0`: freeze the full Python action fixtures and this contract;
- `R1D1`: rematerialize exact affine-grid, spectral, and kinematic foundations
  in the remote Rust lineage, each against compact Python fixtures;
- `R1D2`: implement and admit the 27-event self action;
- `R1D3`: implement and admit 12 elastic plus 3 pair events;
- `R1D4`: combine total action, decomposition, conservation, entropy, support,
  and mutation ledgers;
- `R1D5`: execute the untouched order-60 retained-state holdout.

A failure in a node blocks its successors. No solver, JVP, or Jacobian code may
be added in D-081R1D.

## Claim ceiling

The strongest statement available after `R1D0` alone is:

> The full six-species Python collision-action values and diagnostic ledgers for
> three prospectively fixed order-8 states are frozen as the implementation
> oracle.

It does not establish any Rust full-action parity.
