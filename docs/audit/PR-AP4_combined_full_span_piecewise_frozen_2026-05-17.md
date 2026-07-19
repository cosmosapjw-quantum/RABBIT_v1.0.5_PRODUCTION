# PR-AP4 Combined Full-Span Piecewise-Frozen Source Refresh

Date: 2026-05-17

## Scope

This note records the AP4/AP65 `piecewise_frozen` source-update policy for the
combined angular+`pstf_radial` full-span candidate gate.  The goal is to make
real source refresh executable at diagnostic scale without claiming full
live-RHS BBN support.

## Implementation

- `AugmentedNonLRSCombinedFullSpan3TCandidateGateSpec` accepts
  `source_update_policy="piecewise_frozen"` plus explicit
  `source_update_subspan_ends`.
- The gate runs each subspan through the existing AP65 combined nonlinear
  3T/network solve with `frozen_initial_state` source policy.
- After each subspan, Sigma, A modes, temperatures, and the network abundance
  vector are handed off into the next subspan.
- Source-evaluation accounting is accumulated across subspans and checked
  against `max_pstf_radial_source_evaluations`.
- After the final subspan, the gate recomputes the combined AP41+AP6 source
  once at the true terminal state for diagnostics, so `*_final` source
  observables are terminal-state quantities rather than last-refresh values.
- The initial landing was restricted to fixed electron chemical potential.
  `PR-AP4_combined_full_span_piecewise_charge_neutrality_2026-05-17.md`
  supersedes that limitation by adding evolved charge-asymmetry state handoff
  for charge-neutral e-/e+ mode.

## Numeric Evidence

A real CLI run passed with:

```text
N_span=(0, 1e-4)
source_update_subspan_ends=(5e-5, 1e-4)
method=Radau
max_pstf_radial_source_evaluations=8
max_nfev=5000
```

Terminal values:

```text
source_update_subspan_count = 2
source_evaluations = 2
source_diagnostic_evaluations = 1
source_diagnostics_last_refresh_N = 5e-05
source_diagnostics_terminal_N = 0.0001
source_diagnostics_terminal_minus_last_refresh_N = 5e-05
nfev = 1397
T_gamma_final = 0.7999214320646801
H_rate_s_final = 0.43146274191619854
Xn_final = 0.1300096609235235
collision_dA_abs_max_final = 0.00026527543966857903
radial_offdiagonal_nunu_pair_max_abs_energy_residual_final = 1.1784345878675453e-19
```

A longer nonuniform run also passed with:

```text
N_span=(0, 1e-3)
source_update_subspan_ends=(1e-6, 1e-4, 1e-3)
method=Radau
max_pstf_radial_source_evaluations=8
max_nfev=10000
```

Terminal values:

```text
source_update_subspan_count = 3
source_evaluations = 3
source_diagnostic_evaluations = 1
source_diagnostics_last_refresh_N = 0.0001
source_diagnostics_terminal_N = 0.001
source_diagnostics_terminal_minus_last_refresh_N = 0.0009
nfev = 47
T_gamma_final = 0.7992146796753832
H_rate_s_final = 0.43068978083203713
Xn_final = 0.13005888045355307
collision_dA_abs_max_final = 0.000265067738217371
radial_offdiagonal_nunu_pair_max_abs_energy_residual_final = 7.326834993749698e-20
```

## Boundaries

This is an operator-split source-refresh diagnostic.  It does not implement
full live-RHS collision coupling over BBN spans, QKE, production SMC
validation, or public dispatch.

## Negative Evidence

An equal-width three-chunk `N_span=1e-3` Radau probe became unstable with
non-finite network values.  Longer piecewise ladders therefore require explicit
nonuniform subspan control; the `(1e-6, 1e-4, 1e-3)` ladder above is the
current stable smoke-scale long-row evidence, not a promotion-grade full-BBN
claim.
