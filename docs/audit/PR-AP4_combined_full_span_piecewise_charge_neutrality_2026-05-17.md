# PR-AP4 Combined Full-Span Piecewise Charge-Neutrality Handoff

Date: 2026-05-17

## Scope

This note records the AP4/AP65 follow-up that makes the
`piecewise_frozen` combined angular+`pstf_radial` source-refresh path work with
`electron_chemical_potential_mode="charge_neutrality"`.  The goal is to use
the already-landed finite-mass e-/e+ charge-neutral 3T state in the piecewise
diagnostic gate instead of treating charge-neutral mode as fixed-only.

## Implementation

- The source-only and nonlinear non-LRS 3T weak/network solve shells accept an
  optional `initial_electron_charge_asymmetry_density_MeV3` when
  `electron_chemical_potential_mode="charge_neutrality"`.
- AP4/AP65 piecewise source refresh now carries
  `electron_charge_asymmetry_density_MeV3_final` from one subspan into the
  next subspan's Hubble/RHS initialization and source payload.
- The combined terminal source diagnostic accepts the terminal charge state and
  terminal electron chemical potential so final source observables are
  evaluated from the final charge-neutral bath state.
- The radial collision bridge honors an explicit source-payload
  `electron_chemical_potential_MeV` in charge-neutral mode when it is supplied
  by the evolved charge state, while preserving algebraic charge-neutral
  fallback for ordinary calls.
- Source payload helpers also forward the private
  `_electron_chemical_potential_MeV` override so an explicitly evolved
  zero-`mu_e` charge-neutral state is not confused with the public default
  `0.0` sentinel.
- The full-span gate records
  `source_update_charge_asymmetry_state_handoff=1` for charge-neutral
  piecewise rows.

## Numeric Evidence

A real CLI run passed with:

```text
N_span=(0, 1e-4)
source_update_subspan_ends=(5e-5, 1e-4)
method=Radau
electron_chemical_potential_mode=charge_neutrality
max_pstf_radial_source_evaluations=8
max_nfev=10000
```

Terminal values:

```text
source_update_subspan_count = 2
source_update_charge_asymmetry_state_handoff = 1
source_evaluations = 2
source_diagnostic_evaluations = 1
nfev = 8256
electron_chemical_potential_MeV_final = 3.298132792363573e-10
electron_charge_asymmetry_density_MeV3_final = 6.61672994731945e-11
T_gamma_final = 0.7999214313698554
H_rate_s_final = 0.43146274153219083
Xn_final = 0.13000851100280264
collision_dA_abs_max_final = 0.00026527544839372085
radial_offdiagonal_nunu_pair_max_abs_energy_residual_final = 1.1784345878675453e-19
```

## Boundaries

This is an operator-split charge-neutral source-refresh diagnostic.  It does
not implement fully live collision-source evaluation at every RHS call over BBN
spans, QKE, production SMC validation, public dispatch, or promotion-grade
full-BBN evidence.

## Negative Evidence

A charge-neutral `max_nfev=5000` run reached the valid terminal source path and
kept the source budget passing, but failed only the solve-effort cap with
`nfev=8256`.  The current smoke gate therefore uses `max_nfev=10000`.
