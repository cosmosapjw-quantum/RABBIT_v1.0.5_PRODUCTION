# PR-AP4 Charge-Neutral Total-Derivative Electron Thermodynamics

Date: 2026-05-15

## Scope

This AP4 follow-up upgrades the opt-in charge-neutral finite-mass e-/e+ 3T
thermodynamics path from algebraic `mu_e(T_gamma, X)` evaluation only to a
network-coupled total-derivative photon-bath energy update.

The landed implementation covers the staged LRS, source-only non-LRS, and
nonlinear non-LRS 3T weak/network RHS shells in
`src/rabbit/transport/augmented_typeI_weak_network.py`.

## Physics Change

For `electron_chemical_potential_mode="charge_neutrality"`, each RHS now:

1. Computes the weak/network derivative `dX/dN` from the current augmented
   monopoles and PRIMAT abundance state.
2. Solves the algebraic finite-mass charge-neutral `mu_e` from the current
   positive charge density.
3. Adds the abundance-driven electron charge-asymmetry energy term
   `d rho_em / d(n_e- - n_e+) * d[n_b sum_i Z_i X_i/A_i]/dN` to the
   photon/electron plasma energy equation.
4. Records the final energy-derivative contribution and equivalent
   temperature-RHS correction on the 3T result object.

This keeps the implementation inside the existing algebraic charge-neutrality
closure while making the charge-neutral e-/e+ EOS feedback sensitive to live
network evolution rather than only to the instantaneous abundance state.

## Boundaries

- This is not an independent evolved electron charge-asymmetry state variable.
- This is not an exact finite-density or tensor thermal-QED calculation.
- This is not public forward dispatch.
- This is not a promotion-grade full-span collision-coupled BBN run.
- QKE remains out of scope.

## Evidence

Focused AP4 tests were run in red/green order.  The red run failed because the
plasma override helper had no `dX_dN` input and the charge-neutral contracts
still described the algebraic-only path.  After implementation, the same three
focused tests passed.

Direct helper smoke at `T_gamma = 0.8 MeV`, `eta = 1e-4`, and a
neutron-to-proton test derivative:

- `mu_e = 5.407278613347443e-05 MeV`
- static `plasma_dT_base_dN = -0.7857261820041459`
- total-derivative `plasma_dT_base_dN = -0.78572618257828`
- `plasma_charge_asymmetry_drho_dN_MeV4 = 2.1108534095619534e-09`
- `plasma_charge_asymmetry_dT_correction_dN = -5.741341845659509e-10`

A real short physical charge-neutral solve was not retained as a default gate:
the existing per-RHS charge-neutral root-solve path remains too expensive for
smoke-scale CI, so this PR uses direct helper evidence plus deterministic
fake-solve path tests.
