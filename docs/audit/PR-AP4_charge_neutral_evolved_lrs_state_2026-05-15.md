# PR-AP4 Charge-Neutral Evolved LRS State

Date: 2026-05-15

## Scope

This AP4 follow-up upgrades the staged LRS 3T charge-neutral finite-mass
electron/positron EOS path from a purely algebraic `mu_e(T_gamma, X)` target to
an evolved charge-asymmetry state:

- `electron_charge_asymmetry_density_MeV3` is seeded from
  `eta * n_gamma(T_gamma) * sum_i Z_i X_i / A_i`;
- the LRS 3T RHS uses that state as the charge-neutral `mu_e` target;
- the state derivative includes both photon-temperature dilution/heating and
  the PRIMAT abundance derivative;
- result metadata records the state history, final value, and
  `charge_neutral_positive_charge_density_evolved_v1`.

The fixed-`mu_e` state layout is unchanged.  Non-LRS evolved charge-asymmetry
states, exact finite-`mu_e`/tensor QED, public dispatch, production SMC, and
promotion-grade full-BBN remain out of scope.

## Numeric Smoke

For the deterministic helper check at `T_gamma = 0.8 MeV`, `eta = 1e-4`,
`X = phase1_to_phase2(0.13)`, `dT_gamma/dN = -0.02`, and a neutron-to-proton
test derivative:

- `electron_charge_asymmetry_density_MeV3 = 1.0850368568231983e-05`
- `d(electron_charge_asymmetry_density_MeV3)/dN = 2.304144359748114e-06`
- `n_gamma = 0.1247168800946205`

The solve-path regression uses a deterministic fake `solve_ivp` step to verify
that the LRS 3T state vector carries the evolved charge state and that the
charge-neutral chemical-potential callback receives the evolved state target.

## Verification

- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "charge_asymmetry_state"` -> 2 passed
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "charge_neutral or electron_mu or fixed_electron"` -> 9 passed
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py` -> 30 passed

An attempted real LRS charge-neutral short solve with the finite-mass
charge-neutral path exceeded the intended interactive smoke budget and was
terminated after about two minutes.  That runtime behavior is not promoted into
a pass/fail gate for this stage.
