# PR-AP4 Non-LRS Charge-Neutral Evolved State Audit

Date: 2026-05-15

## Scope

This stage extends the AP4 charge-neutral finite-mass e-/e+ 3T EOS path from
the LRS shell to the source-only non-LRS and nonlinear non-LRS 3T shells.

## Physics Landed

- Charge-neutral source-only non-LRS and nonlinear non-LRS 3T solves now append
  an `electron_charge_asymmetry_density_MeV3` state only when
  `electron_chemical_potential_mode="charge_neutrality"`.
- The initial state is
  `eta * n_gamma(T_gamma) * sum_i Z_i X_i / A_i`.
- Each RHS call uses the evolved charge-asymmetry state as the target passed to
  `charge_neutral_electron_chemical_potential(...)`.
- The state derivative is
  `3 n_Q dT_gamma/(T_gamma dN) + eta n_gamma d(sum_i Z_i X_i/A_i)/dN`, using the
  live photon-temperature derivative and PRIMAT network derivative.
- Result metadata records the state history, final value, and
  `charge_neutral_positive_charge_density_evolved_v1`.

## Boundaries

This closes the staged independent charge-asymmetry state blocker across the
LRS, source-only non-LRS, and nonlinear non-LRS 3T shells.  It does not promote
public dispatch, production SMC, promotion-grade full-span BBN, exact
finite-`mu_e`/tensor QED response, or QKE.

## Verification

- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "charge_asymmetry_state"`
  passed with `2 passed, 20 deselected`.
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "charge_neutral or electron_mu or charge_asymmetry_state"`
  passed with `5 passed, 17 deselected`.
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`
  passed with `22 passed`.
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `79 passed, 3 skipped`.

## Red Test Evidence

The new focused source-only and nonlinear non-LRS tests first failed because
both charge-neutral result contracts remained `not_evolved` and the non-LRS
state vectors did not carry an evolved charge-asymmetry slot.
