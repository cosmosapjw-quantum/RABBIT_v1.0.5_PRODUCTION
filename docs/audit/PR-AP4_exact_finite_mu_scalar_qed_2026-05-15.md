# PR-AP4 Exact Finite-Mu Scalar QED Audit

Date: 2026-05-15

## Scope

This stage adds an opt-in `exact_finite_mu_scalar` QED EOS correction mode for
the electromagnetic plasma thermo entrypoints and staged scalar 3T solve
shells.  It keeps the default finite-mu-scaled isotropic QED mode unchanged.

## Physics Landed

- `rabbit.thermo.qed_eos_exact` accepts signed `mu_e` by replacing the
  zero-chemical-potential occupation sum with
  `f_e(E, mu_e) + f_pos(E, mu_e)`.
- `delta_rho_qed_exact_with_electron_mu(...)` and
  `delta_P_qed_exact_with_electron_mu(...)` expose scalar O(e^2)+O(e^3)
  pressure/energy corrections.
- `qed_delta_rho_with_electron_mu(...)`,
  `qed_delta_pressure_with_electron_mu(...)`,
  `rho_plasma_with_electron_mu(...)`,
  `pressure_plasma_with_electron_mu(...)`,
  `drho_dT_plasma_with_electron_mu(...)`, `hubble_3T(...)`, and
  `coupled_3T_rhs(...)` accept `qed_correction_model="exact_finite_mu_scalar"`.
- `run_augmented_lrs_collisionless_weak_network_3T_solve(...)`,
  `run_augmented_nonlrs_source_collisionless_weak_network_3T_solve(...)`, and
  `run_augmented_nonlrs_nonlinear_collisionless_weak_network_3T_solve(...)`
  accept the same `qed_correction_model` option, pass it through every staged
  Hubble, standard thermo RHS, collision-moment thermo RHS, and charge-neutral
  plasma derivative override, and record `qed_correction_model` plus
  `qed_correction_contract` in result metadata.
- `nudec_coupled` now lazily imports the nu-nu equilibration rate prefactor to
  avoid the direct-import cycle exposed by standalone thermo tests.

## Numeric Smoke

At `T = 0.8 MeV`, `mu_e = 0.2 MeV`:

- `delta_rho_qed_exact = -0.001531317236098999 MeV^4`
- `delta_rho_qed_exact_with_electron_mu = -0.001545790523714566 MeV^4`
- `delta_P_qed_exact = -0.00044680846046272186 MeV^4`
- `delta_P_qed_exact_with_electron_mu = -0.00045775883349570345 MeV^4`

## Boundaries

This is a scalar, isotropic plasma-frame QED EOS mode.  It is not an
anisotropic/tensor QED response, not a default/public full-BBN dispatch mode,
not promotion-grade full-span exact-scalar-QED coupled-solver validation, not
production SMC, and not QKE.

## Verification

- `PYTHONPATH=src pytest -q tests/test_qed_eos_exact_finite_mu.py` passed with
  `4 passed`.
- `PYTHONPATH=src pytest -q tests/test_eos_photon_electron_charge_neutrality.py tests/test_qed_eos_exact_finite_mu.py`
  passed with `14 passed`.
- `PYTHONPATH=src pytest -q tests/test_jax_thermo_provider.py tests/test_species_boltzmann_bridge.py tests/test_isotropic_decoupling_skeleton.py tests/test_nu_nu_3t_equilibration.py tests/test_augmented_typeI_weak_network_3t_solve.py -k "hubble_3T or coupled_3T or electron_mu or charge_neutral"`
  passed with `11 passed, 47 deselected`.
- `PYTHONPATH=src pytest -q tests/test_nu_nu_3t_equilibration.py tests/test_jax_thermo_provider.py tests/test_species_boltzmann_bridge.py tests/test_isotropic_decoupling_skeleton.py tests/test_qed_eos_exact_finite_mu.py tests/test_eos_photon_electron_charge_neutrality.py`
  passed with `42 passed`.
- `PYTHONPATH=src pytest -q tests/test_qed_eos_exact_finite_mu.py tests/test_eos_photon_electron_charge_neutrality.py tests/test_nu_nu_3t_equilibration.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `45 passed, 3 skipped`.
- `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "qed_correction_model or exact_scalar_qed"`
  passed with `4 passed, 52 deselected`.

## Red Test Evidence

The focused exact-scalar QED test first failed during collection because
`qed_delta_pressure_with_electron_mu(...)`,
`delta_rho_qed_exact_with_electron_mu(...)`, and
`delta_P_qed_exact_with_electron_mu(...)` did not exist.
The solve-shell forwarding tests later failed because the staged 3T solvers did
not yet accept `qed_correction_model`.
