# PR-AP4 Charge-Neutral Mu Linear Response

Date: 2026-05-15

## Scope

This AP4 follow-up reduces the cost of the staged charge-neutral e-/e+ 3T
thermodynamics path by adding a finite-mass electron charge-asymmetry
susceptibility and using it as the BBN-scale `mu_e` solve.

## Physics Change

The EOS module now exposes
`electron_charge_asymmetry_susceptibility(T_MeV) =
d(n_e- - n_e+)/dmu_e |_{mu_e=0}` from the same Gauss-Laguerre finite-mass
Fermi-Dirac quadrature used by the e-/e+ density functions.  For BBN-scale
charge targets where `|mu_e|` is in the linear-response regime,
`charge_neutral_electron_chemical_potential(...)` returns
`positive_charge_density / susceptibility` without running the full bracketed
bisection solve.  Larger targets still fall back to the existing finite-mass
bracketed solve.

This is a physics-based fast path, not a hard-coded small-`eta` constant.

## Boundaries

- This does not introduce an independent electron charge-asymmetry state.
- This does not change the finite-mass FD EOS definitions.
- This does not promote public forward dispatch or full-span BBN support.
- QKE remains out of scope.

## Evidence

TDD red run:

- `tests/test_eos_photon_electron_charge_neutrality.py::test_charge_asymmetry_susceptibility_matches_finite_difference`
- `tests/test_eos_photon_electron_charge_neutrality.py::test_charge_neutral_mu_uses_linear_response_for_bbn_scale_targets`

failed at collection because `electron_charge_asymmetry_susceptibility` did not
exist.

Green evidence:

- The susceptibility matches a centered finite-difference derivative at
  `T = 0.8 MeV`.
- A BBN-scale `target = 1e-12 MeV^3` returns through the linear-response path
  without calling the expensive asymmetry residual, and reconstructs the target
  to relative `1e-4`; the remaining error is dominated by subtractive
  cancellation in the diagnostic density difference at such small targets.
- The existing `target = 1e-10 MeV^3` charge-neutral solve test remains within
  its previous tolerance.

Real staged solve smoke:

- LRS charge-neutral 3T solve, `N_span=(0, 1e-7)`, `N_q=3`, `N_mu=8`,
  `WeakQuadrature(4, 4)`, `method=RK23`, `rtol=1e-4`, `atol=1e-8`:
  `success=True`, `nfev=10598`, elapsed `42.4977612849907 s`,
  `mu_final=3.2984396489273045e-10 MeV`,
  `electron_charge_asymmetry_drho_dN_MeV4_final=-1.542004531086532e-20`,
  `electron_charge_asymmetry_dT_correction_dN_final=4.1941224989310315e-21`,
  `T_gamma_final=0.7999999214273857 MeV`, and `Xn_final=1e-25`.

LSODA is not yet the smoke default for this path: the same small solve timed
out under the local 60 s cap.  The remaining runtime is dominated by solver RHS
effort rather than by the charge-neutral `mu_e` bracket solve.
