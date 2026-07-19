# Backend Capability Matrix (v1.0.0)

**Auto-generated from `backend_capabilities.py`. Do not edit manually.**

Historical tier labels preserve scoped evidence. Only `auto`/`scipy` have active runtime authority; JAX endpoint dispatch names are retired and retained JAX entries are non-dispatchable component metadata.

## Dispatch table

| `backend=` | Resolves to | Tier | Weak mode | Notes |
|---|---|---|---|---|
| `"auto"` | `scipy_typeI_reference` | **canonical** | live_f0_cl0_cl2 | Regression-locked SciPy Type-I reference path |
| `"scipy"` | `scipy_typeI_reference` | **canonical** | live_f0_cl0_cl2 | Regression-locked SciPy Type-I reference path |

## Canonical (1)

| Key | Backend | CL | Weak mode | Teff | Notes |
|---|---|---|---|---|---|
| `scipy_typeI_reference` | scipy | 3 | live_f0_cl0_cl2 | no | Regression-locked SciPy Type-I reference path |

## Candidate (10)

| Key | Backend | CL | Weak mode | Teff | Notes |
|---|---|---|---|---|---|
| `scipy_typeI_tier2_per_species` | scipy | 3 | live_f0_cl0_cl3 | no | SciPy Type-I characteristic tier-2 per-species collision pat |
| `scipy_typeI_tier3_weak_budget` | scipy | 3 | live_f0_cl0_cl3 | no | SciPy Type-I CL3 weak-budget surface on top of the transitio |
| `jax_typeI_full_boltzmann_tier3_preflight` | jax | 3 | live_f0_cl0_cl3 | no | JAX tier-3 full-Boltzmann incomplete-decoupling preflight su |
| `jax_classA_geometry` | jax | 0 | not_applicable | no | JAX Class A geometry for all 6 types (WH formalism) |
| `jax_classA_characteristic` | jax | 3 | live_f0_cl0_cl3 | no | v3 |
| `jax_classB_driver` | jax | 0 | born | no | JAX Class B BBN driver (candidate) |
| `jax_tilted_bbn` | jax | 3 | cl0_cl3_equilibrium_fd_optional_boosted_l012 | no | JAX tilted BBN runner (candidate) |
| `jax_tilted_full_coupled` | jax | 3 | cl0_cl3_equilibrium_fd_optional_boosted_l012 | no | Phase γ-3 (Plan §2 |
| `jax_weak_cl3_kernel` | jax | 3 | live_f0_cl0_cl3 | yes | CL3 weak rate kernels (recoil + weak magnetism + K_{r,ℓ}) |
| `jax_rodas5p_solver` | jax | 0 | not_applicable | no | JAX-native Rodas5P (8-stage, order 5(4), SciML convention) |

## Substrate (4)

| Key | Backend | CL | Weak mode | Teff | Notes |
|---|---|---|---|---|---|
| `jax_typeI_liveweak_cl3_tier2_teff_candidate` | jax | 3 | live_f0_cl0_cl3 | no | Deprecated legacy Teff closure |
| `jax_typeI_augmented_pstf_noqke_staging` | jax | 3 | live_monopole_lrs_cl3_moment_input_nonlrs_s2_moment_input | no | Catalog-only augmented-PSTF Type-I no-QKE staging with no pu |
| `jax_classA_driver` | jax | 3 | live_f0_cl0_cl3 | no | JAX Class-A driver via κ-cascade reduced ℓ_max=2 PSTF |
| `jax_curved_hierarchy` | jax | 0 | not_applicable | no | Exact m-decomposed curved PSTF coupling coefficients |
