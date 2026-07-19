# JAX Tier-2/Tier-3 Production Readiness Checklist

> **HISTORICAL (PUB-01, 2026-07-12).** This checklist records an earlier
> bounded-promotion decision. It is provenance-only: no row establishes
> publication or public-production authority. Current authority is the SciPy
> default plus the publication PR programme.

Question:

- Is the CPU-first `jax_advanced` Type-I 3T weak-budget ladder ready as a bounded production surface across CL0-CL3?
- Is the CL3 weak-budget slice on that same surface ready?

## Status

| Surface | Status | Scope note |
|---|---|---|
| Tier-3 JAX 3T weak-budget ladder (`backend="jax_advanced"`, `enable_teff=False`) | `historical bounded implementation` | Explicit diagnostic surface; not publication authority |
| Tier-3 JAX weak budget on 3T (`backend="jax_advanced"`, `correction_level=3`, `enable_teff=False`) | `historical bounded implementation` | Explicit diagnostic CL3 surface; not publication authority |
| Teff on JAX tier-2 (`enable_teff=True`) | `deprecated` | Legacy closure retained only as low-level kernel diagnostics; public runtime rejects it |
| Claim “JAX tier-2/tier-3 are production-ready” | `historical bounded claim; not publication authority` | `jax_advanced` remains explicit; `backend="auto"` is the SciPy reference and PUB-02/G-01 remain open |

## Promotion PRs

### PR-1: Tier-3 promoted-path lock

Status: implemented on the inference path for `backend="jax_advanced"` with `enable_teff=False`.

Required:

- `capability_key = "jax_typeI_tier3_weak_budget"`
- `maturity = "canonical"`
- `runtime_device_policy = "cpu_preferred"`
- `phase1_solver_diagnostics.solver_outcome = "target_reached"`
- `phase2_solver_diagnostics.solver_outcome = "target_reached"`

### PR-2: Tier-3 promoted-path lock

Status: implemented on the inference path for `backend="jax_advanced"` with `correction_level=3` and `enable_teff=False`.

Required:

- `capability_key = "jax_typeI_tier3_weak_budget"`
- `weak_budget_mode = "born+coulomb+sirlin+finite_mass"`
- `correction_budget_channels.finite_mass_recoil_wm = True`

### PR-3: Runtime acceptance gate

Status: implemented for the representative CPU-first CL3 `jax_advanced` solve.

Required:

- representative `jax_advanced` CL3 no-Teff solve remains under the locked wall-clock ceiling
- metadata must still report `runtime_device_contract = "cpu_preferred_exact_runtime_v1"`

### PR-4: Teff deprecated-runtime fence

Status: implemented through capability split and runtime rejection tests.

Required:

- `enable_teff=True` must raise before entering a public forward-solver path
- Teff stays diagnostic/substrate only under `jax_typeI_liveweak_cl3_tier2_teff_candidate`

## Historical wording (now forbidden as a current claim)

The earlier phrase “JAX tier-2/tier-3 are production-ready” described only a
bounded `backend="jax_advanced"` no-Teff CPU-first runtime decision. PUB-01
withdraws it as a current claim: it is not publication validation or public-
production authority, and PUB-02/G-01 remain open.
