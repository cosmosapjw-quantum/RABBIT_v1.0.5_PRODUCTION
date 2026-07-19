# PR-T3A Audit

## Scope

Landed:
- `src/rabbit/jax/q_advection_jax.py`
- `src/rabbit/jax/driver_typeI_full_boltzmann.py`
- `tests/test_pr_t3a_collisionless_driver.py`

This is a **private collisionless shell**, not a new public backend.
Current limits are LRS-only, tier-1 thermo only, `runtime_device_policy="cpu_preferred"` only, and no collision operators.

## Fixed blockers

- Phase-1 init and phase handoff were creating read-only NumPy views from JAX buffers. Replaced with writable `np.array(..., copy=True)` paths.
- JIT tracer leak from `cached_q_advection_operators(...)` being called inside traced code. Operators are now built outside trace and captured by closure.
- `jac_fn` was linearizing `rhs_fn(0.0, y)` regardless of the incoming solver `N`. It now respects `rhs_fn(N, y)`.
- Import-time backend failure under no-visible-GPU hosts. The module now forces CPU platform selection before importing the Rodas5P solver.
- Upwind inflow rows now match the documented clamp contract:
  low-q inflow derivative is zero, high-q inflow uses a zero ghost state.
- The exact-remap PCHIP oracle is now present in `q_advection_jax.py` and is used only as a regression surface.
- The Jacobian payload is no longer a zero low-rank placeholder. The
  passive→active transport coupling now factors exactly through the
  collisionless `stress + transported monopole(q)` moment surface, and
  the materialized `U C V` Jacobian matches dense AD on the bounded
  phase-1 test state.

## Smoke + Regression

- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py`
  - `6 passed in 35.84s`
- `PYTHONPATH=src pytest -q tests/test_pr_t3a_collisionless_driver.py tests/test_j04_jax_rodas5p.py tests/test_jax_characteristic_solver_jacobian_controls.py`
  - `29 passed in 26.14s`

Small smoke (`N_mu=4, N_q=6, CL0`) completed successfully through both phases and surfaced `linear_solver_policy="custom_hook"` in phase 1 and phase 2.

## Bounded reduction checks

- `N_mu=4, N_q=6, Σ_H=0.0`
  - `|ΔY_p| = 6.0e-7`
  - `|ΔD/H| = 9.5e-9`
  - `|ΔX_n| = 3.2e-7`
- `N_mu=12, N_q=20, Σ_H=0.1`
  - `|ΔY_p| = 6.47e-4`
  - `|ΔD/H| = 5.37e-8`
  - `|ΔX_n| = 3.37e-4`
- `N_mu=4, N_q=6, Σ_H=0.3`
  - `|ΔY_p| = 3.10e-2`
  - `|ΔD/H| = 2.99e-6`
  - `|ΔX_n| = 1.60e-2`

All bounded runs preserved `species_identical_max_abs_final <= 4e-16` and final transport support inside `[0, 1]`.

## Adversarial verdict

The landed path is valid as a collisionless tier-3 **preflight shell**. It is not yet suitable for production parity claims:
- coarse-grid high-shear reduction degrades quickly
- `N_mu=12, N_q=20, Σ_H=0.3` exceeded the bounded CPU audit budget and was aborted
- no collision operators are present
- no 3T thermo is present
- no public backend / inference dispatch is wired

Verdict: **conditional pass**. Keep the driver private and candidate-only until high-shear boundedness, collisions, and tier-2 thermo are added.
