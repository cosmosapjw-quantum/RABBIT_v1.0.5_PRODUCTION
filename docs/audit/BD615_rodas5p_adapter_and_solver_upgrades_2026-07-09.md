# BD615 — Rodas5P scipy-compatible adapter + solver upgrades (opt-in lane)

Date: 2026-07-09

Status: **IMPLEMENTED / REGRESSION-LOCKED**. The in-tree Rosenbrock-W solver
(`rabbit.solver.rodas5p`) can now be selected as a driver backend via
`SolverMethod.RODAS5P` and routed through a scipy-`solve_ivp`-shaped adapter.
Default production behavior is **unchanged** (BDF); the RODAS5P lane is opt-in
only. Production *promotion* of RODAS5P remains **BLOCKED** pending the BD616
bake-off wall-time gate (see BD613 §"future method change").

## Problem

BD613 retired the silent Radau→BDF remap and made `PRODUCTION_CONFIG` declare
BDF honestly, but the in-tree `rabbit.solver.rodas5p` ("the R in RABBIT") was
still reachable only as a parity oracle in `test_j04_jax_rodas5p`, never as an
actual driver backend. Two gaps blocked wiring it in:

1. `SolverMethod` had no `RODAS5P` member — the driver could not select it.
2. `rodas5p.solve` returns a `Rodas5PResult` that lacks `.status` and
   `.t_events`, which the driver's target-reached gate and
   `classify_solve_ivp_result` both read from the scipy result object.

Two solver-quality gaps also had to be closed for the lane to be faithful:

3. The event crossing was localized by first-order linear interpolation only,
   leaving an O(1e-4) offset in N on a stiff T-handoff/T-end crossing that
   propagated straight into the endpoint abundances (measured |ΔY_p|=2.67e-4
   before the fix — see below).
4. Each stage solve refactorized the stage matrix W from scratch (8 O(N^3)
   `np.linalg.solve` calls per step).

## Changes

### `src/rabbit/config/solver_config.py`
- Added `SolverMethod.RODAS5P` (value `"RODAS5P"`) — explicitly NOT a scipy
  method — and `_SCIPY_METHODS = {RADAU, BDF, LSODA}`.
- Added `SolverConfig.is_scipy`; `to_scipy_kwargs()` now raises `ValueError`
  fail-loud on a non-scipy method so RODAS5P can never leak into a `solve_ivp`
  call (it must be routed through the adapter).

### `src/rabbit/solver/rodas5p.py` (three independently-verified upgrades)
- **Event direction**: `solve(..., event_direction=0)`; scipy convention
  (`-1` = +→- crossing incl. zero, `+1` = -→+, `0` = either). Default 0
  preserves the historical any-crossing behavior. Helper `_event_fires`.
- **Event refinement**: `_refine_event` bisects the sub-step size on the event
  sign, taking a fresh Rosenbrock sub-step from `(t, y)` with the
  already-computed Jacobian at each probe (~50 O(N^2) back-substitutions per
  event; events fire ~twice per solve). Localizes the crossing to machine
  precision (verified 1.8e-12 vs analytic) instead of the old ~1e-4 linear
  interpolation. THIS is what makes the driver endpoint parity pass.
- **LU-once-per-step**: `_step_sciml`/`_step_hairer` now `lu_factor(W)` once and
  `lu_solve` per stage (one factorization + s back-substitutions instead of s
  factorizations). Non-finite stage results are caught and flagged as a step
  failure (preserving the old singular-W → reject-step semantics).
- **Jacobian reuse (flagged, default off)**: `Rodas5PConfig.jac_reuse_max_steps`
  (default 1 = recompute every step = exact Rosenbrock). >1 reuses a Jacobian
  across accepted steps (a rejected step always forces a fresh Jacobian). The
  FD Jacobian costs N+1 RHS evals/step and dominates at large N, so reuse is the
  main cost lever — but it is an approximation and must be endpoint-parity gated
  before any preset enables it.

### `src/rabbit/solver/rodas5p_adapter.py` (NEW)
`Rodas5PIvpResult` (scipy-`OdeResult`-shaped) + `solve_ivp_rodas5p(...)`.
Synthesizes `.status`/`.t_events` from the `Rodas5PResult` so the driver drives
the in-tree solver through the identical contract — no `Rodas5PResult` change.
Mapping: rtol/atol direct; `max_step` → `max_step_N` and `h_max` (both, strict);
`first_step` → `h_init`; `dense_output=True` → `NotImplementedError` (fail-loud;
the driver never requests a dense interpolant). Status: event → 1 +
`t_events=[[t_cross]]`; reached t_span end → 0 + empty t_events; failure → -1.

### `src/rabbit/drivers/full_coupled_typeI.py`
Factored the two `solve_ivp` call sites into a `_run_phase` seam that branches on
`effective_solver.method is SolverMethod.RODAS5P`, calling the adapter instead of
`solve_ivp` (and skipping the scipy-only `jac_sparsity` hint, recorded as
`metadata['rodas5p_jac_sparsity_unused']`). The BD613 drift guard covers the new
lane unchanged.

## Validation

```
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_rodas5p_upgrades.py tests/test_rodas5p_adapter.py -m "not slow"
  -> 13 passed, 1 deselected in 0.67s

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_rodas5p_adapter.py::test_driver_rodas5p_endpoint_parity_vs_bdf \
  tests/test_j04_jax_rodas5p.py
  -> 21 passed in 17.44s   (driver RODAS5P↔BDF parity + JAX oracle unbroken by LU-once)

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_solver_drift_guard.py -m "not slow"
  -> 9 passed in 17.58s    (BD613 drift guard not regressed)

full-suite collect: 3101 tests, no collection errors.
```

End-to-end driver parity (N_q=8, N_mu=8, tier=1, single process, peak RSS
0.09 GB):
- before event refinement: |ΔY_p| = 2.67e-4  (linear-interp crossing error)
- after event refinement:  |ΔY_p| = 4.61e-9, rel D/H = 2.11e-7  — both far under
  the promotion gate thresholds (|ΔY_p|<1e-5, rel D/H<1e-3).

## Known limitation

The RODAS5P lane uses the solver's dense FD Jacobian (N+1 RHS evals/step,
recomputed every step at the default `jac_reuse_max_steps=1`); `jac_sparsity`
is not consumed. Whether this is wall-time-competitive with BDF at production
N is exactly the BD616 bake-off / BD5b promotion question — not claimed here.

## Cost line

- added_lines: ~470 (solver upgrades + adapter + driver seam + 2 test files + note)
- deleted_lines: ~10 (replaced solve_ivp call sites / np.linalg.solve stage loops)
- files_touched: 4 production (solver_config, rodas5p, rodas5p_adapter[new],
  full_coupled_typeI) + 2 tests[new] + 1 note
- runtime_behavior_changed: no (default BDF lane byte-identical; RODAS5P opt-in only)
- physics_behavior_changed: no
- known_blocker_reduced: yes (in-tree Rodas5P is now a selectable, parity-locked
  driver backend — the precondition for the BD616/BD5b promotion decision)
- blocker_movement_ratio: 0.5
- validation_strengthened: yes (new adapter + solver-upgrade regression locks)
- cost_effectiveness_verdict: ACCEPT
