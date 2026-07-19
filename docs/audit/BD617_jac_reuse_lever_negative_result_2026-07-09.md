# BD617 — Jacobian-reuse lever for RODAS5P: plumbed + measured NEGATIVE

Date: 2026-07-09

Status: **IMPLEMENTED (plumbing) / NEGATIVE RESULT (lever)**. `jac_reuse_max_steps`
is now a `SolverConfig` knob threaded through the adapter into the solver. The
BD616 promotion question — does Jacobian reuse make the RODAS5P endpoint lane
wall-competitive with BDF? — is answered **no**. RODAS5P production promotion
remains **BLOCKED**; BDF stays the default.

## What was tried

BD616 showed RODAS5P is 4.6–5x slower than BDF at the endpoint because the dense
FD Jacobian (N+1 full-RHS evals/step, recomputed every step at the exact-Rosenbrock
default `jac_reuse_max_steps=1`) dominates. The plan's identified unblock lever was
`jac_reuse_max_steps > 1`: reuse a Jacobian across accepted steps to amortize that
cost.

Plumbing (this PR): `SolverConfig.jac_reuse_max_steps` (default 1) →
`solve_ivp_rodas5p(..., jac_reuse_max_steps=)` → `Rodas5PConfig` → the solver's
reuse loop. Correct and unit-tested (`test_adapter_threads_jac_reuse_into_config`:
reuse>1 strictly cuts `njev`).

## Measurement (N_q=16, N_mu=8, tier=1, single process, unbuffered, reps=1)

| lane | wall | steps (p1+p2) | \|ΔY_p\| vs BDF |
|---|---:|---:|---:|
| BDF (baseline) | 1.77 s | 217+430 = 647 | — |
| RODAS5P reuse=1 | 8.25 s | 210+343 = 553 | 6.5e-9 |
| RODAS5P reuse=3 | 35.6 s | 649+1127 = 1776 | 4.7e-9 |
| RODAS5P reuse=5 | 35.8 s | 649+1130 = 1779 | 4.6e-9 |

## Why the lever fails (and what actually would work)

- **Reuse makes it *worse*, not better.** reuse=3/5 explode the step count 3.2x
  (553 → ~1778) and the wall ~4.3x (8.3 s → 35.7 s). A stale Jacobian wrecks the
  step-size controller in this stiff system: the Rosenbrock error estimate
  degrades, steps shrink and get rejected, and the extra steps far outweigh the
  saved Jacobian builds. Endpoint parity stays fine (physics is unchanged), but
  the wall is the opposite of the goal.
- **The barrier is the per-step FD Jacobian, not its frequency.** Tellingly,
  RODAS5P at reuse=1 already takes *fewer* steps than BDF (553 vs 647) yet is
  4.7x slower — because each step spends N+1 full `coupled_rhs` evaluations
  building the dense FD Jacobian, while scipy BDF uses a far cheaper internal
  Jacobian strategy (reuse + the `jac_sparsity` colouring the driver supplies,
  which the in-tree solver ignores).
- **The genuine unblock path** is therefore a *cheaper Jacobian*, not a *staler*
  one: a structured / sparse-coloured / analytic / JVP Jacobian for the in-tree
  solver (the driver already computes `_characteristic_jac_sparsity` for scipy —
  RODAS5P discards it). That is a substantially larger effort and is explicitly
  out of scope here; it is the real prerequisite for any future RODAS5P
  promotion.

## Outcome

- `jac_reuse_max_steps` kept as a correct, tested config knob (default 1 =
  exact/no-change; useful for future experiments and honest about the option).
- RODAS5P promotion: **still BLOCKED**. Not on parity (excellent) but on wall,
  and Jacobian reuse does not move the wall the right way.
- Recorded lever for a future PR: cheap structured Jacobian for the in-tree
  Rosenbrock solver (consume `jac_sparsity`; or analytic/JVP), NOT reuse.

## Cost line

- added_lines: ~40 (SolverConfig field + adapter param + driver thread + 1 test + note)
- deleted_lines: ~2
- files_touched: 3 production (solver_config, rodas5p_adapter, full_coupled_typeI)
  + 1 test + 1 note
- runtime_behavior_changed: no (default jac_reuse_max_steps=1 = prior behavior;
  RODAS5P still opt-in)
- physics_behavior_changed: no
- known_blocker_reduced: yes (the reuse lever is now measured and ruled out — the
  RODAS5P wall blocker is correctly attributed to the per-step FD Jacobian, and
  the real fix is scoped)
- blocker_movement_ratio: 0.3
- validation_strengthened: yes (plumbing test + documented negative measurement)
- cost_effectiveness_verdict: ACCEPT
