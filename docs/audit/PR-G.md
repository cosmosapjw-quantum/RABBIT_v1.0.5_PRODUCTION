# PR-G Audit

## Scope

Add batched LRS characteristic entrypoints on top of Rodas5P without
changing the scalar characteristic driver:

- `run_char_batch_tier1(...)`
- `run_char_batch_tier2(...)`
- `_solve_core_event_masked(...)`

## Verdict

Pass.

What landed:

- event-aware pure-JAX batched solve core with per-lane finish masks
- CPU-first and GPU-retry runtime policy parity with the scalar driver
- tier-1 / tier-2 batch wrappers that preserve scalar observables
- regression locks for batch parity and frozen finished-lane semantics

What did not change:

- scalar `run_full_coupled_typeI_char_jax(...)`
- Rodas5P tableau / stiffness strategy
- characteristic physics scope

## Key results

- `tests/test_pr_g_vmap_batch.py`: `5 passed`
- targeted characteristic/runtime regression bundle: `79 passed`
- local RX 6950 XT throughput crossover occurs around `N≈128`
  rather than the old roadmap estimate `N=64`

## Decision

Keep CPU as the default runtime path. Use GPU only through the explicit
batch helpers and only for medium/large grids.
