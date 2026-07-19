# PR-G Stage 3

## Harsh self-audit

Main risks considered:

1. Slowest-element tax inside the batched while loop
2. Drift of already-finished lanes while later lanes continue stepping
3. Silent regression of the scalar CPU-first path

Outcomes:

- the slowest-element tax is real and visible in the throughput curve
- finished-lane drift is bounded by the solver/event refinement lock in
  `tests/test_pr_g_vmap_batch.py`
- the scalar path remained unchanged and the characteristic/runtime
  regression bundle stayed green

Decision:

- accept the batch path as a separate explicit inference surface
- do not promote GPU for small batches
