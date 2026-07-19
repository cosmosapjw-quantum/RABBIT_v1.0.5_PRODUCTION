# PR-G Stage 1

## Internal code audit

Read and cross-checked:

- `src/rabbit/jax/solver_jax_rodas5p.py`
- `src/rabbit/jax/driver_typeI_char.py`

Findings:

- scalar `_solve_core(...)` was already pure-JAX but had no event path
- scalar `jax_rodas5p_solve(...)` already had correct event refinement,
  but only for one trajectory at a time
- the correct minimal extension was a new cached batched runner rather
  than rewriting the scalar cache path

Result:

- scalar cache path stayed untouched
- batch path got its own event-masked compiled runner
