# PR-J Abort Note

## Status

This note remains accurate for the original PR-J design, but it is now
historical rather than current. The abort applied to the old
25/27-DOF characteristic state with hybrid analytic + reverse-mode
Jacobian assembly.

That line was later superseded by `PR-JL` (`acd9080`), which solved the
underlying cost problem differently:

- removed explicit `I_j` from the characteristic ODE state
- compacted the tier-2 characteristic state to 15 DOF
- kept the characteristic Jacobian path dense-only after compaction
- hardened direct runtime-device fallback and lazy imports for
  no-visible-GPU hosts

Representative bounded CPU numbers after the compaction pass moved to:

| metric | value |
|---|---:|
| warm solve mean | `0.063215 s` |
| phase-2 dense Jacobian | `0.001172 s` |

## Reason for abort

PR-J reached the phase prompt's abort condition:

- analytic Jacobian parity against `jax.jacfwd` was achieved, but
- local warm single-solve performance **regressed instead of improved**.

The prompt explicitly lists "performance regression instead of
improvement" as an abort trigger, so no PR-J commit was made.

## Evidence collected

### 1. Analytic Jacobian parity

Experimental implementation note:

- closed-form geometry / transport / `S` rows
- targeted reverse-mode fill for thermo + network rows

Elementwise comparison against the legacy dense `jax.jacfwd` path
showed excellent numerical agreement:

- phase-1 and phase-2 cases matched with `np.allclose(..., atol=1e-10, rtol=1e-10)`
- largest relative discrepancy measured: `~7.8e-14`
- large absolute differences (`0.25`, `0.0625`) occurred only on
  phase-2 network entries of magnitude `~1e15`, i.e. floating-point
  rounding at machine precision, not a physics mismatch

### 2. Local timing benchmark

Measured on the current sandbox with
`JAX_PLATFORMS=cpu JAX_COMPILATION_CACHE_DIR=/tmp/rabbit_jax_cache`:

| mode | min | mean |
|---|---:|---:|
| `dense + analytic` | 2.199851 s | 2.288836 s |
| `dense + jacfwd` | 1.738424 s | 1.751715 s |
| `block_sparse + analytic` | 2.078340 s | 2.091396 s |
| `block_sparse + jacfwd` | 2.708048 s | 2.798191 s |

Verdict:

- `analytic` was slower than the current production default
  `dense + jacfwd`
- even the best analytic variant (`block_sparse + analytic`) remained
  slower than `dense + jacfwd`

## Interpretation

The hybrid design was mathematically sound but not performant enough:
the targeted reverse-mode fills for thermo + network rows still cost too
much on CPU at the current 25-DOF state size.

In retrospect, this was the correct abort call. The problem was not the
analytic transport algebra itself; it was the cost model of applying a
hybrid Jacobian to a still-too-large state.

## Recommended next step

Do not reopen the original PR-J hybrid design directly. Any future
analytic-Jacobian revisit should start from the compacted post-`PR-JL`
state and benchmark against the standing dense baseline on that newer
surface.

If a new PR-J-like phase is attempted, the concrete targets are:

1. derive the thermo block analytically rather than via reverse mode
2. avoid reintroducing any large passive transport block into the state
3. benchmark against the post-compaction dense baseline, not the old
   25-DOF `dense + jacfwd` surface
