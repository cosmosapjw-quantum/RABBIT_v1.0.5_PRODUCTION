# PR-G Stage 2

## External / contract audit

References checked against the current design:

- JAX `lax.while_loop` / control-flow semantics
- JAX GPU memory allocation environment variables

Operational conclusions:

- per-lane event handling must be expressed as dataflow inside one
  `lax.while_loop`; Python-side loop orchestration would forfeit the
  batch compile win
- GPU VRAM preallocation must still be disabled explicitly before
  import for this workload

No physics formulas changed in this PR.
