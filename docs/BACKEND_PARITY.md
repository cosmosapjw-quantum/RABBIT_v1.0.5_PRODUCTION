# SciPy ↔ JAX Backend Parity

## Resolution (2026-04-18)

The old “JAX has a large Y_p offset versus SciPy by design” story is no longer
the right top-level description for the public Type-I surfaces.

The correct current summary is:

- `backend="jax"` is a bounded canonical tier-1 live-weak surface
- `backend="jax_advanced"` with `enable_teff=False` is a bounded canonical
  no-Teff tier-3 weak-budget surface
- `backend="auto"` resolves to the same SciPy reference capability as
  `backend="scipy"`; this alias is not JAX parity evidence

## Parity posture

### Tier-1 canonical parity

For the promoted tier-1 public surface:

- SciPy and JAX are compared on matched live-weak physics
- FLRW parity is lock-tested
- CL0-CL3 support is available on both sides

This is the canonical `backend="jax"` parity story.

### Tier-3 no-Teff parity

For the promoted no-Teff tier-3 public surface:

- `backend="jax_advanced"` is exercised against the corresponding SciPy cells
- CL0-CL3 support is lock-tested on the bounded cross-check matrix
- the promoted claim is limited to the no-Teff CPU-first surface

This is the canonical `backend="jax_advanced"` parity story.

## What this does not mean

The parity claim does **not** imply:

- Teff is canonical on JAX
- JAX perimeter surfaces are all canonical
- SciPy tier-2/tier-3 transitional collision path is fully superseded

## Current policy summary

- `auto` and `scipy` identify the SciPy reference authority
- JAX parity requires an explicitly named JAX backend on the paired side
- tier-3 no-Teff and perimeter surfaces still retain their explicit scope boundaries
