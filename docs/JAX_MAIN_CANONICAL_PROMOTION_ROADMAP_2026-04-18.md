# JAX Main Canonical Promotion Roadmap

> **HISTORICAL (PUB-01, 2026-07-12).** This records the former JAX-default
> promotion attempt. It is not an active roadmap or current dispatch truth.
> `backend="auto"` now resolves to the SciPy reference; future promotion is
> governed only by `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md` and G-01.

## Scope

This document answers one engineering question:

- What route moved JAX from bounded canonical explicit surfaces to the main
  bounded `backend="auto"` canonical authority above SciPy?

This roadmap is intentionally scoped to:

- CPU-first JAX runtime
- Type-I canonical production chain
- no-Teff promoted surfaces
- no claim about Class A / Class B / tilted perimeter

## Historical state recorded on 2026-04-18

Today the repo has:

- `backend="auto"` as the bounded JAX exact-characteristic canonical authority
- `backend="scipy"` as the reference / fallback canonical authority
- `backend="jax"` as a canonical explicit linearized tier-1 surface
- `backend="jax_characteristic"` / `backend="jax_characteristic_tier2"` as
  canonical explicit exact-characteristic surfaces
- `backend="jax_advanced"` as a canonical explicit no-Teff tier-3 surface

So JAX is canonical on bounded public surfaces and now also owns the bounded
main dispatch authority.

## Promotion target

The target state is:

- `backend="auto"` resolves to a JAX Type-I canonical surface
- SciPy remains available as a reference / fallback backend
- docs, metadata, readiness locks, and benchmark gates all describe that state
  honestly

## Work plan

### PR-A: Canonical contract cleanup

Objective:

- remove stale wording that still describes JAX as purely candidate or reduced
- align docs, registry, and metadata around the actual current bounded JAX
  canonical surfaces

Completion standard:

- no top-level doc contradicts the current `backend_capabilities.py` contract

### PR-B: Load-bearing parity matrix strengthening

Objective:

- expand and tighten the JAX ↔ SciPy parity matrix over the promoted surface

Minimum matrix:

- tier-1 CL0-CL3
- no-Teff tier-3 CL0-CL3
- FLRW cells
- representative anisotropic cells

Completion standard:

- bounded parity is mechanically locked, not described informally

### PR-C: CPU-first runtime hardening

Objective:

- make the CPU-first JAX runtime contract robust enough to serve as the main
  public authority

Scope:

- cold/warm behavior
- prewarm semantics
- cache behavior
- metadata determinism
- failure classification

Completion standard:

- representative bounded JAX solves remain operationally acceptable under the
  existing runtime gates

### PR-D: Dispatch flip

Objective:

- change `backend="auto"` from SciPy-first to JAX-first

Status:

- merged for the bounded CPU-first Type-I exact-characteristic default envelope

Required at flip time:

- explicit fallback policy
- updated hierarchy locks
- updated propagation locks
- no stale doc claiming SciPy is still the auto authority

Completion standard:

- the repo can honestly say that JAX is the main canonical backend

### PR-E: Relock and publication-facing cleanup

Objective:

- relock user-facing docs and release surfaces after the dispatch flip

Scope:

- capability tables
- readiness checklists
- backend parity notes
- README / STATUS
- release smoke and registry sync

Completion standard:

- the repo presents one consistent story from code to docs to tests

## Main risks

### Risk 1: bounded canonical is easier than main canonical

Explicit opt-in canonical surfaces are easier to support than `auto` because
they do not carry the same public default responsibility.

### Risk 2: dispatch flip is a policy change, not just a speed win

Even if JAX is faster, the dispatch flip still needs parity, fallback, and
metadata discipline.

### Risk 3: stale docs can silently undo the promotion

If docs, registry, and tests disagree during the flip, users will read the repo
as less mature than the code actually is.

## Decision rule

Before the dispatch flip:

- “JAX has canonical bounded public surfaces” is allowed
- “JAX is the main default backend” is not yet allowed

After the dispatch flip:

- “JAX is the main canonical backend” becomes allowed
- SciPy should be described as reference / fallback, not the main authority
