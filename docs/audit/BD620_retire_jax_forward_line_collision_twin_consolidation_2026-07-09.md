# BD620 — retire the JAX Type-I forward line (opt-in) → collision/weak twin consolidation

Date: 2026-07-09

Status: **IMPLEMENTED**. The JAX Type-I forward backends are retired from the
active/default surface to opt-in (quarantined, not deleted). scipy is now the
sole active-canonical forward backend. This completes the task-1 collision/weak
twin consolidation: numpy is the single active-canonical collision/weak lane and
the JAX kernels are demoted to opt-in/parity-oracle status.

## Context

BD619 made `auto` default to scipy and recorded the task-1 finding that the JAX
collision/weak kernels are not redundant test-only twins but the compute of the
active JAX drivers — so consolidating them required first retiring the JAX
forward line. The user authorized that retirement, clarifying the verification
standard: the physics is derived, the code-level implementation was never
completed, and there is no external comparison literature — so **internal
consistency (scipy↔JAX parity) is the verification standard**, not external
anchoring.

## Change

`backend_capabilities.QUARANTINED_BACKENDS` extended (BD620) with the JAX Type-I
forward backends: `jax`, `jax_advanced`, `jax_characteristic`,
`jax_characteristic_tier2`, `jax_characteristic_nonlrs`, `jax_ap_unified_tier3`.
Added `ACTIVE_CANONICAL_BACKENDS = {"scipy", "auto"}`.

What retirement means here (identical to BD618's preservation contract):
- The backends stay **registered** (`CAPABILITY_BY_BACKEND` complete — registry
  and surface-honesty tests pass unchanged) and stay **callable** (selectable by
  explicit `backend=` or `RABBIT_AUTO_PREFER_JAX`; future work can use them).
- Their scipy↔JAX **parity/cross-check tests still RUN** — the conftest quarantine
  hook (BD618) only deselects the alt-geometry *driver* suites (driver_classA/
  classB/run_tilted_bbn imports); it does NOT touch the JAX Type-I forward tests.
  So the ~17 parity anchors (`test_backend_parity`, `test_pr_t3b_jax_operator_parity`,
  `test_jax_typeI_characteristic_parity`, `test_jax_collision_operator_parity`, …)
  keep verifying scipy↔JAX consistency — the internal-consistency standard.
- Tier and quarantine-status are ORTHOGONAL: `tier` records validation quality
  (jax_characteristic remains canonical-quality); `QUARANTINED_BACKENDS` records
  active/retired status. scipy is the sole *active* canonical.

## Why this consolidates the collision/weak twin (task 1)

The JAX collision/weak kernels (`collisions_jax`, `nu_nu_scattering_jax`,
`collision_operator_jax`, `collision_ap_preconditioner_jax`, `collision_hm_full_jax`,
`weak_jax`, `weak_live_jax`, …) were the compute of the now-retired JAX drivers.
With the forward line off the active/default surface, those kernels are reached
only via explicit opt-in and the parity tests — i.e. **opt-in/parity-oracle**.
numpy (`nu_e_scattering`, `pair_processes`, `kernels.py`, `weak/*`) is the single
active-canonical collision/weak lane; the JAX lane is checked against it. A
future numpy physics change surfaces as a parity-test signal against the JAX
oracle rather than a silently-required dual edit.

## Verification (internal consistency)

```
pytest tests/test_quarantine_backends.py tests/test_registry_sync.py \
       tests/test_surface_scope_honesty.py -m "not slow"   -> 33 passed, 3 skipped
pytest tests/test_backend_parity.py tests/test_pr_t3b_jax_operator_parity.py -m "not slow"
       -> 19 passed  (scipy<->JAX parity anchors RUN and pass — jax still callable)
full collect: 3108 tests, no errors.
```

No JAX code was deleted; the retirement is a registry/status change plus the
BD619 default flip. Fully reversible: remove entries from QUARANTINED_BACKENDS.

## Cost line

- added_lines: ~50 (QUARANTINED extension + ACTIVE_CANONICAL_BACKENDS + test update + note)
- deleted_lines: ~5
- files_touched: 1 production (backend_capabilities) + 1 test + 1 note
- runtime_behavior_changed: no beyond BD619 (default already scipy; JAX forward
  stays callable on explicit opt-in; nothing removed)
- physics_behavior_changed: no (scipy canonical Type-I surface; JAX preserved)
- known_blocker_reduced: yes (task-1 collision/weak twin consolidated — numpy is
  the single active-canonical lane, JAX demoted to opt-in/parity-oracle)
- blocker_movement_ratio: 0.5
- validation_strengthened: yes (active-canonical lock; parity anchors retained)
- cost_effectiveness_verdict: ACCEPT
