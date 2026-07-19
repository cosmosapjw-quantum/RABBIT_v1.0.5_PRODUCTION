# BD619 — auto→scipy default + collision/weak twin consolidation audit

Date: 2026-07-09

Status: **IMPLEMENTED (auto→scipy)** + **AUDITED / NO-OP (collision twin)**. Two
JAX-removal-direction tasks: (2) make the inference `auto` backend default to
scipy; (1) consolidate collision/weak twins to a numpy single lane. Task 2 is
done; task 1 is honestly a no-op — the safe target does not exist yet.

## Task 2 — `auto` defaults to scipy (IMPLEMENTED)

Production inference already runs scipy (`run_production_inference.py` defaults
`backend='scipy'`), yet `backend='auto'` resolved JAX-first within a bounded
Type-I scope. The `runtime_device` resolver was already built with flip-readiness
metadata (`dispatch_flip_ready`, `candidate_backend_if_auto_flipped`) anticipating
exactly this flip.

Change (`src/rabbit/jax/runtime_device.resolve_typeI_auto_backend`):
- Default `RABBIT_AUTO_BACKEND_POLICY` → `scipy_default_jax_optin`.
- Within the bounded JAX scope, `auto` now returns **scipy** by default and
  records the JAX candidate; it returns `jax_characteristic` only on explicit
  opt-in (`RABBIT_AUTO_PREFER_JAX=1`, or a jax-preferred policy). New reasons:
  `scipy_default_jax_available_optin` (default) / `jax_characteristic_optin_within_scope`
  (opt-in). Preserved all flip metadata; added `scope_jax_optin`.
- `RABBIT_FORCE_BACKEND` and the outside-scope/jax-unavailable paths unchanged.
- `backend_capabilities.CAPABILITY_BY_BACKEND["auto"]` → `SCIPY_TYPEI_REFERENCE`
  (was `JAX_TYPEI_CHARACTERISTIC_TIER1`). Regenerated
  `BACKEND_CAPABILITY_MATRIX.md` + `SUPPORTED_CAPABILITIES.md`.

Tests updated (only 2 files asserted auto→jax): `test_auto_gpu_dispatch_policy.py`
(split into scipy-default + jax-opt-in cases) and
`test_inference_backend_propagation.py` (dispatch key/tier + the N_q=20 scope
case now expect scipy). The bounded JAX characteristic backend remains fully
available and canonical — only the *default* moved to scipy.

## Task 1 — collision/weak twin → numpy single lane (AUDITED, NO SAFE TARGET)

Goal: demote the JAX collision/weak kernels to test-only oracles, leaving numpy
as the single production lane. Audit result: **not achievable safely yet** — the
"twins" are not redundant test-only duplicates; they are the compute kernels of
the still-active JAX driver backends.

Evidence (14 collision/weak JAX modules, production-src importer count):
- `collision_operator_jax` ← `teff_collision_bridge_jax` ← **canonical** `driver_typeI_char`.
- `weak_jax` (10), `weak_live_jax` (11), `weak_live_fused` (1) feed the canonical
  characteristic drivers; `collision_rates_jax` (5) feeds canonical thermo
  (`nudec_coupled_jax`).
- `collisions_jax`, `nu_nu_scattering_jax`, `collision_ap_preconditioner_jax`,
  `collision_hm_full_jax` feed the candidate `driver_typeI_full_boltzmann`
  (`jax_ap_unified_tier3`).
- Only `collision_hm_partner_integration_jax` (+ its `hm_matrix_elements_jax`
  chain) is test-only — and BD614 already established that its shared regression
  test (`test_ap_rosenbrock_full_hm_closed_form.py`) locks the PRODUCTION
  `collision_hm_full_jax`, so it cannot be dropped without losing that lock.

Conclusion: demoting the JAX collision/weak lane to test-only requires FIRST
retiring the JAX forward drivers that consume it (the canonical characteristic
line + candidate full_boltzmann) — a much larger physics/validation decision, not
a mechanical twin-consolidation. Task 2 (auto→scipy default) is the correct
precursor: it moves the default off JAX so a later forward-line retirement
becomes tractable. No code was changed for task 1; the honest finding is
recorded so the twin-maintenance cost is attributed to the right cause.

## Verification

```
# auto resolver flip:
pytest tests/test_auto_gpu_dispatch_policy.py                        -> 6 passed
pytest "tests/test_inference_backend_propagation.py::TestDispatchTable" -> 26 passed
pytest tests/test_inference_backend_propagation.py -k "auto_default_scope or auto_defaults_to_scipy" -> 2 passed
# no NEW failures introduced: the 2 red auto-policy tests
# (test_jax_auto_policy_boundary_transition, test_jax_auto_policy_recording)
# fail identically on clean HEAD c0d1f90 — pre-existing, unrelated to this change.
```

## Cost line

- added_lines: ~70 (resolver flip + capability + test updates + note; matrix/SUPPORTED regenerated)
- deleted_lines: ~10
- files_touched: 2 production (runtime_device, backend_capabilities) + 2 tests
  + 2 rendered docs + 1 note
- runtime_behavior_changed: yes (declared) — `backend='auto'` now resolves to
  scipy by default; JAX characteristic remains available on explicit opt-in
- physics_behavior_changed: no (scipy is a canonical Type-I surface; same physics)
- known_blocker_reduced: yes (default forward path is now scipy, the JAX-removal
  precursor; task-1 twin-consolidation correctly attributed as blocked on the
  jax forward-line retirement)
- blocker_movement_ratio: 0.4
- validation_strengthened: yes (auto-default + opt-in regression locks)
- cost_effectiveness_verdict: ACCEPT
