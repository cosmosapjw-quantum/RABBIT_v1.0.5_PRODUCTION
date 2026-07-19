# BD618 — Retire (quarantine, NOT delete) the alternative-geometry backends

Date: 2026-07-09

Status: **IMPLEMENTED**. The Bianchi Class-A, Class-B, and tilted-FLRW research
backends are retired from the active development surface and **preserved** for
future extension work — quarantined, not deleted. Zero source or test files were
removed; the backends remain registered and callable.

## Decision

Following the JAX-removal direction, the alternative-geometry research lines were
audited. Unlike BD614 (dead/self-referential modules with zero importers), these
are functional, test-covered "candidate"/"substrate" surfaces carrying the
project's general-Bianchi anisotropic-cosmology physics (Types II–IX orthogonal +
tilted). The user's decision: **retire all three geometry lines but do not delete
— isolate them for future extension development.**

Retired-but-preserved backends:
`jax_classA`, `jax_classB`, `jax_tilted`, `jax_tilted_full_coupled`.

Kept canonical line: Type-I only — `scipy`, `jax`, `jax_characteristic`,
`jax_characteristic_tier2`, `jax_advanced` (+ `auto` → characteristic tier-1).

## Why quarantine, not delete

- These are ~1,300 LOC of drivers plus a shared geometry substrate
  (`geometry_classA_jax`, `geometry_classB_jax`, `geometry_bianchi_base`,
  `rhs_classA`) that the tilted line also uses — and ~57 dedicated test files,
  including **34 gold Bianchi/tilted validations** (`gold/test_type*_orthogonal_bbn.py`,
  `gold/test_type*_tilted_bbn.py`). This is validated physics, not dead plumbing.
- Deleting it would destroy the anisotropic-cosmology validation and require
  invasive dispatch/registry/claim-gate surgery. Quarantine preserves everything
  and is fully reversible.

## Mechanism (additive, reversible)

- `src/rabbit/config/backend_capabilities.py`: `QUARANTINED_BACKENDS` frozenset +
  `is_quarantined(backend)`. The backends **stay in `CAPABILITY_BY_BACKEND`**
  (registry remains complete — registry/scope/enumeration tests pass unchanged)
  and **stay callable** (future extension dev can still exercise them; no
  dispatch hard-gate).
- `tests/conftest.py`: `pytest_collection_modifyitems` skips any test module that
  imports a quarantined driver (`driver_classA`/`driver_classB`/`run_tilted_bbn`)
  or is a `*_orthogonal_bbn`/`*_tilted_bbn` gold suite — EXCEPT an explicit infra
  allowlist (registry_sync, surface_scope_honesty, production_gates, claim_gates,
  … ) that must keep validating the still-registered backends. Detection is
  import-string based, not filename-substring, so it is robust. Opt back in with
  `RABBIT_RUN_QUARANTINED=1`.
- `tests/test_quarantine_backends.py`: locks the quarantined set + that the
  backends remain registered (not deleted) + that the canonical line is not
  quarantined.

## Verification

```
# quarantined suites skip by default:
pytest tests/gold/test_typeII_orthogonal_bbn.py tests/test_classB_phase1_smoke.py \
       tests/test_tilted_bbn_smoke.py            -> 40 skipped

# infra/registry + unrelated tests still run and pass (backends still registered):
pytest tests/test_registry_sync.py tests/test_surface_scope_honesty.py \
       tests/test_solver_drift_guard.py -m "not slow"  -> 38 passed, 3 skipped

# reactivation: RABBIT_RUN_QUARANTINED=1 re-selects the full quarantined suite.
```

## Known follow-up (not done here)

The claim gates for the retired lines (`GATE_TILTED_OUTSIDE_TYPE_I_SCALAR`, the
"Full Bianchi-BBN canonical all-11-types" claim in `claim_gates.py`) require the
now-skipped gold node IDs, so under `promotion_check.py` they will read
not-green — which is the honest consequence of retiring those claims. Formally
retiring those gate *definitions* (or marking them quarantined) is a clean-up
left for a follow-up; it does not affect the active Type-I promotion surface.

## Cost line

- added_lines: ~110 (quarantine registry + conftest hook + test + note)
- deleted_lines: 0 (nothing removed — preservation is the point)
- files_touched: 1 production (backend_capabilities) + 1 conftest + 1 test + 1 note
- runtime_behavior_changed: no (backends still registered + callable; only their
  dedicated test suites are deselected by default)
- physics_behavior_changed: no
- known_blocker_reduced: yes (alt-geometry research lines removed from the active
  dev/test surface; JAX-removal direction advanced without destroying physics)
- blocker_movement_ratio: 0.4
- validation_strengthened: yes (quarantine registry lock)
- cost_effectiveness_verdict: ACCEPT
