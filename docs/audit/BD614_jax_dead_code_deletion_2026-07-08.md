# BD614 — Delete Dead + Self-Referential Test-Only JAX Modules

Date: 2026-07-08

## Problem

`src/rabbit/jax/` accumulated a set of modules from earlier development
phases (Phase I/J/χ-4 vector-tilt, characteristic-method Class-A driver,
batched-vmap sweeps, forward-mode Jacobian utilities, an experimental
linear-solve sketch, and a curved-PSTF / Class-B transport substrate) that
are no longer wired into any production code path. Their only consumers are
their own test files — i.e. the tests exist to validate the module, and the
module exists only because the tests import it, with zero production
importers anywhere in `src/`. This is dead-weight twin-maintenance surface:
every future JAX refactor has to reason about whether these modules need to
move too, even though nothing downstream depends on them.

## Deleted files

### Source (10 files, 2407 lines)

| File | Lines | Category |
| --- | --- | --- |
| `src/rabbit/jax/curved_pstf_hierarchy.py` | 294 | dead — zero importers anywhere (exact curved PSTF Boltzmann hierarchy, never wired into a driver) |
| `src/rabbit/jax/rhs_classB.py` | 67 | dead — zero importers anywhere (Class B transport substrate) |
| `src/rabbit/jax/run_classA_bbn.py` | 307 | dead — zero importers anywhere (lightweight GPU-friendly Class A BBN runner). NOTE: `src/rabbit/drivers/classA_driver.py:351` defines a *different* function also named `run_classA_bbn` — that one is untouched and remains production code. |
| `src/rabbit/jax/batch.py` | 306 | test-only — sole importer was `tests/test_j07_batch.py` (batched parameter sweeps via `jax.vmap`) |
| `src/rabbit/jax/jacobian.py` | 265 | test-only — sole importer was `tests/test_j06_jacobian.py` (forward-mode AD Jacobian utilities) |
| `src/rabbit/jax/linear_solve_strategies.py` | 237 | test-only — self-labeled `"""Deprecated experimental JAX linear-solve sketches for future large-W/J regimes."""`; not wired into the default AP65 CPU-JAX/Rodas5P path per its own docstring |
| `src/rabbit/jax/vector_tilt.py` | 194 | test-only — sole importers were `tests/test_vector_tilt.py` and `tests/test_vector_tilt_K3.py` (3-axis vector tilt API, v3.0 Phase J) |
| `src/rabbit/jax/collision_pair_processes_jax.py` | 134 | test-only diagnostic twin — sole importer was `tests/test_pair_processes_jax.py` (JAX port of pair-annihilation rate; the numpy-lane original in `rabbit.collisions.pair_processes` is production) |
| `src/rabbit/jax/classA_characteristic_geodesics.py` | 444 | test-only cluster — importers were `driver_classA_characteristic.py` (its own paired module, also deleted) and the five `test_classA_*characteristic*.py` test files (per-Bianchi-type characteristic integrating factors, v3.0 Phase I) |
| `src/rabbit/jax/driver_classA_characteristic.py` | 159 | test-only — importers were `test_classA_characteristic.py` and `test_classA_typeII_characteristic.py` (characteristic-method Class-A driver, mooted per its own docstring: "Replaces the mooted PSTF-only Class-A path (JAX_CLASSA_DRIVER, demoted to substrate per RABBIT_report §03/§06)") |

### Tests (11 files, 1506 lines)

- `tests/test_j07_batch.py` (8 tests)
- `tests/test_j06_jacobian.py` (7 tests)
- `tests/test_linear_solve_strategies_status.py` (1 test)
- `tests/test_vector_tilt.py` (17 tests)
- `tests/test_vector_tilt_K3.py` (8 tests)
- `tests/test_pair_processes_jax.py` (15 tests)
- `tests/test_classA_characteristic.py` (16 not-slow + 1 slow-deselected)
- `tests/test_classA_typeII_characteristic.py` (9 tests)
- `tests/test_classA_typeVI0_characteristic.py` (7 tests)
- `tests/test_classA_typeVII0_characteristic.py` (5 tests)
- `tests/test_classA_typeVIII_IX_characteristic.py` (9 tests)

**Note on lost coverage**: deleting `tests/test_pair_processes_jax.py` loses a
redundant JAX-side T^5 scaling check on pair-annihilation rate. The primary
lock on this physics is the numpy-lane
`tests/test_deterministic_reference_rate_scaling.py`, which is run in
Validation and is unaffected by this deletion.

## KEEP list (entangled — verified load-bearing, NOT touched)

- `src/rabbit/jax/gradient_bridge.py`, `src/rabbit/jax/solver_diffrax_canonical.py`,
  `src/rabbit/jax/solver_jax_rodas5p_adjoint.py` (AD trio): `tests/test_native_ad_parity.py`
  node IDs are required by claim gates in `src/rabbit/config/claim_gates.py:68-70,94`
  (`GATE_FULL_DIFFERENTIABLE_BBN`, `GATE_GRADIENT_BASED_INFERENCE`).
- `src/rabbit/jax/hm_matrix_elements_jax.py`, `src/rabbit/jax/collision_hm_partner_integration_jax.py`
  (HM pair): `tests/test_ap_rosenbrock_full_hm_closed_form.py` is a shared regression lock on
  production `collision_hm_full_jax`.
- Test files kept as claim-gate / regression-lock infrastructure:
  `tests/test_native_ad_parity.py`, `tests/test_ad_inference_promotion_gates.py`,
  `tests/test_pr06_ad_production.py`, `tests/test_ap_rosenbrock_full_hm_closed_form.py`,
  `tests/test_hm_matrix_elements.py`, `tests/test_hm_partner_integration.py`,
  `tests/test_j08_gradient_bridge.py`, `tests/test_bridge_mode_default_is_rosenbrock_native.py`,
  `tests/test_rodas5p_native_ad.py`.

## Verification

### Pre-deletion importer sweep

For each of the 10 source modules:
`grep -rnE "(from rabbit\.jax\.MODULE|from \.MODULE|import (rabbit\.)?jax\.MODULE|rabbit\.jax\.MODULE)" src tests scripts`

Result for all 10: hits confined to the module's own self-referential docstring
header, its own paired module (`classA_characteristic_geodesics.py` <->
`driver_classA_characteristic.py`, both deleted together), and the deletion-target
test files. Zero hits in any file outside the deletion lists. `src/rabbit/jax/__init__.py`
is empty (re-exports none of the 10 modules) — confirmed clean.

### Post-deletion compile sweep

`venv/bin/python -m py_compile $(git ls-files 'src/**/*.py' | tr '\n' ' ')` — exit 0, no output.

### Post-deletion grep sweep

Same greps as above, re-run after deletion: zero hits in `src/`, `tests/`, `scripts/`.
`src/rabbit/drivers/classA_driver.py:351`'s distinct `run_classA_bbn` function (unrelated
symbol, same name) remains present and untouched, as expected.

### Test collection accounting

- BEFORE: `pytest --collect-only -q -m "not slow"` -> 3059/3190 collected (131 deselected).
- Per-file BEFORE counts for the 11 deleted test files (not-slow): 8+7+1+17+8+15+16+9+7+5+9 = 102
  collected, plus 1 deselected-slow (`test_classA_characteristic.py`) = 103 node IDs total.
- AFTER: see Validation block below — the drop must equal exactly 102 (not-slow collected) /
  103 (all node IDs).

### Targeted suites (post-deletion)

`pytest -q tests/test_jax_canonical_driver.py tests/test_pr_t3b_jax_operator_parity.py tests/test_jax_collision_operator_parity.py` —
result recorded in Validation block.

`PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest tests/test_deterministic_reference_rate_scaling.py tests/test_native_ad_parity.py tests/test_ap_rosenbrock_full_hm_closed_form.py -q` —
result recorded in Validation block (`test_deterministic_reference_rate_scaling.py` carries an
intentional in-flight strict-xfail from BD612 F-1 work, reported as-is, not touched by this PR).

`PYTHONPATH=src venv/bin/python -m pytest tests/test_native_ad_parity.py --collect-only -q` —
claim-gate node IDs still collected (see Validation block).

### promotion_check.py gate table

`PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/promotion_check.py --status` run
BEFORE and AFTER deletion; gate table diff is empty (see Validation block for both outputs).
Confirmed by static grep that no gate in `src/rabbit/config/claim_gates.py` and no logic in
`scripts/promotion_check.py` references any of the 10 deleted modules or 11 deleted test files.

## Validation

<!-- VALIDATION_BLOCK -->

## Cost line

- runtime_behavior_changed: no
- physics_behavior_changed: no
- known_blocker_reduced: yes (JAX twin-maintenance surface reduced ~2.4k source lines)
- blocker_movement_ratio: 0.15
- validation_strengthened: no
- cost_effectiveness_verdict: ACCEPT
