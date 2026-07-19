# BD427 Freedom-Composition Cache Cleanup

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 collision-on
pairwise parity work.  QKE remains out of scope.  This note does not claim
public production, publication readiness, Bianchi validity, resolution
convergence, or default-on optimization.

## Blocker

BD426 showed that collision-on q4 thermal-start pairwise parity does not reach
the cold endpoint for the zero-shear non-LRS branch before host memory becomes
unsafe.  LRS collision-on completed, and the first non-LRS span completed, but
the run was interrupted after RSS reached about 39 GB.

## Code Change

Freedom-composition cases now snapshot the AP65 runtime-cache keys before each
child case and release only the AP65 entries created by that child:

```text
continuous_ap65_radial_grid_cache
continuous_ap65_source_factory_cache
continuous_ap65_live_source_grid_cache
```

The inner freedom-composition cleanup deliberately preserves resolution-level
prewarm entries that existed before the child started and does not clear global
JAX in-process caches.  The existing outer resolution-ladder cleanup still
releases the full child runtime cache after the resolution case finishes.  No
new flag, gate, readiness wrapper, manifest, or physics output transformation was
added.

## Regression Test

`test_bd427_freedom_composition_clears_collision_caches_between_cases` simulates
a direct collision-on LRS composition child followed by a collision-on non-LRS
child.  The fake child builder writes one entry into each AP65 runtime cache.
The regression asserts that the second child sees zero child-owned entries before
its first source call.

`test_bd427_nested_freedom_composition_preserves_resolution_prewarm_cache`
covers the real BD426/BD427 shape: an outer resolution case prewarms collision
caches, then calls nested LRS/non-LRS freedom-composition children with the same
runtime cache.  It asserts the nested cleanup removes child-owned entries while
preserving the outer prewarm entry until the existing outer resolution cleanup
clears it.

RED result before the fix:

```text
At index 1 diff: ('nonlrs_s2', 1, 1, 1) != ('nonlrs_s2', 0, 0, 0)
```

GREEN result after the fix:

```text
2 passed in 2.26s
```

The nearby runtime-cache sharing test still passes for normal nested span rows;
the cleanup is only at the freedom-composition child boundary.

## Long-Run Verification

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_q4_thermal_start_lrs_nonlrs_collision_on_parity.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd416_pr_n2_endpoint_ab/q4_pairwise_collision_on_thermal_case.json \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --weak-correction-level 0 \
  --sigma-plus0 0.0 \
  --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun \
  --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --stop-at-T-gamma-MeV 0.01 \
  --progress-jsonl
```

The run was manually interrupted with SIGINT during the second non-LRS span to
avoid OOM.  `/usr/bin/time -v` recorded:

```text
Command terminated by signal 2
Elapsed wall: 59:16.56
Maximum resident set size: 37418676 KB
```

The final JSON again remained a partial prewarm placeholder, so component-wall
attribution is unavailable and the checker correctly failed:

```text
FAIL component wall attribution
- component attribution has no rows
```

Preserved sidecars:

```text
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_progress.jsonl
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_partial_run_forensics.json
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_bd426_comparison.json
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_time.txt
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_perf_summary.json
diagnostic_outputs/bd427_collision_on_composition_cache_cleanup_q4/bd427_component_wall_check.txt
```

## BD426 Comparison

| metric | BD426 | BD427 | delta |
|---|---:|---:|---:|
| progress rows recovered | 9 | 9 | 0 |
| LRS cold endpoint reached | yes | yes | unchanged |
| non-LRS rows completed | 1 | 1 | unchanged |
| max RSS KB | 39075768 | 37418676 | -1657092 |
| max RSS fraction | 1.000 | 0.9576 | -4.24% |

Recovered physics rows match BD426 for displayed observables.  Selected rows:

| freedom case | span | N_eff_3T BD426 | N_eff_3T BD427 | Yp BD426 | Yp BD427 | builds BD426 | builds BD427 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LRS collision-on | `[3.0, 4.0]` | 3.0347646937660633 | 3.0347646937660633 | 0.24199880575522173 | 0.24199880575522173 | 12795 | 12795 |
| LRS collision-on | `[4.0, 4.8]` | 3.034764639534682 | 3.034764639534682 | 0.24200013462618128 | 0.24200013462618128 | 10242 | 10242 |
| non-LRS collision-on | `[0.0, 1.0]` | 9.330920989913897 | 9.330920989913897 | 1e-30 | 1e-30 | 3561 | 3561 |

## Interpretation

This PR removes a real cross-composition cache-retention bug, and it modestly
reduces peak RSS in the same failed long run.  It does not solve the collision-on
PR-B parity blocker.  The remaining blocker is now sharper:

```text
non-LRS collision-on source/payload memory grows inside the non-LRS child itself,
not only because LRS caches survive across the composition boundary.
```

The next PR should inspect and deflate non-LRS collision payload internals:
angular/radial source factories, S2 angular tables, radial grid cache entry
sizes, and whether non-LRS source payloads retain large per-step arrays that can
be summarized then dropped.

## Validation

```text
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd427_freedom_composition_clears_collision_caches_between_cases \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd427_nested_freedom_composition_preserves_resolution_prewarm_cache \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd271_resolution_runtime_cleanup_clears_child_caches \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd165_dynamic_laguerre_collision_prewarm_uses_runtime_caches \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd194_freedom_composition_forwards_span_progress_callback \
  --tb=short
5 passed

venv/bin/python -m py_compile \
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
passed

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_summarize_perf_artifacts.py --tb=short
19 passed

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py --tb=short
271 passed, 1 skipped, 2 existing deterministic-reference warnings

git diff --check
passed
```

## Cost-Effectiveness Self-Audit

```text
added_lines: 443
deleted_lines: 0
net_lines: 443
files_touched: 2 code/test files, 1 tracked audit note, 1 tracked validation ledger row
token_use_exact: UNAVAILABLE
token_use_basis: harness did not expose an exact token counter
runtime_behavior_changed: yes
physics_behavior_changed: no
known_blocker_reduced: yes
blocker_movement_ratio: 0.25
blocker_movement_scope: freedom-composition cache carryover fixed and same-run max RSS reduced 4.24%, but collision-on non-LRS parity still fails by memory before endpoint
validation_strengthened: yes
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```
