# BD428 q4 Dynamic Laguerre Cache Bounds

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 collision-on
pairwise parity work.  QKE remains out of scope.  This note does not claim
public production, publication readiness, Bianchi validity, resolution
convergence, or default-on optimization.

## Blocker

BD426 and BD427 both showed the same practical blocker: the LRS collision-on
branch reached the cold endpoint, but the zero-shear non-LRS collision-on branch
could not progress beyond the first non-LRS span before host memory became
unsafe.  BD427 removed cross-composition child cache carryover and reduced peak
RSS by 4.24%, but the run still had to be interrupted during the second non-LRS
span.

The q4 case used Gauss-Laguerre dynamic collision payloads but did not set
`source_factory_cache_max_entries` or `radial_grid_cache_max_entries`.  Existing
cache-pruning machinery was already present in
`evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source`, but the FB70
resolver only auto-enabled bounds for high-order Laguerre cases.  For q4, source
factory and radial-grid caches were therefore unbounded unless the case file
manually opted in.

## Code Change

Dynamic q-Laguerre collision cases now receive the existing cache bounds at all
orders unless the case explicitly supplies its own limits:

```text
source_factory_cache_max_entries = 8
radial_grid_cache_max_entries = 72
```

The high-order budget logic remains separate: high-order q-Laguerre still gets
the existing wall/max-step/restart-window controls.  BD428 only changes cache
bounding for dynamic q-Laguerre collision payload construction.  No new flag,
gate, readiness wrapper, manifest, or physics output transformation was added.

## Regression Test

`test_bd428_low_order_dynamic_laguerre_bounds_cache_by_default` replaces the old
BD340 expectation that low-order q-Laguerre cache bounds were opt-in.  RED before
the fix:

```text
KeyError: 'source_factory_cache_max_entries'
```

GREEN after the fix:

```text
3 passed in 3.02s
```

The explicit override test still passes, so user/case-specified cache caps are
not overwritten.

## Long-Run Verification

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_q4_thermal_start_lrs_nonlrs_collision_on_parity.json \
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

`/usr/bin/time -v`:

```text
Elapsed wall: 1:20:11
Maximum resident set size: 11423752 KB
Exit status: 0
```

Post-run artifact tools:

```text
scripts/summarize_perf_artifacts.py ... -> exit 0
scripts/check_component_wall_attribution.py ... -> exit 0
PASS component wall attribution
```

Preserved generated sidecars:

```text
diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_progress.jsonl
diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_perf_summary.json
diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_component_wall_check.txt
diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_time.txt
diagnostic_outputs/bd428_q4_collision_on_cache_bounds/bd428_bd426_bd427_comparison.json
```

## BD426/BD427 Comparison

| metric | BD426 | BD427 | BD428 |
|---|---:|---:|---:|
| process result | SIGINT before endpoint | SIGINT before endpoint | exit 0 |
| elapsed | 57:24.91 | 59:16.56 | 1:20:11 |
| max RSS KB | 39075768 | 37418676 | 11423752 |
| progress rows recovered | 9 | 9 | 16 |
| LRS collision-on endpoint rows | 8 | 8 | 8 |
| non-LRS collision-on endpoint rows | 1 | 1 | 8 |
| non-LRS cold endpoint reached | no | no | yes |
| component attribution | unavailable | unavailable | PASS |

BD428 max RSS is 69.46% lower than BD427 and 70.77% lower than BD426.  The
previous non-LRS memory blocker moved from "cannot progress past the first
non-LRS span safely" to "q4 collision-on LRS/non-LRS pair reaches the cold
endpoint with passing component attribution."

## Endpoint Physics

The final artifact keeps raw observables and reports:

| branch | T_gamma final MeV | N_eff_3T | Yp | Sigma_H |
|---|---:|---:|---:|---:|
| LRS collision-on | 0.009139629953155422 | 3.034764639534682 | 0.24200013462618128 | 5.641674665929347e-31 |
| zero-shear non-LRS collision-on | 0.009139615607474304 | 3.034804605447281 | 0.24201654720110613 | 3.328675517278998e-31 |

The artifact summary reports:

```text
execution_passed = true
controlled_flrw_lrs_nonlrs_cold_endpoint_pair_ready = true
controlled_flrw_lrs_nonlrs_controls_identical_except_geometry = true
controlled_flrw_lrs_nonlrs_default_on_blocker_status = passed_pr_b_neff_floor_and_lrs_nonlrs_parity
controlled_flrw_lrs_nonlrs_neff_delta = 3.996591259891602e-05
controlled_flrw_lrs_nonlrs_default_on_blocker_passed = true
```

Top-level `passed=false` is not a physics or execution failure here.  This run
contains one resolution case, so resolution-convergence readiness remains false:
`blocking_next_step=tighten_resolution_or_solver_tolerance_until_terminal_deltas_converge`.

## Component Wall Attribution

BD428 summary for the final JSON:

| component | wall seconds |
|---|---:|
| total row wall | 4715.05139056209 |
| payload | 2431.3196659345413 |
| phase2 corrector | 1367.095819499169 |
| host Jacobian | 165.15761118230876 |
| outer linear system | 8.063925389142241 |
| JAX compile | 0.0 |
| JAX runtime | 0.0 |
| residual unattributed | 743.4143685569288 |

`W/J` shape remains `[62, 62]`.  Dense linear solve is still not the q4 target.

## Interpretation

BD428 is the first current-head collision-on q4 run in this sequence to complete
both LRS and zero-shear non-LRS endpoint rows under the PR-B thermal-start
pairwise setup.  The concrete blocker moved: unbounded q4 dynamic Laguerre
collision caches were the dominant memory failure mode in BD426/BD427.

This does not validate the solver globally.  It does not prove resolution
convergence, high-q convergence, Bianchi validity, or public readiness.  It also
does not make any optimization default-on outside this private AP65/Rodas5P
staging path.  The next blocker is no longer "non-LRS collision q4 cannot reach
endpoint"; it is now endpoint/resolution convergence and the remaining component
wall profile: payload is still the largest wall bucket, phase2 remains second,
and residual unattributed is still 15.77%.

## Validation

```text
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd428_low_order_dynamic_laguerre_bounds_cache_by_default \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd185_explicit_dynamic_laguerre_cache_bounds_are_not_overridden \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd216_dynamic_laguerre_pair_leg_order_is_case_local_by_default \
  --tb=short
3 passed

BD428 long q4 collision-on pairwise run
exit 0, elapsed 1:20:11, max RSS 11423752 KB

scripts/summarize_perf_artifacts.py diagnostic_outputs/bd428_q4_collision_on_cache_bounds
exit 0

scripts/check_component_wall_attribution.py diagnostic_outputs/bd428_q4_collision_on_cache_bounds
exit 0, PASS component wall attribution

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_typeI_weak_network_3t_solve.py::test_bd185_prune_mutable_cache_entries_preserves_active_keys \
  tests/test_jax_augmented_typeI_replay.py::test_bd185_radial_grid_cache_diagnostics_do_not_affect_fingerprint \
  --tb=short
2 passed

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py --tb=short
271 passed, 1 skipped, 2 existing deterministic-reference warnings

venv/bin/python -m py_compile \
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
passed

git diff --check
passed
```

## Cost-Effectiveness Self-Audit

```text
added_lines: 248
deleted_lines: 9
net_lines: 239
files_touched: 2 code/test files, 1 tracked audit note, 1 tracked validation ledger row
token_use_exact: UNAVAILABLE
token_use_basis: harness did not expose an exact token counter
runtime_behavior_changed: yes
physics_behavior_changed: no
known_blocker_reduced: yes
blocker_movement_ratio: 0.85
blocker_movement_scope: collision-on q4 pairwise run now completes both LRS and zero-shear non-LRS endpoint rows with max RSS reduced from 37.4 GB interrupted to 11.4 GB exit 0
validation_strengthened: yes
cost_effectiveness_verdict: ACCEPT
```
