# BD574 Phase-2 Wall Summary Deflation

Date: 2026-06-26

## Scope

BD574 is a deflation PR.  It does not change physics, runtime policy, endpoint
recipe, or artifact semantics.  After BD571-BD573 eliminated the obvious
cap/slack-only phase-2 endpoint knobs, the branch returns to reducing active
telemetry code surface that directly slows AP65 endpoint work.

The target is
`src/rabbit/validation/augmented_ap65_trace_summary.py`, specifically the
phase-2 wall-time serialization block.  The old code repeated explicit
`float(stats.get(...))` / `int(stats.get(...))` entries for each
`*_wall_seconds_{total,count,max}` field.  The module already had
`_wall_seconds_summary_fields()`, used for dynamic collision payload wall
prefixes.  BD574 extends the same table-driven pattern to phase-2 corrector
wall prefixes.

## Change

- Added `_PHASE2_CORRECTOR_WALL_PREFIXES`.
- Replaced the repeated phase-2 wall-time dict entries with:

```python
**_wall_seconds_summary_fields(
    _wall_seconds_stat_summary,
    _PHASE2_CORRECTOR_WALL_PREFIXES,
),
```

No new runtime gate, manifest, hash, readiness check, or optimization policy was
added.

## Line Cost

Tracked production line delta:

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `src/rabbit/validation/augmented_ap65_trace_summary.py` | 32 | 482 | -450 |

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

## Validation

Commands run:

```bash
python -m py_compile src/rabbit/validation/augmented_ap65_trace_summary.py

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd333_phase2_step_attempt_surfaces_ab2_and_solve_wall_timers \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd360_phase2_adaptive_pair_sums_ab2_counters_across_attempts \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd440_source_evaluation_records_payload_build_subwall \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_totals_pstf_provider_subwalls_from_compact_payload \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_extracted_builder_matches_legacy_wrapper

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_phase2_summary_metrics_sums_wall_timers \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_preserves_phase2_replay_and_corrector_telemetry
```

Results:

- py_compile: PASS
- AP65 RHS targeted trace-summary tests: PASS, 5 tests
- span-ladder phase2 telemetry targeted tests: PASS, 2 tests

The full AP65 RHS file was also started for broader verification:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py
```

Result: PASS, 315 tests, 3 known truncation-guard warnings.

The full span-ladder test file was also run because phase-2 wall fields are
propagated into span-row artifacts:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
```

Result: PASS, 304 tests, 1 skipped, 2 known deterministic-reference warnings.

## Anti-Drift / Cost-Effectiveness

- Runtime behavior changed: no.
- Physics behavior changed: no.
- Raw observables changed/clipped/hidden: no.
- Optimization default enabled: no.
- New gate/wrapper added: no.
- Known blocker reduced: yes, active phase-2 telemetry surface reduced by
  deleting repeated serialization code in one of the highest-churn AP65 files.
- Blocker movement ratio: 0.35.  This does not move endpoint wall directly, but
  it reduces the production code surface that must be audited before the next
  endpoint-facing phase-2 or payload PR.
- Cost-effectiveness verdict: ACCEPT.  The PR is net-negative in production
  code and reuses an existing local helper rather than creating another wrapper.

## Next PR

BD575 should continue deflation only if it deletes active production/test
surface that directly impedes endpoint work.  Otherwise the next runtime PR
should target payload/provider deflation or a genuinely selective phase-2
controller with same-recipe endpoint evidence.  PR-B LRS/non-LRS parity and the
cold `N_eff_3T >= 3.0` floor tripwire remain the default-on blocker.
