# BD575 Source/Host Wall Summary Deflation

Date: 2026-06-26

## Scope

BD575 is a deflation PR.  It continues the BD574 pattern in
`src/rabbit/validation/augmented_ap65_trace_summary.py` by folding remaining
wall-time serialization blocks that all share the same
`*_wall_seconds_{total,count,max}` shape.

This PR does not change physics, runtime policy, endpoint behavior, raw
observables, or default optimization state.  It does not add a gate, manifest,
hash check, readiness wrapper, or new diagnostic layer.

## Change

- Added `_SOURCE_AND_HOST_WALL_PREFIXES`.
- Replaced repeated source-evaluation, nonpayload, boundary, rejected-step,
  host-Jacobian, and nonprobe wall-time dict entries with the existing
  `_wall_seconds_summary_fields()` helper.
- Preserved non-wall provenance fields such as
  `rejected_step_replay_wall_seconds_source` and host JAX unavailable reasons.

## Line Cost

Tracked production line delta:

| File | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| `src/rabbit/validation/augmented_ap65_trace_summary.py` | 19 | 179 | -160 |

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

## Validation

Commands run:

```bash
python -m py_compile src/rabbit/validation/augmented_ap65_trace_summary.py

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_extracted_builder_matches_legacy_wrapper \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_trace_summary_totals_pstf_provider_subwalls_from_compact_payload \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd333_phase2_step_attempt_surfaces_ab2_and_solve_wall_timers \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd440_source_evaluation_records_payload_build_subwall \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd546_dynamic_collision_payload_build_fields_filter_samples

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_preserves_phase2_replay_and_corrector_telemetry \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_preserves_fb69_adaptive_step_telemetry
```

Results:

- py_compile: PASS
- AP65 RHS targeted trace-summary tests: PASS, 5 tests
- span-ladder targeted telemetry tests: PASS, 2 tests

Full-file verification was started:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
```

Results:

- AP65 RHS full file: PASS, 315 tests, 3 known truncation-guard warnings.
- Span-ladder full file: PASS, 304 tests, 1 skipped, 2 known
  deterministic-reference warnings.

## Anti-Drift / Cost-Effectiveness

- Runtime behavior changed: no.
- Physics behavior changed: no.
- Raw observables changed/clipped/hidden: no.
- Optimization default enabled: no.
- New gate/wrapper added: no.
- Known blocker reduced: yes, active source/host telemetry serialization
  surface reduced in the main AP65 trace-summary file.
- Blocker movement ratio: 0.30.  This is not endpoint-wall progress, but it
  removes repeated production telemetry plumbing that directly increases AP65
  review and modification cost.
- Cost-effectiveness verdict: ACCEPT_WITH_LIMITS.  Net-negative production
  code, existing helper reused, no new policy surface.

## Next PR

BD576 is the required five-PR review checkpoint for BD571-BD575.  It must be a
review/adjustment document, not another deflation PR, unless review finds an
immediate must-fix regression.
