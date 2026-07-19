# BD442/BD443 Payload Trace Subwall Attribution

Date: 2026-06-10

Scope: augmented Type-I PSTF no-QKE AP65 performance attribution.  QKE remains
out of scope.  This note records telemetry-surfacing work only; it does not
validate the solver and does not enable any default optimization.

## Claim Ledger

| Claim | Status | Evidence |
| --- | --- | --- |
| `_record_payload_trace` contributes to the source-evaluation envelope and should be reported as a named subwall. | IMPLEMENTED | `source_evaluation_payload_trace_wall_seconds_*` is recorded around `_record_payload_trace` in the RHS source-evaluation path. |
| A unit-level RHS timer test is sufficient to prove q4 artifact surfacing. | DEPRECATED | BD442 showed the direct RHS timer could pass while full q4 artifacts still omitted the field because the full-span ladder wrapper did not whitelist it. |
| Full-span q4 artifact surfacing now forwards payload-trace timing. | VALIDATED for full q4 artifact forwarding | BD443 bounded probe serialized payload-trace fields into the artifact and summarizer output; BD444 then completed the full q4 endpoint pair and surfaced payload-trace timing in the final long-run component table. |
| Raw physics state changed. | FORBIDDEN / not changed | This PR only records wall timers and updates summarization. No physics output clipping or default optimization was added. |

## Implementation

Changed runtime telemetry:

- `source_evaluation_payload_trace_wall_seconds_total`
- `source_evaluation_payload_trace_wall_seconds_count`
- `source_evaluation_payload_trace_wall_seconds_max`

The timer wraps `_record_payload_trace(...)` inside the existing source-evaluation
wall envelope, using the same source-evaluation subwall accounting as payload
build/provenance/JAX/live-RHS/device-get/operator-split timing.

Changed artifact/summarizer surfacing:

- full-span ladder terminal telemetry forwards the new total/count/max fields;
- selected source-evaluation subwall summaries include payload trace;
- `scripts/summarize_perf_artifacts.py` reports `source_evaluation_payload_trace`
  as a reported component, includes it in the
  `source_evaluation_minus_named_subwalls` gap calculation, and lists it in
  timer availability.

## BD442 Full q4 Long Run

Path:

`diagnostic_outputs/bd442_q4_payload_trace_subwall/`

Command:

`diagnostic_outputs/bd442_q4_payload_trace_subwall/bd442_command.txt`

Result:

- process exit: 0
- `/usr/bin/time` elapsed: 49:38.56
- max RSS: 3,129,852 KB
- selected wall: 2913.8878060849966 s
- source-evaluation wall: 1195.7067834298941 s
- source payload build wall: 700.0875332985888 s
- source payload provenance wall: 361.55539384501753 s
- phase-2 corrector wall: 1360.136962491728 s
- dynamic collision payload builds: 12196
- component residual unattributed: 681.447783099662 s
- checker: PASS component wall attribution

Critical finding:

BD442 did **not** surface `source_evaluation_payload_trace_wall_seconds_*` in
the final artifact.  This was a false-green telemetry path: the direct RHS timer
test passed, but the full-span ladder wrapper had a separate whitelist and
selected-summary list that omitted the new subwall.

## BD443 Bounded Forwarding Probe

Path:

`diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper/`

Command:

`diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper/bd443_command.txt`

Result:

- process exit: 1
- expected reason: `wall_time_budget_seconds` bounded partial run; not an endpoint validation
- `/usr/bin/time` elapsed: 2:21.44
- max RSS: 1,434,392 KB
- raw JSON payload-trace hits: 64
- first runtime row payload-trace count: 792
- selected payload-trace wall: 2.717730283853598 s
- component table payload-trace row count: 2
- component table payload-trace wall: 2.717730283853598 s
- timer availability:
  - missing count: 0
  - value count: 2
  - unavailable reason count: 0
- `source_evaluation_minus_named_subwalls` after payload-trace attribution:
  - parent source-evaluation wall: 98.38065973261837 s
  - named child wall: 96.87654200551333 s
  - residual gap: 1.50411772710504 s
  - gap fraction: 0.0152887542245902
- checker: PASS component wall attribution

Interpretation:

BD443 validates the artifact-forwarding fix on a bounded q4 partial artifact.
It does not replace BD442 as a long-run q4 measurement because the run stopped
by wall-time budget and produced non-endpoint rows.  The next fresh full q4 run
should now produce a long-run payload-trace component wall row.

## BD444 Full q4 Forwarding Confirmation

Path:

`diagnostic_outputs/bd444_q4_payload_trace_after_wrapper/`

Command:

`diagnostic_outputs/bd444_q4_payload_trace_after_wrapper/bd444_command.txt`

Result:

- process exit: 0
- `/usr/bin/time` elapsed: 48:47.31
- max RSS: 3,140,488 KB
- final JSON size: 28,192,584 bytes
- summary path:
  `diagnostic_outputs/bd444_q4_payload_trace_after_wrapper/bd444_perf_summary.json`
- component checker:
  `PASS component wall attribution`
- top-level `passed`: false, expected for one-resolution/no-convergence readiness
- `summary.execution_passed`: true
- failed or exception rows: 0
- PR-B blocker status:
  `passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
- controlled FLRW LRS/non-LRS `N_eff_3T` delta:
  7.701442438445838e-06
- controlled floor margin:
  0.034808717967143465

Component wall summary:

| Component | Wall seconds |
| --- | ---: |
| total wall | 2863.7824760479853 |
| attributed wall | 2201.5930251205573 |
| residual unattributed | 662.189450927428 |
| source evaluation | 1170.6737594259903 |
| source payload build | 689.9324971504393 |
| source payload trace | 61.58063135470729 |
| source payload provenance | 353.01989179180237 |
| phase-2 corrector | 1342.248983921425 |

Source-evaluation nested gap after payload-trace attribution:

- parent source-evaluation wall: 1170.6737594259903 s
- named child wall: 1151.4841334907687 s
- residual gap: 19.189625935221557 s
- gap fraction: 0.016391950174599195
- timer availability for payload trace:
  - row count: 16
  - value count: 16
  - missing count: 0

Endpoint rows:

| Freedom key | T_gamma MeV | N_eff_3T | Yp | D/H | Sigma_H | W/J |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `weak_rate_corrections+neutrino_collision_terms` | 0.009139616879151824 | 3.034801016530264 | 0.24200019137645587 | 2.4929358404512697e-05 | 5.516438318504755e-31 | [62, 62] |
| `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | 0.00913961404501975 | 3.0348087179727026 | 0.24201652194550552 | 2.493028169465174e-05 | 3.3286755172789884e-31 | [62, 62] |

Interpretation:

BD444 closes the BD442 false-green blocker for payload-trace surfacing: the same
full q4 family that previously completed without the field now completes with
payload trace visible in final JSON, reported components, nested gap arithmetic,
and timer availability.  This remains performance-attribution evidence, not
solver validation or promotion readiness.

## Validation Commands

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd391_wall_attribution_forwards_source_evaluation_subwalls \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd440_source_evaluation_records_payload_build_subwall \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd392_collision_relax_policy_uses_rhs_only_jax_for_stage_rhs \
  tests/test_summarize_perf_artifacts.py::test_summarizer_computes_component_attribution_residual
```

Result: 4 passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper \
  > diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper/bd443_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper \
  > diagnostic_outputs/bd443_payload_trace_budget_probe_after_wrapper/bd443_component_wall_check.txt
```

Result: checker PASS.  The summary reports `source_evaluation_payload_trace`
with `wall_seconds=2.717730283853598`.

## Remaining Blocker

Closed for payload-trace surfacing by BD444:

- exit 0 endpoint artifact;
- payload trace visible in final JSON;
- component table includes payload trace;
- source-evaluation nested residual measured after payload-trace attribution.

Remaining performance blockers are now downstream of this evidence: the q4 wall
is still dominated by phase-2 corrector wall, source payload build/provenance,
and a large exclusive residual.  Payload trace is no longer dark wall.
