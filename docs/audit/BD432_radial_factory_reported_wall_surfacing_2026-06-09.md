# BD432 radial factory reported-wall surfacing

Date: 2026-06-09

Status: IMPLEMENTED and locally verified. This is an artifact tooling fix, not
solver validation and not a physics-output change.

## Problem

BD429's q4 collision-on LRS/non-LRS endpoint artifact already serialized AP65
payload radial-factory subblock timers under span-level
`fb69_source.summary`. The directory summary merged those fields into
`fields`, but `component_wall_attribution.reported_components` still reported
the radial-factory subwalls as zero.

The cause was a depth mismatch:

- exclusive component attribution correctly selected
  `h_refinement_attempt` rows so residual arithmetic stayed per-attempt;
- source-summary subwalls are span-level totals and are intentionally not copied
  onto child h-refinement attempts;
- reported components were computed only from the selected h-refinement rows.

This made the next payload optimization target look invisible even though the
runtime artifact contained the evidence.

## Fix

`scripts/summarize_perf_artifacts.py` now collects span rows that gained
source-summary overlays and uses them only as a fallback for zero-valued
`reported_components`. Exclusive component buckets and residual unattributed
wall remain based on the selected attribution rows.

The fallback is intentionally reported-only:

- it does not change `components`;
- it does not add to `attributed_wall_seconds_total`;
- it does not change `residual_unattributed_wall_seconds`;
- it records `reported_component_fallback_row_count` and the component names
  that came from the fallback.

## Real artifact check

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py diagnostic_outputs/bd429_q4_reuse_atol_floor > /tmp/bd429_summary_after_bd432.json
```

Extracted values from `/tmp/bd429_summary_after_bd432.json`:

| field | wall seconds | nonzero rows |
|---|---:|---:|
| `payload` | `657.8151970283943` | `16` |
| `payload_pstf_radial_factory` | `155.20703226840124` | `8` |
| `payload_pstf_radial_factory_provider_build` | `128.73353184806183` | `8` |
| `payload_pstf_radial_factory_radial_grid_kwargs` | `14.6129017326748` | `8` |
| `payload_pstf_radial_factory_process_config` | `23.623696720344014` | `8` |

Exclusive attribution remained:

| value | seconds |
|---|---:|
| `total_wall_seconds` | `2770.4276301520877` |
| `attributed_wall_seconds_total` | `2137.7443896837067` |
| `residual_unattributed_wall_seconds` | `632.683240468381` |

The component checker still reports:

```text
PASS component wall attribution
```

## Interpretation

The next payload-side target is now visible from existing long-run evidence:
BD429 spends about `128.7 s` in LRS radial-factory provider build under
reported overlap timers. This does not by itself prove a safe cache/refactor,
because the provider path includes temperature-dependent radial grid kwargs.
It does justify PR-E2 measurement/refactor work against the provider-build and
grid-kwargs split rather than another claim wrapper.

## Cost-effective audit

Verdict: ACCEPT AS ATTRIBUTION/READOUT BLOCKER FIX. This patch should not be
counted as a runtime speedup PR.

| Field | Value |
|---|---|
| Code implementation delta | `+47 / -0 / net +47` lines in `scripts/summarize_perf_artifacts.py` |
| Regression-test delta | `+73 / -0 / net +73` lines in `tests/test_summarize_perf_artifacts.py` |
| Documentation/ledger delta | `+119 / -0 / net +119` lines in this audit note plus `+1` validation-ledger row |
| Token use | `UNAVAILABLE`; this shell/harness does not expose per-PR token usage |
| Runtime speedup | `0%`; no solver code or physical evolution changed |
| Artifact/readout blocker movement | `1/1` for the specific false-zero reported radial-factory subwall |
| Runtime blocker movement | `0/1`; the provider-build subwall is measured but not reduced |
| Gate added | no |
| Raw physics state changed | no |
| Default optimization enabled | no |

The blocker moved here is the attribution/readout blocker: the payload
provider-build target can now be read from existing runtime evidence. The
remaining runtime blocker is still open and requires a separate provider-build
measurement/refactor PR.

Review caveat: the reported fallback is conservative for mixed-schema future
artifacts. If a reported component is already nonzero on selected h-refinement
rows, the span-level fallback for that component is skipped to avoid
double-counting. That can under-report a mixed artifact, but it cannot make
exclusive attribution or the checker falsely pass.

## Validation

| Command | Result | Notes |
|---|---:|---|
| `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_summarize_perf_artifacts.py::test_summarizer_reports_span_source_summary_subwalls_with_attempt_attribution --tb=short` | passed | RED/GREEN regression for h-attempt attribution plus span source-summary reported subwalls. |
| `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_summarize_perf_artifacts.py --tb=short` | passed | Full summarizer suite, `20 passed`. |
| `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py diagnostic_outputs/bd429_q4_reuse_atol_floor > /tmp/bd429_summary_after_bd432.json` | passed | Real long-run artifact now surfaces radial-factory reported subwalls. |
| `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py diagnostic_outputs/bd429_q4_reuse_atol_floor` | passed | Checker still reports `PASS component wall attribution`. |
