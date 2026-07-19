# BD562 Span-Ladder Deflation Review

Date: 2026-06-26

Scope: third-party style review checkpoint for BD558-BD561.  This is a
maintainability review, not a solver validation or endpoint-performance claim.

## Verdict

REQUEST PIVOT AFTER THIS CHECKPOINT.

BD558-BD561 did reduce several overlarge span-ladder helper bodies, but the
aggregate patch still added net code surface and did not move the measured
endpoint wall, activation/cold blocker, PR-B parity blocker, or physics-readout
blocker.  Continuing deflation-only PRs from here would violate the
cost-effective anti-drift rule unless the next extraction directly enables an
endpoint-wall or physics-correctness change.

## PR Ledger

| PR | Commit | Main change | Function movement | Verification |
| --- | --- | --- | --- | --- |
| BD558 | `4151b9a` | Extract endpoint observable/status extraction from `_row_from_source`. | `_row_from_source` 1240 -> 1218. | Full span-ladder: 301 passed, 1 skipped, 2 warnings. |
| BD559 | `07c951a` | Share child terminal/provenance outcome extraction. | `_resolution_case_row_from_child` 1061 -> 1024; `_freedom_composition_row_from_child` 746 -> 709. | Full span-ladder: 302 passed, 1 skipped, 2 warnings. |
| BD560 | `6e652e4` | Split selected payload-build summary assembly. | `_resolution_case_row_from_child` 1024 -> 918. | Full span-ladder: 303 passed, 1 skipped, 2 warnings. |
| BD561 | `012dd64` | Split h-refinement attempt linear-system payload. | `_h_refinement_attempt_summary` 376 -> 305. | Full span-ladder: 304 passed, 1 skipped, 2 warnings. |

## Line Cost

Aggregate BD558-BD561 diff relative to BD557:

| Area | Insertions | Deletions | Net |
| --- | ---: | ---: | ---: |
| Source | 340 | 297 | +43 |
| Tests | 213 | 0 | +213 |
| Docs | 22 | 4 | +18 |
| Total | 575 | 301 | +274 |

Exact token counters: UNAVAILABLE.  The local harness exposes no per-PR token
meter, so no synthetic token estimate is reported.

## Current Largest Functions

Measured after BD561:

| Function | Lines | Review note |
| --- | ---: | --- |
| `_build_augmented_continuous_ap65_full_bbn_span_ladder_artifact_impl` | 5655 | Still the dominant orchestration monolith.  Do not continue shaving it unless the extraction unlocks endpoint runs or physics fixes. |
| `_row_from_source` | 1218 | Still large; contains mixed telemetry/provenance assembly. |
| `_resolution_case_row_from_child` | 918 | Improved but still long. |
| `_freedom_composition_row_from_child` | 709 | Improved but still long. |
| `_attach_h_refinement_metadata` | 505 | Candidate only if h-refinement endpoint evidence requires it. |
| `_h_refinement_attempt_summary` | 305 | Improved enough for now. |

## Adversarial Review Findings

1. Real blocker moved: PARTIAL.  The blocker moved is agent/developer
   navigability, not endpoint wall or physics correctness.
2. Runtime-linked telemetry preserved: YES.  The refactors move existing row
   fields into helpers and tests exercise row-derived values, not schema-only
   placeholders.
3. False-green tooling added: NO.  No new gate, manifest, readiness wrapper, or
   figure/hash gate was added.
4. Raw state preserved: YES by inspection of touched paths.  Endpoint observable
   helper preserves missing/nonfinite values as unavailable fields and does not
   clip raw physics output.
5. Tests meaningful, not count-locks: MIXED-POSITIVE.  The tests cover field
   normalization and provenance, but they are still serialization-level tests.
   They should not be mistaken for endpoint performance or physics validation.
6. Main concern: aggregate net code grew by 274 lines while no endpoint
   improvement was measured.  This is acceptable only as a bounded cleanup
   checkpoint, not as the next repeated PR pattern.

## Verification Evidence

Fresh command after BD561:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
```

Result: `304 passed, 1 skipped, 2 warnings in 76.19s`.  The two warnings are
the existing deterministic-reference overflow/invalid multiply warnings in
`test_bd129_laguerre_resolution_energy_weights_are_unclipped_exp_q`.

Additional BD561 checks:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_bd561_h_refinement_linear_system_payload_normalizes_row_fields \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_pr_c0_h_refinement_attempt_surfaces_runtime_wall_telemetry \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py::test_fb70_refines_failed_span_h_max_and_preserves_attempt_telemetry
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m py_compile \
  src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py \
  tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py
git diff --check
```

Results: targeted tests `3 passed in 0.80s`; `py_compile` passed; `git diff
--check` passed.

## Cost-Effectiveness Verdict

BD558-BD561: ACCEPT WITH PIVOT REQUIRED.

Blocker movement ratio: 0.18.  The sequence reduced four local surfaces and
improved reviewability, but the measured endpoint blocker, PR-B parity/floor
tripwire, payload/provider factory cost, phase-2 warm-start behavior, and
long-run endpoint reliability were not directly improved.

Next PRs must produce quantitative endpoint-wall, endpoint-reach, or
physics-readout evidence.  Segment-only benchmarks remain allowed only when
clearly labeled and used to select the next endpoint experiment.

## Next Queue Recommendation

1. Run or reuse a bounded endpoint recipe to establish a fresh post-BD561
   baseline with raw observable deltas, AB2 counters, rejected counts,
   `N_eff_3T`, `Yp`, `D/H`, wall, and RSS.
2. Implement one endpoint-facing optimization candidate at a time:
   phase2 refined/coarse warm-start, or payload/provider factory deflation.
3. Require before/after endpoint wall evidence for the same recipe before
   calling a PR performance-positive.
4. Keep PR-B parity and cold `N_eff_3T >= 3.0` as default-on blockers.
5. Resume monolith deflation only where it removes code directly on the selected
   endpoint blocker path.
