# BD433-BD436 Provider-Build Subwall Attribution

Date: 2026-06-10

## Scope

This PR does not change physics, collision source values, solver defaults, or
promotion readiness.  It closes a measured attribution blocker from the q4
collision-on endpoint path: `payload_pstf_radial_factory_provider_build` was a
large reported wall bucket, but the provider-internal split was not serialized
through AP65 rows, full span-ladder rows, or the artifact summarizer.

QKE remains out of scope.  This is private augmented Type-I no-QKE AP65/Rodas5P
telemetry work only.

## What Changed

- `src/rabbit/transport/augmented_collision_bridge.py` records provider-build
  diagnostics for:
  - geometric table lookup/build,
  - provider cache-key construction,
  - radial-grid build,
  - moment-weight assembly,
  - batch construction,
  - static-bundle cache store,
  - static-bundle cache hit/enabled counters and process/cache-key counts.
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py` forwards those
  diagnostics through AP65 payload traces and row summaries.
- `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`
  forwards the row fields through terminal wall-attribution rows, nested
  freedom-composition rows, and parent selected summaries.
- `scripts/summarize_perf_artifacts.py` surfaces the provider subwalls as
  reported overlap components.  They are intentionally not added to exclusive
  attribution totals because they sit below `payload_pstf_radial_factory` /
  `provider_build`.

## Long-Run Evidence

Final current-code run:

`diagnostic_outputs/bd436_q4_provider_subwall_parent_summary_fixed/`

Command recorded in `command.txt`.  `/usr/bin/time -v`:

- exit status: `0`
- elapsed: `49:33.27`
- max RSS: `3,136,900 KB`

Artifact structure:

- one final JSON artifact;
- one resolution row;
- two nested freedom-composition rows;
- sixteen nested span rows;
- component checker: `PASS component wall attribution`.

Endpoint rows:

| freedom case | T_gamma MeV | N_eff_3T | Yp | D/H | Sigma_H |
|---|---:|---:|---:|---:|---:|
| weak + collision | 0.009139616879 | 3.0348010165 | 0.2420001914 | 2.492935840e-05 | 5.516438319e-31 |
| weak + non-LRS + collision | 0.009139614045 | 3.0348087180 | 0.2420165219 | 2.493028169e-05 | 3.328675517e-31 |

Top-level `passed=false` remains expected for this one-resolution, no
resolution-convergence artifact.  The executable row status is not failed:
`.summary.execution_passed=true`, `.summary.failed_or_exception_rows=0`,
`.summary.controlled_flrw_lrs_nonlrs_default_on_blocker_passed=true`, and
failure regions are empty.

## Component Wall Table

From `bd436_perf_summary.json`:

| component | wall seconds | rows |
|---|---:|---:|
| total | 2908.487533 | 16 |
| payload | 698.807686 | 16 |
| phase2_corrector | 1361.309689 | 16 |
| host_jacobian | 164.606743 | 16 |
| outer_linear_system | 7.277825 | 16 |
| jax_compile | 0.000000 | 0 |
| jax_runtime | 0.000000 | 0 |
| attributed total | 2232.001943 | 16 |
| residual unattributed | 676.485589 | 16 |

Reported overlap provider/factory split:

| reported component | wall seconds | rows |
|---|---:|---:|
| payload_pstf_radial_factory | 166.454789 | 8 |
| payload_pstf_radial_factory_provider_build | 138.577701 | 8 |
| payload_pstf_radial_provider_grid_build | 81.145256 | 5 |
| payload_pstf_radial_provider_static_bundle_store | 24.213437 | 5 |
| payload_pstf_radial_provider_cache_key | 22.728966 | 8 |
| payload_pstf_radial_factory_process_config | 24.649567 | 8 |
| payload_pstf_radial_factory_radial_grid_kwargs | 15.216273 | 8 |
| payload_pstf_radial_provider_moment_weights | 0.960158 | 5 |
| payload_pstf_radial_provider_geometric_table | 0.379642 | 8 |
| payload_pstf_radial_provider_batch_build | 0.007589 | 5 |

Interpretation: the previously opaque provider-build bucket is dominated by
radial-grid construction, static-bundle store, and cache-key work.  Batch
construction is not a material target in this q4 endpoint workload.

## Negative / Corrected Evidence

- BD433 first attempted the long run without the exact thermal-start flags and
  failed fast with:
  `FLRW thermal neutrino start requires zero shear and zero initial A perturbation`.
  The failed attempt is preserved under
  `diagnostic_outputs/bd433_q4_provider_subwall_collision_on/failed_attempts/`.
- BD434 completed the q4 endpoint workload, but its final artifact still had
  false-zero provider subwalls at parent summary depth.  The child AP65 rows
  had enough information, which localized the propagation bug to the full
  span-ladder terminal/selected summary copier.
- BD435 ran after row-level propagation and proved nonzero provider subwalls in
  nested rows and summarizer output, but top-level selected provider summaries
  still read zero.
- BD436 ran after the nested parent-summary fix and shows nonzero provider
  subwalls at top summary, resolution row, nested span summary, span rows, and
  summarizer reported-component depths.

## Tests

The new tests are not count locks.  They fail if real provider diagnostics do
not travel through the runtime payload path, AP65 trace summary, full span-row
copy, nested parent summary, and summarizer reported-component path.

Key regressions:

- provider diagnostics are attached to real radial providers;
- compact payload summaries propagate through `_record_payload_trace`;
- full span rows propagate provider subwalls from child FB69 rows;
- nested freedom-composition parent summaries aggregate child `span_summary`
  provider subwalls;
- summarizer reports provider subwalls without adding them to exclusive totals.

## Cost-Effectiveness Self-Audit

| item | value |
|---|---|
| added_lines | 882 |
| deleted_lines | 0 |
| net_lines | 882 |
| files_touched | 11 |
| token_use_exact | UNAVAILABLE |
| token_use_basis | Harness did not expose an exact token counter; not fabricated. |
| runtime_behavior_changed | no; telemetry serialization only, no solver default or physics-output path changed |
| physics_behavior_changed | no |
| known_blocker_reduced | yes; `provider_build` is no longer opaque in final endpoint artifacts |
| blocker_movement_ratio | 0.25 |
| validation_strengthened | yes |
| cost_effectiveness_verdict | ACCEPT_WITH_LIMITS |
| cost_basis | `git diff --numstat` tracked diff before adding this audit note: 711 insertions, 0 deletions across 10 files; audit note line count after this table is included: 171 lines; source/tooling insertions: 376; test insertions: 333; validation ledger insertions: 2. |
| runtime speedup claimed | 0% |
| default optimization enabled | no |
| raw state changed/clipped | no |
| new readiness/manifest/hash/figure gate | no |
| remaining blocker | reduce or avoid the measured radial-grid/static-store/cache-key provider-build cost without changing physics outputs |

## Next PR

The next implementation PR should be an opt-in payload provider cache/refactor
experiment aimed at the measured subwalls:

1. prove fixed-state equality for `dQ_nue_pair_N`, `dQ_nux_bank_N`, and
   `dA_modes` with the refactor off/on;
2. target radial-grid/static-bundle/cache-key work first;
3. keep it opt-in until PR-B parity/floor evidence remains clean and fixed-state
   equality passes;
4. do not target batch construction as a primary blocker for q4.
