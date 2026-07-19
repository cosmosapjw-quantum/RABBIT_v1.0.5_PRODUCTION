# BD565 Provider Static-Bundle Cache Negative Result

Date: 2026-06-26

## Status

BD565 was a same-recipe endpoint experiment and is **rejected** as a retained
optimization.  The experimental code was saved as a diagnostic diff and rolled
back before commit.

Artifact directory:

- `diagnostic_outputs/bd565_provider_static_bundle_endpoint/`

Retained evidence:

- `bd565_q4_provider_static_bundle_endpoint.json`
- `bd565_perf_summary.json`
- `bd565_component_check.txt`
- `bd565_time_v.txt`
- `bd565_run.log`
- `checkpoints/`
- `bd565_reverted_code_experiment.diff`

## Experiment

Candidate: enable the existing PSTF radial provider static-bundle cache inside
the live combined angular plus PSTF-radial collision source factory by changing
the live radial builder call from `provider_static_bundle_cache_enabled=False`
to `True`.

Why this was plausible:

- BD563 payload wall was `823.276482 s`.
- The provider runtime sub-wall was `149.317276 s`.
- The cache is already byte-bounded in
  `src/rabbit/transport/augmented_collision_bridge.py` by
  `_PSTF_RADIAL_PROVIDER_BUNDLE_CACHE_MAX_BYTES = 128 * 1024 * 1024` and
  `_PSTF_RADIAL_PROVIDER_BUNDLE_CACHE_MAX_ENTRIES = 64`.

Why it was still risky:

- PR-B parity and cold `N_eff_3T >= 3.0` remain default-on blockers.
- The cache only helps if repeated runtime mass-scale keys actually hit.

## Exact Endpoint Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd565_provider_static_bundle_endpoint/bd565_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd565_provider_static_bundle_endpoint/bd565_q4_provider_static_bundle_endpoint.json \
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
  --phase2-network-ab2-initial-guess-residual-guard-policy adaptive_trust_after_acceptance \
  --phase2-network-newton-jacobian-refresh-policy periodic \
  --phase2-network-newton-jacobian-refresh-interval 4 \
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd565_provider_static_bundle_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd565_provider_static_bundle_endpoint/jax_cache \
  --progress-jsonl
```

## BD563 vs BD565 Endpoint Result

| Metric | BD563 baseline | BD565 experiment | Delta |
| --- | ---: | ---: | ---: |
| exit status | 0 | 0 | 0 |
| `/usr/bin/time` elapsed s | 2535.68 | 2521.59 | -14.09 |
| max RSS KB | 4572180 | 4560308 | -11872 |
| selected row wall s | 2499.378611 | 2485.903417 | -13.475194 |
| phase2 corrector wall s | 1221.549889 | 1199.382130 | -22.167758 |
| payload wall s | 823.276482 | 839.577442 | +16.300960 |
| source nonpayload wall s | 68.368362 | 66.727208 | -1.641153 |
| host Jacobian wall s | 173.619921 | 171.220912 | -2.399009 |
| outer linear wall s | 6.949313 | 6.730759 | -0.218554 |
| residual unattributed wall s | 205.614644 | 202.264966 | -3.349678 |
| total rejected steps | 6 | 6 | 0 |
| total source evals | 87840 | 87840 | 0 |
| dynamic payload builds | 12198 | 12198 | 0 |
| provider runtime wall s | 149.317276 | 144.897606 | -4.419670 |
| static bundle enabled total | 0 | 16 | +16 |
| static bundle cache hit total | 0 | 0 | 0 |

Final all-freedom endpoint row:

| Observable | BD563 | BD565 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | 0.00913961404501975 | 0.00913961404501975 | 0 |
| `Yp` | 0.24201652194490023 | 0.24201652194490023 | 0 |
| `D/H` | 2.493028169464549e-05 | 2.493028169464549e-05 | 0 |
| `N_eff_3T` | 3.0348087179727026 | 3.0348087179727026 | 0 |
| `Sigma_H` | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 0 |

## Interpretation

SUPPORTED:

- Same-recipe endpoint run completed with exit status 0.
- Component attribution checker passed.
- Raw endpoint observables were unchanged to serialized precision.
- The experimental path enabled the static-bundle cache diagnostic at row depth.

CONTRADICTED:

- The target payload wall did not improve.  It worsened by `16.300960 s`.
- The cache did not hit: static bundle hit total stayed `0`.
- Work counts were unchanged: payload builds, source evals, steps, attempts, and
  rejections did not move.

PARTIAL:

- Selected wall improved by `13.475194 s`, but that improvement was carried by
  phase2/residual/host noise rather than the targeted payload/cache mechanism.
  This is not enough evidence to keep a default-on optimization before PR-B.

ACTIONABLE_NOW:

- Do not keep this default-on cache change.
- If provider-bundle caching is revisited, first change the runtime keying or
  reuse pattern so repeated dynamic mass-scale bundles actually hit, and keep
  the path opt-in until PR-B parity and cold floor blockers clear.

## Commands Run

Before endpoint run:

- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_jax_augmented_typeI_replay.py::test_dynamic_source_factory_cache_keeps_radial_factory_temperature_bound_by_default`
  - RED before code change, then PASS after code change.
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_collision_bridge.py::test_pstf_radial_provider_reuses_static_bundle_across_rebuilds tests/test_augmented_collision_bridge.py::test_pstf_radial_provider_can_skip_static_bundle_store_for_live_factories tests/test_augmented_collision_bridge.py::test_bd479_runtime_electron_mass_scale_respects_static_bundle_opt_out tests/test_augmented_collision_bridge.py::test_bd466_runtime_electron_mass_scale_matches_static_factory_state`
  - `4 passed in 2.22s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_collision_bridge.py`
  - `90 passed in 12.35s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_jax_augmented_typeI_replay.py`
  - `75 passed in 87.78s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py`
  - `315 passed, 3 warnings in 116.19s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m py_compile src/rabbit/transport/augmented_typeI_weak_network.py`
  - PASS

Post-run artifact commands:

- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py diagnostic_outputs/bd565_provider_static_bundle_endpoint > diagnostic_outputs/bd565_provider_static_bundle_endpoint/bd565_perf_summary.json`
  - PASS
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py diagnostic_outputs/bd565_provider_static_bundle_endpoint > diagnostic_outputs/bd565_provider_static_bundle_endpoint/bd565_component_check.txt`
  - PASS, output `PASS component wall attribution`

## Cost Effectiveness

Experimental patch size before rollback:

- added lines: 2
- deleted lines: 2
- net lines: 0
- exact token use: UNAVAILABLE, harness does not expose exact per-PR token
  counters to the repository
- blocker movement ratio: rejected despite `0.54%` selected-wall improvement,
  because payload wall regressed and static bundle hits stayed zero
- verdict: REJECTED, no causal payload/provider blocker movement

## Raw State

Raw physics state was not changed in retained code because the experimental
runtime patch was rolled back.  The endpoint artifact is preserved as
diagnostic evidence.

QKE remains out of scope.  No public-production claim is made.  PR-B
LRS/non-LRS parity and cold `N_eff_3T >= 3.0` remain default-on blockers.
