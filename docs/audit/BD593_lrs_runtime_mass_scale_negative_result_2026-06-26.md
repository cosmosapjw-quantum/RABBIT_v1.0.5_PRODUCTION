# BD593 LRS Runtime Mass-Scale Negative Result

Date: 2026-06-26

## Scope

BD593 tested one payload/provider deflation hypothesis after BD592 rejected a
phase-2 ledger accumulator filter:

- add an opt-in `pstf_radial_provider_mass_scale_mode =
  runtime_dynamic_exact`;
- move the live `T_nu_e` finite-mass scale from the LRS radial source-factory
  cache key into source-call kwargs;
- require source-factory reuse at row validation depth only when that opt-in is
  active;
- keep default behavior unchanged before PR-B LRS/non-LRS parity and cold
  `N_eff_3T >= 3.0` floor tripwires.

The code path was implemented and tested, then rejected by the endpoint run.
The rejected code diff is preserved at:

`diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_rejected_code_experiment.diff`

## Commands

Focused red/green and compatibility tests while the code was present:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_jax_augmented_typeI_replay.py::test_dynamic_lrs_collision_payload_reuses_lrs_source_factory_cache \
  tests/test_jax_augmented_typeI_replay.py::test_bd593_dynamic_lrs_runtime_mass_scale_reuses_factory_across_temperature \
  tests/test_jax_augmented_typeI_replay.py::test_dynamic_lrs_collision_payload_hot_loop_minimal_metadata_skips_fingerprints \
  tests/test_jax_augmented_typeI_replay.py::test_dynamic_lrs_collision_payload_forwards_radial_grid_cache_dir
```

Result: 4 passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_collision_bridge.py::test_bd466_pstf_radial_runtime_electron_mass_scale_reuses_provider_bundle \
  tests/test_augmented_collision_bridge.py::test_bd479_runtime_electron_mass_scale_respects_static_bundle_opt_out \
  tests/test_augmented_collision_bridge.py::test_bd466_runtime_electron_mass_scale_matches_static_factory_state \
  tests/test_augmented_collision_bridge.py::test_bd466_runtime_electron_mass_scale_matches_static_radial_gaussian_state
```

Result: 4 passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_lrs_radial_cache_miss_is_not_a_row_failure \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd593_lrs_runtime_mass_scale_requires_source_factory_reuse \
  tests/test_augmented_continuous_ap65_rhs.py::test_fb69_nonlrs_dynamic_cache_miss_still_fails_closed
```

Result: 3 passed.

```bash
python -m py_compile \
  src/rabbit/jax/augmented_typeI_replay.py \
  src/rabbit/validation/augmented_continuous_ap65_rhs.py
```

Result: passed.

Endpoint command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_q4_runtime_mass_scale_endpoint.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_q4_runtime_mass_scale_case.json \
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
  --span-row-checkpoint-dir diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/jax_cache \
  --progress-jsonl
```

Result: exit status 0, elapsed `42:20.30`, max RSS `4250584 KB`.

Post-run:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint \
  > diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint \
  > diagnostic_outputs/bd593_lrs_runtime_mass_scale_endpoint/bd593_component_check.txt
```

Result: both exited 0; component checker output:
`PASS component wall attribution`.

## Endpoint Comparison

Comparison is against BD591 because BD591 is the accepted same-recipe
post-deflation endpoint recheck.  BD592 was already rejected.

| Metric | BD591 | BD593 | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | `41:16.81` | `42:20.30` | `+63.49 s` |
| Max RSS KB | `4564456` | `4250584` | `-313872` |
| Selected wall s | `2441.611232` | `2505.162387` | `+63.551155` |
| Payload wall s | `802.054217` | `811.037835` | `+8.983618` |
| Phase2 corrector wall s | `1196.285249` | `1223.461576` | `+27.176327` |
| Host JVP/Jacobian wall s | `170.215108` | `191.420645` | `+21.205537` |
| Linear-system wall s | `6.664489` | `6.868258` | `+0.203770` |
| Nonpayload source overhead s | `66.084046` | `67.654221` | `+1.570174` |

Raw endpoint observables and executable counters match BD591 exactly at the
selected-summary level:

| Metric | BD591 | BD593 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | `0.00913961404501975` | `0.00913961404501975` | `0` |
| `N_eff_3T` | `3.0348087179727026` | `3.0348087179727026` | `0` |
| `Yp` | `0.24201652194490023` | `0.24201652194490023` | `0` |
| `D/H` | `2.493028169464549e-05` | `2.493028169464549e-05` | `0` |
| `Sigma_H` | `3.3286755172789884e-31` | `3.3286755172789884e-31` | `0` |
| Steps | `10972` | `10972` | `0` |
| Source evaluations | `87840` | `87840` | `0` |
| Stage source evaluations | `76846` | `76846` | `0` |
| Dynamic payload builds | `12198` | `12198` | `0` |
| Stage payload reuse | `75642` | `75642` | `0` |
| AB2 raw-negative count | `8` | `8` | `0` |
| AB2 raw-negative min | `-1.927373191598319e-06` | `-1.927373191598319e-06` | `0` |

## Decision

Rejected.

The opt-in runtime mass-scale provider mode was active and runtime-linked, but
it did not reduce selected endpoint wall.  It reduced max RSS but increased
payload, phase2, and host JVP/Jacobian wall.  The experimental code was
therefore reverted rather than committed.

The artifact top-level `passed` flag remains `false` despite row pass and
`physical_full_bbn_span_ready=true`; the summary blocker is still
`tighten_resolution_or_solver_tolerance_until_terminal_deltas_converge`.
This is not solver validation and must not be reported as publication or
production readiness.

## Cost Line

- experimental added_lines: 143
- experimental deleted_lines: 2
- experimental net_lines: +141
- committed code net_lines: 0
- token_use_exact: UNAVAILABLE
- token_use_basis: harness does not expose an exact per-PR token counter
- runtime_behavior_changed: no, code reverted
- physics_behavior_changed: no, raw observables preserved in the rejected run
- known_blocker_reduced: yes, candidate was falsified with same-recipe
  endpoint evidence
- blocker_movement_ratio: 0.20
- cost_effectiveness_verdict: REJECT_CODE_KEEP_EVIDENCE

## Next

BD594 should avoid another provider mass-scale cache-key variant unless it is
backed by a narrower measurement explaining why host/phase2 wall increased.
The next viable path is either:

1. phase2 refined/coarse warm-start work that removes measured cold-row phase2
   wall without changing raw endpoint state, or
2. a provider deflation experiment that proves a payload reduction on the same
   endpoint recipe before any broader code is kept.

PR-B parity and the cold `N_eff_3T >= 3.0` floor tripwire remain default-on
blockers for any optimization.
