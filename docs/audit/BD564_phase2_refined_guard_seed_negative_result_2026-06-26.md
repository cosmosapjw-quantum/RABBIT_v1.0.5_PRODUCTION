# BD564 Phase-2 Refined Guard Seed Negative Result

Date: 2026-06-26

## Status

BD564 was a measured endpoint experiment and is **rejected** as an
optimization.  The experimental code was saved as an artifact and rolled back
before commit because it did not improve endpoint wall time.

Artifact directory:

- `diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/`

Retained evidence:

- `bd564_q4_ab2_guard_seed_endpoint.json`
- `bd564_perf_summary.json`
- `bd564_component_check.txt`
- `bd564_time_v.txt`
- `bd564_run.log`
- `checkpoints/`
- `bd564_reverted_code_experiment.diff`

## Experiment

Candidate: seed the refined phase-2 attempt's adaptive AB2 residual-guard trust
state from a clean coarse attempt in the same full-step-vs-two-half-steps pair.
This was intentionally narrower than terminal-state warm start: it did not pass
the coarse endpoint state as a Newton initial guess, and it was restricted to
the existing opt-in `adaptive_trust_after_acceptance` policy.

Reason for trying it: BD563 showed phase-2 corrector wall as the largest
same-recipe endpoint component, with `1221.549889 s` attributed phase-2 wall out
of `2499.378611 s` selected row wall.

## Exact Endpoint Command

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/bd564_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/bd564_q4_ab2_guard_seed_endpoint.json \
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
  --span-row-checkpoint-dir diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/jax_cache \
  --progress-jsonl
```

## BD563 vs BD564 Endpoint Result

Same recipe baseline: `docs/audit/BD563_post_deflation_endpoint_baseline_2026-06-26.md`.

| Metric | BD563 baseline | BD564 experiment | Delta |
| --- | ---: | ---: | ---: |
| exit status | 0 | 0 | 0 |
| `/usr/bin/time` elapsed | `42:15.68` | `42:38.50` | `+22.82 s` |
| max RSS KB | 4572180 | 4564148 | -8032 |
| selected row wall s | 2499.378611 | 2522.151432 | +22.772821 |
| phase2 corrector wall s | 1221.549889 | 1238.468794 | +16.918905 |
| payload wall s | 823.276482 | 827.508204 | +4.231722 |
| source nonpayload wall s | 68.368362 | 68.952760 | +0.584399 |
| host Jacobian wall s | 173.619921 | 174.399129 | +0.779208 |
| outer linear wall s | 6.949313 | 6.964645 | +0.015332 |
| residual unattributed wall s | 205.614644 | 205.857900 | +0.243255 |
| total rejected steps | 6 | 6 | 0 |
| total step count | 10972 | 10972 | 0 |
| total source evals | 87840 | 87840 | 0 |
| total stage source evals | 76846 | 76846 | 0 |
| dynamic payload builds | 12198 | 12198 | 0 |

Final all-freedom endpoint row:

| Observable | BD563 | BD564 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | 0.00913961404501975 | 0.00913961404501975 | 0 |
| `Yp` | 0.24201652194490023 | 0.24201652194490023 | 0 |
| `D/H` | 2.493028169464549e-05 | 2.493028169464549e-05 | 0 |
| `N_eff_3T` | 3.0348087179727026 | 3.0348087179727026 | 0 |
| `Sigma_H` | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 0 |

## Interpretation

SUPPORTED:

- The endpoint run completed and the component checker passed.
- Raw endpoint observables were unchanged to the serialized precision.
- Step, attempt, source, rejection, and dynamic payload-build counts were
  unchanged.

CONTRADICTED:

- The candidate did not improve endpoint wall.
- The candidate did not reduce phase-2 wall; phase-2 wall increased by
  `16.918905 s`.

PARTIAL:

- Focused tests showed the seeded path could be wired in a controlled fake
  adaptive pair, but endpoint artifacts did not surface useful seed telemetry
  at row/checkpoint depth.  That makes the behavior hard to audit and not worth
  keeping after the measured regression.

ACTIONABLE_NOW:

- Do not keep BD564's refined-attempt AB2 residual-guard seeding.
- Switch the next endpoint-facing candidate to payload/provider factory
  deflation or a phase-2 change that removes actual work rather than carrying a
  trust counter.

## Commands Run

Experimental RED tests before implementation:

- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_ap65_phase2_corrector.py::test_bd564_phase2_step_attempt_kwargs_pass_ab2_guard_seed_when_supported`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py::test_bd564_bdf2_step_attempt_honors_seeded_adaptive_guard_without_residual_probe`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py::test_bd564_phase2_adaptive_pair_passes_clean_coarse_guard_seed_to_refined`

Experimental GREEN tests before rollback:

- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py::test_bd564_bdf2_step_attempt_honors_seeded_adaptive_guard_without_residual_probe tests/test_augmented_continuous_ap65_rhs.py::test_bd564_phase2_adaptive_pair_passes_clean_coarse_guard_seed_to_refined tests/test_ap65_phase2_corrector.py::test_bd564_phase2_step_attempt_kwargs_pass_ab2_guard_seed_when_supported`
  - `3 passed in 0.60s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_ap65_phase2_corrector.py tests/test_augmented_continuous_ap65_rhs.py::test_bd564_bdf2_step_attempt_honors_seeded_adaptive_guard_without_residual_probe tests/test_augmented_continuous_ap65_rhs.py::test_bd564_phase2_adaptive_pair_passes_clean_coarse_guard_seed_to_refined tests/test_augmented_continuous_ap65_rhs.py::test_bd564_phase2_adaptive_pair_does_not_seed_refined_after_raw_negative_coarse tests/test_augmented_continuous_ap65_rhs.py::test_bd395_ab2_trust_displacement_policy_skips_residual_guard tests/test_augmented_continuous_ap65_rhs.py::test_bd419_ab2_adaptive_trust_residual_guard_resets_after_rejection tests/test_augmented_continuous_ap65_rhs.py::test_bd516_ab2_adaptive_trust_default_threshold_skips_after_four_acceptances tests/test_augmented_continuous_ap65_rhs.py::test_bd360_phase2_adaptive_pair_sums_ab2_counters_across_attempts tests/test_augmented_continuous_ap65_rhs.py::test_bd360_phase2_adaptive_pair_failure_sums_ab2_counters_across_attempts tests/test_augmented_continuous_ap65_rhs.py::test_bd333_phase2_adaptive_pair_splits_background_and_step_attempt_wall_timers tests/test_augmented_continuous_ap65_rhs.py::test_bd333_phase2_adaptive_pair_caches_step_attempt_signature`
  - `24 passed in 2.03s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_rhs.py`
  - `318 passed, 3 warnings in 123.94s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`
  - `304 passed, 1 skipped, 2 warnings in 84.03s`
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m py_compile src/rabbit/validation/ap65_phase2_corrector.py src/rabbit/validation/augmented_continuous_ap65_rhs.py`
  - PASS
- `git diff --check`
  - PASS

Post-run artifact commands:

- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint > diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/bd564_perf_summary.json`
  - PASS
- `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint > diagnostic_outputs/bd564_phase2_refined_guard_seed_endpoint/bd564_component_check.txt`
  - PASS, output `PASS component wall attribution`

## Cost Effectiveness

Experimental patch size before rollback:

- added lines: 403
- deleted lines: 2
- net lines: +401
- exact token use: UNAVAILABLE, harness does not expose exact per-PR token
  counters to the repository
- blocker movement ratio: `-0.91%` selected-row wall regression
  (`+22.772821 s / 2499.378611 s`)
- verdict: REJECTED, because added code surface increased endpoint wall and did
  not reduce any measured work count

## Raw State

Raw physics state was not changed in the retained code because the experimental
runtime patch was rolled back.  The negative endpoint artifact is preserved as
diagnostic evidence only.

QKE remains out of scope.  No public-production claim is made.  PR-B
LRS/non-LRS parity and cold `N_eff_3T >= 3.0` remain default-on blockers.
