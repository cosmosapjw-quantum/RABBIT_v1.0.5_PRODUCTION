# BD446 AB2 Candidate Stats Negative Ablation

Date: 2026-06-10

Scope: augmented Type-I PSTF no-QKE AP65 q4 endpoint performance.  QKE remains
out of scope.  This was a measured optimization hypothesis test, not solver
promotion or public-production validation.

## Hypothesis

BD444 showed a large phase-2 corrector bucket and, inside it,
`phase2_step_attempt_bookkeeping=458.4774511041469 s`.  The tested hypothesis
was:

> The AB2 initial-guess raw-candidate/displacement bookkeeping is dominated by
> repeated tiny NumPy allocations over 9-species vectors; a one-pass helper with
> a reusable raw-guess buffer should reduce the q4 phase-2 bookkeeping wall
> without changing residual-guard policy, raw negative evidence, or endpoint
> observables.

## Local Patch Tested

The working-tree patch, not retained, added
`_phase2_ab2_initial_guess_candidate_stats(...)` in
`src/rabbit/validation/augmented_continuous_ap65_rhs.py` and changed
`_phase2_bdf2_newton_network_step_attempt(...)` to reuse one raw-guess buffer
for AB2 candidate construction and displacement-norm evaluation.

The patch deliberately did not change:

- AB2 residual-guard policy;
- raw negative/nonfinite evidence handling;
- Newton acceptance criteria;
- source payload reuse policy;
- q4 command-line settings.

## Tests

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd446_ab2_candidate_stats_match_vector_formula_without_losing_raw_negatives --tb=short
```

RED before implementation: failed with `AttributeError` because the helper did
not exist.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd446_ab2_candidate_stats_match_vector_formula_without_losing_raw_negatives \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd395_ab2_trust_displacement_policy_skips_residual_guard \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd419_ab2_adaptive_trust_residual_guard_resets_after_rejection \
  tests/test_augmented_continuous_ap65_rhs.py::test_bd333_phase2_step_attempt_surfaces_ab2_and_solve_wall_timers --tb=short
```

Result with the temporary patch: `4 passed`.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m py_compile \
  src/rabbit/validation/augmented_continuous_ap65_rhs.py
```

Result: passed.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q \
  tests/test_augmented_continuous_ap65_rhs.py --tb=short
```

Result with the temporary patch: `294 passed, 3 existing truncation-guard
warnings in 111.55s`.

## Full q4 Run

Path:

`diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats/`

Command:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats/bd446_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats/bd446_q4_ab2_candidate_fast_stats.json \
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
  --source-refresh-stage-collision-payload-reuse-state-atol 1e-12 \
  --progress-jsonl
```

Post-processing:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats \
  > diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats/bd446_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd446_q4_ab2_candidate_fast_stats
```

Results:

- process exit: 0
- `/usr/bin/time` elapsed: 49:20.64
- max RSS: 3,138,676 KB
- final JSON: 27 MB
- component checker: `PASS component wall attribution`
- `summary.execution_passed`: true
- failed or exception rows: 0
- PR-B blocker status:
  `passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
- controlled FLRW LRS/non-LRS `N_eff_3T` delta:
  7.701442438445838e-06

## BD444 vs BD446

| Metric | BD444 baseline | BD446 AB2 candidate stats | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | 48:47.31 | 49:20.64 | +33.33 s |
| component total wall | 2863.7824760479853 | 2896.3595082700485 | +32.577032222063 s |
| attributed wall | 2201.5930251205573 | 2223.501228924259 | +21.908203803701 s |
| residual unattributed | 662.189450927428 | 672.85827934579 | +10.668828418362 s |
| payload | 689.9324971504393 | 691.090342817537 | +1.157845667098 s |
| source evaluation | 1170.6737594259903 | 1180.7087460233015 | +10.034986597311 s |
| payload provenance | 353.01989179180237 | 357.547420337738 | +4.527528545936 s |
| payload trace | 61.58063135470729 | 62.55071886588121 | +0.970087511174 s |
| phase-2 corrector | 1342.248983921425 | 1362.802397580177 | +20.553413658752 s |
| phase-2 step-attempt bookkeeping | 458.4774511041469 | 467.2247014895547 | +8.747250385408 s |
| phase-2 Newton solve call | 407.7310150099802 | 409.4860794171109 | +1.755064407131 s |
| phase-2 AB2 RHS predictor | 168.2392590890522 | 171.98845728876768 | +3.749198199715 s |
| phase-2 AB2 residual guard | 124.24322946520988 | 127.26106284093112 | +3.017833375721 s |
| host Jacobian | 162.60417895158753 | 162.633365972724 | +0.029187021137 s |
| outer linear system | 6.807365097105503 | 6.975122553820256 | +0.167757456715 s |

Endpoint observables matched the BD444 family:

| Freedom key | T_gamma MeV | N_eff_3T | Yp | D/H | Sigma_H | W/J |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `weak_rate_corrections+neutrino_collision_terms` | 0.009139616879151824 | 3.034801016530264 | 0.24200019137645587 | 2.4929358404512697e-05 | 5.516438318504755e-31 | [62, 62] |
| `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | 0.00913961404501975 | 3.0348087179727026 | 0.24201652194550552 | 2.493028169465174e-05 | 3.3286755172789884e-31 | [62, 62] |

## Verdict

CONTRADICTED as a q4 speedup.  The patch preserved endpoint behavior and PR-B
pairwise blocker status, but it worsened the measured runtime:

- total component wall worsened by 32.58 s;
- phase-2 corrector worsened by 20.55 s;
- phase-2 step-attempt bookkeeping worsened by 8.75 s;
- elapsed wall worsened by 33.33 s.

The code change was therefore not retained.  The result is useful negative
evidence: the large phase-2 bookkeeping bucket is not fixed by replacing the
AB2 raw-guess/displacement vector operations with a Python-loop helper.

## Consequence

Do not pursue this one-pass Python helper.  The next phase-2 work should target
algorithmic work that changes the number of expensive operations or removes
larger duplicated calls, not a per-9-vector Python loop.  Candidate areas:

1. reduce phase-2 full-vs-refined attempt duplication if parity can be preserved;
2. move residual-guard/current-residual reuse into the Newton solve without
   increasing Newton iterations;
3. reduce payload build/provider work where the q4 artifact still shows large
   measured wall.
