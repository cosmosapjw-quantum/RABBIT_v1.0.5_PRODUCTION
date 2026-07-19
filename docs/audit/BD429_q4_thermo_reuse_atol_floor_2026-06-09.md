# BD429 q4 Thermo Reuse Absolute Floor

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 collision-on
pairwise parity/performance work. QKE remains out of scope. This does not claim
public production, publication readiness, Bianchi validity, high-q convergence,
resolution convergence, or default-on optimization outside private AP65 staging.

## Blocker

BD428 completed the q4 collision-on LRS/non-LRS cold endpoint pair after cache
bounding, but payload remained the largest wall bucket. Branch counters showed
the mechanism:

| branch | stage evals | payload builds | stage reuse | payload build wall |
|---|---:|---:|---:|---:|
| LRS collision-on | 38423 | 43877 | 43 | 1711.217410 s |
| zero-shear non-LRS collision-on | 38423 | 8397 | 35523 | 720.102255 s |

The replay case stored `stage_collision_payload_reuse_state_rtol=0.01` and
`stage_collision_payload_reuse_state_atol=0.0`. That legacy zero disabled the
RHS thermo-reuse absolute floor, so near-zero state components made scaled
deltas huge and blocked LRS stage reuse.

## Code Change

For dynamic q-Laguerre collision resolution cases with
`stage_collision_payload_policy=thermo_state_tolerance_reuse`, missing or zero
`stage_collision_payload_reuse_state_atol` is now resolved to
`_STAGE_COLLISION_PAYLOAD_THERMO_REUSE_STATE_DEFAULT_ATOL`. The requested config
is still serialized separately, and the auto reason is:

```text
dynamic_q_laguerre_thermo_reuse_absolute_floor
```

Exact/full-state reuse is unchanged: `state_tolerance_reuse` with
`rtol=0.0, atol=0.0` remains exact. No physics output is clipped or hidden. No
new readiness, manifest, hash, figure, or claim gate was added.

## Long-Run Evidence

Before codifying the resolver behavior, BD429 tested the same mechanism with an
explicit CLI override:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd429_q4_reuse_atol_floor/bd429_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd429_q4_reuse_atol_floor/bd429_q4_thermal_start_lrs_nonlrs_collision_on_parity.json \
  --resolution-ladder-cases-json diagnostic_outputs/bd416_pr_n2_endpoint_ab/q4_pairwise_collision_on_thermal_case.json \
  --enabled-freedoms weak_rate_corrections,non_lrs_geometry,neutrino_collision_terms \
  --weak-correction-level 0 --sigma-plus0 0.0 --sigma-minus0 0.0 \
  --initial-np-policy phase1_prerun --phase1-prerun-T-start-MeV 3.0 \
  --phase1-prerun-dN 0.002 \
  --neutrino-thermal-start-policy phase1_thermo_prerun_flrw \
  --initial-A-monopole-offset 0.0 \
  --phase2-activation-validation-mode standard_flrw \
  --stop-at-T-gamma-MeV 0.01 \
  --source-refresh-stage-collision-payload-reuse-state-atol 1e-12 \
  --progress-jsonl
```

Result: exit `0`, elapsed `47:11.96`, max RSS `3136476 KB`. Summarizer and
component checker both exited `0`; checker output was `PASS component wall
attribution`.

Against BD428:

| metric | BD428 legacy zero-atol | BD429 absolute floor | change |
|---|---:|---:|---:|
| elapsed | 1:20:11 | 47:11.96 | -41.1% |
| max RSS KB | 11423752 | 3136476 | -72.5% |
| total component wall | 4715.051391 s | 2770.427630 s | -41.2% |
| payload wall | 2431.319666 s | 657.815197 s | -72.9% |
| phase2 wall | 1367.095819 s | 1316.303678 s | -3.7% |
| residual unattributed | 743.414369 s | 632.683240 s | -14.9% |
| LRS stage reuse | 43 | 37221 | +37178 |
| non-LRS stage reuse | 35523 | 38423 | +2900 |

BD429 endpoints preserved raw observables:

| branch | T_gamma final MeV | N_eff_3T | N_eff_dist | Yp | D/H | Sigma_H |
|---|---:|---:|---:|---:|---:|---:|
| LRS collision-on | 0.009139616879151824 | 3.034801016530264 | 3.0348304440102396 | 0.24200019137645587 | 2.4929358404512697e-05 | 5.516438318504755e-31 |
| zero-shear non-LRS collision-on | 0.00913961404501975 | 3.0348087179727026 | 3.0348087179727 | 0.24201652194550552 | 2.493028169465174e-05 | 3.3286755172789884e-31 |

The artifact reports
`controlled_flrw_lrs_nonlrs_default_on_blocker_status=passed_pr_b_neff_floor_and_lrs_nonlrs_parity`
and `controlled_flrw_lrs_nonlrs_neff_delta=7.701442438445838e-06`.
Top-level `passed=false` remains expected for one-resolution/no-convergence
readiness.

## Current Component Wall

| component | wall seconds |
|---|---:|
| total row wall | 2770.4276301520877 |
| payload | 657.8151970283943 |
| phase2 corrector | 1316.3036778922542 |
| host Jacobian | 157.35264714621007 |
| outer linear system | 6.272867616848089 |
| residual unattributed | 632.683240468381 |

`W/J` remains `[62, 62]`; dense linear solve is not the q4 target. After BD429,
phase2 corrector is again the largest measured wall bucket.

## Validation

```text
Focused resolver/builder tests:
6 passed in 3.59s

py_compile:
passed

Full span-ladder focused suite:
274 passed, 1 skipped, 2 existing deterministic-reference warnings

scripts/summarize_perf_artifacts.py diagnostic_outputs/bd429_q4_reuse_atol_floor:
exit 0

scripts/check_component_wall_attribution.py diagnostic_outputs/bd429_q4_reuse_atol_floor:
exit 0, PASS component wall attribution

git diff --check:
passed
```

## Cost-Effectiveness Line

```text
added_lines: 329
deleted_lines: 9
net_lines: 320
files_touched: 4
token_use_exact: UNAVAILABLE
token_use_basis: Codex harness did not expose exact per-turn token accounting in this workspace.
runtime_behavior_changed: yes
physics_behavior_changed: no
known_blocker_reduced: yes
blocker_movement_ratio: 0.75
validation_strengthened: yes
cost_effectiveness_verdict: ACCEPT
```

The line count includes this audit note, resolver code, tests, and validation
ledger. Net lines are above the preferred budget but below the high-risk band,
and the same-case endpoint run moved the dominant payload wall by more than
30% while preserving raw endpoint state and parity/floor evidence.
