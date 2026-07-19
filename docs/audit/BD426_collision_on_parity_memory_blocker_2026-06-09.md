# BD426 Collision-On Parity Memory Blocker

Date: 2026-06-09

Scope: private augmented Type-I PSTF no-QKE AP65/Rodas5P q4 cold-endpoint
evidence.  QKE remains out of scope.  This note does not claim public
production, publication readiness, Bianchi validity, resolution convergence, or
default-on optimization.

## Question

Does the current head, after the BD425 collision-off PR-B evidence, also pass
the PR-B zero-shear FLRW LRS/non-LRS cold endpoint `N_eff_3T` floor and parity
blocker when AP65 neutrino collision terms are enabled?

## Command

Dry-run controls were first checked with the same output path and the case file
from the BD416 collision-on thermal pairwise attempt:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python \
  scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --dry-run \
  --output diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_q4_thermal_start_lrs_nonlrs_collision_on_parity.json \
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
  --stop-at-T-gamma-MeV 0.01
```

The long run was then started as:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_time.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_q4_thermal_start_lrs_nonlrs_collision_on_parity.json \
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
  --progress-jsonl \
  > diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_stdout.json \
  2> diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_progress.jsonl
```

Companion artifact commands:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd426_pr_b_collision_on_parity_q4 \
  > diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd426_pr_b_collision_on_parity_q4 \
  > diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_component_wall_check.txt
```

The compact progress extract is:

```text
diagnostic_outputs/bd426_pr_b_collision_on_parity_q4/bd426_partial_run_forensics.json
```

## Result

The run was manually interrupted with SIGINT to avoid host OOM after sustained
memory growth.  `/usr/bin/time -v` recorded:

```text
Command terminated by signal 2
Elapsed wall: 57:24.91
Maximum resident set size: 39075768 KB
```

The Codex command session returned exit code `130`.  The `Exit status: 0` field
inside `/usr/bin/time` is not a success claim here; the same file records signal
2 termination.

The final JSON artifact remained the incremental prewarm placeholder:

```text
partial_resolution_ladder_artifact = true
completion_class = dynamic_collision_payload_cache_prewarm_completed_child_pending
```

Therefore the component checker correctly failed:

```text
FAIL component wall attribution
- component attribution has no rows
```

This failure is preserved as evidence.  No final component-wall attribution
exists for BD426 because the full pairwise artifact was not flushed.

## Runtime Progress Recovered From JSONL

`bd426_progress.jsonl` contains a CPU-host JAX/ROCm warning prelude before JSON
events, then nine `span_row_completed` JSON events.  The row payload stores
runtime counters under `solver`.

| freedom case | span | completion | wall s | payload builds | N_eff_3T | Yp | D/H | Sigma_H |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `weak_rate_corrections+neutrino_collision_terms` | `[0.0, 1.0]` | hot | 458.86600295000244 | 6402 | 9.330844557152059 | 1e-30 | 6.0203269046316844e-31 | 4.169862564393419e-29 |
| `weak_rate_corrections+neutrino_collision_terms` | `[1.0, 2.0]` | hot | 444.4609395590378 | 6397 | 5.097317539109222 | 1e-30 | 5.886448689919235e-31 | 1.47942299053715e-29 |
| `weak_rate_corrections+neutrino_collision_terms` | `[2.0, 2.25]` | hot | 116.81687705597142 | 1601 | 4.227169013824763 | 1e-30 | 5.839095275613899e-31 | 1.155415687757622e-29 |
| `weak_rate_corrections+neutrino_collision_terms` | `[2.25, 2.5]` | hot | 113.46885514899623 | 1601 | 3.601119249152914 | 1e-30 | 5.770773868049505e-31 | 8.807350941793004e-30 |
| `weak_rate_corrections+neutrino_collision_terms` | `[2.5, 2.75]` | hot | 314.7561973069678 | 1636 | 3.23410071010404 | 0.1900429926753419 | 0.0031655836782320057 | 6.530460168517527e-30 |
| `weak_rate_corrections+neutrino_collision_terms` | `[2.75, 3.0]` | hot | 451.6947059169761 | 3203 | 3.07989740709401 | 0.24184642166722506 | 0.00011907050621400291 | 5.0418569248375644e-30 |
| `weak_rate_corrections+neutrino_collision_terms` | `[3.0, 4.0]` | hot | 559.0439961289521 | 12795 | 3.0347646937660633 | 0.24199880575522173 | 2.620130546712362e-05 | 1.665871752990435e-30 |
| `weak_rate_corrections+neutrino_collision_terms` | `[4.0, 4.8]` | full endpoint | 449.4690671490389 | 10242 | 3.034764639534682 | 0.24200013462618128 | 2.4929260417469447e-05 | 5.641674665929347e-31 |
| `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | `[0.0, 1.0]` | hot | 431.1013612310053 | 3561 | 9.330920989913897 | 1e-30 | 6.020385165645033e-31 | 2.12158434363584e-31 |

The earlier working assumption that no non-LRS progress event existed was
wrong.  The first zero-shear non-LRS collision-on span completed.  The blocker
is narrower: the pairwise collision-on run is blocked after the first non-LRS
span by continued high memory growth, before the non-LRS branch reaches the
cold endpoint.

## Interpretation

BD426 does not pass collision-on PR-B parity.  It does, however, move the
failure mode from "unknown collision-on pairwise incompletion" to a more precise
runtime blocker:

```text
JSONL-recovered LRS collision-on q4 thermal-start endpoint: completed, floor
passed.
non-LRS zero-shear collision-on q4: first span completed, then the run was
interrupted during continued non-LRS collision work after RSS reached 39.1 GB.
```

The next implementation PR should target memory, not phase-2 wall, for this
specific blocker.  Candidate code hypotheses to inspect first:

1. Freedom-composition children may retain collision payload caches across
   LRS/non-LRS geometry changes even when those caches cannot be reused safely.
2. Non-LRS collision payload/pair-leg scaffolding may be too eager or too
   persistent across spans.
3. Progress/checkpoint flushing currently preserves final observables for
   completed spans but not a recoverable final composition artifact after an
   external interrupt.

Any fix must preserve raw physics outputs and keep the collision-on optimization
state opt-in until the full PR-B floor/parity evidence exists.

## Cost-Effectiveness Self-Audit

```text
added_lines: 181
deleted_lines: 0
net_lines: 181
files_touched: 1 tracked audit note, 1 tracked validation ledger row
token_use_exact: UNAVAILABLE
token_use_basis: harness did not expose an exact token counter
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no
blocker_movement_ratio: 0.25
blocker_movement_scope: failure-mode relocation for collision-on PR-B parity; LRS endpoint and first non-LRS span are now file-backed evidence, but full pairwise parity remains blocked
validation_strengthened: yes
cost_effectiveness_verdict: FAILURE_MODE_RELOCATION
```
