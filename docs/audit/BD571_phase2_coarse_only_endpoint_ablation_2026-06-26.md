# BD571 Phase-2 Coarse-Only Endpoint Ablation

Date: 2026-06-26

## Scope

BD571 reruns the BD563 q4 endpoint matrix recipe with the existing opt-in
`--phase2-network-pair-controller-policy coarse_only_diagnostic` flag.  This is
an endpoint performance ablation, not a default-on optimization, not a
publication/public-production claim, and not QKE validation.

The goal is to quantify the endpoint wall upper bound from skipping the refined
phase-2 pair path while preserving raw observable deltas, AB2 counters, rejected
counts, component attribution, and generated artifacts.  Because the policy is
diagnostic-only and changes raw observables, it remains opt-in.

## Exact Command

```bash
mkdir -p diagnostic_outputs/bd571_phase2_coarse_only_endpoint/checkpoints \
  diagnostic_outputs/bd571_phase2_coarse_only_endpoint/jax_cache

PYTHONPATH=src JAX_PLATFORMS=cpu /usr/bin/time -v \
  -o diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_time_v.txt \
  venv/bin/python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py \
  --output diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_q4_coarse_only_endpoint.json \
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
  --phase2-network-pair-controller-policy coarse_only_diagnostic \
  --stop-at-T-gamma-MeV 0.01 \
  --span-row-checkpoint-dir diagnostic_outputs/bd571_phase2_coarse_only_endpoint/checkpoints \
  --jax-compilation-cache-dir diagnostic_outputs/bd571_phase2_coarse_only_endpoint/jax_cache \
  --progress-jsonl \
  > diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_run.log 2>&1

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/summarize_perf_artifacts.py \
  diagnostic_outputs/bd571_phase2_coarse_only_endpoint \
  > diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_perf_summary.json

PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/check_component_wall_attribution.py \
  diagnostic_outputs/bd571_phase2_coarse_only_endpoint \
  > diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_component_check.txt
```

## Artifact Paths

- Final JSON:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_q4_coarse_only_endpoint.json`
- Perf summary:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_perf_summary.json`
- Component checker:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_component_check.txt`
- `/usr/bin/time -v`:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_time_v.txt`
- Progress log:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/bd571_run.log`
- Span checkpoints:
  `diagnostic_outputs/bd571_phase2_coarse_only_endpoint/checkpoints/`

## Command Results

| Command | Result |
| --- | --- |
| Endpoint run | PASS, exit status 0 |
| Summarizer | PASS, JSON valid |
| Component checker | PASS, `PASS component wall attribution` |

Resource summary from `/usr/bin/time -v`:

| Metric | Value |
| --- | ---: |
| Elapsed wall | `26:25.24` |
| User time | `1580.14 s` |
| System time | `27.90 s` |
| CPU | `101%` |
| Max RSS | `4557888 KB` |
| Exit status | `0` |

## Same-Recipe Comparison: BD563 vs BD571

BD563 is the current accepted-RHS-reuse + periodic4 endpoint baseline.  BD571 is
the same matrix recipe plus `--phase2-network-pair-controller-policy
coarse_only_diagnostic`.

| Metric | BD563 | BD571 | Delta |
| --- | ---: | ---: | ---: |
| `/usr/bin/time` elapsed | `42:15.68` | `26:25.24` | `-15:50.44` |
| `/usr/bin/time` elapsed seconds | 2535.68 | 1585.24 | -950.44 (-37.48%) |
| Max RSS KB | 4572180 | 4557888 | -14292 (-0.31%) |
| Selected wall s | 2499.378611 | 1548.009806 | -951.368805 (-38.06%) |
| Phase2 corrector wall s | 1221.549889 | 254.994739 | -966.555150 (-79.13%) |
| Payload build wall s | 823.276482 | 834.099025 | +10.822543 (+1.32%) |
| Source nonpayload overhead wall s | 68.368362 | 69.625359 | +1.256997 (+1.84%) |
| Host Jacobian wall s | 173.619921 | 175.241279 | +1.621358 (+0.93%) |
| Linear system wall s | 6.949313 | 7.006688 | +0.057375 (+0.83%) |
| Residual unattributed wall s | 205.614644 | 207.042717 | +1.428072 (+0.69%) |
| Source evaluations | 87840 | 87840 | 0 |
| Stage source evaluations | 76846 | 76846 | 0 |
| Payload builds | 12198 | 12198 | 0 |
| Step count | 10972 | 10972 | 0 |
| Sum `n_rejected` over nested span rows | 6 | 6 | 0 |

Interpretation: the endpoint wall gain is almost entirely a phase-2 pair-path
gain.  Payload, source nonpayload, host Jacobian, linear solve, residual wall,
source-evaluation count, payload-build count, and rejected-step count are
effectively unchanged or slightly worse.  This is useful as a phase-2 upper
bound, but not acceptable as a default because raw observables drift.

## Case-Level Endpoint Rows

The command runs the same two freedom cases as BD563.  Case-level wall is the
sum of live span-row checkpoint wall values.

| Case | BD563 wall s | BD571 wall s | Delta | BD563 `Yp` | BD571 `Yp` | BD563 `D/H` | BD571 `D/H` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pairwise `weak_rate_corrections+neutrino_collision_terms` | 1224.311017 | 743.379764 | -480.931252 | 0.242000235664 | 0.242001909403 | 2.492936049701e-05 | 2.494429907588e-05 |
| all `weak_rate_corrections+non_lrs_geometry+neutrino_collision_terms` | 1275.067594 | 804.630042 | -470.437552 | 0.242016521945 | 0.242018195756 | 2.493028169465e-05 | 2.494522084839e-05 |

Both cases preserve step and rejection counts: 5486 steps and 3 rejected steps
per case in both BD563 and BD571.

## Raw Observable Delta

The selected row is the zero-shear non-LRS all-freedoms endpoint row.

| Metric | BD563 | BD571 | Delta |
| --- | ---: | ---: | ---: |
| `T_final_MeV` | 0.00913961404501975 | 0.00913961404501975 | 0 |
| `Yp` | 0.24201652194490023 | 0.2420181957562044 | +1.67381130417e-06 |
| `D/H` | 2.493028169464549e-05 | 2.4945220848394052e-05 | +1.49391537486e-08 |
| `N_eff_3T` | 3.0348087179727026 | 3.0348087179727026 | 0 |
| `Sigma_H` | 3.3286755172789884e-31 | 3.3286755172789884e-31 | 0 |
| AB2 raw negative count | 8 | 4 | -4 |
| AB2 raw negative min | -1.927373191598319e-06 | -1.927373191598319e-06 | 0 |
| AB2 initial-guess rejected total | 33233 | 740 | -32493 |
| AB2 displacement-guard rejected total | 33203 | 728 | -32475 |
| AB2 residual-guard rejected total | 22 | 8 | -14 |
| Phase2 corrector rejected total | 0 | 0 | 0 |

Raw negative AB2 predictor evidence remains serialized and visible.  BD571 does
not clip, hide, or sanitize the changed terminal observables.

## Component Wall Attribution

| Component | BD563 wall s | BD571 wall s | Delta |
| --- | ---: | ---: | ---: |
| Phase2 corrector | 1221.549889 | 254.994739 | -966.555150 |
| Payload | 823.276482 | 834.099025 | +10.822543 |
| Source nonpayload overhead | 68.368362 | 69.625359 | +1.256997 |
| Host Jacobian | 173.619921 | 175.241279 | +1.621358 |
| Outer linear system | 6.949313 | 7.006688 | +0.057375 |
| Residual unattributed | 205.614644 | 207.042717 | +1.428072 |
| Total selected wall | 2499.378611 | 1548.009806 | -951.368805 |

The component checker passed.  No negative or NaN residual was observed.

## Decision

BD571 is a strong phase-2 endpoint-wall upper bound, not a valid default
optimization:

- It reduces selected endpoint wall by `951.37 s` (`38.06%`) and phase-2 wall by
  `966.56 s` (`79.13%`).
- It changes raw terminal `Yp` by `1.67e-06` and `D/H` by `1.49e-08`.
- It does not reduce payload builds, source evaluations, stage evaluations, or
  rejected steps.
- It sharply reduces AB2 guard work because the refined pair path is skipped,
  which is exactly why raw observable drift must be treated seriously.

The next PR should not default this policy on.  It should instead design an
opt-in selective refined/coarse controller that preserves the BD563 raw
observables within an explicit same-recipe endpoint delta budget while retaining
a material fraction of the BD571 phase-2 wall reduction.

## Cost-Effectiveness

Line cost: documentation-only in tracked files for this PR.  Diagnostic outputs
are generated artifacts and are not committed.

Exact token counters: UNAVAILABLE because the harness does not expose per-PR
token accounting.

Blocker movement ratio: 0.65.  BD571 does not change solver behavior, but it
moves the endpoint blocker by identifying a measured, large phase-2 wall upper
bound and quantifying the raw-observable cost of the coarse-only shortcut.

Cost verdict: ACCEPT.  This is not another wrapper or segment-only benchmark;
it is a full endpoint run that directly constrains the next phase-2 controller
PR.
